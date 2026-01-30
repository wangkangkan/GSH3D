import math
import numpy as np
import torch
import torch.nn as nn
#from lib.config import cfg
from torch.nn.functional import normalize as torch_normalize
from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_apply, quaternion_invert, matrix_to_quaternion, quaternion_to_matrix
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from pytorch3d.ops import knn_points

from lib.utils.spherical_harmonics import RGB2SH

scale_activation = torch.exp
scale_inverse_activation = torch.log
use_old_method = True

class SuGaR(nn.Module):
    """Main class for SuGaR models.
    Because SuGaR optimization starts with first optimizing a vanilla Gaussian Splatting model for 7k iterations,
    we built this class as a wrapper of a vanilla Gaussian Splatting model.
    Consequently, a corresponding Gaussian Splatting model trained for 7k iterations must be provided.
    However, this wrapper implementation may not be the most optimal one for memory usage, so we might change it in the future.
    """
    def __init__(
        self, 
        points: torch.Tensor,
        colors: torch.Tensor,
        initialize:bool=True,
        sh_levels:int=4,
        learnable_positions:bool=True,
        triangle_scale:float=2.,
        keep_track_of_knn:bool=False,
        knn_to_track:int=16,
        learn_color_only=False,
        beta_mode='average',  # 'learnable', 'average', 'weighted_average'
        freeze_gaussians=False,
        primitive_types='diamond',  # 'diamond', 'square'
        surface_mesh_to_bind=None,  # Open3D mesh
        surface_mesh_thickness=None,
        learn_surface_mesh_positions=True,
        learn_surface_mesh_opacity=True,
        learn_surface_mesh_scales=True,
        n_gaussians_per_surface_triangle=6,  # 1, 3, 4, 6 or 15
        editable=True,
        faces = None,
        verts=None,
        use_new_verts = False,
        new_verts = None,
        new_faces = None,
        optim_param_name = None,
        # If True, allows for automatically rescaling Gaussians in real time if triangles are deformed from their original shape. 
        # We wrote about this functionality in the paper, and it was previously part of sugar_compositor.py, which we haven't finished cleaning yet for this repo.
        # We now moved it to this script as it is more related to the SuGaR model than to the compositor.
        *args, **kwargs) -> None:
        """
        Args:
            self (GaussianSplattingWrapper): A vanilla Gaussian Splatting model trained for 7k iterations.
            points (torch.Tensor): Initial positions of the Gaussians (not used when wrapping).
            colors (torch.Tensor): Initial colors of the Gaussians (not used when wrapping).
            initialize (bool, optional): Whether to initialize the radiuses. Defaults to True.
            sh_levels (int, optional): Number of spherical harmonics levels to use for the color features. Defaults to 4.
            learnable_positions (bool, optional): Whether to learn the positions of the Gaussians. Defaults to True.
            triangle_scale (float, optional): Scale of the triangles used to replace the Gaussians. Defaults to 2.
            keep_track_of_knn (bool, optional): Whether to keep track of the KNN information for training regularization. Defaults to False.
            knn_to_track (int, optional): Number of KNN to track. Defaults to 16.
            learn_color_only (bool, optional): Whether to learn only the color features. Defaults to False.
            beta_mode (str, optional): Whether to use a learnable beta, or to average the beta values. Defaults to 'average'.
            freeze_gaussians (bool, optional): Whether to freeze the Gaussians. Defaults to False.
            primitive_types (str, optional): Type of primitive to use to replace the Gaussians. Defaults to 'diamond'.
            surface_mesh_to_bind (None, optional): Surface mesh to bind the Gaussians to. Defaults to None.
            surface_mesh_thickness (None, optional): Thickness of the bound Gaussians. Defaults to None.
            learn_surface_mesh_positions (bool, optional): Whether to learn the positions of the bound Gaussians. Defaults to True.
            learn_surface_mesh_opacity (bool, optional): Whether to learn the opacity of the bound Gaussians. Defaults to True.
            learn_surface_mesh_scales (bool, optional): Whether to learn the scales of the bound Gaussians. Defaults to True.
            n_gaussians_per_surface_triangle (int, optional): Number of bound Gaussians per surface triangle. Defaults to 6.
        """
        
        super(SuGaR, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.freeze_gaussians = freeze_gaussians
        
        self.learn_positions = ((not learn_color_only) and learnable_positions) and (not freeze_gaussians)
        self.learn_opacities = (not learn_color_only) and (not freeze_gaussians)
        self.learn_scales = (not learn_color_only) and (not freeze_gaussians)
        self.learn_quaternions = (not learn_color_only) and (not freeze_gaussians)
        self.learnable_positions = learnable_positions
        
        if surface_mesh_to_bind is not None:
            self.learn_surface_mesh_positions = learn_surface_mesh_positions
            self.binded_to_surface_mesh = True
            self.learn_surface_mesh_opacity = learn_surface_mesh_opacity
            self.learn_surface_mesh_scales = learn_surface_mesh_scales
            self.n_gaussians_per_surface_triangle = n_gaussians_per_surface_triangle
            self.editable = editable
            
            self.learn_positions = self.learn_surface_mesh_positions
            self.learn_scales = self.learn_surface_mesh_scales
            self.learn_quaternions = self.learn_surface_mesh_scales
            self.learn_opacities = self.learn_surface_mesh_opacity
            
            # self._surface_mesh_faces = torch.nn.Parameter(
            #     torch.tensor(np.array(surface_mesh_to_bind.triangles)).to(self.device), 
            #     requires_grad=False).to(self.device)
            if faces is not None:
                self._surface_mesh_faces = faces
            else:
                self._surface_mesh_faces = torch.nn.Parameter(
                torch.tensor(np.array(surface_mesh_to_bind.triangles)).to(self.device), 
                requires_grad=False).to(self.device) 
            if surface_mesh_thickness is None:
                surface_mesh_thickness = self.training_cameras.get_spatial_extent() / 1_000_000
            # self.surface_mesh_thickness = torch.nn.Parameter(
                # torch.tensor(surface_mesh_thickness).to(self.device), 
                # requires_grad=False).to(self.device)
            self.surface_mesh_thickness = torch.tensor(surface_mesh_thickness).to(self.device)
                
            print("Binding radiance cloud to surface mesh...")
            if n_gaussians_per_surface_triangle == 1:
                self.surface_triangle_circle_radius = 1. / 2. / np.sqrt(3.)
                self.surface_triangle_bary_coords = torch.tensor(
                    [[1/3, 1/3, 1/3]],
                    dtype=torch.float32,
                    device=self.device,
                )[..., None]
            
            if n_gaussians_per_surface_triangle == 3:
                self.surface_triangle_circle_radius = 1. / 2. / (np.sqrt(3.) + 1.)
                self.surface_triangle_bary_coords = torch.tensor(
                    [[1/2, 1/4, 1/4],
                    [1/4, 1/2, 1/4],
                    [1/4, 1/4, 1/2]],
                    dtype=torch.float32,
                    device=self.device,
                )[..., None]
            
            if n_gaussians_per_surface_triangle == 4:
                self.surface_triangle_circle_radius = 1 / (4. * np.sqrt(3.))
                self.surface_triangle_bary_coords = torch.tensor(
                    [[1/3, 1/3, 1/3],
                    [2/3, 1/6, 1/6],
                    [1/6, 2/3, 1/6],
                    [1/6, 1/6, 2/3]],
                    dtype=torch.float32,
                    device=self.device,
                )[..., None]  # n_gaussians_per_face, 3, 1
                
            if n_gaussians_per_surface_triangle == 6:
                self.surface_triangle_circle_radius = 1 / (4. + 2.*np.sqrt(3.))
                self.surface_triangle_bary_coords = torch.tensor(
                    [[2/3, 1/6, 1/6],
                    [1/6, 2/3, 1/6],
                    [1/6, 1/6, 2/3],
                    [1/6, 5/12, 5/12],
                    [5/12, 1/6, 5/12],
                    [5/12, 5/12, 1/6]],
                    dtype=torch.float32,
                    device=self.device,
                )[..., None]
            
            if n_gaussians_per_surface_triangle == 15:
                self.surface_triangle_circle_radius = 1 / (4. + 2.*np.sqrt(3.))
                self.surface_triangle_bary_coords = torch.tensor(
                    [
                    [2/3, 1/6, 1/6],
                    [1/6, 2/3, 1/6],
                    [1/6, 1/6, 2/3],
                    [3/12, 1/6, 1/6],
                    [1/6, 3/12, 1/6],
                    [1/6, 1/6, 3/12],
                    [1/6, 5/12, 5/12],
                    [5/12, 1/6, 5/12],
                    [5/12, 5/12, 1/6],
                    [1/3, 5/12, 5/12],
                    [5/12, 1/3, 5/12],
                    [5/12, 5/12, 1/3],
                    [8/12, 1/6, 8/12],
                    [8/12, 8/12, 1/6],
                    [1/6, 8/12, 8/12],
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )[..., None]
            if verts is not None:
                points = verts
            else:
                points = torch.tensor(np.array(surface_mesh_to_bind.vertices)).float().to(self.device)
            # verts_normals = torch.tensor(np.array(surface_mesh_to_bind.vertex_normals)).float().to(self.device)
            self._vertex_colors = torch.tensor(np.array(surface_mesh_to_bind.vertex_colors)).float().to(self.device) # NOTE: 初始coarse mesh是有颜色的
            if len(self._vertex_colors) < len(points):
                self._vertex_colors = torch.rand([len(points), 3]).float().to(self.device)
            faces_colors = self._vertex_colors[self._surface_mesh_faces]  # n_faces, 3, n_coords 
            colors = faces_colors[:, None] * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_colors
            colors = colors.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_colors
            colors = colors.reshape(-1, 3)  # n_faces * n_gaussians_per_face, n_colors
            
            self.initpoint = points            
            # self._points = nn.Parameter(points, requires_grad=self.learn_positions).to(self.device)
            self._points = points
            n_points = len(np.array(surface_mesh_to_bind.triangles)) * n_gaussians_per_surface_triangle
            self._n_points = n_points
            
        else:
            self.binded_to_surface_mesh = False
            self._points = nn.Parameter(points, requires_grad=self.learn_positions).to(self.device)
            n_points = len(self._points)
                                   
        self.scale_activation = scale_activation
        self.scale_inverse_activation = scale_inverse_activation
        
        # First gather vertices of all triangles
        faces_verts = self._points[self._surface_mesh_faces]  # n_faces, 3, n_coords
        
        # Then, compute initial scales
        # NOTE: 初始化高斯的尺度
        # scales = (faces_verts - faces_verts[:, [1, 2, 0]]).norm(dim=-1).min(dim=-1)[0] * self.surface_triangle_circle_radius
        # scales = scales.clamp_min(0.0000001).reshape(len(faces_verts), -1, 1).expand(-1, self.n_gaussians_per_surface_triangle, 2).clone().reshape(-1, 2)
        # self._scales = nn.Parameter(
        #     scale_inverse_activation(scales),
        #     requires_grad=self.learn_surface_mesh_scales).to(self.device)
        
        # # We actually don't learn quaternions here, but complex numbers to encode a 2D rotation in the triangle's plane
        # complex_numbers = torch.zeros(self._n_points, 2).to(self.device)
        # complex_numbers[:, 0] = 1.
        # self._quaternions = nn.Parameter(
        #     complex_numbers,
        #     requires_grad=self.learn_surface_mesh_scales).to(self.device)
            
        # Reference scaling factor
        if self.editable:
            if use_old_method:
                self.reference_scaling_factor = (faces_verts - faces_verts.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=-1, keepdim=True)
            else:
                self._reference_points = self._points.clone().detach()
                self._reference_normals = self.surface_mesh.faces_normals_list()[0].clone()
                self.edited_cache = None
        
        # Initialize color features
        # self.sh_levels = sh_levels
        # sh_coordinates_dc = RGB2SH(colors).unsqueeze(dim=1)
        # self._sh_coordinates_dc = nn.Parameter(
        #     sh_coordinates_dc.to(self.device),
        #     requires_grad=True and (not freeze_gaussians)
        # ).to(self.device)
        
        # self._sh_coordinates_rest = nn.Parameter(
        #     torch.zeros(n_points, sh_levels**2 - 1, 3).to(self.device),
        #     requires_grad=True and (not freeze_gaussians)
        # ).to(self.device)

        self.use_new_verts = use_new_verts
        if use_new_verts and new_faces is not None and new_verts is not None:
            self.new_verts = torch.tensor(new_verts).float().to(self.device)
            self.new_faces = torch.tensor(new_faces).long().to(self.device)
            faces_verts = self.new_verts[self.new_faces]
            self.reference_scaling_factor = (faces_verts - faces_verts.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=-1, keepdim=True)
        
        
        # NOTE: 测试clone和prone
        faces_len = len(self._surface_mesh_faces)
        if use_new_verts:
            faces_len = len(new_faces)
        # self._xyz_weight = nn.Parameter(
                    # torch.tensor([[2/3, 1/6, 1/6],
                    # [1/6, 2/3, 1/6],
                    # [1/6, 1/6, 2/3],
                    # [1/6, 5/12, 5/12],
                    # [5/12, 1/6, 5/12],
                    # [5/12, 5/12, 1/6]],
                    # dtype=torch.float32,
                    # device=self.device,).repeat(faces_len, 1),
                # ).to(self.device)
        self._xyz_weight = torch.tensor([[2/3, 1/6, 1/6],
                    [1/6, 2/3, 1/6],
                    [1/6, 1/6, 2/3],
                    [1/6, 5/12, 5/12],
                    [5/12, 1/6, 5/12],
                    [5/12, 5/12, 1/6]],
                    dtype=torch.float32,
                    device=self.device).repeat(faces_len, 1)      
        xyzidx =  [torch.full([n_gaussians_per_surface_triangle], i, dtype=torch.long) for i in range(faces_len)]
        xyzidx = torch.cat(xyzidx).to(self.device)
        self._xyz_idx = xyzidx#nn.Parameter(xyzidx, requires_grad=False).to(self.device)
        
        self.xyz_gradient_accum = torch.zeros((self._xyz_weight.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self._xyz_weight.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self._xyz_weight.shape[0]), device=self.device)
        
        self.optim_param_name = optim_param_name
    
        
        self.load_objUV()
        self.UV_GS = self.get_UV_GS()
        #np.savetxt('gsuv.txt',self.UV_GS.detach().cpu().numpy())
        #np.savetxt('vertuv.txt',self.vert_uv.detach().cpu().numpy())
    
    # @property
    # def device(self):
    #     return self.device
    
    @property
    def n_points(self):
        if not self.binded_to_surface_mesh:
            return len(self._points)
        else:
            return self._n_points
    
    @property
    def gs_vert_idx(self):
    
        used_faces = self._surface_mesh_faces[self._xyz_idx]
        gs_vert_idx = used_faces[:,0]#choosing one vertex for each face
        
        return gs_vert_idx
        
    @property
    def points(self):
        if not self.binded_to_surface_mesh:
            if (not self.learnable_positions) and self.learnable_shifts:
                return self._points + self.max_shift * 2 * (torch.sigmoid(self.shifts) - 0.5)
            else:
                return self._points
        else:
            # First gather vertices of all triangles
            used_faces = self._surface_mesh_faces[self._xyz_idx]
            faces_verts = self._points[used_faces]  # n_faces, 3, n_coords
            
            # Then compute the points using barycenter coordinates in the surface triangles
            # points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
            used_weight = torch.clamp(self._xyz_weight, 1e-6, 1.0)
            #used_weight = torch.sigmoid(self._xyz_weight)
            weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
            used_weight = used_weight / weight_sum
            
            points = faces_verts * used_weight[..., None] 
            
            # points = faces_verts * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
            points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
            
            return points.reshape(-1, 3)  # n_faces * n_gaussians_per_face, n_coords
    
    @property
    def newpoints(self):
        assert self.use_new_verts, "未使用此模式"
        # First gather vertices of all triangles
        # faces_verts = self.new_verts[self.new_faces]  # n_faces, 3, n_coords
        
        # # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        # points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        used_faces = self.new_faces[self._xyz_idx]
        faces_verts = self.new_verts[used_faces]  # n_faces, 3, n_coords
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        used_weight = torch.clamp(self._xyz_weight, 0.0, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum
        
        points = faces_verts * used_weight[:, None]
        points = points.sum(dim=-2)
        return points.reshape(-1, 3)  # n_faces * n_gaussians_per_face, n_coords
            
    @property
    def deltapoint_loss(self):        
        
        deltap = self._points-self.initpoint
        deltaploss = torch.nn.functional.smooth_l1_loss(deltap, torch.zeros_like(deltap)).mean()
        return deltaploss
        
    @property
    def surface_mesh(self):
        # Create a Meshes object
        surface_mesh = Meshes(
            verts=[self._points.to(self.device)],   
            faces=[self._surface_mesh_faces.to(self.device)],
            textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            # verts_normals=[verts_normals.to(rc.device)],
            )
        return surface_mesh
    
    def deform_mesh(self, points):
        # Create a Meshes object
        surface_mesh = Meshes(
            verts=[points],   
            faces=[self._surface_mesh_faces.to(self.device)],
            textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            # verts_normals=[verts_normals.to(rc.device)],
            )
        return surface_mesh
        
    @property
    def sh_coordinates(self):
        return torch.cat([self._sh_coordinates_dc, self._sh_coordinates_rest], dim=1)
    
    @property
    def scaling(self):
        if not self.binded_to_surface_mesh:
            scales = self.scale_activation(self._scales)
        else:
            plane_scales = self.scale_activation(self._scales)
            if self.editable:
                if use_old_method:
                    # Old method described in the original SuGaR paper
                    faces_verts = self._points[self._surface_mesh_faces]
                    faces_centers = faces_verts.mean(dim=1, keepdim=True)
                    scaling_factor = (faces_verts - faces_centers).norm(dim=-1).mean(dim=-1, keepdim=True) / self.reference_scaling_factor
                    plane_scales = plane_scales * scaling_factor[:, None].expand(-1, self.n_gaussians_per_surface_triangle, -1).reshape(-1, 1)
                else:
                    # New method with better scaling
                    if (self.edited_cache is not None) and self.edited_cache.shape[-1]==3:
                        scales = self.edited_cache
                        self.edited_cache = None
                    else:
                        quaternions, scales = self.get_edited_quaternions_and_scales()
                        self.edited_cache = quaternions
                    return scales

            scales = torch.cat([
                self.surface_mesh_thickness * torch.ones(len(self._scales), 1, device=self.device), 
                plane_scales,
                ], dim=-1)
        return scales
    
    # NOTE: 四元数转换
    @property
    def quaternions(self):
        if not self.binded_to_surface_mesh:
            quaternions = self._quaternions
        else:
            if (not self.editable) or (use_old_method):                
                # We compute quaternions to enforce face normals to be the first axis of gaussians
                R_0 = torch.nn.functional.normalize(self.surface_mesh.faces_normals_list()[0], dim=-1)

                # We use the first side of every triangle as the second base axis
                faces_verts = self._points[self._surface_mesh_faces]
                base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)

                # We use the cross product for the last base axis
                base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
                
                # We now apply the learned 2D rotation to the base quaternion
                complex_numbers = torch.nn.functional.normalize(self._quaternions, dim=-1).view(len(self._surface_mesh_faces), self.n_gaussians_per_surface_triangle, 2)
                R_1 = complex_numbers[..., 0:1] * base_R_1[:, None] + complex_numbers[..., 1:2] * base_R_2[:, None]
                R_2 = -complex_numbers[..., 1:2] * base_R_1[:, None] + complex_numbers[..., 0:1] * base_R_2[:, None]

                # We concatenate the three vectors to get the rotation matrix
                R = torch.cat([R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
                            R_1[..., None],
                            R_2[..., None]],
                            dim=-1).view(-1, 3, 3)
                quaternions = matrix_to_quaternion(R)
            else:
                if (self.edited_cache is not None) and self.edited_cache.shape[-1]==4:
                    quaternions = self.edited_cache
                    self.edited_cache = None
                else:
                    quaternions, scales = self.get_edited_quaternions_and_scales()
                    self.edited_cache = scales
            
        return torch.nn.functional.normalize(quaternions, dim=-1)
    
    def get_edited_points(self, points):
        
        used_faces = self._surface_mesh_faces[self._xyz_idx]
        faces_verts = points[used_faces]
        
        # faces_verts = points[self._surface_mesh_faces]  #batch, n_faces, 3, n_coords
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  #batch, n_faces, n_gaussians_per_face, 3, n_coords
        # points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        used_weight = torch.clamp(self._xyz_weight, 1e-6, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum

        points = faces_verts * used_weight[..., None]
        points = points.sum(dim=-2)
        
        return points.reshape(-1, 3)  #batch, n_faces * n_gaussians_per_face, n_coords
    
    def getbatch_edited_points(self, points):
        
        batchsize = points.shape[0]
        used_faces = self._surface_mesh_faces[self._xyz_idx].view(-1)
        faces_verts = points[:,used_faces].view(batchsize,-1,3,3)
        
        # faces_verts = points[self._surface_mesh_faces]  #batch, n_faces, 3, n_coords
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  #batch, n_faces, n_gaussians_per_face, 3, n_coords
        # points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        used_weight = torch.clamp(self._xyz_weight, 1e-6, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum

        points1 = faces_verts * used_weight[..., None]
        points1 = points1.sum(dim=-2)
        
        return points1.reshape(batchsize, -1, 3)  #batch, n_faces * n_gaussians_per_face, n_coords
    
    def getbatch_edited_blendtransform(self, blendtransforms):
        
        batchsize = blendtransforms.shape[0]
        used_faces = self._surface_mesh_faces[self._xyz_idx].view(-1)
        blendtransforms1 = blendtransforms.view(batchsize,6890,16)
        faces_verts = blendtransforms1[:,used_faces].view(batchsize,-1,3,16)
        
        # faces_verts = points[self._surface_mesh_faces]  #batch, n_faces, 3, n_coords
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  #batch, n_faces, n_gaussians_per_face, 3, n_coords
        # points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        used_weight = torch.clamp(self._xyz_weight, 1e-6, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum

        gsblendtransforms = faces_verts * used_weight[..., None]
        gsblendtransforms = gsblendtransforms.sum(dim=-2)
        
        return gsblendtransforms.reshape(batchsize, -1, 4,4)  #batch, n_faces * n_gaussians_per_face, n_coords
    
    def getbatch_edited_blendweights(self, blendweights):
        
        batchsize = blendweights.shape[0]
        used_faces = self._surface_mesh_faces[self._xyz_idx].view(-1)
        blendweights1 = blendweights.view(batchsize,6890,24)
        faces_verts = blendweights1[:,used_faces].view(batchsize,-1,3,24)
        
        # faces_verts = points[self._surface_mesh_faces]  #batch, n_faces, 3, n_coords
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  #batch, n_faces, n_gaussians_per_face, 3, n_coords
        # points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        used_weight = torch.clamp(self._xyz_weight, 1e-6, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum

        gsblendweights = faces_verts * used_weight[..., None]
        gsblendweights = gsblendweights.sum(dim=-2)
        
        return gsblendweights.reshape(batchsize, -1, 24)  #batch, n_faces * n_gaussians_per_face, n_coords
    
    def get_edited_blendweights(self, blendweights):
        
        used_faces = self._surface_mesh_faces[self._xyz_idx]
        faces_verts = blendweights[used_faces]
        
        # faces_verts = points[self._surface_mesh_faces]  #batch, n_faces, 3, n_coords
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  #batch, n_faces, n_gaussians_per_face, 3, n_coords
        # points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        used_weight = torch.clamp(self._xyz_weight, 1e-6, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum

        blendweights = faces_verts * used_weight[..., None]
        blendweights = blendweights.sum(dim=-2)
        
        return blendweights.reshape(-1, 24)  #batch, n_faces * n_gaussians_per_face, n_coords
        
    def load_objUV(self):
        ##############################################################################
        # Load template obj file
        print('Loading obj file')
        with open('smpl_boundry_free_template.obj', 'r') as fin:#smpl_uv_shrunk
            lines = [l
                     for l in fin.readlines()
                     if len(l.split()) > 0
                     and not l.startswith('#')
                     ]

        # Load all vertices (v) and texcoords (vt)
        vertices = []
        texcoords = []

        for line in lines:
            lsp = line.split()
            if lsp[0] == 'v':
                x = float(lsp[1])
                y = float(lsp[2])
                z = float(lsp[3])
                vertices.append((x, y, z))
            elif lsp[0] == 'vt':
                u = float(lsp[1])
                v = float(lsp[2])
                # texcoords.append((1 - v, u))
                texcoords.append((u, v))


        # Stack these into an array
        vertices = np.vstack(vertices).astype(np.float32)
        texcoords = np.vstack(texcoords).astype(np.float32)

        self.UV_coord =  torch.tensor(texcoords).to(self.device)
        
        # Load face data. All lines are of the form:
        # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
        # Store the texcoord faces and a mapping from texcoord faces to vertex faces
        vt_faces = []
        v_num = vertices.shape[0]
        vt_num = texcoords.shape[0]
        vt2v = np.zeros(vt_num).astype('int64') - 1
        v2vt = [None] * v_num
        for i in range(v_num):
            v2vt[i] = set()
        
        vert_uv = np.zeros(v_num).astype('int64') - 1

        for line in lines:
            vs = line.split()
            if vs[0] == 'f':
                v0 = int(vs[1].split('/')[0]) - 1
                v1 = int(vs[2].split('/')[0]) - 1
                v2 = int(vs[3].split('/')[0]) - 1
                vt0 = int(vs[1].split('/')[1]) - 1
                vt1 = int(vs[2].split('/')[1]) - 1
                vt2 = int(vs[3].split('/')[1]) - 1
                vt_faces.append((vt0, vt1, vt2))

                vt2v[vt0] = v0
                vt2v[vt1] = v1
                vt2v[vt2] = v2

                v2vt[v0].add(vt0)
                v2vt[v1].add(vt1)
                v2vt[v2].add(vt2)
                
                vert_uv[v0] = vt0
                vert_uv[v1] = vt1
                vert_uv[v2] = vt2

        vt_faces = np.vstack(vt_faces)
        self.mesh_faces_UV = torch.tensor(vt_faces).to(self.device)
        self.vert_uv = torch.tensor(self.UV_coord[vert_uv]).to(self.device)
        vt_count = np.zeros(v_num)
        for v_id in range(v_num):
            vt_count[v_id] = len(v2vt[v_id])

    def get_UV_GS(self):
        
        UVidx_faces = self.mesh_faces_UV[self._xyz_idx] # face uv corresponding to each GS
        UVcoord_faces = self.UV_coord[UVidx_faces] #uv coordinate of face uv corresponding to each GS

        # faces_verts = points[self._surface_mesh_faces]  #batch, n_faces, 3, n_coords
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  #batch, n_faces, n_gaussians_per_face, 3, n_coords
        # points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        used_weight = torch.clamp(self._xyz_weight, 1e-6, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum
        
        UVcoord_GS = UVcoord_faces * used_weight[..., None]
        UVcoord_GS = UVcoord_GS.sum(dim=-2)
        
        return UVcoord_GS.reshape(-1, 2)  #batch, n_faces * n_gaussians_per_face, n_coords
    
    def quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as quaternions to rotation matrices.

        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))
    
    def get_posed_quaternions(self, points, quat):
        if not self.editable:
            raise ValueError("The model is not editable. Call make_editable() first.")
        points = points.squeeze(0)
        reference_verts = points.clone().detach()[self._surface_mesh_faces]
        # reference_normals = self._reference_normals
        # NOTE: 选择每个高斯点对应的面
        used_faces = self._surface_mesh_faces[self._xyz_idx]
        faces_verts = points[used_faces]
        # faces_verts = points[self._surface_mesh_faces]
        bary_coords = self.surface_triangle_bary_coords#[None]
        surface_mesh = Meshes(
            verts=[points.to(self.device)],   
            faces=[self._surface_mesh_faces.to(self.device)],
            textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            # verts_normals=[verts_normals.to(rc.device)],
            )
        # 对应面的法向量
        faces_normals = surface_mesh.faces_normals_list()[0]
        faces_normals = faces_normals[self._xyz_idx]
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        used_weight = torch.clamp(self._xyz_weight, 0.0, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum
        
        points = faces_verts * used_weight[:, None]
        points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        # Compute initial rotation matrices of faces
        R_0 = torch.nn.functional.normalize(faces_normals, dim=-1)
        base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
        
        initR = torch.cat([
            R_0[..., None],
            base_R_1[..., None],
            base_R_2[..., None]
            ],
            dim=-1).view(-1, 3, 3)
        # R_rotate = self.quaternion_to_matrix(quat.view([-1, 4]))  # 
        # R_rotate = R_rotate.view(-1, 3, 3)#
        
        # R = torch.matmul(R_rotate, initR)
        R = initR
        # R = torch.cat([
        #     R_1[..., None],
        #     R_2[..., None],
        #     R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
        #     ],
        #     dim=-1).view(-1, 3, 3)
        quaternions = matrix_to_quaternion(R)
    
        return quaternions
    
    def get_posed_quaternions_scale2D(self, points, quat, sca):
        if not self.editable:
            raise ValueError("The model is not editable. Call make_editable() first.")
        points = points.squeeze(0)
        reference_verts = points.clone().detach()[self._surface_mesh_faces]
        # reference_normals = self._reference_normals
        # NOTE: 选择每个高斯点对应的面
        used_faces = self._surface_mesh_faces[self._xyz_idx]
        faces_verts = points[used_faces]
        # faces_verts = points[self._surface_mesh_faces]
        bary_coords = self.surface_triangle_bary_coords#[None]
        surface_mesh = Meshes(
            verts=[points.to(self.device)],   
            faces=[self._surface_mesh_faces.to(self.device)],
            textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            # verts_normals=[verts_normals.to(rc.device)],
            )
        # 对应面的法向量
        faces_normals = surface_mesh.faces_normals_list()[0]
        faces_normals = faces_normals[self._xyz_idx]
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        used_weight = torch.clamp(self._xyz_weight, 0.0, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum
        
        points = faces_verts * used_weight[:, None]
        points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        # Compute initial rotation matrices of faces
        R_0 = torch.nn.functional.normalize(faces_normals, dim=-1)
        base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
        
        initR = torch.cat([
            R_0[..., None],
            base_R_1[..., None],
            base_R_2[..., None]
            ],
            dim=-1).view(-1, 3, 3)
        # R_rotate = self.quaternion_to_matrix(quat.view([-1, 4]))  # 
        # R_rotate = R_rotate.view(-1, 3, 3)#
        
        # R = torch.matmul(R_rotate, initR)
        complex_numbers = torch.nn.functional.normalize(quat, dim=-1).view(-1, 2)
        # We apply the adjustment to the complex numbers
        # R_1 = complex_numbers[..., 0:1] * base_R_1[:, None] + complex_numbers[..., 1:2] * base_R_2[:, None]
        # R_2 = -complex_numbers[..., 1:2] * base_R_1[:, None] + complex_numbers[..., 0:1] * base_R_2[:, None]
        R_1 = complex_numbers[..., 0:1] * base_R_1 + complex_numbers[..., 1:2] * base_R_2
        R_2 = -complex_numbers[..., 1:2] * base_R_1 + complex_numbers[..., 0:1] * base_R_2

        # We concatenate the three vectors to get the rotation matrix, and compute the final quaternion
        # R = torch.cat([
        #     R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
        #     R_1[..., None],
        #     R_2[..., None]
        #     ],
        #     dim=-1).view(-1, 3, 3)
        # R = torch.cat([
        #     R_1[..., None],
        #     R_2[..., None],
        #     R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
        #     ],
        #     dim=-1).view(-1, 3, 3)
        R = torch.cat([
            R_0[..., None].clone(),
            R_1[..., None],
            R_2[..., None]
            ],
            dim=-1).view(-1, 3, 3)
        quaternions = matrix_to_quaternion(R)
        
        # =====Adjust scales to the current deformation=====
        # plane_scales = self.scale_activation(sca)
        plane_scales = sca
        faces_centers = faces_verts.mean(dim=1, keepdim=True)
        # self.reference_scaling_factor = (faces_verts - faces_verts.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=-1, keepdim=True)
        #scaling_factor = (faces_verts - faces_centers).norm(dim=-1).mean(dim=-1, keepdim=True) / self.reference_scaling_factor[used_faces]
        scaling_factor = (faces_verts - faces_centers).norm(dim=-1).mean(dim=-1, keepdim=True)
        # plane_scales = plane_scales * scaling_factor[:, None].expand(-1, self.n_gaussians_per_surface_triangle, -1).reshape(-1, 1)

        plane_scales = plane_scales * scaling_factor

        scales = torch.cat([
                self.surface_mesh_thickness * torch.ones(len(plane_scales), 1, device=self.device), 
                plane_scales,
                ], dim=-1)
        # scales = torch.cat([
        #         plane_scales,
        #         self.surface_mesh_thickness * torch.ones(len(self._scales), 1, device=self.device), 
        #         ], dim=-1)
        return quaternions, scales
        
    def get_edited_points_subdivide(self, points, faces):
        """对mesh进行细化后重新得到顶点信息

        Args:
            points (tensor): shape: [N, 3]
            faces (tensor): 细化后模型的面 shape[F, 3]

        Returns:
            tensor: shape: [n_faces * n_gaussians_per_face, 3]
        """

        # faces_verts = points[faces]  #batch, n_faces, 3, n_coords
        
        used_faces = self.new_faces[self._xyz_idx]
        faces_verts = points[used_faces]
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  #batch, n_faces, n_gaussians_per_face, 3, n_coords
        
        used_weight = torch.clamp(self._xyz_weight, 0.0, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum
        
        points = faces_verts * used_weight[:, None]
        
        points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        return points.reshape(-1, 3)  #batch, n_faces * n_gaussians_per_face, n_coords
    
    def get_edited_quaternions_and_scales_with_points(self, points, quat, sca):
        if not self.editable:
            raise ValueError("The model is not editable. Call make_editable() first.")
        points = points.squeeze(0)
        reference_verts = points.clone().detach()[self._surface_mesh_faces]
        # reference_normals = self._reference_normals
        # NOTE: 选择每个高斯点对应的面
        used_faces = self._surface_mesh_faces[self._xyz_idx]
        faces_verts = points[used_faces]
        # faces_verts = points[self._surface_mesh_faces]
        bary_coords = self.surface_triangle_bary_coords#[None]
        surface_mesh = Meshes(
            verts=[points.to(self.device)],   
            faces=[self._surface_mesh_faces.to(self.device)],
            textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            # verts_normals=[verts_normals.to(rc.device)],
            )
        # 对应面的法向量
        faces_normals = surface_mesh.faces_normals_list()[0]
        faces_normals = faces_normals[self._xyz_idx]
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        used_weight = torch.clamp(self._xyz_weight, 0.0, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum
        
        points = faces_verts * used_weight[:, None]
        points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        # Compute initial rotation matrices of faces
        R_0 = torch.nn.functional.normalize(faces_normals, dim=-1)
        base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
        
       
        
        # We compute the complex number representing the learned 2D rotation
        # complex_numbers = torch.nn.functional.normalize(quat, dim=-1).view(len(self._surface_mesh_faces), self.n_gaussians_per_surface_triangle, 2)
        complex_numbers = torch.nn.functional.normalize(quat, dim=-1).view(-1, 2)
        
        # We apply the adjustment to the complex numbers
        # R_1 = complex_numbers[..., 0:1] * base_R_1[:, None] + complex_numbers[..., 1:2] * base_R_2[:, None]
        # R_2 = -complex_numbers[..., 1:2] * base_R_1[:, None] + complex_numbers[..., 0:1] * base_R_2[:, None]
        R_1 = complex_numbers[..., 0:1] * base_R_1 + complex_numbers[..., 1:2] * base_R_2
        R_2 = -complex_numbers[..., 1:2] * base_R_1 + complex_numbers[..., 0:1] * base_R_2

        # We concatenate the three vectors to get the rotation matrix, and compute the final quaternion
        # R = torch.cat([
        #     R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
        #     R_1[..., None],
        #     R_2[..., None]
        #     ],
        #     dim=-1).view(-1, 3, 3)
        R = torch.cat([
            R_0[..., None].clone(),
            R_1[..., None],
            R_2[..., None]
            ],
            dim=-1).view(-1, 3, 3)
        # R = torch.cat([
        #     R_1[..., None],
        #     R_2[..., None],
        #     R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
        #     ],
        #     dim=-1).view(-1, 3, 3)
        quaternions = matrix_to_quaternion(R)
        
        quaternions = torch.nn.functional.normalize(quaternions, dim=-1)
        
        # =====Adjust scales to the current deformation=====
        # plane_scales = self.scale_activation(sca)
        plane_scales = sca
        faces_centers = faces_verts.mean(dim=1, keepdim=True)
        scaling_factor = (faces_verts - faces_centers).norm(dim=-1).mean(dim=-1, keepdim=True) / self.reference_scaling_factor[self._xyz_idx]
        # plane_scales = plane_scales * scaling_factor[:, None].expand(-1, self.n_gaussians_per_surface_triangle, -1).reshape(-1, 1)
        
        #plane_scales = plane_scales * scaling_factor.reshape(-1, 1)
        scales = torch.cat([
                self.surface_mesh_thickness * torch.ones(len(plane_scales), 1, device=self.device), 
                plane_scales,
                ], dim=-1)
        return quaternions, scales
    
    def batch_rodrigues(self, rot_vecs, epsilon=1e-8, dtype=torch.float32):
        ''' Calculates the rotation matrices for a batch of rotation vectors
            Parameters
            ----------
            rot_vecs: torch.tensor Nx3
                array of N axis-angle vectors
            Returns
            -------
            R: torch.tensor Nx3x3
                The rotation matrices for the given axis-angle parameters
        '''

        batch_size = rot_vecs.shape[0]
        device = rot_vecs.device

        angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
        rot_dir = rot_vecs / angle

        cos = torch.unsqueeze(torch.cos(angle), dim=1)
        sin = torch.unsqueeze(torch.sin(angle), dim=1)

        # Bx1 arrays
        rx, ry, rz = torch.split(rot_dir, 1, dim=1)
        K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

        zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
            .view((batch_size, 3, 3))

        ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
        #rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)

        t = torch.bmm(rot_dir.unsqueeze(2), rot_dir.unsqueeze(1))
        rot_mat = cos * ident + (1 - cos) * t + sin * K
        return rot_mat
        
    def get_posed_quaternions_and_scales(self, points, quat, sca):
        if not self.editable:
            raise ValueError("The model is not editable. Call make_editable() first.")
        points = points.squeeze(0)
        reference_verts = points.clone().detach()[self._surface_mesh_faces]
        # reference_normals = self._reference_normals
        # NOTE: 选择每个高斯点对应的面
        used_faces = self._surface_mesh_faces[self._xyz_idx]
        faces_verts = points[used_faces]
        # faces_verts = points[self._surface_mesh_faces]
        bary_coords = self.surface_triangle_bary_coords#[None]
        surface_mesh = Meshes(
            verts=[points.to(self.device)],   
            faces=[self._surface_mesh_faces.to(self.device)],
            textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            # verts_normals=[verts_normals.to(rc.device)],
            )
        # 对应面的法向量
        faces_normals = surface_mesh.faces_normals_list()[0]
        faces_normals = faces_normals[self._xyz_idx]
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        used_weight = torch.clamp(self._xyz_weight, 0.0, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum
        
        points = faces_verts * used_weight[:, None]
        points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        # Compute initial rotation matrices of faces
        R_0 = torch.nn.functional.normalize(faces_normals, dim=-1)
        base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
        
        initR = torch.cat([
            R_0[..., None],        
            base_R_1[..., None],
            base_R_2[..., None]
            ],
            dim=-1).view(-1, 3, 3)
        R_rotate = self.batch_rodrigues(quat.view([-1, 3]))  # 
        R_rotate = R_rotate.view(-1, 3, 3)#
        
        R = torch.matmul(R_rotate, initR)
       
        # R = torch.cat([
        #     R_1[..., None],
        #     R_2[..., None],
        #     R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
        #     ],
        #     dim=-1).view(-1, 3, 3)
        quaternions = matrix_to_quaternion(R) #torch.rand(R_rotate.shape[0],4).to(self.device)
        
        quaternions = torch.nn.functional.normalize(quaternions, dim=-1)
        
        # =====Adjust scales to the current deformation=====
        # plane_scales = self.scale_activation(sca)
        plane_scales = sca
        faces_centers = faces_verts.mean(dim=1, keepdim=True)
        # self.reference_scaling_factor = (faces_verts - faces_verts.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=-1, keepdim=True)
        #scaling_factor = (faces_verts - faces_centers).norm(dim=-1).mean(dim=-1, keepdim=True) / self.reference_scaling_factor[used_faces]
        scaling_factor = (faces_verts - faces_centers).norm(dim=-1).mean(dim=-1, keepdim=True)
        # plane_scales = plane_scales * scaling_factor[:, None].expand(-1, self.n_gaussians_per_surface_triangle, -1).reshape(-1, 1)

        plane_scales = plane_scales# * scaling_factor * 2
        scales = plane_scales
        # scales = torch.cat([
                # self.surface_mesh_thickness * torch.ones(len(plane_scales), 1, device=self.device), 
                # plane_scales,
                # ], dim=-1)
        return quaternions, scales
    
    def get_normals(self, canpoints):
        faces_normals_batch = []
        for bidx in range(0, canpoints.shape[0]):
            surface_mesh = Meshes(
                verts=[canpoints[bidx].to(self.device)],   
                faces=[self._surface_mesh_faces.to(self.device)],
                textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
                # verts_normals=[verts_normals.to(rc.device)],
                )
            # 对应面的法向量
            faces_normals = surface_mesh.faces_normals_list()[0]
            faces_normals = faces_normals[self._xyz_idx]
            faces_normals_batch += [faces_normals[None]]
        faces_normals_batch = torch.cat(faces_normals_batch, 0)  
        return faces_normals_batch       
        
    def get_posed_quaternions_and_scales_normals(self, gs_R, canmeanpoints, canpoints, gspoints, points, quat, sca):
        if not self.editable:
            raise ValueError("The model is not editable. Call make_editable() first.")
        points = points.squeeze(0)
        reference_verts = points.clone().detach()[self._surface_mesh_faces]
        # reference_normals = self._reference_normalsfaces_normals, 
        # NOTE: 选择每个高斯点对应的面
        used_faces = self._surface_mesh_faces[self._xyz_idx]
        faces_verts = points[used_faces]
        # faces_verts = points[self._surface_mesh_faces]
        bary_coords = self.surface_triangle_bary_coords#[None]
        surface_mesh = Meshes(
            verts=[canpoints.to(self.device)],   
            faces=[self._surface_mesh_faces.to(self.device)],
            textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            # verts_normals=[verts_normals.to(rc.device)],
            )
        # 对应面的法向量
        faces_normals = surface_mesh.faces_normals_list()[0]
        faces_normals = faces_normals[self._xyz_idx]
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        used_weight = torch.clamp(self._xyz_weight, 0.0, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum
        
        points = faces_verts * used_weight[:, None]
        points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        # Compute initial rotation matrices of faces
        R_0 = torch.nn.functional.normalize(faces_normals, dim=-1)
        base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
        
        initR = torch.cat([
            R_0[..., None],        
            base_R_1[..., None],
            base_R_2[..., None]
            ],
            dim=-1).view(-1, 3, 3)
        R_rotate = self.batch_rodrigues(quat.view([-1, 3]))  # 
        R_rotate = R_rotate.view(-1, 3, 3)#
        
        # complex_numbers = torch.nn.functional.normalize(quat, dim=-1).view(-1, 2)
        # R_1 = complex_numbers[..., 0:1] * base_R_1 + complex_numbers[..., 1:2] * base_R_2
        # R_2 = -complex_numbers[..., 1:2] * base_R_1 + complex_numbers[..., 0:1] * base_R_2

        # R = torch.cat([
            # R_0[..., None].clone(),
            # R_1[..., None],
            # R_2[..., None]
            # ],
            # dim=-1).view(-1, 3, 3)
            
        R = torch.matmul(R_rotate, initR)
        R = torch.matmul(gs_R, R)
        
        gs_normals = torch.matmul(R_rotate,faces_normals[...,None])
        gs_normals = torch.matmul(gs_R,gs_normals)
        gs_normals = gs_normals.squeeze(-1)
        
        # cansurface_mesh = Meshes(
            # verts=[canpoints.to(self.device)],   
            # faces=[self._surface_mesh_faces.to(self.device)],
            # textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            # # verts_normals=[verts_normals.to(rc.device)],
            # )
        # # 对应面的法向量
        # canfaces_normals = cansurface_mesh.faces_normals_list()[0]
        # cangs_normals = canfaces_normals[self._xyz_idx]
        
        # cangs_normals = torch.matmul(R_rotate,faces_normals[...,None])#faces_normals[...,None]#
        # cangs_normals = cangs_normals.squeeze(-1)
        
        # R = torch.cat([
        #     R_1[..., None],
        #     R_2[..., None],
        #     R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
        #     ],
        #     dim=-1).view(-1, 3, 3)
        quaternions = matrix_to_quaternion(R) #torch.rand(R_rotate.shape[0],4).to(self.device)
        
        quaternions = torch.nn.functional.normalize(quaternions, dim=-1)
        
        # =====Adjust scales to the current deformation=====
        # plane_scales = self.scale_activation(sca)
        plane_scales = sca
        canmeanfaces_verts = canmeanpoints[used_faces]
        faces_centers = canmeanfaces_verts.mean(dim=1, keepdim=True)
        # self.reference_scaling_factor = (faces_verts - faces_verts.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=-1, keepdim=True)
        #scaling_factor = (faces_verts - faces_centers).norm(dim=-1).mean(dim=-1, keepdim=True) / self.reference_scaling_factor[used_faces]
        scaling_factor = (canmeanfaces_verts - faces_centers).norm(dim=-1).mean(dim=-1, keepdim=True)
        # plane_scales = plane_scales * scaling_factor[:, None].expand(-1, self.n_gaussians_per_surface_triangle, -1).reshape(-1, 1)
        
        # facenum = self._surface_mesh_faces.shape[0]
        # gspoints_de = gspoints.clone().detach()
        # gspoints_de = gspoints_de.reshape([-1,facenum,6,3])
        # gspoints_centers = gspoints_de.mean(dim=-2, keepdim=True)
        # scaling_factor = (gspoints_de - gspoints_centers).norm(dim=-1).mean(dim=-1, keepdim=True)
        # scaling_factor = scaling_factor.repeat([1,6,1])
 
        gspoints_de = gspoints.clone().detach()
        knn = knn_points(gspoints_de[None], gspoints_de[None], K=4)
        use_sqrt = True
        use_mean = True
        initial_radius_normalization = 1.  # 1., 0.1
        if use_sqrt:
            knn_dists = torch.sqrt(knn.dists[..., 1:])
        else:
            knn_dists = knn.dists[..., 1:]
        if use_mean:
            radiuses = knn_dists.mean(-1, keepdim=True).clamp_min(0.0000001) * initial_radius_normalization
        else:
            radiuses = knn_dists.min(-1, keepdim=True)[0].clamp_min(0.0000001) * initial_radius_normalization

        #plane_scales = plane_scales * scaling_factor * 50
        plane_scales = plane_scales * radiuses[0] * 150
        
        #scales = plane_scales
        scales = torch.cat([
                self.surface_mesh_thickness * torch.ones(len(plane_scales), 1, device=self.device), 
                plane_scales,
                ], dim=-1)
        return quaternions, scales, gs_normals
        
    def get_posed_quaternions_and_scales2(self, canpoints, gs_A, quat, sca):
        if not self.editable:
            raise ValueError("The model is not editable. Call make_editable() first.")
        
        used_faces = self._surface_mesh_faces[self._xyz_idx]
        faces_verts = canpoints[used_faces]
        # faces_verts = points[self._surface_mesh_faces]
        bary_coords = self.surface_triangle_bary_coords#[None]
        surface_mesh = Meshes(
            verts=[canpoints.to(self.device)],   
            faces=[self._surface_mesh_faces.to(self.device)],
            textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            # verts_normals=[verts_normals.to(rc.device)],
            )
        # 对应面的法向量
        faces_normals = surface_mesh.faces_normals_list()[0]
        faces_normals = faces_normals[self._xyz_idx]
            
        # Compute initial rotation matrices of faces
        R_0 = torch.nn.functional.normalize(faces_normals, dim=-1)
        base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
        
        initR = torch.cat([
            R_0[..., None],
            base_R_1[..., None],
            base_R_2[..., None]
            ],
            dim=-1).view(-1, 3, 3)
            
        R_rotate = self.batch_rodrigues(quat.view([-1, 3]))  # 
        R_rotate = R_rotate.view(-1, 3, 3)#
        
        R1 = torch.matmul(R_rotate, initR)
        R = torch.matmul(gs_A, R1)
       
        # R = torch.cat([
        #     R_1[..., None],
        #     R_2[..., None],
        #     R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
        #     ],
        #     dim=-1).view(-1, 3, 3)
        quaternions = matrix_to_quaternion(R)
        
        # =====Adjust scales to the current deformation=====
        # plane_scales = self.scale_activation(sca)
        plane_scales = sca
        #used_faces = self._surface_mesh_faces[self._xyz_idx]
        #faces_verts = points[used_faces]
        #faces_centers = faces_verts.mean(dim=1, keepdim=True)
        #scaling_factor = (faces_verts - faces_centers).norm(dim=-1).mean(dim=-1, keepdim=True) / self.reference_scaling_factor[self._xyz_idx]
        # plane_scales = plane_scales * scaling_factor[:, None].expand(-1, self.n_gaussians_per_surface_triangle, -1).reshape(-1, 1)
        
        #plane_scales = plane_scales * scaling_factor.reshape(-1, 1)
        scales = plane_scales
        # scales = torch.cat([
                # self.surface_mesh_thickness * torch.ones(len(plane_scales), 1, device=self.device), 
                # plane_scales,
                # ], dim=-1)
        return quaternions, scales, quat
        
    # DEBUG: 此处还未修改！！
    def get_edited_quaternions_and_scales_with_points_subdivide(self, points, faces, quat, sca):
        if not self.editable:
            raise ValueError("The model is not editable. Call make_editable() first.")
        points = points.squeeze(0)
        used_faces = self.new_faces[self._xyz_idx]
        faces_verts = points[used_faces]
        # faces_verts = points[faces]
        bary_coords = self.surface_triangle_bary_coords#[None]
        surface_mesh = Meshes(
            verts=[points.to(self.device)],   
            faces=[faces.to(self.device)],
            # textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            # verts_normals=[verts_normals.to(rc.device)],
            )
        faces_normals = surface_mesh.faces_normals_list()[0]
        faces_normals = faces_normals[used_faces]
        
        # Then compute the points using barycenter coordinates in the surface triangles
        # points = faces_verts[:, None] * bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        # points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        used_weight = torch.clamp(self._xyz_weight, 0.0, 1.0)
        weight_sum = torch.sum(used_weight, dim=-1, keepdim=True)
        used_weight = used_weight / weight_sum
        
        points = faces_verts * used_weight[:, None]
        points = points.sum(dim=-2)
        
        # Compute initial rotation matrices of faces
        R_0 = torch.nn.functional.normalize(faces_normals, dim=-1)
        base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
        
       
        
        # We compute the complex number representing the learned 2D rotation
        # complex_numbers = torch.nn.functional.normalize(quat, dim=-1).view(len(faces), self.n_gaussians_per_surface_triangle, 2)
        complex_numbers = torch.nn.functional.normalize(quat, dim=-1).view(-1, 2)
        # We apply the adjustment to the complex numbers
        # R_1 = complex_numbers[..., 0:1] * base_R_1[:, None] + complex_numbers[..., 1:2] * base_R_2[:, None]
        # R_2 = -complex_numbers[..., 1:2] * base_R_1[:, None] + complex_numbers[..., 0:1] * base_R_2[:, None]
        R_1 = complex_numbers[..., 0:1] * base_R_1 + complex_numbers[..., 1:2] * base_R_2
        R_2 = -complex_numbers[..., 1:2] * base_R_1 + complex_numbers[..., 0:1] * base_R_2

        # We concatenate the three vectors to get the rotation matrix, and compute the final quaternion
        # R = torch.cat([
        #     R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
        #     R_1[..., None],
        #     R_2[..., None]
        #     ],
        #     dim=-1).view(-1, 3, 3)
        # R = torch.cat([
        #     R_1[..., None],
        #     R_2[..., None],
        #     R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
        #     ],
        #     dim=-1).view(-1, 3, 3)
        R = torch.cat([
            R_0[..., None].clone(),
            R_1[..., None],
            R_2[..., None]
            ],
            dim=-1).view(-1, 3, 3)
        quaternions = matrix_to_quaternion(R)
        
        # =====Adjust scales to the current deformation=====
        # plane_scales = self.scale_activation(sca)
        plane_scales = sca
        faces_centers = faces_verts.mean(dim=1, keepdim=True)
        # self.reference_scaling_factor = (faces_verts - faces_verts.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=-1, keepdim=True)
        scaling_factor = (faces_verts - faces_centers).norm(dim=-1).mean(dim=-1, keepdim=True) / self.reference_scaling_factor[used_faces]
        # plane_scales = plane_scales * scaling_factor[:, None].expand(-1, self.n_gaussians_per_surface_triangle, -1).reshape(-1, 1)
        plane_scales = plane_scales * scaling_factor.reshape(-1, 1)
        scales = torch.cat([
                self.surface_mesh_thickness * torch.ones(len(plane_scales), 1, device=self.device), 
                plane_scales,
                ], dim=-1)
        # scales = torch.cat([
        #         plane_scales,
        #         self.surface_mesh_thickness * torch.ones(len(self._scales), 1, device=self.device), 
        #         ], dim=-1)
        return quaternions, scales
    
    def get_edited_quaternions_and_scales(self, verbose=False):
        if not self.editable:
            raise ValueError("The model is not editable. Call make_editable() first.")
        
        reference_verts = self._reference_points[self._surface_mesh_faces]
        reference_normals = self._reference_normals
        faces_verts = self._points[self._surface_mesh_faces]
        bary_coords = self.surface_triangle_bary_coords#[None]
        faces_normals = self.surface_mesh.faces_normals_list()[0]
        
        # Then compute the points using barycenter coordinates in the surface triangles
        points = faces_verts[:, None] * bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        # Compute initial rotation matrices of faces
        R_0 = torch.nn.functional.normalize(faces_normals, dim=-1)
        base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
        
        # =====Adjust rotation to the current deformation=====
        reference_base = torch.nn.functional.normalize(reference_verts[:, 0:1] - reference_verts[:, 1:2], dim=-1)  # n_faces, 1, 3
        reference_axis = torch.nn.functional.normalize(reference_verts - reference_verts.mean(dim=-2, keepdim=True), dim=-1)  # n_faces, 3, 3
        reference_axis[:, 2] = -reference_axis[:, 2]
        
        faces_base = torch.nn.functional.normalize(faces_verts[:, 0:1] - faces_verts[:, 1:2], dim=-1)  # n_faces, 1, 3
        faces_axis = torch.nn.functional.normalize(faces_verts - faces_verts.mean(dim=-2, keepdim=True), dim=-1)  # n_faces, 3, 3
        faces_axis[:, 2] = -faces_axis[:, 2]
        
        # Compute angle between the reference and the current sides
        reference_angles = torch.arccos((reference_axis * reference_base).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.))  # n_faces, 3, 1
        faces_angles = torch.arccos((faces_axis * faces_base).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.))  # n_faces, 3, 1
        angles = faces_angles - reference_angles  # n_faces, 3, 1
        point_angles = (angles[:, None] * bary_coords[None]).sum(dim=-2)  # n_faces, n_gaussians_per_face, 1

        # Compute the complex number that will adjust the rotation the gaussians
        point_adjust_complex = torch.cat([torch.cos(point_angles), torch.sin(point_angles)], dim=-1)
        
        # We compute the complex number representing the learned 2D rotation
        complex_numbers = torch.nn.functional.normalize(self._quaternions, dim=-1).view(len(self._surface_mesh_faces), self.n_gaussians_per_surface_triangle, 2)
        
        # We apply the adjustment to the complex numbers
        x, y = complex_numbers[..., 0].clone(), complex_numbers[..., 1].clone()
        a, b = point_adjust_complex[..., 0], point_adjust_complex[..., 1]
        complex_numbers[..., 0] = x * a - y * b
        complex_numbers[..., 1] = x * b + y * a
        
        # We now apply the 2D rotation to the base quaternion
        R_1 = complex_numbers[..., 0:1] * base_R_1[:, None] + complex_numbers[..., 1:2] * base_R_2[:, None]  # n_faces, n_gaussians_per_face, 3
        R_2 = -complex_numbers[..., 1:2] * base_R_1[:, None] + complex_numbers[..., 0:1] * base_R_2[:, None]  # n_faces, n_gaussians_per_face, 3

        # We concatenate the three vectors to get the rotation matrix, and compute the final quaternion
        R = torch.cat([
            R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
            R_1[..., None],
            R_2[..., None]
            ],
            dim=-1)
        quaternions = matrix_to_quaternion(R)
        
        # =====Adjust scales to the current deformation=====
        all_faces_axis = faces_verts.mean(dim=-2, keepdim=True) - faces_verts  # Shape (n_faces, 3, 3)
        all_faces_axis_norm = torch.norm(all_faces_axis, dim=-1, keepdim=True)  # Shape (n_faces, 3, 1)
        all_faces_axis = torch_normalize(all_faces_axis, dim=-1)  # Shape (n_faces, 3, 3)
        all_faces_orthos = torch.cross(all_faces_axis, faces_normals[:, None], dim=-1)  # Shape (n_faces, 3, 3)
        
        all_reference_axis = reference_verts.mean(dim=-2, keepdim=True) - reference_verts  # Shape (n_faces, 3, 3)
        all_reference_axis_norm = torch.norm(all_reference_axis, dim=-1, keepdim=True)  # Shape (n_faces, 3, 1)
        all_reference_axis = torch_normalize(all_reference_axis, dim=-1)  # Shape (n_faces, 3, 3)
        
        axis_proj_1 = (R_1[..., None, :] * all_faces_axis[:, None]).sum(dim=-1, keepdim=True)  # Shape (n_faces, n_gaussians_per_face, 3, 1)
        ortho_proj_1 = (R_1[..., None, :] * all_faces_orthos[:, None]).sum(dim=-1, keepdim=True)  # Shape (n_faces, n_gaussians_per_face, 3, 1)
        side_proj_2 = (R_2[..., None, :] * all_faces_axis[:, None]).sum(dim=-1, keepdim=True)  # Shape (n_faces, n_gaussians_per_face, 3, 1)
        ortho_proj_2 = (R_2[..., None, :] * all_faces_orthos[:, None]).sum(dim=-1, keepdim=True)  # Shape (n_faces, n_gaussians_per_face, 3, 1)
        
        scaling_1 = torch.sqrt(  (axis_proj_1 * all_faces_axis_norm[:, None] / all_reference_axis_norm[:, None])**2 + ortho_proj_1**2  )  # Shape (n_faces, n_gaussians_per_face, 3, 1)
        scaling_2 = torch.sqrt(  (side_proj_2 * all_faces_axis_norm[:, None] / all_reference_axis_norm[:, None])**2 + ortho_proj_2**2  )  # Shape (n_faces, n_gaussians_per_face, 3, 1)
        
        scaling_1 = (scaling_1 * bary_coords[None]).sum(dim=-2)  # Shape (n_faces, n_gaussians_per_face, 1)
        scaling_2 = (scaling_2 * bary_coords[None]).sum(dim=-2)  # Shape (n_faces, n_gaussians_per_face, 1)
        
        plane_scales = self.scale_activation(self._scales).view(-1, self.n_gaussians_per_surface_triangle, 2)
        plane_scales[..., 0:1] = plane_scales[..., 0:1] * scaling_1
        plane_scales[..., 1:2] = plane_scales[..., 1:2] * scaling_2
        scales = torch.cat([
            self.surface_mesh_thickness * torch.ones(len(self._scales), 1, device=self.device), 
            plane_scales.view(-1, 2),
            ], dim=-1)
        
        if verbose:
            with torch.no_grad():
                print("Computed edited quaternions and scales.")
                print("Mean - std - min - max - median")
                print("Scaling1:", scaling_1.mean().item(), scaling_1.std().item(), scaling_1.min().item(), scaling_1.max().item(), scaling_1.median().item())
                print("Scaling2:", scaling_2.mean().item(), scaling_2.std().item(), scaling_2.min().item(), scaling_2.max().item(), scaling_2.median().item())
        return quaternions, scales
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[:update_filter.shape[0]][update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
    def densify_and_prune(self, human_gs_out, max_grad, min_opacity, extent, max_screen_size, max_n_gs=None, optimizer=None):
       grads = self.xyz_gradient_accum / self.denom
       grads[grads.isnan()] = 0.0
       
       self.opacity_tmp = human_gs_out['gs_opacity']
       self.scales_tmp = human_gs_out['gs_scales']
       self.rotmat_tmp = human_gs_out['quaternions']
        
       max_n_gs = max_n_gs if max_n_gs else self._xyz_weight.shape[0] + 1
        
       if self._xyz_weight.shape[0] <= max_n_gs:
           self.densify_and_clone(grads, max_grad, extent, optimizer)
           self.densify_and_split(grads, max_grad, extent, optimizer=optimizer)

       prune_mask = (self.opacity_tmp < min_opacity).squeeze()
       if max_screen_size:
           big_points_vs = self.max_radii2D > max_screen_size
           big_points_ws = self.scales_tmp.max(dim=1).values > 0.1 * extent
           prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
       self.prune_points(prune_mask, optimizer)
       self.n_gs = self._xyz_weight.shape[0]
       torch.cuda.empty_cache()
    
    def densify_and_clone(self, grads, grad_threshold, scene_extent, optimizer):
        # Extract points that satisfy the gradient condition
        scales = self.scales_tmp
        grad_cond = torch.norm(grads, dim=-1) >= grad_threshold
        scale_cond = torch.max(scales, dim=1).values <= 0.01*scene_extent
        
        selected_pts_mask = torch.where(grad_cond, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, scale_cond)
        
        new_xyz_weight = self._xyz_weight[selected_pts_mask]
        new_xyz_idx = self._xyz_idx[selected_pts_mask]
        # new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask]
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask]
        new_scales_tmp = self.scales_tmp[selected_pts_mask]
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask]
        
        self.densification_postfix(new_xyz_weight, new_xyz_idx, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp, optimizer)
        
    def densification_postfix(self, new_xyz_weight, new_xyz_idx, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp, optimizer):
       
        tensors_dict = {
                    f"{self.optim_param_name}": new_xyz_weight
                    }
        optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict, optimizer)
        # self._xyz = optimizable_tensors["xyz"]
        self._xyz_weight = optimizable_tensors[self.optim_param_name]
        self._xyz_idx = nn.Parameter(torch.cat((self._xyz_idx, new_xyz_idx), dim=0), requires_grad=False)
        # self.scaling_multiplier = torch.cat((self.scaling_multiplier, new_scaling_multiplier), dim=0)
        self.opacity_tmp = torch.cat([self.opacity_tmp, new_opacity_tmp], dim=0)
        self.scales_tmp = torch.cat([self.scales_tmp, new_scales_tmp], dim=0)
        self.rotmat_tmp = torch.cat([self.rotmat_tmp, new_rotmat_tmp], dim=0)
        
        self.xyz_gradient_accum = torch.zeros((self._xyz_weight.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz_weight.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz_weight.shape[0]), device="cuda")

    def cat_tensors_to_optimizer(self, tensors_dict, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"] != self.optim_param_name:
                continue
            #if group["name"] in ["xyz_weight"]:

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)#torch.zeros_like(extension_tensor)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, optimizer = None):
        n_init_points = self._xyz_weight.shape[0]
        scales = self.scales_tmp
        rotation = self.rotmat_tmp
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scales, dim=1).values > 0.01*scene_extent)
        # filter elongated gaussians
        med = scales.median(dim=1, keepdim=True).values 
        stdmed_mask = (((scales - med) / med).squeeze(-1) >= 1.0).any(dim=-1)
        selected_pts_mask = torch.logical_and(selected_pts_mask, stdmed_mask)
        
        stds = scales[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device=self.device)
        # samples = torch.normal(mean=means, std=torch.relu(stds))
        # rots = rotation[selected_pts_mask].repeat(N,1,1)
        # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_xyz = self._xyz_weight[selected_pts_mask].repeat(N, 1) + torch.normal(mean=means, std=0.1*torch.ones_like(means, device=self.device))
        # new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask].repeat(N,1) / (0.8*N)
        new_xyz_idx = self._xyz_idx[selected_pts_mask].repeat(N)
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask].repeat(N,1)
        new_scales_tmp = self.scales_tmp[selected_pts_mask].repeat(N,1)
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask].repeat(N,1)#
        
        self.densification_postfix(new_xyz, new_xyz_idx, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp, optimizer)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, optimizer)
    
    def prune_points(self, mask, optimizer):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, optimizer)

        # self._xyz = optimizable_tensors["xyz"]
        #self._points = optimizable_tensors["points"]
        self._xyz_weight = optimizable_tensors[self.optim_param_name]
        # self.scaling_multiplier = self.scaling_multiplier[valid_points_mask]
        self._xyz_idx = nn.Parameter(self._xyz_idx[valid_points_mask], requires_grad=False)
        self.scales_tmp = self.scales_tmp[valid_points_mask]
        self.opacity_tmp = self.opacity_tmp[valid_points_mask]
        self.rotmat_tmp = self.rotmat_tmp[valid_points_mask]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    def _prune_optimizer(self, mask, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"] != self.optim_param_name:
                continue
            #if group["name"] in ["xyz_weight"]:#"points",
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors