import math
import torch
#from lib.config import cfg
# from .nerf_net_utils import *
# from ... import embedder
import os
import numpy as np
#import neural_renderer as nr
import cv2
import scipy.io as scio
import trimesh
import torch.nn.functional as F

from pytorch3d.renderer.cameras import PerspectiveCameras, OrthographicCameras
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,SoftPhongShader
)
import pytorch3d.structures as struct
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops import subdivide_meshes
from pytorch3d.structures import Meshes,Pointclouds,join_meshes_as_batch
from lib.networks.gs_network_snug import Network as GNS
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from plyfile import PlyData, PlyElement

# from lib.utils.rotations import matrix_to_quaternion, rotation_6d_to_matrix
from lib.utils.loss import l1_loss, ssim
from lib.datasets.utils import get_projection_matrix,get_rigid_transformation
from smpl_utils import init_smpl, get_J, get_shape_pose, batch_rodrigues
import smplx
from smplx.lbs import transform_mat, blend_shapes

def set_pytorch3d_intrinsic_matrix(batchK, H, W):
        # K = batchK[0]
        # fx = -K[0, 0] * 2.0 / W
        # fy = -K[1, 1] * 2.0 / H
        # px = -(K[0, 2] - W / 2.0) * 2.0 / W
        # py = -(K[1, 2] - H / 2.0) * 2.0 / H
        # pytorch3d_K = torch.Tensor([
            # [fx, 0, px, 0],
            # [0, fy, py, 0],
            # [0, 0, 0, 1],
            # [0, 0, 1, 0],
        # ]).to(K.device)
        # return pytorch3d_K[None,...].repeat(batchK.shape[0],1,1)
        K = batchK
        fx = -K[:,0, 0] * 2.0 / W
        fy = -K[:,1, 1] * 2.0 / H
        px = -(K[:,0, 2] - W / 2.0) * 2.0 / W
        py = -(K[:,1, 2] - H / 2.0) * 2.0 / H
        pytorch3d_K = torch.zeros([batchK.shape[0],4,4]).to(K.device)
        pytorch3d_K[:,2,3] = 1
        pytorch3d_K[:,3,2] = 1
        pytorch3d_K[:,0,0] = fx
        pytorch3d_K[:,0,2] = px
        pytorch3d_K[:,1,1] = fy
        pytorch3d_K[:,1,2] = py
        return pytorch3d_K
        
class Renderer:
    def __init__(self, net:GNS):
        self.net = net

        #self.meshrenderer = meshrenderer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # DEBUG: 或许是有细节人体模型的三角面
        # FIXME: 确定是否有用到
        # faces_path = os.path.join(cfg.train_dataset.data_root, 'templatedeform/modeltri.txt')
        # faces = np.loadtxt(faces_path)-1
        # self.faces = torch.LongTensor(faces).to(self.device)
        # self.faces = self.faces[None, :, :]
        
        self.smpl_model = init_smpl(
            model_folder = 'smpl_models',
            model_type = 'smpl',
            gender = 'neutral',
            num_betas = 10
        )
        parents = self.smpl_model.parents#.cpu().numpy()

        self.parents = parents
        self.num_joints = parents.shape[0]
        
        data_root = '../gaussian-nerfcap/data/magdalena/magdalena2000-allviews'
        # smpl模型三角面
        # FIXME: 确定是否有用到
        npfaces = np.loadtxt(os.path.join(data_root, 'templatedeform/smpltri.txt')) - 1
        self.smplfaces = torch.LongTensor(npfaces).to(self.device)
        self.smplfaces = self.smplfaces[None, :, :]
        
        # DEBUG: 实际使用的变形后的人体模型的面
        npfaces = np.loadtxt(os.path.join(data_root, 'templatedeformT/desmpl/desmpltri.txt')) - 1
        self.desmplfaces = torch.LongTensor(npfaces).to(self.device)
        self.desmplfaces = self.desmplfaces[None, :, :]
        
        # FIXME: smpl顶点对应面的方向的索引
        #npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/smpl_vfidx.txt')) - 1
        npfaces = np.loadtxt(os.path.join(data_root, 'templatedeformT/desmpl/desmpl_vfidx.txt')) - 1        
        self.desmpl_vfidx = torch.LongTensor(npfaces).to(self.device)
        
        # FIXME: 有细节人体带服装模型的模板顶点位置
        # DEBUG: 是否有用？ 
        # templateshape_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/templateshapeT.txt')
        # templateshape = np.loadtxt(templateshape_path)
        # templateshape = templateshape
        # self.templateshape = torch.Tensor(templateshape).to(self.device)
        # self.templateshape = self.templateshape[None, :, :]
        
        # 服装面
        npfaces = np.loadtxt(os.path.join(data_root, 'templatedeformT/cloth/clothes_face.txt')) - 1
        self.clothfaces = torch.LongTensor(npfaces).to(self.device)
        self.clothfaces = self.clothfaces[None, :, :]
        
        # 水密
        npfaces = np.loadtxt(os.path.join(data_root, 'templatedeformT/cloth/clothes_watertight_face.txt')) - 1
        self.clothes_watertight_face = torch.LongTensor(npfaces).to(self.device)
        self.clothes_watertight_face = self.clothes_watertight_face[None, :, :]
        
        # 顶点对应面法向量的索引
        npfaces = np.loadtxt(os.path.join(data_root, 'templatedeformT/cloth/cloth_vfidx.txt')) - 1
        self.cloth_vfidx = torch.LongTensor(npfaces).to(self.device)
        templatesmpl_path = os.path.join(data_root,
                                         'templatedeformT/vpersonalshape.txt')  # smpldeform/vpersonalshape
        templatesmpl = np.loadtxt(templatesmpl_path)
        templatesmpl = templatesmpl
        templatesmpl = torch.Tensor(templatesmpl).to(self.device)
        self.templatesmpl = templatesmpl
        # loading template deformation graph
        templateshape_path = os.path.join(data_root, 'templatedeformT/cloth/clothes_vert.txt')
        templatecloth = np.loadtxt(templateshape_path)
        templatecloth = torch.Tensor(templatecloth).to(self.device)
        self.templatecloth = templatecloth
        cloth_ptsdist = torch.cdist(templatecloth[None,...], templatesmpl[None,...], p=2)
        cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        self.cloth_nnvidx = torch.squeeze(cloth_ptsdistmin[1], -1)  # B*P
        
        self.meshface = torch.cat([self.clothfaces, self.desmplfaces+templatecloth.shape[0]], dim=1)
        
        self.l1loss = torch.nn.L1Loss()
        
        # 固定顶点的索引
        attachidx = np.loadtxt(os.path.join(data_root, 'templatedeformT/cloth/attachidx.txt')) - 1
        attachidx = torch.LongTensor(attachidx).to(self.device)
        self.attachtag = torch.zeros_like(templatecloth[:,0])
        self.attachtag[attachidx] = 1
        
        self.submesh = subdivide_meshes.SubdivideMeshes()

    def prepare_sp_input(self, batch):
        # feature, coordinate, shape, batch size
        sp_input = {}

        # coordinate: [N, 4], batch_idx, z, y, x
        # sh = batch['coord'].shape
        # idx = [torch.full([sh[1]], i, dtype=torch.long) for i in range(sh[0])]
        # idx = torch.cat(idx).to(batch['coord'])
        # coord = batch['coord'].view(-1, sh[-1])
        # sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = out_sh.shape[0]

        # used for feature interpolation
        sp_input['bounds'] = batch['bounds']
        sp_input['R'] = batch['R']
        if cfg.train_dataset.human == '0080':
            sp_input['Rh'] = batch['Rh']
        sp_input['Th'] = batch['Th']

        # used for color function
        sp_input['latent_index'] = batch['latent_index']
        sp_input['frame_index'] = batch['frame_index']

        sp_input['vert'] = batch['vert']
        sp_input['A'] = batch['A']
        sp_input['RT'] = batch['RT']
        if "pytorch_RT" in batch.keys():
            sp_input['pytorch_RT'] = batch['pytorch_RT']
        sp_input['K'] = batch['K']
        sp_input['msk'] = batch['msk']
        sp_input['fovx'] = batch['fovx']
        sp_input['fovy'] = batch['fovy']
        sp_input['world_view_transform'] = batch['world_view_transform']
        sp_input['c2w'] = batch['c2w']
        sp_input['full_proj_transform'] = batch['full_proj_transform']
        sp_input['camera_center'] = batch['camera_center']
        sp_input['cam_intrinsics'] = batch['cam_intrinsics']
        
        sp_input['smplpose'] = batch['smplpose']
        sp_input['smplshape'] = batch['smplshape']

        return sp_input 

    def get_pixel_value(self, ray_o, ray_d, near, far,
                        sp_input, batch, coord_silhouette):
        # sampling points along camera rays, geometryzero
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)

        n_batch, n_pixel, n_sample = wpts.shape[:3]
        
        ptsdist = torch.cdist(wpts.view(n_batch,-1,3), self.deformedpersonsmpl, p=2)
        nndist = torch.squeeze(torch.min(ptsdist, 2)[0], -1)  # B*P
        ptsnearsurfacetag_smpl = torch.where(nndist > 0.02, torch.zeros_like(nndist), torch.ones_like(nndist))
        ptsdist = torch.cdist(wpts.view(n_batch,-1,3), self.deformedcloth, p=2)
        nndist = torch.squeeze(torch.min(ptsdist, 2)[0], -1)  # B*P
        ptsnearsurfacetag_cloth = torch.where(nndist > 0.02, torch.zeros_like(nndist), torch.ones_like(nndist))
        

        # viewing direction
        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)

        raw_decoder = lambda x_point, viewdir_val: self.net.calculate_density_color_clothdeformation_layer(
            x_point, viewdir_val, sp_input)

        wpts_raw_smpl, wpts_raw_cloth = self.get_density_color_layer(wpts, viewdir, raw_decoder)

        # volume rendering for wpts
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        #raw = wpts_raw.reshape(-1, n_sample, 4)
        wpts_blending = coord_silhouette.repeat(1,1,n_sample)
        wpts_blending = wpts_blending.reshape(-1, n_sample)
        raw_smpl = wpts_raw_smpl.reshape(-1, n_sample, 4)
        raw_cloth = wpts_raw_cloth.reshape(-1, n_sample, 4)


        raw_smpl[...,3] = raw_smpl[...,3]*ptsnearsurfacetag_smpl.reshape(-1, n_sample)
        raw_cloth[...,3] = raw_cloth[...,3]*ptsnearsurfacetag_cloth.reshape(-1, n_sample)

        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3) 
        
        rgb_map_full, depth_map_full, acc_map_full, weights_full, \
        rgb_map_s, depth_map_s, acc_map_s, weights_s, \
        rgb_map_d, depth_map_d, acc_map_d, weights_d, dynamicness_map = raw2outputs_blend(raw_smpl, raw_cloth, wpts_blending, z_vals, ray_d, cfg.raw_noise_std)
                
        ret = {
            'rgb_map': rgb_map_full.view(n_batch, n_pixel, -1),
            'acc_map': acc_map_full.view(n_batch, n_pixel),
            'weights': weights_full.view(n_batch, n_pixel, -1),
            'depth_map': depth_map_full.view(n_batch, n_pixel),
            'rgb_map_s': rgb_map_s.view(n_batch, n_pixel, -1),
            'rgb_map_d': rgb_map_d.view(n_batch, n_pixel, -1)
        }

        return ret
      
    def computeinterpenetrationloss_posedsmpl(self, graphdeformedverts):

        batch_size = graphdeformedverts.size(0)#self.deformedcloth.size(0)self.net.deformation_network.templateshape[None,...]
        # cloth_ptsdist = torch.cdist(graphdeformedverts, self.net.deformation_network.smplgraphdeformedverts, p=2)
        cloth_ptsdist = torch.cdist(self.deformedcloth, self.deformedpersonsmpl, p=2)
        cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        cloth_nnvidx = torch.squeeze(cloth_ptsdistmin[1], -1)  # B*P
        
        # cloth_mindist = torch.squeeze(cloth_ptsdistmin[0], -1)
        # validtag = torch.where(cloth_mindist > 0.03, torch.zeros_like(cloth_mindist), torch.ones_like(cloth_mindist))
        # validtag = validtag.view(-1).unsqueeze(-1).repeat(1,3)self.deformedcloth.view(-1,3)
        
        # _,trinormal = mesh_face_areas_normals(graphdeformedverts.view(-1,3), self.clothfaces.view(-1,3))
        # trinormal = trinormal.view(batch_size,-1,3)
        # vertnorm = trinormal[:,self.cloth_vfidx,:]
        #vertnorm = torch.gather(trinormal,1,self.cloth_vfidx)
        
        
        ptsnum = graphdeformedverts.size(1)
        templatevtnum = self.deformedpersonsmpl.size(1)
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(batch_size)]
        idx = torch.cat(idx).to(self.device)
        bwidx = cloth_nnvidx.view(-1) + idx
        #deformedpersonsmpl = self.deformedpersonsmpl.view(-1, 3)  #self.deformedcloth.view(-1,3)
        # canpersonsmpl = self.net.deformation_network.smplgraphdeformedverts.view(-1,3)
        # selectpersonsmpl = canpersonsmpl[bwidx.long(), :]
        # selectpersonsmpl_norm = self.net.deformation_network.deformedsmpl_vertnorm[bwidx.long(), :]
              
        # normaldist = (selectpersonsmpl - graphdeformedverts.view(-1,3))*selectpersonsmpl_norm
 
        # interpenetration = torch.sum(normaldist,1)
        # # interpenetration = interpenetration[(interpenetration<0.02)*(interpenetration>0)]
        # # print(interpenetration)
        # #interpenetration = torch.sum((selectdeformedpersonsmpl - graphdeformedverts.view(-1,3))*vertnorm.view(-1,3),1)
        # interpenetrationloss = torch.mean(F.relu(interpenetration),0)
        
        _,trinormal = mesh_face_areas_normals(self.deformedpersonsmpl.view(-1,3), self.desmplfaces[0])
         
        deformedposedsmpl_vertnorm = trinormal[self.desmpl_vfidx,:]

        pospersonsmpl = self.deformedpersonsmpl.view(-1,3)
        selectpospersonsmpl = pospersonsmpl[bwidx.long(), :]
        selectpospersonsmpl_norm = deformedposedsmpl_vertnorm[bwidx.long(), :]
              
        normaldist1 = (selectpospersonsmpl - self.deformedcloth.view(-1,3))*selectpospersonsmpl_norm
 
        interpenetration1 = torch.sum(normaldist1,1)
        interpenetrationloss1 = torch.mean(F.relu(interpenetration1),0)
        
        # return interpenetrationloss+interpenetrationloss1
        return interpenetrationloss1
    # def computeinterpenetrationloss_posedsmpl(self, graphdeformedverts):

    #     batch_size = graphdeformedverts.size(0)
    #     cloth_ptsdist = torch.cdist(graphdeformedverts, self.net.deformation_network.smplgraphdeformedverts, p=2)
    #     cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
    #     cloth_nnvidx = torch.squeeze(cloth_ptsdistmin[1], -1)  # B*P
        
    #     ptsnum = graphdeformedverts.size(1)
    #     templatevtnum = self.deformedpersonsmpl.size(1)
    #     idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(batch_size)]
    #     idx = torch.cat(idx).to(self.device)
    #     bwidx = cloth_nnvidx.view(-1) + idx
    #     canpersonsmpl = self.net.deformation_network.smplgraphdeformedverts.view(-1,3)
    #     selectpersonsmpl = canpersonsmpl[bwidx.long(), :]
    #     selectpersonsmpl_norm = self.net.deformation_network.deformedsmpl_vertnorm[bwidx.long(), :]
              
    #     normaldist = (selectpersonsmpl - graphdeformedverts.view(-1,3))*selectpersonsmpl_norm
 
    #     interpenetration = torch.sum(normaldist,1)
    #     interpenetrationloss = torch.mean(F.relu(interpenetration),0)
        
    #     _,trinormal = mesh_face_areas_normals(self.deformedpersonsmpl.view(-1,3), self.desmplfaces[0])
         
    #     deformedposedsmpl_vertnorm = trinormal[self.desmpl_vfidx,:]

    #     pospersonsmpl = self.deformedpersonsmpl.view(-1,3)
    #     selectpospersonsmpl = pospersonsmpl[bwidx.long(), :]
    #     selectpospersonsmpl_norm = deformedposedsmpl_vertnorm[bwidx.long(), :]
              
    #     normaldist1 = (selectpospersonsmpl - self.deformedcloth.view(-1,3))*selectpospersonsmpl_norm
 
    #     interpenetration1 = torch.sum(normaldist1,1)
    #     interpenetrationloss1 = torch.mean(F.relu(interpenetration1),0)
        
    #     return interpenetrationloss+interpenetrationloss1

    def render_image_gaussian_rasterizer(
        self, 
        verbose=False,
        bg_color = None,
        sh_deg:int=None,
        return_2d_radii = False,
        quaternions=None,
        return_opacities:bool=False,
        return_colors:bool=False,
        positions:torch.Tensor=None,
        K = None,
        cam_poses = None,
        pytorch3d_K = None,
        gs_output = None, 
        ):
        """Render an image using the Gaussian Splatting Rasterizer.

        Args:
            nerf_cameras (CamerasWrapper, optional): _description_. Defaults to None.
            camera_indices (int, optional): _description_. Defaults to 0.
            verbose (bool, optional): _description_. Defaults to False.
            bg_color (_type_, optional): _description_. Defaults to None.
            sh_deg (int, optional): _description_. Defaults to None.
            sh_rotations (torch.Tensor, optional): _description_. Defaults to None.
            compute_color_in_rasterizer (bool, optional): _description_. Defaults to False.
            compute_covariance_in_rasterizer (bool, optional): _description_. Defaults to True.
            return_2d_radii (bool, optional): _description_. Defaults to False.
            quaternions (_type_, optional): _description_. Defaults to None.
            use_same_scale_in_all_directions (bool, optional): _description_. Defaults to False.
            return_opacities (bool, optional): _description_. Defaults to False.
            return_colors (bool, optional): _description_. Defaults to False.
            positions (torch.Tensor, optional): _description_. Defaults to None.
            point_colors (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        if bg_color is None:
            bg_color = torch.Tensor([255.0, 255.0, 255.0]).to(self.device)
        
      
        fovx = 2 * torch.arctan(512 / (2 * K[0,0, 0]))
        fovy = 2 * torch.arctan(512 / (2 * K[0,1, 1]))
        tanfovx = math.tan(fovx * 0.5)
        tanfovy = math.tan(fovy * 0.5)
        # 相机投影矩阵和相机位置 3DGS所需
        # TODO: 手动求NEAR和FAR
        zfar = 100.0 # max(zfar, 100.0)
        znear = 0.01 # min(znear, 0.01)
        c2w = cam_poses[0]#torch.cat([cam_poses[0],torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(self.device).view(1,4)],dim=0)
        world_view_transform = c2w.inverse().transpose(0,1)#c2w.float()#torch.from_numpy(np.linalg.inv(c2w).T)
        projection_matrix = get_projection_matrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).float().to(self.device)
        projection_matrix[..., 2, 0] = - pytorch3d_K[0,0, 2] # DEBUG: 为什么要换？
        projection_matrix[..., 2, 1] = - pytorch3d_K[0,1, 2]
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        
        
        raster_settings = GaussianRasterizationSettings(
            image_height=512,
            image_width=512,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=sh_deg,
            campos=camera_center,
            prefiltered=False,
            debug=False
        )
    
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # shs = gs_output["gs_shs"]
        shs = None
        splat_opacities =  gs_output["gs_opacity"]
        quaternions = gs_output["quaternions"]
        scales = gs_output["gs_scales"]
        # splat_colors = None
        splat_colors = gs_output["gs_shs"]
        cov3D = None
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # screenspace_points = torch.zeros_like(self._points, dtype=self._points.dtype, requires_grad=True, device=self.device) + 0
        screenspace_points = torch.zeros(positions.shape[0], 3, dtype=positions.dtype, requires_grad=True, device=self.device)
        if return_2d_radii:
            try:
                screenspace_points.retain_grad()
            except:
                print("WARNING: return_2d_radii is True, but failed to retain grad of screenspace_points!")
                pass
        means2D = screenspace_points
        
        if verbose:
            print("points", positions.shape)
            print("splat_opacities", splat_opacities.shape)
            print("quaternions", quaternions.shape)
            print("scales", scales.shape)
            print("screenspace_points", screenspace_points.shape)
        
        rendered_image, radii = rasterizer(
            means3D = positions,
            means2D = means2D,
            shs = shs,
            colors_precomp = splat_colors,
            opacities = splat_opacities,
            scales = scales,
            rotations = quaternions,
            cov3D_precomp = cov3D)
        
        if not(return_2d_radii or return_opacities or return_colors):
            return rendered_image.transpose(0, 1).transpose(1, 2)
        
        else:
            outputs = {
                "image": rendered_image.transpose(0, 1).transpose(1, 2),
                "radii": radii,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
            }
            if return_opacities:
                outputs["opacities"] = splat_opacities
            if return_colors:
                outputs["colors"] = splat_colors
        
            return outputs

    def render_deformation0(self, styles, cam_poses, focals, beta, theta, trans):
        
        weight1 = torch.from_numpy(np.array([-0.8199, -0.0786], dtype = np.float32))
        tempclothpara = weight1.view(1,2).to('cuda')
        tempclothpara = 2*torch.sigmoid(tempclothpara)-1 
        zeropara = torch.zeros_like(tempclothpara).to('cuda')
        self.tempclothpara = torch.cat((tempclothpara,zeropara),-1)
        
        data_root = '../../neuralbody-deformation-occupancy-fixnerf-our-generalization/tools/data/magdalena20000-allviews/'        
        params_path = os.path.join(data_root, 'smpl_params',
                                   '{}.npy'.format(0))
        paramshape = np.load(params_path, allow_pickle=True).item()
        paramshape = torch.from_numpy(paramshape['shapes']).to('cuda')
        self.canonparamshape = paramshape.view(1,10)
        clothvert = self.net.cloth_simulation(torch.zeros([1,72]).to('cuda'),self.canonparamshape,self.tempclothpara)
        
        self.net.deformation_network.update_embeddedgraph(clothvert.reshape(-1,3))
        
        self.deformedpersonsmpl, self.deformedcloth = self.net.deformation_network.deformingtemplate_lbs(beta, theta, trans) 
        
        c2w = torch.cat([cam_poses[0],torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(self.device).view(1,4)],dim=0)
        vm = c2w.inverse().float()[None]
        mat_R, mat_T = vm[...,:3, :3].transpose(-2, -1), vm[...,:3, 3]
        projection_matrix = focals
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, 512, 512)

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
                                         
       
        meshvert_def = torch.cat([self.deformedcloth[None], self.deformedpersonsmpl[None]], dim=1)#self.deformedcloth
        
        mesh = struct.Meshes(verts=meshvert_def, faces=self.meshface) 
        
        # mesh_test = trimesh.Trimesh(vertices=meshvert_def[0].detach().cpu().numpy(), faces=self.meshface[0].detach().cpu().numpy())
        # mesh_test.export("mesh_test.obj")
        
        fragments = self.net.rasterizer(mesh, cameras=cameras)
        depth = fragments.zbuf
        face_idx_map = fragments.pix_to_face[..., 0]
        self.rendermask = torch.where(depth > 0, torch.ones_like(depth),torch.zeros_like(depth))
        self.rendermask = self.rendermask.squeeze(-1).float()  
        # mesh_mask = self.rendermask[0].detach().cpu().numpy()
        # cv2.imwrite('mesh_mask.png', mesh_mask*255)
        
        self.silhouette = face_idx_map>=self.clothfaces.shape[1]# body mask

        self.silhouette = ~self.silhouette
       
        self.silhouette = self.silhouette.float()
        
        self.silhouette_smpl = (face_idx_map<self.clothfaces.shape[1])&(face_idx_map>=0)# cloth mask

        self.silhouette_smpl = ~self.silhouette_smpl
       
        self.silhouette_smpl = self.silhouette_smpl.float()
        
        # mesh_mask = self.silhouette[0].detach().cpu().numpy()
        # cv2.imwrite('mesh_maskcloth.png', mesh_mask*255)
        # 3dgs渲染
        tri_feats_smpl = self.net.position_enc_smpl(self.net.sugar_model_smpl.points)
        tri_feats_cloth = self.net.position_enc_cloth(self.net.sugar_model_cloth.points)
        
        style = styles.repeat(tri_feats_smpl.shape[0],1)
        tri_feats_smpl = torch.concatenate([tri_feats_smpl,style],dim=-1)
        style = styles.repeat(tri_feats_cloth.shape[0],1)
        tri_feats_cloth = torch.concatenate([tri_feats_cloth,style],dim=-1)
        
        # import open3d as o3d
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(self.net.sugar_model_cloth.points.cpu().detach().numpy())
        # o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
        
        # tri_feats_smpl = self.net.position_enc_smpl(self.net.sugar_model_smpl.points, sp_input['latent_index'].to(torch.int64)[0])
        # tri_feats_cloth = self.net.position_enc_cloth(self.net.sugar_model_cloth.points, sp_input['latent_index'].to(torch.int64)[0])
        # tri_feats_cloth = self.net.position_enc_cloth(self.net.sugar_model_cloth.newpoints, sp_input['latent_index'].to(torch.int64)[0])
        # tri_feats_cloth = self.net.position_enc_cloth(subclothpoints, sp_input['latent_index'].to(torch.int64)[0])
        appearance_out_smpl = self.net.appearance_dec_smpl(tri_feats_smpl)
        appearance_out_cloth = self.net.appearance_dec_cloth(tri_feats_cloth)
        geometry_out_smpl = self.net.geometry_dec_smpl(tri_feats_smpl)
        geometry_out_cloth = self.net.geometry_dec_cloth(tri_feats_cloth)
        
        # 预测透明度和球谐函数参数(SMPL和CLOTH)
        gs_opacity_smpl = appearance_out_smpl['opacity']
        # gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 16, 3)
        gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 3)
        
        gs_opacity_cloth = appearance_out_cloth['opacity']
        # gs_shs_cloth = appearance_out_cloth['shs'].reshape(-1, 16, 3)
        gs_shs_cloth = appearance_out_cloth['shs'].reshape(-1, 3)

        gs_opacity = torch.concatenate([gs_opacity_smpl, gs_opacity_cloth], dim=0)
        gs_shs = torch.concatenate([gs_shs_smpl, gs_shs_cloth], dim=0)
        
        # 预测旋转和尺度(SMPL和CLOTH)
        rotations_smpl = geometry_out_smpl['rotations']
        gs_scales_smpl = geometry_out_smpl['scales']
        # rotations_smpl = rotation_6d_to_matrix(rotations_smpl)
        # quaternions_smpl = matrix_to_quaternion(rotations_smpl)
        
        rotations_cloth = geometry_out_cloth['rotations']
        gs_scales_cloth = geometry_out_cloth['scales']
        # rotations_cloth = rotation_6d_to_matrix(rotations_cloth)
        # quaternions_cloth = matrix_to_quaternion(rotations_cloth)
        
        points_smpl = self.net.sugar_model_smpl.get_edited_points(self.deformedpersonsmpl.squeeze(0))
        quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_edited_quaternions_and_scales_with_points(points=self.deformedpersonsmpl, 
                                                                                                                    quat=rotations_smpl, 
                                                                                                                    sca=gs_scales_smpl)
        
        points_cloth = self.net.sugar_model_cloth.get_edited_points(self.deformedcloth.squeeze(0))
        quaternions_cloth, gs_scales_cloth = self.net.sugar_model_cloth.get_edited_quaternions_and_scales_with_points(points=self.deformedcloth, 
                                                                                                                    quat=rotations_cloth, 
                                                                                                                    sca=gs_scales_cloth)
        
        # import open3d as o3d
        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(self.deformedcloth.squeeze(0).cpu().detach().numpy())
        # mesh.triangles = o3d.utility.Vector3iVector(self.clothfaces[0].detach().cpu().numpy())
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(points_cloth.cpu().detach().numpy())
        # o3d.visualization.draw_geometries([mesh, point_cloud])
        
        # o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
        # o3d.io.write_triangle_mesh("mesh.obj", mesh)
        
        # 细化模型
        # meshes = Meshes(
        #         verts=[self.deformedcloth.squeeze(0).to(self.device)],   
        #         faces=[self.clothfaces.squeeze(0)],
        # )
        
        # meshes = self.submesh(meshes)
        # points = meshes.verts_list()[0]
        # p_faces = meshes.faces_list()[0]
        # points_cloth = self.net.sugar_model_cloth.get_edited_points_subdivide(points, p_faces.squeeze(0))
        # quaternions_cloth, gs_scales_cloth = self.net.sugar_model_cloth.get_edited_quaternions_and_scales_with_points_subdivide(points=points_cloth, 
        #                                                                                                                         faces=p_faces,
        #                                                                                                                         quat=rotations_cloth, 
        #                                                                                                                         sca=gs_scales_cloth)
        
        # SMPL和CLOTH一起渲染
        points = torch.concatenate([points_smpl, points_cloth], dim=0)
        quaternions = torch.concatenate([quaternions_smpl, quaternions_cloth], dim=0)
        gs_scales = torch.concatenate([gs_scales_smpl, gs_scales_cloth], dim=0)
        
        # output Gaussian
        # gs_shs = torch.zeros_like(gs_shs, dtype=torch.float32).to(self.device)
        # gs_shs[:, 0] = torch.tensor([0.5, 0.1, 0.1], dtype=torch.float32).to(self.device)
        # gs_opacity = torch.ones_like(gs_opacity, dtype=torch.float32).to(self.device)

        # # NOTE: 一起渲染
        # gs_output = {
        #     "points": points,
        #     "quaternions": quaternions,
        #     "gs_scales": gs_scales,
        #     "gs_opacity": gs_opacity,
        #     "gs_shs": gs_shs
        # }
        
        # render_output = self.render_image_gaussian_rasterizer(
        #     sh_deg=4,
        #     quaternions=None,
        #     return_2d_radii=True,
        #     return_colors=True,
        #     return_opacities=True,
        #     positions=points,
        #     sp_input=sp_input,
        #     gs_output=gs_output
        # )
        
        # ret = render_output
        gs_output_cloth = {
            "points": points_cloth,
            "quaternions": quaternions_cloth,
            "gs_scales": gs_scales_cloth,
            "gs_opacity": gs_opacity_cloth,
            "gs_shs": gs_shs_cloth
        }
        render_cloth_output = self.render_image_gaussian_rasterizer(
            sh_deg=4,
            quaternions=None,
            return_2d_radii=True,
            return_colors=True,
            return_opacities=True,
            positions=points_cloth,
            K = focals,
            cam_poses = cam_poses,
            pytorch3d_K = pytorch3d_K,
            gs_output=gs_output_cloth
        )
        # self.cloth_img_loss = l1_loss(render_cloth_output['image'].unsqueeze(0), batch['img'] * self.silhouette.unsqueeze(-1))
        # loss_ssim = 1.0 - ssim(render_cloth_output['image'].unsqueeze(0), self.silhouette.unsqueeze(-1))
        # self.cloth_ssim = loss_ssim * (self.silhouette.sum() / (render_cloth_output['image'].shape[-1] * render_cloth_output['image'].shape[-2]))
        
        gs_output_smpl = {
            "points": points_smpl,
            "quaternions": quaternions_smpl,
            "gs_scales": gs_scales_smpl,
            "gs_opacity": gs_opacity_smpl,
            "gs_shs": gs_shs_smpl
        }
        render_smpl_output = self.render_image_gaussian_rasterizer(
            sh_deg=4,
            quaternions=None,
            return_2d_radii=True,
            return_colors=True,
            return_opacities=True,
            positions=points_smpl,
            K = focals,
            cam_poses = cam_poses,
            pytorch3d_K = pytorch3d_K,
            gs_output=gs_output_smpl
        )
        
        # self.smpl_img_loss = l1_loss(render_smpl_output['image'].unsqueeze(0), batch['img'] * self.silhouette_smpl.unsqueeze(-1))
        # loss_ssim_smpl = 1.0 - ssim(render_smpl_output['image'].unsqueeze(0), self.silhouette_smpl.unsqueeze(-1))
        # self.smpl_ssim = loss_ssim_smpl * (self.silhouette_smpl.sum() / (render_smpl_output['image'].shape[-1] * render_smpl_output['image'].shape[-2]))
        
        # NOTE: 使用 1-blendmask
        #rendered_img = render_smpl_output['image'] * (1-self.silhouette[0].unsqueeze(-1)) + render_cloth_output['image'] * self.silhouette[0].unsqueeze(-1)
        rendered_img = render_smpl_output['image'] * self.silhouette_smpl[0].unsqueeze(-1) + render_cloth_output['image'] * self.silhouette[0].unsqueeze(-1)
        
        rendered_img = rendered_img / 127.5 - 1
        rendered_img = rendered_img[None]#[:,:256,:]

        ret = {
            'image': rendered_img,
            'cloth_scene': render_cloth_output,
            'smpl_scene': render_smpl_output,
            'gs_output_smpl': gs_output_smpl,
            'gs_output_cloth': gs_output_cloth
        }
        # if frame_index == 700:
        # # if True:
            # render_img = ret['image'].cpu().detach().numpy() * 255
            # render_img = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(f"gs_render.jpg", render_img)
            # render_img = render_cloth_output['image'].cpu().detach().numpy() * 255
            # render_img = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(f"gs_render_cloth.jpg", render_img)
            # render_img = render_smpl_output['image'].cpu().detach().numpy() * 255
            # render_img = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(f"gs_render_smpl.jpg", render_img)
        
        # if(epoch % 5 == 0):
        #     self.save_ply(
        #         xyz=points,
        #         f_dc=gs_shs[:, :1],
        #         f_rest=gs_shs[:, 1:],
        #         opacities=gs_opacity,
        #         scale=gs_scales,
        #         rotation=quaternions,
        #         path=os.path.join("tmp", "points_cloud.ply")
        #     )
        
        return ret
        
    def batch_rigid_transform(self, rot_mats, init_J):
        joints = torch.from_numpy(init_J.reshape(1, -1, 3, 1)).cuda()
        parents = self.parents

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]

        transforms_mat = transform_mat(
            rot_mats.reshape(-1, 3, 3),
            rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)

        posed_joints = transforms[:, :, :3, 3]

        joints_homogen = F.pad(joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

        return posed_joints, rel_transforms
        
    def deforming_lbs_newsmpl(self, beta, theta, trans, canonicalsmpl):
        #theta0 = theta.clone()

        #theta0[:,0:3] = 0
              
        theta = batch_rodrigues(theta.reshape(-1, 3)).reshape(1, 24, 3, 3)
        _theta = theta.reshape(1, 24, 3, 3)
        # so = self.smpl_model(betas = beta.reshape(1, 10), body_pose = _theta[:, 1:], global_orient = _theta[:, 0].view(1, 1, 3, 3))
        # smpl_v = so['vertices'].clone().reshape(-1, 3)
        
        init_J = get_J(beta.reshape(1, 10), self.smpl_model)

        #_, rel_transforms = smplx.lbs.batch_rigid_transform(_theta, torch.from_numpy(init_J).cuda(), self.parents)
        #theta_rodrigues = batch_rodrigues(theta0.reshape(-1, 3)).reshape(1, 24, 3, 3)
        _, rel_transforms = self.batch_rigid_transform(_theta, init_J)

        #init_J = get_J(beta.reshape(1, 10), self.smpl_model)

        #_, rel_transforms = self.batch_rigid_transform(theta_rodrigues, init_J)
        # joints = torch.from_numpy(init_J.reshape(-1, 3)).cuda()
        # rel_transforms = self.get_rigid_transformation(theta0.reshape(-1, 3).detach().cpu().numpy(), self.joints, self.parents)
        # rel_transforms = torch.from_numpy(rel_transforms).cuda()self.smplbwsmpl_model.lbs_weights

        smpl_A = torch.matmul(self.smpl_model.lbs_weights.view(-1, self.num_joints), rel_transforms.view(self.num_joints, 16)).view(-1, 4, 4)

        R = smpl_A[:, :3, :3]#including global rotation
        #pts = torch.sum(R * self.templatesmpl[:, :, None], dim=-1)
        pts = torch.matmul(R, canonicalsmpl[:, :, None])
        pts = pts.squeeze(-1) + smpl_A[:, :3, 3]
        deformedsmplvert = pts + trans#.unsqueeze(1)torch.matmul(pts, gR[0].transpose(0, 1))
        
        return deformedsmplvert
        
    def render_deformation(self, styles, cam_poses, focals, beta, theta, trans):
        
        thetazero = batch_rodrigues(torch.zeros_like(theta).reshape(-1, 3)).reshape(1, 24, 3, 3)
        thetazero = thetazero.reshape(1, 24, 3, 3)
        
        so = self.smpl_model(betas = torch.zeros_like(beta).reshape(1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(1, 1, 3, 3))
        canoncalmeansmpl = so['vertices'].clone().reshape(-1, 3)#mean smpl
        
        so = self.smpl_model(betas = beta.reshape(1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(1, 1, 3, 3))
        canoncalsmpl = so['vertices'].clone().reshape(-1, 3)
        
      
        c2w = cam_poses[0]#torch.cat([cam_poses[0],torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(self.device).view(1,4)],dim=0)
        vm = c2w.inverse().float()[None]
        mat_R, mat_T = vm[...,:3, :3].transpose(-2, -1), vm[...,:3, 3]
        projection_matrix = focals
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, 512, 512)

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
                                         
                      
        #self.net.position_enc_smpl.triplanefeature(styles)
        
        #vertoffset, _ = self.net.position_enc_smpl(canoncalmeansmpl)
        
        canpersonalsmpl= canoncalsmpl#+tri_feats_smpl[:,:3]
        
        
        self.deformedpersonsmpl = self.deforming_lbs_newsmpl(beta, theta, trans, canpersonalsmpl)
        
        self.net.sugar_model_smpl._points = canpersonalsmpl
        
        # 3dgs渲染
        tri_feats_smpl = self.net.position_enc_smpl(self.net.sugar_model_smpl.points)
        
        style = styles.repeat(tri_feats_smpl.shape[0],1)
        tri_feats_smpl = torch.concatenate([tri_feats_smpl,style],dim=-1)
        
        appearance_out_smpl = self.net.appearance_dec_smpl(tri_feats_smpl)
        geometry_out_smpl = self.net.geometry_dec_smpl(tri_feats_smpl)
        
        # 预测透明度和球谐函数参数(SMPL和CLOTH)
        gs_opacity_smpl = appearance_out_smpl['opacity']
        # gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 16, 3)
        gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 3)
        
        # 预测旋转和尺度(SMPL和CLOTH)
        rotations_smpl = geometry_out_smpl['rotations']
        gs_scales_smpl = geometry_out_smpl['scales']
        # rotations_smpl = rotation_6d_to_matrix(rotations_smpl)
        # quaternions_smpl = matrix_to_quaternion(rotations_smpl)
        
        
        points_smpl = self.net.sugar_model_smpl.get_edited_points(self.deformedpersonsmpl.squeeze(0))
        quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_edited_quaternions_and_scales_with_points(points=self.deformedpersonsmpl, 
                                                                                                                    quat=rotations_smpl, 
                                                                                                                    sca=gs_scales_smpl)
        
        gs_output_smpl = {
            "points": points_smpl,
            "quaternions": quaternions_smpl,
            "gs_scales": gs_scales_smpl,
            "gs_opacity": gs_opacity_smpl,
            "gs_shs": gs_shs_smpl
        }
        render_smpl_output = self.render_image_gaussian_rasterizer(
            sh_deg=4,
            quaternions=None,
            return_2d_radii=True,
            return_colors=True,
            return_opacities=True,
            positions=points_smpl,
            K = focals,
            cam_poses = cam_poses,
            pytorch3d_K = pytorch3d_K,
            gs_output=gs_output_smpl
        )
        
        # self.smpl_img_loss = l1_loss(render_smpl_output['image'].unsqueeze(0), batch['img'] * self.silhouette_smpl.unsqueeze(-1))
        # loss_ssim_smpl = 1.0 - ssim(render_smpl_output['image'].unsqueeze(0), self.silhouette_smpl.unsqueeze(-1))
        # self.smpl_ssim = loss_ssim_smpl * (self.silhouette_smpl.sum() / (render_smpl_output['image'].shape[-1] * render_smpl_output['image'].shape[-2]))
        
        # NOTE: 使用 1-blendmask
        #rendered_img = render_smpl_output['image'] * (1-self.silhouette[0].unsqueeze(-1)) + render_cloth_output['image'] * self.silhouette[0].unsqueeze(-1)
        rendered_img = render_smpl_output['image']
        rendered_img = rendered_img / 127.5 - 1
        rendered_img = rendered_img[None]#[:,:256,:]

        ret = {
            'image': rendered_img,
            #'cloth_scene': render_cloth_output,
            'smpl_scene': render_smpl_output,
            'gs_output_smpl': gs_output_smpl,
            #'gs_output_cloth': gs_output_cloth
        }
        
        return ret
        
    def save_ply(self, 
                 xyz,
                 f_dc,
                 f_rest,
                 opacities,
                 scale,
                 rotation,
                 path):
        os.makedirs(os.path.dirname(path),exist_ok=True)

        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = f_dc.flatten(start_dim=1).detach().contiguous().cpu().numpy()
        f_rest = f_rest.flatten(start_dim=1).detach().contiguous().cpu().numpy()
        opacities = opacities.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()
        
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(3*15):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)