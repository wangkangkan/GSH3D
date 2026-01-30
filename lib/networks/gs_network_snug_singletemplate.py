import torch.nn as nn
import torch.nn.functional as F
import torch
#from lib.config import cfg
from lib.networks.modules.decoders import SDFDecoder, AppearanceDecoder, GeometryDecoder, DispalceDecoder, PositionEncoder, AppearanceDecoder2, GeometryDecoder2
from lib.networks.modules.triplane import TriPlane as TriPlane
from lib.networks.sugar_model import SuGaR
# from lib.utils.blend_utils import *
# from . import embedder
from lib.utils import net_utils
import os
import numpy as np
import scipy.io as scio
import trimesh
# from sklearn.decomposition import PCA
import scipy.sparse as sp
import pickle
import lib.networks.modules as modules
# from lib.networks.meta_modules import HyperNetwork
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRasterizer
)
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor
)
import smplx
import open3d as o3d

from smpl_utils import init_smpl, get_J, get_shape_pose, batch_rodrigues
from smplx.lbs import transform_mat, blend_shapes

from lib.networks.modules.UVmapfeature import UVmap_GS, UVmap_vert 
 
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        n_features = 32
        triplane_res = 256
        sh_levels = 4
        n_gaussians_per_surface_triangle = 6
    
        self.position_enc_smpl = UVmap_GS()
        
        #self.IncSDF = SDFDecoder(n_features=3)
        
        # self.position_enc_smpl = TriPlane(n_features,
                                 # resX=triplane_res,
                                 # resY=triplane_res,
                                 # resZ=triplane_res)
        # self.position_enc_cloth = TriPlane(n_features,
                                 # resX=triplane_res,
                                 # resY=triplane_res,
                                 # resZ=triplane_res)
                                 
        # self.position_enc_smpl = PositionEncoder(
        #     n_features=cfg.encoder.n_features,
        #     num_train_frame=cfg.num_train_frame
        # )
        # self.position_enc_cloth = PositionEncoder(
        #     n_features=cfg.encoder.n_features,
        #     num_train_frame=cfg.num_train_frame
        # )
        #self.appearance_dec_smpl = AppearanceDecoder(n_features=n_features*3)
        #self.appearance_dec_cloth = AppearanceDecoder(n_features=n_features*3)
        #self.appearance_dec_smplcloth = AppearanceDecoder(n_features=n_features*3)#n_features*3
        
        # self.appearance_dec_smpl = AppearanceDecoder(n_features=cfg.triplane.n_features*3 + cfg.encoder.n_features)
        # self.appearance_dec_cloth = AppearanceDecoder(n_features=cfg.triplane.n_features*3 + cfg.encoder.n_features)
        
        #self.geometry_dec_smpl = GeometryDecoder(n_features=n_features*3)
        #self.geometry_dec_cloth = GeometryDecoder(n_features=n_features*3)
        #self.geometry_dec_smplcloth = GeometryDecoder(n_features=n_features*3)
        #self.displace_dec_smpl = DispalceDecoder(n_features=n_features*3)
        
        # self.geometry_dec_smpl = GeometryDecoder(n_features=cfg.triplane.n_features*3 + cfg.encoder.n_features, use_surface=True)
        # self.geometry_dec_cloth = GeometryDecoder(n_features=cfg.triplane.n_features*3 + cfg.encoder.n_features, use_surface=True)
        
        smpl_obj = "cloth/deformation/smpl/template.obj"
        o3d_mesh_smpl = o3d.io.read_triangle_mesh(smpl_obj)
        
        data_root = 'cloth/deformation'
        
        points = torch.randn(1000, 3, device=self.device)
        colors = torch.rand(1000, 3, device=self.device)
        # NOTE: 原始服装面信息
        npfaces = np.loadtxt(os.path.join(data_root, 'smpl/smpltri.txt')) - 1
        self.desmplfaces = torch.LongTensor(npfaces).to(self.device)
        templatesmpl_path = os.path.join(data_root,
                                         'smpl/templateshape.txt') 
        templatesmpl = np.loadtxt(templatesmpl_path)
        self.templatesmpl = torch.Tensor(templatesmpl).to(self.device)
        
        self.sugar_model_smpl = SuGaR(
            points=points, #nerfmodel.gaussians.get_xyz.data,
            colors=colors,
            initialize=True,
            sh_levels=sh_levels,
            learnable_positions=False,
            triangle_scale=1.0,
            keep_track_of_knn=False,
            knn_to_track=0,
            beta_mode=None,
            freeze_gaussians=False,
            surface_mesh_to_bind=o3d_mesh_smpl,
            surface_mesh_thickness=1e-5,
            learn_surface_mesh_positions=True, # 是否需要优化顶点信息
            learn_surface_mesh_opacity=True,
            learn_surface_mesh_scales=True,
            n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle, # Default:=1
            faces=self.desmplfaces,
            verts=self.templatesmpl,
            optim_param_name="xyz_smpl"
        )
        
        # npfaces = np.loadtxt(os.path.join('./', 'cloth/clothes_face.txt')) - 1
        # self.clothfaces = torch.LongTensor(npfaces).to(self.device)
        
        # # loading template deformation graph
        # templateshape_path = os.path.join('./', 'cloth/clothes_vert.txt')
        # templatecloth = np.loadtxt(templateshape_path)
        # templatecloth = torch.Tensor(templatecloth).to(self.device)
        
        # self.sugar_model_cloth = SuGaR(
            # points=points, #nerfmodel.gaussians.get_xyz.data,
            # colors=colors,
            # initialize=True,
            # sh_levels=sh_levels,
            # learnable_positions=False,
            # triangle_scale=1.0,
            # keep_track_of_knn=False,
            # knn_to_track=0,
            # beta_mode=None,
            # freeze_gaussians=False,
            # surface_mesh_to_bind=o3d_mesh_smpl,
            # surface_mesh_thickness=1e-5,
            # learn_surface_mesh_positions=True, # 是否需要优化顶点信息
            # learn_surface_mesh_opacity=True,
            # learn_surface_mesh_scales=True,
            # n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle, # Default:=1
            # faces=self.clothfaces,
            # verts=templatecloth,
            # optim_param_name="xyz_smpl"
        # )
        
        # #combined mesh 
        # self.meshface = torch.cat([self.clothfaces, self.desmplfaces+templatecloth.shape[0]], dim=0)
        # self.templatemesh = torch.cat([templatecloth, self.templatesmpl], dim=0)
        
        # self.sugar_model_smplcloth = SuGaR(
            # points=points, #nerfmodel.gaussians.get_xyz.data,
            # colors=colors,
            # initialize=True,
            # sh_levels=sh_levels,
            # learnable_positions=False,
            # triangle_scale=1.0,
            # keep_track_of_knn=False,
            # knn_to_track=0,
            # beta_mode=None,
            # freeze_gaussians=False,
            # surface_mesh_to_bind=o3d_mesh_smpl,
            # surface_mesh_thickness=1e-5,
            # learn_surface_mesh_positions=True, # 是否需要优化顶点信息
            # learn_surface_mesh_opacity=True,
            # learn_surface_mesh_scales=True,
            # n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle, # Default:=1
            # faces=self.meshface,
            # verts=self.templatemesh,
            # optim_param_name="xyz_smpl"
        # )
        
        #self.init_renderer()
        #self.deformation_network = VaryingClothDeformationNetwork() 
    
    def init_renderer(self):
        elev = torch.linspace(0, 360, 4)
        azim = torch.linspace(-180, 180, 4)
        
        H = 512
        W = 512
        R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=(int(H), int(W)),
            blur_radius=0,
            #max_faces_per_bin=20000
            #blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=1, 
            #perspective_correct=False,
            # bin_size=int(2 ** max(np.ceil(np.log2(max(cfg.H,cfg.W))) - 4, 4)),
            # faces_per_pixel=1,
            # perspective_correct=True,
            # clip_barycentric_coords=False,
            # cull_backfaces=False,
        )
        self.rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        raster_settings_silhouette = PointsRasterizationSettings(
            image_size=(int(H), int(W)), 
            radius=0.005,
            bin_size=(92 if max(H, W)>1024 and max(H, W)<=2048 else None),
            points_per_pixel=50,
            )   
        self.pcRender=PointsRendererWithFrags(
            rasterizer=PointsRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings_silhouette
            ),
                compositor=AlphaCompositor(background_color=None)
            )#.to(self.device)
            
        self.pcRender_def=PointsRendererWithFrags(
            rasterizer=PointsRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings_silhouette
            ),
                compositor=AlphaCompositor(background_color=None)
            )#.to(self.device)   
        
    def pts_to_can_pts(self, pts, sp_input):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = sp_input['Th']
        pts = pts - Th
        R = sp_input['R']
        pts = torch.matmul(pts, R)
        return pts

      
    def calculate_density_color_clothdeformation_layer(self, wpts, viewdir, sp_input):

        smpl_canpts, cloth_canpts = self.deformation_network.inversedeforming_samplepoints_layer(wpts, sp_input)#, posedirs
        smpllight_pts = embedder.xyz_embedder(smpl_canpts)
        clothlight_pts = embedder.xyz_embedder(cloth_canpts)
        
        alpha_smpl, _ = self.occupancy_network_smpl(smpllight_pts)
        

        alpha_cloth, _ = self.occupancy_network_cloth(clothlight_pts)

        rgb_smpl = self.color_network_smpl(smpllight_pts, viewdir, sp_input)
        rgb_cloth = self.color_network_cloth(clothlight_pts, viewdir, sp_input)
        

        raw_smpl = torch.cat([rgb_smpl, alpha_smpl], -1)
        raw_cloth = torch.cat([rgb_cloth, alpha_cloth], -1)

        return raw_smpl, raw_cloth#,wpts_sdf,  blending
           
    def calculate_cannonicalsdf_layer(self, batch):

        wpts_smpl = batch['meshpts_smpl'][0]
        tpose_smpl = batch['tpose_smpl'][0]
        wpts_smpl = torch.cat([wpts_smpl, tpose_smpl], dim=0)#dim=0
        wpts_smpl.requires_grad_()
      
        sdf_smpl, _ = self.sdf_network_smpl(wpts_smpl)

        #sdf = sdf_nn_output[:, 0]
        #feature_vector = sdf_nn_output[:, 1:]

        # calculate normal

        gradients_smpl = self.sdf_network_smpl.sdf_network.gradient(wpts_smpl).squeeze()
        
        wpts_cloth = batch['meshpts_cloth'][0]
        tpose_cloth = batch['tpose_cloth'][0]
        wpts_cloth = torch.cat([wpts_cloth, tpose_cloth], dim=0)#dim=0
        wpts_cloth.requires_grad_()
        sdf_cloth, _ = self.sdf_network_cloth(wpts_cloth)

        gradients_cloth = self.sdf_network_cloth.sdf_network.gradient(wpts_cloth).squeeze()

        # NOTE: occ_loss_SMPL
        wpts_smpl = torch.cat([batch['meshpts_smpl'], batch['tpose_smpl']], dim=1)
        smpllight_pts = embedder.xyz_embedder(wpts_smpl)
        occupancy_smpl, _ = self.occupancy_network_smpl(smpllight_pts)
        occupancy_smpl = occupancy_smpl[0][:, 0]
        occ = occupancy_smpl[sdf_smpl <= 0]
        free = occupancy_smpl[sdf_smpl > 0]
        label = torch.ones_like(occ).to(self.device)
        occ_loss = torch.nn.functional.binary_cross_entropy(occ, label)
        label = torch.zeros_like(free).to(self.device)
        free_loss = torch.nn.functional.binary_cross_entropy(free, label)
        occ_loss_smpl = occ_loss + free_loss#sdf_smpl, gradients_smpl, occ_loss_smpl,
        
        wpts_cloth = torch.cat([batch['meshpts_cloth'], batch['tpose_cloth']], dim=1)
        clothlight_pts = embedder.xyz_embedder(wpts_cloth)
        occupancy_cloth, _ = self.occupancy_network_cloth(clothlight_pts)
        occupancy_cloth = occupancy_cloth[0][:, 0]
        occ = occupancy_cloth[sdf_cloth <= 0]
        free = occupancy_cloth[sdf_cloth > 0]
        label = torch.ones_like(occ).to(self.device)
        occ_loss = torch.nn.functional.binary_cross_entropy(occ, label)
        label = torch.zeros_like(free).to(self.device)
        free_loss = torch.nn.functional.binary_cross_entropy(free, label)
        occ_loss_cloth = occ_loss + free_loss
        
        return  sdf_smpl, gradients_smpl, occ_loss_smpl, sdf_cloth, gradients_cloth, occ_loss_cloth
        
    def calculate_cannonicalsdf_layer_varyingcloth(self, batch, clothsdf):

        wpts_smpl = batch['meshpts_smpl'][0]
        tpose_smpl = batch['tpose_smpl'][0]
        wpts_smpl = torch.cat([wpts_smpl, tpose_smpl], dim=0)
        #wpts.requires_grad_()

        sdf_smpl = self.sdf_network_smpl(wpts_smpl)

        #sdf = sdf_nn_output[:, 0]
        #feature_vector = sdf_nn_output[:, 1:]

        # calculate normal
        gradients_smpl = self.sdf_network_smpl.sdf_network.gradient(wpts_smpl).squeeze()
        
        wpts_cloth = clothsdf['meshpts_cloth'][0]
        tpose_cloth = clothsdf['tpose_cloth'][0]
        wpts_cloth = torch.cat([wpts_cloth, tpose_cloth], dim=0)

        sdf_cloth = self.sdf_network_cloth(wpts_cloth)

        gradients_cloth = self.sdf_network_cloth.sdf_network.gradient(wpts_cloth).squeeze()
        
        return sdf_smpl, gradients_smpl, sdf_cloth, gradients_cloth    
    
    def update_clothshape(self):
    
        #self.clothvert = self.clothpara(torch.zeros(1).to(torch.int64).to(self.device))*self.initclothvert
        clothpara = self.clothpara(torch.zeros(1).to(torch.int64).to(self.device))
        #self.clothvert = self.pca.inverse_transform(clothpara).reshape([-1, 3])
        clothvert = torch.mm(clothpara, self.components) + self.mean
        self.clothvert = clothvert.reshape([-1, 3])
        #np.savetxt('clothtest.txt', self.clothvert[0].detach().cpu().numpy())
        return self.clothvert
    
    def forward(self, sp_input, grid_coords, viewdir, light_pts):
        __import__('ipdb').set_trace()

class VertDispalceNetwork(nn.Module):
    """
        预测顶点偏移
    """
    def __init__(self):
        super(VertDispalceNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latentdeform = nn.Embedding(cfg.num_train_frame, 128)
        cloth_mesh = trimesh.load(os.path.join("tools/data/dnarendering/cloth_template", "skining.obj"))
        self.template = torch.from_numpy(cloth_mesh.vertices).to(self.device)
        self.faces = torch.from_numpy(cloth_mesh.faces.astype(np.int32)).to(self.device)
        self.edges = self.get_vertex_connectivity(self.faces).to(self.device)
        self.edge_length = self.get_edge_length(self.template, self.edges)
        self.f_normal = self.face_normal(self.template, self.faces)
        self.output_verts = self.template.shape[0]
        
        D = 8
        self.deformskips = [4]
        defW = 1024
        layers = [nn.Linear(128, defW)]  # node coding + latent code
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = defW
            if i in self.deformskips:
                in_channels += 128
            layers += [layer(in_channels, defW)]

        self.deformpara_linears = nn.ModuleList(layers)
        self.deformpara_finallinear = nn.Linear(defW, self.output_verts * 3)
    
    def get_vertex_connectivity(self, faces,dtype=torch.int32):
        '''
        Returns a list of unique edges in the mesh. 
        Each edge contains the indices of the vertices it connects
        返回一个包含mesh中所有边的列表 每条边包含一次
        每条边包含着它所连接的顶点的索引
        '''
        # if torch.is_tensor(faces):
        #     faces = faces.cpu().numpy()

        edges = set()
        for f in faces:
            num_vertices = len(f)
            for i in range(num_vertices):
                j = (i + 1) % num_vertices
                edges.add(tuple(sorted([f[i], f[j]])))

        return torch.tensor(list(edges),dtype=dtype)
    
    def get_edge_length(self, vertices, edges):
        edges = edges.long()
        v0 = torch.gather(vertices, -2, edges[:,0].unsqueeze(1).expand(-1,3))
        v1 = torch.gather(vertices, -2, edges[:,1].unsqueeze(1).expand(-1,3))
        return torch.linalg.norm(v0 - v1, axis=-1)
    
    def face_normal(self,vertices,faces):
        v = vertices
        f = faces
        vertice_num = v.shape[-1]
        indices_v_num = f.shape[-1]
        # 是否为批量 
        is_batch = False
        if len(v.shape) == (len(f.shape) + 1):
            f = torch.tile(f, [v.shape[0],1, 1])
            f = f.reshape(f.shape[0],-1,1)
            f = f.repeat([1,1,vertice_num]).long()
            is_batch = True
        else:
            f = f.reshape(-1,1)
            f = f.repeat(1,vertice_num).long()
        # 获取mesh的三角形的顶点坐标[batch_size,num_faces,3,3]
        triangles = torch.gather(vertices, -2, f)
        if is_batch:
            triangles = triangles.reshape(triangles.shape[0],-1,indices_v_num,vertice_num)
        else:
            triangles = triangles.reshape(-1,indices_v_num,vertice_num)
        
        # 计算法向量
        v0 = triangles[...,0,:]
        v1 = triangles[...,1,:]
        v2 = triangles[...,2,:]
        e1 = (v0 - v1)
        e2 = (v2 - v1)
        face_normals = torch.cross(e2, e1,dim=-1)
        # 单位化
        # if self.normalize:
        normals = face_normals.norm(dim=-1,keepdim=True)
        face_normals = face_normals / normals
        
        return face_normals

    
    def computeLoss(self, vertices, vertices_snug):
        edge_length = self.get_edge_length(vertices=vertices.squeeze(), edges=self.edges)
        edge_loss = torch.sqrt(((edge_length - self.edge_length) ** 2) + 1e-20).mean()
        # edge_loss = torch.sum(torch.norm(edge_length - self.edge_length, dim=-1)) / len(self.edges)
        f_n = self.face_normal(vertices=vertices.squeeze(), faces=self.faces)
        f_n_snug = self.face_normal(vertices=vertices_snug.squeeze(), faces=self.faces)
        # ss = torch.sum(torch.mul(f_n, self.f_normal),dim=-1)
        fn_loss = torch.mean(1 - torch.sum(torch.mul(f_n, f_n_snug),dim=-1))
        
        return edge_loss * 10.0 + fn_loss * 0.3
         
    def forward(self, sp_input):
        latent = self.latentdeform(sp_input['latent_index'].to(torch.int64))  # .type(torch.LongTensor).to(self.device)np.asscalar(np.int16(sp_input['latent_index']))
        h = latent
 
        for i, l in enumerate(self.deformpara_linears):
           h = self.deformpara_linears[i](h)
           h = F.relu(h)
           if i in self.deformskips:
               h = torch.cat([latent, h], -1)
        h = self.deformpara_finallinear(h)
        
        return h.reshape(-1, self.output_verts, 3)
    
class VaryingClothDeformationNetwork(nn.Module):
    def __init__(self):
        super(VaryingClothDeformationNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_root = 'cloth/deformation'

        templatesmpl_path = os.path.join(data_root,
                                         'smpl/vpersonalshape.txt')  # smpldeform/vpersonalshape
        templatesmpl = np.loadtxt(templatesmpl_path)
        self.rawtemplatesmpl = torch.Tensor(templatesmpl).to(self.device)
        #self.smplvtnum = self.templatesmpl.size(0)
        
        npfaces = np.loadtxt(os.path.join(data_root, 'smpl/smpltri.txt')) - 1
        self.smplfaces = torch.LongTensor(npfaces).to(self.device)
        
        _,trinormal = mesh_face_areas_normals(self.rawtemplatesmpl, self.smplfaces)
         
        npfaces = np.loadtxt(os.path.join(data_root, 'smpl/smpl_vfidx.txt')) - 1
        self.smpl_vfidx = torch.LongTensor(npfaces).to(self.device)
        
        self.smpl_vertnorm = trinormal[self.smpl_vfidx,:]
        
        # loading template deformation graph
        # templateshape_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/clothes_vert.txt')
        # templateshape = np.loadtxt(templateshape_path)
        # self.templateshape = torch.Tensor(templateshape).to(self.device)
        #self.templateshape = clothvert
        # self.vtnum = self.templateshape.size(0)
        # modelnodepos_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/modelnodepos.txt')
        # modelnodepos = np.loadtxt(modelnodepos_path)
        # modelnodepos = modelnodepos / 1000.0
        # self.modelnodepos = torch.Tensor(modelnodepos).to(self.device)
        
        modelnodeidx_path = os.path.join(data_root, 'templatedeformT/cloth/nodevidx.txt')
        modelnodeidx = np.loadtxt(modelnodeidx_path)
        self.modelnodeidx = torch.LongTensor(modelnodeidx).to(self.device)
        #self.modelnodepos = self.templateshape[self.modelnodeidx, :]
        
        # modelnodenormal_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/modelnodenormal.txt')
        # modelnodenormal = np.loadtxt(modelnodenormal_path)
        # self.modelnodenormal = torch.Tensor(modelnodenormal).to(self.device)
        
        self.modelnodenum = self.modelnodeidx.size(0)
        modelnodeedge_path = os.path.join(data_root, 'templatedeformT/cloth/modelnodeedge.txt')
        modelnodeedge = np.loadtxt(modelnodeedge_path) - 1
        self.modelnodeedge = torch.LongTensor(modelnodeedge).to(self.device)
        self.modelnodeedgenum = self.modelnodeedge.size(1)
        # modelnode_edgeweight_path = os.path.join(cfg.train_dataset.data_root,
                                                 # 'templatedeformT/cloth/modelnode_edgeweight.txt')
        # modelnode_edgeweight = np.loadtxt(modelnode_edgeweight_path)
        # self.modelnode_edgeweight = torch.Tensor(modelnode_edgeweight).to(self.device)
        modelvertnode_path = os.path.join(data_root, 'templatedeformT/cloth/modelvert_node.txt')
        modelvert_node = np.loadtxt(modelvertnode_path) - 1
        self.modelvert_node = torch.LongTensor(modelvert_node).to(self.device)
        modelvertnodeweight_path = os.path.join(data_root, 'templatedeformT/cloth/modelvert_nodeweight.txt')
        modelvert_nodeweight = np.loadtxt(modelvertnodeweight_path)
        self.modelvert_nodeweight = torch.Tensor(modelvert_nodeweight).to(self.device)
        self.modelvert_nodenum = self.modelvert_node.size(1)
        # self.bw = np.load(os.path.join(data_root, 'bw_weight.npy'))
        # #self.bw[:,0] = 0#global rotation remove
        # self.bw = torch.Tensor(self.bw)[None, ...].to(self.device)
        
        # self.latentdeform = nn.Embedding(cfg.num_train_frame, 128)
        # D = 8
        # self.deformskips = [4]
        # defW = 1024
        # layers = [nn.Linear(128, defW)]  # node coding + latent code
        # for i in range(D - 1):
            # layer = nn.Linear
            # in_channels = defW
            # if i in self.deformskips:
                # in_channels += 128
            # layers += [layer(in_channels, defW)]

        # self.deformpara_linears = nn.ModuleList(layers)
        # self.deformpara_finallinear = nn.Linear(defW, self.modelnodenum * 6)

        
        
        # self.smpl_model = init_smpl(
            # model_folder = 'smpl_models',
            # model_type = 'smpl',
            # gender = 'female',
            # num_betas = 10
        # )
        # parents = self.smpl_model.parents.cpu().numpy()
        # #self.parents = parents
        # self.num_joints = num_joints = parents.shape[0]
        
        # joints = np.load(os.path.join(data_root, 'joints.npy'))
        # #self.joints = torch.Tensor(joints).to(self.device)
        
        # self.joints = joints.astype(np.float32)
        # parents = np.load(os.path.join(data_root, 'parents.npy'), allow_pickle=True)
        # self.parents = np.array(parents)
        
        # self.initsmplgraph()
        
    def initsmplgraph(self):
        
        data_root = 'cloth/deformation'
        
        npfaces = np.loadtxt(os.path.join(data_root, 'templatedeformT/desmpl/desmpl_vfidx.txt')) - 1
        self.desmpl_vfidx = torch.LongTensor(npfaces).to(self.device)
        
        templatesmpl_path = os.path.join(data_root,
                                         'templatedeformT/desmpl/desmplvt.txt')  # smpldeform/vpersonalshape
        templatesmpl = np.loadtxt(templatesmpl_path)
        self.templatesmpl = torch.Tensor(templatesmpl).to(self.device)
        self.smplvtnum = self.templatesmpl.size(0)
        
        smplnodeidx_path = os.path.join(data_root, 'templatedeformT/desmpl/nodevidx.txt')
        smplnodeidx = np.loadtxt(smplnodeidx_path)
        self.smplnodeidx = torch.LongTensor(smplnodeidx).to(self.device)
        self.smplnodepos = self.templatesmpl[self.smplnodeidx, :]
        
        self.smplnodenum = self.smplnodeidx.size(0)
        smplnodeedge_path = os.path.join(data_root, 'templatedeformT/desmpl/modelnodeedge.txt')
        smplnodeedge = np.loadtxt(smplnodeedge_path) - 1
        self.smplnodeedge = torch.LongTensor(smplnodeedge).to(self.device)
        self.smplnodeedgenum = self.smplnodeedge.size(1)
        
        smplvertnode_path = os.path.join(data_root, 'templatedeformT/desmpl/modelvert_node.txt')
        smplvert_node = np.loadtxt(smplvertnode_path) - 1
        self.smplvert_node = torch.LongTensor(smplvert_node).to(self.device)
        smplvertnodeweight_path = os.path.join(data_root, 'templatedeformT/desmpl/modelvert_nodeweight.txt')
        smplvert_nodeweight = np.loadtxt(smplvertnodeweight_path)
        self.smplvert_nodeweight = torch.Tensor(smplvert_nodeweight).to(self.device)
        self.smplvert_nodenum = self.smplvert_node.size(1)
        
           
        npfaces = np.loadtxt(os.path.join(data_root, 'templatedeformT/desmpl/desmpltri.txt')) - 1
        self.desmplfaces = torch.LongTensor(npfaces).to(self.device)
        
        bw = np.load(os.path.join(data_root, 'bw.npy'), allow_pickle=True)
        #bw[:,0] = 0#global rotation remove
        bw = torch.Tensor(bw).to(self.device)
        self.rawbw = bw[None, ...]
        
        # templateshape_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/clothes_vert.txt')
        # fixtemplateshape = np.loadtxt(templateshape_path)
        # fixtemplateshape = torch.Tensor(fixtemplateshape).to(self.device)
        ptsdist = torch.cdist(self.templatesmpl, self.rawtemplatesmpl, p=1)#self.templateshape
        minptsdist = torch.min(ptsdist, 1)
        minptsdistvalue = torch.squeeze(minptsdist[0], -1)
        nnvidx = torch.squeeze(minptsdist[1], -1)  # P
        self.smplbw = bw[nnvidx, :][None, ...]

        # self.smpllatentdeform = nn.Embedding(cfg.num_train_frame, 128)
        # # weight1 = torch.zeros([cfg.num_train_frame,64], dtype=torch.float)
        # # self.latentdeform = nn.Embedding.from_pretrained(weight1, freeze=False)
        # D = 8
        # self.deformskips = [4]
        # defW = 1024
        # layers = [nn.Linear(128, defW)]  # node coding + latent code
        # for i in range(D - 1):
            # layer = nn.Linear
            # in_channels = defW
            # if i in self.deformskips:
                # in_channels += 128
            # layers += [layer(in_channels, defW)]

        # self.smpldeformpara_linears = nn.ModuleList(layers)
        # # # self.deformpara_rotatelinear = nn.Linear(defW, 6)  # 12,self.modelnodenum *
        # # # self.deformpara_transllinear = nn.Linear(defW, 3)#self.modelnodenum *
        # # self.displacelinear  = nn.Linear(defW, 3)
        # # torch.nn.init.constant(self.displacelinear.weight, 0)
        # # torch.nn.init.constant(self.displacelinear.bias, 0)
        # # self.deformpara_linears = nn.ModuleList([nn.Linear(256, 512), nn.Linear(512, 1024)])
        # self.smpldeformpara_finallinear = nn.Linear(defW, self.smplnodenum * 6)
        
    def deformationrigidloss(self):
        # rigid constraint loss
        a1Ta2 = torch.matmul(self.deformation_affine[:, :, :, 0][...,None].transpose(3, 2), self.deformation_affine[:, :, :, 1][...,None])
        a1Ta2 = a1Ta2.view(-1, self.modelnodenum)
        a1Ta2ls = torch.sum(a1Ta2 ** 2)
        a2Ta3 = torch.matmul(self.deformation_affine[:, :, :, 1][...,None].transpose(3, 2), self.deformation_affine[:, :, :, 2][...,None])
        a2Ta3 = a2Ta3.view(-1, self.modelnodenum)
        a2Ta3ls = torch.sum(a2Ta3 ** 2)
        a3Ta1 = torch.matmul(self.deformation_affine[:, :, :, 2][...,None].transpose(3, 2), self.deformation_affine[:, :, :, 0][...,None])
        a3Ta1 = a3Ta1.view(-1, self.modelnodenum)
        a3Ta1ls = torch.sum(a3Ta1 ** 2)
        a1Ta1 = torch.matmul(self.deformation_affine[:, :, :, 0][...,None].transpose(3, 2), self.deformation_affine[:, :, :, 0][...,None])
        a1Ta1 = a1Ta1.view(-1, self.modelnodenum)
        a1Ta1ls = torch.sum((1 - a1Ta1) ** 2)
        a2Ta2 = torch.matmul(self.deformation_affine[:, :, :, 1][...,None].transpose(3, 2), self.deformation_affine[:, :, :, 1][...,None])
        a2Ta2 = a2Ta2.view(-1, self.modelnodenum)
        a2Ta2ls = torch.sum((1 - a2Ta2) ** 2)
        a3Ta3 = torch.matmul(self.deformation_affine[:, :, :, 2][...,None].transpose(3, 2), self.deformation_affine[:, :, :, 2][...,None])
        a3Ta3 = a3Ta3.view(-1, self.modelnodenum)
        a3Ta3ls = torch.sum((1 - a3Ta3) ** 2)
        rigidloss = a1Ta2ls + a2Ta3ls + a3Ta1ls + a1Ta1ls + a2Ta2ls + a3Ta3ls

        return rigidloss

    def deformationsmoothloss(self):
        #smooth constrain loss
        #repmodelnodepos = self.modelnodepos.repeat(self.modelnodeedgenum, 1)  # (nodenum*edgenum)*3
        repmodelnodepos = self.modelnodepos.unsqueeze(1).repeat(1, self.modelnodeedgenum, 1)  # nodenum*edgenum*3
        repmodelnodepos = repmodelnodepos.view([-1, 3])  # (nodenum*edgenum)*3

        self.modelnodeedge = self.modelnodeedge.view([-1])  # (nodenum*edgenum)
        relativepos = repmodelnodepos - self.modelnodepos[self.modelnodeedge, :]  # (nodenum*edgenum)*3
        relativepos = relativepos[None, ..., None]
        deformrelativepos = torch.matmul(self.deformation_affine[:, self.modelnodeedge, :, :],
                                         relativepos)  # B*(nodenum*edgenum)*3*3, #B*(nodenum*edgenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.modelnodepos[self.modelnodeedge,
                                                    :][None, ...] + self.deformation_transl[:, self.modelnodeedge, :]
        deformpos = deformpos.view(-1, self.modelnodenum*self.modelnodeedgenum, 3)  # B*(nodenum*edgenum)*3
        #repnodetransl = self.deformation_transl.repeat(1, self.modelnodeedgenum, 1)# B*(nodenum*edgenum)*3
        repnodetransl = self.deformation_transl.unsqueeze(2).repeat(1, 1, self.modelnodeedgenum, 1)  # B*nodenum*edgenum*3
        repnodetransl = repnodetransl.view([-1, self.modelnodenum*self.modelnodeedgenum, 3])  # B*(nodenum*edgenum)*3

        smoothpos = deformpos - (repmodelnodepos[None, ...] + repnodetransl)
        smoothpos = smoothpos.view(-1, self.modelnodenum, self.modelnodeedgenum, 3)# B*nodenum*edgenum*3
        smoothpos = smoothpos**2
        # weightedsmoothpos = smoothpos * self.modelnode_edgeweight[None, ..., None]  # nodeweight: B*nodenum*edgenum*1
        # weightedsmoothpos = torch.sum(weightedsmoothpos, dim=2)  # B*nodenum*1*3
        # weightedsmoothpos = weightedsmoothpos.squeeze(2)  # B*nodenum*3

        smoothloss = torch.sum(smoothpos)
        return smoothloss

    def deformationsmoothloss_smpl(self):
        #smooth constrain loss
        #repmodelnodepos = self.modelnodepos.repeat(self.modelnodeedgenum, 1)  # (nodenum*edgenum)*3
        repmodelnodepos = self.smplnodepos.unsqueeze(1).repeat(1, self.smplnodeedgenum, 1)  # nodenum*edgenum*3
        repmodelnodepos = repmodelnodepos.view([-1, 3])  # (nodenum*edgenum)*3

        self.smplnodeedge = self.smplnodeedge.view([-1])  # (nodenum*edgenum)
        relativepos = repmodelnodepos - self.smplnodepos[self.smplnodeedge, :]  # (nodenum*edgenum)*3
        relativepos = relativepos[None, ..., None]
        deformrelativepos = torch.matmul(self.deformation_affine_smpl[:, self.smplnodeedge, :, :],
                                         relativepos)  # B*(nodenum*edgenum)*3*3, #B*(nodenum*edgenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.smplnodepos[self.smplnodeedge,
                                                    :][None, ...] + self.deformation_transl_smpl[:, self.smplnodeedge, :]
        deformpos = deformpos.view(-1, self.smplnodenum*self.smplnodeedgenum, 3)  # B*(nodenum*edgenum)*3
        #repnodetransl = self.deformation_transl.repeat(1, self.modelnodeedgenum, 1)# B*(nodenum*edgenum)*3
        repnodetransl = self.deformation_transl_smpl.unsqueeze(2).repeat(1, 1, self.smplnodeedgenum, 1)  # B*nodenum*edgenum*3
        repnodetransl = repnodetransl.view([-1, self.smplnodenum*self.smplnodeedgenum, 3])  # B*(nodenum*edgenum)*3

        smoothpos = deformpos - (repmodelnodepos[None, ...] + repnodetransl)
        smoothpos = smoothpos.view(-1, self.smplnodenum, self.smplnodeedgenum, 3)# B*nodenum*edgenum*3
        smoothpos = smoothpos**2
        # weightedsmoothpos = smoothpos * self.modelnode_edgeweight[None, ..., None]  # nodeweight: B*nodenum*edgenum*1
        # weightedsmoothpos = torch.sum(weightedsmoothpos, dim=2)  # B*nodenum*1*3
        # weightedsmoothpos = weightedsmoothpos.squeeze(2)  # B*nodenum*3

        smoothloss = torch.sum(smoothpos)
        return smoothloss
        
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

    def predicting_deformation(self, sp_input):
        # latent = self.encoder(sp_input['img'].transpose(1,3))
        latent = self.latentdeform(sp_input['latent_index'].to(torch.int64))  # .type(torch.LongTensor).to(self.device)np.asscalar(np.int16(sp_input['latent_index']))
        h = latent
 
        for i, l in enumerate(self.deformpara_linears):
           h = self.deformpara_linears[i](h)
           h = F.relu(h)
           if i in self.deformskips:
               h = torch.cat([latent, h], -1)
        # modelnodepos = sp_input['vert'][:, self.nodevidx, :]
        # node_pts = embedder.node_embedder(modelnodepos)
        ##node_pts = node_pts * self.embedweight
        # latent = latent[..., None].expand(*latent.shape, node_pts.size(1))
        # latent = latent.transpose(-2, -1)
        # node_features = torch.cat((node_pts, latent), dim=2)

        # h = node_features#latent
        # for i, l in enumerate(self.deformpara_linears):
        #    h = self.deformpara_linears[i](h)
        #    h = F.relu(h)
        #    if i in self.deformskips:
        #        h = torch.cat([node_pts, h], -1)

        h = self.deformpara_finallinear(h)
        h = h.view(-1,self.modelnodenum,6)#

        # h1 = self.deformpara_rotatelinear(h)
        # deformation_affine = h1.view(-1,self.modelnodenum,6)
        # h2 = self.deformpara_transllinear(h)
        # deformation_transl = h2.view(-1,self.modelnodenum,3)

        # latent = self.latentdeform(sp_input['latent_index'].to(torch.int64))
        # h = latent.view(-1, self.modelnodenum, 6)
        deformation_affine, deformation_transl = torch.split(h, [3, 3], dim=-1)  # 9
        # deformation_rotate = self.rot6d_to_rotmat(deformation_affine.view([-1, 6]))
        deformation_rotate = self.batch_rodrigues(deformation_affine.view([-1, 3]))  # (B*nodenum)*3-->(B*nodenum)*3*3
        # self.deformation_affine = deformation_rotate.view(-1, self.modelnodenum, 3, 3)
        self.deformation_affine = deformation_rotate.view(-1, self.modelnodenum, 3, 3)#deformation_affine
        self.deformation_transl = deformation_transl.view(-1, self.modelnodenum, 3)
        # self.deformation_affine = deformation_rotate.repeat(self.modelnodenum, 1, 1)[None, ...].to(self.device)
        # self.deformation_transl = deformation_transl.repeat(self.modelnodenum, 1)[None, ...].to(self.device)
        # initR = sp_input['initRT'][:,:3,:].float()
        # initT = sp_input['initRT'][:, 3, :].float()
        # self.deformation_affine = torch.matmul(deformation_rotate,initR)
        # self.deformation_transl = torch.matmul(deformation_rotate,initT.unsqueeze(-1)).squeeze(-1)+deformation_transl

        # frame_index = sp_input['latent_index'].item()
        # if frame_index == 0:
            # self.deformation_affine = torch.eye(3).unsqueeze(0).repeat(self.modelnodenum, 1, 1)[None, ...].to(self.device)
            # self.deformation_transl = torch.zeros([1,3]).repeat(self.modelnodenum, 1)[None, ...].to(self.device)
            # self.deformation_affine = torch.eye(3)[None, ...].to(self.device)
            # self.deformation_transl = torch.zeros([1, 3]).to(self.device)
        # px = np.loadtxt('F:/neuralbody-master/tools/data/people_snapshot/female-3-casual/px.txt')
        # px = torch.FloatTensor(px).to(self.device)
        # h = px.view(-1, self.modelnodenum, 12)
        # deformation_affine, deformation_transl = torch.split(h, [9, 3], dim=-1)
        # self.deformation_affine = deformation_affine.view(-1, self.modelnodenum, 3, 3)
        # self.deformation_transl = deformation_transl.view(-1, self.modelnodenum, 3)
        # print(self.deformation_affine[0].detach().cpu().numpy())
        # print(self.deformation_transl[0].detach().cpu().numpy())
        return self.deformation_affine, self.deformation_transl
     
    def predicting_deformation_smpl(self, sp_input):
        # latent = self.encoder(sp_input['img'].transpose(1,3))
        latent = self.smpllatentdeform(sp_input['latent_index'].to(torch.int64))  # .type(torch.LongTensor).to(self.device)np.asscalar(np.int16(sp_input['latent_index']))
        h = latent
        for i, l in enumerate(self.smpldeformpara_linears):
           h = self.smpldeformpara_linears[i](h)
           h = F.relu(h)
           if i in self.deformskips:
               h = torch.cat([latent, h], -1)
       
        h = self.smpldeformpara_finallinear(h)
        h = h.view(-1,self.smplnodenum,6)#
       
        deformation_affine, deformation_transl = torch.split(h, [3, 3], dim=-1)  # 9
        # deformation_rotate = self.rot6d_to_rotmat(deformation_affine.view([-1, 6]))
        deformation_rotate = self.batch_rodrigues(deformation_affine.view([-1, 3]))  # (B*nodenum)*3-->(B*nodenum)*3*3

        self.deformation_affine_smpl = deformation_rotate.view(-1, self.smplnodenum, 3, 3)#deformation_affine
        self.deformation_transl_smpl = deformation_transl.view(-1, self.smplnodenum, 3)
       
        return self.deformation_affine_smpl, self.deformation_transl_smpl
             
    def deformingtemplate(self):
        #deforming template on all nodes once
        #reptemplateshape = self.templateshape.repeat(self.modelvert_nodenum,1)# (vtnum*nodenum)*3
        reptemplateshape = self.templateshape.unsqueeze(1).repeat(1, self.modelvert_nodenum, 1)  # vtnum*nodenum*3
        # NOTE: 变形3dgs顶点
        reptemplateshape = reptemplateshape.view([-1, 3])# (vtnum*nodenum)*3
        self.modelvert_node =self.modelvert_node.view([-1])# (vtnum*nodenum)
        relativepos = reptemplateshape - self.modelnodepos[self.modelvert_node, :]  # (vtnum*nodenum)*3
        relativepos = relativepos[None, ..., None]
        deformrelativepos = torch.matmul(self.deformation_affine[:, self.modelvert_node, :, :],
                                        relativepos)  # B*(vtnum*nodenum)*3*3, #B*(vtnum*nodenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.modelnodepos[self.modelvert_node,
                                                   :][None, ...] + self.deformation_transl[:, self.modelvert_node, :]#B*(vtnum*nodenum)*3
        deformpos = deformpos.view(-1,self.vtnum,self.modelvert_nodenum,3)#B*vtnum*nodenum*3
        weighteddeformpos = deformpos*self.modelvert_nodeweight[None, ..., None]#nodeweight: B*vtnum*nodenum*1
        weighteddeformpos = torch.sum(weighteddeformpos, dim=2)#B*vtnum*1*3
        self.deformedverts = weighteddeformpos.squeeze(2)#B*vtnum*3

        return self.deformedverts
     
    def deformingtemplate_smpl(self):
        #deforming template on all nodes once
        #reptemplateshape = self.templateshape.repeat(self.modelvert_nodenum,1)# (vtnum*nodenum)*3
        
        reptemplateshape = self.templatesmpl.unsqueeze(1).repeat(1, self.smplvert_nodenum, 1)  # vtnum*nodenum*3
        reptemplateshape = reptemplateshape.view([-1, 3])# (vtnum*nodenum)*3
        self.smplvert_node =self.smplvert_node.view([-1])# (vtnum*nodenum)
        relativepos = reptemplateshape - self.smplnodepos[self.smplvert_node, :]  # (vtnum*nodenum)*3
        relativepos = relativepos[None, ..., None]
        deformrelativepos = torch.matmul(self.deformation_affine_smpl[:, self.smplvert_node, :, :],
                                        relativepos)  # B*(vtnum*nodenum)*3*3, #B*(vtnum*nodenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.smplnodepos[self.smplvert_node,
                                                   :][None, ...] + self.deformation_transl_smpl[:, self.smplvert_node, :]#B*(vtnum*nodenum)*3
        deformpos = deformpos.view(-1,self.smplvtnum,self.smplvert_nodenum,3)#B*vtnum*nodenum*3
        weighteddeformpos = deformpos*self.smplvert_nodeweight[None, ..., None]#nodeweight: B*vtnum*nodenum*1
        weighteddeformpos = torch.sum(weighteddeformpos, dim=2)#B*vtnum*1*3
        smplgraphdeformedverts = weighteddeformpos.squeeze(2)#B*vtnum*3
        
        return smplgraphdeformedverts

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

    def get_rigid_transformation0(self, poses, joints, parents):
        """
        poses: 24 x 3
        joints: 24 x 3
        parents: 24
        """
        rot_mats = batch_rodrigues(poses.reshape(-1, 3))

        # obtain the relative joints
        rel_joints = joints.clone()
        rel_joints[1:] -= joints[parents[1:]]

        # create the transformation matrix
        transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=2)
        padding = torch.zeros([24, 1, 4]).cuda()
        padding[..., 3] = 1
        transforms_mat = torch.cat([transforms_mat, padding], dim=1)

        # rotate each part
        transform_chain = [transforms_mat[0]]
        for i in range(1, parents.shape[0]):
            curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[i])#torch.dot()
            transform_chain.append(curr_res)
        transforms = torch.stack(transform_chain, dim=0)

        # obtain the rigid transformation
        padding = torch.zeros([24, 1]).cuda()
        joints_homogen = torch.cat([joints, padding], dim=1)
        transformed_joints = torch.sum(transforms * joints_homogen[:, None], dim=2)
        transforms[..., 3] = transforms[..., 3] - transformed_joints
        #transforms = transforms.astype(torch.float32)

        return transforms
    
    def batch_rodrigues_np(self, poses):
        """ poses: N x 3
        """
        batch_size = poses.shape[0]
        angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
        rot_dir = poses / angle

        cos = np.cos(angle)[:, None]
        sin = np.sin(angle)[:, None]

        rx, ry, rz = np.split(rot_dir, 3, axis=1)
        zeros = np.zeros([batch_size, 1])
        K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1)
        K = K.reshape([batch_size, 3, 3])

        ident = np.eye(3)[None]
        rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

        return rot_mat
    
    def get_rigid_transformation(self, poses, joints, parents):
        """
        poses: 24 x 3
        joints: 24 x 3
        parents: 24
        """
        rot_mats = self.batch_rodrigues_np(poses)

        # obtain the relative joints
        rel_joints = joints.copy()
        rel_joints[1:] -= joints[parents[1:]]

        # create the transformation matrix
        transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
        padding = np.zeros([24, 1, 4])
        padding[..., 3] = 1
        transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

        # rotate each part
        transform_chain = [transforms_mat[0]]
        for i in range(1, parents.shape[0]):
            curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
            transform_chain.append(curr_res)
        transforms = np.stack(transform_chain, axis=0)

        # obtain the rigid transformation
        padding = np.zeros([24, 1])
        joints_homogen = np.concatenate([joints, padding], axis=1)
        transformed_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
        transforms[..., 3] = transforms[..., 3] - transformed_joints
        transforms = transforms.astype(np.float32)

        return transforms
    
    def deformingtemplate_lbs(self, beta, theta, trans):
        #theta0 = theta.clone()

        #theta0[:,0:3] = 0

        theta = batch_rodrigues(theta.reshape(-1, 3)).reshape(1, 24, 3, 3)
        _theta = theta.reshape(1, 24, 3, 3)
        # so = self.smpl_model(betas = beta.reshape(1, 10), body_pose = _theta[:, 1:], global_orient = _theta[:, 0].view(1, 1, 3, 3))
        # smpl_v = so['vertices'].clone().reshape(-1, 3)
        
        _, rel_transforms = smplx.lbs.batch_rigid_transform(_theta,  torch.from_numpy(self.joints[None]).cuda(), self.parents)
        #theta_rodrigues = batch_rodrigues(theta0.reshape(-1, 3)).reshape(1, 24, 3, 3)

        #init_J = get_J(beta.reshape(1, 10), self.smpl_model)

        #_, rel_transforms = self.batch_rigid_transform(theta_rodrigues, init_J)
        # joints = torch.from_numpy(init_J.reshape(-1, 3)).cuda()
        # rel_transforms = self.get_rigid_transformation(theta0.reshape(-1, 3).detach().cpu().numpy(), self.joints, self.parents)
        # rel_transforms = torch.from_numpy(rel_transforms).cuda()self.smplbwsmpl_model.lbs_weights

        smpl_A = torch.matmul(self.smplbw.view(-1, self.num_joints), rel_transforms.view(self.num_joints, 16)).view(-1, 4, 4)

        R = smpl_A[:, :3, :3]#including global rotation
        #pts = torch.sum(R * self.templatesmpl[:, :, None], dim=-1)
        pts = torch.matmul(R, self.templatesmpl[:, :, None])
        pts = pts.squeeze(-1) + smpl_A[:, :3, 3]
        deformedsmplvert = pts + trans#.unsqueeze(1)torch.matmul(pts, gR[0].transpose(0, 1))
        
        device, dtype = beta.device, beta.dtype
        # homogen_coord = torch.ones([self.templatesmpl.shape[0], 1],
                               # dtype=dtype, device=device)
        # v_posed_homo = torch.cat([self.templatesmpl, homogen_coord], dim=1)
        # v_homo = torch.matmul(smpl_A, torch.unsqueeze(v_posed_homo, dim=-1))
        # deformedsmplvert = v_homo[:, :3, 0]  + trans 
        
        cloth_A = torch.bmm(self.bw.view(1,-1, self.num_joints), rel_transforms.view(1,self.num_joints, 16)).view(-1, 4, 4)
        
        homogen_coord = torch.ones([self.templateshape.shape[0], 1],
                               dtype=dtype, device=device)
        v_posed_homo = torch.cat([self.templateshape, homogen_coord], dim=1)
        v_homo = torch.matmul(cloth_A, torch.unsqueeze(v_posed_homo, dim=-1))
        deformedclothvert = v_homo[:, :3, 0] + trans  
        # R = cloth_A[:, :3, :3]#including global rotation
        # pts = torch.sum(R * self.templateshape[:, :, None], dim=-1)
        # pts = pts.squeeze(-1) + cloth_A[:, :3, 3]
        # #deformedclothvert = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']
        # deformedclothvert = pts + trans#.unsqueeze(1)torch.matmul(pts, gR[0].transpose(0, 1))self.templatesmpl, self.templateshape #
        
        return deformedsmplvert, deformedclothvert
    
    def deforming_lbs_newsmpl(self, beta, theta, trans, canonicalsmpl):
        #theta0 = theta.clone()

        #theta0[:,0:3] = 0
        self.smpl_model = init_smpl(
            model_folder = 'smpl_models',
            model_type = 'smpl',
            gender = 'neutral',
            num_betas = 10
        )
        parents = self.smpl_model.parents.cpu().numpy()
        self.parents = parents
        self.num_joints = parents.shape[0]
        
        theta = batch_rodrigues(theta.reshape(-1, 3)).reshape(1, 24, 3, 3)
        _theta = theta.reshape(1, 24, 3, 3)
        # so = self.smpl_model(betas = beta.reshape(1, 10), body_pose = _theta[:, 1:], global_orient = _theta[:, 0].view(1, 1, 3, 3))
        # smpl_v = so['vertices'].clone().reshape(-1, 3)
        
        init_J = get_J(beta.reshape(1, 10), self.smpl_model)
        _, rel_transforms = smplx.lbs.batch_rigid_transform(_theta, init_J, self.parents)
        #theta_rodrigues = batch_rodrigues(theta0.reshape(-1, 3)).reshape(1, 24, 3, 3)

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
        
    def deformingsmpl_LBS(self, sp_input):

        #embedded deformation on template shape in T pose
        personsmpl = self.templatesmpl[None,...]

        #deforming with LBS of SMPL further
        sh = personsmpl.shape
        #bw = self.bw[None,...]
        A = torch.bmm(self.smplbw, sp_input['A'].view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]#not including global rotation
        pts = torch.sum(R * personsmpl[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        self.deformedpersonsmpl = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']

        return self.deformedpersonsmpl
    
    def deformingcloth_graphdeform_disp_LBS(self, template, cloth_dis, vert_disp, sp_input):

        #embedded deformation on template shape in T pose
        # graphdeformedverts = self.deformingtemplate()

        #self.displace = torch.zeros_like(graphdeformedverts).to(graphdeformedverts)#
        # self.displace = self.displace_network(graphdeformedverts, sp_input)
        # NOTE: 使用snug预测的结果作为位移
        self.displace = cloth_dis
        self.vert_disp = vert_disp
        # graphdeformedverts_disp = graphdeformedverts + self.displace
        graphdeformedverts = template.unsqueeze(0)
        graphdeformedverts_disp = graphdeformedverts + self.displace + self.vert_disp
        sh = graphdeformedverts.shape
        #bw = self.bw[None,...]
        A = torch.bmm(self.bw, sp_input['A'].view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]#not including global rotation
        pts = torch.sum(R * graphdeformedverts_disp[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        if cfg.train_dataset.human == "0080":
            self.deformedcloth = pts + sp_input['Th']
        else:
            self.deformedcloth = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']
        
        
        #deforming with LBS of SMPL further
        # sh = graphdeformedverts.shape
        # #bw = self.bw[None,...]
        
        # A = torch.bmm(self.bw, sp_input['A'].view(sh[0], 24, -1))
        
        # A = A.view(sh[0], -1, 4, 4)
        # R = A[..., :3, :3]
        # pts = torch.sum(R * graphdeformedverts_disp[:, :, None], dim=3)
        # pts = pts + A[..., :3, 3]
        
        # self.deformedcloth = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']
        
        # pts = torch.sum(R * graphdeformedverts[:, :, None], dim=3)
        # pts = pts + A[..., :3, 3]        
        # deformedcloth_nodisp = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']
        
        return self.deformedcloth, graphdeformedverts, graphdeformedverts_disp, self.displace #, deformedcloth_nodisp
         
    def deformingcloth_graphdeform_LBS(self, sp_input):

        #embedded deformation on template shape in T pose
        graphdeformedverts = self.deformingtemplate()
        # graphdeformedverts = self.templateshape.unsqueeze(0)

        #deforming with LBS of SMPL further
        sh = graphdeformedverts.shape
        
        A = torch.bmm(self.bw, sp_input['A'].view(sh[0], 24, -1))
        
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]
        pts = torch.sum(R * graphdeformedverts[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        
        if cfg.train_dataset.human == "0080":
            self.deformedcloth = pts + sp_input['Th']
        else:
            self.deformedcloth = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']
        
        return self.deformedcloth, graphdeformedverts

    def deformingsmpl_graphdeform_LBS_dis(self, sp_input, displace):

        self.smplgraphdeformedverts = self.templatesmpl.unsqueeze(0)
        self.smpldisplace = displace
        self.smplgraphdeformedverts = self.smplgraphdeformedverts + self.smpldisplace
        
        sh = self.smplgraphdeformedverts.shape

        A = torch.bmm(self.smplbw, sp_input['A'].view(sh[0], 24, -1))
        # A = torch.bmm(self.smplbw, A)
        
        
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]
        pts = torch.sum(R * self.smplgraphdeformedverts[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        
        # self.deformedsmpl = torch.matmul(pts, sp_input['R'].transpose(1, 2)) # + sp_input['Th']
        if cfg.train_dataset.human == "0080":
            self.deformedsmpl = pts + sp_input['Th']
        else:
            self.deformedsmpl = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']
        
        _,trinormal = mesh_face_areas_normals(self.smplgraphdeformedverts.view(-1,3), self.desmplfaces)
         
        self.deformedsmpl_vertnorm = trinormal[self.desmpl_vfidx,:]
        
        return self.deformedsmpl, self.smplgraphdeformedverts
    
    def deformingsmpl_graphdeform_LBS(self, sp_input):

        #embedded deformation on template shape in T pose
        self.smplgraphdeformedverts = self.deformingtemplate_smpl()
        # self.smplgraphdeformedverts = self.templatesmpl.unsqueeze(0)
        
        # DEBUG:
        # npvertices = self.smplgraphdeformedverts[0].detach().cpu().numpy()
        # npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeform/smpltri.txt')) - 1
        # mesh = trimesh.Trimesh(npvertices, npfaces)
        # import smplx
        # smpl_model = smplx.SMPL("tools/data/SMPL/SMPL_FEMALE.pkl").to("cuda:0")
        # with torch.no_grad():
        #     smpl_output = smpl_model(
        #         betas = torch.zeros(1, 10).to("cuda:0"),
        #         body_pose = torch.zeros(1, 69).to("cuda:0"),
        #         # body_pose = fullpose[0][1:].reshape([1, -1]),
        #         global_orient = torch.zeros(1, 3).to("cuda:0"),
        #         # transl = transl[0].to("cuda:0"),
        #         return_verts=True
        #     )
        # smplmesh = trimesh.Trimesh(vertices=smpl_output.vertices[0].cpu().numpy(), faces=npfaces)
        # mesh = trimesh.util.concatenate([mesh, smplmesh])
        # mesh.show()
        # DEBUG: END

        #deforming with LBS of SMPL further
        sh = self.smplgraphdeformedverts.shape
        #bw = self.bw[None,...]
        
        # full_pose = torch.cat([sp_input['Rh'].reshape(-1, 3), sp_input['smplpose'][:, 3:]], dim=-1)
        # J_transformed, joint_transforms = get_transformed_joints(sp_input['smplshape'], 
        #                                                             # sp_input['smplpose'], 
        #                                                             full_pose,
        #                                                             self.smpl_model.v_template,
        #                                                             self.smpl_model.shapedirs,
        #                                                             self.smpl_model.J_regressor,
        #                                                             self.smpl_model.parents)
        
        # A = joint_transforms.reshape([-1,24,16])
        A = torch.bmm(self.smplbw, sp_input['A'].view(sh[0], 24, -1))
        # A = torch.bmm(self.smplbw, A)
        
        
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]
        pts = torch.sum(R * self.smplgraphdeformedverts[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        
        # self.deformedsmpl = torch.matmul(pts, sp_input['R'].transpose(1, 2)) # + sp_input['Th']
        if cfg.train_dataset.human == "0080":
            self.deformedsmpl = pts + sp_input['Th']
        else:
            self.deformedsmpl = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']
        
        _,trinormal = mesh_face_areas_normals(self.smplgraphdeformedverts.view(-1,3), self.desmplfaces)
         
        self.deformedsmpl_vertnorm = trinormal[self.desmpl_vfidx,:]
        
        return self.deformedsmpl, self.smplgraphdeformedverts
         
    def deformingcloth_graphdeform(self, sp_input):

        #embedded deformation on template shape in T pose
        graphdeformedverts = self.deformingtemplate()

        return graphdeformedverts
            
    def inversedeforming_samplepoints(self, wpts, sp_input):
    
        #finding the nearest point on the whole template for LBS
        templatevert = torch.cat([self.deformedpersonsmpl,self.deformedcloth],1)
        template_ptsdist = torch.cdist(wpts, templatevert, p=2)
        template_ptsdistmin = torch.min(template_ptsdist, 2)
        template_nnvidx = torch.squeeze(template_ptsdistmin[1], -1)  # B*P
        
        cloth_ptsdist = torch.cdist(wpts, self.deformedcloth, p=2)
        cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        cloth_nnvidx = torch.squeeze(cloth_ptsdistmin[1], -1)  # B*P
        
        pts = self.inversedeforming_samplepoints_LBS(wpts, template_nnvidx, templatevert, sp_input)

        graphdeformpts = self.inversedeforming_samplepoints_graphdeform(pts, cloth_nnvidx, sp_input)
        
        clothcormask = torch.where(torch.le(cloth_ptsdistmin[0],template_ptsdistmin[0])*torch.le(cloth_ptsdistmin[0],0.05), torch.ones_like(cloth_ptsdistmin[0]),torch.zeros_like(cloth_ptsdistmin[0]))
        deformpts = pts+(graphdeformpts-pts)*clothcormask[...,None]
        
        return deformpts
    
    def inversedeforming_samplepoints_layer(self, wpts, sp_input):
    
        #inverse deforming with smpl model, posedirs      
        smpl_ptsdist = torch.cdist(wpts, self.deformedsmpl, p=2)#deformedpersonsmpl
        smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        smpl_nnvidx = torch.squeeze(smpl_ptsdistmin[1], -1)  # B*P
        ptssmpl = self.inversedeforming_samplepoints_LBS(wpts, smpl_nnvidx, self.deformedsmpl, self.smplbw, sp_input)#posedirs, 
        
        smpl_invdeformpts = self.inversedeforming_samplepoints_graphdeform_smpl(ptssmpl, smpl_nnvidx, sp_input)
        
        #inverse deforming with cloth model 
        cloth_ptsdist = torch.cdist(wpts, self.deformedcloth, p=2)
        cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        cloth_nnvidx = torch.squeeze(cloth_ptsdistmin[1], -1)  # B*P     
        pts = self.inversedeforming_samplepoints_LBS(wpts, cloth_nnvidx, self.deformedcloth, self.bw, sp_input)
        
        pts = self.inversedeforming_samplepoints_displace(pts, cloth_nnvidx, sp_input)
        
        # cloth_invdeformpts = self.inversedeforming_samplepoints_graphdeform(pts, cloth_nnvidx, sp_input)
        cloth_invdeformpts = pts
        
        return smpl_invdeformpts, cloth_invdeformpts
    
    def inversedeforming_samplepoints_LBS_smpl(self, wpts, nnvidx, templatevert, bw, v_posed, sp_input):
        #inversely deforming sample points to cannonical frame
        ptsnum = wpts.size(1)
        
        templatevtnum = templatevert.size(1)

        #world points to posed points
        if cfg.train_dataset.human == "0080":
            pts = torch.matmul(wpts - sp_input['Th'])
        else:
            pts = torch.matmul(wpts - sp_input['Th'], sp_input['R'])
        
        #transform points from the pose space to the T pose

        #bw = self.bw[nnvidx.view([-1]),:].view(-1,ptsum,24) #bw: B, 24, n_points
        #bw = self.bw[None,...]#B, n_points, 24.permute(0, 2, 1)
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(wpts.size(0))]
        idx = torch.cat(idx).to(self.device)
        bwidx = nnvidx.view(-1) + idx
        bw1 = bw.view(-1, 24)  #self.templatebw
        selectbw = bw1[bwidx.long(), :]
        selectbw = selectbw.view(-1, ptsnum, 24)

        sh = pts.shape
        A = torch.bmm(selectbw, sp_input['A'].view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)#A: n_batch, 24, 4, 4
        pts = pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3])
        pts = torch.sum(R_inv * pts[:, :, None], dim=3)
                
        # Rs = batch_rodrigues(theta.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
        # pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        # v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, 3)
        selectv_posed = v_posed.view(-1, 3)[bwidx.long(), :].view(-1, ptsnum, 3)
        pts = pts - selectv_posed
        
        return pts
        
    def inversedeforming_samplepoints_LBS(self, wpts, nnvidx, templatevert, bw, sp_input):
        #inversely deforming sample points to cannonical frame
        ptsnum = wpts.size(1)
        
        templatevtnum = templatevert.size(1)

        #world points to posed points
        if cfg.train_dataset.human == "0080":
            pts = wpts - sp_input['Th']
        else:
            pts = torch.matmul(wpts - sp_input['Th'], sp_input['R'])
        #transform points from the pose space to the T pose

        #bw = self.bw[nnvidx.view([-1]),:].view(-1,ptsum,24) #bw: B, 24, n_points
        #bw = self.bw[None,...]#B, n_points, 24.permute(0, 2, 1)
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(wpts.size(0))]
        idx = torch.cat(idx).to(self.device)
        bwidx = nnvidx.view(-1) + idx
        bw1 = bw.view(-1, 24)  #self.templatebw
        selectbw = bw1[bwidx.long(), :]
        selectbw = selectbw.view(-1, ptsnum, 24)

        sh = pts.shape
        A = torch.bmm(selectbw, sp_input['A'].view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)#A: n_batch, 24, 4, 4
        pts = pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3].cpu()).to(self.device)
        pts = torch.sum(R_inv * pts[:, :, None], dim=3)
        
        return pts
    
    def inversedeforming_samplepoints_LBS_updatepose(self, wpts, nnvidx, templatevert, bw, sp_input):
        #inversely deforming sample points to cannonical frame
        ptsnum = wpts.size(1)
        
        templatevtnum = templatevert.size(1)

        #world points to posed points
        transl = self.displace(sp_input['latent_index'].to(torch.int64))
        #rot, transl = torch.split(latent, [3, 3], dim=-1) 
        #rotation = self.batch_rodrigues(rot.view([-1, 3]))
        h = transl.unsqueeze(1)
        #pts = torch.matmul(wpts - sp_input['Th']-h, rotation)
        pts = wpts - sp_input['Th']-h
        
        #bw = self.bw[nnvidx.view([-1]),:].view(-1,ptsum,24) #bw: B, 24, n_points
        #bw = self.bw[None,...]#B, n_points, 24.permute(0, 2, 1)
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(wpts.size(0))]
        idx = torch.cat(idx).to(self.device)
        bwidx = nnvidx.view(-1) + idx
        bw1 = bw.view(-1, 24)  #self.templatebw
        selectbw = bw1[bwidx.long(), :]
        selectbw = selectbw.view(-1, ptsnum, 24)

        sh = pts.shape
        #A = torch.bmm(selectbw, sp_input['A'].view(sh[0], 24, -1))
        A = torch.bmm(selectbw, self.J_A.view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)#A: n_batch, 24, 4, 4
        pts = pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3])
        pts = torch.sum(R_inv * pts[:, :, None], dim=3)
        
        return pts
    
    def inversedeforming_samplepoints_displace_smpl(self, pts, nnvidx, sp_input):
        ptsnum = pts.size(1)
        
        templatevtnum = self.smplgraphdeformedverts.size(0)

        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(pts.size(0))]
        idx = torch.cat(idx).to(self.device)
        nidx = nnvidx.view(-1) + idx
        
        selectv_posed = self.smpldisplace.view(-1, 3)[nidx.long(), :].view(-1, ptsnum, 3)
        pts = pts - selectv_posed
        
        return pts

    def inversedeforming_samplepoints_displace(self, pts, nnvidx, sp_input):
        ptsnum = pts.size(1)
        
        templatevtnum = self.templateshape.size(0)

        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(pts.size(0))]
        idx = torch.cat(idx).to(self.device)
        nidx = nnvidx.view(-1) + idx
        
        selectv_posed = self.vert_disp.view(-1, 3)[nidx.long(), :].view(-1, ptsnum, 3)
        pts = pts - selectv_posed
        selectv_posed = self.displace.view(-1, 3)[nidx.long(), :].view(-1, ptsnum, 3)
        pts = pts - selectv_posed
        
        return pts
        
    def inversedeforming_samplepoints_graphdeform(self, pts, nnvidx, sp_input):
        #inversely deforming sample points to cannonical frame
        ptsnum = pts.size(1)

        sh = pts.shape
       
        #inverse embedded deformation
        #repwpts = wpts.repeat(1, self.modelvert_nodenum, 1)  # B*(vtnum*nodenum)*3
        repwpts = pts.unsqueeze(2).repeat(1, 1, self.modelvert_nodenum, 1)# B*vtnum*nodenum*3
        self.modelvert_node = self.modelvert_node.view([-1,self.modelvert_nodenum])
        ptsnode = self.modelvert_node[nnvidx.view([-1]),:] # (B*P)*nodenum
        ptsnode = ptsnode.view([-1])# (B*P*nodenum); the influence nodes for each pt

        sh = ptsnum * self.modelvert_nodenum
        idx = [torch.full([sh], i * self.modelnodenum, dtype=torch.long) for i in range(pts.size(0))]
        idx = torch.cat(idx).to(self.device)
        ptsnodeidx = ptsnode + idx# batch idx of the influence nodes, used for retrieving deformation (affine and transl) of each batch sample
        ptsnodeidx = ptsnodeidx.long()
        deformtransl = self.deformation_transl.view(-1, 3)# (B*modelnodenum)*3
        selectdeformtransl = deformtransl[ptsnodeidx, :]
        selectdeformtransl = selectdeformtransl.view(-1, ptsnum,self.modelvert_nodenum, 3)  # B*(vtnum*nodenum)*3

        repwpts = repwpts.view([-1, ptsnum,self.modelvert_nodenum, 3])
        nodepos = self.modelnodepos[ptsnode, :].view([-1, ptsnum, self.modelvert_nodenum, 3])
        relativepos = repwpts - nodepos - selectdeformtransl  # B*vtnum*nodenum*3

        deformaffine = self.deformation_affine.view(-1, 3, 3)
        selectdeformaffine = deformaffine[ptsnodeidx, :, :]  # (B*vtnum*nodenum)*3*3
        selectdeformaffine = selectdeformaffine.view(-1, ptsnum, self.modelvert_nodenum, 3, 3)  # B*vtnum*nodenum*3*3
        deformednodepos = torch.matmul(selectdeformaffine, nodepos[..., None])# B*vtnum*nodenum*3*1
        relativepos = relativepos + deformednodepos.squeeze(-1)# B*vtnum*nodenum*3

        ptsnodeweight = self.modelvert_nodeweight[nnvidx.view([-1]), :]  # (B*vtnum)*nodenum
        ptsnodeweight = ptsnodeweight.view([-1, ptsnum, self.modelvert_nodenum,1])
        weightedrelativepts = relativepos * ptsnodeweight  # B*vtnum*nodenum*3
        weightedrelativepts = torch.sum(weightedrelativepts, dim=2)  # B*vtnum*1*3
        weightedrelativepts = weightedrelativepts.squeeze(2)

        weighteddeformaffine = selectdeformaffine * ptsnodeweight[...,None]
        weighteddeformaffine = torch.sum(weighteddeformaffine, dim=2)  # B*vtnum*1*3*3
        weighteddeformaffine = weighteddeformaffine.squeeze(2)
        a = weighteddeformaffine.view(-1, 3, 3)# (B*vtnum)*3*3
        c = a.cpu().inverse().to(self.device)
        inversedeformaffine = c.view(-1, ptsnum, 3, 3)# B*vtnum*3*3

        #inversedeformaffine = weighteddeformaffine.transpose(3,2)

        deformpts = torch.matmul(inversedeformaffine, weightedrelativepts[..., None])  # B*vtnum*3*1
        deformpts = deformpts.squeeze(-1)# B*vtnum*3

        return deformpts
    
    def inversedeforming_samplepoints_graphdeform_smpl(self, pts, nnvidx, sp_input):
        #inversely deforming sample points to cannonical frame
        ptsnum = pts.size(1)

        sh = pts.shape
       
        #inverse embedded deformation
        #repwpts = wpts.repeat(1, self.modelvert_nodenum, 1)  # B*(vtnum*nodenum)*3
        repwpts = pts.unsqueeze(2).repeat(1, 1, self.smplvert_nodenum, 1)# B*vtnum*nodenum*3
        self.smplvert_node = self.smplvert_node.view([-1,self.smplvert_nodenum])
        ptsnode = self.smplvert_node[nnvidx.view([-1]),:] # (B*P)*nodenum
        ptsnode = ptsnode.view([-1])# (B*P*nodenum); the influence nodes for each pt

        sh = ptsnum * self.smplvert_nodenum
        idx = [torch.full([sh], i * self.smplnodenum, dtype=torch.long) for i in range(pts.size(0))]
        idx = torch.cat(idx).to(self.device)
        ptsnodeidx = ptsnode + idx# batch idx of the influence nodes, used for retrieving deformation (affine and transl) of each batch sample
        ptsnodeidx = ptsnodeidx.long()
        deformtransl = self.deformation_transl_smpl.view(-1, 3)# (B*modelnodenum)*3
        selectdeformtransl = deformtransl[ptsnodeidx, :]
        selectdeformtransl = selectdeformtransl.view(-1, ptsnum,self.smplvert_nodenum, 3)  # B*(vtnum*nodenum)*3

        repwpts = repwpts.view([-1, ptsnum,self.smplvert_nodenum, 3])
        nodepos = self.smplnodepos[ptsnode, :].view([-1, ptsnum, self.smplvert_nodenum, 3])
        relativepos = repwpts - nodepos - selectdeformtransl  # B*vtnum*nodenum*3

        deformaffine = self.deformation_affine_smpl.view(-1, 3, 3)
        selectdeformaffine = deformaffine[ptsnodeidx, :, :]  # (B*vtnum*nodenum)*3*3
        selectdeformaffine = selectdeformaffine.view(-1, ptsnum, self.smplvert_nodenum, 3, 3)  # B*vtnum*nodenum*3*3
        deformednodepos = torch.matmul(selectdeformaffine, nodepos[..., None])# B*vtnum*nodenum*3*1
        relativepos = relativepos + deformednodepos.squeeze(-1)# B*vtnum*nodenum*3

        ptsnodeweight = self.smplvert_nodeweight[nnvidx.view([-1]), :]  # (B*vtnum)*nodenum
        ptsnodeweight = ptsnodeweight.view([-1, ptsnum, self.smplvert_nodenum,1])
        weightedrelativepts = relativepos * ptsnodeweight  # B*vtnum*nodenum*3
        weightedrelativepts = torch.sum(weightedrelativepts, dim=2)  # B*vtnum*1*3
        weightedrelativepts = weightedrelativepts.squeeze(2)

        weighteddeformaffine = selectdeformaffine * ptsnodeweight[...,None]
        weighteddeformaffine = torch.sum(weighteddeformaffine, dim=2)  # B*vtnum*1*3*3
        weighteddeformaffine = weighteddeformaffine.squeeze(2)
        a = weighteddeformaffine.view(-1, 3, 3)# (B*vtnum)*3*3
        c = a.cpu().inverse().to(self.device)
        inversedeformaffine = c.view(-1, ptsnum, 3, 3)# B*vtnum*3*3

        #inversedeformaffine = weighteddeformaffine.transpose(3,2)

        deformpts = torch.matmul(inversedeformaffine, weightedrelativepts[..., None])  # B*vtnum*3*1
        deformpts = deformpts.squeeze(-1)# B*vtnum*3

        return deformpts
        
    def barycentricmapping_observe2canon(self, wpts, canvert, posedverts, meshface, vertfaceidx):
        
        # coordinate construction for observed model
        v1 = posedverts[:, meshface[:, 0], :]
        v2 = posedverts[:, meshface[:, 1], :]
        v3 = posedverts[:, meshface[:, 2], :]
        d1 = v2 - v1
        d1 = F.normalize(d1, p=2, dim=2)
        d2 = v3 - v1
        d2 = F.normalize(d2, p=2, dim=2)
        d3 = torch.cross(d1, d2)  # direction in or out
        d3 = F.normalize(d3, p=2, dim=2)  # B*T*3
        posedcoord = torch.cat((d1[..., None], d2[..., None], d3[..., None]),
                               dim=3)  # B*T*3*3, each triangle coordinate has 3 direction, last dim is direction

        # nearest triangle for input points
        # vs = torch.squeeze(torch.mean(torch.cat(v1[...,None], v2[...,None], v3[...,None]),-1),-1)
        nwpts = torch.cdist(wpts, posedverts, p=2)

        # nv1 = v1.view(v1.size(0), -1, v1.size(1))#B*3*T
        # nwpts = torch.matmul(wpts,nv1)#B*P*T, wpts: B*P*3
        triidx1 = torch.squeeze(torch.min(nwpts, 2)[1], -1)  # B*P

        triidx = vertfaceidx[triidx1.view(-1)].view(-1, wpts.shape[1])

        sh = triidx.shape
        idx = [torch.full([sh[1]], i * v1.size(1), dtype=torch.long) for i in range(sh[0])]
        idx = torch.cat(idx).to(triidx)
        triidx = triidx.view(-1)
        triidx = triidx + idx
        posedcoord = posedcoord.view(-1, 3, 3)
        selectcoord = posedcoord[triidx, :, :]
        selectcoord = selectcoord.view(-1, sh[1], 3, 3)
        # selectcoord = posedcoord[triidx[:,0],triidx[:,1],:,:]# B*P*3*3
        # selectcoord = torch.index_select(posedcoord, 1, triidx)

        # the projected position in each direction of nearest triangle coord
        v1 = v1.view(-1, 3)
        nv1 = v1[triidx, :]
        nv1 = nv1.view(-1, sh[1], 3)
        # projcoord = torch.matmul((wpts-nv1).unsqueeze(2),torch.transpose(selectcoord, 3,2))# B*P*1*3, B*P*3*3->B*P*1*3
        projcoord = torch.matmul((wpts - nv1).unsqueeze(2), selectcoord)

        # coordinate construction for canonical model
        # canvert_detail = canvert_detail.view(batchsize, 6890, 3)
        v1 = canvert[:, meshface[:, 0], :]
        v2 = canvert[:, meshface[:, 1], :]
        v3 = canvert[:, meshface[:, 2], :]
        d1 = v2 - v1
        d1 = F.normalize(d1, p=2, dim=2)
        d2 = v3 - v1
        d2 = F.normalize(d2, p=2, dim=2)
        d3 = torch.cross(d1, d2)  # direction in or out
        d3 = F.normalize(d3, p=2, dim=2)
        cancoord = torch.cat((d1[..., None], d2[..., None], d3[..., None]), dim=3)  # B*T*3*3

        # obtaining the corresponding points of input wpts in canonical frame
        cancoord = cancoord.view(-1, 3, 3)
        canselectcoord = cancoord[triidx, :, :]
        canselectcoord = canselectcoord.view(-1, sh[1], 3, 3)
        # canselectcoord = cancoord[triidx, :, :]  # B*P*3*3
        # canselectcoord = torch.index_select(cancoord, 1, triidx)
        # canwpts = torch.matmul(projcoord, canselectcoord)  # B*P*1*3, B*P*3*3->B*P*1*3
        canwpts = torch.matmul(projcoord, torch.transpose(canselectcoord, 3, 2))
        canwpts = canwpts.squeeze(2)  # B*P*3
        v1 = v1.view(-1, 3)
        nv1 = v1[triidx, :]
        nv1 = nv1.view(-1, sh[1], 3)
        canwpts = canwpts + nv1  # absolute position in canonical frame

        return canwpts
        
    def inverseLBS_simulatedcloth(self, clothvert, sp_input):
        #inversely deforming sample points to cannonical frame
        ptsnum = clothvert.size(1)
        
        #world points to posed points
        pts = torch.matmul(clothvert - sp_input['Th'], sp_input['R'])
        #transform points from the pose space to the T pose

        sh = pts.shape
        A = torch.bmm(self.bw, sp_input['A'].view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)#A: n_batch, 24, 4, 4
        pts = pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3])
        pts = torch.sum(R_inv * pts[:, :, None], dim=3)
        
        return pts
    
    def LBS_simulatedcloth(self, clothvert, sp_input):

        sh = clothvert.shape
        A = torch.bmm(self.bw, sp_input['A'].view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]#not including global rotation
        pts = torch.sum(R * clothvert[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        if cfg.train_dataset.human == "0080":
            deformedclothvert = pts + sp_input['Th']
        else:
            deformedclothvert = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']

        return deformedclothvert
    
    def batch_global_rigid_transformation(self, Rs, Js, parent):
        N = Rs.shape[0]
        
        root_rotation = Rs[:, 0, :, :]
        Js = torch.unsqueeze(Js, -1)

        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            t_homo = torch.cat([t, torch.autograd.Variable(torch.ones(N, 1, 1)).cuda()], dim = 1)            
            return torch.cat([R_homo, t_homo], 2)
        
        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]

        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)
            results.append(res_here)

        results = torch.stack(results, dim = 1)

        new_J = results[:, :, :3, 3]
        Js_w0 = torch.cat([Js, torch.autograd.Variable(torch.zeros(N, 24, 1, 1)).cuda()], dim = 2)
        init_bone = torch.matmul(results, Js_w0)
        init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
        A = results - init_bone
        
        return A# new_J, A
        
    def LBS_simulatedcloth_updatepose(self, clothvert, sp_input):

        sh = clothvert.shape
        
        inctheta_frm = self.inctheta(sp_input['latent_index'].to(torch.int64))
        theta = sp_input['theta'] + inctheta_frm#including global rotation
        Rs = self.batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        self.J_A = self.batch_global_rigid_transformation(Rs, self.joints, self.parents)
        A = torch.bmm(self.bw, self.J_A.view(sh[0], 24, -1))
        
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]
        pts = torch.sum(R * clothvert[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        
        transl = self.displace(sp_input['latent_index'].to(torch.int64))
        #rot, transl = torch.split(latent, [3, 3], dim=-1) 
        #rotation = self.batch_rodrigues(rot.view([-1, 3]))
        h = transl.unsqueeze(1)
        deformedclothvert = pts+ sp_input['Th']+h
       
        return deformedclothvert
        
    def update_embeddedgraph(self, clothvert):
    
        self.templateshape = clothvert
        self.vtnum = self.templateshape.size(0)
        self.modelnodepos = self.templateshape[self.modelnodeidx, :]
    
    def update_canonsmpl(self, smplvert):
    
        self.templatesmpl = smplvert
        
    def inversedeforming_samplepoints_LBS_graphdeform(self, wpts, sp_input):
        #inversely deforming sample points to cannonical frame
        ptsnum = wpts.size(1)
        ptsdist = torch.cdist(wpts, self.deformedverts, p=2)
        nnvidx = torch.squeeze(torch.min(ptsdist, 2)[1], -1)  # B*P

        templatevtnum = self.deformedverts.size(1)

        #world points to posed points
        pts = torch.matmul(wpts - sp_input['Th'], sp_input['R'])
        #transform points from the pose space to the T pose

        #bw = self.bw[nnvidx.view([-1]),:].view(-1,ptsum,24) #bw: B, 24, n_points
        #bw = self.bw[None,...]#B, n_points, 24.permute(0, 2, 1)
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(wpts.size(0))]
        idx = torch.cat(idx).to(self.device)
        bwidx = nnvidx.view(-1) + idx
        bw1 = self.bw.view(-1, 24)  #
        selectbw = bw1[bwidx.long(), :]
        selectbw = selectbw.view(-1, ptsnum, 24)

        sh = pts.shape
        A = torch.bmm(selectbw, sp_input['A'].view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)#A: n_batch, 24, 4, 4
        pts = pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3])
        pts = torch.sum(R_inv * pts[:, :, None], dim=3)

        #inverse embedded deformation
        #repwpts = wpts.repeat(1, self.modelvert_nodenum, 1)  # B*(vtnum*nodenum)*3
        repwpts = pts.unsqueeze(2).repeat(1, 1, self.modelvert_nodenum, 1)# B*vtnum*nodenum*3
        self.modelvert_node = self.modelvert_node.view([-1,self.modelvert_nodenum])
        ptsnode = self.modelvert_node[nnvidx.view([-1]),:] # (B*P)*nodenum
        ptsnode = ptsnode.view([-1])# (B*P*nodenum); the influence nodes for each pt

        sh = ptsnum * self.modelvert_nodenum
        idx = [torch.full([sh], i * self.modelnodenum, dtype=torch.long) for i in range(pts.size(0))]
        idx = torch.cat(idx).to(self.device)
        ptsnodeidx = ptsnode + idx# batch idx of the influence nodes, used for retrieving deformation (affine and transl) of each batch sample
        ptsnodeidx = ptsnodeidx.long()
        deformtransl = self.deformation_transl.view(-1, 3)# (B*modelnodenum)*3
        selectdeformtransl = deformtransl[ptsnodeidx, :]
        selectdeformtransl = selectdeformtransl.view(-1, ptsnum,self.modelvert_nodenum, 3)  # B*(vtnum*nodenum)*3

        repwpts = repwpts.view([-1, ptsnum,self.modelvert_nodenum, 3])
        nodepos = self.modelnodepos[ptsnode, :].view([-1, ptsnum, self.modelvert_nodenum, 3])
        relativepos = repwpts - nodepos - selectdeformtransl  # B*vtnum*nodenum*3

        deformaffine = self.deformation_affine.view(-1, 3, 3)
        selectdeformaffine = deformaffine[ptsnodeidx, :, :]  # (B*vtnum*nodenum)*3*3
        selectdeformaffine = selectdeformaffine.view(-1, ptsnum, self.modelvert_nodenum, 3, 3)  # B*vtnum*nodenum*3*3
        deformednodepos = torch.matmul(selectdeformaffine, nodepos[..., None])# B*vtnum*nodenum*3*1
        relativepos = relativepos + deformednodepos.squeeze(-1)# B*vtnum*nodenum*3

        ptsnodeweight = self.modelvert_nodeweight[nnvidx.view([-1]), :]  # (B*vtnum)*nodenum
        ptsnodeweight = ptsnodeweight.view([-1, ptsnum, self.modelvert_nodenum,1])
        weightedrelativepts = relativepos * ptsnodeweight  # B*vtnum*nodenum*3
        weightedrelativepts = torch.sum(weightedrelativepts, dim=2)  # B*vtnum*1*3
        weightedrelativepts = weightedrelativepts.squeeze(2)

        weighteddeformaffine = selectdeformaffine * ptsnodeweight[...,None]
        weighteddeformaffine = torch.sum(weighteddeformaffine, dim=2)  # B*vtnum*1*3*3
        weighteddeformaffine = weighteddeformaffine.squeeze(2)
        a = weighteddeformaffine.view(-1, 3, 3)# (B*vtnum)*3*3
        c = a.inverse()
        inversedeformaffine = c.view(-1, ptsnum, 3, 3)# B*vtnum*3*3

        #inversedeformaffine = weighteddeformaffine.transpose(3,2)

        deformpts = torch.matmul(inversedeformaffine, weightedrelativepts[..., None])  # B*vtnum*3*1
        deformpts = deformpts.squeeze(-1)# B*vtnum*3

        return deformpts

def batch_rodrigues(theta):
    # theta N x 3
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat
          
class OccupancyNetwork(nn.Module):
    def __init__(self):
        super(OccupancyNetwork, self).__init__()

        self.actvn = nn.ReLU()

        self.skips = [4]
        D = 8
        W = 256
        input_ch = 63
        input_ch_views = 27
        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = W
            if i in self.skips:
                in_channels += input_ch
            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)
        self.alpha_linear = nn.Linear(W, 1)
        self.alpha_linear.bias.data.fill_(0.693)

    def forward(self, light_pts):
        #light_pts = embedder.xyz_embedder(wpts)
        h = light_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([light_pts, h], -1)
        alpha = self.alpha_linear(h)
        occupancy = 1 - torch.exp(-torch.relu(alpha))
        return occupancy, h
            
class ClothSimulation(nn.Module):
    def __init__(self):
        super(ClothSimulation, self).__init__()
        
        data_root = 'cloth/deformation'
        templatecloth_path = os.path.join(data_root, 'cloth/clothes_vert.txt')
        templatecloth = np.loadtxt(templatecloth_path)

        vtnum = templatecloth.shape[0]
        
        self.model = FullyConnected(
            input_size=72+10+4, output_size=vtnum * 3,
            num_layers=3,
            hidden_size=1024)
            
    def forward(self, thetas, betas, gammas):
    
        pred_verts = self.model(torch.cat((thetas, betas, gammas), dim=1))
        
        return pred_verts.view(thetas.shape[0], -1, 3)

class FullyConnected(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=None):
        super(FullyConnected, self).__init__()
        net = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.2),
        ]
        for i in range(num_layers - 2):
            net.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
            ])
        net.extend([
            nn.Linear(hidden_size, output_size),
        ])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
        
class ColorNetwork(nn.Module):
    def __init__(self):
        super(ColorNetwork, self).__init__()

        input_ch = 63
        input_ch_views = 27
        D = 8
        W = 256

        self.skips = [4]

        layers = [nn.Linear(input_ch, W)]#+1
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = W
            if i in self.skips:
                in_channels += input_ch#+1
            layers += [layer(in_channels, W)]
        self.pts_linears = nn.ModuleList(layers)
        
        #D = 3
        #W = 256
        #self.skips = []
        #layers = [nn.Linear(W, W)]
        #for i in range(D - 1):
        #    layer = nn.Linear
        #    in_channels = W
        #    if i in self.skips:
        #        in_channels += W
        #    layers += [layer(in_channels, W)]
        #self.prefeature_linear = nn.ModuleList(layers)input_ch_views + 

        self.views_linears = nn.ModuleList([nn.Linear(W, W // 2)])#input channel needs to change,
        self.feature_linear = nn.Linear(W, W)
        self.rgb_linear = nn.Linear(W // 2, 3)
        self.latent_fc = nn.Linear(384, 256)#384

        self.latent = nn.Embedding(cfg.num_train_frame, 128)

    def forward(self, light_pts, viewdir, sp_input):

        #light_pts = embedder.xyz_embedder(wpts)
        #input_h = h
        #for i, l in enumerate(self.prefeature_linear):
        #    h = self.prefeature_linear[i](h)
        #    h = F.relu(h)
        #    if i in self.skips:
        #        h = torch.cat([input_h, h], -1)
        
        n_batch, n_point = light_pts.shape[:2]
        # alpha = occupancy.reshape(-1, n_point)
        # # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        # weights = alpha * torch.cumprod(
            # torch.cat(
                # [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
                # -1), -1)[:, :-1]
        # weights = weights.view([n_batch, n_point,1])

        #input_h = torch.cat((light_pts, weights), dim=2)
        input_h = light_pts
        h = input_h
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_h, h], -1)
                
        features = self.feature_linear(h)

        latent = self.latent(sp_input['latent_index'])
        latent = latent[..., None].expand(*latent.shape, h.size(1))
        latent = latent.transpose(-2,-1)
        features = torch.cat((features, latent), dim=2)
        features = self.latent_fc(features)

        #viewdir = embedder.view_embedder(viewdir)
        #viewdir = viewdir.transpose(1, 2)

        #h = torch.cat((features, viewdir), dim=2)
        h = features
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)

        return rgb

class PointsRendererWithFrags(torch.nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.
    """

    def __init__(self, rasterizer, compositor):
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images,fragments