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
from torch import nn

import pytorch3d
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
from smpl_utils import init_smpl, get_J, get_shape_pose, batch_rodrigues, get_J_batch_cpu
import smplx
from smplx.lbs import transform_mat, blend_shapes

from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from skimage import io
    
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

def depth_double_to_normal(points1, points2):
    points = torch.stack([points1, points2],dim=0)
    output = torch.zeros_like(points)
    dx = points[...,2:, 1:-1] - points[...,:-2, 1:-1]
    dy = points[...,1:-1, 2:] - points[...,1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=1), dim=1)
    output[...,1:-1, 1:-1] = normal_map
    return output

def depth_to_points(depth, intrinsics, height, width):
    """
    将深度图转换为3D点云。
    
    参数:
    - depth: 深度图Tensor，形状为(1, height, width)
    - intrinsics: 相机内参Tensor，形状为(3, 3)
    - height, width: 深度图的尺寸
    
    返回:
    - points: 3D点云Tensor，形状为(N, 3)，其中N是点的数量
    """
    # 创建网格坐标
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width))
    xx = xx.float().to('cuda')
    yy = yy.float().to('cuda')
    
    # 将深度图转换为3D点云
    zz = depth[0]
    points = torch.stack([xx, yy, zz], dim=-1)  # Nx3
    
    # 转换到世界坐标系
    intrinsics_inv = torch.inverse(intrinsics)
    points = torch.matmul(points, intrinsics_inv.transpose(1,0))  # Nx3
    
    return points
    
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
        
        # with open('assets/smpl_template_sdf.npy', 'rb') as f:
            # sdf_voxels = np.load(f)
        # self.sdf_voxels = torch.from_numpy(sdf_voxels).reshape(1, 1, 128, 128, 128).cuda()
        # self.sdf_voxels = self.sdf_voxels.permute(0, 1, 4, 3, 2)
        
                    
        self.smpl_model = init_smpl(
            model_folder = 'smpl_models',
            model_type = 'smpl',
            gender = 'neutral',#
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
              
        self.l1loss = torch.nn.L1Loss()
               
        #self.submesh = subdivide_meshes.SubdivideMeshes()
        
        self.thetazero = batch_rodrigues(torch.zeros([8,72]).reshape(-1, 3)).reshape(-1, 24, 3, 3).to(self.device)
        
        so = self.smpl_model(betas = torch.zeros([8,10]).reshape(-1, 10).to(self.device), body_pose = self.thetazero[:, 1:], global_orient = self.thetazero[:, 0].view(-1, 1, 3, 3))
        self.canoncalmeansmpl = so['vertices'].clone().reshape(-1, 6890, 3)#mean smpl
        
        # mesh_test = trimesh.Trimesh(vertices=self.canoncalmeansmpl[0].detach().cpu().numpy(), faces=self.smplfaces[0].detach().cpu().numpy())
        # mesh_test.export("meansmpl.obj")
        
        facesegidx = np.loadtxt('lib/networks/modules/facesegidx.txt')
        self.facesegidx = torch.Tensor(facesegidx).to('cuda')
        faces_len = self.smplfaces.shape[1]
        gssegidx =  [torch.full([6], self.facesegidx[i], dtype=torch.long) for i in range(faces_len)]
        self.gssegidx = torch.cat(gssegidx).to('cuda')
        self.gsnum = self.gssegidx.shape[0]
        self.bodysegidx = self.gssegidx==0
        self.legsegidx = self.gssegidx==1
        self.headsegidx = self.gssegidx==2
        
        bodymaskimg = cv2.imread('lib/networks/modules/bodymask.png')
        self.bodymask = torch.from_numpy(bodymaskimg.transpose((2, 0, 1))).to(self.device)
        self.bodymask = self.bodymask/255
        headmaskimg = cv2.imread('lib/networks/modules/headmask.png')
        self.headmask = torch.from_numpy(headmaskimg.transpose((2, 0, 1))).to(self.device)
        self.headmask = self.bodymask/255
        legmaskimg = cv2.imread('lib/networks/modules/legmask.png')
        self.legmask = torch.from_numpy(legmaskimg.transpose((2, 0, 1))).to(self.device)
        self.legmask = self.legmask/255        
        #self.gspoints_meansmpl = self.net.sugar_model_smpl.getbatch_edited_points(self.canoncalmeansmpl)
        
        # min_xyz, _ = torch.min(self.canoncalmeansmpl, axis=1)
        # max_xyz, _ = torch.max(self.canoncalmeansmpl, axis=1)
        # min_xyz = min_xyz - 0.3
        # max_xyz = max_xyz + 0.3
        
        # gridsample = 20000
               
        # vals = torch.rand(1,gridsample, 3).to(self.canoncalmeansmpl)
        # self.samplepts_smpl = (max_xyz[:,None] - min_xyz[:,None]) * vals + min_xyz[:,None]
        
        self.blendweights_gs = self.net.sugar_model_smpl.get_edited_blendweights(self.smpl_model.lbs_weights.view(-1, self.num_joints))
        
        self.resH = 512#256#256#
        self.resW = 256#256#128#
        
        self.highrestag = 1
        self.lowrestag = 1
        
            
        # self.cloth_simulation = ClothSimulation()
        # state_dict = torch.load('../gaussian-nerfcap/data/trained_model/cloth/lin.pth.tar')
        # self.cloth_simulation.model.load_state_dict(state_dict)
        # self.cloth_simulation = self.cloth_simulation.cuda()
        # pretrained_model = torch.load('../gaussian-nerfcap/data/trained_model/cloth/latest.pth')
        # self.tempclothpara0 = nn.Embedding.from_pretrained(pretrained_model['net']['tempclothpara.weight'], freeze=True)
        
        # self.cloth_simulation_prior = ClothSimulation()
        # self.cloth_simulation_prior.model.load_state_dict(state_dict)
        # self.cloth_simulation_prior = self.cloth_simulation_prior.cuda()
        # for param in self.cloth_simulation_prior.parameters():
            # param.requires_grad = False
        self.saveiter = 0
        
        # self.bw = np.loadtxt(os.path.join('./', 'cloth/skinweightnew.txt'))
        # self.bw = torch.Tensor(self.bw)[None, ...].to('cuda')
        
        # npfaces = np.loadtxt(os.path.join('./', 'cloth/clothes_face.txt')) - 1
        # self.clothfaces = torch.LongTensor(npfaces).to('cuda')
        # self.clothfaces = self.clothfaces[None, :, :]
        
        # # loading template deformation graph
        # templateshape_path = os.path.join('./', 'cloth/clothes_vert.txt')
        # templatecloth = np.loadtxt(templateshape_path)
        # self.templatecloth = torch.Tensor(templatecloth).to('cuda')
        
        # #combined mesh 
        # self.meshface = torch.cat([self.clothfaces, self.smplfaces+templatecloth.shape[0]], dim=1)
        
        # self.init_clothdeformgraph()    
        
        # self.clothpararange = torch.from_numpy(np.array([[-4, 4],[-2, 2]], dtype = np.float32))
        # self.clothpararange = self.clothpararange.view(2,2).to('cuda')
        
        # weight1 = torch.from_numpy(np.array([-0.8199, -0.0786], dtype = np.float32))
        # tempclothpara = weight1.view(1,2).to('cuda')
        # tempclothpara = 2*torch.sigmoid(tempclothpara)-1 
        # zeropara = torch.zeros_like(tempclothpara).to('cuda')
        # tempclothpara = torch.cat((tempclothpara,zeropara),-1)
        # tempclothpara = tempclothpara.view(-1,4).repeat(4,1)
        
        # canoncalmeancloth = self.cloth_simulation(torch.zeros([4,72]).reshape(-1, 72).to(self.device),torch.zeros([4,10]).reshape(-1, 10).to(self.device),tempclothpara)
        
        # cloth_ptsdist = torch.cdist(canoncalmeancloth[0:1], self.canoncalmeansmpl[0:1], p=2)#self.rawtemplatesmpl[None,...]
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # self.cloth_nnvidx = torch.squeeze(cloth_ptsdistmin[1], -1)  # B*P

        # npfaces = np.loadtxt(os.path.join('./cloth/deformation', 'smpl/smpl_vfidx.txt')) - 1
        # self.smpl_vfidx = torch.LongTensor(npfaces).to('cuda')
        
        
        # attachidx = np.loadtxt(os.path.join('./', 'cloth/attachidx.txt')) - 1
        # attachidx = torch.LongTensor(attachidx).to(self.device)
        # self.attachvertidx = torch.cat([attachidx.view(-1,1), self.cloth_nnvidx[0,attachidx].view(-1,1)], dim=-1)        
        
        
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

    def init_clothdeformgraph(self):

        npfaces = np.loadtxt(os.path.join('./cloth/deformation', 'smpl/smpl_vfidx.txt')) - 1
        self.smpl_vfidx = torch.LongTensor(npfaces).to('cuda')
                       
        # _,trinormal = mesh_face_areas_normals(self.templatesmpl, self.smplfaces)            
        # self.canonsmpl_vertnorm = trinormal[self.smpl_vfidx,:]
        
        modelnodeidx_path = os.path.join('./cloth/deformation', 'cloth/nodevidx.txt')
        modelnodeidx = np.loadtxt(modelnodeidx_path)
        self.modelnodeidx = torch.LongTensor(modelnodeidx).to('cuda')
        #self.modelnodepos = self.templateshape[self.modelnodeidx, :]
        
        # modelnodenormal_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/modelnodenormal.txt')
        # modelnodenormal = np.loadtxt(modelnodenormal_path)
        # self.modelnodenormal = torch.Tensor(modelnodenormal).to(self.device)
        
        self.modelnodenum = self.modelnodeidx.size(0)
        modelnodeedge_path = os.path.join('./cloth/deformation', 'cloth/modelnodeedge.txt')
        modelnodeedge = np.loadtxt(modelnodeedge_path) - 1
        self.modelnodeedge = torch.LongTensor(modelnodeedge).to('cuda')
        self.modelnodeedgenum = self.modelnodeedge.size(1)
        # modelnode_edgeweight_path = os.path.join(cfg.train_dataset.data_root,
                                                 # 'templatedeformT/cloth/modelnode_edgeweight.txt')
        # modelnode_edgeweight = np.loadtxt(modelnode_edgeweight_path)
        # self.modelnode_edgeweight = torch.Tensor(modelnode_edgeweight).to(self.device)
        modelvertnode_path = os.path.join('./cloth/deformation', 'cloth/modelvert_node.txt')
        modelvert_node = np.loadtxt(modelvertnode_path) - 1
        self.modelvert_node = torch.LongTensor(modelvert_node).to('cuda')
        modelvertnodeweight_path = os.path.join('./cloth/deformation', 'cloth/modelvert_nodeweight.txt')
        modelvert_nodeweight = np.loadtxt(modelvertnodeweight_path)
        self.modelvert_nodeweight = torch.Tensor(modelvert_nodeweight).to('cuda')
        self.modelvert_nodenum = self.modelvert_node.size(1)
        # nodevidx_path = os.path.join(cfg.train_dataset.data_root, 'smpldeform/nodevidx.txt')
        # nodevidx = np.loadtxt(nodevidx_path)
        # self.nodevidx = torch.LongTensor(nodevidx).to(self.device)
        #
        #
        # bw = np.load(os.path.join(cfg.train_dataset.data_root, 'bw.npy'), allow_pickle=True)
        # bw = torch.Tensor(bw).to(self.device)
        # ptsdist = torch.cdist(self.templateshape, self.templatesmpl, p=2)
        # nnvidx = torch.squeeze(torch.min(ptsdist, 1)[1], -1)  # P
        # self.bw = bw[nnvidx, :][None, ...]

        # hipidx_path = os.path.join('./',
                                   # '/hipidx.txt')  # smpldeform/vpersonalshape
        # hipidx = np.loadtxt(hipidx_path) - 1
        # hipidx = torch.LongTensor(hipidx).to(self.device)
        # skirtidx_path = os.path.join(cfg.train_dataset.data_root,
                                     # 'templatedeformT/cloth/skirtidx.txt')  # smpldeform/vpersonalshape
        # skirtidx = np.loadtxt(skirtidx_path) - 1
        # skirtidx = torch.LongTensor(skirtidx).to(self.device)

        # bw = np.load(os.path.join(cfg.train_dataset.data_root, 'bw.npy'), allow_pickle=True)
        # bw = torch.Tensor(bw).to(self.device)
        # self.smplbw = bw[None, ...]
        
        # templateshape_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/clothes_vert.txt')
        # fixtemplateshape = np.loadtxt(templateshape_path)
        # fixtemplateshape = torch.Tensor(fixtemplateshape).to(self.device)
        # ptsdist = torch.cdist(fixtemplateshape, self.templatesmpl, p=2)#self.templateshape
        # minptsdist = torch.min(ptsdist, 1)
        # minptsdistvalue = torch.squeeze(minptsdist[0], -1)
        # nnvidx = torch.squeeze(minptsdist[1], -1)  # P
        # self.bw = bw[nnvidx, :]
        # np.savetxt(os.path.join(cfg.train_dataset.data_root,
                                     # 'templatedeformT/cloth/nnbw.txt'), self.bw.detach().cpu().numpy())
        
        self.bw = np.loadtxt(os.path.join('./cloth/deformation', 'cloth/skinweightnew.txt'))
        self.bw = torch.Tensor(self.bw)[None, ...].to('cuda')
 
        # bwhip = bw[hipidx, :]
        # bwhipmean = torch.mean(bwhip, 0)
        # self.bw[skirtidx, :] = bwhipmean 
        # self.bw = self.bw[None, ...]

        # weight1 = torch.zeros([cfg.num_train_frame,3], dtype=torch.float)
        # self.displace = nn.Embedding.from_pretrained(weight1, freeze=False)
        
        # weight2 = torch.zeros([cfg.num_train_frame,72], dtype=torch.float)
        # self.inctheta = nn.Embedding.from_pretrained(weight2, freeze=False)
        # joints = np.load(os.path.join(cfg.train_dataset.data_root, 'joints.npy'))
        # self.joints = torch.Tensor(joints).to(self.device)
        # self.joints = self.joints[None,...]
        # parents = np.load(os.path.join(cfg.train_dataset.data_root, 'parents.npy'), allow_pickle=True)
        # self.parents = torch.LongTensor(parents).to(self.device)
        
        #self.latentdeform = nn.Embedding(cfg.num_train_frame, 128)
        # weight1 = torch.zeros([cfg.num_train_frame,64], dtype=torch.float)
        # self.latentdeform = nn.Embedding.from_pretrained(weight1, freeze=False)
        D = 4
        self.deformskips = [4]
        defW = 256
        layers = [nn.Linear(86, defW)]  # node coding + latent code
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = defW
            if i in self.deformskips:
                in_channels += 128
            layers += [layer(in_channels, defW)]

        self.deformpara_linears = nn.ModuleList(layers)
        self.deformpara_linears = self.deformpara_linears.to('cuda')
        # # self.deformpara_rotatelinear = nn.Linear(defW, 6)  # 12,self.modelnodenum *
        # # self.deformpara_transllinear = nn.Linear(defW, 3)#self.modelnodenum *
        # self.displacelinear  = nn.Linear(defW, 3)
        # torch.nn.init.constant(self.displacelinear.weight, 0)
        # torch.nn.init.constant(self.displacelinear.bias, 0)
        # self.deformpara_linears = nn.ModuleList([nn.Linear(256, 512), nn.Linear(512, 1024)])
        self.deformpara_finallinear = nn.Linear(defW, self.modelnodenum * 6)
        self.deformpara_finallinear = self.deformpara_finallinear.to('cuda')
        torch.nn.init.constant(self.deformpara_finallinear.weight, 0)
        torch.nn.init.constant(self.deformpara_finallinear.bias, 0)
        
    def init_smpldeformgraph(self):
        
        # npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/desmpl/desmpl_vfidx.txt')) - 1
        # self.desmpl_vfidx = torch.LongTensor(npfaces).to(self.device)
        
        templatesmpl_path = os.path.join('./cloth/deformation', 'smpl/vpersonalshape.txt')  # smpldeform/vpersonalshape
        templatesmpl = np.loadtxt(templatesmpl_path)
        self.templatesmpl = torch.Tensor(templatesmpl).to('cuda')
        self.smplvtnum = self.templatesmpl.size(0)
        
        smplnodeidx_path = os.path.join('./cloth/deformation', 'smpl/nodevidx.txt')
        smplnodeidx = np.loadtxt(smplnodeidx_path)
        self.smplnodeidx = torch.LongTensor(smplnodeidx).to('cuda')
        self.smplnodepos = self.templatesmpl[self.smplnodeidx, :]
        
        self.smplnodenum = self.smplnodeidx.size(0)
        smplnodeedge_path = os.path.join('./cloth/deformation', 'smpl/modelnodeedge.txt')
        smplnodeedge = np.loadtxt(smplnodeedge_path) - 1
        self.smplnodeedge = torch.LongTensor(smplnodeedge).to('cuda')
        self.smplnodeedgenum = self.smplnodeedge.size(1)
        
        smplvertnode_path = os.path.join('./cloth/deformation', 'smpl/modelvert_node.txt')
        smplvert_node = np.loadtxt(smplvertnode_path) - 1
        self.smplvert_node = torch.LongTensor(smplvert_node).to('cuda')
        smplvertnodeweight_path = os.path.join('./cloth/deformation', 'smpl/modelvert_nodeweight.txt')
        smplvert_nodeweight = np.loadtxt(smplvertnodeweight_path)
        self.smplvert_nodeweight = torch.Tensor(smplvert_nodeweight).to('cuda')
        self.smplvert_nodenum = self.smplvert_node.size(1)
        
           
        # npfaces = np.loadtxt(os.path.join('./cloth/deformation', 'smpl/desmpltri.txt')) - 1
        # self.desmplfaces = torch.LongTensor(npfaces).to(self.device)
        
        # bw = np.load(os.path.join(cfg.train_dataset.data_root, 'bw.npy'), allow_pickle=True)
        # bw = torch.Tensor(bw).to(self.device)
        # #bw = bw[None, ...]
        
        # # templateshape_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/clothes_vert.txt')
        # # fixtemplateshape = np.loadtxt(templateshape_path)
        # # fixtemplateshape = torch.Tensor(fixtemplateshape).to(self.device)
        # ptsdist = torch.cdist(self.templatesmpl, self.rawtemplatesmpl, p=1)#self.templateshape
        # minptsdist = torch.min(ptsdist, 1)
        # minptsdistvalue = torch.squeeze(minptsdist[0], -1)
        # nnvidx = torch.squeeze(minptsdist[1], -1)  # P
        # self.smplbw = bw[nnvidx, :][None, ...]
        
        #self.smpllatentdeform = nn.Embedding(cfg.num_train_frame, 128)
        # weight1 = torch.zeros([cfg.num_train_frame,64], dtype=torch.float)
        # self.latentdeform = nn.Embedding.from_pretrained(weight1, freeze=False)
        D = 4
        self.deformskips = [4]
        defW = 256
        layers = [nn.Linear(82, defW)]  # node coding + latent code
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = defW
            if i in self.deformskips:
                in_channels += 128
            layers += [layer(in_channels, defW)]

        self.smpldeformpara_linears = nn.ModuleList(layers)
        # # self.deformpara_rotatelinear = nn.Linear(defW, 6)  # 12,self.modelnodenum *
        # # self.deformpara_transllinear = nn.Linear(defW, 3)#self.modelnodenum *
        # self.displacelinear  = nn.Linear(defW, 3)
        # torch.nn.init.constant(self.displacelinear.weight, 0)
        # torch.nn.init.constant(self.displacelinear.bias, 0)
        # self.deformpara_linears = nn.ModuleList([nn.Linear(256, 512), nn.Linear(512, 1024)])
        self.smpldeformpara_finallinear = nn.Linear(defW, self.smplnodenum * 6)
        torch.nn.init.constant(self.smpldeformpara_finallinear.weight, 0)
        torch.nn.init.constant(self.smpldeformpara_finallinear.bias, 0)
        
    def update_embeddedgraph_cloth(self, clothvert):
    
        self.templateshape = clothvert
        self.vtnum = self.templateshape.size(1)
        self.modelnodepos = self.templateshape[:,self.modelnodeidx, :]
    
    def update_embeddedgraph_smpl(self, smplvert):
    
        self.templatesmpl = smplvert
        self.smplnodepos = self.templatesmpl[:,self.smplnodeidx, :]
        
    def update_canonsmpl(self, smplvert):
    
        self.rawtemplatesmpl = smplvert
        
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
        
    def predicting_deformation(self, clothlatent):
        # latent = self.encoder(sp_input['img'].transpose(1,3))
        latent = clothlatent#self.latentdeform(sp_input['latent_index'].to(torch.int64))  # .type(torch.LongTensor).to(self.device)np.asscalar(np.int16(sp_input['latent_index']))
        h = latent
 
        for i, l in enumerate(self.deformpara_linears):
           h = self.deformpara_linears[i](h)
           h = F.relu(h)
           if i in self.deformskips:
               h = torch.cat([latent, h], -1)
        
        h = self.deformpara_finallinear(h)
        h = h.view(-1,self.modelnodenum,6)#

        deformation_affine, deformation_transl = torch.split(h, [3, 3], dim=-1)  # 9
        # deformation_rotate = self.rot6d_to_rotmat(deformation_affine.view([-1, 6]))
        deformation_rotate = self.batch_rodrigues(deformation_affine.view([-1, 3]))  # (B*nodenum)*3-->(B*nodenum)*3*3
        # self.deformation_affine = deformation_rotate.view(-1, self.modelnodenum, 3, 3)
        self.deformation_affine = deformation_rotate.view(-1, self.modelnodenum, 3, 3)#deformation_affine
        self.deformation_transl = deformation_transl.view(-1, self.modelnodenum, 3)
        
        return self.deformation_affine, self.deformation_transl

    def predicting_deformation_smpl(self, latent):
        # latent = self.encoder(sp_input['img'].transpose(1,3))
        #latent = self.smpllatentdeform(sp_input['latent_index'].to(torch.int64))  # .type(torch.LongTensor).to(self.device)np.asscalar(np.int16(sp_input['latent_index']))
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
        
    def deformingtemplate(self, templateshape):
        #deforming template on all nodes once
        #reptemplateshape = self.templateshape.repeat(self.modelvert_nodenum,1)# (vtnum*nodenum)*3
        bs = templateshape.shape[0]
        reptemplateshape = templateshape.unsqueeze(2).repeat(1, 1, self.modelvert_nodenum, 1)  # B*vtnum*nodenum*3
        reptemplateshape = reptemplateshape.view([bs, -1, 3])# B*(vtnum*nodenum)*3
        self.modelvert_node =self.modelvert_node.view([-1])# (vtnum*nodenum)
        relativepos = reptemplateshape - self.modelnodepos[:,self.modelvert_node, :]  # B*(vtnum*nodenum)*3
        relativepos = relativepos[..., None]
        deformrelativepos = torch.matmul(self.deformation_affine[:, self.modelvert_node, :, :],
                                        relativepos)  # B*(vtnum*nodenum)*3*3, #B*(vtnum*nodenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.modelnodepos[:,self.modelvert_node,
                                                   :] + self.deformation_transl[:, self.modelvert_node, :]#B*(vtnum*nodenum)*3
        deformpos = deformpos.view(-1,self.vtnum,self.modelvert_nodenum,3)#B*vtnum*nodenum*3
        weighteddeformpos = deformpos*self.modelvert_nodeweight[None, ..., None].repeat(bs,1,1,1)#nodeweight: B*vtnum*nodenum*1
        weighteddeformpos = torch.sum(weighteddeformpos, dim=2)#B*vtnum*1*3
        deformedverts = weighteddeformpos.squeeze(2)#B*vtnum*3

        return deformedverts
    
    def deformingtemplate_smpl(self, templatesmpl):
        #deforming template on all nodes once
        #reptemplateshape = self.templateshape.repeat(self.modelvert_nodenum,1)# (vtnum*nodenum)*3
        bs = templatesmpl.shape[0]
        reptemplateshape = templatesmpl.unsqueeze(2).repeat(1, 1, self.smplvert_nodenum, 1)  # B*vtnum*nodenum*3
        reptemplateshape = reptemplateshape.view([bs, -1, 3])# B*(vtnum*nodenum)*3
        self.smplvert_node =self.smplvert_node.view([-1])# (vtnum*nodenum)
        relativepos = reptemplateshape - self.smplnodepos[:,self.smplvert_node, :]  # B*(vtnum*nodenum)*3
        relativepos = relativepos[..., None]
        deformrelativepos = torch.matmul(self.deformation_affine_smpl[:, self.smplvert_node, :, :],
                                        relativepos)  # B*(vtnum*nodenum)*3*3, #B*(vtnum*nodenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.smplnodepos[:,self.smplvert_node,
                                                   :] + self.deformation_transl_smpl[:, self.smplvert_node, :]#B*(vtnum*nodenum)*3
        deformpos = deformpos.view(-1,self.smplvtnum,self.smplvert_nodenum,3)#B*vtnum*nodenum*3
        weighteddeformpos = deformpos*self.smplvert_nodeweight[None, ..., None].repeat(bs,1,1,1)#nodeweight: B*vtnum*nodenum*1
        weighteddeformpos = torch.sum(weighteddeformpos, dim=2)#B*vtnum*1*3
        smplgraphdeformedverts = weighteddeformpos.squeeze(2)#B*vtnum*3

        return smplgraphdeformedverts
        
    def deformationsmoothloss(self):
        #smooth constrain loss
        #repmodelnodepos = self.modelnodepos.repeat(self.modelnodeedgenum, 1)  # (nodenum*edgenum)*3
        bs = self.modelnodepos.shape[0]
        repmodelnodepos = self.modelnodepos.unsqueeze(2).repeat(1, 1, self.modelnodeedgenum, 1)  # B*nodenum*edgenum*3
        repmodelnodepos = repmodelnodepos.view([bs, -1, 3])  # B*(nodenum*edgenum)*3

        self.modelnodeedge = self.modelnodeedge.view([-1])  # (nodenum*edgenum)
        relativepos = repmodelnodepos - self.modelnodepos[:, self.modelnodeedge, :]  # B*(nodenum*edgenum)*3
        relativepos = relativepos[..., None]
        deformrelativepos = torch.matmul(self.deformation_affine[:, self.modelnodeedge, :, :],
                                         relativepos)  # B*(nodenum*edgenum)*3*3, #B*(nodenum*edgenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.modelnodepos[:, self.modelnodeedge,
                                                    :] + self.deformation_transl[:, self.modelnodeedge, :]
        deformpos = deformpos.view(-1, self.modelnodenum*self.modelnodeedgenum, 3)  # B*(nodenum*edgenum)*3
        #repnodetransl = self.deformation_transl.repeat(1, self.modelnodeedgenum, 1)# B*(nodenum*edgenum)*3
        repnodetransl = self.deformation_transl.unsqueeze(2).repeat(1, 1, self.modelnodeedgenum, 1)  # B*nodenum*edgenum*3
        repnodetransl = repnodetransl.view([-1, self.modelnodenum*self.modelnodeedgenum, 3])  # B*(nodenum*edgenum)*3

        smoothpos = deformpos - (repmodelnodepos + repnodetransl)
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
        bs = self.modelnodepos.shape[0]
        repmodelnodepos = self.smplnodepos.unsqueeze(2).repeat(1, 1, self.smplnodeedgenum, 1)  # B*nodenum*edgenum*3
        repmodelnodepos = repmodelnodepos.view([bs, -1, 3])  # B*(nodenum*edgenum)*3

        self.smplnodeedge = self.smplnodeedge.view([-1])  # (nodenum*edgenum)
        relativepos = repmodelnodepos - self.smplnodepos[:,self.smplnodeedge, :]  # B*(nodenum*edgenum)*3
        relativepos = relativepos[..., None]
        deformrelativepos = torch.matmul(self.deformation_affine_smpl[:, self.smplnodeedge, :, :],
                                         relativepos)  # B*(nodenum*edgenum)*3*3, #B*(nodenum*edgenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.smplnodepos[:,self.smplnodeedge,
                                                    :] + self.deformation_transl_smpl[:, self.smplnodeedge, :]
        deformpos = deformpos.view(-1, self.smplnodenum*self.smplnodeedgenum, 3)  # B*(nodenum*edgenum)*3
        #repnodetransl = self.deformation_transl.repeat(1, self.modelnodeedgenum, 1)# B*(nodenum*edgenum)*3
        repnodetransl = self.deformation_transl_smpl.unsqueeze(2).repeat(1, 1, self.smplnodeedgenum, 1)  # B*nodenum*edgenum*3
        repnodetransl = repnodetransl.view([-1, self.smplnodenum*self.smplnodeedgenum, 3])  # B*(nodenum*edgenum)*3

        smoothpos = deformpos - (repmodelnodepos + repnodetransl)
        smoothpos = smoothpos.view(-1, self.smplnodenum, self.smplnodeedgenum, 3)# B*nodenum*edgenum*3
        smoothpos = smoothpos**2
        # weightedsmoothpos = smoothpos * self.modelnode_edgeweight[None, ..., None]  # nodeweight: B*nodenum*edgenum*1
        # weightedsmoothpos = torch.sum(weightedsmoothpos, dim=2)  # B*nodenum*1*3
        # weightedsmoothpos = weightedsmoothpos.squeeze(2)  # B*nodenum*3

        smoothloss = torch.sum(smoothpos)
        return smoothloss
        
    def deformingcloth_graphdeform_LBS(self, templateshape, A, Th):

        #embedded deformation on template shape in T pose
        graphdeformedverts = self.deformingtemplate(templateshape)

        #deforming with LBS of SMPL further
        sh = graphdeformedverts.shape
        #bw = self.bw[None,...]
               
        A = torch.bmm(self.bw.repeat(sh[0],1,1), A.view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]#including global rotation
        pts = torch.sum(R * graphdeformedverts[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        #deformedclothvert = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']
        deformedcloth = pts + Th.unsqueeze(1)
        
        return deformedcloth, graphdeformedverts
            
    def deformingsmpl_graphdeform_LBS(self, templatesmpl, A, Th):

        #embedded deformation on template shape in T pose
        smplgraphdeformedverts = self.deformingtemplate_smpl(templatesmpl)

        #deforming with LBS of SMPL further
        sh = smplgraphdeformedverts.shape
        #bw = self.bw[None,...]
        #deforming with LBS of SMPL further
       
        #embedded deformation on template shape in T pose
        personsmpl = smplgraphdeformedverts#.repeat(sh[0],1,1)  
        
        #bw = self.bw[None,...]
        A = torch.bmm(self.rawsmplbw.repeat(sh[0],1,1), A.view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]#including global rotation
        pts = torch.sum(R * personsmpl[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        #rawdeformedsmpl = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']

        deformedsmpl = pts + Th.unsqueeze(1)
                
        return deformedsmpl, smplgraphdeformedverts
        
    def computeinterpenetrationloss_posedsmpl(self, graphdeformedverts, smplgraphdeformedverts, posedsmpl, posedcloth):

        batch_size = graphdeformedverts.size(0)#self.deformedcloth.size(0)self.net.deformation_network.templateshape[None,...]
        # cloth_ptsdist = torch.cdist(graphdeformedverts, smplgraphdeformedverts, p=2)#self.rawtemplatesmpl[None,...]
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_nnvidx = torch.squeeze(cloth_ptsdistmin[1], -1)  # B*P
        cloth_nnvidx = self.cloth_nnvidx.repeat(batch_size,1)
        # cloth_mindist = torch.squeeze(cloth_ptsdistmin[0], -1)
        # validtag = torch.where(cloth_mindist > 0.03, torch.zeros_like(cloth_mindist), torch.ones_like(cloth_mindist))
        # validtag = validtag.view(-1).unsqueeze(-1).repeat(1,3)self.deformedcloth.view(-1,3)
        
        # _,trinormal = mesh_face_areas_normals(graphdeformedverts.view(-1,3), self.clothfaces.view(-1,3))
        # trinormal = trinormal.view(batch_size,-1,3)
        # vertnorm = trinormal[:,self.cloth_vfidx,:]
        #vertnorm = torch.gather(trinormal,1,self.cloth_vfidx)
        
        
        ptsnum = graphdeformedverts.size(1)
        templatevtnum = smplgraphdeformedverts.size(1)
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(batch_size)]
        idx = torch.cat(idx).to('cuda')
        bwidx = cloth_nnvidx.view(-1) #+ idx
        #deformedpersonsmpl = self.deformedpersonsmpl.view(-1, 3)  #self.deformedcloth.view(-1,3)
        canpersonsmpl = smplgraphdeformedverts.view(-1,3)
        selectcanonsmpl = canpersonsmpl[bwidx.long(), :]
        
        canonsmpl_vertnorm = []
        for i in range(0, batch_size):
            _,trinormal = mesh_face_areas_normals(smplgraphdeformedverts[i].view(-1,3), self.smplfaces[0])            
            canonsmpl_vertnorm.append(trinormal[self.smpl_vfidx,:])           
        canonsmpl_vertnorm = torch.cat(canonsmpl_vertnorm)
        
        selectcanonsmpl_norm = canonsmpl_vertnorm[bwidx.long(), :]
   
        normaldist = (selectcanonsmpl - graphdeformedverts.view(-1,3))*selectcanonsmpl_norm
 
        interpenetration = torch.sum(normaldist,1)
        # interpenetration = interpenetration[(interpenetration<0.02)*(interpenetration>0)]
        # print(interpenetration)
        #interpenetration = torch.sum((selectdeformedpersonsmpl - graphdeformedverts.view(-1,3))*vertnorm.view(-1,3),1)
        interpenetrationloss = torch.mean(F.relu(interpenetration),0)
        
        bwidx = cloth_nnvidx.view(-1) + idx
        
        posedsmpl_vertnorm = []
        for i in range(0, batch_size):
            _,trinormal = mesh_face_areas_normals(posedsmpl[i].view(-1,3), self.smplfaces[0])
            posedsmpl_vertnorm.append(trinormal[self.smpl_vfidx,:])
        posedsmpl_vertnorm = torch.cat(posedsmpl_vertnorm)

        pospersonsmpl = posedsmpl.view(-1,3)
        selectpospersonsmpl = pospersonsmpl[bwidx.long(), :]
        selectpospersonsmpl_norm = posedsmpl_vertnorm[bwidx.long(), :]
              
        normaldist1 = (selectpospersonsmpl - posedcloth.view(-1,3))*selectpospersonsmpl_norm
 
        interpenetration1 = torch.sum(normaldist1,1)
        interpenetrationloss1 = torch.mean(F.relu(interpenetration1),0)
        
        return interpenetrationloss+interpenetrationloss1
    
    def setting_gaussian_rasterizer(
        self, 
        bg_color = None,
        sh_deg:int=None,
        K = None,
        cam_poses = None,
        pytorch3d_K = None,
        ):        
        
        if bg_color is None:
            bg_color = torch.Tensor([255, 255, 255]).to(self.device)
        
      
        fovx = 2 * torch.arctan(self.resW / (2 * K[0,0, 0]))#256
        fovy = 2 * torch.arctan(self.resH / (2 * K[0,1, 1]))
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
            image_height=self.resH,
            image_width=self.resW,#256
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
    
        # raster_settings = GaussianRasterizationSettings(
            # image_height=self.resH,
            # image_width=self.resW,
            # tanfovx=tanfovx,
            # tanfovy=tanfovy,
            # kernel_size = 0.0,
            # bg=bg_color,
            # scale_modifier=1.,
            # viewmatrix=world_view_transform,
            # projmatrix=full_proj_transform,
            # sh_degree=sh_deg,
            # campos=camera_center,
            # prefiltered=False,
            # require_coord = True,
            # require_depth = True,
            # debug=False
        # )
        self.rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
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
        return_depth:bool=False
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
            bg_color = torch.Tensor([255, 255, 255]).to(self.device)
        
      
        fovx = 2 * torch.arctan(self.resW / (2 * K[0,0, 0]))#256
        fovy = 2 * torch.arctan(self.resH / (2 * K[0,1, 1]))
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
            image_height=self.resH,
            image_width=self.resW,#256
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
    
        # raster_settings = GaussianRasterizationSettings(
            # image_height=self.resH,
            # image_width=self.resW,
            # tanfovx=tanfovx,
            # tanfovy=tanfovy,
            # kernel_size = 0.0,
            # bg=bg_color,
            # scale_modifier=1.,
            # viewmatrix=world_view_transform,
            # projmatrix=full_proj_transform,
            # sh_degree=sh_deg,
            # campos=camera_center,
            # prefiltered=False,
            # require_coord = True,
            # require_depth = True,
            # debug=False
        # )
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
        
        # # if return_depth:
            # # splat_depth = gs_output["gs_depth"]
            # # splat_normal = gs_output["gs_normal"]
            # # splat_mask = torch.zeros_like(gs_output["gs_normal"])
            
            # # rendered_depth, _ = rasterizer(
                # # means3D = positions,
                # # means2D = means2D,
                # # shs = shs,
                # # colors_precomp = splat_depth,
                # # opacities = splat_opacities,
                # # scales = scales,
                # # rotations = quaternions,
                # # cov3D_precomp = cov3D)
            
            # # rendered_normal, _ = rasterizer(
                # # means3D = positions,
                # # means2D = means2D,
                # # shs = shs,
                # # colors_precomp = splat_normal,
                # # opacities = splat_opacities,
                # # scales = scales,
                # # rotations = quaternions,
                # # cov3D_precomp = cov3D)
                
            # # rendered_mask, _ = rasterizer(
                # # means3D = positions,
                # # means2D = means2D,
                # # shs = shs,
                # # colors_precomp = splat_mask,
                # # opacities = splat_opacities,
                # # scales = scales,
                # # rotations = quaternions,
                # # cov3D_precomp = cov3D)    
            
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
            # if return_depth:
                # outputs["depth"] = rendered_depth
                # outputs["normal"] = rendered_normal
                # outputs["mask"] = rendered_mask
                
            return outputs
            
        # rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
        # means3D = positions,
        # means2D = means2D,
        # shs = shs,
        # colors_precomp = splat_colors,
        # opacities = splat_opacities,
        # scales = scales,
        # rotations = quaternions,
        # cov3D_precomp = cov3D)

        # # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # # They will be excluded from value updates used in the splitting criteria.
        # return {"image": rendered_image.transpose(0, 1).transpose(1, 2),
                # "mask": rendered_alpha,
                # "expected_coord": rendered_expected_coord,
                # "median_coord": rendered_median_coord,
                # "expected_depth": rendered_expected_depth,
                # "median_depth": rendered_median_depth,
                # "viewspace_points": means2D,
                # "visibility_filter" : radii > 0,
                # "radii": radii,
                # "normal":rendered_normal,
                # }    

    def render_image_gaussian_rasterizer_depth(
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
        return_depth:bool=False
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
            bg_color = torch.Tensor([255, 255, 255]).to(self.device)
        
      
        fovx = 2 * torch.arctan(self.resW / (2 * K[0,0, 0]))#256
        fovy = 2 * torch.arctan(self.resH / (2 * K[0,1, 1]))
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
            image_height=self.resH,
            image_width=self.resW,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size = 0.0,
            bg=bg_color,
            scale_modifier=1.,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=sh_deg,
            campos=camera_center,
            prefiltered=False,
            require_coord = True,
            require_depth = True,
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
        
                   
        rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
        means3D = positions,
        means2D = means2D,
        shs = shs,
        colors_precomp = splat_colors,
        opacities = splat_opacities,
        scales = scales,
        rotations = quaternions,
        cov3D_precomp = cov3D)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"image": rendered_image.transpose(0, 1).transpose(1, 2),
                "mask": rendered_alpha,
                "expected_coord": rendered_expected_coord,
                "median_coord": rendered_median_coord,
                "expected_depth": rendered_expected_depth,
                "median_depth": rendered_median_depth,
                "viewspace_points": means2D,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "normal":rendered_normal,
                }
                
    def render_image_gaussian_rasterizer0(
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
        return_depth:bool=False
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
        
        rendered_image, radii = self.rasterizer(
            means3D = positions,
            means2D = means2D,
            shs = shs,
            colors_precomp = splat_colors,
            opacities = splat_opacities,
            scales = scales,
            rotations = quaternions,
            cov3D_precomp = cov3D)
        
        # if return_depth:
            # splat_depth = gs_output["gs_depth"]
            # splat_normal = gs_output["gs_normal"]
            # splat_mask = torch.zeros_like(gs_output["gs_normal"])
            
            # rendered_depth, _ = rasterizer(
                # means3D = positions,
                # means2D = means2D,
                # shs = shs,
                # colors_precomp = splat_depth,
                # opacities = splat_opacities,
                # scales = scales,
                # rotations = quaternions,
                # cov3D_precomp = cov3D)
            
            # rendered_normal, _ = rasterizer(
                # means3D = positions,
                # means2D = means2D,
                # shs = shs,
                # colors_precomp = splat_normal,
                # opacities = splat_opacities,
                # scales = scales,
                # rotations = quaternions,
                # cov3D_precomp = cov3D)
                
            # rendered_mask, _ = rasterizer(
                # means3D = positions,
                # means2D = means2D,
                # shs = shs,
                # colors_precomp = splat_mask,
                # opacities = splat_opacities,
                # scales = scales,
                # rotations = quaternions,
                # cov3D_precomp = cov3D)    
            
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
            # if return_depth:
                # outputs["depth"] = rendered_depth
                # outputs["normal"] = rendered_normal
                # outputs["mask"] = rendered_mask
                
            return outputs
            
        # rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
        # means3D = positions,
        # means2D = means2D,
        # shs = shs,
        # colors_precomp = splat_colors,
        # opacities = splat_opacities,
        # scales = scales,
        # rotations = quaternions,
        # cov3D_precomp = cov3D)

        # # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # # They will be excluded from value updates used in the splitting criteria.
        # return {"image": rendered_image.transpose(0, 1).transpose(1, 2),
                # "mask": rendered_alpha,
                # "expected_coord": rendered_expected_coord,
                # "median_coord": rendered_median_coord,
                # "expected_depth": rendered_expected_depth,
                # "median_depth": rendered_median_depth,
                # "viewspace_points": means2D,
                # "visibility_filter" : radii > 0,
                # "radii": radii,
                # "normal":rendered_normal,
                # }
                
    def batch_rigid_transform(self, rot_mats, init_J):
        batchsize = init_J.shape[0]
        joints = init_J.reshape(batchsize, -1, 3, 1)#torch.from_numpy(init_J.reshape(1, -1, 3, 1)).cuda()
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
        
        return deformedsmplvert, smpl_A
    
    def deforming_lbs_newsmpl_batch(self, beta, theta, trans, canonicalsmpl):
        #theta0 = theta.clone()

        #theta0[:,0:3] = 0
        batchsize = beta.shape[0]
        theta = batch_rodrigues(theta.reshape(-1, 3))#.reshape(batchsize, 24, 3, 3)
        _theta = theta.reshape(batchsize, 24, 3, 3)
        # so = self.smpl_model(betas = beta.reshape(1, 10), body_pose = _theta[:, 1:], global_orient = _theta[:, 0].view(1, 1, 3, 3))
        # smpl_v = so['vertices'].clone().reshape(-1, 3)
        
        init_J = get_J_batch_cpu(beta.reshape(-1, 10), self.smpl_model)

        #_, rel_transforms = smplx.lbs.batch_rigid_transform(_theta, torch.from_numpy(init_J).cuda(), self.parents)
        #theta_rodrigues = batch_rodrigues(theta0.reshape(-1, 3)).reshape(1, 24, 3, 3)
        posed_joints, rel_transforms = self.batch_rigid_transform(_theta, init_J)

        #init_J = get_J(beta.reshape(1, 10), self.smpl_model)

        #_, rel_transforms = self.batch_rigid_transform(theta_rodrigues, init_J)
        # joints = torch.from_numpy(init_J.reshape(-1, 3)).cuda()
        # rel_transforms = self.get_rigid_transformation(theta0.reshape(-1, 3).detach().cpu().numpy(), self.joints, self.parents)
        # rel_transforms = torch.from_numpy(rel_transforms).cuda()self.smplbwsmpl_model.lbs_weights

        smpl_A = torch.matmul(self.smpl_model.lbs_weights.view(-1, self.num_joints), rel_transforms.view(batchsize, self.num_joints, 16)).view(batchsize, -1, 4, 4)

        R = smpl_A[:, :, :3, :3]#including global rotation
        #pts = torch.sum(R * self.templatesmpl[:, :, None], dim=-1)
        pts = torch.matmul(R, canonicalsmpl[:, :, :, None])
        pts = pts.squeeze(-1) + smpl_A[:, :, :3, 3]

        deformedsmplvert = pts + trans.unsqueeze(1)#torch.matmul(pts, gR[0].transpose(0, 1))
        
        self.posed_joints = posed_joints+trans.unsqueeze(1)

        return deformedsmplvert, smpl_A, rel_transforms
    
    def deforming_lbs_body_clothing_batch(self, beta, theta, trans, canonicalsmpl, canonicalcloth):
        #theta0 = theta.clone()

        #theta0[:,0:3] = 0
        batchsize = beta.shape[0]
        theta = batch_rodrigues(theta.reshape(-1, 3))#.reshape(batchsize, 24, 3, 3)
        _theta = theta.reshape(batchsize, 24, 3, 3)
        # so = self.smpl_model(betas = beta.reshape(1, 10), body_pose = _theta[:, 1:], global_orient = _theta[:, 0].view(1, 1, 3, 3))
        # smpl_v = so['vertices'].clone().reshape(-1, 3)
        
        init_J = get_J_batch_cpu(beta.reshape(-1, 10), self.smpl_model)

        #_, rel_transforms = smplx.lbs.batch_rigid_transform(_theta, torch.from_numpy(init_J).cuda(), self.parents)
        #theta_rodrigues = batch_rodrigues(theta0.reshape(-1, 3)).reshape(1, 24, 3, 3)
        _, rel_transforms = self.batch_rigid_transform(_theta, init_J)

        #init_J = get_J(beta.reshape(1, 10), self.smpl_model)

        #_, rel_transforms = self.batch_rigid_transform(theta_rodrigues, init_J)
        # joints = torch.from_numpy(init_J.reshape(-1, 3)).cuda()
        # rel_transforms = self.get_rigid_transformation(theta0.reshape(-1, 3).detach().cpu().numpy(), self.joints, self.parents)
        # rel_transforms = torch.from_numpy(rel_transforms).cuda()self.smplbwsmpl_model.lbs_weights

        smpl_A = torch.matmul(self.smpl_model.lbs_weights.view(-1, self.num_joints), rel_transforms.view(batchsize, self.num_joints, 16)).view(batchsize, -1, 4, 4)

        R = smpl_A[:, :, :3, :3]#including global rotation
        #pts = torch.sum(R * self.templatesmpl[:, :, None], dim=-1)
        pts = torch.matmul(R, canonicalsmpl[:, :, :, None])
        pts = pts.squeeze(-1) + smpl_A[:, :, :3, 3]

        deformedsmplvert = pts + trans.unsqueeze(1)#torch.matmul(pts, gR[0].transpose(0, 1))

        cloth_A = torch.matmul(self.bw.view(-1, self.num_joints), rel_transforms.view(batchsize,self.num_joints, 16)).view(batchsize, -1, 4, 4)
        clothR = cloth_A[:, :, :3, :3]#including global rotation
        clothpts = torch.matmul(clothR, canonicalcloth[:, :, :, None])
        clothpts = clothpts.squeeze(-1) + cloth_A[:, :, :3, 3]

        deformedsclothvert = clothpts + trans.unsqueeze(1)#torch.matmul(pts, gR[0].transpose(0, 1))


        return deformedsmplvert, deformedsclothvert
        
    def deforming_lbs_gsdisp(self, gsdisp, smpl_A, trans):
           
        gs_A = smpl_A[self.net.sugar_model_smpl.gs_vert_idx]
        
        R = gs_A[:, :3, :3]#including global rotation
        #pts = torch.sum(R * self.templatesmpl[:, :, None], dim=-1)

        pts = torch.matmul(R, gsdisp[:, :, None])
        pts = pts.squeeze(-1) #+ gs_A[:, :3, 3]
        return pts, R
    
    def deforming_lbs_gsdisp_batch(self, gsdisp, smpl_A, trans):
           
        gs_A = smpl_A[:,self.net.sugar_model_smpl.gs_vert_idx]
        
        R = gs_A[:, :, :3, :3]#including global rotation
        #pts = torch.sum(R * self.templatesmpl[:, :, None], dim=-1)

        pts = torch.matmul(R, gsdisp[:, :, :, None])
        pts = pts.squeeze(-1) #+ gs_A[:, :3, 3]
        return pts, R
    
    def deforming_lbs_gs_batch(self, points_gs, gs_A, trans):
           
        R = gs_A[:, :, :3, :3]#including global rotation
        #pts = torch.sum(R * self.templatesmpl[:, :, None], dim=-1)

        pts = torch.matmul(R, points_gs[:, :, :, None])
        pts = pts.squeeze(-1) + gs_A[:, :, :3, 3]
        
        deformedgs = pts + trans.unsqueeze(1)
        
        return deformedgs
    
    def GMRobustError(self, x,c):
        #if square:,square=False
            #return 2.*x/(c*c)/(x/(c*c)+4)
        # else:
        return 2.*x*x/(c*c)/(x*x/(c*c)+4)
        
    def tv_loss(self, img, tv_weight):
        """
        Compute total variation loss.
        
        Inputs:
        - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
        - tv_weight: Scalar giving the weight w_t to use for the TV loss.
        
        Returns:
        - loss: PyTorch Variable holding a scalar giving the total variation loss
          for img weighted by tv_weight.
        """
        # Your implementation should be vectorized and not require any loops!
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # tv1 = torch.sum((img[:,:,:,1:] - img[:,:,:,:-1])**2)
        # tv2 = torch.sum((img[:,:,1:] - img[:,:,:-1])**2)
        tv1 = torch.sum(self.GMRobustError(img[:,:,:,1:] - img[:,:,:,:-1],0.5))
        tv2 = torch.sum(self.GMRobustError(img[:,:,1:] - img[:,:,:-1],0.5))
        t_v_loss = tv_weight * (tv1 + tv2)
        return t_v_loss
    
    def render_deformation_gs(self, styles, cam_poses, focals, beta, theta, trans):
        
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
                                         
                      
        self.net.position_enc_smpl.triplanefeature(styles)
        
        #obtaining gs displacement
        self.net.sugar_model_smpl._points = canoncalmeansmpl#
        gspoints_meansmpl = self.net.sugar_model_smpl.get_edited_points(self.net.sugar_model_smpl.points)
        gsoffset = self.net.position_enc_smpl.fetch_disp(gspoints_meansmpl)
        
        canpersonalsmpl= canoncalsmpl#+vertoffset+tri_feats_smpl[:,:3]
        
        self.incdisploss = 20000.0*torch.nn.functional.smooth_l1_loss(gsoffset, torch.zeros_like(gsoffset)).mean()
        
        self.deformedpersonsmpl, smpl_A = self.deforming_lbs_newsmpl(beta, theta, trans, canpersonalsmpl)
        
        #posing gs displacement
        posedgsdisp = self.deforming_lbs_gsdisp(gsoffset, smpl_A, trans)
        
        #mesh = struct.Meshes(verts=self.deformedpersonsmpl[None], faces=self.smplfaces) 
        
        # mesh_test = trimesh.Trimesh(vertices=meshvert_def[0].detach().cpu().numpy(), faces=self.meshface[0].detach().cpu().numpy())
        # mesh_test.export("mesh_test.obj")
        
        # fragments = self.net.rasterizer(mesh, cameras=cameras)
        # depth = fragments.zbuf
        # face_idx_map = fragments.pix_to_face[..., 0]
        # self.rendermask = torch.where(depth > 0, torch.ones_like(depth),torch.zeros_like(depth))
        # self.rendermask = self.rendermask.squeeze(-1).float()  
        # mesh_mask = self.rendermask[0].detach().cpu().numpy()
        # cv2.imwrite('mesh_mask.png', mesh_mask*255)
        
        
        self.net.sugar_model_smpl._points = canoncalmeansmpl#canpersonalsmpl#new mesh point

        points_meansmpl = self.net.sugar_model_smpl.get_edited_points(self.net.sugar_model_smpl.points)  #canoncalmeansmpl 

        #self.net.sugar_model_smpl._points = canpersonalsmpl#for computing mesh normal        
        # 3dgs渲染_,
        tri_feats_smpl = self.net.position_enc_smpl(points_meansmpl)#gaussian center with mean smpl vertex for fetching feature
        
        style = styles.repeat(tri_feats_smpl.shape[0],1)
        tri_feats_smpl = torch.concatenate([tri_feats_smpl,style],dim=-1)
       
        # import open3d as o3d
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(self.net.sugar_model_cloth.points.cpu().detach().numpy())
        # o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
        
        # tri_feats_smpl = self.net.position_enc_smpl(self.net.sugar_model_smpl.points, sp_input['latent_index'].to(torch.int64)[0])
        # tri_feats_cloth = self.net.position_enc_cloth(self.net.sugar_model_cloth.points, sp_input['latent_index'].to(torch.int64)[0])
        # tri_feats_cloth = self.net.position_enc_cloth(self.net.sugar_model_cloth.newpoints, sp_input['latent_index'].to(torch.int64)[0])
        # tri_feats_cloth = self.net.position_enc_cloth(subclothpoints, sp_input['latent_index'].to(torch.int64)[0])
     
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
        # quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_edited_quaternions_and_scales_with_points(points=self.deformedpersonsmpl, 
                                                                                                                    # quat=rotations_smpl, 
                                                                                                                    # sca=gs_scales_smpl)
        
        quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_posed_quaternions_and_scales(points=self.deformedpersonsmpl, 
                                                                                                      quat=rotations_smpl, 
                                                                                                      sca=gs_scales_smpl)
        
        points_smpl = points_smpl + posedgsdisp#adding gs displacement in the posed space
        
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
        
        if 0:
            rendered_expected_coord: torch.Tensor = render_smpl_output["expected_coord"]
            rendered_median_coord: torch.Tensor = render_smpl_output["median_coord"]
            rendered_normal: torch.Tensor = render_smpl_output["normal"]
            depth_middepth_normal = depth_double_to_normal(rendered_expected_coord, rendered_median_coord)
            depth_ratio = 0.6
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
            self.depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
            
        # self.smpl_img_loss = l1_loss(render_smpl_output['image'].unsqueeze(0), batch['img'] * self.silhouette_smpl.unsqueeze(-1))
        # loss_ssim_smpl = 1.0 - ssim(render_smpl_output['image'].unsqueeze(0), self.silhouette_smpl.unsqueeze(-1))
        # self.smpl_ssim = loss_ssim_smpl * (self.silhouette_smpl.sum() / (render_smpl_output['image'].shape[-1] * render_smpl_output['image'].shape[-2]))
        
        # NOTE: 使用 1-blendmask
        
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

    def render_deformation_UV(self, styles, cam_poses, focals, beta, theta, trans):
        
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
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, 512, 256)

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
                                         
                     
        self.net.position_enc_smpl.uvgsfeature(styles)
        
        self.net.sugar_model_smpl._points = canoncalmeansmpl#canpersonalsmpl#new mesh point

        points_meansmpl = self.net.sugar_model_smpl.get_edited_points(canoncalmeansmpl)  #self.net.sugar_model_smpl.points canoncalmeansmpl 

        gs_uv = self.net.sugar_model_smpl.UV_GS

        gs_pos, gs_rot, gs_scale, gs_color, gs_opacity = self.net.position_enc_smpl(gs_uv)#gaussian center with mean smpl vertex for fetching feature
        
        #plane_opacity = self.net.position_enc_smpl.plane_opacity
        #self.opacity_loss = (torch.log(gs_opacity)+torch.log(1-gs_opacity)).abs().mean()
        self.opacity_loss = torch.mean(
                    torch.log(0.1 + self.net.position_enc_smpl.plane_opacity) +
                    torch.log(0.1 + 1. - self.net.position_enc_smpl.plane_opacity) - -2.20727)
        
        self.postv_loss = self.tv_loss(self.net.position_enc_smpl.plane_pos, 0.1)
        
        self.scale_loss = 0.1*torch.norm(self.net.position_enc_smpl.plane_scale-0.01).mean()
                    
        # 预测透明度和球谐函数参数(SMPL和CLOTH) 
        gs_opacity_smpl = gs_opacity#torch.ones_like(points_meansmpl)[...,0:1]#
        # gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 16, 3)
        gs_shs_smpl = gs_color

        # 预测旋转和尺度(SMPL和CLOTH)
        rotations_smpl = gs_rot
        gs_scales_smpl = gs_scale
        #gs_xyz_smpl = geometry_out_smpl['xyz']
       
        canpersonalsmpl= canoncalsmpl#+vertoffset+tri_feats_smpl[:,:3]
        
        #self.incdisploss = 20000.0*torch.nn.functional.smooth_l1_loss(gs_pos, torch.zeros_like(gs_pos)).mean()
        self.incdisploss = 0.1*torch.nn.functional.smooth_l1_loss(gs_pos, torch.zeros_like(gs_pos),reduction='sum', beta=0.1)#.mean()
        
        self.deformedpersonsmpl, smpl_A = self.deforming_lbs_newsmpl(beta, theta, trans, canpersonalsmpl)

        #posing gs displacement
        posedgsdisp, gs_A = self.deforming_lbs_gsdisp(gs_pos, smpl_A, trans)
        
        #mesh = struct.Meshes(verts=self.deformedpersonsmpl[None], faces=self.smplfaces) 
        
        # mesh_test = trimesh.Trimesh(vertices=meshvert_def[0].detach().cpu().numpy(), faces=self.meshface[0].detach().cpu().numpy())
        # mesh_test.export("mesh_test.obj")
        
        # fragments = self.net.rasterizer(mesh, cameras=cameras)
        # depth = fragments.zbuf
        # face_idx_map = fragments.pix_to_face[..., 0]
        # self.rendermask = torch.where(depth > 0, torch.ones_like(depth),torch.zeros_like(depth))
        # self.rendermask = self.rendermask.squeeze(-1).float()  
        # mesh_mask = self.rendermask[0].detach().cpu().numpy()
        # cv2.imwrite('mesh_mask.png', mesh_mask*255)
        
                    
        points_smpl = self.net.sugar_model_smpl.get_edited_points(self.deformedpersonsmpl.squeeze(0))
        #quaternions_smpl = self.net.sugar_model_smpl.get_posed_quaternions(points=self.deformedpersonsmpl, quat=0)
        
        #quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_posed_quaternions_scale2D(points=self.deformedpersonsmpl, quat=rotations_smpl, sca=gs_scales_smpl)
        
        # quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_edited_quaternions_and_scales_with_points(points=self.deformedpersonsmpl, 
                                                                                                                    # quat=rotations_smpl, 
                                                                                                                    # sca=gs_scales_smpl)
        
        quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_posed_quaternions_and_scales(points=self.deformedpersonsmpl, 
                                                                                                      quat=rotations_smpl, 
                                                                                                      sca=gs_scales_smpl)
        # quaternions_smpl, gs_scales_smpl, gs_quat = self.net.sugar_model_smpl.get_posed_quaternions_and_scales2(canpoints=canpersonalsmpl,
                                                                                                      # gs_A=gs_A, 
                                                                                                      # quat=rotations_smpl, 
                                                                                                      # sca=gs_scales_smpl)
        #self.incrotloss = 20000.0*torch.nn.functional.smooth_l1_loss(rotations_smpl, torch.zeros_like(rotations_smpl)).mean()
        self.incrotloss = 0.1*torch.nn.functional.smooth_l1_loss(rotations_smpl, torch.zeros_like(rotations_smpl),reduction='sum')#.mean()
        
        points_smpl = points_smpl + posedgsdisp#adding gs displacement in the posed space
        #quaternions_smpl = rotations_smpl
        
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
        
        if 0:
            rendered_expected_coord: torch.Tensor = render_smpl_output["expected_coord"]
            rendered_median_coord: torch.Tensor = render_smpl_output["median_coord"]
            rendered_normal: torch.Tensor = render_smpl_output["normal"]
            depth_middepth_normal = depth_double_to_normal(rendered_expected_coord, rendered_median_coord)
            depth_ratio = 0.6
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
            self.depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
            
        # self.smpl_img_loss = l1_loss(render_smpl_output['image'].unsqueeze(0), batch['img'] * self.silhouette_smpl.unsqueeze(-1))
        # loss_ssim_smpl = 1.0 - ssim(render_smpl_output['image'].unsqueeze(0), self.silhouette_smpl.unsqueeze(-1))
        # self.smpl_ssim = loss_ssim_smpl * (self.silhouette_smpl.sum() / (render_smpl_output['image'].shape[-1] * render_smpl_output['image'].shape[-2]))
        
        # NOTE: 使用 1-blendmask
        
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
    
    def batch_generatefeature_multiparts(self, styles, cam_poses, focals, beta, theta, trans):
        
        thetazero = batch_rodrigues(torch.zeros_like(theta).reshape(-1, 3)).reshape(-1, 24, 3, 3)
        
        # so = self.smpl_model(betas = torch.zeros_like(beta).reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        # canoncalmeansmpl = so['vertices'].clone().reshape(-1, 6890, 3)#mean smpl
        
        so = self.smpl_model(betas = beta.reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        self.canoncalsmpl = so['vertices'].clone().reshape(-1, 6890, 3)

        # c2w = cam_poses#[0]torch.cat([cam_poses[0],torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(self.device).view(1,4)],dim=0)
        # vm = c2w.inverse().float()#[None]
        # mat_R, mat_T = vm[...,:3, :3].transpose(-2, -1), vm[...,:3, 3]
        # projection_matrix = focals
        # pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, 512, 256)

        # cameras = PerspectiveCameras(device='cuda',
                                         # K=pytorch3d_K,
                                         # R=mat_R,
                                         # T=mat_T)
                                         
                     
        self.net.position_enc_smpl.uvgsfeature(styles)
        
        #self.net.sugar_model_smpl._points = canoncalmeansmpl#canpersonalsmpl#new mesh point

        #points_meansmpl = self.net.sugar_model_smpl.getbatch_edited_points(canoncalmeansmpl)  #self.net.sugar_model_smpl.points canoncalmeansmpl 

        gs_uv = self.net.sugar_model_smpl.UV_GS

        gs_incpos, gs_rot, gs_scale, gs_color, gs_opacity = self.net.position_enc_smpl.forward_batch(gs_uv)#gaussian center with mean smpl vertex for fetching feature
        
        #plane_opacity = self.net.position_enc_smpl.plane_opacity, gs_bw
        #self.opacity_loss = (torch.log(gs_opacity)+torch.log(1-gs_opacity)).abs().mean()
        # self.opacity_loss = torch.mean(
                    # torch.log(0.1 + self.net.position_enc_smpl.plane_opacity) +
                    # torch.log(0.1 + 1. - self.net.position_enc_smpl.plane_opacity) - -2.20727)
        
        allopacity_loss = []         
        for seg in range(0, 3):
            plane_opacity = self.net.position_enc_smpl.getspecattr('planeopacity', seg)
            opacity_loss = torch.mean(
                    torch.log(0.1 + plane_opacity) +
                    torch.log(0.1 + 1. - plane_opacity) - -2.20727)
            allopacity_loss += [opacity_loss.view([-1])]           
        self.opacity_loss = torch.cat(allopacity_loss, 0).sum()
        
        #self.postv_loss = self.tv_loss(self.net.position_enc_smpl.plane_pos, 0.1)
      
        allpostv_loss = []         
        for seg in range(0, 3):
            plane_pos = self.net.position_enc_smpl.getspecattr('planepos', seg)
            postv_loss = self.tv_loss(plane_pos, 0.1)
            allpostv_loss += [postv_loss.view([-1])]  
        # globalplane_pos = self.bodymask[None]*self.net.position_enc_smpl.getspecattr('planepos', 0) + \
                    # self.legmask[None]*self.net.position_enc_smpl.getspecattr('planepos', 1) + \
                    # self.headmask[None]*self.net.position_enc_smpl.getspecattr('planepos', 2)
        # gbpostv_loss = self.tv_loss(globalplane_pos, 0.1)#global incpos TV loss
        # allpostv_loss += [gbpostv_loss.view([-1])]                    
        self.postv_loss = torch.cat(allpostv_loss, 0).sum()
        
        #self.scale_loss = 0.1*torch.norm(self.net.position_enc_smpl.plane_scale1-(-4.9)).mean()#(-4.7)-5.10.012
        
        allscale_loss  = []         
        for seg in range(0, 3):
            plane_scale1 = self.net.position_enc_smpl.getspecattr('planescale1', seg)
            scale_loss = 0.1*torch.norm(plane_scale1-(-4.9)).mean()
            allscale_loss += [scale_loss.view([-1])]           
        self.scale_loss = torch.cat(allscale_loss, 0).sum()
        
        # 预测透明度和球谐函数参数(SMPL和CLOTH) 
        self.gs_opacity_smpl = gs_opacity#torch.ones_like(gs_color[...,0:1])#
        # gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 16, 3)
        self.gs_shs_smpl = gs_color

        # 预测旋转和尺度(SMPL和CLOTH)
        self.rotations_smpl = gs_rot
        self.gs_scales_smpl = gs_scale
        #gs_xyz_smpl = geometry_out_smpl['xyz']
       
        #canpersonalsmpl= canoncalsmpl#+vertoffset+tri_feats_smpl[:,:3]
        
        #self.incdisploss = 20000.0*torch.nn.functional.smooth_l1_loss(gs_pos, torch.zeros_like(gs_pos)).mean()
        #self.incdisploss = 0.001*torch.nn.functional.smooth_l1_loss(self.net.position_enc_smpl.plane_pos, torch.zeros_like(self.net.position_enc_smpl.plane_pos),reduction='sum', beta=0.1)#.mean()
        
        allincdisploss  = []         
        for seg in range(0, 3):
            plane_pos = self.net.position_enc_smpl.getspecattr('planepos', seg)
            incdisploss = 0.001*torch.nn.functional.smooth_l1_loss(plane_pos, torch.zeros_like(plane_pos),reduction='sum', beta=0.1)
            allincdisploss += [incdisploss.view([-1])]           
        self.incdisploss = torch.cat(allincdisploss, 0).sum()
        
        #self.incrotloss = 20000.0*torch.nn.functional.smooth_l1_loss(rotations_smpl, torch.zeros_like(rotations_smpl)).mean()
        #self.incrotloss = 0.1*torch.nn.functional.smooth_l1_loss(self.net.position_enc_smpl.plane_rot, torch.zeros_like(self.net.position_enc_smpl.plane_rot),reduction='sum')#.mean()
        
        allincrotloss  = []         
        for seg in range(0, 3):
            plane_rot = self.net.position_enc_smpl.getspecattr('planerot', seg)
            incrotloss = 0.1*torch.nn.functional.smooth_l1_loss(plane_rot, torch.zeros_like(plane_rot),reduction='sum')
            allincrotloss += [incrotloss.view([-1])]           
        self.incrotloss = torch.cat(allincrotloss, 0).sum()
        
        # self.incbwloss = 10.0*torch.nn.functional.smooth_l1_loss(self.net.position_enc_smpl.plane_bw, torch.zeros_like(self.net.position_enc_smpl.plane_bw),reduction='sum')#.mean()
        # self.bwtv_loss = self.tv_loss(self.net.position_enc_smpl.plane_bw, 0.1)
        
        #self.deformedpersonsmpl, smpl_A = self.deforming_lbs_newsmpl_batch(beta, theta, trans, canpersonalsmpl)
        self.deformedpersonsmpl, smpl_A, rel_transforms = self.deforming_lbs_newsmpl_batch(beta, theta, trans, self.canoncalsmpl)
        
        points_gs = self.net.sugar_model_smpl.getbatch_edited_points(self.canoncalsmpl)
        self.canonpersongs = points_gs + gs_incpos
        
        #self.blendtransform_gs = self.net.sugar_model_smpl.getbatch_edited_blendtransform(smpl_A)
        # lbsweights_gs = self.blendweights_gs[None]+gs_bw
        # lbsweights_gs = lbsweights_gs/(lbsweights_gs.sum(dim=-1).unsqueeze(-1))
        # lbsweights_gs = F.softmax(F.gelu(lbsweights_gs))
        #lbsweights_gs = F.softmax(F.gelu(gs_bw))
        #self.incbwloss = 1.0*torch.nn.functional.smooth_l1_loss(lbsweights_gs, self.blendweights_gs[None],reduction='sum')#.mean()
        
        #self.blendweights_gs = self.net.sugar_model_smpl.get_edited_blendweights(self.smpl_model.lbs_weights.view(-1, self.num_joints))
        batchsize = styles.shape[0]
        self.blendtransform_gs = torch.matmul(self.blendweights_gs, rel_transforms.view(batchsize, self.num_joints, 16)).view(batchsize, -1, 4, 4)

        #posing gs displacement
        #self.posedgsdisp, gs_A = self.deforming_lbs_gsdisp_batch(gs_pos, smpl_A, trans)
        self.posedgs = self.deforming_lbs_gs_batch(self.canonpersongs, self.blendtransform_gs, trans)
    
    def batch_generatefeature_multiparts_fusion(self, styles, cam_poses, focals, beta, theta, trans):
        
        thetazero = batch_rodrigues(torch.zeros_like(theta).reshape(-1, 3)).reshape(-1, 24, 3, 3)
        
        # so = self.smpl_model(betas = torch.zeros_like(beta).reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        # canoncalmeansmpl = so['vertices'].clone().reshape(-1, 6890, 3)#mean smpl
        
        so = self.smpl_model(betas = beta.reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        self.canoncalsmpl = so['vertices'].clone().reshape(-1, 6890, 3)
                     
        self.net.position_enc_smpl.uvgsfeature(styles)
        
        #self.net.sugar_model_smpl._points = canoncalmeansmpl#canpersonalsmpl#new mesh point

        #points_meansmpl = self.net.sugar_model_smpl.getbatch_edited_points(canoncalmeansmpl)  #self.net.sugar_model_smpl.points canoncalmeansmpl 

        gs_uv = self.net.sugar_model_smpl.UV_GS

        gs_incpos, gs_rot, gs_scale, gs_color, gs_opacity, gs_incpos_fusion, gs_rot_fusion, gs_scale_fusion, gs_color_fusion, gs_opacity_fusion = self.net.position_enc_smpl.forward_batch(gs_uv)#gaussian center with mean smpl vertex for fetching feature
        
        #plane_opacity = self.net.position_enc_smpl.plane_opacity, gs_bw
        #self.opacity_loss = (torch.log(gs_opacity)+torch.log(1-gs_opacity)).abs().mean()
        self.opacity_loss_fusion = torch.mean(
                    torch.log(0.1 + self.net.position_enc_smpl.plane_opacity) +
                    torch.log(0.1 + 1. - self.net.position_enc_smpl.plane_opacity) - -2.20727)
        
        allopacity_loss = []         
        for seg in range(0, 3):
            plane_opacity = self.net.position_enc_smpl.getspecattr('planeopacity', seg)
            opacity_loss = torch.mean(
                    torch.log(0.1 + plane_opacity) +
                    torch.log(0.1 + 1. - plane_opacity) - -2.20727)
            allopacity_loss += [opacity_loss.view([-1])]           
        self.opacity_loss = torch.cat(allopacity_loss, 0).sum() + self.opacity_loss_fusion
        
        self.postv_loss_fusion = self.tv_loss(self.net.position_enc_smpl.plane_pos, 0.1)
      
        allpostv_loss = []         
        for seg in range(0, 3):
            plane_pos = self.net.position_enc_smpl.getspecattr('planepos', seg)
            postv_loss = self.tv_loss(plane_pos, 0.1)
            allpostv_loss += [postv_loss.view([-1])]  
        # globalplane_pos = self.bodymask[None]*self.net.position_enc_smpl.getspecattr('planepos', 0) + \
                    # self.legmask[None]*self.net.position_enc_smpl.getspecattr('planepos', 1) + \
                    # self.headmask[None]*self.net.position_enc_smpl.getspecattr('planepos', 2)
        # gbpostv_loss = self.tv_loss(globalplane_pos, 0.1)#global incpos TV loss
        # allpostv_loss += [gbpostv_loss.view([-1])]                    
        self.postv_loss = torch.cat(allpostv_loss, 0).sum() + self.postv_loss_fusion
        
        self.scale_loss_fusion = 0.1*torch.norm(self.net.position_enc_smpl.plane_scale1-(-4.9)).mean()#(-4.7)-5.10.012
        
        allscale_loss  = []         
        for seg in range(0, 3):
            plane_scale1 = self.net.position_enc_smpl.getspecattr('planescale1', seg)
            scale_loss = 0.1*torch.norm(plane_scale1-(-4.9)).mean()
            allscale_loss += [scale_loss.view([-1])]           
        self.scale_loss = torch.cat(allscale_loss, 0).sum() + self.scale_loss_fusion
        
        # 预测透明度和球谐函数参数(SMPL和CLOTH) 
        self.gs_opacity_smpl = gs_opacity#torch.ones_like(gs_color[...,0:1])#
        # gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 16, 3)
        self.gs_shs_smpl = gs_color

        # 预测旋转和尺度(SMPL和CLOTH)
        self.rotations_smpl = gs_rot
        self.gs_scales_smpl = gs_scale
        #gs_xyz_smpl = geometry_out_smpl['xyz']
       
        self.gs_opacity_smpl_fusion = gs_opacity_fusion#torch.ones_like(gs_color[...,0:1])#
        self.gs_shs_smpl_fusion = gs_color_fusion

        self.rotations_smpl_fusion = gs_rot_fusion
        self.gs_scales_smpl_fusion = gs_scale_fusion
        
        #canpersonalsmpl= canoncalsmpl#+vertoffset+tri_feats_smpl[:,:3]
        
        #self.incdisploss = 20000.0*torch.nn.functional.smooth_l1_loss(gs_pos, torch.zeros_like(gs_pos)).mean()
        self.incdisploss_fusion = 0.001*torch.nn.functional.smooth_l1_loss(self.net.position_enc_smpl.plane_pos, torch.zeros_like(self.net.position_enc_smpl.plane_pos),reduction='sum', beta=0.1)#.mean()
        
        allincdisploss  = []         
        for seg in range(0, 3):
            plane_pos = self.net.position_enc_smpl.getspecattr('planepos', seg)
            incdisploss = 0.001*torch.nn.functional.smooth_l1_loss(plane_pos, torch.zeros_like(plane_pos),reduction='sum', beta=0.1)
            allincdisploss += [incdisploss.view([-1])]           
        self.incdisploss = torch.cat(allincdisploss, 0).sum() + self.incdisploss_fusion
        
        #self.incrotloss = 20000.0*torch.nn.functional.smooth_l1_loss(rotations_smpl, torch.zeros_like(rotations_smpl)).mean()
        self.incrotloss_fusion = 0.1*torch.nn.functional.smooth_l1_loss(self.net.position_enc_smpl.plane_rot, torch.zeros_like(self.net.position_enc_smpl.plane_rot),reduction='sum')#.mean()
        
        allincrotloss  = []         
        for seg in range(0, 3):
            plane_rot = self.net.position_enc_smpl.getspecattr('planerot', seg)
            incrotloss = 0.1*torch.nn.functional.smooth_l1_loss(plane_rot, torch.zeros_like(plane_rot),reduction='sum')
            allincrotloss += [incrotloss.view([-1])]           
        self.incrotloss = torch.cat(allincrotloss, 0).sum() + self.incrotloss_fusion
        
        # self.incbwloss = 10.0*torch.nn.functional.smooth_l1_loss(self.net.position_enc_smpl.plane_bw, torch.zeros_like(self.net.position_enc_smpl.plane_bw),reduction='sum')#.mean()
        # self.bwtv_loss = self.tv_loss(self.net.position_enc_smpl.plane_bw, 0.1)
        
        #self.deformedpersonsmpl, smpl_A = self.deforming_lbs_newsmpl_batch(beta, theta, trans, canpersonalsmpl)
        self.deformedpersonsmpl, smpl_A, rel_transforms = self.deforming_lbs_newsmpl_batch(beta, theta, trans, self.canoncalsmpl)
        
        points_gs = self.net.sugar_model_smpl.getbatch_edited_points(self.canoncalsmpl)
        self.canonpersongs = points_gs + gs_incpos
        
        self.canonpersongs_fusion = points_gs + gs_incpos_fusion
        #self.blendtransform_gs = self.net.sugar_model_smpl.getbatch_edited_blendtransform(smpl_A)
        # lbsweights_gs = self.blendweights_gs[None]+gs_bw
        # lbsweights_gs = lbsweights_gs/(lbsweights_gs.sum(dim=-1).unsqueeze(-1))
        # lbsweights_gs = F.softmax(F.gelu(lbsweights_gs))
        #lbsweights_gs = F.softmax(F.gelu(gs_bw))
        #self.incbwloss = 1.0*torch.nn.functional.smooth_l1_loss(lbsweights_gs, self.blendweights_gs[None],reduction='sum')#.mean()
        
        #self.blendweights_gs = self.net.sugar_model_smpl.get_edited_blendweights(self.smpl_model.lbs_weights.view(-1, self.num_joints))
        batchsize = styles.shape[0]
        self.blendtransform_gs = torch.matmul(self.blendweights_gs, rel_transforms.view(batchsize, self.num_joints, 16)).view(batchsize, -1, 4, 4)

        #posing gs displacement
        #self.posedgsdisp, gs_A = self.deforming_lbs_gsdisp_batch(gs_pos, smpl_A, trans)
        self.posedgs = self.deforming_lbs_gs_batch(self.canonpersongs, self.blendtransform_gs, trans)
        
        self.posedgs_fusion = self.deforming_lbs_gs_batch(self.canonpersongs_fusion, self.blendtransform_gs, trans)
    
	
	def edit_Gaussianattributes(self, gs_rot, gs_scale, gs_color, gs_opacity):
	
		facesegidx_edit = np.loadtxt('lib/networks/modules/facesegidx_edit.txt')
        self.facesegidx_edit = torch.Tensor(facesegidx_edit).to('cuda')
        faces_len = self.smplfaces.shape[1]
        gssegidx =  [torch.full([6], self.facesegidx_edit[i], dtype=torch.long) for i in range(faces_len)]
        gssegidx = torch.cat(gssegidx).to('cuda')
        self.editgsidx = gssegidx==0#upper body
		
		self.gs_opacity_smpl[:,editgsidx] = gs_opacity[:,editgsidx]

		self.gs_shs_smpl[:,editgsidx] = gs_color[:,editgsidx]

		# 预测旋转和尺度(SMPL和CLOTH)
		self.rotations_smpl[:,editgsidx] = gs_rot[:,editgsidx]
		self.gs_scales_smpl[:,editgsidx] = gs_scale[:,editgsidx]
	
    def batch_generatefeature(self, styles, cam_poses, focals, beta, theta, trans):
        
        #thetazero = batch_rodrigues(torch.zeros_like(theta).reshape(-1, 3)).reshape(-1, 24, 3, 3)
        
        # so = self.smpl_model(betas = torch.zeros_like(beta).reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        # canoncalmeansmpl = so['vertices'].clone().reshape(-1, 6890, 3)#mean smpl
        batchsize = styles.shape[0]
        so = self.smpl_model(betas = beta.reshape(-1, 10), body_pose = self.thetazero[:batchsize, 1:], global_orient = self.thetazero[:batchsize, 0].view(-1, 1, 3, 3))
        self.canoncalsmpl = so['vertices'].clone().reshape(-1, 6890, 3)

        # c2w = cam_poses#[0]torch.cat([cam_poses[0],torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(self.device).view(1,4)],dim=0)
        # vm = c2w.inverse().float()#[None]
        # mat_R, mat_T = vm[...,:3, :3].transpose(-2, -1), vm[...,:3, 3]
        # projection_matrix = focals
        # pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, 512, 256)

        # cameras = PerspectiveCameras(device='cuda',
                                         # K=pytorch3d_K,
                                         # R=mat_R,
                                         # T=mat_T)
                                         
                     
        self.net.position_enc_smpl.uvgsfeature(styles)
        
        #self.net.sugar_model_smpl._points = canoncalmeansmpl#canpersonalsmpl#new mesh point

        #points_meansmpl = self.net.sugar_model_smpl.getbatch_edited_points(canoncalmeansmpl)  #self.net.sugar_model_smpl.points canoncalmeansmpl 

        gs_uv = self.net.sugar_model_smpl.UV_GS

        gs_incpos, gs_rot, gs_scale, gs_color, gs_opacity = self.net.position_enc_smpl.forward_batch(gs_uv)#gaussian center with mean smpl vertex for fetching feature
        
        #plane_opacity = self.net.position_enc_smpl.plane_opacity, gs_bw
        #self.opacity_loss = (torch.log(gs_opacity)+torch.log(1-gs_opacity)).abs().mean()
        # self.opacity_loss = torch.mean(
                    # torch.log(0.1 + self.net.position_enc_smpl.plane_opacity) +
                    # torch.log(0.1 + 1. - self.net.position_enc_smpl.plane_opacity) - -2.20727)
       
        self.opacity_loss = 0.1*torch.norm(self.net.position_enc_smpl.plane_opacity-1).mean()
        
        self.postv_loss = self.tv_loss(self.net.position_enc_smpl.plane_pos, 0.1)

        self.scale_loss = 0.1*torch.norm(self.net.position_enc_smpl.plane_scale1-(-4.9)).mean()#(-4.7)-5.10.012
               
        self.gs_incpos_smpl = gs_incpos     
        # 预测透明度和球谐函数参数(SMPL和CLOTH) 
        self.gs_opacity_smpl = gs_opacity#torch.ones_like(gs_color[...,0:1])#
        # gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 16, 3)
        self.gs_shs_smpl = gs_color

        # 预测旋转和尺度(SMPL和CLOTH)
        self.rotations_smpl = gs_rot
        self.gs_scales_smpl = gs_scale
        #gs_xyz_smpl = geometry_out_smpl['xyz']
       
        #canpersonalsmpl= canoncalsmpl#+vertoffset+tri_feats_smpl[:,:3]
        
        #self.incdisploss = 20000.0*torch.nn.functional.smooth_l1_loss(gs_pos, torch.zeros_like(gs_pos)).mean()
        self.incdisploss = 0.001*torch.nn.functional.smooth_l1_loss(self.net.position_enc_smpl.plane_pos, torch.zeros_like(self.net.position_enc_smpl.plane_pos),reduction='sum', beta=0.1)#.mean()
        
        #self.incrotloss = 20000.0*torch.nn.functional.smooth_l1_loss(rotations_smpl, torch.zeros_like(rotations_smpl)).mean()
        self.incrotloss = 0.1*torch.nn.functional.smooth_l1_loss(self.net.position_enc_smpl.plane_rot, torch.zeros_like(self.net.position_enc_smpl.plane_rot),reduction='sum')#.mean()
        
        # self.incbwloss = 10.0*torch.nn.functional.smooth_l1_loss(self.net.position_enc_smpl.plane_bw, torch.zeros_like(self.net.position_enc_smpl.plane_bw),reduction='sum')#.mean()
        # self.bwtv_loss = self.tv_loss(self.net.position_enc_smpl.plane_bw, 0.1)
        
        #self.deformedpersonsmpl, smpl_A = self.deforming_lbs_newsmpl_batch(beta, theta, trans, canpersonalsmpl)
        self.deformedpersonsmpl, smpl_A, rel_transforms = self.deforming_lbs_newsmpl_batch(beta, theta, trans, self.canoncalsmpl)
        
        #self.cannormals = self.net.sugar_model_smpl.get_normals(self.canoncalsmpl)
        
        points_gs = self.net.sugar_model_smpl.getbatch_edited_points(self.canoncalsmpl)
        self.canonpersongs = points_gs + gs_incpos
        #self.canonpersongs = points_gs + self.cannormals*gs_incpos
        
        #self.blendtransform_gs = self.net.sugar_model_smpl.getbatch_edited_blendtransform(smpl_A)
        # lbsweights_gs = self.blendweights_gs[None]+gs_bw
        # lbsweights_gs = lbsweights_gs/(lbsweights_gs.sum(dim=-1).unsqueeze(-1))
        # lbsweights_gs = F.softmax(F.gelu(lbsweights_gs))
        #lbsweights_gs = F.softmax(F.gelu(gs_bw))
        #self.incbwloss = 1.0*torch.nn.functional.smooth_l1_loss(lbsweights_gs, self.blendweights_gs[None],reduction='sum')#.mean()
        
        #self.blendweights_gs = self.net.sugar_model_smpl.get_edited_blendweights(self.smpl_model.lbs_weights.view(-1, self.num_joints))
        batchsize = styles.shape[0]
        self.blendtransform_gs = torch.matmul(self.blendweights_gs, rel_transforms.view(batchsize, self.num_joints, 16)).view(batchsize, -1, 4, 4)

        #posing gs displacement
        #self.posedgsdisp, gs_A = self.deforming_lbs_gsdisp_batch(gs_pos, smpl_A, trans)
        self.posedgs = self.deforming_lbs_gs_batch(self.canonpersongs, self.blendtransform_gs, trans)
        
    def gsrendering(self, gradtag, bidx, cam_poses, focals):
        
        if self.resH==256: #progressive training
            focals[:,0:2,:] = focals[:,0:2,:]/2     
            
        c2w = cam_poses#[0]torch.cat([cam_poses[0],torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(self.device).view(1,4)],dim=0)
        vm = c2w.inverse().float()#[None]styles, 
        mat_R, mat_T = vm[...,:3, :3].transpose(-2, -1), vm[...,:3, 3]
        projection_matrix = focals
                    
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, self.resH, self.resW)

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
        

        # if self.resW == 128 and self.lowrestag == 1:
            # self.setting_gaussian_rasterizer(sh_deg=4,
            # K = focals,
            # cam_poses = cam_poses,
            # pytorch3d_K = pytorch3d_K
            # )
            # self.lowrestag = 0
            
        # if self.resW == 256 and self.highrestag == 1:
            # self.setting_gaussian_rasterizer(sh_deg=4,
            # K = focals,
            # cam_poses = cam_poses,
            # pytorch3d_K = pytorch3d_K
            # )
            # self.highrestag = 0
          
        #points_smpl = self.net.sugar_model_smpl.get_edited_points(self.deformedpersonsmpl[bidx])faces_normals=self.cannormals[bidx],
        quaternions_smpl, gs_scales_smpl, gs_normals = self.net.sugar_model_smpl.get_posed_quaternions_and_scales_normals(gs_R=self.blendtransform_gs[:, :, :3, :3][bidx],canmeanpoints=self.canoncalmeansmpl[0],canpoints=self.canoncalsmpl[bidx], gspoints=self.posedgs[bidx], points=self.deformedpersonsmpl[bidx], 
                                                                                                          quat=self.rotations_smpl[bidx], 
                                                                                                          sca=self.gs_scales_smpl[bidx])
        # quaternions_smpl, gs_scales_smpl, gs_quat = self.net.sugar_model_smpl.get_posed_quaternions_and_scales2(canpoints=canpersonalsmpl,
                                                                                                      # gs_A=gs_A, 
                                                                                                      # quat=rotations_smpl, 
                                                                                                      # sca=gs_scales_smpl)
        
        points_smpl_disp = self.posedgs[bidx]#points_smpl + self.posedgsdisp[bidx]#adding gs displacement in the posed space
        #quaternions_smpl = rotations_smpl
        
        
        # query_pts = self.canonpersongs[bidx].reshape(1, 1, 1, -1, 3) / 1.3
        # template_sdf = torch.nn.functional.grid_sample(
            # self.sdf_voxels, query_pts,
            # padding_mode = 'border', align_corners = True
        # ).reshape(-1)
        
        if 0:#gradtag==1:#self.net.position_enc_smpl.scalelinear.weight.requires_grad == True:
            query_pts = self.canonpersongs[bidx][::3][None]
            query_pts.requires_grad_(True)

            incsdf = self.net.IncSDF(query_pts,styles)
            gspoints_sdf = incsdf#template_sdf + 

            d_output_smpl = torch.ones_like(gspoints_sdf, requires_grad=False, device=gspoints_sdf.device)#
            sdfgradients_gspoints = torch.autograd.grad(outputs=gspoints_sdf,
                                          inputs=query_pts,
                                           grad_outputs=d_output_smpl,
                                          create_graph=True,
                                           retain_graph=True,
                                           only_inputs=True)[0]
          
            normals_sdf=F.normalize(sdfgradients_gspoints,dim=-1)
            normalerr = 1-(gs_normals[::3] * normals_sdf).sum(dim=-1).abs()#normal in the canonical space
            normal_loss = 0.1*self.l1loss(normalerr,torch.zeros_like(normalerr))
            # dot=(gs_normals[::3]*normals_sdf).sum(dim=-1, keepdim=True)
            # angle=torch.acos(torch.clamp(dot, -1.0+1e-6, 1.0-1e-6))
            # normalerr=angle/math.pi #map to [0,1 range]
            # normal_loss=normalerr.mean()
        
            #sdfloss = 0.1*torch.nn.functional.smooth_l1_loss(gspoints_sdf, torch.zeros_like(gspoints_sdf),reduction='sum')#.mean()
            surface_sdfloss = 10*self.l1loss(gspoints_sdf,torch.zeros_like(gspoints_sdf))
            
            self.samplepts_smpl.requires_grad_(True)
            sampleptssdf = self.net.IncSDF(self.samplepts_smpl,styles)
            
            s_output_smpl = torch.ones_like(sampleptssdf, requires_grad=False, device=sampleptssdf.device)#
            sdfgradients_samplepoints = torch.autograd.grad(outputs=sampleptssdf,
                                          inputs=self.samplepts_smpl,
                                           grad_outputs=s_output_smpl,
                                          create_graph=True,
                                           retain_graph=True,
                                           only_inputs=True)[0]
            grad_loss = 0.5*(torch.norm(sdfgradients_samplepoints, dim=-1) - 1.0)**2#
            grad_loss = grad_loss.mean()#0.0001.sum()
            
            curvature_loss = self.get_sdf_and_curvature_1d_precomputed_gradient_normal_based(styles, self.samplepts_smpl, sdfgradients_samplepoints)
        else:
            normal_loss = torch.zeros(1).to(self.device)
            surface_sdfloss = torch.zeros(1).to(self.device)
            grad_loss = torch.zeros(1).to(self.device)
            curvature_loss = torch.zeros(1).to(self.device)
            
        if 0:#self.saveiter%200==0 and bidx==0:
            mface = torch.arange(0, points_smpl_disp.shape[0]).reshape(3,-1)
            mesh_test = trimesh.Trimesh(vertices=points_smpl_disp.detach().cpu().numpy(), faces=mface.numpy())
            mesh_test.export("mesh_gs_batch.obj")
            # points_n = points_smpl_disp+gs_normals*0.1
            # mesh_test = trimesh.Trimesh(vertices=points_n.detach().cpu().numpy(), faces=mface.numpy())
            # mesh_test.export("mesh_gs_batch_n.obj")
        self.saveiter = self.saveiter+1
        
        gs_output_smpl = {
            "points": points_smpl_disp,
            "quaternions": quaternions_smpl,
            "gs_scales": gs_scales_smpl,
            "gs_opacity": self.gs_opacity_smpl[bidx],
            "gs_shs": self.gs_shs_smpl[bidx]
        }
        if 0:
            gs_normals=torch.nn.functional.normalize(gs_normals, dim=-1)
            gs_output_smpl["gs_normal"] = gs_normals
            gs_output_smpl["gs_depth"] = points_smpl_disp#[:,2:3].repeat(1,3) #extrinsics is Identity, need transforming to local camera frame
            
        render_smpl_output = self.render_image_gaussian_rasterizer(
            sh_deg=4,
            quaternions=None,
            return_2d_radii=True,
            return_colors=True,
            return_opacities=True,
            positions=points_smpl_disp,
            K = focals,
            cam_poses = cam_poses,
            pytorch3d_K = pytorch3d_K,
            gs_output=gs_output_smpl,
            return_depth = True
        )
        if 0:
            rendered_normal = render_smpl_output["normal"]
            rendered_depth = render_smpl_output["depth"][2:3,...]
            rendered_mask = render_smpl_output["mask"]
            rendered_depth[rendered_mask[2:3,...]==255] = 0
            rendered_normal[rendered_mask==255] = 0
            depthpoints = depth_to_points(rendered_depth, focals[0], self.resH, self.resW)
            depthpoints = depthpoints.permute(2,0,1)
            outputnormal = torch.zeros_like(depthpoints)
            dx = depthpoints[...,2:, 1:-1] - depthpoints[...,:-2, 1:-1]
            dy = depthpoints[...,1:-1, 2:] - depthpoints[...,1:-1, :-2]
            normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=0), dim=0)#local camera frame
            outputnormal[...,1:-1, 1:-1] = normal_map
            outputnormal[rendered_mask==255]=0
            normal_error_map = 1 - (rendered_normal * outputnormal).sum(dim=0)
            depth_normal_loss = normal_error_map.mean()
            if (self.saveiter-1)%200==0 and bidx==0:
                rendernormalimg = rendered_normal.permute(1,2,0).detach().cpu().numpy()
                cv2.imwrite('rendernormal.png', (0.5 * rendernormalimg + 0.5)*255)
                rendernormalimg = outputnormal.permute(1,2,0).detach().cpu().numpy()
                cv2.imwrite('renderdepthnormal.png', (0.5 * rendernormalimg + 0.5)*255)
                rendernormalimg = rendered_mask.permute(1,2,0).detach().cpu().numpy()
                cv2.imwrite('rendered_mask.png', rendernormalimg*255)
                renderdepth = rendered_depth.permute(1,2,0).detach().cpu().numpy()               
                # renderdepth1 = renderdepth[renderdepth>0]
                # renderdepth1 = renderdepth1 - renderdepth1.min()
                # renderdepth[renderdepth>0] = renderdepth1
                cv2.imwrite('renderdepthimg.png', renderdepth*20)

        if 0:
            rendered_expected_coord: torch.Tensor = render_smpl_output["expected_coord"]
            rendered_median_coord: torch.Tensor = render_smpl_output["median_coord"]
            rendered_normal: torch.Tensor = render_smpl_output["normal"]
            
            depth_middepth_normal = depth_double_to_normal(rendered_expected_coord, rendered_median_coord)
            depth_ratio = 0.6
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
            if bidx==0:#(self.saveiter-1)%200==0 and bidx==0:
                rendernormalimg = rendered_normal.permute(1,2,0).detach().cpu().numpy()
                #print(rendered_normal[rendered_normal.abs()>0.01])
                cv2.imwrite('mesh_rendernormalimg.png', (0.5 * rendernormalimg + 0.5)*255)
                rendernormalimg = depth_middepth_normal[0].permute(1,2,0).detach().cpu().numpy()
                cv2.imwrite('mesh_depthnormalimg.png', (0.5 * rendernormalimg + 0.5)*255)
                self.renderdepth = render_smpl_output["expected_depth"].permute(1,2,0)
                renderdepth = self.renderdepth.detach().cpu().numpy()
                io.imsave('mesh_renderdepth.tif', renderdepth*1000)
                renderdepth = render_smpl_output["expected_depth"].permute(1,2,0).detach().cpu().numpy()
                renderdepth1 = renderdepth[renderdepth>0]
                renderdepth1 = renderdepth1 - renderdepth1.min()
                renderdepth[renderdepth>0] = renderdepth1
                cv2.imwrite('mesh_renderdepthimg.png', renderdepth*100)
            
            depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
            
        # self.smpl_img_loss = l1_loss(render_smpl_output['image'].unsqueeze(0), batch['img'] * self.silhouette_smpl.unsqueeze(-1))
        # loss_ssim_smpl = 1.0 - ssim(render_smpl_output['image'].unsqueeze(0), self.silhouette_smpl.unsqueeze(-1))
        # self.smpl_ssim = loss_ssim_smpl * (self.silhouette_smpl.sum() / (render_smpl_output['image'].shape[-1] * render_smpl_output['image'].shape[-2]))
        
        # NOTE: 使用 1-blendmask
        
        rendered_img = render_smpl_output['image']
        
        # renderimg = rendered_img.detach().cpu().numpy()
        # cv2.imwrite('renderimg{:02d}.png'.format(bidx), renderimg)
    
        rendered_img = rendered_img / 127.5 - 1
        #rendered_img = rendered_img*2 - 1
        rendered_img = rendered_img[None]

        ret = {
            'image': rendered_img,
            #'cloth_scene': render_cloth_output,
            #'smpl_scene': render_smpl_output,
            #'gs_output_smpl': gs_output_smpl,
            #'gs_output_cloth': gs_output_cloth
        }
        return ret#, depth_normal_loss, normal_loss, surface_sdfloss, grad_loss, curvature_loss
    
    def gsrendering_multiparts(self, gradtag, bidx, cam_poses, focals):
    
        c2w = cam_poses#[0]torch.cat([cam_poses[0],torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(self.device).view(1,4)],dim=0)
        vm = c2w.inverse().float()#[None]styles, 
        mat_R, mat_T = vm[...,:3, :3].transpose(-2, -1), vm[...,:3, 3]
        projection_matrix = focals
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, self.resH, self.resW)

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
                                         
        #points_smpl = self.net.sugar_model_smpl.get_edited_points(self.deformedpersonsmpl[bidx])
        quaternions_smpl, gs_scales_smpl, gs_normals = self.net.sugar_model_smpl.get_posed_quaternions_and_scales_normals(gs_R=self.blendtransform_gs[:, :, :3, :3][bidx],canmeanpoints=self.canoncalmeansmpl[0],canpoints=self.canoncalsmpl[bidx],points=self.deformedpersonsmpl[bidx], 
                                                                                                          quat=self.rotations_smpl[bidx], 
                                                                                                          sca=self.gs_scales_smpl[bidx])
        # quaternions_smpl, gs_scales_smpl, gs_quat = self.net.sugar_model_smpl.get_posed_quaternions_and_scales2(canpoints=canpersonalsmpl,
                                                                                                      # gs_A=gs_A, 
                                                                                                      # quat=rotations_smpl, 
                                                                                                      # sca=gs_scales_smpl)
        
        points_smpl_disp = self.posedgs[bidx]#points_smpl + self.posedgsdisp[bidx]#adding gs displacement in the posed space
        #quaternions_smpl = rotations_smpl
        
        
        # query_pts = self.canonpersongs[bidx].reshape(1, 1, 1, -1, 3) / 1.3
        # template_sdf = torch.nn.functional.grid_sample(
            # self.sdf_voxels, query_pts,
            # padding_mode = 'border', align_corners = True
        # ).reshape(-1)
        
        if self.saveiter%200==0 and bidx==0:
            mface = torch.arange(0, points_smpl_disp.shape[0]).reshape(3,-1)
            mesh_test = trimesh.Trimesh(vertices=points_smpl_disp.detach().cpu().numpy(), faces=mface.numpy())
            mesh_test.export("mesh_gs_multi.obj")
            # points_n = points_smpl_disp+gs_normals*0.1
            # mesh_test = trimesh.Trimesh(vertices=points_n.detach().cpu().numpy(), faces=mface.numpy())
            # mesh_test.export("mesh_gs_batch_n.obj")
        self.saveiter = self.saveiter+1
        
        rendered_multi_img = []
        for seg in range(0,3):
        
            if seg==0:
                gsidx = self.bodysegidx
            if seg==1:
                gsidx = self.legsegidx
            if seg==2:
                gsidx = self.headsegidx

            gs_output_smpl = {
                "points": points_smpl_disp[gsidx],
                "quaternions": quaternions_smpl[gsidx],
                "gs_scales": gs_scales_smpl[gsidx],
                "gs_opacity": self.gs_opacity_smpl[bidx][gsidx],
                "gs_shs": self.gs_shs_smpl[bidx][gsidx]
            }
            
            render_smpl_output = self.render_image_gaussian_rasterizer(
                sh_deg=4,
                quaternions=None,
                return_2d_radii=True,
                return_colors=True,
                return_opacities=True,
                positions=points_smpl_disp[gsidx],
                K = focals,
                cam_poses = cam_poses,
                pytorch3d_K = pytorch3d_K,
                gs_output=gs_output_smpl,
                return_depth = True
            )
           
            
            rendered_img = render_smpl_output['image']
            
            # renderimg = rendered_img.detach().cpu().numpy()
            # cv2.imwrite('renderimg{:02d}.png'.format(bidx), renderimg)
        
            rendered_img = rendered_img / 127.5 - 1
            rendered_img = rendered_img[None]
            
            rendered_multi_img += [rendered_img]
                 
        rendered_multi_img = torch.cat(rendered_multi_img, 0)#3*H*W*3

        ret = {
            'image': rendered_multi_img[None],#rendered_img,
            #'cloth_scene': render_cloth_output,
            #'smpl_scene': render_smpl_output,
            #'gs_output_smpl': gs_output_smpl,
            #'gs_output_cloth': gs_output_cloth
        }
        return ret#, depth_normal_loss, normal_loss, surface_sdfloss, grad_loss, curvature_loss
    
    def gsrendering_multiparts_fusion(self, gradtag, bidx, cam_poses, focals):
    
        c2w = cam_poses#[0]torch.cat([cam_poses[0],torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(self.device).view(1,4)],dim=0)
        vm = c2w.inverse().float()#[None]styles, 
        mat_R, mat_T = vm[...,:3, :3].transpose(-2, -1), vm[...,:3, 3]
        projection_matrix = focals
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, self.resH, self.resW)

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
                                         
        #points_smpl = self.net.sugar_model_smpl.get_edited_points(self.deformedpersonsmpl[bidx])
        quaternions_smpl, gs_scales_smpl, gs_normals = self.net.sugar_model_smpl.get_posed_quaternions_and_scales_normals(gs_R=self.blendtransform_gs[:, :, :3, :3][bidx],canmeanpoints=self.canoncalmeansmpl[0],canpoints=self.canoncalsmpl[bidx],points=self.deformedpersonsmpl[bidx], 
                                                                                                          quat=self.rotations_smpl[bidx], 
                                                                                                          sca=self.gs_scales_smpl[bidx])
        # quaternions_smpl, gs_scales_smpl, gs_quat = self.net.sugar_model_smpl.get_posed_quaternions_and_scales2(canpoints=canpersonalsmpl,
                                                                                                      # gs_A=gs_A, 
                                                                                                      # quat=rotations_smpl, 
                                                                                                      # sca=gs_scales_smpl)
        
        points_smpl_disp = self.posedgs[bidx]#points_smpl + self.posedgsdisp[bidx]#adding gs displacement in the posed space
        #quaternions_smpl = rotations_smpl
        
        
        # query_pts = self.canonpersongs[bidx].reshape(1, 1, 1, -1, 3) / 1.3
        # template_sdf = torch.nn.functional.grid_sample(
            # self.sdf_voxels, query_pts,
            # padding_mode = 'border', align_corners = True
        # ).reshape(-1)
        
        if self.saveiter%200==0 and bidx==0:
            mface = torch.arange(0, points_smpl_disp.shape[0]).reshape(3,-1)
            mesh_test = trimesh.Trimesh(vertices=points_smpl_disp.detach().cpu().numpy(), faces=mface.numpy())
            mesh_test.export("mesh_gs_multi_fusion.obj")
            # points_n = points_smpl_disp+gs_normals*0.1
            # mesh_test = trimesh.Trimesh(vertices=points_n.detach().cpu().numpy(), faces=mface.numpy())
            # mesh_test.export("mesh_gs_batch_n.obj")
        self.saveiter = self.saveiter+1
        
        rendered_multi_img = []
        for seg in range(0,3):
        
            if seg==0:
                gsidx = self.bodysegidx
            if seg==1:
                gsidx = self.legsegidx
            if seg==2:
                gsidx = self.headsegidx

            gs_output_smpl = {
                "points": points_smpl_disp[gsidx],
                "quaternions": quaternions_smpl[gsidx],
                "gs_scales": gs_scales_smpl[gsidx],
                "gs_opacity": self.gs_opacity_smpl[bidx][gsidx],
                "gs_shs": self.gs_shs_smpl[bidx][gsidx]
            }
            
            render_smpl_output = self.render_image_gaussian_rasterizer(
                sh_deg=4,
                quaternions=None,
                return_2d_radii=True,
                return_colors=True,
                return_opacities=True,
                positions=points_smpl_disp[gsidx],
                K = focals,
                cam_poses = cam_poses,
                pytorch3d_K = pytorch3d_K,
                gs_output=gs_output_smpl,
                return_depth = True
            )
           
            
            rendered_img = render_smpl_output['image']
            
            # renderimg = rendered_img.detach().cpu().numpy()
            # cv2.imwrite('renderimg{:02d}.png'.format(bidx), renderimg)
        
            rendered_img = rendered_img / 127.5 - 1
            rendered_img = rendered_img[None]
            
            rendered_multi_img += [rendered_img]
                 
        rendered_multi_img = torch.cat(rendered_multi_img, 0)#3*H*W*3

        quaternions_smpl_fusion, gs_scales_smpl_fusion, gs_normals_fusion = self.net.sugar_model_smpl.get_posed_quaternions_and_scales_normals(gs_R=self.blendtransform_gs[:, :, :3, :3][bidx],canmeanpoints=self.canoncalmeansmpl[0],canpoints=self.canoncalsmpl[bidx],points=self.deformedpersonsmpl[bidx], 
                                                                                                          quat=self.rotations_smpl_fusion[bidx], 
                                                                                                          sca=self.gs_scales_smpl_fusion[bidx])
        
        points_smpl_disp_fusion = self.posedgs_fusion[bidx]
        
        gs_output_smpl_fusion = {
                "points": points_smpl_disp_fusion,
                "quaternions": quaternions_smpl_fusion,
                "gs_scales": gs_scales_smpl_fusion,
                "gs_opacity": self.gs_opacity_smpl_fusion[bidx],
                "gs_shs": self.gs_shs_smpl_fusion[bidx]
            }
            
        render_smpl_output_fusion = self.render_image_gaussian_rasterizer(
            sh_deg=4,
            quaternions=None,
            return_2d_radii=True,
            return_colors=True,
            return_opacities=True,
            positions=points_smpl_disp_fusion,
            K = focals,
            cam_poses = cam_poses,
            pytorch3d_K = pytorch3d_K,
            gs_output=gs_output_smpl_fusion,
            return_depth = True
        )
       
        
        rendered_img_fusion = render_smpl_output_fusion['image']
        
        # renderimg = rendered_img.detach().cpu().numpy()
        # cv2.imwrite('renderimg{:02d}.png'.format(bidx), renderimg)
    
        rendered_img_fusion = rendered_img_fusion / 127.5 - 1
        rendered_img_fusion = rendered_img_fusion[None]
        
        rendered_multi_img = torch.cat([rendered_multi_img, rendered_img_fusion], 0)#4*H*W*3
        
        ret = {
            'image': rendered_multi_img[None],#rendered_img,
            #'cloth_scene': render_cloth_output,
            #'smpl_scene': render_smpl_output,
            #'gs_output_smpl': gs_output_smpl,
            #'gs_output_cloth': gs_output_cloth
        }
        return ret#, depth_normal_loss, normal_loss, surface_sdfloss, grad_loss, curvature_loss
        
    def get_sdf_and_curvature_1d_precomputed_gradient_normal_based(self, styles, points, sdf_gradients):
        #get the curvature along a certain random direction for each point
        #does it by computing the normal at a shifted point on the tangent plant and then computing a dot produt


        #to the original positions, add also a tiny epsilon 
        nr_points_original=points.shape[0]
        epsilon=1e-4
        rand_directions=torch.randn_like(points)
        rand_directions=F.normalize(rand_directions,dim=-1)

        #instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
        normals=F.normalize(sdf_gradients,dim=-1)
        # normals=normals.detach()
        tangent=torch.cross(normals, rand_directions)
        rand_directions=tangent #set the random moving direction to be the tangent direction now
        

        points_shifted=points.clone()+rand_directions*epsilon
        
        #get the gradient at the shifted point
        #sdf_shifted, sdf_gradients_shifted, feat_shifted=self.get_sdf_and_gradient(points_shifted, iter_nr) 
        
        points_shifted.requires_grad_(True)
        sdf_shifted = self.net.IncSDF(points_shifted,styles)
        
        s_output_smpl = torch.ones_like(sdf_shifted, requires_grad=False, device=sdf_shifted.device)#
        sdf_gradients_shifted = torch.autograd.grad(outputs=sdf_shifted,
                                      inputs=points_shifted,
                                       grad_outputs=s_output_smpl,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
                                           
        normals_shifted=F.normalize(sdf_gradients_shifted,dim=-1)

        dot=(normals*normals_shifted).sum(dim=-1, keepdim=True)
        #the dot would assign low weight importance to normals that are almost the same, and increasing error the more they deviate. So it's something like and L2 loss. But we want a L1 loss so we get the angle, and then we map it to range [0,1]
        angle=torch.acos(torch.clamp(dot, -1.0+1e-6, 1.0-1e-6)) #goes to range 0 when the angle is the same and pi when is opposite


        curvature=angle/math.pi #map to [0,1 range]

        loss_curvature=curvature.mean()
        
        return loss_curvature
        
    def batch_render_deformation_UV(self, styles, cam_poses, focals, beta, theta, trans):
        
        thetazero = batch_rodrigues(torch.zeros_like(theta).reshape(-1, 3)).reshape(-1, 24, 3, 3)
        
        so = self.smpl_model(betas = torch.zeros_like(beta).reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        canoncalmeansmpl = so['vertices'].clone().reshape(-1, 6890, 3)#mean smpl
        
        so = self.smpl_model(betas = beta.reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        canoncalsmpl = so['vertices'].clone().reshape(-1, 6890, 3)

        c2w = cam_poses#[0]torch.cat([cam_poses[0],torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(self.device).view(1,4)],dim=0)
        vm = c2w.inverse().float()#[None]
        mat_R, mat_T = vm[...,:3, :3].transpose(-2, -1), vm[...,:3, 3]
        projection_matrix = focals
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, 512, 256)

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
                                         
                     
        self.net.position_enc_smpl.uvgsfeature(styles)
        
        #self.net.sugar_model_smpl._points = canoncalmeansmpl#canpersonalsmpl#new mesh point

        #points_meansmpl = self.net.sugar_model_smpl.getbatch_edited_points(canoncalmeansmpl)  #self.net.sugar_model_smpl.points canoncalmeansmpl 

        gs_uv = self.net.sugar_model_smpl.UV_GS

        gs_pos, gs_rot, gs_scale, gs_color, gs_opacity = self.net.position_enc_smpl.forward_batch(gs_uv)#gaussian center with mean smpl vertex for fetching feature
        
        #plane_opacity = self.net.position_enc_smpl.plane_opacity
        #self.opacity_loss = (torch.log(gs_opacity)+torch.log(1-gs_opacity)).abs().mean()
        self.opacity_loss = torch.mean(
                    torch.log(0.1 + self.net.position_enc_smpl.plane_opacity) +
                    torch.log(0.1 + 1. - self.net.position_enc_smpl.plane_opacity) - -2.20727)
        
        self.postv_loss = self.tv_loss(self.net.position_enc_smpl.plane_pos, 0.1)
        
        self.scale_loss = 0.1*torch.norm(self.net.position_enc_smpl.plane_scale-0.01).mean()
                    
        # 预测透明度和球谐函数参数(SMPL和CLOTH) 
        gs_opacity_smpl = gs_opacity#torch.ones_like(points_meansmpl)[...,0:1]#
        # gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 16, 3)
        gs_shs_smpl = gs_color

        # 预测旋转和尺度(SMPL和CLOTH)
        rotations_smpl = gs_rot
        gs_scales_smpl = gs_scale
        #gs_xyz_smpl = geometry_out_smpl['xyz']
       
        canpersonalsmpl= canoncalsmpl#+vertoffset+tri_feats_smpl[:,:3]
        
        #self.incdisploss = 20000.0*torch.nn.functional.smooth_l1_loss(gs_pos, torch.zeros_like(gs_pos)).mean()
        self.incdisploss = 0.1*torch.nn.functional.smooth_l1_loss(self.net.position_enc_smpl.plane_pos, torch.zeros_like(self.net.position_enc_smpl.plane_pos),reduction='sum', beta=0.1)#.mean()
        
        #self.incrotloss = 20000.0*torch.nn.functional.smooth_l1_loss(rotations_smpl, torch.zeros_like(rotations_smpl)).mean()
        self.incrotloss = 0.1*torch.nn.functional.smooth_l1_loss(self.net.position_enc_smpl.plane_rot, torch.zeros_like(self.net.position_enc_smpl.plane_rot),reduction='sum')#.mean()
        
        # self.deformedpersonsmpl, smpl_A = self.deforming_lbs_newsmpl_batch(beta, theta, trans, canpersonalsmpl)

        # #posing gs displacement
        # posedgsdisp, gs_A = self.deforming_lbs_gsdisp_batch(gs_pos, smpl_A, trans)
        
        #mesh = struct.Meshes(verts=self.deformedpersonsmpl[None], faces=self.smplfaces) 
        
        # mesh_test = trimesh.Trimesh(vertices=meshvert_def[0].detach().cpu().numpy(), faces=self.meshface[0].detach().cpu().numpy())
        # mesh_test.export("mesh_test.obj")
        
        # fragments = self.net.rasterizer(mesh, cameras=cameras)
        # depth = fragments.zbuf
        # face_idx_map = fragments.pix_to_face[..., 0]
        # self.rendermask = torch.where(depth > 0, torch.ones_like(depth),torch.zeros_like(depth))
        # self.rendermask = self.rendermask.squeeze(-1).float()  
        # mesh_mask = self.rendermask[0].detach().cpu().numpy()
        # cv2.imwrite('mesh_mask.png', mesh_mask*255)
        
                    
                
        rendered_img_batch = []
        batchsize = beta.shape[0]
        for bidx in range(0, batchsize):
            
            self.deformedpersonsmpl, smpl_A = self.deforming_lbs_newsmpl(beta[bidx:bidx+1], theta[bidx:bidx+1], trans[bidx:bidx+1], canpersonalsmpl[bidx])

            posedgsdisp, _ = self.deforming_lbs_gsdisp(gs_pos[bidx], smpl_A, trans[bidx])
            
            points_smpl = self.net.sugar_model_smpl.get_edited_points(self.deformedpersonsmpl)
            #quaternions_smpl = self.net.sugar_model_smpl.get_posed_quaternions(points=self.deformedpersonsmpl, quat=0)
            
            #quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_posed_quaternions_scale2D(points=self.deformedpersonsmpl, quat=rotations_smpl, sca=gs_scales_smpl)
            
            # quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_edited_quaternions_and_scales_with_points(points=self.deformedpersonsmpl, 
                                                                                                                        # quat=rotations_smpl, 
                                                                                                                        # sca=gs_scales_smpl)
            
            quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_posed_quaternions_and_scales(points=self.deformedpersonsmpl, 
                                                                                                          quat=rotations_smpl[bidx], 
                                                                                                          sca=gs_scales_smpl[bidx])
            # quaternions_smpl, gs_scales_smpl, gs_quat = self.net.sugar_model_smpl.get_posed_quaternions_and_scales2(canpoints=canpersonalsmpl,
                                                                                                          # gs_A=gs_A, 
                                                                                                          # quat=rotations_smpl, 
                                                                                                          # sca=gs_scales_smpl)
            
            points_smpl_disp = points_smpl + posedgsdisp#adding gs displacement in the posed space
            #quaternions_smpl = rotations_smpl
            
            gs_output_smpl = {
                "points": points_smpl_disp,
                "quaternions": quaternions_smpl,
                "gs_scales": gs_scales_smpl,
                "gs_opacity": gs_opacity_smpl[bidx],
                "gs_shs": gs_shs_smpl[bidx]
            }
            render_smpl_output = self.render_image_gaussian_rasterizer(
                sh_deg=4,
                quaternions=None,
                return_2d_radii=True,
                return_colors=True,
                return_opacities=True,
                positions=points_smpl_disp,
                K = focals,
                cam_poses = cam_poses,
                pytorch3d_K = pytorch3d_K,
                gs_output=gs_output_smpl
            )
            
            if 0:
                rendered_expected_coord: torch.Tensor = render_smpl_output["expected_coord"]
                rendered_median_coord: torch.Tensor = render_smpl_output["median_coord"]
                rendered_normal: torch.Tensor = render_smpl_output["normal"]
                depth_middepth_normal = depth_double_to_normal(rendered_expected_coord, rendered_median_coord)
                depth_ratio = 0.6
                normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
                self.depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
                
            # self.smpl_img_loss = l1_loss(render_smpl_output['image'].unsqueeze(0), batch['img'] * self.silhouette_smpl.unsqueeze(-1))
            # loss_ssim_smpl = 1.0 - ssim(render_smpl_output['image'].unsqueeze(0), self.silhouette_smpl.unsqueeze(-1))
            # self.smpl_ssim = loss_ssim_smpl * (self.silhouette_smpl.sum() / (render_smpl_output['image'].shape[-1] * render_smpl_output['image'].shape[-2]))
            
            # NOTE: 使用 1-blendmask
            
            rendered_img = render_smpl_output['image']
            
            # renderimg = rendered_img.detach().cpu().numpy()
            # cv2.imwrite('renderimg{:02d}.png'.format(bidx), renderimg)
        
            rendered_img = rendered_img / 127.5 - 1
            rendered_img_batch += [rendered_img[None]]
        
        rendered_img_batch = torch.cat(rendered_img_batch, 0)
        ret = {
            'image': rendered_img_batch,
            #'cloth_scene': render_cloth_output,
            #'smpl_scene': render_smpl_output,
            #'gs_output_smpl': gs_output_smpl,
            #'gs_output_cloth': gs_output_cloth
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
    
    def batch_generateGS_body_clothing(self, styles, cam_poses, focals, beta, theta, trans):
        
        thetazero = batch_rodrigues(torch.zeros_like(theta).reshape(-1, 3)).reshape(-1, 24, 3, 3)
        
        so = self.smpl_model(betas = torch.zeros_like(beta).reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        canoncalmeansmpl = so['vertices'].clone().reshape(-1, 6890, 3)#mean smpl
        
        so = self.smpl_model(betas = beta.reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        canoncalsmpl = so['vertices'].clone().reshape(-1, 6890, 3)
             
        weight1 = torch.from_numpy(np.array([-0.8199, -0.0786], dtype = np.float32))
        tempclothpara = weight1.view(1,2).to('cuda')
        tempclothpara = 2*torch.sigmoid(tempclothpara)-1 
        zeropara = torch.zeros_like(tempclothpara).to('cuda')
        tempclothpara = torch.cat((tempclothpara,zeropara),-1)
        tempclothpara = tempclothpara.view(-1,4).repeat(theta.shape[0],1)
        
        canoncalcloth = self.cloth_simulation(torch.zeros_like(theta),beta,tempclothpara)
        #clothvert = self.net.cloth_simulation(theta,beta,self.tempclothpara)
        
        canoncalmeancloth = self.cloth_simulation(torch.zeros_like(theta),torch.zeros_like(beta),tempclothpara)
        canoncalmean_smplcloth = torch.cat([canoncalmeancloth, canoncalmeansmpl], dim=1)
        
        deformedsmpl, deformedcloth = self.deforming_lbs_body_clothing_batch(beta, theta, trans, canoncalsmpl, canoncalcloth)
        
        self.deformed_smplcloth = torch.cat([deformedcloth, deformedsmpl], dim=1)
        
        mesh_test = trimesh.Trimesh(vertices=self.deformed_smplcloth[0].detach().cpu().numpy(), faces=self.meshface[0].detach().cpu().numpy())
        mesh_test.export("mesh_test.obj")
        
        points_meansmplcloth = self.net.sugar_model_smplcloth.get_edited_points(canoncalmean_smplcloth[0])  #self.net.sugar_model_smpl.points canoncalmeansmpl 
        points_meansmplcloth = points_meansmplcloth.repeat(theta.shape[0],1,1)
        #self.net.sugar_model_smpl._points = canpersonalsmpl#for computing mesh normal        
        # 3dgs渲染_,
        tri_feats_smplcloth = self.net.position_enc_smplcloth.forward_batch(points_meansmplcloth)#gaussian center with mean smpl vertex for fetching feature
        
        style = styles[:,None].repeat(1,tri_feats_smplcloth.shape[1],1)
        tri_feats_smplcloth = torch.concatenate([tri_feats_smplcloth,style],dim=-1)
       
        #tri_feats_smpl = styles.repeat(points_meansmpl.shape[0],1)

        # import open3d as o3d
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(self.net.sugar_model_cloth.points.cpu().detach().numpy())
        # o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
        
        appearance_out_smplcloth = self.net.appearance_dec_smplcloth(tri_feats_smplcloth)
       
        geometry_out_smplcloth = self.net.geometry_dec_smplcloth(tri_feats_smplcloth)
        
        
        # 预测透明度和球谐函数参数(SMPL和CLOTH)
        self.gs_opacity_smplcloth = appearance_out_smplcloth['opacity']
        # gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 16, 3)
        self.gs_shs_smplcloth = appearance_out_smplcloth['shs'].reshape(-1, 3)

        # 预测旋转和尺度(SMPL和CLOTH)
        self.rotations_smplcloth = geometry_out_smplcloth['rotations']
        self.gs_scales_smplcloth = geometry_out_smplcloth['scales']
        #gs_xyz_smpl = geometry_out_smpl['xyz']
       
        self.opacity_loss = torch.mean(
                    torch.log(0.1 + self.gs_opacity_smplcloth) +
                    torch.log(0.1 + 1. -  self.gs_opacity_smplcloth) - -2.20727)
                    
        #self.scale_loss = 0.1*torch.norm(geometry_out_smplcloth['scales1']-(-4.7)).mean()#0.012
        self.scale_loss = 0.1*torch.norm(self.gs_scales_smplcloth-0.02).mean()#0.012
                  
    def gsrendering_body_clothing(self,bidx,cam_poses, focals):
    
        c2w = cam_poses#[0]torch.cat([cam_poses[0],torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(self.device).view(1,4)],dim=0)
        vm = c2w.inverse().float()#[None]
        mat_R, mat_T = vm[...,:3, :3].transpose(-2, -1), vm[...,:3, 3]
        projection_matrix = focals
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, 512, 256)

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
                                         
        points_smplcloth = self.net.sugar_model_smplcloth.get_edited_points(self.deformed_smplcloth[bidx])
        quaternions_smplcloth, gs_scales_smplcloth = self.net.sugar_model_smplcloth.get_edited_quaternions_and_scales_with_points(points=self.deformed_smplcloth[bidx], 
                                                                                                                    quat=self.rotations_smplcloth[bidx], 
                                                                                                                    sca=self.gs_scales_smplcloth[bidx])
        
        
        gs_output_smpl = {
            "points": points_smplcloth,
            "quaternions": quaternions_smplcloth,
            "gs_scales": gs_scales_smplcloth,
            "gs_opacity": self.gs_opacity_smplcloth[bidx],
            "gs_shs": self.gs_shs_smplcloth[bidx]
        }
        render_smplcloth_output = self.render_image_gaussian_rasterizer(
            sh_deg=4,
            quaternions=None,
            return_2d_radii=True,
            return_colors=True,
            return_opacities=True,
            positions=points_smplcloth,
            K = focals,
            cam_poses = cam_poses,
            pytorch3d_K = pytorch3d_K,
            gs_output=gs_output_smpl
        )
        
        
        rendered_img = render_smplcloth_output['image']
        
        # renderimg = rendered_img.detach().cpu().numpy()
        # cv2.imwrite('renderimg{:02d}.png'.format(bidx), renderimg)
    
        rendered_img = rendered_img / 127.5 - 1
        rendered_img = rendered_img[None]

        ret = {
            'image': rendered_img,
            #'cloth_scene': render_cloth_output,
            #'smpl_scene': render_smpl_output,
            #'gs_output_smpl': gs_output_smpl,
            #'gs_output_cloth': gs_output_cloth
        }
        return ret
    
    def batch_generatemesh_body_clothing_separate(self, clothweight, cam_poses, focals, beta, theta, trans):
        
        thetazero = batch_rodrigues(torch.zeros_like(theta).reshape(-1, 3)).reshape(-1, 24, 3, 3)
        
        so = self.smpl_model(betas = torch.zeros_like(beta).reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        canoncalmeansmpl = so['vertices'].clone().reshape(-1, 6890, 3)#mean smpl
        
        so = self.smpl_model(betas = beta.reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        canoncalsmpl = so['vertices'].clone().reshape(-1, 6890, 3)
        
        #clothweight = torch.rand(theta.shape[0], 2).to('cuda')
        tempclothpara = (self.clothpararange[:,1][None] - self.clothpararange[:,0][None]) * clothweight + self.clothpararange[:,0][None]
        tempclothpara = 2*torch.sigmoid(tempclothpara)-1 
        zeropara = torch.zeros_like(tempclothpara).to('cuda')
        tempclothpara = torch.cat((tempclothpara,zeropara),-1)
       
        
        # weight1 = torch.from_numpy(np.array([-0.8199, -0.0786], dtype = np.float32))
        # tempclothpara = weight1.view(1,2).to('cuda')
        # tempclothpara = 2*torch.sigmoid(tempclothpara)-1 
        # zeropara = torch.zeros_like(tempclothpara).to('cuda')
        # tempclothpara = torch.cat((tempclothpara,zeropara),-1)
        # tempclothpara = tempclothpara.view(-1,4).repeat(theta.shape[0],1)
        
        canoncalcloth = self.cloth_simulation_prior(torch.zeros_like(theta),torch.zeros_like(beta),tempclothpara)
        #clothvert = self.net.cloth_simulation(theta,beta,self.tempclothpara)
        
        deformfeature = torch.cat([theta,beta,tempclothpara],dim=-1)
        self.deformation_affine, self.deformation_transl = self.predicting_deformation(deformfeature)
        #self.deformation_affine_smpl, self.deformation_transl_smpl = self.predicting_deformation_smpl(smpllatent)
    
        #clothvert = self.net.update_clothshape()
        self.update_embeddedgraph_cloth(canoncalcloth)
        #self.update_embeddedgraph_smpl(self.rawtemplatesmpl.repeat(batch_size,1,1))
        
        self.defsmoothloss = self.deformationsmoothloss()
        #self.smoothloss_smpl = self.deformationsmoothloss_smpl()
        
        graphdeformedverts_cloth = self.deformingtemplate(canoncalcloth)

        newcanoncalcloth = self.cloth_simulation(torch.zeros_like(theta),beta,tempclothpara)
        self.simulloss = self.l1loss(graphdeformedverts_cloth, newcanoncalcloth)
 
        # #self.deltadeformloss_smpl = self.deform_crit(smplgraphdeformedverts,self.rawtemplatesmpl.repeat(batch_size,1,1))#torch.zeros_like(smplgraphdeformedverts)
        # self.deltadeformloss = self.l1loss(graphdeformedverts_cloth, canoncalcloth)#torch.zeros_like(graphdeformedverts)
                
        self.deformedsmpl, self.deformedcloth = self.deforming_lbs_body_clothing_batch(beta, theta, trans, canoncalsmpl, graphdeformedverts_cloth)#canoncalcloth
 
        self.interploss = self.computeinterpenetrationloss_posedsmpl(graphdeformedverts_cloth, canoncalsmpl, self.deformedsmpl, self.deformedcloth)
 
        self.deformed_smplcloth = torch.cat([self.deformedcloth, self.deformedsmpl], dim=1)
        
        if self.saveiter%500==0:
            mesh_test = trimesh.Trimesh(vertices=self.deformed_smplcloth[0].detach().cpu().numpy(), faces=self.meshface[0].detach().cpu().numpy())
            mesh_test.export("mesh_test.obj")
        self.saveiter = self.saveiter+1
            
        attachptsdist = torch.sqrt(torch.sum((self.deformedcloth[:,self.attachvertidx[:,0]]-self.deformedsmpl[:,self.attachvertidx[:,1]])**2,-1))       
        self.attach_loss = torch.sum((attachptsdist-0.01)**2)       
    
    def batch_generateGS_body(self, styles, cam_poses, focals, beta, theta, trans):
        
        thetazero = batch_rodrigues(torch.zeros_like(theta).reshape(-1, 3)).reshape(-1, 24, 3, 3)
        
        # so = self.smpl_model(betas = torch.zeros_like(beta).reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        # canoncalmeansmpl = so['vertices'].clone().reshape(-1, 6890, 3)#mean smpl
        
        so = self.smpl_model(betas = beta.reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        canoncalsmpl = so['vertices'].clone().reshape(-1, 6890, 3)
        canoncalmeansmpl = self.canoncalmeansmpl.clone()
        
        points_meansmpl = self.net.sugar_model_smpl.get_edited_points(canoncalmeansmpl[0])  #self.net.sugar_model_smpl.points canoncalmeansmpl 
        
        points_meansmpl = points_meansmpl.repeat(theta.shape[0],1,1)
        
        tri_feats_smpl = self.net.position_enc_smpl.forward_batch(points_meansmpl)#gaussian center with mean smpl vertex for fetching feature
        
        style = styles[:,None].repeat(1,tri_feats_smpl.shape[1],1)
        tri_feats_smpl = torch.concatenate([tri_feats_smpl,style],dim=-1)
        
        appearance_out_smpl = self.net.appearance_dec_smpl(tri_feats_smpl)
       
        geometry_out_smpl = self.net.geometry_dec_smpl(tri_feats_smpl)
        
        self.gs_opacity_smpl = appearance_out_smpl['opacity']
        # gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 16, 3)
        self.gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 3)

        self.rotations_smpl = geometry_out_smpl['rotations']
        self.gs_scales_smpl = geometry_out_smpl['scales']
        #gs_xyz_smpl = geometry_out_smpl['xyz']
       
        tri_feats_smplvert = self.net.position_enc_smpl.forward_batch(canoncalmeansmpl)#gaussian center with mean smpl vertex for fetching feature
        
        style = styles[:,None].repeat(1,tri_feats_smplvert.shape[1],1)
        tri_feats_smplvert = torch.concatenate([tri_feats_smplvert,style],dim=-1)
        
        self.displace_smpl = self.net.displace_dec_smpl(tri_feats_smplvert)
        
        # self.net.position_enc_incsmpl.uvvertfeature(styles)
        
        # vert_uv = self.net.sugar_model_smpl.vert_uv

        # self.displace_smpl = self.net.position_enc_incsmpl.forward_batch(vert_uv)#gaussian center with mean smpl vertex for fetching feature
        
        displaceverts_smpl =  canoncalmeansmpl + self.displace_smpl#canoncalsmpl
        
        self.deformedsmpl, _ = self.deforming_lbs_newsmpl_batch(beta, theta, trans, displaceverts_smpl)

        if self.saveiter%100==0:
            mesh_test = trimesh.Trimesh(vertices=self.deformedsmpl[0].detach().cpu().numpy(), faces=self.smplfaces[0].detach().cpu().numpy())
            mesh_test.export("mesh_test_batch.obj")
        self.saveiter = self.saveiter+1
        #net.position_enc_incsmpl.plane_pos
        self.incdisploss = 0.01*torch.nn.functional.smooth_l1_loss(self.displace_smpl, torch.zeros_like(self.displace_smpl),reduction='sum', beta=0.1)#.mean()
        
        #self.mormalsmoothloss = 0
        #self.laplacsmoothloss = 0
        #for i in range(0, self.deformedsmpl.shape[0]):
        # i=0
        # self.mormalsmoothloss =  8.0 * mesh_normal_consistency(self.net.sugar_model_smpl.deform_mesh(self.deformedsmpl[i]))                
        # self.laplacsmoothloss = 20.0 * mesh_laplacian_smoothing(self.net.sugar_model_smpl.deform_mesh(self.deformedsmpl[i]), method="uniform")
        
        allmormalsmoothloss = []
        alllaplacsmoothloss = []
        for i in range(0, self.deformedsmpl.shape[0]):
            mormalsmoothloss =  10.0 * mesh_normal_consistency(self.net.sugar_model_smpl.deform_mesh(self.deformedsmpl[i]))                
            laplacsmoothloss = 5.0 * mesh_laplacian_smoothing(self.net.sugar_model_smpl.deform_mesh(self.deformedsmpl[i]), method="uniform")
            allmormalsmoothloss += [mormalsmoothloss.view([-1])]
            alllaplacsmoothloss += [laplacsmoothloss.view([-1])]
        allmormalsmoothloss = torch.cat(allmormalsmoothloss, 0)                  
        alllaplacsmoothloss = torch.cat(alllaplacsmoothloss, 0)
        self.mormalsmoothloss = allmormalsmoothloss.sum()
        self.laplacsmoothloss = alllaplacsmoothloss.sum()
        
        self.opacity_loss = torch.mean(
                    torch.log(0.1 + self.gs_opacity_smpl) +
                    torch.log(0.1 + 1. -  self.gs_opacity_smpl) - -2.20727)
        
        #self.scale_loss = 0.1*torch.norm(geometry_out_smplcloth['scales1']-(-4.7)).mean()#0.012
        self.scale_loss = 0.1*torch.norm(self.gs_scales_smpl-0.02).mean()#0.012
        
    def batch_generateGS_body_clothing_separate(self, clothweight, styles, cam_poses, focals, beta, theta, trans):
        
        thetazero = batch_rodrigues(torch.zeros_like(theta).reshape(-1, 3)).reshape(-1, 24, 3, 3)
        
        so = self.smpl_model(betas = torch.zeros_like(beta).reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        canoncalmeansmpl = so['vertices'].clone().reshape(-1, 6890, 3)#mean smpl
        
        so = self.smpl_model(betas = beta.reshape(-1, 10), body_pose = thetazero[:, 1:], global_orient = thetazero[:, 0].view(-1, 1, 3, 3))
        canoncalsmpl = so['vertices'].clone().reshape(-1, 6890, 3)
        
        #clothweight = torch.rand(theta.shape[0], 2).to('cuda')
        tempclothpara = (self.clothpararange[:,1][None] - self.clothpararange[:,0][None]) * clothweight + self.clothpararange[:,0][None]
        tempclothpara = 2*torch.sigmoid(tempclothpara)-1 
        zeropara = torch.zeros_like(tempclothpara).to('cuda')
        tempclothpara = torch.cat((tempclothpara,zeropara),-1)
       
        
        # weight1 = torch.from_numpy(np.array([-0.8199, -0.0786], dtype = np.float32))
        # tempclothpara = weight1.view(1,2).to('cuda')
        # tempclothpara = 2*torch.sigmoid(tempclothpara)-1 
        # zeropara = torch.zeros_like(tempclothpara).to('cuda')
        # tempclothpara = torch.cat((tempclothpara,zeropara),-1)
        # tempclothpara = tempclothpara.view(-1,4).repeat(theta.shape[0],1)
        
        canoncalcloth = self.cloth_simulation(torch.zeros_like(theta),beta,tempclothpara)
        #clothvert = self.net.cloth_simulation(theta,beta,self.tempclothpara)
        
        # self.deformation_affine, self.deformation_transl = self.predicting_deformation(styles)
        # #self.deformation_affine_smpl, self.deformation_transl_smpl = self.predicting_deformation_smpl(smpllatent)
    
        # #clothvert = self.net.update_clothshape()
        # self.update_embeddedgraph_cloth(canoncalcloth)
        # #self.update_embeddedgraph_smpl(self.rawtemplatesmpl.repeat(batch_size,1,1))
        
        # self.defsmoothloss = self.deformationsmoothloss()
        # #self.smoothloss_smpl = self.deformationsmoothloss_smpl()
        
        # graphdeformedverts_cloth = self.deformingtemplate(canoncalcloth)

        # #self.deltadeformloss_smpl = self.deform_crit(smplgraphdeformedverts,self.rawtemplatesmpl.repeat(batch_size,1,1))#torch.zeros_like(smplgraphdeformedverts)
        # self.deltadeformloss = self.l1loss(graphdeformedverts_cloth, canoncalcloth)#torch.zeros_like(graphdeformedverts)
        
        
        canoncalmeancloth = self.cloth_simulation(torch.zeros_like(theta),torch.zeros_like(beta),tempclothpara)
        canoncalmean_smplcloth = torch.cat([canoncalmeancloth, canoncalmeansmpl], dim=1)
        
        # self.deformedsmpl, self.deformedcloth = self.deforming_lbs_body_clothing_batch(beta, theta, trans, canoncalsmpl, graphdeformedverts_cloth)#canoncalcloth
 
        # self.interploss = self.computeinterpenetrationloss_posedsmpl(graphdeformedverts_cloth, canoncalsmpl, self.deformedsmpl, self.deformedcloth)
 
        # self.deformed_smplcloth = torch.cat([self.deformedcloth, self.deformedsmpl], dim=1)
        
        # mesh_test = trimesh.Trimesh(vertices=self.deformed_smplcloth[0].detach().cpu().numpy(), faces=self.meshface[0].detach().cpu().numpy())
        # mesh_test.export("mesh_test.obj")
        
        points_meansmpl = self.net.sugar_model_smpl.get_edited_points(canoncalmeansmpl[0])  #self.net.sugar_model_smpl.points canoncalmeansmpl 
        points_meansmpl = points_meansmpl.repeat(theta.shape[0],1,1)
        
        points_meancloth = self.net.sugar_model_cloth.getbatch_edited_points(canoncalmeancloth)  #self.net.sugar_model_smpl.points canoncalmeansmpl 
        #points_meancloth = points_meancloth.repeat(theta.shape[0],1,1)
        
        #self.net.sugar_model_smpl._points = canpersonalsmpl#for computing mesh normal        
        # 3dgs渲染_,
        tri_feats_smpl = self.net.position_enc_smpl.forward_batch(points_meansmpl)#gaussian center with mean smpl vertex for fetching feature
        tri_feats_cloth = self.net.position_enc_cloth.forward_batch(points_meancloth)#gaussian center with mean smpl vertex for fetching feature
        
        style = styles[:,None].repeat(1,tri_feats_smpl.shape[1],1)
        tri_feats_smpl = torch.concatenate([tri_feats_smpl,style],dim=-1)
        
        style = styles[:,None].repeat(1,tri_feats_cloth.shape[1],1)
        tri_feats_cloth = torch.concatenate([tri_feats_cloth,style],dim=-1)
        
        #tri_feats_smpl = styles.repeat(points_meansmpl.shape[0],1)

        # import open3d as o3d
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(self.net.sugar_model_cloth.points.cpu().detach().numpy())
        # o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
        
        appearance_out_smpl = self.net.appearance_dec_smpl(tri_feats_smpl)
       
        geometry_out_smpl = self.net.geometry_dec_smpl(tri_feats_smpl)
        
        appearance_out_cloth = self.net.appearance_dec_cloth(tri_feats_cloth)
       
        geometry_out_cloth = self.net.geometry_dec_cloth(tri_feats_cloth)

        self.gs_opacity_smpl = appearance_out_smpl['opacity']
        # gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 16, 3)
        self.gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 3)

        self.rotations_smpl = geometry_out_smpl['rotations']
        self.gs_scales_smpl = geometry_out_smpl['scales']
        #gs_xyz_smpl = geometry_out_smpl['xyz']
       
        self.gs_opacity_cloth = appearance_out_cloth['opacity']
        # gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 16, 3)
        self.gs_shs_cloth = appearance_out_cloth['shs'].reshape(-1, 3)

        self.rotations_cloth = geometry_out_cloth['rotations']
        self.gs_scales_cloth = geometry_out_cloth['scales']
        #self.gs_xyz_cloth = geometry_out_cloth['xyz']
        
        tri_feats_clothvert = self.net.position_enc_cloth.forward_batch(canoncalmeancloth)#gaussian center with mean smpl vertex for fetching feature
        
        style = styles[:,None].repeat(1,tri_feats_clothvert.shape[1],1)
        tri_feats_clothvert = torch.concatenate([tri_feats_clothvert,style],dim=-1)
        
        self.displace_cloth = self.net.displace_dec_cloth(tri_feats_clothvert)
        
        displaceverts_cloth = canoncalcloth + self.displace_cloth
        
        self.deformedsmpl, self.deformedcloth = self.deforming_lbs_body_clothing_batch(beta, theta, trans, canoncalsmpl, displaceverts_cloth)#canoncalcloth
 
        self.interploss = self.computeinterpenetrationloss_posedsmpl(displaceverts_cloth, canoncalsmpl, self.deformedsmpl, self.deformedcloth)
 
        self.deformed_smplcloth = torch.cat([self.deformedcloth, self.deformedsmpl], dim=1)
        
        self.incdisploss = 0.001*torch.nn.functional.smooth_l1_loss(self.displace_cloth, torch.zeros_like(self.displace_cloth),reduction='sum', beta=0.1)#.mean()
        
        attachptsdist = torch.sqrt(torch.sum((self.deformedcloth[:,self.attachvertidx[:,0]]-self.deformedsmpl[:,self.attachvertidx[:,1]])**2,-1))       
        self.attach_loss = torch.sum((attachptsdist-0.01)**2)
        
        self.mormalsmoothloss = 0
        self.laplacsmoothloss = 0
        for i in range(0, self.deformedcloth.shape[0]):
            self.mormalsmoothloss +=  1.0 * mesh_normal_consistency(self.net.sugar_model_cloth.deform_mesh(self.deformedcloth[i]))                
            self.laplacsmoothloss += 1.0 * mesh_laplacian_smoothing(self.net.sugar_model_cloth.deform_mesh(self.deformedcloth[i]), method="uniform")
                          
        opacity_loss_smpl = torch.mean(
                    torch.log(0.1 + self.gs_opacity_smpl) +
                    torch.log(0.1 + 1. -  self.gs_opacity_smpl) - -2.20727)
        opacity_loss_cloth = torch.mean(
                    torch.log(0.1 + self.gs_opacity_cloth) +
                    torch.log(0.1 + 1. -  self.gs_opacity_cloth) - -2.20727)
        self.opacity_loss = opacity_loss_smpl + opacity_loss_cloth
        
        #self.scale_loss = 0.1*torch.norm(geometry_out_smplcloth['scales1']-(-4.7)).mean()#0.012
        scale_loss_smpl = 0.1*torch.norm(self.gs_scales_smpl-0.02).mean()#0.012
        scale_loss_cloth = 0.1*torch.norm(self.gs_scales_cloth-0.02).mean()#
        self.scale_loss = scale_loss_smpl + scale_loss_cloth#
        
    def render_clothmask(self, faces: torch.Tensor, verts: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        """
        # Build meshes.
        if len(verts.shape) == 2:
            verts = verts[None]
        if len(faces.shape) == 2:
            faces = faces[None]
            faces = faces.repeat(verts.shape[0], 1, 1)
        meshes = pytorch3d.structures.Meshes(verts=verts, faces=faces)
        # Render entire batch.

        vm = extrinsics
        mat_R, mat_T = vm[...,:3, :3].transpose(-2, -1), vm[...,:3, 3]
        projection_matrix = intrinsics
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, 512, 512)

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
                                         
        rasterizer=pytorch3d.renderer.MeshRasterizer(
                cameras=cameras, 
                raster_settings=pytorch3d.renderer.RasterizationSettings(
                    image_size=(512, 512),
                    blur_radius=0.0, 
                    faces_per_pixel=1, 
                    # cull_backfaces=False,
                    # z_clip_value=None,
                    # cull_to_frustum=False,
                    # perspective_correct=True,
                ),            
            )
            
        
        # localverts = torch.matmul(meshes.verts_padded(), mat_R) + mat_T.unsqueeze(1)
        # pixel2d = torch.matmul(projection_matrix, localverts.transpose(-2, -1))
        # pixel2d = pixel2d.transpose(-2, -1)
        # pixel2d = torch.divide(pixel2d,pixel2d[...,2].unsqueeze(-1))
           
        # Rasterize.
        # raster = rasterizer(meshes)
        
        # depth = raster.zbuf[...,0]          
        # rendermask = torch.where(depth > 0, torch.ones_like(depth),torch.zeros_like(depth))     
        # mask_render = rendermask[0].detach().cpu().numpy()
        # mask_path = 'test/smplmask{:04d}.png'.format(frame_index)       
        # cv2.imwrite(mask_path, mask_render * 255) , dtype=torch.long
        
        fragments = rasterizer(meshes)
        depth = fragments.zbuf
        face_idx_map = fragments.pix_to_face[..., 0]
        #idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(batch_size)]
        #idx = [torch.Tensor(i * faces.shape[1]) for i in range(self.clothfaces.shape[0])]
        #idx = torch.cat(idx).to(self.device)
        idx = []
        for i in range(0, faces.shape[0]):
            idx0 = torch.full([face_idx_map.shape[1],face_idx_map.shape[2]],i * faces.shape[1], dtype=torch.long)
            idx.append(idx0[None])
        idx = torch.cat(idx).to(face_idx_map)

        face_idx_map = face_idx_map - idx
        
        silhouette0 = face_idx_map>=self.clothfaces.shape[1]# body mask

        silhouette = ~silhouette0
       
        #silhouette = face_idx_map<self.clothfaces.shape[0]# cloth mask
       
        silhouette = silhouette.squeeze(-1).float()
        
        silhouette_smpl = (face_idx_map<self.clothfaces.shape[1])&(face_idx_map>=0)# cloth mask

        silhouette_smpl = ~silhouette_smpl
       
        silhouette_smpl = silhouette_smpl.float()
        
        #meshsilhouette = torch.where(depth > 0, torch.ones_like(depth),torch.zeros_like(depth))
        
        return silhouette[:,:,:256], silhouette_smpl[:,:,:256]
        
    def gsrendering_body_clothing_separate(self,bidx,cam_poses, focals):
           
        c2w = cam_poses#[0]torch.cat([cam_poses[0],torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(self.device).view(1,4)],dim=0)
        vm = c2w.inverse().float()#[None]
        mat_R, mat_T = vm[...,:3, :3].transpose(-2, -1), vm[...,:3, 3]
        projection_matrix = focals
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, 512, 256)

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
                                         
        points_smpl = self.net.sugar_model_smpl.get_edited_points(self.deformedsmpl[bidx])
        quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_edited_quaternions_and_scales_with_points(points=self.deformedsmpl[bidx], 
                                                                                                                    quat=self.rotations_smpl[bidx], 
                                                                                                                    sca=self.gs_scales_smpl[bidx])
        
        
        gs_output_smpl = {
            "points": points_smpl,
            "quaternions": quaternions_smpl,
            "gs_scales": gs_scales_smpl,
            "gs_opacity": self.gs_opacity_smpl[bidx],
            "gs_shs": self.gs_shs_smpl[bidx]
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
        
        
        rendered_img_smpl = render_smpl_output['image']
        
        # renderimg = rendered_img.detach().cpu().numpy()
        # cv2.imwrite('renderimg{:02d}.png'.format(bidx), renderimg)
    
        #rendered_img_smpl = rendered_img_smpl / 127.5 - 1
        rendered_img_smpl = rendered_img_smpl[None]

        points_cloth = self.net.sugar_model_cloth.get_edited_points(self.deformedcloth[bidx])
        quaternions_cloth, gs_scales_cloth = self.net.sugar_model_cloth.get_edited_quaternions_and_scales_with_points(points=self.deformedcloth[bidx], 
                                                                                                                    quat=self.rotations_cloth[bidx], 
                                                                                                                    sca=self.gs_scales_cloth[bidx])
        
        
        gs_output_cloth = {
            "points": points_cloth,
            "quaternions": quaternions_cloth,
            "gs_scales": gs_scales_cloth,
            "gs_opacity": self.gs_opacity_cloth[bidx],
            "gs_shs": self.gs_shs_cloth[bidx]
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
        
        
        rendered_img_cloth = render_cloth_output['image']
        
        # renderimg = rendered_img.detach().cpu().numpy()
        # cv2.imwrite('renderimg{:02d}.png'.format(bidx), renderimg).repeat(self.deformed_smplcloth.shape[0],1,1)
    
        #rendered_img_cloth = rendered_img_cloth / 127.5 - 1
        rendered_img_cloth = rendered_img_cloth[None]
        
        
        silhouette, silhouette_smpl = self.render_clothmask(self.meshface, self.deformed_smplcloth[bidx:bidx+1], cam_poses, focals)

        #rendered_img = (1-silhouette_smpl.unsqueeze(-1))*rendered_img_cloth + (1-silhouette.unsqueeze(-1))*rendered_img_smpl
        rendered_img = silhouette.unsqueeze(-1)*rendered_img_cloth + (1-silhouette.unsqueeze(-1))*rendered_img_smpl
        
        # mask_render = silhouette[0].detach().cpu().numpy()   
        # cv2.imwrite('silhouette.png', mask_render * 255) 
        
        rendered_img = rendered_img / 127.5 - 1
        
        ret = {
            'image': rendered_img,
            #'cloth_scene': render_cloth_output,
            #'smpl_scene': render_smpl_output,
            #'gs_output_smpl': gs_output_smpl,
            #'gs_output_cloth': gs_output_cloth
        }
        return ret
    
    def gsrendering_body(self,bidx,cam_poses, focals):
           
        c2w = cam_poses#[0]torch.cat([cam_poses[0],torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(self.device).view(1,4)],dim=0)
        vm = c2w.inverse().float()#[None]
        mat_R, mat_T = vm[...,:3, :3].transpose(-2, -1), vm[...,:3, 3]
        projection_matrix = focals
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, 512, self.res)#,256

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
                                         
        points_smpl = self.net.sugar_model_smpl.get_edited_points(self.deformedsmpl[bidx])
        quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_edited_quaternions_and_scales_with_points(points=self.deformedsmpl[bidx], 
                                                                                                                    quat=self.rotations_smpl[bidx], 
                                                                                                                    sca=self.gs_scales_smpl[bidx])
        
        
        gs_output_smpl = {
            "points": points_smpl,
            "quaternions": quaternions_smpl,
            "gs_scales": gs_scales_smpl,
            "gs_opacity": self.gs_opacity_smpl[bidx],
            "gs_shs": self.gs_shs_smpl[bidx]
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
        
        
        rendered_img_smpl = render_smpl_output['image']
        
        # renderimg = rendered_img.detach().cpu().numpy()
        # cv2.imwrite('renderimg{:02d}.png'.format(bidx), renderimg)
    
        rendered_img_smpl = rendered_img_smpl / 127.5 - 1
        rendered_img_smpl = rendered_img_smpl[None]


        ret = {
            'image': rendered_img_smpl,
            #'cloth_scene': render_cloth_output,
            #'smpl_scene': render_smpl_output,
            #'gs_output_smpl': gs_output_smpl,
            #'gs_output_cloth': gs_output_cloth
        }
        return ret
        
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
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, 512, 256)

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
                                         
                     
        #self.net.position_enc_smpl.triplanefeature(styles)
        
        self.net.sugar_model_smpl._points = canoncalmeansmpl#canpersonalsmpl#new mesh point

        points_meansmpl = self.net.sugar_model_smpl.get_edited_points(canoncalmeansmpl)  #self.net.sugar_model_smpl.points canoncalmeansmpl 

        #self.net.sugar_model_smpl._points = canpersonalsmpl#for computing mesh normal        
        # 3dgs渲染_,
        tri_feats_smpl = self.net.position_enc_smpl(points_meansmpl)#gaussian center with mean smpl vertex for fetching feature
        
        style = styles.repeat(tri_feats_smpl.shape[0],1)
        tri_feats_smpl = torch.concatenate([tri_feats_smpl,style],dim=-1)
       
        #tri_feats_smpl = styles.repeat(points_meansmpl.shape[0],1)

        # import open3d as o3d
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(self.net.sugar_model_cloth.points.cpu().detach().numpy())
        # o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
        
        appearance_out_smpl = self.net.appearance_dec_smpl(tri_feats_smpl)
       
        geometry_out_smpl = self.net.geometry_dec_smpl(tri_feats_smpl)
        
        
        # 预测透明度和球谐函数参数(SMPL和CLOTH)
        gs_opacity_smpl = appearance_out_smpl['opacity']
        # gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 16, 3)
        gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 3)

        # 预测旋转和尺度(SMPL和CLOTH)
        rotations_smpl = geometry_out_smpl['rotations']
        gs_scales_smpl = geometry_out_smpl['scales']
        #gs_xyz_smpl = geometry_out_smpl['xyz']
       
        self.opacity_loss = torch.mean(
                    torch.log(0.1 + gs_opacity_smpl) +
                    torch.log(0.1 + gs_opacity_smpl) - -2.20727)
                    
        canpersonalsmpl= canoncalsmpl#+vertoffset+tri_feats_smpl[:,:3]
        
        self.incdisploss = 0#20000.0*torch.nn.functional.smooth_l1_loss(gs_xyz_smpl, torch.zeros_like(gs_xyz_smpl)).mean()
        
        self.deformedpersonsmpl, smpl_A = self.deforming_lbs_newsmpl(beta, theta, trans, canpersonalsmpl)

        #posing gs displacement
        #posedgsdisp, gs_A = self.deforming_lbs_gsdisp(gs_xyz_smpl, smpl_A, trans)
        
        #mesh = struct.Meshes(verts=self.deformedpersonsmpl[None], faces=self.smplfaces) 
        
        # mesh_test = trimesh.Trimesh(vertices=meshvert_def[0].detach().cpu().numpy(), faces=self.meshface[0].detach().cpu().numpy())
        # mesh_test.export("mesh_test.obj")
        
        # fragments = self.net.rasterizer(mesh, cameras=cameras)
        # depth = fragments.zbuf
        # face_idx_map = fragments.pix_to_face[..., 0]
        # self.rendermask = torch.where(depth > 0, torch.ones_like(depth),torch.zeros_like(depth))
        # self.rendermask = self.rendermask.squeeze(-1).float()  
        # mesh_mask = self.rendermask[0].detach().cpu().numpy()
        # cv2.imwrite('mesh_mask.png', mesh_mask*255)
        
                    
        points_smpl = self.net.sugar_model_smpl.get_edited_points(self.deformedpersonsmpl.squeeze(0))
        quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_edited_quaternions_and_scales_with_points(points=self.deformedpersonsmpl, 
                                                                                                                    quat=rotations_smpl, 
                                                                                                                    sca=gs_scales_smpl)
        
        # quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_posed_quaternions_and_scales(points=self.deformedpersonsmpl, 
                                                                                                      # quat=rotations_smpl, 
                                                                                                      # sca=gs_scales_smpl)
        # quaternions_smpl, gs_scales_smpl, gs_quat = self.net.sugar_model_smpl.get_posed_quaternions_and_scales2(canpoints=canpersonalsmpl,
                                                                                                      # gs_A=gs_A, 
                                                                                                      # quat=rotations_smpl, 
                                                                                                      # sca=gs_scales_smpl)
        self.incrotloss = 0#20000.0*torch.nn.functional.smooth_l1_loss(rotations_smpl, torch.zeros_like(rotations_smpl)).mean()
        
        #points_smpl = points_smpl + posedgsdisp#adding gs displacement in the posed space
        
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
        
        if 0:
            rendered_expected_coord: torch.Tensor = render_smpl_output["expected_coord"]
            rendered_median_coord: torch.Tensor = render_smpl_output["median_coord"]
            rendered_normal: torch.Tensor = render_smpl_output["normal"]
            depth_middepth_normal = depth_double_to_normal(rendered_expected_coord, rendered_median_coord)
            depth_ratio = 0.6
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
            self.depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
            
        # self.smpl_img_loss = l1_loss(render_smpl_output['image'].unsqueeze(0), batch['img'] * self.silhouette_smpl.unsqueeze(-1))
        # loss_ssim_smpl = 1.0 - ssim(render_smpl_output['image'].unsqueeze(0), self.silhouette_smpl.unsqueeze(-1))
        # self.smpl_ssim = loss_ssim_smpl * (self.silhouette_smpl.sum() / (render_smpl_output['image'].shape[-1] * render_smpl_output['image'].shape[-2]))
        
        # NOTE: 使用 1-blendmask
        
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
        
    def render_deformation0(self, styles, cam_poses, focals, beta, theta, trans):
        
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
                                         
                      
        self.net.position_enc_smpl.triplanefeature(styles)
        
        #obtaining vert displacement
        vertoffset = self.net.position_enc_smpl.fetch_disp(canoncalmeansmpl)
        
        canpersonalsmpl= canoncalsmpl+vertoffset#+tri_feats_smpl[:,:3]
        
        self.incdisploss = 20000.0*torch.nn.functional.smooth_l1_loss(vertoffset, torch.zeros_like(vertoffset)).mean()
        
        self.deformedpersonsmpl, smpl_A = self.deforming_lbs_newsmpl(beta, theta, trans, canpersonalsmpl)
        
        #posing gs displacement
        #posedgsdisp = self.deforming_lbs_gsdisp(gsoffset, smpl_A, trans)
        
        #mesh = struct.Meshes(verts=self.deformedpersonsmpl[None], faces=self.smplfaces) 

        mesh_test = trimesh.Trimesh(vertices=self.deformedpersonsmpl.detach().cpu().numpy(), faces=self.smplfaces[0].detach().cpu().numpy())
        mesh_test.export("mesh_test.obj")
        
        # fragments = self.net.rasterizer(mesh, cameras=cameras)
        # depth = fragments.zbuf
        # face_idx_map = fragments.pix_to_face[..., 0]
        # self.rendermask = torch.where(depth > 0, torch.ones_like(depth),torch.zeros_like(depth))
        # self.rendermask = self.rendermask.squeeze(-1).float()  
        # mesh_mask = self.rendermask[0].detach().cpu().numpy()
        # cv2.imwrite('mesh_mask.png', mesh_mask*255)
        
        
        self.net.sugar_model_smpl._points = canoncalmeansmpl#canpersonalsmpl#new mesh point

        points_meansmpl = self.net.sugar_model_smpl.get_edited_points(self.net.sugar_model_smpl.points)  #canoncalmeansmpl 

        #self.net.sugar_model_smpl._points = canpersonalsmpl#for computing mesh normal        
        # 3dgs渲染_,
        tri_feats_smpl = self.net.position_enc_smpl(points_meansmpl)#gaussian center with mean smpl vertex for fetching feature
        
        style = styles.repeat(tri_feats_smpl.shape[0],1)
        tri_feats_smpl = torch.concatenate([tri_feats_smpl,style],dim=-1)
       
        # import open3d as o3d
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(self.net.sugar_model_cloth.points.cpu().detach().numpy())
        # o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
        
        # tri_feats_smpl = self.net.position_enc_smpl(self.net.sugar_model_smpl.points, sp_input['latent_index'].to(torch.int64)[0])
        # tri_feats_cloth = self.net.position_enc_cloth(self.net.sugar_model_cloth.points, sp_input['latent_index'].to(torch.int64)[0])
        # tri_feats_cloth = self.net.position_enc_cloth(self.net.sugar_model_cloth.newpoints, sp_input['latent_index'].to(torch.int64)[0])
        # tri_feats_cloth = self.net.position_enc_cloth(subclothpoints, sp_input['latent_index'].to(torch.int64)[0])
     
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
        
        # quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_posed_quaternions_and_scales(points=self.deformedpersonsmpl, 
                                                                                                      # quat=rotations_smpl, 
                                                                                                      # sca=gs_scales_smpl)
        
        #points_smpl = points_smpl + posedgsdisp#adding gs displacement in the posed space
        
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
        
        if 0:
            rendered_expected_coord: torch.Tensor = render_smpl_output["expected_coord"]
            rendered_median_coord: torch.Tensor = render_smpl_output["median_coord"]
            rendered_normal: torch.Tensor = render_smpl_output["normal"]
            depth_middepth_normal = depth_double_to_normal(rendered_expected_coord, rendered_median_coord)
            depth_ratio = 0.6
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
            self.depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
            
        # self.smpl_img_loss = l1_loss(render_smpl_output['image'].unsqueeze(0), batch['img'] * self.silhouette_smpl.unsqueeze(-1))
        # loss_ssim_smpl = 1.0 - ssim(render_smpl_output['image'].unsqueeze(0), self.silhouette_smpl.unsqueeze(-1))
        # self.smpl_ssim = loss_ssim_smpl * (self.silhouette_smpl.sum() / (render_smpl_output['image'].shape[-1] * render_smpl_output['image'].shape[-2]))
        
        # NOTE: 使用 1-blendmask
        
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
        
class ClothSimulation(nn.Module):
    def __init__(self):
        super(ClothSimulation, self).__init__()

        templatecloth_path = os.path.join('./', 'cloth/clothes_vert.txt')
        templatecloth = np.loadtxt(templatecloth_path)

        vtnum = templatecloth.shape[0]
        
        self.model = FullyConnected(
            input_size=72+10+4, output_size=vtnum * 3,
            num_layers=3,
            hidden_size=1024)
            
    def forward(self, thetas, betas, gammas):

        pred_verts = self.model(torch.cat((thetas, betas, gammas), dim=1))
        
        return pred_verts.view(thetas.shape[0], -1, 3)       
        