from typing import Union, List, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from libraries.NARF.base_deepcap import NARFBase
from libraries.NeRF.nerf import calc_density_and_color_from_feature
from libraries.NeRF.net import StyledMLP, MLP
from libraries.NeRF.utils import StyledConv1d, multi_part_positional_encoding, in_cube, to_local
#from libraries.custom_stylegan2.net import EqualConv1d
from libraries.triplane.sampling import sample_feature, sample_triplane_part_prob, sample_weighted_feature_v2
from libraries.triplane.triplane_nerf import prepare_triplane_generator, calc_density_and_color_from_feature

from libraries.superresolution import Superresolution_freesize as Superresolution #SuperresolutionHybrid8X
from libraries.custom_stylegan2.torch_utils.ops import grid_sample_gradfix

# import libraries.modules as modules
# from libraries.meta_modules import HyperNetwork
from math import sqrt, exp

from . import embedder
from warping_utils import surface_field, surface_field_clothing, smpl_helper, clothing_helper
from smplx.utils import SMPLOutput
from libraries.stylegan2_ada_pytorch.training.networks import FullyConnectedLayer
import trimesh
from warping_utils import math_utils
from smplx.body_models import SMPL
from libraries.smpl_utils import get_shape
import os

from smplx.lbs import blend_shapes, vertices2joints

import glob
from tqdm import tqdm
import scipy.io

from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals

from warping_utils.ray_marcher import MipRayMarcher2

from models.transforms_deepcap import RealImgToPatch, FlexGridRaySampler
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# import faulthandler
# faulthandler.enable()

import pytorch3d
import pytorch3d.structures
import pytorch3d.renderer

from pytorch3d.renderer.cameras import PerspectiveCameras, OrthographicCameras
from pytorch3d.renderer import (
    FoVOrthographicCameras, FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,SoftPhongShader
)

from pytorch3d.structures import Pointclouds
import pytorch3d.structures as struct
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer,
    SoftSilhouetteShader,
    TexturesVertex,
    BlendParams,
    PointsRasterizationSettings,
    # PointsRenderer,
    PointsRasterizer,
    PointLights,
    AlphaCompositor
)
import cv2

def cdf_Phi_s(x, s):
    return torch.sigmoid(x * s)

def sdf_to_alpha(sdf: torch.Tensor, s):
    # [(B), N_rays, N_pts]
    cdf = cdf_Phi_s(sdf, s)
    # [(B), N_rays, N_pts-1]
    # TODO: check sanity.
    opacity_alpha = (cdf[..., :-1] - cdf[..., 1:]) / (cdf[..., :-1] + 1e-10)
    
    opacity_alpha = torch.cat(
            [opacity_alpha,
             opacity_alpha[:,:, -1][:,:,None]],
            -1)  # [(B), N_rays, N_pts]
            
    opacity_alpha = torch.clamp_min(opacity_alpha, 0)
    return opacity_alpha

def sdf_to_alpha2(sdf: torch.Tensor, s):
    opacity_alpha = torch.sigmoid(sdf * -1/s)*1/s
    opacity_alpha = torch.clamp_min(opacity_alpha, 0)
    return opacity_alpha

def sdf_to_w(sdf: torch.Tensor, s):
    device = sdf.device
    # [(B), N_rays, N_pts-1]
    opacity_alpha = sdf_to_alpha(sdf, s)

    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
            1.0 - opacity_alpha + 1e-10,
        ],
        dim=-1,
    )

    # [(B), N_rays, N_pts-1]
    visibility_weights = (
        opacity_alpha * torch.cumprod(shifted_transparency, dim=-1)[..., :-1]
    )

    return opacity_alpha, visibility_weights

def alpha_to_w(alpha: torch.Tensor):
    device = alpha.device
    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*alpha.shape[:-1], 1], device=device),
            1.0 - alpha + 1e-10,
        ],
        dim=-1,
    )

    # [(B), N_rays, N_pts-1]
    visibility_weights = alpha * torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return visibility_weights
    
def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0]]], dtype=torch.float32)
    # torch.tensor([[[1, 0, 0],
                                # [0, 1, 0],
                                # [0, 0, 1]],
                                # [[1, 0, 0],
                                # [0, 0, 1],
                                # [0, 1, 0]],
                                # [[0, 0, 1],
                                # [1, 0, 0],
                                # [0, 1, 0]]], dtype=torch.float32)
def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)

    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=3, box_warp_pre_deform=False):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)
    #coordmean = torch.mean(v_posed,1)
    #coordinates = coordinates - coordmean[:,None,:]
    #print(coordinates)
    #print(coordinates.min(1)[0],coordinates.max(1)[0])
    #print(coordinates[torch.abs(coordinates)<=5])    
    #if not box_warp_pre_deform:       
    coordinates = (1/2) * coordinates # TODO: add specific box bounds
    #smvert = (1/2) * smvert
    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    #proj_coordinates = project_onto_planes(plane_axes, smvert)
    #projected_coordinates = projected_coordinates.permute(0,2,1,3)#torch.zeros_like(projected_coordinates)
    #proj_coordinates = projected_coordinates.squeeze(1)
    # proj_coordinates[...,1] = (proj_coordinates[...,1]+1)/2*512
    # proj_coordinates[...,0] = (proj_coordinates[...,0]+1)/2*512
    # proj_coordinates = proj_coordinates.clamp(min=0,max=511)
    # proj_coordinates = proj_coordinates.long()
    # projrgb = projected_coordinates.new_zeros([512,512,3], dtype=torch.float32)
    # for i in range(0,proj_coordinates.shape[1],1):   
        # projrgb[proj_coordinates[0,i,1],proj_coordinates[0,i,0],:] = 255
    # projrgb = projrgb.detach().cpu().numpy()
    # cv2.imwrite('test/proj0smpl.png', projrgb)

    #output_features = plane_features.new_zeros([3,3,coordinates.shape[1],32], dtype=torch.float32)
    #output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    output_features = grid_sample_gradfix.grid_sample(plane_features, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

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
        
class TriPlaneNARF(NARFBase):
    def __init__(self, z_dim: Union[int, List[int]] = 256, num_bone=1,
                 bone_length=True, parent=None, num_bone_param=None, view_dependent: bool = False):
        assert bone_length
        self.tri_plane_based = True
        self.w_dim = 512#512
        self.feat_dim = 16#32
        #self.no_selector = config.no_selector config,
        super(TriPlaneNARF, self).__init__(z_dim, num_bone, bone_length, parent, num_bone_param, view_dependent)
        #self.initialize_network()
        
        # self.incSDFnetwork_smpl = IncSDFNetwork()
        # self.incSDFnetwork_cloth = IncSDFNetwork()
        
        #self.SDFnetwork = SDFNetwork(10)
        self.Densitynetwork_smpl = DensityNetwork()
        self.Densitynetwork_cloth = DensityNetwork()
        
        # pretrained_model = torch.load('./models_0010000.pt')
        # model_dict = self.Densitynetwork.state_dict()        
        # pretrained_dict = {k[20:]: v for k, v in pretrained_model['g'].items() if k[20:] in model_dict}#20
        # # for k, v in pretrained_model['g'].items():
            # # print(k[20:])        
        # model_dict.update(pretrained_dict)
        # self.Densitynetwork.load_state_dict(model_dict)
        # for param in self.Densitynetwork.parameters():
            # param.requires_grad = False
            
        # self.SDFnetwork_cloth = SDFNetwork(14)
        # self.SDFnetwork_smpl = SDFNetwork(10)
        #pretrained_model = torch.load('snapshot_latest260000.pth')
        # model_dict = self.SDFnetwork_cloth.state_dict()        
        # pretrained_dict = {k[22:]: v for k, v in pretrained_model['gen'].items() if k[22:] in model_dict}
        # # for k, v in pretrained_model['gen'].items():
            # # print(v)        
        # model_dict.update(pretrained_dict)
        # self.SDFnetwork_cloth.load_state_dict(model_dict)
        
        # model_dict = self.SDFnetwork.state_dict() #_smpl       
        # pretrained_dict = {k[21:]: v for k, v in pretrained_model['gen'].items() if k[21:] in model_dict}       
        # model_dict.update(pretrained_dict)
        # self.SDFnetwork.load_state_dict(model_dict)#_smpl
        
        # for param in self.SDFnetwork_cloth.parameters():
            # param.requires_grad = False
        # for param in self.SDFnetwork_smpl.parameters():
            # param.requires_grad = False
            
        #self.decoder = ColorNetwork()#OSGDecoder_fixgeometry_color(80)
        #self.decoder_cloth = ColorNetwork()#OSGDecoder_fixgeometry_color(80)
        #self.decoder = SirenGenerator(D=5, W=128, style_dim=128, input_ch=3, input_ch_views=3, output_ch=4,output_features=False)
          
        # self.colorlatent_smpl = nn.Embedding(1, 16)
        # self.colorlatent_cloth = nn.Embedding(1, 16)
        # self.latentws = nn.Embedding(1, 512)
        
        # multires = 6
        # embed_fn, input_ch = embedder.get_embedder(multires, input_dims=3)
        # self.embed_fn_fine = embed_fn
        # # Deform-Net
        # self.sdf_network_cloth=modules.SingleBVPNet(type='relu',mode='mlp', hidden_features=256, num_hidden_layers=3, in_features=72,out_features=1)
        # # Hyper-Net
        # self.hyper_net_cloth = HyperNetwork(hyper_in_features=2, hyper_hidden_layers=3, hyper_hidden_features=128,hypo_module=self.sdf_network_cloth)
        # # Deform-Net
        # self.sdf_network_smpl=modules.SingleBVPNet(type='relu',mode='mlp', hidden_features=256, num_hidden_layers=3, in_features=72,out_features=1)
        # # Hyper-Net
        # self.hyper_net_smpl = HyperNetwork(hyper_in_features=10, hyper_hidden_layers=3, hyper_hidden_features=128,hypo_module=self.sdf_network_smpl)
                              
        #self.decoder = OSGDecoder(16)
        #self.decoder_cloth = OSGDecoder(16)
        # self.plane_axes = generate_planes()
        # self.plane_axes = self.plane_axes.to('cuda')
        # smpl_base = smpl_helper.load_smpl_model(smpl_helper.get_smpl_data_path('m'))
        # self.smpl_reduced = smpl_helper.SMPLSimplified.build_from_template(smpl_base, growth_offset=0.0)
        # self.surface_field = surface_field.SurfaceField(self.smpl_reduced)
        
        # self.smplreducedfaces = torch.LongTensor(self.smpl_reduced.faces).to('cuda')
        
        # self.displace_smpl = DisplaceNetwork()
        # self.displace_cloth = DisplaceNetwork()
        #self.superresolution = Superresolution(16, 512, sr_num_fp16_res=4, sr_antialias=True)   
        
        #self.latent = nn.Embedding(1, 512)
        
        self._register_avg_smpl()
        
        self.ray_marcher = MipRayMarcher2()
        SMPL_MODEL_PATH = "./smpl_data"#"./smpl_models/smpl"#"./smpl_data"NEUTRAL
        self.smpl = SMPL(model_path=SMPL_MODEL_PATH, gender='FEMALE', batch_size=2)
        #self.surface_field = surface_field.SurfaceField(self.smpl)
        
        # smpl_reduced_canon_vert = get_shape(self.smpl, body_pose=self.smpl_avg_body_pose.expand(3, -1).view(3,23,3),
                                      # global_orient=self.smpl_avg_orient.expand(3, -1)[:,None])
        # smpl_reduced_canon_vert *= self.smpl_avg_scale
        # #smpl_reduced_canon_vert += self.smpl_avg_transl.expand(3, -1)[:,None]                                      
        # self.smpl_reduced_canon = SMPLOutput(vertices=smpl_reduced_canon_vert.to('cuda'))
        
        npfaces = self.smpl.faces
        npfaces = npfaces.astype(np.int32)
        self.smplfaces = torch.LongTensor(npfaces).to('cuda')
        
        #self.laplacedensity = LaplaceDensity(params_init={'beta': 0.1}) 
        #sdf to density
        self.ln_s = nn.Parameter(0.1 * torch.ones(1))
        # self.ln_s_cloth = nn.Parameter(0.1 * torch.ones(1))
        # self.ln_s = nn.Parameter(
            # data=torch.Tensor([np.log(0.05) / 1.0]).to('cuda'),
            # requires_grad=True,
        # )
       
        # self.ln_s_cloth = nn.Parameter(
            # data=torch.Tensor([np.log(0.05) / 1.0]).to('cuda'),
            # requires_grad=True,
        # )
        #cloth model
        self.cloth_simulation = ClothSimulation()
        state_dict = torch.load('./cloth/lin.pth.tar')
        self.cloth_simulation.model.load_state_dict(state_dict)
        self.cloth_simulation = self.cloth_simulation.cuda()
        for param in self.cloth_simulation.parameters():
            param.requires_grad = False
         
        templatesmpl_path = os.path.join('./',
                                         'cloth/vpersonalshape.txt')  # smpldeform/vpersonalshape
        templatesmpl = np.loadtxt(templatesmpl_path)
        self.rawtemplatesmpl = torch.Tensor(templatesmpl).to('cuda')
            
        bw = np.load(os.path.join('./', 'cloth/bw.npy'), allow_pickle=True)
        bw = torch.Tensor(bw).to('cuda')
        self.rawsmplbw = bw[None, ...]
        
        self.bw = np.loadtxt(os.path.join('./', 'cloth/skinweightnew.txt'))
        self.bw = torch.Tensor(self.bw)[None, ...].to('cuda')
        
        npfaces = np.loadtxt(os.path.join('./', 'cloth/clothes_face.txt')) - 1
        self.clothfaces = torch.LongTensor(npfaces).to('cuda')
        #self.clothfaces = self.clothfaces[None, :, :]
        
        npfaces = np.loadtxt(os.path.join('./', 'cloth/clothes_watertight_face.txt')) - 1
        self.clothes_watertight_face = torch.LongTensor(npfaces).to('cuda')
        #self.clothes_watertight_face = self.clothes_watertight_face[None, :, :]
        
        # loading template deformation graph
        templateshape_path = os.path.join('./', 'cloth/clothes_vert.txt')
        templatecloth = np.loadtxt(templateshape_path)
        self.templatecloth = torch.Tensor(templatecloth).to('cuda')
        
        #combined mesh 
        self.meshface = torch.cat([self.clothfaces[None, :, :], self.smplfaces[None, :, :]+templatecloth.shape[0]], dim=1)
        
        # self.cloth_reduced = clothing_helper.ClothSimplified.build_from_template(self.templatecloth.detach().cpu(), self.clothfaces.detach().cpu(), growth_offset=0.0)
        # self.cloth_surface_field = surface_field_clothing.SurfaceField(self.cloth_reduced.v_template,self.cloth_reduced.faces)

        weight1 = torch.from_numpy(np.array([-0.8199, -0.0786], dtype = np.float32))
        weight1 = weight1.view(1,2)
        self.rawtempclothpara = nn.Embedding.from_pretrained(weight1, freeze=True)
        tempclothpara = self.rawtempclothpara(torch.zeros(1).to(torch.int64)).to('cuda')
        #print(tempclothpara)
        #clothpara = 2*torch.sigmoid(clothpara)-1 
        tempclothpara = 2*torch.sigmoid(tempclothpara)-1 
        zeropara = torch.zeros_like(tempclothpara).to('cuda')
        #clothpara = torch.cat((clothpara,zeropara),-1)
        self.tempclothpara = torch.cat((tempclothpara,zeropara),-1)
        #print(tempclothpara)
        
        #simulatedcanoncloth = self.net.cloth_simulation(sp_input['smplpose'],sp_input['smplshape'],tempclothpara)
        #self.simulatedcloth =self.net.deformation_network.LBS_simulatedcloth(simulatedcanoncloth, sp_input)
        data_root = '../../neuralbody-deformation-occupancy-fixnerf-our-generalization/tools/data/magdalena20000-allviews/'        
        params_path = os.path.join(data_root, 'smpl_params',
                                   '{}.npy'.format(0))
        paramshape = np.load(params_path, allow_pickle=True).item()
        paramshape = torch.from_numpy(paramshape['shapes']).to('cuda')
        self.canonparamshape = paramshape.view(1,10)
        self.canonclothvert = self.cloth_simulation(torch.zeros([1,72]).to('cuda'),self.canonparamshape,self.tempclothpara)
        #self.update_embeddedgraph(self.canonclothvert[0])
        self.canon_clothvert = self.canonclothvert.clone()
        
        self.shapepara = torch.cat([self.tempclothpara,self.canonparamshape],dim=1)
        
        smploutput = self.smpl.forward(self.canonparamshape.detach().cpu(), self.smpl_avg_body_pose,self.smpl_avg_orient, self.smpl_avg_transl)#        
        self.update_canonsmpl(smploutput.vertices)
    
        self.templateshape_blend_shapes = blend_shapes(self.canonparamshape.view(-1, 10).detach().cpu(), self.smpl.shapedirs).view(-1, 3)
        self.templateshape_blend_shapes = self.templateshape_blend_shapes.to('cuda')

        # self.smpl_reduced_canon_vert = self.smpl_reduced.simplifysmpl(self.rawtemplatesmpl).to('cuda')
        # self.cloth_reduced_canon = self.cloth_reduced(self.canonclothvert.detach().cpu()).to('cuda')
        self.rawtemplatesmpl = self.rawtemplatesmpl.to('cuda')
        #np.savetxt('./test/vert.txt',self.smpl_reduced_canon_vert[0].detach().cpu())
        
        # self.canoncombinedmeshvert = torch.cat([self.canonclothvert, self.rawtemplatesmpl], dim=1)
        # self.combinedmesh_reduced = clothing_helper.ClothSimplified.build_from_template(self.canoncombinedmeshvert[0].detach().cpu(), self.meshface[0].detach().cpu(), growth_offset=0.0)
        # self.combinedmesh_surface_field = surface_field_clothing.SurfaceField(self.combinedmesh_reduced.v_template,self.combinedmesh_reduced.faces)
        # self.combinedmesh_reduced_canon = self.combinedmesh_reduced(self.canoncombinedmeshvert.detach().cpu()).to('cuda')
        
        # self.combinedmesh_reduced_faces = self.combinedmesh_reduced.faces.to('cuda')
        # self.reducedcloth_faces = self.cloth_reduced.faces.to('cuda')
        
        # npvertices = self.canonclothvert[0].detach().cpu().numpy()                        
        # npfaces = self.clothes_watertight_face.detach().cpu().numpy() 
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/cloth_watertight.obj')
        # mesh.export(result_path)
        
        #self.allbeta = self.readSMPLshape()
        #self.allbeta = self.allbeta[0:2,:]
        self.smplshapepararange = torch.from_numpy(np.array([[-2, 2],[-2, 2]], dtype = np.float32))
        self.smplshapepararange = self.smplshapepararange.view(2,2).to('cuda')
        
        self.clothpararange = torch.from_numpy(np.array([[-4, 4],[-2, 2]], dtype = np.float32))
        self.clothpararange = self.clothpararange.view(2,2).to('cuda')
        
        self.ray_sampler = FlexGridRaySampler(N_samples=64*64,
                                     min_scale=0.01,
                                     max_scale=1.0,
                                     scale_anneal=0.0025)        
        self.beta_network = BetaNetwork()
        
        self.deform_crit = torch.nn.L1Loss()
        
        outvfidx_path = os.path.join('./',
                                         'cloth/deformation/smpl/outvfidx.txt')  
        outvfidx = np.loadtxt(outvfidx_path) - 1
        self.outvfidx = torch.LongTensor(outvfidx).to('cuda')
        
        self.istraining = False
        #self.init_clothdeformgraph()
        #self.init_smpldeformgraph()
        
    def sample_canonshape(self, batch_size, smplweight, clothweight):
        
        # shapeidx = torch.randint(self.allbeta.shape[0],(1,batch_size))#batchsize,shapeidx,clothweight
        # sampledshape = self.allbeta[shapeidx.view(-1),:]
        # self.canonparamshape = torch.from_numpy(sampledshape).to('cuda')
        # self.canonparamshape = self.canonparamshape.view(batch_size,-1)
        
        #smplweight = torch.rand(batch_size, 2).to('cuda')
        tempsmplpara = (self.smplshapepararange[:,1][None] - self.smplshapepararange[:,0][None]) * smplweight + self.smplshapepararange[:,0][None]
        zeropara = torch.zeros([batch_size, 8],dtype=torch.float).to('cuda')
        self.canonparamshape = torch.cat((tempsmplpara,zeropara),-1)
        
        #clothweight = torch.rand(batch_size, 2).to('cuda')
        tempclothpara = (self.clothpararange[:,1][None] - self.clothpararange[:,0][None]) * clothweight + self.clothpararange[:,0][None]
        tempclothpara = 2*torch.sigmoid(tempclothpara)-1 
        zeropara = torch.zeros_like(tempclothpara).to('cuda')
        self.tempclothpara = torch.cat((tempclothpara,zeropara),-1)
        
        self.canonclothvert = self.cloth_simulation(torch.zeros([batch_size,72]).to('cuda'),self.canonparamshape,self.tempclothpara)#.repeat(batch_size,1)
        #self.update_embeddedgraph(self.canonclothvert[0]).repeat(batch_size,1)
        self.canon_clothvert = self.canonclothvert.clone()
               
        self.shapepara = torch.cat([self.tempclothpara,self.canonparamshape],dim=1)
        
        smploutput = self.smpl.forward(self.canonparamshape, self.smpl_avg_body_pose.expand(batch_size, -1),self.smpl_avg_orient.expand(batch_size, -1), self.smpl_avg_transl.expand(batch_size, -1))#        
        self.update_canonsmpl(smploutput.vertices)
        #self.rawtemplatesmpl = self.rawtemplatesmpl.to('cuda')
    
    def sample_canonshape_smpl(self, batch_size):
        
        # shapeidx = torch.randint(self.allbeta.shape[0],(1,batch_size))#batchsize,shapeidx,clothweight
        # sampledshape = self.allbeta[shapeidx.view(-1),:]
        # self.canonparamshape = torch.from_numpy(sampledshape).to('cuda')
        # self.canonparamshape = self.canonparamshape.view(batch_size,-1), smplweight, clothweight
        
        smplweight = torch.rand(batch_size, 2).to('cuda')
        tempsmplpara = (self.smplshapepararange[:,1][None] - self.smplshapepararange[:,0][None]) * smplweight + self.smplshapepararange[:,0][None]
        zeropara = torch.zeros([batch_size, 8],dtype=torch.float).to('cuda')
        self.canonparamshape = torch.cat((tempsmplpara,zeropara),-1)
        
        #smploutput = self.smpl.forward(self.canonparamshape, self.smpl_avg_body_pose.expand(batch_size, -1),self.smpl_avg_orient.expand(batch_size, -1), self.smpl_avg_transl.expand(batch_size, -1))#        
        #self.update_canonsmpl(smploutput.vertices)
        #self.rawtemplatesmpl = self.rawtemplatesmpl.to('cuda')
        smploutput = self.smpl.forward(self.canonparamshape, self.smpl_avg_body_pose.expand(batch_size, -1),self.smpl_avg_orient.expand(batch_size, -1), self.smpl_avg_transl.expand(batch_size, -1))#        
        self.update_canonsmpl(smploutput.vertices)
    
    def sdf_to_alpha(self, sdf, beta):
        x = -sdf
        
        # select points whose x is smaller than 0: 1 / beta * 0.5 * exp(x/beta)
        ind0 = x <= 0
        val0 = 1 / beta * (0.5 * torch.exp(x[ind0] / beta))

        # select points whose x is bigger than 0: 1 / beta * (1 - 0.5 * exp(-x/beta))
        ind1 = x > 0
        val1 = 1 / beta * (1 - 0.5 * torch.exp(-x[ind1] / beta))

        val = torch.zeros_like(sdf)
        val[ind0] = val0
        val[ind1] = val1
        return val
        
    def _register_avg_smpl(self):
        # if rendering_kwargs['cfg_name'] == 'aist':
            # avg_body_pose = torch.from_numpy(np.array(consts.AIST_BODYPOSE_AVG)[None, ...]).float().contiguous()
            # avg_orient = torch.from_numpy(np.array(consts.AIST_ORIENT_AVG)[None, ...]).float().contiguous()
            # avg_betas = torch.from_numpy(np.array(consts.AIST_BETAS_AVG)[None, ...]).float().contiguous()
            # avg_transl = torch.from_numpy(np.array(consts.AIST_TRANSL)[None, ...]).float().contiguous()
            # avg_scale = torch.from_numpy(np.array(consts.SURREAL_SCALE)).float().contiguous()
        # elif rendering_kwargs['cfg_name'] == 'surreal_new':
            # avg_body_pose = torch.from_numpy(np.array(consts.SURREAL_BODYPOSE_AVG)[None, ...]).float().contiguous()
            # avg_orient = torch.from_numpy(np.array(consts.SURREAL_ORIENT_AVG)[None, ...]).float().contiguous()
            # avg_betas = torch.from_numpy(np.array(consts.SURREAL_BETAS_AVG)[None, ...]).float().contiguous()
            # avg_transl = torch.from_numpy(np.array(consts.AIST_TRANSL)[None, ...]).float().contiguous()  # None
            # avg_scale = torch.from_numpy(np.array(consts.SURREAL_SCALE)).float().contiguous()  # None
        # elif rendering_kwargs['cfg_name'] == 'surreal':
            # avg_body_pose = torch.from_numpy(np.array(consts.SURREAL_BODYPOSE_AVG)[None, ...]).float().contiguous()
            # avg_orient = torch.from_numpy(np.array(consts.SURREAL_ORIENT_AVG)[None, ...]).float().contiguous()
            # avg_betas = torch.from_numpy(np.array(consts.SURREAL_BETAS_AVG)[None, ...]).float().contiguous()
            # avg_transl = torch.from_numpy(np.array(consts.SURREAL_TRANSL)[None, ...]).float().contiguous()
            # avg_scale = torch.from_numpy(np.array(consts.SURREAL_SCALE)).float().contiguous()
        # elif rendering_kwargs['cfg_name'] == 'aist_rescaled':
            # avg_body_pose = torch.from_numpy(np.array(consts.AIST_BODYPOSE_AVG)[None, ...]).float().contiguous()
            # avg_orient = torch.from_numpy(np.array(consts.AIST_ORIENT_AVG)[None, ...]).float().contiguous()
            # avg_betas = torch.from_numpy(np.array(consts.AIST_BETAS_AVG)[None, ...]).float().contiguous()
            # avg_transl = torch.from_numpy(np.array(consts.AIST_TRANSL)[None, ...]).float().contiguous()
            # avg_scale = torch.from_numpy(np.array(consts.AIST_SCALE)).float().contiguous()
        # elif rendering_kwargs['cfg_name'] == 'shhq' or rendering_kwargs['cfg_name'] == 'deepfashion':
            # avg_body_pose = torch.from_numpy(np.array(consts.SHHQ_BODYPOSE_AVG)[None, ...]).float().contiguous()
            # avg_orient = torch.from_numpy(np.array(consts.SHHQ_ORIENT_AVG)[None, ...]).float().contiguous()
            # avg_betas = torch.from_numpy(np.array(consts.SHHQ_BETAS_AVG)[None, ...]).float().contiguous()
            # avg_transl = torch.from_numpy(np.array(consts.SHHQ_TRANSL)[None, ...]).float().contiguous()
            # avg_scale = torch.from_numpy(np.array(consts.SHHQ_SCALE)).float().contiguous()
        # else:
        print("Using T-pose as canonical pose")
        avg_body_pose = torch.zeros((1, 69)).contiguous()
        avg_orient = torch.zeros((1, 3)).contiguous()
        avg_betas = torch.zeros((1, 10)).contiguous()
        avg_transl = torch.zeros((1, 3)).contiguous()
        avg_scale = torch.ones((1,)).contiguous()

        self.register_buffer('smpl_avg_body_pose', avg_body_pose)
        self.register_buffer('smpl_avg_orient', avg_orient)
        self.register_buffer('smpl_avg_betas', avg_betas)
        self.register_buffer('smpl_avg_transl', avg_transl)
        self.register_buffer('smpl_avg_scale', avg_scale)
        self._avg_pose_initialized = True
    
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
        # # self.deformpara_rotatelinear = nn.Linear(defW, 6)  # 12,self.modelnodenum *
        # # self.deformpara_transllinear = nn.Linear(defW, 3)#self.modelnodenum *
        # self.displacelinear  = nn.Linear(defW, 3)
        # torch.nn.init.constant(self.displacelinear.weight, 0)
        # torch.nn.init.constant(self.displacelinear.bias, 0)
        # self.deformpara_linears = nn.ModuleList([nn.Linear(256, 512), nn.Linear(512, 1024)])
        self.deformpara_finallinear = nn.Linear(defW, self.modelnodenum * 6)
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
        cloth_ptsdist = torch.cdist(graphdeformedverts, smplgraphdeformedverts, p=2)#self.rawtemplatesmpl[None,...]
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
        templatevtnum = smplgraphdeformedverts.size(1)
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(batch_size)]
        idx = torch.cat(idx).to('cuda')
        bwidx = cloth_nnvidx.view(-1) #+ idx
        #deformedpersonsmpl = self.deformedpersonsmpl.view(-1, 3)  #self.deformedcloth.view(-1,3)
        canpersonsmpl = smplgraphdeformedverts.view(-1,3)
        selectcanonsmpl = canpersonsmpl[bwidx.long(), :]
        
        canonsmpl_vertnorm = []
        for i in range(0, batch_size):
            _,trinormal = mesh_face_areas_normals(smplgraphdeformedverts[i].view(-1,3), self.smplfaces)            
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
            _,trinormal = mesh_face_areas_normals(posedsmpl[i].view(-1,3), self.smplfaces)
            posedsmpl_vertnorm.append(trinormal[self.smpl_vfidx,:])
        posedsmpl_vertnorm = torch.cat(posedsmpl_vertnorm)

        pospersonsmpl = posedsmpl.view(-1,3)
        selectpospersonsmpl = pospersonsmpl[bwidx.long(), :]
        selectpospersonsmpl_norm = posedsmpl_vertnorm[bwidx.long(), :]
              
        normaldist1 = (selectpospersonsmpl - posedcloth.view(-1,3))*selectpospersonsmpl_norm
 
        interpenetration1 = torch.sum(normaldist1,1)
        interpenetrationloss1 = torch.mean(F.relu(interpenetration1),0)
        
        return interpenetrationloss+interpenetrationloss1
    
    def readSMPLshape(self):

        betas_cache = []
        video_path = glob.glob(f"{'../../cloth_simulation/SURREAL_v1/cmu/testour'}/*/*/*.mp4")
        with torch.no_grad():
            for path in tqdm(video_path):
                ann_path = path[:-4] + "_info.mat"
                annot = scipy.io.loadmat(ann_path)
                betas = annot["shape"][None, :, 0]
                betas_cache.append(betas.reshape(-1))
            betas_cache = np.array(betas_cache)
        return betas_cache
        
    def initialize_network(self):
        if self.config.constant_triplane:
            self.tri_plane = nn.Parameter(torch.zeros(1, 32 * 3 + self.num_bone * 3, 256, 256))
            self.tri_plane_gen = lambda z, *args, **kwargs: self.tri_plane.expand(z.shape[0], -1, -1, -1)
        elif self.config.constant_trimask:
            self.generator = self.prepare_stylegan2(self.feat_dim * 3)
            lr_mul = self.config.constant_trimask_lr_mul
            self.tri_plane = nn.Parameter(torch.zeros(1, self.num_bone * 3, 256, 256) / lr_mul)
            self.tri_plane_gen = lambda z, *args, **kwargs: torch.cat(
                [self.generator(z, *args, **kwargs),
                 self.tri_plane.expand(z.shape[0], -1, -1, -1) * lr_mul], dim=1)
        elif self.config.deformation_field:
            self.tri_plane = nn.Parameter(torch.zeros(1, 32 * 3 + self.num_bone * 3, 256, 256))
            self.flow_generator = self.prepare_stylegan2(2 * 3)

            def warp(z, *args, **kwargs):
                bs = z.shape[0]
                tri_plane_size = 256
                flow = self.flow_generator(z, *args, **kwargs)
                flow = flow.reshape(bs * 3, 2, tri_plane_size, tri_plane_size).permute(0, 2, 3, 1)
                arange = torch.arange(tri_plane_size, device=z.device)
                grid = torch.stack(torch.meshgrid(arange, arange)[::-1], dim=2) + 0.5
                grid = (grid + flow) / 128 - 1  # warped grid in [-1, 1], (3B, 256, 256, 2)
                tri_plane = self.tri_plane.expand(z.shape[0], -1, -1, -1)
                warped_feature = F.grid_sample(tri_plane[:, :32 * 3].reshape(bs * 3, 32, tri_plane_size,
                                                                             tri_plane_size), grid)
                warped_feature = warped_feature.reshape(bs, 32 * 3, tri_plane_size, tri_plane_size)
                tri_plane = torch.cat([warped_feature, tri_plane[:, 32 * 3:]], dim=1)
                return tri_plane

            self.tri_plane_gen = warp
        elif self.config.selector_mlp:
            self.generator = self.prepare_stylegan2(self.feat_dim * 3)
            self.tri_plane_gen = lambda z, *args, **kwargs: torch.cat(
                [self.generator(z, *args, **kwargs),
                 z.new_zeros(z.shape[0], self.num_bone * 3, 256, 256)], dim=1)

            self.selector = nn.Sequential(EqualConv1d(3 * self.num_bone * self.num_frequency_for_position * 2,
                                                      10 * self.num_bone, 1, groups=self.num_bone),
                                          nn.ReLU(inplace=True),
                                          EqualConv1d(10 * self.num_bone, self.num_bone, 1,
                                                      groups=self.num_bone))
        else:
            #self.tri_plane_gen = self.prepare_stylegan2((self.feat_dim + self.num_bone) * 3)
            self.tri_plane_gen = self.prepare_stylegan2((self.feat_dim) * 3)#without deformation

        # if self.view_dependent:
            # self.density_fc = StyledConv1d(32, 1, self.z2_dim)
            # self.mlp = StyledMLP(32 + 3 * self.num_frequency_for_other * 2, 64, 3, style_dim=self.z2_dim)
        # else:
            # self.mlp = StyledMLP(32, 64, 4, style_dim=self.z2_dim)

    def prepare_stylegan2(self, in_channels):
        # return prepare_triplane_generator(
            # self.z_dim, self.w_dim, in_channels,
            # self.num_frequency_for_other * 2 * self.num_bone)
        
        return prepare_triplane_generator(
            self.z_dim, self.w_dim, in_channels,0)# without pose condition    

    def register_canonical_pose(self, pose: np.ndarray) -> None:
        """ register canonical pose.

        Args:
            pose: array of (24, 4, 4)

        Returns:

        """
        assert self.origin_location in ["center", "center_fixed", "center+head"]
        coordinate = pose[:, :3, 3]
        length = np.linalg.norm(coordinate[1:] - coordinate[self.parent_id[1:]], axis=1)  # (23, )

        canonical_joints = pose[1:, :3, 3]  # (n_bone, 3)
        canonical_parent_joints = pose[self.parent_id[1:], :3, 3]  # (n_bone, 3)
        self.register_buffer('canonical_joints', torch.tensor(canonical_joints, dtype=torch.float32))
        self.register_buffer('canonical_parent_joints', torch.tensor(canonical_parent_joints, dtype=torch.float32))

        if self.origin_location == "center":
            # move origins to parts' center (self.origin_location == "center)
            pose = np.concatenate([pose[1:, :, :3],
                                   (pose[1:, :, 3:] +
                                    pose[self.parent_id[1:], :, 3:]) / 2], axis=-1)  # (23, 4, 4)
        elif self.origin_location == "center_fixed":
            pose = np.concatenate([pose[self.parent_id[1:], :, :3],
                                   (pose[1:, :, 3:] +
                                    pose[self.parent_id[1:], :, 3:]) / 2], axis=-1)  # (23, 4, 4)
        elif self.origin_location == "center+head":
            length = np.concatenate([length, np.ones(1, )])  # (24,)
            head_id = 15
            _pose = np.concatenate([pose[self.parent_id[1:], :, :3],
                                    (pose[1:, :, 3:] +
                                     pose[self.parent_id[1:], :, 3:]) / 2], axis=-1)  # (23, 4, 4)
            pose = np.concatenate([_pose, pose[head_id][None]])  # (24, 4, 4)

        self.register_buffer('canonical_bone_length', torch.tensor(length, dtype=torch.float32))
        self.register_buffer('canonical_pose', torch.tensor(pose, dtype=torch.float32))

    def calc_weight(self, tri_plane_weights: torch.Tensor, position: torch.Tensor, position_validity: torch.Tensor,
                    mode="prod"):
        """
        return part prob. MLP/triplane/constant
        :param tri_plane_weights:
        :param position:
        :param position_validity:
        :param mode:
        :return:
        """
        bs, n_bone, _, n = position.shape
        if self.no_selector:
            weight = torch.ones(bs, n_bone, n, device=position.device) / n_bone

        elif hasattr(self, "selector"):  # use selector
            position = position.reshape(bs, n_bone * 3, n)
            encoded_p = multi_part_positional_encoding(position, self.num_frequency_for_position, self.num_bone)
            h = self.selector(encoded_p)
            weight = torch.softmax(h, dim=1)  # (B, n_bone, n)
        else:  # tri-plane based
            weight = sample_triplane_part_prob(tri_plane_weights, position, position_validity, mode=mode,
                                               clamp_mask=self.config.clamp_mask)

        return weight

    def to_local_and_canonical(self, points, pose_to_camera, bone_length):
        """transform points to local and canonical coordinate

        Args:
            points:
            pose_to_camera:
            bone_length: (B, n_bone, 1)

        Returns:

        """     
        # to local coordinate
        R = pose_to_camera[:, :, :3, :3]  # (B, n_bone, 3, 3)
        inv_R = R.permute(0, 1, 3, 2)
        t = pose_to_camera[:, :, :3, 3:]  # (B, n_bone, 3, 1)
        local_points = torch.matmul(inv_R, points[:, None] - t)  # (B, n_bone, 3, n)

        # to canonical coordinate
        canonical_scale = (self.canonical_bone_length[:, None] / bone_length / self.coordinate_scale)[:, :, :, None]
        canonical_points = local_points * canonical_scale

        canonical_R = self.canonical_pose[:, :3, :3].unsqueeze(0).repeat(local_points.shape[0],1,1,1)  # (n_bone, 3, 3)
        canonical_t = self.canonical_pose[:, :3, 3:].unsqueeze(0).repeat(local_points.shape[0],1,1,1)  # (n_bone, 3, 1)

        canonical_points = torch.matmul(canonical_R, canonical_points) + canonical_t

        # reshape local
        bs, n_bone, _, n = local_points.shape
        local_points = local_points.reshape(bs, n_bone * 3, n)
        return local_points, canonical_points

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse
    
    def rasterize_eg3d(self, faces: torch.Tensor, verts: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
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
                    blur_radius=0.0001,#0.0, 
                    faces_per_pixel=16, 
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
        raster = rasterizer(meshes)
        
        depth = raster.zbuf[...,0]          
        rendermask = torch.where(depth > 0, torch.ones_like(depth),torch.zeros_like(depth))     
        # mask_render = rendermask[0].detach().cpu().numpy()
        # mask_path = 'test/smplmask{:04d}.png'.format(frame_index)       
        # cv2.imwrite(mask_path, mask_render * 255) 
        return rendermask#rendermask[:,:240,:]        
    
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
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, 128, 128)

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
                                         
        rasterizer=pytorch3d.renderer.MeshRasterizer(
                cameras=cameras, 
                raster_settings=pytorch3d.renderer.RasterizationSettings(
                    image_size=(128, 128),
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
        
        silhouette0 = face_idx_map>=self.clothfaces.shape[0]# body mask

        silhouette = ~silhouette0
       
        #silhouette = face_idx_map<self.clothfaces.shape[0]# cloth mask
       
        silhouette = silhouette.squeeze(-1).float()
        
        meshsilhouette = torch.where(depth > 0, torch.ones_like(depth),torch.zeros_like(depth))
        
        return silhouette, meshsilhouette#rendermask[:,:240,:]
        
    #@torch.no_grad()
    def inversedeforming_samplepoints_LBS(self, wpts, nnvidx, templatevert, A, Th, beta, theta):
        #samplepts: sampled points in the world space
        #wpts: sampled points after invtransorm_surreal which applied inv LBS
              
        #inversely deforming sample points to cannonical frame
        ptsnum = wpts.size(1)
        batchsize = wpts.size(0)
        
        templatevtnum = templatevert.size(1)

        #world points to posed points
        #pts = wpts
        #pts = torch.matmul(wpts - sp_input['Th'], sp_input['R'])
        pts = wpts - Th.unsqueeze(1)
        #transform points from the pose space to the T pose

        #bw = self.bw[nnvidx.view([-1]),:].view(-1,ptsum,24) #bw: B, 24, n_points
        #bw = self.bw[None,...]#B, n_points, 24.permute(0, 2, 1)
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(wpts.size(0))]
        idx = torch.cat(idx).to(wpts.device)
        tidx = nnvidx.view(-1) + idx
        
        #T1 = T.view(-1,4,4)
        A = torch.bmm(self.rawsmplbw.repeat(batchsize,1,1), A.view(batchsize, 24, -1))
        A = A.view(batchsize, -1, 4, 4)
        T1 = A.view(-1,4,4)
        
        T1[..., :3, :3] = torch.inverse(T1[..., :3, :3])

        selectT = T1[tidx.long(), ...]
        selectT = selectT.view(-1, ptsnum, 4,4)

        # sh = pts.shape
        # #A = torch.bmm(selectbw, sp_input['A'].view(sh[0], 24, -1))
        # #selectT = selectT.view(sh[0], -1, 4, 4)#A: n_batch, 24, 4, 4
        pts = pts - selectT[..., :3, 3]
        R_inv = selectT[..., :3, :3]#torch.inverse(selectT[..., :3, :3])

        pts = torch.sum(R_inv * pts[:, :, None], dim=3)
        
        shape_blend_shapes = blend_shapes(beta.view(-1, 10), self.smpl.shapedirs).view(batchsize,-1, 3)

        thetarotation=self.batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        ident = torch.eye(3).cuda()
        pose_feature = (thetarotation[:, 1:].view(batchsize, -1, 3, 3) - ident).view(batchsize, -1)
        pose_blend_shapes = torch.matmul(pose_feature, self.smpl.posedirs).view(batchsize,-1, 3)
        all_blend_shapes = pose_blend_shapes + shape_blend_shapes
        
        pts = pts - all_blend_shapes.view(-1,3)[tidx.long(), ...].view(batchsize,-1, 3)+self.templateshape_blend_shapes.view(-1,3).repeat(batchsize,1)[tidx.long(), ...].view(batchsize,-1, 3)
        
        return pts
    
    def inversedeforming_samplepoints_cloth(self, wpts, nnvidx, templatevert, A, Th):
        #samplepts: sampled points in the world space
        #wpts: sampled points after invtransorm_surreal which applied inv LBS
              
        #inversely deforming sample points to cannonical frame
        ptsnum = wpts.size(1)
        batchsize = wpts.size(0)
        
        templatevtnum = templatevert.size(1)

        #world points to posed points
        #pts = wpts
        #pts = torch.matmul(wpts - sp_input['Th'], sp_input['R'])
        pts = wpts - Th.unsqueeze(1)
        #transform points from the pose space to the T pose

        #bw = self.bw[nnvidx.view([-1]),:].view(-1,ptsum,24) #bw: B, 24, n_points
        #bw = self.bw[None,...]#B, n_points, 24.permute(0, 2, 1)
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(wpts.size(0))]
        idx = torch.cat(idx).to(wpts.device)
        tidx = nnvidx.view(-1) + idx
        
        #T1 = T.view(-1,4,4)
        A = torch.bmm(self.bw.repeat(batchsize,1,1), A.view(batchsize, 24, -1))
        A = A.view(batchsize, -1, 4, 4)
        T1 = A.view(-1,4,4)
        
        T1[..., :3, :3] = torch.inverse(T1[..., :3, :3])

        selectT = T1[tidx.long(), ...]
        selectT = selectT.view(-1, ptsnum, 4,4)

        # sh = pts.shape
        # #A = torch.bmm(selectbw, sp_input['A'].view(sh[0], 24, -1))
        # #selectT = selectT.view(sh[0], -1, 4, 4)#A: n_batch, 24, 4, 4
        pts = pts - selectT[..., :3, 3]
        R_inv = selectT[..., :3, :3]#torch.inverse(selectT[..., :3, :3])

        pts = torch.sum(R_inv * pts[:, :, None], dim=3)
        
        return pts
    
    def inversedeforming_samplepoints_graphdeform(self, pts, nnvidx):
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
        idx = torch.cat(idx).to('cuda')
        ptsnodeidx = ptsnode + idx# batch idx of the influence nodes, used for retrieving deformation (affine and transl) of each batch sample
        ptsnodeidx = ptsnodeidx.long()
        deformtransl = self.deformation_transl.view(-1, 3)# (B*modelnodenum)*3
        selectdeformtransl = deformtransl[ptsnodeidx, :]
        selectdeformtransl = selectdeformtransl.view(-1, ptsnum,self.modelvert_nodenum, 3)  # B*(vtnum*nodenum)*3

        repwpts = repwpts.view([-1, ptsnum,self.modelvert_nodenum, 3])
        nodepos = self.modelnodepos.view(-1, 3)[ptsnodeidx, :].view([-1, ptsnum, self.modelvert_nodenum, 3])
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
    
    def inversedeforming_samplepoints_graphdeform2(self, pts, nnvidx, modelnodepos, deformation_affine, deformation_transl):
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
        idx = torch.cat(idx).to('cuda')
        ptsnodeidx = ptsnode + idx# batch idx of the influence nodes, used for retrieving deformation (affine and transl) of each batch sample
        ptsnodeidx = ptsnodeidx.long()
        deformtransl = deformation_transl.view(-1, 3)# (B*modelnodenum)*3
        selectdeformtransl = deformtransl[ptsnodeidx, :]
        selectdeformtransl = selectdeformtransl.view(-1, ptsnum,self.modelvert_nodenum, 3)  # B*(vtnum*nodenum)*3

        repwpts = repwpts.view([-1, ptsnum,self.modelvert_nodenum, 3])
        nodepos = modelnodepos.view(-1, 3)[ptsnodeidx, :].view([-1, ptsnum, self.modelvert_nodenum, 3])
        relativepos = repwpts - nodepos - selectdeformtransl  # B*vtnum*nodenum*3

        deformaffine = deformation_affine.view(-1, 3, 3)
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
        
    def inversedeforming_samplepoints_graphdeform_smpl(self, pts, nnvidx):
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
        idx = torch.cat(idx).to('cuda')
        ptsnodeidx = ptsnode + idx# batch idx of the influence nodes, used for retrieving deformation (affine and transl) of each batch sample
        ptsnodeidx = ptsnodeidx.long()
        deformtransl = self.deformation_transl_smpl.view(-1, 3)# (B*modelnodenum)*3
        selectdeformtransl = deformtransl[ptsnodeidx, :]
        selectdeformtransl = selectdeformtransl.view(-1, ptsnum,self.smplvert_nodenum, 3)  # B*(vtnum*nodenum)*3

        repwpts = repwpts.view([-1, ptsnum,self.smplvert_nodenum, 3])
        nodepos = self.smplnodepos.view(-1, 3)[ptsnodeidx, :].view([-1, ptsnum, self.smplvert_nodenum, 3])
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
        c = a.inverse()
        inversedeformaffine = c.view(-1, ptsnum, 3, 3)# B*vtnum*3*3

        #inversedeformaffine = weighteddeformaffine.transpose(3,2)

        deformpts = torch.matmul(inversedeformaffine, weightedrelativepts[..., None])  # B*vtnum*3*1
        deformpts = deformpts.squeeze(-1)# B*vtnum*3

        return deformpts
           
    def invtransform_surreal(self, wpts, shift, trans):
    
        # invs axis_transform
        wpts = wpts[:, :, [2, 0, 1]] * torch.tensor([-1, -1, -1]).view(1,3)[:, None].to(wpts.device)
        wpts -= shift[:,None]
        wpts = torch.matmul(torch.inverse(trans[:,:3,:3]), wpts.permute(0,2,1))      
        wpts = wpts.permute(0,2,1)
        
        return wpts
        # smpl_reduced_current.vertices = torch.matmul(trans[:,:3,:3], smpl_reduced_current.vertices.permute(0,2,1))
        # smpl_reduced_current.vertices = smpl_reduced_current.vertices.permute(0,2,1)
        # smpl_reduced_current.vertices += shift[:,None]
        # # axis_transform
        # smpl_reduced_current.vertices = smpl_reduced_current.vertices[:, :, [1, 2, 0]] * torch.tensor([-1, -1, -1]).view(1,3)[:, None].to(betas.device)
    
    def tensor_expand(self, bin_img, ksize=7):
        # 图像膨胀
        # bin_img = bin_img[0]
        B, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = torch.nn.functional.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

        # 将原图 unfold 成 patch
        patches = bin_img.unfold(dimension=1, size=ksize, step=1)
        patches = patches.unfold(dimension=2, size=ksize, step=1)
        # B x C x H x W x k x k

        # 取每个 patch 中最小的值，i.e., 0
        res, _ = patches.reshape(B, H, W, -1).max(dim=-1)
        return res
        
    def deformingsmpl_LBS(self, A, Th):

        #deforming with LBS of SMPL further
        sh = A.shape
        
        #embedded deformation on template shape in T pose
        personsmpl = self.rawtemplatesmpl.repeat(sh[0],1,1)  
        
        #bw = self.bw[None,...]
        A = torch.bmm(self.rawsmplbw.repeat(sh[0],1,1), A.view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]#including global rotation
        pts = torch.sum(R * personsmpl[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        #rawdeformedsmpl = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']

        rawdeformedsmpl = pts + Th.unsqueeze(1)

        return rawdeformedsmpl
        
    def deformingcloth_LBS(self, clothvert, A, Th):

        sh = clothvert.shape
        A = torch.bmm(self.bw.repeat(sh[0],1,1), A.view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]#including global rotation
        pts = torch.sum(R * clothvert[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        #deformedclothvert = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']
        deformedclothvert = pts + Th.unsqueeze(1)

        return deformedclothvert
        
    def nerf_rendering(self, frameidx, rawimg, Nc, Nf, z_rend, planes, smpl_params, smpl_translate, smplscale, pose_to_world, extrinsics, intrinsic, bone_mask, ray_origins, ray_directions, near, far, mask_at_box):

        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]

        #smpl_betas = smpl_params[:, 72:82]
        smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        
         # Create stratified depth samples
        ray_start = near
        ray_end = far
        depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
        bs_expand = batch_size 
         
        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       
        
        # smpl_reduced_current = self.smpl_reduced(betas=smpl_betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=smpl_translate)
        # smpl_reduced_canon = self.smpl_reduced(betas=self.smpl_avg_betas.expand(bs_expand, -1),
                                               # body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1),
                                               # global_orient=self.smpl_avg_orient.expand(bs_expand, -1),
                                               # transl=self.smpl_avg_transl.expand(bs_expand, -1))

        # smpl_reduced_current.transl = smpl_translate
        # smpl_reduced_canon.transl = self.smpl_avg_transl.expand(bs_expand, -1)
        # smpl_reduced_current.vertices *= smplscale#self.smpl_avg_scale
        # smpl_reduced_canon.vertices *= self.smpl_avg_scale
        # print(smplscale)
        
        #reduced smpl
        smpl_reduced_current = self.smpl_reduced(betas=self.smpl.betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=self.smpl_avg_transl.expand(bs_expand, -1))
        smpl_reduced_current.vertices *= smplscale[:,None][...,None].repeat(1,690,3)
        smpl_reduced_current.vertices += smpl_translate[:,None] / 100
        smpl_reduced_canon = self.smpl_reduced(betas=self.smpl.betas, body_pose=self.smpl_avg_body_pose.expand(batch_size, -1),
                                      global_orient=self.smpl_avg_orient.expand(batch_size, -1), transl=self.smpl_avg_transl.expand(bs_expand, -1))
        smpl_reduced_canon.vertices *= smplscale[:,None][...,None].repeat(1,690,3)
                
        #raw smpl
        # smpl_reduced_current_vert = get_shape(self.smpl, body_pose=smpl_body_pose.view(batch_size,23,3),
                                      # global_orient=smpl_orient[:,None])  
        # smpl_reduced_current_vert *= smplscale
        # smpl_reduced_current_vert += smpl_translate[:,None] / 100
        # smpl_reduced_current = SMPLOutput(vertices=smpl_reduced_current_vert)
        
        # smpl_reduced_canon_vert = get_shape(self.smpl, body_pose=self.smpl_avg_body_pose.expand(batch_size, -1).view(batch_size,23,3),
                                      # global_orient=self.smpl_avg_orient.expand(batch_size, -1)[:,None])
        # smpl_reduced_canon_vert *= smplscale
        # #smpl_reduced_canon_vert += self.smpl_avg_transl.expand(3, -1)[:,None]                                      
        # smpl_reduced_canon = SMPLOutput(vertices=smpl_reduced_canon_vert)
        
        #rendermask = self.rasterize_eg3d(self.smpl_reduced.faces_t, smpl_reduced_current.vertices, extrinsics, intrinsic)
        # for i in range(0,batch_size):
            # mask_render = rendermask[i].detach().cpu().numpy()
            # mask_path = 'test/smplmask{:04d}.png'.format(frameidx[i].item())       
            # cv2.imwrite(mask_path, mask_render * 255)      
            # raw_img = rawimg.permute(0,2,3,1)[i].detach().cpu().numpy()
            # im_path = 'test/rawimg{:04d}.png'.format(frameidx[i].item())       
            # cv2.imwrite(im_path, raw_img)
        #bonemask = bone_mask[0].detach().cpu().numpy()
        #cv2.imwrite('test/bonemask.png', bonemask * 255)
        
        # smpl_reduced_canon_vert = get_shape(self.smpl, body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1).view(batch_size,23,3),
                                      # global_orient=self.smpl_avg_orient.expand(bs_expand, -1)[:,None])
        # smpl_reduced_canon_vert *= self.smpl_avg_scale
        # smpl_reduced_canon_vert += self.smpl_avg_transl.expand(bs_expand, -1)[:,None]                                      
        # smpl_reduced_canon = SMPLOutput(vertices=smpl_reduced_canon_vert)

        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates.obj')   
        
        # npvertices = smpl_reduced_canon.vertices[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/canonmesh.obj')
        
        # npvertices = smpl_reduced_current.vertices[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/currmesh.obj')
        
        mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        
        sample_coordinates = self.get_canonical_coordinates(
            sample_coordinates,
            smpl_src=smpl_reduced_current,
            smpl_dst=smpl_reduced_canon,
            mask=mask
        )
        

        # t = mask_at_box.reshape(-1,128,128)
        # bone_mask = bone_mask*t
        # bonemask = bone_mask[0].detach().cpu().numpy()
        # cv2.imwrite('test/bonemask.png', bonemask * 255)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates_can.obj') 
        
        # R = extrinsics[:,:3,:3] 
        # T = extrinsics[:,:3,3:] 
        # localvert = torch.matmul(R,smpl_reduced_current.vertices.permute(0,2,1))+T
        # projpixel = torch.matmul(intrinsic, localvert)
        # projpixel = projpixel/localvert[:,2,:]
        # print('smpl:',projpixel)
        # globaljoint = pose_to_world[:,:, :3, 3]
        # localvert = torch.matmul(R,globaljoint.permute(0,2,1))+T
        # projpixel = torch.matmul(intrinsic, localvert)
        # projpixel = projpixel/localvert[:,2,:]
        # print('joint:',projpixel)
        
        #mask = mask_at_box.unsqueeze(-1).unsqueeze(-1).repeat(1,1,Nc,1).view(sample_coordinates.shape[0],sample_coordinates.shape[1],1)
        
        rgb_vals = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],32], dtype=torch.float32)#normlized to [-1,1]
        sigma_vals = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32) * 1e3
        #print(sample_coordinates.shape,mask.shape,sample_coordinates[mask].shape)
        #color, density = self.fetchingnerf(planes, sample_coordinates[0][mask[0]][None,...])
 
        color, density = self.fetchingnerf(planes, sample_coordinates)
        # sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros')
        # sampled_features = sampled_features.mean(1)
        # density, color = calc_density_and_color_from_feature(self, sampled_features.permute(0,2,1), z_rend, sample_coordinates)
        
        #density = self.density_activation(density) * 10
        # density = density.permute(0,2,1)
        # color = color.permute(0,2,1)

        density *= mask.unsqueeze(-1)
        
        rgb_vals = color#[mask]
        sigma_vals = density#[mask]
        
        colors_coarse = rgb_vals
        densities_coarse = sigma_vals
        
        # colors_coarse = out['rgb']
        # densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        #print(colors_coarse[densities_coarse.repeat(1,1,1,3)>0])
        # Mask out invalid samples (optional).
        is_sample_valid = None
        # if smpl_clip_depths is not None:
            # is_sample_valid = self.get_sample_mask(sample_depths=depths_coarse, min_max_depths=smpl_clip_depths)
            # densities_coarse = densities_coarse - 1000 * (1-is_sample_valid.float())

        # Fine Pass
        N_importance = 0#64#rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, batch_size)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            if 0:#rendering_options['box_warp_pre_deform']:
                sample_coordinates = (2 / rendering_options['box_warp']) * sample_coordinates
            sample_coordinates = self.get_canonical_coordinates(
                                sample_coordinates,
                                smpl_src=smpl_reduced_current,
                                smpl_dst=smpl_reduced_canon
                                )
            colors_fine, densities_fine = self.fetchingnerf(planes, sample_coordinates)
            # out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            # colors_fine = out['rgb']
            # densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            # Mask out invalid samples (optional).
            if 0:#smpl_clip_depths is not None:
                is_sample_valid = self.get_sample_mask(sample_depths=depths_fine, min_max_depths=smpl_clip_depths)
                densities_fine = densities_fine - 1000 * (1-is_sample_valid.float())
                #colors_fine = colors_fine * is_sample_valid.float()

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(all_colors, all_densities, all_depths, batch_size)
        else:
            rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, batch_size)
        #print(rgb_final.shape,is_valid.shape)rgb_final, depth_final, weights
        # rgb = rgb_final.reshape(2,64,64,32)
        # print(rgb.shape)
        #print(rgb_final[depth_final.squeeze(-1)>0])
        if is_sample_valid is not None: depth_final = is_sample_valid.any(-2).float()
        
        return rendered_color, rendered_mask, rendered_disparity#rgb_final, depth_final, weights.sum(2)

    def nerf_rendering_surreal(self, frameidx, rawimg, Nc, Nf, z_rend, planes, latentws, smpl_params, betas, trans, shift, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):

        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]

        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        
         # Create stratified depth samples
        ray_start = near
        ray_end = far
        depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
        bs_expand = batch_size 
         
        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       
        
        # smpl_reduced_current = self.smpl_reduced(betas=smpl_betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=smpl_translate)
        # smpl_reduced_canon = self.smpl_reduced(betas=self.smpl_avg_betas.expand(bs_expand, -1),
                                               # body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1),
                                               # global_orient=self.smpl_avg_orient.expand(bs_expand, -1),
                                               # transl=self.smpl_avg_transl.expand(bs_expand, -1))

        # smpl_reduced_current.transl = smpl_translate
        # smpl_reduced_canon.transl = self.smpl_avg_transl.expand(bs_expand, -1)
        # smpl_reduced_current.vertices *= smplscale#self.smpl_avg_scale
        # smpl_reduced_canon.vertices *= self.smpl_avg_scale
        # print(smplscale)
        #with torch.no_grad():
        #reduced smpl        
        smpl_reduced_current = self.smpl_reduced(betas=betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=self.smpl_avg_transl.expand(bs_expand, -1))
      
        smpl_reduced_canon = self.smpl_reduced(betas=betas, body_pose=self.smpl_avg_body_pose.expand(batch_size, -1),
                                      global_orient=self.smpl_avg_orient.expand(batch_size, -1), transl=self.smpl_avg_transl.expand(bs_expand, -1))
        
        smpl_reduced_current.vertices = torch.matmul(trans[:,:3,:3], smpl_reduced_current.vertices.permute(0,2,1))
        smpl_reduced_current.vertices = smpl_reduced_current.vertices.permute(0,2,1)
        smpl_reduced_current.vertices += shift[:,None]
        # axis_transform
        smpl_reduced_current.vertices = smpl_reduced_current.vertices[:, :, [1, 2, 0]] * torch.tensor([-1, -1, -1]).view(1,3)[:, None].to(betas.device)
        
        smpl_reduced_canon.vertices = smpl_reduced_canon.vertices[:, :, [1, 2, 0]] * torch.tensor([-1, -1, -1]).view(1,3)[:, None].to(betas.device)

        #smpl model, vert transformation
        # smplvert, A, T = get_shape(self.smpl, body_pose=smpl_body_pose.view(batch_size,23,3),
                                       # global_orient=smpl_orient[:,None]) 
        # smplvert = torch.matmul(trans[:,:3,:3], smplvert.permute(0,2,1))
        # smplvert = smplvert.permute(0,2,1)
        # smplvert += shift[:,None]
        # # axis_transform
        # smplvert = smplvert[:, :, [1, 2, 0]] * torch.tensor([-1, -1, -1]).view(1,3)[:, None].to(betas.device)
        # mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])

        # nnvidx = mask.new_zeros([mask.shape[0],mask.shape[1]], dtype=torch.long)              
        # for i in range(0, smplvert.shape[0], 1):
            # # tnum = int(sample_coordinates[i][mask[i]].shape[0]/2)
            # # pts = sample_coordinates[i][mask[i]][:tnum]                       
            # # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][:tnum], smplvert[i], p=2)#deformedpersonsmpl
            # # smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            # # vidx = torch.squeeze(smpl_ptsdistmin[1], -1)  # B*P
            # # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][tnum:], smplvert[i], p=2)#deformedpersonsmpl
            # # smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            # # vidx1 = torch.squeeze(smpl_ptsdistmin[1], -1)
            # # vidx = torch.cat((vidx,vidx1),0)
            # # nnvidx[i][mask[i]] = vidx

            # vidx = []
            # tnum = int(sample_coordinates[i][mask[i]].shape[0]/4)          
            # for j in range(4):             
                # if j==3:
                    # #smpl_ptsdist = ((sample_coordinates[i][mask[i]][j*tnum:][:,None,:] - smplvert[i][None,:,:])**2).sum(-1) # Pts x V       
                    # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][j*tnum:], smplvert[i], p=2)
                # else:
                    # #smpl_ptsdist = ((sample_coordinates[i][mask[i]][j*tnum:(j+1)*tnum][:,None,:] - smplvert[i][None,:,:])**2).sum(-1) # Pts x V       
                    # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][j*tnum:(j+1)*tnum], smplvert[i], p=2)#deformedpersonsmpl
                # smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
                # tidx = torch.squeeze(smpl_ptsdistmin[1], -1)  # B*P
                # vidx += [tidx]
            # vidx = torch.cat(vidx, 0)
            # nnvidx[i][mask[i]] = vidx
            
        # sample_coordinates = self.invtransform_surreal(sample_coordinates, shift, trans)
        
        # sample_coordinates = self.inversedeforming_samplepoints_LBS(sample_coordinates, nnvidx, smplvert, T)
        
        # mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        # #sample_coordinates[~mask] = sample_coordinates[~mask] + 10
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates.obj')   
        
        # npvertices = self.smpl.v_template.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/canonmesh.obj')
        
        #raw smpl
        # smpl_reduced_current_vert = get_shape(self.smpl, body_pose=smpl_body_pose.view(batch_size,23,3),
                                      # global_orient=smpl_orient[:,None])  
        # smpl_reduced_current_vert *= smplscale
        # smpl_reduced_current_vert += smpl_translate[:,None] / 100
        # smpl_reduced_current = SMPLOutput(vertices=smpl_reduced_current_vert)
        
        # smpl_reduced_canon_vert = get_shape(self.smpl, body_pose=self.smpl_avg_body_pose.expand(batch_size, -1).view(batch_size,23,3),
                                      # global_orient=self.smpl_avg_orient.expand(batch_size, -1)[:,None])
        # smpl_reduced_canon_vert *= smplscale
        # #smpl_reduced_canon_vert += self.smpl_avg_transl.expand(3, -1)[:,None]                                      
        # smpl_reduced_canon = SMPLOutput(vertices=smpl_reduced_canon_vert)self.smpl_reduced.faces_t, smpl_reduced_current.vertices
        
        # rendermask = self.rasterize_eg3d(self.smplfaces, smplvert, extrinsics, intrinsic)
        # for i in range(0,1):
            # mask_render = rendermask[i].detach().cpu().numpy()
            # mask_path = 'test/smplmask{:04d}.png'.format(frameidx[i].item())       
            # cv2.imwrite(mask_path, mask_render * 255)            
            # raw_img = rawimg.permute(0,2,3,1)[i].detach().cpu().numpy()
            # im_path = 'test/rawimg{:04d}.png'.format(frameidx[i].item())       
            # cv2.imwrite(im_path, raw_img)
        #bonemask = bone_mask[0].detach().cpu().numpy()
        #cv2.imwrite('test/bonemask.png', bonemask * 255)
        
        # smpl_reduced_canon_vert = get_shape(self.smpl, body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1).view(batch_size,23,3),
                                      # global_orient=self.smpl_avg_orient.expand(bs_expand, -1)[:,None])
        # smpl_reduced_canon_vert *= self.smpl_avg_scale
        # smpl_reduced_canon_vert += self.smpl_avg_transl.expand(bs_expand, -1)[:,None]                                      
        # smpl_reduced_canon = SMPLOutput(vertices=smpl_reduced_canon_vert)

        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinatesraw.obj')   
        
        # npvertices = smpl_reduced_current.vertices[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/currmesh.obj')
        
        mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        #displacement
        # smplpara = torch.cat([smpl_params, betas], -1)
        # samplepts_disp =  torch.zeros_like(sample_coordinates)   
        # for i in range(smpl_params.shape[0]):
            # samplepts_disp[i][mask[i]] = self.displace(sample_coordinates[i][mask[i]], latentws[i][0], smplpara[i])
            # #print(samplepts_disp[i][mask[i]])
            # #sample_coordinates[i][mask[i]] = sample_coordinates[i][mask[i]]+samplepts_disp[i][mask[i]]
        # #samplepts_disp = self.displace(sample_coordinates, latentws, smplpara)
        samplepts_disp = 0
        
        sample_coordinates = self.get_canonical_coordinates(
            sample_coordinates,
            smpl_src=smpl_reduced_current,
            smpl_dst=smpl_reduced_canon,  
            mask = mask            
        )
        # for i in range(smpl_params.shape[0]):
            # sample_coordinates[i][mask[i]] = sample_coordinates[i][mask[i]]+samplepts_disp[i][mask[i]]
               
        #sample_coordinates[~mask] = 10
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates.obj')   
        
        # npvertices = smpl_reduced_canon.vertices[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/canonmesh.obj')
        
        # smvert = self.get_canonical_coordinates(
            # smpl_reduced_current.vertices,
            # smpl_src=smpl_reduced_current,
            # smpl_dst=smpl_reduced_canon          
        # )
        #smvert = smpl_reduced_canon.vertices
        # t = mask_at_box.reshape(-1,128,128)
        # bone_mask = bone_mask*t
        # bonemask = bone_mask[0].detach().cpu().numpy()
        # cv2.imwrite('test/bonemask.png', bonemask * 255)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates_can.obj') 
        
        # R = extrinsics[:,:3,:3] 
        # T = extrinsics[:,:3,3:] 
        # localvert = torch.matmul(R,smpl_reduced_current.vertices.permute(0,2,1))+T
        # projpixel = torch.matmul(intrinsic, localvert)
        # projpixel = projpixel/localvert[:,2,:]
        # print('smpl:',projpixel)
        # globaljoint = pose_to_world[:,:, :3, 3]
        # localvert = torch.matmul(R,globaljoint.permute(0,2,1))+T
        # projpixel = torch.matmul(intrinsic, localvert)
        # projpixel = projpixel/localvert[:,2,:]
        # print('joint:',projpixel)
        
        #mask = mask_at_box.unsqueeze(-1).unsqueeze(-1).repeat(1,1,Nc,1).view(sample_coordinates.shape[0],sample_coordinates.shape[1],1)
               
        # rgb_vals = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],32], dtype=torch.float32)#normlized to [-1,1]
        # sigma_vals = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32) * 1e3
        # #print(sample_coordinates.shape,mask.shape,sample_coordinates[mask].shape)
        #color, density = self.fetchingnerf(planes, sample_coordinates[0][mask[0]][None,...])
 
        color, density = self.fetchingnerf(planes, sample_coordinates)
        # sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros')
        # sampled_features = sampled_features.mean(1)
        # density, color = calc_density_and_color_from_feature(self, sampled_features.permute(0,2,1), z_rend, sample_coordinates)
        
        #density = self.density_activation(density) * 10
        # density = density.permute(0,2,1)
        # color = color.permute(0,2,1)

        density *= mask.unsqueeze(-1)
        
        rgb_vals = color#[mask]
        sigma_vals = density#[mask]
        
        colors_coarse = rgb_vals
        densities_coarse = sigma_vals
        
        # colors_coarse = out['rgb']
        # densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        #print(colors_coarse[densities_coarse.repeat(1,1,1,3)>0])
        # Mask out invalid samples (optional).
        is_sample_valid = None
        # if smpl_clip_depths is not None:
            # is_sample_valid = self.get_sample_mask(sample_depths=depths_coarse, min_max_depths=smpl_clip_depths)
            # densities_coarse = densities_coarse - 1000 * (1-is_sample_valid.float())

        # Fine Pass
        N_importance = 0#64#rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, batch_size)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            if 0:#rendering_options['box_warp_pre_deform']:
                sample_coordinates = (2 / rendering_options['box_warp']) * sample_coordinates
            sample_coordinates = self.get_canonical_coordinates(
                                sample_coordinates,
                                smpl_src=smpl_reduced_current,
                                smpl_dst=smpl_reduced_canon
                                )
            colors_fine, densities_fine = self.fetchingnerf(planes, sample_coordinates)
            # out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            # colors_fine = out['rgb']
            # densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            # Mask out invalid samples (optional).
            if 0:#smpl_clip_depths is not None:
                is_sample_valid = self.get_sample_mask(sample_depths=depths_fine, min_max_depths=smpl_clip_depths)
                densities_fine = densities_fine - 1000 * (1-is_sample_valid.float())
                #colors_fine = colors_fine * is_sample_valid.float()

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(all_colors, all_densities, all_depths, batch_size)
        else:
            rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, batch_size)
             
        #print(rgb_final.shape,is_valid.shape)rgb_final, depth_final, weights
        # rgb = rgb_final.reshape(2,64,64,32)
        # print(rgb.shape)
        #print(rgb_final[depth_final.squeeze(-1)>0])
        if is_sample_valid is not None: depth_final = is_sample_valid.any(-2).float()

        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp#rgb_final, depth_final, weights.sum(2)
    
    def samplingpoint_learningmeshsdf_smpl0(self, templatesmpl, templatecloth):
    
        smplfaces = self.smplreducedfaces
        
        smplfacevert1 = templatesmpl[:,smplfaces[:,0],:] 
        smplfacevert2 = templatesmpl[:,smplfaces[:,1],:]      
        smplfacevert3 = templatesmpl[:,smplfaces[:,2],:] 
        
        # smplfacevert1 = smplfacevert1.repeat(2,1)
        # smplfacevert2 = smplfacevert2.repeat(2,1)
        # smplfacevert3 = smplfacevert3.repeat(2,1)
        
        numscale = 10        
        n_sample = smplfaces.shape[0]*numscale
        faceweight = torch.rand(n_sample, 3)
        faceweight = faceweight.to(templatesmpl)
        faceweightsum = torch.sum(faceweight,1)
        faceweight = faceweight/faceweightsum[:,None]
        #meshpts_smpl = smplfacevert1*faceweight[:,0][None,:][...,None]+smplfacevert2*faceweight[:,1][None,:][...,None]+smplfacevert3*faceweight[:,2][None,:][...,None]
        meshpts_smpl = smplfacevert1.repeat(1,numscale,1)*faceweight[:,0][None,:][...,None]+smplfacevert2.repeat(1,numscale,1)*faceweight[:,1][None,:][...,None]+smplfacevert3.repeat(1,numscale,1)*faceweight[:,2][None,:][...,None]
        
        clothfaces = self.cloth_reduced.faces
        #clothfaces = self.clothes_watertight_face
        
        clothfacevert1 = templatecloth[:,clothfaces[:,0],:] 
        clothfacevert2 = templatecloth[:,clothfaces[:,1],:]      
        clothfacevert3 = templatecloth[:,clothfaces[:,2],:] 
    
        n_sample = clothfaces.shape[0]*numscale
        faceweight = torch.rand(n_sample, 3)
        faceweight = faceweight.to(templatecloth)
        faceweightsum = torch.sum(faceweight,1)
        faceweight = faceweight/faceweightsum[:,None]
        #meshpts_smpl = smplfacevert1*faceweight[:,0][None,:][...,None]+smplfacevert2*faceweight[:,1][None,:][...,None]+smplfacevert3*faceweight[:,2][None,:][...,None]
        meshpts_cloth = clothfacevert1.repeat(1,numscale,1)*faceweight[:,0][None,:][...,None]+clothfacevert2.repeat(1,numscale,1)*faceweight[:,1][None,:][...,None]+clothfacevert3.repeat(1,numscale,1)*faceweight[:,2][None,:][...,None]
        

        # _,trinormal = mesh_face_areas_normals(templatesmpl.view(-1,3), self.smplfaces.view(-1,3))
        # #trinormal = trinormal.view(batch_size,-1,3)
        # sdf_smpl = torch.zeros([n_sample]).to(self.device)
        # normal_smpl = trinormal#.repeat(2,1)[ind2].astype(np.float32)
        
        min_xyz, _ = torch.min(templatesmpl, axis=1)
        max_xyz, _ = torch.max(templatesmpl, axis=1)
        min_xyz = min_xyz - 0.3
        max_xyz = max_xyz + 0.3
       
        gridsample = n_sample*5
        # x_vals = torch.random.uniform(0, 1, n_sample//2)
        # y_vals = torch.random.uniform(0, 1, n_sample//2)
        # z_vals = torch.random.uniform(0, 1, n_sample//2)
        # vals = np.stack([x_vals, y_vals, z_vals], axis=1)        
        vals = torch.rand(templatesmpl.shape[0],gridsample, 3).to(templatesmpl)

        samplepts = (max_xyz[:,None] - min_xyz[:,None]) * vals + min_xyz[:,None]
        
        # npvertices = meshpts_cloth[0].detach().cpu().numpy()
        # npfaces = clothfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/meshpts_cloth.obj')
        # mesh.export(result_path)
        
        # npvertices = templatecloth[0].detach().cpu().numpy()
        # npfaces = clothfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/templatecloth.obj')
        # mesh.export(result_path)
        
        # npvertices = meshpts_smpl[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/meshpts_smpl.obj')
        # mesh.export(result_path)
        
        # npvertices = templatesmpl[0].detach().cpu().numpy()
        # npfaces = self.smplreducedfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/smpl_template.obj')
        # mesh.export(result_path)
        
        # npvertices = samplepts[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/samplepts.obj')
        # mesh.export(result_path)
        
        return meshpts_smpl, samplepts, meshpts_cloth
    
    def samplingpoint_learningmeshsdf(self, templatesmpl, templatecloth):
      
        smplfaces = self.smplfaces[self.outvfidx,:]#self.smplreducedfaces

        smplfacevert1 = templatesmpl[:,smplfaces[:,0],:] 
        smplfacevert2 = templatesmpl[:,smplfaces[:,1],:]      
        smplfacevert3 = templatesmpl[:,smplfaces[:,2],:] 
        
        # smplfacevert1 = smplfacevert1.repeat(2,1)
        # smplfacevert2 = smplfacevert2.repeat(2,1)
        # smplfacevert3 = smplfacevert3.repeat(2,1)
        
        numscale = 10        
        n_sample = smplfaces.shape[0]*numscale
        faceweight = torch.rand(n_sample, 3)
        faceweight = faceweight.to(templatesmpl)
        faceweightsum = torch.sum(faceweight,1)
        faceweight = faceweight/faceweightsum[:,None]
        #meshpts_smpl = smplfacevert1*faceweight[:,0][None,:][...,None]+smplfacevert2*faceweight[:,1][None,:][...,None]+smplfacevert3*faceweight[:,2][None,:][...,None]
        meshpts_smpl = smplfacevert1.repeat(1,numscale,1)*faceweight[:,0][None,:][...,None]+smplfacevert2.repeat(1,numscale,1)*faceweight[:,1][None,:][...,None]+smplfacevert3.repeat(1,numscale,1)*faceweight[:,2][None,:][...,None]
        
        batch_size = templatesmpl.shape[0]
        trinormal_smpl = []
        for i in range(0, batch_size):
            _,trinormal = mesh_face_areas_normals(templatesmpl[i], self.smplfaces)
            trinormal_smpl.append(trinormal[None])#[self.smpl_vfidx,:])
        trinormal_smpl = torch.cat(trinormal_smpl)
        trinormal_cloth = []
        for i in range(0, batch_size):
            _,trinormal = mesh_face_areas_normals(templatecloth[i], self.clothes_watertight_face)
            trinormal_cloth.append(trinormal[None])#[self.smpl_vfidx,:])
        trinormal_cloth = torch.cat(trinormal_cloth)
        meshpts_normal_smpl = trinormal_smpl[:,self.outvfidx,:].repeat(1,numscale,1)
        meshpts_normal_cloth = trinormal_cloth.repeat(1,numscale,1)
        meshpts_normal = torch.cat([meshpts_normal_smpl,meshpts_normal_cloth], dim=1)
        
        # _,trinormal_smpl = mesh_face_areas_normals(templatesmpl[0].view(-1,3), smplfaces)
        # batch_size = templatesmpl.shape[0]
        # meshpts_normal_smpl = trinormal_smpl[None].repeat(batch_size,numscale,1)
        
        
        #clothfaces = self.reducedcloth_faces
        clothfaces = self.clothes_watertight_face
        
        clothfacevert1 = templatecloth[:,clothfaces[:,0],:] 
        clothfacevert2 = templatecloth[:,clothfaces[:,1],:]      
        clothfacevert3 = templatecloth[:,clothfaces[:,2],:] 
    
        n_sample = clothfaces.shape[0]*numscale
        faceweight = torch.rand(n_sample, 3)
        faceweight = faceweight.to(templatecloth)
        faceweightsum = torch.sum(faceweight,1)
        faceweight = faceweight/faceweightsum[:,None]
        #meshpts_smpl = smplfacevert1*faceweight[:,0][None,:][...,None]+smplfacevert2*faceweight[:,1][None,:][...,None]+smplfacevert3*faceweight[:,2][None,:][...,None]
        meshpts_cloth = clothfacevert1.repeat(1,numscale,1)*faceweight[:,0][None,:][...,None]+clothfacevert2.repeat(1,numscale,1)*faceweight[:,1][None,:][...,None]+clothfacevert3.repeat(1,numscale,1)*faceweight[:,2][None,:][...,None]
        
        # _,trinormal_cloth = mesh_face_areas_normals(templatecloth[0].view(-1,3), self.clothes_watertight_face)
        # meshpts_normal_cloth = trinormal_cloth[None].repeat(batch_size,numscale,1)
        # meshpts_normal = torch.cat([meshpts_normal_smpl,meshpts_normal_cloth], dim=1)
        
        # _,trinormal = mesh_face_areas_normals(templatesmpl.view(-1,3), self.smplfaces.view(-1,3))
        # #trinormal = trinormal.view(batch_size,-1,3)
        # sdf_smpl = torch.zeros([n_sample]).to(self.device)
        # normal_smpl = trinormal#.repeat(2,1)[ind2].astype(np.float32)
        
        min_xyz, _ = torch.min(templatesmpl, axis=1)
        max_xyz, _ = torch.max(templatesmpl, axis=1)
        min_xyz = min_xyz - 0.3
        max_xyz = max_xyz + 0.3
        
        gridsample = n_sample//10#*5
        # x_vals = torch.random.uniform(0, 1, n_sample//2)
        # y_vals = torch.random.uniform(0, 1, n_sample//2)
        # z_vals = torch.random.uniform(0, 1, n_sample//2)
        # vals = np.stack([x_vals, y_vals, z_vals], axis=1)        
        vals = torch.rand(templatesmpl.shape[0],gridsample, 3).to(templatesmpl)
        samplepts_smpl = (max_xyz[:,None] - min_xyz[:,None]) * vals + min_xyz[:,None]
        
        # smpl_ptsdist = torch.cdist(samplepts_smpl, templatesmpl, p=2)
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # smpl_neartag = smpl_distmin<0.05
        # samplepts_smpl[smpl_neartag] = 10
        
        # min_xyz1, _ = torch.min(templatecloth, axis=1)
        # max_xyz1, _ = torch.max(templatecloth, axis=1)
        # min_xyz1 = min_xyz1 - 0.3
        # max_xyz1 = max_xyz1 + 0.3
        vals1 = torch.rand(templatecloth.shape[0],gridsample, 3).to(templatecloth)
        samplepts_cloth = (max_xyz[:,None] - min_xyz[:,None]) * vals1 + min_xyz[:,None]
        
        # cloth_ptsdist = torch.cdist(samplepts_cloth, templatecloth, p=2)
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_distmin = cloth_ptsdistmin[0]
        # cloth_neartag = cloth_distmin<0.05
        # samplepts_cloth[cloth_neartag] = 10
        
        # from pytorch3d.ops import mesh_face_normal, line_mesh_distance
        # mesh = pytorch3d.structures.Meshes(verts=templatecloth, faces=self.clothes_watertight_face[None,...].repeat(templatecloth.shape[0],1,1))
        # face_normals = mesh_face_normal(mesh, normalize=True)
        # distances, closest_points, closest_faces = line_mesh_distance(mesh, samplepts_cloth)
        
        # npvertices = meshpts_cloth[0].detach().cpu().numpy()
        # npfaces = clothfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/meshpts_cloth.obj')
        # mesh.export(result_path)
        
        # npvertices = templatecloth[0].detach().cpu().numpy()
        # npfaces = clothfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/templatecloth.obj')
        # mesh.export(result_path)
        
        # npvertices = meshpts_smpl[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/meshpts_smpl.obj')
        # mesh.export(result_path)
        
        # npvertices = templatesmpl[0].detach().cpu().numpy()
        # npfaces = self.smplreducedfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/smpl_template.obj')
        # mesh.export(result_path)
        
        # npvertices = samplepts_smpl[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/samplepts_smpl.obj')
        # mesh.export(result_path)
        
        # npvertices = samplepts_cloth[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/samplepts_cloth.obj')
        # mesh.export(result_path)
        
        return meshpts_smpl, meshpts_cloth, samplepts_smpl, samplepts_cloth, meshpts_normal
    
    def samplingpoint_learningmeshsdf_smpl(self, templatesmpl):
      
        smplfaces = self.smplfaces#[self.outvfidx,:]#self.smplreducedfaces

        smplfacevert1 = templatesmpl[:,smplfaces[:,0],:] 
        smplfacevert2 = templatesmpl[:,smplfaces[:,1],:]      
        smplfacevert3 = templatesmpl[:,smplfaces[:,2],:] 
        
        # smplfacevert1 = smplfacevert1.repeat(2,1)
        # smplfacevert2 = smplfacevert2.repeat(2,1)
        # smplfacevert3 = smplfacevert3.repeat(2,1)
        
        numscale = 10        
        n_sample = smplfaces.shape[0]*numscale
        faceweight = torch.rand(n_sample, 3)
        faceweight = faceweight.to(templatesmpl)
        faceweightsum = torch.sum(faceweight,1)
        faceweight = faceweight/faceweightsum[:,None]
        #meshpts_smpl = smplfacevert1*faceweight[:,0][None,:][...,None]+smplfacevert2*faceweight[:,1][None,:][...,None]+smplfacevert3*faceweight[:,2][None,:][...,None]
        meshpts_smpl = smplfacevert1.repeat(1,numscale,1)*faceweight[:,0][None,:][...,None]+smplfacevert2.repeat(1,numscale,1)*faceweight[:,1][None,:][...,None]+smplfacevert3.repeat(1,numscale,1)*faceweight[:,2][None,:][...,None]
        
        batch_size = templatesmpl.shape[0]
        trinormal_smpl = []
        for i in range(0, batch_size):
            _,trinormal = mesh_face_areas_normals(templatesmpl[i], self.smplfaces)
            trinormal_smpl.append(trinormal[None])#[self.smpl_vfidx,:])
        trinormal_smpl = torch.cat(trinormal_smpl)
        
        meshpts_normal_smpl = trinormal_smpl.repeat(1,numscale,1)#[:,self.outvfidx,:]
        
        meshpts_normal = meshpts_normal_smpl#torch.cat([meshpts_normal_smpl,meshpts_normal_cloth], dim=1)
        
        # _,trinormal_smpl = mesh_face_areas_normals(templatesmpl[0].view(-1,3), smplfaces)
        # batch_size = templatesmpl.shape[0]
        # meshpts_normal_smpl = trinormal_smpl[None].repeat(batch_size,numscale,1)
        
        min_xyz, _ = torch.min(templatesmpl, axis=1)
        max_xyz, _ = torch.max(templatesmpl, axis=1)
        min_xyz = min_xyz - 0.3
        max_xyz = max_xyz + 0.3
        
        gridsample = n_sample//10#*5
        # x_vals = torch.random.uniform(0, 1, n_sample//2)
        # y_vals = torch.random.uniform(0, 1, n_sample//2)
        # z_vals = torch.random.uniform(0, 1, n_sample//2)
        # vals = np.stack([x_vals, y_vals, z_vals], axis=1)        
        vals = torch.rand(templatesmpl.shape[0],gridsample, 3).to(templatesmpl)
        samplepts_smpl = (max_xyz[:,None] - min_xyz[:,None]) * vals + min_xyz[:,None]
        
        return meshpts_smpl, samplepts_smpl, meshpts_normal
            
    def meshdensity_deepcap0(self, planes, smpl_params, betas, Th, extrinsics, intrinsic, local_cube_coordinates):

        R = extrinsics[:,:3,:3] 
        T = extrinsics[:,:3,3:]

        global_cube_coordinates = torch.bmm(local_cube_coordinates - T.permute(0,2,1), R)

        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        
        batch_size = R.shape[0]
        bs_expand = batch_size 
              
        smpl_reduced_current = self.smpl_reduced(betas=betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=self.smpl_avg_transl.expand(bs_expand, -1))
      
        smpl_reduced_canon = self.smpl_reduced(betas=betas, body_pose=self.smpl_avg_body_pose.expand(batch_size, -1),
                                      global_orient=self.smpl_avg_orient.expand(batch_size, -1), transl=self.smpl_avg_transl.expand(bs_expand, -1))
        
        smpl_reduced_current.vertices += Th[:,None]
        
       
        canonical_cube_coordinates = self.get_canonical_coordinates(
            global_cube_coordinates,
            smpl_src=smpl_reduced_current,
            smpl_dst=smpl_reduced_canon          
        )
        
        color, density = self.fetchingnerf(planes, canonical_cube_coordinates)
        
        #rgb_vals = color#[mask]
        #sigma_vals = density#[mask]
        
        #colors_coarse = rgb_vals
        #densities_coarse = sigma_vals
        
        #colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        
        densities_coarse = density.reshape(batch_size, -1, 1)
        
        return densities_coarse
    
    def meshdensity_deepcap(self, frameidx, i, planes, smpl_params, betas, Th, A, extrinsics, intrinsic, local_cube_coordinates):

        R = extrinsics[:,:3,:3] 
        T = extrinsics[:,:3,3:]

        global_cube_coordinates = torch.bmm(local_cube_coordinates - T.permute(0,2,1), R)

        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        
        batch_size = R.shape[0]
        bs_expand = batch_size 
              
        #deforming with cloth and smpl
        smpl_current = self.deformingsmpl_LBS(A, Th)
        smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)
        
        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)
        cloth_reduced_current = self.cloth_reduced(cloth_current)
        
        if i==0:
            npvertices = smpl_current[0].detach().cpu().numpy()
            npfaces = self.smplfaces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            result_path = os.path.join('test/smpl_current{:04d}.obj'.format(frameidx[0].item()))
            mesh.export(result_path)
            npvertices = cloth_current[0].detach().cpu().numpy()
            npfaces = self.clothfaces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            result_path = os.path.join('test/cloth_current{:04d}.obj'.format(frameidx[0].item()))
            mesh.export(result_path)
        
       
        combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        canonical_cube_coordinates = self.get_canonical_coordinates_clothvertex(
            global_cube_coordinates,
            combinedmesh_reduced_current,
            self.combinedmesh_reduced_canon.repeat(batch_size,1,1)           
        )
        
        # smpl_ptsdist = torch.cdist(global_cube_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
        # cloth_ptsdist = torch.cdist(global_cube_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_distmin = cloth_ptsdistmin[0]
        # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
        
        # cloth_neartag = cloth_distmin<smpl_distmin
        
        # sample_coordinates_smpl = self.get_canonical_coordinates(
            # global_cube_coordinates,
            # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1))            
        # )

        # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
            # global_cube_coordinates,
            # cloth_reduced_current,
            # self.cloth_reduced_canon.repeat(batch_size,1,1)
        # )
        # canonical_cube_coordinates = sample_coordinates_smpl
        # canonical_cube_coordinates.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
        # canonical_cube_coordinates = canonical_cube_coordinates.view(batch_size,-1,3)
        
        
        color, density = self.fetchingnerf(planes, canonical_cube_coordinates)
        
        densities_coarse = density.reshape(batch_size, -1, 1)
        
        return densities_coarse
    
    def canonmeshdensity_deepcap_doublelayer(self, frameidx, i, smplweight,clothweight, planes, smpl_params, betas, Th, A, extrinsics, intrinsic, cube_coordinates):

        batch_size = smpl_params.shape[0]       
        
        self.sample_canonshape(batch_size,smplweight,clothweight)
        
        #density_smpl, density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], cube_coordinates, cube_coordinates)
        density_cloth,_ = self.SDFnetwork_cloth(cube_coordinates,self.shapepara)#.repeat(batch_size,1)

        density_smpl,_ = self.SDFnetwork_smpl(cube_coordinates,self.canonparamshape)

        if i==0:
            npvertices = self.rawtemplatesmpl[0].detach().cpu().numpy()
            npfaces = self.smplfaces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            result_path = os.path.join('test/smpl_canon{:04d}.obj'.format(frameidx[0].item()))
            mesh.export(result_path)
            npvertices = self.canonclothvert[0].detach().cpu().numpy()
            npfaces = self.clothfaces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            result_path = os.path.join('test/cloth_canon{:04d}.obj'.format(frameidx[0].item()))
            mesh.export(result_path)
            
        # smpl_ptsdist = torch.cdist(global_cube_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
        # cloth_ptsdist = torch.cdist(global_cube_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_distmin = cloth_ptsdistmin[0]
        # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
        
        # cloth_neartag = cloth_distmin<smpl_distmin
        
        # sample_coordinates_smpl = self.get_canonical_coordinates(
            # global_cube_coordinates,
            # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1))            
        # )

        # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
            # global_cube_coordinates,
            # cloth_reduced_current,
            # self.cloth_reduced_canon.repeat(batch_size,1,1)
        # )
        # canonical_cube_coordinates = sample_coordinates_smpl
        # canonical_cube_coordinates.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
        # canonical_cube_coordinates = canonical_cube_coordinates.view(batch_size,-1,3)
        
        
        # color, density = self.fetchingnerf(planes, canonical_cube_coordinates)
        #density_smpl = sdf_to_alpha(density_smpl.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s * 1.0))  
        
        density_smpl = density_smpl.reshape(batch_size, -1, 1)
        density_cloth = density_cloth.reshape(batch_size, -1, 1)
        
        return density_smpl, density_cloth
    
    def canonmeshdensity_deepcap_doublelayer_fixgeometry(self, frameidx, i, smplweight,clothweight, planes, smpl_params, betas, Th, A, extrinsics, intrinsic, cube_coordinates):

        batch_size = smpl_params.shape[0]       
        
        #self.sample_canonshape(batch_size,smplweight,clothweight)
        
        #density_smpl, density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], cube_coordinates, cube_coordinates)
        density_cloth,_ = self.SDFnetwork_cloth(cube_coordinates,self.shapepara.repeat(batch_size,1))#.repeat(batch_size,1)

        density_smpl,_ = self.SDFnetwork_smpl(cube_coordinates,self.canonparamshape.repeat(batch_size,1))

        # incsdf_smpl = self.incSDFnetwork_smpl(cube_coordinates)
        # incsdf_cloth = self.incSDFnetwork_cloth(cube_coordinates)
        # density_smpl = density_smpl+incsdf_smpl     
        # density_cloth = density_cloth+incsdf_cloth
        
        # sdf_cloth,_ = self.SDFnetwork_cloth(self.canonclothvert,self.shapepara.repeat(batch_size,1))#.repeat(batch_size,1)
        # print(sdf_cloth[0])
        
        if i==0:
            npvertices = self.rawtemplatesmpl[0].detach().cpu().numpy()
            npfaces = self.smplfaces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            result_path = os.path.join('result/deepcap/test/smpl_canon{:04d}.obj'.format(frameidx[0].item()))
            mesh.export(result_path)
            npvertices = self.canonclothvert[0].detach().cpu().numpy()
            npfaces = self.clothfaces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            result_path = os.path.join('result/deepcap/test/cloth_canon{:04d}.obj'.format(frameidx[0].item()))
            mesh.export(result_path)
            
        density_smpl = density_smpl.reshape(batch_size, -1, 1)
        density_cloth = density_cloth.reshape(batch_size, -1, 1)
        
        return density_smpl, density_cloth

    def canonmeshdensity_deepcap_singlelayer_fixgeometry(self, frameidx, i, smplweight,clothweight, planes, smpl_params, betas, Th, A, extrinsics, intrinsic, cube_coordinates):

        batch_size = smpl_params.shape[0]       
        
        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        batch_size = smpl_params.shape[0]
        
        #self.sample_canonshape(batch_size)
        
        smploutput = self.smpl.forward(betas, smpl_body_pose, smpl_orient, Th)#        
        smpl_current = smploutput.vertices
        
        nnvidxtag = [] 
        smpl_nnvidx = [] 
        cloth_nnvidx = []
        sptneartag = []        
        for i in range(0, cube_coordinates.shape[1], 1000):
            smpl_ptsdist = torch.cdist(cube_coordinates[0, i:i + 1000, :], smpl_current[0], p=2)#[mask[i]]deformedpersonsmpl
            smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            smpl_distmin = smpl_ptsdistmin[0]
            smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1)#[mask[i]]
            smpl_nnvidx.append(smpl_minidx)
            
            # cloth_ptsdist = torch.cdist(sample_coordinates[i], cloth_current[i], p=2)#deformedpersonsmpl
            # cloth_ptsdistmin = torch.min(cloth_ptsdist, 1)
            # cloth_distmin = cloth_ptsdistmin[0]
            # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1)
            # cloth_nnvidx.append(cloth_minidx[None])
            # idxtag = cloth_distmin<smpl_distmin
            # nnvidxtag.append(idxtag[None])
            
            # ptdist = torch.logical_or(cloth_distmin<0.03, smpl_distmin<0.03)
            ptdist = smpl_distmin<0.05
            sptneartag.append(ptdist)
        
        sptneartag = torch.cat(sptneartag)[None]   
        smpl_nnvidx = torch.cat(smpl_nnvidx) [None]   
        # cloth_nnvidx = torch.cat(cloth_nnvidx)
        # nnvidxtag = torch.cat(nnvidxtag)        #[mask].view(batch_size,-1,3)
        cube_coordinates_can = self.inversedeforming_samplepoints_LBS(cube_coordinates, smpl_nnvidx, smpl_current, A, Th, betas, smpl_params)
        
        #self.sample_canonshape(batch_size,smplweight,clothweight)
        
        #density_smpl, density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], cube_coordinates, cube_coordinates)
        #density_cloth,_ = self.SDFnetwork_cloth(cube_coordinates,self.shapepara.repeat(batch_size,1))#.repeat(batch_size,1)

        #density,_ = self.SDFnetwork(cube_coordinates,self.canonparamshape.repeat(batch_size,1))
        #density = self.Densitynetwork(cube_coordinates)
        density = self.Densitynetwork(cube_coordinates_can)
        color, incdensity = self.decoder(cube_coordinates_can, smplweight[:,:128])
        density = density + incdensity       
        # incsdf_smpl = self.incSDFnetwork_smpl(cube_coordinates)
        # incsdf_cloth = self.incSDFnetwork_cloth(cube_coordinates)
        # density_smpl = density_smpl+incsdf_smpl     
        # density_cloth = density_cloth+incsdf_cloth
        density[~sptneartag] = 10
        # sdf_cloth,_ = self.SDFnetwork_cloth(self.canonclothvert,self.shapepara.repeat(batch_size,1))#.repeat(batch_size,1)
        # print(sdf_cloth[0])
        
        # if i==0:
            # npvertices = self.rawtemplatesmpl[0].detach().cpu().numpy()
            # npfaces = self.smplfaces.detach().cpu().numpy()
            # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            # result_path = os.path.join('result/deepcap/test/smpl_canon{:04d}.obj'.format(frameidx[0].item()))
            # mesh.export(result_path)
            # npvertices = self.canonclothvert[0].detach().cpu().numpy()
            # npfaces = self.clothfaces.detach().cpu().numpy()
            # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            # result_path = os.path.join('result/deepcap/test/cloth_canon{:04d}.obj'.format(frameidx[0].item()))
            # mesh.export(result_path)
            
        # beta = self.beta_network().clamp(1e-9, 1e6)
        # density = self.sdf_to_alpha(density.reshape(batch_size, -1), beta)
        
        #density = sdf_to_alpha2(density.reshape(batch_size, -1), torch.exp(self.ln_s))  #      
        density = density.reshape(batch_size, -1, 1)
        #density_cloth = density_cloth.reshape(batch_size, -1, 1)
        
        return density, density
        
    def meshdensity_deepcap_doublelayer(self, frameidx, i, planes, smpl_params, betas, Th, A, extrinsics, intrinsic, local_cube_coordinates):

        R = extrinsics[:,:3,:3] 
        T = extrinsics[:,:3,3:]

        global_cube_coordinates = torch.bmm(local_cube_coordinates - T.permute(0,2,1), R)

        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        
        batch_size = R.shape[0]
        bs_expand = batch_size 
              
        #deforming with cloth and smpl
        smpl_current = self.deformingsmpl_LBS(A, Th)
        smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)
        
        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)
        cloth_reduced_current = self.cloth_reduced(cloth_current)
        
        if i==0:
            npvertices = smpl_current[0].detach().cpu().numpy()
            npfaces = self.smplfaces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            result_path = os.path.join('test/smpl_current{:04d}.obj'.format(frameidx[0].item()))
            mesh.export(result_path)
            npvertices = cloth_current[0].detach().cpu().numpy()
            npfaces = self.clothfaces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            result_path = os.path.join('test/cloth_current{:04d}.obj'.format(frameidx[0].item()))
            mesh.export(result_path)
        
       
        # combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        # combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        # canonical_cube_coordinates = self.get_canonical_coordinates_clothvertex(
            # global_cube_coordinates,
            # combinedmesh_reduced_current,
            # self.combinedmesh_reduced_canon.repeat(batch_size,1,1)           
        # )
        
        sample_coordinates_smpl = self.get_canonical_coordinates(
            global_cube_coordinates,
            smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1))            
        )

        sample_coordinates = self.get_canonical_coordinates_clothvertex(
            global_cube_coordinates,
            cloth_reduced_current,
            self.cloth_reduced_canon.repeat(batch_size,1,1)          
        )
        
        color_smpl, density_smpl, color_cloth, density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        
        # smpl_ptsdist = torch.cdist(global_cube_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
        # cloth_ptsdist = torch.cdist(global_cube_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_distmin = cloth_ptsdistmin[0]
        # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
        
        # cloth_neartag = cloth_distmin<smpl_distmin
        
        # sample_coordinates_smpl = self.get_canonical_coordinates(
            # global_cube_coordinates,
            # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1))            
        # )

        # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
            # global_cube_coordinates,
            # cloth_reduced_current,
            # self.cloth_reduced_canon.repeat(batch_size,1,1)
        # )
        # canonical_cube_coordinates = sample_coordinates_smpl
        # canonical_cube_coordinates.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
        # canonical_cube_coordinates = canonical_cube_coordinates.view(batch_size,-1,3)
        
        
        # color, density = self.fetchingnerf(planes, canonical_cube_coordinates)
        #density_smpl = sdf_to_alpha(density_smpl.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s * 1.0))  
        
        density_smpl = density_smpl.reshape(batch_size, -1, 1)
        density_cloth = density_cloth.reshape(batch_size, -1, 1)
        
        return density_smpl, density_cloth
        
    def doublenerf_rendering_deepcap(self, frameidx, rawimg, Nc, Nf, z_rend, planes, latentws, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):

        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]

        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        
         # Create stratified depth samples
        ray_start = near
        ray_end = far
        depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
        bs_expand = batch_size 
         
        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       
        
        #self.sample_canonshape(batch_size)
        
        #deforming with cloth and smpl
        smpl_current = self.deformingsmpl_LBS(A, Th)
        smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)
 
        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)
        cloth_reduced_current = self.cloth_reduced(cloth_current)
        
        mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        
        combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        #combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        
        # npvertices = combinedmeshvert[0].detach().cpu().numpy()                        
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = f"test/combined0_{frameidx[0].item():0>4}.obj"
        # mesh.export(result_path)
        
        # data_root = '../../neuralbody-deformation-occupancy-fixnerf-our-generalization/tools/data/magdalena20000-allviews/'
        # smplvert_path = os.path.join(data_root, 'vertices',
                                   # '{}.npy'.format(frameidx[0].item()))
        # smplvert = np.load(smplvert_path).astype(np.float32)
        # smplmesh = trimesh.Trimesh(smplvert, self.smplfaces.detach().cpu().numpy(), process=False)
        # result_path = f"test/smpl0_{frameidx[0].item():0>4}.obj"
        # smplmesh.export(result_path)
                    
        with torch.no_grad(): 
            clothmask = self.render_clothmask(self.meshface.repeat(batch_size,1,1), combinedmeshvert, extrinsics, intrinsic)
            renderclothmask = clothmask.unsqueeze(-1).repeat(1,1,1,Nc).view(batch_size,-1,Nc,1)
        
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # combined_distmin = sample_coordinates.new_zeros([batch_size,sample_coordinates.shape[1]], dtype=torch.float32)              
        # for i in range(0, batch_size, 1):                   
            # combined_ptsdist = torch.cdist(sample_coordinates[i][mask[i]], combinedmesh_reduced_current[i], p=2)#deformedpersonsmpl
            # combined_ptsdistmin = torch.min(combined_ptsdist, 1)           
            # ptsdist = torch.squeeze(combined_ptsdistmin[0], -1)  # B*P 
            # print(ptsdist)            
            # combined_distmin[i][mask[i]] = ptsdist
            
        # meshhumanmask = combined_distmin<0.05
        
        # mesh_mask = meshhumanmask.reshape(batch_size, 128, 128,48)#low resolution for nerf rendering
        # mesh_mask = mesh_mask.sum(-1)                 
        # mesh_mask = mesh_mask>0
        
        # mesh_mask = clothmask[0].detach().cpu().numpy()
        # cv2.imwrite('test/cloth_mask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        
        # sample_coordinates = self.get_canonical_coordinates_clothvertex(
            # sample_coordinates,
            # combinedmesh_reduced_current,
            # self.combinedmesh_reduced_canon.repeat(batch_size,1,1),  
            # mask            
        # )
        # with torch.no_grad(): 
            # mesh_mask = self.rasterize_eg3d(self.combinedmesh_reduced_faces, combinedmesh_reduced_current, extrinsics, intrinsic)
            # meshhumanmask = mesh_mask.unsqueeze(-1).repeat(1,1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        
        # combined_distmin = sample_coordinates.new_zeros([batch_size,sample_coordinates.shape[1]], dtype=torch.float32)              
        # for i in range(0, batch_size, 1):
            # tnum = int(sample_coordinates[i].shape[0]/2)                     
            # combined_ptsdist = torch.cdist(sample_coordinates[i][:tnum], combinedmesh_reduced_current[i], p=2)#deformedpersonsmpl
            # combined_ptsdistmin = torch.min(combined_ptsdist, 1)           
            # ptsdist = torch.squeeze(combined_ptsdistmin[0], -1)  # B*P            
            # combined_ptsdist = torch.cdist(sample_coordinates[i][tnum:], combinedmesh_reduced_current[i], p=2)#deformedpersonsmpl
            # combined_ptsdistmin1 = torch.min(combined_ptsdist, 1)
            # ptsdist1 = torch.squeeze(combined_ptsdistmin1[0], -1)
            # ptsdist = torch.cat((ptsdist,ptsdist1),0)
            # combined_distmin[i] = ptsdist
            # print(ptsdist[mask[i]])
                
        
        # npvertices = combinedmesh_reduced_current[0].detach().cpu().numpy()
        # npfaces = self.combinedmesh_reduced.faces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/deformedverts_reduced{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = self.combinedmesh_reduced_canon[0].detach().cpu().numpy()
        # npfaces = self.combinedmesh_reduced.faces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/canverts_reduced{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        
        #interp_loss = self.computeinterpenetrationloss_posedsmpl(canondeformedcloth, smpl_current, cloth_current)
                
        # meshvert_def = torch.cat([cloth_current, smpl_current], dim=1)#self.deformedcloth
        # npvertices = meshvert_def[0].detach().cpu().numpy()
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/deformedverts{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = smpl_current[0].detach().cpu().numpy()
        # npfaces = self.smplfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/smpl_current{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = cloth_current[0].detach().cpu().numpy()
        # npfaces = self.clothfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/cloth_current{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
         
        # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
        # cloth_ptsdist = torch.cdist(sample_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_distmin = cloth_ptsdistmin[0]
        # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
        
        sample_coordinates_smpl = self.get_canonical_coordinates(
            sample_coordinates,
            smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)),  
            mask = mask            
        )

        sample_coordinates = self.get_canonical_coordinates_clothvertex(
            sample_coordinates,
            cloth_reduced_current,
            self.cloth_reduced_canon.repeat(batch_size,1,1),  
            mask            
        )
        
        # sample_coordinates = sample_coordinates_smpl
        # sample_coordinates.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
        # sample_coordinates = sample_coordinates.view(batch_size,-1,3)
        
        
        # meshvert_def = torch.cat([self.canonclothvert, self.rawtemplatesmpl], dim=1)#self.deformedcloth
        # npvertices = meshvert_def[0].detach().cpu().numpy()
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/canverts{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = self.rawtemplatesmpl[0].detach().cpu().numpy()
        # npfaces = self.smplfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/smpl_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = self.canonclothvert[0].detach().cpu().numpy()
        # npfaces = self.clothfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/cloth_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/samplecan_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        
        #smpl model, vert transformation
        # smplvert, A, T = get_shape(self.smpl, body_pose=smpl_body_pose.view(batch_size,23,3),
                                       # global_orient=smpl_orient[:,None]) 
        # smplvert = torch.matmul(trans[:,:3,:3], smplvert.permute(0,2,1))
        # smplvert = smplvert.permute(0,2,1)
        # smplvert += shift[:,None]
        # # axis_transform
        # smplvert = smplvert[:, :, [1, 2, 0]] * torch.tensor([-1, -1, -1]).view(1,3)[:, None].to(betas.device)
        # mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])

        # nnvidx = mask.new_zeros([mask.shape[0],mask.shape[1]], dtype=torch.long)              
        # for i in range(0, smplvert.shape[0], 1):
            # # tnum = int(sample_coordinates[i][mask[i]].shape[0]/2)
            # # pts = sample_coordinates[i][mask[i]][:tnum]                       
            # # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][:tnum], smplvert[i], p=2)#deformedpersonsmpl
            # # smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            # # vidx = torch.squeeze(smpl_ptsdistmin[1], -1)  # B*P
            # # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][tnum:], smplvert[i], p=2)#deformedpersonsmpl
            # # smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            # # vidx1 = torch.squeeze(smpl_ptsdistmin[1], -1)
            # # vidx = torch.cat((vidx,vidx1),0)
            # # nnvidx[i][mask[i]] = vidx

            # vidx = []
            # tnum = int(sample_coordinates[i][mask[i]].shape[0]/4)          
            # for j in range(4):             
                # if j==3:
                    # #smpl_ptsdist = ((sample_coordinates[i][mask[i]][j*tnum:][:,None,:] - smplvert[i][None,:,:])**2).sum(-1) # Pts x V       
                    # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][j*tnum:], smplvert[i], p=2)
                # else:
                    # #smpl_ptsdist = ((sample_coordinates[i][mask[i]][j*tnum:(j+1)*tnum][:,None,:] - smplvert[i][None,:,:])**2).sum(-1) # Pts x V       
                    # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][j*tnum:(j+1)*tnum], smplvert[i], p=2)#deformedpersonsmpl
                # smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
                # tidx = torch.squeeze(smpl_ptsdistmin[1], -1)  # B*P
                # vidx += [tidx]
            # vidx = torch.cat(vidx, 0)
            # nnvidx[i][mask[i]] = vidx
            
        # sample_coordinates = self.invtransform_surreal(sample_coordinates, shift, trans)
        
        # sample_coordinates = self.inversedeforming_samplepoints_LBS(sample_coordinates, nnvidx, smplvert, T)
        
        # mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        # #sample_coordinates[~mask] = sample_coordinates[~mask] + 10
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates.obj')   
        
        # npvertices = self.smpl.v_template.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/canonmesh.obj')
        
        #raw smpl
        # smpl_reduced_current_vert = get_shape(self.smpl, body_pose=smpl_body_pose.view(batch_size,23,3),
                                      # global_orient=smpl_orient[:,None])  
        # smpl_reduced_current_vert *= smplscale
        # smpl_reduced_current_vert += smpl_translate[:,None] / 100
        # smpl_reduced_current = SMPLOutput(vertices=smpl_reduced_current_vert)
        
        # smpl_reduced_canon_vert = get_shape(self.smpl, body_pose=self.smpl_avg_body_pose.expand(batch_size, -1).view(batch_size,23,3),
                                      # global_orient=self.smpl_avg_orient.expand(batch_size, -1)[:,None])
        # smpl_reduced_canon_vert *= smplscale
        # #smpl_reduced_canon_vert += self.smpl_avg_transl.expand(3, -1)[:,None]                                      
        # smpl_reduced_canon = SMPLOutput(vertices=smpl_reduced_canon_vert)self.smpl_reduced.faces_t, smpl_reduced_current.vertices
        
        # rendermask = self.rasterize_eg3d(self.smplfaces, smplvert, extrinsics, intrinsic)
        # for i in range(0,1):
            # mask_render = rendermask[i].detach().cpu().numpy()
            # mask_path = 'test/smplmask{:04d}.png'.format(frameidx[i].item())       
            # cv2.imwrite(mask_path, mask_render * 255)            
            # raw_img = rawimg.permute(0,2,3,1)[i].detach().cpu().numpy()
            # im_path = 'test/rawimg{:04d}.png'.format(frameidx[i].item())       
            # cv2.imwrite(im_path, raw_img)
        #bonemask = bone_mask[0].detach().cpu().numpy()
        #cv2.imwrite('test/bonemask.png', bonemask * 255)
        
        # smpl_reduced_canon_vert = get_shape(self.smpl, body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1).view(batch_size,23,3),
                                      # global_orient=self.smpl_avg_orient.expand(bs_expand, -1)[:,None])
        # smpl_reduced_canon_vert *= self.smpl_avg_scale
        # smpl_reduced_canon_vert += self.smpl_avg_transl.expand(bs_expand, -1)[:,None]                                      
        # smpl_reduced_canon = SMPLOutput(vertices=smpl_reduced_canon_vert)

        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinatesraw.obj')   
        
        # npvertices = smpl_reduced_current.vertices[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/currmesh.obj')
        
        #mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        
        #displacement
        # smplpara = torch.cat([smpl_params, betas], -1)
        # samplepts_disp =  torch.zeros_like(sample_coordinates)   
        # for i in range(smpl_params.shape[0]):
            # samplepts_disp[i][mask[i]] = self.displace(sample_coordinates[i][mask[i]], latentws[i][0], smplpara[i])
            # #print(samplepts_disp[i][mask[i]])
            # #sample_coordinates[i][mask[i]] = sample_coordinates[i][mask[i]]+samplepts_disp[i][mask[i]]
        # #samplepts_disp = self.displace(sample_coordinates, latentws, smplpara)
        samplepts_disp = 0
        
        # sample_coordinates = self.get_canonical_coordinates(
            # sample_coordinates,
            # smpl_src=smpl_reduced_current,
            # smpl_dst=smpl_reduced_canon,  
            # mask = mask            
        # )
        
        # for i in range(smpl_params.shape[0]):
            # sample_coordinates[i][mask[i]] = sample_coordinates[i][mask[i]]+samplepts_disp[i][mask[i]]
               
        #sample_coordinates[~mask] = 10
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates.obj')   
        
        # npvertices = smpl_reduced_canon.vertices[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/canonmesh.obj')
        
        # smvert = self.get_canonical_coordinates(
            # smpl_reduced_current.vertices,
            # smpl_src=smpl_reduced_current,
            # smpl_dst=smpl_reduced_canon          
        # )
        #smvert = smpl_reduced_canon.vertices
        # t = mask_at_box.reshape(-1,128,128)
        # bone_mask = bone_mask*t
        # bonemask = bone_mask[0].detach().cpu().numpy()
        # cv2.imwrite('test/bonemask.png', bonemask * 255)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates_can.obj') 
        
        # R = extrinsics[:,:3,:3] 
        # T = extrinsics[:,:3,3:] 
        # localvert = torch.matmul(R,smpl_reduced_current.vertices.permute(0,2,1))+T
        # projpixel = torch.matmul(intrinsic, localvert)
        # projpixel = projpixel/localvert[:,2,:]
        # print('smpl:',projpixel)
        # globaljoint = pose_to_world[:,:, :3, 3]
        # localvert = torch.matmul(R,globaljoint.permute(0,2,1))+T
        # projpixel = torch.matmul(intrinsic, localvert)
        # projpixel = projpixel/localvert[:,2,:]
        # print('joint:',projpixel)
        
        #mask = mask_at_box.unsqueeze(-1).unsqueeze(-1).repeat(1,1,Nc,1).view(sample_coordinates.shape[0],sample_coordinates.shape[1],1)
               
        # rgb_vals = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],32], dtype=torch.float32)#normlized to [-1,1]
        # sigma_vals = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32) * 1e3
        # #print(sample_coordinates.shape,mask.shape,sample_coordinates[mask].shape)
        #color, density = self.fetchingnerf(planes, sample_coordinates[0][mask[0]][None,...])
        
        #sample_coordinates.requires_grad_(True)
        #color, density = self.fetchingnerf(planes, sample_coordinates)
        
        color_smpl, density_smpl, color_cloth, density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        
        color_smpl = color_smpl.reshape(batch_size, num_rays, samples_per_ray, color_smpl.shape[-1])
        density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)              
        density_smpl = sdf_to_alpha(density_smpl.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s * 1.0))  
        density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        color_cloth = color_cloth.reshape(batch_size, num_rays, samples_per_ray, color_cloth.shape[-1])
        density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)              
        density_cloth = sdf_to_alpha(density_cloth.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s_cloth * 1.0))  
        density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color_smpl, density_smpl, color_cloth, density_cloth, renderclothmask, depths_coarse, batch_size)
        
        # sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros')
        # sampled_features = sampled_features.mean(1)
        # density, color = calc_density_and_color_from_feature(self, sampled_features.permute(0,2,1), z_rend, sample_coordinates)
        
        #density = self.density_activation(density) * 10
        # density = density.permute(0,2,1)
        # color = color.permute(0,2,1)
        
        #density *= meshhumanmask.unsqueeze(-1)
        
        #density *= mask.unsqueeze(-1)
        
        #if 0:
        # canonical sdf regularization smpl_reduced_canon.verticesself.smpl_reduced_canon_vert
        meshpts_smpl, meshpts_cloth, samplepts_smpl, samplepts_cloth, meshpts_normal = self.samplingpoint_learningmeshsdf(self.rawtemplatesmpl.repeat(batch_size,1,1), self.canon_clothvert.repeat(batch_size,1,1))#cloth_reduced_canon
        
        #samplepts.requires_grad_(True)
        #_, samplepts_density = self.fetchingnerf(planes, samplepts)
        # samplepts = torch.cat([samplepts_smpl,samplepts_cloth,meshpts_smpl,meshpts_cloth],dim=1)
        # samplepts.requires_grad_(True)
        
        samplepts_smpl.requires_grad_(True)
        samplepts_cloth.requires_grad_(True)
        _, samplepts_density_smpl, _, samplepts_density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], samplepts_smpl, samplepts_cloth)
        
        offmesh_sdf_smpl = samplepts_density_smpl.reshape(batch_size,-1)
        offmesh_sdf_cloth = samplepts_density_cloth.reshape(batch_size,-1)
        offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)
        
        d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      inputs=samplepts_smpl,
                                       grad_outputs=d_output_smpl,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      inputs=samplepts_cloth,
                                       grad_outputs=d_output_cloth,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)
        
        #_, meshpts_smpl_sdf = self.fetchingnerf(planes, meshpts_smpl)
        #_, meshpts_cloth_sdf = self.fetchingnerf(planes, meshpts_cloth)
        meshpts_smpl.requires_grad_(True)
        meshpts_cloth.requires_grad_(True)
        _, meshpts_smpl_sdf, _, meshpts_cloth_sdf = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], meshpts_smpl, meshpts_cloth)
        meshpts_smpl_sdf = meshpts_smpl_sdf.reshape(batch_size,-1)
        meshpts_cloth_sdf = meshpts_cloth_sdf.reshape(batch_size,-1)
        #print(meshpts_smpl_sdf[0])       
        # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth,meshpts_smpl_sdf,meshpts_cloth_sdf],dim=1)
                
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdf_gradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=samplepts,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
                                       
        # sampleptsnum = samplepts_smpl.shape[1]+samplepts_cloth.shape[1]
        # sdfgradients =  sdf_gradients[:,:sampleptsnum,:]          
        # meshptsgradients =  sdf_gradients[:,sampleptsnum:,:]   
        
        output_smpl = torch.ones_like(meshpts_smpl_sdf, requires_grad=False, device=meshpts_smpl_sdf.device)#
        meshptsgradients_smpl = torch.autograd.grad(outputs=meshpts_smpl_sdf,
                                      inputs=meshpts_smpl,
                                       grad_outputs=output_smpl,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        output_cloth = torch.ones_like(meshpts_cloth_sdf, requires_grad=False, device=meshpts_cloth_sdf.device)#
        meshptsgradients_cloth = torch.autograd.grad(outputs=meshpts_cloth_sdf,
                                      inputs=meshpts_cloth,
                                       grad_outputs=output_cloth,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        meshptsgradients = torch.cat([meshptsgradients_smpl,meshptsgradients_cloth],dim=1)
        #meshptsgradients = 0
        
        # offmesh_sdf = 0
        # sdfgradients = 0
        # meshpts_smpl_sdf = 0
        # meshpts_cloth_sdf = 0
        # meshptsgradients = 0
        # meshpts_normal = 0
        
        # offmesh_sdf = density.reshape(batch_size,-1)
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdfgradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=sample_coordinates,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # meshpts_smpl_sdf = 0                               
                                       
        
        # rgb_vals = color#[mask]
        # sigma_vals = density#[mask]
        
        # colors_coarse = rgb_vals
        # densities_coarse = sigma_vals
        
        # # colors_coarse = out['rgb']
        # # densities_coarse = out['sigma']
        # colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        # densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        
       
        # densities_coarse = sdf_to_alpha(densities_coarse.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s * 1.0))  
        # densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        #print(colors_coarse[densities_coarse.repeat(1,1,1,3)>0])
        # Mask out invalid samples (optional).
        is_sample_valid = None
        # if smpl_clip_depths is not None:
            # is_sample_valid = self.get_sample_mask(sample_depths=depths_coarse, min_max_depths=smpl_clip_depths)
            # densities_coarse = densities_coarse - 1000 * (1-is_sample_valid.float())

        # Fine Pass
        N_importance = 0#64#rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, batch_size)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            if 0:#rendering_options['box_warp_pre_deform']:
                sample_coordinates = (2 / rendering_options['box_warp']) * sample_coordinates
            sample_coordinates = self.get_canonical_coordinates(
                                sample_coordinates,
                                smpl_src=smpl_reduced_current,
                                smpl_dst=smpl_reduced_canon
                                )
            colors_fine, densities_fine = self.fetchingnerf(planes, sample_coordinates)
            # out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            # colors_fine = out['rgb']
            # densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            # Mask out invalid samples (optional).
            if 0:#smpl_clip_depths is not None:
                is_sample_valid = self.get_sample_mask(sample_depths=depths_fine, min_max_depths=smpl_clip_depths)
                densities_fine = densities_fine - 1000 * (1-is_sample_valid.float())
                #colors_fine = colors_fine * is_sample_valid.float()

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(all_colors, all_densities, all_depths, batch_size)
        # else:
            # rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, batch_size)
             
        #print(rgb_final.shape,is_valid.shape)rgb_final, depth_final, weights
        # rgb = rgb_final.reshape(2,64,64,32)
        # print(rgb.shape)
        #print(rgb_final[depth_final.squeeze(-1)>0])
        if is_sample_valid is not None: depth_final = is_sample_valid.any(-2).float()

        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients#rgb_final, depth_final, weights.sum(2)
    
    def doublenerf_rendering_deepcap_fixgeometry_color_chunk(self, frameidx, rawimg, Nc, Nf, z_rend, planes, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):
                        
        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        batch_size = smpl_params.shape[0]
        
        #self.sample_canonshape(batch_size)
        
        #deforming with cloth and smpl
        smpl_current = self.deformingsmpl_LBS(A, Th)
        smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)

        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)#
        cloth_reduced_current = self.cloth_reduced(cloth_current)
        
       
        combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        #combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        
        # npvertices = combinedmeshvert[0].detach().cpu().numpy()                        
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = f"test/combined_{frameidx[0].item():0>4}.obj"
        # mesh.export(result_path)
        
        # data_root = '../../neuralbody-deformation-occupancy-fixnerf-our-generalization/tools/data/magdalena20000-allviews/'
        # smplvert_path = os.path.join(data_root, 'vertices',
                                   # '{}.npy'.format(frameidx[0].item()))
        # smplvert = np.load(smplvert_path).astype(np.float32)
        # smplmesh = trimesh.Trimesh(smplvert, self.smplfaces.detach().cpu().numpy(), process=False)
        # result_path = f"test/smpl_{frameidx[0].item():0>4}.obj"
        # smplmesh.export(result_path)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
                
        with torch.no_grad(): 
            clothmask, meshmask = self.render_clothmask(self.meshface.repeat(batch_size,1,1), combinedmeshvert, extrinsics, intrinsic)
            renderclothmask = clothmask#.unsqueeze(-1).repeat(1,1,1,Nc).view(batch_size,-1,Nc,1)
        # mesh_mask = clothmask[0].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        # mesh_mask = clothmask[1].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[1].item()), mesh_mask*255)
        
        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        ray_start = near
        ray_end = far
        # k_iter = self.ray_sampler.iterations // 1000 * 3        
        # print(self.ray_sampler.max_scale * exp(-k_iter*self.ray_sampler.scale_anneal))
        rays_o_s = []
        rays_d_s = []
        ray_start_s = []
        ray_end_s = []
        renderclothmask_s = []
        for i in range(batch_size):       
            rays_o_i, rays_d_i, ray_start_i, ray_end_i, renderclothmask_i, _, _ = self.ray_sampler(512, 512, ray_origins[i].view(512,512,3), ray_directions[i].view(512,512,3), ray_start[i].view(512,512,1), ray_end[i].view(512,512,1), renderclothmask[i], meshmask[i])
            rays_o_s.append(rays_o_i[None])
            rays_d_s.append(rays_d_i[None])
            ray_start_s.append(ray_start_i[None])
            ray_end_s.append(ray_end_i[None])
            renderclothmask_s.append(renderclothmask_i[None])
        allray_origins = torch.cat(rays_o_s)        
        allray_directions = torch.cat(rays_d_s)
        allray_start = torch.cat(ray_start_s)
        allray_end = torch.cat(ray_end_s)
        allrenderclothmask = torch.cat(renderclothmask_s)
        # renderclothmask0 = renderclothmask[0].reshape(32,32)#low resolution for nerf rendering
        # renderclothmask0 = renderclothmask0.detach().cpu().numpy()
        # cv2.imwrite('test/real_mask0.png', renderclothmask0*255)
        allrenderclothmask = allrenderclothmask.unsqueeze(-1).repeat(1,1,Nc,1)
        # Create stratified depth samples
        
        n_batch, n_pixel = allray_origins.shape[:2]
        chunk = 2048
        rendered_color = []
        rendered_mask = []
        rendered_disparity = []
        weights = []
        for i in range(0, n_pixel, chunk):
            ray_origins = allray_origins[:, i:i + chunk]
            ray_directions = allray_directions[:, i:i + chunk]
            ray_start = allray_start[:, i:i + chunk]
            ray_end = allray_end[:, i:i + chunk]
            renderclothmask = allrenderclothmask[:, i:i + chunk]
            
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

            _, num_rays, samples_per_ray, _ = depths_coarse.shape
            bs_expand = batch_size 
                   
            # Coarse Pass
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       
            
            #mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
            
            
            sample_coordinates_smpl = self.get_canonical_coordinates(
                sample_coordinates,
                smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
                smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)) 
                #mask = mask            
            )

            sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
                sample_coordinates,
                cloth_reduced_current,
                self.cloth_reduced_canon.repeat(batch_size,1,1)
                #mask            
            )
            
            #displacement
            # smplpara = torch.cat([smpl_params, betas], -1)
            # samplepts_disp_smpl = self.displace_smpl(sample_coordinates, smplpara)
            # samplepts_disp_cloth = self.displace_cloth(sample_coordinates, smplpara)
            # samplepts_disp = torch.cat([samplepts_disp_smpl, samplepts_disp_cloth], 1)
            # sample_coordinates_smpl = sample_coordinates_smpl+samplepts_disp_smpl
            # sample_coordinates_cloth = sample_coordinates_cloth+samplepts_disp_cloth
            samplepts_disp = 0
            
            # meshvert_def = torch.cat([self.canonclothvert, self.rawtemplatesmpl], dim=1)#self.deformedcloth
            # npvertices = meshvert_def[0].detach().cpu().numpy()
            # npfaces = self.meshface[0].detach().cpu().numpy()
            # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            # result_path = os.path.join('test/canverts{:04d}.obj'.format(frameidx[0].item()))
            # mesh.export(result_path)
            # npvertices = self.rawtemplatesmpl[0].detach().cpu().numpy()
            # npfaces = self.smplfaces.detach().cpu().numpy()
            # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            # result_path = os.path.join('test/smpl_can{:04d}.obj'.format(frameidx[0].item()))
            # mesh.export(result_path)
            # npvertices = self.canonclothvert[0].detach().cpu().numpy()
            # npfaces = self.clothfaces.detach().cpu().numpy()
            # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            # result_path = os.path.join('test/cloth_can{:04d}.obj'.format(frameidx[0].item()))
            # mesh.export(result_path)
            # npvertices = sample_coordinates_smpl[0].detach().cpu().numpy()                        
            # npfaces = self.smpl.faces
            # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            # result_path = os.path.join('test/samplecan_coordinatessmpl{:04d}.obj'.format(frameidx[0].item()))
            # mesh.export(result_path)
            # npvertices = sample_coordinates_cloth[0].detach().cpu().numpy()                        
            # npfaces = self.smpl.faces
            # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            # result_path = os.path.join('test/samplecan_coordinatescloth{:04d}.obj'.format(frameidx[0].item()))
            # mesh.export(result_path)
            
            
            #print(torch.isnan(sample_coordinates_cloth).any() or torch.isinf(sample_coordinates_cloth).any())
            density_cloth, feature_cloth = self.SDFnetwork_cloth(sample_coordinates_cloth,self.shapepara.repeat(batch_size,1))#tempclothpara     
            
            #density_cloth = torch.nan_to_num(density_cloth, nan=0.0, posinf=0.0, neginf=0.0)
            
            density_smpl, feature_smpl = self.SDFnetwork_smpl(sample_coordinates_smpl,self.canonparamshape.repeat(batch_size,1))
            
            # samplepts_incsdf_smpl = self.incSDFnetwork_smpl(sample_coordinates_smpl)
            # samplepts_incsdf_cloth = self.incSDFnetwork_cloth(sample_coordinates_cloth)
            # #samplepts_disp = torch.cat([samplepts_incsdf_smpl, samplepts_incsdf_cloth], 1)
            # density_smpl = density_smpl+samplepts_incsdf_smpl
            # density_cloth = density_cloth+samplepts_incsdf_cloth
            
            # sample_coordinates_smpl.requires_grad_(True)
            # sample_coordinates_cloth.requires_grad_(True)
            # #samplepts_density_smpl, samplepts_density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], samplepts_smpl, samplepts_cloth)
            # samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
            # samplepts_density_smpl,_ = self.SDFnetwork_smpl(samplepts_smpl,self.canonparamshape)

            # offmesh_sdf_smpl = density_smpl.reshape(batch_size,-1)
            # offmesh_sdf_cloth = density_cloth.reshape(batch_size,-1)
            # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)
            
            # d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
            # sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                          # inputs=sample_coordinates_smpl,
                                           # grad_outputs=d_output_smpl,
                                          # create_graph=True,
                                           # retain_graph=True,
                                           # only_inputs=True)[0]
            # d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
            # sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                          # inputs=sample_coordinates_cloth,
                                           # grad_outputs=d_output_cloth,
                                          # create_graph=True,
                                           # retain_graph=True,
                                           # only_inputs=True)[0]
            # sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)
            
            #density_smpl = torch.nan_to_num(density_smpl, nan=0.0, posinf=0.0, neginf=0.0)
            # density_cloth[density_cloth.abs()>0.5] = 0.5
            # density_smpl[density_smpl.abs()>0.5] = 0.5
            smplpara = torch.cat([smpl_params, betas], -1)
            
            colorlatentsmpl = self.colorlatent_smpl(torch.LongTensor([0]).to('cuda'))
            colorlatentcloth = self.colorlatent_cloth(torch.LongTensor([0]).to('cuda'))
            feature_cloth = torch.cat([feature_cloth,colorlatentcloth[:,None].repeat(feature_cloth.shape[0],feature_cloth.shape[1],1)],dim=-1)
            feature_smpl = torch.cat([feature_smpl,colorlatentsmpl[:,None].repeat(feature_smpl.shape[0],feature_smpl.shape[1],1)],dim=-1)
            color_cloth = self.decoder_cloth(sample_coordinates_cloth,z_rend)#feature_cloth
            color_smpl = self.decoder(sample_coordinates_smpl,z_rend)#feature_smpl
                
            # density_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32)#normlized to [-1,1]
            # density_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32) * 1e3
            # color_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32)#normlized to [-1,1]
            # color_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32) * 1e3
            # density_cloth[mask] = density_cloth0
            # density_smpl[mask] = density_smpl0
            # color_cloth[mask] = color_cloth0
            # color_smpl[mask] = color_smpl0
            
            #color_smpl, color_cloth = self.fetchingdoublenerf_fixgeometry_color(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
            #color_smpl, density_smpl, color_cloth, density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
            
            color_smpl = color_smpl.reshape(batch_size, num_rays, samples_per_ray, color_smpl.shape[-1])
            density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)              
            #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
            density_smpl = sdf_to_alpha2(density_smpl.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #
            density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)
            #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
            #density_smplmesh, _ = self.SDFnetwork_smpl(self.smpl_reduced_canon_vert.repeat(batch_size,1,1),self.canonparamshape.repeat(batch_size,1))

            #density_smplmesh=sdf_to_alpha2(density_smplmesh, 0.05)
            #print(torch.exp(self.ln_s * 1.0),density_smplmesh[0])
            
            color_cloth = color_cloth.reshape(batch_size, num_rays, samples_per_ray, color_cloth.shape[-1])
            density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)  
            #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())        
            density_cloth = sdf_to_alpha2(density_cloth.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s_cloth))  #
            density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)
            #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())  

            rendered_color_chunk, rendered_mask_chunk, rendered_disparity_chunk, weights_chunk = self.ray_marcher(color_smpl, density_smpl, color_cloth, density_cloth, renderclothmask, depths_coarse, batch_size)
            
            rendered_color.append(rendered_color_chunk)
            rendered_mask.append(rendered_mask_chunk)
            rendered_disparity.append(rendered_disparity_chunk)
            weights.append(weights_chunk)
            
            #if 0:    
            offmesh_sdf = 0
            sdfgradients = 0
            meshpts_smpl_sdf = 0
            meshpts_cloth_sdf = 0
            meshptsgradients = 0
            meshpts_normal = 0
        
        rendered_color = torch.cat(rendered_color, dim=1)
        rendered_mask = torch.cat(rendered_mask, dim=1)
        rendered_disparity = torch.cat(rendered_disparity, dim=1)
        weights = torch.cat(weights, dim=1)
        
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients#rgb_final, depth_final, weights.sum(2)
    
    def doublenerf_rendering_deepcap_fixgeometry_color(self, frameidx, rawimg, Nc, Nf, z_rend, planes, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):
        
        # R = extrinsics[:,:3,:3] 
        # T = extrinsics[:,:3,3:]

        # pose_to_camera = torch.bmm(pose_to_world[..., :3, 3], R.permute(0,2,1)) + T.permute(0,2,1)
        # pose_2d = torch.matmul(pose_to_camera, intrinsic.permute(0,2,1))
        # pose_2d = pose_2d[:, :, :2] / pose_2d[:, :, 2][...,None] 
        # location = pose_2d[:, 15, :2]
        
        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        batch_size = smpl_params.shape[0]
        
        #self.sample_canonshape(batch_size)
        
        #deforming with cloth and smpl
        smpl_current = self.deformingsmpl_LBS(A, Th)
        smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)

        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)#
        cloth_reduced_current = self.cloth_reduced(cloth_current)
        
       
        combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        #combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        
        # npvertices = combinedmeshvert[0].detach().cpu().numpy()                        
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = f"test/combined_{frameidx[0].item():0>4}.obj"
        # mesh.export(result_path)
        
        # data_root = '../../neuralbody-deformation-occupancy-fixnerf-our-generalization/tools/data/magdalena20000-allviews/'
        # smplvert_path = os.path.join(data_root, 'vertices',
                                   # '{}.npy'.format(frameidx[0].item()))
        # smplvert = np.load(smplvert_path).astype(np.float32)
        # smplmesh = trimesh.Trimesh(smplvert, self.smplfaces.detach().cpu().numpy(), process=False)
        # result_path = f"test/smpl_{frameidx[0].item():0>4}.obj"
        # smplmesh.export(result_path)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
                
        with torch.no_grad(): 
            clothmask, meshmask = self.render_clothmask(self.meshface.repeat(batch_size,1,1), combinedmeshvert, extrinsics, intrinsic)
            renderclothmask = clothmask#.unsqueeze(-1).repeat(1,1,1,Nc).view(batch_size,-1,Nc,1)
        # mesh_mask = clothmask[0].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        # mesh_mask = clothmask[1].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[1].item()), mesh_mask*255)
        
        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        ray_start = near
        ray_end = far
        # k_iter = self.ray_sampler.iterations // 1000 * 3        
        # print(self.ray_sampler.max_scale * exp(-k_iter*self.ray_sampler.scale_anneal))
        rays_o_s = []
        rays_d_s = []
        ray_start_s = []
        ray_end_s = []
        renderclothmask_s = []
        for i in range(batch_size):       
            rays_o_i, rays_d_i, ray_start_i, ray_end_i, renderclothmask_i, _, _ = self.ray_sampler(512, 512, ray_origins[i].view(512,512,3), ray_directions[i].view(512,512,3), ray_start[i].view(512,512,1), ray_end[i].view(512,512,1), renderclothmask[i], meshmask[i])#,location[i]
            rays_o_s.append(rays_o_i[None])
            rays_d_s.append(rays_d_i[None])
            ray_start_s.append(ray_start_i[None])
            ray_end_s.append(ray_end_i[None])
            renderclothmask_s.append(renderclothmask_i[None])
        ray_origins = torch.cat(rays_o_s)        
        ray_directions = torch.cat(rays_d_s)
        ray_start = torch.cat(ray_start_s)
        ray_end = torch.cat(ray_end_s)
        renderclothmask = torch.cat(renderclothmask_s)
        # renderclothmask0 = renderclothmask[0].reshape(32,32)#low resolution for nerf rendering
        # renderclothmask0 = renderclothmask0.detach().cpu().numpy()
        # cv2.imwrite('test/real_mask0.png', renderclothmask0*255)
        renderclothmask = renderclothmask.unsqueeze(-1).repeat(1,1,Nc,1)
        # Create stratified depth samples
        depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

        _, num_rays, samples_per_ray, _ = depths_coarse.shape
        bs_expand = batch_size 
               
        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       
        
        #mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        
        
        sample_coordinates_smpl = self.get_canonical_coordinates(
            sample_coordinates,
            smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)) 
            #mask = mask            
        )

        sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
            sample_coordinates,
            cloth_reduced_current,
            self.cloth_reduced_canon.repeat(batch_size,1,1)
            #mask            
        )
        
        #displacement
        # smplpara = torch.cat([smpl_params, betas], -1)
        # samplepts_disp_smpl = self.displace_smpl(sample_coordinates, smplpara)
        # samplepts_disp_cloth = self.displace_cloth(sample_coordinates, smplpara)
        # samplepts_disp = torch.cat([samplepts_disp_smpl, samplepts_disp_cloth], 1)
        # sample_coordinates_smpl = sample_coordinates_smpl+samplepts_disp_smpl
        # sample_coordinates_cloth = sample_coordinates_cloth+samplepts_disp_cloth
        samplepts_disp = 0
        
        # meshvert_def = torch.cat([self.canonclothvert, self.rawtemplatesmpl], dim=1)#self.deformedcloth
        # npvertices = meshvert_def[0].detach().cpu().numpy()
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/canverts{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = self.rawtemplatesmpl[0].detach().cpu().numpy()
        # npfaces = self.smplfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/smpl_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = self.canonclothvert[0].detach().cpu().numpy()
        # npfaces = self.clothfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/cloth_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = sample_coordinates_smpl[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/samplecan_coordinatessmpl{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = sample_coordinates_cloth[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/samplecan_coordinatescloth{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        
        sample_coordinates_smpl.requires_grad_(True)
        sample_coordinates_cloth.requires_grad_(True)
        #print(torch.isnan(sample_coordinates_cloth).any() or torch.isinf(sample_coordinates_cloth).any())
        density_cloth, feature_cloth = self.SDFnetwork_cloth(sample_coordinates_cloth,self.shapepara.repeat(batch_size,1))#tempclothpara     
        
        #density_cloth = torch.nan_to_num(density_cloth, nan=0.0, posinf=0.0, neginf=0.0)
        
        density_smpl, feature_smpl = self.SDFnetwork_smpl(sample_coordinates_smpl,self.canonparamshape.repeat(batch_size,1))
        
     
        # samplepts_incsdf_smpl = self.incSDFnetwork_smpl(sample_coordinates_smpl)
        # samplepts_incsdf_cloth = self.incSDFnetwork_cloth(sample_coordinates_cloth)
        # samplepts_disp = torch.cat([samplepts_incsdf_smpl, samplepts_incsdf_cloth], 1)#samplepts_incsdf_smpl
        # density_smpl = density_smpl+samplepts_incsdf_smpl
        # density_cloth = density_cloth+samplepts_incsdf_cloth
        samplepts_disp = 0
        
        # sample_coordinates_cloth.requires_grad_(True)
        # #samplepts_density_smpl, samplepts_density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], samplepts_smpl, samplepts_cloth)
        # samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
        # samplepts_density_smpl,_ = self.SDFnetwork_smpl(samplepts_smpl,self.canonparamshape)

        # offmesh_sdf_smpl = samplepts_incsdf_smpl.reshape(batch_size,-1)#density_smpl
        # offmesh_sdf_cloth = samplepts_incsdf_cloth.reshape(batch_size,-1)#density_cloth
        # #offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)#offmesh_sdf_smpl
        # offmesh_sdf = torch.cat([density_smpl.reshape(batch_size,-1),density_cloth.reshape(batch_size,-1)],dim=1)
        #offmesh_sdf = 0
        
        offmesh_sdf_smpl = density_smpl.reshape(batch_size,-1)#density_smpl
        offmesh_sdf_cloth = density_cloth.reshape(batch_size,-1)#density_cloth
        offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)#offmesh_sdf_smpl
        
        d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      inputs=sample_coordinates_smpl,
                                       grad_outputs=d_output_smpl,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      inputs=sample_coordinates_cloth,
                                       grad_outputs=d_output_cloth,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)#sdfgradients_smpl
        #sdfgradients = 0
        
        #density_smpl = torch.nan_to_num(density_smpl, nan=0.0, posinf=0.0, neginf=0.0)
        # density_cloth[density_cloth.abs()>0.5] = 0.5
        # density_smpl[density_smpl.abs()>0.5] = 0.5
        
        smplpara = torch.cat([smpl_params, betas], -1)
        
        # colorlatentsmpl = self.colorlatent_smpl(torch.LongTensor([0]).to('cuda'))
        # colorlatentcloth = self.colorlatent_cloth(torch.LongTensor([0]).to('cuda'))
        # feature_cloth = torch.cat([feature_cloth,colorlatentcloth[:,None].repeat(feature_cloth.shape[0],feature_cloth.shape[1],1)],dim=-1)
        # feature_smpl = torch.cat([feature_smpl,colorlatentsmpl[:,None].repeat(feature_smpl.shape[0],feature_smpl.shape[1],1)],dim=-1)
        color_cloth = self.decoder_cloth(sample_coordinates_cloth, z_rend)#feature_cloth
        color_smpl = self.decoder(sample_coordinates_smpl, z_rend)#feature_smpl
            
        # density_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32)#normlized to [-1,1]
        # density_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32) * 1e3
        # color_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32)#normlized to [-1,1]
        # color_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32) * 1e3
        # density_cloth[mask] = density_cloth0
        # density_smpl[mask] = density_smpl0
        # color_cloth[mask] = color_cloth0
        # color_smpl[mask] = color_smpl0
        
        #color_smpl, color_cloth = self.fetchingdoublenerf_fixgeometry_color(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        #color_smpl, density_smpl, color_cloth, density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        
        color_smpl = color_smpl.reshape(batch_size, num_rays, samples_per_ray, color_smpl.shape[-1])
        density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)              
        #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())

        density_smpl = sdf_to_alpha2(density_smpl.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #
        density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)
        #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
        #density_smplmesh, _ = self.SDFnetwork_smpl(self.smpl_reduced_canon_vert.repeat(batch_size,1,1),self.canonparamshape.repeat(batch_size,1))

        #density_smplmesh=sdf_to_alpha2(density_smplmesh, 0.05)
        #print(torch.exp(self.ln_s * 1.0),density_smplmesh[0])
        
        color_cloth = color_cloth.reshape(batch_size, num_rays, samples_per_ray, color_cloth.shape[-1])
        density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)  
        #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())        
        density_cloth = sdf_to_alpha2(density_cloth.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s_cloth))  #
        density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)
        #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())  

        rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color_smpl, density_smpl, color_cloth, density_cloth, renderclothmask, depths_coarse, batch_size)
        
        #if 0:    
        #offmesh_sdf = 0
        #sdfgradients = 0
        meshpts_smpl_sdf = 0
        meshpts_cloth_sdf = 0
        meshptsgradients = 0
        meshpts_normal = 0
        
        
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients#rgb_final, depth_final, weights.sum(2)
    
    def doublenerf_rendering_deepcap_geometry(self, frameidx, rawimg, Nc, Nf, z_rend, planes, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):

        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]

        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        
        batch_size = smpl_params.shape[0]
        
        #self.sample_canonshape(batch_size)
        
        #deforming with cloth and smpl
        #smpl_current = self.deformingsmpl_LBS(A, Th)
        #smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)
        
        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        #cloth_current = self.deformingcloth_LBS(self.canonclothvert, A, Th)#.repeat(batch_size,1,1)
        #cloth_reduced_current = self.cloth_reduced(cloth_current)
        
        #if 0:
        # canonical sdf regularization smpl_reduced_canon.verticesself.smpl_reduced_canon_vert
        with torch.no_grad(): 
            meshpts_smpl, meshpts_cloth, samplepts_smpl, samplepts_cloth, meshpts_normal = self.samplingpoint_learningmeshsdf(self.rawtemplatesmpl, self.canon_clothvert)#.repeat(batch_size,1,1)cloth_reduced_canon
        
        #samplepts.requires_grad_(True)
        #_, samplepts_density = self.fetchingnerf(planes, samplepts)
        # samplepts = torch.cat([samplepts_smpl,samplepts_cloth,meshpts_smpl,meshpts_cloth],dim=1)
        # samplepts.requires_grad_(True)
        
        # self.embed_fn_fine = embed_fn
        # # Deform-Net
        # self.sdf_network_cloth=modules.SingleBVPNet(type='relu',mode='mlp', hidden_features=256, num_hidden_layers=3, in_features=72,out_features=1)
        # # Hyper-Net
        # self.hyper_net_cloth = HyperNetwork(hyper_in_features=2, hyper_hidden_layers=3, hyper_hidden_features=128,hypo_module=self.sdf_network_cloth)
        # # Deform-Net
        # self.sdf_network_smpl=modules.SingleBVPNet(type='relu',mode='mlp', hidden_features=256, num_hidden_layers=3, in_features=72,out_features=1)
        # # Hyper-Net
        # self.hyper_net_smpl = HyperNetwork(hyper_in_features=10, hyper_hidden_layers=3, hyper_hidden_features=128,hypo_module=self.sdf_network_smpl)
                
        samplepts_smpl.requires_grad_(True)
        samplepts_cloth.requires_grad_(True)
        #samplepts_density_smpl, samplepts_density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], samplepts_smpl, samplepts_cloth)
        samplepts_density_cloth = self.Densitynetwork_cloth(samplepts_cloth)#,self.shapeparatempclothpara
        samplepts_density_smpl = self.Densitynetwork_smpl(samplepts_smpl)#,self.canonparamshape

        # hypo_params_cloth = self.hyper_net_cloth(self.tempclothpara)
        # embed_samplepts_cloth = self.embed_fn_fine(samplepts_cloth)
        # print(embed_samplepts_cloth.shape)
        # samplepts_density_cloth = self.sdf_network_cloth(embed_samplepts_cloth, params=hypo_params_cloth)
        
        # hypo_params_smpl = self.hyper_net_smpl(self.canonparamshape)
        # embed_samplepts_smpl = self.embed_fn_fine(samplepts_smpl)
        # samplepts_density_smpl = self.sdf_network_cloth(embed_samplepts_smpl, params=hypo_params_smpl)

        offmesh_sdf_smpl = samplepts_density_smpl.reshape(batch_size,-1)
        offmesh_sdf_cloth = samplepts_density_cloth.reshape(batch_size,-1)
        offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)
        #print(offmesh_sdf_cloth[0][offmesh_sdf_cloth[0]<0])
        
        d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      inputs=samplepts_smpl,
                                       grad_outputs=d_output_smpl,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      inputs=samplepts_cloth,
                                       grad_outputs=d_output_cloth,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)
        
        #_, meshpts_smpl_sdf = self.fetchingnerf(planes, meshpts_smpl)
        #_, meshpts_cloth_sdf = self.fetchingnerf(planes, meshpts_cloth)
        meshpts_smpl.requires_grad_(True)
        meshpts_cloth.requires_grad_(True)
        #meshpts_smpl_sdf, meshpts_cloth_sdf = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], meshpts_smpl, meshpts_cloth)
        meshpts_cloth_sdf = self.Densitynetwork_cloth(meshpts_cloth)#,self.shapepara,self.canonparamshape
        meshpts_smpl_sdf = self.Densitynetwork_smpl(meshpts_smpl)
        
        # embed_meshpts_cloth = self.embed_fn_fine(meshpts_cloth_sdf)
        # meshpts_cloth_sdf = self.sdf_network_cloth(embed_meshpts_cloth, params=hypo_params_cloth)
        
        # embed_meshpts_smpl = self.embed_fn_fine(meshpts_smpl_sdf)
        # meshpts_smpl_sdf = self.sdf_network_cloth(embed_meshpts_smpl, params=hypo_params_smpl)


        meshpts_smpl_sdf = meshpts_smpl_sdf.reshape(batch_size,-1)
        meshpts_cloth_sdf = meshpts_cloth_sdf.reshape(batch_size,-1)
        #print(meshpts_smpl_sdf[0])       
        # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth,meshpts_smpl_sdf,meshpts_cloth_sdf],dim=1)
                
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdf_gradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=samplepts,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
                                       
        # sampleptsnum = samplepts_smpl.shape[1]+samplepts_cloth.shape[1]
        # sdfgradients =  sdf_gradients[:,:sampleptsnum,:]          
        # meshptsgradients =  sdf_gradients[:,sampleptsnum:,:]   
        
        output_smpl = torch.ones_like(meshpts_smpl_sdf, requires_grad=False, device=meshpts_smpl_sdf.device)#
        meshptsgradients_smpl = torch.autograd.grad(outputs=meshpts_smpl_sdf,
                                      inputs=meshpts_smpl,
                                       grad_outputs=output_smpl,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        output_cloth = torch.ones_like(meshpts_cloth_sdf, requires_grad=False, device=meshpts_cloth_sdf.device)#
        meshptsgradients_cloth = torch.autograd.grad(outputs=meshpts_cloth_sdf,
                                      inputs=meshpts_cloth,
                                       grad_outputs=output_cloth,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        meshptsgradients = torch.cat([meshptsgradients_smpl,meshptsgradients_cloth],dim=1)
        #meshptsgradients = 0
        
        # offmesh_sdf = 0
        # sdfgradients = 0
        # meshpts_smpl_sdf = 0
        # meshpts_cloth_sdf = 0
        # meshptsgradients = 0
        # meshpts_normal = 0
        
        rendered_color = 0
        rendered_mask = 0
        rendered_disparity = 0
        samplepts_disp = 0
       
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients#rgb_final, depth_final, weights.sum(2)
    
    def singlenerf_rendering_deepfashion_geometry_smpl(self, frameidx, rawimg, Nc, Nf, z_rend, planes, latentws, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):

        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]

        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        
        batch_size = smpl_params.shape[0]
        
        self.sample_canonshape_smpl(batch_size)
               
        with torch.no_grad(): 
            meshpts_smpl, samplepts_smpl, meshpts_normal = self.samplingpoint_learningmeshsdf_smpl(self.rawtemplatesmpl)#.repeat(batch_size,1,1)cloth_reduced_canon
        
        # #deforming with cloth and smpl
        # smpl_current = self.deformingsmpl_LBS(A, Th)
        # smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)
        
        # # self.predicting_deformation(latentws)
        # # canondeformedcloth = self.deformingtemplate()
        # # deform_smoothloss = self.deformationsmoothloss()
        
        # cloth_current = self.deformingcloth_LBS(self.canonclothvert, A, Th)#.repeat(batch_size,1,1)
        # cloth_reduced_current = self.cloth_reduced(cloth_current)
        
        #if 0:
        # canonical sdf regularization smpl_reduced_canon.verticesself.smpl_reduced_canon_vert
        # with torch.no_grad(): 
            # meshpts_smpl, meshpts_cloth, samplepts_smpl, samplepts_cloth, meshpts_normal = self.samplingpoint_learningmeshsdf(self.rawtemplatesmpl, self.canon_clothvert)#.repeat(batch_size,1,1)cloth_reduced_canon
        
        #samplepts.requires_grad_(True)
        #_, samplepts_density = self.fetchingnerf(planes, samplepts)
        # samplepts = torch.cat([samplepts_smpl,samplepts_cloth,meshpts_smpl,meshpts_cloth],dim=1)
        # samplepts.requires_grad_(True)
        
        # self.embed_fn_fine = embed_fn
        # # Deform-Net
        # self.sdf_network_cloth=modules.SingleBVPNet(type='relu',mode='mlp', hidden_features=256, num_hidden_layers=3, in_features=72,out_features=1)
        # # Hyper-Net
        # self.hyper_net_cloth = HyperNetwork(hyper_in_features=2, hyper_hidden_layers=3, hyper_hidden_features=128,hypo_module=self.sdf_network_cloth)
        # # Deform-Net
        # self.sdf_network_smpl=modules.SingleBVPNet(type='relu',mode='mlp', hidden_features=256, num_hidden_layers=3, in_features=72,out_features=1)
        # # Hyper-Net
        # self.hyper_net_smpl = HyperNetwork(hyper_in_features=10, hyper_hidden_layers=3, hyper_hidden_features=128,hypo_module=self.sdf_network_smpl)
                
        samplepts_smpl.requires_grad_(True)
        #samplepts_cloth.requires_grad_(True)
        #samplepts_density_smpl, samplepts_density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], samplepts_smpl, samplepts_cloth)
        #samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
        samplepts_density_smpl,_ = self.SDFnetwork_smpl(samplepts_smpl,self.canonparamshape)

        # hypo_params_cloth = self.hyper_net_cloth(self.tempclothpara)
        # embed_samplepts_cloth = self.embed_fn_fine(samplepts_cloth)
        # print(embed_samplepts_cloth.shape)
        # samplepts_density_cloth = self.sdf_network_cloth(embed_samplepts_cloth, params=hypo_params_cloth)
        
        # hypo_params_smpl = self.hyper_net_smpl(self.canonparamshape)
        # embed_samplepts_smpl = self.embed_fn_fine(samplepts_smpl)
        # samplepts_density_smpl = self.sdf_network_cloth(embed_samplepts_smpl, params=hypo_params_smpl)

        offmesh_sdf_smpl = samplepts_density_smpl.reshape(batch_size,-1)
        #offmesh_sdf_cloth = samplepts_density_cloth.reshape(batch_size,-1)
        offmesh_sdf = offmesh_sdf_smpl#torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)
        #print(offmesh_sdf_cloth[0][offmesh_sdf_cloth[0]<0])
        
        d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      inputs=samplepts_smpl,
                                       grad_outputs=d_output_smpl,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        # d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        # sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      # inputs=samplepts_cloth,
                                       # grad_outputs=d_output_cloth,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        sdfgradients = sdfgradients_smpl#torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)
        
        #_, meshpts_smpl_sdf = self.fetchingnerf(planes, meshpts_smpl)
        #_, meshpts_cloth_sdf = self.fetchingnerf(planes, meshpts_cloth)
        meshpts_smpl.requires_grad_(True)
        #meshpts_cloth.requires_grad_(True)
        #meshpts_smpl_sdf, meshpts_cloth_sdf = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], meshpts_smpl, meshpts_cloth)
        #meshpts_cloth_sdf,_ = self.SDFnetwork_cloth(meshpts_cloth,self.shapepara)
        meshpts_smpl_sdf,_ = self.SDFnetwork_smpl(meshpts_smpl,self.canonparamshape)
        
        # embed_meshpts_cloth = self.embed_fn_fine(meshpts_cloth_sdf)
        # meshpts_cloth_sdf = self.sdf_network_cloth(embed_meshpts_cloth, params=hypo_params_cloth)
        
        # embed_meshpts_smpl = self.embed_fn_fine(meshpts_smpl_sdf)
        # meshpts_smpl_sdf = self.sdf_network_cloth(embed_meshpts_smpl, params=hypo_params_smpl)


        meshpts_smpl_sdf = meshpts_smpl_sdf.reshape(batch_size,-1)
        #meshpts_cloth_sdf = meshpts_cloth_sdf.reshape(batch_size,-1)
        #print(meshpts_smpl_sdf[0])       
        # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth,meshpts_smpl_sdf,meshpts_cloth_sdf],dim=1)
                
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdf_gradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=samplepts,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
                                       
        # sampleptsnum = samplepts_smpl.shape[1]+samplepts_cloth.shape[1]
        # sdfgradients =  sdf_gradients[:,:sampleptsnum,:]          
        # meshptsgradients =  sdf_gradients[:,sampleptsnum:,:]   
        
        output_smpl = torch.ones_like(meshpts_smpl_sdf, requires_grad=False, device=meshpts_smpl_sdf.device)#
        meshptsgradients_smpl = torch.autograd.grad(outputs=meshpts_smpl_sdf,
                                      inputs=meshpts_smpl,
                                       grad_outputs=output_smpl,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        # output_cloth = torch.ones_like(meshpts_cloth_sdf, requires_grad=False, device=meshpts_cloth_sdf.device)#
        # meshptsgradients_cloth = torch.autograd.grad(outputs=meshpts_cloth_sdf,
                                      # inputs=meshpts_cloth,
                                       # grad_outputs=output_cloth,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        meshptsgradients = meshptsgradients_smpl#torch.cat([meshptsgradients_smpl,meshptsgradients_cloth],dim=1)
        #meshptsgradients = 0
        
        # offmesh_sdf = 0
        # sdfgradients = 0
        # meshpts_smpl_sdf = 0
        # meshpts_cloth_sdf = 0
        # meshptsgradients = 0
        # meshpts_normal = 0
        
        rendered_color = 0
        rendered_mask = 0
        rendered_disparity = 0
        samplepts_disp = 0
       
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients#rgb_final, depth_final, weights.sum(2)
    
    def singlenerf_rendering_deepfashion_geometry(self, frameidx, cropimg, Nc, Nf, z_rend, planes, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):
        
        batch_size = smpl_params.shape[0]           
        with torch.no_grad(): 
            meshpts_smpl, samplepts_smpl, meshpts_normal = self.samplingpoint_learningmeshsdf_smpl(self.rawtemplatesmpl)#.repeat(batch_size,1,1)cloth_reduced_canon
        
        #samplepts.requires_grad_(True)
        #_, samplepts_density = self.fetchingnerf(planes, samplepts)
        # samplepts = torch.cat([samplepts_smpl,samplepts_cloth,meshpts_smpl,meshpts_cloth],dim=1)
        # samplepts.requires_grad_(True)
        
        samplepts_smpl.requires_grad_(True)
        #samplepts_cloth.requires_grad_(True)
        
        #samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
        samplepts_density_smpl = self.Densitynetwork(samplepts_smpl)

        # hypo_params_cloth = self.hyper_net_cloth(self.tempclothpara)
        # embed_samplepts_cloth = self.embed_fn_fine(samplepts_cloth)
        # print(embed_samplepts_cloth.shape)
        # samplepts_density_cloth = self.sdf_network_cloth(embed_samplepts_cloth, params=hypo_params_cloth)
        
        # hypo_params_smpl = self.hyper_net_smpl(self.canonparamshape)
        # embed_samplepts_smpl = self.embed_fn_fine(samplepts_smpl)
        # samplepts_density_smpl = self.sdf_network_cloth(embed_samplepts_smpl, params=hypo_params_smpl)

        offmesh_sdf_smpl = samplepts_density_smpl.reshape(batch_size,-1)
        #offmesh_sdf_cloth = samplepts_density_cloth.reshape(batch_size,-1)
        offmesh_sdf = offmesh_sdf_smpl#torch.cat([,offmesh_sdf_cloth],dim=1)

        d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      inputs=samplepts_smpl,
                                       grad_outputs=d_output_smpl,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        # # d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        # # sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      # # inputs=samplepts_cloth,
                                       # # grad_outputs=d_output_cloth,
                                      # # create_graph=True,
                                       # # retain_graph=True,
                                       # # only_inputs=True)[0]
        # # sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)
        sdfgradients = sdfgradients_smpl
        
        #_, meshpts_smpl_sdf = self.fetchingnerf(planes, meshpts_smpl)
        #_, meshpts_cloth_sdf = self.fetchingnerf(planes, meshpts_cloth)
        meshpts_smpl.requires_grad_(True)
        #meshpts_cloth.requires_grad_(True)
        #meshpts_smpl_sdf, meshpts_cloth_sdf = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], meshpts_smpl, meshpts_cloth)
        #meshpts_cloth_sdf = self.Densitynetwork(meshpts_cloth)
        meshpts_smpl_sdf = self.Densitynetwork(meshpts_smpl)
        
        # embed_meshpts_cloth = self.embed_fn_fine(meshpts_cloth_sdf)
        # meshpts_cloth_sdf = self.sdf_network_cloth(embed_meshpts_cloth, params=hypo_params_cloth)
        
        # embed_meshpts_smpl = self.embed_fn_fine(meshpts_smpl_sdf)
        # meshpts_smpl_sdf = self.sdf_network_cloth(embed_meshpts_smpl, params=hypo_params_smpl)


        meshpts_smpl_sdf = meshpts_smpl_sdf.reshape(batch_size,-1)
        #meshpts_cloth_sdf = meshpts_cloth_sdf.reshape(batch_size,-1)
        #print(meshpts_smpl_sdf[0])       
        # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth,meshpts_smpl_sdf,meshpts_cloth_sdf],dim=1)
                
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdf_gradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=samplepts,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
                                       
        # sampleptsnum = samplepts_smpl.shape[1]+samplepts_cloth.shape[1]
        # sdfgradients =  sdf_gradients[:,:sampleptsnum,:]          
        # meshptsgradients =  sdf_gradients[:,sampleptsnum:,:]   
        
        output_smpl = torch.ones_like(meshpts_smpl_sdf, requires_grad=False, device=meshpts_smpl_sdf.device)#
        meshptsgradients_smpl = torch.autograd.grad(outputs=meshpts_smpl_sdf,
                                      inputs=meshpts_smpl,
                                       grad_outputs=output_smpl,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        # output_cloth = torch.ones_like(meshpts_cloth_sdf, requires_grad=False, device=meshpts_cloth_sdf.device)#
        # meshptsgradients_cloth = torch.autograd.grad(outputs=meshpts_cloth_sdf,
                                      # inputs=meshpts_cloth,
                                       # grad_outputs=output_cloth,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        meshptsgradients = meshptsgradients_smpl#torch.cat([meshptsgradients_smpl,meshptsgradients_cloth],dim=1)
        #meshptsgradients = 0
        
        #if 0:    
        #offmesh_sdf = 0
        #sdfgradients = 0
        #meshpts_smpl_sdf = 0
        meshpts_cloth_sdf = 0
        #meshptsgradients = 0
        #meshpts_normal = 0
        
        rendered_color = 0
        rendered_mask = 0
        rendered_disparity = 0
        samplepts_disp = 0
        
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients#rgb_final, depth_final, weights.sum(2)
    
    def singlenerf_rendering_deepcap_geometry_color_chunk(self, frameidx, cropimg, Nc, Nf, z_rend, planes, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):
                        
        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        batch_size = smpl_params.shape[0]
        
        #self.sample_canonshape(batch_size)
        
        #deforming with cloth and smpl
        smpl_current = self.deformingsmpl_LBS(A, Th)
        smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)

        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)#
        cloth_reduced_current = self.cloth_reduced(cloth_current)
        
       
        combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        #combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        
        # npvertices = combinedmeshvert[0].detach().cpu().numpy()                        
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = f"test/combined_{frameidx[0].item():0>4}.obj"
        # mesh.export(result_path)
        
        # data_root = '../../neuralbody-deformation-occupancy-fixnerf-our-generalization/tools/data/magdalena20000-allviews/'
        # smplvert_path = os.path.join(data_root, 'vertices',
                                   # '{}.npy'.format(frameidx[0].item()))
        # smplvert = np.load(smplvert_path).astype(np.float32)
        # smplmesh = trimesh.Trimesh(smplvert, self.smplfaces.detach().cpu().numpy(), process=False)
        # result_path = f"test/smpl_{frameidx[0].item():0>4}.obj"
        # smplmesh.export(result_path)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
                
        # with torch.no_grad(): 
            # clothmask, meshmask = self.render_clothmask(self.meshface.repeat(batch_size,1,1), combinedmeshvert, extrinsics, intrinsic)
            # renderclothmask = clothmask#.unsqueeze(-1).repeat(1,1,1,Nc).view(batch_size,-1,Nc,1)
        # mesh_mask = clothmask[0].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        # mesh_mask = clothmask[1].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[1].item()), mesh_mask*255)
        renderclothmask = 0
        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        ray_start = near
        ray_end = far
        # k_iter = self.ray_sampler.iterations // 1000 * 3        
        # print(self.ray_sampler.max_scale * exp(-k_iter*self.ray_sampler.scale_anneal))
        # rays_o_s = []
        # rays_d_s = []
        # ray_start_s = []
        # ray_end_s = []
        # #renderclothmask_s = []renderclothmask_i,
        # for i in range(batch_size):       
            # rays_o_i, rays_d_i, ray_start_i, ray_end_i, _, _ = self.ray_sampler(512, 512, ray_origins[i].view(512,512,3), ray_directions[i].view(512,512,3), ray_start[i].view(512,512,1), ray_end[i].view(512,512,1), cropimg[i])#renderclothmask[i], meshmask[i]
            # rays_o_s.append(rays_o_i[None])
            # rays_d_s.append(rays_d_i[None])
            # ray_start_s.append(ray_start_i[None])
            # ray_end_s.append(ray_end_i[None])
            # #renderclothmask_s.append(renderclothmask_i[None])
        # allray_origins = torch.cat(rays_o_s)        
        # allray_directions = torch.cat(rays_d_s)
        # allray_start = torch.cat(ray_start_s)
        # allray_end = torch.cat(ray_end_s)
        # allrenderclothmask = torch.cat(renderclothmask_s)
        
        # renderclothmask0 = renderclothmask[0].reshape(32,32)#low resolution for nerf rendering
        # renderclothmask0 = renderclothmask0.detach().cpu().numpy()
        # cv2.imwrite('test/real_mask0.png', renderclothmask0*255)
        #allrenderclothmask = allrenderclothmask.unsqueeze(-1).repeat(1,1,Nc,1)
        
        allray_origins = ray_origins       
        allray_directions = ray_directions
        allray_start = ray_start
        allray_end = ray_end
        #allrenderclothmask = renderclothmask
        
        # Create stratified depth samples
        
        n_batch, n_pixel = allray_origins.shape[:2]
        chunk = 2048
        rendered_color = []
        rendered_mask = []
        rendered_disparity = []
        weights = []
        for i in range(0, n_pixel, chunk):
            ray_origins = allray_origins[:, i:i + chunk]
            ray_directions = allray_directions[:, i:i + chunk]
            ray_start = allray_start[:, i:i + chunk]
            ray_end = allray_end[:, i:i + chunk]
            renderclothmask = 0#allrenderclothmask[:, i:i + chunk]
            
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

            _, num_rays, samples_per_ray, _ = depths_coarse.shape
            bs_expand = batch_size 
                   
            # Coarse Pass
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       
            
            #mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
            
            
            # sample_coordinates_smpl = self.get_canonical_coordinates(
                # sample_coordinates,
                # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
                # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)) 
                # #mask = mask            
            # )
            # #sample_coordinates = sample_coordinates_smpl
            
            # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
                # sample_coordinates,
                # cloth_reduced_current,
                # self.cloth_reduced_canon.repeat(batch_size,1,1)
                # #mask            
            # )
            
            # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
            # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
            # smpl_distmin = smpl_ptsdistmin[0]
            # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
            # cloth_ptsdist = torch.cdist(sample_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
            # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
            # cloth_distmin = cloth_ptsdistmin[0]
            # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
                        
            # cloth_neartag = cloth_distmin<smpl_distmin
           
            # sample_coordinates = sample_coordinates_smpl
            # sample_coordinates.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
            # sample_coordinates = sample_coordinates.view(batch_size,-1,3)
            
            nnvidxtag = [] 
            smpl_nnvidx = [] 
            cloth_nnvidx = []
            sptneartag = []            
            for i in range(0, batch_size, 1):
                smpl_ptsdist = torch.cdist(sample_coordinates[i], smpl_current[i], p=2)#deformedpersonsmpl
                smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
                smpl_distmin = smpl_ptsdistmin[0]
                smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1)
                smpl_nnvidx.append(smpl_minidx[None])
                
                cloth_ptsdist = torch.cdist(sample_coordinates[i], cloth_current[i], p=2)#deformedpersonsmpl
                cloth_ptsdistmin = torch.min(cloth_ptsdist, 1)
                cloth_distmin = cloth_ptsdistmin[0]
                cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1)
                cloth_nnvidx.append(cloth_minidx[None])
                idxtag = cloth_distmin<smpl_distmin
                nnvidxtag.append(idxtag[None])
                
                ptdist = torch.logical_or(cloth_distmin<0.03, smpl_distmin<0.03)
                sptneartag.append(ptdist[None])
        
            sptneartag = torch.cat(sptneartag).view(batch_size,-1,Nc,1)    
        
            smpl_nnvidx = torch.cat(smpl_nnvidx)   
            cloth_nnvidx = torch.cat(cloth_nnvidx)
            nnvidxtag = torch.cat(nnvidxtag)        
            sample_coordinates_can = self.inversedeforming_samplepoints_LBS(sample_coordinates, smpl_nnvidx, smpl_current, A, Th)
            sample_coordinates_can2 = self.inversedeforming_samplepoints_cloth(sample_coordinates, cloth_nnvidx, cloth_current, A, Th)       
            sample_coordinates_can[nnvidxtag] = sample_coordinates_can2[nnvidxtag]
            
            #displacement
            # smplpara = torch.cat([smpl_params, betas], -1)
            # samplepts_disp_smpl = self.displace_smpl(sample_coordinates, smplpara)
            # samplepts_disp_cloth = self.displace_cloth(sample_coordinates, smplpara)
            # samplepts_disp = torch.cat([samplepts_disp_smpl, samplepts_disp_cloth], 1)
            # sample_coordinates_smpl = sample_coordinates_smpl+samplepts_disp_smpl
            # sample_coordinates_cloth = sample_coordinates_cloth+samplepts_disp_cloth
            samplepts_disp = 0
            
            #density, feature = self.SDFnetwork(sample_coordinates, self.canonparamshape.repeat(batch_size,1))#
            density = self.Densitynetwork(sample_coordinates_can)
            # samplepts_incsdf_smpl = self.incSDFnetwork_smpl(sample_coordinates_smpl)
            # samplepts_incsdf_cloth = self.incSDFnetwork_cloth(sample_coordinates_cloth)
            # samplepts_disp = torch.cat([samplepts_incsdf_smpl, samplepts_incsdf_cloth], 1)#samplepts_incsdf_smpl
            # density_smpl = density_smpl+samplepts_incsdf_smpl
            # density_cloth = density_cloth+samplepts_incsdf_cloth
            samplepts_disp = 0
            
            
            smplpara = torch.cat([smpl_params, betas], -1)
            
            color = self.decoder(sample_coordinates_can, z_rend)
            
            color = color.reshape(batch_size, num_rays, samples_per_ray, color.shape[-1])
            density = density.reshape(batch_size, num_rays, samples_per_ray, 1)              
            #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
            
            # density = sdf_to_alpha2(density.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #torch.exp()
            # density = density.reshape(batch_size, num_rays, samples_per_ray, 1)
            
            beta = self.beta_network().clamp(1e-9, 1e6)
            alpha = self.sdf_to_alpha(density.reshape(batch_size, num_rays, samples_per_ray), beta)
            alpha = alpha.reshape(batch_size, num_rays, samples_per_ray, 1)
            
            alpha[~sptneartag] = 0
            #rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color_smpl, density_smpl, color_cloth, density_cloth, renderclothmask, depths_coarse, batch_size)
            rendered_color_chunk, rendered_mask_chunk, rendered_disparity_chunk, weights_chunk = self.ray_marcher(color, alpha, depths_coarse, batch_size)
                
            rendered_color.append(rendered_color_chunk)
            rendered_mask.append(rendered_mask_chunk)
            rendered_disparity.append(rendered_disparity_chunk)
            weights.append(weights_chunk)
            
            #if 0:    
            offmesh_sdf = 0
            sdfgradients = 0
            meshpts_smpl_sdf = 0
            meshpts_cloth_sdf = 0
            meshptsgradients = 0
            meshpts_normal = 0
        
        rendered_color = torch.cat(rendered_color, dim=1)
        rendered_mask = torch.cat(rendered_mask, dim=1)
        rendered_disparity = torch.cat(rendered_disparity, dim=1)
        weights = torch.cat(weights, dim=1)
        
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients#rgb_final, depth_final, weights.sum(2)
    
    def singlenerf_rendering_deepcap_geometry_color(self, frameidx, cropimg, Nc, Nf, z_rend, planes, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):
        
        R = extrinsics[:,:3,:3] 
        T = extrinsics[:,:3,3:]

        pose_to_camera = torch.bmm(pose_to_world[..., :3, 3], R.permute(0,2,1)) + T.permute(0,2,1)
        pose_2d = torch.matmul(pose_to_camera, intrinsic.permute(0,2,1))
        pose_2d = pose_2d[:, :, :2] / pose_2d[:, :, 2][...,None] 
        #location = pose_2d[:, 15, :2]
        # jointidx = torch.randint(24,(1,))[0]
        # location = pose_2d[:, jointidx, :2]
        
        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        batch_size = smpl_params.shape[0]
        
        #self.sample_canonshape(batch_size)
        
        #deforming with cloth and smpl
        smpl_current = self.deformingsmpl_LBS(A, Th)
        smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)

        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)#
        cloth_reduced_current = self.cloth_reduced(cloth_current)
        
       
        combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        #combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        
        # npvertices = combinedmeshvert[0].detach().cpu().numpy()                        
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = f"test/combined_{frameidx[0].item():0>4}.obj"
        # mesh.export(result_path)
        
        # data_root = '../../neuralbody-deformation-occupancy-fixnerf-our-generalization/tools/data/magdalena20000-allviews/'
        # smplvert_path = os.path.join(data_root, 'vertices',
                                   # '{}.npy'.format(frameidx[0].item()))
        # smplvert = np.load(smplvert_path).astype(np.float32)
        # smplmesh = trimesh.Trimesh(smplvert, self.smplfaces.detach().cpu().numpy(), process=False)
        # result_path = f"test/smpl_{frameidx[0].item():0>4}.obj"
        # smplmesh.export(result_path)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
                
        # with torch.no_grad(): 
            # clothmask, meshmask = self.render_clothmask(self.meshface.repeat(batch_size,1,1), combinedmeshvert, extrinsics, intrinsic)
            # renderclothmask = clothmask#.unsqueeze(-1).repeat(1,1,1,Nc).view(batch_size,-1,Nc,1)
        # mesh_mask = clothmask[0].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        # mesh_mask = clothmask[1].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[1].item()), mesh_mask*255)
        renderclothmask = 0
        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        ray_start = near
        ray_end = far
        # k_iter = self.ray_sampler.iterations // 1000 * 3        
        # print(self.ray_sampler.max_scale * exp(-k_iter*self.ray_sampler.scale_anneal))
        rays_o_s = []
        rays_d_s = []
        ray_start_s = []
        ray_end_s = []
        mask_at_box_s = []
        #renderclothmask_s = [], renderclothmask_i, mask_at_box_i
        for i in range(batch_size):       
            rays_o_i, rays_d_i, ray_start_i, ray_end_i, _, _ = self.ray_sampler(256, 512, ray_origins[i].view(256,512,3), ray_directions[i].view(256,512,3), ray_start[i].view(256,512,1), ray_end[i].view(256,512,1), cropimg[i],pose_2d[i])#, 'local', mask_at_box[i].view(512,512,1)renderclothmask[i], meshmask[i]
            rays_o_s.append(rays_o_i[None])
            rays_d_s.append(rays_d_i[None])
            ray_start_s.append(ray_start_i[None])
            ray_end_s.append(ray_end_i[None])
            #mask_at_box_s.append(mask_at_box_i[None])
            #renderclothmask_s.append(renderclothmask_i[None])
        #mask_at_box = torch.cat(mask_at_box_s)            
        ray_origins = torch.cat(rays_o_s)#[mask_at_box].view(batch_size,-1,3)        
        ray_directions = torch.cat(rays_d_s)#[mask_at_box].view(batch_size,-1,3) 
        ray_start = torch.cat(ray_start_s)#[mask_at_box].view(batch_size,-1,1) 
        ray_end = torch.cat(ray_end_s)#[mask_at_box].view(batch_size,-1,1) 

        #renderclothmask = torch.cat(renderclothmask_s)
        
        # renderclothmask0 = renderclothmask[0].reshape(32,32)#low resolution for nerf rendering
        # renderclothmask0 = renderclothmask0.detach().cpu().numpy()
        # cv2.imwrite('test/real_mask0.png', renderclothmask0*255)
        
        #renderclothmask = renderclothmask.unsqueeze(-1).repeat(1,1,Nc,1)
        
        # Create stratified depth samples
        depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

        _, num_rays, samples_per_ray, _ = depths_coarse.shape
        bs_expand = batch_size 
               
        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       

        #mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        #sample_coordinates.requires_grad_(True)
        
        # sample_coordinates_smpl = self.get_canonical_coordinates(
            # sample_coordinates,
            # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)) 
            # #mask = mask            
        # )
        # #sample_coordinates = sample_coordinates_smpl
        # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
            # sample_coordinates,
            # cloth_reduced_current,
            # self.cloth_reduced_canon.repeat(batch_size,1,1)
            # #mask            
        # )
        
        # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
        # cloth_ptsdist = torch.cdist(sample_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_distmin = cloth_ptsdistmin[0]
        # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
                    
        # cloth_neartag = cloth_distmin<smpl_distmin
       
        # sample_coordinates_can = sample_coordinates_smpl
        # sample_coordinates_can.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
        # sample_coordinates_can = sample_coordinates_can.view(batch_size,-1,3)
        
        # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_current, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # nnvidx = torch.squeeze(smpl_ptsdistmin[1], -1)
        
        nnvidxtag = [] 
        smpl_nnvidx = [] 
        cloth_nnvidx = []
        sptneartag = []        
        for i in range(0, batch_size, 1):
            smpl_minidx = sample_coordinates.new_zeros([sample_coordinates.shape[1]], dtype=torch.long)              
            smpl_ptsdist = torch.cdist(sample_coordinates[i], smpl_current[i], p=2)#[mask[i]]deformedpersonsmpl
            smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            smpl_distmin = smpl_ptsdistmin[0]
            smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1)#[mask[i]]
            smpl_nnvidx.append(smpl_minidx[None])
            
            cloth_ptsdist = torch.cdist(sample_coordinates[i], cloth_current[i], p=2)#deformedpersonsmpl
            cloth_ptsdistmin = torch.min(cloth_ptsdist, 1)
            cloth_distmin = cloth_ptsdistmin[0]
            cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1)
            cloth_nnvidx.append(cloth_minidx[None])
            idxtag = cloth_distmin<smpl_distmin
            nnvidxtag.append(idxtag[None])
            
            ptdist = torch.logical_or(cloth_distmin<0.03, smpl_distmin<0.03)
            sptneartag.append(ptdist[None])
        
        sptneartag = torch.cat(sptneartag).view(batch_size,-1,Nc,1)            
        smpl_nnvidx = torch.cat(smpl_nnvidx)   
        cloth_nnvidx = torch.cat(cloth_nnvidx)
        nnvidxtag = torch.cat(nnvidxtag)        #[mask].view(batch_size,-1,3)
        sample_coordinates_can = self.inversedeforming_samplepoints_LBS(sample_coordinates, smpl_nnvidx, smpl_current, A, Th)
        sample_coordinates_can2 = self.inversedeforming_samplepoints_cloth(sample_coordinates, cloth_nnvidx, cloth_current, A, Th)       
        sample_coordinates_can[nnvidxtag] = sample_coordinates_can2[nnvidxtag]
        
        #displacement
        # smplpara = torch.cat([smpl_params, betas], -1)
        # samplepts_disp_smpl = self.displace_smpl(sample_coordinates, smplpara)
        # samplepts_disp_cloth = self.displace_cloth(sample_coordinates, smplpara)
        # samplepts_disp = torch.cat([samplepts_disp_smpl, samplepts_disp_cloth], 1)
        # sample_coordinates_smpl = sample_coordinates_smpl+samplepts_disp_smpl
        # sample_coordinates_cloth = sample_coordinates_cloth+samplepts_disp_cloth
        samplepts_disp = 0
        
        # meshvert_def = torch.cat([self.canonclothvert, self.rawtemplatesmpl], dim=1)#self.deformedcloth
        # npvertices = meshvert_def[0].detach().cpu().numpy()
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/canverts{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = self.rawtemplatesmpl[0].detach().cpu().numpy()
        # npfaces = self.smplfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/smpl_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = smpl_current[0].detach().cpu().numpy()
        # npfaces = self.smplfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/smpl{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # # npvertices = self.canonclothvert[0].detach().cpu().numpy()
        # # npfaces = self.clothfaces.detach().cpu().numpy()
        # # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # # result_path = os.path.join('test/cloth_can{:04d}.obj'.format(frameidx[0].item()))
        # # mesh.export(result_path)
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = sample_coordinates_can[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/sample_coordinates_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        
        # sample_coordinates_smpl.requires_grad_(True)
        # sample_coordinates_cloth.requires_grad_(True)
        # #print(torch.isnan(sample_coordinates_cloth).any() or torch.isinf(sample_coordinates_cloth).any())
        # density_cloth, feature_cloth = self.SDFnetwork_cloth(sample_coordinates_cloth,self.shapepara.repeat(batch_size,1))#tempclothpara     
        
        # #density_cloth = torch.nan_to_num(density_cloth, nan=0.0, posinf=0.0, neginf=0.0)
        
        # density_smpl, feature_smpl = self.SDFnetwork_smpl(sample_coordinates_smpl,self.canonparamshape.repeat(batch_size,1))
        
        #sample_coordinates_can.requires_grad_(True)
        #density, feature = self.SDFnetwork(sample_coordinates, self.canonparamshape.repeat(batch_size,1))#
        density = self.Densitynetwork(sample_coordinates_can)
        
        # samplepts_incsdf_smpl = self.incSDFnetwork_smpl(sample_coordinates_smpl)
        # samplepts_incsdf_cloth = self.incSDFnetwork_cloth(sample_coordinates_cloth)
        # samplepts_disp = torch.cat([samplepts_incsdf_smpl, samplepts_incsdf_cloth], 1)#samplepts_incsdf_smpl
        # density_smpl = density_smpl+samplepts_incsdf_smpl
        # density_cloth = density_cloth+samplepts_incsdf_cloth
        samplepts_disp = 0
        
        # sample_coordinates_cloth.requires_grad_(True)
        # #samplepts_density_smpl, samplepts_density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], samplepts_smpl, samplepts_cloth)
        # samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
        # samplepts_density_smpl,_ = self.SDFnetwork_smpl(samplepts_smpl,self.canonparamshape)

        # offmesh_sdf_smpl = samplepts_incsdf_smpl.reshape(batch_size,-1)#density_smpl
        # offmesh_sdf_cloth = samplepts_incsdf_cloth.reshape(batch_size,-1)#density_cloth
        # #offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)#offmesh_sdf_smpl
        # offmesh_sdf = torch.cat([density_smpl.reshape(batch_size,-1),density_cloth.reshape(batch_size,-1)],dim=1)
        #offmesh_sdf = 0
        
        # offmesh_sdf_smpl = density_smpl.reshape(batch_size,-1)#density_smpl
        # offmesh_sdf_cloth = density_cloth.reshape(batch_size,-1)#density_cloth
        # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)#offmesh_sdf_smpl
        
        
        # d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        # sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      # inputs=sample_coordinates_smpl,
                                       # grad_outputs=d_output_smpl,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        # sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      # inputs=sample_coordinates_cloth,
                                       # grad_outputs=d_output_cloth,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)#sdfgradients_smpl
        #sdfgradients = 0
        
        # offmesh_sdf = density.reshape(batch_size,-1)
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdfgradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=sample_coordinates,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
                                       
        #density_smpl = torch.nan_to_num(density_smpl, nan=0.0, posinf=0.0, neginf=0.0)
        # density_cloth[density_cloth.abs()>0.5] = 0.5
        # density_smpl[density_smpl.abs()>0.5] = 0.5
        #print(torch.norm(sdfgradients[0], dim=-1))
        smplpara = torch.cat([smpl_params, betas], -1)
        
        # colorlatentsmpl = self.colorlatent_smpl(torch.LongTensor([0]).to('cuda'))
        # colorlatentcloth = self.colorlatent_cloth(torch.LongTensor([0]).to('cuda'))
        # feature_cloth = torch.cat([feature_cloth,colorlatentcloth[:,None].repeat(feature_cloth.shape[0],feature_cloth.shape[1],1)],dim=-1)
        # feature_smpl = torch.cat([feature_smpl,colorlatentsmpl[:,None].repeat(feature_smpl.shape[0],feature_smpl.shape[1],1)],dim=-1)
        # color_cloth = self.decoder_cloth(sample_coordinates_cloth, z_rend)#feature_cloth
        # color_smpl = self.decoder(sample_coordinates_smpl, z_rend)#feature_smpl
        
        color = self.decoder(sample_coordinates_can, z_rend)
        
        # density_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32)#normlized to [-1,1]
        # density_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32) * 1e3
        # color_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32)#normlized to [-1,1]
        # color_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32) * 1e3
        # density_cloth[mask] = density_cloth0
        # density_smpl[mask] = density_smpl0
        # color_cloth[mask] = color_cloth0
        # color_smpl[mask] = color_smpl0
        
        #color_smpl, color_cloth = self.fetchingdoublenerf_fixgeometry_color(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        #color_smpl, density_smpl, color_cloth, density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        
        color = color.reshape(batch_size, num_rays, samples_per_ray, color.shape[-1])
        density = density.reshape(batch_size, num_rays, samples_per_ray, 1)              
        #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
        #density = self.laplacedensity(density.reshape(batch_size, -1,1))

        # density = sdf_to_alpha2(density.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #
        # density = density.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        beta = self.beta_network().clamp(1e-9, 1e6)
        alpha = self.sdf_to_alpha(density.reshape(batch_size, num_rays, samples_per_ray), beta)
        alpha = alpha.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        alpha[~sptneartag] = 0
        
        # alpha = sdf_to_alpha(density.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s * 1.0))  
        # alpha = alpha.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        # color_smpl = color_smpl.reshape(batch_size, num_rays, samples_per_ray, color_smpl.shape[-1])
        # density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)              
        # #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())

        # density_smpl = sdf_to_alpha2(density_smpl.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #
        # density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)
        # #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
        # #density_smplmesh, _ = self.SDFnetwork_smpl(self.smpl_reduced_canon_vert.repeat(batch_size,1,1),self.canonparamshape.repeat(batch_size,1))

        # #density_smplmesh=sdf_to_alpha2(density_smplmesh, 0.05)
        # #print(torch.exp(self.ln_s * 1.0),density_smplmesh[0])
        
        # color_cloth = color_cloth.reshape(batch_size, num_rays, samples_per_ray, color_cloth.shape[-1])
        # density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)  
        # #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())        
        # density_cloth = sdf_to_alpha2(density_cloth.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s_cloth))  #
        # density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)
        # #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())  

        #rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color_smpl, density_smpl, color_cloth, density_cloth, renderclothmask, depths_coarse, batch_size)
        rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color, alpha, depths_coarse, batch_size)
        # rendered_color_full = rendered_color.new_zeros([batch_size,self.ray_sampler.N_samples_sqrt**2,3])
        # rendered_color_full[mask_at_box] = rendered_color.reshape(batch_size, -1, 3)
        # rendered_color = rendered_color_full.reshape(batch_size, -1)
        
        with torch.no_grad(): 
            meshpts_smpl, meshpts_cloth, samplepts_smpl, samplepts_cloth, meshpts_normal = self.samplingpoint_learningmeshsdf(self.rawtemplatesmpl, self.canon_clothvert)#.repeat(batch_size,1,1)cloth_reduced_canon
        
        #samplepts.requires_grad_(True)
        #_, samplepts_density = self.fetchingnerf(planes, samplepts)
        # samplepts = torch.cat([samplepts_smpl,samplepts_cloth,meshpts_smpl,meshpts_cloth],dim=1)
        # samplepts.requires_grad_(True)
        
        samplepts_smpl.requires_grad_(True)
        #samplepts_cloth.requires_grad_(True)
        
        #samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
        samplepts_density_smpl = self.Densitynetwork(samplepts_smpl)

        # hypo_params_cloth = self.hyper_net_cloth(self.tempclothpara)
        # embed_samplepts_cloth = self.embed_fn_fine(samplepts_cloth)
        # print(embed_samplepts_cloth.shape)
        # samplepts_density_cloth = self.sdf_network_cloth(embed_samplepts_cloth, params=hypo_params_cloth)
        
        # hypo_params_smpl = self.hyper_net_smpl(self.canonparamshape)
        # embed_samplepts_smpl = self.embed_fn_fine(samplepts_smpl)
        # samplepts_density_smpl = self.sdf_network_cloth(embed_samplepts_smpl, params=hypo_params_smpl)

        offmesh_sdf_smpl = samplepts_density_smpl.reshape(batch_size,-1)
        #offmesh_sdf_cloth = samplepts_density_cloth.reshape(batch_size,-1)
        offmesh_sdf = offmesh_sdf_smpl#torch.cat([,offmesh_sdf_cloth],dim=1)

        d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      inputs=samplepts_smpl,
                                       grad_outputs=d_output_smpl,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        # # d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        # # sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      # # inputs=samplepts_cloth,
                                       # # grad_outputs=d_output_cloth,
                                      # # create_graph=True,
                                       # # retain_graph=True,
                                       # # only_inputs=True)[0]
        # # sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)
        sdfgradients = sdfgradients_smpl
        
        #_, meshpts_smpl_sdf = self.fetchingnerf(planes, meshpts_smpl)
        #_, meshpts_cloth_sdf = self.fetchingnerf(planes, meshpts_cloth)
        # meshpts_smpl.requires_grad_(True)
        # meshpts_cloth.requires_grad_(True)
        #meshpts_smpl_sdf, meshpts_cloth_sdf = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], meshpts_smpl, meshpts_cloth)
        meshpts_cloth_sdf = self.Densitynetwork(meshpts_cloth)
        meshpts_smpl_sdf = self.Densitynetwork(meshpts_smpl)
        
        # embed_meshpts_cloth = self.embed_fn_fine(meshpts_cloth_sdf)
        # meshpts_cloth_sdf = self.sdf_network_cloth(embed_meshpts_cloth, params=hypo_params_cloth)
        
        # embed_meshpts_smpl = self.embed_fn_fine(meshpts_smpl_sdf)
        # meshpts_smpl_sdf = self.sdf_network_cloth(embed_meshpts_smpl, params=hypo_params_smpl)


        meshpts_smpl_sdf = meshpts_smpl_sdf.reshape(batch_size,-1)
        meshpts_cloth_sdf = meshpts_cloth_sdf.reshape(batch_size,-1)
        #print(meshpts_smpl_sdf[0])       
        # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth,meshpts_smpl_sdf,meshpts_cloth_sdf],dim=1)
                
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdf_gradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=samplepts,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
                                       
        # sampleptsnum = samplepts_smpl.shape[1]+samplepts_cloth.shape[1]
        # sdfgradients =  sdf_gradients[:,:sampleptsnum,:]          
        # meshptsgradients =  sdf_gradients[:,sampleptsnum:,:]   
        
        # output_smpl = torch.ones_like(meshpts_smpl_sdf, requires_grad=False, device=meshpts_smpl_sdf.device)#
        # meshptsgradients_smpl = torch.autograd.grad(outputs=meshpts_smpl_sdf,
                                      # inputs=meshpts_smpl,
                                       # grad_outputs=output_smpl,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # output_cloth = torch.ones_like(meshpts_cloth_sdf, requires_grad=False, device=meshpts_cloth_sdf.device)#
        # meshptsgradients_cloth = torch.autograd.grad(outputs=meshpts_cloth_sdf,
                                      # inputs=meshpts_cloth,
                                       # grad_outputs=output_cloth,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # meshptsgradients = torch.cat([meshptsgradients_smpl,meshptsgradients_cloth],dim=1)
        #meshptsgradients = 0
        
        #if 0:    
        #offmesh_sdf = 0
        #sdfgradients = 0
        #meshpts_smpl_sdf = 0
        #meshpts_cloth_sdf = 0
        meshptsgradients = 0
        meshpts_normal = 0
        
        
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients#rgb_final, depth_final, weights.sum(2)
    
    def singlenerf_rendering_deepfashion_geometry_color_chunk(self, frameidx, cropimg, Nc, Nf, z_rend, planes, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):
                        
        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        batch_size = smpl_params.shape[0]
        
        #self.sample_canonshape(batch_size)
        
        #deforming with cloth and smpl
        # smpl_current = self.deformingsmpl_LBS(A, Th)
        # smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)
        smploutput = self.smpl.forward(betas, smpl_body_pose, smpl_orient, Th)#        
        smpl_current = smploutput.vertices
        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        # cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)#
        # cloth_reduced_current = self.cloth_reduced(cloth_current)
        
       
        # combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        #combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        
        # npvertices = combinedmeshvert[0].detach().cpu().numpy()                        
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = f"test/combined_{frameidx[0].item():0>4}.obj"
        # mesh.export(result_path)
        
        # data_root = '../../neuralbody-deformation-occupancy-fixnerf-our-generalization/tools/data/magdalena20000-allviews/'
        # smplvert_path = os.path.join(data_root, 'vertices',
                                   # '{}.npy'.format(frameidx[0].item()))
        # smplvert = np.load(smplvert_path).astype(np.float32)
        # smplmesh = trimesh.Trimesh(smplvert, self.smplfaces.detach().cpu().numpy(), process=False)
        # result_path = f"test/smpl_{frameidx[0].item():0>4}.obj"
        # smplmesh.export(result_path)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
                
        # with torch.no_grad(): 
            # clothmask, meshmask = self.render_clothmask(self.meshface.repeat(batch_size,1,1), combinedmeshvert, extrinsics, intrinsic)
            # renderclothmask = clothmask#.unsqueeze(-1).repeat(1,1,1,Nc).view(batch_size,-1,Nc,1)
        # mesh_mask = clothmask[0].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        # mesh_mask = clothmask[1].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[1].item()), mesh_mask*255)
        renderclothmask = 0
        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        ray_start = near
        ray_end = far
        # k_iter = self.ray_sampler.iterations // 1000 * 3        
        # print(self.ray_sampler.max_scale * exp(-k_iter*self.ray_sampler.scale_anneal))
        # rays_o_s = []
        # rays_d_s = []
        # ray_start_s = []
        # ray_end_s = []
        # #renderclothmask_s = []renderclothmask_i,
        # for i in range(batch_size):       
            # rays_o_i, rays_d_i, ray_start_i, ray_end_i, _, _ = self.ray_sampler(512, 512, ray_origins[i].view(512,512,3), ray_directions[i].view(512,512,3), ray_start[i].view(512,512,1), ray_end[i].view(512,512,1), cropimg[i])#renderclothmask[i], meshmask[i]
            # rays_o_s.append(rays_o_i[None])
            # rays_d_s.append(rays_d_i[None])
            # ray_start_s.append(ray_start_i[None])
            # ray_end_s.append(ray_end_i[None])
            # #renderclothmask_s.append(renderclothmask_i[None])
        # allray_origins = torch.cat(rays_o_s)        
        # allray_directions = torch.cat(rays_d_s)
        # allray_start = torch.cat(ray_start_s)
        # allray_end = torch.cat(ray_end_s)
        # allrenderclothmask = torch.cat(renderclothmask_s)
        
        # renderclothmask0 = renderclothmask[0].reshape(32,32)#low resolution for nerf rendering
        # renderclothmask0 = renderclothmask0.detach().cpu().numpy()
        # cv2.imwrite('test/real_mask0.png', renderclothmask0*255)
        #allrenderclothmask = allrenderclothmask.unsqueeze(-1).repeat(1,1,Nc,1)
        
        allray_origins = ray_origins       
        allray_directions = ray_directions
        allray_start = ray_start
        allray_end = ray_end
        #allrenderclothmask = renderclothmask
        
        # Create stratified depth samples

        n_batch, n_pixel = allray_origins.shape[:2]
        chunk = 2048
        rendered_color = []
        rendered_mask = []
        rendered_disparity = []
        weights = []
        for i in range(0, n_pixel, chunk):
            ray_origins = allray_origins[:, i:i + chunk]
            ray_directions = allray_directions[:, i:i + chunk]
            ray_start = allray_start[:, i:i + chunk]
            ray_end = allray_end[:, i:i + chunk]
            renderclothmask = 0#allrenderclothmask[:, i:i + chunk]
            
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

            _, num_rays, samples_per_ray, _ = depths_coarse.shape
            bs_expand = batch_size 
                   
            # Coarse Pass
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       
            
            #mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
            
            
            # sample_coordinates_smpl = self.get_canonical_coordinates(
                # sample_coordinates,
                # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
                # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)) 
                # #mask = mask            
            # )
            # #sample_coordinates = sample_coordinates_smpl
            
            # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
                # sample_coordinates,
                # cloth_reduced_current,
                # self.cloth_reduced_canon.repeat(batch_size,1,1)
                # #mask            
            # )
            
            # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
            # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
            # smpl_distmin = smpl_ptsdistmin[0]
            # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
            # cloth_ptsdist = torch.cdist(sample_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
            # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
            # cloth_distmin = cloth_ptsdistmin[0]
            # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
                        
            # cloth_neartag = cloth_distmin<smpl_distmin
           
            # sample_coordinates = sample_coordinates_smpl
            # sample_coordinates.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
            # sample_coordinates = sample_coordinates.view(batch_size,-1,3)
            
            nnvidxtag = [] 
            smpl_nnvidx = [] 
            cloth_nnvidx = []
            sptneartag = []            
            for i in range(0, batch_size, 1):
                smpl_ptsdist = torch.cdist(sample_coordinates[i], smpl_current[i], p=2)#deformedpersonsmpl
                smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
                smpl_distmin = smpl_ptsdistmin[0]
                smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1)
                smpl_nnvidx.append(smpl_minidx[None])
                
                # cloth_ptsdist = torch.cdist(sample_coordinates[i], cloth_current[i], p=2)#deformedpersonsmpl
                # cloth_ptsdistmin = torch.min(cloth_ptsdist, 1)
                # cloth_distmin = cloth_ptsdistmin[0]
                # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1)
                # cloth_nnvidx.append(cloth_minidx[None])
                # idxtag = cloth_distmin<smpl_distmin
                # nnvidxtag.append(idxtag[None])
                
                ptdist = smpl_distmin<0.03#torch.logical_or(cloth_distmin<0.03, smpl_distmin<0.03)
                sptneartag.append(ptdist[None])
        
            sptneartag = torch.cat(sptneartag).view(batch_size,-1,Nc,1)    
        
            smpl_nnvidx = torch.cat(smpl_nnvidx)   
            # cloth_nnvidx = torch.cat(cloth_nnvidx)
            # nnvidxtag = torch.cat(nnvidxtag)        
            sample_coordinates_can = self.inversedeforming_samplepoints_LBS(sample_coordinates, smpl_nnvidx, smpl_current, A, Th)
            # sample_coordinates_can2 = self.inversedeforming_samplepoints_cloth(sample_coordinates, cloth_nnvidx, cloth_current, A, Th)       
            # sample_coordinates_can[nnvidxtag] = sample_coordinates_can2[nnvidxtag]
            
            #displacement
            # smplpara = torch.cat([smpl_params, betas], -1)
            # samplepts_disp_smpl = self.displace_smpl(sample_coordinates, smplpara)
            # samplepts_disp_cloth = self.displace_cloth(sample_coordinates, smplpara)
            # samplepts_disp = torch.cat([samplepts_disp_smpl, samplepts_disp_cloth], 1)
            # sample_coordinates_smpl = sample_coordinates_smpl+samplepts_disp_smpl
            # sample_coordinates_cloth = sample_coordinates_cloth+samplepts_disp_cloth
            samplepts_disp = 0
            
            #density, feature = self.SDFnetwork(sample_coordinates, self.canonparamshape.repeat(batch_size,1))#
            #density = self.Densitynetwork(sample_coordinates_can)
            color, density = self.decoder(sample_coordinates_can, z_rend)
            # samplepts_incsdf_smpl = self.incSDFnetwork_smpl(sample_coordinates_smpl)
            # samplepts_incsdf_cloth = self.incSDFnetwork_cloth(sample_coordinates_cloth)
            # samplepts_disp = torch.cat([samplepts_incsdf_smpl, samplepts_incsdf_cloth], 1)#samplepts_incsdf_smpl
            # density_smpl = density_smpl+samplepts_incsdf_smpl
            # density_cloth = density_cloth+samplepts_incsdf_cloth
            samplepts_disp = 0
            
            
            #smplpara = torch.cat([smpl_params, betas], -1)
            
            #color = self.decoder(sample_coordinates_can, z_rend)
            
            color = color.reshape(batch_size, num_rays, samples_per_ray, color.shape[-1])
            density = density.reshape(batch_size, num_rays, samples_per_ray, 1)              
            #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
            
            # density = sdf_to_alpha2(density.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #torch.exp()
            # density = density.reshape(batch_size, num_rays, samples_per_ray, 1)
            alpha = density
            # beta = self.beta_network().clamp(1e-9, 1e6)
            # alpha = self.sdf_to_alpha(density.reshape(batch_size, num_rays, samples_per_ray), beta)
            # alpha = alpha.reshape(batch_size, num_rays, samples_per_ray, 1)
            
            # alpha[~sptneartag] = 0
            #rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color_smpl, density_smpl, color_cloth, density_cloth, renderclothmask, depths_coarse, batch_size)
            rendered_color_chunk, rendered_mask_chunk, rendered_disparity_chunk, weights_chunk = self.ray_marcher(color, alpha, depths_coarse, batch_size)
                
            rendered_color.append(rendered_color_chunk)
            rendered_mask.append(rendered_mask_chunk)
            rendered_disparity.append(rendered_disparity_chunk)
            weights.append(weights_chunk)
            
            #if 0:    
            offmesh_sdf = 0
            sdfgradients = 0
            meshpts_smpl_sdf = 0
            meshpts_cloth_sdf = 0
            meshptsgradients = 0
            meshpts_normal = 0
        
        rendered_color = torch.cat(rendered_color, dim=1)
        rendered_mask = torch.cat(rendered_mask, dim=1)
        rendered_disparity = torch.cat(rendered_disparity, dim=1)
        weights = torch.cat(weights, dim=1)
        
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients#rgb_final, depth_final, weights.sum(2)
    
    def singlenerf_rendering_deepfashion_geometry_color(self, frameidx, cropimg, Nc, Nf, z_rend, planes, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):
        
        R = extrinsics[:,:3,:3] 
        T = extrinsics[:,:3,3:]

        pose_to_camera = torch.bmm(pose_to_world[..., :3, 3], R.permute(0,2,1)) + T.permute(0,2,1)
        pose_2d = torch.matmul(pose_to_camera, intrinsic.permute(0,2,1))
        pose_2d = pose_2d[:, :, :2] / pose_2d[:, :, 2][...,None] 
        #location = pose_2d[:, 15, :2]
        # jointidx = torch.randint(24,(1,))[0]
        # location = pose_2d[:, jointidx, :2]
        
        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        batch_size = smpl_params.shape[0]
        
        #self.sample_canonshape(batch_size)
        
        #deforming with cloth and smpl
        # smpl_current = self.deformingsmpl_LBS(A, Th)
        # smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)

        smploutput = self.smpl.forward(betas, smpl_body_pose, smpl_orient, Th)#        
        smpl_current = smploutput.vertices
        
        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        # cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)#
        # cloth_reduced_current = self.cloth_reduced(cloth_current)
        
       
        # combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        #combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        
        # npvertices = combinedmeshvert[0].detach().cpu().numpy()                        
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = f"test/combined_{frameidx[0].item():0>4}.obj"
        # mesh.export(result_path)
        
        # data_root = '../../neuralbody-deformation-occupancy-fixnerf-our-generalization/tools/data/magdalena20000-allviews/'
        # smplvert_path = os.path.join(data_root, 'vertices',
                                   # '{}.npy'.format(frameidx[0].item()))
        # smplvert = np.load(smplvert_path).astype(np.float32)
        # smplmesh = trimesh.Trimesh(smplvert, self.smplfaces.detach().cpu().numpy(), process=False)
        # result_path = f"test/smpl_{frameidx[0].item():0>4}.obj"
        # smplmesh.export(result_path)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        
        # meshmask = self.rasterize_eg3d(self.smplfaces, smpl_current, extrinsics, intrinsic) 
        # mesh_mask = meshmask[0][:,:256].detach().cpu().numpy()
        # cv2.imwrite('result/deepcap/test/rendermeshmask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        
        # smpl_camera = torch.bmm(smpl_current, R.permute(0,2,1)) + T.permute(0,2,1)
        # smpl_2d = torch.matmul(smpl_camera, intrinsic.permute(0,2,1))
        # smpl_2d = smpl_2d[:, :, :2] / smpl_2d[:, :, 2][...,None] 
        # for t in range(0, 6890):
           # cropimg[0,:,smpl_2d[0,t,1].long(),smpl_2d[0,t,0].long()] = 0  
           # cropimg[1,:,smpl_2d[1,t,1].long(),smpl_2d[1,t,0].long()] = 0   
        # crop_img = cropimg.cpu().numpy()[0].transpose(1, 2, 0)
        # crop_img = crop_img * 127.5 + 127.5
        # cv2.imwrite('result/deepcap/test/crop_img{:04d}.png'.format(frameidx[0].item()), crop_img)    
        
        #with torch.no_grad(): 
            #clothmask, meshmask = self.render_clothmask(self.meshface.repeat(batch_size,1,1), combinedmeshvert, extrinsics, intrinsic)
            # renderclothmask = clothmask#.unsqueeze(-1).repeat(1,1,1,Nc).view(batch_size,-1,Nc,1)
        # mesh_mask = clothmask[0].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        # mesh_mask = clothmask[1].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[1].item()), mesh_mask*255)
        renderclothmask = 0
        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        ray_start = near
        ray_end = far
        # k_iter = self.ray_sampler.iterations // 1000 * 3        
        # print(self.ray_sampler.max_scale * exp(-k_iter*self.ray_sampler.scale_anneal))
        rays_o_s = []
        rays_d_s = []
        ray_start_s = []
        ray_end_s = []
        mask_at_box_s = []
        #renderclothmask_s = [], renderclothmask_i, mask_at_box_i
        for i in range(batch_size):       
            rays_o_i, rays_d_i, ray_start_i, ray_end_i, _, _ = self.ray_sampler(512, 256, ray_origins[i].view(512,256,3), ray_directions[i].view(512,256,3), ray_start[i].view(512,256,1), ray_end[i].view(512,256,1), cropimg[i],pose_2d[i])#, 'local', mask_at_box[i].view(512,512,1)renderclothmask[i], meshmask[i]
            rays_o_s.append(rays_o_i[None])
            rays_d_s.append(rays_d_i[None])
            ray_start_s.append(ray_start_i[None])
            ray_end_s.append(ray_end_i[None])
            #mask_at_box_s.append(mask_at_box_i[None])
            #renderclothmask_s.append(renderclothmask_i[None])
        #mask_at_box = torch.cat(mask_at_box_s)            
        ray_origins = torch.cat(rays_o_s)#[mask_at_box].view(batch_size,-1,3)        
        ray_directions = torch.cat(rays_d_s)#[mask_at_box].view(batch_size,-1,3) 
        ray_start = torch.cat(ray_start_s)#[mask_at_box].view(batch_size,-1,1) 
        ray_end = torch.cat(ray_end_s)#[mask_at_box].view(batch_size,-1,1) 

        #renderclothmask = torch.cat(renderclothmask_s)
        
        # renderclothmask0 = renderclothmask[0].reshape(32,32)#low resolution for nerf rendering
        # renderclothmask0 = renderclothmask0.detach().cpu().numpy()
        # cv2.imwrite('test/real_mask0.png', renderclothmask0*255)
        
        #renderclothmask = renderclothmask.unsqueeze(-1).repeat(1,1,Nc,1)
        
        # Create stratified depth samples
        depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

        _, num_rays, samples_per_ray, _ = depths_coarse.shape
        bs_expand = batch_size 
               
        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       

        #mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        #sample_coordinates.requires_grad_(True)
        
        # sample_coordinates_smpl = self.get_canonical_coordinates(
            # sample_coordinates,
            # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)) 
            # #mask = mask            
        # )
        # #sample_coordinates = sample_coordinates_smpl
        # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
            # sample_coordinates,
            # cloth_reduced_current,
            # self.cloth_reduced_canon.repeat(batch_size,1,1)
            # #mask            
        # )
        
        # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
        # cloth_ptsdist = torch.cdist(sample_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_distmin = cloth_ptsdistmin[0]
        # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
                    
        # cloth_neartag = cloth_distmin<smpl_distmin
       
        # sample_coordinates_can = sample_coordinates_smpl
        # sample_coordinates_can.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
        # sample_coordinates_can = sample_coordinates_can.view(batch_size,-1,3)
        
        # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_current, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # nnvidx = torch.squeeze(smpl_ptsdistmin[1], -1)
        
        nnvidxtag = [] 
        smpl_nnvidx = [] 
        cloth_nnvidx = []
        sptneartag = []        
        for i in range(0, batch_size, 1):
            smpl_minidx = sample_coordinates.new_zeros([sample_coordinates.shape[1]], dtype=torch.long)              
            smpl_ptsdist = torch.cdist(sample_coordinates[i], smpl_current[i], p=2)#[mask[i]]deformedpersonsmpl
            smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            smpl_distmin = smpl_ptsdistmin[0]
            smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1)#[mask[i]]
            smpl_nnvidx.append(smpl_minidx[None])
            
            # cloth_ptsdist = torch.cdist(sample_coordinates[i], cloth_current[i], p=2)#deformedpersonsmpl
            # cloth_ptsdistmin = torch.min(cloth_ptsdist, 1)
            # cloth_distmin = cloth_ptsdistmin[0]
            # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1)
            # cloth_nnvidx.append(cloth_minidx[None])
            # idxtag = cloth_distmin<smpl_distmin
            # nnvidxtag.append(idxtag[None])
            
            # ptdist = torch.logical_or(cloth_distmin<0.03, smpl_distmin<0.03)
            ptdist = smpl_distmin<0.03
            sptneartag.append(ptdist[None])
        
        sptneartag = torch.cat(sptneartag).view(batch_size,-1,Nc,1)            
        smpl_nnvidx = torch.cat(smpl_nnvidx)   
        # cloth_nnvidx = torch.cat(cloth_nnvidx)
        # nnvidxtag = torch.cat(nnvidxtag)        #[mask].view(batch_size,-1,3)
        sample_coordinates_can = self.inversedeforming_samplepoints_LBS(sample_coordinates, smpl_nnvidx, smpl_current, A, Th)
        #sample_coordinates_can2 = self.inversedeforming_samplepoints_cloth(sample_coordinates, cloth_nnvidx, cloth_current, A, Th)       
        #sample_coordinates_can[nnvidxtag] = sample_coordinates_can2[nnvidxtag]
        
        #displacement
        # smplpara = torch.cat([smpl_params, betas], -1)
        # samplepts_disp_smpl = self.displace_smpl(sample_coordinates, smplpara)
        # samplepts_disp_cloth = self.displace_cloth(sample_coordinates, smplpara)
        # samplepts_disp = torch.cat([samplepts_disp_smpl, samplepts_disp_cloth], 1)
        # sample_coordinates_smpl = sample_coordinates_smpl+samplepts_disp_smpl
        # sample_coordinates_cloth = sample_coordinates_cloth+samplepts_disp_cloth
        samplepts_disp = 0
        
        # meshvert_def = torch.cat([self.canonclothvert, self.rawtemplatesmpl], dim=1)#self.deformedcloth
        # npvertices = meshvert_def[0].detach().cpu().numpy()
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/canverts{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = self.rawtemplatesmpl[0].detach().cpu().numpy()
        # npfaces = self.smplfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/smpl_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = smpl_current[0].detach().cpu().numpy()
        # npfaces = self.smplfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/smpl{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # # npvertices = self.canonclothvert[0].detach().cpu().numpy()
        # # npfaces = self.clothfaces.detach().cpu().numpy()
        # # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # # result_path = os.path.join('test/cloth_can{:04d}.obj'.format(frameidx[0].item()))
        # # mesh.export(result_path)
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = sample_coordinates_can[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/sample_coordinates_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        
        # sample_coordinates_smpl.requires_grad_(True)
        # sample_coordinates_cloth.requires_grad_(True)
        # #print(torch.isnan(sample_coordinates_cloth).any() or torch.isinf(sample_coordinates_cloth).any())
        # density_cloth, feature_cloth = self.SDFnetwork_cloth(sample_coordinates_cloth,self.shapepara.repeat(batch_size,1))#tempclothpara     
        
        # #density_cloth = torch.nan_to_num(density_cloth, nan=0.0, posinf=0.0, neginf=0.0)
        
        # density_smpl, feature_smpl = self.SDFnetwork_smpl(sample_coordinates_smpl,self.canonparamshape.repeat(batch_size,1))
        
        #sample_coordinates_can.requires_grad_(True)
        #density, feature = self.SDFnetwork(sample_coordinates, self.canonparamshape.repeat(batch_size,1))#
        #density = self.Densitynetwork(sample_coordinates_can)
        color, density = self.decoder(sample_coordinates_can, z_rend)
        # samplepts_incsdf_smpl = self.incSDFnetwork_smpl(sample_coordinates_smpl)
        # samplepts_incsdf_cloth = self.incSDFnetwork_cloth(sample_coordinates_cloth)
        # samplepts_disp = torch.cat([samplepts_incsdf_smpl, samplepts_incsdf_cloth], 1)#samplepts_incsdf_smpl
        # density_smpl = density_smpl+samplepts_incsdf_smpl
        # density_cloth = density_cloth+samplepts_incsdf_cloth
        samplepts_disp = 0
        
        # sample_coordinates_cloth.requires_grad_(True)
        # #samplepts_density_smpl, samplepts_density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], samplepts_smpl, samplepts_cloth)
        # samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
        # samplepts_density_smpl,_ = self.SDFnetwork_smpl(samplepts_smpl,self.canonparamshape)

        # offmesh_sdf_smpl = samplepts_incsdf_smpl.reshape(batch_size,-1)#density_smpl
        # offmesh_sdf_cloth = samplepts_incsdf_cloth.reshape(batch_size,-1)#density_cloth
        # #offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)#offmesh_sdf_smpl
        # offmesh_sdf = torch.cat([density_smpl.reshape(batch_size,-1),density_cloth.reshape(batch_size,-1)],dim=1)
        #offmesh_sdf = 0
        
        # offmesh_sdf_smpl = density_smpl.reshape(batch_size,-1)#density_smpl
        # offmesh_sdf_cloth = density_cloth.reshape(batch_size,-1)#density_cloth
        # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)#offmesh_sdf_smpl
        
        
        # d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        # sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      # inputs=sample_coordinates_smpl,
                                       # grad_outputs=d_output_smpl,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        # sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      # inputs=sample_coordinates_cloth,
                                       # grad_outputs=d_output_cloth,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)#sdfgradients_smpl
        #sdfgradients = 0
        
        # offmesh_sdf = density.reshape(batch_size,-1)
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdfgradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=sample_coordinates,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
                                       
        #density_smpl = torch.nan_to_num(density_smpl, nan=0.0, posinf=0.0, neginf=0.0)
        # density_cloth[density_cloth.abs()>0.5] = 0.5
        # density_smpl[density_smpl.abs()>0.5] = 0.5
        #print(torch.norm(sdfgradients[0], dim=-1))
        smplpara = torch.cat([smpl_params, betas], -1)
        
        # colorlatentsmpl = self.colorlatent_smpl(torch.LongTensor([0]).to('cuda'))
        # colorlatentcloth = self.colorlatent_cloth(torch.LongTensor([0]).to('cuda'))
        # feature_cloth = torch.cat([feature_cloth,colorlatentcloth[:,None].repeat(feature_cloth.shape[0],feature_cloth.shape[1],1)],dim=-1)
        # feature_smpl = torch.cat([feature_smpl,colorlatentsmpl[:,None].repeat(feature_smpl.shape[0],feature_smpl.shape[1],1)],dim=-1)
        # color_cloth = self.decoder_cloth(sample_coordinates_cloth, z_rend)#feature_cloth
        # color_smpl = self.decoder(sample_coordinates_smpl, z_rend)#feature_smpl
        
        #color = self.decoder(sample_coordinates_can, z_rend)
        
        # density_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32)#normlized to [-1,1]
        # density_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32) * 1e3
        # color_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32)#normlized to [-1,1]
        # color_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32) * 1e3
        # density_cloth[mask] = density_cloth0
        # density_smpl[mask] = density_smpl0
        # color_cloth[mask] = color_cloth0
        # color_smpl[mask] = color_smpl0
        
        #color_smpl, color_cloth = self.fetchingdoublenerf_fixgeometry_color(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        #color_smpl, density_smpl, color_cloth, density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        
        color = color.reshape(batch_size, num_rays, samples_per_ray, color.shape[-1])
        density = density.reshape(batch_size, num_rays, samples_per_ray, 1)              
        #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
        #density = self.laplacedensity(density.reshape(batch_size, -1,1))

        # density = sdf_to_alpha2(density.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #
        # density = density.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        # beta = self.beta_network().clamp(1e-9, 1e6)
        # alpha = self.sdf_to_alpha(density.reshape(batch_size, num_rays, samples_per_ray), beta)
        # alpha = alpha.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        # alpha[~sptneartag] = 0
        alpha = density
        # alpha = sdf_to_alpha(density.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s * 1.0))  
        # alpha = alpha.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        # color_smpl = color_smpl.reshape(batch_size, num_rays, samples_per_ray, color_smpl.shape[-1])
        # density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)              
        # #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())

        # density_smpl = sdf_to_alpha2(density_smpl.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #
        # density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)
        # #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
        # #density_smplmesh, _ = self.SDFnetwork_smpl(self.smpl_reduced_canon_vert.repeat(batch_size,1,1),self.canonparamshape.repeat(batch_size,1))

        # #density_smplmesh=sdf_to_alpha2(density_smplmesh, 0.05)
        # #print(torch.exp(self.ln_s * 1.0),density_smplmesh[0])
        
        # color_cloth = color_cloth.reshape(batch_size, num_rays, samples_per_ray, color_cloth.shape[-1])
        # density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)  
        # #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())        
        # density_cloth = sdf_to_alpha2(density_cloth.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s_cloth))  #
        # density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)
        # #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())  

        #rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color_smpl, density_smpl, color_cloth, density_cloth, renderclothmask, depths_coarse, batch_size)
        rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color, alpha, depths_coarse, batch_size)
        # rendered_color_full = rendered_color.new_zeros([batch_size,self.ray_sampler.N_samples_sqrt**2,3])
        # rendered_color_full[mask_at_box] = rendered_color.reshape(batch_size, -1, 3)
        # rendered_color = rendered_color_full.reshape(batch_size, -1)
        
        # with torch.no_grad(): 
            # meshpts_smpl, meshpts_cloth, samplepts_smpl, samplepts_cloth, meshpts_normal = self.samplingpoint_learningmeshsdf(self.rawtemplatesmpl, self.canon_clothvert)#.repeat(batch_size,1,1)cloth_reduced_canon
        
        # #samplepts.requires_grad_(True)
        # #_, samplepts_density = self.fetchingnerf(planes, samplepts)
        # # samplepts = torch.cat([samplepts_smpl,samplepts_cloth,meshpts_smpl,meshpts_cloth],dim=1)
        # # samplepts.requires_grad_(True)
        
        # samplepts_smpl.requires_grad_(True)
        # #samplepts_cloth.requires_grad_(True)
        
        # #samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
        # samplepts_density_smpl = self.Densitynetwork(samplepts_smpl)

        # # hypo_params_cloth = self.hyper_net_cloth(self.tempclothpara)
        # # embed_samplepts_cloth = self.embed_fn_fine(samplepts_cloth)
        # # print(embed_samplepts_cloth.shape)
        # # samplepts_density_cloth = self.sdf_network_cloth(embed_samplepts_cloth, params=hypo_params_cloth)
        
        # # hypo_params_smpl = self.hyper_net_smpl(self.canonparamshape)
        # # embed_samplepts_smpl = self.embed_fn_fine(samplepts_smpl)
        # # samplepts_density_smpl = self.sdf_network_cloth(embed_samplepts_smpl, params=hypo_params_smpl)

        # offmesh_sdf_smpl = samplepts_density_smpl.reshape(batch_size,-1)
        # #offmesh_sdf_cloth = samplepts_density_cloth.reshape(batch_size,-1)
        # offmesh_sdf = offmesh_sdf_smpl#torch.cat([,offmesh_sdf_cloth],dim=1)

        # d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        # sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      # inputs=samplepts_smpl,
                                       # grad_outputs=d_output_smpl,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # # # d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        # # # sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      # # # inputs=samplepts_cloth,
                                       # # # grad_outputs=d_output_cloth,
                                      # # # create_graph=True,
                                       # # # retain_graph=True,
                                       # # # only_inputs=True)[0]
        # # # sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)
        # sdfgradients = sdfgradients_smpl
        
        # #_, meshpts_smpl_sdf = self.fetchingnerf(planes, meshpts_smpl)
        # #_, meshpts_cloth_sdf = self.fetchingnerf(planes, meshpts_cloth)
        # # meshpts_smpl.requires_grad_(True)
        # # meshpts_cloth.requires_grad_(True)
        # #meshpts_smpl_sdf, meshpts_cloth_sdf = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], meshpts_smpl, meshpts_cloth)
        # meshpts_cloth_sdf = self.Densitynetwork(meshpts_cloth)
        # meshpts_smpl_sdf = self.Densitynetwork(meshpts_smpl)
        
        # # embed_meshpts_cloth = self.embed_fn_fine(meshpts_cloth_sdf)
        # # meshpts_cloth_sdf = self.sdf_network_cloth(embed_meshpts_cloth, params=hypo_params_cloth)
        
        # # embed_meshpts_smpl = self.embed_fn_fine(meshpts_smpl_sdf)
        # # meshpts_smpl_sdf = self.sdf_network_cloth(embed_meshpts_smpl, params=hypo_params_smpl)


        # meshpts_smpl_sdf = meshpts_smpl_sdf.reshape(batch_size,-1)
        # meshpts_cloth_sdf = meshpts_cloth_sdf.reshape(batch_size,-1)
        # #print(meshpts_smpl_sdf[0])       
        # # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth,meshpts_smpl_sdf,meshpts_cloth_sdf],dim=1)
                
        # # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # # sdf_gradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # # inputs=samplepts,
                                       # # grad_outputs=d_output,
                                      # # create_graph=True,
                                       # # retain_graph=True,
                                       # # only_inputs=True)[0]
                                       
        # # sampleptsnum = samplepts_smpl.shape[1]+samplepts_cloth.shape[1]
        # # sdfgradients =  sdf_gradients[:,:sampleptsnum,:]          
        # # meshptsgradients =  sdf_gradients[:,sampleptsnum:,:]   
        
        # # output_smpl = torch.ones_like(meshpts_smpl_sdf, requires_grad=False, device=meshpts_smpl_sdf.device)#
        # # meshptsgradients_smpl = torch.autograd.grad(outputs=meshpts_smpl_sdf,
                                      # # inputs=meshpts_smpl,
                                       # # grad_outputs=output_smpl,
                                      # # create_graph=True,
                                       # # retain_graph=True,
                                       # # only_inputs=True)[0]
        # # output_cloth = torch.ones_like(meshpts_cloth_sdf, requires_grad=False, device=meshpts_cloth_sdf.device)#
        # # meshptsgradients_cloth = torch.autograd.grad(outputs=meshpts_cloth_sdf,
                                      # # inputs=meshpts_cloth,
                                       # # grad_outputs=output_cloth,
                                      # # create_graph=True,
                                       # # retain_graph=True,
                                       # # only_inputs=True)[0]
        # # meshptsgradients = torch.cat([meshptsgradients_smpl,meshptsgradients_cloth],dim=1)
        # #meshptsgradients = 0
        
        #if 0:    
        offmesh_sdf = 0
        sdfgradients = 0
        meshpts_smpl_sdf = 0
        meshpts_cloth_sdf = 0
        meshptsgradients = 0
        meshpts_normal = 0
        
        
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients#rgb_final, depth_final, weights.sum(2)
    
    def singlenerf_rendering_deepfashion_fixgeometry_color_chunk(self, frameidx, cropimg, Nc, Nf, z_rend, planes, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):
                        
        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        batch_size = smpl_params.shape[0]
        
        #self.sample_canonshape(batch_size)
        
        #deforming with cloth and smpl
        # smpl_current = self.deformingsmpl_LBS(A, Th)
        # smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)
        smploutput = self.smpl.forward(betas, smpl_body_pose, smpl_orient, Th)#        
        smpl_current = smploutput.vertices
        #smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)
        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        # cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)#
        # cloth_reduced_current = self.cloth_reduced(cloth_current)
        
       
        # combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        #combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        
        # npvertices = combinedmeshvert[0].detach().cpu().numpy()                        
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = f"test/combined_{frameidx[0].item():0>4}.obj"
        # mesh.export(result_path)
        
        # data_root = '../../neuralbody-deformation-occupancy-fixnerf-our-generalization/tools/data/magdalena20000-allviews/'
        # smplvert_path = os.path.join(data_root, 'vertices',
                                   # '{}.npy'.format(frameidx[0].item()))
        # smplvert = np.load(smplvert_path).astype(np.float32)
        # smplmesh = trimesh.Trimesh(smplvert, self.smplfaces.detach().cpu().numpy(), process=False)
        # result_path = f"test/smpl_{frameidx[0].item():0>4}.obj"
        # smplmesh.export(result_path)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
                
        # with torch.no_grad(): 
            # clothmask, meshmask = self.render_clothmask(self.meshface.repeat(batch_size,1,1), combinedmeshvert, extrinsics, intrinsic)
            # renderclothmask = clothmask#.unsqueeze(-1).repeat(1,1,1,Nc).view(batch_size,-1,Nc,1)
        # mesh_mask = clothmask[0].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        # mesh_mask = clothmask[1].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[1].item()), mesh_mask*255)
        renderclothmask = 0
        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        ray_start = near
        ray_end = far
        # k_iter = self.ray_sampler.iterations // 1000 * 3        
        # print(self.ray_sampler.max_scale * exp(-k_iter*self.ray_sampler.scale_anneal))
        # rays_o_s = []
        # rays_d_s = []
        # ray_start_s = []
        # ray_end_s = []
        # #renderclothmask_s = []renderclothmask_i,
        # for i in range(batch_size):       
            # rays_o_i, rays_d_i, ray_start_i, ray_end_i, _, _ = self.ray_sampler(512, 512, ray_origins[i].view(512,512,3), ray_directions[i].view(512,512,3), ray_start[i].view(512,512,1), ray_end[i].view(512,512,1), cropimg[i])#renderclothmask[i], meshmask[i]
            # rays_o_s.append(rays_o_i[None])
            # rays_d_s.append(rays_d_i[None])
            # ray_start_s.append(ray_start_i[None])
            # ray_end_s.append(ray_end_i[None])
            # #renderclothmask_s.append(renderclothmask_i[None])
        # allray_origins = torch.cat(rays_o_s)        
        # allray_directions = torch.cat(rays_d_s)
        # allray_start = torch.cat(ray_start_s)
        # allray_end = torch.cat(ray_end_s)
        # allrenderclothmask = torch.cat(renderclothmask_s)
        
        # renderclothmask0 = renderclothmask[0].reshape(32,32)#low resolution for nerf rendering
        # renderclothmask0 = renderclothmask0.detach().cpu().numpy()
        # cv2.imwrite('test/real_mask0.png', renderclothmask0*255)
        #allrenderclothmask = allrenderclothmask.unsqueeze(-1).repeat(1,1,Nc,1)
        
        allray_origins = ray_origins       
        allray_directions = ray_directions
        allray_start = ray_start
        allray_end = ray_end
        #allrenderclothmask = renderclothmask
        
        # Create stratified depth samples

        n_batch, n_pixel = allray_origins.shape[:2]
        chunk = 2048
        rendered_color = []
        rendered_mask = []
        rendered_disparity = []
        weights = []
        for i in range(0, n_pixel, chunk):
            ray_origins = allray_origins[:, i:i + chunk]
            ray_directions = allray_directions[:, i:i + chunk]
            ray_start = allray_start[:, i:i + chunk]
            ray_end = allray_end[:, i:i + chunk]
            renderclothmask = 0#allrenderclothmask[:, i:i + chunk]
            
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

            _, num_rays, samples_per_ray, _ = depths_coarse.shape
            bs_expand = batch_size 
                   
            # Coarse Pass
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       
            
            #mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
            
            
            # sample_coordinates_can = self.get_canonical_coordinates(
                # sample_coordinates,
                # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
                # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)) 
                # #mask = mask            
            # )
            # #sample_coordinates = sample_coordinates_smpl
            
            # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
                # sample_coordinates,
                # cloth_reduced_current,
                # self.cloth_reduced_canon.repeat(batch_size,1,1)
                # #mask            
            # )
            
            # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
            # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
            # smpl_distmin = smpl_ptsdistmin[0]
            # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
            # cloth_ptsdist = torch.cdist(sample_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
            # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
            # cloth_distmin = cloth_ptsdistmin[0]
            # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
                        
            # cloth_neartag = cloth_distmin<smpl_distmin
           
            # sample_coordinates = sample_coordinates_smpl
            # sample_coordinates.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
            # sample_coordinates = sample_coordinates.view(batch_size,-1,3)
            
            nnvidxtag = [] 
            smpl_nnvidx = [] 
            cloth_nnvidx = []
            sptneartag = []            
            for i in range(0, batch_size, 1):
                smpl_ptsdist = torch.cdist(sample_coordinates[i], smpl_current[i], p=2)#deformedpersonsmpl
                smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
                smpl_distmin = smpl_ptsdistmin[0]
                smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1)
                smpl_nnvidx.append(smpl_minidx[None])
                
                # cloth_ptsdist = torch.cdist(sample_coordinates[i], cloth_current[i], p=2)#deformedpersonsmpl
                # cloth_ptsdistmin = torch.min(cloth_ptsdist, 1)
                # cloth_distmin = cloth_ptsdistmin[0]
                # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1)
                # cloth_nnvidx.append(cloth_minidx[None])
                # idxtag = cloth_distmin<smpl_distmin
                # nnvidxtag.append(idxtag[None])
                
                ptdist = smpl_distmin<0.04#torch.logical_or(cloth_distmin<0.03, smpl_distmin<0.03)
                sptneartag.append(ptdist[None])
        
            sptneartag = torch.cat(sptneartag).view(batch_size,-1,Nc,1)    
        
            smpl_nnvidx = torch.cat(smpl_nnvidx)   
            # cloth_nnvidx = torch.cat(cloth_nnvidx)
            # nnvidxtag = torch.cat(nnvidxtag)        
            sample_coordinates_can = self.inversedeforming_samplepoints_LBS(sample_coordinates, smpl_nnvidx, smpl_current, A, Th, betas, smpl_params)
            # sample_coordinates_can2 = self.inversedeforming_samplepoints_cloth(sample_coordinates, cloth_nnvidx, cloth_current, A, Th)       
            # sample_coordinates_can[nnvidxtag] = sample_coordinates_can2[nnvidxtag]
            
            #displacement
            # smplpara = torch.cat([smpl_params, betas], -1)
            # samplepts_disp_smpl = self.displace_smpl(sample_coordinates, smplpara)
            # samplepts_disp_cloth = self.displace_cloth(sample_coordinates, smplpara)
            # samplepts_disp = torch.cat([samplepts_disp_smpl, samplepts_disp_cloth], 1)
            # sample_coordinates_smpl = sample_coordinates_smpl+samplepts_disp_smpl
            # sample_coordinates_cloth = sample_coordinates_cloth+samplepts_disp_cloth
            samplepts_disp = 0
            
            #density, feature = self.SDFnetwork(sample_coordinates, self.canonparamshape.repeat(batch_size,1))#
            #density = self.Densitynetwork(sample_coordinates_can)
            density = self.Densitynetwork(sample_coordinates_can)
            #incdensity = self.IncDensitynetwork(sample_coordinates_can)
            color, incdensity = self.decoder(sample_coordinates_can, z_rend[:,:128])
            density = density + incdensity
            #color, density = self.decoder(sample_coordinates_can, z_rend)
            # samplepts_incsdf_smpl = self.incSDFnetwork_smpl(sample_coordinates_smpl)
            # samplepts_incsdf_cloth = self.incSDFnetwork_cloth(sample_coordinates_cloth)
            # samplepts_disp = torch.cat([samplepts_incsdf_smpl, samplepts_incsdf_cloth], 1)#samplepts_incsdf_smpl
            # density_smpl = density_smpl+samplepts_incsdf_smpl
            # density_cloth = density_cloth+samplepts_incsdf_cloth
            samplepts_disp = 0
            
            
            #smplpara = torch.cat([smpl_params, betas], -1)
            
            #color = self.decoder(sample_coordinates_can, z_rend)
            
            color = color.reshape(batch_size, num_rays, samples_per_ray, color.shape[-1])
            density = density.reshape(batch_size, num_rays, samples_per_ray, 1)              
            #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
            
            # density = sdf_to_alpha2(density.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #torch.exp()
            # density = density.reshape(batch_size, num_rays, samples_per_ray, 1)
            #alpha = density
            beta = self.beta_network().clamp(1e-9, 1e6)
            alpha = self.sdf_to_alpha(density.reshape(batch_size, num_rays, samples_per_ray), beta)
            alpha = alpha.reshape(batch_size, num_rays, samples_per_ray, 1)
            
            alpha[~sptneartag] = 0
            #rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color_smpl, density_smpl, color_cloth, density_cloth, renderclothmask, depths_coarse, batch_size)
            rendered_color_chunk, rendered_mask_chunk, rendered_disparity_chunk, weights_chunk = self.ray_marcher(color, alpha, depths_coarse, batch_size)
                
            rendered_color.append(rendered_color_chunk)
            rendered_mask.append(rendered_mask_chunk)
            rendered_disparity.append(rendered_disparity_chunk)
            weights.append(weights_chunk)
            
            #if 0:    
            offmesh_sdf = 0
            sdfgradients = 0
            meshpts_smpl_sdf = 0
            meshpts_cloth_sdf = 0
            meshptsgradients = 0
            meshpts_normal = 0
        
        rendered_color = torch.cat(rendered_color, dim=1)
        rendered_mask = torch.cat(rendered_mask, dim=1)
        rendered_disparity = torch.cat(rendered_disparity, dim=1)
        weights = torch.cat(weights, dim=1)
        
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients#rgb_final, depth_final, weights.sum(2)
    
    def singlenerf_rendering_deepfashion_fixgeometry_color(self, frameidx, cropimg, Nc, Nf, z_rend, planes, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):
        
        # R = extrinsics[:,:3,:3] 
        # T = extrinsics[:,:3,3:]

        # pose_to_camera = torch.bmm(pose_to_world[..., :3, 3], R.permute(0,2,1)) + T.permute(0,2,1)
        # pose_2d = torch.matmul(pose_to_camera, intrinsic.permute(0,2,1))
        # pose_2d = pose_2d[:, :, :2] / pose_2d[:, :, 2][...,None] 
        #location = pose_2d[:, 15, :2]
        # jointidx = torch.randint(24,(1,))[0]
        # location = pose_2d[:, jointidx, :2]
        
        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        batch_size = smpl_params.shape[0]
        
        #self.sample_canonshape(batch_size)
        
        #deforming with cloth and smpl
        # smpl_current = self.deformingsmpl_LBS(A, Th)
        # smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)

        smploutput = self.smpl.forward(betas, smpl_body_pose, smpl_orient, Th)#        
        smpl_current = smploutput.vertices
        #smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)
        
        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        # cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)#
        # cloth_reduced_current = self.cloth_reduced(cloth_current)
        
       
        # combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        #combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        
        # npvertices = combinedmeshvert[0].detach().cpu().numpy()                        
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = f"test/combined_{frameidx[0].item():0>4}.obj"
        # mesh.export(result_path)
        
        # data_root = '../../neuralbody-deformation-occupancy-fixnerf-our-generalization/tools/data/magdalena20000-allviews/'
        # smplvert_path = os.path.join(data_root, 'vertices',
                                   # '{}.npy'.format(frameidx[0].item()))
        # smplvert = np.load(smplvert_path).astype(np.float32)
        # smplmesh = trimesh.Trimesh(smplvert, self.smplfaces.detach().cpu().numpy(), process=False)
        # result_path = f"test/smpl_{frameidx[0].item():0>4}.obj"
        # smplmesh.export(result_path)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        
        # meshmask = self.rasterize_eg3d(self.smplfaces, smpl_current, extrinsics, intrinsic) 
        # mesh_mask = meshmask[0][:,:256].detach().cpu().numpy()
        # cv2.imwrite('result/deepcap/test/rendermeshmask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        
        # smpl_camera = torch.bmm(smpl_current, R.permute(0,2,1)) + T.permute(0,2,1)
        # smpl_2d = torch.matmul(smpl_camera, intrinsic.permute(0,2,1))
        # smpl_2d = smpl_2d[:, :, :2] / smpl_2d[:, :, 2][...,None] 
        # for t in range(0, 6890):
           # cropimg[0,:,smpl_2d[0,t,1].long(),smpl_2d[0,t,0].long()] = 0  
           # cropimg[1,:,smpl_2d[1,t,1].long(),smpl_2d[1,t,0].long()] = 0   
        # crop_img = cropimg.cpu().numpy()[0].transpose(1, 2, 0)
        # crop_img = crop_img * 127.5 + 127.5
        # cv2.imwrite('result/deepcap/test/crop_img{:04d}.png'.format(frameidx[0].item()), crop_img)    
        
        #with torch.no_grad(): 
            #clothmask, meshmask = self.render_clothmask(self.meshface.repeat(batch_size,1,1), combinedmeshvert, extrinsics, intrinsic)
            # renderclothmask = clothmask#.unsqueeze(-1).repeat(1,1,1,Nc).view(batch_size,-1,Nc,1)
        # mesh_mask = clothmask[0].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        # mesh_mask = clothmask[1].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[1].item()), mesh_mask*255)
        renderclothmask = 0
        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        ray_start = near
        ray_end = far
        # k_iter = self.ray_sampler.iterations // 1000 * 3        
        # print(self.ray_sampler.max_scale * exp(-k_iter*self.ray_sampler.scale_anneal))
        # rays_o_s = []
        # rays_d_s = []
        # ray_start_s = []
        # ray_end_s = []
        # mask_at_box_s = []
        # #renderclothmask_s = [], renderclothmask_i, mask_at_box_i
        # for i in range(batch_size):       
            # rays_o_i, rays_d_i, ray_start_i, ray_end_i, _, _ = self.ray_sampler(512, 256, ray_origins[i].view(512,256,3), ray_directions[i].view(512,256,3), ray_start[i].view(512,256,1), ray_end[i].view(512,256,1), cropimg[i],pose_2d[i])#, 'local', mask_at_box[i].view(512,512,1)renderclothmask[i], meshmask[i]
            # rays_o_s.append(rays_o_i[None])
            # rays_d_s.append(rays_d_i[None])
            # ray_start_s.append(ray_start_i[None])
            # ray_end_s.append(ray_end_i[None])
            # #mask_at_box_s.append(mask_at_box_i[None])
            # #renderclothmask_s.append(renderclothmask_i[None])
        # #mask_at_box = torch.cat(mask_at_box_s)            
        # ray_origins = torch.cat(rays_o_s)#[mask_at_box].view(batch_size,-1,3)        
        # ray_directions = torch.cat(rays_d_s)#[mask_at_box].view(batch_size,-1,3) 
        # ray_start = torch.cat(ray_start_s)#[mask_at_box].view(batch_size,-1,1) 
        # ray_end = torch.cat(ray_end_s)#[mask_at_box].view(batch_size,-1,1) 

        #renderclothmask = torch.cat(renderclothmask_s)
        
        # renderclothmask0 = renderclothmask[0].reshape(32,32)#low resolution for nerf rendering
        # renderclothmask0 = renderclothmask0.detach().cpu().numpy()
        # cv2.imwrite('test/real_mask0.png', renderclothmask0*255)
        
        #renderclothmask = renderclothmask.unsqueeze(-1).repeat(1,1,Nc,1)
        
        # Create stratified depth samples
        depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

        _, num_rays, samples_per_ray, _ = depths_coarse.shape
        bs_expand = batch_size 
               
        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        #sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       
        mask_at_box = self.tensor_expand(mask_at_box)
        mask = mask_at_box.unsqueeze(-1).repeat(1,1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        #sample_coordinates.requires_grad_(True)unsqueeze(-1).repeat(1,1,Nc)
        
        # sample_coordinates_can = self.get_canonical_coordinates(
            # sample_coordinates,
            # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)) 
            # #mask = mask            
        # )
        # #sample_coordinates = sample_coordinates_smpl
        # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
            # sample_coordinates,
            # cloth_reduced_current,
            # self.cloth_reduced_canon.repeat(batch_size,1,1)
            # #mask            
        # )
        
        # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
        # cloth_ptsdist = torch.cdist(sample_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_distmin = cloth_ptsdistmin[0]
        # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
                    
        # cloth_neartag = cloth_distmin<smpl_distmin
       
        # sample_coordinates_can = sample_coordinates_smpl
        # sample_coordinates_can.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
        # sample_coordinates_can = sample_coordinates_can.view(batch_size,-1,3)
        
        # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_current, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # nnvidx = torch.squeeze(smpl_ptsdistmin[1], -1)
        #sample_coordinates.requires_grad_(True)
        
        sample_coordinates = sample_coordinates[mask][None]

        if self.istraining:
           sample_coordinates.requires_grad_(True)
        nnvidxtag = [] 
        smpl_nnvidx = [] 
        cloth_nnvidx = []
        sptneartag = []       
                
        for i in range(0, batch_size, 1):
            #smpl_minidx = sample_coordinates.new_zeros([sample_coordinates.shape[1]], dtype=torch.long)              
            smpl_ptsdist = torch.cdist(sample_coordinates[i], smpl_current[i], p=2)#[mask[i]]deformedpersonsmpl
            smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            smpl_distmin = smpl_ptsdistmin[0]
            smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1)#[mask[i]]
            smpl_nnvidx.append(smpl_minidx[None])
            
            # cloth_ptsdist = torch.cdist(sample_coordinates[i], cloth_current[i], p=2)#deformedpersonsmpl
            # cloth_ptsdistmin = torch.min(cloth_ptsdist, 1)
            # cloth_distmin = cloth_ptsdistmin[0]
            # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1)
            # cloth_nnvidx.append(cloth_minidx[None])
            # idxtag = cloth_distmin<smpl_distmin
            # nnvidxtag.append(idxtag[None])
            
            # ptdist = torch.logical_or(cloth_distmin<0.03, smpl_distmin<0.03)
            ptdist = smpl_distmin<0.05
            sptneartag.append(ptdist[None])
        
        sptneartag = torch.cat(sptneartag).view(batch_size,-1,Nc,1)            
        smpl_nnvidx = torch.cat(smpl_nnvidx)   
        # cloth_nnvidx = torch.cat(cloth_nnvidx)
        # nnvidxtag = torch.cat(nnvidxtag)        #[mask].view(batch_size,-1,3)
        sample_coordinates_can = self.inversedeforming_samplepoints_LBS(sample_coordinates, smpl_nnvidx, smpl_current, A, Th, betas, smpl_params)
        #sample_coordinates_can2 = self.inversedeforming_samplepoints_cloth(sample_coordinates, cloth_nnvidx, cloth_current, A, Th)       
        #sample_coordinates_can[nnvidxtag] = sample_coordinates_can2[nnvidxtag]
        
        #displacement
        # smplpara = torch.cat([smpl_params, betas], -1)
        # samplepts_disp_smpl = self.displace_smpl(sample_coordinates, smplpara)
        # samplepts_disp_cloth = self.displace_cloth(sample_coordinates, smplpara)
        # samplepts_disp = torch.cat([samplepts_disp_smpl, samplepts_disp_cloth], 1)
        # sample_coordinates_smpl = sample_coordinates_smpl+samplepts_disp_smpl
        # sample_coordinates_cloth = sample_coordinates_cloth+samplepts_disp_cloth
        samplepts_disp = 0
        
        # meshvert_def = torch.cat([self.canonclothvert, self.rawtemplatesmpl], dim=1)#self.deformedcloth
        # npvertices = meshvert_def[0].detach().cpu().numpy()
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/canverts{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = self.rawtemplatesmpl[0].detach().cpu().numpy()
        # npfaces = self.smplfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/smpl_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = smpl_current[0].detach().cpu().numpy()
        # npfaces = self.smplfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/smpl{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # # npvertices = self.canonclothvert[0].detach().cpu().numpy()
        # # npfaces = self.clothfaces.detach().cpu().numpy()
        # # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # # result_path = os.path.join('test/cloth_can{:04d}.obj'.format(frameidx[0].item()))
        # # mesh.export(result_path)
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = sample_coordinates_can[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/sample_coordinates_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        
        # sample_coordinates_smpl.requires_grad_(True)
        # sample_coordinates_cloth.requires_grad_(True)
        # #print(torch.isnan(sample_coordinates_cloth).any() or torch.isinf(sample_coordinates_cloth).any())
        # density_cloth, feature_cloth = self.SDFnetwork_cloth(sample_coordinates_cloth,self.shapepara.repeat(batch_size,1))#tempclothpara     
        
        # #density_cloth = torch.nan_to_num(density_cloth, nan=0.0, posinf=0.0, neginf=0.0)
        
        # density_smpl, feature_smpl = self.SDFnetwork_smpl(sample_coordinates_smpl,self.canonparamshape.repeat(batch_size,1))
        # if self.istraining:
            # sample_coordinates_can.requires_grad_(True)
        #density, feature = self.SDFnetwork(sample_coordinates, self.canonparamshape.repeat(batch_size,1))#
        #density = self.Densitynetwork(sample_coordinates_can)
        #color, density = self.decoder(sample_coordinates_can, z_rend)
        density = self.Densitynetwork(sample_coordinates_can)
        #incdensity = self.IncDensitynetwork(sample_coordinates_can)
        color, incdensity = self.decoder(sample_coordinates_can, z_rend[:,:128])
        density = density + incdensity
        
        # samplepts_incsdf_smpl = self.incSDFnetwork_smpl(sample_coordinates_smpl)
        # samplepts_incsdf_cloth = self.incSDFnetwork_cloth(sample_coordinates_cloth)
        # samplepts_disp = torch.cat([samplepts_incsdf_smpl, samplepts_incsdf_cloth], 1)#samplepts_incsdf_smpl
        # density_smpl = density_smpl+samplepts_incsdf_smpl
        # density_cloth = density_cloth+samplepts_incsdf_cloth
        samplepts_disp = 0
        
        # sample_coordinates_cloth.requires_grad_(True)
        # #samplepts_density_smpl, samplepts_density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], samplepts_smpl, samplepts_cloth)
        # samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
        # samplepts_density_smpl,_ = self.SDFnetwork_smpl(samplepts_smpl,self.canonparamshape)

        # offmesh_sdf_smpl = samplepts_incsdf_smpl.reshape(batch_size,-1)#density_smpl
        # offmesh_sdf_cloth = samplepts_incsdf_cloth.reshape(batch_size,-1)#density_cloth
        # #offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)#offmesh_sdf_smpl
        # offmesh_sdf = torch.cat([density_smpl.reshape(batch_size,-1),density_cloth.reshape(batch_size,-1)],dim=1)
        #offmesh_sdf = 0
        
        # offmesh_sdf_smpl = density_smpl.reshape(batch_size,-1)#density_smpl
        # offmesh_sdf_cloth = density_cloth.reshape(batch_size,-1)#density_cloth
        # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)#offmesh_sdf_smpl
        
        
        # d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        # sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      # inputs=sample_coordinates_smpl,
                                       # grad_outputs=d_output_smpl,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        # sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      # inputs=sample_coordinates_cloth,
                                       # grad_outputs=d_output_cloth,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)#sdfgradients_smpl
        #sdfgradients = 0
        
        # offmesh_sdf = density.reshape(batch_size,-1)
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdfgradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=sample_coordinates,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
                                       
        #density_smpl = torch.nan_to_num(density_smpl, nan=0.0, posinf=0.0, neginf=0.0)
        # density_cloth[density_cloth.abs()>0.5] = 0.5
        # density_smpl[density_smpl.abs()>0.5] = 0.5
        #print(torch.norm(sdfgradients[0], dim=-1))
        smplpara = torch.cat([smpl_params, betas], -1)
        
        # colorlatentsmpl = self.colorlatent_smpl(torch.LongTensor([0]).to('cuda'))
        # colorlatentcloth = self.colorlatent_cloth(torch.LongTensor([0]).to('cuda'))
        # feature_cloth = torch.cat([feature_cloth,colorlatentcloth[:,None].repeat(feature_cloth.shape[0],feature_cloth.shape[1],1)],dim=-1)
        # feature_smpl = torch.cat([feature_smpl,colorlatentsmpl[:,None].repeat(feature_smpl.shape[0],feature_smpl.shape[1],1)],dim=-1)
        # color_cloth = self.decoder_cloth(sample_coordinates_cloth, z_rend)#feature_cloth
        # color_smpl = self.decoder(sample_coordinates_smpl, z_rend)#feature_smpl
        
        #color = self.decoder(sample_coordinates_can, z_rend)
        
        # density_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32)#normlized to [-1,1]
        # density_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32) * 1e3
        # color_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32)#normlized to [-1,1]
        # color_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32) * 1e3
        # density_cloth[mask] = density_cloth0
        # density_smpl[mask] = density_smpl0
        # color_cloth[mask] = color_cloth0
        # color_smpl[mask] = color_smpl0
        
        #color_smpl, color_cloth = self.fetchingdoublenerf_fixgeometry_color(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        #color_smpl, density_smpl, color_cloth, density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        
        color = color.reshape(batch_size, -1, samples_per_ray, color.shape[-1])#num_rays
        density = density.reshape(batch_size, -1, samples_per_ray, 1)              
        #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
        #density = self.laplacedensity(density.reshape(batch_size, -1,1))

        # density = sdf_to_alpha2(density.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #
        # density = density.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        # beta = self.beta_network().clamp(1e-9, 1e6)
        # alpha = self.sdf_to_alpha(density.reshape(batch_size, num_rays, samples_per_ray), beta)
        # alpha = alpha.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        alpha = torch.sigmoid(-density.reshape(batch_size, -1, samples_per_ray,1) / self.ln_s) / self.ln_s
        alpha[~sptneartag] = 0
        
        color_full = color.new_zeros([batch_size,self.ray_sampler.N_samples_sqrt*2*self.ray_sampler.N_samples_sqrt*Nc,3])
        alpha_full = density.new_zeros([batch_size,self.ray_sampler.N_samples_sqrt*2*self.ray_sampler.N_samples_sqrt*Nc,1])
        color_full[mask] = color.reshape(batch_size, -1, color.shape[-1])
        alpha_full[mask] = alpha.reshape(batch_size, -1, 1)   
        color = color_full.reshape(batch_size, -1, samples_per_ray, color.shape[-1])#num_rays
        alpha = alpha_full.reshape(batch_size, -1, samples_per_ray, 1) 
                
        #alpha = density
        # alpha = sdf_to_alpha(density.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s * 1.0))  
        # alpha = alpha.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        # color_smpl = color_smpl.reshape(batch_size, num_rays, samples_per_ray, color_smpl.shape[-1])
        # density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)              
        # #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())

        # density_smpl = sdf_to_alpha2(density_smpl.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #
        # density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)
        # #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
        # #density_smplmesh, _ = self.SDFnetwork_smpl(self.smpl_reduced_canon_vert.repeat(batch_size,1,1),self.canonparamshape.repeat(batch_size,1))

        # #density_smplmesh=sdf_to_alpha2(density_smplmesh, 0.05)
        # #print(torch.exp(self.ln_s * 1.0),density_smplmesh[0])
        
        # color_cloth = color_cloth.reshape(batch_size, num_rays, samples_per_ray, color_cloth.shape[-1])
        # density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)  
        # #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())        
        # density_cloth = sdf_to_alpha2(density_cloth.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s_cloth))  #
        # density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)
        # #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())  

        #rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color_smpl, density_smpl, color_cloth, density_cloth, renderclothmask, depths_coarse, batch_size)
        rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color, alpha, depths_coarse, batch_size)
        # rendered_color_full = rendered_color.new_zeros([batch_size,self.ray_sampler.N_samples_sqrt*2*self.ray_sampler.N_samples_sqrt,3])
        # rendered_color_full[mask_at_box] = rendered_color.reshape(batch_size, -1, 3)
        # rendered_color = rendered_color_full.reshape(batch_size, -1)
        if self.istraining:
            # with torch.no_grad(): 
                # meshpts_smpl, samplepts_smpl, meshpts_normal = self.samplingpoint_learningmeshsdf_smpl(self.rawtemplatesmpl)#.repeat(batch_size,1,1)cloth_reduced_canon
            meshpts_normal = 0
            #samplepts.requires_grad_(True)
            #_, samplepts_density = self.fetchingnerf(planes, samplepts)
            # samplepts = torch.cat([samplepts_smpl,samplepts_cloth,meshpts_smpl,meshpts_cloth],dim=1)
            # samplepts.requires_grad_(True)
            
            #sample_coordinates_can.requires_grad_(True)
            #samplepts_cloth.requires_grad_(True)
            
            #samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
            #samplepts_density_smpl = self.Densitynetwork(samplepts_smpl)
            # sample_can = sample_coordinates_can#[:,0:-1:20]
            # sample_can.requires_grad_(True)
            # _, samplepts_density_smpl = self.decoder(sample_can, z_rend[:,:128])
            samplepts_density_smpl = incdensity#
            samplepts_disp = samplepts_density_smpl
            
            # hypo_params_cloth = self.hyper_net_cloth(self.tempclothpara)
            # embed_samplepts_cloth = self.embed_fn_fine(samplepts_cloth)
            # print(embed_samplepts_cloth.shape)
            # samplepts_density_cloth = self.sdf_network_cloth(embed_samplepts_cloth, params=hypo_params_cloth)
            
            # hypo_params_smpl = self.hyper_net_smpl(self.canonparamshape)
            # embed_samplepts_smpl = self.embed_fn_fine(samplepts_smpl)
            # samplepts_density_smpl = self.sdf_network_cloth(embed_samplepts_smpl, params=hypo_params_smpl)

            offmesh_sdf_smpl = samplepts_density_smpl.reshape(batch_size,-1)
            #offmesh_sdf_cloth = samplepts_density_cloth.reshape(batch_size,-1)
            offmesh_sdf = offmesh_sdf_smpl#torch.cat([,offmesh_sdf_cloth],dim=1)

            d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
            sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                          inputs=sample_coordinates,
                                           grad_outputs=d_output_smpl,
                                          create_graph=True,
                                           retain_graph=True,
                                           only_inputs=True)[0]
            # # d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
            # # sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                          # # inputs=samplepts_cloth,
                                           # # grad_outputs=d_output_cloth,
                                          # # create_graph=True,
                                           # # retain_graph=True,
                                           # # only_inputs=True)[0]
            # # sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)
            sdfgradients = sdfgradients_smpl
            
            #_, meshpts_smpl_sdf = self.fetchingnerf(planes, meshpts_smpl)
            #_, meshpts_cloth_sdf = self.fetchingnerf(planes, meshpts_cloth)
            # meshpts_smpl.requires_grad_(True)
            # #meshpts_cloth.requires_grad_(True)
            # #meshpts_smpl_sdf, meshpts_cloth_sdf = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], meshpts_smpl, meshpts_cloth)
            # #meshpts_cloth_sdf = self.Densitynetwork(meshpts_cloth)
            # meshpts_smpl_sdf = self.Densitynetwork(meshpts_smpl)
            
            # # embed_meshpts_cloth = self.embed_fn_fine(meshpts_cloth_sdf)
            # # meshpts_cloth_sdf = self.sdf_network_cloth(embed_meshpts_cloth, params=hypo_params_cloth)
            
            # # embed_meshpts_smpl = self.embed_fn_fine(meshpts_smpl_sdf)
            # # meshpts_smpl_sdf = self.sdf_network_cloth(embed_meshpts_smpl, params=hypo_params_smpl)


            # meshpts_smpl_sdf = meshpts_smpl_sdf.reshape(batch_size,-1)
            #meshpts_cloth_sdf = meshpts_cloth_sdf.reshape(batch_size,-1)
            #print(meshpts_smpl_sdf[0])       
            # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth,meshpts_smpl_sdf,meshpts_cloth_sdf],dim=1)
                    
            # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
            # sdf_gradients = torch.autograd.grad(outputs=offmesh_sdf,
                                          # inputs=samplepts,
                                           # grad_outputs=d_output,
                                          # create_graph=True,
                                           # retain_graph=True,
                                           # only_inputs=True)[0]
                                           
            # sampleptsnum = samplepts_smpl.shape[1]+samplepts_cloth.shape[1]
            # sdfgradients =  sdf_gradients[:,:sampleptsnum,:]          
            # meshptsgradients =  sdf_gradients[:,sampleptsnum:,:]   
            
            # output_smpl = torch.ones_like(meshpts_smpl_sdf, requires_grad=False, device=meshpts_smpl_sdf.device)#
            # meshptsgradients_smpl = torch.autograd.grad(outputs=meshpts_smpl_sdf,
                                          # inputs=meshpts_smpl,
                                           # grad_outputs=output_smpl,
                                          # create_graph=True,
                                           # retain_graph=True,
                                           # only_inputs=True)[0]
            # # output_cloth = torch.ones_like(meshpts_cloth_sdf, requires_grad=False, device=meshpts_cloth_sdf.device)#
            # # meshptsgradients_cloth = torch.autograd.grad(outputs=meshpts_cloth_sdf,
                                          # # inputs=meshpts_cloth,
                                           # # grad_outputs=output_cloth,
                                          # # create_graph=True,
                                           # # retain_graph=True,
                                           # # only_inputs=True)[0]
            # meshptsgradients = meshptsgradients_smpl#torch.cat([meshptsgradients_smpl,meshptsgradients_cloth],dim=1)
            meshptsgradients = 0
        else:
            offmesh_sdf = 0
            sdfgradients = 0
            meshptsgradients = 0
            meshpts_normal = 0
        #if 0:    
        #offmesh_sdf = 0
        #sdfgradients = 0
        meshpts_smpl_sdf = 0
        meshpts_cloth_sdf = 0
        #meshptsgradients = 0
        #meshpts_normal = 0
               
        
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients#rgb_final, depth_final, weights.sum(2)
    
    def singlenerf_rendering_deepcap_geometry_color_deformation(self, frameidx, cropimg, Nc, Nf, z_rend, planes, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):
        
        # R = extrinsics[:,:3,:3] 
        # T = extrinsics[:,:3,3:]

        # pose_to_camera = torch.bmm(pose_to_world[..., :3, 3], R.permute(0,2,1)) + T.permute(0,2,1)
        # pose_2d = torch.matmul(pose_to_camera, intrinsic.permute(0,2,1))
        # pose_2d = pose_2d[:, :, :2] / pose_2d[:, :, 2][...,None] 
        # location = pose_2d[:, 15, :2]
        
        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        batch_size = smpl_params.shape[0]
        
        #smpllatent = torch.cat([smpl_params, betas],dim=1)
        clothlatent = torch.cat([smpl_params, betas, self.tempclothpara.repeat(batch_size,1)], -1)

        self.deformation_affine, self.deformation_transl = self.predicting_deformation(clothlatent)
        #self.deformation_affine_smpl, self.deformation_transl_smpl = self.predicting_deformation_smpl(smpllatent)
    
        #clothvert = self.net.update_clothshape()
        self.update_embeddedgraph_cloth(self.canonclothvert.repeat(batch_size,1,1))
        #self.update_embeddedgraph_smpl(self.rawtemplatesmpl.repeat(batch_size,1,1))
        
        self.smoothloss = self.deformationsmoothloss()
        #self.smoothloss_smpl = self.deformationsmoothloss_smpl()

        #smpl_current, smplgraphdeformedverts = self.deformingsmpl_graphdeform_LBS(self.rawtemplatesmpl.repeat(batch_size,1,1),A, Th)
        smpl_current = self.deformingsmpl_LBS(A, Th)
        
        cloth_current, graphdeformedverts = self.deformingcloth_graphdeform_LBS(self.canonclothvert.repeat(batch_size,1,1),A, Th)
        
        #self.deltadeformloss_smpl = self.deform_crit(smplgraphdeformedverts,self.rawtemplatesmpl.repeat(batch_size,1,1))#torch.zeros_like(smplgraphdeformedverts)
        self.deltadeformloss = self.deform_crit(graphdeformedverts,self.canonclothvert.repeat(batch_size,1,1))#torch.zeros_like(graphdeformedverts)
        
        #self.interploss = self.computeinterpenetrationloss_posedsmpl(graphdeformedverts, self.rawtemplatesmpl.repeat(batch_size,1,1), smpl_current, cloth_current)
        
        if frameidx[0].item()%500==0:
            npvertices = smpl_current[0].detach().cpu().numpy()
            npfaces = self.smplfaces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            result_path = os.path.join('result/deepcap/test/smpl{:04d}.obj'.format(frameidx[0].item()))
            mesh.export(result_path)
            npvertices = cloth_current[0].detach().cpu().numpy()
            npfaces = self.clothfaces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
            result_path = os.path.join('result/deepcap/test/cloth{:04d}.obj'.format(frameidx[0].item()))
            mesh.export(result_path)
        
        # #deforming with cloth and smpl
        # smpl_current = self.deformingsmpl_LBS(A, Th)
        # smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)

        # # self.predicting_deformation(latentws)
        # # canondeformedcloth = self.deformingtemplate()
        # # deform_smoothloss = self.deformationsmoothloss()
        
        # cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)#
        # cloth_reduced_current = self.cloth_reduced(cloth_current)
        
       
        # combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        #combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        
        # npvertices = combinedmeshvert[0].detach().cpu().numpy()                        
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = f"test/combined_{frameidx[0].item():0>4}.obj"
        # mesh.export(result_path)
        
        # data_root = '../../neuralbody-deformation-occupancy-fixnerf-our-generalization/tools/data/magdalena20000-allviews/'
        # smplvert_path = os.path.join(data_root, 'vertices',
                                   # '{}.npy'.format(frameidx[0].item()))
        # smplvert = np.load(smplvert_path).astype(np.float32)
        # smplmesh = trimesh.Trimesh(smplvert, self.smplfaces.detach().cpu().numpy(), process=False)
        # result_path = f"test/smpl_{frameidx[0].item():0>4}.obj"
        # smplmesh.export(result_path)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
                
        # with torch.no_grad(): 
            # clothmask, meshmask = self.render_clothmask(self.meshface.repeat(batch_size,1,1), combinedmeshvert, extrinsics, intrinsic)
            # renderclothmask = clothmask#.unsqueeze(-1).repeat(1,1,1,Nc).view(batch_size,-1,Nc,1)
        # mesh_mask = clothmask[0].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        # mesh_mask = clothmask[1].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[1].item()), mesh_mask*255)
        renderclothmask = 0
        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        ray_start = near
        ray_end = far
        # k_iter = self.ray_sampler.iterations // 1000 * 3        
        # print(self.ray_sampler.max_scale * exp(-k_iter*self.ray_sampler.scale_anneal))
        rays_o_s = []
        rays_d_s = []
        ray_start_s = []
        ray_end_s = []
        mask_at_box_s = []
        #renderclothmask_s = [], renderclothmask_i, mask_at_box_i
        for i in range(batch_size):       
            rays_o_i, rays_d_i, ray_start_i, ray_end_i, _, _ = self.ray_sampler(512, 512, ray_origins[i].view(512,512,3), ray_directions[i].view(512,512,3), ray_start[i].view(512,512,1), ray_end[i].view(512,512,1), cropimg[i])#, 'local', mask_at_box[i].view(512,512,1)renderclothmask[i], meshmask[i],location[i]
            rays_o_s.append(rays_o_i[None])
            rays_d_s.append(rays_d_i[None])
            ray_start_s.append(ray_start_i[None])
            ray_end_s.append(ray_end_i[None])
            #mask_at_box_s.append(mask_at_box_i[None])
            #renderclothmask_s.append(renderclothmask_i[None])
        #mask_at_box = torch.cat(mask_at_box_s)            
        ray_origins = torch.cat(rays_o_s)#[mask_at_box].view(batch_size,-1,3)        
        ray_directions = torch.cat(rays_d_s)#[mask_at_box].view(batch_size,-1,3) 
        ray_start = torch.cat(ray_start_s)#[mask_at_box].view(batch_size,-1,1) 
        ray_end = torch.cat(ray_end_s)#[mask_at_box].view(batch_size,-1,1) 

        #renderclothmask = torch.cat(renderclothmask_s)
        
        # renderclothmask0 = renderclothmask[0].reshape(32,32)#low resolution for nerf rendering
        # renderclothmask0 = renderclothmask0.detach().cpu().numpy()
        # cv2.imwrite('test/real_mask0.png', renderclothmask0*255)
        
        #renderclothmask = renderclothmask.unsqueeze(-1).repeat(1,1,Nc,1)
        
        # Create stratified depth samples
        depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

        _, num_rays, samples_per_ray, _ = depths_coarse.shape
        bs_expand = batch_size 
               
        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       

        #mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        #sample_coordinates.requires_grad_(True)
        
        # sample_coordinates_smpl = self.get_canonical_coordinates(
            # sample_coordinates,
            # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)) 
            # #mask = mask            
        # )
        # #sample_coordinates = sample_coordinates_smpl
        # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
            # sample_coordinates,
            # cloth_reduced_current,
            # self.cloth_reduced_canon.repeat(batch_size,1,1)
            # #mask            
        # )
        
        # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
        # cloth_ptsdist = torch.cdist(sample_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_distmin = cloth_ptsdistmin[0]
        # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
                    
        # cloth_neartag = cloth_distmin<smpl_distmin
       
        # sample_coordinates_can = sample_coordinates_smpl
        # sample_coordinates_can.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
        # sample_coordinates_can = sample_coordinates_can.view(batch_size,-1,3)
        
        # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_current, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # nnvidx = torch.squeeze(smpl_ptsdistmin[1], -1)
        
        sample_coordinates_can = torch.zeros_like(sample_coordinates)
        nnvidxtag = [] 
        smpl_nnvidx = [] 
        cloth_nnvidx = [] 
        sptneartag = []  
        with torch.no_grad():         
            for i in range(0, batch_size, 1):
                #smpl_minidx = sample_coordinates.new_zeros([sample_coordinates.shape[1]], dtype=torch.long)              
                smpl_ptsdist = torch.cdist(sample_coordinates[i], smpl_current[i], p=2)#[mask[i]]deformedpersonsmpl
                smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
                smpl_distmin = smpl_ptsdistmin[0]
                smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1)#[mask[i]]
                #smpl_nnvidx.append(smpl_minidx[None])
                
                cloth_ptsdist = torch.cdist(sample_coordinates[i], cloth_current[i], p=2)#deformedpersonsmpl
                cloth_ptsdistmin = torch.min(cloth_ptsdist, 1)
                cloth_distmin = cloth_ptsdistmin[0]
                cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1)
                #cloth_nnvidx.append(cloth_minidx[None])
                idxtag = cloth_distmin<smpl_distmin
                #nnvidxtag.append(idxtag[None])
                
                ptdist = torch.logical_or(cloth_distmin<0.03, smpl_distmin<0.03)
                sptneartag.append(ptdist[None])
                if ~idxtag.sum().item()>0:
                    sample_coordinates_can[i][~idxtag] = self.inversedeforming_samplepoints_LBS(sample_coordinates[i][~idxtag][None], smpl_minidx[~idxtag][None], smpl_current, A[i][None], Th[i][None])[0]
                #sample_coordinates_can = self.inversedeforming_samplepoints_graphdeform_smpl(sample_coordinates_can, smpl_nnvidx)        
                if idxtag.sum().item()>0:
                    sample_coordinates_can2 = self.inversedeforming_samplepoints_cloth(sample_coordinates[i][idxtag][None], cloth_minidx[idxtag][None], cloth_current, A[i][None], Th[i][None])       
                    sample_coordinates_can2 = self.inversedeforming_samplepoints_graphdeform2(sample_coordinates_can2, cloth_minidx[idxtag][None], self.modelnodepos[i], self.deformation_affine[i], self.deformation_transl[i])            
                    sample_coordinates_can[i][idxtag] = sample_coordinates_can2[0]#[nnvidxtag]
                
                       
        sptneartag = torch.cat(sptneartag).view(batch_size,-1,Nc,1)         
        # smpl_nnvidx = torch.cat(smpl_nnvidx)   
        # cloth_nnvidx = torch.cat(cloth_nnvidx)
        # nnvidxtag = torch.cat(nnvidxtag)        #[mask].view(batch_size,-1,3)
        
        

        
        #displacement
        # smplpara = torch.cat([smpl_params, betas], -1)
        # samplepts_disp_smpl = self.displace_smpl(sample_coordinates, smplpara)
        # samplepts_disp_cloth = self.displace_cloth(sample_coordinates, smplpara)
        # samplepts_disp = torch.cat([samplepts_disp_smpl, samplepts_disp_cloth], 1)
        # sample_coordinates_smpl = sample_coordinates_smpl+samplepts_disp_smpl
        # sample_coordinates_cloth = sample_coordinates_cloth+samplepts_disp_cloth
        samplepts_disp = 0
        
        # meshvert_def = torch.cat([self.canonclothvert, self.rawtemplatesmpl], dim=1)#self.deformedcloth
        # npvertices = meshvert_def[0].detach().cpu().numpy()
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/canverts{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = self.rawtemplatesmpl[0].detach().cpu().numpy()
        # npfaces = self.smplfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/smpl_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = smpl_current[0].detach().cpu().numpy()
        # npfaces = self.smplfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/smpl{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # # npvertices = self.canonclothvert[0].detach().cpu().numpy()
        # # npfaces = self.clothfaces.detach().cpu().numpy()
        # # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # # result_path = os.path.join('test/cloth_can{:04d}.obj'.format(frameidx[0].item()))
        # # mesh.export(result_path)
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = sample_coordinates_can[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('result/deepcap/test/sample_coordinates_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        
        # sample_coordinates_smpl.requires_grad_(True)
        # sample_coordinates_cloth.requires_grad_(True)
        # #print(torch.isnan(sample_coordinates_cloth).any() or torch.isinf(sample_coordinates_cloth).any())
        # density_cloth, feature_cloth = self.SDFnetwork_cloth(sample_coordinates_cloth,self.shapepara.repeat(batch_size,1))#tempclothpara     
        
        # #density_cloth = torch.nan_to_num(density_cloth, nan=0.0, posinf=0.0, neginf=0.0)
        
        # density_smpl, feature_smpl = self.SDFnetwork_smpl(sample_coordinates_smpl,self.canonparamshape.repeat(batch_size,1))
        
        #sample_coordinates_can.requires_grad_(True)
        #density, feature = self.SDFnetwork(sample_coordinates, self.canonparamshape.repeat(batch_size,1))#
        density = self.Densitynetwork(sample_coordinates_can)
        
        # samplepts_incsdf_smpl = self.incSDFnetwork_smpl(sample_coordinates_smpl)
        # samplepts_incsdf_cloth = self.incSDFnetwork_cloth(sample_coordinates_cloth)
        # samplepts_disp = torch.cat([samplepts_incsdf_smpl, samplepts_incsdf_cloth], 1)#samplepts_incsdf_smpl
        # density_smpl = density_smpl+samplepts_incsdf_smpl
        # density_cloth = density_cloth+samplepts_incsdf_cloth
        samplepts_disp = 0
        
        # sample_coordinates_cloth.requires_grad_(True)
        # #samplepts_density_smpl, samplepts_density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], samplepts_smpl, samplepts_cloth)
        # samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
        # samplepts_density_smpl,_ = self.SDFnetwork_smpl(samplepts_smpl,self.canonparamshape)

        # offmesh_sdf_smpl = samplepts_incsdf_smpl.reshape(batch_size,-1)#density_smpl
        # offmesh_sdf_cloth = samplepts_incsdf_cloth.reshape(batch_size,-1)#density_cloth
        # #offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)#offmesh_sdf_smpl
        # offmesh_sdf = torch.cat([density_smpl.reshape(batch_size,-1),density_cloth.reshape(batch_size,-1)],dim=1)
        #offmesh_sdf = 0
        
        # offmesh_sdf_smpl = density_smpl.reshape(batch_size,-1)#density_smpl
        # offmesh_sdf_cloth = density_cloth.reshape(batch_size,-1)#density_cloth
        # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)#offmesh_sdf_smpl
        
        
        # d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        # sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      # inputs=sample_coordinates_smpl,
                                       # grad_outputs=d_output_smpl,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        # sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      # inputs=sample_coordinates_cloth,
                                       # grad_outputs=d_output_cloth,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)#sdfgradients_smpl
        #sdfgradients = 0
        
        # offmesh_sdf = density.reshape(batch_size,-1)
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdfgradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=sample_coordinates,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
                                       
        #density_smpl = torch.nan_to_num(density_smpl, nan=0.0, posinf=0.0, neginf=0.0)
        # density_cloth[density_cloth.abs()>0.5] = 0.5
        # density_smpl[density_smpl.abs()>0.5] = 0.5
        #print(torch.norm(sdfgradients[0], dim=-1))
        smplpara = torch.cat([smpl_params, betas], -1)
        
        # colorlatentsmpl = self.colorlatent_smpl(torch.LongTensor([0]).to('cuda'))
        # colorlatentcloth = self.colorlatent_cloth(torch.LongTensor([0]).to('cuda'))
        # feature_cloth = torch.cat([feature_cloth,colorlatentcloth[:,None].repeat(feature_cloth.shape[0],feature_cloth.shape[1],1)],dim=-1)
        # feature_smpl = torch.cat([feature_smpl,colorlatentsmpl[:,None].repeat(feature_smpl.shape[0],feature_smpl.shape[1],1)],dim=-1)
        # color_cloth = self.decoder_cloth(sample_coordinates_cloth, z_rend)#feature_cloth
        # color_smpl = self.decoder(sample_coordinates_smpl, z_rend)#feature_smpl
        
        color = self.decoder(sample_coordinates_can, z_rend)
        
        # density_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32)#normlized to [-1,1]
        # density_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32) * 1e3
        # color_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32)#normlized to [-1,1]
        # color_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32) * 1e3
        # density_cloth[mask] = density_cloth0
        # density_smpl[mask] = density_smpl0
        # color_cloth[mask] = color_cloth0
        # color_smpl[mask] = color_smpl0
        
        #color_smpl, color_cloth = self.fetchingdoublenerf_fixgeometry_color(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        #color_smpl, density_smpl, color_cloth, density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        
        color = color.reshape(batch_size, num_rays, samples_per_ray, color.shape[-1])
        density = density.reshape(batch_size, num_rays, samples_per_ray, 1)              
        #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
        #density = self.laplacedensity(density.reshape(batch_size, -1,1))

        # density = sdf_to_alpha2(density.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #
        # density = density.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        beta = self.beta_network().clamp(1e-9, 1e6)
        alpha = self.sdf_to_alpha(density.reshape(batch_size, num_rays, samples_per_ray), beta)
        alpha = alpha.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        #alpha[~sptneartag] = 0
        
        # alpha = sdf_to_alpha(density.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s * 1.0))  
        # alpha = alpha.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        # color_smpl = color_smpl.reshape(batch_size, num_rays, samples_per_ray, color_smpl.shape[-1])
        # density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)              
        # #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())

        # density_smpl = sdf_to_alpha2(density_smpl.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #
        # density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)
        # #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
        # #density_smplmesh, _ = self.SDFnetwork_smpl(self.smpl_reduced_canon_vert.repeat(batch_size,1,1),self.canonparamshape.repeat(batch_size,1))

        # #density_smplmesh=sdf_to_alpha2(density_smplmesh, 0.05)
        # #print(torch.exp(self.ln_s * 1.0),density_smplmesh[0])
        
        # color_cloth = color_cloth.reshape(batch_size, num_rays, samples_per_ray, color_cloth.shape[-1])
        # density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)  
        # #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())        
        # density_cloth = sdf_to_alpha2(density_cloth.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s_cloth))  #
        # density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)
        # #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())  

        #rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color_smpl, density_smpl, color_cloth, density_cloth, renderclothmask, depths_coarse, batch_size)
        rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color, alpha, depths_coarse, batch_size)
        # rendered_color_full = rendered_color.new_zeros([batch_size,self.ray_sampler.N_samples_sqrt**2,3])
        # rendered_color_full[mask_at_box] = rendered_color.reshape(batch_size, -1, 3)
        # rendered_color = rendered_color_full.reshape(batch_size, -1)
        
        with torch.no_grad(): 
            meshpts_smpl, meshpts_cloth, samplepts_smpl, samplepts_cloth, meshpts_normal = self.samplingpoint_learningmeshsdf(self.rawtemplatesmpl, self.canon_clothvert)#.repeat(batch_size,1,1)cloth_reduced_canon
        
        #samplepts.requires_grad_(True)
        #_, samplepts_density = self.fetchingnerf(planes, samplepts)
        # samplepts = torch.cat([samplepts_smpl,samplepts_cloth,meshpts_smpl,meshpts_cloth],dim=1)
        # samplepts.requires_grad_(True)
        
        samplepts_smpl.requires_grad_(True)
        #samplepts_cloth.requires_grad_(True)
        
        #samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
        samplepts_density_smpl = self.Densitynetwork(samplepts_smpl)

        # hypo_params_cloth = self.hyper_net_cloth(self.tempclothpara)
        # embed_samplepts_cloth = self.embed_fn_fine(samplepts_cloth)
        # print(embed_samplepts_cloth.shape)
        # samplepts_density_cloth = self.sdf_network_cloth(embed_samplepts_cloth, params=hypo_params_cloth)
        
        # hypo_params_smpl = self.hyper_net_smpl(self.canonparamshape)
        # embed_samplepts_smpl = self.embed_fn_fine(samplepts_smpl)
        # samplepts_density_smpl = self.sdf_network_cloth(embed_samplepts_smpl, params=hypo_params_smpl)

        offmesh_sdf_smpl = samplepts_density_smpl.reshape(batch_size,-1)
        #offmesh_sdf_cloth = samplepts_density_cloth.reshape(batch_size,-1)
        offmesh_sdf = offmesh_sdf_smpl#torch.cat([,offmesh_sdf_cloth],dim=1)

        d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      inputs=samplepts_smpl,
                                       grad_outputs=d_output_smpl,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        # # d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        # # sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      # # inputs=samplepts_cloth,
                                       # # grad_outputs=d_output_cloth,
                                      # # create_graph=True,
                                       # # retain_graph=True,
                                       # # only_inputs=True)[0]
        # # sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)
        sdfgradients = sdfgradients_smpl
        
        #_, meshpts_smpl_sdf = self.fetchingnerf(planes, meshpts_smpl)
        #_, meshpts_cloth_sdf = self.fetchingnerf(planes, meshpts_cloth)
        # meshpts_smpl.requires_grad_(True)
        # meshpts_cloth.requires_grad_(True)
        #meshpts_smpl_sdf, meshpts_cloth_sdf = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], meshpts_smpl, meshpts_cloth)
        meshpts_cloth_sdf = self.Densitynetwork(meshpts_cloth)
        meshpts_smpl_sdf = self.Densitynetwork(meshpts_smpl)
        
        # embed_meshpts_cloth = self.embed_fn_fine(meshpts_cloth_sdf)
        # meshpts_cloth_sdf = self.sdf_network_cloth(embed_meshpts_cloth, params=hypo_params_cloth)
        
        # embed_meshpts_smpl = self.embed_fn_fine(meshpts_smpl_sdf)
        # meshpts_smpl_sdf = self.sdf_network_cloth(embed_meshpts_smpl, params=hypo_params_smpl)


        meshpts_smpl_sdf = meshpts_smpl_sdf.reshape(batch_size,-1)
        meshpts_cloth_sdf = meshpts_cloth_sdf.reshape(batch_size,-1)
        #print(meshpts_smpl_sdf[0])       
        # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth,meshpts_smpl_sdf,meshpts_cloth_sdf],dim=1)
                
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdf_gradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=samplepts,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
                                       
        # sampleptsnum = samplepts_smpl.shape[1]+samplepts_cloth.shape[1]
        # sdfgradients =  sdf_gradients[:,:sampleptsnum,:]          
        # meshptsgradients =  sdf_gradients[:,sampleptsnum:,:]   
        
        # output_smpl = torch.ones_like(meshpts_smpl_sdf, requires_grad=False, device=meshpts_smpl_sdf.device)#
        # meshptsgradients_smpl = torch.autograd.grad(outputs=meshpts_smpl_sdf,
                                      # inputs=meshpts_smpl,
                                       # grad_outputs=output_smpl,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # output_cloth = torch.ones_like(meshpts_cloth_sdf, requires_grad=False, device=meshpts_cloth_sdf.device)#
        # meshptsgradients_cloth = torch.autograd.grad(outputs=meshpts_cloth_sdf,
                                      # inputs=meshpts_cloth,
                                       # grad_outputs=output_cloth,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # meshptsgradients = torch.cat([meshptsgradients_smpl,meshptsgradients_cloth],dim=1)
        #meshptsgradients = 0
        
        #if 0:    
        #offmesh_sdf = 0
        #sdfgradients = 0
        #meshpts_smpl_sdf = 0
        #meshpts_cloth_sdf = 0
        meshptsgradients = 0
        meshpts_normal = 0
        
        
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients#rgb_final, depth_final, weights.sum(2)
    
    def singlenerf_rendering_deepcap_twodiscriminator(self, frameidx, cropimg, Nc, Nf, z_rend, planes, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):
        
        # R = extrinsics[:,:3,:3] 
        # T = extrinsics[:,:3,3:]

        # pose_to_camera = torch.bmm(pose_to_world[..., :3, 3], R.permute(0,2,1)) + T.permute(0,2,1)
        # pose_2d = torch.matmul(pose_to_camera, intrinsic.permute(0,2,1))
        # pose_2d = pose_2d[:, :, :2] / pose_2d[:, :, 2][...,None] 
        # location = pose_2d[:, 15, :2]
        
        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        batch_size = smpl_params.shape[0]
        
        #self.sample_canonshape(batch_size)
        
        #deforming with cloth and smpl
        smpl_current = self.deformingsmpl_LBS(A, Th)
        smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)

        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)#
        cloth_reduced_current = self.cloth_reduced(cloth_current)
        
       
        combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        #combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        
        # with torch.no_grad(): 
            # clothmask, meshmask = self.render_clothmask(self.meshface.repeat(batch_size,1,1), combinedmeshvert, extrinsics, intrinsic)
            # renderclothmask = clothmask#.unsqueeze(-1).repeat(1,1,1,Nc).view(batch_size,-1,Nc,1)
        # mesh_mask = clothmask[0].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        # mesh_mask = clothmask[1].detach().cpu().numpy()
        # cv2.imwrite('test/renderclothmask{:04d}.png'.format(frameidx[1].item()), mesh_mask*255)
        renderclothmask = 0
        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        ray_start = near
        ray_end = far
        # k_iter = self.ray_sampler.iterations // 1000 * 3        
        # print(self.ray_sampler.max_scale * exp(-k_iter*self.ray_sampler.scale_anneal))
        
        #global generation
        rays_o_s = []
        rays_d_s = []
        ray_start_s = []
        ray_end_s = []
        #renderclothmask_s = []renderclothmask_i,
        for i in range(batch_size):       
            rays_o_i, rays_d_i, ray_start_i, ray_end_i, _, _ = self.ray_sampler(512, 512, ray_origins[i].view(512,512,3), ray_directions[i].view(512,512,3), ray_start[i].view(512,512,1), ray_end[i].view(512,512,1), cropimg[i] ,'global')#renderclothmask[i], meshmask[i],location[i]
            rays_o_s.append(rays_o_i[None])
            rays_d_s.append(rays_d_i[None])
            ray_start_s.append(ray_start_i[None])
            ray_end_s.append(ray_end_i[None])
            #renderclothmask_s.append(renderclothmask_i[None])
        ray_origins_global = torch.cat(rays_o_s)        
        ray_directions_global = torch.cat(rays_d_s)
        ray_start_global = torch.cat(ray_start_s)
        ray_end_global = torch.cat(ray_end_s)
        # renderclothmask = torch.cat(renderclothmask_s)
        
        # renderclothmask0 = renderclothmask[0].reshape(32,32)#low resolution for nerf rendering
        # renderclothmask0 = renderclothmask0.detach().cpu().numpy()
        # cv2.imwrite('test/real_mask0.png', renderclothmask0*255)
        
        #renderclothmask = renderclothmask.unsqueeze(-1).repeat(1,1,Nc,1)
        
        # Create stratified depth samples
        depths_coarse = self.sample_stratified(ray_origins_global, ray_start_global, ray_end_global, Nc)

        _, num_rays, samples_per_ray, _ = depths_coarse.shape
        bs_expand = batch_size 
               
        # Coarse Pass
        sample_coordinates = (ray_origins_global.unsqueeze(-2) + depths_coarse * ray_directions_global.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions_global.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       
        
        #mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        
        
        # sample_coordinates_smpl = self.get_canonical_coordinates(
            # sample_coordinates,
            # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)) 
            # #mask = mask            
        # )
        # #sample_coordinates = sample_coordinates_smpl
        # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
            # sample_coordinates,
            # cloth_reduced_current,
            # self.cloth_reduced_canon.repeat(batch_size,1,1)
            # #mask            
        # )
        
        # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
        # cloth_ptsdist = torch.cdist(sample_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_distmin = cloth_ptsdistmin[0]
        # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
                    
        # cloth_neartag = cloth_distmin<smpl_distmin
       
        # sample_coordinates = sample_coordinates_smpl
        # sample_coordinates.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
        # sample_coordinates = sample_coordinates.view(batch_size,-1,3)
        sample_coordinates.requires_grad_(True)
        nnvidx = []              
        for i in range(0, batch_size, 1):
            smpl_ptsdist = torch.cdist(sample_coordinates[i], smpl_current[i], p=2)#deformedpersonsmpl
            smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            smpl_distmin = smpl_ptsdistmin[0]
            smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1)
            nnvidx.append(smpl_minidx[None])
        nnvidx = torch.cat(nnvidx)    
        sample_coordinates_can = self.inversedeforming_samplepoints_LBS(sample_coordinates, nnvidx, smpl_current, A, Th)
        
        #displacement
        # smplpara = torch.cat([smpl_params, betas], -1)
        # samplepts_disp_smpl = self.displace_smpl(sample_coordinates, smplpara)
        # samplepts_disp_cloth = self.displace_cloth(sample_coordinates, smplpara)
        # samplepts_disp = torch.cat([samplepts_disp_smpl, samplepts_disp_cloth], 1)
        # sample_coordinates_smpl = sample_coordinates_smpl+samplepts_disp_smpl
        # sample_coordinates_cloth = sample_coordinates_cloth+samplepts_disp_cloth
        samplepts_disp = 0
        
        # sample_coordinates_smpl.requires_grad_(True)
        # sample_coordinates_cloth.requires_grad_(True)
        # #print(torch.isnan(sample_coordinates_cloth).any() or torch.isinf(sample_coordinates_cloth).any())
        # density_cloth, feature_cloth = self.SDFnetwork_cloth(sample_coordinates_cloth,self.shapepara.repeat(batch_size,1))#tempclothpara     
        
        # #density_cloth = torch.nan_to_num(density_cloth, nan=0.0, posinf=0.0, neginf=0.0)
        
        # density_smpl, feature_smpl = self.SDFnetwork_smpl(sample_coordinates_smpl,self.canonparamshape.repeat(batch_size,1))
        
        #sample_coordinates.requires_grad_(True)
        #density, feature = self.SDFnetwork(sample_coordinates, self.canonparamshape.repeat(batch_size,1))#
        density = self.Densitynetwork(sample_coordinates_can)
        
        # samplepts_incsdf_smpl = self.incSDFnetwork_smpl(sample_coordinates_smpl)
        # samplepts_incsdf_cloth = self.incSDFnetwork_cloth(sample_coordinates_cloth)
        # samplepts_disp = torch.cat([samplepts_incsdf_smpl, samplepts_incsdf_cloth], 1)#samplepts_incsdf_smpl
        # density_smpl = density_smpl+samplepts_incsdf_smpl
        # density_cloth = density_cloth+samplepts_incsdf_cloth
        samplepts_disp = 0
        
        # sample_coordinates_cloth.requires_grad_(True)
        # #samplepts_density_smpl, samplepts_density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], samplepts_smpl, samplepts_cloth)
        # samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
        # samplepts_density_smpl,_ = self.SDFnetwork_smpl(samplepts_smpl,self.canonparamshape)

        # offmesh_sdf_smpl = samplepts_incsdf_smpl.reshape(batch_size,-1)#density_smpl
        # offmesh_sdf_cloth = samplepts_incsdf_cloth.reshape(batch_size,-1)#density_cloth
        # #offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)#offmesh_sdf_smpl
        # offmesh_sdf = torch.cat([density_smpl.reshape(batch_size,-1),density_cloth.reshape(batch_size,-1)],dim=1)
        #offmesh_sdf = 0
        
        # offmesh_sdf_smpl = density_smpl.reshape(batch_size,-1)#density_smpl
        # offmesh_sdf_cloth = density_cloth.reshape(batch_size,-1)#density_cloth
        # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)#offmesh_sdf_smpl
        
        
        # d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        # sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      # inputs=sample_coordinates_smpl,
                                       # grad_outputs=d_output_smpl,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        # sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      # inputs=sample_coordinates_cloth,
                                       # grad_outputs=d_output_cloth,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)#sdfgradients_smpl
        #sdfgradients = 0
        
        offmesh_sdf = density.reshape(batch_size,-1)
        d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        sdfgradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      inputs=sample_coordinates,
                                       grad_outputs=d_output,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
                                       
        #density_smpl = torch.nan_to_num(density_smpl, nan=0.0, posinf=0.0, neginf=0.0)
        # density_cloth[density_cloth.abs()>0.5] = 0.5
        # density_smpl[density_smpl.abs()>0.5] = 0.5
        
        smplpara = torch.cat([smpl_params, betas], -1)
        
        # colorlatentsmpl = self.colorlatent_smpl(torch.LongTensor([0]).to('cuda'))
        # colorlatentcloth = self.colorlatent_cloth(torch.LongTensor([0]).to('cuda'))
        # feature_cloth = torch.cat([feature_cloth,colorlatentcloth[:,None].repeat(feature_cloth.shape[0],feature_cloth.shape[1],1)],dim=-1)
        # feature_smpl = torch.cat([feature_smpl,colorlatentsmpl[:,None].repeat(feature_smpl.shape[0],feature_smpl.shape[1],1)],dim=-1)
        # color_cloth = self.decoder_cloth(sample_coordinates_cloth, z_rend)#feature_cloth
        # color_smpl = self.decoder(sample_coordinates_smpl, z_rend)#feature_smpl
        
        color = self.decoder(sample_coordinates_can, z_rend)
        
        # density_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32)#normlized to [-1,1]
        # density_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32) * 1e3
        # color_cloth = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32)#normlized to [-1,1]
        # color_smpl = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],16], dtype=torch.float32) * 1e3
        # density_cloth[mask] = density_cloth0
        # density_smpl[mask] = density_smpl0
        # color_cloth[mask] = color_cloth0
        # color_smpl[mask] = color_smpl0
        
        #color_smpl, color_cloth = self.fetchingdoublenerf_fixgeometry_color(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        #color_smpl, density_smpl, color_cloth, density_cloth = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], sample_coordinates_smpl, sample_coordinates)
        
        color = color.reshape(batch_size, num_rays, samples_per_ray, color.shape[-1])
        density = density.reshape(batch_size, num_rays, samples_per_ray, 1)              
        #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())

        # density = sdf_to_alpha2(density.reshape(batch_size, num_rays, samples_per_ray), self.ln_s)  #torch.exp()
        # density = density.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        beta = self.beta_network().clamp(1e-9, 1e6)
        density = self.sdf_to_alpha(density.reshape(batch_size, num_rays, samples_per_ray), beta)
        density = density.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        # color_smpl = color_smpl.reshape(batch_size, num_rays, samples_per_ray, color_smpl.shape[-1])
        # density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)              
        # #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())

        # density_smpl = sdf_to_alpha2(density_smpl.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s))  #
        # density_smpl = density_smpl.reshape(batch_size, num_rays, samples_per_ray, 1)
        # #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())
        # #density_smplmesh, _ = self.SDFnetwork_smpl(self.smpl_reduced_canon_vert.repeat(batch_size,1,1),self.canonparamshape.repeat(batch_size,1))

        # #density_smplmesh=sdf_to_alpha2(density_smplmesh, 0.05)
        # #print(torch.exp(self.ln_s * 1.0),density_smplmesh[0])
        
        # color_cloth = color_cloth.reshape(batch_size, num_rays, samples_per_ray, color_cloth.shape[-1])
        # density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)  
        # #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())        
        # density_cloth = sdf_to_alpha2(density_cloth.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s_cloth))  #
        # density_cloth = density_cloth.reshape(batch_size, num_rays, samples_per_ray, 1)
        # #print(torch.isnan(density_cloth).any() or torch.isinf(density_cloth).any())  

        #rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color_smpl, density_smpl, color_cloth, density_cloth, renderclothmask, depths_coarse, batch_size)
        rendered_color_global, rendered_mask_global, rendered_disparity_global, weights_global = self.ray_marcher(color, density, depths_coarse, batch_size)
             
        with torch.no_grad(): 
            meshpts_smpl, meshpts_cloth, samplepts_smpl, samplepts_cloth, meshpts_normal = self.samplingpoint_learningmeshsdf(self.rawtemplatesmpl, self.canon_clothvert)#.repeat(batch_size,1,1)cloth_reduced_canon
        
        #samplepts.requires_grad_(True)
        #_, samplepts_density = self.fetchingnerf(planes, samplepts)
        # samplepts = torch.cat([samplepts_smpl,samplepts_cloth,meshpts_smpl,meshpts_cloth],dim=1)
        # samplepts.requires_grad_(True)
        
        #samplepts_smpl.requires_grad_(True)
        #samplepts_cloth.requires_grad_(True)
        
        #samplepts_density_cloth,_ = self.SDFnetwork_cloth(samplepts_cloth,self.shapepara)#tempclothpara
        #samplepts_density_smpl = self.Densitynetwork(samplepts_smpl)

        # hypo_params_cloth = self.hyper_net_cloth(self.tempclothpara)
        # embed_samplepts_cloth = self.embed_fn_fine(samplepts_cloth)
        # print(embed_samplepts_cloth.shape)
        # samplepts_density_cloth = self.sdf_network_cloth(embed_samplepts_cloth, params=hypo_params_cloth)
        
        # hypo_params_smpl = self.hyper_net_smpl(self.canonparamshape)
        # embed_samplepts_smpl = self.embed_fn_fine(samplepts_smpl)
        # samplepts_density_smpl = self.sdf_network_cloth(embed_samplepts_smpl, params=hypo_params_smpl)

        # offmesh_sdf_smpl = samplepts_density_smpl.reshape(batch_size,-1)
        # #offmesh_sdf_cloth = samplepts_density_cloth.reshape(batch_size,-1)
        # offmesh_sdf = offmesh_sdf_smpl#torch.cat([,offmesh_sdf_cloth],dim=1)

        # d_output_smpl = torch.ones_like(offmesh_sdf_smpl, requires_grad=False, device=offmesh_sdf_smpl.device)#
        # sdfgradients_smpl = torch.autograd.grad(outputs=offmesh_sdf_smpl,
                                      # inputs=samplepts_smpl,
                                       # grad_outputs=d_output_smpl,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # # d_output_cloth = torch.ones_like(offmesh_sdf_cloth, requires_grad=False, device=offmesh_sdf_cloth.device)#
        # # sdfgradients_cloth = torch.autograd.grad(outputs=offmesh_sdf_cloth,
                                      # # inputs=samplepts_cloth,
                                       # # grad_outputs=d_output_cloth,
                                      # # create_graph=True,
                                       # # retain_graph=True,
                                       # # only_inputs=True)[0]
        # # sdfgradients = torch.cat([sdfgradients_smpl,sdfgradients_cloth],dim=1)
        # sdfgradients = sdfgradients_smpl
        
        #_, meshpts_smpl_sdf = self.fetchingnerf(planes, meshpts_smpl)
        #_, meshpts_cloth_sdf = self.fetchingnerf(planes, meshpts_cloth)
        # meshpts_smpl.requires_grad_(True)
        # meshpts_cloth.requires_grad_(True)
        #meshpts_smpl_sdf, meshpts_cloth_sdf = self.fetchingdoublenerf(planes[:,:,:16], planes[:,:,16:], meshpts_smpl, meshpts_cloth)
        meshpts_cloth_sdf = self.Densitynetwork(meshpts_cloth)
        meshpts_smpl_sdf = self.Densitynetwork(meshpts_smpl)
        
        # embed_meshpts_cloth = self.embed_fn_fine(meshpts_cloth_sdf)
        # meshpts_cloth_sdf = self.sdf_network_cloth(embed_meshpts_cloth, params=hypo_params_cloth)
        
        # embed_meshpts_smpl = self.embed_fn_fine(meshpts_smpl_sdf)
        # meshpts_smpl_sdf = self.sdf_network_cloth(embed_meshpts_smpl, params=hypo_params_smpl)


        meshpts_smpl_sdf = meshpts_smpl_sdf.reshape(batch_size,-1)
        meshpts_cloth_sdf = meshpts_cloth_sdf.reshape(batch_size,-1)
        #print(meshpts_smpl_sdf[0])       
        # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth,meshpts_smpl_sdf,meshpts_cloth_sdf],dim=1)
                
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdf_gradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=samplepts,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
                                       
        # sampleptsnum = samplepts_smpl.shape[1]+samplepts_cloth.shape[1]
        # sdfgradients =  sdf_gradients[:,:sampleptsnum,:]          
        # meshptsgradients =  sdf_gradients[:,sampleptsnum:,:]   
        
        # output_smpl = torch.ones_like(meshpts_smpl_sdf, requires_grad=False, device=meshpts_smpl_sdf.device)#
        # meshptsgradients_smpl = torch.autograd.grad(outputs=meshpts_smpl_sdf,
                                      # inputs=meshpts_smpl,
                                       # grad_outputs=output_smpl,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # output_cloth = torch.ones_like(meshpts_cloth_sdf, requires_grad=False, device=meshpts_cloth_sdf.device)#
        # meshptsgradients_cloth = torch.autograd.grad(outputs=meshpts_cloth_sdf,
                                      # inputs=meshpts_cloth,
                                       # grad_outputs=output_cloth,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # meshptsgradients = torch.cat([meshptsgradients_smpl,meshptsgradients_cloth],dim=1)
        #meshptsgradients = 0
        
        #if 0:    
        #offmesh_sdf = 0
        #sdfgradients = 0
        #meshpts_smpl_sdf = 0
        #meshpts_cloth_sdf = 0
        meshptsgradients = 0
        meshpts_normal = 0
        
        #local generation
        rays_o_s = []
        rays_d_s = []
        ray_start_s = []
        ray_end_s = []
        #renderclothmask_s = []renderclothmask_i,
        for i in range(batch_size):       
            rays_o_i, rays_d_i, ray_start_i, ray_end_i, _, _ = self.ray_sampler(512, 512, ray_origins[i].view(512,512,3), ray_directions[i].view(512,512,3), ray_start[i].view(512,512,1), ray_end[i].view(512,512,1), cropimg[i] ,'local')#renderclothmask[i], meshmask[i],location[i]
            rays_o_s.append(rays_o_i[None])
            rays_d_s.append(rays_d_i[None])
            ray_start_s.append(ray_start_i[None])
            ray_end_s.append(ray_end_i[None])
            #renderclothmask_s.append(renderclothmask_i[None])
        ray_origins_local = torch.cat(rays_o_s)        
        ray_directions_local = torch.cat(rays_d_s)
        ray_start_local = torch.cat(ray_start_s)
        ray_end_local = torch.cat(ray_end_s)
        # renderclothmask = torch.cat(renderclothmask_s)
        
        # renderclothmask0 = renderclothmask[0].reshape(32,32)#low resolution for nerf rendering
        # renderclothmask0 = renderclothmask0.detach().cpu().numpy()
        # cv2.imwrite('test/real_mask0.png', renderclothmask0*255)
        
        #renderclothmask = renderclothmask.unsqueeze(-1).repeat(1,1,Nc,1)
        
        # Create stratified depth samples
        depths_coarse = self.sample_stratified(ray_origins_local, ray_start_local, ray_end_local, Nc)

        _, num_rays, samples_per_ray, _ = depths_coarse.shape
        bs_expand = batch_size 
               
        # Coarse Pass
        sample_coordinates2 = (ray_origins_local.unsqueeze(-2) + depths_coarse * ray_directions_local.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions_local.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       
        
        #mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        
        
        # sample_coordinates_smpl = self.get_canonical_coordinates(
            # sample_coordinates,
            # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)) 
            # #mask = mask            
        # )
        # #sample_coordinates = sample_coordinates_smpl
        # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
            # sample_coordinates,
            # cloth_reduced_current,
            # self.cloth_reduced_canon.repeat(batch_size,1,1)
            # #mask            
        # )
        
        # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
        # cloth_ptsdist = torch.cdist(sample_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_distmin = cloth_ptsdistmin[0]
        # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
                    
        # cloth_neartag = cloth_distmin<smpl_distmin
       
        # sample_coordinates = sample_coordinates_smpl
        # sample_coordinates.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
        # sample_coordinates = sample_coordinates.view(batch_size,-1,3)
        
        sample_coordinates2.requires_grad_(True)
        nnvidx = []              
        for i in range(0, batch_size, 1):
            smpl_ptsdist = torch.cdist(sample_coordinates2[i], smpl_current[i], p=2)#deformedpersonsmpl
            smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            smpl_distmin = smpl_ptsdistmin[0]
            smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1)
            nnvidx.append(smpl_minidx[None])
        nnvidx = torch.cat(nnvidx)    
        sample_coordinates_can2 = self.inversedeforming_samplepoints_LBS(sample_coordinates2, nnvidx, smpl_current, A, Th)
        
        #sample_coordinates.requires_grad_(True)
        #density, feature = self.SDFnetwork(sample_coordinates, self.canonparamshape.repeat(batch_size,1))#
        density_local = self.Densitynetwork(sample_coordinates_can2)
        
        #offmesh_sdf = 0
        
        # offmesh_sdf_smpl = density_smpl.reshape(batch_size,-1)#density_smpl
        # offmesh_sdf_cloth = density_cloth.reshape(batch_size,-1)#density_cloth
        # offmesh_sdf = torch.cat([offmesh_sdf_smpl,offmesh_sdf_cloth],dim=1)#offmesh_sdf_smpl
        
        #sdfgradients = 0
        
        offmesh_sdf_local = density_local.reshape(batch_size,-1)
        d_output = torch.ones_like(offmesh_sdf_local, requires_grad=False, device=offmesh_sdf.device)#
        sdfgradients_local = torch.autograd.grad(outputs=offmesh_sdf_local,
                                      inputs=sample_coordinates2,
                                       grad_outputs=d_output,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
                                       
        offmesh_sdf = torch.cat([offmesh_sdf,offmesh_sdf_local],dim=1)
        sdfgradients = torch.cat([sdfgradients,sdfgradients_local],dim=1)
        
        smplpara = torch.cat([smpl_params, betas], -1)
        
        color_local = self.decoder(sample_coordinates_can2, z_rend)
        
        color_local = color_local.reshape(batch_size, num_rays, samples_per_ray, color.shape[-1])
        density_local = density_local.reshape(batch_size, num_rays, samples_per_ray, 1)              
        #print(torch.isnan(density_smpl).any() or torch.isinf(density_smpl).any())

        # density_local = sdf_to_alpha2(density_local.reshape(batch_size, num_rays, samples_per_ray), self.ln_s)  #torch.exp()
        # density_local = density_local.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        beta = self.beta_network().clamp(1e-9, 1e6)
        density_local = self.sdf_to_alpha(density_local.reshape(batch_size, num_rays, samples_per_ray), beta)
        density_local = density_local.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        #rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(color_smpl, density_smpl, color_cloth, density_cloth, renderclothmask, depths_coarse, batch_size)
        rendered_color_local, rendered_mask_local, rendered_disparity_local, weights_local = self.ray_marcher(color_local, density_local, depths_coarse, batch_size)
        
        return rendered_color_global, rendered_mask_global, rendered_disparity_global, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients,rendered_color_local#rgb_final, depth_final, weights.sum(2)
            
    def nerf_rendering_deepcap(self, frameidx, rawimg, Nc, Nf, z_rend, planes, latentws, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):

        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]

        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        
         # Create stratified depth samples
        ray_start = near
        ray_end = far
        depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
        bs_expand = batch_size 
         
        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       
        
        # smpl_reduced_current = self.smpl_reduced(betas=smpl_betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=smpl_translate)
        # smpl_reduced_canon = self.smpl_reduced(betas=self.smpl_avg_betas.expand(bs_expand, -1),
                                               # body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1),
                                               # global_orient=self.smpl_avg_orient.expand(bs_expand, -1),
                                               # transl=self.smpl_avg_transl.expand(bs_expand, -1))

        # smpl_reduced_current.transl = smpl_translate
        # smpl_reduced_canon.transl = self.smpl_avg_transl.expand(bs_expand, -1)
        # smpl_reduced_current.vertices *= smplscale#self.smpl_avg_scale
        # smpl_reduced_canon.vertices *= self.smpl_avg_scale
        # print(smplscale)
        #with torch.no_grad():
        
        #reduced smpl        self.canonparamshape
        # smpl_reduced_current = self.smpl_reduced(betas=betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=self.smpl_avg_transl.expand(bs_expand, -1))
      
        # smpl_reduced_canon = self.smpl_reduced(betas=betas, body_pose=self.smpl_avg_body_pose.expand(batch_size, -1),
                                      # global_orient=self.smpl_avg_orient.expand(batch_size, -1), transl=self.smpl_avg_transl.expand(bs_expand, -1))
        
        # smpl_reduced_current.vertices += Th[:,None]
       
        #deforming with cloth and smpl
        smpl_current = self.deformingsmpl_LBS(A, Th)
        smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)
        
        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        cloth_current = self.deformingcloth_LBS(self.canonclothvert.repeat(batch_size,1,1), A, Th)
        cloth_reduced_current = self.cloth_reduced(cloth_current)
        
        mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        
        combinedmeshvert = torch.cat([cloth_current, smpl_current], dim=1)
        combinedmesh_reduced_current = self.combinedmesh_reduced(combinedmeshvert)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # combined_distmin = sample_coordinates.new_zeros([batch_size,sample_coordinates.shape[1]], dtype=torch.float32)              
        # for i in range(0, batch_size, 1):                   
            # combined_ptsdist = torch.cdist(sample_coordinates[i][mask[i]], combinedmesh_reduced_current[i], p=2)#deformedpersonsmpl
            # combined_ptsdistmin = torch.min(combined_ptsdist, 1)           
            # ptsdist = torch.squeeze(combined_ptsdistmin[0], -1)  # B*P 
            # print(ptsdist)            
            # combined_distmin[i][mask[i]] = ptsdist
            
        # meshhumanmask = combined_distmin<0.05
        
        # mesh_mask = meshhumanmask.reshape(batch_size, 128, 128,48)#low resolution for nerf rendering
        # mesh_mask = mesh_mask.sum(-1)                 
        # mesh_mask = mesh_mask>0
        
        # mesh_mask = mesh_mask[0].detach().cpu().numpy()
        # cv2.imwrite('test/mesh_mask{:04d}.png'.format(frameidx[0].item()), mesh_mask*255)
        
        sample_coordinates = self.get_canonical_coordinates_clothvertex(
            sample_coordinates,
            combinedmesh_reduced_current,
            self.combinedmesh_reduced_canon.repeat(batch_size,1,1),  
            mask            
        )
        # with torch.no_grad(): 
            # mesh_mask = self.rasterize_eg3d(self.combinedmesh_reduced_faces, combinedmesh_reduced_current, extrinsics, intrinsic)
            # meshhumanmask = mesh_mask.unsqueeze(-1).repeat(1,1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        
        # combined_distmin = sample_coordinates.new_zeros([batch_size,sample_coordinates.shape[1]], dtype=torch.float32)              
        # for i in range(0, batch_size, 1):
            # tnum = int(sample_coordinates[i].shape[0]/2)                     
            # combined_ptsdist = torch.cdist(sample_coordinates[i][:tnum], combinedmesh_reduced_current[i], p=2)#deformedpersonsmpl
            # combined_ptsdistmin = torch.min(combined_ptsdist, 1)           
            # ptsdist = torch.squeeze(combined_ptsdistmin[0], -1)  # B*P            
            # combined_ptsdist = torch.cdist(sample_coordinates[i][tnum:], combinedmesh_reduced_current[i], p=2)#deformedpersonsmpl
            # combined_ptsdistmin1 = torch.min(combined_ptsdist, 1)
            # ptsdist1 = torch.squeeze(combined_ptsdistmin1[0], -1)
            # ptsdist = torch.cat((ptsdist,ptsdist1),0)
            # combined_distmin[i] = ptsdist
            # print(ptsdist[mask[i]])
                
        
        # npvertices = combinedmesh_reduced_current[0].detach().cpu().numpy()
        # npfaces = self.combinedmesh_reduced.faces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/deformedverts_reduced{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = self.combinedmesh_reduced_canon[0].detach().cpu().numpy()
        # npfaces = self.combinedmesh_reduced.faces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/canverts_reduced{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        
        #interp_loss = self.computeinterpenetrationloss_posedsmpl(canondeformedcloth, smpl_current, cloth_current)
                
        # meshvert_def = torch.cat([cloth_current, smpl_current], dim=1)#self.deformedcloth
        # npvertices = meshvert_def[0].detach().cpu().numpy()
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/deformedverts{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = smpl_current[0].detach().cpu().numpy()
        # npfaces = self.smplfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/smpl_current{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = cloth_current[0].detach().cpu().numpy()
        # npfaces = self.clothfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/cloth_current{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/sample_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
         
        # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
        # cloth_ptsdist = torch.cdist(sample_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_distmin = cloth_ptsdistmin[0]
        # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
        
        # smpl_distmin = sample_coordinates.new_zeros([batch_size,sample_coordinates.shape[1]], dtype=torch.long)              
        # for i in range(0, batch_size, 1):
            # tnum = int(sample_coordinates[i].shape[0]/2)                     
            # smpl_ptsdist = torch.cdist(sample_coordinates[i][:tnum], smpl_reduced_current_vert[i], p=2)#deformedpersonsmpl
            # smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            # ptsdist = torch.squeeze(smpl_ptsdistmin[0], -1)  # B*P
            # smpl_ptsdist = torch.cdist(sample_coordinates[i][tnum:], smpl_reduced_current_vert[i], p=2)#deformedpersonsmpl
            # smpl_ptsdistmin1 = torch.min(smpl_ptsdist, 1)
            # ptsdist1 = torch.squeeze(smpl_ptsdistmin1[0], -1)
            # ptsdist = torch.cat((ptsdist,ptsdist1),0)
            # smpl_distmin[i] = ptsdist
        # cloth_distmin = sample_coordinates.new_zeros([batch_size,sample_coordinates.shape[1]], dtype=torch.long)              
        # for i in range(0, batch_size, 1):
            # tnum = int(sample_coordinates[i].shape[0]/2)                      
            # cloth_ptsdist = torch.cdist(sample_coordinates[i][:tnum], cloth_reduced_current[i], p=2)#deformedpersonsmpl
            # cloth_ptsdistmin = torch.min(cloth_ptsdist, 1)
            # ptsdist = torch.squeeze(cloth_ptsdistmin[0], -1)  # B*P
            # cloth_ptsdist = torch.cdist(sample_coordinates[i][tnum:], cloth_reduced_current[i], p=2)#deformedpersonsmpl
            # cloth_ptsdistmin1 = torch.min(cloth_ptsdist, 1)
            # ptsdist1 = torch.squeeze(cloth_ptsdistmin1[0], -1)
            # ptsdist = torch.cat((ptsdist,ptsdist1),0)
            # cloth_distmin[i] = ptsdist
            
        # cloth_neartag = cloth_distmin<smpl_distmin
        
        # mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        
        # sample_coordinates_smpl = self.get_canonical_coordinates(
            # sample_coordinates,
            # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)),  
            # mask = mask            
        # )

        # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertex(
            # sample_coordinates,
            # cloth_reduced_current,
            # self.cloth_reduced_canon.repeat(batch_size,1,1),  
            # mask            
        # )
        # sample_coordinates = sample_coordinates_smpl
        # sample_coordinates.view(-1,3)[cloth_neartag.view(-1)] = sample_coordinates_cloth.view(-1,3)[cloth_neartag.view(-1)]
        # sample_coordinates = sample_coordinates.view(batch_size,-1,3)
        
        
        # meshvert_def = torch.cat([self.canonclothvert, self.rawtemplatesmpl], dim=1)#self.deformedcloth
        # npvertices = meshvert_def[0].detach().cpu().numpy()
        # npfaces = self.meshface[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/canverts{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = self.rawtemplatesmpl[0].detach().cpu().numpy()
        # npfaces = self.smplfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/smpl_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = self.canonclothvert[0].detach().cpu().numpy()
        # npfaces = self.clothfaces.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/cloth_can{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # result_path = os.path.join('test/samplecan_coordinates{:04d}.obj'.format(frameidx[0].item()))
        # mesh.export(result_path)
        
        #smpl model, vert transformation
        # smplvert, A, T = get_shape(self.smpl, body_pose=smpl_body_pose.view(batch_size,23,3),
                                       # global_orient=smpl_orient[:,None]) 
        # smplvert = torch.matmul(trans[:,:3,:3], smplvert.permute(0,2,1))
        # smplvert = smplvert.permute(0,2,1)
        # smplvert += shift[:,None]
        # # axis_transform
        # smplvert = smplvert[:, :, [1, 2, 0]] * torch.tensor([-1, -1, -1]).view(1,3)[:, None].to(betas.device)
        # mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])

        # nnvidx = mask.new_zeros([mask.shape[0],mask.shape[1]], dtype=torch.long)              
        # for i in range(0, smplvert.shape[0], 1):
            # # tnum = int(sample_coordinates[i][mask[i]].shape[0]/2)
            # # pts = sample_coordinates[i][mask[i]][:tnum]                       
            # # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][:tnum], smplvert[i], p=2)#deformedpersonsmpl
            # # smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            # # vidx = torch.squeeze(smpl_ptsdistmin[1], -1)  # B*P
            # # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][tnum:], smplvert[i], p=2)#deformedpersonsmpl
            # # smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            # # vidx1 = torch.squeeze(smpl_ptsdistmin[1], -1)
            # # vidx = torch.cat((vidx,vidx1),0)
            # # nnvidx[i][mask[i]] = vidx

            # vidx = []
            # tnum = int(sample_coordinates[i][mask[i]].shape[0]/4)          
            # for j in range(4):             
                # if j==3:
                    # #smpl_ptsdist = ((sample_coordinates[i][mask[i]][j*tnum:][:,None,:] - smplvert[i][None,:,:])**2).sum(-1) # Pts x V       
                    # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][j*tnum:], smplvert[i], p=2)
                # else:
                    # #smpl_ptsdist = ((sample_coordinates[i][mask[i]][j*tnum:(j+1)*tnum][:,None,:] - smplvert[i][None,:,:])**2).sum(-1) # Pts x V       
                    # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][j*tnum:(j+1)*tnum], smplvert[i], p=2)#deformedpersonsmpl
                # smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
                # tidx = torch.squeeze(smpl_ptsdistmin[1], -1)  # B*P
                # vidx += [tidx]
            # vidx = torch.cat(vidx, 0)
            # nnvidx[i][mask[i]] = vidx
            
        # sample_coordinates = self.invtransform_surreal(sample_coordinates, shift, trans)
        
        # sample_coordinates = self.inversedeforming_samplepoints_LBS(sample_coordinates, nnvidx, smplvert, T)
        
        # mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        # #sample_coordinates[~mask] = sample_coordinates[~mask] + 10
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates.obj')   
        
        # npvertices = self.smpl.v_template.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/canonmesh.obj')
        
        #raw smpl
        # smpl_reduced_current_vert = get_shape(self.smpl, body_pose=smpl_body_pose.view(batch_size,23,3),
                                      # global_orient=smpl_orient[:,None])  
        # smpl_reduced_current_vert *= smplscale
        # smpl_reduced_current_vert += smpl_translate[:,None] / 100
        # smpl_reduced_current = SMPLOutput(vertices=smpl_reduced_current_vert)
        
        # smpl_reduced_canon_vert = get_shape(self.smpl, body_pose=self.smpl_avg_body_pose.expand(batch_size, -1).view(batch_size,23,3),
                                      # global_orient=self.smpl_avg_orient.expand(batch_size, -1)[:,None])
        # smpl_reduced_canon_vert *= smplscale
        # #smpl_reduced_canon_vert += self.smpl_avg_transl.expand(3, -1)[:,None]                                      
        # smpl_reduced_canon = SMPLOutput(vertices=smpl_reduced_canon_vert)self.smpl_reduced.faces_t, smpl_reduced_current.vertices
        
        # rendermask = self.rasterize_eg3d(self.smplfaces, smplvert, extrinsics, intrinsic)
        # for i in range(0,1):
            # mask_render = rendermask[i].detach().cpu().numpy()
            # mask_path = 'test/smplmask{:04d}.png'.format(frameidx[i].item())       
            # cv2.imwrite(mask_path, mask_render * 255)            
            # raw_img = rawimg.permute(0,2,3,1)[i].detach().cpu().numpy()
            # im_path = 'test/rawimg{:04d}.png'.format(frameidx[i].item())       
            # cv2.imwrite(im_path, raw_img)
        #bonemask = bone_mask[0].detach().cpu().numpy()
        #cv2.imwrite('test/bonemask.png', bonemask * 255)
        
        # smpl_reduced_canon_vert = get_shape(self.smpl, body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1).view(batch_size,23,3),
                                      # global_orient=self.smpl_avg_orient.expand(bs_expand, -1)[:,None])
        # smpl_reduced_canon_vert *= self.smpl_avg_scale
        # smpl_reduced_canon_vert += self.smpl_avg_transl.expand(bs_expand, -1)[:,None]                                      
        # smpl_reduced_canon = SMPLOutput(vertices=smpl_reduced_canon_vert)

        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinatesraw.obj')   
        
        # npvertices = smpl_reduced_current.vertices[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/currmesh.obj')
        
        #mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        
        #displacement
        # smplpara = torch.cat([smpl_params, betas], -1)
        # samplepts_disp =  torch.zeros_like(sample_coordinates)   
        # for i in range(smpl_params.shape[0]):
            # samplepts_disp[i][mask[i]] = self.displace(sample_coordinates[i][mask[i]], latentws[i][0], smplpara[i])
            # #print(samplepts_disp[i][mask[i]])
            # #sample_coordinates[i][mask[i]] = sample_coordinates[i][mask[i]]+samplepts_disp[i][mask[i]]
        # #samplepts_disp = self.displace(sample_coordinates, latentws, smplpara)
        samplepts_disp = 0
        
        # sample_coordinates = self.get_canonical_coordinates(
            # sample_coordinates,
            # smpl_src=smpl_reduced_current,
            # smpl_dst=smpl_reduced_canon,  
            # mask = mask            
        # )
        
        # for i in range(smpl_params.shape[0]):
            # sample_coordinates[i][mask[i]] = sample_coordinates[i][mask[i]]+samplepts_disp[i][mask[i]]
               
        #sample_coordinates[~mask] = 10
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates.obj')   
        
        # npvertices = smpl_reduced_canon.vertices[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/canonmesh.obj')
        
        # smvert = self.get_canonical_coordinates(
            # smpl_reduced_current.vertices,
            # smpl_src=smpl_reduced_current,
            # smpl_dst=smpl_reduced_canon          
        # )
        #smvert = smpl_reduced_canon.vertices
        # t = mask_at_box.reshape(-1,128,128)
        # bone_mask = bone_mask*t
        # bonemask = bone_mask[0].detach().cpu().numpy()
        # cv2.imwrite('test/bonemask.png', bonemask * 255)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates_can.obj') 
        
        # R = extrinsics[:,:3,:3] 
        # T = extrinsics[:,:3,3:] 
        # localvert = torch.matmul(R,smpl_reduced_current.vertices.permute(0,2,1))+T
        # projpixel = torch.matmul(intrinsic, localvert)
        # projpixel = projpixel/localvert[:,2,:]
        # print('smpl:',projpixel)
        # globaljoint = pose_to_world[:,:, :3, 3]
        # localvert = torch.matmul(R,globaljoint.permute(0,2,1))+T
        # projpixel = torch.matmul(intrinsic, localvert)
        # projpixel = projpixel/localvert[:,2,:]
        # print('joint:',projpixel)
        
        #mask = mask_at_box.unsqueeze(-1).unsqueeze(-1).repeat(1,1,Nc,1).view(sample_coordinates.shape[0],sample_coordinates.shape[1],1)
               
        # rgb_vals = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],32], dtype=torch.float32)#normlized to [-1,1]
        # sigma_vals = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32) * 1e3
        # #print(sample_coordinates.shape,mask.shape,sample_coordinates[mask].shape)
        #color, density = self.fetchingnerf(planes, sample_coordinates[0][mask[0]][None,...])
        
        #sample_coordinates.requires_grad_(True)
        color, density = self.fetchingnerf(planes, sample_coordinates)
        # sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros')
        # sampled_features = sampled_features.mean(1)
        # density, color = calc_density_and_color_from_feature(self, sampled_features.permute(0,2,1), z_rend, sample_coordinates)
        
        #density = self.density_activation(density) * 10
        # density = density.permute(0,2,1)
        # color = color.permute(0,2,1)
        
        #density *= meshhumanmask.unsqueeze(-1)
        
        #density *= mask.unsqueeze(-1)
        
        # canonical sdf regularization smpl_reduced_canon.vertices
        meshpts_smpl, samplepts, meshpts_cloth = self.samplingpoint_learningmeshsdf_smpl(self.smpl_reduced_canon_vert.repeat(batch_size,1,1), self.cloth_reduced_canon.repeat(batch_size,1,1))
        
        samplepts.requires_grad_(True)
        _, samplepts_density = self.fetchingnerf(planes, samplepts)
        offmesh_sdf = samplepts_density.reshape(batch_size,-1)
        
        d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        sdfgradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      inputs=samplepts,
                                       grad_outputs=d_output,
                                      create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        
        _, meshpts_smpl_sdf = self.fetchingnerf(planes, meshpts_smpl)
        _, meshpts_cloth_sdf = self.fetchingnerf(planes, meshpts_cloth)
        
        # offmesh_sdf = 0
        # sdfgradients = 0
        # meshpts_smpl_sdf = 0
        # meshpts_cloth_sdf = 0
        
        # offmesh_sdf = density.reshape(batch_size,-1)
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdfgradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=sample_coordinates,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # meshpts_smpl_sdf = 0                               
                                       
        
        rgb_vals = color#[mask]
        sigma_vals = density#[mask]
        
        colors_coarse = rgb_vals
        densities_coarse = sigma_vals
        
        # colors_coarse = out['rgb']
        # densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        
       
        densities_coarse = sdf_to_alpha(densities_coarse.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s * 1.0))  
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        #print(colors_coarse[densities_coarse.repeat(1,1,1,3)>0])
        # Mask out invalid samples (optional).
        is_sample_valid = None
        # if smpl_clip_depths is not None:
            # is_sample_valid = self.get_sample_mask(sample_depths=depths_coarse, min_max_depths=smpl_clip_depths)
            # densities_coarse = densities_coarse - 1000 * (1-is_sample_valid.float())

        # Fine Pass
        N_importance = 0#64#rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, batch_size)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            if 0:#rendering_options['box_warp_pre_deform']:
                sample_coordinates = (2 / rendering_options['box_warp']) * sample_coordinates
            sample_coordinates = self.get_canonical_coordinates(
                                sample_coordinates,
                                smpl_src=smpl_reduced_current,
                                smpl_dst=smpl_reduced_canon
                                )
            colors_fine, densities_fine = self.fetchingnerf(planes, sample_coordinates)
            # out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            # colors_fine = out['rgb']
            # densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            # Mask out invalid samples (optional).
            if 0:#smpl_clip_depths is not None:
                is_sample_valid = self.get_sample_mask(sample_depths=depths_fine, min_max_depths=smpl_clip_depths)
                densities_fine = densities_fine - 1000 * (1-is_sample_valid.float())
                #colors_fine = colors_fine * is_sample_valid.float()

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(all_colors, all_densities, all_depths, batch_size)
        else:
            rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, batch_size)
             
        #print(rgb_final.shape,is_valid.shape)rgb_final, depth_final, weights
        # rgb = rgb_final.reshape(2,64,64,32)
        # print(rgb.shape)
        #print(rgb_final[depth_final.squeeze(-1)>0])
        if is_sample_valid is not None: depth_final = is_sample_valid.any(-2).float()

        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf#rgb_final, depth_final, weights.sum(2)
            
    def nerf_rendering_deepcap0(self, frameidx, rawimg, Nc, Nf, z_rend, planes, latentws, smpl_params, betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box):

        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]

        #smpl_betas = smpl_params[:, 72:82]
        #smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        
         # Create stratified depth samples
        ray_start = near
        ray_end = far
        depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, Nc)

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
        bs_expand = batch_size 
         
        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)       
        
        # smpl_reduced_current = self.smpl_reduced(betas=smpl_betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=smpl_translate)
        # smpl_reduced_canon = self.smpl_reduced(betas=self.smpl_avg_betas.expand(bs_expand, -1),
                                               # body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1),
                                               # global_orient=self.smpl_avg_orient.expand(bs_expand, -1),
                                               # transl=self.smpl_avg_transl.expand(bs_expand, -1))

        # smpl_reduced_current.transl = smpl_translate
        # smpl_reduced_canon.transl = self.smpl_avg_transl.expand(bs_expand, -1)
        # smpl_reduced_current.vertices *= smplscale#self.smpl_avg_scale
        # smpl_reduced_canon.vertices *= self.smpl_avg_scale
        # print(smplscale)
        #with torch.no_grad():
        #reduced smpl        self.canonparamshape
        smpl_reduced_current = self.smpl_reduced(betas=betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=self.smpl_avg_transl.expand(bs_expand, -1))
      
        smpl_reduced_canon = self.smpl_reduced(betas=betas, body_pose=self.smpl_avg_body_pose.expand(batch_size, -1),
                                      global_orient=self.smpl_avg_orient.expand(batch_size, -1), transl=self.smpl_avg_transl.expand(bs_expand, -1))
        
        smpl_reduced_current.vertices += Th[:,None]
        
        # canpts = self.get_canonical_coordinates(
            # smpl_reduced_current.vertices,
            # smpl_src=smpl_reduced_current,
            # smpl_dst=smpl_reduced_canon           
        # )
        # npvertices = canpts[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/canpts.obj')
        # npvertices = smpl_reduced_canon.vertices[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/smpl_reduced_canon.obj')
        
        # smpl_current0 = self.smpl_reduced.rawSMPL(betas=betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=self.smpl_avg_transl.expand(bs_expand, -1))
        # smpl_current0.vertices += Th[:,None]
      
        # smpl_current = self.deformingsmpl_LBS(A, Th)
        
        # npvertices = smpl_current0.vertices[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/smpl_current0.obj')   
        
        # npvertices = smpl_current[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/smpl_current.obj')
        
        #deforming with cloth and smpl
        # smpl_current = self.deformingsmpl_LBS(A, Th)
        # smpl_reduced_current_vert = self.smpl_reduced.simplifysmpl(smpl_current)
        
        # self.predicting_deformation(latentws)
        # canondeformedcloth = self.deformingtemplate()
        # deform_smoothloss = self.deformationsmoothloss()
        
        # cloth_current = self.deformingcloth_LBS(canondeformedcloth, A, Th)
        # cloth_reduced_current = self.cloth_reduced(cloth_current)
        
        #interp_loss = self.computeinterpenetrationloss_posedsmpl(canondeformedcloth, smpl_current, cloth_current)
        
        # smpl_ptsdist = torch.cdist(sample_coordinates, smpl_reduced_current_vert, p=2)#deformedpersonsmpl
        # smpl_ptsdistmin = torch.min(smpl_ptsdist, 2)
        # smpl_distmin = smpl_ptsdistmin[0]
        # smpl_minidx = torch.squeeze(smpl_ptsdistmin[1], -1) 
        # cloth_ptsdist = torch.cdist(sample_coordinates, cloth_reduced_current, p=2)#deformedpersonsmpl
        # cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        # cloth_distmin = cloth_ptsdistmin[0]
        # cloth_minidx = torch.squeeze(cloth_ptsdistmin[1], -1) 
        
        # cloth_neartag = cloth_ptsdist<smpl_distmin
        
        # sample_coordinates_smpl = self.get_canonical_coordinates(
            # sample_coordinates,
            # smpl_src=SMPLOutput(vertices=smpl_reduced_current_vert),
            # smpl_dst=SMPLOutput(vertices=self.smpl_reduced_canon_vert.repeat(batch_size,1,1)),  
            # mask = mask            
        # )
        
        # sample_coordinates_cloth = self.get_canonical_coordinates_clothvertices(
            # sample_coordinates,
            # src_vertices=cloth_reduced_current,
            # dst_vertices=self.cloth_reduced_canon.repeat(batch_size,1,1)),  
            # mask = mask            
        # )
        # sample_coordinates = sample_coordinates_smpl
        # sample_coordinates.view(-1,3)[cloth_neartag] = sample_coordinates_cloth.view(-1,3)[cloth_neartag]
        # sample_coordinates = sample_coordinates.view(batch_size,-1,3)
        
        
        #smpl model, vert transformation
        # smplvert, A, T = get_shape(self.smpl, body_pose=smpl_body_pose.view(batch_size,23,3),
                                       # global_orient=smpl_orient[:,None]) 
        # smplvert = torch.matmul(trans[:,:3,:3], smplvert.permute(0,2,1))
        # smplvert = smplvert.permute(0,2,1)
        # smplvert += shift[:,None]
        # # axis_transform
        # smplvert = smplvert[:, :, [1, 2, 0]] * torch.tensor([-1, -1, -1]).view(1,3)[:, None].to(betas.device)
        # mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])

        # nnvidx = mask.new_zeros([mask.shape[0],mask.shape[1]], dtype=torch.long)              
        # for i in range(0, smplvert.shape[0], 1):
            # # tnum = int(sample_coordinates[i][mask[i]].shape[0]/2)
            # # pts = sample_coordinates[i][mask[i]][:tnum]                       
            # # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][:tnum], smplvert[i], p=2)#deformedpersonsmpl
            # # smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            # # vidx = torch.squeeze(smpl_ptsdistmin[1], -1)  # B*P
            # # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][tnum:], smplvert[i], p=2)#deformedpersonsmpl
            # # smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
            # # vidx1 = torch.squeeze(smpl_ptsdistmin[1], -1)
            # # vidx = torch.cat((vidx,vidx1),0)
            # # nnvidx[i][mask[i]] = vidx

            # vidx = []
            # tnum = int(sample_coordinates[i][mask[i]].shape[0]/4)          
            # for j in range(4):             
                # if j==3:
                    # #smpl_ptsdist = ((sample_coordinates[i][mask[i]][j*tnum:][:,None,:] - smplvert[i][None,:,:])**2).sum(-1) # Pts x V       
                    # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][j*tnum:], smplvert[i], p=2)
                # else:
                    # #smpl_ptsdist = ((sample_coordinates[i][mask[i]][j*tnum:(j+1)*tnum][:,None,:] - smplvert[i][None,:,:])**2).sum(-1) # Pts x V       
                    # smpl_ptsdist = torch.cdist(sample_coordinates[i][mask[i]][j*tnum:(j+1)*tnum], smplvert[i], p=2)#deformedpersonsmpl
                # smpl_ptsdistmin = torch.min(smpl_ptsdist, 1)
                # tidx = torch.squeeze(smpl_ptsdistmin[1], -1)  # B*P
                # vidx += [tidx]
            # vidx = torch.cat(vidx, 0)
            # nnvidx[i][mask[i]] = vidx
            
        # sample_coordinates = self.invtransform_surreal(sample_coordinates, shift, trans)
        
        # sample_coordinates = self.inversedeforming_samplepoints_LBS(sample_coordinates, nnvidx, smplvert, T)
        
        # mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        # #sample_coordinates[~mask] = sample_coordinates[~mask] + 10
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates.obj')   
        
        # npvertices = self.smpl.v_template.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/canonmesh.obj')
        
        #raw smpl
        # smpl_reduced_current_vert = get_shape(self.smpl, body_pose=smpl_body_pose.view(batch_size,23,3),
                                      # global_orient=smpl_orient[:,None])  
        # smpl_reduced_current_vert *= smplscale
        # smpl_reduced_current_vert += smpl_translate[:,None] / 100
        # smpl_reduced_current = SMPLOutput(vertices=smpl_reduced_current_vert)
        
        # smpl_reduced_canon_vert = get_shape(self.smpl, body_pose=self.smpl_avg_body_pose.expand(batch_size, -1).view(batch_size,23,3),
                                      # global_orient=self.smpl_avg_orient.expand(batch_size, -1)[:,None])
        # smpl_reduced_canon_vert *= smplscale
        # #smpl_reduced_canon_vert += self.smpl_avg_transl.expand(3, -1)[:,None]                                      
        # smpl_reduced_canon = SMPLOutput(vertices=smpl_reduced_canon_vert)self.smpl_reduced.faces_t, smpl_reduced_current.vertices
        
        # rendermask = self.rasterize_eg3d(self.smplfaces, smplvert, extrinsics, intrinsic)
        # for i in range(0,1):
            # mask_render = rendermask[i].detach().cpu().numpy()
            # mask_path = 'test/smplmask{:04d}.png'.format(frameidx[i].item())       
            # cv2.imwrite(mask_path, mask_render * 255)            
            # raw_img = rawimg.permute(0,2,3,1)[i].detach().cpu().numpy()
            # im_path = 'test/rawimg{:04d}.png'.format(frameidx[i].item())       
            # cv2.imwrite(im_path, raw_img)
        #bonemask = bone_mask[0].detach().cpu().numpy()
        #cv2.imwrite('test/bonemask.png', bonemask * 255)
        
        # smpl_reduced_canon_vert = get_shape(self.smpl, body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1).view(batch_size,23,3),
                                      # global_orient=self.smpl_avg_orient.expand(bs_expand, -1)[:,None])
        # smpl_reduced_canon_vert *= self.smpl_avg_scale
        # smpl_reduced_canon_vert += self.smpl_avg_transl.expand(bs_expand, -1)[:,None]                                      
        # smpl_reduced_canon = SMPLOutput(vertices=smpl_reduced_canon_vert)

        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinatesraw.obj')   
        
        # npvertices = smpl_reduced_current.vertices[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/currmesh.obj')
        
        mask = mask_at_box.unsqueeze(-1).repeat(1,1,Nc).view(sample_coordinates.shape[0],sample_coordinates.shape[1])
        #displacement
        # smplpara = torch.cat([smpl_params, betas], -1)
        # samplepts_disp =  torch.zeros_like(sample_coordinates)   
        # for i in range(smpl_params.shape[0]):
            # samplepts_disp[i][mask[i]] = self.displace(sample_coordinates[i][mask[i]], latentws[i][0], smplpara[i])
            # #print(samplepts_disp[i][mask[i]])
            # #sample_coordinates[i][mask[i]] = sample_coordinates[i][mask[i]]+samplepts_disp[i][mask[i]]
        # #samplepts_disp = self.displace(sample_coordinates, latentws, smplpara)
        samplepts_disp = 0
        
        sample_coordinates = self.get_canonical_coordinates(
            sample_coordinates,
            smpl_src=smpl_reduced_current,
            smpl_dst=smpl_reduced_canon,  
            mask = mask            
        )
        # for i in range(smpl_params.shape[0]):
            # sample_coordinates[i][mask[i]] = sample_coordinates[i][mask[i]]+samplepts_disp[i][mask[i]]
               
        #sample_coordinates[~mask] = 10
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # npfaces = self.smpl.faces
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates.obj')   
        
        # npvertices = smpl_reduced_canon.vertices[0].detach().cpu().numpy()
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/canonmesh.obj')
        
        # smvert = self.get_canonical_coordinates(
            # smpl_reduced_current.vertices,
            # smpl_src=smpl_reduced_current,
            # smpl_dst=smpl_reduced_canon          
        # )
        #smvert = smpl_reduced_canon.vertices
        # t = mask_at_box.reshape(-1,128,128)
        # bone_mask = bone_mask*t
        # bonemask = bone_mask[0].detach().cpu().numpy()
        # cv2.imwrite('test/bonemask.png', bonemask * 255)
        
        # npvertices = sample_coordinates[0].detach().cpu().numpy()                        
        # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # mesh.export('test/sample_coordinates_can.obj') 
        
        # R = extrinsics[:,:3,:3] 
        # T = extrinsics[:,:3,3:] 
        # localvert = torch.matmul(R,smpl_reduced_current.vertices.permute(0,2,1))+T
        # projpixel = torch.matmul(intrinsic, localvert)
        # projpixel = projpixel/localvert[:,2,:]
        # print('smpl:',projpixel)
        # globaljoint = pose_to_world[:,:, :3, 3]
        # localvert = torch.matmul(R,globaljoint.permute(0,2,1))+T
        # projpixel = torch.matmul(intrinsic, localvert)
        # projpixel = projpixel/localvert[:,2,:]
        # print('joint:',projpixel)
        
        #mask = mask_at_box.unsqueeze(-1).unsqueeze(-1).repeat(1,1,Nc,1).view(sample_coordinates.shape[0],sample_coordinates.shape[1],1)
               
        # rgb_vals = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],32], dtype=torch.float32)#normlized to [-1,1]
        # sigma_vals = -sample_coordinates.new_zeros([sample_coordinates.shape[0],sample_coordinates.shape[1],1], dtype=torch.float32) * 1e3
        # #print(sample_coordinates.shape,mask.shape,sample_coordinates[mask].shape)
        #color, density = self.fetchingnerf(planes, sample_coordinates[0][mask[0]][None,...])
        
        #sample_coordinates.requires_grad_(True)
        color, density = self.fetchingnerf(planes, sample_coordinates)
        # sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros')
        # sampled_features = sampled_features.mean(1)
        # density, color = calc_density_and_color_from_feature(self, sampled_features.permute(0,2,1), z_rend, sample_coordinates)
        
        #density = self.density_activation(density) * 10
        # density = density.permute(0,2,1)
        # color = color.permute(0,2,1)

        #density *= mask.unsqueeze(-1)
        
        # canonical sdf regularization
        meshpts_smpl, samplepts = self.samplingpoint_learningmeshsdf_smpl(smpl_reduced_canon.vertices)
        
        #samplepts.requires_grad_(True)
        #_, samplepts_density = self.fetchingnerf(planes, samplepts)
        #offmesh_sdf = samplepts_density.reshape(batch_size,-1)
        offmesh_sdf = 0
        #d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdfgradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=samplepts,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        sdfgradients = 0
        _, meshpts_smpl_sdf = self.fetchingnerf(planes, meshpts_smpl)
        #meshpts_smpl_sdf = 0
        
        # offmesh_sdf = density.reshape(batch_size,-1)
        # d_output = torch.ones_like(offmesh_sdf, requires_grad=False, device=offmesh_sdf.device)#
        # sdfgradients = torch.autograd.grad(outputs=offmesh_sdf,
                                      # inputs=sample_coordinates,
                                       # grad_outputs=d_output,
                                      # create_graph=True,
                                       # retain_graph=True,
                                       # only_inputs=True)[0]
        # meshpts_smpl_sdf = 0                               
                                       
        
        rgb_vals = color#[mask]
        sigma_vals = density#[mask]
        
        colors_coarse = rgb_vals
        densities_coarse = sigma_vals
        
        # colors_coarse = out['rgb']
        # densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        
       
        densities_coarse = sdf_to_alpha(densities_coarse.reshape(batch_size, num_rays, samples_per_ray), torch.exp(self.ln_s * 1.0))  
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        #print(colors_coarse[densities_coarse.repeat(1,1,1,3)>0])
        # Mask out invalid samples (optional).
        is_sample_valid = None
        # if smpl_clip_depths is not None:
            # is_sample_valid = self.get_sample_mask(sample_depths=depths_coarse, min_max_depths=smpl_clip_depths)
            # densities_coarse = densities_coarse - 1000 * (1-is_sample_valid.float())

        # Fine Pass
        N_importance = 0#64#rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, batch_size)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            if 0:#rendering_options['box_warp_pre_deform']:
                sample_coordinates = (2 / rendering_options['box_warp']) * sample_coordinates
            sample_coordinates = self.get_canonical_coordinates(
                                sample_coordinates,
                                smpl_src=smpl_reduced_current,
                                smpl_dst=smpl_reduced_canon
                                )
            colors_fine, densities_fine = self.fetchingnerf(planes, sample_coordinates)
            # out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            # colors_fine = out['rgb']
            # densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            # Mask out invalid samples (optional).
            if 0:#smpl_clip_depths is not None:
                is_sample_valid = self.get_sample_mask(sample_depths=depths_fine, min_max_depths=smpl_clip_depths)
                densities_fine = densities_fine - 1000 * (1-is_sample_valid.float())
                #colors_fine = colors_fine * is_sample_valid.float()

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(all_colors, all_densities, all_depths, batch_size)
        else:
            rendered_color, rendered_mask, rendered_disparity, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, batch_size)
             
        #print(rgb_final.shape,is_valid.shape)rgb_final, depth_final, weights
        # rgb = rgb_final.reshape(2,64,64,32)
        # print(rgb.shape)
        #print(rgb_final[depth_final.squeeze(-1)>0])
        if is_sample_valid is not None: depth_final = is_sample_valid.any(-2).float()

        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf#rgb_final, depth_final, weights.sum(2)
        
    def image_synthesis(self, Nc, Nf, model_input: Dict):
                                                            
        z, z_rend = model_input["z"], model_input["z_rend"]
        truncation_psi =  model_input["truncation_psi"]
        
        frameidx = model_input["frameidx"]
        rawimg = model_input["rawimg"]
        
        smplposes, smpltrans = model_input["smplposes"], model_input["smpltrans"]
        ray_origins, ray_directions, near, far, mask_at_box = model_input["ray_o"], model_input["ray_d"],model_input["near"], model_input["far"],model_input["mask_at_box"]
        extrinsics = model_input["extrinsics"]
        intrinsic = model_input["intrinsic"]
        
        smplscale = model_input["smplscale"]
        
        pose_to_world = model_input["pose_to_world"]
        
        bone_length = model_input["bone_length"]
        bone_mask = model_input["bone_mask"]
        
        planes = self.compute_tri_plane_feature(z, bone_length, truncation_psi)#self.compute_tri_plane_feature_our(z, truncation_psi)  
        
        self.buffers_tensors["tri_plane_feature"] = planes
        
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        rendered_color, rendered_mask, rendered_disparity = self.nerf_rendering(frameidx,rawimg,Nc,Nf, z_rend,planes, smplposes.view(-1,24*3), smpltrans, smplscale, pose_to_world, extrinsics, intrinsic, bone_mask, ray_origins, ray_directions, near, far, mask_at_box)
        
        return rendered_color, rendered_mask, rendered_disparity
    
    def image_synthesis_surreal(self, Nc, Nf, model_input: Dict):
                                                            
        z, z_rend = model_input["z"], model_input["z_rend"]
        truncation_psi =  model_input["truncation_psi"]
        
        frameidx = model_input["frameidx"]
        rawimg = model_input["rawimg"]
        
        poses, betas = model_input["poses"],model_input["betas"]
        trans, shift, joints2D = model_input["trans"],model_input["shift"],model_input["joints2D"]
        
        ray_origins, ray_directions, near, far, mask_at_box = model_input["ray_o"], model_input["ray_d"],model_input["near"], model_input["far"],model_input["mask_at_box"]
        extrinsics = model_input["extrinsics"]
        intrinsic = model_input["intrinsic"]
        
        pose_to_world = model_input["pose_to_world"]
        
        bone_length = model_input["bone_length"]
        
        #cond_smplpara = torch.cat([poses.view(-1,24*3),betas],1)     
        #cond_smplpara = torch.cat([trans.view(-1,4*4),pose_to_world.view(-1,24*4*4),betas],1) 
        
        planes, latentws = self.compute_tri_plane_feature_our(z, truncation_psi) #self.compute_tri_plane_feature(z, bone_length, truncation_psi)
        
        self.buffers_tensors["tri_plane_feature"] = planes
        
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # from datetime import datetime
        # filename = datetime.now().strftime('%Y%m%d')#%H%M%S
        # feature = planes[0,0,:3].permute(1,2,0).detach().cpu().numpy()
        # cv2.imwrite('test/feature/'+filename+'feature1.png', feature)
        # feature = planes[0,1,:3].permute(1,2,0).detach().cpu().numpy()
        # cv2.imwrite('test/feature/'+filename+'feature2.png', feature)
        # feature = planes[0,2,:3].permute(1,2,0).detach().cpu().numpy()
        # cv2.imwrite('test/feature/'+filename+'feature3.png', feature)
        
        rendered_feature, rendered_mask, rendered_disparity, samplepts_disp = self.nerf_rendering_surreal(frameidx,rawimg,Nc,Nf, z_rend,planes, latentws, poses.view(-1,24*3), betas, trans, shift, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box)
        batchsize = rendered_feature.shape[0]
        rendered_feature = rendered_feature.reshape(batchsize, -1, 32)
        rendered_color = rendered_feature[..., :3]  
        #    
        feature_image = rendered_feature.permute(0, 2, 1).reshape(batchsize, rendered_feature.shape[-1], 90, 90).contiguous()
        rgb_image = feature_image[:, :3]     
        #ws = self.ws.unsqueeze(1).repeat([1, 8, 1])
        sr_image = self.superresolution(rgb_image, feature_image, latentws)
        
        #rgb_image = torch.nn.functional.interpolate(rgb_image, size=(180, 180), mode='bilinear', align_corners=False)
        img = {}  
        img['image'] = sr_image
        img['image_raw'] = rgb_image#low resolution
  
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, img
    
    def image_synthesis_deepcap_geometry(self, Nc, Nf, model_input: Dict):
                                                            
        z, z_rend = model_input["z"], model_input["z_rend"]
        truncation_psi =  model_input["truncation_psi"]
        
        frameidx = model_input["frameidx"]
        rawimg = model_input["rawimg"]
        
        poses, betas = model_input["poses"],model_input["betas"]
        Th, joints2D = model_input["Th"],model_input["joints2D"]
        A = model_input["A"]
        
        ray_origins, ray_directions, near, far, mask_at_box = model_input["ray_o"], model_input["ray_d"],model_input["near"], model_input["far"],model_input["mask_at_box"]
        extrinsics = model_input["extrinsics"]
        intrinsic = model_input["intrinsic"]
        
        pose_to_world = model_input["pose_to_world"]
        
        bone_length = model_input["bone_length"]
        
        #cond_smplpara = torch.cat([poses.view(-1,24*3),betas],1)     
        #cond_smplpara = torch.cat([trans.view(-1,4*4),pose_to_world.view(-1,24*4*4),betas],1) 
        
        #fix_z = self.latent(torch.LongTensor([0]).to('cuda'))fix_z.repeat(A.shape[0],1)self.shapepara.repeat(A.shape[0],1)

        #planes, latentws = self.compute_tri_plane_feature_our(z, truncation_psi) #self.compute_tri_plane_feature(z, bone_length, truncation_psi)
        # latentws = self.latentws(torch.LongTensor([0]).to('cuda'))
        # latentws = latentws.unsqueeze(1).repeat([A.shape[0], 8, 1])
        planes = 0
        self.buffers_tensors["tri_plane_feature"] = planes
        
        # planes_cloth = planes[:,16*3:,:,:]
        # planes = planes[:,:16*3,:,:]       
        # planes = planes.view(len(planes), 3, 16, planes.shape[-2], planes.shape[-1])
        # planes_cloth = planes_cloth.view(len(planes), 3, 16, planes.shape[-2], planes.shape[-1])
        # planes = torch.cat([planes,planes_cloth],dim=2)
        
        #planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # from datetime import datetime
        # filename = datetime.now().strftime('%Y%m%d')#%H%M%S
        # feature = planes[0,0,:3].permute(1,2,0).detach().cpu().numpy()
        # cv2.imwrite('test/feature/'+filename+'feature1.png', feature)
        # feature = planes[0,1,:3].permute(1,2,0).detach().cpu().numpy()
        # cv2.imwrite('test/feature/'+filename+'feature2.png', feature)
        # feature = planes[0,2,:3].permute(1,2,0).detach().cpu().numpy()
        # cv2.imwrite('test/feature/'+filename+'feature3.png', feature)
        
        # with torch.no_grad(): 
           # rendered_feature, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf,meshpts_cloth_sdf,meshpts_normal,meshptsgradients = self.doublenerf_rendering_deepcap_fixgeometry_color_chunk(frameidx,rawimg,Nc,Nf, z_rend,planes, poses.view(-1,24*3), betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box)
        #rendered_feature, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf,meshpts_cloth_sdf,meshpts_normal,meshptsgradients = self.doublenerf_rendering_deepcap_fixgeometry_color(frameidx,rawimg,Nc,Nf, z_rend,planes, poses.view(-1,24*3), betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box)
        
        # with torch.no_grad(): 
           # rendered_feature, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf,meshpts_cloth_sdf,meshpts_normal,meshptsgradients = self.singlenerf_rendering_deepfashion_fixgeometry_color_chunk(frameidx,rawimg,Nc,Nf, z_rend,planes, poses.view(-1,24*3), betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box)
        
        rendered_feature, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf,meshpts_cloth_sdf,meshpts_normal,meshptsgradients = self.singlenerf_rendering_deepfashion_fixgeometry_color(frameidx,rawimg,Nc,Nf, z_rend,planes, poses.view(-1,24*3), betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box)
        
        #rendered_feature, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf,meshpts_cloth_sdf,meshpts_normal,meshptsgradients = self.singlenerf_rendering_deepcap_geometry_color_deformation(frameidx,rawimg,Nc,Nf, z_rend,planes, poses.view(-1,24*3), betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box)
        
        #rendered_feature, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf,meshpts_cloth_sdf,meshpts_normal,meshptsgradients,rendered_color_local = self.singlenerf_rendering_deepcap_twodiscriminator(frameidx,rawimg,Nc,Nf, z_rend,planes, poses.view(-1,24*3), betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box)
        
        #rendered_feature, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf,meshpts_cloth_sdf,meshpts_normal,meshptsgradients = self.doublenerf_rendering_deepcap(frameidx,rawimg,Nc,Nf, z_rend,planes, latentws, poses.view(-1,24*3), betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box)
        
        batchsize = rendered_feature.shape[0]
        rendered_feature = rendered_feature.reshape(batchsize, -1, 3)#16
        rendered_color = rendered_feature[..., :3]  
        #    
        feature_image = rendered_feature.permute(0, 2, 1).reshape(batchsize, rendered_feature.shape[-1], self.ray_sampler.N_samples_sqrt*2,self.ray_sampler.N_samples_sqrt).contiguous()#256, 256
        rgb_image = feature_image[:, :3]     
        #ws = self.ws.unsqueeze(1).repeat([1, 8, 1])
        
        # rendered_feature_local = rendered_color_local.reshape(batchsize, -1, 3)#16
        # feature_image_local = rendered_feature_local.permute(0, 2, 1).reshape(batchsize, rendered_feature_local.shape[-1], self.ray_sampler.N_samples_sqrt,self.ray_sampler.N_samples_sqrt).contiguous()#256, 256
        # rgb_image_local = feature_image_local[:, :3] 
        # rgb_image = torch.cat([rgb_image,rgb_image_local],dim=1)
        
        # fgcolor = rgb_image.permute(0,2,3,1)[0].detach().cpu().numpy()
        # fgcolor = cv2.cvtColor(fgcolor, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('result/deepcap/rendercolor.png', fgcolor*127.5+127.5)
        
        #sr_image = self.superresolution(rgb_image, feature_image, latentws)
        
        #rgb_image = torch.nn.functional.interpolate(rgb_image, size=(180, 180), mode='bilinear', align_corners=False)
        img = {}  
        img['image'] = rgb_image#sr_image
        img['image_raw'] = rgb_image#low resolution 
        
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients, img
    
    def image_synthesis_deepcap(self, Nc, Nf, model_input: Dict):#
                                                            
        z, z_rend = model_input["z"], model_input["z_rend"]
        truncation_psi =  model_input["truncation_psi"]
        
        frameidx = model_input["frameidx"]
        rawimg = model_input["rawimg"]
        
        poses, betas = model_input["poses"],model_input["betas"]
        Th, joints2D = model_input["Th"],model_input["joints2D"]
        A = model_input["A"]
        
        ray_origins, ray_directions, near, far, mask_at_box = model_input["ray_o"], model_input["ray_d"],model_input["near"], model_input["far"],model_input["mask_at_box"]
        extrinsics = model_input["extrinsics"]
        intrinsic = model_input["intrinsic"]
        
        pose_to_world = model_input["pose_to_world"]
        
        bone_length = model_input["bone_length"]
        
        #cond_smplpara = torch.cat([poses.view(-1,24*3),betas],1)     
        #cond_smplpara = torch.cat([trans.view(-1,4*4),pose_to_world.view(-1,24*4*4),betas],1) 
        
        #fix_z = self.latent(torch.LongTensor([0]).to('cuda'))#self.shapepara.repeat(A.shape[0],1)

        #planes, latentws = self.compute_tri_plane_feature_our(fix_z.repeat(A.shape[0],1), truncation_psi) #self.compute_tri_plane_feature(z, bone_length, truncation_psi)
        planes = 0
        latentws = 0
        self.buffers_tensors["tri_plane_feature"] = planes
        
        # planes_cloth = planes[:,16*3:,:,:]
        # planes = planes[:,:16*3,:,:]       
        # planes = planes.view(len(planes), 3, 16, planes.shape[-2], planes.shape[-1])
        # planes_cloth = planes_cloth.view(len(planes), 3, 16, planes.shape[-2], planes.shape[-1])
        # planes = torch.cat([planes,planes_cloth],dim=2)singlenerf_rendering_deepfashion_geometry_smpl
        
        rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf,meshpts_cloth_sdf,meshpts_normal,meshptsgradients = self.doublenerf_rendering_deepcap_geometry(frameidx,rawimg,Nc,Nf, z_rend,planes, poses.view(-1,24*3), betas, Th, A, joints2D, pose_to_world, extrinsics, intrinsic, ray_origins, ray_directions, near, far, mask_at_box)
        img = {}  
  
        return rendered_color, rendered_mask, rendered_disparity, samplepts_disp, sdfgradients,offmesh_sdf,meshpts_smpl_sdf, meshpts_cloth_sdf, meshpts_normal,meshptsgradients, img
            
    def calc_density_and_color_canonicalframe(self, position: torch.Tensor, pose_to_camera: torch.Tensor,
                                                    ray_direction: torch.Tensor, model_input: Dict):
                                                            
        z, z_rend = model_input["z"], model_input["z_rend"]
        truncation_psi =  model_input["truncation_psi"]
        bone_length = model_input["bone_length"]
        
        smplposes, smpltrans = model_input["smplposes"], model_input["smpltrans"]
        smplscale = model_input["smplscale"]
        
        planes = self.compute_tri_plane_feature_our(z, truncation_psi)  
        
        self.buffers_tensors["tri_plane_feature"] = planes
        
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
       
        sample_coordinates = self.obtaining_canonicalpoints(position.permute(0,2,1), smplposes.view(-1,24*3), smpltrans, smplscale)
        
        color, density = self.fetchingnerf(planes, sample_coordinates)
        
        if self.config.multiply_density_with_triplane_wieght:
            density = self.density_activation(density) * (10 * weight.max(dim=1, keepdim=True)[0])
        else:
            density = self.density_activation(density) * 10
        
        # to local and canonical coordinate (challenge: this is heavy (B, n_bone * 3, n))
        local_points, canonical_points = self.to_local_and_canonical(position, pose_to_camera, bone_length)

        in_cube_p = in_cube(local_points)  # (B, n_bone, n)
        in_cube_p = in_cube_p * (canonical_points.abs() < 1).all(dim=2)  # (B, n_bone, n)
        in_cube_mask = in_cube_p.any(dim=1, keepdim=True)
        in_cube_mask = in_cube_mask.squeeze(1)[...,None]
        density *= in_cube_mask  # density is 0 if not in cube
        
        return density, color
                
    def calc_density_and_color_from_camera_coord_v2(self, position: torch.Tensor, pose_to_camera: torch.Tensor,
                                                    ray_direction: torch.Tensor, model_input: Dict):
        """compute density from positions in camera coordinate

        :param position: (B, 3, n), n is a very large number of points sampled
        :param pose_to_camera:
        :param ray_direction:
        :param model_input: dictionary of model input
        :return: density of input positions
        """
        bone_length, z, z_rend = model_input["bone_length"], model_input["z"], model_input["z_rend"]
        tri_plane_feature, truncation_psi = model_input.get("tri_plane_feature"), model_input["truncation_psi"]

        if self.tri_plane_based:
            if tri_plane_feature is None:
                z = self.compute_tri_plane_feature(z, bone_length, truncation_psi)
            else:
                z = tri_plane_feature
            self.buffers_tensors["tri_plane_feature"] = z
            if not self.training:
                self.temporal_state["tri_plane_feature"] = z
      
        # to local and canonical coordinate (challenge: this is heavy (B, n_bone * 3, n))
        local_points, canonical_points = self.to_local_and_canonical(position, pose_to_camera, bone_length)

        in_cube_p = in_cube(local_points)  # (B, n_bone, n)
        in_cube_p = in_cube_p * (canonical_points.abs() < 1).all(dim=2)  # (B, n_bone, n)
        density, color = self.backbone(canonical_points, in_cube_p, z, z_rend, bone_length, "weight_feature",
                                       ray_direction)
        density *= in_cube_p.any(dim=1, keepdim=True)  # density is 0 if not in cube

        if not self.training:
            self.temporal_state.update({
                "canonical_fine_points": canonical_points,
                "in_cube": in_cube(local_points),
            })
        return density, color

    def backbone(self, p: torch.Tensor, position_validity: torch.Tensor, tri_plane_feature: torch.Tensor,
                 z_rend: torch.Tensor, bone_length: torch.Tensor, mode: str = "weight_feature",
                 ray_direction: Optional[torch.Tensor] = None):
        """

        Args:
            p: position in canonical coordinate, (B, n_bone, 3, n)
            position_validity: bool tensor for validity of p, (B, n_bone, n)
            tri_plane_feature:
            z_rend: (B, dim)
            bone_length: (B, n_bone)
            mode: "weight_feature" or "weight_position"
            ray_direction: not None if color is view dependent
        Returns:

        """
        # don't support mip-nerf rendering
        assert isinstance(p, torch.Tensor)
        assert bone_length is not None
        assert mode in ["weight_position", "weight_feature"]

        bs, n_bone, _, n = p.shape

        # Make the invalid position outside the range of -1 to 1 (all invalid positions become 2)
        masked_position = p * position_validity[:, :, None] + 2 * ~position_validity[:, :, None]

        weight = self.calc_weight(tri_plane_feature[:, 32 * 3:].reshape(bs * n_bone, 3, 256, 256),
                                  masked_position, position_validity)  # (bs, n_bone, n)

        if not self.training:
            self.temporal_state.update({
                "weight": weight,
            })
        if weight.requires_grad:
            if not hasattr(self, "buffers_tensors"):
                self.buffers_tensors = {}

            self.buffers_tensors["mask_weight"] = weight

        # default mode is "weight_feature"
        # weighted sum of tri-plane features
        if mode == "weight_feature":
            feature = sample_weighted_feature_v2(self.feat_dim, tri_plane_feature[:, :32 * 3], masked_position,
                                                 weight, position_validity,
                                                 clamp_mask=self.config.clamp_mask)  # (B, 32, n)
        # canonical position based
        elif mode == "weight_position":
            weighted_position_validity = position_validity.any(dim=1)[:, None]
            weighted_position = (p * weight[:, :, None]).sum(dim=1)  # (bs, 3, n)
            # Make the invalid position outside the range of -1 to 1 (all invalid positions become 2)
            weighted_position = weighted_position * weighted_position_validity + 2 * ~weighted_position_validity
            feature = sample_feature(tri_plane_feature[:, :32 * 3], weighted_position,
                                     clamp_mask=self.config.clamp_mask, )  # (B, 32, n)
        else:
            raise ValueError()

        density, color = calc_density_and_color_from_feature(self, feature, z_rend, ray_direction)

        if self.config.multiply_density_with_triplane_wieght:
            density = self.density_activation(density) * (10 * weight.max(dim=1, keepdim=True)[0])
        else:
            density = self.density_activation(density) * 10
        return density, color

    def compute_tri_plane_feature(self, z, bone_length, truncation_psi=1):
        """
        Generate triplane features with stylegan
        :param z:
        :param bone_length:
        :param truncation_psi:
        :return:
        """
        # generate tri-plane feature conditioned on z and bone_length
        encoded_length = multi_part_positional_encoding(bone_length, self.num_frequency_for_other,
                                                        num_bone=self.num_bone)
        tri_plane_feature = self.tri_plane_gen(z, encoded_length[:, :, 0],
                                               truncation_psi=truncation_psi)  # (B, (32 + n_bone) * 3, h, w)
        return tri_plane_feature

    def compute_tri_plane_feature_our(self, z, truncation_psi=1):
        """
        Generate triplane features with stylegan
        :param z:
        :param bone_length:
        :param truncation_psi:
        :return:
        """
        # generate tri-plane feature conditioned on z
        tri_plane_feature, latentws = self.tri_plane_gen(z,0,
                                               truncation_psi=truncation_psi)  # (B, (32 + n_bone) * 3, h, w)
        return tri_plane_feature, latentws
        
    def fetchingnerf(self, planes, sample_coordinates):
        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros')
        #print(sampled_features.shape)
        rgb, sigma = self.decoder(sampled_features)
        # if options.get('density_noise', 0) > 0:
            # out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        return rgb, sigma
    
    def fetchingdoublenerf(self, planes_smpl, planes_cloth, sample_coordinates_smpl, sample_coordinates_cloth):
        sampled_features = sample_from_planes(self.plane_axes, planes_smpl, sample_coordinates_smpl, padding_mode='zeros')

        sigma = self.decoder(sampled_features)
        
        sampled_features = sample_from_planes(self.plane_axes, planes_cloth, sample_coordinates_cloth, padding_mode='zeros')
        
        sigma_cloth = self.decoder_cloth(sampled_features)
        # if options.get('density_noise', 0) > 0:rgb_cloth, 
            # out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        return sigma, sigma_cloth
    
    def fetchingdoublenerf_fixgeometry_color(self, planes_smpl, planes_cloth, sample_coordinates_smpl, sample_coordinates_cloth):
        sampled_features = sample_from_planes(self.plane_axes, planes_smpl, sample_coordinates_smpl, padding_mode='zeros')

        rgb = self.decoder(sampled_features)
        
        sampled_features = sample_from_planes(self.plane_axes, planes_cloth, sample_coordinates_cloth, padding_mode='zeros')
        
        rgb_cloth = self.decoder_cloth(sampled_features)
        # if options.get('density_noise', 0) > 0:rgb_cloth, 
            # out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        return rgb, rgb_cloth
        
    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities
        
    def obtaining_canonicalpoints(self, sample_coordinates, smpl_params, smpl_translate, smplscale):

        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        #smpl_betas = smpl_params[:, 72:82]
        smpl_betas = smpl_params.new_zeros([smpl_params.shape[0],10], dtype=torch.float32)
        #smpl_translate = smpl_params[:, 82:85]
        batch_size = smpl_params.shape[0]
        #bs_expand = smpl_params.shape[0]
        # smpl_reduced_current = self.smpl_reduced(betas=smpl_betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=smpl_translate)
        # smpl_reduced_canon = self.smpl_reduced(betas=self.smpl_avg_betas.expand(bs_expand, -1),
                                               # body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1),
                                               # global_orient=self.smpl_avg_orient.expand(bs_expand, -1),
                                               # transl=self.smpl_avg_transl.expand(bs_expand, -1))
        # # npvertices = smpl_reduced_canon.vertices[0].detach().cpu().numpy()
        # # npfaces = self.smpl_reduced.faces_t.detach().cpu().numpy()
        # # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # # mesh.export('test/canonmesh.obj')
        
        # # npvertices = smpl_reduced_current.vertices[0].detach().cpu().numpy()
        # # npfaces = self.smpl_reduced.faces_t.detach().cpu().numpy()
        # # mesh = trimesh.Trimesh(npvertices, npfaces, process=False)
        # # mesh.export('test/currmesh.obj')
        
        # smpl_reduced_current.transl = smpl_translate
        # smpl_reduced_canon.transl = self.smpl_avg_transl.expand(bs_expand, -1)
        # smpl_reduced_current.vertices *= self.smpl_avg_scale
        # smpl_reduced_canon.vertices *= self.smpl_avg_scale

        # sample_coordinates = self.get_canonical_coordinates(
            # sample_coordinates,
            # smpl_src=smpl_reduced_current,
            # smpl_dst=smpl_reduced_canon
        # )
        
        smpl_reduced_current_vert = get_shape(self.smpl, body_pose=smpl_body_pose.view(batch_size,23,3),
                                      global_orient=smpl_orient[:,None])  
        smpl_reduced_current_vert *= smplscale
        smpl_reduced_current_vert += smpl_translate[:,None] / 100
        smpl_reduced_current = SMPLOutput(vertices=smpl_reduced_current_vert)
        
        sample_coordinates = self.get_canonical_coordinates(
            sample_coordinates,
            smpl_src=smpl_reduced_current,
            smpl_dst=self.smpl_reduced_canon
        )
        return sample_coordinates

    #@torch.no_grad()
    def get_canonical_coordinates(self, coordinates,
                                  smpl_src: SMPLOutput, smpl_dst: SMPLOutput,
                                  mask=None):
        """
        # coordinates: bs x N x 3
        """
        coordinates_out = coordinates.clone()
        
        if mask is None:
            for i in range(smpl_src.vertices.shape[0]):
                coordinates_out[i, :] = self.surface_field(pts=coordinates[i, :],
                                                           smpl_data=SMPLOutput(vertices=smpl_src.vertices[i:i + 1]),
                                                           smpl_data_0=SMPLOutput(vertices=smpl_dst.vertices[i:i + 1]))
        else:
            for i in range(smpl_src.vertices.shape[0]):
                coordinates_out[i][mask[i]] = self.surface_field(pts=coordinates[i][mask[i]],
                                                                 smpl_data=SMPLOutput(vertices=smpl_src.vertices[i:i + 1]),
                                                                 smpl_data_0=SMPLOutput(vertices=smpl_dst.vertices[i:i + 1]))

        if mask is not None:
            # Handle the otuside case - mask these coordinates to be elsewhere.
            coordinates_out[~mask] = 0#coordinates[~mask] + 10

        return coordinates_out

    #@torch.no_grad()
    def get_canonical_coordinates_clothvertex(self, coordinates, src_vertices, dst_vertices, mask=None):
        """
        # coordinates: bs x N x 3
        """
        coordinates_out = coordinates.clone()
        
        if mask is None:
            for i in range(src_vertices.shape[0]):
                coordinates_out[i, :] = self.cloth_surface_field(coordinates[i, :],
                                                           src_vertices[i:i + 1],
                                                           dst_vertices[i:i + 1])
        else:
            for i in range(src_vertices.shape[0]):
                coordinates_out[i][mask[i]] = self.cloth_surface_field(coordinates[i][mask[i]],
                                                                 src_vertices[i:i + 1],
                                                                 dst_vertices[i:i + 1])

        if mask is not None:
            # Handle the otuside case - mask these coordinates to be elsewhere.
            coordinates_out[~mask] = 0#coordinates[~mask] + 10

        return coordinates_out

class BetaNetwork(nn.Module):
    def __init__(self):
        super(BetaNetwork, self).__init__()
        init_val = 0.1
        self.register_parameter('beta', nn.Parameter(torch.tensor(init_val)))

    def forward(self):
        beta = self.beta
        # beta = torch.exp(self.beta).to(x)
        return beta
        
class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)
        
class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta

        # if sdf<0:
            # return alpha * (1 - 0.5 * torch.expm1(sdf/ beta))
        # else:
            # return alpha * 0.5 * torch.expm1(-sdf/ beta)
        
        # density = torch.where(sdf<0, alpha * (1 - 0.5 * torch.expm1(sdf/ beta)),alpha * 0.5 * torch.expm1(-sdf/ beta))
        # return density
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta
        
class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, std_init=1, freq_init=False, is_first=False):
        super().__init__()
        if is_first:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-1 / in_dim, 1 / in_dim))
        elif freq_init:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-np.sqrt(6 / in_dim) / 25, np.sqrt(6 / in_dim) / 25))
        else:
            self.weight = nn.Parameter(0.25 * nn.init.kaiming_normal_(torch.randn(out_dim, in_dim), a=0.2, mode='fan_in', nonlinearity='leaky_relu'))

        self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), a=-np.sqrt(1/in_dim), b=np.sqrt(1/in_dim)))

        self.bias_init = bias_init
        self.std_init = std_init

    def forward(self, input):
        out = self.std_init * F.linear(input, self.weight, bias=self.bias) + self.bias_init

        return out

# Siren layer with frequency modulation and offset
class FiLMSiren(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, is_first=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        if is_first:
            self.weight = nn.Parameter(torch.empty(out_channel, in_channel).uniform_(-1 / 3, 1 / 3))
        else:
            self.weight = nn.Parameter(torch.empty(out_channel, in_channel).uniform_(-np.sqrt(6 / in_channel) / 25, np.sqrt(6 / in_channel) / 25))

        self.bias = nn.Parameter(nn.Parameter(nn.init.uniform_(torch.empty(out_channel), a=-np.sqrt(1/in_channel), b=np.sqrt(1/in_channel))))
        self.activation = torch.sin

        self.gamma = LinearLayer(style_dim, out_channel, bias_init=30, std_init=15)
        self.beta = LinearLayer(style_dim, out_channel, bias_init=0, std_init=0.25)

    def forward_with_gamma_beta(self, input, gamma, beta):
        out = F.linear(input, self.weight, bias=self.bias)
        out = self.activation(gamma * out + beta)

        return out

    def forward(self, input, style):
        batch, features = style.shape
        out = F.linear(input, self.weight, bias=self.bias)
        gamma = self.gamma(style).view(batch, 1, -1)
        beta = self.beta(style).view(batch, 1, -1)

        out = self.activation(gamma * out + beta)

        return out

# Siren Generator Model
class SirenGenerator(nn.Module):
    def __init__(self, D=8, W=256, style_dim=256, input_ch=3, input_ch_views=3, output_ch=4,
                 output_features=True):
        super(SirenGenerator, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.style_dim = style_dim
        self.output_features = output_features

        self.pts_linears = nn.ModuleList(
            [FiLMSiren(3, W, style_dim=style_dim, is_first=True)] + \
            [FiLMSiren(W, W, style_dim=style_dim) for i in range(D-1)])

        self.views_linears = FiLMSiren(W, W,#input_ch_views + 
                                       style_dim=style_dim)
        self.rgb_linear = LinearLayer(W, 3, freq_init=True)
        self.sigma_linear = LinearLayer(W, 1, freq_init=True)
        self.sigma_linear.bias.data.fill_(0)#

    def mapping_network(self, style):
        batch, _ = style.shape
        gamma_list = [
            self.pts_linears[i].gamma(style).view(batch, 1, -1)
            for i in range(len(self.pts_linears))
        ] + [self.views_linears.gamma(style).view(batch, 1, -1),]
        beta_list = [
            self.pts_linears[i].beta(style).view(batch, 1, -1)
            for i in range(len(self.pts_linears))
        ] + [self.views_linears.beta(style).view(batch, 1, -1),]
        return torch.stack(gamma_list, 0), torch.stack(beta_list, 0)

    def forward_with_gamma_beta(self, x, gamma_list, beta_list):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        mlp_out = input_pts.contiguous()
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i].forward_with_gamma_beta(mlp_out, gamma_list[i], beta_list[i])

        sdf = self.sigma_linear(mlp_out)

        mlp_out = torch.cat([mlp_out, input_views], -1)
        out_features = self.views_linears.forward_with_gamma_beta(mlp_out, gamma_list[-1], beta_list[-1])
        rgb = self.rgb_linear(out_features)

        outputs = torch.cat([rgb, sdf], -1)
        if self.output_features:
            outputs = torch.cat([outputs, out_features], -1)

        return outputs

    def forward(self, x, styles):
        #input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        mlp_out = x.contiguous()#input_pts
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i](mlp_out, styles)


        sdf = self.sigma_linear(mlp_out)

        #mlp_out = torch.cat([mlp_out, input_views], -1)
        out_features = self.views_linears(mlp_out, styles)
        rgb = self.rgb_linear(out_features)
        rgb = torch.sigmoid(rgb)
        # outputs = torch.cat([rgb, sdf], -1)
        # if self.output_features:
            # outputs = torch.cat([outputs, out_features], -1)

        return rgb, sdf
        
class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=1),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1, lr_multiplier=1)#3+ 16
        )
        
    def forward(self, sampled_features):
        # Aggregate features

        x = sampled_features.mean(1)

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        #rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        #rgb = torch.sigmoid(x[..., 1:])#torch.tanh(x[..., 1:])rgb, 

        sigma = x[..., 0:1]
        return sigma

class DensityNetwork(nn.Module):
    def __init__(self):
        super(DensityNetwork, self).__init__()

        #input_ch = 63
        input_ch_views = 27
        D = 3
        W = 128

        self.skips = [4]
        embed_fn, input_ch = embedder.get_embedder(6,
                                                       input_dims=3)
        self.embed_fn_fine = embed_fn
        
        layers = [nn.Linear(input_ch, W)]#+1+82+256
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = W
            if i in self.skips:
                in_channels += input_ch#+1
            layers += [layer(in_channels, W)]
        self.pts_linears = nn.ModuleList(layers)
        
               
        self.views_linears = nn.ModuleList([nn.Linear(W, W // 2)])#input channel needs to change,
        
        #self.feature_linear = nn.Linear(W, W)
        self.density_linear = nn.Linear(W // 2, 1)
        self.density_linear.bias.data.fill_(0.693)
        
        #self.latent_fc = nn.Linear(128, 128)#384
        # for m in self.modules():
            # if isinstance(m,nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu') 
        #self.latent = nn.Embedding(cfg.num_train_frame, 128)

    def forward(self, wpts):

        light_pts = self.embed_fn_fine(wpts)
        
        #input_h = h#light_pts, viewdir, sp_input
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
        
        #input_h = torch.cat([light_pts, smplpara[:,None].repeat(1,n_point,1)], -1)
        #input_h = torch.cat((light_pts, weights), dim=2)
        input_h = light_pts
        h = input_h
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            #print(self.pts_linears[i].weight.data)
            h = F.relu(h)
            
            if i in self.skips:
                h = torch.cat([input_h, h], -1)
             
        #features = self.feature_linear(h)

        # latent = self.latent(sp_input['latent_index'])
        # latent = latent[..., None].expand(*latent.shape, h.size(1))
        # latent = latent.transpose(-2,-1)
        # features = torch.cat((features, latent), dim=2)
        #features = self.latent_fc(features)

        #viewdir = embedder.view_embedder(viewdir)
        #viewdir = viewdir.transpose(1, 2)

        #h = torch.cat((features, viewdir), dim=2)
        #h = features
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)
        density = self.density_linear(h)

        return density
        
class ColorNetwork(nn.Module):
    def __init__(self):
        super(ColorNetwork, self).__init__()

        #input_ch = 63
        input_ch_views = 27
        D = 3
        W = 128

        self.skips = [4]
        embed_fn, input_ch = embedder.get_embedder(6,
                                                       input_dims=3)
        self.embed_fn_fine = embed_fn
        
        layers = [nn.Linear(input_ch, W)]#+1+82+256
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = W
            if i in self.skips:
                in_channels += input_ch#+1
            layers += [layer(in_channels, W)]
        self.pts_linears = nn.ModuleList(layers)
        
               
        self.views_linears = nn.ModuleList([nn.Linear(W, W // 2)])#input channel needs to change,
        
        #self.feature_linear = nn.Linear(W, W)
        self.rgb_linear = nn.Linear(W // 2, 3)
        
        #self.latent_fc = nn.Linear(128, 128)#384
        # for m in self.modules():
            # if isinstance(m,nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu') 
        #self.latent = nn.Embedding(cfg.num_train_frame, 128)

    def forward(self, wpts, smplpara):

        light_pts = self.embed_fn_fine(wpts)
        
        #input_h = h#light_pts, viewdir, sp_input
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
        
        #input_h = torch.cat([light_pts, smplpara[:,None].repeat(1,n_point,1)], -1)
        #input_h = torch.cat((light_pts, weights), dim=2)
        input_h = light_pts
        h = input_h
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            #print(self.pts_linears[i].weight.data)
            h = F.relu(h)
            
            if i in self.skips:
                h = torch.cat([input_h, h], -1)
             
        #features = self.feature_linear(h)

        # latent = self.latent(sp_input['latent_index'])
        # latent = latent[..., None].expand(*latent.shape, h.size(1))
        # latent = latent.transpose(-2,-1)
        # features = torch.cat((features, latent), dim=2)
        #features = self.latent_fc(features)

        #viewdir = embedder.view_embedder(viewdir)
        #viewdir = viewdir.transpose(1, 2)

        #h = torch.cat((features, viewdir), dim=2)
        #h = features
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)
        rgb = self.rgb_linear(h)
        rgb = torch.sigmoid(rgb)

        return rgb
        
class OSGDecoder_fixgeometry_color(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden_dim = 64
        embed_fn, self.input_ch = embedder.get_embedder(6,
                                                       input_dims=3)
        self.embed_fn_fine = embed_fn
            
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(self.input_ch, self.hidden_dim, lr_multiplier=1),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 16, lr_multiplier=1)#3+ 16
        )
        
    def forward0(self, sampled_features):
        # Aggregate features

        x = sampled_features.mean(1)

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        #rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        #rgb = torch.sigmoid(x[..., 1:])#torch.tanh(x[..., 1:])rgb, 
        rgb = torch.sigmoid(x)

        #sigma = x[..., 0:1]
        return rgb
    
    def forward(self, inputs):
        # Aggregate features,sampled_features
        x = self.embed_fn_fine(inputs)
        print(torch.isnan(x).any() or torch.isinf(x).any())
        # x = sampled_features.mean(1)
        #x = sampled_features
        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        print(torch.isnan(x).any() or torch.isinf(x).any())
        x = x.view(N, M, -1)
        #rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        #rgb = torch.sigmoid(x[..., 1:])#torch.tanh(x[..., 1:])rgb, 
        rgb = torch.sigmoid(x)
        print(torch.isnan(rgb).any() or torch.isinf(rgb).any())
        print('1')
        #sigma = x[..., 0:1]
        return rgb
        
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

class SDFNetwork(nn.Module):
    def __init__(self, latentdim):#
        super(SDFNetwork, self).__init__()

        self.sdf_network = SDFNetwork_(latentdim)#

    def forward(self, wpts, latent):#
        # calculate sdf
        sdf_nn_output = self.sdf_network(wpts, latent)#
        sdf = sdf_nn_output[..., 0]
        features = sdf_nn_output[..., 1:65]
        return sdf, features

class SDFNetwork_(nn.Module):
    def __init__(self, latentdim):#
        super(SDFNetwork_, self).__init__()

        d_in = 3
        d_out = 257
        d_hidden = 256
        n_layers = 2

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        multires = 6
        if multires > 0:
            embed_fn, input_ch = embedder.get_embedder(multires,
                                                       input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch+latentdim

        skip_in = [4]
        bias = 0.5
        scale = 1
        geometric_init = True
        weight_norm = True
        activation = 'softplus'

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs, latent):#
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        latent = latent[:,None].repeat(1,inputs.shape[1],1)
        x = torch.cat([inputs, latent], dim=-1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)#1

            x = lin(x)
        
            if l < self.num_layers - 2:
                x = self.activation(x)
      
        return torch.cat([x[..., :1] / self.scale, x[..., 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[..., :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)

class incSDFNetwork(nn.Module):
    def __init__(self):
        super(incSDFNetwork, self).__init__()

        self.sdf_network = incSDFNetwork_()

    def forward(self, wpts):
        # calculate sdf
        sdf_nn_output = self.sdf_network(wpts)
        sdf = sdf_nn_output
        return sdf

class incSDFNetwork_(nn.Module):
    def __init__(self):
        super(incSDFNetwork_, self).__init__()

        d_in = 3
        d_out = 1
        d_hidden = 256
        n_layers = 2

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        multires = 6
        if multires > 0:
            embed_fn, input_ch = embedder.get_embedder(multires,
                                                       input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        skip_in = [4]
        bias = 0.5
        scale = 1
        geometric_init = True
        weight_norm = True
        activation = 'softplus'

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)        
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)#1

            x = lin(x)
        
            if l < self.num_layers - 2:
                x = self.activation(x)
      
        return x[...,0] / self.scale

    def sdf(self, x):
        return self.forward(x)[..., :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)
          
class DisplaceNetwork(nn.Module):
    def __init__(self):
        super(DisplaceNetwork, self).__init__()

        self.actvn = nn.ReLU()
        
        outdim = 3
        self.skips = [4]
        D = 3
        W = 128
        input_ch = 63
        input_ch_views = 27
        layers = [nn.Linear(145, W)]#input_ch
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = W
            if i in self.skips:
                in_channels += 128#input_ch
            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)
        #self.feature_linear = nn.Linear(W+128, W)
             
        self.disp_linear = nn.Linear(W, outdim)
        self.disp_linear.bias.data.fill_(0.0)
        
        # for i, l in enumerate(self.pts_linears):
            # torch.nn.init.constant_(self.pts_linears[i].weight, 0)
            # torch.nn.init.constant_(self.pts_linears[i].bias, 0)
        # torch.nn.init.constant_(self.feature_linear.weight, 0)
        # torch.nn.init.constant_(self.feature_linear.bias, 0)
        
        # torch.nn.init.constant_(self.disp_linear.weight, 0)
        # torch.nn.init.constant_(self.disp_linear.bias, 0)
        
        #self.latent = nn.Embedding(cfg.num_train_frame, 128)
       

    def forward(self, wpts, smplpara):
        light_pts = embedder.xyz_embedder(wpts)
        ptnum = wpts.shape[1]
        #latent = self.latent(sp_input['latent_index'])
        # latent = latent[..., None].expand(*latent.shape, wpts.size(1))
        # latent = latent.transpose(-2,-1)latent[None].repeat(ptnum,1),latent,
        #h = latent
        h = torch.cat([light_pts, smplpara[:,None].repeat(1,ptnum,1)], -1)
        #h = light_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([latent, h], -1)#light_pts
    
        # latent = self.latent(sp_input['latent_index'])
        # latent = latent[..., None].expand(*latent.shape, h.size(1))
        # latent = latent.transpose(-2,-1)
        # h = torch.cat((h, latent), dim=2)

        #h = self.feature_linear(h)

        disp = self.disp_linear(h)

        return disp

class IncSDFNetwork(nn.Module):
    def __init__(self):
        super(IncSDFNetwork, self).__init__()

        self.actvn = nn.ReLU()
        
        outdim = 1
        self.skips = [4]
        D = 3
        W = 128
        input_ch = 63
        input_ch_views = 27
        layers = [nn.Linear(63, W)]#input_ch
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = W
            if i in self.skips:
                in_channels += 128#input_ch
            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)
        #self.feature_linear = nn.Linear(W+128, W)
             
        self.disp_linear = nn.Linear(W, outdim)
        self.disp_linear.bias.data.fill_(0.0)
        
        for i, l in enumerate(self.pts_linears):
            torch.nn.init.constant_(self.pts_linears[i].weight, 0)
            torch.nn.init.constant_(self.pts_linears[i].bias, 0)
        # torch.nn.init.constant_(self.feature_linear.weight, 0)
        # torch.nn.init.constant_(self.feature_linear.bias, 0)
        
        torch.nn.init.constant_(self.disp_linear.weight, 0)
        torch.nn.init.constant_(self.disp_linear.bias, 0)
        
        #self.latent = nn.Embedding(cfg.num_train_frame, 128)
       

    def forward(self, wpts):
        light_pts = embedder.xyz_embedder(wpts)
        ptnum = wpts.shape[1]
        #latent = self.latent(sp_input['latent_index'])
        # latent = latent[..., None].expand(*latent.shape, wpts.size(1))
        # latent = latent.transpose(-2,-1)latent[None].repeat(ptnum,1),
        #h = latent
        #h = torch.cat([light_pts, smplpara[:,None].repeat(1,ptnum,1)], -1)
        h = light_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([latent, h], -1)#light_pts
    
        # latent = self.latent(sp_input['latent_index'])
        # latent = latent[..., None].expand(*latent.shape, h.size(1))
        # latent = latent.transpose(-2,-1)
        # h = torch.cat((h, latent), dim=2)

        #h = self.feature_linear(h)

        disp = self.disp_linear(h)

        return disp[...,0]
        
class MLPNARF(NARFBase):
    def __init__(self, config, z_dim: Union[int, List[int]] = 256, num_bone=1,
                 bone_length=False, parent=None, num_bone_param=None, view_dependent: bool = True):
        assert config.origin_location in ["center", "center_fixed"]
        self.tri_plane_based = False
        super(MLPNARF, self).__init__(config, z_dim, num_bone, bone_length, parent, num_bone_param, view_dependent)
        self.initialize_network()

    def initialize_network(self):
        hidden_size = self.hidden_size

        # selector
        hidden_dim_for_mask = 10
        self.selector = nn.Sequential(nn.Conv1d(3 * self.num_frequency_for_position * 2 * self.num_bone,
                                                hidden_dim_for_mask * self.num_bone, 1, groups=self.num_bone),
                                      nn.ReLU(inplace=True),
                                      nn.Conv1d(hidden_dim_for_mask * self.num_bone, self.num_bone, 1,
                                                groups=self.num_bone),
                                      nn.Softmax(dim=1))

        if self.config.model_type == "dnarf":
            self.deformation_field = MLP((self.num_bone * 3 + 1) * self.num_frequency_for_position * 2, hidden_size,
                                         self.num_bone * 3, num_layers=8, skips=(4,))
            self.density_mlp = MLP(self.num_bone * 3 * self.num_frequency_for_position * 2, hidden_size, hidden_size,
                                   num_layers=8, skips=(4,))
        elif self.config.model_type == "tnarf":
            self.density_mlp = StyledMLP(self.num_bone * 3 * self.num_frequency_for_position * 2, hidden_size,
                                         hidden_size, style_dim=self.z_dim, num_layers=8)
        elif self.config.model_type == "narf":
            self.density_mlp = MLP(self.num_bone * 3 * self.num_frequency_for_position * 2, hidden_size,
                                   hidden_size, num_layers=8, skips=(4,))

        self.density_fc = StyledConv1d(self.hidden_size, 1, self.z2_dim)
        if self.view_dependent:
            self.mlp = StyledMLP(self.hidden_size + 3 * self.num_frequency_for_other * 2, self.hidden_size // 2,
                                 3, style_dim=self.z2_dim)
        else:
            self.mlp = StyledMLP(self.hidden_size, self.hidden_size // 2, 3, style_dim=self.z2_dim)

    def calc_density_and_color_from_camera_coord_v2(self, position: torch.Tensor, pose_to_camera: torch.Tensor,
                                                    ray_direction: torch.Tensor, model_input: Dict = {}):
        """compute density from positions in camera coordinate

        :param position:
        :param pose_to_camera:
        :param bone_length:
        :param z:
        :param z_rend:
        :return: density of input positions
        """
        bone_length, z, z_rend = model_input["bone_length"], model_input["z"], model_input["z_rend"]

        local_points = to_local(position, pose_to_camera)

        in_cube_p = in_cube(local_points)  # (B, n_bone, n)
        density, color = self.backbone(local_points, in_cube_p, z, z_rend, bone_length, ray_direction)
        density *= in_cube_p.any(dim=1, keepdim=True)
        return density, color

    def backbone(self, p: torch.Tensor, position_validity: torch.Tensor, z: torch.Tensor,
                 z_rend: torch.Tensor, bone_length: torch.Tensor,
                 ray_direction: Optional[torch.Tensor] = None):
        """

        Args:
            p: position in local coordinate, (B, n_bone, 3, n)
            position_validity: bool tensor for validity of p, (B, n_bone, n)
            z: (B, dim)
            z_rend: (B, dim)
            bone_length: (B, n_bone)
            # mode: "weight_feature" or "weight_position"
            ray_direction: not None if color is view dependent
        Returns:

        """
        # don't support mip-nerf rendering
        assert isinstance(p, torch.Tensor)
        assert bone_length is not None
        # assert mode in ["weight_position", "weight_feature"]
        encoded_p = multi_part_positional_encoding(p, self.num_frequency_for_position, self.num_bone)
        prob = self.selector(encoded_p)

        encoded_p = encoded_p * torch.repeat_interleave(prob, 3 * self.num_frequency_for_position * 2, dim=1)

        if self.config.model_type == "dnarf":
            expand_z = z[:, :, None].expand(-1, -1, p.shape[-1])
            dp = self.deformation_field(torch.cat([encoded_p, expand_z], dim=1))  # (B, num_bone * 3, n)
            p = p + dp
            encoded_p = multi_part_positional_encoding(p, self.num_frequency_for_position, self.num_bone)

        if self.config.model_type == "tnarf":
            feature = self.density_mlp(encoded_p, z)
        else:
            feature = self.density_mlp(encoded_p)

        density, color = calc_density_and_color_from_feature(self, feature, z_rend, ray_direction)
        return density, color
