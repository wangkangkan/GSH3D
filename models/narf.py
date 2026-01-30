from typing import Union, List, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from libraries.NARF.base import NARFBase
from libraries.NeRF.nerf import calc_density_and_color_from_feature
from libraries.NeRF.net import StyledMLP, MLP
from libraries.NeRF.utils import StyledConv1d, multi_part_positional_encoding, in_cube, to_local
from libraries.custom_stylegan2.net import EqualConv1d
from libraries.triplane.sampling import sample_feature, sample_triplane_part_prob, sample_weighted_feature_v2
from libraries.triplane.triplane_nerf import prepare_triplane_generator, calc_density_and_color_from_feature

from libraries.superresolution import Superresolution_freesize as Superresolution

from . import embedder
from warping_utils import surface_field, smpl_helper
from smplx.utils import SMPLOutput
from libraries.stylegan2_ada_pytorch.training.networks import FullyConnectedLayer
import trimesh
from warping_utils import math_utils
from smplx.body_models import SMPL
from libraries.smpl_utils import get_shape

from warping_utils.ray_marcher import MipRayMarcher2
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def set_pytorch3d_intrinsic_matrix(batchK, H, W):
        K = batchK[0]
        fx = -K[0, 0] * 2.0 / W
        fy = -K[1, 1] * 2.0 / H
        px = -(K[0, 2] - W / 2.0) * 2.0 / W
        py = -(K[1, 2] - H / 2.0) * 2.0 / H
        pytorch3d_K = torch.Tensor([
            [fx, 0, px, 0],
            [0, fy, py, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]).to(K.device)
        return pytorch3d_K[None,...].repeat(batchK.shape[0],1,1)
        
class TriPlaneNARF(NARFBase):
    def __init__(self, config, z_dim: Union[int, List[int]] = 256, num_bone=1,
                 bone_length=True, parent=None, num_bone_param=None, view_dependent: bool = False):
        assert bone_length
        self.tri_plane_based = True
        self.w_dim = 512#512
        self.feat_dim = 32#32
        self.no_selector = config.no_selector
        super(TriPlaneNARF, self).__init__(config, z_dim, num_bone, bone_length, parent, num_bone_param, view_dependent)
        self.initialize_network()
        
        self.decoder = OSGDecoder(32)
        self.plane_axes = generate_planes()
        self.plane_axes = self.plane_axes.to('cuda')
        smpl_base = smpl_helper.load_smpl_model(smpl_helper.get_smpl_data_path('m'))
        self.smpl_reduced = smpl_helper.SMPLSimplified.build_from_template(smpl_base, growth_offset=0.0)
        self.surface_field = surface_field.SurfaceField(self.smpl_reduced)
        
        #self.displace = DisplaceNetwork()
        self.superresolution = Superresolution(32, 180, sr_num_fp16_res=4, sr_antialias=True)   
        
        self._register_avg_smpl()
        
        self.ray_marcher = MipRayMarcher2()
        SMPL_MODEL_PATH = "./smpl_data"
        self.smpl = SMPL(model_path=SMPL_MODEL_PATH, gender='MALE', batch_size=1)
        #self.surface_field = surface_field.SurfaceField(self.smpl)
        
        # smpl_reduced_canon_vert = get_shape(self.smpl, body_pose=self.smpl_avg_body_pose.expand(3, -1).view(3,23,3),
                                      # global_orient=self.smpl_avg_orient.expand(3, -1)[:,None])
        # smpl_reduced_canon_vert *= self.smpl_avg_scale
        # #smpl_reduced_canon_vert += self.smpl_avg_transl.expand(3, -1)[:,None]                                      
        # self.smpl_reduced_canon = SMPLOutput(vertices=smpl_reduced_canon_vert.to('cuda'))
        
        npfaces = self.smpl.faces
        npfaces = npfaces.astype(np.int32)
        self.smplfaces = torch.LongTensor(npfaces).to('cuda')

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
            self.z_dim, self.w_dim, in_channels,0)#without pose condition    

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
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(projection_matrix, 320, 320)

        cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K,
                                         R=mat_R,
                                         T=mat_T)
                                         
        rasterizer=pytorch3d.renderer.MeshRasterizer(
                cameras=cameras, 
                raster_settings=pytorch3d.renderer.RasterizationSettings(
                    image_size=(320, 320),
                    blur_radius=0.0, 
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
        return rendermask[:,:240,:]        
    
    @torch.no_grad()
    def inversedeforming_samplepoints_LBS(self, wpts, nnvidx, templatevert, T):
        #samplepts: sampled points in the world space
        #wpts: sampled points after invtransorm_surreal which applied inv LBS
              
        #inversely deforming sample points to cannonical frame
        ptsnum = wpts.size(1)
        
        templatevtnum = templatevert.size(1)

        #world points to posed points
        pts = wpts
        #pts = torch.matmul(wpts - sp_input['Th'], sp_input['R'])
        #transform points from the pose space to the T pose

        #bw = self.bw[nnvidx.view([-1]),:].view(-1,ptsum,24) #bw: B, 24, n_points
        #bw = self.bw[None,...]#B, n_points, 24.permute(0, 2, 1)
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(wpts.size(0))]
        idx = torch.cat(idx).to(wpts.device)
        tidx = nnvidx.view(-1) + idx
        T1 = T.view(-1,4,4)
        
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

    @torch.no_grad()
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
            coordinates_out[~mask] = coordinates[~mask] + 10

        return coordinates_out

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=1),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + 32, lr_multiplier=1)#3
        )
        
    def forward(self, sampled_features):
        # Aggregate features

        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        #rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        rgb = torch.sigmoid(x[..., 1:])#torch.tanh(x[..., 1:])

        sigma = x[..., 0:1]
        return rgb, sigma

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
        layers = [nn.Linear(657, W)]#input_ch
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
       

    def forward(self, wpts, latent, smplpara):
        light_pts = embedder.xyz_embedder(wpts)
        ptnum = wpts.shape[0]
        #latent = self.latent(sp_input['latent_index'])
        # latent = latent[..., None].expand(*latent.shape, wpts.size(1))
        # latent = latent.transpose(-2,-1)
        #h = latent
        h = torch.cat([light_pts, latent[None].repeat(ptnum,1), smplpara[None].repeat(ptnum,1)], -1)
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
