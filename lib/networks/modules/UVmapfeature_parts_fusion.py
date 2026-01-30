#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os

EPS = 1e-3

import sys
sys.path.append("libraries/stylegan2_ada_pytorch")
import dnnlib

def prepare_triplane_generator(z_dim, w_dim, out_channels, c_dim=0):
    G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=z_dim, w_dim=w_dim,
                               mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict(use_noise=False))
    G_kwargs.synthesis_kwargs.channel_base = 32768
    G_kwargs.synthesis_kwargs.channel_max = 512
    G_kwargs.mapping_kwargs.num_layers = 8
    G_kwargs.synthesis_kwargs.num_fp16_res = 0
    G_kwargs.synthesis_kwargs.conv_clamp = None

    g_common_kwargs = dict(c_dim=c_dim,
                           img_resolution=256, img_channels=out_channels)
    gen = dnnlib.util.construct_class_by_name(**G_kwargs, **g_common_kwargs)
    return gen

def prepare_triplane_generator_fusion(z_dim, w_dim, out_channels, c_dim=0):
    G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator_multiparts', z_dim=z_dim, w_dim=w_dim,
                               mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict(use_noise=False))
    G_kwargs.synthesis_kwargs.channel_base = 32768
    G_kwargs.synthesis_kwargs.channel_max = 512
    G_kwargs.mapping_kwargs.num_layers = 8
    G_kwargs.synthesis_kwargs.num_fp16_res = 0
    G_kwargs.synthesis_kwargs.conv_clamp = None

    g_common_kwargs = dict(c_dim=c_dim,
                           img_resolution=256, img_channels=out_channels)
    gen = dnnlib.util.construct_class_by_name(**G_kwargs, **g_common_kwargs)
    return gen
    
class TriPlane(nn.Module):
    def __init__(self, features=32, resX=256, resY=256, resZ=256):
        super().__init__()
        self.plane_xy = nn.Parameter(torch.randn(1, features, resX, resY))
        self.plane_xz = nn.Parameter(torch.randn(1, features, resX, resZ))
        self.plane_yz = nn.Parameter(torch.randn(1, features, resY, resZ))
        self.dim = features
        self.n_input_dims = 3
        self.n_output_dims = 3 * features
        self.center = 0.0
        self.scale = 5.0

    def forward(self, x):
        x = (x - self.center) / self.scale + 0.6

        assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 3)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        feat_xy = F.grid_sample(self.plane_xy, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_xz = F.grid_sample(self.plane_xz, coords[..., [0, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_yz = F.grid_sample(self.plane_yz, coords[..., [1, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
        feat = feat.reshape(*shape[:-1], 3 * self.dim)
        return feat
        
class TriPlane_varying(nn.Module):
    def __init__(self, features=32, resX=256, resY=256, resZ=256):
        super().__init__()
        self.plane_xy = nn.Parameter(torch.randn(1, features, resX, resY))
        self.plane_xz = nn.Parameter(torch.randn(1, features, resX, resZ))
        self.plane_yz = nn.Parameter(torch.randn(1, features, resY, resZ))
        self.dim = features#+3
        self.n_input_dims = 3
        self.n_output_dims = 3 * features
        self.center = 0.0
        self.scale = 5.0
              
        self.tri_plane_gen = prepare_triplane_generator(128, 512, self.dim*3,0)#

    def triplanefeature(self, styles):

        planes, latentws = self.tri_plane_gen(styles,0,truncation_psi=1)
        
        planes = planes.view(len(planes), 3, self.dim, planes.shape[-2], planes.shape[-1])
        
        self.plane_xy_d = planes[:,0] 
        self.plane_xz_d = planes[:,1]
        self.plane_yz_d = planes[:,2]
    
    def fetch_disp(self, x):
        x = (x - self.center) / self.scale + 0.6

        assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 3)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        feat_z = F.grid_sample(self.plane_xy_d, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_y = F.grid_sample(self.plane_xz_d, coords[..., [0, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_x = F.grid_sample(self.plane_yz_d, coords[..., [1, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)

        feat = torch.cat([feat_x, feat_y, feat_z], dim=1)
        feat = feat.reshape(*shape[:-1], 3 * self.dim)#
        return feat
        
    def forward(self, x):
        x = (x - self.center) / self.scale + 0.6

        assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 3)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        feat_xy = F.grid_sample(self.plane_xy, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_xz = F.grid_sample(self.plane_xz, coords[..., [0, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_yz = F.grid_sample(self.plane_yz, coords[..., [1, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)

        feat = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
        feat = feat.reshape(*shape[:-1], 3 * self.dim)
        return feat
        # feat = torch.cat([feat_xy[:,3:], feat_xz[:,3:], feat_yz[:,3:]], dim=1)
        # feat = feat.reshape(*shape[:-1], 3 * (self.dim-3))
        # offset = feat_xy[:,:3] + feat_xz[:,:3] + feat_yz[:,:3]
        # return offset, feat     

class UVmap_GS(nn.Module):
    def __init__(self, features=32, resX=256, resY=256, resZ=256):
        super().__init__()
        
        self.dim = 13#37#8#features#+3
        self.n_input_dims = 3
        self.n_output_dims = 3 * features
        self.center = 0.0
        self.scale = 5.0
        
        data_root = '../gaussian-nerfcap/data/magdalena/magdalena2000-allviews'        
        npfaces = np.loadtxt(os.path.join(data_root, 'templatedeformT/smpl/smpltri.txt')) - 1
        self.smplfaces = torch.LongTensor(npfaces).to('cuda')
        facesegidx = np.loadtxt('lib/networks/modules/facesegidx.txt')
        self.facesegidx = torch.Tensor(facesegidx).to('cuda')
        faces_len = self.smplfaces.shape[0]
        gssegidx =  [torch.full([6], self.facesegidx[i], dtype=torch.long) for i in range(faces_len)]
        self.gssegidx = torch.cat(gssegidx).to('cuda')
        self.gsnum = self.gssegidx.shape[0]
        
        for seg in range(0,3):
            #feat_plane_gen = prepare_triplane_generator(512, 512, self.dim,0)#128
            #setattr(self, f'featgen{seg}', feat_plane_gen)

            poslinear = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)        
            torch.nn.init.constant(poslinear.weight, 0)
            torch.nn.init.constant(poslinear.bias, 0)
            setattr(self, f'poslinear{seg}', poslinear)
            
            rotlinear = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1) 
            torch.nn.init.constant(rotlinear.weight, 0)
            torch.nn.init.constant(rotlinear.bias, 0)
            setattr(self, f'rotlinear{seg}', rotlinear)
            
            scalelinear = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1) 
            torch.nn.init.constant(scalelinear.weight, 0)
            torch.nn.init.constant(scalelinear.bias, -4.9) #-4.7-5.1
            setattr(self, f'scalelinear{seg}', scalelinear)
        
        # self.bwlinear = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=0) 
        # torch.nn.init.constant(self.bwlinear.weight, 0)
        # torch.nn.init.constant(self.bwlinear.bias, 0) 
        
        self.tri_plane_gen = prepare_triplane_generator_fusion(512, 512, self.dim,0)#128

        self.poslinear = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0)        
        torch.nn.init.constant(self.poslinear.weight, 0)
        torch.nn.init.constant(self.poslinear.bias, 0)

        self.rotlinear = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0) 
        torch.nn.init.constant(self.rotlinear.weight, 0)
        torch.nn.init.constant(self.rotlinear.bias, 0)
        
        self.scalelinear = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=0) 
        torch.nn.init.constant(self.scalelinear.weight, 0)
        torch.nn.init.constant(self.scalelinear.bias, -4.9) #-4.7-5.1
        
        self.sigm_activation = nn.Sigmoid()

    def uvgsfeature(self, styles):
            
        planes_fusion, planes_parts = self.tri_plane_gen(styles,0,truncation_psi=1)
        
        planes_fusion = planes_fusion.view(len(planes_fusion), self.dim, planes_fusion.shape[-2], planes_fusion.shape[-1])
        
        for seg in range(0,3):
            #feat_plane_gen = getattr(self, f'featgen{seg}')
            #planes, latentws = feat_plane_gen(styles,0,truncation_psi=1)
            planes = planes_parts[seg]
            planes = planes.view(len(planes), self.dim, planes.shape[-2], planes.shape[-1])
            
            plane_pos = planes[:,0:3] 
            plane_scale = planes[:,3:5]
            plane_rot = planes[:,5:8]
            plane_color = planes[:,8:11]
            plane_opacity = planes[:,12:13]
            #self.plane_bw = planes[:,13:37]
            
            scalelinear = getattr(self, f'scalelinear{seg}')
            plane_scale1 = scalelinear(plane_scale) 
            setattr(self, f'planescale1{seg}', plane_scale1)
            plane_scale = torch.clamp(torch.exp(plane_scale1), max=0.03)#
            setattr(self, f'planescale{seg}', plane_scale)
            #self.plane_scale = torch.sigmoid(self.plane_scale)*0.03
            #self.plane_scale = torch.clamp(self.scalelinear(self.plane_scale), max=0.02, min=0.0003)
            plane_opacity  = torch.sigmoid(plane_opacity)
            setattr(self, f'planeopacity{seg}', plane_opacity)
            
            plane_color = torch.sigmoid(plane_color)*(1 + 2*0.001) - 0.001 # follow mip-nerf
            plane_color = plane_color*255
            setattr(self, f'planecolor{seg}', plane_color)
            
            poslinear = getattr(self, f'poslinear{seg}')   
            plane_pos = poslinear(plane_pos) 
            setattr(self, f'planepos{seg}', plane_pos)
            
            #self.plane_bw = self.bwlinear(self.plane_bw) 
            rotlinear = getattr(self, f'rotlinear{seg}')   
            plane_rot = rotlinear(plane_rot) 
            setattr(self, f'planerot{seg}', plane_rot)
            
            #self.plane_rot=torch.nn.functional.normalize(self.plane_rot)
        
        self.plane_pos = planes_fusion[:,0:3] 
        self.plane_scale = planes_fusion[:,3:5]
        self.plane_rot = planes_fusion[:,5:8]
        self.plane_color = planes_fusion[:,8:11]
        self.plane_opacity = planes_fusion[:,12:13]
        #self.plane_bw = planes[:,12:26]
        
        self.plane_scale1 = self.scalelinear(self.plane_scale) 
        self.plane_scale = torch.clamp(torch.exp(self.plane_scale1), max=0.03)#
        #self.plane_scale = torch.sigmoid(self.plane_scale)*0.03
        #self.plane_scale = torch.clamp(self.scalelinear(self.plane_scale), max=0.02, min=0.0003)
        self.plane_opacity  = torch.sigmoid(self.plane_opacity)
        self.plane_color = torch.sigmoid(self.plane_color)*(1 + 2*0.001) - 0.001 # follow mip-nerf
        self.plane_color = self.plane_color*255
           
        self.plane_pos = self.poslinear(self.plane_pos) 
        
        #self.plane_bw = self.bwlinear(self.plane_bw) 
        
        self.plane_rot = self.rotlinear(self.plane_rot) 
    
    def fetch_disp(self, x):
        x = (x - self.center) / self.scale + 0.6

        assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 3)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        feat_z = F.grid_sample(self.plane_xy_d, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_y = F.grid_sample(self.plane_xz_d, coords[..., [0, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_x = F.grid_sample(self.plane_yz_d, coords[..., [1, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)

        feat = torch.cat([feat_x, feat_y, feat_z], dim=1)
        feat = feat.reshape(*shape[:-1], 3 * self.dim)#
        return feat
    
    def getspecattr(self, attri_name, seg):
        return getattr(self, f'{attri_name}{seg}')
        
    def forward_batch(self, x):
        #x = (x - self.center) / self.scale + 0.6
        plane_scale = getattr(self, f'planescale{0}')
        batchsize = plane_scale.shape[0]
        
        # segidx = self.gssegidx==0
        # uvseg = x[segidx]
        # np.savetxt('uvseg0.txt', uvseg.detach().cpu().numpy())
        # segidx = self.gssegidx==1
        # uvseg = x[segidx]
        # np.savetxt('uvseg1.txt', uvseg.detach().cpu().numpy())
        # segidx = self.gssegidx==2
        # uvseg = x[segidx]
        # np.savetxt('uvseg2.txt', uvseg.detach().cpu().numpy())
        
        assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 2).repeat(batchsize,1,1,1)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        
        #directly combine
        gs_pos = torch.zeros(batchsize,self.gsnum,3).to('cuda')
        gs_scale = torch.zeros(batchsize,self.gsnum,2).to('cuda')
        gs_rot = torch.zeros(batchsize,self.gsnum,3).to('cuda')
        gs_color = torch.zeros(batchsize,self.gsnum,3).to('cuda')
        gs_opacity = torch.zeros(batchsize,self.gsnum,1).to('cuda')
        for seg in range(0,3):
            plane_pos = getattr(self, f'planepos{seg}')
            plane_scale = getattr(self, f'planescale{seg}')
            plane_rot = getattr(self, f'planerot{seg}')            
            plane_color = getattr(self, f'planecolor{seg}')
            plane_opacity = getattr(self, f'planeopacity{seg}')
            segidx = self.gssegidx==seg
            gs_pos[:,segidx,:] = F.grid_sample(plane_pos, coords[:,segidx,:,:], align_corners=True)[..., 0].transpose(1, 2)
            gs_scale[:,segidx,:]  = F.grid_sample(plane_scale, coords[:,segidx,:,:], align_corners=True)[..., 0].transpose(1, 2)
            gs_rot[:,segidx,:]  = F.grid_sample(plane_rot, coords[:,segidx,:,:], align_corners=True)[..., 0].transpose(1, 2)
            gs_color[:,segidx,:]  = F.grid_sample(plane_color, coords[:,segidx,:,:], align_corners=True)[..., 0].transpose(1, 2)
            gs_opacity[:,segidx,:]  = F.grid_sample(plane_opacity, coords[:,segidx,:,:], align_corners=True)[..., 0].transpose(1, 2)
            #gs_bw = F.grid_sample(self.plane_bw, coords[..., [0, 1]], align_corners=True)[..., 0].transpose(1, 2)
        
        #fuse to a single map
        gs_pos_fusion = F.grid_sample(self.plane_pos, coords[..., [0, 1]], align_corners=True)[..., 0].transpose(1, 2)
        gs_scale_fusion = F.grid_sample(self.plane_scale, coords[..., [0, 1]], align_corners=True)[..., 0].transpose(1, 2)
        gs_rot_fusion = F.grid_sample(self.plane_rot, coords[..., [0, 1]], align_corners=True)[..., 0].transpose(1, 2)
        gs_color_fusion = F.grid_sample(self.plane_color, coords[..., [0, 1]], align_corners=True)[..., 0].transpose(1, 2)
        gs_opacity_fusion = F.grid_sample(self.plane_opacity, coords[..., [0, 1]], align_corners=True)[..., 0].transpose(1, 2)
        
        # gs_pos = self.poslinear(gs_pos) 
        # gs_rot = self.rotlinear(gs_rot) 
        
        #gs_pos = self.poslinear(gs_pos) 
        #gs_pos = 0.2*torch.tanh(gs_pos)#vert offset
        
        #gs_scale = torch.sigmoid(gs_scale)/30
        #gs_opacity = self.sigm_activation(gs_opacity)
        # gs_pos_batch = []
        # gs_rot_batch = []
        # gs_scale_batch = []
        # gs_color_batch = []
        # gs_opacity_batch = []
        # for bidx in range(0, batchsize):
            # gs_pos = F.grid_sample(self.plane_pos[bidx:bidx+1], coords[..., [0, 1]], align_corners=True)[..., 0].transpose(1, 2)
            # gs_scale = F.grid_sample(self.plane_scale[bidx:bidx+1], coords[..., [0, 1]], align_corners=True)[..., 0].transpose(1, 2)
            # gs_rot = F.grid_sample(self.plane_rot[bidx:bidx+1], coords[..., [0, 1]], align_corners=True)[..., 0].transpose(1, 2)
            # gs_color = F.grid_sample(self.plane_color[bidx:bidx+1], coords[..., [0, 1]], align_corners=True)[..., 0].transpose(1, 2)
            # gs_opacity = F.grid_sample(self.plane_opacity[bidx:bidx+1], coords[..., [0, 1]], align_corners=True)[..., 0].transpose(1, 2)

            
            # gs_pos = self.poslinear(gs_pos) 
            # gs_rot = self.rotlinear(gs_rot) 
            
            # gs_pos_batch += [gs_pos]
            # gs_rot_batch += [gs_rot]
            # gs_scale_batch += [gs_scale]
            # gs_color_batch += [gs_color]
            # gs_opacity_batch += [gs_opacity]
        # gs_pos = torch.cat(gs_pos_batch, 0)
        # gs_rot = torch.cat(gs_rot_batch, 0)
        # gs_scale = torch.cat(gs_scale_batch, 0)
        # gs_color = torch.cat(gs_color_batch, 0)
        # gs_opacity = torch.cat(gs_opacity_batch, 0)
        
        return gs_pos,gs_rot,gs_scale, gs_color, gs_opacity, gs_pos_fusion,gs_rot_fusion,gs_scale_fusion, gs_color_fusion, gs_opacity_fusion#, gs_bw   
        
    def forward(self, x):
        #x = (x - self.center) / self.scale + 0.6

        assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        #x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 2)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        gs_pos = F.grid_sample(self.plane_pos, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        gs_scale = F.grid_sample(self.plane_scale, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        gs_rot = F.grid_sample(self.plane_rot, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        gs_color = F.grid_sample(self.plane_color, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        gs_opacity = F.grid_sample(self.plane_opacity, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)

        
        gs_pos = self.poslinear(gs_pos) 
        gs_rot = self.rotlinear(gs_rot) 
        
        #gs_pos = self.poslinear(gs_pos) 
        #gs_pos = 0.02*torch.tanh(gs_pos)#vert offset
        
        #gs_scale = torch.sigmoid(gs_scale)/30
        #gs_opacity = self.sigm_activation(gs_opacity)
        
        return gs_pos,gs_rot,gs_scale, gs_color, gs_opacity     
        
class UVmap_vert(nn.Module):
    def __init__(self, features=32, resX=256, resY=256, resZ=256):
        super().__init__()
        
        self.dim = 3#8#features#+3
        self.n_input_dims = 3
        self.n_output_dims = 3 * features
        self.center = 0.0
        self.scale = 5.0
              
        self.tri_plane_gen = prepare_triplane_generator(128, 512, self.dim,0)#
        
        #self.poslinear  = nn.Linear(3, 3)
        self.poslinear = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0)        
        torch.nn.init.constant(self.poslinear.weight, 0)
        torch.nn.init.constant(self.poslinear.bias, 0)
                   
        self.sigm_activation = nn.Sigmoid()

    def uvvertfeature(self, styles):

        planes, latentws = self.tri_plane_gen(styles,0,truncation_psi=1)
        
        self.plane_pos = planes.view(len(planes), self.dim, planes.shape[-2], planes.shape[-1])
                   
        self.plane_pos = self.poslinear(self.plane_pos)        
    
     
    def forward_batch(self, x):
        #x = (x - self.center) / self.scale + 0.6
        batchsize = self.plane_pos.shape[0]
        assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 2).repeat(batchsize,1,1,1)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        vert_incpos = F.grid_sample(self.plane_pos, coords[..., [0, 1]], align_corners=True)[..., 0].transpose(1, 2)
       
        return vert_incpos
        
    def forward(self, x):
        #x = (x - self.center) / self.scale + 0.6

        assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        #x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 2)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        gs_pos = F.grid_sample(self.plane_pos, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        gs_scale = F.grid_sample(self.plane_scale, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        gs_rot = F.grid_sample(self.plane_rot, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        gs_color = F.grid_sample(self.plane_color, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        gs_opacity = F.grid_sample(self.plane_opacity, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)

        
        gs_pos = self.poslinear(gs_pos) 
        gs_rot = self.rotlinear(gs_rot) 
        
        #gs_pos = self.poslinear(gs_pos) 
        #gs_pos = 0.02*torch.tanh(gs_pos)#vert offset
        
        #gs_scale = torch.sigmoid(gs_scale)/30
        #gs_opacity = self.sigm_activation(gs_opacity)
        
        return gs_pos,gs_rot,gs_scale, gs_color, gs_opacity             