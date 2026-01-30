#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
        #x = (x - self.center) / self.scale + 0.6
        x[:,1] = x[:,1] + 0.2
        assert x.max() <= 1 + EPS and x.min() >= -1-EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        #x = x * 2 - 1
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
        
    def forward_batch(self, x):
        batchsize = x.shape[0]
        #x = (x - self.center) / self.scale + 0.6
        x[...,1] = x[...,1] + 0.2
        assert x.max() <= 1 + EPS and x.min() >= -1-EPS, f"x must be in [-1, 1], got {x.min()} and {x.max()}"
        #x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(batchsize, -1, 1, 3)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        feat_xy = F.grid_sample(self.plane_xy.repeat(batchsize, 1, 1, 1), coords[..., [0, 1]], align_corners=True)[..., 0].transpose(1, 2)
        feat_xz = F.grid_sample(self.plane_xz.repeat(batchsize, 1, 1, 1), coords[..., [0, 2]], align_corners=True)[..., 0].transpose(1, 2)
        feat_yz = F.grid_sample(self.plane_yz.repeat(batchsize, 1, 1, 1), coords[..., [1, 2]], align_corners=True)[..., 0].transpose(1, 2)
        feat = torch.cat([feat_xy, feat_xz, feat_yz], dim=-1)
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
        #x = (x - self.center) / self.scale + 0.6
        x[:,1] = x[:,1] + 0.2
        assert x.max() <= 1 + EPS and x.min() >= -1-EPS, f"x must be in [-1, 1], got {x.min()} and {x.max()}"
        #x = x * 2 - 1
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