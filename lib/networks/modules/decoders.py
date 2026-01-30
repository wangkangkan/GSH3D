#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.networks.modules.triplane import TriPlane
#from lib.config import cfg

from .activation import SineActivation


act_fn_dict = {
    'softplus': torch.nn.Softplus(),
    'relu': torch.nn.ReLU(),
    'sine': SineActivation(omega_0=30),
    'gelu': torch.nn.GELU(),
    'tanh': torch.nn.Tanh(),
}

class PositionEncoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, num_train_frame = 1, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
            
        # self.net = torch.nn.Sequential(
        #     nn.Linear(3, self.hidden_dim),
        #     act_fn_dict[act],
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     act_fn_dict[act],
        # )
        n_features = 32
        triplane_res = 256
        self.position_enc = TriPlane(n_features,
                                        resX=triplane_res,
                                        resY=triplane_res,
                                        resZ=triplane_res)
        
        # self.final_layer = nn.Linear(hidden_dim, n_features)
        self.latentdeform = nn.Embedding(num_train_frame, n_features)
        
    def forward(self, x, frame_index):

        num_v = x.shape[0]
        # position_code = self.final_layer(self.net(x))
        position_code = self.position_enc(x)
        latent = self.latentdeform(frame_index)
        latent = latent[None].repeat(num_v, 1)
        
        x = torch.concatenate([position_code, latent], dim=-1)
        return x
 

class AppearanceDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
            
        self.net = torch.nn.Sequential(
            nn.Linear(n_features+128, self.hidden_dim),#
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.opacity = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        # self.shs = nn.Linear(hidden_dim, 16*3)
        self.shs = nn.Linear(hidden_dim, 3)
        
    def forward(self, x):

        x1 = self.net(x)
        shs = self.shs(x1)
        shs = torch.sigmoid(shs)*(1 + 2*0.001) - 0.001 # follow mip-nerf
        shs = shs*255
        #shs = torch.sigmoid(shs)*255
        # shs = 2*shs-1
        opacity = self.opacity(x1)
        return {'shs': shs, 'opacity': opacity}
    

class DeformationDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=128, weight_norm=True, act='gelu', disable_posedirs=False):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.sine = SineActivation(omega_0=30)
        self.disable_posedirs = disable_posedirs
        
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.skinning_linear = nn.Linear(hidden_dim, hidden_dim)
        self.skinning = nn.Linear(hidden_dim, 24)
        
        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
            
        # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
        if not disable_posedirs:
            self.blendshapes = nn.Linear(hidden_dim, 3 * 207)
            torch.nn.init.constant_(self.blendshapes.bias, 0.0)
            torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        
    def forward(self, x):
        x = self.net(x)
        if not self.disable_posedirs:
            posedirs = self.blendshapes(x)
            posedirs = posedirs.reshape(207, -1)
            
        lbs_weights = self.skinning(F.gelu(self.skinning_linear(x)))
        lbs_weights = F.gelu(lbs_weights)
        
        return {
            'lbs_weights': lbs_weights,
            'posedirs': posedirs if not self.disable_posedirs else None,
        }
    

class GeometryDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=128, act='gelu', cloth=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = torch.nn.Sequential(
            nn.Linear(n_features+128, self.hidden_dim),#
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.clothtag = cloth
        if self.clothtag:
            self.xyz = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 3))
            self.xyzlinear  = nn.Linear(3, 3)
            torch.nn.init.constant(self.xyzlinear.weight, 0)
            torch.nn.init.constant(self.xyzlinear.bias, 0)
        self.rotations = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 2))
        # self.rotations = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 6))
        self.scales = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 2)) # 此处预测平面的尺度信息
        # self.scalelinear  = nn.Linear(2, 2)
        # torch.nn.init.constant(self.scalelinear.weight, 0)
        # torch.nn.init.constant(self.scalelinear.bias, 0.02)
        
    def forward(self, x):
        if self.clothtag:
            xyz = self.xyzlinear(self.xyz(x))
        else:
            xyz = []
        rotations = self.rotations(x)
        #scales = F.gelu(self.scales(x))
        #scales = self.scalelinear(self.scales(x)) 
        #scales = torch.clamp(torch.exp(scales1), max=0.04)#
        scales = torch.sigmoid(self.scales(x))/30#/50
        # scales = F.relu(self.scales(x))
        # scales = self.scales(x)
        return {
            'xyz': xyz,
            'rotations': rotations,
            'scales': scales
            #'scales1': scales1,
        }

class DispalceDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=128, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = torch.nn.Sequential(
            nn.Linear(n_features+128, self.hidden_dim),#
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )

        self.xyz = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 3))
        self.xyzlinear  = nn.Linear(3, 3)
        torch.nn.init.constant(self.xyzlinear.weight, 0)
        torch.nn.init.constant(self.xyzlinear.bias, 0)

    def forward(self, x):

        xyz = self.xyzlinear(self.xyz(x))

        return xyz
        
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
class AppearanceDecoder2(nn.Module):
    def __init__(self, n_features, D=8, W=256, style_dim=128, hidden_dim=64, act='gelu'):
        super(AppearanceDecoder2, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = n_features
        self.style_dim = style_dim

        # self.net = torch.nn.Sequential(
            # nn.Linear(n_features, hidden_dim),
            # act_fn_dict[act],
            # nn.Linear(hidden_dim, hidden_dim),
            # act_fn_dict[act],
        # )
        
        self.pts_linears = nn.ModuleList(
            [FiLMSiren(self.input_ch, hidden_dim, style_dim=style_dim, is_first=True)] + \
            [FiLMSiren(hidden_dim, hidden_dim, style_dim=style_dim) for i in range(D-1)])

        self.views_linears = FiLMSiren(hidden_dim, hidden_dim,
                                       style_dim=style_dim)
        self.shs = LinearLayer(hidden_dim, 3, freq_init=True)
        self.opacity = LinearLayer(hidden_dim, 1, freq_init=True)
        
        #self.opacity = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # self.shs = nn.Linear(hidden_dim, 16*3)
        #self.shs = nn.Linear(hidden_dim, 3)

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
        mlp_out = x
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i](mlp_out, styles)

        opacity = self.opacity(mlp_out)#self.net(x)
        opacity = torch.sigmoid(opacity)
        # shs = 2*shs-1
        out_features = self.views_linears(mlp_out, styles)
        shs = self.shs(out_features)
        
        #opacity = self.opacity(self.net(x))
        return {'shs': shs[0], 'opacity': opacity[0]}

# Siren Generator Model
class GeometryDecoder2(nn.Module):
    def __init__(self, n_features, D=8, W=256, style_dim=128, hidden_dim=64, act='gelu'):
        super(GeometryDecoder2, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = n_features
        self.style_dim = style_dim

        # self.net = torch.nn.Sequential(
            # nn.Linear(n_features, hidden_dim),
            # act_fn_dict[act],
            # nn.Linear(hidden_dim, hidden_dim),
            # act_fn_dict[act],
        # )
        
        self.pts_linears = nn.ModuleList(
            [FiLMSiren(self.input_ch, hidden_dim, style_dim=style_dim, is_first=True)] + \
            [FiLMSiren(hidden_dim, hidden_dim, style_dim=style_dim) for i in range(D-1)])

        self.views_linears = FiLMSiren(hidden_dim, hidden_dim,
                                       style_dim=style_dim)
        self.views_linears1 = FiLMSiren(hidden_dim, hidden_dim,
                                       style_dim=style_dim)
                                       
        self.xyz = LinearLayer(hidden_dim, 3, freq_init=True)
        self.rotations = LinearLayer(hidden_dim, 3, freq_init=True)
        self.scales = LinearLayer(hidden_dim, 3, freq_init=True)
        
        
        #self.opacity = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # self.shs = nn.Linear(hidden_dim, 16*3)
        #self.shs = nn.Linear(hidden_dim, 3)

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
        mlp_out = x
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i](mlp_out, styles)

        xyz = self.xyz(mlp_out)
        
        out_features = self.views_linears(mlp_out, styles)
        rotations = self.rotations(out_features)
        
        out_features1 = self.views_linears1(mlp_out, styles)
        scales = self.scales(out_features1)
        scales = torch.sigmoid(scales)/30
        
        return {
            'xyz': xyz[0],
            'rotations': rotations[0],
            'scales': scales[0],
        }
        
class SDFDecoder(nn.Module):
    def __init__(self, n_features, D=8, W=256, style_dim=128, hidden_dim=64, act='gelu'):
        super(SDFDecoder, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = n_features
        self.style_dim = style_dim
        
        self.pts_linears = nn.ModuleList(
            [FiLMSiren(self.input_ch, W, style_dim=style_dim, is_first=True)] + \
            [FiLMSiren(W, W, style_dim=style_dim) for i in range(D-1)])
        
        self.sigma_linear = LinearLayer(W, 1, freq_init=True)

    
    def forward(self, x, styles):
        mlp_out = x
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i](mlp_out, styles)

        sdf = self.sigma_linear(mlp_out)
        
        return sdf 