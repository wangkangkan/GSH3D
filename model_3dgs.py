import torch.nn as nn
#from lib.config import cfg
import torch
from lib.networks.renderer import if_clight_renderer_occupancy_singletemplate as if_clight_renderer_occupancy
from lib.networks.gs_network_snug_singletemplate import Network#
from torch.nn import functional as F
import numpy as np
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

class GSHumanGenerator(nn.Module):
    def __init__(self):
        super(GSHumanGenerator, self).__init__()

        self.net = Network()
        self.renderer = if_clight_renderer_occupancy.Renderer(self.net)
        
        # for param in self.net.cloth_simulation.parameters():
             # param.requires_grad = False   

        # self.style_dim = 128
        # layers = []
        # for i in range(3):
            # layers.append(
                # MappingLinear(self.style_dim, self.style_dim, activation="fused_lrelu")
            # )

        # self.style = nn.Sequential(*layers)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def styles_and_noise_forward(self, styles, inject_index=None, truncation=1,
                                 truncation_latent=None, input_is_latent=False):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent[0] + truncation * (style - truncation_latent[0])
                )

            styles = style_t

        return styles
    
    def mean_latent(self, n_latent, device):
        latent_in = torch.randn(n_latent, self.style_dim, device=device)
        renderer_latent = self.style(latent_in)
        renderer_latent_mean = renderer_latent.mean(0, keepdim=True)
        #if self.full_pipeline:
        #    decoder_latent_mean = self.decoder.mean_latent(renderer_latent)
        #else:
        decoder_latent_mean = None

        return [renderer_latent_mean, decoder_latent_mean]
        
    def forward(self, gradtag, styles, cam_poses, focals, beta, theta, trans,
                inject_index=None, truncation=1, truncation_latent=None,
                input_is_latent=False):
        
        #latent = self.styles_and_noise_forward(styles, inject_index, truncation, truncation_latent, input_is_latent)
                                                    
        #styles = [self.style(s) for s in styles]
        
        #sdfstyles = self.style(styles)
        
        #ret = self.renderer.batch_render_deformation_UV(latent, cam_poses, focals, beta, theta, trans)#[0]
        #out = (None, ret['image'])
        
        # self.renderer.batch_generatefeature(latent[0], cam_poses, focals, beta, theta, trans)
        
        # rendered_img_batch = []
        # for bidx in range(0, latent.shape[0]):
            # ret = self.renderer.gsrendering(bidx, cam_poses[bidx:bidx+1], focals[bidx:bidx+1])
            # rendered_img_batch += [ret['image']]
        # rendered_img_batch = torch.cat(rendered_img_batch, 0)
        # ret = {"image": rendered_img_batch}
        # styles = styles.to(self.renderer.thetazero.device)
        # cam_poses = cam_poses.to(self.renderer.thetazero.device)
        # focals = focals.to(self.renderer.thetazero.device)
        # beta = beta.to(self.renderer.thetazero.device)
        # theta = theta.to(self.renderer.thetazero.device)
        # trans = trans.to(self.renderer.thetazero.device)
        # print('1:',beta.device)
        latent = styles
        #self.renderer.batch_generategs_body_clothing_separate(clothweight, latent[0], cam_poses, focals, beta, theta, trans)
        self.renderer.batch_generatefeature(latent, cam_poses, focals, beta, theta, trans)#[0]
        
        #ret = self.renderer.gsrendering_body(0, cam_poses[0:1], focals[0:1])
        #rendered_img_batch = ret['image']
        
        alldepth_normal_loss = [] 
        allnormal_loss = [] 
        allsurface_sdfloss = [] 
        allgrad_loss = []
        allcurvature_loss = []        
        rendered_img_batch = []
        for bidx in range(0, latent.shape[0]):#[0]sdfstyles[bidx:bidx+1],, normal_loss, surface_sdfloss, grad_loss, curvature_loss
            ret = self.renderer.gsrendering(gradtag, bidx, cam_poses[bidx:bidx+1], focals[bidx:bidx+1])
            rendered_img_batch += [ret['image']]
            #alldepth_normal_loss += [depth_normal_loss.view([-1])]
            # allnormal_loss += [normal_loss.view([-1])]
            # allsurface_sdfloss += [surface_sdfloss.view([-1])]
            # allgrad_loss += [grad_loss.view([-1])]
            # allcurvature_loss += [curvature_loss.view([-1])]
        #self.depthnormal_loss = torch.cat(alldepth_normal_loss, 0).sum()
        # self.normal_loss = torch.cat(allnormal_loss, 0).sum() 
        # self.surface_sdfloss = torch.cat(allsurface_sdfloss, 0).sum() 
        # self.grad_loss = torch.cat(allgrad_loss, 0).sum()  
        # self.curvature_loss = torch.cat(allcurvature_loss, 0).sum()        
        rendered_img_batch = torch.cat(rendered_img_batch, 0)
        ret = {"image": rendered_img_batch}
        
        return ret

    def feature_edit(self, gradtag, styles, cam_poses, focals, beta, theta, trans, gs_incpos, gs_rot, gs_scale, gs_color, gs_opacity):
        
        #latent = self.styles_and_noise_forward(styles, inject_index, truncation, truncation_latent, input_is_latent)
                                                    
        #styles = [self.style(s) for s in styles]
        latent = styles
        #sdfstyles = self.style(styles)
        
        #ret = self.renderer.batch_render_deformation_UV(latent, cam_poses, focals, beta, theta, trans)#[0]
        #out = (None, ret['image'])
        
        # self.renderer.batch_generatefeature(latent[0], cam_poses, focals, beta, theta, trans)
        
        # rendered_img_batch = []
        # for bidx in range(0, latent.shape[0]):
            # ret = self.renderer.gsrendering(bidx, cam_poses[bidx:bidx+1], focals[bidx:bidx+1])
            # rendered_img_batch += [ret['image']]
        # rendered_img_batch = torch.cat(rendered_img_batch, 0)
        # ret = {"image": rendered_img_batch}
        
        #self.renderer.batch_generategs_body_clothing_separate(clothweight, latent[0], cam_poses, focals, beta, theta, trans)
        self.renderer.batch_editfeature(latent, cam_poses, focals, beta, theta, trans, gs_incpos, gs_rot, gs_scale, gs_color, gs_opacity)#[0]
        
        #ret = self.renderer.gsrendering_body(0, cam_poses[0:1], focals[0:1])
        #rendered_img_batch = ret['image']
        
        alldepth_normal_loss = [] 
        allnormal_loss = [] 
        allsurface_sdfloss = [] 
        allgrad_loss = []
        allcurvature_loss = []        
        rendered_img_batch = []
        for bidx in range(0, latent.shape[0]):#[0]sdfstyles[bidx:bidx+1],, normal_loss, surface_sdfloss, grad_loss, curvature_loss
            ret = self.renderer.gsrendering(gradtag, bidx, cam_poses[bidx:bidx+1], focals[bidx:bidx+1])
            rendered_img_batch += [ret['image']]
            #alldepth_normal_loss += [depth_normal_loss.view([-1])]
            # allnormal_loss += [normal_loss.view([-1])]
            # allsurface_sdfloss += [surface_sdfloss.view([-1])]
            # allgrad_loss += [grad_loss.view([-1])]
            # allcurvature_loss += [curvature_loss.view([-1])]
        #self.depthnormal_loss = torch.cat(alldepth_normal_loss, 0).sum()
        # self.normal_loss = torch.cat(allnormal_loss, 0).sum() 
        # self.surface_sdfloss = torch.cat(allsurface_sdfloss, 0).sum() 
        # self.grad_loss = torch.cat(allgrad_loss, 0).sum()  
        # self.curvature_loss = torch.cat(allcurvature_loss, 0).sum()        
        rendered_img_batch = torch.cat(rendered_img_batch, 0)
        ret = {"image": rendered_img_batch}
        
        return ret
        
class MappingLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None, is_last=False):
        super().__init__()
        if is_last:
            weight_std = 0.25
        else:
            weight_std = 1

        self.weight = nn.Parameter(weight_std * nn.init.kaiming_normal_(torch.empty(out_dim, in_dim), a=0.2, mode='fan_in', nonlinearity='leaky_relu'))

        if bias:
            self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), a=-np.sqrt(1/in_dim), b=np.sqrt(1/in_dim)))
        else:
            self.bias = None

        self.activation = activation

    def forward(self, input):
        if self.activation != None:
            out = F.linear(input, self.weight)
            out = fused_leaky_relu(out, self.bias, scale=1)
        else:
            out = F.linear(input, self.weight, bias=self.bias)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )