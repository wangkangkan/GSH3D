import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import if_clight_renderer_occupancy
from lib.config import cfg
from lib.utils.loss import ssim


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = if_clight_renderer_occupancy.Renderer(self.net)
        
        for param in self.net.cloth_simulation.parameters():
             param.requires_grad = False   
             
        # TODO: 重新训练
        self.net.tempclothpara.weight.requires_grad = False 
        

 
        self.acc_crit = torch.nn.functional.smooth_l1_loss

        self.msk2mse = lambda x, y: torch.mean((x - y) ** 2)
        
        self.sdf_crit = torch.nn.L1Loss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def l1_loss(self, network_output, gt, mask=None):
        if mask is not None:
            return torch.abs((network_output - gt)).sum() / mask.sum()
        else:
            return torch.abs((network_output - gt)).mean()
    
    def img2mse(self, x, y, M=None):
        if M == None:
            return torch.mean((x - y) ** 2)
        else:
            return torch.sum((x - y) ** 2 * M) / (torch.sum(M) + 1e-8) / x.shape[-1]     
    def forward(self, batch, epoch):
            
        ret = self.renderer.render_deformation(batch, epoch)
        
        scalar_stats = {}
        loss = 0

        # mask = batch['mask_at_box']
        # msk = batch['msk'].float()
        # img_loss = self.img2mse(ret['image'].unsqueeze(0), batch['img'] * msk.unsqueeze(-1))
        # scalar_stats.update({'img_loss': img_loss})
        # loss += 200.0 *img_loss
        msk = batch['msk'].float()
        img_loss = self.l1_loss(ret['image'].unsqueeze(0), batch['img'] * msk.unsqueeze(-1))
        scalar_stats.update({'img_loss': img_loss})
        loss += 5.0 *img_loss
        
        loss_ssim = 1.0 - ssim(ret['image'].unsqueeze(0), batch['img'] * msk.unsqueeze(-1))
        loss_ssim = loss_ssim * (msk.sum() / (ret['image'].shape[-1] * ret['image'].shape[-2]))
        scalar_stats.update({'loss_ssim': loss_ssim})
        loss += 1.0 *loss_ssim
        
        # scalar_stats.update({'cloth_img_loss': self.renderer.cloth_img_loss})
        # loss += 5.0 * self.renderer.cloth_img_loss
        # scalar_stats.update({'cloth_ssim_loss': self.renderer.cloth_ssim})
        # loss += 1.0 * self.renderer.cloth_ssim
        # scalar_stats.update({'smpl_img_loss': self.renderer.smpl_img_loss})
        # loss += 5.0 * self.renderer.cloth_img_loss
        # scalar_stats.update({'smpl_ssim_loss': self.renderer.smpl_ssim})
        # loss += 1.0 * self.renderer.smpl_ssim
        # import cv2
        # import numpy as np
        # color_mask = ret['image'].detach().cpu().numpy() * 255
        # gt_msk = (batch['img'] * msk.unsqueeze(-1)).detach().cpu().numpy()[0] * 255
        # merge_msk = cv2.add(color_mask.astype(np.uint8), gt_msk.astype(np.uint8))
        # cv2.imwrite(f"merge.png", merge_msk)

        
        # DEBUG: mask损失
        # scalar_stats.update({'IoUloss': self.renderer.IoUloss})
        # loss += 10.0 *self.renderer.IoUloss#10

        scalar_stats.update({'IoUloss_def': self.renderer.IoUloss_def})
        loss += 30.0 *self.renderer.IoUloss_def#30
        # DEBUG: END
        
        # scalar_stats.update({'attach_loss': self.renderer.attach_loss})
        # loss += 0.1 *self.renderer.attach_loss#0.1
        # DEBUG: END
        
        # smplimg_loss = self.img2mse(ret['rgb_map_s'], batch['rgb'], 1-self.renderer.coordclothrendermask)#coord_silhouette
        # scalar_stats.update({'smplimg_loss': smplimg_loss})
        # loss += 1.0 *smplimg_loss
        
        # clothimg_loss = self.img2mse(ret['rgb_map_d'], batch['rgb'], self.renderer.coord_silhouette)#
        # scalar_stats.update({'clothimg_loss': clothimg_loss})
        # loss += 1.0 *clothimg_loss

        #---------------
        # DEBUG: 刚性损失
        loss += 1.0 * self.renderer.smoothloss#5.0
        # loss += 5.0 * self.renderer.smoothloss
        scalar_stats.update({'smoothloss': self.renderer.smoothloss})
        
        loss += 0.1 * self.renderer.smoothloss_smpl#5.0
        # # loss += 1.0 * self.renderer.smoothloss_smpl
        scalar_stats.update({'smoothloss_smpl': self.renderer.smoothloss_smpl})
        # DEBUG: END
        
        # NOTE: disp碰撞损失
        loss += 100.0 * self.renderer.interploss_graphdeform_disp
        scalar_stats.update({'interploss_graphdeform_disp': self.renderer.interploss_graphdeform_disp})
        
        # NOTE: 服装模拟损失
        scalar_stats.update({'graphdeform_loss': self.renderer.graphdeform_loss})
        loss += 0.01 *self.renderer.graphdeform_loss
        
        

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
