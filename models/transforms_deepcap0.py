import torch
from math import sqrt, exp

#from submodules.nerf_pytorch.run_nerf_helpers_mod import get_rays, get_rays_ortho


class ImgToPatch(object):
    def __init__(self, ray_sampler, hwf):
        self.ray_sampler = ray_sampler
        self.hwf = hwf      # camera intrinsics

    def __call__(self, img):
        rgbs = []
        for img_i in img:
            pose = torch.eye(4)         # use dummy pose to infer pixel values
            _, selected_idcs, pixels_i = self.ray_sampler(H=self.hwf[0], W=self.hwf[1], focal=self.hwf[2], pose=pose)
            if selected_idcs is not None:
                rgbs_i = img_i.flatten(1, 2).t()[selected_idcs]
            else:
                rgbs_i = torch.nn.functional.grid_sample(img_i.unsqueeze(0), 
                                     pixels_i.unsqueeze(0), mode='bilinear', align_corners=True)[0]
                rgbs_i = rgbs_i.flatten(1, 2).t()
            rgbs.append(rgbs_i)

        rgbs = torch.cat(rgbs, dim=0)       # (B*N)x3

        return rgbs

class RealImgToPatch(object):
    def __init__(self, RaySampler):
        self.ray_sampler = RaySampler
        #self.hwf = hwf      # camera intrinsics

    def __call__(self, img, H, W):#, strtag, location
        rgbs = []
        #for img_i in img:
        for i in range(img.shape[0]): 
            pose = torch.eye(4)         # use dummy pose to infer pixel values
            selected_idcs, pixels_i = self.ray_sampler.realimage_sampler(H, W, img[i])#, strtag, location[i]
            if selected_idcs is not None:
                rgbs_i = img[i].flatten(1, 2).t()[selected_idcs]
            else:
                pixels_i = pixels_i.to(img)
                rgbs_i = torch.nn.functional.grid_sample(img[i].unsqueeze(0), 
                                     pixels_i.unsqueeze(0), mode='bilinear', align_corners=True)[0]
                rgbs_i = rgbs_i.flatten(1, 2).t()
            rgbs.append(rgbs_i)

        rgbs = torch.cat(rgbs, dim=0)       # (B*N)x3

        return rgbs
            
class RaySampler(object):
    def __init__(self, N_samples, orthographic=False):
        super(RaySampler, self).__init__()
        self.N_samples = N_samples
        self.scale = torch.ones(1,).float()
        self.return_indices = True
        self.orthographic = orthographic

    def __call__(self, H, W, rays_o, rays_d, ray_start, ray_end, img):#renderclothmask,rendermeshmask,location
        # if self.orthographic:
            # size_h, size_w = focal      # Hacky
            # rays_o, rays_d = get_rays_ortho(H, W, pose, size_h, size_w)
        # else:
            # rays_o, rays_d = get_rays(H, W, focal, pose), focal, pose

        #select_inds = self.sample_rays(H, W)
        select_inds = self.sample_rays_mask(H, W, img.permute(1,2,0)[...,0]*127.5+127.5)#rendermeshmask,location
        #select_inds = select_inds.to(rays_o)
        
        if self.return_indices:
            rays_o = rays_o.view(-1, 3)[select_inds]
            rays_d = rays_d.view(-1, 3)[select_inds]

            h = (select_inds // W) / float(H) - 0.5
            w = (select_inds %  W) / float(W) - 0.5

            hw = torch.stack([h,w]).t()

        else:
            
            rays_o = torch.nn.functional.grid_sample(rays_o.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_d = torch.nn.functional.grid_sample(rays_d.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            
            rays_o = rays_o.permute(1,2,0).view(-1, 3)
            rays_d = rays_d.permute(1,2,0).view(-1, 3)

            ray_start = torch.nn.functional.grid_sample(ray_start.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            ray_end = torch.nn.functional.grid_sample(ray_end.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            ray_start = ray_start.permute(1,2,0).view(-1, 1)
            ray_end = ray_end.permute(1,2,0).view(-1, 1)
            
            # renderclothmask = renderclothmask[...,None]
            # renderclothmask = torch.nn.functional.grid_sample(renderclothmask.permute(2,0,1).unsqueeze(0), 
                                 # select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            # #renderclothmask = renderclothmask.flatten(1, 2).t()
            # renderclothmask = renderclothmask.permute(1,2,0).view(-1, 1)
            # #renderclothmask = clothmaskrepeat(1,Nc).view(-1,Nc,1)
            
            hw = select_inds
            select_inds = None

        return rays_o, rays_d, ray_start, ray_end, select_inds, hw#, renderclothmasktorch.stack([rays_o, rays_d])

    def realimage_sampler(self, H, W, img):#, location
        
        #select_inds = self.sample_rays(H, W)
        select_inds = self.sample_rays_mask(H, W, img.permute(1,2,0)[...,0]*127.5+127.5)#, location

        if self.return_indices:          
            h = (select_inds // W) / float(H) - 0.5
            w = (select_inds %  W) / float(W) - 0.5

            hw = torch.stack([h,w]).t()

        else:
          
            hw = select_inds
            select_inds = None

        return select_inds, hw
            
    def sample_rays(self, H, W):
        raise NotImplementedError

class RaySampler_globallocal(object):
    def __init__(self, N_samples, orthographic=False):
        super(RaySampler_globallocal, self).__init__()
        self.N_samples = N_samples
        self.scale = torch.ones(1,).float()
        self.return_indices = True
        self.orthographic = orthographic

    def __call__(self, H, W, rays_o, rays_d, ray_start, ray_end, mask_at_box, img, strtag):#,location
        # if self.orthographic:
            # size_h, size_w = focal      # Hacky
            # rays_o, rays_d = get_rays_ortho(H, W, pose, size_h, size_w)
        # else:
            # rays_o, rays_d = get_rays(H, W, focal, pose), focal, pose

        #select_inds = self.sample_rays(H, W)
        if strtag=='global':
            select_inds = self.sample_rays_global(H, W, img.permute(1,2,0)[...,0]*127.5+127.5)#,location
        if strtag=='local':
            select_inds = self.sample_rays_local(H, W, img.permute(1,2,0)[...,0]*127.5+127.5)
        #select_inds = select_inds.to(rays_o)
        
        if self.return_indices:
            rays_o = rays_o.view(-1, 3)[select_inds]
            rays_d = rays_d.view(-1, 3)[select_inds]

            h = (select_inds // W) / float(H) - 0.5
            w = (select_inds %  W) / float(W) - 0.5

            hw = torch.stack([h,w]).t()

        else:
            
            rays_o = torch.nn.functional.grid_sample(rays_o.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_d = torch.nn.functional.grid_sample(rays_d.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            
            rays_o = rays_o.permute(1,2,0).view(-1, 3)
            rays_d = rays_d.permute(1,2,0).view(-1, 3)

            ray_start = torch.nn.functional.grid_sample(ray_start.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            ray_end = torch.nn.functional.grid_sample(ray_end.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            ray_start = ray_start.permute(1,2,0).view(-1, 1)
            ray_end = ray_end.permute(1,2,0).view(-1, 1)
            
            mask_at_box = torch.nn.functional.grid_sample(mask_at_box.permute(2,0,1).unsqueeze(0).float(), 
                                 select_inds.unsqueeze(0), mode='nearest', align_corners=True)[0]
            mask_at_box = mask_at_box.permute(1,2,0).view(-1).bool()
            
            # renderclothmask = renderclothmask[...,None]
            # renderclothmask = torch.nn.functional.grid_sample(renderclothmask.permute(2,0,1).unsqueeze(0), 
                                 # select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            # #renderclothmask = renderclothmask.flatten(1, 2).t()
            # renderclothmask = renderclothmask.permute(1,2,0).view(-1, 1)
            # #renderclothmask = clothmaskrepeat(1,Nc).view(-1,Nc,1), renderclothmask
            
            hw = select_inds
            select_inds = None

        return rays_o, rays_d, ray_start, ray_end, mask_at_box, select_inds, hw#torch.stack([rays_o, rays_d])

    def realimage_sampler(self, H, W, img, strtag):#, location
        
        #select_inds = self.sample_rays(H, W)
        if strtag=='global':
            select_inds = self.sample_rays_global(H, W, img.permute(1,2,0)[...,0]*127.5+127.5)#, location
        if strtag=='local':
            select_inds = self.sample_rays_local(H, W, img.permute(1,2,0)[...,0]*127.5+127.5)#, location

        if self.return_indices:          
            h = (select_inds // W) / float(H) - 0.5
            w = (select_inds %  W) / float(W) - 0.5

            hw = torch.stack([h,w]).t()

        else:
          
            hw = select_inds
            select_inds = None

        return select_inds, hw
            
    def sample_rays(self, H, W):
        raise NotImplementedError
        
class FullRaySampler(RaySampler):
    def __init__(self, **kwargs):
        super(FullRaySampler, self).__init__(N_samples=None, **kwargs)

    def sample_rays(self, H, W):
        return torch.arange(0, H*W)


class FlexGridRaySampler(RaySampler):
    def __init__(self, N_samples, random_shift=True, random_scale=True, min_scale=0.25, max_scale=1., scale_anneal=-1,
                 **kwargs):
        self.N_samples_sqrt = int(sqrt(N_samples))
        super(FlexGridRaySampler, self).__init__(self.N_samples_sqrt**2, **kwargs)

        self.random_shift = random_shift
        self.random_scale = random_scale

        self.min_scale = min_scale
        self.max_scale = max_scale

        # nn.functional.grid_sample grid value range in [-1,1]
        self.w, self.h = torch.meshgrid([torch.linspace(-1,1,self.N_samples_sqrt),
                                         torch.linspace(-1,1,self.N_samples_sqrt)])
        self.h = self.h.unsqueeze(2)
        self.w = self.w.unsqueeze(2)

        # directly return grid for grid_sample
        self.return_indices = False

        self.iterations = 0
        self.scale_anneal = scale_anneal

    def sample_rays(self, H, W):

        if self.scale_anneal>0:
            k_iter = self.iterations // 1000 * 3
            min_scale = max(self.min_scale, self.max_scale * exp(-k_iter*self.scale_anneal))
            #print(self.iterations,k_iter,min_scale)
            min_scale = min(0.9, min_scale)
        else:
            min_scale = self.min_scale

        scale = 1
        if self.random_scale:
            scale = torch.Tensor(1).uniform_(0.1, self.max_scale)#min_scale
            h = self.h * scale 
            w = self.w * scale 

        if self.random_shift:
            max_offset = 1-scale.item()
            h_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2
            w_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2
            
            h += h_offset
            w += w_offset

        self.scale = scale

        return torch.cat([h, w], dim=2)
        # return torch.cat([self.h, self.w], dim=2)
        
    def sample_rays_mask(self, H, W, rendermask):#,location

        # randomoffset = []
        # for i in range(batch_size):
            # nonzero_indices = torch.nonzero(rendermask[i])
            # randidx = torch.randint(nonzero_indices.shape[0],(1,))
            # randomoffset.append([(nonzero_indices[randidx, 0]-256)/256, (nonzero_indices[randidx, 1]-256)/256])
        # randomoffset = torch.cat(randomoffset)
        
        if self.N_samples_sqrt==512:#full resolution
            self.h = self.h.to(rendermask)
            self.w = self.w.to(rendermask)
            return torch.cat([self.h, self.w], dim=2)
            
        else:
            nonzero_indices = torch.nonzero(rendermask)        
            randidx = torch.randint(nonzero_indices.shape[0],(1,))
            h_offset = (nonzero_indices[randidx, 1]-256)/256
            w_offset = (nonzero_indices[randidx, 0]-256)/256
            
            #specific location
            # h_offset = (location[0]-256)/256
            # w_offset = (location[1]-256)/256

            max_offset = max(abs(h_offset),abs(w_offset))
            
            max_scale = 1-max_offset.item()
            if max_scale<=0.05:#0.125
                max_scale = 1
                h_offset = 0
                w_offset = 0
            
            scale = torch.Tensor(1).uniform_(0.05, max_scale)#min_scale,0.06#0.125

            h = self.h * scale 
            w = self.w * scale 
            h = h.to(h_offset)
            w = w.to(w_offset)
            
            h += h_offset
            w += w_offset

            return torch.cat([h, w], dim=2)
    
    def sample_rays_local(self, H, W, humanmask):#,location

        if self.N_samples_sqrt==H:#full resolution
            self.h = self.h.to(humanmask)
            self.w = self.w.to(humanmask)
            return torch.cat([self.h, self.w], dim=2)
            
        else:
            nonzero_indices = torch.nonzero(humanmask)        
            randidx = torch.randint(nonzero_indices.shape[0],(1,))
            h_offset = (nonzero_indices[randidx, 1]-H/2)/(H/2)
            w_offset = (nonzero_indices[randidx, 0]-W/2)/(W/2)
            
            #specific location
            # h_offset = (location[0]-256)/256
            # w_offset = (location[1]-256)/256

            max_offset = max(abs(h_offset),abs(w_offset))
            
            max_scale = 1-max_offset.item()
            scalether = self.N_samples_sqrt/H
            if max_scale<=scalether:
                max_scale = 1
                h_offset = 0
                w_offset = 0
            
            scale = scalether#torch.Tensor(1).uniform_(scalether, max_scale)#min_scale,0.06#0.125

            h = self.h * scale 
            w = self.w * scale 
            h = h.to(h_offset)
            w = w.to(w_offset)
            
            h += h_offset
            w += w_offset

            return torch.cat([h, w], dim=2)    
    
    def sample_rays_global(self, H, W, humanmask):#,location

        if self.N_samples_sqrt==H:#full resolution
            self.h = self.h.to(humanmask)
            self.w = self.w.to(humanmask)
            return torch.cat([self.h, self.w], dim=2)
            
        else:
            
            h = self.h  
            w = self.w
            h = h.to(humanmask)
            w = w.to(humanmask)
            
            return torch.cat([h, w], dim=2)      