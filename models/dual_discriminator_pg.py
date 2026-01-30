import numpy as np
import torch
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
from models.networks_stylegan2_pg import DiscriminatorBlock, DiscriminatorEpilogue
from munch import Munch

def training_schedule(
    cur_nimg,
    dataset_resolution,
    phase_idx,
    lod_initial_resolution  = 256,        # Image resolution used at the beginning.
    lod_training_kimg       = [0.001,3000],      # Thousands of real images to show before doubling the resolution.
    lod_transition_kimg     = [0.005,0],      # Thousands of real images to show when fading in new layers.
):
    cur_kimg = cur_nimg / 1000.0
    if cur_kimg > (sum(lod_training_kimg[:(phase_idx+1)]) + sum(lod_transition_kimg[:(phase_idx+1)])):
        phase_idx  = min(phase_idx + 1, len(lod_training_kimg)-1)
    cur_lod_training_kimg = lod_training_kimg[phase_idx]
    cur_lod_transition_kimg = lod_transition_kimg[phase_idx]
    phase_kimg = (cur_kimg - (sum(lod_training_kimg[:phase_idx]) + sum(lod_transition_kimg[:phase_idx]))) if phase_idx > 0 else cur_kimg
    lod = np.log2(dataset_resolution)
    lod -= np.floor(np.log2(lod_initial_resolution))
    lod -= phase_idx
    if cur_lod_transition_kimg > 0:
        lod -= max(phase_kimg - cur_lod_training_kimg, 0.0) / cur_lod_transition_kimg
    lod = max(lod, 0.0)
    return lod, phase_idx

@persistence.persistent_class
class DualDiscriminator(torch.nn.Module):
    def __init__(self,                       # Conditioning label (C) dimensionality.
        img_resolution      = (512, 256),    # Input resolution.
        img_channels        = 3,             # Number of input color channels.
        architecture        = 'resnet',      # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,         # Overall multiplier for the number of channels.
        channel_max         = 512,           # Maximum number of channels in any layer.
        num_fp16_res        = 4,             # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,           # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = 0,             # Dimensionality of mapped conditioning label, None = default.
        epilogue_kwargs     = {},            # Arguments for DiscriminatorEpilogue.
        label_type          = 'smpl',
        final_res           = 4,             # 512情况下 to 8*8
        is_down             = True,
        progressive_training= True,
        ds_pg               = True,
    ):
        super().__init__()
        if isinstance(img_resolution, Munch):
            img_resolution = img_resolution.renderer_spatial_output_dim
        elif isinstance(img_resolution, tuple) and all(isinstance(item, Munch) for item in img_resolution):
            img_resolution = img_resolution[0].renderer_spatial_output_dim
        if isinstance(img_resolution, list):
            img_resolution = tuple(img_resolution)
        elif not isinstance(img_resolution, tuple):
            img_resolution = (img_resolution, img_resolution)

        self.final_res = final_res if isinstance(final_res, tuple) else (final_res, final_res)
        self.is_down = is_down
        self.ds_pg = ds_pg
        self.label_type = label_type

        self.img_resolution = img_resolution if isinstance(img_resolution, tuple) else (img_resolution, img_resolution)
        #print(f"img_resolution type: {type(self.img_resolution)}, value: {self.img_resolution}")
        
        self.img_resolution_log2 = int(np.log2(self.img_resolution[0] // self.final_res[0]) + 2)

        self.img_channels = img_channels
        
        if self.img_resolution[0]==self.img_resolution[1]:
            self.block_resolutions = [(2 ** i, 2 ** i) for i in range(self.img_resolution_log2, 2, -1)]
        else:
            self.block_resolutions = [(2 ** i, 2 ** (i - 1)) for i in range(self.img_resolution_log2, 2, -1)]

        channels_dict = {res: min(channel_base // res[0], channel_max) for res in self.block_resolutions + [self.final_res]}
        
        for res in self.block_resolutions:
            half_res = (res[0] // 2, res[1] // 2)
            if half_res not in channels_dict:
                channels_dict[half_res] = min(channel_base // half_res[0], channel_max)

        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        common_kwargs = dict(img_channels=self.img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res[0] < self.img_resolution[0] else 0
            tmp_channels = channels_dict[res] 
            out_channels = channels_dict[(res[0] // 2, res[1] // 2)]
            use_fp16 = (res[0] >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, with_rgb=progressive_training, ds_pg=ds_pg, **common_kwargs)
            setattr(self, f'b{res[0]}x{res[1]}', block)
            cur_layer_idx += block.num_layers
        
        final_res_int = self.final_res[0] if isinstance(self.final_res, tuple) else self.final_res
        if self.img_resolution[0]==self.img_resolution[1]:
            self.b4 = DiscriminatorEpilogue(channels_dict[self.final_res], cmap_dim=cmap_dim, resolution=(final_res_int, final_res_int), **epilogue_kwargs, **common_kwargs)        
        else:
            self.b4 = DiscriminatorEpilogue(channels_dict[self.final_res], cmap_dim=cmap_dim, resolution=(final_res_int, final_res_int // 2), **epilogue_kwargs, **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))

    def forward(self, img, lod=None):
        if lod is None:
            lod = 0
        
        x = None
        # print(f'block_resolutions: {self.block_resolutions}')
        for res_log2 in range(self.img_resolution_log2, 2, -1):
            #res = (2 ** (res_log2), 2 ** (res_log2 - 1))
            
            #print(self.img_resolution, res_log2, (2 ** (res_log2 - 2)),self.final_res)
            if self.img_resolution[0]==self.img_resolution[1]:
                #res = (2 ** (res_log2), 2 ** (res_log2))
                res = (2 ** (res_log2), 2 ** (res_log2))
            else:
                res = (2 ** (res_log2), 2 ** (res_log2 - 1))
            cur_lod = self.img_resolution_log2 - res_log2
            if lod < cur_lod + 1:
                # block = getattr(self, f'b{res}')
                block = getattr(self, f'b{res[0]}x{res[1]}')
                if cur_lod <= lod < cur_lod + 1:
                    #print(f'cur_lod:{cur_lod}, lod:{lod}, input{res}, first')
                    x, img = block(x, img, alpha=1e4)
                elif cur_lod -1 < lod < cur_lod:
                    alpha = lod -  np.floor(lod)
                    #print(f'cur_lod:{cur_lod}, lod:{lod}, input{res}, second')
                    x, img = block(x, img, alpha=alpha)
                else:
                    #print(f'cur_lod:{cur_lod}, lod:{lod}, input{res}, third')
                    x, img = block(x, img, alpha=None)

            if self.is_down:
                if lod > cur_lod and (img.shape[2]==512 or img.shape[2]==128):#only run for 512 transition stage (multi discri)
                    #print(f'cur_lod:{cur_lod}, lod:{lod}, downsample!')
                    img = torch.nn.functional.avg_pool2d(img, kernel_size=2, stride=2, padding=0)
                    #print(f'After downsample, img shape: {img.shape}')
            else:
                if cur_lod < lod < cur_lod + 1:
                    img = torch.nn.functional.avg_pool2d(img, kernel_size=2, stride=2, padding=0)

        cmap = None

        x = self.b4(x, img, cmap)
        gan_preds = x[:,0:1]
        gan_preds = gan_preds.view(-1, 1)
        pose_pred = None

        return gan_preds, pose_pred