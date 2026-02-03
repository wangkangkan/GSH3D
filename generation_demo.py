import os
import torch
import imageio
import numpy as np
from tqdm import tqdm
from torchvision import transforms, utils
from scipy.spatial.transform import Rotation as R

from options import BaseOptions
from model_3dgs import GSHumanGenerator as Generator
from dataset_deepfashion_gs import DeepFashionDataset, DemoDataset
from utils import requires_grad

PANNING_ANGLE = np.pi / 3


def generate(opt, dataset, g_ema, device, is_video=False):
    requires_grad(g_ema, False)
    g_ema.eval()
    
    num_views = getattr(opt, 'num_views_per_id', 3)
    if num_views < 1:
        num_views = 3
    
    print(f"Generating {opt.identities} identities...")
    
    for i in tqdm(range(opt.identities)):
        sample_z = torch.randn(1, opt.style_dim, device=device)
        sample_trans, sample_beta, sample_theta = dataset.sample_smpl_param(1, device, val=False)
        sample_cam_extrinsics, sample_focals = dataset.get_camera_extrinsics(1, device, val=False)
        
        if is_video:
            video_list = []
            
            for k in tqdm(range(120), desc=f"Identity {i}", leave=False):
                if k < 30:
                    angle = (PANNING_ANGLE / 2) * (k / 30)
                elif k >= 30 and k < 90:
                    angle = PANNING_ANGLE / 2 - PANNING_ANGLE * ((k - 30) / 60)
                else:
                    angle = -PANNING_ANGLE / 2 * ((120 - k) / 30)
                
                delta = R.from_rotvec(angle * np.array([0, 1, 0]))
                r = R.from_rotvec(sample_theta[0, :3].cpu().numpy())
                new_r = delta * r
                new_sample_theta = sample_theta.clone()
                new_sample_theta[0, :3] = torch.from_numpy(new_r.as_rotvec()).to(device)
                
                with torch.no_grad():
                    out = g_ema(0, sample_z,
                               sample_cam_extrinsics,
                               sample_focals,
                               sample_beta,
                               new_sample_theta,
                               sample_trans)
                    
                    rgb_image = out['image'].detach().cpu()[..., :3]
                    g_ema.zero_grad()
                    frame = ((rgb_image.numpy() + 1) / 2. * 255.).clip(0, 255).astype(np.uint8)
                    video_list.append(frame)
            
            all_img = np.concatenate(video_list, 0)
            video_path = os.path.join(opt.results_dst_dir, 'videos', f'video_{str(i).zfill(7)}.mp4')
            imageio.mimwrite(video_path, all_img, fps=30, quality=8)
            
        else:
            img_list = []
            
            for k in range(num_views):
                if num_views == 1:
                    angle = 0
                elif num_views == 3:
                    angles = [-np.pi/8, 0, np.pi/8]
                    angle = angles[k]
                else:
                    angle = (k / (num_views - 1) - 0.5) * (np.pi / 4)
                
                delta = R.from_rotvec(angle * np.array([0, 1, 0]))
                r = R.from_rotvec(sample_theta[0, :3].cpu().numpy())
                new_r = delta * r
                new_sample_theta = sample_theta.clone()
                new_sample_theta[0, :3] = torch.from_numpy(new_r.as_rotvec()).to(device)
                
                with torch.no_grad():
                    out = g_ema(0, sample_z,
                               sample_cam_extrinsics,
                               sample_focals,
                               sample_beta,
                               new_sample_theta,
                               sample_trans)
                    
                    rgb_image = out['image'].detach().cpu()[..., :3].permute(0, 3, 1, 2)
                    g_ema.zero_grad()
                    img_list.append(rgb_image)
            
            combined_images = torch.cat(img_list, 0)
            save_path = os.path.join(opt.results_dst_dir, 'images', f'{str(i).zfill(7)}.png')
            utils.save_image(combined_images,
                           save_path,
                           nrow=num_views,
                           normalize=True,
                           range=(-1, 1),
                           padding=0)
    
    print(f"Done. Results saved to: {opt.results_dst_dir}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = BaseOptions().parse()
    
    opt.inference.size = opt.model.size
    opt.inference.camera = opt.camera
    opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.inference.style_dim = opt.model.style_dim
    
    checkpoints_dir = os.path.join('checkpoint', opt.experiment.expname, 'volume_renderer')
    checkpoint_path = os.path.join(checkpoints_dir, f'models_{opt.experiment.ckpt.zfill(7)}.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model not found: {checkpoint_path}")
        return
    
    result_model_dir = f'iter_{opt.experiment.ckpt.zfill(7)}'
    results_dir_basename = os.path.join(opt.inference.results_dir, opt.experiment.expname)
    opt.inference.results_dst_dir = os.path.join(results_dir_basename, result_model_dir)
    
    os.makedirs(opt.inference.results_dst_dir, exist_ok=True)
    if opt.rendering.render_video:
        os.makedirs(os.path.join(opt.inference.results_dst_dir, 'videos'), exist_ok=True)
    else:
        os.makedirs(os.path.join(opt.inference.results_dst_dir, 'images'), exist_ok=True)
    
    print(f"Experiment: {opt.experiment.expname}")
    print(f"Checkpoint: {opt.experiment.ckpt}")
    print(f"Output: {opt.inference.results_dst_dir}")
    
    print(f"Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    
    g_ema = Generator().to(device)
    
    pretrained_weights_dict = checkpoint["g_ema"]
    model_dict = g_ema.state_dict()
    
    for k, v in pretrained_weights_dict.items():
        if k in model_dict and v.size() == model_dict[k].size():
            model_dict[k] = v
    
    g_ema.load_state_dict(model_dict)
    g_ema.eval()
    
    print("Model loaded.")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    ])
    
    dataset = None
    if hasattr(opt.dataset, 'dataset_path') and opt.dataset.dataset_path:
        if os.path.exists(opt.dataset.dataset_path):
            file_list = os.path.join(opt.dataset.dataset_path, 'train_list.txt')
            if os.path.exists(file_list):
                dataset = DeepFashionDataset(
                    opt.dataset.dataset_path, 
                    transform, 
                    opt.model.size,
                    opt.model.renderer_spatial_output_dim, 
                    file_list
                )
    
    if dataset is None:
        dataset = DemoDataset()
    
    generate(opt.inference, dataset, g_ema, device, opt.rendering.render_video)


if __name__ == "__main__":
    main()
