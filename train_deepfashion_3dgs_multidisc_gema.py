import os
import random
import yaml
import torch
import warnings
import numpy as np
import torch.distributed as dist

from PIL import Image
from tqdm import tqdm
from typing import Optional

from torch.utils import data
from operator import itemgetter
from torch.nn import functional as F
from torch import nn, autograd, optim
from torchvision import transforms, utils
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, Sampler

from losses import *
from options import BaseOptions
from augment import AugmentPipe
from calculate_fid import get_fid
from dataset_deepfashion_gs import DeepFashionDataset, DemoDataset
from model import VolumeRenderDiscriminator
from libraries.stylegan2_ada_pytorch.training.networks import Discriminator
from model_3dgs import GSHumanGenerator as Generator
from distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size
from utils import set_emagen, data_sampler, requires_grad, requires_grad1, accumulate, accumulate1, sample_data, make_noise, mixing_noise, generate_camera_params
from models.transforms_localimg import RealImgToPatch, FlexGridRaySampler
import cv2
from skimage import io
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
warnings.filterwarnings("ignore")

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

def human_densification(generator, guassian_opt, human_gs_out_smpl, human_gs_out_cloth,
                            visibility_filter_smpl, visibility_filter_cloth,
                            radii_smpl, radii_cloth,
                            viewspace_point_tensor_smpl, viewspace_point_tensor_cloth,
                            iteration, optimizer):
    generator.net.sugar_model_smpl.max_radii2D[visibility_filter_smpl] = torch.max(
        generator.net.sugar_model_smpl.max_radii2D[visibility_filter_smpl], 
        radii_smpl[visibility_filter_smpl]
    )
    
    generator.net.sugar_model_cloth.max_radii2D[visibility_filter_cloth] = torch.max(
        generator.net.sugar_model_cloth.max_radii2D[visibility_filter_cloth], 
        radii_cloth[visibility_filter_cloth]
    )
    
    generator.net.sugar_model_smpl.add_densification_stats(viewspace_point_tensor_smpl, visibility_filter_smpl)
    generator.net.sugar_model_cloth.add_densification_stats(viewspace_point_tensor_cloth, visibility_filter_cloth)

    if iteration > guassian_opt.densify_from_iter and iteration % guassian_opt.densification_interval == 0:

        size_threshold = 20
        generator.net.sugar_model_smpl.densify_and_prune(
            human_gs_out_smpl,
            guassian_opt.densify_grad_threshold, 
            min_opacity=guassian_opt.prune_min_opacity, 
            extent=guassian_opt.densify_extent, 
            max_screen_size=size_threshold,
            max_n_gs=guassian_opt.max_n_gaussians,
            optimizer=optimizer
        )            

        generator.net.sugar_model_cloth.densify_and_prune(
            human_gs_out_cloth,
            guassian_opt.densify_grad_threshold, 
            min_opacity=guassian_opt.prune_min_opacity, 
            extent=guassian_opt.densify_extent, 
            max_screen_size=size_threshold,
            max_n_gs=guassian_opt.max_n_gaussians,
            optimizer=optimizer
        )
        return True        
    return False

def Gaussiananchor(generator, g_ema):

    anchorpos_loss = 1*torch.norm(generator.renderer.gs_incpos_smpl-g_ema.renderer.gs_incpos_smpl, p=2, dim=-1).mean()
    
    anchorrot_loss = 1*torch.norm(generator.renderer.rotations_smpl-g_ema.renderer.rotations_smpl, p=2, dim=-1).mean()
    anchorscale_loss = 1*torch.norm(generator.renderer.gs_scales_smpl-g_ema.renderer.gs_scales_smpl, p=2, dim=-1).mean()
    anchorcolor_loss = 1*torch.norm(generator.renderer.gs_shs_smpl/255-g_ema.renderer.gs_shs_smpl/255, p=2, dim=-1).mean()
    
    anchoropacity_loss = 0.1*torch.norm(generator.renderer.gs_opacity_smpl-g_ema.renderer.gs_opacity_smpl, p=2, dim=-1).mean()
    
    print(anchorpos_loss.item(), anchorrot_loss.item(), anchorscale_loss.item(), anchorcolor_loss.item(), anchoropacity_loss.item())    
    anchor_loss = anchorpos_loss + anchorrot_loss + anchorscale_loss + anchorcolor_loss + anchoropacity_loss
    
    return anchor_loss

def train(opt, experiment_opt, guassian_opt, _loader_dict, generator, discriminator, discriminator_head,discriminator_body,discriminator_leg, g_optim, d_optim, g_ema, device, img_to_patch, img_to_patch_body):
    dataset, train_sampler = _loader_dict
    _loader = data.DataLoader(
        dataset,
        batch_size=opt.batch,
        sampler=train_sampler,
        num_workers=8,
        drop_last=True,
    )
    loader = sample_data(_loader)

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_eikonal = torch.tensor(0.0, device=device)
    g_minimal_surface = torch.tensor(0.0, device=device)

    g_loss_val = 0
    loss_dict = {}

    if opt.distributed:
        g_module = generator.module
        d_module = discriminator.module
        dhead_module = discriminator_head.module
        dbody_module = discriminator_body.module
        dleg_module = discriminator_leg.module
    else:
        g_module = generator
        d_module = discriminator
        dhead_module = discriminator_head
        dbody_module = discriminator_body
        dleg_module = discriminator_leg
        
    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = torch.cuda.FloatTensor(opt.val_n_sample, opt.style_dim, device=device).normal_()

    sample_trans, sample_beta, sample_theta = _loader.dataset.sample_smpl_param(opt.val_n_sample, device)
    sample_cam_extrinsics, sample_focals = _loader.dataset.get_camera_extrinsics(opt.val_n_sample, device)
    pbar = range(opt.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=opt.start_iter, dynamic_ncols=True, smoothing=0.01)

    for idx in pbar:
        i = idx + opt.start_iter
        if i > opt.iter:
            print("Done!")
            break

        imgchannel = 3
        # Train discriminator
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        discriminator.zero_grad()
        requires_grad(discriminator_head, True)
        discriminator_head.zero_grad()
        requires_grad(discriminator_body, True)
        discriminator_body.zero_grad()
        requires_grad(discriminator_leg, True)
        discriminator_leg.zero_grad()
        
        _, real_imgs, cur_trans, cur_beta, cur_theta = next(loader)
        
        real_imgs = real_imgs.to(device)
        
        noise = torch.cuda.FloatTensor(opt.batch, opt.style_dim, device=device).normal_()
        cur_trans = cur_trans.to(device)
        cur_beta = cur_beta.to(device)
        cur_theta = cur_theta.to(device)
        cam_extrinsics, focal = _loader.dataset.get_camera_extrinsics(opt.batch, device)
        gen_imgs = []
        for j in range(0, opt.batch, opt.chunk):
            curr_noise = [n[j:j+opt.chunk] for n in noise]
            out = generator(0,noise[j:j+opt.chunk],
                            cam_extrinsics[j:j+opt.chunk],
                            focal[j:j+opt.chunk],
                            cur_beta[j:j+opt.chunk],
                            cur_theta[j:j+opt.chunk],
                            cur_trans[j:j+opt.chunk])

            gen_imgs += [out['image']]

        gen_imgs = torch.cat(gen_imgs, 0)

        R = cam_extrinsics[:,:3,:3] 
        T = cam_extrinsics[:,:3,3:]
        localjointpose = torch.bmm(g_module.renderer.posed_joints, R.permute(0,2,1)) + T.permute(0,2,1)
        joint_2d = torch.matmul(localjointpose, focal.permute(0,2,1))
        joint_2d = joint_2d[:, :, :2] / joint_2d[:, :, 2][...,None] 
        real_location = joint_2d[:, 15, :2]  
        gen_headimgs = img_to_patch(gen_imgs.permute(0, 3, 1, 2),g_module.renderer.resH,g_module.renderer.resW,real_location)
        gen_headimgs = gen_headimgs.reshape(-1,img_to_patch.ray_sampler.H_samples,img_to_patch.ray_sampler.W_samples,imgchannel).permute(0,3,1,2)
        real_headimgs = img_to_patch(real_imgs,g_module.renderer.resH,g_module.renderer.resW,real_location)
        real_headimgs = real_headimgs.reshape(-1,img_to_patch.ray_sampler.H_samples,img_to_patch.ray_sampler.W_samples,imgchannel).permute(0,3,1,2)
        
        if torch.rand(1).item()>0.5:
            real_location = joint_2d[:, 20, :2]
        else:
            real_location = joint_2d[:, 21, :2]
        
        gen_bodyimgs = img_to_patch_body(gen_imgs.permute(0, 3, 1, 2),g_module.renderer.resH,g_module.renderer.resW,real_location)
        gen_bodyimgs = gen_bodyimgs.reshape(-1,img_to_patch_body.ray_sampler.H_samples,img_to_patch_body.ray_sampler.W_samples,imgchannel).permute(0,3,1,2)
        real_bodyimgs = img_to_patch_body(real_imgs,g_module.renderer.resH,g_module.renderer.resW,real_location)
        real_bodyimgs = real_bodyimgs.reshape(-1,img_to_patch_body.ray_sampler.H_samples,img_to_patch_body.ray_sampler.W_samples,imgchannel).permute(0,3,1,2)
        
        real_location = (joint_2d[:, 7, :2] +joint_2d[:, 8, :2])/2
        real_location[:,1] -= 10        
        gen_legimgs = img_to_patch(gen_imgs.permute(0, 3, 1, 2),g_module.renderer.resH,g_module.renderer.resW,real_location)
        gen_legimgs = gen_legimgs.reshape(-1,img_to_patch.ray_sampler.H_samples,img_to_patch.ray_sampler.W_samples,imgchannel).permute(0,3,1,2)
        real_legimgs = img_to_patch(real_imgs,g_module.renderer.resH,g_module.renderer.resW,real_location)
        real_legimgs = real_legimgs.reshape(-1,img_to_patch.ray_sampler.H_samples,img_to_patch.ray_sampler.W_samples,imgchannel).permute(0,3,1,2)

        fake_pred, _ = discriminator(gen_imgs.detach().permute(0, 3, 1, 2).contiguous())
        fake_pred_head, _ = discriminator_head(gen_headimgs.detach())
        fake_pred_body, _ = discriminator_body(gen_bodyimgs.detach())
        fake_pred_leg, _ = discriminator_leg(gen_legimgs.detach())

        real_imgs.requires_grad = True
        real_pred, _ = discriminator(real_imgs)
        d_gan_loss = d_logistic_loss(real_pred, fake_pred)
        grad_penalty = d_r1_loss(real_pred, real_imgs)
        r1_loss = opt.r1 * 0.5 * grad_penalty
        
        real_headimgs.requires_grad = True
        real_pred_head, _ = discriminator_head(real_headimgs)
        d_gan_loss_head = d_logistic_loss(real_pred_head, fake_pred_head)
        grad_penalty_head = d_r1_loss(real_pred_head, real_headimgs)
        r1_loss_head = opt.r1 * 0.5 * grad_penalty_head
        
        real_bodyimgs.requires_grad = True
        real_pred_body, _ = discriminator_body(real_bodyimgs)
        d_gan_loss_body = d_logistic_loss(real_pred_body, fake_pred_body)
        grad_penalty_body = d_r1_loss(real_pred_body, real_bodyimgs)
        r1_loss_body = opt.r1 * 0.5 * grad_penalty_body
        
        real_legimgs.requires_grad = True
        real_pred_leg, _ = discriminator_leg(real_legimgs)
        d_gan_loss_leg = d_logistic_loss(real_pred_leg, fake_pred_leg)
        grad_penalty_leg = d_r1_loss(real_pred_leg, real_legimgs)
        r1_loss_leg = opt.r1 * 0.5 * grad_penalty_leg
        
        d_loss = d_gan_loss + r1_loss + d_gan_loss_head + r1_loss_head + d_gan_loss_body + r1_loss_body + d_gan_loss_leg + r1_loss_leg
        d_loss.backward()
        d_optim.step()

        loss_dict["d"] = d_gan_loss + d_gan_loss_head + d_gan_loss_body + d_gan_loss_leg
        loss_dict["r1"] = r1_loss + r1_loss_head + r1_loss_body + r1_loss_leg
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        # Train Generator
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(discriminator_head, False)
        requires_grad(discriminator_body, False)
        requires_grad(discriminator_leg, False)
        
        noise = torch.cuda.FloatTensor(opt.batch, opt.style_dim, device=device).normal_()
        _, _, cur_trans, cur_beta, cur_theta = next(loader)
        cur_trans = cur_trans.to(device)
        cur_beta = cur_beta.to(device)
        cur_theta = cur_theta.to(device)
        cam_extrinsics, focal = _loader.dataset.get_camera_extrinsics(opt.batch, device)
        for j in range(0, opt.batch, opt.chunk):
            curr_noise = [n[j:j+opt.chunk] for n in noise]
            out = generator(1,noise[j:j+opt.chunk],
                            cam_extrinsics[j:j+opt.chunk],
                            focal[j:j+opt.chunk],
                            cur_beta[j:j+opt.chunk],
                            cur_theta[j:j+opt.chunk],
                            cur_trans[j:j+opt.chunk])
                            
            fake_img = out['image']
        
            R = cam_extrinsics[:,:3,:3] 
            T = cam_extrinsics[:,:3,3:]
            localjointpose = torch.bmm(g_module.renderer.posed_joints, R.permute(0,2,1)) + T.permute(0,2,1)
            joint_2d = torch.matmul(localjointpose, focal.permute(0,2,1))
            joint_2d = joint_2d[:, :, :2] / joint_2d[:, :, 2][...,None] 
            real_location = joint_2d[:, 15, :2]  
            fake_headimgs = img_to_patch(fake_img.permute(0, 3, 1, 2),g_module.renderer.resH,g_module.renderer.resW,real_location)
            fake_headimgs = fake_headimgs.reshape(-1,img_to_patch.ray_sampler.H_samples,img_to_patch.ray_sampler.W_samples,imgchannel).permute(0,3,1,2)
            if torch.rand(1).item()>0.5:
                real_location = joint_2d[:, 20, :2]
            else:
                real_location = joint_2d[:, 21, :2]            
            fake_bodyimgs = img_to_patch_body(fake_img.permute(0, 3, 1, 2),g_module.renderer.resH,g_module.renderer.resW,real_location)
            fake_bodyimgs = fake_bodyimgs.reshape(-1,img_to_patch_body.ray_sampler.H_samples,img_to_patch_body.ray_sampler.W_samples,imgchannel).permute(0,3,1,2)
            real_location = (joint_2d[:, 7, :2] +joint_2d[:, 8, :2])/2  
            real_location[:,1] -= 10            
            fake_legimgs = img_to_patch(fake_img.permute(0, 3, 1, 2),g_module.renderer.resH,g_module.renderer.resW,real_location)
            fake_legimgs = fake_legimgs.reshape(-1,img_to_patch.ray_sampler.H_samples,img_to_patch.ray_sampler.W_samples,imgchannel).permute(0,3,1,2)
                    
            fake_pred, _ = discriminator(fake_img.permute(0, 3, 1, 2).contiguous())
            g_gan_loss = g_nonsaturating_loss(fake_pred)
            
            fake_pred_head, _ = discriminator_head(fake_headimgs)
            g_gan_loss_head = g_nonsaturating_loss(fake_pred_head)
            fake_pred_body, _ = discriminator_body(fake_bodyimgs)
            g_gan_loss_body = g_nonsaturating_loss(fake_pred_body)
            fake_pred_leg, _ = discriminator_leg(fake_legimgs)
            g_gan_loss_leg = g_nonsaturating_loss(fake_pred_leg)
            
            incdisploss = g_module.renderer.incdisploss
            incrotloss = g_module.renderer.incrotloss
            opacity_loss = g_module.renderer.opacity_loss
            postv_loss = g_module.renderer.postv_loss
            scale_loss = g_module.renderer.scale_loss

            g_loss = g_gan_loss + g_gan_loss_head + g_gan_loss_body + g_gan_loss_leg + incdisploss + scale_loss+ opacity_loss+ incrotloss+ postv_loss

            g_loss.backward()

        if 0:
            gs_out_smpl = out['gs_output_smpl']
            gs_out_cloth = out['gs_output_cloth']
            
            render_pkg_smpl = out['smpl_scene']
            render_pkg_smpl['human_viewspace_points'] = render_pkg_smpl['viewspace_points'][:gs_out_smpl['points'].shape[0]]
            render_pkg_smpl['human_viewspace_points'].grad = render_pkg_smpl['viewspace_points'].grad[:gs_out_smpl['points'].shape[0]]
            render_pkg_cloth = out['cloth_scene']
            render_pkg_cloth['human_viewspace_points'] = render_pkg_cloth['viewspace_points'][:gs_out_cloth['points'].shape[0]]
            render_pkg_cloth['human_viewspace_points'].grad = render_pkg_cloth['viewspace_points'].grad[:gs_out_cloth['points'].shape[0]]
            
            with torch.no_grad():
                desifytag = human_densification(
                    generator, guassian_opt, 
                    human_gs_out_smpl=gs_out_smpl,
                    human_gs_out_cloth=gs_out_cloth,
                    visibility_filter_smpl=render_pkg_smpl['visibility_filter'],
                    visibility_filter_cloth=render_pkg_cloth['visibility_filter'],
                    radii_smpl=render_pkg_smpl['radii'],
                    radii_cloth=render_pkg_cloth['radii'],
                    viewspace_point_tensor_smpl=render_pkg_smpl['human_viewspace_points'],
                    viewspace_point_tensor_cloth=render_pkg_cloth['human_viewspace_points'],
                    iteration=i,
                    optimizer=g_optim
                )
        else:
            desifytag = False
        g_optim.step()
        generator.zero_grad()
        loss_dict["g"] = g_gan_loss + g_gan_loss_head + g_gan_loss_body + g_gan_loss_leg
        
        loss_dict['incdisploss'] =  incdisploss
        loss_dict['incrotloss'] =  incrotloss
        loss_dict['opacityloss'] =  opacity_loss
        loss_dict['postvloss'] =  postv_loss
        loss_dict['scaleloss'] =  scale_loss
        accumulate(g_ema, g_module, accum)
    
        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        
        incdisp_loss = loss_reduced["incdisploss"].mean().item()
        
        opacityloss = loss_reduced["opacityloss"].mean().item()
        scaleloss = loss_reduced["scaleloss"].mean().item()
        incrot_loss = loss_reduced["incrotloss"].mean().item()
        postvloss = loss_reduced["postvloss"].mean().item()

        if opt.adjust_gamma:
            if opt.r1 >= opt.gamma_lb and i % 10000 == 0 and i != 0:
                opt.r1 = opt.r1 // 2

        if get_rank() == 0:
            pbar.set_description(
                (f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; incdloss: {incdisp_loss:.4f};scaleloss: {scaleloss:.4f}; opacloss: {opacityloss:.4f};incrloss: {incrot_loss:.4f}; postvloss: {postvloss:.4f};r1: {opt.r1} {r1_val:.4f}")
            )

            if i % 1000 == 0:
                with torch.no_grad():
                    # 创建保存目录
                    sample_dir = os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'volume_renderer', 'samples')
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # 使用 g_ema 生成8张样本
                    val_n_sample = 8
                    sample_z = torch.cuda.FloatTensor(val_n_sample, opt.style_dim, device=device).normal_()
                    sample_trans, sample_beta, sample_theta = _loader.dataset.sample_smpl_param(val_n_sample, device)
                    sample_cam_extrinsics, sample_focals = _loader.dataset.get_camera_extrinsics(val_n_sample, device)
                    
                    out = g_ema(0, sample_z,
                                sample_cam_extrinsics,
                                sample_focals,
                                sample_beta,
                                sample_theta,
                                sample_trans)
                    
                    curr_sample = out['image']
                    # 生成样本 (8张)
                    samples = curr_sample.cpu().permute(0, 3, 1, 2)[:, :3, ...]
                    # 拼接当前训练的假图
                    samples = torch.cat([samples, fake_img.permute(0, 3, 1, 2)[:, :3, ...].cpu()], 0)
                    # 拼接真实图片
                    samples = torch.cat([samples, real_imgs.cpu()[:, :3, ...]], 0)
                    # 保存: 8张生成样本 + batch张假图 + batch张真图，每行8张
                    utils.save_image(samples,
                        os.path.join(sample_dir, f"{str(i).zfill(7)}.png"),
                        nrow=8,
                        normalize=True, range=(-1, 1))

            if i % 5000 == 0 or (i < 10000 and i % 5000 == 0):
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "dhead": dhead_module.state_dict(),
                        "dbody": dbody_module.state_dict(),
                        "dleg": dleg_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                    },
                    os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'volume_renderer', f"models_{str(i).zfill(7)}.pt")
                )
                print('Successfully saved checkpoint for iteration {}.'.format(i))

if __name__ == "__main__":
    device = "cuda"
    opt = BaseOptions().parse()
    opt.model.freeze_renderer = False
    opt.training.camera = opt.camera
    opt.training.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.training.style_dim = opt.model.style_dim
    opt.training.with_sdf = not opt.rendering.no_sdf
    if opt.training.with_sdf and opt.training.min_surf_lambda > 0:
        opt.rendering.return_sdf = True
    opt.rendering.no_features_output = True
    opt.training.sphere_init = False

    torch.autograd.set_detect_anomaly(True)

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.training.distributed = n_gpu > 1

    if opt.training.distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # create checkpoints directories
    os.makedirs(os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'volume_renderer'), exist_ok=True)
    os.makedirs(os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'volume_renderer', 'samples'), exist_ok=True)

    discriminator = VolumeRenderDiscriminator(opt.model.renderer_spatial_output_dim).to(device)
    discriminator_head = VolumeRenderDiscriminator([128,128]).to(device)
    discriminator_body = VolumeRenderDiscriminator([128,128]).to(device)
    discriminator_leg = VolumeRenderDiscriminator([128,128]).to(device)
    generator = Generator().to(device)
    g_ema = Generator().to(device)

    g_ema.eval()
    accumulate(g_ema, generator, 0)
    
    g_optim = optim.Adam(generator.parameters(), lr=opt.training.glr, betas=(0, 0.9))
    params = list(discriminator.parameters()) + list(discriminator_head.parameters()) + list(discriminator_body.parameters()) + list(discriminator_leg.parameters())
    d_optim = optim.Adam(params, lr=opt.training.dlr, betas=(0, 0.9))
    
    opt.training.start_iter = 0

    if opt.experiment.continue_training and opt.experiment.ckpt is not None:
        ckpt_path = os.path.join(opt.training.checkpoints_dir,
                                 opt.experiment.expname,
                                 'volume_renderer/models_{}.pt'.format(opt.experiment.ckpt.zfill(7)))
        if get_rank() == 0:
            print("load model:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        try:
            opt.training.start_iter = int(opt.experiment.ckpt) + 1
        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"], strict=True)
        discriminator.load_state_dict(ckpt["d"], strict=True)
        discriminator_head.load_state_dict(ckpt["dhead"], strict=True)
        discriminator_body.load_state_dict(ckpt["dbody"], strict=True)
        discriminator_leg.load_state_dict(ckpt["dleg"], strict=True)
        g_ema.load_state_dict(ckpt["g_ema"])
        if "g_optim" in ckpt.keys():
            g_optim.load_state_dict(ckpt["g_optim"])
            d_optim.load_state_dict(ckpt["d_optim"])

    if opt.training.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            broadcast_buffers=True,
            find_unused_parameters=True,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        discriminator_head= nn.parallel.DistributedDataParallel(
            discriminator_head,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        
        discriminator_body= nn.parallel.DistributedDataParallel(
            discriminator_body,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        
        discriminator_leg= nn.parallel.DistributedDataParallel(
            discriminator_leg,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])

    dataset = DeepFashionDataset(opt.dataset.dataset_path, transform, opt.model.size,
                                opt.model.renderer_spatial_output_dim,
                                os.path.join(opt.dataset.dataset_path, 'train_list.txt'),
                                white_bg=opt.rendering.white_bg,
                                random_flip=opt.dataset.random_flip,
                                gaussian_weighted_sampler=opt.dataset.gaussian_weighted_sampler,
                                sampler_std=opt.dataset.sampler_std)

    if 0:
        sampler = WeightedRandomSampler(dataset.weights, len(dataset.weights))
        if opt.training.distributed:
            train_sampler = DistributedSamplerWrapper(sampler)
        else:
            train_sampler = sampler
    else:
        train_sampler = data_sampler(dataset, shuffle=True, distributed=opt.training.distributed)

    opt.training.dataset_name = opt.dataset.dataset_path.lower()

    patch_sampler = FlexGridRaySampler(128, 128)
    img_to_patch = RealImgToPatch(patch_sampler)

    patch_sampler_body = FlexGridRaySampler(128,128)
    img_to_patch_body = RealImgToPatch(patch_sampler_body)
    
    # save options
    opt_path = os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'volume_renderer', f"opt.yaml")
    with open(opt_path,'w') as f:
        yaml.safe_dump(opt, f)

    # 直接运行训练
    train(opt.training, opt.experiment, opt.guassian, (dataset, train_sampler),  generator, discriminator, discriminator_head,discriminator_body,discriminator_leg,
          g_optim, d_optim, g_ema, device,  img_to_patch, img_to_patch_body)
