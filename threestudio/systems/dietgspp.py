import os
from dataclasses import dataclass, field
from tqdm import tqdm
import copy
import pathlib
import random
import pickle

import torch
import torch.nn.functional as F
import torch.nn as nn

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.typing import *
from threestudio.utils.criterion import PSNR, NIQE

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from threestudio.utils.sr_esrnet import SFTNet, default_init_weights

import lpips
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

# Gaussian Splatting
from random import randint
from argparse import ArgumentParser, Namespace
from threestudio.systems.gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams
from threestudio.systems.gaussian_splatting.scene import Scene, GaussianModel, DiffGaussianModel
from threestudio.systems.gaussian_splatting.gaussian_renderer import render, render_latent_feature
from threestudio.systems.gaussian_splatting.utils.wavelet_utils import wavelet_decomposition
from threestudio.utils.loss_utils import l1_loss
from threestudio.utils.event_utils import *

import pyiqa

from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray

@threestudio.register("dietgspp")
class DietGSPP(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        vis_interval: int = 200
        
        start_sr_step: int = 20000
        num_sr_steps: int = 10000
        num_sync_steps: int = 10000
        sr_batch_size: int = 1
        patch_size: int = 128
 
    cfg: Config

    def configure(self):
        torch.set_float32_matmul_precision('medium') # test

        super().configure()

        # load prompt processor
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
        self.prompt_processor.configure_text_encoder()
        self.prompt_processor.destroy_text_encoder()
        self.prompt_processor_output = self.prompt_processor()
        print(f"VRAM usage after removing text_encoder: {torch.cuda.memory_allocated(0) / (1024 ** 3)}GB")

        # load guidance
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance.vae.enable_tiling()

        for p in self.guidance.parameters():
            p.requires_grad=False
        print(f"Guidance weights frozen.")

        self.stage_interval = self.cfg.num_sr_steps+self.cfg.num_sync_steps

        # lpips loss
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').to(self.device)
        
        # Gaussian Splatting
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        
        args = parser.parse_args([])
        args.load_point = self.cfg.gaussian.load_point
        args.source_path = self.cfg.gaussian.dataroot
        args.model_path = self.cfg.gaussian.model_path

        self.pipe = pp.extract(args)
        dataset = lp.extract(args)
        
        self.gaussians = GaussianModel(dataset.sh_degree)
        self.scene = Scene(dataset, self.gaussians)
        
        self.gaussians._xyz.requires_grad = False
        self.gaussians._features_dc.requires_grad = False
        self.gaussians._features_rest.requires_grad = False
        self.gaussians._opacity.requires_grad = False
        self.gaussians._scaling.requires_grad = False
        self.gaussians._rotation.requires_grad = False
        
        point_num = self.gaussians._xyz.shape[0]
        
        self.scene_name = os.path.basename(self.cfg.gaussian.dataroot)
        
        class Param(nn.Module):
            def __init__(self, point_num):
                super(Param, self).__init__()
                    
                self.cache = []
                    
                latents = torch.zeros((point_num, 4)).float().cuda()
                self.latents = nn.Parameter(latents.contiguous().requires_grad_(True))
                    
            def forward(self):
                return self.latents

        self.param = Param(point_num)
        
    def training_step(self, batch, batch_idx):
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if self.global_step == 0:
            self.image_latents = []
            self.org_image = []
            
            viewpoint_stack = self.scene.getTrainCameras().copy()
            for c in viewpoint_stack:            
                with torch.no_grad():
                    render_pkg = render(c.moving_poses[4], self.gaussians, self.pipe, background)
                    img = render_pkg['render'].permute(1, 2, 0).unsqueeze(0)
                    img_remap= F.interpolate(img.movedim(-1,1), scale_factor=4, mode='bicubic', align_corners=False).movedim(1,-1)        
                    img_remap = (img_remap * 2.0) - 1.0
                    
                    image_id = int(c.image_name)
          
                    comp_latent = self.guidance.encode_images(img_remap.movedim(-1,1)).movedim(1,-1)
            
                self.image_latents.append(comp_latent)
                self.org_image.append(img)
            
            self.image_latents = torch.cat(self.image_latents, dim=0)
            self.org_image = torch.cat(self.org_image, dim=0)
            
        stage_step = self.global_step

        viewpoint_stack = self.scene.getTrainCameras().copy()
    
        N = len(viewpoint_stack)
        _, bh, bw, _ = self.org_image.shape
        B = 4
        
        image_latents = []
        latents_h = []
        org_image = []
        
        random_idx = np.random.permutation(N)[:B]
        
        for i in random_idx:
            image_latents.append(self.image_latents[i].unsqueeze(0))
            org_image.append(self.org_image[i].unsqueeze(0))
            
            render_feature_pkg = render_latent_feature(viewpoint_stack[i].moving_poses[4], self.gaussians, self.pipe, background, self.param()) # FIXME:
            latents_h.append(render_feature_pkg['render'].permute(1, 2, 0).unsqueeze(0))
        
        image_latents = torch.cat(image_latents, dim=0)
        latents_h = torch.cat(latents_h, dim=0)
        org_image = torch.cat(org_image, dim=0)
        
        if stage_step % (N//B) == 0: # update by epoch 
            self.guidance.timestep_annealing(stage_step)
        
        # 2. let latents = self.render_latents + self.param.latents
        latent = image_latents + latents_h
        lr_mask = None

        # sample patch from each image
        latent_crop, org_image_crop, _, crop_box_lst = self.sample_patch(latent=latent,
                                                image=org_image,
                                                mask=lr_mask,
                                                patch_size=self.cfg.patch_size)
        
        # 3. run guidance to get gradient and loss
        latents_noisy, pred_latents_noisy = self.guidance(latent_crop, org_image_crop, self.prompt_processor_output) #[NCHW]
            
        latents_noisy = latents_noisy.movedim(1,-1)
        pred_latents_noisy = pred_latents_noisy.movedim(1,-1)

        # logging
        self.log('t', self.guidance.last_timestep.to(torch.float), prog_bar=True)
        self.log('cfg', self.guidance.cfg.guidance_scale, prog_bar=True)
        lightning_optimizer = self.optimizers()  # self = your model
        for param_group in lightning_optimizer.optimizer.param_groups:
            lr = param_group['lr']
        self.log('lr', lr, prog_bar=True)

        loss = 0.0
        loss_rsd = F.l1_loss(latents_noisy, pred_latents_noisy, reduction="mean") * B
        loss_rsd *= self.C(self.cfg.loss.lambda_rsd)
        loss += loss_rsd
        self.log('train/loss_rsd', loss_rsd.item(), prog_bar=True)

        if stage_step % self.cfg.vis_interval == 0:
            test_viewpoint_stack = self.scene.getTestCameras().copy()
            
            with torch.no_grad():
                gs_image = []
                for c in test_viewpoint_stack:
                    render_pkg = render(c, self.gaussians, self.pipe, background)
                    gs_image.append(render_pkg['render'].permute(1, 2, 0).unsqueeze(0))
                gs_image = torch.cat(gs_image, dim=0)
            
                self.param.cache.append(self.param().data.clone())
            
                finals = gs_image

                for i in range(len(self.param.cache)):
                    latents_h = [] 
                    for c in test_viewpoint_stack:
                        render_feature_pkg = render_latent_feature(c, self.gaussians, self.pipe, background, self.param.cache[i])
                        latents_h.append(render_feature_pkg['render'].permute(1, 2, 0).unsqueeze(0))
                    latents_h = torch.cat(latents_h, dim=0)
                    
                    images = []
                    for j in tqdm(range(len(gs_image))): 
                        finals_remap = F.interpolate(finals[j].unsqueeze(0).movedim(-1,1), scale_factor=4, mode='bicubic', align_corners=False).movedim(1,-1)        
                        finals_remap = (finals_remap * 2.0) - 1.0           
                        image_latents = self.guidance.encode_images(finals_remap.movedim(-1,1)).movedim(1,-1)
                    
                        latent = image_latents + latents_h[j].unsqueeze(0)
                        image = self.guidance.decode_latents(latent.movedim(-1,1)).movedim(1,-1) # NHWC
                
                        image = F.interpolate(image.movedim(-1,1), scale_factor=0.25, mode='bicubic', align_corners=False).movedim(1,-1)         
                        image = torch.clamp(image, 0.0, 1.0)

                        images.append(image)

                    finals = []
                    for j in tqdm(range(len(images))):
                        i_h, i_l = wavelet_decomposition(images[j][0].permute(2, 0, 1))
                        r_h, r_l = wavelet_decomposition(gs_image[j].permute(2, 0, 1))
                
                        final = (i_h + r_l)

                        if self.scene_name == "blurfigures":
                            with torch.no_grad():
                                ssim_map, ssim_full = ssim(
                                    rgb2gray(final.permute(1, 2, 0).cpu().numpy()), 
                                    rgb2gray(gs_image[j].cpu().numpy()), 
                                    full=True, 
                                    data_range=1.0
                                )
                            idx = torch.from_numpy(ssim_full < 0.95).unsqueeze(0).repeat(3, 1, 1)
                            final[idx] = gs_image[j].permute(2, 0, 1)[idx]

                        image_filename=f"sr_train/it{self.global_step}-{i}-{j}.png"
                        plt.imsave(self.get_save_path(image_filename), torch.clamp(final, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()) # FIXME:
                
                        finals.append(final)
                    finals = torch.stack(finals, dim=0).movedim(1,-1)
                
                del self.param.cache[-1]

            # torch.save(self.param.state_dict(), os.path.join("checkpoint/blurtanabata", str(stage_step) + ".pt"))

        return {
            'loss': loss
        }
    
    def evaluate(self, batch, batch_idx, image_filename, stage):
        return 0., 0.


    def validation_step(self, batch, batch_idx):
        # sample random index
        idx = random.randint(0,len(self.dataset.poses)-1)  

        psnr, niqe = self.evaluate(batch, idx, image_filename=f"val/it{self.global_step}-{batch_idx}.png", stage='val')
        self.validation_step_psnr.append(psnr)
        self.validation_step_niqe.append(niqe)


    def test_step(self, batch, batch_idx): 
        psnr, niqe, lpips3, lpips15 = self.evaluate(batch, batch_idx, image_filename=f"test/it{self.global_step}-{batch_idx}.png", stage='test')
        self.test_step_lpips3.append(lpips3)
        self.test_step_lpips15.append(lpips15)      
        self.log('test/lpips3', lpips3, prog_bar=True, rank_zero_only=True)         
        self.log('test/lpips15', lpips15, prog_bar=True, rank_zero_only=True)        

        self.test_step_psnr.append(psnr)
        self.test_step_niqe.append(niqe)
        self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)         
        self.log('test/niqe', niqe, prog_bar=True, rank_zero_only=True)     


    def on_validation_epoch_end(self):
        psnr = torch.stack(self.validation_step_psnr)
        psnr = torch.mean(psnr)
        self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)      
        self.validation_step_psnr.clear()  # free memory

        niqe = torch.stack(self.validation_step_niqe)
        niqe = torch.mean(niqe)
        self.log('val/niqe', niqe, prog_bar=True, rank_zero_only=True)      
        self.validation_step_niqe.clear()  # free memory


    def on_test_epoch_end(self):
        psnr = torch.stack(self.test_step_psnr)
        psnr = torch.mean(psnr)
        self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    
        self.test_step_psnr.clear()  # free memory

        niqe = torch.stack(self.test_step_niqe)
        niqe = torch.mean(niqe)
        self.log('test/niqe', niqe, prog_bar=True, rank_zero_only=True)    
        self.test_step_niqe.clear()  # free memory

        lpips3 = torch.stack(self.test_step_lpips3)
        lpips3 = torch.mean(lpips3)
        self.log('test/lpips3', lpips3, prog_bar=True, rank_zero_only=True)    
        self.test_step_lpips3.clear()  # free memory

        lpips15 = torch.stack(self.test_step_lpips15)
        lpips15 = torch.mean(lpips15)
        self.log('test/lpips15', lpips15, prog_bar=True, rank_zero_only=True)    
        self.test_step_lpips15.clear()  # free memory

        self.save_img_sequence(
            f"it{self.global_step}",
            f"test/",
            '(\d+)\.png',
            save_format='mp4',
            fps=30
        )

    def sample_patch(self, latent, image, mask, patch_size):
        
        B,H,W,_ = latent.shape

        ph=pw=patch_size

        # get masked crop coords
        coord_y, coord_x = torch.meshgrid(torch.arange(H,device=self.device), torch.arange(W,device=self.device), indexing='ij')
        coord_x = coord_x.flatten()
        coord_y = coord_y.flatten()
        x_mask = torch.all(torch.stack([coord_x>(pw//2), coord_x<(W-pw//2)]),dim=0).unsqueeze(0).expand(B,-1) #B,HW
        y_mask = torch.all(torch.stack([coord_y>(ph//2), coord_y<(H-ph//2)]),dim=0).unsqueeze(0).expand(B,-1) #B,HW

        if mask is not None:
            fg_mask = mask.reshape(B,-1) # B,HW
            comp_masks = torch.all(torch.stack([x_mask,y_mask,fg_mask]),dim=0)
            fg_masks = fg_mask.view(B,H,W,1)
        else:
            comp_masks = torch.all(torch.stack([x_mask,y_mask]),dim=0)

        rays = image.view(B,H,W,3)
        latent = latent.view(B,H,W,4)
    
        patch_rays_list = []
        patch_latent_list = []
        patch_mask_list = []

        crop_box_lst = []
        for i, comp_mask in enumerate(comp_masks):
            valid_coord_x = coord_x[comp_mask]
            valid_coord_y = coord_y[comp_mask]
            assert len(valid_coord_x) == len(valid_coord_y)
            sample_idx = random.randint(0,len(valid_coord_x)-1)
            h_sample = valid_coord_y[sample_idx]
            w_sample = valid_coord_x[sample_idx]

            # edge sampling for LLFF dataset
            if not self.dataset.apply_mask:
                h_sample = random.randint(0,H)
                w_sample = random.randint(0,W)
                if h_sample < (ph//2):
                    h_sample = ph//2
                if w_sample < (pw//2):
                    w_sample = pw//2
                if h_sample > (H-ph//2):
                    h_sample = (H-ph//2)
                if w_sample > (W-pw//2):
                    w_sample = (W-pw//2)

            center = (h_sample,w_sample)
            crop_size = (ph,pw)
            crop_box = (int(center[1]-crop_size[1]/2), int(center[0]-crop_size[0]/2), int(center[1]+crop_size[1]/2), int(center[0]+crop_size[0]/2)) #left,up,right,down
        
            patch_rays = rays[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,3) #(128*128,3)
            patch_latent = latent[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,4) #(128*128,1)
   

            patch_rays_list.append(patch_rays)
            patch_latent_list.append(patch_latent)

            if self.dataset.apply_mask:
                patch_mask = fg_masks[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,1) #(128*128,1)
                patch_mask_list.append(patch_mask)

            crop_box_lst.append(crop_box)
        
        patch_rays = torch.stack(patch_rays_list)
        patch_latent = torch.stack(patch_latent_list)

        patch_rays = patch_rays.reshape(B,ph,pw,3)
        patch_latent = patch_latent.reshape(B,ph,pw,4)

        if self.dataset.apply_mask:
            patch_mask = torch.stack(patch_mask_list)
            patch_mask = patch_mask.reshape(B,ph,pw,1)
        else:
            patch_mask = None

        return patch_latent, patch_rays, patch_mask, crop_box_lst


    def sample_patch_broad(self, latent, image, mask, patch_size):
        
        B,H,W,_ = latent.shape

        ph=pw=patch_size

        # get masked crop coords
        coord_y, coord_x = torch.meshgrid(torch.arange(H,device=self.device), torch.arange(W,device=self.device), indexing='ij')
        coord_x = coord_x.flatten()
        coord_y = coord_y.flatten()
        x_mask = torch.all(torch.stack([coord_x>(pw//2), coord_x<(W-pw//2)]),dim=0).unsqueeze(0).expand(B,-1) #B,HW
        y_mask = torch.all(torch.stack([coord_y>(ph//2), coord_y<(H-ph//2)]),dim=0).unsqueeze(0).expand(B,-1) #B,HW
        
        if mask is not None:
            fg_mask = mask.reshape(B,-1) # B,HW
            comp_masks = torch.all(torch.stack([x_mask,y_mask,fg_mask]),dim=0)
            fg_masks = fg_mask.view(B,H,W,1)
        else:
            comp_masks = torch.all(torch.stack([x_mask,y_mask]),dim=0)
            
        valid_coord_x = coord_x[comp_masks[0]]
        valid_coord_y = coord_y[comp_masks[0]]
        assert len(valid_coord_x) == len(valid_coord_y)
        sample_idx = random.randint(0,len(valid_coord_x)-1)
        h_sample = valid_coord_y[sample_idx]
        w_sample = valid_coord_x[sample_idx]

        # edge sampling for LLFF dataset
        if not self.dataset.apply_mask:
            h_sample = random.randint(0,H)
            w_sample = random.randint(0,W)
            if h_sample < (ph//2):
                h_sample = ph//2
            if w_sample < (pw//2):
                w_sample = pw//2
            if h_sample > (H-ph//2):
                h_sample = (H-ph//2)
            if w_sample > (W-pw//2):
                w_sample = (W-pw//2)

        center = (h_sample,w_sample)
        crop_size = (ph,pw)
        crop_box = (int(center[1]-crop_size[1]/2), int(center[0]-crop_size[0]/2), int(center[1]+crop_size[1]/2), int(center[0]+crop_size[0]/2))
            
        rays = image.view(B,H,W,3)
        latent = latent.view(B,H,W,4)
    
        patch_rays_list = []
        patch_latent_list = []
        patch_mask_list = []

        for i in range(len(comp_masks)):
            patch_rays = rays[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,3) #(128*128,3)
            patch_latent = latent[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,4) #(128*128,1)
   

            patch_rays_list.append(patch_rays)
            patch_latent_list.append(patch_latent)

            if self.dataset.apply_mask:
                patch_mask = fg_masks[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,1) #(128*128,1)
                patch_mask_list.append(patch_mask)
        
        patch_rays = torch.stack(patch_rays_list)
        patch_latent = torch.stack(patch_latent_list)

        patch_rays = patch_rays.reshape(B,ph,pw,3)
        patch_latent = patch_latent.reshape(B,ph,pw,4)

        if self.dataset.apply_mask:
            patch_mask = torch.stack(patch_mask_list)
            patch_mask = patch_mask.reshape(B,ph,pw,1)
        else:
            patch_mask = None

        return patch_latent, patch_rays, patch_mask, crop_box

