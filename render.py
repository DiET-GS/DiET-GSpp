#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from os import makedirs
from PIL import Image

import torchvision
import torchvision.transforms.functional as tf

import threestudio
from threestudio.systems.gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from threestudio.systems.gaussian_splatting.gaussian_renderer import GaussianModel
from threestudio.systems.gaussian_splatting.gaussian_renderer import render, render_latent_feature
from threestudio.systems.gaussian_splatting.scene import Scene

from threestudio.systems.gaussian_splatting.utils.loss_utils import ssim
from threestudio.systems.gaussian_splatting.lpipsPyTorch import lpips
from threestudio.systems.gaussian_splatting.utils.image_utils import psnr
from threestudio.systems.gaussian_splatting.utils.config import load_config
from threestudio.systems.gaussian_splatting.utils.wavelet_utils import wavelet_decomposition

from argparse import ArgumentParser
import pyiqa
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity
from skimage.color import rgb2gray

cfg = load_config("configs/dietgspp_render.yaml")

prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(cfg.system.prompt_processor)
prompt_processor.configure_text_encoder()
prompt_processor.destroy_text_encoder()
prompt_processor_output = prompt_processor()

guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
guidance.vae.enable_tiling()
for p in guidance.parameters():
    p.requires_grad=False

  
def readImages(path):
    render = Image.open(path)
    render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
    return render

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    scene_name = os.path.basename(cfg.system.gaussian.dataroot)
    render_path = os.path.join(os.path.dirname(model_path), name, "renders")
    gts_path = os.path.join(os.path.dirname(model_path), name, "gt")
    
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    clipiqa_test = 0.0
    musiq_test = 0.0
    
    clipiqa = pyiqa.create_metric('clipiqa').cuda()
    musiq = pyiqa.create_metric('musiq').cuda()
    
    params = torch.load(cfg.system.gaussian.latent_path)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        with torch.no_grad():
            rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], 0.0, 1.0)
            gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)
            
            dietgs_image = rendering.clone()
            
            rendering = F.interpolate(rendering.unsqueeze(0), scale_factor=4, mode='bicubic', align_corners=False)
            rendering = (rendering * 2.0) - 1.0
            image_latent = guidance.encode_images(rendering).movedim(1,-1)
            
            render_feature_pkg = render_latent_feature(view, gaussians, pipeline, background, params['latents'].data)
            latent_h = render_feature_pkg['render'].permute(1, 2, 0).unsqueeze(0)
            
            latent = image_latent + latent_h
            
            dietgspp_image = guidance.decode_latents(latent.movedim(-1,1))
            dietgspp_image = F.interpolate(dietgspp_image, scale_factor=0.25, mode='bicubic', align_corners=False).movedim(1,-1)         
            dietgspp_image = torch.clamp(dietgspp_image, 0.0, 1.0)

            i_h, i_l = wavelet_decomposition(dietgspp_image[0].permute(2, 0, 1))
            r_h, r_l = wavelet_decomposition(dietgs_image)
            
            final = (i_h + r_l)
            
            if scene_name == "blurfigures":
                ssim_map, ssim_full = structural_similarity(
                    rgb2gray(final.permute(1, 2, 0).cpu().numpy()), 
                    rgb2gray(dietgs_image.permute(1, 2, 0).cpu().numpy()), 
                    full=True, 
                    data_range=1.0
                )
                ssim_mask = torch.from_numpy(ssim_full < 0.95).unsqueeze(0).repeat(3, 1, 1)
                final[ssim_mask] = dietgs_image[ssim_mask]

            final = torch.clamp(final, 0.0, 1.0)
            
        plt.imsave(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"), final.permute(1, 2, 0).detach().cpu().numpy())
        plt.imsave(os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"), gt.permute(1, 2, 0).detach().cpu().numpy())
            
    for f in sorted(os.listdir(render_path)):
        render_image_path = os.path.join(render_path, f)
        gt_image_path = os.path.join(gts_path, f)
        
        final = readImages(render_image_path)
        gt = readImages(gt_image_path)
        
        psnr_test += psnr(final[0], gt[0]).mean().double()
        ssim_test += ssim(final[0], gt[0]).mean().double()
        lpips_test += lpips(final[0], gt[0]).mean().double()
        clipiqa_test += clipiqa(final)
        musiq_test += musiq(final)

    lpips_test /= len(views)
    psnr_test /= len(views)
    ssim_test /= len(views)
    clipiqa_test /= len(views)
    musiq_test /= len(views)

    print("  SSIM : {}".format(ssim_test))
    print("  PSNR : {}".format(psnr_test))
    print("  LPIPS: {}".format(lpips_test))
    print("  MUSIQ: {}".format(musiq_test.item()))
    print("  CLIP-IQA: {}".format(clipiqa_test.item()))
    print("")


def render_sets(dataset : ModelParams, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args([])
    args.load_point = cfg.system.gaussian.load_point
    args.source_path = cfg.system.gaussian.dataroot
    args.model_path = cfg.system.gaussian.model_path
    
    render_sets(lp.extract(args), pp.extract(args), True, False)