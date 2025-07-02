<p align="center">
  <h1 align="center">DiET-GS++ </h1>
  <p align="center">
  The repository for DiET-GS++. DiET-GS++ refines the edge details and fine-grained details of <a href="https://github.com/DiET-GS/DiET-GS">DiET-GS</a> by fully leveraging the pretrained diffusion prior.
  </p>
  <div align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
  </div>
</p>

## Installation
### Dependencies :memo:
The main dependencies of the project are the following:
```yaml
python: 3.9
cuda: 11.3
```
You can set up a conda environment as follows:
```
conda create name -n dietgspp python=3.9
conda activate dietgspp

pip install torch==2.0.1 torchvision==0.15.2

pip install -r requirements.txt

pip install nvdiffrast # Use the absolute path if there is any error
pip install threestudio/systems/gaussian_splatting/submodules/diff-gaussian-rasterization
pip install threestudio/systems/gaussian_splatting/submodules/diff-gaussian-rasterization_contrastive_f
pip install threestudio/systems/gaussian_splatting/submodules/simple-knn
```

## Per-scene Optimization of DiET-GS++ (Stage 2)

Before you optimize the DiET-GS++, you need to first train the DiET-GS in <a href="https://github.com/DiET-GS/DiET-GS">this repository</a>. Or, we also provide the pretrained weights of DiET-GS in `pretrained` folder so that you can directly use them for Stage 2 optimization.

Then, enter the path of scene data and weight of pretrained DiET-GS in `configs/dietgspp.yaml` file. 

Below snippet is the example where our official DiET-GS weights are used (see `model_path`).
```
# In configs/dietgspp.yaml file,
...

system_type: "dietgspp"
system:
  gaussian:
    dataroot: "/Username/DiET-GS/data/ev-deblurnerf_cdavis/blurbatteries" # <- HERE
    model_path: "pretrained/ev-deblurnerf_cdavis/blurbatteries" # <- Here
    load_point: true

...
```

Run the script below:
```
python launch.py --config configs/dietgspp.yaml --train
```
The learned parameters from Stage 2 will be saved to the `checkpoint/${scene_name}` folder every 200 iterations. Among the stored checkpoints, you may select the best one based on the MUSIQ and CLIP-IQA scores.

## Render DiET-GS++

After the optimization of DiET-GS++, you can render the novel views with pretrained DiET-GS++. Enter the path of scene data, pretrained weight of DiET-GS, and pretrained latent parameters of DiET-GS++ in `configs/dietgspp_render.yaml`. 

Below snippet is the example where our official DiET-GS weights are used (see `model_path`).
```
# In configs/dietgspp_render.yaml file,
...

  gaussian:
    dataroot: "/Username/DiET-GS/data/ev-deblurnerf_cdavis/blurbatteries" # <- HERE
    model_path: pretrained/ev-deblurnerf_cdavis/blurbatteries # <- HERE
    latent_path: checkpoint/blurbatteries/2400.pt # <- HERE
    load_point: True

...
```

To render the novel views and evaluate their quality, run the below script:
```
python render.py
```

If the code is successfully run, you can check the stored outputs in `pretrained/${scene_name}/test/renders` folder. Also, you may check the quantitative results in the terminal:
```
SSIM : 0.9124267578125
PSNR : 33.524009704589844
LPIPS: 0.043877677619457246
MUSIQ: 49.69626998901367
CLIP-IQA: 0.2598043382167816
```

## Citation
If you find our code or paper useful, please cite
```bibtex
@inproceedings{lee2025diet,
  title={DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting},
  author={Lee, Seungjun and Lee, Gim Hee},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={21739--21749},
  year={2025}
}
```
