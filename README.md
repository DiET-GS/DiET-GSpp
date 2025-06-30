<p align="center">
  <h1 align="center">DiET-GS++ </h1>
  <p align="center">
  The repository for DiET-GS++. DiET-GS++ refines the edge details and fine-grained details of <a href="https://github.com/DiET-GS/DiET-GS">DiET-GS</a> by fully leveraging the pretrained diffusion prior.
  </p>
  <div align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
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

pip install threestudio/systems/gaussian_splatting/submodules/diff-gaussian-rasterization
pip install threestudio/systems/gaussian_splatting/submodules/diff-gaussian-rasterization_contrastive_f
pip install threestudio/systems/gaussian_splatting/submodules/simple-knn
```