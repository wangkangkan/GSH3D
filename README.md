# GSH3D: Efficient 3D Gaussian Human Generation from 2D Image Collections

Official implementation for GSH3D, a 3D Gaussian Splatting based human generation model trained on 2D image collections.

Questions and discussions are welcome.

## Installation

### Environment Setup

**Tested on Ubuntu 20.04. CUDA 11.8 is required if you follow the steps below.**

#### Conda Environment

```bash
# Enter project directory
cd GSH3D

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate gsh3d
```

#### Install Gaussian Splatting Dependencies

```bash
# Install Gaussian rasterizer
cd submodules/diff-gaussian-rasterization/
pip install -e .

## Data Preparation

### SMPL Model

1. Download SMPL model from [SMPL official website](https://smpl.is.tue.mpg.de/)
2. Place the model file in the following structure:

```text
smpl_models/
└── smpl/
    └── SMPL_NEUTRAL.pkl
```

### DeepFashion Dataset

Prepare the DeepFashion dataset with the following structure:

```text
DeepFashion/
├── images/           # Image files
├── segm/             # Segmentation masks
├── smpl.pkl          # SMPL parameters
└── train_list.txt    # Training image list
```

## Run the Code

### Training

```bash
python train_deepfashion_3dgs_multidisc_gema.py \
    --batch 8 \
    --chunk 8 \
    --expname train_deepfashion \
    --dataset_path ../autodl-tmp/DeepFashion \
    --depth 5 \
    --width 128 \
    --style_dim 512 \
    --renderer_spatial_output_dim 512 256 \
    --input_ch_views 3 \
    --white_bg \
    --r1 15 \
    --voxhuman_name eva3d_deepfashion \
    --random_flip \
    --eikonal_lambda 0.5 \
    --small_aug \
    --iter 1000000 \
    --adjust_gamma \
    --gamma_lb 20 \
    --min_surf_lambda 1.5 \
    --deltasdf \
    --gaussian_weighted_sampler \
    --sampler_std 15 \
    --N_samples 48
```

### Inference

Generate multi-view images from trained model:

```bash
# Generate multi-view static images (default 3 views)
python generation_demo.py \
    --expname xxxx \
    --ckpt xxxx \
    --identities 16 \
    --style_dim 512 \
    --renderer_spatial_output_dim 512 256

# Generate rotation videos
python generation_demo.py \
    --expname xxxx \
    --ckpt xxxx \
    --identities 8 \
    --style_dim 512 \
    --renderer_spatial_output_dim 512 256 \
    --render_video
```

## Citation


## Acknowledgements

