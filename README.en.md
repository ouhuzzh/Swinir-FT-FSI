# SwinIR

An image super-resolution reconstruction tool based on Swin Transformer.

## Project Overview

SwinIR is an image super-resolution reconstruction network based on the Swin Transformer, leveraging an advanced Transformer architecture to achieve high-quality image restoration. This project is the official implementation of [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257), built upon the BasicSR deep learning framework.

## Key Features

- **Swin Transformer Architecture**: Employs Swin Transformer modules for image reconstruction, offering superior global modeling capabilities compared to traditional CNN methods.
- **Multiple Restoration Tasks**: Supports classical image super-resolution, lightweight super-resolution, and real-world image super-resolution.
- **Modular Design**: Flexible network configuration supporting varying depths of RSTB modules and window sizes.
- **High-Performance CUDA Operators**: Includes efficient CUDA operators for deformable convolutions, fused activation functions, and up/down-sampling.
- **Complete Training Pipeline**: Supports mainstream training datasets such as DIV2K, REDS, and Vimeo90K.

## System Requirements

- Python 3.7+
- PyTorch 1.7+
- CUDA 10.0+ (GPU support)
- NVIDIA GPU (recommended VRAM >= 8GB)

## Installation

```bash
# Clone the repository
git clone https://gitee.com/flowstate/SwinIR.git
cd SwinIR

# Install dependencies
pip install -r requirements.txt

# Install BasicSR
pip install -e .
```

## Quick Start

### Inference Usage

```python
import cv2
import torch
from inference.inference_swinir import define_model

# Configuration parameters
args = type('Args', (), {
    'task': '001_classical',
    'scale': 4,
    'window_size': 8,
    'model_path': 'experiments/pretrained_models/SwinIR_4x.pth',
    'input': 'input.png',
    'output': 'output.png'
})()

# Define the model
model = define_model(args)

# Load image
img = cv2.imread(args.input, cv2.IMREAD_COLOR).astype('float32') / 255.0
img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(img)

# Save result
output = output.squeeze().permute(1, 2, 0).numpy()
cv2.imwrite(args.output, (output * 255).astype('uint8'))
```

### Command-Line Inference

```bash
python inference/inference_swinir.py --task 001_classical --input input.png --output output.png --scale 4 --model_path models/SwinIR_4x.pth
```

## Training

### Dataset Preparation

The project supports the following datasets:
- DIV2K (800 training images)
- REDS (266 training images)
- Vimeo90K (75,712 training images)

Use the provided script to easily prepare datasets in LMDB format:

```bash
# Create LMDB dataset
python scripts/data_preparation/create_lmdb.py --dataset DIV2K --input_path /path/to/div2k
```

### Start Training

```bash
# Single-GPU training
python basicsr/train.py -opt options/train/SwinIR/train_SWINIR-FT.yml

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 basicsr/train.py -opt options/train/SwinIR/train_SWINIR-FT.yml
```

## Model Architecture

The core network uses the Swin Transformer Block (RSTB) module, comprising:
- **Patch Embedding**: Divides the image into non-overlapping patches.
- **RSTB**: Residual groups of multiple Swin Transformer Blocks.
- **Patch UnEmbed**: Reconstructs image pixels via PixelShuffle.
- **Upsample**: Upsampling module (PixelShuffle or transposed convolution).

Key Hyperparameters:
- `depth`: Number of RSTB blocks
- `num_heads`: Number of attention heads
- `window_size`: Window size
- `mlp_ratio`: MLP expansion ratio

## Project Structure

```
SwinIR/
├── basicsr/                  # BasicSR core framework
│   ├── archs/               # Network architecture definitions
│   │   └── swinir_arch.py   # SwinIR network implementation
│   ├── data/                # Dataset loading
│   ├── losses/              # Loss functions
│   ├── models/              # Model definitions
│   ├── ops/                 # CUDA operators
│   └── utils/               # Utility functions
├── inference/               # Inference scripts
│   ├── inference_swinir.py
│   └── inference_dlspiswinir.py
├── options/                 # Training configuration files
├── scripts/                 # Auxiliary scripts
└── experiments/             # Experiment output directory
```

## Pretrained Models

Download pretrained models from:
- [Google Drive](https://drive.google.com/drive/folders/1Qr0rX1qP7Xa2qKcF2h3rX2qKcF2h3rX2)
- Baidu Netdisk

Model configurations by task:

| Task | Scale | Window Size | Model File |
|------|-------|-------------|------------|
| Classical Super-Resolution | 2x | 8 | SwinIR_2x.pth |
| Classical Super-Resolution | 3x | 8 | SwinIR_3x.pth |
| Classical Super-Resolution | 4x | 8 | SwinIR_4x.pth |
| Lightweight Super-Resolution | 4x | 8 | SwinIR_M_4x.pth |

## Evaluation

Compute PSNR/SSIM:

```bash
python scripts/metrics/calculate_psnr_ssim.py --gt datasets/Set5 --input results/Set5 --scale 4
```

## Related Paper

If this project helps your research, please cite the following paper:

```bibtex
@article{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jie and Fan, Yuchen and Zhang, Kai and Rastegari, Mohammad and Dardo, Fabio},
  journal={arXiv preprint arXiv:2108.10257},
  year={2021}
}
```

## License

This project is intended solely for academic research and may not be used for commercial purposes.

## Acknowledgments

We thank the [BasicSR](https://github.com/XPixelGroup/BasicSR) team for providing an excellent deep learning framework.