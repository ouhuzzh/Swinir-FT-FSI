# SwinIR-FT

Research code for Fourier single-pixel image reconstruction based on the Swin Transformer and the BasicSR framework.

## Overview

This repository contains the core training and inference code used in our paper experiments. It is intended for reproducing and extending a Transformer-based pipeline for Fourier single-pixel image reconstruction. The public release mainly includes:

- `basicsr/`: network architectures, datasets, loss functions, and training/testing logic
- `inference/`: inference scripts
- `options/`: training configurations
- `scripts/`: data preparation, metric computation, and utility scripts

To keep the repository lightweight and suitable for public release, local experiment outputs, TensorBoard logs, datasets, and model weights are not included.

## Highlights

- Built on top of the `BasicSR` training framework
- Uses `SwinIR` as the main backbone for image reconstruction/restoration
- Includes the training configurations and inference scripts required for paper reproduction
- Can be adapted to different local dataset layouts and experiment directories

## Repository Status

- This repository is currently maintained for paper reproduction and academic use
- Only the core codebase is included by default; large datasets and pretrained weights are excluded
- The repository is currently maintained by the author, and only the author pushes directly to the main branch

## Method Pipeline

The overall workflow can be summarized as follows:

1. Prepare paired low-quality observations and target images
2. Build the training and validation pipeline with `DLFSPIDataset`
3. Configure the model, losses, and optimization settings in `options/train/SwinIR/train_SWINIR-FT.yml`
4. Train the model with `basicsr/train.py`
5. Run inference and save outputs with the scripts in `inference/`

## Environment

- Python 3.8 or later
- PyTorch 1.7 or later
- CUDA 10.0 or later if GPU training/inference is used

## Installation

```bash
git clone <your-repository-url>
cd <repo-name>
pip install -r requirements.txt
```

If you need a GPU-enabled PyTorch build, it is recommended to first install a `torch` / `torchvision` version that matches your local CUDA environment, and then run:

```bash
pip install -r requirements.txt
```

## Data and Weights

The following directories are intentionally excluded from version control and should be prepared locally in your own environment:

- `datasets/`
- `inputs/`
- `pretrained_models/`
- `experiments/`
- `result/`
- `tb_logger/`

## Training

The main training configuration is located at [`options/train/SwinIR/train_SWINIR-FT.yml`](</D:/A/SR/BasicSR - 副本/options/train/SwinIR/train_SWINIR-FT.yml>).

Single-GPU training:

```bash
python basicsr/train.py -opt options/train/SwinIR/train_SWINIR-FT.yml
```

Multi-GPU training:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 basicsr/train.py -opt options/train/SwinIR/train_SWINIR-FT.yml
```

Before training, please update the configuration file with your local paths:

- `dataroot_gt`
- `dataroot_lq`
- `pretrain_network_g` if you want to load pretrained weights

## Inference

Two inference scripts are currently provided:

- [`inference/inference_swinir.py`](</D:/A/SR/BasicSR - 副本/inference/inference_swinir.py>)
- [`inference/inference_dlspiswinir.py`](</D:/A/SR/BasicSR - 副本/inference/inference_dlspiswinir.py>)

Example command:

```bash
python inference/inference_swinir.py --task color_dn --model_path <model_path> --input <input_dir> --output <output_dir>
```

It is recommended to explicitly specify:

- `--model_path`
- `--input`
- `--output`
- `--task`

You can find the full argument definitions directly in the scripts via `argparse`.

## Project Structure

```text
.
├── basicsr/
├── inference/
├── options/
├── scripts/
├── datasets/            # local dataset directory, ignored by default
├── pretrained_models/   # local weight directory, ignored by default
├── experiments/         # training outputs, ignored by default
├── result/              # inference outputs, ignored by default
└── tb_logger/           # TensorBoard logs, ignored by default
```

## Reproducibility Notes

- For the first public release, it is a good idea to add dataset download instructions in the README or release notes
- Pretrained models are better distributed separately, for example through cloud storage or GitHub Releases
- If you later add the paper link, you can place the title, author list, and arXiv/journal link near the top of the README

## Citation

If you use this repository in academic work, please consider citing:

- your paper
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [SwinIR](https://github.com/JingyunLiang/SwinIR)

You can replace the following placeholder with the BibTeX entry for your paper:

```bibtex
@article{your_paper_key,
  title   = {Your Paper Title},
  author  = {Your Name},
  journal = {To be added},
  year    = {2026}
}
```

## License

This repository is released under the [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) license.

This choice was made to stay compatible with the upstream projects that this codebase clearly builds upon, especially BasicSR and SwinIR. Please also follow the license and citation requirements of those upstream projects.

## Contribution Policy

At the current stage, this repository is mainly maintained for paper reproduction and author-driven updates.

- Issues are welcome for bug reports and discussion
- Unless otherwise stated, only the author maintains and pushes directly to the main branch
- If you later decide to accept pull requests, a `CONTRIBUTING.md` file can be added

## Acknowledgements

Thanks to [BasicSR](https://github.com/XPixelGroup/BasicSR) for the underlying training framework.
Thanks to [SwinIR](https://github.com/JingyunLiang/SwinIR) for the model design and public implementation.
