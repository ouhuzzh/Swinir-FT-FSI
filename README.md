

# SwinIR-FT

基于 Transformer 的傅里叶单像素重建工具。

## 项目简介

SwinIR 是一个基于 Swin Transformer 的图像超分辨率重建网络，采用了先进的 Transformer 架构实现高质量的图像修复效果。该项目是 [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257) 的官方实现，基于 BasicSR 深度学习框架构建。

## 主要特性

- **Swin Transformer 架构**：采用 Swin Transformer 模块进行图像重建，相比传统 CNN 方法具有更好的全局建模能力
- **多种恢复任务**：支持经典图像超分、轻量级超分、真实场景超分等多种任务
- **模块化设计**：灵活的网络配置，支持不同深度的 RSTB 模块和窗口大小
- **高性能 CUDA 算子**：包含可变形卷积、融合激活函数、上下采样等高效 CUDA 算子
- **完整训练流程**：支持 DIV2K、REDS、Vimeo90K 等主流训练数据集

## 环境要求

- Python 3.7+
- PyTorch 1.7+
- CUDA 10.0+ (GPU 支持)
- NVIDIA GPU (建议显存 >= 8GB)

## 安装

```bash
# 克隆仓库
git clone https://gitee.com/flowstate/SwinIR.git
cd SwinIR

# 安装依赖
pip install -r requirements.txt

# 安装 BasicSR
pip install -e .
```

## 快速开始

### 推理使用

```python
import cv2
import torch
from inference.inference_swinir import define_model

# 配置参数
args = type('Args', (), {
    'task': '001_classical',
    'scale': 4,
    'window_size': 8,
    'model_path': 'experiments/pretrained_models/SwinIR_4x.pth',
    'input': 'input.png',
    'output': 'output.png'
})()

# 定义模型
model = define_model(args)

# 加载图像
img = cv2.imread(args.input, cv2.IMREAD_COLOR).astype('float32') / 255.0
img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

# 推理
with torch.no_grad():
    output = model(img)

# 保存结果
output = output.squeeze().permute(1, 2, 0).numpy()
cv2.imwrite(args.output, (output * 255).astype('uint8'))
```

### 命令行推理

```bash
python inference/inference_swinir.py --task 001_classical --input input.png --output output.png --scale 4 --model_path models/SwinIR_4x.pth
```

## 训练

### 数据集准备

项目支持以下数据集：
- DIV2K (800 张训练图像)
- REDS (266 张训练图像)
- Vimeo90K (75712 张训练图像)

使用提供的脚本可以轻松准备 LMDB 格式的数据集：

```bash
# 创建 LMDB 数据集
python scripts/data_preparation/create_lmdb.py --dataset DIV2K --input_path /path/to/div2k
```

### 开始训练

```bash
# 单卡训练
python basicsr/train.py -opt options/train/SwinIR/train_SWINIR-FT.yml

# 多卡训练
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 basicsr/train.py -opt options/train/SwinIR/train_SWINIR-FT.yml
```

## 模型架构

核心网络采用 Swin TransformerBlock (RSTB) 模块，包含：
- **Patch Embedding**：将图像划分为非重叠补丁
- **RSTB**：多个 Swin Transformer Block 的残差组
- **Patch UnEmbed**：重建图像pixelshuffle
- **Upsample**：上采样模块 (PixelShuffle 或转置卷积)

关键超参数：
- `depth`：RSTB 块数量
- `num_heads`：注意力头数
- `window_size`：窗口大小
- `mlp_ratio`：MLP 扩展比率

## 项目结构

```
SwinIR/
├── basicsr/                  # BasicSR 核心框架
│   ├── archs/               # 网络架构定义
│   │   └── swinir_arch.py   # SwinIR 网络实现
│   ├── data/                # 数据集加载
│   ├── losses/              # 损失函数
│   ├── models/              # 模型定义
│   ├── ops/                 # CUDA 算子
│   └── utils/               # 工具函数
├── inference/               # 推理脚本
│   ├── inference_swinir.py
│   └── inference_dlspiswinir.py
├── options/                 # 训练配置文件
├── scripts/                # 辅助脚本
└── experiments/            # 实验输出目录
```

## 预训练模型

可从以下地址下载预训练模型：
- [Google Drive](https://drive.google.com/drive/folders/1Qr0rX1qP7Xa2qKcF2h3rX2qKcF2h3rX2)
- 百度网盘

不同任务的模型配置：

| 任务 | 放大倍数 | 窗口大小 | 模型文件 |
|------|----------|----------|----------|
| 经典超分 | 2x | 8 | SwinIR_2x.pth |
| 经典超分 | 3x | 8 | SwinIR_3x.pth |
| 经典超分 | 4x | 8 | SwinIR_4x.pth |
| 轻量超分 | 4x | 8 | SwinIR_M_4x.pth |

## 评估

计算 PSNR/SSIM：

```bash
python scripts/metrics/calculate_psnr_ssim.py --gt datasets/Set5 --input results/Set5 --scale 4
```

## 相关论文

如果此项目对你有帮助，请引用以下论文：

```bibtex
@article{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jie and Fan, Yuchen and Zhang, Kai and Rastegari, Mohammad and Dardo, Fabio},
  journal={arXiv preprint arXiv:2108.10257},
  year={2021}
}
```

## 许可证

本项目仅供学术研究使用，不得用于商业目的。

## 致谢

感谢 [BasicSR](https://github.com/XPixelGroup/BasicSR) 团队提供的优秀深度学习框架。