

# SwinIR-FT

基于 Transformer 的傅里叶单像素重建工具。

## 项目简介

SwinIR-FT 是一个基于 Swin Transformer 的傅里叶单像素图像重建网络，采用了先进的 Transformer 架构实现高质量的图像修复效果。该项目基于 BasicSR 深度学习框架构建。



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


# 安装 BasicSR
pip install -e .
```

## 快速开始

```

### 命令行推理

```bash
 python inference/inference_swinir.py --task color_dn --model_path  --input --output 
```

## 训练

### 数据集准备

```bash
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
- 通过网盘分享的文件：net_g_55000.pth
链接: https://pan.baidu.com/s/1owKCp12lyC_QOQ01aRfWUQ?pwd=h1ig 提取码: h1ig

## 许可证

本项目仅供学术研究使用，不得用于商业目的。

## 致谢

感谢 [BasicSR](https://github.com/XPixelGroup/BasicSR) 团队提供的优秀深度学习框架。