# SwinIR-FT

基于 Swin Transformer 与 BasicSR 框架实现的傅里叶单像素图像重建研究代码。

## 项目简介

本仓库整理了论文实验所需的核心训练与推理代码，目标是复现和扩展基于 Transformer 的傅里叶单像素图像重建流程。当前公开内容主要覆盖：

- `basicsr/`：网络结构、数据集、损失函数、训练/测试逻辑
- `inference/`：推理脚本
- `options/`：训练配置
- `scripts/`：数据准备、指标计算和辅助脚本

为便于公开发布，仓库中不包含本地实验输出、TensorBoard 日志、数据集和模型权重等大文件。

## 亮点

- 基于 `BasicSR` 训练框架组织实验流程
- 采用 `SwinIR` 主干网络进行图像重建/恢复
- 保留论文复现实验所需的训练配置与推理脚本
- 支持按照本地数据路径重新组织训练与测试目录

## 仓库状态

- 当前仓库主要服务于论文复现与学术交流
- 默认只保留核心代码，不附带大体积数据和权重
- 目前由作者本人维护，默认仅作者直接向主分支推送

## 方法概览

整体流程可概括为：

1. 准备低质量观测与目标图像配对数据
2. 使用 `DLFSPIDataset` 构建训练/验证数据加载流程
3. 通过 `options/train/SwinIR/train_SWINIR-FT.yml` 配置模型、损失与训练参数
4. 使用 `basicsr/train.py` 进行训练
5. 使用 `inference/` 下脚本进行推理与结果保存

## 环境要求

- Python 3.8 及以上
- PyTorch 1.7 及以上
- CUDA 10.0 及以上（如使用 GPU）

## 安装

```bash
git clone <your-repository-url>
cd <repo-name>
pip install -r requirements.txt
```

如果你需要 GPU 版本的 PyTorch，建议先根据本机 CUDA 版本安装匹配的 `torch` / `torchvision`，再执行：

```bash
pip install -r requirements.txt
```

## 数据与权重

以下目录默认不随仓库提交，请按你的实验环境自行准备：

- `datasets/`
- `inputs/`
- `pretrained_models/`
- `experiments/`
- `result/`
- `tb_logger/`

## 训练

训练配置位于 [`options/train/SwinIR/train_SWINIR-FT.yml`](</D:/A/SR/BasicSR - 副本/options/train/SwinIR/train_SWINIR-FT.yml>)。

单卡训练：

```bash
python basicsr/train.py -opt options/train/SwinIR/train_SWINIR-FT.yml
```

多卡训练：

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 basicsr/train.py -opt options/train/SwinIR/train_SWINIR-FT.yml
```

训练前请先在配置文件中填写：

- `dataroot_gt`
- `dataroot_lq`
- `pretrain_network_g`（如需要加载预训练权重）

## 推理

仓库中提供了两个推理脚本：

- [`inference/inference_swinir.py`](</D:/A/SR/BasicSR - 副本/inference/inference_swinir.py>)
- [`inference/inference_dlspiswinir.py`](</D:/A/SR/BasicSR - 副本/inference/inference_dlspiswinir.py>)

示例命令：

```bash
python inference/inference_swinir.py --task color_dn --model_path <model_path> --input <input_dir> --output <output_dir>
```

建议显式指定以下参数：

- `--model_path`
- `--input`
- `--output`
- `--task`

具体参数定义可直接查看脚本中的 `argparse` 配置。

## 项目结构

```text
.
├── basicsr/
├── inference/
├── options/
├── scripts/
├── datasets/            # 本地数据目录，默认忽略
├── pretrained_models/   # 本地权重目录，默认忽略
├── experiments/         # 训练输出目录，默认忽略
├── result/              # 推理结果目录，默认忽略
└── tb_logger/           # TensorBoard 日志目录，默认忽略
```

## 复现建议

- 首次公开时建议在 release 或 README 中补充数据下载说明
- 预训练模型建议单独放到网盘或 GitHub Releases
- 若后续补论文链接，可在 README 首页直接加入论文标题、作者和 arXiv/期刊链接

## 引用

如果你在学术工作中使用了本仓库，建议同时引用：

- 你的论文
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [SwinIR](https://github.com/JingyunLiang/SwinIR)

你可以在这里补充你论文对应的 BibTeX：

```bibtex
@article{your_paper_key,
  title   = {Your Paper Title},
  author  = {Your Name},
  journal = {To be added},
  year    = {2026}
}
```

## 许可证

本仓库采用 [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) 许可证发布。

这是基于仓库当前代码明显继承自 BasicSR 和 SwinIR 而作出的兼容整理；对应的上游项目也请一并遵守其许可证与引用要求。

## 贡献说明

当前阶段以论文复现和作者维护为主。

- 欢迎通过 issue 反馈问题
- 如无特殊说明，默认仅作者维护并直接推送主分支
- 如果后续开放 PR，我可以再帮你补 `CONTRIBUTING.md`

## 致谢

感谢 [BasicSR](https://github.com/XPixelGroup/BasicSR) 提供基础框架支持。
感谢 [SwinIR](https://github.com/JingyunLiang/SwinIR) 提供模型设计与公开实现。
