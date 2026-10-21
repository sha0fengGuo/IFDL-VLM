# IFDL-VLM — CVPR 2026 Findings

[English](README.md) · [中文](README_CN.md)

> IFDL-VLM 官方代码：一个用于图像伪造检测与定位的两阶段视觉语言模型流程。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#安装) [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE) [![Paper](https://img.shields.io/badge/CVPR%202026%20Findings-论文-228b22)](https://arxiv.org/abs/2603.12930)

## 项目简介

IFDL-VLM 研究视觉语言模型中的语义先验应当如何服务于图像伪造检测与定位。仓库包含两个阶段：

- `stage1/`：基于早期 SIDA 代码路径的检测与定位代码。
- `stage2/`：基于 LLaVA 的训练与推理代码，包括区域感知训练选项。

论文：[Rethinking VLMs for Image Forgery Detection and Localization](https://arxiv.org/abs/2603.12930) · 代码：[本仓库](https://github.com/sha0fengGuo/IFDL-VLM)

数据集、模型权重和实验材料将与源代码分开提供。

## 目录结构

```text
IFDL-VLM/
├── stage1/                  # Stage-1 训练、测试、模型和数据工具
├── stage2/                  # LLaVA-based 训练、推理和评估代码
├── REPRODUCTION.md          # English 复现说明
├── REPRODUCTION_CN.md       # 中文复现说明
├── README.md                # English 项目说明
└── README_CN.md             # 中文项目说明
```

## 安装

Stage 1 和 Stage 2 的 CUDA/PyTorch 依赖可能冲突，建议分别建立环境。依赖记录在：

- `stage1/requirements.txt`、`stage1/environment.yml`
- `stage2/requirements.txt`、`stage2/environment.yml`

```bash
git clone https://github.com/sha0fengGuo/IFDL-VLM.git
cd IFDL-VLM
conda create -n ifdl-vlm python=3.10 -y
conda activate ifdl-vlm
```

分阶段安装：

```bash
python -m pip install -r stage1/requirements.txt
# Stage 2 建议使用独立环境
python -m pip install -r stage2/requirements.txt
```

依赖文件包含 CUDA 相关 PyTorch 包。如果目标机器的 CUDA 工具链不同，请按实际环境调整，并在复现实验记录中保存最终版本。

## 数据和模型路径

数据集和模型权重不包含在仓库中，请放在仓库外，并向脚本传入绝对路径。

Stage 1 需要 SID_SET 风格的数据目录和 SAM/CLIP 视觉组件。Stage 2 需要类似下面的目录：

```text
/path/to/data/
├── train.json          # 对话或指令记录
└── images/             # train.json 引用的图像
```

具体标注格式必须和对应阶段的数据加载器一致。不要提交私有数据、密钥或模型权重。

## 快速开始

### Stage 1 训练

先修改 `stage1/train_simple.sh` 中的路径和 GPU 设置：

```bash
cd stage1
bash train_simple.sh
```

至少需要检查 `DATASET_DIR`、`VISION_PRETRAINED`、`EXP_NAME` 和 DeepSpeed GPU 选择。

### Stage 1 评估

```bash
cd stage1
CUDA_VISIBLE_DEVICES=0 bash test.sh
```

运行前修改 `test.sh` 中的 `--ckpt_path`、`--dataset_dir`、`--vision_tower` 和 `--image_size`。

### Stage 2 训练

```bash
cd stage2
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=/absolute/path/to/train.json \
IMAGE_FOLDER=/absolute/path/to/images \
OUTPUT_DIR=/absolute/path/to/output \
PRETRAIN_MM_MLP_ADAPTER=/absolute/path/to/mm_projector.bin \
bash train.sh
```

默认启动脚本使用 `liuhaotian/llava-v1.5-13b`、336px CLIP ViT-L/14、LoRA、区域感知训练和四个训练 epoch。改变这些设置时，请在实验记录中注明。

### Stage 2 单图推理

```bash
cd stage2
python inference.py \
  --model-path /absolute/path/to/model_or_lora_dir \
  --image-path /absolute/path/to/image.jpg \
  --mask-path /absolute/path/to/mask.png \
  --query "Describe whether the image is real, synthetic, or tampered."
```

完整输入说明请看 [REPRODUCTION_CN.md](REPRODUCTION_CN.md) 和 [stage2/README.md](stage2/README.md)。

## 复现状态

复现说明记录公开代码所需的环境、路径、命令和实验元数据。

- [English reproduction guide](REPRODUCTION.md)
- [中文复现说明](REPRODUCTION_CN.md)

## 复现记录清单

每次实验至少记录：

- commit SHA 和使用的环境文件
- CUDA、PyTorch、Transformers、DeepSpeed 版本
- 数据集、划分、预处理和标注版本
- 基础模型和 checkpoint 路径或公开下载地址
- GPU 型号与数量、batch size、梯度累积和随机种子
- 完整命令和输出目录
- 评估脚本、指标和各数据集结果

## 引用、许可与联系

```bibtex
@article{guo2026rethinking,
  title   = {Rethinking VLMs for Image Forgery Detection and Localization},
  author  = {Guo, Shaofeng and Cui, Jiequan and Hong, Richang},
  journal = {arXiv preprint arXiv:2603.12930},
  year    = {2026}
}
```

代码使用 [MIT License](LICENSE)。问题和合作请求可以提交 GitHub Issue，或发送至 `shaofengGuo@mail.hfut.edu.cn`。

项目基于 [LLaVA](https://github.com/haotian-liu/LLaVA) 和早期 [SIDA](https://github.com/hzlsaber/SIDA) 代码路径。
