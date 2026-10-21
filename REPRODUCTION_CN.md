# IFDL-VLM 复现说明

[English](REPRODUCTION.md) · [中文](REPRODUCTION_CN.md) · [项目 README](README_CN.md)

本文记录公开代码所需的环境、路径、命令和实验元数据。

## 1. 环境

```bash
git clone https://github.com/sha0fengGuo/IFDL-VLM.git
cd IFDL-VLM
conda create -n ifdl-stage1 python=3.10 -y
conda activate ifdl-stage1
python -m pip install -r stage1/requirements.txt
```

如果 CUDA/PyTorch 依赖冲突，请为 Stage 2 单独建立环境：

```bash
conda create -n ifdl-stage2 python=3.10 -y
conda activate ifdl-stage2
python -m pip install -r stage2/requirements.txt
```

使用 `git rev-parse HEAD` 记录代码版本，并保存最终依赖版本。

## 2. Stage 1

Stage 1 需要 SID_SET 风格数据，以及启动脚本引用的 SAM/CLIP 视觉权重。运行前修改 `stage1/train_simple.sh` 中的：

- `DATASET_DIR`
- `VISION_PRETRAINED`
- `EXP_NAME`
- batch、epoch、learning rate
- DeepSpeed GPU 设置

```bash
cd stage1
bash train_simple.sh
```

评估前修改 `stage1/test.sh` 中的 checkpoint、数据集、图像大小和 vision tower：

```bash
CUDA_VISIBLE_DEVICES=0 bash test.sh
```

## 3. Stage 2

示例输入目录：

```text
/path/to/data/
├── train.json
└── images/
```

从 `stage2/` 运行训练：

```bash
cd stage2
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=/absolute/path/to/train.json \
IMAGE_FOLDER=/absolute/path/to/images \
OUTPUT_DIR=/absolute/path/to/output \
PRETRAIN_MM_MLP_ADAPTER=/absolute/path/to/mm_projector.bin \
bash train.sh
```

启动脚本默认使用 LLaVA-1.5-13B、336px CLIP ViT-L/14、LoRA、区域感知训练、4 个 epoch、单卡 batch size 1 和 `5e-5` learning rate。

单图推理：

```bash
python inference.py \
  --model-path /absolute/path/to/model_or_lora_dir \
  --image-path /absolute/path/to/image.jpg \
  --mask-path /absolute/path/to/mask.png \
  --query "Describe whether the image is real, synthetic, or tampered."
```

## 4. 实验记录

每次运行保存：

```text
commit_sha=
environment_file=
cuda=
pytorch=
transformers=
deepseed=
base_model=
checkpoint=
dataset=
split=
preprocessing=
seed=
gpu=
command=
metric_script=
output_dir=
```

完整运行前先做一个 batch 或一个 step 的 smoke test，并将日志和生成结果放到独立输出目录。
