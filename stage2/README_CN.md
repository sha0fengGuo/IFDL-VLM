# Stage 2（基于 LLaVA）

[English](README.md) · [中文](README_CN.md) · [项目复现说明](../REPRODUCTION_CN.md)

Stage 2 保留 LLaVA 风格的代码结构，包含模型、训练、推理和评估组件。

## 包含内容

- `llava/` 核心代码
- `inference.py`
- `train.sh`、`train_mem.py`
- `model_load.py`
- `environment.yml`、`requirements.txt`

## 训练

```bash
cd stage2
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=/absolute/path/to/train.json \
IMAGE_FOLDER=/absolute/path/to/images \
OUTPUT_DIR=/absolute/path/to/output \
PRETRAIN_MM_MLP_ADAPTER=/absolute/path/to/mm_projector.bin \
bash train.sh
```

## 推理

```bash
python inference.py \
  --model-path /path/to/model_or_lora_dir \
  --image-path /path/to/image.jpg \
  --mask-path /path/to/mask.png \
  --query "Describe whether the image is real, synthetic, or tampered."
```

数据、checkpoint 和完整复现限制请看 [中文复现说明](../REPRODUCTION_CN.md)。
