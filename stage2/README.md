# Stage 2 (LLaVA-based)

This folder keeps the Stage-2 codebase with the original LLaVA-style structure preserved as much as possible.

## Included

- `llava/` core package (model, train, serve, eval code)
- `inference.py`
- `train.sh`, `train_mem.py`
- `model_load.py`

## Excluded

- datasets
- checkpoints/weights
- logs/results artifacts

## Environment

Use Python 3.10+ and install required dependencies (PyTorch, Transformers, PEFT, etc.) according to your target CUDA/runtime setup.

## Training

`train.sh` uses configurable paths via environment variables:

- `DATA_PATH` (default: `./data/train.json`)
- `IMAGE_FOLDER` (default: `./data/images`)
- `OUTPUT_DIR` (default: `./outputs/50region_lora_baseline`)
- `PRETRAIN_MM_MLP_ADAPTER` (default: `./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin`)
- `CUDA_VISIBLE_DEVICES` (default: `0`)

Example:

```bash
DATA_PATH=/path/to/train.json \
IMAGE_FOLDER=/path/to/images \
OUTPUT_DIR=./outputs/exp1 \
PRETRAIN_MM_MLP_ADAPTER=/path/to/mm_projector.bin \
CUDA_VISIBLE_DEVICES=0 \
bash train.sh
```

## Inference

```bash
python inference.py \
  --model-path /path/to/model_or_lora_dir \
  --image-path /path/to/image.jpg \
  --mask-path /path/to/mask.png \
  --query "Describe whether the image is real, synthetic, or tampered."
```
