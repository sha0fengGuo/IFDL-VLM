# IFDL-VLM reproduction guide

[English](REPRODUCTION.md) · [中文](REPRODUCTION_CN.md) · [Project README](README.md)

This guide records the environments, paths, commands, and metadata needed to run the released code.

## 1. Setup

```bash
git clone https://github.com/sha0fengGuo/IFDL-VLM.git
cd IFDL-VLM
conda create -n ifdl-stage1 python=3.10 -y
conda activate ifdl-stage1
python -m pip install -r stage1/requirements.txt
```

Use a separate environment for Stage 2 when CUDA/PyTorch dependencies conflict:

```bash
conda create -n ifdl-stage2 python=3.10 -y
conda activate ifdl-stage2
python -m pip install -r stage2/requirements.txt
```

Record the commit with `git rev-parse HEAD` and keep the final package versions with the run.

## 2. Stage 1

Stage 1 expects a SID_SET-style dataset and the SAM/CLIP vision weights referenced by the launcher. Edit `stage1/train_simple.sh` before running:

- `DATASET_DIR`
- `VISION_PRETRAINED`
- `EXP_NAME`
- batch/epoch/learning-rate settings
- DeepSpeed GPU selection

```bash
cd stage1
bash train_simple.sh
```

For evaluation, edit `stage1/test.sh` with the checkpoint, dataset, image size, and vision tower:

```bash
CUDA_VISIBLE_DEVICES=0 bash test.sh
```

## 3. Stage 2

The example input layout is:

```text
/path/to/data/
├── train.json
└── images/
```

Run training from `stage2/`:

```bash
cd stage2
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=/absolute/path/to/train.json \
IMAGE_FOLDER=/absolute/path/to/images \
OUTPUT_DIR=/absolute/path/to/output \
PRETRAIN_MM_MLP_ADAPTER=/absolute/path/to/mm_projector.bin \
bash train.sh
```

The launcher defaults to LLaVA-1.5-13B, CLIP ViT-L/14 at 336px, LoRA, region-aware training, four epochs, batch size 1, and learning rate `5e-5`.

Single-image inference:

```bash
python inference.py \
  --model-path /absolute/path/to/model_or_lora_dir \
  --image-path /absolute/path/to/image.jpg \
  --mask-path /absolute/path/to/mask.png \
  --query "Describe whether the image is real, synthetic, or tampered."
```

## 4. Run record

For each run, save:

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

Run a one-batch or one-step smoke test before a full schedule, and keep logs and generated outputs in a separate output directory.
