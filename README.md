# IFDL-VLM — CVPR 2026 Findings

[English](README.md) · [中文](README_CN.md)

> Official code for **IFDL-VLM**, a two-stage vision-language pipeline for image forgery detection and localization.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#installation) [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE) [![Paper](https://img.shields.io/badge/CVPR%202026%20Findings-paper-228b22)](https://arxiv.org/abs/2603.12930)

## Overview

IFDL-VLM studies how vision-language models should use semantic priors for image forgery detection and localization. The release contains two stages:

- `stage1/`: the Stage-1 detection/localization code, based on the earlier SIDA code path.
- `stage2/`: the Stage-2 LLaVA-based training and inference code, including the region-aware training option.

Paper: [Rethinking VLMs for Image Forgery Detection and Localization](https://arxiv.org/abs/2603.12930) · Code release: [this repository](https://github.com/sha0fengGuo/IFDL-VLM)

Datasets, checkpoints, and experiment assets are provided separately from the source code.

For dataset access and the original dataset protocols, please refer to the upstream repositories:

- [SIDA](https://github.com/hzlsaber/SIDA) for the SID-Set data and related preparation details.
- [FakeShield](https://github.com/zhipeixu/FakeShield) for the FakeShield data and protocol.

The datasets are not redistributed in this repository. Follow the licenses and download instructions of the respective upstream projects.

## Checkpoints and pretrained weights

The conference-version checkpoints are currently shared through Quark Drive:

- [Download IFDL-VLM-ckpts from Quark Drive](https://pan.quark.cn/s/f6e20cd50685?pwd=t4kC) (extraction code: `t4kC`)
- [Download the Stage-2 LLaVA-13B checkpoint separately](https://pan.quark.cn/s/772e7c439ec9?pwd=5k5b) (extraction code: `5k5b`)
- The folder contains the Stage-1 SID-Set checkpoint, the Stage-1 FakeShield checkpoint, and the Stage-2 region-aware LLaVA-13B LoRA checkpoint.
- A Google Drive mirror will be added in a later update. The download location may be updated without changing the checkpoint names.

The release package corresponds to the following files:

```text
ifdl-vlm-stage1-sidset.bin       # SID-Set classification and localization
ifdl-vlm-stage1-fakeshield.bin  # FakeShield localization
stage2_llava13b_regionaware_alpha0.5_epoch4/
                                 # Stage-2 explanation LoRA checkpoint
```

### Download pretrained SAM weight

Stage 1 uses the SAM ViT-H pretrained weight. You can download it with:

```bash
mkdir -p weight
wget https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth -P weight/
```

### Download the CLIP vision encoder

The released configuration uses `openai/clip-vit-large-patch14-336` (336px CLIP ViT-L/14). Download it from Hugging Face with:

```bash
huggingface-cli download openai/clip-vit-large-patch14-336 \
  --local-dir weight/clip-vit-large-patch14-336
```

Model page: [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)

## Repository layout

```text
IFDL-VLM/
├── stage1/
│   ├── model/                 # Stage-1 model and segmentation components
│   ├── utils/                 # Dataset and utility code
│   ├── train_simple.py        # Stage-1 training entry point
│   ├── test_simple.py         # Stage-1 evaluation entry point
│   ├── train_simple.sh        # Example DeepSpeed launcher
│   └── test.sh                # Example evaluation launcher
├── stage2/
│   ├── llava/                 # LLaVA-based model, training, serving, and eval code
│   ├── train_mem.py           # Training entry point
│   ├── inference.py           # Single-image inference entry point
│   ├── train.sh               # Example Stage-2 launcher
│   └── model_load.py          # Model-loading utilities
├── REPRODUCTION.md            # Detailed English reproduction guide
├── REPRODUCTION_CN.md         # 中文复现说明
└── README_CN.md               # 中文项目说明
```

## Installation

Use separate environments for Stage 1 and Stage 2 when their CUDA/PyTorch requirements conflict. The repository records the currently used dependency sets in:

- `stage1/requirements.txt` and `stage1/environment.yml`
- `stage2/requirements.txt` and `stage2/environment.yml`

The short setup is:

```bash
git clone https://github.com/sha0fengGuo/IFDL-VLM.git
cd IFDL-VLM
conda create -n ifdl-vlm python=3.10 -y
conda activate ifdl-vlm
```

Install one stage at a time:

```bash
# Stage 1
python -m pip install -r stage1/requirements.txt

# Stage 2 (prefer a separate environment)
python -m pip install -r stage2/requirements.txt
```

The requirements files contain CUDA-specific PyTorch packages. Adjust them for the CUDA toolchain on the target machine if installation fails; record the final versions before reporting reproduction results.

## Data and model paths

Dataset and model paths are configured outside the source tree and passed as absolute paths to the scripts.

Stage 1 expects a SID_SET-style dataset directory and a SAM/CLIP vision setup. Stage 2 expects:

```text
/path/to/data/
├── train.json          # conversation/instruction records
└── images/             # image files referenced by train.json
```

The exact annotation format must match the loader used by the selected stage. Do not commit private data, credentials, or model weights.

## Quick start

### Stage 1 training

Edit the paths and GPU list in `stage1/train_simple.sh`, then run:

```bash
cd stage1
bash train_simple.sh
```

The script is an example launcher. In particular, update `DATASET_DIR`, `VISION_PRETRAINED`, `EXP_NAME`, and the DeepSpeed GPU selection before running it.

### Stage 1 evaluation

```bash
cd stage1
CUDA_VISIBLE_DEVICES=0 bash test.sh
```

Update `--ckpt_path`, `--dataset_dir`, `--vision_tower`, and `--image_size` in `test.sh` first.

### Stage 2 training

From the `stage2/` directory:

```bash
cd stage2
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=/absolute/path/to/train.json \
IMAGE_FOLDER=/absolute/path/to/images \
OUTPUT_DIR=/absolute/path/to/output \
PRETRAIN_MM_MLP_ADAPTER=/absolute/path/to/mm_projector.bin \
bash train.sh
```

The default launcher uses `liuhaotian/llava-v1.5-13b`, CLIP ViT-L/14 at 336px, LoRA, region-aware training, and four training epochs. Change these settings only after recording the change in your experiment log.

### Stage 2 single-image inference

```bash
cd stage2
python inference.py \
  --model-path /absolute/path/to/model_or_lora_dir \
  --image-path /absolute/path/to/image.jpg \
  --mask-path /absolute/path/to/mask.png \
  --query "Describe whether the image is real, synthetic, or tampered."
```

For the full command context and expected inputs, see [REPRODUCTION.md](REPRODUCTION.md) and [stage2/README.md](stage2/README.md).

## Reproduction status

The reproduction guide records the expected environments, paths, commands, and run metadata for the released code.

- [English reproduction guide](REPRODUCTION.md)
- [中文复现说明](REPRODUCTION_CN.md)

## Release checklist

Before calling a run reproducible, record:

- commit SHA and environment file used
- CUDA, PyTorch, Transformers, and DeepSpeed versions
- dataset name, split, preprocessing, and annotation version
- base model and checkpoint path (or public download URL)
- GPU type/count, batch size, gradient accumulation, and random seed
- exact command line and output directory
- evaluation script, metrics, and per-dataset results

## Citation

```bibtex
@article{guo2026rethinking,
  title   = {Rethinking VLMs for Image Forgery Detection and Localization},
  author  = {Guo, Shaofeng and Cui, Jiequan and Hong, Richang},
  journal = {arXiv preprint arXiv:2603.12930},
  year    = {2026}
}
```

## License and contact

The source code is released under the [MIT License](LICENSE). For questions, open a GitHub issue or email `shaofengGuo@mail.hfut.edu.cn`.

## Acknowledgements

This project builds upon [LLaVA](https://github.com/haotian-liu/LLaVA) and the earlier [SIDA](https://github.com/hzlsaber/SIDA) code path.
