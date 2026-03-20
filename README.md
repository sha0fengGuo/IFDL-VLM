# IFDL-VLM - CVPR 2026 Findings

> Official codebase for **IFDL-VLM** (two-stage image forgery detection and localization training & inference pipeline).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red)](#installation)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CVPR 2026 Findings](https://img.shields.io/badge/CVPR-2026_Findings-228b22)](https://arxiv.org/abs/2603.12930)

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Roadmap / TODO](#-roadmap--todo)
- [News](#-news)
- [Environment & Installation](#-environment--installation)
- [Data Preparation](#-data-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Reproducibility Checklist](#-reproducibility-checklist)
- [Model Zoo](#-model-zoo)
- [Results](#-results)
- [FAQ](#-faq)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [Contact](#-contact)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🧠 Overview
This repository contains two stages of code for the **IFDL-VLM** project:

- `stage1/`: Stage-1 code (from the earlier SIDA submission), including training and testing scripts. Baseline reference: [SIDA](https://github.com/hzlsaber/SIDA?tab=readme-ov-file).
- `stage2/`: Stage-2 code based on LLaVA, including training/inference pipelines and the core `llava` package.

> 🎉 Accepted at **CVPR 2026 Findings**. Paper: [arXiv:2603.12930](https://arxiv.org/abs/2603.12930)

> Current repo focuses on **code release**. Datasets, checkpoints, and logs are not included.

---

## 📁 Project Structure
```text
IFDL-VLM/
├── stage1/                  # Stage-1 training / testing code
├── stage2/                  # Stage-2 (LLaVA-based) training / inference code
└── README.md
```

See detailed docs:
- `stage1/README.md`
- `stage2/README.md`

---

## 🗺️ Roadmap / TODO
- [x] Release Stage-1 and Stage-2 training/inference code.
- [ ] Release cleaned training/validation split scripts.
- [ ] Release pretrained checkpoints.
- [ ] Release evaluation benchmark scripts and fixed seeds.
- [ ] Provide one-command reproduction pipeline.

---

## 📰 News
- **2026-03**: Paper accepted to **CVPR 2026 Findings**. Preprint: [arXiv:2603.12930](https://arxiv.org/abs/2603.12930)
- **2026-03**: Repository structure consolidated and root README improved.
- *(Add future updates here, e.g., checkpoint release / paper acceptance / benchmark updates.)*

---

## ⚙️ Environment & Installation

### 1) Create environment
```bash
conda create -n ifdl-vlm python=3.10 -y
conda activate ifdl-vlm
```

### 2) Install dependencies
Because Stage-1 and Stage-2 may use different dependency sets, please install according to each stage:

```bash
# Stage-1 dependencies
cd stage1
# e.g. pip install -r requirements.txt

# Stage-2 dependencies
cd ../stage2
# e.g. pip install -r requirements.txt
```

> If CUDA/torch versions conflict between stages, consider using separate conda environments.

---

## 🗂️ Data Preparation

### Expected directory layout (example)
```text
/path/to/datasets/
├── dataset_a/
│   ├── images/
│   └── annotations/
└── dataset_b/
    ├── images/
    └── annotations/
```

### Notes
- Pass dataset paths via CLI arguments or environment variables.
- Do **not** commit datasets to this repository.
- Keep private/proprietary data outside this repo.

---

## 🏋️ Training
Please follow stage-specific instructions:

### Stage-1
```bash
cd stage1
# bash scripts/train_stage1.sh
```

### Stage-2
```bash
cd stage2
# bash scripts/train_stage2.sh
```

> Replace script names/arguments with those provided in each stage directory.

---

## 📏 Evaluation
```bash
cd stage1
# bash scripts/eval_stage1.sh

cd ../stage2
# bash scripts/eval_stage2.sh
```

Recommended report items for paper reproduction:
- checkpoint path
- random seed
- dataset split/version
- evaluation metric implementation
- hardware (GPU type and count)

---

## 🔍 Inference
```bash
cd stage2
# python -m llava.serve.cli --model-path <ckpt_path> ...
```

For batch inference, add your own wrapper script and save outputs to a non-versioned path (e.g., `./outputs/` ignored by git).

---

## ✅ Reproducibility Checklist
- [ ] Fixed random seed for training/evaluation.
- [ ] Exact dependency versions recorded.
- [ ] Data split scripts and IDs released.
- [ ] Training logs and tensorboard curves archived.
- [ ] Final checkpoints with SHA256 checksums provided.

---

## 🧩 Model Zoo
| Model | Stage | Description | Link |
|---|---|---|---|
| IFDL-VLM-Stage1 | Stage-1 | Base training checkpoint | TBA |
| IFDL-VLM-Stage2 | Stage-2 | Instruction-tuned checkpoint | TBA |

> Add links after release (Google Drive / HuggingFace / ModelScope).

---

## 📊 Results
| Setting | Dataset | Metric | Score |
|---|---|---|---|
| Stage-1 baseline | TBA | TBA | TBA |
| Stage-2 full model | TBA | TBA | TBA |

> Please fill with the exact numbers reported in your paper / appendix.

---

## ❓ FAQ
**Q1: Why are datasets/checkpoints not included?**  
A: Due to data license / storage constraints, only source code is currently released.

**Q2: Can I train Stage-1 and Stage-2 in one environment?**  
A: It depends on dependency compatibility. Using separate environments is safer.

**Q3: Where should I report bugs?**  
A: Please open an issue with environment details, reproduction steps, and logs.

---

## 🤝 Contributing
Contributions are welcome!

- Open an issue for bug reports or feature requests.
- Submit a pull request with clear description and reproducible changes.
- Keep commits small and focused.

---

## 📚 Citation
If you find this project helpful, please cite:

```bibtex
@article{guo2026rethinking,
  title   = {Rethinking VLMs for Image Forgery Detection and Localization},
  author  = {Guo, Shaofeng and Cui, Jiequan and Hong, Richang},
  journal = {arXiv preprint arXiv:2603.12930},
  year    = {2026}
}
```

---

## 📮 Contact
For questions and collaboration requests:
- Open a GitHub issue
- Or contact the maintainers via email: shaofengGuo@mail.hfut.edu.cn

---

## 📄 License
Licensed under **MIT**. See [LICENSE](LICENSE) for full terms.

---

## 🙏 Acknowledgements
This project builds upon and/or references the following open-source efforts:
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- Stage-1 baseline code from earlier SIDA submission ([SIDA](https://github.com/hzlsaber/SIDA?tab=readme-ov-file))

