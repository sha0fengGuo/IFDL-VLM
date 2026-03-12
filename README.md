# IFDL_VLM

This repository contains two stages of code for the IFDL VLM project.

- `stage1/`: Stage-1 code (from the earlier SIDA code submission), including training and testing scripts.
- `stage2/`: Stage-2 code based on LLaVA, including training/inference pipelines and the core `llava` package.

## Scope

Only source code and minimal run scripts are included.

Excluded from this repository:
- datasets
- model weights/checkpoints
- logs and cached artifacts

## Quick Start

1. Create a Python environment (Python 3.10+ recommended).
2. Install dependencies for each stage as needed.
3. Follow stage-specific READMEs:
   - `stage1/README.md`
   - `stage2/README.md`

## Notes

- Paths are configurable and should be passed through CLI arguments or environment variables.
- Do not commit local datasets or weights.
