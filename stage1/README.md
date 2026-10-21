# Stage 1

[English](README.md) · [中文](README_CN.md)

This folder contains Stage-1 training/testing code.

## Included

- `train_simple.py`, `train_simple.sh`
- `test_simple.py`, `test.sh`
- `model/`, `utils/`
- `environment.yml`

## Runtime inputs

- dataset and annotation paths
- checkpoint and vision-weight paths
- log/output directories

## Run

Adjust paths in your shell commands to your local dataset locations.

Example:

```bash
python train_simple.py --help
python test_simple.py --help
```
