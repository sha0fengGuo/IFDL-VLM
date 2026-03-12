# Stage 1

This folder contains Stage-1 training/testing code.

## Included

- `train_simple.py`, `train_simple.sh`
- `test_simple.py`, `test.sh`
- `model/`, `utils/`
- `environment.yml`

## Excluded

- datasets
- checkpoints/weights
- logs/cache files

## Run

Adjust paths in your shell commands to your local dataset locations.

Example:

```bash
python train_simple.py --help
python test_simple.py --help
```
