# Stage 1

[English](README.md) · [中文](README_CN.md) · [项目复现说明](../REPRODUCTION_CN.md)

Stage 1 包含基于 SIDA 代码路径的图像伪造检测与定位训练、测试和数据工具。

## 包含内容

- `train_simple.py`、`train_simple.sh`
- `test_simple.py`、`test.sh`
- `model/`、`utils/`
- `environment.yml`、`requirements.txt`

## 不包含内容

- 数据集和私有标注
- checkpoint 和模型权重
- 日志、缓存和运行输出

## 运行前准备

先修改 `train_simple.sh` 和 `test.sh` 中的数据集、视觉预训练权重、checkpoint、图像大小以及 GPU 设置。具体参数和记录要求请看 [中文复现说明](../REPRODUCTION_CN.md)。

```bash
python train_simple.py --help
python test_simple.py --help
```
