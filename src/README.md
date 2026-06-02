# Source Code

This folder contains the Python implementation for the blood cell classification pipeline.

## Main Files

| File | Purpose |
|---|---|
| `check_dataset.py` | Validates dataset structure and image counts. |
| `smoke_test.py` | Runs a fast non-training test to verify image loading and preprocessing. |
| `config.py` | Stores project constants, image size, class names, and label mappings. |
| `dataset_utils.py` | Resolves dataset paths and creates dataset summaries. |
| `data_loader.py` | Loads, resizes, normalizes, and labels images. |
| `model.py` | Defines the CNN architecture. |
| `train.py` | Trains and evaluates the model. |
| `main.py` | Command-line training entry point. |
| `predict.py` | Runs inference on a single image. |
| `visualization.py` | Generates accuracy/loss curves and confusion matrices. |

## Quick Test

```bash
python src/check_dataset.py --data-dir data/dataset2-master
python src/smoke_test.py --data-dir data/dataset2-master --task multiclass --samples-per-class 5
```

## Train

```bash
python src/main.py --data-dir data/dataset2-master --task multiclass --epochs 20
```
