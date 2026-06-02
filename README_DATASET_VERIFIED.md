# Blood Cell Classification - Dataset Verified Version

This version was tested against the uploaded `dataset2-master.zip` structure.

## Verified Dataset Counts

The uploaded dataset was correctly detected with this structure:

```text
dataset2-master/
└── images/
    ├── TRAIN/
    │   ├── EOSINOPHIL/   2497 images
    │   ├── LYMPHOCYTE/   2483 images
    │   ├── MONOCYTE/     2478 images
    │   └── NEUTROPHIL/   2499 images
    └── TEST/
        ├── EOSINOPHIL/    623 images
        ├── LYMPHOCYTE/    620 images
        ├── MONOCYTE/      620 images
        └── NEUTROPHIL/    624 images
```

Total verified images:

- TRAIN: 9,957
- TEST: 2,487
- Total: 12,444

The dataset also contains `TEST_SIMPLE`, but this implementation uses `TRAIN` and `TEST` by default.

## Recommended Environment

Use **Google Colab** or Python **3.10 / 3.11**. TensorFlow may not work correctly on unsupported Python versions.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Validate Dataset

```bash
python src/check_dataset.py --data-dir data/dataset2-master
```

## Run a Fast Smoke Test Without TensorFlow

This verifies paths, image loading, resizing, normalization, and labels without training the CNN.

```bash
python src/smoke_test.py --data-dir data/dataset2-master --samples-per-class 5
```

## Quick CNN Test

Use a small subset first:

```bash
python src/main.py --data-dir data/dataset2-master --task multiclass --epochs 1 --limit-per-class 50
```

## Full Training

```bash
python src/main.py --data-dir data/dataset2-master --task multiclass --epochs 20
```

## Binary Mononuclear vs Polynuclear Training

```bash
python src/main.py --data-dir data/dataset2-master --task nuclear --epochs 20
```

## Outputs

```text
outputs/
├── accuracy_curve.png
├── loss_curve.png
├── confusion_matrix.png
├── classification_report.txt
├── dataset_summary.json
├── training_logs.npy
└── trained_model.keras
```
