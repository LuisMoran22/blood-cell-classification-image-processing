# Blood Cell Classification using Biomedical Image Processing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Project Type](https://img.shields.io/badge/Project-Biomedical%20Image%20Processing-lightgrey)](#)

This repository presents a biomedical engineering project focused on **blood smear microscopy image classification** using image processing and deep learning.

The project classifies white blood cell microscopy images into four categories and also supports a secondary binary classification task based on nuclear morphology.

---

## Project Overview

The objective of this project is to support automated blood cell analysis through biomedical image processing and convolutional neural networks.

### Classification Tasks

#### 1. Four-class white blood cell classification

- Eosinophil
- Lymphocyte
- Monocyte
- Neutrophil

#### 2. Binary nuclear morphology classification

- Mononuclear cells
- Polynuclear cells

This project is relevant to biomedical engineering because it addresses automated microscopy image interpretation, a task associated with hematology support systems, digital pathology, and AI-assisted clinical workflows.

---

## Dataset

This project uses the **Blood Cell Images** dataset by Paul Mooney, available on Kaggle.

The dataset contains approximately 12,500 augmented blood cell images distributed across four white blood cell classes:

- `EOSINOPHIL`
- `LYMPHOCYTE`
- `MONOCYTE`
- `NEUTROPHIL`

Expected local dataset structure:

```text
data/
└── dataset2-master/
    └── dataset2-master/
        └── images/
            ├── TRAIN/
            │   ├── EOSINOPHIL/
            │   ├── LYMPHOCYTE/
            │   ├── MONOCYTE/
            │   └── NEUTROPHIL/
            └── TEST/
                ├── EOSINOPHIL/
                ├── LYMPHOCYTE/
                ├── MONOCYTE/
                └── NEUTROPHIL/
```

The full dataset is not included in this repository. Please download it from the original dataset source and place it locally inside the `data/` folder.

---

## Verified Dataset Counts

The dataset structure was verified with the following image counts:

| Split | Eosinophil | Lymphocyte | Monocyte | Neutrophil | Total |
|---|---:|---:|---:|---:|---:|
| Train | 2497 | 2483 | 2478 | 2499 | 9957 |
| Test | 623 | 620 | 620 | 624 | 2487 |
| **Total** | **3120** | **3103** | **3098** | **3123** | **12444** |

---

## Methodology

The implemented workflow includes:

1. Loading microscopy images from the dataset folders.
2. Resizing images to `80 x 60` pixels.
3. Converting images from BGR to RGB.
4. Normalizing pixel values to the range `[0, 1]`.
5. Encoding labels for multiclass or binary classification.
6. Training a convolutional neural network using TensorFlow/Keras.
7. Evaluating performance with accuracy, loss curves, classification reports, and confusion matrices.

---

## Model Architecture

The CNN model includes:

- Convolutional layers
- Batch normalization
- Max pooling
- Dropout regularization
- Dense classification layers
- Softmax output

The model is implemented in:

```text
src/model.py
```

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── assets/
│   ├── figures/
│   └── results/
├── data/
│   └── README.md
├── docs/
│   ├── dataset_notes.md
│   ├── methodology.md
│   └── project_summary.md
├── models/
├── notebooks/
├── outputs/
│   └── README.md
└── src/
    ├── README.md
    ├── __init__.py
    ├── check_dataset.py
    ├── config.py
    ├── data_loader.py
    ├── dataset_utils.py
    ├── main.py
    ├── model.py
    ├── predict.py
    ├── smoke_test.py
    ├── train.py
    └── visualization.py
```

---

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

If OpenCV fails on Windows, reinstall it with:

```bash
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-python
```

---

## Usage

### 1. Check dataset structure

```bash
python src/check_dataset.py --data-dir data/dataset2-master
```

### 2. Run a fast smoke test

```bash
python src/smoke_test.py --data-dir data/dataset2-master --task multiclass --samples-per-class 5
```

### 3. Train the four-class model

```bash
python src/main.py --data-dir data/dataset2-master --task multiclass --epochs 20
```

### 4. Train the binary nuclear morphology model

```bash
python src/main.py --data-dir data/dataset2-master --task nuclear --epochs 20
```

### 5. Predict a single image

```bash
python src/predict.py --model outputs/trained_model.keras --image path/to/image.jpeg
```

---

## Outputs

After training, the project generates:

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

---

## Visual Results

### Example Cell Classes

![Four cell class examples](assets/figures/four_cell_classes_examples.png)

### Preprocessing Example

![Preprocessing and histogram](assets/figures/preprocessing_histogram.png)

### Classification Output Example

![Detection examples](assets/figures/four_class_detection_examples.png)

---

## Technical Skills Demonstrated

- Biomedical image processing
- Microscopy image analysis
- Deep learning with CNNs
- TensorFlow/Keras model training
- Image preprocessing with OpenCV
- Model evaluation with confusion matrices and classification reports
- Healthcare-oriented AI workflow

---

## Future Work

- Add Grad-CAM or saliency map visualizations.
- Compare CNN results with transfer learning architectures.
- Add a lightweight web demo for image inference.
- Improve model reproducibility with fixed seeds and experiment tracking.
- Evaluate model performance on external blood smear datasets.

---

## License

This repository is licensed under the MIT License.

Dataset rights belong to the original dataset creators. Please refer to the original dataset page for usage rights and redistribution conditions.
