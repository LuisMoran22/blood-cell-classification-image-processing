# Methodology

This document describes the project workflow based on the available exported results.

## 1. Problem Definition

Manual blood smear analysis is a relevant task in biomedical and clinical contexts. This project aimed to support the automated identification and classification of white blood cells from microscopy images.

## 2. Input Data

The available figures show blood smear microscopy images containing white blood cells and red blood cells. The repository currently includes exported result images, but not the raw dataset.

## 3. Preprocessing

The uploaded archive includes a preprocessing visualization with an input blood smear sample and a pixel intensity histogram. This suggests that intensity distribution analysis was used to understand image characteristics before segmentation or classification.

Potential preprocessing operations may include:

- Image resizing
- Color or intensity normalization
- Noise reduction
- Contrast adjustment
- Region-of-interest extraction

These steps should be confirmed once the original code is added.

## 4. Detection and Region Extraction

The output visualizations show multiple regions marked with bounding boxes, indicating that the workflow included a cell localization or candidate-region extraction stage.

## 5. Classification Tasks

Two classification setups are documented in the available results:

### Four-class classification

Classes:

- Neutrophil
- Eosinophil
- Monocyte
- Lymphocyte

### Binary classification

Classes:

- Mononuclear
- Polynuclear

## 6. Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

Representative four-class experiments show accuracies of approximately 0.83–0.84 in the included classification report screenshots. Binary mononuclear/polynuclear experiments show stronger performance in the available screenshots, with representative accuracies around 0.89–0.96 depending on the run.

## 7. Reproducibility Notes

The current repository is not yet fully reproducible because the source code, raw data, model files, and environment configuration are missing from the uploaded archive.

To complete the methodology section, add:

- Original code
- Model type or algorithm used
- Dataset source and labeling process
- Training/validation/test split
- Hyperparameters
- Preprocessing parameters
- Exact dependency versions
