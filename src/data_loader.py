"""Data loading utilities adapted for the dataset2-master image structure."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

from config import CLASS_NAMES, CLASS_TO_INDEX, IMAGE_SIZE, NUCLEAR_GROUPS
from dataset_utils import resolve_images_dir


def _image_files(class_dir: Path) -> list[Path]:
    return sorted([
        file for file in class_dir.iterdir()
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ])


def load_split(
    split_dir: str | Path,
    task: str = "multiclass",
    limit_per_class: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load images from one split directory: TRAIN or TEST."""
    split_dir = Path(split_dir)
    X, y = [], []

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name

        if not class_dir.exists():
            print(f"Warning: missing class folder: {class_dir}")
            continue

        image_files = _image_files(class_dir)
        if limit_per_class is not None:
            image_files = image_files[:limit_per_class]

        for image_path in tqdm(image_files, desc=f"Loading {split_dir.name}/{class_name}"):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: could not read image: {image_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)

            if task == "nuclear":
                label = NUCLEAR_GROUPS[class_name]
            else:
                label = CLASS_TO_INDEX[class_name]

            X.append(image)
            y.append(label)

    if not X:
        raise ValueError(f"No images were loaded from {split_dir}")

    X = np.asarray(X, dtype=np.float32) / 255.0
    y = np.asarray(y, dtype=np.int64)

    return X, y


def load_dataset(
    data_dir: str | Path,
    task: str = "multiclass",
    limit_per_class: int | None = None,
):
    """Load TRAIN and TEST data from the dataset root."""
    images_dir = resolve_images_dir(data_dir)

    X_train, y_train = load_split(images_dir / "TRAIN", task=task, limit_per_class=limit_per_class)
    X_test, y_test = load_split(images_dir / "TEST", task=task, limit_per_class=limit_per_class)

    label_names = ["Mononuclear", "Polynuclear"] if task == "nuclear" else CLASS_NAMES

    return X_train, y_train, X_test, y_test, label_names
