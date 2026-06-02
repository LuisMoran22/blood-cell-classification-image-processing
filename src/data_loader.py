"""Data loading utilities adapted for the dataset2-master image structure."""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
from tqdm import tqdm
from config import CLASS_NAMES, CLASS_TO_INDEX, IMAGE_SIZE, NUCLEAR_GROUPS
from dataset_utils import resolve_images_dir, IMAGE_EXTENSIONS

def _valid_image_files(class_dir: Path, limit_per_class: int | None = None):
    files = sorted([f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and not f.name.startswith('._')])
    if limit_per_class is not None:
        files = files[:limit_per_class]
    return files

def load_split(split_dir: str | Path, task: str = "multiclass", limit_per_class: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load images from one split directory: TRAIN or TEST."""
    split_dir = Path(split_dir)
    X, y = [], []
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            print(f"Warning: missing class folder: {class_dir}")
            continue
        image_files = _valid_image_files(class_dir, limit_per_class=limit_per_class)
        for image_path in tqdm(image_files, desc=f"Loading {split_dir.name}/{class_name}"):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: could not read image: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)
            label = NUCLEAR_GROUPS[class_name] if task == "nuclear" else CLASS_TO_INDEX[class_name]
            X.append(image); y.append(label)
    if not X:
        raise ValueError(f"No images were loaded from {split_dir}")
    X = np.asarray(X, dtype=np.float32) / 255.0
    y = np.asarray(y, dtype=np.int64)
    return X, y

def load_dataset(data_dir: str | Path, task: str = "multiclass", limit_per_class: int | None = None):
    """Load TRAIN and TEST data from the dataset root."""
    images_dir = resolve_images_dir(data_dir)
    X_train, y_train = load_split(images_dir / "TRAIN", task=task, limit_per_class=limit_per_class)
    X_test, y_test = load_split(images_dir / "TEST", task=task, limit_per_class=limit_per_class)
    label_names = ["Mononuclear", "Polynuclear"] if task == "nuclear" else CLASS_NAMES
    return X_train, y_train, X_test, y_test, label_names
