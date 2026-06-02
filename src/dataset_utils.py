"""Dataset discovery and validation utilities."""
from __future__ import annotations
from pathlib import Path
import json
from config import CLASS_NAMES
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def resolve_images_dir(data_dir: str | Path) -> Path:
    """Find the folder that contains TRAIN and TEST directories."""
    data_dir = Path(data_dir)
    candidates = [
        data_dir,
        data_dir / "images",
        data_dir / "dataset2-master" / "images",
        data_dir / "dataset2-master" / "dataset2-master" / "images",
        data_dir / "dataset-master" / "images",
    ]
    for candidate in candidates:
        if (candidate / "TRAIN").exists() and (candidate / "TEST").exists():
            return candidate
    raise FileNotFoundError("Could not find a valid dataset folder containing TRAIN and TEST. Checked: " + str([str(c) for c in candidates]))

def count_images(class_dir: Path) -> int:
    if not class_dir.exists(): return 0
    return len([f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and not f.name.startswith('._')])

def summarize_dataset(data_dir: str | Path) -> dict:
    images_dir = resolve_images_dir(data_dir)
    summary = {"images_dir": str(images_dir), "splits": {}}
    splits = ["TRAIN", "TEST"]
    if (images_dir / "TEST_SIMPLE").exists():
        splits.append("TEST_SIMPLE")
    for split in splits:
        split_dir = images_dir / split
        summary["splits"][split] = {class_name: count_images(split_dir / class_name) for class_name in CLASS_NAMES}
    return summary

def save_dataset_summary(data_dir: str | Path, output_path: str | Path) -> dict:
    summary = summarize_dataset(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary
