"""Dataset discovery and validation utilities."""

from __future__ import annotations

from pathlib import Path
import json

from config import CLASS_NAMES


def resolve_images_dir(data_dir: str | Path) -> Path:
    """Find the folder that contains TRAIN and TEST directories."""
    data_dir = Path(data_dir)

    candidates = [
        data_dir,
        data_dir / "images",
        data_dir / "dataset2-master" / "images",
        data_dir / "dataset2-master" / "dataset2-master" / "images",
    ]

    for candidate in candidates:
        if (candidate / "TRAIN").exists() and (candidate / "TEST").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find a valid dataset folder containing TRAIN and TEST. "
        f"Checked: {[str(candidate) for candidate in candidates]}"
    )


def summarize_dataset(data_dir: str | Path) -> dict:
    """Return a summary of available images by split and class."""
    images_dir = resolve_images_dir(data_dir)
    summary = {"images_dir": str(images_dir), "splits": {}}

    for split in ["TRAIN", "TEST", "TEST_SIMPLE"]:
        split_dir = images_dir / split
        if not split_dir.exists():
            continue

        summary["splits"][split] = {}
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = len([
                    file for file in class_dir.iterdir()
                    if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                ])
            else:
                count = 0
            summary["splits"][split][class_name] = count

    return summary


def save_dataset_summary(data_dir: str | Path, output_path: str | Path) -> dict:
    """Save dataset summary to JSON."""
    summary = summarize_dataset(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary
