"""Predict the class of a single blood cell image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import CLASS_NAMES, IMAGE_SIZE


def load_image(image_path: str | Path):
    """Load, resize, and normalize one image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32") / 255.0

    return np.expand_dims(image, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one blood cell image.")
    parser.add_argument("--model", required=True, help="Path to trained .keras model.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--classes", nargs="+", default=CLASS_NAMES, help="Class names in training order.")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    image = load_image(args.image)

    probabilities = model.predict(image)[0]
    class_index = int(np.argmax(probabilities))

    print(f"Predicted class: {args.classes[class_index]}")
    print(f"Confidence: {probabilities[class_index]:.4f}")


if __name__ == "__main__":
    main()
