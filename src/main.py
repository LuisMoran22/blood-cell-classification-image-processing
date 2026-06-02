"""Command-line script to train the blood cell CNN."""

from __future__ import annotations

import argparse

from train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN for blood cell image classification.")
    parser.add_argument("--data-dir", required=True, help="Path to dataset root or images folder.")
    parser.add_argument("--task", choices=["multiclass", "nuclear"], default="multiclass")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--limit-per-class", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        data_dir=args.data_dir,
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        limit_per_class=args.limit_per_class,
    )


if __name__ == "__main__":
    main()
