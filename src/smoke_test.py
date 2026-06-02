"""Fast non-TensorFlow smoke test for dataset reading and preprocessing."""
from __future__ import annotations
import argparse
import numpy as np
from data_loader import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Smoke test dataset loading without TensorFlow.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--task", choices=["multiclass", "nuclear"], default="multiclass")
    parser.add_argument("--samples-per-class", type=int, default=5)
    args = parser.parse_args()
    X_train, y_train, X_test, y_test, class_names = load_dataset(args.data_dir, task=args.task, limit_per_class=args.samples_per_class)
    print("Class names:", class_names)
    print("X_train:", X_train.shape, X_train.dtype, "range:", float(X_train.min()), float(X_train.max()))
    print("y_train:", y_train.shape, "labels:", sorted(np.unique(y_train).tolist()))
    print("X_test:", X_test.shape, X_test.dtype, "range:", float(X_test.min()), float(X_test.max()))
    print("y_test:", y_test.shape, "labels:", sorted(np.unique(y_test).tolist()))
    assert X_train.shape[1:] == (60, 80, 3)
    assert X_test.shape[1:] == (60, 80, 3)
    assert 0.0 <= float(X_train.min()) <= 1.0 and 0.0 <= float(X_train.max()) <= 1.0
    print("Smoke test passed.")
if __name__ == "__main__": main()
