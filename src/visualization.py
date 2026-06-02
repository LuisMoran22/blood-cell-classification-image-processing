"""Visualization functions for model evaluation."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curve(history, output_dir: str | Path) -> None:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    acc_key = "accuracy" if "accuracy" in history.history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
    plt.figure(figsize=(8, 6)); plt.plot(history.history[acc_key], label="train"); plt.plot(history.history[val_acc_key], label="validation")
    plt.title("Model Accuracy"); plt.ylabel("Accuracy"); plt.xlabel("Epoch"); plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(output_dir / "accuracy_curve.png", dpi=300); plt.close()
    plt.figure(figsize=(8, 6)); plt.plot(history.history["loss"], label="train"); plt.plot(history.history["val_loss"], label="validation")
    plt.title("Model Loss"); plt.ylabel("Loss"); plt.xlabel("Epoch"); plt.legend(loc="upper right"); plt.tight_layout(); plt.savefig(output_dir / "loss_curve.png", dpi=300); plt.close()

def plot_confusion_matrix(cm, class_names, output_dir: str | Path) -> None:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label"); plt.xlabel("Predicted Label"); plt.title("Confusion Matrix"); plt.tight_layout(); plt.savefig(output_dir / "confusion_matrix.png", dpi=300); plt.close()
