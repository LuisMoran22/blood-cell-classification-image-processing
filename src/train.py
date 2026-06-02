"""Training pipeline for blood cell classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from config import OUTPUT_DIR
from data_loader import load_dataset
from dataset_utils import save_dataset_summary
from model import build_cnn
from visualization import plot_confusion_matrix, plot_learning_curve


class MetricsCheckpoint(tf.keras.callbacks.Callback):
    """Save epoch logs to a numpy file."""

    def __init__(self, savepath: str | Path):
        super().__init__()
        self.savepath = Path(savepath)
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))
        self.savepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.savepath, self.history)


def train_model(
    data_dir: str | Path,
    task: str = "multiclass",
    epochs: int = 20,
    batch_size: int = 32,
    output_dir: str | Path = OUTPUT_DIR,
    limit_per_class: int | None = None,
):
    """Train and evaluate the model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_dataset_summary(data_dir, output_dir / "dataset_summary.json")

    X_train, y_train, X_test, y_test, class_names = load_dataset(
        data_dir,
        task=task,
        limit_per_class=limit_per_class,
    )

    num_classes = len(class_names)
    y_train_hot = to_categorical(y_train, num_classes=num_classes)
    y_test_hot = to_categorical(y_test, num_classes=num_classes)

    model = build_cnn(num_classes=num_classes)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )

    history = model.fit(
        datagen.flow(X_train, y_train_hot, batch_size=batch_size),
        steps_per_epoch=max(1, len(X_train) // batch_size),
        epochs=epochs,
        validation_data=(X_test, y_test_hot),
        callbacks=[
            early_stopping,
            reduce_lr,
            MetricsCheckpoint(output_dir / "training_logs.npy"),
        ],
    )

    score = model.evaluate(X_test, y_test_hot, verbose=0)
    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}")

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    print(report)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    cm = confusion_matrix(y_test, y_pred)

    plot_learning_curve(history, output_dir)
    plot_confusion_matrix(cm, class_names, output_dir)

    model.save(output_dir / "trained_model.keras")

    return model, history
