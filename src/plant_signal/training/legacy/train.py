"""Train the Keras model and save evaluation artifacts."""

from __future__ import annotations

import json
import os
import random
import argparse
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from plant_signal.config import (
    ARTIFACT_DIR,
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASS_NAMES,
    CONFUSION_MATRIX_PNG,
    DATA_CSV,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    INT8_MODEL_PATH,
    LEARNING_RATE,
    NUM_CLASSES,
    PREPROCESS_PATH,
    SEED,
    TRAINING_CURVES_PNG,
    TRAINING_HISTORY_CSV,
    WINDOW_SIZE,
)
from plant_signal.training.legacy.dataset import prepare_datasets
from plant_signal.training.legacy.model import build_model


def set_reproducible_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _trainable_parameter_count(model: tf.keras.Model) -> int:
    return int(sum(np.prod(weight.shape) for weight in model.trainable_weights))


def _file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def _plot_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    dataframe = pd.DataFrame(history.history)
    dataframe.to_csv(TRAINING_HISTORY_CSV, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    axes[0].plot(dataframe.index + 1, dataframe["accuracy"], label="train")
    axes[0].plot(dataframe.index + 1, dataframe["val_accuracy"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(dataframe.index + 1, dataframe["loss"], label="train")
    axes[1].plot(dataframe.index + 1, dataframe["val_loss"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> np.ndarray:
    labels = list(range(NUM_CLASSES))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[CLASS_NAMES[index] for index in labels],
    )
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=150)
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return matrix


def _warn_if_dataset_small(train_samples: int, session_count: int) -> None:
    if session_count < 6 or train_samples < 300:
        print(
            "WARNING: Dataset is small. Reported accuracy may be unstable; "
            "use more independent sessions for real experiments."
        )


def train_model(csv_path: Path = DATA_CSV) -> None:
    set_reproducible_seed()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = prepare_datasets(
        csv_path=csv_path,
        preprocess_path=PREPROCESS_PATH,
        fit_preprocess=True,
        save_fitted_preprocess=True,
        verbose=True,
    )

    _warn_if_dataset_small(len(bundle.y_train), len(bundle.train_sessions))

    model = build_model(input_shape=(WINDOW_SIZE, 1), num_classes=NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nModel summary:")
    model.summary()
    print(f"Total parameters: {model.count_params()}")
    print(f"Trainable parameters: {_trainable_parameter_count(model)}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        bundle.x_train,
        bundle.y_train,
        validation_data=(bundle.x_val, bundle.y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    # Keep the checkpoint and the in-memory restored weights aligned.
    model.save(BEST_MODEL_PATH)
    _plot_history(history, TRAINING_CURVES_PNG)

    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    train_loss, train_acc = best_model.evaluate(bundle.x_train, bundle.y_train, verbose=0)
    val_loss, val_acc = best_model.evaluate(bundle.x_val, bundle.y_val, verbose=0)
    test_loss, test_acc = best_model.evaluate(bundle.x_test, bundle.y_test, verbose=0)

    test_probs = best_model.predict(bundle.x_test, verbose=0)
    test_pred = np.argmax(test_probs, axis=1)
    matrix = _save_confusion_matrix(bundle.y_test, test_pred, CONFUSION_MATRIX_PNG)
    report = classification_report(
        bundle.y_test,
        test_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=[CLASS_NAMES[index] for index in range(NUM_CLASSES)],
        zero_division=0,
    )

    metrics_path = ARTIFACT_DIR / "train_metrics.json"
    metrics = {
        "train_accuracy": float(train_acc),
        "validation_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "train_loss": float(train_loss),
        "validation_loss": float(val_loss),
        "test_loss": float(test_loss),
        "total_parameters": int(best_model.count_params()),
        "trainable_parameters": _trainable_parameter_count(best_model),
        "train_samples": int(len(bundle.y_train)),
        "validation_samples": int(len(bundle.y_val)),
        "test_samples": int(len(bundle.y_test)),
        "train_sessions": len(bundle.train_sessions),
        "validation_sessions": len(bundle.val_sessions),
        "test_sessions": len(bundle.test_sessions),
        "confusion_matrix": matrix.tolist(),
        "preprocess": bundle.preprocess,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    keras_size = _file_size(BEST_MODEL_PATH)
    int8_size = _file_size(INT8_MODEL_PATH)

    print("\nAccuracy:")
    print(f"  train: {train_acc:.4f}")
    print(f"  val:   {val_acc:.4f}")
    print(f"  test:  {test_acc:.4f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix:")
    print(matrix)
    print(f"Keras model file size: {keras_size if keras_size is not None else 'not found'} bytes")
    print(f"INT8 TFLite model file size: {int8_size if int8_size is not None else 'not generated yet'}")
    print(f"Saved best model: {BEST_MODEL_PATH}")
    print(f"Saved preprocess parameters: {PREPROCESS_PATH}")
    print(f"Saved training curves: {TRAINING_CURVES_PNG}")
    print(f"Saved confusion matrix image: {CONFUSION_MATRIX_PNG}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Keras plant signal model.")
    parser.add_argument("--csv", type=Path, default=DATA_CSV, help="Training CSV with session_id,label,p0..p127.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(csv_path=args.csv)


if __name__ == "__main__":
    main()
