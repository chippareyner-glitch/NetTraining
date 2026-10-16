"""Evaluate the saved Keras model on the session-held-out test split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from plant_signal.config import BEST_MODEL_PATH, CLASS_NAMES, DATA_CSV, NUM_CLASSES, PREPROCESS_PATH
from plant_signal.training.legacy.dataset import prepare_datasets


def evaluate_model(csv_path: Path = DATA_CSV) -> None:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Best Keras model not found: {BEST_MODEL_PATH}. Run train.py first.")

    bundle = prepare_datasets(
        csv_path=csv_path,
        preprocess_path=PREPROCESS_PATH,
        fit_preprocess=False,
        save_fitted_preprocess=False,
        verbose=True,
    )

    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    probs = model.predict(bundle.x_test, verbose=0)
    pred = np.argmax(probs, axis=1)

    accuracy = accuracy_score(bundle.y_test, pred)
    precision_macro = precision_score(bundle.y_test, pred, average="macro", zero_division=0)
    recall_macro = recall_score(bundle.y_test, pred, average="macro", zero_division=0)
    f1_macro = f1_score(bundle.y_test, pred, average="macro", zero_division=0)
    matrix = confusion_matrix(bundle.y_test, pred, labels=list(range(NUM_CLASSES)))

    print("\nKeras test metrics:")
    print(f"accuracy:  {accuracy:.4f}")
    print(f"precision: {precision_macro:.4f} (macro)")
    print(f"recall:    {recall_macro:.4f} (macro)")
    print(f"F1:        {f1_macro:.4f} (macro)")
    print("\nPer-class sample counts:")
    counts = np.bincount(bundle.y_test.astype(np.int64), minlength=NUM_CLASSES)
    for index in range(NUM_CLASSES):
        print(f"  {index} {CLASS_NAMES[index]}: {int(counts[index])}")
    print("\nConfusion matrix:")
    print(matrix)
    print("\nClassification report:")
    print(
        classification_report(
            bundle.y_test,
            pred,
            labels=list(range(NUM_CLASSES)),
            target_names=[CLASS_NAMES[index] for index in range(NUM_CLASSES)],
            zero_division=0,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved Keras model.")
    parser.add_argument("--csv", type=Path, default=DATA_CSV, help="Training/test CSV with session_id,label,p0..p127.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(csv_path=args.csv)


if __name__ == "__main__":
    main()
