"""One-command pipeline to rebuild the training dataset and update model artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from plant_signal.config import (
    ARTIFACT_DIR,
    C_MODEL_CC_PATH,
    C_MODEL_H_PATH,
    INT8_MODEL_PATH,
    REAL_DATA_CSV,
    ROOT_DIR,
    WINDOW_SIZE,
)


PYTHON = Path(sys.executable)


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def summarize_dataset(csv_path: Path) -> None:
    dataframe = pd.read_csv(csv_path)
    counts = dataframe["label"].value_counts().sort_index().to_dict()
    session_count = int(dataframe["session_id"].nunique())
    print("\n=== Dataset summary ===")
    print(f"CSV: {csv_path}")
    print(f"Rows: {len(dataframe)}")
    print(f"Sessions: {session_count}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Class counts: {counts}")
    if int(counts.get(2, 0)) == 0:
        print("WARNING: HoldTouch(label=2) has 0 windows. The updated model will not learn touch yet.")


def summarize_artifacts() -> None:
    metrics_path = ARTIFACT_DIR / "train_metrics.json"
    print("\n=== Updated artifacts ===")
    for path in [metrics_path, INT8_MODEL_PATH, C_MODEL_CC_PATH, C_MODEL_H_PATH]:
        if path.exists():
            print(f"{path}: {path.stat().st_size} bytes")
        else:
            print(f"{path}: missing")

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print("\n=== Accuracy ===")
        print(f"train: {float(metrics.get('train_accuracy', 0.0)):.4f}")
        print(f"val:   {float(metrics.get('validation_accuracy', 0.0)):.4f}")
        print(f"test:  {float(metrics.get('test_accuracy', 0.0)):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild dataset, train, quantize, test, and export MCU model files.")
    parser.add_argument("--output", type=Path, default=REAL_DATA_CSV, help="Window dataset CSV to generate and train on.")
    parser.add_argument("--event-step", type=int, default=WINDOW_SIZE, help="Tap/Touch window step.")
    parser.add_argument("--rest-step", type=int, default=WINDOW_SIZE, help="Rest window step.")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip Keras evaluate.py after training.")
    parser.add_argument("--skip-tflite-test", action="store_true", help="Skip test_tflite.py after conversion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output

    run_step("Sync completed labels into data_labeled", [str(PYTHON), "-m", "plant_signal.annotation.export"])
    run_step(
        "Build window dataset",
        [
            str(PYTHON),
            "-m", "plant_signal.annotation.windows",
            "--output",
            str(output_csv),
            "--event-step",
            str(args.event_step),
            "--rest-step",
            str(args.rest_step),
        ],
    )
    summarize_dataset(output_csv)
    run_step("Train Keras model", [str(PYTHON), "-m", "plant_signal.training.legacy.train", "--csv", str(output_csv)])
    if not args.skip_evaluate:
        run_step("Evaluate Keras model", [str(PYTHON), "-m", "plant_signal.training.legacy.evaluate", "--csv", str(output_csv)])
    run_step("Convert to full INT8 TFLite", [str(PYTHON), "-m", "plant_signal.training.legacy.quantize", "--csv", str(output_csv)])
    if not args.skip_tflite_test:
        run_step("Test INT8 TFLite model", [str(PYTHON), "-m", "plant_signal.training.legacy.tflite_test", "--csv", str(output_csv)])
    run_step("Export TFLite model to C arrays", [str(PYTHON), "-m", "plant_signal.training.legacy.export_c"])
    summarize_artifacts()


if __name__ == "__main__":
    main()
