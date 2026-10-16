"""Convert the trained Keras model to a fully INT8 TFLite model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from plant_signal.config import BEST_MODEL_PATH, DATA_CSV, INT8_MODEL_PATH, NUM_CLASSES, PREPROCESS_PATH, REPRESENTATIVE_DATASET_SIZE, WINDOW_SIZE
from plant_signal.training.legacy.dataset import prepare_datasets


def _representative_dataset(x_train: np.ndarray):
    count = min(REPRESENTATIVE_DATASET_SIZE, len(x_train))
    if count <= 0:
        raise ValueError("Representative dataset is empty.")

    def generator():
        for index in range(count):
            yield [x_train[index : index + 1].astype(np.float32)]

    return generator


def _fail_if_not_full_int8(interpreter: tf.lite.Interpreter) -> None:
    floating_tensors = []
    for tensor in interpreter.get_tensor_details():
        dtype = tensor.get("dtype")
        if dtype in (np.float16, np.float32, np.float64):
            floating_tensors.append((tensor.get("name", "<unnamed>"), str(dtype)))
    if floating_tensors:
        raise RuntimeError(f"TFLite model still contains floating-point tensors: {floating_tensors}")


def _verify_tflite_model(model_path: Path) -> tuple[dict, dict]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if input_details["dtype"] != np.int8:
        raise RuntimeError(f"Expected int8 input, got {input_details['dtype']}")
    if output_details["dtype"] != np.int8:
        raise RuntimeError(f"Expected int8 output, got {output_details['dtype']}")
    if list(input_details["shape"]) != [1, WINDOW_SIZE, 1]:
        raise RuntimeError(f"Expected input shape [1, {WINDOW_SIZE}, 1], got {input_details['shape']}")
    if int(output_details["shape"][-1]) != NUM_CLASSES:
        raise RuntimeError(f"Expected output class count {NUM_CLASSES}, got {output_details['shape']}")

    _fail_if_not_full_int8(interpreter)
    return input_details, output_details


def convert_model(csv_path: Path = DATA_CSV) -> None:
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
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset(bundle.x_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    INT8_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    INT8_MODEL_PATH.write_bytes(tflite_model)

    input_details, output_details = _verify_tflite_model(INT8_MODEL_PATH)
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    print("\nINT8 TFLite conversion succeeded.")
    print(f"Input scale: {input_scale}")
    print(f"Input zero_point: {input_zero_point}")
    print(f"Output scale: {output_scale}")
    print(f"Output zero_point: {output_zero_point}")
    print(f"TFLite model size: {INT8_MODEL_PATH.stat().st_size} bytes")
    print(f"Saved: {INT8_MODEL_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the Keras model to full INT8 TFLite.")
    parser.add_argument("--csv", type=Path, default=DATA_CSV, help="Training CSV for representative data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_model(csv_path=args.csv)


if __name__ == "__main__":
    main()
