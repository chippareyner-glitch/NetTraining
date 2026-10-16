"""Run the INT8 TFLite model on the test split and compare with Keras."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from plant_signal.config import BEST_MODEL_PATH, CLASS_NAMES, DATA_CSV, INT8_MODEL_PATH, NUM_CLASSES, PREPROCESS_PATH
from plant_signal.training.legacy.dataset import prepare_datasets


def _quantize_input(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f"Input quantization scale must be positive, got {scale}")
    quantized = np.round(x / scale + zero_point)
    quantized = np.clip(quantized, -128, 127)
    return quantized.astype(np.int8)


def _dequantize_output(y: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f"Output quantization scale must be positive, got {scale}")
    return scale * (y.astype(np.float32) - float(zero_point))


def _run_tflite(interpreter: tf.lite.Interpreter, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    if input_details["dtype"] != np.int8 or output_details["dtype"] != np.int8:
        raise RuntimeError("TFLite model input and output must both be int8.")

    pred_labels: list[int] = []
    dequantized_outputs: list[np.ndarray] = []
    for sample in x_test:
        sample_batch = sample[np.newaxis, ...].astype(np.float32)
        quantized = _quantize_input(sample_batch, float(input_scale), int(input_zero_point))
        interpreter.set_tensor(input_details["index"], quantized)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details["index"])
        output = _dequantize_output(raw_output, float(output_scale), int(output_zero_point))
        dequantized_outputs.append(output[0])
        pred_labels.append(int(np.argmax(output[0])))

    return np.array(pred_labels, dtype=np.int64), np.vstack(dequantized_outputs)


def test_tflite_model(csv_path: Path = DATA_CSV) -> None:
    if not INT8_MODEL_PATH.exists():
        raise FileNotFoundError(f"INT8 model not found: {INT8_MODEL_PATH}. Run convert_int8.py first.")
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Best Keras model not found: {BEST_MODEL_PATH}. Run train.py first.")

    bundle = prepare_datasets(
        csv_path=csv_path,
        preprocess_path=PREPROCESS_PATH,
        fit_preprocess=False,
        save_fitted_preprocess=False,
        verbose=True,
    )

    keras_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    keras_probs = keras_model.predict(bundle.x_test, verbose=0)
    keras_pred = np.argmax(keras_probs, axis=1)
    keras_acc = accuracy_score(bundle.y_test, keras_pred)

    interpreter = tf.lite.Interpreter(model_path=str(INT8_MODEL_PATH))
    interpreter.allocate_tensors()
    tflite_pred, _ = _run_tflite(interpreter, bundle.x_test)
    tflite_acc = accuracy_score(bundle.y_test, tflite_pred)
    consistency = accuracy_score(keras_pred, tflite_pred)
    matrix = confusion_matrix(bundle.y_test, tflite_pred, labels=list(range(NUM_CLASSES)))

    print("\nINT8 TFLite test metrics:")
    print(f"Keras accuracy:       {keras_acc:.4f}")
    print(f"INT8 TFLite accuracy: {tflite_acc:.4f}")
    print(f"Keras/INT8 prediction consistency: {consistency:.4f}")
    if tflite_acc + 0.05 < keras_acc:
        print("WARNING: INT8 accuracy dropped by more than 0.05 compared with Keras.")

    print("\nConfusion matrix:")
    print(matrix)
    print("\nClassification report:")
    print(
        classification_report(
            bundle.y_test,
            tflite_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=[CLASS_NAMES[index] for index in range(NUM_CLASSES)],
            zero_division=0,
        )
    )

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print("Input quantization:", input_details["quantization"])
    print("Output quantization:", output_details["quantization"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the full INT8 TFLite model.")
    parser.add_argument("--csv", type=Path, default=DATA_CSV, help="Training/test CSV with session_id,label,p0..p127.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_tflite_model(csv_path=args.csv)


if __name__ == "__main__":
    main()
