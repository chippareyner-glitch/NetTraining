"""Pure NumPy preprocessing and quantization shared by realtime monitors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

DEFAULT_WINDOW_LEN = 128


def infer_model_time_len(input_details: dict[str, Any] | None) -> int:
    if not input_details:
        return DEFAULT_WINDOW_LEN
    shape = [int(dim) for dim in list(input_details.get("shape", []))]
    dims = [dim for dim in shape if dim > 0]
    if len(dims) >= 3 and dims[-1] == 1:
        return int(dims[-2])
    if len(dims) >= 2:
        return int(dims[-1] if dims[0] == 1 else dims[0])
    if len(dims) == 1:
        return int(dims[0])
    return DEFAULT_WINDOW_LEN


def infer_default_window_len(input_details: dict[str, Any] | None, preprocess_config: dict[str, Any]) -> int:
    raw_window_size = preprocess_config.get("raw_window_size")
    if raw_window_size:
        return int(raw_window_size)
    return infer_model_time_len(input_details)


def median_filter_3(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return values.astype(np.float32)
    padded = np.pad(values.astype(np.float32), (1, 1), mode="edge")
    stacked = np.vstack([padded[:-2], padded[1:-1], padded[2:]])
    return np.median(stacked, axis=0).astype(np.float32)


def resize_1d(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if target_len <= 0:
        return values.astype(np.float32)
    if values.size == target_len:
        return values.astype(np.float32)
    if values.size == 0:
        return np.zeros((target_len,), dtype=np.float32)
    if values.size == 1:
        return np.full((target_len,), float(values[0]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=values.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(x_new, x_old, values).astype(np.float32)


def preprocess_window(raw_window: np.ndarray, target_len: int, config: dict[str, Any]) -> np.ndarray:
    values = np.asarray(raw_window, dtype=np.float32).reshape(-1)

    # Prefer the exported training preprocess config when it is present.
    if config:
        if bool(config.get("median_filter_3", False)):
            values = median_filter_3(values)

        downsample_factor = int(config.get("downsample_factor") or 1)
        downsample_method = str(config.get("downsample_method") or "mean").lower()
        if downsample_factor > 1 and values.size >= downsample_factor:
            usable = (values.size // downsample_factor) * downsample_factor
            grouped = values[:usable].reshape(-1, downsample_factor)
            if downsample_method == "mean":
                values = grouped.mean(axis=1).astype(np.float32)
            else:
                values = grouped[:, 0].astype(np.float32)

        baseline_points = int(config.get("baseline_points") or min(16, max(1, values.size)))
        baseline_points = max(1, min(baseline_points, values.size))
        baseline = float(np.median(values[:baseline_points]))
        values = values - baseline

        unit_text = str(config.get("unit", "")).lower()
        if "mv" in unit_text:
            values = values / 1000.0

        scale = float(config.get("scale_mv_abs_p99_train") or config.get("scale_abs_p99_train") or 0.0)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = float(np.std(values))
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0
        values = values / scale

        clip_range = config.get("clip_range") or [-1.0, 1.0]
        try:
            low, high = float(clip_range[0]), float(clip_range[1])
        except Exception:
            low, high = -1.0, 1.0
        values = np.clip(values, low, high)
        return resize_1d(values, target_len).astype(np.float32)

    # Fallback realtime normalization when training preprocess is unavailable.
    median = float(np.median(values)) if values.size else 0.0
    values = values - median
    std = float(np.std(values))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    values = np.clip(values / std, -6.0, 6.0)
    return resize_1d(values, target_len).astype(np.float32)


def model_input_array(processed: np.ndarray, input_details: dict[str, Any]) -> np.ndarray:
    shape = [int(dim) for dim in list(input_details.get("shape", []))]
    if not shape:
        shape = [1, processed.size]
    concrete_shape = [1 if dim <= 0 else dim for dim in shape]

    if len(concrete_shape) == 3:
        arr = processed.reshape(1, concrete_shape[1], concrete_shape[2])
    elif len(concrete_shape) == 2:
        arr = processed.reshape(1, concrete_shape[1])
    elif len(concrete_shape) == 1:
        arr = processed.reshape(concrete_shape)
    else:
        arr = processed.reshape(concrete_shape)

    dtype = input_details["dtype"]
    if np.issubdtype(dtype, np.floating):
        return arr.astype(dtype)

    scale, zero_point = input_details.get("quantization", (0.0, 0))
    scale = float(scale or 0.0)
    zero_point = int(zero_point or 0)
    if scale <= 0:
        return arr.astype(dtype)

    info = np.iinfo(dtype)
    quantized = np.round(arr / scale + zero_point)
    quantized = np.clip(quantized, info.min, info.max)
    return quantized.astype(dtype)


def dequantize_output(raw_output: np.ndarray, output_details: dict[str, Any]) -> np.ndarray:
    dtype = output_details["dtype"]
    output = np.asarray(raw_output)
    if np.issubdtype(dtype, np.floating):
        return output.astype(np.float32)
    scale, zero_point = output_details.get("quantization", (0.0, 0))
    scale = float(scale or 0.0)
    zero_point = int(zero_point or 0)
    if scale <= 0:
        return output.astype(np.float32)
    return scale * (output.astype(np.float32) - float(zero_point))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def impact_score_from_output(output: np.ndarray) -> float:
    values = np.asarray(output, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    if values.size == 1:
        value = float(values[0])
        if 0.0 <= value <= 1.0:
            return value
        return sigmoid(value)

    rest_impact = values[:2].astype(np.float32)
    if np.all(rest_impact >= -1e-5) and np.all(rest_impact <= 1.00001):
        return float(rest_impact[1])
    shifted = rest_impact - float(np.max(rest_impact))
    exp_values = np.exp(shifted)
    probs = exp_values / float(np.sum(exp_values))
    return float(probs[1])
