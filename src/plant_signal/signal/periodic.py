"""周期干扰估计、模板生成和模板抵消的纯算法函数。

本模块不依赖 Streamlit，后续可以直接集成到其他采集或标注页面。
所有函数都会复制输入数据并返回新数组，绝不原地覆盖原始采集数据。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12


def _as_float_array(values: Any, *, name: str = "values") -> np.ndarray:
    """转换为一维 float 数组，并用中位数替换非有限值。"""

    array = np.asarray(values, dtype=float).reshape(-1).copy()
    if array.size == 0:
        raise ValueError(f"{name} 为空，无法分析。")

    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError(f"{name} 中没有有效数值。")

    if not finite.all():
        fill_value = float(np.median(array[finite]))
        array[~finite] = fill_value

    return array


def median_filter_3(values: Any) -> np.ndarray:
    """3 点中值滤波，仅用于可选的孤立毛刺抑制，不覆盖原始数据。"""

    array = _as_float_array(values)
    if array.size < 3:
        return array.copy()
    padded = np.pad(array, (1, 1), mode="edge")
    windows = np.column_stack([padded[:-2], padded[1:-1], padded[2:]])
    return np.median(windows, axis=1)


def _odd_window(samples: int, minimum: int = 3) -> int:
    samples = max(minimum, int(round(samples)))
    if samples % 2 == 0:
        samples += 1
    return samples


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """使用 pandas 的滚动中位数去除慢趋势，避免手写低效循环。"""

    window = _odd_window(window)
    series = pd.Series(values)
    return series.rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)


def _detrend_for_period(values: np.ndarray, fs: float, max_period_s: float) -> np.ndarray:
    centered = values - float(np.median(values))
    trend_window = _odd_window(max_period_s * fs * 4.0)
    slow_trend = _rolling_median(centered, trend_window)
    detrended = centered - slow_trend
    return detrended - float(np.median(detrended))


def _autocorrelation_fft(values: np.ndarray, max_lag: int) -> np.ndarray:
    """FFT 自相关，适合 60～120 秒这类较长静息数据。"""

    centered = values - float(np.mean(values))
    n = centered.size
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, fft_size)
    corr = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_size)[: max_lag + 1]
    counts = np.arange(n, n - max_lag - 1, -1, dtype=float)
    corr = corr / np.maximum(counts, 1.0)
    if abs(corr[0]) < EPS:
        raise ValueError("信号能量过低，无法估计自相关周期。")
    return corr / corr[0]


def _local_peak_indices(y: np.ndarray) -> np.ndarray:
    if y.size < 3:
        return np.array([], dtype=int)
    middle = y[1:-1]
    mask = (middle > y[:-2]) & (middle >= y[2:])
    return np.flatnonzero(mask) + 1


def estimate_period(
    values,
    fs,
    min_period_s=0.5,
    max_period_s=3.0,
):
    """
    返回周期估计结果和诊断信息。

    返回值为 (result, diagnostics)。result 是可 JSON 化的摘要，
    diagnostics 包含绘图需要的去趋势波形、自相关曲线和候选峰。
    """

    fs = float(fs)
    min_period_s = float(min_period_s)
    max_period_s = float(max_period_s)
    if fs <= 0:
        raise ValueError("采样率 fs 必须大于 0。")
    if min_period_s <= 0 or max_period_s <= min_period_s:
        raise ValueError("周期搜索范围不合法，请确认最小周期和最大周期。")

    raw = _as_float_array(values)
    min_lag = max(2, int(round(min_period_s * fs)))
    max_lag = min(raw.size - 2, int(round(max_period_s * fs)))
    if max_lag <= min_lag:
        raise ValueError("数据长度不足，无法覆盖指定的周期搜索范围。")

    detrended = _detrend_for_period(raw, fs, max_period_s)
    if float(np.std(detrended)) < EPS:
        raise ValueError("去趋势后的信号变化过小，无法估计周期。")

    autocorr = _autocorrelation_fft(detrended, max_lag)
    search = autocorr[min_lag : max_lag + 1]
    peak_indices = _local_peak_indices(search) + min_lag
    if peak_indices.size == 0:
        best_lag = int(np.argmax(search) + min_lag)
        peak_indices = np.array([best_lag], dtype=int)
    else:
        best_lag = int(peak_indices[np.argmax(autocorr[peak_indices])])

    strength = float(autocorr[best_lag])
    useful_peak_mask = autocorr[peak_indices] >= max(0.10, 0.35 * strength)
    useful_peaks = peak_indices[useful_peak_mask]
    spacings = np.diff(useful_peaks).astype(float) if useful_peaks.size >= 2 else np.array([], dtype=float)

    if spacings.size:
        spacing_median = float(np.median(spacings))
        spacing_std = float(np.std(spacings, ddof=1)) if spacings.size > 1 else 0.0
        period_cv = float(spacing_std / spacing_median) if spacing_median > EPS else 0.0
    else:
        spacing_median = float(best_lag)
        spacing_std = 0.0
        period_cv = 0.0

    # 自相关最高峰作为主估计；峰间距作为稳定性诊断。
    period_samples = max(2, int(round(best_lag)))
    period_seconds = float(period_samples / fs)
    result = {
        "period_seconds": period_seconds,
        "period_samples": int(period_samples),
        "frequency_hz": float(fs / period_samples),
        "autocorrelation_strength": strength,
        "peak_spacing_median_samples": spacing_median,
        "peak_spacing_std_samples": spacing_std,
        "period_cv": period_cv,
        "period_lengths_samples": spacings.tolist(),
        "warning": "当前周期漂移较大，固定模板抵消可能效果不稳定。" if period_cv > 0.10 else "",
    }

    diagnostics = {
        "fs": fs,
        "detrended_values": detrended,
        "autocorrelation_lags": np.arange(autocorr.size, dtype=int),
        "autocorrelation": autocorr,
        "peak_lags": peak_indices,
        "useful_peak_lags": useful_peaks,
        "min_lag": int(min_lag),
        "max_lag": int(max_lag),
    }
    return result, diagnostics


def _resample_template(template: np.ndarray, target_len: int) -> np.ndarray:
    target_len = max(2, int(target_len))
    if target_len == template.size:
        return template.copy()
    src_x = np.linspace(0.0, 1.0, template.size, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
    extended_x = np.r_[src_x, 1.0]
    extended_y = np.r_[template, template[0]]
    resized = np.interp(dst_x, extended_x, extended_y)
    return resized - float(np.mean(resized))


def _periodic_reference(
    length: int,
    template: np.ndarray,
    phase_shift_samples: int = 0,
    period_samples: int | None = None,
    start_index: int = 0,
) -> np.ndarray:
    """按全局样本索引生成重复模板，避免假设每段数据都从相位 0 开始。"""

    period_samples = int(period_samples or template.size)
    one_period = _resample_template(template, period_samples)
    indices = (np.arange(length, dtype=int) + int(start_index) + int(phase_shift_samples)) % period_samples
    return one_period[indices]


def _best_roll_shift(segment: np.ndarray, reference: np.ndarray, max_shift: int) -> int:
    segment_centered = segment - float(np.median(segment))
    reference_centered = reference - float(np.median(reference))
    if float(np.std(segment_centered)) < EPS or float(np.std(reference_centered)) < EPS:
        return 0

    best_shift = 0
    best_score = -np.inf
    for shift in range(-max_shift, max_shift + 1):
        shifted = np.roll(segment_centered, shift)
        score = float(np.dot(shifted, reference_centered))
        if score > best_score:
            best_score = score
            best_shift = shift
    return int(best_shift)


def build_period_template(
    values,
    fs,
    period_samples,
    align_ratio=0.1,
):
    """
    将静息数据切成周期、互相关对齐，
    并逐点取中位数生成模板。

    返回值为 (model, diagnostics)。model 可传给 remove_periodic_noise。
    """

    fs = float(fs)
    period_samples = int(round(period_samples))
    align_ratio = float(align_ratio)
    if fs <= 0:
        raise ValueError("采样率 fs 必须大于 0。")
    if period_samples < 3:
        raise ValueError("周期点数过少，无法生成模板。")

    raw = _as_float_array(values)
    usable_len = (raw.size // period_samples) * period_samples
    if usable_len < period_samples * 3:
        raise ValueError("静息数据不足 3 个完整周期，无法生成稳健模板。")

    trend = _rolling_median(raw, period_samples * 5)
    detrended = raw - trend
    detrended = detrended - float(np.median(detrended))
    trimmed = detrended[:usable_len]
    segments = trimmed.reshape(-1, period_samples)

    rough_reference = np.median(segments, axis=0)
    max_shift = max(1, int(round(period_samples * align_ratio)))
    shifts: list[int] = []
    aligned_segments = []
    for segment in segments:
        shift = _best_roll_shift(segment, rough_reference, max_shift)
        shifts.append(shift)
        aligned_segments.append(np.roll(segment, shift))

    aligned = np.asarray(aligned_segments, dtype=float)
    template = np.median(aligned, axis=0)
    template = template - float(np.mean(template))

    model = {
        "fs": fs,
        "period_samples": int(period_samples),
        "period_seconds": float(period_samples / fs),
        "frequency_hz": float(fs / period_samples),
        "template": template,
        "creation_time": datetime.now().isoformat(timespec="seconds"),
        "source_rest_file": "",
        "period_cv": None,
        "autocorrelation_strength": None,
        "config": {
            "align_ratio": align_ratio,
        },
    }
    diagnostics = {
        "detrended_values": detrended,
        "segments": segments,
        "aligned_segments": aligned,
        "alignment_shifts": np.asarray(shifts, dtype=int),
        "template": template,
    }
    return model, diagnostics


def _robust_scale_and_baseline(
    measured: np.ndarray,
    reference: np.ndarray,
    scale_min: float,
    scale_max: float,
    fixed_scale: float | None = None,
) -> tuple[float, float, float]:
    """迭代异常值剔除的稳健拟合，b 只用于估计 a 和评分。"""

    measured = measured.astype(float, copy=False)
    reference = reference.astype(float, copy=False)
    valid = np.isfinite(measured) & np.isfinite(reference)
    if valid.sum() < 3 or float(np.std(reference[valid])) < EPS:
        scale = 0.0 if fixed_scale is None else float(fixed_scale)
        baseline = float(np.median(measured[valid])) if valid.any() else 0.0
        score = float("inf")
        return scale, baseline, score

    mask = valid.copy()
    scale = 0.0 if fixed_scale is None else float(fixed_scale)
    baseline = float(np.median(measured[mask] - scale * reference[mask]))

    for _ in range(4):
        if fixed_scale is None:
            y = measured[mask] - float(np.median(measured[mask]))
            r = reference[mask] - float(np.mean(reference[mask]))
            denom = float(np.dot(r, r))
            scale = float(np.dot(y, r) / denom) if denom > EPS else 0.0
            scale = float(np.clip(scale, scale_min, scale_max))
        baseline = float(np.median(measured[mask] - scale * reference[mask]))
        residual = measured - baseline - scale * reference
        residual_center = float(np.median(residual[mask]))
        mad = float(np.median(np.abs(residual[mask] - residual_center)))
        if mad < EPS:
            break
        new_mask = valid & (np.abs(residual - residual_center) <= 3.5 * 1.4826 * mad)
        if new_mask.sum() < max(3, int(0.25 * valid.sum())):
            break
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    if fixed_scale is not None:
        scale = float(np.clip(fixed_scale, scale_min, scale_max))
    residual = measured - baseline - scale * reference
    residual_center = float(np.median(residual[mask])) if mask.any() else 0.0
    score = float(np.median(np.abs(residual[mask] - residual_center))) if mask.any() else float("inf")
    return scale, baseline, score


def _estimate_initial_phase(values: np.ndarray, template: np.ndarray, period_samples: int) -> int:
    length = min(values.size, max(period_samples * 2, period_samples))
    chunk = values[:length]
    chunk_centered = chunk - float(np.median(chunk))
    best_phase = 0
    best_score = -np.inf
    for phase in range(period_samples):
        reference = _periodic_reference(length, template, phase, period_samples, start_index=0)
        score = float(np.dot(chunk_centered, reference))
        if score > best_score:
            best_score = score
            best_phase = phase
    return int(best_phase)


def _candidate_offsets(max_abs_offset: int, max_count: int = 41) -> np.ndarray:
    max_abs_offset = int(max_abs_offset)
    if max_abs_offset <= 0:
        return np.array([0], dtype=int)
    if 2 * max_abs_offset + 1 <= max_count:
        return np.arange(-max_abs_offset, max_abs_offset + 1, dtype=int)
    offsets = np.linspace(-max_abs_offset, max_abs_offset, max_count)
    return np.unique(np.round(offsets).astype(int))


def remove_periodic_noise(
    values,
    fs,
    model,
    block_periods=4,
    max_period_drift_ratio=0.05,
    scale_min=0.0,
    scale_max=3.0,
    freeze_scale=False,
    calibration_seconds=5.0,
):
    """
    返回：
    cleaned_values
    estimated_noise
    scale_history
    phase_history
    diagnostics
    """

    fs = float(fs)
    if fs <= 0:
        raise ValueError("采样率 fs 必须大于 0。")

    measured = _as_float_array(values)
    template = _as_float_array(model.get("template"), name="template")
    template = template - float(np.mean(template))
    period_samples = int(round(model.get("period_samples", template.size)))
    if period_samples < 3 or template.size < 3:
        raise ValueError("模板周期点数不合法。")

    block_periods = max(1, int(block_periods))
    block_size = max(period_samples, int(round(period_samples * block_periods)))
    drift_samples = max(0, int(round(period_samples * float(max_period_drift_ratio))))
    period_offsets = _candidate_offsets(drift_samples, max_count=7)
    phase_offsets = _candidate_offsets(drift_samples, max_count=41)
    candidate_periods = np.unique(np.clip(period_samples + period_offsets, 3, None).astype(int))

    initial_phase = _estimate_initial_phase(measured, template, period_samples)
    phase_center = initial_phase
    frozen_scale: float | None = None

    if freeze_scale:
        calibration_len = min(measured.size, max(period_samples, int(round(float(calibration_seconds) * fs))))
        reference = _periodic_reference(calibration_len, template, initial_phase, period_samples, start_index=0)
        frozen_scale, _, _ = _robust_scale_and_baseline(
            measured[:calibration_len],
            reference,
            float(scale_min),
            float(scale_max),
            fixed_scale=None,
        )

    estimated_noise = np.zeros_like(measured, dtype=float)
    scale_history = np.zeros_like(measured, dtype=float)
    phase_history = np.zeros_like(measured, dtype=float)
    block_records: list[dict[str, Any]] = []

    for start in range(0, measured.size, block_size):
        end = min(measured.size, start + block_size)
        block = measured[start:end]
        best: dict[str, Any] | None = None

        for candidate_period in candidate_periods:
            for phase_offset in phase_offsets:
                candidate_phase = int(phase_center + phase_offset)
                reference = _periodic_reference(
                    block.size,
                    template,
                    candidate_phase,
                    int(candidate_period),
                    start_index=start,
                )
                scale, baseline, score = _robust_scale_and_baseline(
                    block,
                    reference,
                    float(scale_min),
                    float(scale_max),
                    fixed_scale=frozen_scale,
                )
                if best is None or score < best["score"]:
                    best = {
                        "period_samples": int(candidate_period),
                        "phase_shift_samples": int(candidate_phase),
                        "scale": float(scale),
                        "baseline": float(baseline),
                        "score": float(score),
                        "reference": reference,
                    }

        if best is None:
            raise ValueError("模板拟合失败，无法生成抵消结果。")

        noise = best["scale"] * best["reference"]
        estimated_noise[start:end] = noise
        scale_history[start:end] = best["scale"]
        phase_history[start:end] = best["phase_shift_samples"]
        phase_center = int(best["phase_shift_samples"])
        block_records.append(
            {
                "block_start": int(start),
                "block_end": int(end),
                "block_center_time_s": float((start + end) / 2.0 / fs),
                "period_samples": int(best["period_samples"]),
                "scale": float(best["scale"]),
                "phase_shift_samples": int(best["phase_shift_samples"]),
                "baseline_for_fit": float(best["baseline"]),
                "fit_mad": float(best["score"]),
            }
        )

    cleaned_values = measured - estimated_noise
    diagnostics = {
        "initial_phase_shift_samples": int(initial_phase),
        "frozen_scale": frozen_scale,
        "block_records": block_records,
        "block_periods": int(block_periods),
        "max_period_drift_ratio": float(max_period_drift_ratio),
        "scale_min": float(scale_min),
        "scale_max": float(scale_max),
    }
    return cleaned_values, estimated_noise, scale_history, phase_history, diagnostics


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def save_template(path, model):
    """保存 period_template.csv、period_template.npz 和 period_template_config.json。"""

    target = Path(path)
    if target.suffix.lower() == ".npz":
        directory = target.parent
        npz_path = target
        csv_path = target.with_suffix(".csv")
        json_path = target.with_name(f"{target.stem}_config.json")
    else:
        directory = target
        npz_path = directory / "period_template.npz"
        csv_path = directory / "period_template.csv"
        json_path = directory / "period_template_config.json"

    directory.mkdir(parents=True, exist_ok=True)
    template = _as_float_array(model.get("template"), name="template")
    fs = float(model.get("fs"))
    period_samples = int(model.get("period_samples", template.size))
    period_seconds = float(model.get("period_seconds", period_samples / fs))

    template_df = pd.DataFrame(
        {
            "template_index": np.arange(template.size, dtype=int),
            "template_time_s": np.arange(template.size, dtype=float) / fs,
            "template_value": template,
        }
    )
    template_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    np.savez(
        npz_path,
        fs=fs,
        period_samples=period_samples,
        period_seconds=period_seconds,
        template=template,
        creation_time=str(model.get("creation_time", datetime.now().isoformat(timespec="seconds"))),
        source_rest_file=str(model.get("source_rest_file", "")),
        period_cv=float(model.get("period_cv") or 0.0),
        autocorrelation_strength=float(model.get("autocorrelation_strength") or 0.0),
    )

    config = {
        "fs": fs,
        "period_samples": period_samples,
        "period_seconds": period_seconds,
        "frequency_hz": float(model.get("frequency_hz", fs / period_samples)),
        "creation_time": str(model.get("creation_time", "")),
        "source_rest_file": str(model.get("source_rest_file", "")),
        "period_cv": model.get("period_cv"),
        "autocorrelation_strength": model.get("autocorrelation_strength"),
        "config": model.get("config", {}),
    }
    json_path.write_text(json.dumps(_json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": csv_path, "npz": npz_path, "json": json_path}


def _npz_scalar(npz: Any, key: str, default: Any = None) -> Any:
    if key not in npz:
        return default
    value = npz[key]
    if np.asarray(value).shape == ():
        return value.item()
    return value


def load_template(path):
    """从目录或 .npz 文件载入模板。"""

    source = Path(path)
    if source.is_dir():
        npz_path = source / "period_template.npz"
        json_path = source / "period_template_config.json"
    else:
        npz_path = source
        json_path = source.with_name(f"{source.stem}_config.json")

    if not npz_path.exists():
        raise FileNotFoundError(f"找不到模板文件：{npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        model = {
            "fs": float(_npz_scalar(data, "fs")),
            "period_samples": int(_npz_scalar(data, "period_samples")),
            "period_seconds": float(_npz_scalar(data, "period_seconds")),
            "template": np.asarray(data["template"], dtype=float),
            "creation_time": str(_npz_scalar(data, "creation_time", "")),
            "source_rest_file": str(_npz_scalar(data, "source_rest_file", "")),
            "period_cv": float(_npz_scalar(data, "period_cv", 0.0)),
            "autocorrelation_strength": float(_npz_scalar(data, "autocorrelation_strength", 0.0)),
            "config": {},
        }

    if json_path.exists():
        try:
            config = json.loads(json_path.read_text(encoding="utf-8"))
            model["config"] = config.get("config", config)
        except json.JSONDecodeError:
            model["config"] = {}

    return model
