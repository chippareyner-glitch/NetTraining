"""Run Rest/Impact training experiments with fixed session splits.

The script only reads data_labeled/ and writes outputs. It does not modify
annotate_trials.py or source data files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from plant_signal.training.binary import (
    THRESHOLD_VALUES,
    _as_json_ready,
    _median_filter_3,
    build_model,
    build_threshold_report,
    choose_session_split,
    evaluate_threshold,
    export_c_header,
    export_int8_tflite,
    label_counts,
    load_windows,
    plot_confusion_matrix,
    plot_training_curve,
    recommend_threshold,
    save_threshold_outputs,
    set_reproducible_seed,
)


BASELINE_SESSION_COUNT = 27


@dataclass
class ExperimentConfig:
    name: str
    group: str
    class_weight_mode: str
    raw_window_size: int = 128
    stride: int = 32
    model_input_size: int | None = None
    downsample_factor: int = 1
    downsample_method: str = "none"
    hard_negative: bool = False


def log(message: str) -> None:
    print(f"[实验] {message}")


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_as_json_ready(obj), ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def class_weights_for(mode: str, y_train: np.ndarray) -> dict[int, float]:
    rest = int(np.sum(y_train == 0))
    impact = int(np.sum(y_train == 1))
    total = len(y_train)
    if rest <= 0 or impact <= 0:
        return {0: 1.0, 1: 1.0}

    if mode == "no_class_weight":
        return {0: 1.0, 1: 1.0}
    if mode == "inverse_class_weight":
        return {0: float(total / (2.0 * rest)), 1: float(total / (2.0 * impact))}
    if mode == "sqrt_class_weight":
        return {0: 1.0, 1: float(math.sqrt(rest / impact))}
    if mode.startswith("manual_impact_weight_"):
        return {0: 1.0, 1: float(mode.rsplit("_", 1)[-1])}
    raise ValueError(f"Unknown class_weight_mode: {mode}")


def sample_weights(y: np.ndarray, class_weight: dict[int, float], extra: np.ndarray | None = None) -> np.ndarray:
    weights = np.asarray([class_weight[int(label)] for label in y], dtype=np.float32)
    if extra is not None:
        weights = weights * extra.astype(np.float32)
    return weights


def split_raw_windows(windows, split_sessions: dict[str, list[str]], split_name: str):
    mask = windows.meta["session_name"].isin(split_sessions[split_name]).to_numpy()
    return (
        windows.raw_windows_uV[mask],
        windows.labels[mask],
        windows.meta.loc[mask].reset_index(drop=True),
    )


def downsample_1d(values: np.ndarray, factor: int, method: str) -> np.ndarray:
    if factor == 1:
        return values
    usable = (len(values) // factor) * factor
    values = values[:usable]
    reshaped = values.reshape(-1, factor)
    if method == "mean":
        return reshaped.mean(axis=1).astype(np.float32)
    if method == "take":
        return reshaped[:, 0].astype(np.float32)
    raise ValueError(f"Unknown downsample method: {method}")


def preprocess_raw_windows(
    raw_windows_uV: np.ndarray,
    *,
    scale_mv: float | None,
    baseline_points: int,
    use_median_filter: bool,
    downsample_factor: int,
    downsample_method: str,
) -> tuple[np.ndarray, float]:
    processed = []
    for window in raw_windows_uV:
        filtered = _median_filter_3(window) if use_median_filter else window.astype(np.float32)
        filtered = downsample_1d(filtered, downsample_factor, downsample_method)
        baseline_count = min(max(1, baseline_points), len(filtered))
        baseline_uV = float(np.median(filtered[:baseline_count]))
        processed.append(((filtered - baseline_uV) / 1000.0).astype(np.float32))

    x_mv = np.stack(processed).astype(np.float32)
    if scale_mv is None:
        scale_mv = float(np.percentile(np.abs(x_mv), 99))
        if not np.isfinite(scale_mv) or scale_mv <= 1e-6:
            scale_mv = 1.0

    x_scaled = np.clip(x_mv / scale_mv, -1.0, 1.0).astype(np.float32)
    return x_scaled[..., np.newaxis], float(scale_mv)


def make_preprocess_config(config: ExperimentConfig, scale_mv: float, baseline_points: int) -> dict[str, Any]:
    input_size = config.model_input_size or (config.raw_window_size // config.downsample_factor)
    return {
        "raw_window_size": config.raw_window_size,
        "model_input_size": input_size,
        "stride": config.stride,
        "impact_window_threshold": 0.20,
        "median_filter_3": True,
        "baseline_points": baseline_points,
        "baseline_method": "median of first baseline_points samples after optional downsample",
        "unit": "mV after baseline subtraction",
        "scale_mv_abs_p99_train": scale_mv,
        "clip_range": [-1.0, 1.0],
        "downsample_factor": config.downsample_factor,
        "downsample_method": config.downsample_method,
        "input_shape": [input_size, 1],
    }


def fit_model(
    config: ExperimentConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray,
    exp_dir: Path,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    set_reproducible_seed(seed)
    input_size = int(x_train.shape[1])
    model = build_model(window_size=input_size, learning_rate=1e-3)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        sample_weight=weights,
        verbose=2,
    )
    model.save(exp_dir / "model.keras")
    plot_training_curve(history, exp_dir / "training_curve.png")
    return model, history


def save_predictions(meta: pd.DataFrame, y_true: np.ndarray, probs: np.ndarray, threshold: float, path: Path) -> None:
    out = meta.copy()
    out["true_label"] = y_true.astype(int)
    out["prob_rest"] = probs[:, 0].astype(float)
    out["prob_impact"] = probs[:, 1].astype(float)
    out["pred_label"] = (probs[:, 1] >= threshold).astype(int)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def eval_event_level(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    consecutive_windows: int,
    cooldown_s: float,
) -> dict[str, Any]:
    df = meta.copy()
    df["true_label"] = y_true.astype(int)
    df["prob_impact"] = probs[:, 1].astype(float)
    total_segments = 0
    detected_segments = 0
    delays = []
    false_triggers = 0
    total_triggers = 0

    for _, session_df in df.groupby("session_name", sort=False):
        session_df = session_df.sort_values(["window_start_index"]).reset_index(drop=True)
        triggers = []
        consecutive = 0
        cooldown_until = -1e9

        for idx, row in session_df.iterrows():
            start_time = float(row.get("window_start_time_s", idx))
            if start_time < cooldown_until:
                if row["prob_impact"] >= threshold:
                    consecutive += 1
                else:
                    consecutive = 0
                continue

            if row["prob_impact"] >= threshold:
                consecutive += 1
            else:
                consecutive = 0

            if consecutive >= consecutive_windows:
                triggers.append({"time": start_time, "idx": int(idx), "assigned": False})
                total_triggers += 1
                cooldown_until = start_time + cooldown_s
                consecutive = 0

        segments = []
        in_segment = False
        start_time = 0.0
        end_time = 0.0
        for idx, row in session_df.iterrows():
            is_impact = int(row["true_label"]) == 1
            row_start = float(row.get("window_start_time_s", idx))
            row_end = float(row.get("window_end_time_s", row_start))
            if is_impact and not in_segment:
                in_segment = True
                start_time = row_start
                end_time = row_end
            elif is_impact and in_segment:
                end_time = row_end
            elif not is_impact and in_segment:
                segments.append({"start": start_time, "end": end_time})
                in_segment = False
        if in_segment:
            segments.append({"start": start_time, "end": end_time})

        total_segments += len(segments)
        for segment in segments:
            match = None
            for trigger in triggers:
                if trigger["assigned"]:
                    continue
                if trigger["time"] >= segment["start"] and trigger["time"] <= segment["end"] + 0.5:
                    match = trigger
                    break
            if match is not None:
                match["assigned"] = True
                detected_segments += 1
                delays.append(float(match["time"] - segment["start"]))

        false_triggers += sum(1 for trigger in triggers if not trigger["assigned"])

    return {
        "threshold": float(threshold),
        "consecutive_windows": int(consecutive_windows),
        "cooldown_s": float(cooldown_s),
        "rest_false_trigger_count": int(false_triggers),
        "total_trigger_count": int(total_triggers),
        "impact_event_count": int(total_segments),
        "detected_impact_event_count": int(detected_segments),
        "impact_detection_rate": float(detected_segments / total_segments) if total_segments else 0.0,
        "mean_trigger_delay_s": float(np.mean(delays)) if delays else None,
    }


def build_event_report(meta: pd.DataFrame, y_true: np.ndarray, probs: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    rows = []
    for n in (1, 2, 3):
        for cooldown in (0.3, 0.5, 1.0):
            rows.append(eval_event_level(meta, y_true, probs, threshold, n, cooldown))
    return rows


def run_single_experiment(
    config: ExperimentConfig,
    split_sessions: dict[str, list[str]],
    data_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    quantize: bool,
) -> dict[str, Any]:
    exp_dir = out_dir / "experiments" / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log(f"开始实验: {config.name}")

    windows = load_windows(data_dir, config.raw_window_size, config.stride, impact_threshold=0.20)
    raw_train, y_train, meta_train = split_raw_windows(windows, split_sessions, "train")
    raw_val, y_val, meta_val = split_raw_windows(windows, split_sessions, "val")
    raw_test, y_test, meta_test = split_raw_windows(windows, split_sessions, "test")

    x_train, scale_mv = preprocess_raw_windows(
        raw_train,
        scale_mv=None,
        baseline_points=16,
        use_median_filter=True,
        downsample_factor=config.downsample_factor,
        downsample_method=config.downsample_method,
    )
    x_val, _ = preprocess_raw_windows(
        raw_val,
        scale_mv=scale_mv,
        baseline_points=16,
        use_median_filter=True,
        downsample_factor=config.downsample_factor,
        downsample_method=config.downsample_method,
    )
    x_test, _ = preprocess_raw_windows(
        raw_test,
        scale_mv=scale_mv,
        baseline_points=16,
        use_median_filter=True,
        downsample_factor=config.downsample_factor,
        downsample_method=config.downsample_method,
    )

    preprocess_config = make_preprocess_config(config, scale_mv, baseline_points=16)
    json_dump(exp_dir / "preprocess_config.json", preprocess_config)

    cw = class_weights_for(config.class_weight_mode, y_train)
    base_weights = sample_weights(y_train, cw)

    hard_negative_info = None
    if config.hard_negative:
        initial_dir = exp_dir / "initial_model"
        initial_dir.mkdir(parents=True, exist_ok=True)
        initial_model, _ = fit_model(
            config,
            x_train,
            y_train,
            x_val,
            y_val,
            base_weights,
            initial_dir,
            epochs,
            batch_size,
            patience,
            seed,
        )
        train_probs = initial_model.predict(x_train, verbose=0)
        val_probs = initial_model.predict(x_val, verbose=0)
        rest_train_probs = train_probs[y_train == 0, 1]
        if len(rest_train_probs):
            cutoff = max(0.5, float(np.percentile(rest_train_probs, 80)))
        else:
            cutoff = 0.5
        hard_train = (y_train == 0) & (train_probs[:, 1] >= cutoff)
        hard_val = (y_val == 0) & (val_probs[:, 1] >= cutoff)
        extra = np.ones_like(y_train, dtype=np.float32)
        extra[hard_train] = 4.0
        base_weights = sample_weights(y_train, cw, extra=extra)
        hard_negative_info = {
            "initial_class_weight_mode": config.class_weight_mode,
            "hard_negative_cutoff_prob_impact": cutoff,
            "hard_train_rest_windows": int(np.sum(hard_train)),
            "hard_val_rest_windows_observed": int(np.sum(hard_val)),
            "hard_negative_weight_multiplier": 4.0,
            "note": "Only train hard negatives are upweighted in the second model; test set is never used.",
        }
        json_dump(exp_dir / "hard_negative_info.json", hard_negative_info)

    model, _ = fit_model(
        config,
        x_train,
        y_train,
        x_val,
        y_val,
        base_weights,
        exp_dir,
        epochs,
        batch_size,
        patience,
        seed + 17 if config.hard_negative else seed,
    )

    val_probs = model.predict(x_val, verbose=0)
    test_probs = model.predict(x_test, verbose=0)
    val_threshold_rows = build_threshold_report(y_val, val_probs)
    decision_config = recommend_threshold(val_threshold_rows)
    decision_config["note"] = "recommended by threshold sweep on validation set"
    threshold = float(decision_config["impact_threshold"])
    test_threshold_rows = build_threshold_report(y_test, test_probs)
    selected_test = evaluate_threshold(y_test, test_probs[:, 1], threshold)
    default_test = evaluate_threshold(y_test, test_probs[:, 1], 0.50)

    save_threshold_outputs(
        val_threshold_rows,
        decision_config,
        exp_dir / "val_threshold_report.csv",
        exp_dir / "val_threshold_report.json",
        exp_dir / "decision_config.json",
    )
    pd.DataFrame(test_threshold_rows).to_csv(exp_dir / "test_threshold_report.csv", index=False, encoding="utf-8-sig")
    json_dump(exp_dir / "test_threshold_report.json", test_threshold_rows)
    save_predictions(meta_val, y_val, val_probs, threshold, exp_dir / "val_window_predictions.csv")
    save_predictions(meta_test, y_test, test_probs, threshold, exp_dir / "test_window_predictions.csv")
    plot_confusion_matrix(np.asarray(selected_test["confusion_matrix"]), exp_dir / "confusion_matrix.png")

    event_rows = build_event_report(meta_test, y_test, test_probs, threshold)
    pd.DataFrame(event_rows).to_csv(exp_dir / "event_level_report.csv", index=False, encoding="utf-8-sig")

    quant_info = None
    if quantize:
        tflite_path = exp_dir / "model_int8.tflite"
        quant_info = export_int8_tflite(model, x_train, tflite_path, representative_count=512)
        json_dump(exp_dir / "quantization_info.json", quant_info)

    report = {
        "experiment": config.__dict__,
        "class_weight": cw,
        "hard_negative": hard_negative_info,
        "split_window_counts": {
            "train": {"total": int(len(y_train)), **label_counts(y_train)},
            "val": {"total": int(len(y_val)), **label_counts(y_val)},
            "test": {"total": int(len(y_test)), **label_counts(y_test)},
        },
        "preprocess": preprocess_config,
        "parameter_count": int(model.count_params()),
        "val_recommended_decision": decision_config,
        "test_at_recommended_threshold": selected_test,
        "test_at_threshold_0_50": default_test,
        "quantization": quant_info,
    }
    json_dump(exp_dir / "report.json", report)

    summary = {
        "experiment": config.name,
        "group": config.group,
        "class_weight_mode": config.class_weight_mode,
        "raw_window_size": config.raw_window_size,
        "stride": config.stride,
        "model_input_size": int(x_train.shape[1]),
        "downsample_factor": config.downsample_factor,
        "downsample_method": config.downsample_method,
        "hard_negative": bool(config.hard_negative),
        "train_windows": int(len(y_train)),
        "train_rest_windows": int(np.sum(y_train == 0)),
        "train_impact_windows": int(np.sum(y_train == 1)),
        "val_windows": int(len(y_val)),
        "val_rest_windows": int(np.sum(y_val == 0)),
        "val_impact_windows": int(np.sum(y_val == 1)),
        "test_windows": int(len(y_test)),
        "test_rest_windows": int(np.sum(y_test == 0)),
        "test_impact_windows": int(np.sum(y_test == 1)),
        "parameter_count": int(model.count_params()),
        "keras_size_bytes": int((exp_dir / "model.keras").stat().st_size),
        "tflite_size_bytes": int((exp_dir / "model_int8.tflite").stat().st_size) if (exp_dir / "model_int8.tflite").exists() else None,
        "recommended_threshold": threshold,
        "val_fpr": float(decision_config["metrics"]["rest_false_positive_rate"]),
        "val_impact_recall": float(decision_config["metrics"]["impact_recall"]),
        "val_impact_f1": float(decision_config["metrics"]["impact_f1"]),
        "test_default_accuracy": float(default_test["accuracy"]),
        "test_default_fpr": float(default_test["rest_false_positive_rate"]),
        "test_default_impact_recall": float(default_test["impact_recall"]),
        "test_selected_accuracy": float(selected_test["accuracy"]),
        "test_selected_precision": float(selected_test["impact_precision"]),
        "test_selected_impact_recall": float(selected_test["impact_recall"]),
        "test_selected_impact_f1": float(selected_test["impact_f1"]),
        "test_selected_fpr": float(selected_test["rest_false_positive_rate"]),
        "test_selected_fnr": float(selected_test["impact_false_negative_rate"]),
        "test_selected_confusion_matrix": json.dumps(selected_test["confusion_matrix"]),
        "experiment_dir": str(exp_dir),
    }
    log(
        f"完成 {config.name}: threshold={threshold:.2f}, "
        f"test_fpr={summary['test_selected_fpr']:.4f}, "
        f"test_recall={summary['test_selected_impact_recall']:.4f}"
    )
    return summary


def make_fixed_split(data_dir: Path, out_dir: Path, seed: int) -> dict[str, Any]:
    windows = load_windows(data_dir, 128, 32, impact_threshold=0.20)
    split_sessions, warnings = choose_session_split(windows.meta, seed=seed)
    session_names = sorted(windows.meta["session_name"].unique().tolist())
    impact_sessions = sorted(
        session for session in session_names if bool((windows.meta.loc[windows.meta["session_name"] == session, "true_label"] == 1).any())
    )
    split = {
        "seed": seed,
        "data_dir": str(data_dir),
        "session_count": len(session_names),
        "baseline_session_count_before_new_data": BASELINE_SESSION_COUNT,
        "new_data_detected": len(session_names) > BASELINE_SESSION_COUNT,
        "all_sessions": session_names,
        "impact_sessions": impact_sessions,
        "rest_only_sessions": [session for session in session_names if session not in impact_sessions],
        "split_sessions": split_sessions,
        "warnings": warnings,
    }
    json_dump(out_dir / "session_split.json", split)
    return split


def experiment_list(full: bool) -> list[ExperimentConfig]:
    priority = [
        ExperimentConfig("no_class_weight", "class_weight", "no_class_weight"),
        ExperimentConfig("sqrt_class_weight", "class_weight", "sqrt_class_weight"),
        ExperimentConfig("manual_impact_weight_5", "class_weight", "manual_impact_weight_5"),
        ExperimentConfig("window_size_256_sqrt", "window_size", "sqrt_class_weight", raw_window_size=256, stride=64),
        ExperimentConfig(
            "downsample_2x_window128_stride32",
            "downsample",
            "sqrt_class_weight",
            raw_window_size=256,
            stride=32,
            model_input_size=128,
            downsample_factor=2,
            downsample_method="mean",
        ),
        ExperimentConfig("hard_negative_sqrt", "hard_negative", "sqrt_class_weight", hard_negative=True),
    ]
    if not full:
        return priority

    return [
        ExperimentConfig("no_class_weight", "class_weight", "no_class_weight"),
        ExperimentConfig("inverse_class_weight", "class_weight", "inverse_class_weight"),
        ExperimentConfig("sqrt_class_weight", "class_weight", "sqrt_class_weight"),
        ExperimentConfig("manual_impact_weight_3", "class_weight", "manual_impact_weight_3"),
        ExperimentConfig("manual_impact_weight_5", "class_weight", "manual_impact_weight_5"),
        ExperimentConfig("manual_impact_weight_8", "class_weight", "manual_impact_weight_8"),
        ExperimentConfig("manual_impact_weight_10", "class_weight", "manual_impact_weight_10"),
        ExperimentConfig("window_size_256_sqrt", "window_size", "sqrt_class_weight", raw_window_size=256, stride=64),
        ExperimentConfig("window_size_384_sqrt", "window_size", "sqrt_class_weight", raw_window_size=384, stride=96),
        ExperimentConfig(
            "downsample_2x_window128_stride32",
            "downsample",
            "sqrt_class_weight",
            raw_window_size=256,
            stride=32,
            model_input_size=128,
            downsample_factor=2,
            downsample_method="mean",
        ),
        ExperimentConfig("hard_negative_sqrt", "hard_negative", "sqrt_class_weight", hard_negative=True),
    ]


def select_best(summary_df: pd.DataFrame) -> pd.Series:
    eligible = summary_df[
        (summary_df["val_fpr"] <= 0.05)
        & (summary_df["val_impact_recall"] >= 0.75)
    ]
    if len(eligible):
        return eligible.sort_values(
            ["val_impact_f1", "val_impact_recall", "val_fpr"],
            ascending=[False, False, True],
        ).iloc[0]

    recall_ok = summary_df[summary_df["val_impact_recall"] >= 0.75]
    if len(recall_ok):
        return recall_ok.sort_values(
            ["val_fpr", "val_impact_f1", "val_impact_recall"],
            ascending=[True, False, False],
        ).iloc[0]

    return summary_df.sort_values(
        ["val_impact_f1", "val_fpr"],
        ascending=[False, True],
    ).iloc[0]


def copy_best_artifacts(best_row: pd.Series, best_dir: Path) -> None:
    exp_dir = Path(str(best_row["experiment_dir"]))
    best_dir.mkdir(parents=True, exist_ok=True)

    copies = {
        "model.keras": "model.keras",
        "model_int8.tflite": "model_int8.tflite",
        "preprocess_config.json": "preprocess_config.json",
        "quantization_info.json": "quantization_info.json",
        "decision_config.json": "decision_config.json",
        "report.json": "report.json",
        "confusion_matrix.png": "confusion_matrix.png",
        "val_threshold_report.csv": "threshold_report.csv",
        "test_threshold_report.csv": "test_threshold_report.csv",
        "event_level_report.csv": "event_level_report.csv",
    }
    for src_name, dst_name in copies.items():
        src = exp_dir / src_name
        if src.exists():
            shutil.copy2(src, best_dir / dst_name)

    export_c_header(best_dir / "model_int8.tflite", best_dir / "model_data.h")


def write_deploy_readme(best_row: pd.Series, best_dir: Path) -> None:
    preprocess = json.loads((best_dir / "preprocess_config.json").read_text(encoding="utf-8-sig"))
    quant = json.loads((best_dir / "quantization_info.json").read_text(encoding="utf-8-sig"))
    decision = json.loads((best_dir / "decision_config.json").read_text(encoding="utf-8-sig"))
    event_path = best_dir / "event_level_report.csv"
    event_note = "建议先使用单窗口阈值判定。"
    if event_path.exists():
        events = pd.read_csv(event_path)
        n2 = events[(events["consecutive_windows"] == 2) & (events["cooldown_s"] == 0.5)]
        n1 = events[(events["consecutive_windows"] == 1) & (events["cooldown_s"] == 0.5)]
        if len(n2) and len(n1):
            if int(n2.iloc[0]["rest_false_trigger_count"]) <= int(n1.iloc[0]["rest_false_trigger_count"]):
                if float(n2.iloc[0]["impact_detection_rate"]) >= float(n1.iloc[0]["impact_detection_rate"]):
                    event_note = "建议使用连续 2 个窗口确认 Impact，并设置约 0.5s cooldown。"

    text = f"""# Impact Binary Deployment Notes

推荐模型实验：`{best_row['experiment']}`

## 输入与预处理

- 模型输入长度：{preprocess['model_input_size']} 点，形状 `[1, {preprocess['model_input_size']}, 1]`
- 原始窗口长度：{preprocess['raw_window_size']} 点
- stride：{preprocess['stride']} 点
- 3 点中值滤波：需要
- 降采样：factor = {preprocess['downsample_factor']}，method = `{preprocess['downsample_method']}`
- baseline：对预处理后窗口前 {preprocess['baseline_points']} 点取中位数
- 单位转换：`x_mv = (x_uV - baseline_uV) / 1000.0`
- scale：{preprocess['scale_mv_abs_p99_train']:.8f} mV
- 裁剪：`[-1, 1]`

## int8 输入量化

TFLite 输入量化参数：

- input_scale：{quant['input_scale']}
- input_zero_point：{quant['input_zero_point']}

MCU 上可按下面方式把 float 输入量化为 int8：

```c
int8_t q = (int8_t)roundf(x_float / input_scale + input_zero_point);
```

其中 `x_float` 是已经完成 baseline、mV 转换、scale 归一化并裁剪到 `[-1, 1]` 的值。

## 推理后处理

模型仍输出 Rest / Impact 两类分数，阈值不要写入模型。

- 推荐 Impact 阈值：{decision['impact_threshold']}
- 判定规则：`prob_impact >= impact_threshold -> Impact else Rest`
- 连续窗口建议：{event_note}

如果现场误报多，可以继续提高阈值或启用连续 2 窗口确认。
如果现场漏检多，可以降低阈值或关闭连续窗口确认。

## 输出文件

- `model_int8.tflite`
- `model_data.h`
- `preprocess_config.json`
- `quantization_info.json`
- `decision_config.json`
- `report.json`
"""
    (best_dir / "README_deploy.md").write_text(text, encoding="utf-8-sig", newline="\n")


def write_summary_md(summary_df: pd.DataFrame, best_row: pd.Series, out_dir: Path) -> None:
    lowest_fpr = summary_df.sort_values("test_selected_fpr", ascending=True).iloc[0]
    highest_recall = summary_df.sort_values("test_selected_impact_recall", ascending=False).iloc[0]

    best_event_path = Path(str(best_row["experiment_dir"])) / "event_level_report.csv"
    recommend_consecutive = "暂不建议"
    if best_event_path.exists():
        events = pd.read_csv(best_event_path)
        n1 = events[(events["consecutive_windows"] == 1) & (events["cooldown_s"] == 0.5)]
        n2 = events[(events["consecutive_windows"] == 2) & (events["cooldown_s"] == 0.5)]
        if len(n1) and len(n2):
            if (
                int(n2.iloc[0]["rest_false_trigger_count"]) <= int(n1.iloc[0]["rest_false_trigger_count"])
                and float(n2.iloc[0]["impact_detection_rate"]) >= float(n1.iloc[0]["impact_detection_rate"])
            ):
                recommend_consecutive = "建议使用连续 2 窗口确认"

    text = f"""# Rest / Impact 实验汇总

## 结论

1. Rest false positive rate 最低的实验：`{lowest_fpr['experiment']}`，test FPR = {lowest_fpr['test_selected_fpr']:.4f}，Impact recall = {lowest_fpr['test_selected_impact_recall']:.4f}。
2. Impact recall 最高的实验：`{highest_recall['experiment']}`，Impact recall = {highest_recall['test_selected_impact_recall']:.4f}，test FPR = {highest_recall['test_selected_fpr']:.4f}。
3. 推荐最终使用模型：`{best_row['experiment']}`。
4. 推荐 Impact 阈值：{best_row['recommended_threshold']:.2f}。
5. 连续窗口确认建议：{recommend_consecutive}。
6. 建议继续补充更多困难静息数据，尤其是有周期波动、有毛刺、手靠近但无弹击、环境干扰。
7. 建议补充弱弹击数据，防止提高阈值或连续确认后漏检轻微弹击。
8. 模型参数量约 {int(best_row['parameter_count'])}，int8 TFLite 大小约 {int(best_row['tflite_size_bytes']) if not pd.isna(best_row['tflite_size_bytes']) else 'N/A'} bytes，适合继续导出到 CH32H417 TFLM 验证。

## 推荐模型 test 指标

- Accuracy：{best_row['test_selected_accuracy']:.4f}
- Impact precision：{best_row['test_selected_precision']:.4f}
- Impact recall：{best_row['test_selected_impact_recall']:.4f}
- Impact F1：{best_row['test_selected_impact_f1']:.4f}
- Rest false positive rate：{best_row['test_selected_fpr']:.4f}
- Impact false negative rate：{best_row['test_selected_fnr']:.4f}

## 说明

所有实验使用同一个 `session_split.json`。阈值只基于 val set 选择，test set 只用于最终报告。
"""
    (out_dir / "summary.md").write_text(text, encoding="utf-8-sig", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-split Rest/Impact experiments.")
    parser.add_argument("--data_dir", default="data_labeled")
    parser.add_argument("--out_dir", default="outputs/impact_binary_experiments")
    parser.add_argument("--best_dir", default="outputs/impact_binary_best")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="Run only the six priority experiments.")
    parser.add_argument("--skip_quantize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    best_dir = Path(args.best_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    set_reproducible_seed(args.seed)

    split_info = make_fixed_split(data_dir, out_dir, args.seed)
    if not split_info["new_data_detected"]:
        log("警告：session 数没有超过旧基线 27，请确认新增数据是否放入 data_labeled/。")
    log(f"固定 split 已保存: {out_dir / 'session_split.json'}")
    log(f"session 总数: {split_info['session_count']}, Impact session: {len(split_info['impact_sessions'])}, Rest-only session: {len(split_info['rest_only_sessions'])}")

    configs = experiment_list(full=not args.quick)
    summaries = []
    for config in configs:
        summaries.append(
            run_single_experiment(
                config,
                split_info["split_sessions"],
                data_dir,
                out_dir,
                args.epochs,
                args.batch_size,
                args.patience,
                args.seed,
                quantize=not args.skip_quantize,
            )
        )

    summary_df = pd.DataFrame(summaries)
    best_row = select_best(summary_df)
    summary_df["selected_best_by_val"] = summary_df["experiment"] == best_row["experiment"]
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    best_event = Path(str(best_row["experiment_dir"])) / "event_level_report.csv"
    if best_event.exists():
        shutil.copy2(best_event, out_dir / "event_level_report.csv")

    copy_best_artifacts(best_row, best_dir)
    write_deploy_readme(best_row, best_dir)
    write_summary_md(summary_df, best_row, out_dir)

    log(f"summary.csv: {out_dir / 'summary.csv'}")
    log(f"summary.md: {out_dir / 'summary.md'}")
    log(f"最佳模型已导出: {best_dir}")
    log(
        f"推荐实验 {best_row['experiment']}: threshold={best_row['recommended_threshold']:.2f}, "
        f"test_fpr={best_row['test_selected_fpr']:.4f}, "
        f"test_recall={best_row['test_selected_impact_recall']:.4f}"
    )


if __name__ == "__main__":
    main()
