"""Train a Rest/Impact event detector with recovery tails ignored.

This script does not edit data_labeled/ or annotate_trials.py. It derives an
event-detection training view from samples_labeled.csv at runtime:

- Impact positive event: a window contains impact_start.
- Impact valid response interval: impact_start .. impact_start + 255 raw samples.
- Recovery tail ignore interval: impact_start + 256 .. impact_start + 499.
- Windows touching ignored rows or recovery tails are excluded from training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

from plant_signal.training.binary import (
    THRESHOLD_VALUES,
    _as_json_ready,
    _median_filter_3,
    _session_and_trial,
    build_model,
    build_threshold_report,
    choose_session_split,
    evaluate_threshold,
    export_c_header,
    export_int8_tflite,
    label_name_to_binary,
    plot_confusion_matrix,
    plot_training_curve,
    recommend_threshold,
    set_reproducible_seed,
)


RAW_WINDOW_SIZE = 256
STRIDE = 32
DOWNSAMPLE_FACTOR = 2
MODEL_INPUT_SIZE = RAW_WINDOW_SIZE // DOWNSAMPLE_FACTOR
BASELINE_POINTS = 16
IMPACT_RESPONSE_POINTS = 256
TAIL_IGNORE_END_POINTS = 500
CLASS_NAMES = ["Rest", "Impact"]
SIGNAL_COLUMNS = ["voltage_uV", "raw_value"]
REQUIRED_COLUMNS = ["sample_index", "time_s", "label_id", "label_name", "train_use"]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass
class EventWindowBundle:
    raw_windows_uV: np.ndarray
    labels: np.ndarray
    meta: pd.DataFrame
    events: pd.DataFrame
    stats: dict[str, Any]
    warnings: list[str]


def log(message: str) -> None:
    print(f"[事件训练] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[警告] {message}", flush=True)


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_as_json_ready(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def label_counts(values: np.ndarray) -> dict[str, int]:
    counter = Counter(int(v) for v in values.tolist())
    return {"rest": int(counter.get(0, 0)), "impact": int(counter.get(1, 0))}


def find_impact_events(df: pd.DataFrame) -> list[dict[str, int]]:
    """Find contiguous human-labeled Impact segments and use their first point as impact_start."""
    impact_mask = (df["binary_label"] == 1) & (df["train_use"] == 1)
    events: list[dict[str, int]] = []
    active = False
    start_idx = None
    end_idx = None
    prev_idx = None

    for _, row in df.loc[impact_mask].iterrows():
        sample_idx = int(row["sample_index"])
        if not active or prev_idx is None or sample_idx != prev_idx + 1:
            if active and start_idx is not None and end_idx is not None:
                events.append(
                    {
                        "impact_start_index": int(start_idx),
                        "annotated_impact_end_index": int(end_idx),
                        "annotated_impact_len": int(end_idx - start_idx + 1),
                    }
                )
            active = True
            start_idx = sample_idx
        end_idx = sample_idx
        prev_idx = sample_idx

    if active and start_idx is not None and end_idx is not None:
        events.append(
            {
                "impact_start_index": int(start_idx),
                "annotated_impact_end_index": int(end_idx),
                "annotated_impact_len": int(end_idx - start_idx + 1),
            }
        )
    return events


def load_impact_events_from_labels(labels_path: Path) -> list[dict[str, int]]:
    """Prefer labels.json intervals because samples_labeled.csv can contain tiny gaps."""
    if not labels_path.exists():
        return []
    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    events: list[dict[str, int]] = []
    for item in payload.get("intervals", []):
        start = item.get("start_sample")
        end = item.get("end_sample", start)
        label_name = item.get("label_name")
        label_id = item.get("label_id")
        is_impact = label_name_to_binary(label_name) == 1 or int(label_id or -999) == 1
        if not is_impact or start is None:
            continue
        start_idx = int(start)
        end_idx = int(end if end is not None else start_idx)
        events.append(
            {
                "impact_start_index": start_idx,
                "annotated_impact_end_index": end_idx,
                "annotated_impact_len": int(max(1, end_idx - start_idx + 1)),
                "event_source": str(labels_path.name),
            }
        )
    return events


def contiguous_position_ranges(sample_indices: np.ndarray) -> list[tuple[int, int]]:
    """Return [start_pos, end_pos) ranges where sample_index increments by exactly 1."""
    if len(sample_indices) == 0:
        return []
    starts = [0]
    gaps = np.flatnonzero(np.diff(sample_indices.astype(np.int64)) != 1)
    starts.extend((gaps + 1).tolist())
    ends = (gaps + 1).tolist()
    ends.append(len(sample_indices))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def downsample_mean_2x(values: np.ndarray) -> np.ndarray:
    usable = (len(values) // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
    return values[:usable].reshape(-1, DOWNSAMPLE_FACTOR).mean(axis=1).astype(np.float32)


def preprocess_event_windows(
    raw_windows_uV: np.ndarray,
    scale_mv: float | None,
) -> tuple[np.ndarray, float]:
    processed = []
    for window in raw_windows_uV:
        filtered = _median_filter_3(window.astype(np.float32))
        downsampled = downsample_mean_2x(filtered)
        baseline_count = min(BASELINE_POINTS, len(downsampled))
        baseline_uV = float(np.median(downsampled[:baseline_count]))
        processed.append(((downsampled - baseline_uV) / 1000.0).astype(np.float32))

    x_mv = np.stack(processed).astype(np.float32)
    if scale_mv is None:
        scale_mv = float(np.percentile(np.abs(x_mv), 99))
        if not np.isfinite(scale_mv) or scale_mv <= 1e-6:
            scale_mv = 1.0

    x_scaled = np.clip(x_mv / scale_mv, -1.0, 1.0).astype(np.float32)
    return x_scaled[..., np.newaxis], float(scale_mv)


def overlap_range(window_start: int, window_end: int, interval_start: int, interval_end: int) -> bool:
    return window_start <= interval_end and window_end >= interval_start


def load_event_windows(data_dir: Path) -> EventWindowBundle:
    csv_files = sorted(data_dir.rglob("samples_labeled.csv"))
    if not csv_files:
        raise FileNotFoundError(f"没有找到 samples_labeled.csv: {data_dir}")

    raw_windows: list[np.ndarray] = []
    labels: list[int] = []
    meta_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    stats: dict[str, Any] = {
        "csv_file_count": len(csv_files),
        "session_count": 0,
        "signal_column_usage": Counter(),
        "original_label_points_train_use_1": Counter(),
        "window_label_distribution": Counter(),
        "ignored_original_train_use_or_unknown_windows": 0,
        "ignored_tail_windows": 0,
        "ignored_positive_zone_without_start_windows": 0,
        "too_short_contiguous_segments": 0,
        "candidate_windows_before_ignore": 0,
        "candidate_rest_phase_note": "Rest windows are taken by fixed stride over clean Rest-only spans, so periodic rising, falling, peak and valley phases are all kept when present in the session.",
    }

    sessions_seen: set[str] = set()
    event_id = 0
    seen_candidate_windows: set[tuple[str, int, int]] = set()

    for csv_path in csv_files:
        rel_path = csv_path.relative_to(data_dir).as_posix()
        session_name, trial_name = _session_and_trial(data_dir, csv_path)
        sessions_seen.add(session_name)

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            warnings.append(f"{rel_path}: 读取失败，已跳过: {exc}")
            continue

        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        signal_col = next((col for col in SIGNAL_COLUMNS if col in df.columns), None)
        if missing:
            warnings.append(f"{rel_path}: 缺少必要列 {missing}，已跳过")
            continue
        if signal_col is None:
            warnings.append(f"{rel_path}: 缺少 voltage_uV/raw_value，已跳过")
            continue

        stats["signal_column_usage"][signal_col] += 1
        df = df.copy()
        df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce")
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
        df["train_use"] = pd.to_numeric(df["train_use"], errors="coerce")
        df[signal_col] = pd.to_numeric(df[signal_col], errors="coerce")
        df["binary_label"] = df["label_name"].map(label_name_to_binary)
        df = df.dropna(subset=["sample_index", "time_s", "train_use", signal_col]).copy()
        if df.empty:
            warnings.append(f"{rel_path}: 有效数值行为空，已跳过")
            continue

        df["sample_index"] = df["sample_index"].astype(np.int64)
        df["train_use"] = df["train_use"].astype(int)
        df = df.sort_values("sample_index").reset_index(drop=True)
        stats["original_label_points_train_use_1"].update(
            "ignored" if pd.isna(value) else str(int(value))
            for value in df.loc[df["train_use"] == 1, "binary_label"].tolist()
        )

        file_events = load_impact_events_from_labels(csv_path.parent / "labels.json")
        if not file_events:
            file_events = find_impact_events(df)
            for event in file_events:
                event["event_source"] = "samples_labeled.csv"
        local_event_ranges: list[dict[str, int]] = []
        for event in file_events:
            event = dict(event)
            event["event_id"] = event_id
            event["session_name"] = session_name
            event["trial_name"] = trial_name
            event["source_file"] = rel_path
            event["positive_interval_start"] = int(event["impact_start_index"])
            event["positive_interval_end"] = int(event["impact_start_index"] + IMPACT_RESPONSE_POINTS - 1)
            event["tail_ignore_start"] = int(event["impact_start_index"] + IMPACT_RESPONSE_POINTS)
            event["tail_ignore_end"] = int(event["impact_start_index"] + TAIL_IGNORE_END_POINTS - 1)
            event["positive_windows"] = 0
            event["tail_ignored_windows"] = 0
            event_rows.append(event)
            local_event_ranges.append(event)
            event_id += 1

        ignore_row = (df["train_use"] != 1) | df["binary_label"].isna()
        sample_indices = df["sample_index"].to_numpy(dtype=np.int64)
        signal = df[signal_col].to_numpy(dtype=np.float32)
        times = df["time_s"].to_numpy(dtype=np.float64)

        def add_candidate_window(pos: int) -> None:
            end_pos = pos + RAW_WINDOW_SIZE
            if pos < 0 or end_pos > len(sample_indices):
                return

            win_start_idx = int(sample_indices[pos])
            win_end_idx = int(sample_indices[end_pos - 1])
            key = (rel_path, win_start_idx, win_end_idx)
            if key in seen_candidate_windows:
                return
            seen_candidate_windows.add(key)
            stats["candidate_windows_before_ignore"] += 1

            window_ignore = bool(ignore_row.iloc[pos:end_pos].any())
            tail_events = [
                e
                for e in local_event_ranges
                if overlap_range(win_start_idx, win_end_idx, int(e["tail_ignore_start"]), int(e["tail_ignore_end"]))
            ]
            contains_events = [
                e
                for e in local_event_ranges
                if win_start_idx <= int(e["impact_start_index"]) <= win_end_idx
            ]
            overlaps_positive_without_start = any(
                overlap_range(
                    win_start_idx,
                    win_end_idx,
                    int(e["positive_interval_start"]),
                    int(e["positive_interval_end"]),
                )
                and e not in contains_events
                for e in local_event_ranges
            )

            if tail_events:
                stats["ignored_tail_windows"] += 1
                for event in tail_events:
                    event_rows[int(event["event_id"])]["tail_ignored_windows"] += 1
                return

            if window_ignore:
                stats["ignored_original_train_use_or_unknown_windows"] += 1
                return

            if contains_events:
                label = 1
                event_ids = []
                for event in contains_events:
                    event_rows[int(event["event_id"])]["positive_windows"] += 1
                    event_ids.append(int(event["event_id"]))
            elif overlaps_positive_without_start:
                stats["ignored_positive_zone_without_start_windows"] += 1
                return
            else:
                label = 0
                event_ids = []

            raw_windows.append(signal[pos:end_pos].astype(np.float32))
            labels.append(label)
            stats["window_label_distribution"][str(label)] += 1
            meta_rows.append(
                {
                    "session_name": session_name,
                    "trial_name": trial_name,
                    "source_file": rel_path,
                    "signal_column": signal_col,
                    "window_start_index": win_start_idx,
                    "window_end_index": win_end_idx,
                    "window_start_time_s": float(times[pos]),
                    "window_end_time_s": float(times[end_pos - 1]),
                    "true_label": int(label),
                    "event_ids": ";".join(str(eid) for eid in event_ids),
                }
            )

        for seg_start, seg_end in contiguous_position_ranges(sample_indices):
            seg_len = seg_end - seg_start
            if seg_len < RAW_WINDOW_SIZE:
                stats["too_short_contiguous_segments"] += 1
                continue

            segment_indices = sample_indices[seg_start:seg_end]
            for event in local_event_ranges:
                impact_start = int(event["impact_start_index"])
                local_positions = np.flatnonzero(segment_indices == impact_start)
                if len(local_positions) == 0:
                    continue

                event_pos = seg_start + int(local_positions[0])
                min_start_pos = max(seg_start, event_pos - RAW_WINDOW_SIZE + 1)
                max_start_pos = min(event_pos, seg_end - RAW_WINDOW_SIZE)
                if min_start_pos > max_start_pos:
                    continue

                starts = list(range(min_start_pos, max_start_pos + 1, STRIDE))
                starts.extend([min_start_pos, max_start_pos])
                for pos in sorted(set(starts)):
                    add_candidate_window(pos)

            for pos in range(seg_start, seg_end - RAW_WINDOW_SIZE + 1, STRIDE):
                add_candidate_window(pos)

    if not raw_windows:
        raise RuntimeError("事件切片后没有可训练窗口，请检查标签和 train_use。")

    events_df = pd.DataFrame(event_rows)
    meta_df = pd.DataFrame(meta_rows)
    stats["session_count"] = len(sessions_seen)
    stats["impact_event_count"] = int(len(events_df))
    stats["signal_column_usage"] = dict(stats["signal_column_usage"])
    stats["original_label_points_train_use_1"] = dict(stats["original_label_points_train_use_1"])
    stats["window_label_distribution"] = dict(stats["window_label_distribution"])
    stats["rest_windows"] = int(np.sum(np.asarray(labels) == 0))
    stats["impact_windows"] = int(np.sum(np.asarray(labels) == 1))
    stats["events_with_zero_positive_windows"] = (
        int((events_df["positive_windows"] == 0).sum()) if not events_df.empty else 0
    )

    return EventWindowBundle(
        raw_windows_uV=np.stack(raw_windows).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        meta=meta_df,
        events=events_df,
        stats=_as_json_ready(stats),
        warnings=warnings,
    )


def load_or_make_split(
    meta: pd.DataFrame,
    data_dir: Path,
    out_dir: Path,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, Any], list[str]]:
    warnings: list[str] = []
    current_sessions = set(meta["session_name"].unique().tolist())
    prior_split_path = Path("outputs/impact_binary_experiments/session_split.json")
    split_payload: dict[str, Any] | None = None

    if prior_split_path.exists():
        prior = read_json(prior_split_path)
        split_sessions = prior.get("split_sessions", {})
        split_all = set()
        for values in split_sessions.values():
            split_all.update(values)
        if current_sessions.issubset(split_all):
            split_payload = {
                "source": str(prior_split_path),
                "note": "Reused fixed split from previous experiments.",
                "split_sessions": split_sessions,
                "warnings": prior.get("warnings", []),
            }
            log(f"复用固定 session 划分: {prior_split_path}")
        else:
            missing = sorted(current_sessions - split_all)
            warnings.append(f"旧 split 缺少当前 session: {missing}，已重新生成 split。")

    if split_payload is None:
        split_sessions, split_warnings = choose_session_split(meta, seed=seed)
        warnings.extend(split_warnings)
        split_payload = {
            "source": "generated_by_train_impact_event_detector.py",
            "data_dir": str(data_dir),
            "seed": seed,
            "split_sessions": split_sessions,
            "warnings": split_warnings,
        }
        log("没有可复用的固定 split，已按 session 重新生成。")

    split_sessions = split_payload["split_sessions"]
    out_payload = dict(split_payload)
    out_payload["session_count"] = len(current_sessions)
    out_payload["all_sessions"] = sorted(current_sessions)
    json_dump(out_dir / "session_split.json", out_payload)
    return split_sessions, out_payload, warnings


def split_raw_windows(bundle: EventWindowBundle, split_sessions: dict[str, list[str]], name: str):
    mask = bundle.meta["session_name"].isin(split_sessions.get(name, [])).to_numpy()
    return (
        bundle.raw_windows_uV[mask],
        bundle.labels[mask],
        bundle.meta.loc[mask].reset_index(drop=True),
    )


def class_weight_sqrt(y_train: np.ndarray) -> dict[int, float]:
    rest = int(np.sum(y_train == 0))
    impact = int(np.sum(y_train == 1))
    if rest <= 0 or impact <= 0:
        return {0: 1.0, 1: 1.0}
    return {0: 1.0, 1: float(math.sqrt(rest / impact))}


def sample_weights(y_train: np.ndarray, class_weight: dict[int, float]) -> np.ndarray:
    return np.asarray([class_weight[int(label)] for label in y_train], dtype=np.float32)


def save_threshold_outputs(
    threshold_rows: list[dict[str, Any]],
    decision_config: dict[str, Any],
    csv_path: Path,
    json_path: Path,
    decision_path: Path | None = None,
) -> None:
    csv_rows = []
    for row in threshold_rows:
        csv_row = {key: value for key, value in row.items() if key != "confusion_matrix"}
        csv_row["confusion_matrix"] = json.dumps(row["confusion_matrix"], ensure_ascii=False)
        csv_rows.append(csv_row)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_dump(json_path, threshold_rows)
    if decision_path is not None:
        json_dump(decision_path, decision_config)


def recommend_event_threshold(threshold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in threshold_rows
        if row["rest_false_positive_rate"] <= 0.03 and row["impact_recall"] >= 0.75
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (row["impact_f1"], row["impact_recall"], -row["rest_false_positive_rate"]),
        )
        reason = "meets Rest false positive rate <= 3% and Impact recall >= 75%; selected highest Impact F1"
    else:
        relaxed = [
            row
            for row in threshold_rows
            if row["rest_false_positive_rate"] <= 0.03 and row["impact_recall"] >= 0.70
        ]
        if relaxed:
            selected = max(
                relaxed,
                key=lambda row: (row["impact_f1"], row["impact_recall"], -row["rest_false_positive_rate"]),
            )
            reason = "no threshold met both strict targets; selected a low-FPR compromise with validation Impact recall >= 70%"
        else:
            recall_ok = [row for row in threshold_rows if row["impact_recall"] >= 0.75]
            if recall_ok:
                selected = min(
                    recall_ok,
                    key=lambda row: (row["rest_false_positive_rate"], -row["impact_f1"], -row["impact_recall"]),
                )
                reason = "no threshold met the FPR target; selected the lowest Rest false positive rate while keeping Impact recall >= 75%"
            else:
                selected = max(
                    threshold_rows,
                    key=lambda row: (row["impact_f1"], -row["rest_false_positive_rate"], row["impact_recall"]),
                )
                reason = "no threshold kept Impact recall >= 75%; selected best Impact F1 compromise"

    return {
        "class_names": CLASS_NAMES,
        "impact_threshold": float(selected["threshold"]),
        "rule": "prob_impact >= impact_threshold -> Impact else Rest",
        "note": "recommended by threshold sweep on validation set",
        "selection_reason": reason,
        "metrics": selected,
    }


def save_predictions(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    pred = (probs[:, 1] >= threshold).astype(np.int64)
    out = meta.copy()
    out["true_label"] = y_true.astype(int)
    out["pred_label"] = pred.astype(int)
    out["prob_rest"] = probs[:, 0].astype(float)
    out["prob_impact"] = probs[:, 1].astype(float)
    first_cols = [
        "session_name",
        "window_start_index",
        "window_end_index",
        "true_label",
        "pred_label",
        "prob_rest",
        "prob_impact",
    ]
    rest_cols = [col for col in out.columns if col not in first_cols]
    out[first_cols + rest_cols].to_csv(path, index=False, encoding="utf-8-sig")


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    set_reproducible_seed(seed)
    model = build_model(window_size=int(x_train.shape[1]), learning_rate=1e-3)
    callbacks = []
    if patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=False,
            )
        )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        sample_weight=weights,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(out_dir / "model.keras")
    plot_training_curve(history, out_dir / "training_curve.png")
    return model, history


def make_preprocess_config(scale_mv: float) -> dict[str, Any]:
    return {
        "task": "Rest/Impact event detection",
        "raw_window_size": RAW_WINDOW_SIZE,
        "stride": STRIDE,
        "model_input_size": MODEL_INPUT_SIZE,
        "input_shape": [MODEL_INPUT_SIZE, 1],
        "median_filter_3": True,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "downsample_method": "mean",
        "baseline_points": BASELINE_POINTS,
        "baseline_method": "median of first 16 downsampled points",
        "unit": "mV after baseline subtraction",
        "scale_mv_abs_p99_train": float(scale_mv),
        "clip_range": [-1.0, 1.0],
        "impact_event_rule": "positive window contains impact_start",
        "impact_response_points_raw": IMPACT_RESPONSE_POINTS,
        "tail_ignore_rule": "windows overlapping impact_start+256 through impact_start+499 are excluded",
    }


def make_deploy_readme(out_dir: Path, report: dict[str, Any]) -> None:
    decision = report["decision_config"]
    preprocess = report["preprocess_config"]
    text = f"""# Impact Event Detector 部署说明

这是 Rest / Impact 二分类事件检测模型。模型检测弹击发生后的初始响应，不把恢复长尾当作 Impact 或 Rest 训练。

## 输入预处理

- 原始窗口长度: {preprocess["raw_window_size"]} 点
- stride: {preprocess["stride"]} 点
- 3 点中值滤波: 开启
- 2x 降采样: 每 2 点求平均，输入变成 {preprocess["model_input_size"]} 点
- baseline: 取降采样后前 {preprocess["baseline_points"]} 点中位数
- 单位转换: `(uV - baseline_uV) / 1000.0`
- scale: `{preprocess["scale_mv_abs_p99_train"]:.8f}` mV
- clip: `[-1, 1]`

## int8 输入量化

先得到 float 输入 `x`，范围约为 `[-1, 1]`，再按 TFLite 输入量化参数转换:

`x_int8 = round(x / input_scale + input_zero_point)`

量化参数见 `quantization_info.json`。

## 判定阈值

- 推荐 Impact 阈值: `{decision["impact_threshold"]:.2f}`
- 后处理规则: `prob_impact >= threshold` 判为 Impact，否则判为 Rest
- 如果现场静息误报多，可以提高阈值。
- 如果现场弱弹击漏检多，可以降低阈值。

TFLite 模型仍输出两个类别分数，阈值请放在单片机推理后的后处理逻辑里。
"""
    (out_dir / "README_deploy.md").write_text(text, encoding="utf-8-sig", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train event-based Rest/Impact detector.")
    parser.add_argument("--data_dir", default="data_labeled", help="包含 samples_labeled.csv 的目录")
    parser.add_argument("--out_dir", default="outputs/impact_event_detector", help="输出目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=0, help="0 表示事件检测脚本默认跑满 epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    log("开始事件切片：Impact 只取 impact_start 附近，恢复尾巴窗口会被忽略。")
    bundle = load_event_windows(data_dir)
    bundle.events.to_csv(out_dir / "impact_event_window_counts.csv", index=False, encoding="utf-8-sig")
    bundle.meta.to_csv(out_dir / "all_event_windows.csv", index=False, encoding="utf-8-sig")
    json_dump(out_dir / "event_slice_stats.json", bundle.stats)

    for message in bundle.warnings:
        warn(message)
    if bundle.stats["events_with_zero_positive_windows"]:
        warn(f"有 {bundle.stats['events_with_zero_positive_windows']} 个 Impact 事件没有生成正样本窗口。")

    log(
        "事件切片完成: "
        f"Impact事件={bundle.stats['impact_event_count']}，"
        f"Rest窗口={bundle.stats['rest_windows']}，"
        f"Impact窗口={bundle.stats['impact_windows']}，"
        f"尾巴ignore窗口={bundle.stats['ignored_tail_windows']}"
    )

    split_sessions, split_payload, split_warnings = load_or_make_split(bundle.meta, data_dir, out_dir, args.seed)
    for message in split_warnings:
        warn(message)

    raw_train, y_train, meta_train = split_raw_windows(bundle, split_sessions, "train")
    raw_val, y_val, meta_val = split_raw_windows(bundle, split_sessions, "val")
    raw_test, y_test, meta_test = split_raw_windows(bundle, split_sessions, "test")

    if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
        raise RuntimeError("train/val/test 至少有一个 split 没有窗口，无法训练。")
    if np.sum(y_train == 1) == 0:
        raise RuntimeError("train split 没有 Impact 窗口，无法训练二分类模型。")
    if np.sum(y_val == 1) == 0:
        warn("val split 没有 Impact 窗口，阈值推荐会不稳定。")
    if np.sum(y_test == 1) == 0:
        warn("test split 没有 Impact 窗口，最终 Impact recall 无法可信评估。")

    x_train, scale_mv = preprocess_event_windows(raw_train, scale_mv=None)
    x_val, _ = preprocess_event_windows(raw_val, scale_mv=scale_mv)
    x_test, _ = preprocess_event_windows(raw_test, scale_mv=scale_mv)
    preprocess_config = make_preprocess_config(scale_mv)
    json_dump(out_dir / "preprocess_config.json", preprocess_config)

    class_weight = class_weight_sqrt(y_train)
    weights = sample_weights(y_train, class_weight)
    log(f"训练窗口: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    log(f"窗口类别: train={label_counts(y_train)}, val={label_counts(y_val)}, test={label_counts(y_test)}")
    log(f"使用 sqrt class_weight: {class_weight}")

    model, _ = train_model(
        x_train,
        y_train,
        x_val,
        y_val,
        weights,
        out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )

    val_probs = model.predict(x_val, verbose=0)
    test_probs = model.predict(x_test, verbose=0)
    val_threshold_rows = build_threshold_report(y_val, val_probs)
    decision_config = recommend_event_threshold(val_threshold_rows)
    threshold = float(decision_config["impact_threshold"])

    test_threshold_rows = build_threshold_report(y_test, test_probs)
    default_test = evaluate_threshold(y_test, test_probs[:, 1], 0.50)
    selected_test = evaluate_threshold(y_test, test_probs[:, 1], threshold)
    pred_selected = (test_probs[:, 1] >= threshold).astype(np.int64)
    selected_matrix = np.asarray(selected_test["confusion_matrix"])

    save_threshold_outputs(
        val_threshold_rows,
        decision_config,
        out_dir / "val_threshold_report.csv",
        out_dir / "val_threshold_report.json",
        out_dir / "decision_config.json",
    )
    save_threshold_outputs(
        test_threshold_rows,
        decision_config,
        out_dir / "threshold_report.csv",
        out_dir / "threshold_report.json",
        None,
    )
    save_predictions(meta_val, y_val, val_probs, threshold, out_dir / "val_window_predictions.csv")
    save_predictions(meta_test, y_test, test_probs, threshold, out_dir / "window_predictions.csv")
    plot_confusion_matrix(selected_matrix, out_dir / "confusion_matrix.png")

    quant_info = export_int8_tflite(model, x_train, out_dir / "model_int8.tflite", representative_count=512)
    json_dump(out_dir / "quantization_info.json", quant_info)
    header_len = export_c_header(out_dir / "model_int8.tflite", out_dir / "model_data.h")

    report = {
        "task": "Rest/Impact event detection",
        "class_names": CLASS_NAMES,
        "event_slice_config": {
            "raw_window_size": RAW_WINDOW_SIZE,
            "stride": STRIDE,
            "positive_rule": "window contains impact_start",
            "impact_effective_interval": "impact_start to impact_start + 255 raw samples",
            "tail_ignore_interval": "impact_start + 256 to impact_start + 499 raw samples",
            "source_data_modified": False,
        },
        "event_slice_stats": bundle.stats,
        "split_sessions": split_sessions,
        "split_payload": split_payload,
        "split_window_counts": {
            "train": {"total": int(len(y_train)), **label_counts(y_train)},
            "val": {"total": int(len(y_val)), **label_counts(y_val)},
            "test": {"total": int(len(y_test)), **label_counts(y_test)},
        },
        "preprocess_config": preprocess_config,
        "class_weight_mode": "sqrt_class_weight",
        "class_weight": class_weight,
        "parameter_count": int(model.count_params()),
        "test_accuracy_at_recommended_threshold": float(accuracy_score(y_test, pred_selected)),
        "test_confusion_matrix_at_recommended_threshold": selected_matrix.tolist(),
        "test_at_threshold_0_50": default_test,
        "test_at_recommended_threshold": selected_test,
        "decision_config": decision_config,
        "threshold_values": THRESHOLD_VALUES,
        "quantization_info": quant_info,
        "model_data_h_bytes": int(header_len),
    }
    json_dump(out_dir / "report.json", report)
    make_deploy_readme(out_dir, report)

    # Also place a deploy-ready copy in the existing "best" output directory for convenience.
    best_dir = Path("outputs/impact_binary_best_event")
    best_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "model.keras",
        "model_int8.tflite",
        "model_data.h",
        "preprocess_config.json",
        "quantization_info.json",
        "decision_config.json",
        "report.json",
        "confusion_matrix.png",
        "threshold_report.csv",
        "README_deploy.md",
        "impact_event_window_counts.csv",
    ]:
        shutil.copy2(out_dir / name, best_dir / name)

    log("训练完成，关键输出已保存。")
    log(f"推荐阈值: {threshold:.2f}")
    log(f"默认0.50测试结果: FPR={default_test['rest_false_positive_rate']:.4f}, Impact recall={default_test['impact_recall']:.4f}")
    log(
        "推荐阈值测试结果: "
        f"FPR={selected_test['rest_false_positive_rate']:.4f}, "
        f"Impact recall={selected_test['impact_recall']:.4f}, "
        f"precision={selected_test['impact_precision']:.4f}, "
        f"F1={selected_test['impact_f1']:.4f}"
    )
    log(f"test混淆矩阵: {selected_matrix.tolist()}")


if __name__ == "__main__":
    main()
