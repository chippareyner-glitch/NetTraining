"""Train a small binary Rest/Impact model and export TFLite Micro artifacts."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


# Change these defaults when you want to try a different slice length or stride.
DEFAULT_WINDOW_SIZE = 128
DEFAULT_STRIDE = 32
THRESHOLD_VALUES = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

REQUIRED_COLUMNS = ["sample_index", "time_s", "label_id", "label_name", "train_use"]
SIGNAL_COLUMNS = ["voltage_uV", "raw_value"]
CLASS_NAMES = {0: "Rest", 1: "Impact"}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass
class WindowBundle:
    raw_windows_uV: np.ndarray
    labels: np.ndarray
    meta: pd.DataFrame
    point_stats: dict[str, Any]
    warnings: list[str]


@dataclass
class SplitBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    meta_train: pd.DataFrame
    x_val: np.ndarray
    y_val: np.ndarray
    meta_val: pd.DataFrame
    x_test: np.ndarray
    y_test: np.ndarray
    meta_test: pd.DataFrame
    split_sessions: dict[str, list[str]]
    warnings: list[str]
    preprocess_config: dict[str, Any]


def log(message: str) -> None:
    print(f"[训练] {message}")


def warn(message: str) -> None:
    print(f"[警告] {message}")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _label_candidates(value: Any) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value)
    candidates = [text]

    # Repair common mojibake if a CSV was decoded through the wrong locale.
    for source_encoding in ("gbk", "latin1"):
        try:
            repaired = text.encode(source_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired not in candidates:
            candidates.append(repaired)

    return candidates


def label_name_to_binary(value: Any) -> int | None:
    """Map label_name by the requested rule: Rest/静息 -> 0, Impact/弹击 -> 1."""
    for candidate in _label_candidates(value):
        lowered = candidate.lower()
        if "rest" in lowered or "静息" in candidate:
            return 0
        if "impact" in lowered or "弹击" in candidate:
            return 1
    return None


def _session_and_trial(data_dir: Path, csv_path: Path) -> tuple[str, str]:
    try:
        rel = csv_path.relative_to(data_dir)
        parts = rel.parts
    except ValueError:
        parts = csv_path.parts

    session_name = parts[0] if len(parts) >= 3 else csv_path.parent.parent.name
    trial_name = parts[1] if len(parts) >= 3 else csv_path.parent.name
    return session_name, trial_name


def _as_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _as_json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_as_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_windows(
    data_dir: Path,
    window_size: int,
    stride: int,
    impact_threshold: float,
) -> WindowBundle:
    csv_files = sorted(data_dir.rglob("samples_labeled.csv"))
    if not csv_files:
        raise FileNotFoundError(f"没有找到 samples_labeled.csv: {data_dir}")

    raw_windows: list[np.ndarray] = []
    labels: list[int] = []
    meta_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    point_stats: dict[str, Any] = {
        "csv_file_count": len(csv_files),
        "session_count": 0,
        "signal_column_usage": Counter(),
        "train_use_distribution": Counter(),
        "label_points_all": Counter(),
        "label_points_train_use_1": Counter(),
        "window_label_distribution": Counter(),
        "session_point_stats": defaultdict(lambda: {"rest_points": 0, "impact_points": 0, "windows": 0}),
    }

    sessions_seen: set[str] = set()

    for csv_path in csv_files:
        rel_path = csv_path.relative_to(data_dir).as_posix()
        session_name, trial_name = _session_and_trial(data_dir, csv_path)
        sessions_seen.add(session_name)

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - defensive diagnostics
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

        point_stats["signal_column_usage"][signal_col] += 1

        df = df.copy()
        df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce")
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
        df["train_use"] = pd.to_numeric(df["train_use"], errors="coerce")
        df[signal_col] = pd.to_numeric(df[signal_col], errors="coerce")
        df["binary_label"] = df["label_name"].map(label_name_to_binary)

        point_stats["train_use_distribution"].update(
            df["train_use"].dropna().astype(int).astype(str).tolist()
        )
        point_stats["label_points_all"].update(
            "ignored" if pd.isna(value) else str(int(value)) for value in df["binary_label"].tolist()
        )

        clean = df.dropna(subset=["sample_index", "time_s", "train_use", signal_col]).copy()
        use_df = clean.loc[clean["train_use"] == 1].copy()
        use_df = use_df.loc[use_df["binary_label"].notna()].copy()
        use_df["binary_label"] = use_df["binary_label"].astype(int)

        point_stats["label_points_train_use_1"].update(
            str(int(value)) for value in use_df["binary_label"].tolist()
        )
        point_stats["session_point_stats"][session_name]["rest_points"] += int((use_df["binary_label"] == 0).sum())
        point_stats["session_point_stats"][session_name]["impact_points"] += int((use_df["binary_label"] == 1).sum())

        if use_df.empty:
            warnings.append(f"{rel_path}: train_use == 1 且标签可识别的样本为空，已跳过窗口切片")
            continue

        use_df = use_df.sort_values("sample_index").reset_index(drop=True)
        indices = use_df["sample_index"].to_numpy(dtype=np.int64)
        segment_ids = np.cumsum(np.r_[0, np.diff(indices) != 1])

        file_windows = 0
        for _, segment in use_df.groupby(segment_ids, sort=False):
            if len(segment) < window_size:
                continue

            signal_uV = segment[signal_col].to_numpy(dtype=np.float32)
            sample_indices = segment["sample_index"].to_numpy(dtype=np.int64)
            time_s = segment["time_s"].to_numpy(dtype=np.float64)
            binary_labels = segment["binary_label"].to_numpy(dtype=np.int64)

            for start in range(0, len(segment) - window_size + 1, stride):
                end = start + window_size
                window_labels = binary_labels[start:end]
                impact_ratio = float(np.mean(window_labels == 1))
                window_label = 1 if impact_ratio >= impact_threshold else 0
                raw_windows.append(signal_uV[start:end].copy())
                labels.append(window_label)
                file_windows += 1
                point_stats["window_label_distribution"][str(window_label)] += 1
                point_stats["session_point_stats"][session_name]["windows"] += 1

                meta_rows.append(
                    {
                        "session_name": session_name,
                        "trial_name": trial_name,
                        "source_file": rel_path,
                        "window_start_index": int(sample_indices[start]),
                        "window_end_index": int(sample_indices[end - 1]),
                        "window_start_time_s": float(time_s[start]),
                        "window_end_time_s": float(time_s[end - 1]),
                        "impact_point_ratio": impact_ratio,
                        "true_label": int(window_label),
                        "signal_column": signal_col,
                    }
                )

        if file_windows == 0:
            warnings.append(f"{rel_path}: 连续可训练样本不足 {window_size} 点，没有生成窗口")

    if not raw_windows:
        raise RuntimeError("没有生成任何训练窗口，请检查标签、train_use 或 window_size。")

    point_stats["session_count"] = len(sessions_seen)
    point_stats["signal_column_usage"] = dict(point_stats["signal_column_usage"])
    point_stats["train_use_distribution"] = dict(point_stats["train_use_distribution"])
    point_stats["label_points_all"] = dict(point_stats["label_points_all"])
    point_stats["label_points_train_use_1"] = dict(point_stats["label_points_train_use_1"])
    point_stats["window_label_distribution"] = dict(point_stats["window_label_distribution"])
    point_stats["session_point_stats"] = {
        name: dict(stats) for name, stats in sorted(point_stats["session_point_stats"].items())
    }

    return WindowBundle(
        raw_windows_uV=np.stack(raw_windows).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        meta=pd.DataFrame(meta_rows),
        point_stats=_as_json_ready(point_stats),
        warnings=warnings,
    )


def choose_session_split(
    meta: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[dict[str, list[str]], list[str]]:
    warnings: list[str] = []
    sessions = sorted(meta["session_name"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n_sessions = len(sessions)

    if n_sessions < 3:
        warnings.append(
            f"session 数量只有 {n_sessions} 个，无法合理按 70/15/15 划分 train/val/test。"
        )

    session_has_impact = {
        session: bool((meta.loc[meta["session_name"] == session, "true_label"] == 1).any())
        for session in sessions
    }
    impact_sessions = [session for session in sessions if session_has_impact[session]]
    if not impact_sessions:
        warnings.append("没有任何 Impact 窗口；本次只能检查流程，不能训练有效弹击分类器。")
    elif len(impact_sessions) < 3:
        warnings.append(
            f"Impact session 只有 {len(impact_sessions)} 个，验证/测试指标可能非常不稳定。"
        )

    if n_sessions <= 1:
        return {"train": sessions, "val": [], "test": []}, warnings

    test_count = max(1, int(round(n_sessions * test_ratio)))
    val_count = max(1, int(round(n_sessions * val_ratio))) if n_sessions >= 3 else 0
    while test_count + val_count >= n_sessions and val_count > 0:
        val_count -= 1
    while test_count + val_count >= n_sessions and test_count > 0:
        test_count -= 1

    remaining = sessions.copy()
    test_sessions: list[str] = []
    if impact_sessions and test_count > 0:
        chosen = next(session for session in remaining if session_has_impact[session])
        test_sessions.append(chosen)
        remaining.remove(chosen)

    while len(test_sessions) < test_count and remaining:
        test_sessions.append(remaining.pop(0))

    val_sessions: list[str] = []
    remaining_impact = [session for session in remaining if session_has_impact[session]]
    if val_count > 0 and len(remaining_impact) >= 2:
        chosen = remaining_impact[0]
        val_sessions.append(chosen)
        remaining.remove(chosen)

    while len(val_sessions) < val_count and remaining:
        val_sessions.append(remaining.pop(0))

    train_sessions = remaining
    if not train_sessions and val_sessions:
        train_sessions.append(val_sessions.pop())
    if not train_sessions and test_sessions:
        train_sessions.append(test_sessions.pop())

    split = {
        "train": sorted(train_sessions),
        "val": sorted(val_sessions),
        "test": sorted(test_sessions),
    }

    for split_name, split_sessions in split.items():
        if not split_sessions:
            warnings.append(f"{split_name} split 没有 session。")
            continue
        split_meta = meta.loc[meta["session_name"].isin(split_sessions)]
        label_counts = Counter(split_meta["true_label"].astype(int).tolist())
        if label_counts.get(1, 0) == 0:
            warnings.append(f"{split_name} split 没有 Impact 窗口。")
        if label_counts.get(0, 0) == 0:
            warnings.append(f"{split_name} split 没有 Rest 窗口。")

    if impact_sessions:
        test_meta = meta.loc[meta["session_name"].isin(split["test"])]
        if int((test_meta["true_label"] == 1).sum()) == 0:
            warnings.append("测试集没有 Impact 窗口；请增加 Impact session 后重新训练。")

    return split, warnings


def _median_filter_3(window: np.ndarray) -> np.ndarray:
    padded = np.pad(window, (1, 1), mode="edge")
    stacked = np.stack([padded[:-2], padded[1:-1], padded[2:]], axis=0)
    return np.median(stacked, axis=0).astype(np.float32)


def preprocess_raw_windows(
    raw_windows_uV: np.ndarray,
    scale_mv: float | None,
    baseline_points: int,
    use_median_filter: bool,
) -> tuple[np.ndarray, float | None]:
    processed: list[np.ndarray] = []

    for window in raw_windows_uV:
        filtered = _median_filter_3(window) if use_median_filter else window.astype(np.float32)
        baseline_count = min(max(1, baseline_points), len(filtered))
        baseline_uV = float(np.median(filtered[:baseline_count]))
        x_mv = (filtered - baseline_uV) / 1000.0
        processed.append(x_mv.astype(np.float32))

    x_mv_all = np.stack(processed).astype(np.float32)
    fitted_scale = scale_mv
    if fitted_scale is None:
        fitted_scale = float(np.percentile(np.abs(x_mv_all), 99))
        if not np.isfinite(fitted_scale) or fitted_scale <= 1e-6:
            fitted_scale = 1.0

    x_scaled = np.clip(x_mv_all / fitted_scale, -1.0, 1.0).astype(np.float32)
    return x_scaled[..., np.newaxis], fitted_scale


def build_split_bundle(
    windows: WindowBundle,
    split_sessions: dict[str, list[str]],
    baseline_points: int,
    use_median_filter: bool,
    window_size: int,
    stride: int,
    impact_threshold: float,
) -> SplitBundle:
    def by_sessions(split_name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        mask = windows.meta["session_name"].isin(split_sessions[split_name]).to_numpy()
        return windows.raw_windows_uV[mask], windows.labels[mask], windows.meta.loc[mask].reset_index(drop=True)

    raw_train, y_train, meta_train = by_sessions("train")
    raw_val, y_val, meta_val = by_sessions("val")
    raw_test, y_test, meta_test = by_sessions("test")

    if len(raw_train) == 0:
        raise RuntimeError("训练集没有窗口，无法训练。")
    if len(raw_test) == 0:
        raise RuntimeError("测试集没有窗口，无法评估。")

    x_train, scale_mv = preprocess_raw_windows(
        raw_train,
        scale_mv=None,
        baseline_points=baseline_points,
        use_median_filter=use_median_filter,
    )
    x_val, _ = preprocess_raw_windows(
        raw_val,
        scale_mv=scale_mv,
        baseline_points=baseline_points,
        use_median_filter=use_median_filter,
    ) if len(raw_val) else (np.empty((0, window_size, 1), dtype=np.float32), scale_mv)
    x_test, _ = preprocess_raw_windows(
        raw_test,
        scale_mv=scale_mv,
        baseline_points=baseline_points,
        use_median_filter=use_median_filter,
    )

    preprocess_config = {
        "window_size": window_size,
        "stride": stride,
        "impact_threshold": impact_threshold,
        "median_filter_3": bool(use_median_filter),
        "baseline_points": baseline_points,
        "baseline_method": "median of first baseline_points samples per window",
        "unit": "mV after baseline subtraction",
        "scale_mv_abs_p99_train": float(scale_mv),
        "clip_range": [-1.0, 1.0],
        "input_shape": [window_size, 1],
    }

    return SplitBundle(
        x_train=x_train,
        y_train=y_train,
        meta_train=meta_train,
        x_val=x_val,
        y_val=y_val,
        meta_val=meta_val,
        x_test=x_test,
        y_test=y_test,
        meta_test=meta_test,
        split_sessions=split_sessions,
        warnings=[],
        preprocess_config=preprocess_config,
    )


def compute_class_weight(y_train: np.ndarray) -> dict[int, float]:
    counts = Counter(int(value) for value in y_train.tolist())
    total = len(y_train)
    class_weight: dict[int, float] = {}
    for class_id in (0, 1):
        count = counts.get(class_id, 0)
        if count > 0:
            class_weight[class_id] = float(total / (2.0 * count))
    return class_weight


def build_model(window_size: int, learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, 1)),
            tf.keras.layers.Conv1D(8, kernel_size=5, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(16, kernel_size=5, activation="relu", padding="same"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_curve(history: tf.keras.callbacks.History, output_path: Path) -> None:
    hist = pd.DataFrame(history.history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    axes[0].plot(hist.index + 1, hist["loss"], label="train")
    if "val_loss" in hist:
        axes[0].plot(hist.index + 1, hist["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(hist.index + 1, hist["accuracy"], label="train")
    if "val_accuracy" in hist:
        axes[1].plot(hist.index + 1, hist["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=150)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Rest", "Impact"])
    ax.set_yticks([0, 1], labels=["Rest", "Impact"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix")

    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    pred = np.argmax(probs, axis=1).astype(np.int64)
    matrix = confusion_matrix(y_true, pred, labels=[0, 1])
    accuracy = accuracy_score(y_true, pred)

    precision_each, recall_each, f1_each, support_each = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[0, 1],
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        pred,
        average="macro",
        zero_division=0,
    )
    precision_impact, recall_impact, f1_impact, _ = precision_recall_fscore_support(
        y_true,
        pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    tn, fp, fn, tp = matrix.ravel()
    rest_false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_impact": float(precision_impact),
        "recall_impact": float(recall_impact),
        "f1_impact": float(f1_impact),
        "impact_recall": float(recall_impact),
        "rest_false_positive_rate": rest_false_positive_rate,
        "per_class": {
            "Rest": {
                "precision": float(precision_each[0]),
                "recall": float(recall_each[0]),
                "f1": float(f1_each[0]),
                "support": int(support_each[0]),
            },
            "Impact": {
                "precision": float(precision_each[1]),
                "recall": float(recall_each[1]),
                "f1": float(f1_each[1]),
                "support": int(support_each[1]),
            },
        },
        "confusion_matrix_labels": ["Rest", "Impact"],
        "confusion_matrix": matrix.tolist(),
    }
    return metrics, matrix, pred


def evaluate_threshold(y_true: np.ndarray, prob_impact: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob_impact >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]

    accuracy = float(accuracy_score(y_true, pred)) if len(y_true) else 0.0
    impact_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    impact_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    impact_f1 = (
        float(2.0 * impact_precision * impact_recall / (impact_precision + impact_recall))
        if (impact_precision + impact_recall) > 0
        else 0.0
    )
    rest_false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    impact_false_negative_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": matrix.tolist(),
        "accuracy": accuracy,
        "impact_precision": impact_precision,
        "impact_recall": impact_recall,
        "impact_f1": impact_f1,
        "rest_false_positive_rate": rest_false_positive_rate,
        "impact_false_negative_rate": impact_false_negative_rate,
    }


def build_threshold_report(y_true: np.ndarray, probs: np.ndarray) -> list[dict[str, Any]]:
    prob_impact = probs[:, 1].astype(np.float32)
    return [evaluate_threshold(y_true, prob_impact, threshold) for threshold in THRESHOLD_VALUES]


def recommend_threshold(threshold_rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        recall_ok = [row for row in threshold_rows if row["impact_recall"] >= 0.75]
        if recall_ok:
            selected = min(
                recall_ok,
                key=lambda row: (row["rest_false_positive_rate"], -row["impact_f1"], -row["impact_recall"]),
            )
            reason = "no threshold met both targets; selected lowest Rest false positive rate while keeping Impact recall >= 75%"
        else:
            selected = max(
                threshold_rows,
                key=lambda row: (row["impact_f1"], -row["rest_false_positive_rate"], row["impact_recall"]),
            )
            reason = "no threshold kept Impact recall >= 75%; selected best Impact F1 compromise"

    return {
        "class_names": ["Rest", "Impact"],
        "impact_threshold": float(selected["threshold"]),
        "rule": "prob_impact >= impact_threshold -> Impact else Rest",
        "note": "recommended by threshold sweep on test set",
        "selection_reason": reason,
        "metrics": selected,
    }


def save_threshold_outputs(
    threshold_rows: list[dict[str, Any]],
    decision_config: dict[str, Any],
    csv_path: Path,
    json_path: Path,
    decision_path: Path,
) -> None:
    csv_rows = []
    for row in threshold_rows:
        csv_row = {key: value for key, value in row.items() if key != "confusion_matrix"}
        csv_row["confusion_matrix"] = json.dumps(row["confusion_matrix"], ensure_ascii=False)
        csv_rows.append(csv_row)

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(_as_json_ready(threshold_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    decision_path.write_text(
        json.dumps(_as_json_ready(decision_config), ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def representative_dataset(x_train: np.ndarray, max_count: int) -> Callable[[], Any]:
    count = min(max_count, len(x_train))
    if count <= 0:
        raise RuntimeError("Representative dataset 为空，无法量化。")

    def generator():
        for index in range(count):
            yield [x_train[index : index + 1].astype(np.float32)]

    return generator


def fail_if_not_full_int8(interpreter: tf.lite.Interpreter) -> None:
    floating_tensors = []
    for tensor in interpreter.get_tensor_details():
        dtype = tensor.get("dtype")
        if dtype in (np.float16, np.float32, np.float64):
            floating_tensors.append((tensor.get("name", "<unnamed>"), str(dtype)))
    if floating_tensors:
        raise RuntimeError(f"TFLite 模型仍包含浮点 tensor: {floating_tensors}")


def export_int8_tflite(
    model: tf.keras.Model,
    x_train: np.ndarray,
    output_path: Path,
    representative_count: int,
) -> dict[str, Any]:
    log("开始导出 full integer int8 TFLite 模型。")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(x_train, representative_count)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    if input_details["dtype"] != np.int8:
        raise RuntimeError(f"TFLite 输入不是 int8: {input_details['dtype']}")
    if output_details["dtype"] != np.int8:
        raise RuntimeError(f"TFLite 输出不是 int8: {output_details['dtype']}")
    fail_if_not_full_int8(interpreter)

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]
    info = {
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
        "input_dtype": str(input_details["dtype"]),
        "output_dtype": str(output_details["dtype"]),
        "input_shape": input_details["shape"].astype(int).tolist(),
        "output_shape": output_details["shape"].astype(int).tolist(),
        "model_size_bytes": int(output_path.stat().st_size),
        "representative_count": int(min(representative_count, len(x_train))),
    }
    return info


def export_c_header(tflite_path: Path, header_path: Path) -> int:
    data = tflite_path.read_bytes()
    bytes_per_line = 16
    lines = []
    for offset in range(0, len(data), bytes_per_line):
        chunk = data[offset : offset + bytes_per_line]
        lines.append("  " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")

    content = f"""#ifndef IMPACT_BINARY_MODEL_DATA_H_
#define IMPACT_BINARY_MODEL_DATA_H_

const unsigned char g_model[] = {{
{os.linesep.join(lines)}
}};

const int g_model_len = {len(data)};

#endif  // IMPACT_BINARY_MODEL_DATA_H_
"""
    header_path.write_text(content, encoding="utf-8", newline="\n")
    return len(data)


def save_window_predictions(
    meta_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
) -> None:
    output = meta_test.copy()
    output["true_label"] = y_true.astype(int)
    output["pred_label"] = y_pred.astype(int)
    output["prob_rest"] = probs[:, 0].astype(float)
    output["prob_impact"] = probs[:, 1].astype(float)

    first_columns = [
        "session_name",
        "window_start_index",
        "window_end_index",
        "true_label",
        "pred_label",
        "prob_rest",
        "prob_impact",
    ]
    remaining_columns = [col for col in output.columns if col not in first_columns]
    output = output[first_columns + remaining_columns]
    output.to_csv(output_path, index=False, encoding="utf-8-sig")


def print_random_test_windows(
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    seed: int,
    count: int = 5,
) -> None:
    rng = np.random.default_rng(seed)
    sample_count = min(count, len(x_test))
    if sample_count <= 0:
        return
    choices = rng.choice(len(x_test), size=sample_count, replace=False)
    log("随机抽取 test window 检查预处理输入:")
    for index in choices:
        first_values = ", ".join(f"{value:.4f}" for value in x_test[index, :10, 0].tolist())
        log(
            f"  true={int(y_test[index])}, pred={int(y_pred[index])}, "
            f"prob_rest={probs[index, 0]:.4f}, prob_impact={probs[index, 1]:.4f}, "
            f"前10点=[{first_values}]"
        )


def label_counts(values: np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in values.tolist())
    return {"Rest": int(counts.get(0, 0)), "Impact": int(counts.get(1, 0))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Rest/Impact 二分类模型并导出 TFLM 资源。")
    parser.add_argument("--data_dir", default="data_labeled", help="标注数据目录")
    parser.add_argument("--out_dir", default="outputs/impact_binary", help="输出目录")
    parser.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE, help="窗口长度，默认 128")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="窗口步长，默认 32")
    parser.add_argument("--impact_threshold", type=float, default=0.20, help="窗口内 Impact 点占比阈值")
    parser.add_argument("--baseline_points", type=int, default=16, help="窗口前 N 点中位数作为基线")
    parser.add_argument("--epochs", type=int, default=80, help="最大训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--representative_count", type=int, default=512, help="TFLite representative dataset 窗口数")
    parser.add_argument("--no_median_filter", action="store_true", help="关闭 3 点中值滤波")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.keras"
    tflite_path = out_dir / "model_int8.tflite"
    header_path = out_dir / "model_data.h"
    report_path = out_dir / "report.json"
    preprocess_path = out_dir / "preprocess_config.json"
    quantization_path = out_dir / "quantization_info.json"
    confusion_png_path = out_dir / "confusion_matrix.png"
    curve_png_path = out_dir / "training_curve.png"
    predictions_path = out_dir / "window_predictions.csv"
    threshold_csv_path = out_dir / "threshold_report.csv"
    threshold_json_path = out_dir / "threshold_report.json"
    decision_config_path = out_dir / "decision_config.json"

    log(f"读取数据目录: {data_dir}")
    windows = load_windows(
        data_dir=data_dir,
        window_size=args.window_size,
        stride=args.stride,
        impact_threshold=args.impact_threshold,
    )
    for message in windows.warnings:
        warn(message)

    log(f"生成窗口数: {len(windows.labels)}，窗口标签分布: {label_counts(windows.labels)}")
    log(f"session 数量: {windows.point_stats['session_count']}")

    split_sessions, split_warnings = choose_session_split(windows.meta, seed=args.seed)
    for message in split_warnings:
        warn(message)

    split = build_split_bundle(
        windows=windows,
        split_sessions=split_sessions,
        baseline_points=args.baseline_points,
        use_median_filter=not args.no_median_filter,
        window_size=args.window_size,
        stride=args.stride,
        impact_threshold=args.impact_threshold,
    )

    log(f"训练 session: {len(split.split_sessions['train'])}, 验证 session: {len(split.split_sessions['val'])}, 测试 session: {len(split.split_sessions['test'])}")
    log(f"训练窗口: {len(split.y_train)} {label_counts(split.y_train)}")
    log(f"验证窗口: {len(split.y_val)} {label_counts(split.y_val)}")
    log(f"测试窗口: {len(split.y_test)} {label_counts(split.y_test)}")

    if label_counts(split.y_test)["Impact"] == 0:
        warn("测试集中没有 Impact 窗口，请增加 Impact session 后再信任指标。")
    if label_counts(split.y_train)["Impact"] == 0:
        warn("训练集中没有 Impact 窗口，模型无法真正学习弹击。")

    preprocess_path.write_text(
        json.dumps(_as_json_ready(split.preprocess_config), ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    log(f"预处理配置已保存: {preprocess_path}")

    class_weight = compute_class_weight(split.y_train)
    log(f"类别权重 class_weight: {class_weight}")

    model = build_model(window_size=args.window_size, learning_rate=args.learning_rate)
    log("模型结构:")
    model.summary(print_fn=lambda line: log(f"  {line}"))

    callbacks: list[tf.keras.callbacks.Callback] = []
    monitor = "val_loss" if len(split.y_val) > 0 else "loss"
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=args.patience,
            restore_best_weights=True,
        )
    )

    validation_data = (split.x_val, split.y_val) if len(split.y_val) > 0 else None
    log("开始训练。")
    history = model.fit(
        split.x_train,
        split.y_train,
        validation_data=validation_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight if class_weight else None,
        verbose=2,
    )

    model.save(model_path)
    log(f"Keras 模型已保存: {model_path}")
    plot_training_curve(history, curve_png_path)
    log(f"训练曲线已保存: {curve_png_path}")

    train_eval = model.evaluate(split.x_train, split.y_train, verbose=0)
    val_eval = model.evaluate(split.x_val, split.y_val, verbose=0) if len(split.y_val) else [None, None]
    test_eval = model.evaluate(split.x_test, split.y_test, verbose=0)

    probs = model.predict(split.x_test, verbose=0)
    metrics, matrix, pred = evaluate_predictions(split.y_test, probs)
    plot_confusion_matrix(matrix, confusion_png_path)
    save_window_predictions(split.meta_test, split.y_test, pred, probs, predictions_path)
    threshold_rows = build_threshold_report(split.y_test, probs)
    decision_config = recommend_threshold(threshold_rows)
    save_threshold_outputs(
        threshold_rows,
        decision_config,
        threshold_csv_path,
        threshold_json_path,
        decision_config_path,
    )
    log(f"混淆矩阵已保存: {confusion_png_path}")
    log(f"测试窗口预测已保存: {predictions_path}")
    log(f"threshold report saved: {threshold_csv_path}")
    log(f"decision config saved: {decision_config_path}")

    quant_info = export_int8_tflite(model, split.x_train, tflite_path, args.representative_count)
    quantization_path.write_text(
        json.dumps(_as_json_ready(quant_info), ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    log(f"int8 TFLite 已保存: {tflite_path}")
    log(f"量化信息已保存: {quantization_path}")

    c_header_len = export_c_header(tflite_path, header_path)
    log(f"C 数组头文件已保存: {header_path} ({c_header_len} bytes)")

    report = {
        "task": "Rest / Impact binary classification",
        "class_mapping": {"0": "Rest / 静息", "1": "Impact / 弹击"},
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "window_config": {
            "window_size": args.window_size,
            "stride": args.stride,
            "impact_threshold": args.impact_threshold,
        },
        "data_stats": windows.point_stats,
        "split_sessions": split.split_sessions,
        "split_window_counts": {
            "train": {"total": int(len(split.y_train)), **label_counts(split.y_train)},
            "val": {"total": int(len(split.y_val)), **label_counts(split.y_val)},
            "test": {"total": int(len(split.y_test)), **label_counts(split.y_test)},
        },
        "class_weight": {str(key): value for key, value in class_weight.items()},
        "preprocess": split.preprocess_config,
        "model": {
            "keras_path": str(model_path),
            "tflite_int8_path": str(tflite_path),
            "c_header_path": str(header_path),
            "parameter_count": int(model.count_params()),
            "keras_size_bytes": int(model_path.stat().st_size),
            "tflite_size_bytes": int(tflite_path.stat().st_size),
        },
        "train_eval": {"loss": None if train_eval[0] is None else float(train_eval[0]), "accuracy": None if train_eval[1] is None else float(train_eval[1])},
        "val_eval": {"loss": None if val_eval[0] is None else float(val_eval[0]), "accuracy": None if val_eval[1] is None else float(val_eval[1])},
        "test_eval": {"loss": float(test_eval[0]), "accuracy": float(test_eval[1])},
        "test_metrics": metrics,
        "threshold_sweep": threshold_rows,
        "recommended_decision": decision_config,
        "quantization": quant_info,
        "warnings": windows.warnings + split_warnings,
        "artifacts": {
            "preprocess_config": str(preprocess_path),
            "report_json": str(report_path),
            "confusion_matrix_png": str(confusion_png_path),
            "training_curve_png": str(curve_png_path),
            "window_predictions_csv": str(predictions_path),
            "threshold_report_csv": str(threshold_csv_path),
            "threshold_report_json": str(threshold_json_path),
            "decision_config_json": str(decision_config_path),
            "quantization_info_json": str(quantization_path),
        },
    }
    report_path.write_text(
        json.dumps(_as_json_ready(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    log(f"训练报告已保存: {report_path}")

    log(
        "测试集指标: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision_impact={metrics['precision_impact']:.4f}, "
        f"impact_recall={metrics['impact_recall']:.4f}, "
        f"f1_impact={metrics['f1_impact']:.4f}, "
        f"Rest false positive rate={metrics['rest_false_positive_rate']:.4f}"
    )
    log(f"混淆矩阵 [[Rest->Rest, Rest->Impact], [Impact->Rest, Impact->Impact]]: {matrix.tolist()}")
    log(
        "recommended threshold: "
        f"{decision_config['impact_threshold']:.2f}, "
        f"Rest false positive rate={decision_config['metrics']['rest_false_positive_rate']:.4f}, "
        f"Impact recall={decision_config['metrics']['impact_recall']:.4f}, "
        f"Impact F1={decision_config['metrics']['impact_f1']:.4f}"
    )
    print_random_test_windows(split.x_test, split.y_test, pred, probs, seed=args.seed)


if __name__ == "__main__":
    main()
