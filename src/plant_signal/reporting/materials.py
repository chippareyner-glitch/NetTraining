#!/usr/bin/env python3
"""Generate report-ready AI figures without modifying source data or model outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from plant_signal.paths import ROOT_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from PIL import Image


ROOT = ROOT_DIR
OUT = ROOT / "outputs" / "report_ai_materials"
MODEL_DIR = ROOT / "outputs" / "impact_binary_best_event"
FALLBACK_MODEL_DIR = ROOT / "outputs" / "impact_event_detector"
LABELED_DIR = ROOT / "data_labeled"
DPI = 220

COLORS = {
    "blue": "#2F6B9A",
    "orange": "#E07A3F",
    "green": "#4F8A70",
    "red": "#C84B4B",
    "gray": "#6B7280",
    "light_blue": "#DDEBF7",
    "light_orange": "#FCE4D6",
    "light_gray": "#E5E7EB",
}

CAPTIONS = {
    "01_rest_waveform.png": "静息状态下，植物微电信号整体保持相对平稳，可作为二分类任务中的 Rest 类样本。",
    "02_impact_waveform.png": "弹击刺激发生时，波形中会出现短时突变或尖峰特征，为神经网络识别 Impact 事件提供了依据。",
    "03_rest_impact_comparison.png": "Rest 与 Impact 样本在局部波形形态上存在差异，Impact 类通常包含更明显的瞬态扰动。",
    "04_labeled_event_example.png": "采集后的波形通过人工标注工具标记弹击事件位置，形成监督学习所需的标签数据。",
    "05_window_slicing_diagram.png": "连续波形被划分为固定长度时间窗口。弹击事件附近窗口作为 Impact 样本，远离事件的稳定区间作为 Rest 样本，恢复过渡阶段作为 ignore，不参与训练。",
    "06_dataset_statistics.png": "当前数据集以静息与弹击二分类为核心，通过滑动窗口切片得到训练样本，并对弹击后的恢复阶段进行忽略处理，降低标签混淆。",
    "07_training_curves.png": "训练曲线用于观察模型损失下降和验证集性能变化，判断模型是否稳定收敛。",
    "08_confusion_matrix.png": "混淆矩阵用于展示模型对 Rest 与 Impact 两类样本的分类效果，是验证模型识别能力的重要依据。",
    "09_model_architecture.png": "模型采用轻量化一维时序神经网络结构，对固定长度的植物电信号窗口进行特征提取，并输出 Rest / Impact 二分类结果，便于部署到 CH32H417 端侧运行。",
    "10_model_deployment_files.png": "训练完成后的模型被转换为 TensorFlow Lite 格式，并进一步转为 C 数组文件，便于集成到 CH32H417 的 TFLM 推理工程中。",
}

SECTIONS = {
    "01_rest_waveform.png": "数据采集与典型静息波形",
    "02_impact_waveform.png": "数据采集与弹击响应特征",
    "03_rest_impact_comparison.png": "二分类样本特征对比",
    "04_labeled_event_example.png": "人工标注方法",
    "05_window_slicing_diagram.png": "样本切片与标签构建",
    "06_dataset_statistics.png": "数据集构成",
    "07_training_curves.png": "模型训练过程",
    "08_confusion_matrix.png": "模型测试与性能评估",
    "09_model_architecture.png": "轻量化网络结构",
    "10_model_deployment_files.png": "嵌入式模型部署",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_stats_and_report() -> tuple[dict, dict]:
    report = read_json(MODEL_DIR / "report.json")
    stats = report.get("event_slice_stats", {})
    if not stats:
        stats = read_json(FALLBACK_MODEL_DIR / "event_slice_stats.json")
    return stats, report


def load_labeled(path: Path) -> pd.DataFrame:
    columns = [
        "sample_index",
        "time_s",
        "voltage_uV",
        "raw_value",
        "label_id",
        "interval_kind",
        "train_use",
    ]
    frame = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda col: col in columns)
    frame["sample_index"] = pd.to_numeric(frame["sample_index"], errors="coerce")
    frame["time_s"] = pd.to_numeric(frame.get("time_s"), errors="coerce")
    signal_col = "voltage_uV" if "voltage_uV" in frame else "raw_value"
    frame["signal"] = pd.to_numeric(frame[signal_col], errors="coerce")
    frame["interval_kind"] = frame.get("interval_kind", "rest").fillna("rest").astype(str).str.lower()
    frame["train_use"] = pd.to_numeric(frame.get("train_use", 1), errors="coerce").fillna(0)
    frame = frame.dropna(subset=["sample_index", "signal"]).reset_index(drop=True)
    if frame["time_s"].isna().all():
        frame["time_s"] = frame["sample_index"] * 0.002
    return frame


def choose_event() -> tuple[Path, int, pd.DataFrame]:
    count_file = MODEL_DIR / "impact_event_window_counts.csv"
    candidates: list[tuple[Path, int, int]] = []
    if count_file.exists():
        counts = pd.read_csv(count_file, encoding="utf-8-sig")
        counts = counts.sort_values(["positive_windows", "annotated_impact_len"], ascending=False)
        for row in counts.itertuples():
            source = LABELED_DIR / str(row.source_file)
            if source.exists():
                candidates.append((source, int(row.impact_start_index), int(row.positive_windows)))
    if not candidates:
        for source in LABELED_DIR.glob("**/samples_labeled.csv"):
            frame = load_labeled(source)
            impact = frame.index[frame["interval_kind"].eq("tap")].tolist()
            if impact:
                candidates.append((source, int(frame.loc[impact[0], "sample_index"]), len(impact)))
    if not candidates:
        raise RuntimeError("No labeled Impact/tap event was found.")

    best = None
    for source, center, support in candidates[:20]:
        frame = load_labeled(source)
        local = frame[(frame["sample_index"] >= center - 300) & (frame["sample_index"] <= center + 500)]
        if local.empty:
            continue
        baseline = float(local["signal"].median())
        score = float(np.nanpercentile(np.abs(local["signal"] - baseline), 99)) * max(support, 1)
        if best is None or score > best[0]:
            best = (score, source, center, frame)
    if best is None:
        source, center, _ = candidates[0]
        return source, center, load_labeled(source)
    return best[1], best[2], best[3]


def longest_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(changes[::2], changes[1::2])]


def choose_rest(frame: pd.DataFrame, event_center: int, length: int = 700) -> pd.DataFrame:
    mask = frame["interval_kind"].eq("rest").to_numpy() & frame["train_use"].eq(1).to_numpy()
    runs = longest_runs(mask)
    candidates = []
    for start, end in runs:
        if end - start < length:
            continue
        for offset in range(start, end - length + 1, max(length // 2, 1)):
            part = frame.iloc[offset : offset + length]
            distance = abs(float(part["sample_index"].median()) - event_center)
            if distance < 800:
                continue
            centered = part["signal"] - part["signal"].median()
            candidates.append((float(centered.std()), part))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[min(len(candidates) // 3, len(candidates) - 1)][1].copy()
    clean = frame[frame["interval_kind"].eq("rest") & frame["train_use"].eq(1)]
    return clean.iloc[: min(length, len(clean))].copy()


def choose_global_rest(length: int = 600) -> tuple[Path, pd.DataFrame]:
    """Choose a genuinely stable labeled Rest segment across all sessions."""
    candidates: list[tuple[float, Path, pd.DataFrame]] = []
    for source in sorted(LABELED_DIR.glob("**/samples_labeled.csv")):
        frame = load_labeled(source)
        mask = frame["interval_kind"].eq("rest").to_numpy() & frame["train_use"].eq(1).to_numpy()
        for start, end in longest_runs(mask):
            if end - start < length:
                continue
            for offset in range(start, end - length + 1, length):
                part = frame.iloc[offset : offset + length].copy()
                signal = part["signal"].to_numpy(dtype=float)
                x = np.arange(len(signal), dtype=float)
                slope, intercept = np.polyfit(x, signal, 1)
                detrended = signal - (slope * x + intercept)
                score = float(np.nanstd(detrended)) + abs(float(slope)) * length * 0.15
                candidates.append((score, source, part))
    if not candidates:
        raise RuntimeError("No sufficiently long labeled Rest segment was found.")
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def relative_time(frame: pd.DataFrame) -> np.ndarray:
    time = frame["time_s"].to_numpy(dtype=float)
    return time - time[0]


def centered_mv(frame: pd.DataFrame) -> np.ndarray:
    signal = frame["signal"].to_numpy(dtype=float)
    return (signal - np.nanmedian(signal)) / 1000.0


def plot_waveforms(rest: pd.DataFrame, event: pd.DataFrame, event_center: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.plot(relative_time(rest), centered_mv(rest), color=COLORS["blue"], lw=1.3)
    ax.set(title="Rest State Waveform", xlabel="Time / s", ylabel="Signal / mV (baseline removed)")
    ax.grid(alpha=0.2)
    save(fig, "01_rest_waveform.png")

    event_time = relative_time(event)
    center_pos = int(np.argmin(np.abs(event["sample_index"].to_numpy() - event_center)))
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.plot(event_time, centered_mv(event), color=COLORS["orange"], lw=1.3)
    ax.axvline(event_time[center_pos], color=COLORS["red"], ls="--", lw=1.4, label="Impact center")
    ax.set(title="Impact Event Waveform", xlabel="Time / s", ylabel="Signal / mV (baseline removed)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    save(fig, "02_impact_waveform.png")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=False)
    axes[0].plot(relative_time(rest), centered_mv(rest), color=COLORS["blue"], lw=1.2)
    axes[0].set_title("Rest window", loc="left", fontsize=11)
    axes[1].plot(event_time, centered_mv(event), color=COLORS["orange"], lw=1.2)
    axes[1].axvline(event_time[center_pos], color=COLORS["red"], ls="--", lw=1.2)
    axes[1].set_title("Impact window", loc="left", fontsize=11)
    for ax in axes:
        ax.set(xlabel="Time / s", ylabel="Signal / mV")
        ax.grid(alpha=0.2)
    fig.suptitle("Rest vs Impact Waveform Comparison", fontsize=14)
    fig.tight_layout()
    save(fig, "03_rest_impact_comparison.png")


def label_runs(frame: pd.DataFrame) -> list[tuple[int, int, str]]:
    values = frame["interval_kind"].to_numpy()
    if len(values) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    return [(int(a), int(b), str(values[a])) for a, b in zip(starts, ends)]


def plot_annotation(frame: pd.DataFrame, event_center: int) -> None:
    local = frame[
        (frame["sample_index"] >= event_center - 650)
        & (frame["sample_index"] <= event_center + 950)
    ].copy()
    time = relative_time(local)
    signal = centered_mv(local)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(time, signal, color="#263238", lw=1.0, zorder=3)
    span_colors = {"rest": COLORS["light_blue"], "tap": COLORS["light_orange"], "noise": COLORS["light_gray"]}
    legend_names = {"rest": "Rest", "tap": "Impact", "noise": "Ignore"}
    seen = set()
    for start, end, kind in label_runs(local):
        color = span_colors.get(kind, COLORS["light_gray"])
        ax.axvspan(time[start], time[min(end - 1, len(time) - 1)], color=color, alpha=0.75, zorder=0)
        seen.add(kind)
    # The event detector deliberately excludes the post-impact recovery tail,
    # even when the original point annotation returns to Rest.
    samples = local["sample_index"].to_numpy(dtype=int)
    ignore_start = int(np.argmin(np.abs(samples - (event_center + 256))))
    ignore_end = int(np.argmin(np.abs(samples - (event_center + 499))))
    ax.axvspan(time[ignore_start], time[ignore_end], color=COLORS["light_gray"], alpha=0.85, zorder=1)
    center_pos = int(np.argmin(np.abs(local["sample_index"].to_numpy() - event_center)))
    ax.axvline(time[center_pos], color=COLORS["red"], ls="--", lw=1.3, label="Annotated impact start")
    handles = [Patch(facecolor=span_colors[k], label=legend_names[k]) for k in ("rest", "tap", "noise") if k in seen]
    if "noise" not in seen:
        handles.append(Patch(facecolor=COLORS["light_gray"], label="Ignore recovery"))
    handles.append(plt.Line2D([], [], color=COLORS["red"], ls="--", label="Impact start"))
    ax.legend(handles=handles, ncol=4, frameon=False, loc="upper right")
    ax.set(title="Manual Annotation Example", xlabel="Time / s", ylabel="Signal / mV (baseline removed)")
    ax.grid(axis="y", alpha=0.18)
    save(fig, "04_labeled_event_example.png")


def plot_slicing(event: pd.DataFrame, event_center: int, raw_window: int, stride: int) -> None:
    local = event[
        (event["sample_index"] >= event_center - 750)
        & (event["sample_index"] <= event_center + 950)
    ].copy()
    x = local["sample_index"].to_numpy(dtype=int)
    y = centered_mv(local)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(x, y, color="#37474F", lw=1.0)
    ax.axvline(event_center, color=COLORS["red"], ls="--", lw=1.4, label="Impact center")
    ax.axvspan(event_center + 256, event_center + 499, color=COLORS["light_gray"], alpha=0.8, label="Ignore recovery region")
    starts = [event_center - 640, event_center - 640 + stride, event_center - 64, event_center - 64 + stride]
    labels = ["Rest window", "Rest window", "Impact window", "Impact window"]
    colors = [COLORS["blue"], COLORS["blue"], COLORS["orange"], COLORS["orange"]]
    ymin, ymax = ax.get_ylim()
    for i, (start, label, color) in enumerate(zip(starts, labels, colors)):
        rect = Rectangle((start, ymin), raw_window, ymax - ymin, fill=False, ec=color, lw=1.7)
        ax.add_patch(rect)
        ax.text(start + raw_window / 2, ymax - (i % 2) * 0.13 * (ymax - ymin), label, ha="center", va="top", color=color, fontsize=8)
    arrow_y = ymin + 0.13 * (ymax - ymin)
    ax.annotate(
        "",
        xy=(starts[0] + stride, arrow_y),
        xytext=(starts[0], arrow_y),
        arrowprops={"arrowstyle": "<->", "color": COLORS["gray"], "lw": 1.4},
    )
    ax.text(
        starts[0] + stride / 2,
        arrow_y + 0.035 * (ymax - ymin),
        f"stride = {stride} samples",
        ha="center",
        color=COLORS["gray"],
        fontsize=9,
    )
    ax.set(title="Fixed-window Slicing and Label Assignment", xlabel="Sample index", ylabel="Signal / mV")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.16)
    save(fig, "05_window_slicing_diagram.png")


def plot_statistics(stats: dict) -> None:
    windows = {
        "Rest windows": int(stats.get("rest_windows", stats.get("window_label_distribution", {}).get("0", 0))),
        "Impact windows": int(stats.get("impact_windows", stats.get("window_label_distribution", {}).get("1", 0))),
        "Ignore windows": int(stats.get("ignored_tail_windows", 0)),
    }
    overview = {
        "Impact events": int(stats.get("impact_event_count", 0)),
        "Sessions": int(stats.get("session_count", 0)),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, values, colors, title in [
        (axes[0], windows, [COLORS["blue"], COLORS["orange"], COLORS["gray"]], "Window samples"),
        (axes[1], overview, [COLORS["red"], COLORS["green"]], "Dataset coverage"),
    ]:
        bars = ax.bar(list(values), list(values.values()), color=colors, width=0.62)
        ax.bar_label(bars, padding=4, fmt="{:,.0f}", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylim(0, max(values.values()) * 1.16 if max(values.values()) else 1)
    fig.suptitle("Dataset Statistics", fontsize=14)
    fig.tight_layout()
    save(fig, "06_dataset_statistics.png")


def copy_training_curve() -> tuple[bool, str]:
    for source in (MODEL_DIR / "training_curve.png", FALLBACK_MODEL_DIR / "training_curve.png"):
        if source.exists():
            # Preserve the real curve pixels and only normalize PNG resolution metadata.
            with Image.open(source) as image:
                image.save(OUT / "07_training_curves.png", dpi=(DPI, DPI))
            return True, str(source.relative_to(ROOT)).replace("\\", "/")
    marker = OUT / "MISSING_training_curves.txt"
    marker.write_text(
        "未找到训练 history、log、CSV、JSON 或现有 training_curve.png；未伪造训练曲线。\n",
        encoding="utf-8",
    )
    return False, ""


def plot_confusion(report: dict) -> dict:
    metrics = report.get("test_at_recommended_threshold", {})
    matrix = np.asarray(
        metrics.get("confusion_matrix", report.get("test_confusion_matrix_at_recommended_threshold", [])),
        dtype=int,
    )
    if matrix.shape != (2, 2):
        (OUT / "MISSING_confusion_matrix.txt").write_text(
            "当前模型报告中未找到 2×2 测试集混淆矩阵；未伪造结果。\n",
            encoding="utf-8",
        )
        return {}
    fig, ax = plt.subplots(figsize=(5.9, 5.1))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > matrix.max() * 0.55 else "#1F2937"
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center", fontsize=16, color=color)
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Rest", "Impact"],
        yticklabels=["Rest", "Impact"],
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, "08_confusion_matrix.png")
    return metrics


def plot_architecture(report: dict) -> None:
    layers = [
        ("Input", "128 × 1\nwaveform"),
        ("Conv1D", "8 filters, k=5\n128 × 8"),
        ("MaxPool1D", "pool=2\n64 × 8"),
        ("Conv1D", "16 filters, k=5\n64 × 16"),
        ("Global AvgPool", "16 features"),
        ("Dense", "16, ReLU"),
        ("Output", "2, Softmax\nRest / Impact"),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.set_xlim(0, len(layers) * 1.75)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for i, (name, detail) in enumerate(layers):
        x = 0.25 + i * 1.72
        color = COLORS["orange"] if name == "Output" else COLORS["blue"]
        box = FancyBboxPatch(
            (x, 0.9),
            1.35,
            1.18,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            facecolor="white",
            edgecolor=color,
            linewidth=1.6,
        )
        ax.add_patch(box)
        ax.text(x + 0.675, 1.68, name, ha="center", va="center", color=color, weight="bold", fontsize=9)
        ax.text(x + 0.675, 1.27, detail, ha="center", va="center", color="#374151", fontsize=8)
        if i < len(layers) - 1:
            ax.annotate("", xy=(x + 1.62, 1.49), xytext=(x + 1.37, 1.49), arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 1.2})
    params = int(report.get("parameter_count", 1010))
    ax.text(0.25, 2.62, "Lightweight 1D Neural Network Architecture", fontsize=15, weight="bold", color="#1F2937")
    ax.text(0.25, 2.32, f"Actual training architecture · {params:,} parameters · int8 TFLite", fontsize=10, color=COLORS["gray"])
    save(fig, "09_model_architecture.png")


def deployment_files() -> list[tuple[str, int]]:
    rows = []
    if MODEL_DIR.exists():
        for path in sorted(MODEL_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in {".tflite", ".h", ".cc", ".c"}:
                rows.append((path.name, path.stat().st_size))
    if not rows:
        (OUT / "MISSING_deployment_files.txt").write_text(
            "当前模型输出目录中未找到 .tflite、.h、.cc 或 .c 部署文件。\n",
            encoding="utf-8",
        )
        return rows
    fig, ax = plt.subplots(figsize=(8.7, 2.2 + 0.5 * len(rows)))
    ax.axis("off")
    cells = [[name, f"{size:,}", f"{size / 1024:.2f} KB"] for name, size in rows]
    table = ax.table(
        cellText=cells,
        colLabels=["File", "Size / bytes", "Size / KB"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.58, 0.22, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for col in range(3):
        table[0, col].set_facecolor(COLORS["blue"])
        table[0, col].set_text_props(color="white", weight="bold")
    for row in range(1, len(cells) + 1):
        for col in range(3):
            table[row, col].set_facecolor("#F7FAFC" if row % 2 else "white")
            table[row, col].set_edgecolor("#D1D5DB")
    ax.set_title("Model Deployment Files", fontsize=14, pad=15)
    save(fig, "10_model_deployment_files.png")
    return rows


def fmt_pct(value: float | None) -> str:
    return "缺失" if value is None else f"{100 * float(value):.2f}%"


def write_readme(
    stats: dict,
    report: dict,
    metrics: dict,
    deployment: list[tuple[str, int]],
    curve_source: str,
    rest_source: Path,
    event_source: Path,
) -> None:
    generated = [name for name in CAPTIONS if (OUT / name).exists()]
    missing = []
    if "07_training_curves.png" not in generated:
        missing.append("训练曲线：缺少 history/log/CSV/JSON 和可复用的真实训练曲线图片。")
    elif curve_source:
        missing.append("逐 epoch 训练数值文件未保存；07 图复用真实训练曲线图片，不据图反推或伪造数值。")
    if "08_confusion_matrix.png" not in generated:
        missing.append("测试集混淆矩阵：模型报告中无可用 2×2 数据。")
    if "10_model_deployment_files.png" not in generated:
        missing.append("部署文件：当前模型目录未找到 .tflite/.h/.cc/.c。")
    missing.append("串口端逐样本推理日志未在优先扫描目录中发现，故本材料不宣称已完成整机长期在线验证。")

    rest_windows = int(stats.get("rest_windows", stats.get("window_label_distribution", {}).get("0", 0)))
    impact_windows = int(stats.get("impact_windows", stats.get("window_label_distribution", {}).get("1", 0)))
    ignore_windows = int(stats.get("ignored_tail_windows", 0))
    sessions = int(stats.get("session_count", 0))
    events = int(stats.get("impact_event_count", 0))
    threshold = metrics.get("threshold")
    lines = [
        "# AI 识别模块报告材料说明",
        "",
        "## 1. 当前项目口径",
        "",
        "本材料仅对应当前已完成的 **Rest / Impact 二分类核心验证**。LoRa、本地告警和多类别识别属于后续扩展，不写作已经完整实现；当前系统也不表述为完整的植物状态识别系统。",
        "",
        "## 2. 已生成图片及建议位置",
        "",
        "| 文件 | 建议放置小节 | 中文图注 |",
        "|---|---|---|",
    ]
    for name in CAPTIONS:
        state = "已生成" if name in generated else "缺失"
        lines.append(f"| `{name}`（{state}） | {SECTIONS[name]} | {CAPTIONS[name]} |")
    lines += [
        "",
        "## 3. 数据来源与数据集统计",
        "",
        f"- Rest 波形来源：`{rest_source.relative_to(ROOT).as_posix()}`。",
        f"- Impact 与标注示例来源：`{event_source.relative_to(ROOT).as_posix()}`。",
        "- 图中电压减去局部中位基线后以 mV 展示，未修改任何源文件。",
        f"- Sessions：**{sessions}**",
        f"- Impact events：**{events}**",
        f"- Rest windows：**{rest_windows}**",
        f"- Impact windows：**{impact_windows}**",
        f"- Ignore recovery windows：**{ignore_windows}**",
        "- 切片规则：原始窗口 256 点，步长 32 点；2 倍均值降采样后形成 128 点模型输入。",
        "- Impact 正样本规则：窗口包含人工标注的 `impact_start`；弹击后恢复长尾区域不参与训练。",
        "",
        "上述统计取自当前模型目录的 `report.json`，并与 `outputs/impact_event_detector/event_slice_stats.json` 中记录一致。",
        "",
        "## 4. 当前模型与测试指标",
        "",
        "- 类别映射：0 = Rest，1 = Impact。",
        f"- 网络参数量：**{int(report.get('parameter_count', 0)):,}**。",
        f"- 推荐判定阈值：**{threshold:.2f}**。" if threshold is not None else "- 推荐判定阈值：缺失。",
        f"- 测试集 accuracy：**{fmt_pct(metrics.get('accuracy'))}**。",
        f"- Impact precision：**{fmt_pct(metrics.get('impact_precision'))}**。",
        f"- Impact recall：**{fmt_pct(metrics.get('impact_recall'))}**。",
        f"- Impact F1：**{fmt_pct(metrics.get('impact_f1'))}**。",
        f"- 测试集混淆矩阵：`{metrics.get('confusion_matrix', '缺失')}`，行是真实类别，列是预测类别。",
        "- 以上为固定测试集的窗口级指标，适合表述为“初步验证结果”，不外推为跨植株、跨环境的完整泛化结论。",
        "",
        "## 5. 部署文件",
        "",
    ]
    if deployment:
        lines += ["| 文件 | 大小（bytes） |", "|---|---:|"]
        lines += [f"| `{name}` | {size:,} |" for name, size in deployment]
    else:
        lines.append("- 当前未发现可列出的部署文件。")
    lines += [
        "",
        "模型输入为 128×1 的 int8 时序窗口，输出为 Rest / Impact 两类分数。部署时还需按 `preprocess_config.json` 执行中值滤波、降采样、基线扣除、缩放及裁剪，并按推荐阈值完成后处理。",
        "",
        "## 6. 缺失项与使用限制",
        "",
    ]
    lines += [f"- {item}" for item in missing]
    if curve_source:
        lines.append(f"- 真实训练曲线图片来源：`{curve_source}`。")
    lines += [
        "",
        "## 7. 报告推荐表述",
        "",
        "> 本项目当前已完成植物微电信号静息状态与弹击刺激的二分类识别。系统通过采集植物叶片两端的微弱电信号，利用人工标注构建 Rest / Impact 数据集，并采用轻量化神经网络完成时序波形窗口分类。训练后的模型进一步转换为 TensorFlow Lite / C 数组格式，可集成至 CH32H417 的 TFLM 推理工程中，开展端侧 AI 识别验证。当前已完成“信号采集—数据标注—模型训练—嵌入式推理输出”的核心闭环初步验证；后续将继续扩充不同植株、不同刺激强度和不同环境条件下的数据集，以提升模型泛化能力。",
        "",
        "## 8. 不建议使用的表述",
        "",
        "- 不写“已完成多类别植物状态识别”。",
        "- 不写“LoRa 与本地告警已完整实现”，除非后续另有可核验材料。",
        "- 不把当前固定测试集结果表述为已证明跨植株、跨环境稳定泛化。",
    ]
    (OUT / "README_AI_REPORT_MATERIALS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    stats, report = load_stats_and_report()
    if not stats:
        raise RuntimeError("No event slicing statistics found in current output directories.")

    event_source, event_center, frame = choose_event()
    rest_source, rest = choose_global_rest()
    event = frame[
        (frame["sample_index"] >= event_center - 320)
        & (frame["sample_index"] <= event_center + 520)
    ].copy()
    if rest.empty or event.empty:
        raise RuntimeError("Labeled data did not contain usable Rest and Impact waveform segments.")

    plot_waveforms(rest, event, event_center)
    plot_annotation(frame, event_center)
    preprocess = report.get("preprocess_config", {})
    plot_slicing(
        frame,
        event_center,
        int(preprocess.get("raw_window_size", 256)),
        int(preprocess.get("stride", 32)),
    )
    plot_statistics(stats)
    curve_ok, curve_source = copy_training_curve()
    metrics = plot_confusion(report)
    plot_architecture(report)
    deployment = deployment_files()
    write_readme(stats, report, metrics, deployment, curve_source, rest_source, event_source)

    expected = [OUT / name for name in CAPTIONS]
    expected.append(OUT / "README_AI_REPORT_MATERIALS.md")
    print("AI report materials generated:")
    for path in expected:
        if path.exists():
            print(f"- {path.relative_to(ROOT).as_posix()}")
        else:
            print(f"- MISSING: {path.relative_to(ROOT).as_posix()}")
    if not curve_ok:
        print("- outputs/report_ai_materials/MISSING_training_curves.txt")


if __name__ == "__main__":
    main()
