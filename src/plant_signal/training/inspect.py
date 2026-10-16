"""Inspect labeled rest/impact CSV data before training.

This script only reads data under --data_dir. It does not modify source data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["sample_index", "time_s", "label_id", "label_name", "train_use"]
SIGNAL_COLUMNS = ["voltage_uV", "raw_value"]
CLASS_NAMES = {0: "Rest / 静息", 1: "Impact / 弹击"}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _log(message: str) -> None:
    print(f"[检查] {message}")


def _label_candidates(value: Any) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value)
    candidates = [text]

    # Some old exports may contain mojibake such as "闈欐伅" for "静息".
    for source_encoding in ("gbk", "latin1"):
        try:
            repaired = text.encode(source_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired not in candidates:
            candidates.append(repaired)

    return candidates


def label_name_to_binary(value: Any) -> int | None:
    """Map label_name to 0/1 strictly by the requested label-name rule."""
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def inspect_data(data_dir: Path) -> dict[str, Any]:
    csv_files = sorted(data_dir.rglob("samples_labeled.csv"))
    anomalies: list[dict[str, Any]] = []
    label_rows = Counter()
    label_points_all = Counter()
    label_points_train = Counter()
    train_use_counter = Counter()
    sessions: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "files": 0,
            "points": 0,
            "train_use_points": 0,
            "rest_points": 0,
            "impact_points": 0,
            "ignored_train_points": 0,
            "has_impact": False,
        }
    )
    signal_column_counter = Counter()
    file_summaries: list[dict[str, Any]] = []

    for csv_path in csv_files:
        rel_path = csv_path.relative_to(data_dir).as_posix()
        session_name, trial_name = _session_and_trial(data_dir, csv_path)
        file_anomalies: list[str] = []

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            anomalies.append({"file": rel_path, "issues": [f"读取失败: {exc}"]})
            continue

        missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        signal_col = next((col for col in SIGNAL_COLUMNS if col in df.columns), None)

        if missing_required:
            file_anomalies.append(f"缺少必要列: {missing_required}")
        if signal_col is None:
            file_anomalies.append("缺少信号列: voltage_uV/raw_value")
        if len(df) == 0:
            file_anomalies.append("空文件")

        if file_anomalies:
            anomalies.append({"file": rel_path, "issues": file_anomalies})
            continue

        signal_column_counter[signal_col] += 1
        sessions[session_name]["files"] += 1
        sessions[session_name]["points"] += int(len(df))

        train_use = pd.to_numeric(df["train_use"], errors="coerce")
        sample_index = pd.to_numeric(df["sample_index"], errors="coerce")
        signal = pd.to_numeric(df[signal_col], errors="coerce")

        if train_use.isna().any():
            file_anomalies.append(f"train_use 非数字/空值: {int(train_use.isna().sum())} 行")
        if sample_index.isna().any():
            file_anomalies.append(f"sample_index 非数字/空值: {int(sample_index.isna().sum())} 行")
        if signal.isna().any():
            file_anomalies.append(f"{signal_col} 非数字/空值: {int(signal.isna().sum())} 行")
        if sample_index.duplicated().any():
            file_anomalies.append(f"sample_index 重复: {int(sample_index.duplicated().sum())} 行")

        sorted_index = sample_index.dropna().to_numpy()
        if len(sorted_index) > 1:
            diffs = np.diff(sorted_index)
            if np.any(diffs < 0):
                file_anomalies.append("sample_index 非单调递增")
            gap_count = int(np.sum(diffs > 1))
        else:
            gap_count = 0

        train_use_counter.update(train_use.dropna().astype(int).astype(str).tolist())
        mapped = df["label_name"].map(label_name_to_binary)

        for row in df[["label_id", "label_name", "train_use"]].drop_duplicates().itertuples(index=False):
            label_rows[(str(row.label_id), str(row.label_name), str(row.train_use))] += 1

        label_points_all.update(
            "ignored" if pd.isna(value) else str(int(value)) for value in mapped.tolist()
        )

        use_mask = train_use == 1
        train_df = df.loc[use_mask].copy()
        train_mapped = mapped.loc[use_mask]
        sessions[session_name]["train_use_points"] += int(use_mask.sum())
        sessions[session_name]["rest_points"] += int((train_mapped == 0).sum())
        sessions[session_name]["impact_points"] += int((train_mapped == 1).sum())
        sessions[session_name]["ignored_train_points"] += int(train_mapped.isna().sum())
        sessions[session_name]["has_impact"] = sessions[session_name]["impact_points"] > 0
        label_points_train.update(
            "ignored" if pd.isna(value) else str(int(value)) for value in train_mapped.tolist()
        )

        if int(use_mask.sum()) == 0:
            file_anomalies.append("没有 train_use == 1 的样本")
        if int((train_mapped == 1).sum()) == 0:
            file_anomalies.append("该文件没有 Impact 点，可作为纯静息数据")
        if int(train_mapped.isna().sum()) > 0:
            file_anomalies.append(f"train_use == 1 中有未识别标签: {int(train_mapped.isna().sum())} 点")

        file_summary = {
            "file": rel_path,
            "session_name": session_name,
            "trial_name": trial_name,
            "rows": int(len(df)),
            "train_use_1_rows": int(use_mask.sum()),
            "rest_points": int((train_mapped == 0).sum()),
            "impact_points": int((train_mapped == 1).sum()),
            "ignored_train_points": int(train_mapped.isna().sum()),
            "sample_index_gaps": gap_count,
            "signal_column": signal_col,
        }
        file_summaries.append(file_summary)

        if file_anomalies:
            anomalies.append({"file": rel_path, "issues": file_anomalies})

    session_summary = {
        name: {
            **info,
            "has_impact": bool(info["impact_points"] > 0),
        }
        for name, info in sorted(sessions.items())
    }

    result = {
        "data_dir": str(data_dir),
        "csv_file_count": len(csv_files),
        "session_count": len(session_summary),
        "signal_column_usage": dict(signal_column_counter),
        "train_use_distribution": dict(train_use_counter),
        "unique_label_rows": {
            f"label_id={key[0]}, label_name={key[1]}, train_use={key[2]}": count
            for key, count in label_rows.items()
        },
        "point_distribution_all_rows": dict(label_points_all),
        "point_distribution_train_use_1": dict(label_points_train),
        "class_names": CLASS_NAMES,
        "sessions": session_summary,
        "impact_session_count": int(sum(1 for item in session_summary.values() if item["has_impact"])),
        "rest_only_session_count": int(sum(1 for item in session_summary.values() if not item["has_impact"])),
        "files": file_summaries,
        "anomalies": anomalies,
    }
    return _json_ready(result)


def print_summary(result: dict[str, Any]) -> None:
    _log(f"数据目录: {result['data_dir']}")
    _log(f"samples_labeled.csv 文件数: {result['csv_file_count']}")
    _log(f"session 数量: {result['session_count']}")
    _log(f"信号列使用情况: {result['signal_column_usage']}")
    _log(f"train_use 分布: {result['train_use_distribution']}")
    _log(f"全部点标签分布: {result['point_distribution_all_rows']}")
    _log(f"train_use == 1 标签分布: {result['point_distribution_train_use_1']}")
    _log(
        f"Impact session: {result['impact_session_count']}，"
        f"纯静息 session: {result['rest_only_session_count']}"
    )

    if result["anomalies"]:
        _log("发现需要注意的文件:")
        for item in result["anomalies"]:
            _log(f"  - {item['file']}: {'; '.join(item['issues'])}")
    else:
        _log("没有发现异常文件。")

    _log("逐 session 点数:")
    for name, info in result["sessions"].items():
        _log(
            f"  {name}: files={info['files']}, train_use={info['train_use_points']}, "
            f"Rest={info['rest_points']}, Impact={info['impact_points']}, "
            f"ignored={info['ignored_train_points']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 data_labeled 下的标注数据。")
    parser.add_argument("--data_dir", default="data_labeled", help="标注数据目录")
    parser.add_argument("--out_json", default="", help="可选：保存检查结果 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    result = inspect_data(data_dir)
    print_summary(result)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        _log(f"检查结果已保存: {out_path}")


if __name__ == "__main__":
    main()
