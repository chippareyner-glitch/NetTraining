"""Streamlit app for acquiring and annotating raw plant signal trials."""

from __future__ import annotations

import json
import gc
import os
import re
import stat
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from plant_signal.ui.collection import render_collection_page
from plant_signal.config import ANNOTATIONS_DIR, RAW_DATA_DIR, ROOT_DIR
from plant_signal.annotation.export import (
    DEFAULT_OUTPUT_DIR,
    rebuild_manifest_from_exports,
    remove_exported_trial,
    sync_labeled_trial_export,
)
from plant_signal.acquisition.trials import metadata_for_trial


BASE_RAW_COLUMNS = ["sample_index", "timestamp_ms", "time_s"]
TAG_ANNOTATIONS_DIR = ANNOTATIONS_DIR / "trial_tags"
TAG_NAME_MAP = {
    "impact": "弹击开始",
    "tap": "弹击开始",
    "impact_start": "弹击开始",
    "impact_end": "弹击结束",
    "touch_start": "触摸开始",
    "touch_end": "触摸结束",
    "response_start": "响应开始",
    "response_end": "响应结束",
    "recovery_end": "恢复结束",
    "noise_start": "噪声开始",
    "noise_end": "噪声结束",
    "ignore_start": "忽略开始",
    "ignore_end": "忽略结束",
    "custom": "自定义标记",
    "rest": "自定义标记",
    "transition": "响应开始",
    "noise": "噪声开始",
    "unlabeled": "忽略开始",
}
TAG_CHOICES = [
    "弹击开始",
    "弹击结束",
    "触摸开始",
    "触摸结束",
    "响应开始",
    "响应结束",
    "恢复结束",
    "噪声开始",
    "噪声结束",
    "忽略开始",
    "忽略结束",
    "自定义标记",
]
TAG_PAIR_RULES = [
    ("弹击开始", "弹击结束"),
    ("触摸开始", "触摸结束"),
    ("响应开始", "响应结束"),
    ("噪声开始", "噪声结束"),
    ("忽略开始", "忽略结束"),
]
KEYBOARD_TAGS = {
    "1": "弹击开始",
    "2": "弹击结束",
    "3": "触摸开始",
    "4": "触摸结束",
    "5": "噪声开始",
    "6": "噪声结束",
    "7": "忽略开始",
    "8": "忽略结束",
    "9": "响应开始",
    "0": "响应结束",
}
SHORTCUT_QUERY_KEYS = ["标签快捷键", "标注文件", "标签事件"]
ARMED_TAG_KEY = "armed_tag"
ARMED_TAG_FILE_KEY = "armed_tag_file"
PURE_REST_SESSION_TYPE = "pure_rest"
PURE_REST_STATUS_LABEL = "Pure Rest / 纯静息"
KEYBOARD_SHORTCUT_COMPONENT_DIR = Path(__file__).resolve().parent / "components" / "keyboard_shortcuts"
keyboard_shortcut_component = components.declare_component(
    "keyboard_shortcuts",
    path=str(KEYBOARD_SHORTCUT_COMPONENT_DIR),
)
STATUS_PRIORITY = {
    "未标注": 0,
    "部分标注": 1,
    PURE_REST_STATUS_LABEL: 2,
    "已标注": 3,
    "异常": 4,
}
DELETE_RETRY_DELAYS_S = (0.05, 0.1, 0.2, 0.4, 0.8, 1.2)


def ensure_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    TAG_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


def relative_path_string(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def parse_session_trial(path: Path) -> tuple[str, str]:
    """Parse IDs from old flat files and new data_raw/session/trial files."""
    if RAW_DATA_DIR.resolve() in path.resolve().parents and path.parent.resolve() != RAW_DATA_DIR.resolve():
        return path.parent.name, path.stem

    stem = path.stem
    match = re.search(r"(?P<session>[A-Za-z]\d+)[_-](?P<trial>[A-Za-z]\d+)", stem)
    if match:
        return match.group("session").upper(), match.group("trial").upper()

    parts = [part for part in re.split(r"[_\-\s]+", stem) if part]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return path.parent.name or "unknown_session", stem


@st.cache_data(show_spinner=False)
def load_raw_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError("原始 CSV 没有样本。")
    if "sample_index" not in dataframe.columns:
        raise ValueError("原始 CSV 缺少必需列：sample_index")

    dataframe = dataframe.copy()
    if "timestamp_ms" not in dataframe.columns:
        if "time_s" not in dataframe.columns:
            raise ValueError("原始 CSV 必须包含 timestamp_ms 或 time_s。")
        dataframe["timestamp_ms"] = pd.to_numeric(dataframe["time_s"], errors="raise") * 1000.0
    if "time_s" not in dataframe.columns:
        dataframe["time_s"] = pd.to_numeric(dataframe["timestamp_ms"], errors="raise") / 1000.0

    for column in BASE_RAW_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")

    signal_column = ""
    signal_type = ""
    signal_label = ""
    signal_unit = ""
    if "voltage_uV" in dataframe.columns and pd.to_numeric(dataframe["voltage_uV"], errors="coerce").notna().any():
        signal_column = "voltage_uV"
        signal_type = "voltage"
        signal_label = "电压"
        signal_unit = "微伏"
    elif "current_uA" in dataframe.columns and pd.to_numeric(dataframe["current_uA"], errors="coerce").notna().any():
        signal_column = "current_uA"
        signal_type = "current"
        signal_label = "电流"
        signal_unit = "微安"
    elif "raw_value" in dataframe.columns and pd.to_numeric(dataframe["raw_value"], errors="coerce").notna().any():
        signal_column = "raw_value"
        signal_type = "raw"
        signal_label = "原始值"
        signal_unit = ""
    elif "adc_raw" in dataframe.columns and pd.to_numeric(dataframe["adc_raw"], errors="coerce").notna().any():
        signal_column = "adc_raw"
        signal_type = "raw"
        signal_label = "原始ADC"
        signal_unit = ""
    else:
        raise ValueError("原始 CSV 必须包含 voltage_uV、current_uA、raw_value 或 adc_raw 中的至少一列。")

    dataframe[signal_column] = pd.to_numeric(dataframe[signal_column], errors="raise")
    dataframe["signal_value"] = dataframe[signal_column]
    dataframe["signal_type"] = signal_type
    dataframe["signal_label"] = signal_label
    dataframe["signal_unit"] = signal_unit
    dataframe["signal_column"] = signal_column
    if "adc_raw" in dataframe.columns:
        dataframe["adc_raw"] = pd.to_numeric(dataframe["adc_raw"], errors="coerce")

    if dataframe[[*BASE_RAW_COLUMNS, "signal_value"]].isna().any().any():
        raise ValueError("原始 CSV 的必需信号列中存在空值。")
    if dataframe["sample_index"].duplicated().any():
        raise ValueError("原始 CSV 中存在重复采样点。")

    dataframe = dataframe.sort_values("sample_index").reset_index(drop=True)
    if not dataframe["time_s"].is_monotonic_increasing:
        raise ValueError("原始 CSV 的时间列必须单调递增。")
    return dataframe


def signal_display_info(dataframe: pd.DataFrame) -> dict[str, str]:
    return {
        "label": str(dataframe["signal_label"].iloc[0]) if "signal_label" in dataframe.columns else "信号",
        "unit": str(dataframe["signal_unit"].iloc[0]) if "signal_unit" in dataframe.columns else "",
        "column": str(dataframe["signal_column"].iloc[0]) if "signal_column" in dataframe.columns else "signal_value",
    }


def nearest_sample_from_time(dataframe: pd.DataFrame, time_s: float) -> int:
    index = (dataframe["time_s"] - float(time_s)).abs().idxmin()
    return int(dataframe.iloc[index]["sample_index"])


def nearest_sample_from_index(dataframe: pd.DataFrame, sample_index: int) -> int:
    index = (dataframe["sample_index"] - int(sample_index)).abs().idxmin()
    return int(dataframe.iloc[index]["sample_index"])


def tag_annotation_path(session_id: str, trial_id: str) -> Path:
    safe_session = re.sub(r"[^A-Za-z0-9_-]+", "_", session_id)
    safe_trial = re.sub(r"[^A-Za-z0-9_-]+", "_", trial_id)
    return TAG_ANNOTATIONS_DIR / safe_session / f"{safe_trial}.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def tag_to_chinese(tag: Any) -> str:
    text = str(tag or "").strip()
    return TAG_NAME_MAP.get(text, text if text in TAG_CHOICES else "自定义标记")


def normalize_tag_record(tag: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(tag)
    normalized["tag"] = tag_to_chinese(normalized.get("tag"))
    return normalized


def load_tag_annotation(session_id: str, trial_id: str) -> tuple[dict[str, Any] | None, str | None]:
    path = tag_annotation_path(session_id, trial_id)
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"标注JSON无法读取：{exc}"
    if not isinstance(data, dict):
        return None, "标注JSON不是对象"
    tags = data.get("tags", [])
    if not isinstance(tags, list):
        return None, "tags 字段不是列表"
    data["tags"] = [normalize_tag_record(item) for item in tags if isinstance(item, dict)]
    return data, None


def _raw_path_from_file_key(file_key: str) -> Path:
    path = Path(file_key)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def _set_labeled_export_message(level: str, message: str) -> None:
    try:
        st.session_state.labeled_export_message = {"level": level, "message": message}
    except Exception:
        pass


def sync_labeled_export_after_annotation_save(
    session_id: str,
    trial_id: str,
    file_key: str,
    annotation_path: Path,
    completed: bool,
) -> None:
    raw_path = _raw_path_from_file_key(file_key)
    if not raw_path.exists():
        if completed:
            _set_labeled_export_message(
                "error",
                f"\u6807\u6ce8\u5df2\u4fdd\u5b58\uff0c\u4f46\u672a\u540c\u6b65\u5230\u8bad\u7ec3\u6587\u4ef6\u5939\uff1a\u627e\u4e0d\u5230\u539f\u59cbCSV {raw_path}",
            )
        return

    if not completed:
        try:
            removed = remove_exported_trial(DEFAULT_OUTPUT_DIR, session_id, trial_id)
            if removed:
                rebuild_manifest_from_exports(DEFAULT_OUTPUT_DIR)
                _set_labeled_export_message(
                    "info",
                    f"\u5df2\u4ece\u8bad\u7ec3\u6587\u4ef6\u5939\u79fb\u9664\u672a\u5b8c\u6210\u7684\u5bfc\u51fa\uff1a{session_id}/{trial_id}",
                )
        except Exception as exc:  # noqa: BLE001
            _set_labeled_export_message(
                "warning",
                f"\u6807\u6ce8\u5df2\u4fdd\u5b58\uff0c\u4f46\u6e05\u7406\u8bad\u7ec3\u6587\u4ef6\u5939\u65e7\u5bfc\u51fa\u5931\u8d25\uff1a{exc}",
            )
        return

    try:
        result = sync_labeled_trial_export(raw_path, annotation_path, DEFAULT_OUTPUT_DIR)
    except Exception as exc:  # noqa: BLE001 - surface export errors without losing the annotation.
        _set_labeled_export_message(
            "error",
            f"\u6807\u6ce8\u5df2\u4fdd\u5b58\uff0c\u4f46\u540c\u6b65\u5230\u8bad\u7ec3\u6587\u4ef6\u5939\u5931\u8d25\uff1a{exc}",
        )
        return

    if completed and result.get("exported"):
        _set_labeled_export_message(
            "success",
            f"\u5df2\u540c\u6b65\u5230\u8bad\u7ec3\u6587\u4ef6\u5939\uff1a{result.get('export_dir', DEFAULT_OUTPUT_DIR)}",
        )


def render_labeled_export_message() -> None:
    message = st.session_state.pop("labeled_export_message", None)
    if not isinstance(message, dict):
        return
    text = str(message.get("message", ""))
    if not text:
        return
    level = str(message.get("level", "info"))
    if level == "success":
        st.success(text)
    elif level == "warning":
        st.warning(text)
    elif level == "error":
        st.error(text)
    else:
        st.info(text)


def save_tag_annotation(session_id: str, trial_id: str, file_key: str, data: dict[str, Any]) -> None:
    path = tag_annotation_path(session_id, trial_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_created = data.get("created_at") or now_iso()
    tags = [normalize_tag_record(item) for item in data.get("tags", []) if isinstance(item, dict)]
    completed = bool(data.get("completed", False))
    rest_all = bool(completed and not tags)
    data = {
        "version": 1,
        "session_id": session_id,
        "trial_id": trial_id,
        "file_path": file_key,
        "completed": completed,
        "background_label": "静息",
        "rest_all": rest_all,
        "default_label": "静息",
        "unlabeled_policy": "未被事件、噪声或忽略区间覆盖的采样点按静息处理",
        "tags": tags,
        "created_at": existing_created,
        "updated_at": now_iso(),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    sync_labeled_export_after_annotation_save(session_id, trial_id, file_key, path, completed)


def valid_tags_for_dataframe(tags: list[dict[str, Any]], dataframe: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    valid: list[dict[str, Any]] = []
    first_time = float(dataframe["time_s"].iloc[0])
    last_time = float(dataframe["time_s"].iloc[-1])
    samples = set(dataframe["sample_index"].astype(int).tolist())
    for index, tag in enumerate(tags):
        if not isinstance(tag, dict):
            errors.append(f"标记 {index} 不是对象")
            continue
        try:
            time_s = float(tag.get("time_s"))
            sample_index = int(tag.get("sample_index"))
        except (TypeError, ValueError):
            errors.append(f"标记 {index} 缺少有效时间或采样点")
            continue
        if time_s < first_time or time_s > last_time or sample_index not in samples:
            errors.append(f"标记 {index} 超出波形范围")
            continue
        valid.append(tag)
    return valid, errors


def is_completed_trial_csv(path: Path) -> bool:
    if path.name.startswith(".") or path.name.endswith(".tmp") or path.name.endswith(".partial.csv"):
        return False
    if path.parent.resolve() != RAW_DATA_DIR.resolve():
        return re.fullmatch(r"trial_\d+\.csv", path.name, flags=re.IGNORECASE) is not None
    return path.suffix.lower() == ".csv"


def find_annotation_trial_files() -> list[Path]:
    ensure_directories()
    return sorted(
        [path for path in RAW_DATA_DIR.rglob("*.csv") if is_completed_trial_csv(path)],
        key=lambda item: relative_path_string(item).lower(),
    )


def trial_row_from_file(path: Path) -> dict[str, Any]:
    file_key = relative_path_string(path)
    session_id, trial_id = parse_session_trial(path)
    metadata = metadata_for_trial(path)
    session_metadata: dict[str, Any] = {}
    try:
        session_metadata = json.loads((path.parent / "session.json").read_text(encoding="utf-8"))
    except Exception:
        session_metadata = {}
    is_pure_rest = (
        metadata.get("session_type") == PURE_REST_SESSION_TYPE
        or bool(metadata.get("pure_rest_confirmed"))
        or session_metadata.get("session_type") == PURE_REST_SESSION_TYPE
        or bool(session_metadata.get("pure_rest_confirmed"))
    )
    annotation, annotation_error = load_tag_annotation(session_id, trial_id)

    row: dict[str, Any] = {
        "path": path,
        "file_key": file_key,
        "session_id": session_id,
        "trial_id": trial_id,
        "file_name": path.name,
        "采集时间": str(metadata.get("start_time", "")),
        "实际时长": metadata.get("actual_duration_s", ""),
        "样本数量": metadata.get("sample_count", ""),
        "试次类型": metadata.get("trial_type", ""),
        "标注状态": "未标注",
        "已有标记数量": 0,
        "最后标注时间": "",
        "异常原因": "",
        "最后修改时间": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }

    try:
        dataframe = load_raw_csv(str(path))
    except Exception as exc:  # noqa: BLE001
        row["标注状态"] = "异常"
        row["异常原因"] = f"原始CSV无法读取：{exc}"
        return row

    if not row["实际时长"]:
        row["实际时长"] = float(dataframe["time_s"].iloc[-1] - dataframe["time_s"].iloc[0])
    if not row["样本数量"]:
        row["样本数量"] = len(dataframe)

    if is_pure_rest:
        row["标注状态"] = PURE_REST_STATUS_LABEL
        row["已有Tag数量"] = 0
        row["最后标注时间"] = str(
            metadata.get("pure_rest_confirmed_at")
            or session_metadata.get("pure_rest_confirmed_at")
            or metadata.get("end_time", "")
        )
        return row

    if annotation_error:
        row["标注状态"] = "异常"
        row["异常原因"] = annotation_error
        return row
    if annotation is None:
        return row

    tags = annotation.get("tags", [])
    valid_tags, tag_errors = valid_tags_for_dataframe(tags, dataframe)
    row["已有标记数量"] = len(valid_tags)
    row["最后标注时间"] = str(annotation.get("updated_at", ""))
    if tag_errors:
        row["标注状态"] = "异常"
        row["异常原因"] = "; ".join(tag_errors[:3])
    elif annotation.get("completed") is True:
        row["标注状态"] = "已标注"
    elif not valid_tags:
        row["标注状态"] = "未标注"
    else:
        row["标注状态"] = "部分标注"
    return row


def load_trial_rows() -> list[dict[str, Any]]:
    return [trial_row_from_file(path) for path in find_annotation_trial_files()]


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def trial_delete_targets(path: Path) -> list[Path]:
    path = Path(path).resolve()
    if not path_is_within(path, RAW_DATA_DIR):
        raise ValueError("只能删除 data_raw 目录中的 Trial 文件。")
    if not is_completed_trial_csv(path):
        raise ValueError("只能删除已经完成采集的 Trial CSV。")

    session_id, trial_id = parse_session_trial(path)
    candidates: list[Path] = [
        path,
        path.with_suffix(".json"),
        path.with_name(f"{path.stem}.partial.csv"),
        path.with_name(f"{path.stem}.partial.csv.tmp"),
        tag_annotation_path(session_id, trial_id),
    ]
    candidates.extend(path.parent.glob(f".{path.stem}.partial.csv.*.tmp"))
    candidates.extend(path.parent.glob(f"{path.stem}.partial.csv.*.tmp"))

    targets: list[Path] = []
    seen: set[Path] = set()
    allowed_roots = [RAW_DATA_DIR, TAG_ANNOTATIONS_DIR]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.exists():
            continue
        if not any(path_is_within(resolved, root) for root in allowed_roots):
            raise ValueError(f"拒绝删除工作区外文件：{resolved}")
        if not resolved.is_file():
            raise ValueError(f"拒绝删除非文件路径：{resolved}")
        seen.add(resolved)
        targets.append(resolved)
    return targets


def clear_annotation_caches() -> None:
    try:
        load_raw_csv.clear()
    except Exception:
        pass


def unlink_with_retry(path: Path) -> None:
    last_error: PermissionError | None = None
    for index, delay_s in enumerate((0.0, *DELETE_RETRY_DELAYS_S)):
        if delay_s:
            time.sleep(delay_s)
        try:
            if not path.exists():
                return
            os.chmod(path, stat.S_IWRITE)
            path.unlink()
            return
        except PermissionError as exc:
            last_error = exc
            gc.collect()
            if index >= len(DELETE_RETRY_DELAYS_S):
                break
    if last_error is not None:
        raise PermissionError(
            f"无法删除：{path}。这个文件可能正被 Excel、记事本、另一个网页标签页、"
            "旧的 Python/Streamlit 进程或杀毒/同步软件占用。请关闭占用程序后再试。"
        ) from last_error


def cleanup_empty_trial_parents(raw_path: Path, annotation_path: Path | None) -> list[Path]:
    deleted: list[Path] = []
    raw_dir = raw_path.resolve().parent
    if raw_dir != RAW_DATA_DIR.resolve() and raw_dir.exists():
        remaining_trials = [
            item
            for item in raw_dir.iterdir()
            if item.is_file() and item.name.lstrip(".").lower().startswith("trial_")
        ]
        if not remaining_trials:
            session_json = raw_dir / "session.json"
            if session_json.exists() and session_json.is_file():
                unlink_with_retry(session_json)
                deleted.append(session_json.resolve())
            try:
                raw_dir.rmdir()
                deleted.append(raw_dir.resolve())
            except OSError:
                pass

    if annotation_path is not None:
        annotation_dir = annotation_path.resolve().parent
        if annotation_dir != TAG_ANNOTATIONS_DIR.resolve() and annotation_dir.exists():
            try:
                annotation_dir.rmdir()
                deleted.append(annotation_dir.resolve())
            except OSError:
                pass
    return deleted


def delete_completed_trial_data(path: Path) -> list[Path]:
    path = Path(path).resolve()
    session_id, trial_id = parse_session_trial(path)
    annotation_path = tag_annotation_path(session_id, trial_id)
    targets = trial_delete_targets(path)
    deleted: list[Path] = []
    clear_annotation_caches()
    for target in targets:
        unlink_with_retry(target)
        deleted.append(target)
    try:
        if remove_exported_trial(DEFAULT_OUTPUT_DIR, session_id, trial_id):
            rebuild_manifest_from_exports(DEFAULT_OUTPUT_DIR)
            deleted.append((DEFAULT_OUTPUT_DIR / session_id / trial_id).resolve())
    except Exception:
        pass
    deleted.extend(cleanup_empty_trial_parents(path, annotation_path))
    return deleted


def delete_request_key(file_key: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, file_key).hex


def clear_armed_tag() -> None:
    st.session_state.pop(ARMED_TAG_KEY, None)
    st.session_state.pop(ARMED_TAG_FILE_KEY, None)


def select_trial_for_annotation(path: Path) -> None:
    clear_armed_tag()
    st.session_state.selected_annotation_file = str(path)
    st.session_state.annotation_view = "detail"


def clear_selected_trial() -> None:
    clear_armed_tag()
    st.session_state.annotation_view = "list"
    st.session_state.selected_annotation_file = ""


def nearest_sample_from_selected_point(dataframe: pd.DataFrame, point: dict[str, Any]) -> dict[str, Any] | None:
    customdata = point.get("customdata")
    try:
        if isinstance(customdata, list):
            sample_index = int(customdata[0])
        else:
            sample_index = int(customdata)
    except (TypeError, ValueError):
        try:
            sample_index = nearest_sample_from_time(dataframe, float(point.get("x")))
        except (TypeError, ValueError):
            return None
    sample_index = nearest_sample_from_index(dataframe, sample_index)
    row = dataframe[dataframe["sample_index"] == sample_index].iloc[0]
    return {
        "sample_index": sample_index,
        "time_s": float(row["time_s"]),
        "signal_value": float(row["signal_value"]),
    }


def status_badge(status: str) -> str:
    colors = {
        "未标注": "#64748b",
        "部分标注": "#d97706",
        PURE_REST_STATUS_LABEL: "#0f766e",
        "已标注": "#15803d",
        "异常": "#dc2626",
    }
    color = colors.get(status, "#64748b")
    return f"<span style='color:{color};font-weight:700'>{status}</span>"


def sorted_trial_rows(rows: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if sort_by == "按采集时间":
        return sorted(rows, key=lambda row: str(row.get("采集时间", "")), reverse=True)
    if sort_by == "按文件名":
        return sorted(rows, key=lambda row: str(row.get("file_name", "")).lower())
    if sort_by == "按最后修改时间":
        return sorted(rows, key=lambda row: str(row.get("最后修改时间", "")), reverse=True)
    return sorted(
        rows,
        key=lambda row: (
            STATUS_PRIORITY.get(str(row.get("标注状态", "异常")), 9),
            str(row.get("采集时间", "")),
        ),
    )


def row_matches_search(row: dict[str, Any], search_text: str) -> bool:
    if not search_text.strip():
        return True
    haystack = " ".join(
        str(row.get(key, ""))
        for key in ["session_id", "trial_id", "file_name", "试次类型"]
    ).lower()
    return search_text.strip().lower() in haystack


def next_unfinished_row(rows: list[dict[str, Any]], start_path: Path | None = None) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("标注状态") in {"未标注", "部分标注"}]
    if not candidates:
        return None
    if start_path is None:
        return candidates[0]
    all_paths = [Path(row["path"]).resolve() for row in rows]
    try:
        start_index = all_paths.index(start_path.resolve())
    except ValueError:
        return candidates[0]
    for offset in range(1, len(rows) + 1):
        row = rows[(start_index + offset) % len(rows)]
        if row.get("标注状态") in {"未标注", "部分标注"}:
            return row
    return None


def dataframe_selected_rows(state: Any) -> list[int]:
    selection = getattr(state, "selection", None)
    if selection is None and isinstance(state, dict):
        selection = state.get("selection")
    rows = getattr(selection, "rows", None)
    if rows is None and isinstance(selection, dict):
        rows = selection.get("rows")
    return list(rows or [])


def render_trial_delete_confirmation(
    path: Path,
    session_id: str,
    trial_id: str,
    file_key: str,
    *,
    return_to_list: bool,
) -> None:
    request_key = delete_request_key(file_key)
    if st.session_state.get("delete_trial_request") != file_key:
        return

    targets = trial_delete_targets(path)
    target_names = [relative_path_string(target) for target in targets]
    st.warning(
        f"确认删除 {session_id} / {trial_id}？这会删除原始 CSV、Trial 元数据、残留 partial 文件和对应标注 JSON，操作不可恢复。"
    )
    if target_names:
        st.caption("将删除：" + "；".join(target_names))
    confirm = st.checkbox(
        "我确认删除这个 Trial 的原始数据和标注文件",
        key=f"confirm_delete_trial_{request_key}",
    )
    action_cols = st.columns([1, 1, 6])
    if action_cols[0].button("确认删除", disabled=not confirm, key=f"do_delete_trial_{request_key}"):
        try:
            deleted = delete_completed_trial_data(path)
        except PermissionError as exc:
            st.error(str(exc))
            st.info("处理办法：关闭可能打开该 CSV 的 Excel/记事本、其他浏览器标签页，或结束旧的 Python/Streamlit 进程后，再点一次确认删除。")
        except Exception as exc:  # noqa: BLE001
            st.error(f"删除失败：{exc}")
        else:
            st.session_state.delete_trial_notice = (
                f"已删除 {session_id} / {trial_id}，共删除 {len(deleted)} 个路径。"
            )
            st.session_state.pop("delete_trial_request", None)
            clear_armed_tag()
            st.session_state.pop(f"selected_point_{file_key}", None)
            st.session_state.pop(f"highlighted_tag_{file_key}", None)
            st.session_state.pop(f"last_added_tag_{file_key}", None)
            if return_to_list:
                clear_selected_trial()
            st.rerun()
    if action_cols[1].button("取消", key=f"cancel_delete_trial_{request_key}"):
        st.session_state.pop("delete_trial_request", None)
        st.rerun()


def render_trial_list_page() -> None:
    st.title("数据标注")
    st.caption("先从列表中选择一个已完成试次，再进入波形标注。")

    delete_notice = st.session_state.pop("delete_trial_notice", "")
    if delete_notice:
        st.success(delete_notice)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    status_filter = col1.selectbox("状态筛选", ["全部", "未标注", "部分标注", PURE_REST_STATUS_LABEL, "已标注", "异常"], index=0)
    search_text = col2.text_input("搜索", placeholder="实验 / 试次 / 文件名 / 试次类型")
    sort_by = col3.selectbox("排序", ["按标注状态", "按采集时间", "按文件名", "按最后修改时间"], index=0)
    if col4.button("刷新文件列表"):
        st.rerun()

    rows = load_trial_rows()
    display_rows = [
        row
        for row in rows
        if (status_filter == "全部" or row.get("标注状态") == status_filter)
        and row_matches_search(row, search_text)
    ]
    display_rows = sorted_trial_rows(display_rows, sort_by)

    unfinished = next_unfinished_row(sorted_trial_rows(rows, "按标注状态"))
    if st.button("打开下一个未标注文件", disabled=unfinished is None):
        select_trial_for_annotation(Path(unfinished["path"]))
        st.rerun()

    st.write(f"共 {len(rows)} 个完成试次，当前显示 {len(display_rows)} 个。")
    if not display_rows:
        st.info("没有符合条件的试次。")
        return

    header = st.columns([1.1, 1.4, 1.0, 1.5, 1.4, 0.9, 0.9, 1.1, 0.9, 1.3, 0.8, 0.8])
    for col, text in zip(
        header,
        ["状态", "实验", "试次", "文件名", "采集时间", "时长", "样本", "试次类型", "标记数", "最后标注", "打开", "删除"],
    ):
        col.markdown(f"**{text}**")

    for row in display_rows:
        row_path = Path(row["path"])
        row_file_key = relative_path_string(row_path)
        request_key = delete_request_key(row_file_key)
        cols = st.columns([1.1, 1.4, 1.0, 1.5, 1.4, 0.9, 0.9, 1.1, 0.9, 1.3, 0.8, 0.8])
        cols[0].markdown(status_badge(str(row["标注状态"])), unsafe_allow_html=True)
        cols[1].write(str(row["session_id"]))
        cols[2].write(str(row["trial_id"]))
        cols[3].write(str(row["file_name"]))
        cols[4].write(str(row["采集时间"])[:19])
        duration = row.get("实际时长", "")
        cols[5].write("" if duration == "" else f"{float(duration):.3f}s")
        cols[6].write(row.get("样本数量", ""))
        cols[7].write(row.get("试次类型", ""))
        cols[8].write(row.get("已有标记数量", 0))
        cols[9].write(str(row.get("最后标注时间", ""))[:19])
        if cols[10].button("打开", key=f"open_{row_file_key}"):
            select_trial_for_annotation(row_path)
            st.rerun()
        if cols[11].button("删除", key=f"delete_trial_{request_key}"):
            st.session_state.delete_trial_request = row_file_key
            st.rerun()
        if row.get("异常原因"):
            st.caption(f"{row['file_name']}：{row['异常原因']}")
        render_trial_delete_confirmation(
            row_path,
            str(row["session_id"]),
            str(row["trial_id"]),
            row_file_key,
            return_to_list=False,
        )


def tag_dataframe(tags: list[dict[str, Any]]) -> pd.DataFrame:
    if not tags:
        return pd.DataFrame(columns=["编号", "时间", "采样点", "标签", "备注", "创建时间"])
    rows = []
    for item in tags:
        rows.append(
            {
                "编号": item.get("id", ""),
                "时间": float(item.get("time_s", 0.0)),
                "采样点": int(item.get("sample_index", 0)),
                "标签": tag_to_chinese(item.get("tag")),
                "备注": item.get("note", ""),
                "创建时间": item.get("created_at", ""),
            }
        )
    return pd.DataFrame(rows)


def build_tag_figure(
    dataframe: pd.DataFrame,
    tags: list[dict[str, Any]],
    file_key: str,
    pending_point: dict[str, Any] | None = None,
    highlighted_tag_id: str | None = None,
) -> go.Figure:
    signal_info = signal_display_info(dataframe)
    signal_label = signal_info["label"]
    signal_unit = signal_info["unit"]
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=dataframe["time_s"],
            y=dataframe["signal_value"],
            customdata=dataframe["sample_index"],
            mode="lines+markers",
            name="波形",
            line={"color": "#2563eb", "width": 1},
            marker={"size": 4, "opacity": 0.18, "color": "#2563eb"},
            hovertemplate=f"时间=%{{x:.4f}}秒<br>{signal_label}=%{{y:.4f}}{signal_unit}<br>采样点=%{{customdata}}<extra></extra>",
        )
    )
    for tag in tags:
        sample_index = int(tag["sample_index"])
        point = dataframe[dataframe["sample_index"] == sample_index]
        if point.empty:
            continue
        x = float(point.iloc[0]["time_s"])
        y = float(point.iloc[0]["signal_value"])
        label = tag_to_chinese(tag["tag"])
        is_highlighted = str(tag.get("id", "")) == str(highlighted_tag_id or "")
        line_color = "#f59e0b" if is_highlighted else "#dc2626"
        marker_color = "#f59e0b" if is_highlighted else "#dc2626"
        marker_size = 15 if is_highlighted else 9
        line_width = 3 if is_highlighted else 1
        figure.add_vline(x=x, line_color=line_color, line_dash="dot", line_width=line_width)
        figure.add_trace(
            go.Scattergl(
                x=[x],
                y=[y],
                mode="markers+text",
                name=f"高亮：{label}" if is_highlighted else label,
                text=[label],
                textposition="top center",
                marker={"size": marker_size, "color": marker_color, "line": {"color": "#111827", "width": 2 if is_highlighted else 0}},
                hovertemplate=f"{label}<br>时间=%{{x:.4f}}秒<br>采样点={sample_index}<extra></extra>",
            )
        )
    if pending_point is not None:
        pending_time = float(pending_point["time_s"])
        pending_sample = int(pending_point["sample_index"])
        pending_value = float(pending_point.get("signal_value", 0.0))
        figure.add_vline(
            x=pending_time,
            line_color="#f59e0b",
            line_dash="dash",
            line_width=2,
            annotation_text="待确认",
            annotation_position="top right",
        )
        figure.add_trace(
            go.Scattergl(
                x=[pending_time],
                y=[pending_value],
                mode="markers+text",
                name="待确认位置",
                text=["待确认"],
                textposition="bottom center",
                marker={"size": 10, "color": "#f59e0b"},
                hovertemplate=f"待确认<br>时间=%{{x:.4f}}秒<br>{signal_label}=%{{y:.4f}}{signal_unit}<br>采样点={pending_sample}<extra></extra>",
            )
        )
    reset_count = int(st.session_state.get(f"plot_reset_{file_key}", 0))
    figure.update_layout(
        height=650,
        margin={"l": 40, "r": 24, "t": 28, "b": 42},
        xaxis_title="时间（秒）",
        yaxis_title=f"{signal_label}（{signal_unit}）" if signal_unit else signal_label,
        hovermode="x unified",
        dragmode="pan",
        clickmode="event+select",
        uirevision=f"{file_key}:{reset_count}",
        xaxis={
            "title": "时间（秒）",
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "cursor",
            "spikedash": "dot",
            "spikecolor": "rgba(120,120,120,0.85)",
            "spikethickness": 1,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    return figure


def tag_pair_warnings(tags: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    by_name = {name: 0 for pair in TAG_PAIR_RULES for name in pair}
    for item in tags:
        name = tag_to_chinese(item.get("tag"))
        if name in by_name:
            by_name[name] += 1
    for start_name, end_name in TAG_PAIR_RULES:
        if by_name[start_name] != by_name[end_name]:
            warnings.append(f"{start_name} 与 {end_name} 数量不一致：{by_name[start_name]} / {by_name[end_name]}")
    return warnings


def make_tag(
    selected_point: dict[str, Any],
    tag: str,
    note: str = "",
) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "time_s": float(selected_point["time_s"]),
        "sample_index": int(selected_point["sample_index"]),
        "tag": tag_to_chinese(tag),
        "note": note,
        "created_at": now_iso(),
    }


def remember_last_added_tag(file_key: str, tag: dict[str, Any]) -> None:
    tag_id = str(tag.get("id", ""))
    if tag_id:
        st.session_state[f"last_added_tag_{file_key}"] = tag_id


def remove_last_added_tag(tags: list[dict[str, Any]], file_key: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not tags:
        return tags, None

    last_id = str(st.session_state.get(f"last_added_tag_{file_key}", "") or "")
    if last_id:
        for index in range(len(tags) - 1, -1, -1):
            if str(tags[index].get("id", "")) == last_id:
                removed = tags[index]
                remaining = [*tags[:index], *tags[index + 1 :]]
                st.session_state.pop(f"last_added_tag_{file_key}", None)
                return remaining, removed

    removed = tags[-1]
    remaining = tags[:-1]
    st.session_state.pop(f"last_added_tag_{file_key}", None)
    return remaining, removed


def bump_annotation_widget_epoch(file_key: str) -> None:
    st.session_state[f"plot_widget_epoch_{file_key}"] = int(
        st.session_state.get(f"plot_widget_epoch_{file_key}", 0)
    ) + 1
    st.session_state[f"tag_table_epoch_{file_key}"] = int(
        st.session_state.get(f"tag_table_epoch_{file_key}", 0)
    ) + 1


def shortcut_help_text() -> str:
    ordered = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    return "　".join(f"{key} {KEYBOARD_TAGS[key]}" for key in ordered) + "　P 撤销上一个标记　Esc 取消选点"


def install_keyboard_shortcuts(
    file_key: str,
    enabled: bool,
    has_pending_point: bool = False,
    initial_tag: str = "",
) -> dict[str, Any] | None:
    component_key = "keyboard_shortcuts_disabled"
    if file_key:
        component_key = f"keyboard_shortcuts_{uuid.uuid5(uuid.NAMESPACE_URL, file_key).hex}"
    value = keyboard_shortcut_component(
        fileKey=file_key,
        enabled=enabled,
        hasPendingPoint=has_pending_point,
        initialTag=initial_tag,
        tagByKey=KEYBOARD_TAGS,
        helpText=shortcut_help_text(),
        key=component_key,
        default=None,
    )
    return value if isinstance(value, dict) else None


def clear_shortcut_query_params() -> None:
    for name in SHORTCUT_QUERY_KEYS:
        try:
            del st.query_params[name]
        except KeyError:
            pass


def shortcut_event_from_key(key: str) -> dict[str, str] | None:
    if key == "Escape":
        return {"action": "cancel"}
    if str(key).lower() == "p":
        return {"action": "undo"}

    tag = KEYBOARD_TAGS.get(str(key))
    if tag is None:
        return None
    return {"action": "tag", "tag": tag}


def consume_shortcut_event(file_key: str, component_value: dict[str, Any] | None = None) -> dict[str, str] | None:
    seen_key = "keyboard_shortcut_seen_event"
    if isinstance(component_value, dict) and component_value.get("source") == "plant-keyboard-shortcuts":
        event_id = str(component_value.get("eventId") or "")
        key = str(component_value.get("key") or "")
        target_file = str(component_value.get("fileKey") or "")
        if event_id and key and target_file == file_key and st.session_state.get(seen_key) != event_id:
            st.session_state[seen_key] = event_id
            clear_shortcut_query_params()
            return shortcut_event_from_key(key)

    params = st.query_params
    key = params.get("标签快捷键")
    target_file = params.get("标注文件")
    event_id = params.get("标签事件")
    if isinstance(key, list):
        key = key[-1] if key else None
    if isinstance(target_file, list):
        target_file = target_file[-1] if target_file else None
    if isinstance(event_id, list):
        event_id = event_id[-1] if event_id else None
    if not key or target_file != file_key:
        return None

    if event_id and st.session_state.get(seen_key) == str(event_id):
        clear_shortcut_query_params()
        return None
    if event_id:
        st.session_state[seen_key] = str(event_id)

    clear_shortcut_query_params()
    return shortcut_event_from_key(str(key))


def render_single_file_annotation_page(path: Path) -> None:
    file_key = relative_path_string(path)
    session_id, trial_id = parse_session_trial(path)
    dataframe = load_raw_csv(str(path))
    metadata = metadata_for_trial(path)
    session_metadata: dict[str, Any] = {}
    try:
        session_metadata = json.loads((path.parent / "session.json").read_text(encoding="utf-8"))
    except Exception:
        session_metadata = {}
    is_pure_rest = (
        metadata.get("session_type") == PURE_REST_SESSION_TYPE
        or bool(metadata.get("pure_rest_confirmed"))
        or session_metadata.get("session_type") == PURE_REST_SESSION_TYPE
        or bool(session_metadata.get("pure_rest_confirmed"))
    )
    annotation, annotation_error = load_tag_annotation(session_id, trial_id)
    if annotation is None:
        annotation = {
            "session_id": session_id,
            "trial_id": trial_id,
            "file_path": file_key,
            "completed": False,
            "tags": [],
        }
    tags, tag_errors = valid_tags_for_dataframe(
        [normalize_tag_record(item) for item in list(annotation.get("tags", []))],
        dataframe,
    )
    if annotation_error or tag_errors:
        status = "异常"
    elif is_pure_rest:
        status = PURE_REST_STATUS_LABEL
    elif annotation.get("completed") is True:
        status = "已标注"
    elif tags:
        status = "部分标注"
    else:
        status = "未标注"

    all_rows = sorted_trial_rows(load_trial_rows(), "按文件名")
    all_paths = [Path(row["path"]).resolve() for row in all_rows]
    current_index = all_paths.index(path.resolve()) if path.resolve() in all_paths else 0

    top = st.columns([1.1, 1.1, 1.5, 0.9, 0.9, 0.9, 1.1, 0.9, 0.9, 1.1, 1.0])
    top[0].markdown(f"**实验**  \n{session_id}")
    top[1].markdown(f"**试次**  \n{trial_id}")
    top[2].markdown(f"**文件名**  \n{path.name}")
    top[3].markdown(f"**采样率**  \n{metadata.get('sample_rate_hz', '')}")
    top[4].markdown(f"**时长**  \n{float(dataframe['time_s'].iloc[-1] - dataframe['time_s'].iloc[0]):.3f}s")
    top[5].markdown(f"**样本数**  \n{len(dataframe)}")
    top[6].markdown(f"**状态**  \n{status}")
    if top[7].button("上一个", disabled=current_index <= 0):
        select_trial_for_annotation(Path(all_rows[current_index - 1]["path"]))
        st.rerun()
    if top[8].button("下一个", disabled=current_index >= len(all_rows) - 1):
        select_trial_for_annotation(Path(all_rows[current_index + 1]["path"]))
        st.rerun()
    next_row = next_unfinished_row(all_rows, path)
    if top[9].button("下个未标注", disabled=next_row is None):
        select_trial_for_annotation(Path(next_row["path"]))
        st.rerun()
    if top[10].button("返回列表"):
        clear_selected_trial()
        st.rerun()

    if annotation_error:
        st.error(annotation_error)
    for error in tag_errors:
        st.error(error)
    for warning in tag_pair_warnings(tags):
        st.warning(f"配对警告：{warning}")

    if st.button("恢复全图"):
        st.session_state[f"plot_reset_{file_key}"] = int(st.session_state.get(f"plot_reset_{file_key}", 0)) + 1
        st.rerun()

    delete_cols = st.columns([1.2, 5])
    if delete_cols[0].button("删除当前 Trial", key=f"delete_current_trial_{delete_request_key(file_key)}"):
        st.session_state.delete_trial_request = file_key
        st.rerun()
    render_trial_delete_confirmation(
        path,
        session_id,
        trial_id,
        file_key,
        return_to_list=True,
    )

    render_labeled_export_message()

    shortcut_message = st.session_state.pop("shortcut_message", "")
    if shortcut_message:
        st.info(shortcut_message)

    selected_point = st.session_state.get(f"selected_point_{file_key}")
    signal_info = signal_display_info(dataframe)
    if selected_point is not None and "signal_value" not in selected_point:
        row = dataframe[dataframe["sample_index"] == int(selected_point["sample_index"])]
        if not row.empty:
            selected_point["signal_value"] = float(row.iloc[0]["signal_value"])

    st.session_state.pop(f"armed_tag_{file_key}", None)
    st.session_state.pop(f"active_tag_{file_key}", None)
    if st.session_state.get(ARMED_TAG_FILE_KEY) != file_key:
        clear_armed_tag()
    armed_tag = st.session_state.get(ARMED_TAG_KEY, "")
    shortcut_value = install_keyboard_shortcuts(
        file_key,
        enabled=True,
        has_pending_point=selected_point is not None,
        initial_tag=str(armed_tag or ""),
    )
    shortcut_event = consume_shortcut_event(file_key, shortcut_value)
    if shortcut_event:
        selected_for_shortcut = st.session_state.get(f"selected_point_{file_key}")
        if shortcut_event["action"] == "cancel":
            st.session_state.pop(f"selected_point_{file_key}", None)
            clear_armed_tag()
            bump_annotation_widget_epoch(file_key)
            st.session_state.shortcut_message = "已取消当前选点和当前待用标签。"
            st.rerun()
        elif shortcut_event["action"] == "undo":
            updated_tags, removed_tag = remove_last_added_tag(tags, file_key)
            if removed_tag is None:
                st.session_state.shortcut_message = "没有可撤销的标记。"
                st.rerun()
            annotation["tags"] = updated_tags
            annotation["completed"] = False
            save_tag_annotation(session_id, trial_id, file_key, annotation)
            removed_id = str(removed_tag.get("id", ""))
            if st.session_state.get(f"highlighted_tag_{file_key}") == removed_id:
                st.session_state.pop(f"highlighted_tag_{file_key}", None)
            st.session_state.pop(f"selected_point_{file_key}", None)
            tags = updated_tags
            bump_annotation_widget_epoch(file_key)
            st.session_state.shortcut_message = (
                f"已撤销上一个标记：{tag_to_chinese(removed_tag.get('tag'))} "
                f"{float(removed_tag.get('time_s', 0.0)):.6f} 秒。"
            )
            st.rerun()
        elif shortcut_event["action"] == "tag":
            shortcut_tag = shortcut_event["tag"]
            if isinstance(selected_for_shortcut, dict):
                if "signal_value" not in selected_for_shortcut:
                    row = dataframe[dataframe["sample_index"] == int(selected_for_shortcut["sample_index"])]
                    if not row.empty:
                        selected_for_shortcut["signal_value"] = float(row.iloc[0]["signal_value"])
                new_tag = make_tag(selected_for_shortcut, shortcut_tag)
                tags = [*tags, new_tag]
                annotation["tags"] = tags
                annotation["completed"] = False
                save_tag_annotation(session_id, trial_id, file_key, annotation)
                st.session_state.pop(f"selected_point_{file_key}", None)
                clear_armed_tag()
                st.session_state[f"highlighted_tag_{file_key}"] = new_tag["id"]
                remember_last_added_tag(file_key, new_tag)
                bump_annotation_widget_epoch(file_key)
                st.session_state.shortcut_message = f"已添加：{shortcut_tag}；当前未选择标签。"
                st.rerun()
            else:
                st.session_state[ARMED_TAG_KEY] = shortcut_tag
                st.session_state[ARMED_TAG_FILE_KEY] = file_key
                armed_tag = shortcut_tag
                st.session_state.shortcut_message = f"当前待用标签：{shortcut_tag}。请在波形图上单击一次。"

    if armed_tag:
        st.success(f"当前待用标签：{armed_tag}。点击波形后只添加一次，添加成功后会自动取消。")
    else:
        st.info("当前未选择标签。请先按数字键选择待用标签，然后单击波形添加一次标记；也可以直接单击波形后添加少用标签。")

    highlighted_tag_id = st.session_state.get(f"highlighted_tag_{file_key}")
    figure = build_tag_figure(dataframe, tags, file_key, pending_point=selected_point, highlighted_tag_id=highlighted_tag_id)
    plot_epoch = int(st.session_state.get(f"plot_widget_epoch_{file_key}", 0))
    plot_state = st.plotly_chart(
        figure,
        use_container_width=True,
        key=f"tag_plot_{file_key}_{plot_epoch}",
        on_select="rerun",
        selection_mode="points",
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
    )
    selection = getattr(plot_state, "selection", None)
    points = getattr(selection, "points", []) if selection is not None else []
    if points:
        selected = nearest_sample_from_selected_point(dataframe, dict(points[-1]))
        if selected is not None:
            if armed_tag:
                new_tag = make_tag(selected, str(armed_tag))
                tags = [*tags, new_tag]
                annotation["tags"] = tags
                annotation["completed"] = False
                save_tag_annotation(session_id, trial_id, file_key, annotation)
                st.session_state.pop(f"selected_point_{file_key}", None)
                clear_armed_tag()
                st.session_state[f"highlighted_tag_{file_key}"] = new_tag["id"]
                remember_last_added_tag(file_key, new_tag)
                bump_annotation_widget_epoch(file_key)
                st.session_state.shortcut_message = f"已添加：{armed_tag}；当前未选择标签。"
                st.rerun()
            elif selected != st.session_state.get(f"selected_point_{file_key}"):
                st.session_state[f"selected_point_{file_key}"] = selected
                bump_annotation_widget_epoch(file_key)
                st.rerun()

    selected_point = st.session_state.get(f"selected_point_{file_key}")
    edit_cols = st.columns([1.2, 1.0, 1.1, 2.0, 1.0, 1.1])
    if selected_point:
        edit_cols[0].metric("当前选中时间", f"{float(selected_point['time_s']):.6f} 秒")
        edit_cols[1].metric("采样点", int(selected_point["sample_index"]))
        edit_cols[2].metric(signal_info["label"], f"{float(selected_point.get('signal_value', 0.0)):.6f} {signal_info['unit']}")
        st.warning("当前未选择待用标签。可以按数字键直接把这个位置标为对应标签，或选择少用标签并点击按钮。")
    else:
        edit_cols[0].info("点击波形上的采样点")
        edit_cols[1].write("")
        edit_cols[2].write("")
    tag = edit_cols[3].selectbox("少用标签", ["恢复结束", "自定义标记"])
    note = edit_cols[4].text_input("备注", value="")
    if edit_cols[5].button("添加少用标签", disabled=selected_point is None):
        new_tag = make_tag(selected_point, tag, note=note)
        annotation["tags"] = [*tags, new_tag]
        annotation["completed"] = False
        save_tag_annotation(session_id, trial_id, file_key, annotation)
        if f"selected_point_{file_key}" in st.session_state:
            del st.session_state[f"selected_point_{file_key}"]
        clear_armed_tag()
        st.session_state[f"highlighted_tag_{file_key}"] = new_tag["id"]
        remember_last_added_tag(file_key, new_tag)
        bump_annotation_widget_epoch(file_key)
        st.rerun()

    st.subheader("已保存标记")
    if tags:
        st.caption("保存规则：标记覆盖的事件按标签处理；没有被事件、噪声或忽略区间覆盖的其余部分默认按静息处理。")
        table_state = st.dataframe(
            tag_dataframe(tags),
            use_container_width=True,
            hide_index=True,
            key=f"tag_table_{file_key}_{int(st.session_state.get(f'tag_table_epoch_{file_key}', 0))}",
            on_select="rerun",
            selection_mode="single-row",
        )
        selected_rows = dataframe_selected_rows(table_state)
        if selected_rows:
            selected_index = int(selected_rows[0])
            if 0 <= selected_index < len(tags):
                selected_id = str(tags[selected_index].get("id", ""))
                if selected_id and selected_id != st.session_state.get(f"highlighted_tag_{file_key}"):
                    st.session_state[f"highlighted_tag_{file_key}"] = selected_id
                    st.rerun()
        delete_options = [f"{item['id']} | {item['time_s']:.6f}秒 | {tag_to_chinese(item['tag'])}" for item in tags]
        del_cols = st.columns([3, 1])
        selected_delete = del_cols[0].selectbox("删除标记", delete_options)
        if del_cols[1].button("删除所选"):
            delete_id = selected_delete.split(" | ", 1)[0]
            annotation["tags"] = [item for item in tags if item["id"] != delete_id]
            annotation["completed"] = False
            if st.session_state.get(f"last_added_tag_{file_key}") == delete_id:
                st.session_state.pop(f"last_added_tag_{file_key}", None)
            if st.session_state.get(f"highlighted_tag_{file_key}") == delete_id:
                st.session_state.pop(f"highlighted_tag_{file_key}", None)
            st.session_state.pop(f"selected_point_{file_key}", None)
            save_tag_annotation(session_id, trial_id, file_key, annotation)
            bump_annotation_widget_epoch(file_key)
            st.rerun()
    else:
        if annotation.get("completed") is True and annotation.get("rest_all") is True:
            st.success("已标注完成：整段静息。")
        else:
            st.info("还没有事件标记。如果这段采集全程都是静息，可以直接点击“标注完成（整段静息）”。")

    complete_label = "标注完成（整段静息）" if not tags else "标注完成"
    if st.button(complete_label, disabled=annotation.get("completed") is True):
        annotation["tags"] = tags
        annotation["completed"] = True
        annotation["rest_all"] = not bool(tags)
        annotation["default_label"] = "静息"
        annotation["unlabeled_policy"] = "未被事件、噪声或忽略区间覆盖的采样点按静息处理"
        save_tag_annotation(session_id, trial_id, file_key, annotation)
        bump_annotation_widget_epoch(file_key)
        st.rerun()

    if annotation.get("completed"):
        if st.button("取消标注完成状态"):
            annotation["tags"] = tags
            annotation["completed"] = False
            annotation["rest_all"] = False
            annotation["default_label"] = "静息"
            annotation["unlabeled_policy"] = "未被事件、噪声或忽略区间覆盖的采样点按静息处理"
            save_tag_annotation(session_id, trial_id, file_key, annotation)
            bump_annotation_widget_epoch(file_key)
            st.rerun()


def render_annotation_page() -> None:
    selected = st.session_state.get("selected_annotation_file")
    if st.session_state.get("annotation_view") == "detail" and selected:
        path = Path(selected)
        if path.exists():
            render_single_file_annotation_page(path)
            return
        clear_selected_trial()

    clear_shortcut_query_params()
    install_keyboard_shortcuts("", enabled=False)
    render_trial_list_page()


def main() -> None:
    st.set_page_config(page_title="Plant Signal Data Tool", layout="wide")
    ensure_directories()

    pending_page = st.session_state.pop("pending_app_page", None)
    if pending_page in {"数据采集", "数据标注"}:
        st.session_state.app_page = pending_page

    page = st.sidebar.radio("功能", ["数据采集", "数据标注"], key="app_page")
    if page == "数据采集":
        clear_shortcut_query_params()
        install_keyboard_shortcuts("", enabled=False)
        render_collection_page()
    else:
        render_annotation_page()


if __name__ == "__main__":
    main()
