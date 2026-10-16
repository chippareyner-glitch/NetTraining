"""Streamlit data acquisition page."""

from __future__ import annotations

import time
import uuid
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plant_signal.config import ANNOTATIONS_DIR
from plant_signal.annotation.export import DEFAULT_OUTPUT_DIR, sync_labeled_trial_export
from plant_signal.acquisition.serial_reader import SERIAL_BUSY_MESSAGE, SerialCollector, SerialPortInUseError, SerialPortOpenError, list_serial_ports
from plant_signal.acquisition.trials import (
    delete_partial_trial,
    finalize_trial,
    list_partial_trials,
    list_sessions,
    load_json,
    make_session_id,
    now_iso,
    paths_from_state,
    paths_to_state,
    relative_path_string,
    recover_partial_trial,
    reserve_trial,
    write_partial_csv,
)


TRIAL_TYPES = ["静止", "弹击", "触摸", "混合动作", "其他"]
EVENT_TYPES = {
    "弹击": "tap",
    "触摸开始": "touch_start",
    "触摸结束": "touch_end",
    "异常": "anomaly",
    "自定义备注": "note",
}
TAG_ANNOTATIONS_DIR = ANNOTATIONS_DIR / "trial_tags"
PURE_REST_SESSION_TYPE = "pure_rest"
PURE_REST_STATUS_LABEL = "Pure Rest / 纯静息"
COLLECTOR_CODE_VERSION = "ch0-voltage-uV-115200-post-filter-v1"


def _is_completed_trial_csv(path: Path) -> bool:
    if path.name.startswith(".") or path.name.endswith(".tmp") or path.name.endswith(".partial.csv"):
        return False
    return path.suffix.lower() == ".csv" and path.name.lower().startswith("trial_")


def _tag_annotation_path(session_id: str, trial_id: str) -> Path:
    safe_session = "".join(char if char.isalnum() or char in "_-" else "_" for char in session_id)
    safe_trial = "".join(char if char.isalnum() or char in "_-" else "_" for char in trial_id)
    return TAG_ANNOTATIONS_DIR / safe_session / f"{safe_trial}.json"


def _write_completed_rest_annotation(final_path: Path) -> Path:
    session_id = final_path.parent.name
    trial_id = final_path.stem
    annotation_path = _tag_annotation_path(session_id, trial_id)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation = {
        "version": 1,
        "session_id": session_id,
        "trial_id": trial_id,
        "file_path": relative_path_string(final_path),
        "completed": True,
        "background_label": "\u9759\u606f",
        "rest_all": True,
        "default_label": "\u9759\u606f",
        "unlabeled_policy": "\u6574\u6bb5\u539f\u59cb\u6570\u636e\u7531\u7528\u6237\u786e\u8ba4\u4f5c\u4e3a\u9759\u606f\uff0c\u4e0d\u81ea\u52a8\u751f\u6210\u523a\u6fc0\u6807\u7b7e\u3002",
        "tags": [],
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    annotation_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    return annotation_path


def _update_session_metadata(session_dir: Path, updates: dict[str, Any]) -> None:
    session_json = session_dir / "session.json"
    metadata = load_json(session_json)
    metadata.update(updates)
    metadata["updated_time"] = now_iso()
    session_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _session_status_summary(record: Any) -> dict[str, Any]:
    session_dir = Path(record.path)
    completed_trials = sorted(
        [path for path in session_dir.glob("trial_*.csv") if _is_completed_trial_csv(path)],
        key=lambda item: item.name.lower(),
    )
    partial_trials = {
        path
        for pattern in ("trial_*.partial.csv*", ".trial_*.partial.csv*.tmp")
        for path in session_dir.glob(pattern)
        if path.name.endswith(".partial.csv") or path.name.endswith(".tmp")
    }

    pure_rest_confirmed = (
        record.metadata.get("session_type") == PURE_REST_SESSION_TYPE
        or bool(record.metadata.get("pure_rest_confirmed"))
    )
    annotated = False
    for trial_path in completed_trials:
        trial_metadata = load_json(trial_path.with_suffix(".json"))
        if trial_metadata.get("session_type") == PURE_REST_SESSION_TYPE or trial_metadata.get("pure_rest_confirmed"):
            pure_rest_confirmed = True
        session_id = str(trial_metadata.get("session_id") or record.session_id or session_dir.name)
        trial_id = str(trial_metadata.get("trial_id") or trial_path.stem)
        annotation = load_json(_tag_annotation_path(session_id, trial_id))
        if annotation.get("completed") is True:
            annotated = True

    if pure_rest_confirmed and completed_trials:
        status = PURE_REST_STATUS_LABEL
    elif annotated:
        status = "已标注 Session"
    else:
        status = "未完成 Session"

    return {
        "Session": record.session_id,
        "状态": status,
        "完成Trial数": len(completed_trials),
        "未完成Trial数": len(partial_trials),
        "Session类型": record.metadata.get("session_type", ""),
        "更新时间": record.metadata.get("updated_time", ""),
    }


def _render_session_status_list() -> None:
    records = list_sessions()
    if not records:
        return
    rows = [_session_status_summary(record) for record in records]
    status_order = {PURE_REST_STATUS_LABEL: 0, "已标注 Session": 1, "未完成 Session": 2}
    rows.sort(key=lambda row: (status_order.get(str(row["状态"]), 9), str(row["Session"]).lower()))
    with st.expander("Session 列表状态", expanded=False):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _streamlit_session_id() -> str:
    if "streamlit_serial_session_id" not in st.session_state:
        st.session_state.streamlit_serial_session_id = uuid.uuid4().hex[:12]
    return str(st.session_state.streamlit_serial_session_id)


def _collector() -> SerialCollector:
    session_id = _streamlit_session_id()
    collector = st.session_state.get("serial_collector")
    collector_version = st.session_state.get("serial_collector_code_version", "")
    if collector is None or not hasattr(collector, "set_owner_id") or collector_version != COLLECTOR_CODE_VERSION:
        if collector is not None and hasattr(collector, "close"):
            try:
                collector.close()
            except Exception:
                pass
        collector = SerialCollector(owner_id=session_id)
        st.session_state.serial_collector = collector
        st.session_state.serial_collector_code_version = COLLECTOR_CODE_VERSION
    collector.set_owner_id(session_id)
    return collector


def _active_paths():
    state = st.session_state.get("active_trial_paths")
    return paths_from_state(state) if state else None


def _set_active_paths(paths) -> None:
    st.session_state.active_trial_paths = paths_to_state(paths) if paths else None


def _save_partial_snapshot(collector: SerialCollector) -> None:
    paths = _active_paths()
    if paths is None:
        return
    samples = collector.get_samples_snapshot()
    if not samples:
        return
    write_partial_csv(paths.partial_csv, samples)
    st.session_state.last_partial_save_monotonic = time.monotonic()


def _metadata_from_state(status: dict[str, Any]) -> dict[str, Any]:
    info = st.session_state.get("active_trial_info", {})
    requested_duration = info.get("requested_duration_s")
    actual_duration = status.get("current_time_s") or status.get("elapsed_s") or 0.0
    return {
        "start_time": status.get("start_time", ""),
        "end_time": status.get("end_time", ""),
        "requested_duration_s": requested_duration,
        "actual_duration_s": float(actual_duration),
        "sample_rate_hz": float(info.get("sample_rate_hz", status.get("sample_rate_hz", 500))),
        "serial_port": status.get("port", ""),
        "baudrate": int(status.get("baudrate", 0) or 0),
        "trial_type": info.get("trial_type", ""),
        "note": info.get("note", ""),
        "parse_error_count": int(status.get("parse_error_count", 0)),
        "receive_rate_hz": float(status.get("receive_rate_hz", 0.0)),
        "duplicate_seq_count": int(status.get("duplicate_seq_count", 0)),
        "seq_jump_count": int(status.get("seq_jump_count", 0)),
        "estimated_lost_samples": int(status.get("estimated_lost_samples", 0)),
        "source_mode": status.get("mode", ""),
        "signal_type": status.get("signal_type", "voltage"),
        "signal_unit": status.get("signal_unit", "uV"),
        "signal_column": (
            "voltage_uV"
            if status.get("signal_type", "voltage") == "voltage"
            else "current_uA"
            if status.get("signal_type") == "current"
            else "raw_value"
        ),
    }


def _finalize_active_trial(
    collector: SerialCollector,
    extra_metadata: dict[str, Any] | None = None,
    events_override: list[dict[str, Any]] | None = None,
) -> Path:
    status = collector.get_status()
    if status.get("collecting"):
        collector.stop_collection()
        status = collector.get_status()

    paths = _active_paths()
    if paths is None:
        raise RuntimeError("没有正在保存的 Trial。")
    samples = collector.get_samples_snapshot()
    events = events_override if events_override is not None else collector.get_events_snapshot()
    metadata = _metadata_from_state(status)
    if extra_metadata:
        metadata = {**metadata, **extra_metadata}
    final_path = finalize_trial(paths, samples, metadata, events)
    if extra_metadata and extra_metadata.get("session_type"):
        _update_session_metadata(
            paths.session_dir,
            {
                "session_type": extra_metadata.get("session_type"),
                "pure_rest_confirmed": bool(extra_metadata.get("pure_rest_confirmed", False)),
                "pure_rest_confirmed_at": now_iso(),
                "last_pure_rest_trial_id": paths.trial_id,
            },
        )
    st.session_state.last_saved_trial = str(final_path)
    st.session_state.selected_raw_file_path = str(final_path)
    _set_active_paths(None)
    st.session_state.active_trial_info = {}
    return final_path


def _session_inputs(sample_rate_hz: float) -> tuple[str, dict[str, Any]]:
    records = list_sessions()
    options = ["新建 Session", *[record.session_id for record in records]]
    choice = st.selectbox("Session", options, key="collection_session_choice")
    selected = next((record for record in records if record.session_id == choice), None)

    if selected is not None:
        st.caption(f"使用已有 Session：`{selected.session_id}`")
        with st.expander("查看 Session 元数据", expanded=False):
            st.json(selected.metadata or {"session_id": selected.session_id})
        metadata = {
            **selected.metadata,
            "sample_rate_hz": float(sample_rate_hz),
        }
        return selected.session_id, metadata

    col1, col2 = st.columns(2)
    with col1:
        session_name = st.text_input("Session 名称或编号", value="", placeholder="例如 session_plant01")
        plant_id = st.text_input("植物编号", value="", placeholder="例如 plant01")
        electrode_position = st.text_input("电极位置", value="")
    with col2:
        amplifier_gain = st.text_input("放大倍数", value="")
        ads_gain = st.text_input("ADS 增益", value="")
        session_note = st.text_area("实验备注", value="", height=80)

    session_id = make_session_id(session_name=session_name, plant_id=plant_id)
    st.caption(f"将创建：`{session_id}`")
    metadata = {
        "session_name": session_name,
        "plant_id": plant_id,
        "electrode_position": electrode_position,
        "amplifier_gain": amplifier_gain,
        "ads_gain": ads_gain,
        "sample_rate_hz": float(sample_rate_hz),
        "note": session_note,
    }
    return session_id, metadata


def _render_connection_controls(collector: SerialCollector, sample_rate_hz: float) -> None:
    st.subheader("串口连接")
    if "source_mode_v2" not in st.session_state:
        st.session_state.source_mode_v2 = "真实串口"
    source_mode = st.radio("数据源", ["真实串口", "模拟数据"], horizontal=True, key="source_mode_v2")
    status = collector.get_status()
    session_id = _streamlit_session_id()

    if source_mode == "真实串口":
        if "serial_ports" not in st.session_state:
            st.session_state.serial_ports = list_serial_ports()
        if st.button("刷新串口列表"):
            st.session_state.serial_ports = list_serial_ports()

        ports = st.session_state.serial_ports
        port_labels = [f"{item.device} - {item.description}" for item in ports]
        port_devices = [item.device for item in ports]
        if not port_devices:
            st.warning("未发现可用串口。如果只是测试页面，请切换到模拟数据。")
            selected_port = ""
        else:
            selected_label = st.selectbox("当前可用串口", port_labels)
            selected_port = port_devices[port_labels.index(selected_label)]

        baudrate = st.number_input("波特率", min_value=1, value=115200, step=9600)
        cols = st.columns(2)
        if cols[0].button("打开串口", disabled=not selected_port):
            try:
                current = collector.get_status()
                if current["collecting"]:
                    st.warning("正在采集，请先停止采集后再切换串口。")
                elif (
                    current["connected"]
                    and current.get("mode") == "serial"
                    and current.get("port") == selected_port
                    and int(current.get("baudrate", 0) or 0) == int(baudrate)
                ):
                    st.info("串口已经打开。")
                else:
                    if current["connected"]:
                        collector.close()
                    collector.open_serial(selected_port, int(baudrate))
                    st.success(f"串口已连接：{selected_port} @ {int(baudrate)}")
                    st.rerun()
            except (PermissionError, SerialPortInUseError, SerialPortOpenError) as exc:
                message = str(exc) or SERIAL_BUSY_MESSAGE
                if isinstance(exc, PermissionError):
                    message = SERIAL_BUSY_MESSAGE
                st.error(f"打开串口失败：{message}")
            except Exception as exc:  # noqa: BLE001
                text = str(exc)
                if "PermissionError" in text or "拒绝访问" in text or "Access is denied" in text:
                    st.error(f"打开串口失败：{SERIAL_BUSY_MESSAGE}")
                else:
                    st.error(f"打开串口失败：{exc}")
        if cols[1].button("关闭串口", disabled=not status["connected"]):
            collector.close()
            st.rerun()
    else:
        st.warning("当前使用的是模拟数据，不会读取电脑串口。采集真实设备数据时请选择“真实串口”。")
        simulate_errors = st.checkbox("模拟解析错误行", value=False)
        cols = st.columns(2)
        if cols[0].button("启动模拟源", disabled=status["connected"]):
            collector.open_simulated(sample_rate_hz=float(sample_rate_hz), simulate_errors=simulate_errors)
            st.success("模拟源已启动")
            st.rerun()
        if cols[1].button("关闭模拟源", disabled=not status["connected"]):
            collector.close()
            st.rerun()

    status = collector.get_status()
    if status["connected"]:
        st.success(f"串口已连接：{status['port']}")
    else:
        st.info("串口断开")
    if status["last_error"]:
        st.error(status["last_error"])
    debug_info = collector.get_debug_info(session_id)
    debug_cols = st.columns(3)
    debug_cols[0].caption(f"会话：`{debug_info['当前Streamlit会话标识']}`")
    debug_cols[1].caption(f"串口对象：{'存在' if debug_info['当前串口对象是否存在'] else '无'} / is_open={debug_info['is_open状态']}")
    debug_cols[2].caption(
        f"线程：{debug_info['当前读取线程名称'] or '无'} / 存活={debug_info['读取线程是否存活']}"
    )
    with st.expander("串口调试信息", expanded=False):
        st.json(debug_info)
    raw_lines = collector.get_recent_raw_lines()
    diag_cols = st.columns([1, 1, 2])
    diag_cols[0].metric("收到原始行", int(status.get("raw_line_count", 0)))
    diag_cols[1].metric("最近原始行", str(status.get("last_raw_line", ""))[:24] or "无")
    if diag_cols[2].button("刷新串口诊断"):
        st.rerun()
    with st.expander("最近原始串口行", expanded=bool(raw_lines and not status.get("sample_count"))):
        if raw_lines:
            st.dataframe(pd.DataFrame(raw_lines), use_container_width=True, hide_index=True)
        else:
            st.info("还没有收到任何串口行。若 MCU 正在发送，请检查 COM 口、波特率、接线和是否被其他程序占用。")


def _render_status(status: dict[str, Any], active_trial_id: str) -> None:
    st.subheader("实时状态")
    cols = st.columns(4)
    cols[0].metric("已采集时间", f"{float(status['current_time_s']):.3f} s")
    cols[1].metric("有效样本数", int(status["sample_count"]))
    cols[2].metric("估算接收速率", f"{float(status['receive_rate_hz']):.1f} Hz")
    cols[3].metric("设置采样率", f"{float(status['sample_rate_hz']):.1f} Hz")

    cols = st.columns(4)
    cols[0].metric("解析错误数", int(status["parse_error_count"]))
    cols[0].caption(f"已过滤异常值：{int(status.get('filtered_count', 0))}")
    signal_type = str(status.get("signal_type", "voltage"))
    signal_unit = str(status.get("signal_unit", "uV"))
    signal_name = "电压" if signal_type == "voltage" else "电流" if signal_type == "current" else "原始ADC"
    unit_text = "微伏" if signal_unit == "uV" else "微安" if signal_unit == "uA" else signal_unit
    signal_value = status.get("signal_value")
    signal_text = "无" if signal_value is None else f"{float(signal_value):.4f}"
    signal_label = f"当前{signal_name}" + (f"（{unit_text}）" if unit_text else "")
    cols[1].metric(signal_label, signal_text)
    cols[2].metric("正在采集", "是" if status["collecting"] else "否")
    cols[3].metric("当前 Trial", active_trial_id or "无")

    if status["seq_jump_count"] or status["duplicate_seq_count"]:
        st.warning(
            "检测到 MCU 序号异常："
            f"跳变 {status['seq_jump_count']} 次，重复 {status['duplicate_seq_count']} 次，"
            f"估算丢失 {status['estimated_lost_samples']} 点。"
        )
    if status["parse_error_count"]:
        st.warning("存在解析错误，可在下方展开查看最近错误行。")


def _render_waveform(samples: list[dict[str, Any]], window_s: float) -> None:
    st.subheader("实时波形")
    if not samples:
        st.info("开始采集后这里会显示实时波形。")
        return

    dataframe = pd.DataFrame(samples)
    if "voltage_uV" in dataframe.columns and dataframe["voltage_uV"].notna().any():
        value_column = "voltage_uV"
        trace_name = "电压"
        yaxis_title = "电压（微伏）"
        hover_unit = "微伏"
    elif "current_uA" in dataframe.columns and dataframe["current_uA"].notna().any():
        value_column = "current_uA"
        trace_name = "电流"
        yaxis_title = "电流（微安）"
        hover_unit = "微安"
    elif "raw_value" in dataframe.columns and dataframe["raw_value"].notna().any():
        value_column = "raw_value"
        trace_name = "原始ADC"
        yaxis_title = "原始ADC值"
        hover_unit = ""
    else:
        st.info("已收到数据，但还没有可显示的电压、电流或原始ADC数值。")
        return
    max_time = float(dataframe["time_s"].max())
    tail = dataframe[dataframe["time_s"] >= max(0.0, max_time - float(window_s))]

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=tail["time_s"],
            y=tail[value_column],
            mode="lines",
            name=trace_name,
            line={"color": "#2563eb", "width": 1},
            hovertemplate=(
                f"时间=%{{x:.4f}} 秒<br>{trace_name}=%{{y:.4f}}"
                + (f" {hover_unit}" if hover_unit else "")
                + "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        height=360,
        margin={"l": 40, "r": 24, "t": 20, "b": 40},
        xaxis_title="时间（秒）",
        yaxis_title=yaxis_title,
        hovermode="x unified",
    )
    st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False, "responsive": True})


def _render_event_controls(collector: SerialCollector, collecting: bool) -> None:
    st.subheader("动作标记")
    custom_note = st.text_input("自定义备注", value="", key="event_note")
    cols = st.columns(len(EVENT_TYPES))
    for column, (label, event_type) in zip(cols, EVENT_TYPES.items()):
        if column.button(label, disabled=not collecting):
            note = custom_note if event_type == "note" else ""
            collector.add_event(event_type, note=note)
            st.rerun()

    events = collector.get_events_snapshot()
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)


def _render_partial_trials() -> None:
    partials = list_partial_trials()
    if not partials:
        return

    with st.expander("未完成 Trial", expanded=False):
        labels = [str(path.relative_to(Path.cwd())) if path.is_absolute() else str(path) for path in partials]
        selected = st.selectbox("未完成文件", labels)
        path = partials[labels.index(selected)]
        cols = st.columns(2)
        if cols[0].button("恢复为可用 Trial"):
            try:
                final_path = recover_partial_trial(path)
                st.success(f"已恢复：{final_path}")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"恢复失败：{exc}")
        confirm_delete = st.checkbox("确认删除所选未完成 Trial")
        if cols[1].button("删除未完成 Trial", disabled=not confirm_delete):
            delete_partial_trial(path)
            st.warning("已删除未完成 Trial。")
            st.rerun()


def render_collection_page() -> None:
    st.title("数据采集")
    st.caption("从串口或模拟源接收 ADS 电压数据，保存为可直接标注的 Trial。")

    collector = _collector()

    st.sidebar.subheader("刷新")
    auto_refresh = st.sidebar.checkbox("采集时自动刷新", value=True)
    refresh_interval_s = st.sidebar.slider("刷新间隔秒", 0.5, 3.0, 1.0, 0.5)

    st.subheader("采样设置")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    sample_rate_hz = sample_col1.number_input("采样率 Hz", min_value=1.0, value=500.0, step=50.0)
    collection_mode = sample_col2.radio("采集模式", ["自动停止", "手动停止"], horizontal=True)
    requested_duration_s = sample_col3.number_input("自动停止时长 秒", min_value=0.1, value=10.0, step=1.0)
    auto_finalize = st.checkbox("自动停止后保存为完成 Trial", value=True, disabled=collection_mode != "自动停止")

    _render_connection_controls(collector, sample_rate_hz=float(sample_rate_hz))

    st.subheader("Session 信息")
    session_id, session_metadata = _session_inputs(sample_rate_hz=float(sample_rate_hz))
    _render_session_status_list()

    st.subheader("Trial 信息")
    trial_col1, trial_col2 = st.columns([1, 2])
    trial_type = trial_col1.selectbox("Trial 类型", TRIAL_TYPES)
    trial_note = trial_col2.text_area("本次备注", value="", height=80)

    status = collector.get_status()
    active_paths = _active_paths()
    active_trial_id = active_paths.trial_id if active_paths else ""

    if (
        status["collecting"]
        and st.session_state.get("active_trial_info", {}).get("collection_mode") == "自动停止"
        and float(status["current_time_s"]) >= float(st.session_state.get("active_trial_info", {}).get("requested_duration_s", 0))
    ):
        collector.stop_collection()
        _save_partial_snapshot(collector)
        if st.session_state.get("active_trial_info", {}).get("auto_finalize"):
            try:
                final_path = _finalize_active_trial(collector)
                st.success(f"采集完成并已保存：{final_path}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"保存失败：{exc}")
        st.rerun()

    st.subheader("采集控制")
    control_cols = st.columns(5)
    can_start = bool(status["connected"]) and not bool(status["collecting"]) and active_paths is None
    if control_cols[0].button("开始采集", type="primary", disabled=not can_start):
        try:
            paths = reserve_trial(session_id, session_metadata)
            collector.start_collection(float(sample_rate_hz))
            _set_active_paths(paths)
            st.session_state.active_trial_info = {
                "sample_rate_hz": float(sample_rate_hz),
                "collection_mode": collection_mode,
                "requested_duration_s": float(requested_duration_s) if collection_mode == "自动停止" else None,
                "auto_finalize": bool(auto_finalize and collection_mode == "自动停止"),
                "trial_type": trial_type,
                "note": trial_note,
            }
            st.session_state.last_partial_save_monotonic = 0.0
            st.success(f"正在采集：{paths.session_id}/{paths.trial_id}")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"开始采集失败：{exc}")

    if control_cols[1].button("停止采集", disabled=not status["collecting"]):
        collector.stop_collection()
        _save_partial_snapshot(collector)
        st.success("采集已停止，可保存或放弃本次 Trial。")
        st.rerun()

    can_save = active_paths is not None and not collector.get_status()["collecting"] and status["sample_count"] > 0
    if control_cols[2].button("保存并进入标注", disabled=not can_save):
        try:
            final_path = _finalize_active_trial(collector)
            st.success(f"已保存：{final_path}")
            st.session_state.pending_app_page = "数据标注"
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"保存失败：{exc}")

    if control_cols[3].button("保存为纯静息 Session", disabled=not can_save):
        try:
            final_path = _finalize_active_trial(
                collector,
                extra_metadata={
                    "session_type": PURE_REST_SESSION_TYPE,
                    "pure_rest_confirmed": True,
                    "trial_type": PURE_REST_STATUS_LABEL,
                    "rest_all": True,
                    "default_label": "静息",
                    "unlabeled_policy": "整段原始数据由用户确认作为静息，不自动生成刺激标签。",
                },
                events_override=[],
            )
            annotation_path = _write_completed_rest_annotation(final_path)
            sync_labeled_trial_export(final_path, annotation_path, DEFAULT_OUTPUT_DIR)
            st.success(f"已保存为纯静息 Session：{final_path}")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"保存纯静息失败：{exc}")

    confirm_discard = st.checkbox("确认放弃当前 Trial")
    if control_cols[4].button("放弃本次采集", disabled=active_paths is None or not confirm_discard):
        if collector.get_status()["collecting"]:
            collector.stop_collection()
        paths = _active_paths()
        if paths and paths.partial_csv.exists():
            paths.partial_csv.unlink()
        collector.discard_collection()
        _set_active_paths(None)
        st.session_state.active_trial_info = {}
        st.warning("已放弃当前 Trial。")
        st.rerun()

    status = collector.get_status()
    active_paths = _active_paths()
    active_trial_id = active_paths.trial_id if active_paths else ""
    _render_status(status, active_trial_id)

    samples = collector.get_samples_snapshot()
    if active_paths is not None and samples:
        last_save = float(st.session_state.get("last_partial_save_monotonic", 0.0))
        if time.monotonic() - last_save >= 1.0:
            try:
                _save_partial_snapshot(collector)
            except Exception as exc:  # noqa: BLE001
                st.error(f"保存临时文件失败：{exc}")

    invalid_lines = collector.get_invalid_lines()
    with st.expander("最近解析错误行", expanded=bool(invalid_lines)):
        if invalid_lines:
            st.dataframe(pd.DataFrame(invalid_lines), use_container_width=True, hide_index=True)
        else:
            st.write("暂无解析错误。")

    live_window_s = st.slider("实时波形显示最近秒数", 1.0, 20.0, 5.0, 1.0)
    _render_waveform(samples, window_s=float(live_window_s))
    _render_event_controls(collector, collecting=bool(status["collecting"]))
    _render_partial_trials()

    last_saved = st.session_state.get("last_saved_trial")
    if last_saved:
        st.success(f"最近保存 Trial：`{last_saved}`")

    if status["collecting"] and auto_refresh:
        time.sleep(float(refresh_interval_s))
        st.rerun()
