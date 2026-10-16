"""Streamlit realtime serial monitor for Impact event detection.

Run:
    python run.py monitor
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from plant_signal.paths import ROOT_DIR
from typing import Any

import numpy as np

from plant_signal.monitoring.protocol import parse_signal_line, parse_classification_line

from plant_signal.monitoring.inference import (
    infer_model_time_len,
    infer_default_window_len,
    median_filter_3,
    resize_1d,
    preprocess_window,
    model_input_array,
    dequantize_output,
    sigmoid,
    impact_score_from_output,
)
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


APP_DIR = ROOT_DIR
MODEL_SEARCH_DIRS = [
    APP_DIR / "outputs" / "impact_binary_best_event",
    APP_DIR / "outputs" / "impact_event_detector",
]
DEFAULT_PORT = ""
DEFAULT_BAUDRATE = 115200
DEFAULT_SAMPLE_RATE_HZ = 220.0
DEFAULT_DISPLAY_SECONDS = 10.0
DEFAULT_WINDOW_LEN = 128
DEFAULT_STRIDE = 16
DEFAULT_IMPACT_THRESHOLD = 0.6
DEFAULT_COOLDOWN_SECONDS = 0.5
SPIKE_ABS_THRESHOLD = 4_900_000.0
MAX_BUFFER_POINTS = 60_000
MAX_RECENT_RAW_LINES = 30
MAX_SCORE_POINTS = 2_000
MAX_IMPACT_EVENTS = 30
MAX_CLASSIFICATION_LOGS = 50
MAX_SERIAL_DEBUG_LOGS = 50


def init_state() -> None:
    defaults: dict[str, Any] = {
        "serial_obj": None,
        "serial_connected": False,
        "monitoring": False,
        "serial_error": "",
        "serial_port": DEFAULT_PORT,
        "baudrate": DEFAULT_BAUDRATE,
        "sample_indices": [],
        "sample_times": [],
        "sample_values": [],
        "recent_raw_lines": [],
        "raw_line_count": 0,
        "parse_error_count": 0,
        "spike_drop_count": 0,
        "accepted_count": 0,
        "last_value": None,
        "last_infer_accepted_count": 0,
        "score_history": [],
        "last_score": None,
        "last_impact_monotonic": 0.0,
        "last_impact_time": "",
        "impact_events": [],
        "model_path_loaded": "",
        "model_load_error": "",
        "interpreter": None,
        "input_details": None,
        "output_details": None,
        "preprocess_config": {},
        "decision_config": {},
        "model_window_default": DEFAULT_WINDOW_LEN,
        "mcu_class_id": None,
        "mcu_class_label": "",
        "mcu_confidence": None,
        "mcu_last_recognition_time": "",
        "mcu_impact_count": 0,
        "mcu_classification_logs": [],
        "serial_debug_logs": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def list_serial_ports() -> list[str]:
    try:
        from serial.tools import list_ports  # type: ignore[import-not-found]
    except Exception:
        return []
    return [str(port.device) for port in list_ports.comports()]


def close_serial() -> None:
    serial_obj = st.session_state.get("serial_obj")
    if serial_obj is not None:
        try:
            serial_obj.close()
        except Exception:
            pass
    st.session_state.serial_obj = None
    st.session_state.serial_connected = False
    st.session_state.monitoring = False


def connect_serial(port: str, baudrate: int) -> None:
    close_serial()
    try:
        import serial  # type: ignore[import-not-found]

        serial_obj = serial.Serial(
            port=str(port).strip(),
            baudrate=int(baudrate),
            timeout=0.02,
            write_timeout=0.2,
        )
        try:
            serial_obj.reset_input_buffer()
        except Exception:
            pass
        st.session_state.serial_obj = serial_obj
        st.session_state.serial_connected = True
        st.session_state.serial_error = ""
        st.session_state.serial_port = str(port).strip()
        st.session_state.baudrate = int(baudrate)
    except Exception as exc:
        st.session_state.serial_obj = None
        st.session_state.serial_connected = False
        st.session_state.monitoring = False
        st.session_state.serial_error = f"连接串口失败: {exc}"


def append_classification_result(result: dict[str, Any]) -> None:
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    previous_label = str(st.session_state.mcu_class_label or "").upper()
    label = str(result["label"]).upper()
    if label == "IMPACT" and previous_label != "IMPACT":
        st.session_state.mcu_impact_count += 1

    st.session_state.mcu_class_id = int(result["class_id"])
    st.session_state.mcu_class_label = label
    st.session_state.mcu_confidence = int(result["confidence"])
    st.session_state.mcu_last_recognition_time = now_text
    st.session_state.mcu_classification_logs.append(
        {
            "识别时间": now_text,
            "类别编号": int(result["class_id"]),
            "识别结果": "静息" if label == "REST" else "检测到弹击" if label == "IMPACT" else label,
            "置信度": int(result["confidence"]),
        }
    )


def trim_buffers() -> None:
    point_count = len(st.session_state.sample_values)
    if point_count > MAX_BUFFER_POINTS:
        drop = point_count - MAX_BUFFER_POINTS
        st.session_state.sample_indices = st.session_state.sample_indices[drop:]
        st.session_state.sample_times = st.session_state.sample_times[drop:]
        st.session_state.sample_values = st.session_state.sample_values[drop:]
    if len(st.session_state.recent_raw_lines) > MAX_RECENT_RAW_LINES:
        st.session_state.recent_raw_lines = st.session_state.recent_raw_lines[-MAX_RECENT_RAW_LINES:]
    if len(st.session_state.score_history) > MAX_SCORE_POINTS:
        st.session_state.score_history = st.session_state.score_history[-MAX_SCORE_POINTS:]
    if len(st.session_state.impact_events) > MAX_IMPACT_EVENTS:
        st.session_state.impact_events = st.session_state.impact_events[-MAX_IMPACT_EVENTS:]
    if len(st.session_state.mcu_classification_logs) > MAX_CLASSIFICATION_LOGS:
        st.session_state.mcu_classification_logs = st.session_state.mcu_classification_logs[-MAX_CLASSIFICATION_LOGS:]
    if len(st.session_state.serial_debug_logs) > MAX_SERIAL_DEBUG_LOGS:
        st.session_state.serial_debug_logs = st.session_state.serial_debug_logs[-MAX_SERIAL_DEBUG_LOGS:]


def append_serial_line(raw_line: str, now_mono: float, *, allow_v_prefix: bool = False) -> None:
    text = raw_line.replace("\x00", "").strip()
    if not text:
        return

    st.session_state.raw_line_count += 1
    st.session_state.recent_raw_lines.append(
        {
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "raw_line": text,
        }
    )

    classification = parse_classification_line(text)
    if classification is not None:
        append_classification_result(classification)
        trim_buffers()
        return

    value = parse_signal_line(text, allow_v_prefix=allow_v_prefix)
    if value is None:
        st.session_state.parse_error_count += 1
        st.session_state.serial_debug_logs.append(
            {
                "时间": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "内容": text,
                "原因": "无法解析为波形数据或分类结果",
            }
        )
        trim_buffers()
        return
    if abs(value) > SPIKE_ABS_THRESHOLD:
        st.session_state.spike_drop_count += 1
        trim_buffers()
        return

    sample_index = int(st.session_state.accepted_count)
    st.session_state.sample_indices.append(sample_index)
    st.session_state.sample_times.append(float(now_mono))
    st.session_state.sample_values.append(float(value))
    st.session_state.accepted_count += 1
    st.session_state.last_value = float(value)
    trim_buffers()


def read_serial_lines(max_lines: int = 300, *, allow_v_prefix: bool = False) -> int:
    serial_obj = st.session_state.get("serial_obj")
    if serial_obj is None or not st.session_state.get("serial_connected", False):
        return 0

    read_count = 0
    try:
        if hasattr(serial_obj, "is_open") and not serial_obj.is_open:
            raise RuntimeError("串口已断开")

        while read_count < max_lines:
            waiting = int(getattr(serial_obj, "in_waiting", 0) or 0)
            if waiting <= 0:
                break
            raw = serial_obj.readline()
            if not raw:
                break
            if isinstance(raw, bytes):
                line = raw.decode("utf-8", errors="replace")
            else:
                line = str(raw)
            append_serial_line(line, time.monotonic(), allow_v_prefix=allow_v_prefix)
            read_count += 1
    except Exception as exc:
        st.session_state.serial_error = f"读取串口失败: {exc}"
        close_serial()
    return read_count


def clear_runtime_buffers() -> None:
    st.session_state.sample_indices = []
    st.session_state.sample_times = []
    st.session_state.sample_values = []
    st.session_state.recent_raw_lines = []
    st.session_state.raw_line_count = 0
    st.session_state.parse_error_count = 0
    st.session_state.spike_drop_count = 0
    st.session_state.accepted_count = 0
    st.session_state.last_value = None
    st.session_state.last_infer_accepted_count = 0
    st.session_state.score_history = []
    st.session_state.last_score = None
    st.session_state.last_impact_monotonic = 0.0
    st.session_state.last_impact_time = ""
    st.session_state.impact_events = []
    st.session_state.mcu_class_id = None
    st.session_state.mcu_class_label = ""
    st.session_state.mcu_confidence = None
    st.session_state.mcu_last_recognition_time = ""
    st.session_state.mcu_impact_count = 0
    st.session_state.mcu_classification_logs = []
    st.session_state.serial_debug_logs = []


def find_tflite_models() -> list[Path]:
    found: list[Path] = []
    for directory in MODEL_SEARCH_DIRS:
        if directory.exists():
            found.extend(sorted(directory.glob("*.tflite")))
    return found


def read_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def load_tflite_model(model_path: Path) -> None:
    if not model_path.exists():
        st.session_state.interpreter = None
        st.session_state.input_details = None
        st.session_state.output_details = None
        st.session_state.model_load_error = f"未找到模型: {model_path}"
        st.session_state.model_path_loaded = ""
        return

    try:
        try:
            import tensorflow as tf  # type: ignore[import-not-found]

            interpreter = tf.lite.Interpreter(model_path=str(model_path))
        except Exception:
            from tflite_runtime.interpreter import Interpreter  # type: ignore[import-not-found]

            interpreter = Interpreter(model_path=str(model_path))

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        preprocess_config = read_json_if_exists(model_path.parent / "preprocess_config.json")
        decision_config = read_json_if_exists(model_path.parent / "decision_config.json")

        st.session_state.interpreter = interpreter
        st.session_state.input_details = input_details
        st.session_state.output_details = output_details
        st.session_state.preprocess_config = preprocess_config
        st.session_state.decision_config = decision_config
        st.session_state.model_path_loaded = str(model_path)
        st.session_state.model_load_error = ""
        st.session_state.model_window_default = infer_default_window_len(input_details, preprocess_config)
    except Exception as exc:
        st.session_state.interpreter = None
        st.session_state.input_details = None
        st.session_state.output_details = None
        st.session_state.preprocess_config = {}
        st.session_state.decision_config = {}
        st.session_state.model_path_loaded = ""
        st.session_state.model_load_error = f"加载模型失败: {exc}"


def run_single_inference(window_len: int, threshold: float, cooldown_s: float) -> None:
    interpreter = st.session_state.get("interpreter")
    input_details = st.session_state.get("input_details")
    output_details = st.session_state.get("output_details")
    if interpreter is None or input_details is None or output_details is None:
        return
    if len(st.session_state.sample_values) < window_len:
        return

    accepted_count = int(st.session_state.accepted_count)
    if accepted_count <= st.session_state.last_infer_accepted_count:
        return

    raw_window = np.asarray(st.session_state.sample_values[-window_len:], dtype=np.float32)
    target_len = infer_model_time_len(input_details)
    processed = preprocess_window(raw_window, target_len, st.session_state.get("preprocess_config", {}))
    model_input = model_input_array(processed, input_details)

    try:
        interpreter.set_tensor(input_details["index"], model_input)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details["index"])
        output = dequantize_output(raw_output, output_details)
        score = impact_score_from_output(output)
    except Exception as exc:
        st.session_state.model_load_error = f"推理失败: {exc}"
        return

    now_mono = time.monotonic()
    now_label = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    start_index = int(st.session_state.sample_indices[-window_len])
    end_index = int(st.session_state.sample_indices[-1])
    st.session_state.last_score = float(score)
    st.session_state.last_infer_accepted_count = accepted_count
    st.session_state.score_history.append(
        {
            "time_mono": now_mono,
            "time": now_label,
            "score": float(score),
            "threshold": float(threshold),
        }
    )

    if score >= threshold and (now_mono - float(st.session_state.last_impact_monotonic)) >= cooldown_s:
        st.session_state.last_impact_monotonic = now_mono
        st.session_state.last_impact_time = now_label
        st.session_state.impact_events.append(
            {
                "time": now_label,
                "score": round(float(score), 4),
                "window_start": start_index,
                "window_end": end_index,
            }
        )
    trim_buffers()


def maybe_run_inference(window_len: int, stride: int, threshold: float, cooldown_s: float) -> None:
    if len(st.session_state.sample_values) < window_len:
        return
    delta = int(st.session_state.accepted_count) - int(st.session_state.last_infer_accepted_count)
    if delta >= max(1, int(stride)):
        run_single_inference(window_len, threshold, cooldown_s)


def estimate_receive_rate(display_seconds: float = 5.0) -> float:
    times = st.session_state.sample_times
    if len(times) < 2:
        return 0.0
    now_mono = time.monotonic()
    recent = [float(t) for t in times if now_mono - float(t) <= display_seconds]
    if len(recent) < 2:
        return 0.0
    duration = recent[-1] - recent[0]
    if duration <= 0:
        return 0.0
    return (len(recent) - 1) / duration


def model_status_text() -> str:
    if st.session_state.get("interpreter") is None:
        if st.session_state.get("model_load_error"):
            return st.session_state.model_load_error
        return "未加载模型"
    return f"已加载: {Path(st.session_state.model_path_loaded).name}"


def serial_status_text() -> str:
    if st.session_state.get("serial_connected", False):
        status = f"已连接 {st.session_state.serial_port} @ {st.session_state.baudrate}"
        if st.session_state.get("monitoring", False):
            return status + "，监测中"
        return status + "，已暂停"
    return "未连接"


def render_model_details() -> None:
    input_details = st.session_state.get("input_details")
    output_details = st.session_state.get("output_details")
    if input_details is None or output_details is None:
        return
    rows = [
        {
            "item": "input shape",
            "value": str(list(input_details.get("shape", []))),
        },
        {
            "item": "output shape",
            "value": str(list(output_details.get("shape", []))),
        },
        {
            "item": "input dtype",
            "value": str(input_details.get("dtype")),
        },
        {
            "item": "output dtype",
            "value": str(output_details.get("dtype")),
        },
        {
            "item": "input quantization",
            "value": str(input_details.get("quantization")),
        },
        {
            "item": "output quantization",
            "value": str(output_details.get("quantization")),
        },
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_waveform(display_seconds: float) -> None:
    now_mono = time.monotonic()
    times = np.asarray(st.session_state.sample_times, dtype=np.float64)
    values = np.asarray(st.session_state.sample_values, dtype=np.float32)
    if times.size == 0:
        st.info("暂无有效串口数据")
        return
    mask = times >= now_mono - display_seconds
    times = times[mask]
    values = values[mask]
    if times.size == 0:
        st.info("最近显示窗口内暂无数据")
        return

    x = times - times[-1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=values,
            mode="lines",
            line=dict(color="#2563eb", width=1.5),
            name="Signal",
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="相对当前时间 (s)",
        yaxis_title="Signal value",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_score_curve(display_seconds: float, threshold: float) -> None:
    rows = st.session_state.score_history
    if not rows:
        st.info("暂无推理结果")
        return
    now_mono = time.monotonic()
    df = pd.DataFrame(rows)
    df = df[df["time_mono"] >= now_mono - max(display_seconds, 10.0)].copy()
    if df.empty:
        st.info("最近窗口内暂无推理结果")
        return
    df["relative_s"] = df["time_mono"] - float(df["time_mono"].iloc[-1])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["relative_s"],
            y=df["score"],
            mode="lines+markers",
            line=dict(color="#dc2626", width=2),
            marker=dict(size=5),
            name="Impact score",
        )
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="#111827", annotation_text="threshold")
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="相对当前时间 (s)",
        yaxis_title="Impact score",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def rerun_later(refresh_ms: int) -> None:
    time.sleep(max(0.05, float(refresh_ms) / 1000.0))
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def render_mcu_classification_panel() -> None:
    label = str(st.session_state.mcu_class_label or "").upper()
    confidence = st.session_state.mcu_confidence
    if label == "IMPACT":
        st.error("当前识别状态：检测到弹击")
    elif label == "REST":
        st.success("当前识别状态：静息")
    elif label:
        st.info(f"当前识别状态：{label}")
    else:
        st.info("当前识别状态：等待 MCU 分类结果")

    cols = st.columns(4)
    cols[0].metric("MCU 类别", "暂无" if st.session_state.mcu_class_id is None else str(st.session_state.mcu_class_id))
    cols[1].metric("MCU 置信度", "暂无" if confidence is None else str(confidence))
    cols[2].metric("最近识别时间", st.session_state.mcu_last_recognition_time or "暂无")
    cols[3].metric("弹击事件计数", int(st.session_state.mcu_impact_count))

    log_cols = st.columns(2)
    with log_cols[0]:
        st.subheader("最近识别日志")
        if st.session_state.mcu_classification_logs:
            logs = pd.DataFrame(st.session_state.mcu_classification_logs)
            st.dataframe(logs.iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("暂未收到分类结果。")
    with log_cols[1]:
        st.subheader("串口解析调试日志")
        if st.session_state.serial_debug_logs:
            logs = pd.DataFrame(st.session_state.serial_debug_logs)
            st.dataframe(logs.iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("暂无解析失败内容。")


def main(*, enable_v_format: bool = False) -> None:
    st.set_page_config(page_title="Impact 神经网络实时检测", layout="wide")
    init_state()

    st.title("Impact 神经网络实时检测")

    model_candidates = find_tflite_models()
    candidate_labels = [str(path.relative_to(APP_DIR)) for path in model_candidates]

    st.subheader("1. 神经网络模型")
    model_cols = st.columns([3, 2, 1])
    selected_label = ""
    if candidate_labels:
        selected_label = model_cols[0].selectbox(
            "选择已训练好的 .tflite 神经网络",
            candidate_labels,
            index=0,
            key="main_model_select",
        )
        st.session_state.model_load_error = "" if st.session_state.get("model_load_error") == "未找到模型" else st.session_state.model_load_error
    else:
        model_cols[0].warning("未找到模型")
        st.session_state.model_load_error = "未找到模型"

    manual_model_path = model_cols[1].text_input(
        "手动输入 .tflite 模型路径",
        value="",
        key="main_manual_model_path",
    )
    chosen_model_path = Path(manual_model_path).expanduser() if manual_model_path.strip() else None
    if chosen_model_path is None and selected_label:
        chosen_model_path = APP_DIR / selected_label
    if chosen_model_path is not None:
        chosen_model_path = chosen_model_path if chosen_model_path.is_absolute() else APP_DIR / chosen_model_path
        if model_cols[2].button("加载模型", type="primary", use_container_width=True):
            load_tflite_model(chosen_model_path)
    else:
        model_cols[2].button("加载模型", disabled=True, use_container_width=True)

    with st.sidebar:
        st.header("串口设置")
        st.caption("支持解析: A0=123.45μA、CH0 Voltage=123.45 uV" + ("、V     12345 uV" if enable_v_format else ""))
        ports = list_serial_ports()
        if ports:
            default_index = ports.index(st.session_state.serial_port) if st.session_state.serial_port in ports else 0
            selected_port = st.selectbox("可用串口", ports, index=default_index)
        else:
            selected_port = st.session_state.serial_port
            st.caption("未枚举到串口，可手动输入。")
        port = st.text_input("串口号", value=str(selected_port or DEFAULT_PORT))
        baudrate = st.number_input("波特率", min_value=1, max_value=2_000_000, value=int(st.session_state.baudrate), step=9600)

        c1, c2 = st.columns(2)
        if c1.button("连接串口", use_container_width=True):
            connect_serial(port, int(baudrate))
        if c2.button("断开串口", use_container_width=True):
            close_serial()

        c3, c4 = st.columns(2)
        if c3.button("开始监测", use_container_width=True):
            if st.session_state.serial_connected:
                st.session_state.monitoring = True
                st.session_state.serial_error = ""
            else:
                st.session_state.serial_error = "请先连接串口。"
        if c4.button("暂停监测", use_container_width=True):
            st.session_state.monitoring = False

        if st.button("清空缓冲", use_container_width=True):
            clear_runtime_buffers()

        st.header("显示与推理")
        display_seconds = st.number_input("显示最近秒数", min_value=1.0, max_value=120.0, value=DEFAULT_DISPLAY_SECONDS, step=1.0)
        sample_rate_hz = st.number_input("采样率估计 (Hz)", min_value=1.0, max_value=5000.0, value=DEFAULT_SAMPLE_RATE_HZ, step=10.0)
        refresh_ms = st.slider("页面刷新间隔 (ms)", min_value=80, max_value=2000, value=200, step=20)

        st.header("推理参数")
        if st.session_state.get("interpreter") is None:
            st.caption("先在主页面顶部选择并加载神经网络模型。")
        else:
            st.caption(f"当前模型: {Path(st.session_state.model_path_loaded).name}")
        model_default_window = int(st.session_state.get("model_window_default") or DEFAULT_WINDOW_LEN)
        window_len = st.number_input("滑动窗口长度 (点)", min_value=8, max_value=8192, value=model_default_window, step=8)
        stride = st.number_input("滑窗步长 (点)", min_value=1, max_value=4096, value=DEFAULT_STRIDE, step=1)
        threshold = st.slider("Impact 阈值", min_value=0.0, max_value=1.0, value=DEFAULT_IMPACT_THRESHOLD, step=0.01)
        cooldown_s = st.number_input("报警冷却时间 (s)", min_value=0.0, max_value=30.0, value=DEFAULT_COOLDOWN_SECONDS, step=0.1)

    if st.session_state.monitoring:
        read_serial_lines(allow_v_prefix=enable_v_format)
        maybe_run_inference(int(window_len), int(stride), float(threshold), float(cooldown_s))

    st.subheader("MCU 分类结果")
    render_mcu_classification_panel()

    recent_impact = (time.monotonic() - float(st.session_state.last_impact_monotonic)) <= max(2.0, float(cooldown_s))
    if recent_impact:
        st.error(f"Impact detected  最近一次: {st.session_state.last_impact_time}")
    else:
        st.success("No Impact detected")

    status_cols = st.columns(4)
    status_cols[0].metric("串口状态", serial_status_text())
    status_cols[1].metric("模型状态", model_status_text())
    status_cols[2].metric("最近 Impact score", "N/A" if st.session_state.last_score is None else f"{st.session_state.last_score:.3f}")
    status_cols[3].metric("接收频率估计", f"{estimate_receive_rate():.1f} Hz")

    if st.session_state.serial_error:
        st.warning(st.session_state.serial_error)
    if st.session_state.model_load_error and st.session_state.get("interpreter") is None:
        st.warning(st.session_state.model_load_error)

    metric_cols = st.columns(5)
    metric_cols[0].metric("有效点数", int(st.session_state.accepted_count))
    metric_cols[1].metric("缓冲点数", len(st.session_state.sample_values))
    metric_cols[2].metric("最近值", "N/A" if st.session_state.last_value is None else f"{st.session_state.last_value:.3f}")
    metric_cols[3].metric("解析失败行", int(st.session_state.parse_error_count))
    metric_cols[4].metric("异常尖峰丢弃", int(st.session_state.spike_drop_count))

    with st.expander("神经网络输入/输出信息", expanded=True):
        render_model_details()
        if st.session_state.get("preprocess_config"):
            st.caption(f"已读取 preprocess_config.json: {st.session_state.preprocess_config}")
        if st.session_state.get("decision_config"):
            st.caption(f"已读取 decision_config.json: {st.session_state.decision_config}")

        st.subheader("实时检测波形（A0 / CH0 Voltage" + (" / V raw_uV" if enable_v_format else "") + "）")
    render_waveform(float(display_seconds))

    st.subheader("实时神经网络输出（Impact score）")
    render_score_curve(float(display_seconds), float(threshold))

    lower_cols = st.columns(2)
    with lower_cols[0]:
        st.subheader("最近串口原始行")
        if st.session_state.recent_raw_lines:
            st.dataframe(pd.DataFrame(st.session_state.recent_raw_lines), use_container_width=True, hide_index=True)
        else:
            st.info("暂无原始串口行")

    with lower_cols[1]:
        st.subheader("Impact 事件记录")
        if st.session_state.impact_events:
            events_df = pd.DataFrame(st.session_state.impact_events)
            st.dataframe(events_df.iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("暂无 Impact 事件")

    if st.session_state.monitoring:
        rerun_later(int(refresh_ms))


if __name__ == "__main__":
    main()
