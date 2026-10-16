"""CH32H417 serial waveform and classification result monitor.

Run:
    python run.py mcu
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DEFAULT_PORT = ""
DEFAULT_BAUDRATE = 115200
MAX_WAVE_POINTS = 20_000
MAX_CLASS_LOGS = 100
MAX_DEBUG_LOGS = 100
IMPACT_ALARM_HOLD_SECONDS = 2.0

VOLTAGE_PATTERN = re.compile(
    r"^\s*CH0\s+Voltage\s*=\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*uV\s*$",
    re.IGNORECASE,
)
CLASS_PATTERN = re.compile(
    r"^\s*CLS\s*=\s*(?P<class_id>-?\d+)\s*,\s*"
    r"LABEL\s*=\s*(?P<label>[A-Za-z_]+)\s*,\s*"
    r"CONF\s*=\s*(?P<confidence>\d+)\s*$",
    re.IGNORECASE,
)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "mcu_serial": None,
        "mcu_connected": False,
        "mcu_running": False,
        "mcu_error": "",
        "mcu_port": DEFAULT_PORT,
        "mcu_baudrate": DEFAULT_BAUDRATE,
        "wave_indices": [],
        "wave_values": [],
        "sample_count": 0,
        "current_class_id": None,
        "current_label": "",
        "current_confidence": None,
        "last_recognition_time": "",
        "impact_count": 0,
        "impact_alarm_until": 0.0,
        "classification_logs": [],
        "debug_logs": [],
        "last_raw_line": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def list_ports() -> list[str]:
    try:
        from serial.tools import list_ports as serial_list_ports

        return [str(port.device) for port in serial_list_ports.comports()]
    except Exception:
        return []


def close_serial() -> None:
    serial_obj = st.session_state.get("mcu_serial")
    if serial_obj is not None:
        try:
            serial_obj.close()
        except Exception:
            pass
    st.session_state.mcu_serial = None
    st.session_state.mcu_connected = False
    st.session_state.mcu_running = False


def open_serial(port: str, baudrate: int) -> None:
    serial_obj = st.session_state.get("mcu_serial")
    if serial_obj is not None and bool(getattr(serial_obj, "is_open", False)):
        st.session_state.mcu_error = "串口已经打开，无需重复连接。"
        return

    close_serial()
    try:
        import serial

        serial_obj = serial.Serial(
            port=port.strip(),
            baudrate=int(baudrate),
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.02,
        )
        serial_obj.reset_input_buffer()
        st.session_state.mcu_serial = serial_obj
        st.session_state.mcu_connected = True
        st.session_state.mcu_port = port.strip()
        st.session_state.mcu_baudrate = int(baudrate)
        st.session_state.mcu_error = ""
    except PermissionError:
        close_serial()
        st.session_state.mcu_error = (
            "串口可能正被其他程序或另一个网页会话占用，请关闭串口助手、"
            "其他网页标签或旧的 Python 进程后重试。"
        )
    except Exception as exc:
        close_serial()
        st.session_state.mcu_error = f"打开串口失败：{exc}"


def clear_data() -> None:
    st.session_state.wave_indices = []
    st.session_state.wave_values = []
    st.session_state.sample_count = 0
    st.session_state.current_class_id = None
    st.session_state.current_label = ""
    st.session_state.current_confidence = None
    st.session_state.last_recognition_time = ""
    st.session_state.impact_count = 0
    st.session_state.impact_alarm_until = 0.0
    st.session_state.classification_logs = []
    st.session_state.debug_logs = []
    st.session_state.last_raw_line = ""


def display_label(label: str) -> str:
    normalized = label.upper()
    if normalized == "REST":
        return "静息"
    if normalized == "IMPACT":
        return "检测到弹击"
    return normalized


def process_line(line: str) -> None:
    text = line.replace("\x00", "").strip()
    if not text:
        return
    st.session_state.last_raw_line = text

    class_match = CLASS_PATTERN.fullmatch(text)
    if class_match is not None:
        label = class_match.group("label").upper()
        previous_label = str(st.session_state.current_label or "").upper()
        if label == "IMPACT" and previous_label != "IMPACT":
            st.session_state.impact_count += 1
        if label == "IMPACT":
            st.session_state.impact_alarm_until = time.monotonic() + IMPACT_ALARM_HOLD_SECONDS

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        class_id = int(class_match.group("class_id"))
        confidence = int(class_match.group("confidence"))
        st.session_state.current_class_id = class_id
        st.session_state.current_label = label
        st.session_state.current_confidence = confidence
        st.session_state.last_recognition_time = now
        st.session_state.classification_logs.append(
            {
                "识别时间": now,
                "类别编号": class_id,
                "识别状态": display_label(label),
                "置信度": confidence,
                "原始标签": label,
            }
        )
        st.session_state.classification_logs = st.session_state.classification_logs[-MAX_CLASS_LOGS:]
        return

    voltage_match = VOLTAGE_PATTERN.fullmatch(text)
    if voltage_match is not None:
        value = float(voltage_match.group("value"))
        index = int(st.session_state.sample_count)
        st.session_state.wave_indices.append(index)
        st.session_state.wave_values.append(value)
        st.session_state.sample_count += 1
        st.session_state.wave_indices = st.session_state.wave_indices[-MAX_WAVE_POINTS:]
        st.session_state.wave_values = st.session_state.wave_values[-MAX_WAVE_POINTS:]
        return

    st.session_state.debug_logs.append(
        {
            "时间": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "无法解析的内容": text,
        }
    )
    st.session_state.debug_logs = st.session_state.debug_logs[-MAX_DEBUG_LOGS:]


def read_available_lines(max_lines: int = 1000) -> None:
    serial_obj = st.session_state.get("mcu_serial")
    if serial_obj is None or not bool(getattr(serial_obj, "is_open", False)):
        st.session_state.mcu_connected = False
        st.session_state.mcu_running = False
        return

    try:
        for _ in range(max_lines):
            if int(getattr(serial_obj, "in_waiting", 0) or 0) <= 0:
                break
            raw = serial_obj.readline()
            if not raw:
                break
            process_line(raw.decode("ascii", errors="replace"))
    except Exception as exc:
        st.session_state.mcu_error = f"读取串口失败：{exc}"
        close_serial()


def render_waveform(display_points: int) -> None:
    indices = st.session_state.wave_indices[-display_points:]
    values = st.session_state.wave_values[-display_points:]
    if not values:
        st.info("尚未收到电压数据。")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=indices,
            y=values,
            mode="lines",
            name="CH0 电压",
            line={"color": "#2563eb", "width": 1.5},
        )
    )
    fig.update_layout(
        height=430,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis_title="样本序号",
        yaxis_title="电压（微伏）",
        template="plotly_white",
        uirevision="mcu-waveform",
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


def render_classification() -> None:
    label = str(st.session_state.current_label or "").upper()
    alarm_active = time.monotonic() < float(st.session_state.impact_alarm_until)
    if alarm_active:
        st.markdown(
            """
            <style>
            @keyframes impact-pulse {
                0%, 100% {
                    box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.85),
                                0 0 28px 10px rgba(255, 220, 0, 0.90);
                    transform: scale(1);
                }
                50% {
                    box-shadow: 0 0 0 8px rgba(255, 255, 255, 1),
                                0 0 55px 22px rgba(255, 230, 0, 1);
                    transform: scale(1.015);
                }
            }
            .stApp {
                background: #b00020 !important;
            }
            [data-testid="stSidebar"] {
                background: #65000f !important;
            }
            .impact-alarm {
                animation: impact-pulse 0.55s ease-in-out infinite;
                background: #ff1f1f;
                border: 5px solid #fff200;
                color: #ffffff;
                font-size: 42px;
                font-weight: 800;
                letter-spacing: 0;
                margin: 14px 8px 28px;
                padding: 24px 18px;
                text-align: center;
            }
            </style>
            <div class="impact-alarm">检测到弹击</div>
            """,
            unsafe_allow_html=True,
        )
    elif label == "REST":
        st.success("当前识别状态：静息")
    elif label:
        st.info(f"当前识别状态：{display_label(label)}")
    else:
        st.info("当前识别状态：等待 MCU 发送分类结果")

    cols = st.columns(4)
    class_id = st.session_state.current_class_id
    confidence = st.session_state.current_confidence
    cols[0].metric("类别编号", "暂无" if class_id is None else str(class_id))
    cols[1].metric("置信度", "暂无" if confidence is None else str(confidence))
    cols[2].metric("最近识别时间", st.session_state.last_recognition_time or "暂无")
    cols[3].metric("弹击事件计数", int(st.session_state.impact_count))


@st.fragment(run_every=0.2)
def live_panel(display_points: int) -> None:
    if st.session_state.mcu_running:
        read_available_lines()

    render_classification()
    st.subheader("实时电压波形")
    render_waveform(display_points)

    left, right = st.columns(2)
    with left:
        st.subheader("最近识别日志")
        if st.session_state.classification_logs:
            logs = pd.DataFrame(st.session_state.classification_logs)
            st.dataframe(logs.iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("暂无识别日志。")
    with right:
        st.subheader("串口解析调试日志")
        if st.session_state.debug_logs:
            logs = pd.DataFrame(st.session_state.debug_logs)
            st.dataframe(logs.iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("暂无解析失败内容。")


def main() -> None:
    st.set_page_config(page_title="MCU 分类结果实时监测", layout="wide")
    init_state()
    st.title("MCU 分类结果实时监测")
    st.caption("同时接收 CH0 电压波形和 CH32H417 神经网络分类结果")

    with st.sidebar:
        st.header("串口设置")
        ports = list_ports()
        selected_port = st.selectbox(
            "串口",
            ports if ports else [st.session_state.mcu_port],
            index=0,
        )
        port = st.text_input("串口名称", value=selected_port)
        baudrate = st.number_input(
            "波特率",
            min_value=1,
            max_value=2_000_000,
            value=int(st.session_state.mcu_baudrate),
            step=9600,
        )

        connect_col, close_col = st.columns(2)
        if connect_col.button("打开串口", type="primary", use_container_width=True):
            open_serial(port, int(baudrate))
        if close_col.button("关闭串口", use_container_width=True):
            close_serial()

        run_col, pause_col = st.columns(2)
        if run_col.button("开始接收", use_container_width=True):
            if st.session_state.mcu_connected:
                st.session_state.mcu_running = True
                st.session_state.mcu_error = ""
            else:
                st.session_state.mcu_error = "请先打开串口。"
        if pause_col.button("暂停接收", use_container_width=True):
            st.session_state.mcu_running = False

        if st.button("清空显示数据", use_container_width=True):
            clear_data()

        display_points = st.slider("波形显示点数", 200, 20_000, 3000, 200)
        serial_obj = st.session_state.get("mcu_serial")
        is_open = bool(serial_obj is not None and getattr(serial_obj, "is_open", False))
        st.write(f"串口状态：{'已打开' if is_open else '未打开'}")
        st.write(f"接收状态：{'正在接收' if st.session_state.mcu_running else '已暂停'}")
        st.write(f"累计波形点数：{st.session_state.sample_count}")

    if st.session_state.mcu_error:
        st.warning(st.session_state.mcu_error)

    with st.expander("MCU 发送格式", expanded=False):
        st.code(
            "CH0 Voltage=886218.562500 uV\n"
            "CLS=0,LABEL=REST,CONF=81\n"
            "CLS=1,LABEL=IMPACT,CONF=127",
            language="text",
        )

    live_panel(int(display_points))


if __name__ == "__main__":
    main()
