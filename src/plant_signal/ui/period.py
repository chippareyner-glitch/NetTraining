"""独立 Streamlit 小工具：周期干扰测量与静息模板抵消测试。"""

from __future__ import annotations

import io
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from plant_signal.paths import ROOT_DIR
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plant_signal.signal.periodic import (
    build_period_template,
    estimate_period,
    load_template,
    median_filter_3,
    remove_periodic_noise,
    save_template,
)

try:
    import serial
    from serial import SerialException
except ImportError:  # pragma: no cover - 页面会给出中文提示
    serial = None

    class SerialException(Exception):
        pass


BASE_DIR = ROOT_DIR
EXPORT_DIR = BASE_DIR / "period_noise_exports"
FLOAT_PATTERN = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?")
PREFERRED_COLUMNS = (
    "raw_value",
    "current_uA",
    "voltage_uV",
    "adc_raw",
    "value",
    "A0",
    "median_value",
)


def parse_last_float(text: str | bytes) -> float | None:
    """从一行串口文本中提取最后一个合法浮点数。"""

    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    matches = FLOAT_PATTERN.findall(str(text))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _new_buffer(source: str = "串口") -> dict[str, Any]:
    return {
        "values": [],
        "start_time": None,
        "end_time": None,
        "valid_lines": 0,
        "invalid_lines": 0,
        "source": source,
    }


class SerialBufferManager:
    """跨 Streamlit 重跑持久存在的串口资源和后台读取线程。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._serial = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._mode: str | None = None
        self._port = ""
        self._baudrate = 0
        self._timeout = 0.1
        self._last_error = ""
        self._last_line = ""
        self._total_valid = 0
        self._total_invalid = 0
        self._buffers = {
            "rest": _new_buffer(),
            "test": _new_buffer(),
        }

    def connect(self, port: str, baudrate: int, timeout: float = 0.1) -> str:
        if serial is None:
            raise RuntimeError("当前环境未安装 pyserial，请先安装后再连接串口。")

        with self._lock:
            if self._serial is not None and getattr(self._serial, "is_open", False):
                return "串口已经连接。"

        try:
            ser = serial.Serial(port=port, baudrate=int(baudrate), timeout=float(timeout))
        except PermissionError as exc:
            raise RuntimeError(f"没有权限打开串口 {port}，请确认串口未被其他程序占用。原始错误：{exc}") from exc
        except SerialException as exc:
            raise RuntimeError(f"打开串口 {port} 失败，请检查 COM 口和波特率。原始错误：{exc}") from exc

        self._stop_event.clear()
        with self._lock:
            self._serial = ser
            self._port = port
            self._baudrate = int(baudrate)
            self._timeout = float(timeout)
            self._last_error = ""

        self._thread = threading.Thread(target=self._read_loop, name="period-noise-serial-reader", daemon=True)
        self._thread.start()
        return f"已连接 {port}。"

    def disconnect(self) -> str:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        with self._lock:
            ser = self._serial
            self._serial = None
            self._thread = None
            self._mode = None

        if ser is not None:
            try:
                ser.close()
            except SerialException:
                pass
        return "串口已断开，后台读取线程已停止。"

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                ser = self._serial
            if ser is None:
                time.sleep(0.05)
                continue

            try:
                raw_line = ser.readline()
            except (PermissionError, SerialException) as exc:
                with self._lock:
                    self._last_error = f"串口读取失败：{exc}"
                break

            if not raw_line:
                continue

            text = raw_line.decode("utf-8", errors="replace").strip()
            value = parse_last_float(text)
            now = time.perf_counter()
            with self._lock:
                self._last_line = text
                active_mode = self._mode
                if value is None:
                    self._total_invalid += 1
                    if active_mode in self._buffers:
                        self._buffers[active_mode]["invalid_lines"] += 1
                        self._buffers[active_mode]["end_time"] = now
                    continue

                self._total_valid += 1
                if active_mode in self._buffers:
                    buffer = self._buffers[active_mode]
                    if buffer["start_time"] is None:
                        buffer["start_time"] = now
                    buffer["end_time"] = now
                    buffer["values"].append(float(value))
                    buffer["valid_lines"] += 1

    def start_collection(self, kind: str) -> None:
        with self._lock:
            if kind not in self._buffers:
                raise ValueError("未知采集类型。")
            if self._buffers[kind]["start_time"] is None:
                now = time.perf_counter()
                self._buffers[kind]["start_time"] = now
                self._buffers[kind]["end_time"] = now
            self._buffers[kind]["source"] = "串口"
            self._mode = kind

    def stop_collection(self, kind: str) -> None:
        with self._lock:
            if self._mode == kind:
                self._mode = None
            if kind in self._buffers:
                self._buffers[kind]["end_time"] = time.perf_counter()

    def clear_collection(self, kind: str) -> None:
        with self._lock:
            if self._mode == kind:
                self._mode = None
            self._buffers[kind] = _new_buffer()

    def load_collection(self, kind: str, values: np.ndarray, source: str) -> None:
        array = np.asarray(values, dtype=float).reshape(-1)
        finite = np.isfinite(array)
        array = array[finite]
        now = time.perf_counter()
        with self._lock:
            self._buffers[kind] = {
                "values": array.tolist(),
                "start_time": now,
                "end_time": now,
                "valid_lines": int(array.size),
                "invalid_lines": int((~finite).sum()),
                "source": source,
            }
            if self._mode == kind:
                self._mode = None

    def clear_serial_input(self) -> str:
        with self._lock:
            ser = self._serial
            self._total_valid = 0
            self._total_invalid = 0
            self._last_line = ""
        if ser is not None:
            try:
                ser.reset_input_buffer()
            except SerialException as exc:
                raise RuntimeError(f"清空串口接收缓冲区失败：{exc}") from exc
        return "串口接收缓冲区已清空。"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            buffers = {}
            for key, buffer in self._buffers.items():
                copied = dict(buffer)
                copied["values"] = np.asarray(buffer["values"], dtype=float)
                buffers[key] = copied
            connected = self._serial is not None and getattr(self._serial, "is_open", False)
            return {
                "connected": connected,
                "mode": self._mode,
                "port": self._port,
                "baudrate": self._baudrate,
                "timeout": self._timeout,
                "last_error": self._last_error,
                "last_line": self._last_line,
                "total_valid": self._total_valid,
                "total_invalid": self._total_invalid,
                "rest": buffers["rest"],
                "test": buffers["test"],
            }


@st.cache_resource
def get_serial_manager() -> SerialBufferManager:
    return SerialBufferManager()


def ensure_state() -> None:
    defaults = {
        "period_model": None,
        "period_result": None,
        "period_diagnostics": None,
        "template_diagnostics": None,
        "test_result": None,
        "last_saved_paths": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def read_values_from_csv_upload(uploaded_file) -> tuple[np.ndarray, str]:
    raw_bytes = uploaded_file.getvalue()
    source_name = uploaded_file.name

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception:
        df = pd.DataFrame()

    if not df.empty:
        for column in PREFERRED_COLUMNS:
            if column in df.columns:
                values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    return values, source_name

        numeric_candidates: list[tuple[int, str, np.ndarray]] = []
        ignored = {"sample_index", "time_s", "timestamp_ms", "host_elapsed_s", "device_seq", "device_tick_us"}
        for column in df.columns:
            if str(column) in ignored:
                continue
            values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            finite_count = int(np.isfinite(values).sum())
            if finite_count:
                numeric_candidates.append((finite_count, str(column), values))
        if numeric_candidates:
            _, _, values = max(numeric_candidates, key=lambda item: item[0])
            values = values[np.isfinite(values)]
            return values, source_name

        if "raw_line" in df.columns:
            parsed = [parse_last_float(item) for item in df["raw_line"].astype(str)]
            values = np.asarray([item for item in parsed if item is not None], dtype=float)
            if values.size:
                return values, source_name

    text = raw_bytes.decode("utf-8", errors="replace")
    parsed_values = []
    for line in text.splitlines():
        value = parse_last_float(line)
        if value is not None:
            parsed_values.append(value)
    values = np.asarray(parsed_values, dtype=float)
    if values.size == 0:
        raise ValueError("CSV 中没有找到可解析的数值列或浮点数行。")
    return values, source_name


def theoretical_duration(values: np.ndarray, fs: float) -> float:
    return float(values.size / fs) if fs > 0 else 0.0


def actual_duration(buffer: dict[str, Any]) -> float:
    start = buffer.get("start_time")
    end = buffer.get("end_time")
    if start is None or end is None or end <= start:
        return 0.0
    return float(end - start)


def estimated_sps(buffer: dict[str, Any]) -> float:
    duration = actual_duration(buffer)
    if duration <= 0:
        return 0.0
    return float(buffer["valid_lines"] / duration)


def plot_indices(length: int, max_points: int = 8000) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    if length <= max_points:
        return np.arange(length, dtype=int)
    step = int(np.ceil(length / max_points))
    return np.arange(0, length, step, dtype=int)


def make_signal_figure(title: str, series: list[tuple[str, np.ndarray, str]], fs: float, max_points: int = 8000) -> go.Figure:
    fig = go.Figure()
    max_len = max((len(values) for _, values, _ in series), default=0)
    idx = plot_indices(max_len, max_points)
    x = idx / fs if fs > 0 else idx

    for name, values, color in series:
        values = np.asarray(values, dtype=float)
        valid_idx = idx[idx < values.size]
        valid_x = valid_idx / fs if fs > 0 else valid_idx
        fig.add_trace(
            go.Scattergl(
                x=valid_x,
                y=values[valid_idx],
                mode="lines",
                name=name,
                line=dict(color=color, width=1.4),
            )
        )

    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=20, r=20, t=45, b=35),
        xaxis_title="时间 (s)" if fs > 0 else "样本点",
        yaxis_title="数值",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_autocorr_figure(diagnostics: dict[str, Any], fs: float) -> go.Figure:
    lags = np.asarray(diagnostics["autocorrelation_lags"], dtype=float)
    autocorr = np.asarray(diagnostics["autocorrelation"], dtype=float)
    idx = plot_indices(autocorr.size, max_points=6000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lags[idx] / fs, y=autocorr[idx], mode="lines", name="自相关"))
    peak_lags = np.asarray(diagnostics.get("peak_lags", []), dtype=float)
    if peak_lags.size:
        peak_lags = peak_lags[peak_lags < autocorr.size]
        fig.add_trace(
            go.Scatter(
                x=peak_lags / fs,
                y=autocorr[peak_lags.astype(int)],
                mode="markers",
                name="候选峰",
                marker=dict(color="#d62728", size=6),
            )
        )
    fig.update_layout(
        title="自相关曲线",
        height=340,
        margin=dict(l=20, r=20, t=45, b=35),
        xaxis_title="滞后时间 (s)",
        yaxis_title="归一化自相关",
        hovermode="x unified",
    )
    return fig


def make_segments_figure(template_diag: dict[str, Any], fs: float) -> go.Figure:
    aligned = np.asarray(template_diag["aligned_segments"], dtype=float)
    template = np.asarray(template_diag["template"], dtype=float)
    period = template.size
    x = np.arange(period, dtype=float) / fs
    fig = go.Figure()
    if aligned.ndim == 2 and aligned.shape[0] > 0:
        max_segments = min(80, aligned.shape[0])
        indices = np.linspace(0, aligned.shape[0] - 1, max_segments).round().astype(int)
        for idx in np.unique(indices):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=aligned[idx],
                    mode="lines",
                    name="周期片段",
                    line=dict(color="rgba(90, 120, 150, 0.18)", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=template,
            mode="lines",
            name="中位数周期模板",
            line=dict(color="#d62728", width=2.4),
        )
    )
    fig.update_layout(
        title="周期片段叠加与最终模板",
        height=360,
        margin=dict(l=20, r=20, t=45, b=35),
        xaxis_title="周期内时间 (s)",
        yaxis_title="去趋势值",
        hovermode="x unified",
    )
    return fig


def metrics_for_interval(raw: np.ndarray, cleaned: np.ndarray, noise: np.ndarray, fs: float, start_s: float, end_s: float) -> dict[str, float]:
    start = max(0, int(round(start_s * fs)))
    end = min(raw.size, int(round(end_s * fs)))
    if end <= start:
        raise ValueError("静息区间无有效样本，请检查起止秒数。")

    raw_seg = raw[start:end]
    cleaned_seg = cleaned[start:end]
    noise_seg = noise[start:end]

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    raw_std = float(np.std(raw_seg))
    cleaned_std = float(np.std(cleaned_seg))
    raw_ptp = float(np.ptp(raw_seg))
    cleaned_ptp = float(np.ptp(cleaned_seg))
    return {
        "raw_std": raw_std,
        "cleaned_std": cleaned_std,
        "raw_ptp": raw_ptp,
        "cleaned_ptp": cleaned_ptp,
        "std_reduction_pct": float((raw_std - cleaned_std) / raw_std * 100.0) if raw_std > 1e-12 else float("nan"),
        "ptp_reduction_pct": float((raw_ptp - cleaned_ptp) / raw_ptp * 100.0) if raw_ptp > 1e-12 else float("nan"),
        "raw_template_corr": corr(raw_seg, noise_seg),
        "cleaned_template_corr": corr(cleaned_seg, noise_seg),
    }


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def rest_dataframe(values: np.ndarray, fs: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_index": np.arange(values.size, dtype=int),
            "time_s": np.arange(values.size, dtype=float) / fs,
            "raw_value": values,
        }
    )


def test_result_dataframe(result: dict[str, np.ndarray], fs: float) -> pd.DataFrame:
    raw = np.asarray(result["raw_value"], dtype=float)
    return pd.DataFrame(
        {
            "sample_index": np.arange(raw.size, dtype=int),
            "time_s": np.arange(raw.size, dtype=float) / fs,
            "raw_value": raw,
            "median_value": np.asarray(result["median_value"], dtype=float),
            "estimated_periodic_noise": np.asarray(result["estimated_periodic_noise"], dtype=float),
            "cleaned_value": np.asarray(result["cleaned_value"], dtype=float),
            "template_scale": np.asarray(result["template_scale"], dtype=float),
            "phase_shift_samples": np.asarray(result["phase_shift_samples"], dtype=float),
        }
    )


def save_dataframe_with_timestamp(df: pd.DataFrame, prefix: str) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EXPORT_DIR / f"{prefix}_{timestamp}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def show_buffer_metrics(buffer: dict[str, Any], fs: float) -> None:
    values = np.asarray(buffer["values"], dtype=float)
    duration = actual_duration(buffer)
    cols = st.columns(6)
    cols[0].metric("已采集点数", f"{values.size}")
    cols[1].metric("理论时长", f"{theoretical_duration(values, fs):.2f} s")
    cols[2].metric("实际接收时长", f"{duration:.2f} s" if duration > 0 else "-")
    cols[3].metric("有效行", f"{buffer['valid_lines']}")
    cols[4].metric("无效行", f"{buffer['invalid_lines']}")
    cols[5].metric("估计 SPS", f"{estimated_sps(buffer):.1f}" if duration > 0 else "-")


def show_period_result(result: dict[str, Any]) -> None:
    cols = st.columns(6)
    cols[0].metric("估计周期", f"{result['period_seconds']:.4f} s")
    cols[1].metric("周期点数", f"{result['period_samples']}")
    cols[2].metric("频率", f"{result['frequency_hz']:.4f} Hz")
    cols[3].metric("自相关峰值", f"{result['autocorrelation_strength']:.3f}")
    cols[4].metric("峰距中位数", f"{result['peak_spacing_median_samples']:.1f}")
    cols[5].metric("周期 CV", f"{result['period_cv'] * 100.0:.2f}%")
    if result.get("warning"):
        st.warning(result["warning"])


def main() -> None:
    st.set_page_config(page_title="周期干扰测量与模板抵消测试", layout="wide")
    ensure_state()
    manager = get_serial_manager()

    st.title("周期干扰测量与模板抵消测试")

    st.header("1. 串口与采样设置")
    settings_cols = st.columns([1.2, 1.2, 1.2, 1.0])
    port = settings_cols[0].text_input("COM 口", value="")
    baudrate = settings_cols[1].number_input("波特率", min_value=300, max_value=4000000, value=115200, step=9600)
    fs = float(settings_cols[2].number_input("实际采样率 SPS", min_value=1.0, value=500.0, step=100.0))
    timeout = float(settings_cols[3].number_input("串口超时 (s)", min_value=0.01, max_value=5.0, value=0.1, step=0.01))

    button_cols = st.columns([1, 1, 1, 4])
    if button_cols[0].button("连接串口", key="connect_serial"):
        try:
            st.success(manager.connect(port, int(baudrate), timeout))
        except Exception as exc:
            st.error(f"连接串口失败：{exc}")

    if button_cols[1].button("断开串口", key="disconnect_serial"):
        st.info(manager.disconnect())

    if button_cols[2].button("清空接收缓冲区", key="clear_serial_buffer"):
        try:
            st.info(manager.clear_serial_input())
        except Exception as exc:
            st.error(str(exc))

    snapshot = manager.snapshot()
    if snapshot["connected"]:
        st.success(f"串口已连接：{snapshot['port']}，波特率 {snapshot['baudrate']}，当前采集模式：{snapshot['mode'] or '未采集'}")
    else:
        st.info("串口未连接。可以直接导入 CSV 做离线分析。")
    if snapshot["last_error"]:
        st.error(snapshot["last_error"])
    if snapshot["last_line"]:
        st.caption(f"最近一行：{snapshot['last_line']}")

    st.header("2. 静息校准")
    rest_cols = st.columns([1, 1, 1, 1.4, 1.4, 1.2])
    if rest_cols[0].button("开始采集静息数据", key="start_rest"):
        manager.start_collection("rest")
        st.info("已开始采集静息数据。")
    if rest_cols[1].button("停止静息采集", key="stop_rest"):
        manager.stop_collection("rest")
        st.info("已停止静息采集。")
    if rest_cols[2].button("清空静息数据", key="clear_rest"):
        manager.clear_collection("rest")
        st.session_state["period_model"] = None
        st.session_state["period_result"] = None
        st.session_state["period_diagnostics"] = None
        st.session_state["template_diagnostics"] = None
        st.info("静息数据已清空。")

    rest_upload = rest_cols[3].file_uploader("从 CSV 载入静息数据", type=["csv", "txt"], key="rest_csv_upload", label_visibility="collapsed")
    if rest_cols[4].button("载入静息 CSV", key="load_rest_csv"):
        if rest_upload is None:
            st.warning("请先选择静息 CSV 文件。")
        else:
            try:
                values, source = read_values_from_csv_upload(rest_upload)
                manager.load_collection("rest", values, source)
                st.success(f"已载入静息数据：{values.size} 点。")
            except Exception as exc:
                st.error(f"载入静息 CSV 失败：{exc}")

    snapshot = manager.snapshot()
    rest_buffer = snapshot["rest"]
    st.caption(f"静息数据来源：{rest_buffer['source']}。建议采集 60～120 秒。")
    show_buffer_metrics(rest_buffer, fs)
    rest_values = np.asarray(rest_buffer["values"], dtype=float)
    if rest_values.size:
        st.plotly_chart(
            make_signal_figure("静息原始波形（显示降采样，算法仍使用完整数据）", [("静息原始", rest_values, "#1f77b4")], fs),
            use_container_width=True,
        )

    with st.expander("周期分析参数", expanded=True):
        param_cols = st.columns(4)
        min_period_s = float(param_cols[0].number_input("最小周期 (s)", min_value=0.05, value=0.5, step=0.05))
        max_period_s = float(param_cols[1].number_input("最大周期 (s)", min_value=0.1, value=3.0, step=0.1))
        use_rest_median = param_cols[2].checkbox("周期分析使用 3 点中值滤波", value=False)
        align_ratio = float(param_cols[3].number_input("模板对齐搜索比例", min_value=0.01, max_value=0.5, value=0.1, step=0.01))

    analysis_cols = st.columns([1.2, 1.2, 1.2, 4])
    if analysis_cols[0].button("分析周期并生成模板", key="analyze_rest"):
        try:
            if rest_values.size < max(10, int(max_period_s * fs * 3)):
                raise ValueError("静息数据太短，建议至少包含 3 个最大周期。")
            analysis_values = median_filter_3(rest_values) if use_rest_median else rest_values.copy()
            period_result, period_diag = estimate_period(analysis_values, fs, min_period_s, max_period_s)
            model, template_diag = build_period_template(rest_values, fs, period_result["period_samples"], align_ratio)
            model["period_cv"] = period_result["period_cv"]
            model["autocorrelation_strength"] = period_result["autocorrelation_strength"]
            model["source_rest_file"] = str(rest_buffer["source"])
            model["config"].update(
                {
                    "min_period_s": min_period_s,
                    "max_period_s": max_period_s,
                    "use_median_filter_for_period": use_rest_median,
                    "align_ratio": align_ratio,
                }
            )
            st.session_state["period_model"] = model
            st.session_state["period_result"] = period_result
            st.session_state["period_diagnostics"] = period_diag
            st.session_state["template_diagnostics"] = template_diag
            st.success("周期分析完成，已生成当前周期模板。")
        except Exception as exc:
            st.error(f"周期分析失败：{exc}")

    if analysis_cols[1].button("保存周期模板", key="save_period_template"):
        model = st.session_state.get("period_model")
        if model is None:
            st.warning("请先分析周期并生成模板。")
        else:
            try:
                paths = save_template(BASE_DIR, model)
                st.session_state["last_saved_paths"] = {k: str(v) for k, v in paths.items()}
                st.success("周期模板已保存为 period_template.csv、period_template.npz 和 period_template_config.json。")
            except Exception as exc:
                st.error(f"保存周期模板失败：{exc}")

    template_upload = analysis_cols[2].file_uploader("载入模板 NPZ", type=["npz"], key="template_npz_upload", label_visibility="collapsed")
    if template_upload is not None and analysis_cols[3].button("从 NPZ 载入周期模板", key="load_template_npz"):
        try:
            temp_path = EXPORT_DIR / "_uploaded_period_template.npz"
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(template_upload.getvalue())
            st.session_state["period_model"] = load_template(temp_path)
            st.success("已从 NPZ 载入周期模板。")
        except Exception as exc:
            st.error(f"载入周期模板失败：{exc}")

    if st.session_state["period_result"] is not None:
        show_period_result(st.session_state["period_result"])

    if st.session_state["period_diagnostics"] is not None and st.session_state["template_diagnostics"] is not None:
        tabs = st.tabs(["去趋势波形", "自相关曲线", "周期片段与模板", "最终模板"])
        period_diag = st.session_state["period_diagnostics"]
        template_diag = st.session_state["template_diagnostics"]
        with tabs[0]:
            st.plotly_chart(
                make_signal_figure("去趋势后的静息波形", [("去趋势", np.asarray(period_diag["detrended_values"], dtype=float), "#2ca02c")], fs),
                use_container_width=True,
            )
        with tabs[1]:
            st.plotly_chart(make_autocorr_figure(period_diag, fs), use_container_width=True)
        with tabs[2]:
            st.plotly_chart(make_segments_figure(template_diag, fs), use_container_width=True)
        with tabs[3]:
            model = st.session_state["period_model"]
            if model is not None:
                template = np.asarray(model["template"], dtype=float)
                st.plotly_chart(
                    make_signal_figure("最终中位数周期模板", [("周期模板", template, "#d62728")], fs),
                    use_container_width=True,
                )

    st.header("3. 测试数据采集")
    test_cols = st.columns([1, 1, 1, 1.4, 1.4])
    if test_cols[0].button("开始测试采集", key="start_test"):
        manager.start_collection("test")
        st.info("已开始测试采集。")
    if test_cols[1].button("停止测试采集", key="stop_test"):
        manager.stop_collection("test")
        st.info("已停止测试采集。")
    if test_cols[2].button("清空测试数据", key="clear_test"):
        manager.clear_collection("test")
        st.session_state["test_result"] = None
        st.info("测试数据已清空。")
    test_upload = test_cols[3].file_uploader("从 CSV 载入测试数据", type=["csv", "txt"], key="test_csv_upload", label_visibility="collapsed")
    if test_cols[4].button("载入测试 CSV", key="load_test_csv"):
        if test_upload is None:
            st.warning("请先选择测试 CSV 文件。")
        else:
            try:
                values, source = read_values_from_csv_upload(test_upload)
                manager.load_collection("test", values, source)
                st.success(f"已载入测试数据：{values.size} 点。")
            except Exception as exc:
                st.error(f"载入测试 CSV 失败：{exc}")

    snapshot = manager.snapshot()
    test_buffer = snapshot["test"]
    st.caption(f"测试数据来源：{test_buffer['source']}。")
    show_buffer_metrics(test_buffer, fs)
    test_values = np.asarray(test_buffer["values"], dtype=float)
    if test_values.size:
        st.plotly_chart(
            make_signal_figure("原始测试波形（显示降采样，算法仍使用完整数据）", [("原始测试", test_values, "#1f77b4")], fs),
            use_container_width=True,
        )

    with st.expander("模板抵消参数", expanded=True):
        remove_cols = st.columns(7)
        block_periods = int(remove_cols[0].number_input("每块周期数", min_value=1, max_value=12, value=4, step=1))
        drift_ratio = float(remove_cols[1].number_input("周期漂移比例", min_value=0.0, max_value=0.2, value=0.05, step=0.01))
        scale_min = float(remove_cols[2].number_input("幅度 a 最小值", value=0.0, step=0.1))
        scale_max = float(remove_cols[3].number_input("幅度 a 最大值", value=3.0, step=0.1))
        freeze_scale = remove_cols[4].checkbox("冻结模板幅度", value=False)
        calibration_seconds = float(remove_cols[5].number_input("冻结校准秒数", min_value=0.1, value=5.0, step=0.5))
        use_test_median = remove_cols[6].checkbox("测试使用 3 点中值滤波", value=False)

    if st.button("使用当前模板处理", key="process_test"):
        try:
            model = st.session_state.get("period_model")
            if model is None:
                raise ValueError("还没有当前周期模板，请先在静息校准中生成或载入模板。")
            if test_values.size < int(model["period_samples"]):
                raise ValueError("测试数据太短，至少需要一个周期长度。")
            fit_values = median_filter_3(test_values) if use_test_median else test_values.copy()
            _, estimated_noise, scale_history, phase_history, diagnostics = remove_periodic_noise(
                fit_values,
                fs,
                model,
                block_periods=block_periods,
                max_period_drift_ratio=drift_ratio,
                scale_min=scale_min,
                scale_max=scale_max,
                freeze_scale=freeze_scale,
                calibration_seconds=calibration_seconds,
            )
            cleaned_values = test_values - estimated_noise
            st.session_state["test_result"] = {
                "raw_value": test_values.copy(),
                "median_value": fit_values,
                "estimated_periodic_noise": estimated_noise,
                "cleaned_value": cleaned_values,
                "template_scale": scale_history,
                "phase_shift_samples": phase_history,
                "diagnostics": diagnostics,
                "used_median_filter": use_test_median,
            }
            st.success("模板抵消处理完成。")
        except Exception as exc:
            st.error(f"测试数据处理失败：{exc}")

    st.header("4. 结果对比与导出")
    result = st.session_state.get("test_result")
    if result is None:
        st.info("完成测试处理后，这里会显示抵消前后对比、指标和导出按钮。")
    else:
        raw = np.asarray(result["raw_value"], dtype=float)
        median_value = np.asarray(result["median_value"], dtype=float)
        noise = np.asarray(result["estimated_periodic_noise"], dtype=float)
        cleaned = np.asarray(result["cleaned_value"], dtype=float)
        st.plotly_chart(
            make_signal_figure(
                "原始测试、估计周期干扰与模板抵消后波形",
                [
                    ("原始测试", raw, "#1f77b4"),
                    ("估计周期干扰", noise, "#ff7f0e"),
                    ("抵消后", cleaned, "#2ca02c"),
                ],
                fs,
            ),
            use_container_width=True,
        )

        if result.get("used_median_filter"):
            st.plotly_chart(
                make_signal_figure(
                    "测试中值滤波前后对比（原始数据未覆盖）",
                    [("原始测试", raw, "#1f77b4"), ("3 点中值", median_value, "#9467bd")],
                    fs,
                ),
                use_container_width=True,
            )

        tabs = st.tabs(["原始波形", "抵消后波形", "残差波形", "幅度系数 a", "相位偏移 tau"])
        with tabs[0]:
            st.plotly_chart(make_signal_figure("原始波形", [("原始", raw, "#1f77b4")], fs), use_container_width=True)
        with tabs[1]:
            st.plotly_chart(make_signal_figure("抵消后波形", [("抵消后", cleaned, "#2ca02c")], fs), use_container_width=True)
        with tabs[2]:
            st.plotly_chart(make_signal_figure("残差波形（原始 - 估计周期干扰）", [("残差", cleaned, "#d62728")], fs), use_container_width=True)
        block_records = result["diagnostics"].get("block_records", [])
        block_time = np.asarray([item["block_center_time_s"] for item in block_records], dtype=float)
        block_scale = np.asarray([item["scale"] for item in block_records], dtype=float)
        block_phase = np.asarray([item["phase_shift_samples"] for item in block_records], dtype=float)
        with tabs[3]:
            fig = go.Figure(go.Scatter(x=block_time, y=block_scale, mode="lines+markers", name="幅度系数 a"))
            fig.update_layout(height=320, xaxis_title="时间 (s)", yaxis_title="a", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        with tabs[4]:
            fig = go.Figure(go.Scatter(x=block_time, y=block_phase, mode="lines+markers", name="相位偏移 tau"))
            fig.update_layout(height=320, xaxis_title="时间 (s)", yaxis_title="tau (samples)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        duration_s = raw.size / fs
        metric_cols = st.columns(2)
        quiet_start = float(metric_cols[0].number_input("指标静息区间起始秒", min_value=0.0, max_value=max(0.0, duration_s), value=0.0, step=0.5))
        default_end = min(duration_s, max(quiet_start + 1.0, 5.0))
        quiet_end = float(metric_cols[1].number_input("指标静息区间结束秒", min_value=0.0, max_value=max(0.0, duration_s), value=default_end, step=0.5))
        try:
            metric = metrics_for_interval(raw, cleaned, noise, fs, quiet_start, quiet_end)
            metric_display_cols = st.columns(4)
            metric_display_cols[0].metric("原始标准差", f"{metric['raw_std']:.6g}")
            metric_display_cols[1].metric("抵消后标准差", f"{metric['cleaned_std']:.6g}")
            metric_display_cols[2].metric("标准差降低", f"{metric['std_reduction_pct']:.2f}%")
            metric_display_cols[3].metric("原始/模板相关", f"{metric['raw_template_corr']:.3f}")
            metric_display_cols = st.columns(4)
            metric_display_cols[0].metric("原始峰峰值", f"{metric['raw_ptp']:.6g}")
            metric_display_cols[1].metric("抵消后峰峰值", f"{metric['cleaned_ptp']:.6g}")
            metric_display_cols[2].metric("峰峰值降低", f"{metric['ptp_reduction_pct']:.2f}%")
            metric_display_cols[3].metric("抵消后/模板相关", f"{metric['cleaned_template_corr']:.3f}")
            if metric["cleaned_std"] > metric["raw_std"]:
                st.error("抵消后标准差反而变大，请谨慎使用当前模板或重新校准静息模板。")
        except Exception as exc:
            st.warning(f"指标计算失败：{exc}")

        export_cols = st.columns([1.2, 1.2, 1.2, 4])
        rest_df = rest_dataframe(rest_values, fs) if rest_values.size else None
        test_df = test_result_dataframe(result, fs)
        if export_cols[0].button("保存静息原始 CSV", key="save_rest_raw"):
            if rest_df is None:
                st.warning("当前没有静息数据可保存。")
            else:
                path = save_dataframe_with_timestamp(rest_df, "rest_raw")
                st.success(f"已保存：{path}")
        if export_cols[1].button("保存测试结果 CSV", key="save_test_processed"):
            path = save_dataframe_with_timestamp(test_df, "test_processed")
            st.success(f"已保存：{path}")
        if export_cols[2].button("保存周期模板", key="save_template_bottom"):
            model = st.session_state.get("period_model")
            if model is None:
                st.warning("当前没有周期模板可保存。")
            else:
                paths = save_template(BASE_DIR, model)
                st.success(f"已保存模板：{paths['npz']}")

        download_cols = st.columns(2)
        if rest_df is not None:
            download_cols[0].download_button(
                "下载静息原始 CSV",
                dataframe_to_csv_bytes(rest_df),
                file_name=f"rest_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_rest_raw",
            )
        download_cols[1].download_button(
            "下载测试结果 CSV",
            dataframe_to_csv_bytes(test_df),
            file_name=f"test_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_test_processed",
        )


if __name__ == "__main__":
    main()
