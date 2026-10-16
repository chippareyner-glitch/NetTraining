"""Threaded serial acquisition for Streamlit pages."""

from __future__ import annotations

import math
import random
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from plant_signal.acquisition.parser import parse_serial_line


@dataclass(frozen=True)
class SerialPortInfo:
    device: str
    description: str


@dataclass(frozen=True)
class SerialSample:
    sample_index: int
    time_s: float
    host_elapsed_s: float
    raw_line: str
    signal_type: str
    voltage_uV: float | None = None
    current_uA: float | None = None
    raw_value: float | None = None
    device_seq: int | None = None
    device_tick_us: int | None = None


@dataclass(frozen=True)
class CollectionEvent:
    event_type: str
    time_s: float
    host_time: str
    note: str = ""


SERIAL_BUSY_MESSAGE = "串口可能正被其他程序或另一个网页会话占用，请关闭串口助手、其他网页标签或旧的Python进程后重试。"
ADS_SPIKE_THRESHOLD_UV = 4_900_000.0
ADS_LOW_THRESHOLD_UV = 500.0
ADS_JUMP_THRESHOLD_UV = 10_000.0


class SerialPortInUseError(RuntimeError):
    """Raised when a port is already owned by another collector in this process."""


class SerialPortOpenError(RuntimeError):
    """Raised when pyserial cannot open the requested port."""


def list_serial_ports() -> list[SerialPortInfo]:
    """Return available serial ports.

    pyserial is optional so the app can still run in simulation mode.
    """

    try:
        from serial.tools import list_ports  # type: ignore[import-not-found]
    except Exception:
        return []

    return [
        SerialPortInfo(device=str(port.device), description=str(port.description or port.device))
        for port in list_ports.comports()
    ]


class SerialCollector:
    """Owns a serial/simulated source, a background reader, and sample buffers."""

    _registry_lock = threading.RLock()
    _open_serial_by_port: dict[str, "SerialCollector"] = {}

    def __init__(self, owner_id: str = "") -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial: Any | None = None
        self._owner_id = owner_id
        self._mode = "closed"
        self._port = ""
        self._baudrate = 0
        self._connected = False
        self._collecting = False
        self._last_error = ""
        self._last_open_call_site = ""
        self._samples: list[SerialSample] = []
        self._events: list[CollectionEvent] = []
        self._invalid_lines: deque[dict[str, str]] = deque(maxlen=50)
        self._recent_raw_lines: deque[dict[str, Any]] = deque(maxlen=20)
        self._raw_line_count = 0
        self._parse_error_count = 0
        self._filtered_count = 0
        self._sample_rate_hz = 500.0
        self._collection_start_monotonic = 0.0
        self._collection_start_time = ""
        self._collection_end_time = ""
        self._last_device_seq: int | None = None
        self._first_device_tick_us: int | None = None
        self._duplicate_seq_count = 0
        self._seq_jump_count = 0
        self._estimated_lost_samples = 0
        self._simulate_errors = False

    @staticmethod
    def _port_key(port: str) -> str:
        return str(port or "").strip().upper()

    @staticmethod
    def _format_call_site() -> str:
        stack = traceback.extract_stack(limit=10)
        for frame in reversed(stack[:-1]):
            filename = str(frame.filename).replace("\\", "/")
            if not filename.endswith("serial_reader.py"):
                return f"{filename.rsplit('/', 1)[-1]}:{frame.lineno} in {frame.name}"
        frame = stack[-2]
        return f"{frame.filename}:{frame.lineno} in {frame.name}"

    @staticmethod
    def _is_permission_error(exc: BaseException) -> bool:
        current: BaseException | None = exc
        while current is not None:
            if isinstance(current, PermissionError):
                return True
            current = current.__cause__ or current.__context__
        text = str(exc)
        return "PermissionError" in text or "拒绝访问" in text or "Access is denied" in text

    def set_owner_id(self, owner_id: str) -> None:
        with self._lock:
            self._owner_id = str(owner_id or "")

    def _serial_is_open_unlocked(self) -> bool:
        return bool(self._serial is not None and getattr(self._serial, "is_open", False))

    def _thread_is_alive_unlocked(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def _is_active_for_registry(self) -> bool:
        with self._lock:
            return bool(self._serial_is_open_unlocked() or self._thread_is_alive_unlocked() or self._connected)

    def _release_port_registration(self, port: str) -> None:
        key = self._port_key(port)
        if not key:
            return
        with self._registry_lock:
            if self._open_serial_by_port.get(key) is self:
                del self._open_serial_by_port[key]

    def _reserve_port_registration(self, port: str) -> None:
        key = self._port_key(port)
        with self._registry_lock:
            owner = self._open_serial_by_port.get(key)
            if owner is self:
                return
            if owner is not None:
                if owner._is_active_for_registry():
                    owner_debug = owner.get_debug_info()
                    owner_id = owner_debug.get("当前Streamlit会话标识") or "未知"
                    raise SerialPortInUseError(
                        f"当前进程中 {port} 已被另一个网页会话占用（会话 {owner_id}）。{SERIAL_BUSY_MESSAGE}"
                    )
                del self._open_serial_by_port[key]
            self._open_serial_by_port[key] = self

    def open_serial(self, port: str, baudrate: int) -> None:
        if not port:
            raise ValueError("serial port is required")
        if baudrate <= 0:
            raise ValueError("baudrate must be positive")

        port = str(port)
        baudrate = int(baudrate)
        call_site = self._format_call_site()
        with self._lock:
            if (
                self._mode == "serial"
                and self._port_key(self._port) == self._port_key(port)
                and self._baudrate == baudrate
                and self._serial_is_open_unlocked()
            ):
                self._last_open_call_site = call_site
                self._last_error = ""
                return
            needs_close = bool(self._serial is not None or self._thread is not None or self._connected)

        if needs_close:
            self.close()
        self._reserve_port_registration(port)
        try:
            import serial  # type: ignore[import-not-found]
        except Exception as exc:
            self._release_port_registration(port)
            raise RuntimeError("pyserial is not installed; use simulation mode or install pyserial.") from exc

        serial_obj = None
        try:
            serial_obj = serial.Serial(port=port, baudrate=baudrate, timeout=0.2)
            with self._lock:
                self._serial = serial_obj
                self._mode = "serial"
                self._port = port
                self._baudrate = baudrate
                self._connected = True
                self._last_error = ""
                self._last_open_call_site = call_site
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._serial_loop, name=f"serial-reader-{port}", daemon=True)
                thread = self._thread
            thread.start()
        except Exception as exc:
            if serial_obj is not None:
                try:
                    serial_obj.close()
                except Exception:
                    pass
            self._release_port_registration(port)
            with self._lock:
                self._thread = None
                self._serial = None
                self._connected = False
                self._collecting = False
                self._mode = "closed"
                self._port = ""
                self._baudrate = 0
                self._last_open_call_site = call_site
                self._last_error = SERIAL_BUSY_MESSAGE if self._is_permission_error(exc) else str(exc)
            if self._is_permission_error(exc):
                raise SerialPortOpenError(SERIAL_BUSY_MESSAGE) from exc
            raise SerialPortOpenError(f"无法打开串口 {port}：{exc}") from exc

    def open_simulated(self, sample_rate_hz: float = 500.0, simulate_errors: bool = False) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

        self.close()
        with self._lock:
            self._mode = "simulation"
            self._port = "SIMULATED"
            self._baudrate = 0
            self._connected = True
            self._sample_rate_hz = float(sample_rate_hz)
            self._simulate_errors = bool(simulate_errors)
            self._last_error = ""
            self._last_open_call_site = self._format_call_site()
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._simulation_loop, name=f"serial-simulation-{self._owner_id or 'session'}", daemon=True)
            thread = self._thread
        thread.start()

    def close(self) -> None:
        with self._lock:
            self._stop_event.set()
            self._collecting = False
            thread = self._thread
            serial_obj = self._serial
            port = self._port
            mode = self._mode

        if serial_obj is not None:
            try:
                cancel_read = getattr(serial_obj, "cancel_read", None)
                if callable(cancel_read):
                    cancel_read()
            except Exception:
                pass
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)

        if serial_obj is not None:
            try:
                serial_obj.close()
            except Exception as exc:
                with self._lock:
                    self._last_error = f"serial close failed: {exc}"
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.5)

        with self._lock:
            self._thread = None
            self._serial = None
            self._connected = False
            self._collecting = False
            if self._mode != "closed":
                self._mode = "closed"
            self._port = ""
            self._baudrate = 0
        if mode == "serial":
            self._release_port_registration(port)

    def start_collection(self, sample_rate_hz: float) -> str:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        with self._lock:
            if not self._connected:
                raise RuntimeError("serial source is not connected")
            if self._collecting:
                raise RuntimeError("collection is already running")
            self._samples = []
            self._events = []
            self._invalid_lines.clear()
            self._recent_raw_lines.clear()
            self._raw_line_count = 0
            self._parse_error_count = 0
            self._filtered_count = 0
            self._duplicate_seq_count = 0
            self._seq_jump_count = 0
            self._estimated_lost_samples = 0
            self._last_device_seq = None
            self._first_device_tick_us = None
            self._sample_rate_hz = float(sample_rate_hz)
            self._collection_start_monotonic = time.monotonic()
            self._collection_start_time = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
            self._collection_end_time = ""
            self._collecting = True
            return self._collection_start_time

    def stop_collection(self) -> str:
        with self._lock:
            if not self._collecting and self._collection_end_time:
                return self._collection_end_time
            self._collecting = False
            self._collection_end_time = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
            return self._collection_end_time

    def discard_collection(self) -> None:
        with self._lock:
            self._collecting = False
            self._samples = []
            self._events = []
            self._invalid_lines.clear()
            self._parse_error_count = 0
            self._filtered_count = 0
            self._collection_start_time = ""
            self._collection_end_time = ""

    def add_event(self, event_type: str, note: str = "") -> CollectionEvent:
        with self._lock:
            latest_time = self._samples[-1].time_s if self._samples else 0.0
            event = CollectionEvent(
                event_type=event_type,
                time_s=float(latest_time),
                host_time=datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
                note=str(note or ""),
            )
            self._events.append(event)
            return event

    def get_samples_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            samples, filtered_count = self._filtered_samples_unlocked()
            self._filtered_count = filtered_count
            return [asdict(sample) for sample in samples]

    def get_events_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(event) for event in self._events]

    def get_invalid_lines(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._invalid_lines)

    def get_recent_raw_lines(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._recent_raw_lines)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            samples, filtered_count = self._filtered_samples_unlocked()
            self._filtered_count = filtered_count
            sample_count = len(samples)
            latest = samples[-1] if samples else None
            if self._collecting:
                elapsed_s = max(0.0, time.monotonic() - self._collection_start_monotonic)
            elif latest is not None:
                elapsed_s = float(latest.host_elapsed_s)
            else:
                elapsed_s = 0.0
            receive_rate_hz = float(sample_count / elapsed_s) if elapsed_s > 0 else 0.0
            signal_type = latest.signal_type if latest else "voltage"
            signal_unit = "uV" if signal_type == "voltage" else "uA" if signal_type == "current" else ""
            signal_value = None
            if latest is not None:
                if latest.signal_type == "voltage":
                    signal_value = latest.voltage_uV
                elif latest.signal_type == "current":
                    signal_value = latest.current_uA
                else:
                    signal_value = latest.raw_value
            return {
                "mode": self._mode,
                "port": self._port,
                "baudrate": self._baudrate,
                "connected": self._connected,
                "collecting": self._collecting,
                "last_error": self._last_error,
                "sample_rate_hz": self._sample_rate_hz,
                "sample_count": sample_count,
                "elapsed_s": elapsed_s,
                "receive_rate_hz": receive_rate_hz,
                "parse_error_count": self._parse_error_count,
                "filtered_count": filtered_count,
                "raw_sample_count": len(self._samples),
                "raw_line_count": self._raw_line_count,
                "last_raw_line": self._recent_raw_lines[-1]["text"] if self._recent_raw_lines else "",
                "signal_type": signal_type,
                "signal_unit": signal_unit,
                "signal_value": signal_value,
                "voltage_uV": latest.voltage_uV if latest else None,
                "current_uA": latest.current_uA if latest else None,
                "raw_value": latest.raw_value if latest else None,
                "current_time_s": latest.time_s if latest else 0.0,
                "start_time": self._collection_start_time,
                "end_time": self._collection_end_time,
                "duplicate_seq_count": self._duplicate_seq_count,
                "seq_jump_count": self._seq_jump_count,
                "estimated_lost_samples": self._estimated_lost_samples,
            }

    def get_debug_info(self, streamlit_session_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            _, filtered_count = self._filtered_samples_unlocked()
            self._filtered_count = filtered_count
            serial_exists = self._serial is not None
            serial_is_open = self._serial_is_open_unlocked()
            thread_name = self._thread.name if self._thread is not None else ""
            thread_alive = self._thread_is_alive_unlocked()
            owner_id = streamlit_session_id or self._owner_id
            return {
                "当前Streamlit会话标识": owner_id,
                "当前采集控制器对象": hex(id(self)),
                "当前串口对象是否存在": serial_exists,
                "is_open状态": serial_is_open,
                "当前读取线程名称": thread_name,
                "读取线程是否存活": thread_alive,
                "当前是否正在采集": self._collecting,
                "连接模式": self._mode,
                "当前端口": self._port,
                "当前波特率": self._baudrate,
                "已收到原始行数": self._raw_line_count,
                "已过滤异常值数": self._filtered_count,
                "最后原始行": self._recent_raw_lines[-1]["text"] if self._recent_raw_lines else "",
                "打开串口的调用位置": self._last_open_call_site,
                "最后错误": self._last_error,
            }

    def _serial_loop(self) -> None:
        failure: Exception | None = None
        while not self._stop_event.is_set():
            try:
                serial_obj = self._serial
                if serial_obj is None:
                    break
                line = serial_obj.readline()
                if not line:
                    continue
                self._handle_line(line)
            except Exception as exc:
                failure = exc
                break
        if failure is not None:
            self._handle_reader_failure(failure)

    def _handle_reader_failure(self, exc: Exception) -> None:
        with self._lock:
            serial_obj = self._serial
            port = self._port
            self._last_error = f"串口读取失败或已断开：{exc}"
            self._serial = None
            self._connected = False
            self._collecting = False
            self._mode = "closed"
            self._port = ""
            self._baudrate = 0
        if serial_obj is not None:
            try:
                serial_obj.close()
            except Exception:
                pass
        self._release_port_registration(port)

    def _simulation_loop(self) -> None:
        sample_index = 0
        rng = random.Random()
        next_at = time.monotonic()
        while not self._stop_event.is_set():
            with self._lock:
                rate = max(1.0, float(self._sample_rate_hz))
                simulate_errors = self._simulate_errors
                collecting = self._collecting
            if not collecting:
                sample_index = 0
                next_at = time.monotonic() + 1.0 / rate
                time.sleep(0.02)
                continue
            now = time.monotonic()
            if now < next_at:
                time.sleep(min(0.02, next_at - now))
                continue
            if simulate_errors and sample_index > 0 and sample_index % 137 == 0:
                line = "BROKEN_LINE"
            else:
                t = sample_index / rate
                baseline = 937_000.0 + 800.0 * math.sin(2.0 * math.pi * 0.3 * t)
                noise = rng.uniform(-150.0, 150.0)
                tap = -7_000.0 * math.exp(-0.5 * ((t - 2.0) / 0.025) ** 2)
                touch = 4_000.0 if 4.0 <= t <= 6.4 else 0.0
                recovery = 1_800.0 * math.exp(-(t - 6.4) / 0.9) if t > 6.4 else 0.0
                voltage = baseline + noise + tap + touch + recovery
                line = f"CH0 Voltage={voltage:10f} uV"
            self._handle_line(line)
            sample_index += 1
            next_at += 1.0 / rate

    @staticmethod
    def _raw_line_hex(line: str | bytes) -> str:
        if isinstance(line, bytes):
            data = line[:48]
        else:
            data = str(line).encode("utf-8", errors="replace")[:48]
        return data.hex(" ")

    def _record_raw_line_unlocked(self, line: str | bytes, parsed: Any) -> None:
        self._raw_line_count += 1
        self._recent_raw_lines.append(
            {
                "host_time": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
                "text": parsed.raw_line,
                "hex": self._raw_line_hex(line),
                "valid": bool(parsed.valid),
                "signal_type": parsed.signal_type or "",
                "raw_value": "" if parsed.raw_value is None else parsed.raw_value,
                "voltage_uV": "" if parsed.voltage_uV is None else parsed.voltage_uV,
                "current_uA": "" if parsed.current_uA is None else parsed.current_uA,
                "error": parsed.error or "",
                "filtered": False,
                "filter_reason": "",
                "collecting": bool(self._collecting),
            }
        )

    def _handle_line(self, line: str | bytes) -> None:
        parsed = parse_serial_line(line)
        with self._lock:
            self._record_raw_line_unlocked(line, parsed)
            if not self._collecting:
                return
            if not parsed.valid or (
                parsed.voltage_uV is None and parsed.current_uA is None and parsed.raw_value is None
            ):
                self._parse_error_count += 1
                self._invalid_lines.append(
                    {
                        "host_time": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
                        "line": parsed.raw_line,
                        "error": parsed.error or "parse failed",
                    }
                )
                return

            sample_index = len(self._samples)
            host_elapsed_s = max(0.0, time.monotonic() - self._collection_start_monotonic)
            time_s = self._time_for_sample(sample_index, parsed.device_tick_us)
            self._update_sequence_stats(parsed.device_seq)
            signal_type = parsed.signal_type or (
                "voltage" if parsed.voltage_uV is not None else "current" if parsed.current_uA is not None else "raw"
            )
            self._samples.append(
                SerialSample(
                    sample_index=sample_index,
                    time_s=float(time_s),
                    host_elapsed_s=float(host_elapsed_s),
                    raw_line=parsed.raw_line,
                    signal_type=signal_type,
                    voltage_uV=None if parsed.voltage_uV is None else float(parsed.voltage_uV),
                    current_uA=None if parsed.current_uA is None else float(parsed.current_uA),
                    raw_value=None if parsed.raw_value is None else float(parsed.raw_value),
                    device_seq=parsed.device_seq,
                    device_tick_us=parsed.device_tick_us,
                )
            )

    @staticmethod
    def _voltage_filter_reason(sample: SerialSample, previous_raw_voltage_uV: float | None) -> str:
        if sample.voltage_uV is None:
            return ""
        voltage_uV = float(sample.voltage_uV)
        if voltage_uV > ADS_SPIKE_THRESHOLD_UV:
            return f"voltage_uV > {ADS_SPIKE_THRESHOLD_UV:.0f}"
        if voltage_uV < ADS_LOW_THRESHOLD_UV:
            return f"voltage_uV < {ADS_LOW_THRESHOLD_UV:.0f}"
        if previous_raw_voltage_uV is not None and abs(voltage_uV - previous_raw_voltage_uV) > ADS_JUMP_THRESHOLD_UV:
            return f"abs(voltage_uV - previous_voltage_uV) > {ADS_JUMP_THRESHOLD_UV:.0f}"
        return ""

    def _filtered_samples_unlocked(self) -> tuple[list[SerialSample], int]:
        filtered_samples: list[SerialSample] = []
        filtered_count = 0
        previous_raw_voltage_uV: float | None = None
        for sample in self._samples:
            reason = self._voltage_filter_reason(sample, previous_raw_voltage_uV)
            if reason:
                filtered_count += 1
            else:
                filtered_samples.append(sample)
            if sample.voltage_uV is not None:
                previous_raw_voltage_uV = float(sample.voltage_uV)
        return filtered_samples, filtered_count

    def _time_for_sample(self, sample_index: int, device_tick_us: int | None) -> float:
        if device_tick_us is not None:
            if self._first_device_tick_us is None:
                self._first_device_tick_us = int(device_tick_us)
            elapsed_us = int(device_tick_us) - int(self._first_device_tick_us)
            if elapsed_us >= 0:
                return elapsed_us / 1_000_000.0
        return sample_index / float(self._sample_rate_hz)

    def _update_sequence_stats(self, device_seq: int | None) -> None:
        if device_seq is None:
            return
        if self._last_device_seq is None:
            self._last_device_seq = int(device_seq)
            return

        expected = self._last_device_seq + 1
        if int(device_seq) == self._last_device_seq:
            self._duplicate_seq_count += 1
        elif int(device_seq) != expected:
            self._seq_jump_count += 1
            if int(device_seq) > expected:
                self._estimated_lost_samples += int(device_seq) - expected
        self._last_device_seq = int(device_seq)
