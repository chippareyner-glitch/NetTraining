"""Parsing helpers for ADS signal samples received from serial text lines."""

from __future__ import annotations

import re
from dataclasses import dataclass


_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_CH0_VOLTAGE_PATTERN = re.compile(
    rf"\bCH0\s+Voltage\s*=\s*(?P<value>{_NUMBER_PATTERN})\s*uV\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedSerialLine:
    """A parsed serial line.

    Current STM32 firmware sends one ASCII line in this form:
    CH0 Voltage=886218.562500 uV
    """

    valid: bool
    raw_line: str
    signal_type: str | None = None
    voltage_uV: float | None = None
    current_uA: float | None = None
    raw_value: float | None = None
    device_seq: int | None = None
    device_tick_us: int | None = None
    error: str | None = None


def parse_serial_line(raw_line: str | bytes) -> ParsedSerialLine:
    """Parse one MCU serial line.

    Supported format:
    - CH0 Voltage=886218.562500 uV

    The value between Voltage= and uV is parsed as voltage_uV.
    """

    if isinstance(raw_line, bytes):
        text = raw_line.decode("utf-8", errors="replace")
    else:
        text = str(raw_line)

    text = text.replace("\x00", "").strip()
    if not text:
        return ParsedSerialLine(valid=False, raw_line=text, error="empty line")

    voltage_match = _CH0_VOLTAGE_PATTERN.search(text)
    if voltage_match is None:
        return ParsedSerialLine(
            valid=False,
            raw_line=text,
            error="missing CH0 Voltage=<float> uV format",
        )

    try:
        voltage_uV = float(voltage_match.group("value"))
    except ValueError as exc:
        return ParsedSerialLine(
            valid=False,
            raw_line=text,
            error=f"invalid voltage_uV value: {exc}",
        )

    return ParsedSerialLine(
        valid=True,
        raw_line=text,
        signal_type="voltage",
        voltage_uV=voltage_uV,
    )
