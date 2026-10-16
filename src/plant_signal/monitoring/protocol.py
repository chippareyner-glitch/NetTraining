"""Pure parsers for monitor serial formats; no Streamlit or serial-port access."""
from __future__ import annotations

import re
from typing import Any

A0_PATTERN = re.compile(
    r'\bA0\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:uA|μA|µA)?\b',
    re.IGNORECASE,
)
CH0_VOLTAGE_PATTERN = re.compile(
    r'\bCH0\s+Voltage\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:uV|μV|µV)?\b',
    re.IGNORECASE,
)
V_RAW_UV_PATTERN = re.compile(
    r'^\s*V\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:uV|μV|µV)?\s*$',
    re.IGNORECASE,
)
CLASSIFICATION_PATTERN = re.compile(
    r'^\s*CLS\s*=\s*(?P<class_id>-?\d+)\s*,\s*'
    r'LABEL\s*=\s*(?P<label>[A-Za-z_]+)\s*,\s*'
    r'CONF\s*=\s*(?P<confidence>\d+)\s*$',
    re.IGNORECASE,
)


def parse_signal_line(raw_line: str, *, allow_v_prefix: bool = False) -> float | None:
    text = raw_line.replace('\x00', '').strip()
    match = A0_PATTERN.search(text)
    if match is None:
        match = CH0_VOLTAGE_PATTERN.search(text)
    if match is None and allow_v_prefix:
        match = V_RAW_UV_PATTERN.search(text)
    if match is None:
        return None
    try:
        return float(match.group('value'))
    except ValueError:
        return None


def parse_classification_line(raw_line: str) -> dict[str, Any] | None:
    text = raw_line.replace('\x00', '').strip()
    match = CLASSIFICATION_PATTERN.fullmatch(text)
    if match is None:
        return None
    return {
        'class_id': int(match.group('class_id')),
        'label': match.group('label').upper(),
        'confidence': int(match.group('confidence')),
    }
