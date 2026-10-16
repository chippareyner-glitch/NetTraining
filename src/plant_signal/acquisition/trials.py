"""Session and Trial file management for raw acquisition data."""

from __future__ import annotations

import json
import os
import re
import csv
import time
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from plant_signal.config import RAW_DATA_DIR, ROOT_DIR


TRIAL_COLUMNS = [
    "sample_index",
    "time_s",
    "raw_value",
    "voltage_uV",
    "current_uA",
    "timestamp_ms",
    "adc_raw",
    "host_elapsed_s",
    "raw_line",
    "device_seq",
    "device_tick_us",
]

_SAVE_LOCK = threading.RLock()
_REPLACE_RETRY_DELAYS_S = (0.05, 0.1, 0.2, 0.4, 0.8, 1.2)


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrialPaths:
    session_id: str
    trial_id: str
    session_dir: Path
    partial_csv: Path
    final_csv: Path
    metadata_json: Path


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def relative_path_string(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def sanitize_identifier(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text or "").strip()).strip("_")
    return cleaned or fallback


def make_session_id(session_name: str = "", plant_id: str = "") -> str:
    date_part = datetime.now().strftime("%Y%m%d_%H%M%S")
    if session_name.strip():
        base = sanitize_identifier(session_name, f"session_{date_part}")
        if not base.lower().startswith("session"):
            base = f"session_{base}"
    else:
        base = f"session_{date_part}"

    plant = sanitize_identifier(plant_id, "")
    if plant and plant.lower() not in base.lower():
        base = f"{base}_{plant}"
    return base


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_sessions() -> list[SessionRecord]:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sessions: list[SessionRecord] = []
    for path in sorted(RAW_DATA_DIR.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir():
            continue
        metadata = load_json(path / "session.json")
        session_id = str(metadata.get("session_id") or path.name)
        sessions.append(SessionRecord(session_id=session_id, path=path, metadata=metadata))
    return sessions


def ensure_session(session_id: str, metadata: dict[str, Any]) -> Path:
    safe_session_id = sanitize_identifier(session_id, make_session_id())
    session_dir = RAW_DATA_DIR / safe_session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    session_json = session_dir / "session.json"
    existing = load_json(session_json)
    merged = {
        **existing,
        **metadata,
        "session_id": safe_session_id,
        "updated_time": now_iso(),
    }
    if "created_time" not in merged:
        merged["created_time"] = now_iso()
    session_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return session_dir


def _trial_number_from_name(path: Path) -> int | None:
    match = re.search(r"trial_(\d+)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def next_trial_id(session_dir: Path) -> str:
    used: list[int] = []
    for path in session_dir.glob("trial_*"):
        number = _trial_number_from_name(path)
        if number is not None:
            used.append(number)
    next_number = max(used, default=0) + 1
    return f"trial_{next_number:03d}"


def reserve_trial(session_id: str, session_metadata: dict[str, Any]) -> TrialPaths:
    session_dir = ensure_session(session_id, session_metadata)
    trial_id = next_trial_id(session_dir)
    return TrialPaths(
        session_id=session_dir.name,
        trial_id=trial_id,
        session_dir=session_dir,
        partial_csv=session_dir / f"{trial_id}.partial.csv",
        final_csv=session_dir / f"{trial_id}.csv",
        metadata_json=session_dir / f"{trial_id}.json",
    )


def paths_from_state(state: dict[str, str]) -> TrialPaths:
    session_dir = Path(state["session_dir"])
    return TrialPaths(
        session_id=str(state["session_id"]),
        trial_id=str(state["trial_id"]),
        session_dir=session_dir,
        partial_csv=Path(state["partial_csv"]),
        final_csv=Path(state["final_csv"]),
        metadata_json=Path(state["metadata_json"]),
    )


def paths_to_state(paths: TrialPaths) -> dict[str, str]:
    return {
        "session_id": paths.session_id,
        "trial_id": paths.trial_id,
        "session_dir": str(paths.session_dir),
        "partial_csv": str(paths.partial_csv),
        "final_csv": str(paths.final_csv),
        "metadata_json": str(paths.metadata_json),
    }


def samples_to_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        sample_index = int(sample.get("sample_index", index))
        time_s = float(sample.get("time_s", 0.0))
        voltage_uV = sample.get("voltage_uV")
        current_uA = sample.get("current_uA")
        sample_raw_value = sample.get("raw_value")
        voltage_value = None if voltage_uV in (None, "") else float(voltage_uV)
        current_value = None if current_uA in (None, "") else float(current_uA)
        raw_value = None if sample_raw_value in (None, "") else float(sample_raw_value)
        if raw_value is None:
            raw_value = voltage_value if voltage_value is not None else current_value
        rows.append(
            {
                "sample_index": sample_index,
                "time_s": time_s,
                "raw_value": raw_value,
                "voltage_uV": voltage_value,
                "current_uA": current_value,
                "timestamp_ms": time_s * 1000.0,
                "adc_raw": raw_value,
                "host_elapsed_s": float(sample.get("host_elapsed_s", 0.0)),
                "raw_line": str(sample.get("raw_line", "")),
                "device_seq": sample.get("device_seq"),
                "device_tick_us": sample.get("device_tick_us"),
            }
        )
    return rows


def samples_to_dataframe(samples: list[dict[str, Any]]) -> pd.DataFrame:
    rows = samples_to_rows(samples)
    return pd.DataFrame(rows, columns=TRIAL_COLUMNS)


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in TRIAL_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())


def _replace_with_retry(source: Path, destination: Path) -> None:
    last_error: PermissionError | None = None
    for index, delay_s in enumerate((0.0, *_REPLACE_RETRY_DELAYS_S)):
        if delay_s:
            time.sleep(delay_s)
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            last_error = exc
            if index >= len(_REPLACE_RETRY_DELAYS_S):
                break
    if last_error is not None:
        raise last_error


def _safe_unlink(path: Path) -> None:
    if not path.exists():
        return
    try:
        path.unlink()
    except OSError:
        pass


def _write_csv_atomic_or_direct(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    try:
        _write_csv_rows(temp_path, rows)
        _replace_with_retry(temp_path, path)
    except PermissionError:
        # Some Windows file watchers/security tools keep newly-created files
        # locked long enough that rename/replace fails. Direct overwrite is
        # less elegant, but it keeps the collected data usable.
        _write_csv_rows(path, rows)
    finally:
        _safe_unlink(temp_path)


def write_partial_csv(partial_csv: Path, samples: list[dict[str, Any]]) -> None:
    rows = samples_to_rows(samples)
    with _SAVE_LOCK:
        _write_csv_rows(partial_csv, rows)


def finalize_trial(
    paths: TrialPaths,
    samples: list[dict[str, Any]],
    metadata: dict[str, Any],
    events: list[dict[str, Any]],
) -> Path:
    if not samples:
        raise ValueError("cannot save an empty trial")
    with _SAVE_LOCK:
        rows = samples_to_rows(samples)
        _write_csv_rows(paths.final_csv, rows)
        _safe_unlink(paths.partial_csv)

    completed_metadata = {
        **metadata,
        "trial_id": paths.trial_id,
        "session_id": paths.session_id,
        "file_path": relative_path_string(paths.final_csv),
        "sample_count": int(len(samples)),
        "events": events,
        "status": "complete",
    }
    paths.metadata_json.write_text(
        json.dumps(completed_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths.final_csv


def list_partial_trials() -> list[Path]:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    partials = set(RAW_DATA_DIR.rglob("*.partial.csv"))
    partial_keys = {(path.parent, path.name.replace(".partial.csv", "")) for path in partials}

    temp_paths: set[Path] = set()
    for path in RAW_DATA_DIR.rglob("*.partial.csv.tmp"):
        key = (path.parent, path.name.replace(".partial.csv.tmp", ""))
        if key not in partial_keys:
            temp_paths.add(path)
    for path in RAW_DATA_DIR.rglob("*.partial.csv.*.tmp"):
        visible_name = path.name.lstrip(".")
        key = (path.parent, visible_name.split(".partial.csv.", 1)[0])
        if key not in partial_keys:
            temp_paths.add(path)

    return sorted(partials | temp_paths, key=lambda item: str(item).lower())


def delete_partial_trial(path: Path) -> None:
    path = Path(path)
    if path.exists() and (
        path.name.endswith(".partial.csv")
        or path.name.endswith(".partial.csv.tmp")
        or (".partial.csv." in path.name and path.name.endswith(".tmp"))
    ):
        path.unlink()


def recover_partial_trial(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"partial trial not found: {path}")

    if path.name.endswith(".partial.csv"):
        partial_path = path
        final_path = path.with_name(path.name.replace(".partial.csv", ".csv"))
    elif path.name.endswith(".partial.csv.tmp"):
        partial_path = path.with_name(path.name.replace(".partial.csv.tmp", ".partial.csv"))
        final_path = path.with_name(path.name.replace(".partial.csv.tmp", ".csv"))
        os.replace(path, partial_path)
    elif ".partial.csv." in path.name and path.name.endswith(".tmp"):
        visible_name = path.name.lstrip(".")
        partial_name = visible_name.split(".partial.csv.", 1)[0] + ".partial.csv"
        partial_path = path.with_name(partial_name)
        final_path = path.with_name(partial_name.replace(".partial.csv", ".csv"))
        os.replace(path, partial_path)
    else:
        raise FileNotFoundError(f"partial trial not found: {path}")

    if final_path.exists():
        raise FileExistsError(f"final trial already exists: {final_path}")
    os.replace(partial_path, final_path)
    metadata_path = final_path.with_suffix(".json")
    if not metadata_path.exists():
        dataframe = pd.read_csv(final_path)
        metadata = {
            "trial_id": final_path.stem,
            "session_id": final_path.parent.name,
            "file_path": relative_path_string(final_path),
            "sample_count": int(len(dataframe)),
            "status": "recovered_from_partial",
            "recovered_time": now_iso(),
            "events": [],
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_path


def metadata_for_trial(csv_path: Path) -> dict[str, Any]:
    metadata_path = Path(csv_path).with_suffix(".json")
    return load_json(metadata_path)
