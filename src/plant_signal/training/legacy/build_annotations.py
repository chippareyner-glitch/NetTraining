"""Build 128-point training windows from annotated raw trial CSV files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from plant_signal.config import (
    ANNOTATION_CSV,
    ANNOTATION_EDGE_MARGIN_SECONDS,
    CLASS_NAMES,
    REAL_DATA_CSV,
    ROOT_DIR,
    SAMPLING_RATE_HZ,
    SEED,
    WINDOW_SIZE,
)


BASE_RAW_COLUMNS = ["sample_index", "timestamp_ms", "time_s"]
ANNOTATION_COLUMNS = [
    "session_id",
    "trial_id",
    "file_path",
    "tap_sample",
    "hold_start_sample",
    "hold_end_sample",
]


@dataclass(frozen=True)
class WindowCandidate:
    session_id: str
    label: int
    values: np.ndarray


def resolve_data_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_raw_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    dataframe = pd.read_csv(path)
    if "sample_index" not in dataframe.columns:
        raise ValueError(f"Raw CSV {path} is missing required column: sample_index")

    dataframe = dataframe.copy()
    if "timestamp_ms" not in dataframe.columns:
        if "time_s" not in dataframe.columns:
            raise ValueError(f"Raw CSV {path} must contain timestamp_ms or time_s.")
        dataframe["timestamp_ms"] = pd.to_numeric(dataframe["time_s"], errors="raise") * 1000.0
    if "time_s" not in dataframe.columns:
        dataframe["time_s"] = pd.to_numeric(dataframe["timestamp_ms"], errors="raise") / 1000.0
    for column in BASE_RAW_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")

    if "voltage_uV" in dataframe.columns and pd.to_numeric(dataframe["voltage_uV"], errors="coerce").notna().any():
        signal_column = "voltage_uV"
    elif "current_uA" in dataframe.columns and pd.to_numeric(dataframe["current_uA"], errors="coerce").notna().any():
        signal_column = "current_uA"
    elif "raw_value" in dataframe.columns and pd.to_numeric(dataframe["raw_value"], errors="coerce").notna().any():
        signal_column = "raw_value"
    elif "adc_raw" in dataframe.columns and pd.to_numeric(dataframe["adc_raw"], errors="coerce").notna().any():
        signal_column = "adc_raw"
    else:
        raise ValueError(f"Raw CSV {path} must contain voltage_uV, current_uA, raw_value, or adc_raw.")
    dataframe[signal_column] = pd.to_numeric(dataframe[signal_column], errors="raise")
    dataframe["signal_value"] = dataframe[signal_column]

    if dataframe[[*BASE_RAW_COLUMNS, "signal_value"]].isna().any().any():
        raise ValueError(f"Raw CSV {path} contains NaN values.")
    if dataframe["sample_index"].duplicated().any():
        raise ValueError(f"Raw CSV {path} contains duplicate sample_index values.")

    return dataframe.sort_values("sample_index").reset_index(drop=True)


def load_annotations(annotation_path: Path) -> pd.DataFrame:
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation CSV not found: {annotation_path}\n"
            "Run the Streamlit annotator first and save annotations."
        )

    dataframe = pd.read_csv(annotation_path)
    missing = [column for column in ANNOTATION_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Annotation CSV is missing required columns: {missing}")

    dataframe = dataframe[ANNOTATION_COLUMNS].copy()
    for column in ["tap_sample", "hold_start_sample", "hold_end_sample"]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise").astype(int)

    if dataframe.empty:
        raise ValueError("Annotation CSV is empty.")
    return dataframe


def nearest_position_for_sample(dataframe: pd.DataFrame, sample_index: int) -> int:
    samples = dataframe["sample_index"].to_numpy(dtype=np.int64)
    return int(np.argmin(np.abs(samples - sample_index)))


def extract_window(dataframe: pd.DataFrame, center_sample: int) -> tuple[np.ndarray, int, int] | None:
    if len(dataframe) < WINDOW_SIZE:
        return None

    center_position = nearest_position_for_sample(dataframe, center_sample)
    start_position = center_position - WINDOW_SIZE // 2
    start_position = max(0, min(start_position, len(dataframe) - WINDOW_SIZE))
    end_position = start_position + WINDOW_SIZE
    values = dataframe["signal_value"].iloc[start_position:end_position].to_numpy(dtype=np.float32)
    if len(values) != WINDOW_SIZE:
        return None

    start_sample = int(dataframe["sample_index"].iloc[start_position])
    end_sample = int(dataframe["sample_index"].iloc[end_position - 1])
    return values, start_sample, end_sample


def _range_overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def make_tap_windows(dataframe: pd.DataFrame, session_id: str, tap_sample: int) -> list[WindowCandidate]:
    candidates: list[WindowCandidate] = []
    offsets = [-WINDOW_SIZE // 4, -WINDOW_SIZE // 8, 0, WINDOW_SIZE // 8, WINDOW_SIZE // 4]
    for offset in offsets:
        extracted = extract_window(dataframe, tap_sample + offset)
        if extracted is None:
            continue
        values, _, _ = extracted
        candidates.append(WindowCandidate(session_id=session_id, label=1, values=values))
    return candidates


def make_hold_windows(
    dataframe: pd.DataFrame,
    session_id: str,
    hold_start_sample: int,
    hold_end_sample: int,
    margin_samples: int,
) -> list[WindowCandidate]:
    candidates: list[WindowCandidate] = []
    safe_start = hold_start_sample + margin_samples + WINDOW_SIZE // 2
    safe_end = hold_end_sample - margin_samples - WINDOW_SIZE // 2
    if safe_start > safe_end:
        return candidates

    step = max(1, WINDOW_SIZE // 2)
    for center_sample in range(safe_start, safe_end + 1, step):
        extracted = extract_window(dataframe, center_sample)
        if extracted is None:
            continue
        values, window_start, window_end = extracted
        if window_start <= hold_start_sample + margin_samples:
            continue
        if window_end >= hold_end_sample - margin_samples:
            continue
        candidates.append(WindowCandidate(session_id=session_id, label=2, values=values))
    return candidates


def make_rest_windows(
    dataframe: pd.DataFrame,
    session_id: str,
    tap_sample: int,
    hold_start_sample: int,
    hold_end_sample: int,
    margin_samples: int,
) -> list[WindowCandidate]:
    candidates: list[WindowCandidate] = []
    samples = dataframe["sample_index"].to_numpy(dtype=np.int64)
    first_center = int(samples[0] + WINDOW_SIZE // 2)
    last_center = int(samples[-1] - WINDOW_SIZE // 2)
    forbidden_ranges = [
        (tap_sample - margin_samples, tap_sample + margin_samples),
        (hold_start_sample - margin_samples, hold_end_sample + margin_samples),
    ]

    for center_sample in range(first_center, last_center + 1, WINDOW_SIZE):
        extracted = extract_window(dataframe, center_sample)
        if extracted is None:
            continue
        values, window_start, window_end = extracted
        if any(_range_overlaps(window_start, window_end, start, end) for start, end in forbidden_ranges):
            continue
        candidates.append(WindowCandidate(session_id=session_id, label=0, values=values))
    return candidates


def candidates_from_annotation(row: pd.Series, sample_rate_hz: int) -> list[WindowCandidate]:
    raw_path = resolve_data_path(str(row["file_path"]))
    dataframe = load_raw_csv(raw_path)
    session_id = str(row["session_id"])
    tap_sample = int(row["tap_sample"])
    hold_start_sample = int(row["hold_start_sample"])
    hold_end_sample = int(row["hold_end_sample"])

    if hold_start_sample >= hold_end_sample:
        raise ValueError(f"Invalid HoldTouch range in {raw_path}: start must be less than end.")

    margin_samples = int(round(float(sample_rate_hz) * ANNOTATION_EDGE_MARGIN_SECONDS))
    candidates: list[WindowCandidate] = []
    candidates.extend(make_tap_windows(dataframe, session_id, tap_sample))
    candidates.extend(make_hold_windows(dataframe, session_id, hold_start_sample, hold_end_sample, margin_samples))
    candidates.extend(make_rest_windows(dataframe, session_id, tap_sample, hold_start_sample, hold_end_sample, margin_samples))
    return candidates


def balance_candidates(
    candidates: list[WindowCandidate],
    max_windows_per_class: int | None,
    seed: int,
) -> list[WindowCandidate]:
    by_label = {label: [item for item in candidates if item.label == label] for label in CLASS_NAMES}
    counts = {label: len(items) for label, items in by_label.items()}
    missing = [f"{label} {CLASS_NAMES[label]}" for label, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Could not generate windows for class(es): {missing}. Check annotations and hold duration.")

    balanced_count = min(counts.values())
    if max_windows_per_class is not None:
        balanced_count = min(balanced_count, max_windows_per_class)
    if balanced_count <= 0:
        raise ValueError("Balanced dataset would be empty.")

    rng = np.random.default_rng(seed)
    balanced: list[WindowCandidate] = []
    for label, items in by_label.items():
        indices = rng.permutation(len(items))[:balanced_count]
        balanced.extend(items[int(index)] for index in indices)

    shuffled_indices = rng.permutation(len(balanced))
    return [balanced[int(index)] for index in shuffled_indices]


def write_dataset(candidates: list[WindowCandidate], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for candidate in candidates:
        row: dict[str, float | int | str] = {
            "session_id": candidate.session_id,
            "label": candidate.label,
        }
        for index, value in enumerate(candidate.values):
            row[f"p{index}"] = float(value)
        rows.append(row)

    columns = ["session_id", "label", *[f"p{i}" for i in range(WINDOW_SIZE)]]
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


def build_dataset(
    annotation_path: Path = ANNOTATION_CSV,
    output_path: Path = REAL_DATA_CSV,
    sample_rate_hz: int = SAMPLING_RATE_HZ,
    max_windows_per_class: int | None = None,
) -> dict[str, int]:
    annotations = load_annotations(annotation_path)
    all_candidates: list[WindowCandidate] = []
    for _, row in annotations.iterrows():
        generated = candidates_from_annotation(row, sample_rate_hz=sample_rate_hz)
        all_candidates.extend(generated)

    raw_counts = {label: sum(1 for item in all_candidates if item.label == label) for label in CLASS_NAMES}
    balanced = balance_candidates(all_candidates, max_windows_per_class=max_windows_per_class, seed=SEED)
    balanced_counts = {label: sum(1 for item in balanced if item.label == label) for label in CLASS_NAMES}
    write_dataset(balanced, output_path)

    print(f"Read annotations: {len(annotations)}")
    print("Generated raw window candidates:")
    for label, count in raw_counts.items():
        print(f"  {label} {CLASS_NAMES[label]}: {count}")
    print("Balanced output windows:")
    for label, count in balanced_counts.items():
        print(f"  {label} {CLASS_NAMES[label]}: {count}")
    print(f"Wrote dataset: {output_path}")
    return {"total": len(balanced), **{str(label): count for label, count in balanced_counts.items()}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build plant_real_dataset.csv from trial annotations.")
    parser.add_argument("--annotations", type=Path, default=ANNOTATION_CSV, help="Annotation CSV path.")
    parser.add_argument("--output", type=Path, default=REAL_DATA_CSV, help="Output training CSV path.")
    parser.add_argument("--sample-rate", type=int, default=SAMPLING_RATE_HZ, help="Sampling rate in SPS.")
    parser.add_argument(
        "--max-windows-per-class",
        type=int,
        default=None,
        help="Optional cap after balancing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_rate <= 0:
        raise ValueError("sample-rate must be positive.")
    build_dataset(
        annotation_path=args.annotations,
        output_path=args.output,
        sample_rate_hz=args.sample_rate,
        max_windows_per_class=args.max_windows_per_class,
    )


if __name__ == "__main__":
    main()
