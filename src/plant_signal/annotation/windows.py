"""Build plant_real_dataset.csv from per-trial tag JSON annotations."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plant_signal.config import ANNOTATIONS_DIR, CLASS_NAMES, RAW_DATA_DIR, REAL_DATA_CSV, WINDOW_SIZE


LABEL_REST = 0
LABEL_TAP = 1
LABEL_HOLD_TOUCH = 2

TAG_ANNOTATIONS_DIR = ANNOTATIONS_DIR / "trial_tags"

TAG_ALIASES = {
    "弹击开始": "tap_start",
    "弹击结束": "tap_end",
    "impact": "tap_start",
    "tap": "tap_start",
    "tap_start": "tap_start",
    "tap_end": "tap_end",
    "impact_start": "tap_start",
    "impact_end": "tap_end",
    "触摸开始": "touch_start",
    "触摸结束": "touch_end",
    "touch_start": "touch_start",
    "touch_end": "touch_end",
    "噪声开始": "noise_start",
    "噪声结束": "noise_end",
    "noise": "noise_start",
    "noise_start": "noise_start",
    "noise_end": "noise_end",
    "异常开始": "noise_start",
    "异常结束": "noise_end",
    "anomaly_start": "noise_start",
    "anomaly_end": "noise_end",
    "忽略开始": "ignore_start",
    "忽略结束": "ignore_end",
    "ignore_start": "ignore_start",
    "ignore_end": "ignore_end",
    "unlabeled": "ignore_start",
}

PAIR_BY_START_TAG = {
    "tap_start": ("tap", "tap_end"),
    "touch_start": ("touch", "touch_end"),
    "noise_start": ("noise", "noise_end"),
    "ignore_start": ("ignore", "ignore_end"),
}
PAIR_BY_END_TAG = {end_tag: (kind, start_tag) for start_tag, (kind, end_tag) in PAIR_BY_START_TAG.items()}
LABEL_BY_KIND = {
    "tap": LABEL_TAP,
    "touch": LABEL_HOLD_TOUCH,
}


@dataclass(frozen=True)
class Interval:
    kind: str
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class Window:
    session_id: str
    label: int
    values: tuple[float, ...]


def is_trial_csv(path: Path) -> bool:
    if path.name.startswith(".") or path.name.endswith(".tmp") or path.name.endswith(".partial.csv"):
        return False
    return path.suffix.lower() == ".csv" and path.name.lower().startswith("trial_")


def iter_trial_csvs(raw_dir: Path) -> list[Path]:
    return sorted((path for path in raw_dir.glob("*/trial_*.csv") if is_trial_csv(path)), key=lambda item: str(item))


def load_raw_signal(csv_path: Path) -> tuple[list[int], list[float]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        missing = [name for name in ("sample_index", "voltage_uV") if name not in fieldnames]
        if missing:
            raise ValueError(f"{csv_path} is missing required column(s): {missing}")

        rows: list[tuple[int, float]] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                sample_index = int(float(str(row["sample_index"]).strip()))
                voltage = float(str(row["voltage_uV"]).strip())
            except Exception as exc:  # noqa: BLE001 - include row context in the raised error.
                raise ValueError(f"{csv_path}:{row_number} has invalid sample_index or voltage_uV") from exc
            if not math.isfinite(voltage):
                raise ValueError(f"{csv_path}:{row_number} has non-finite voltage_uV: {voltage}")
            rows.append((sample_index, voltage))

    if not rows:
        raise ValueError(f"{csv_path} contains no samples.")

    rows.sort(key=lambda item: item[0])
    samples = [sample for sample, _ in rows]
    if len(samples) != len(set(samples)):
        raise ValueError(f"{csv_path} contains duplicate sample_index values.")
    values = [value for _, value in rows]
    return samples, values


def load_completed_annotation(annotation_path: Path) -> dict[str, Any] | None:
    if not annotation_path.exists():
        return None
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if data.get("completed") is not True:
        return None
    return data


def normalize_tag(tag: Any) -> str | None:
    if tag is None:
        return None
    return TAG_ALIASES.get(str(tag).strip())


def tag_sample_index(tag: dict[str, Any]) -> int | None:
    value = tag.get("sample_index")
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def pair_tag_intervals(tags: list[dict[str, Any]]) -> tuple[list[Interval], Counter[str]]:
    pending_starts: dict[str, list[int]] = {kind: [] for kind, _ in PAIR_BY_START_TAG.values()}
    intervals: list[Interval] = []
    stats: Counter[str] = Counter()

    sortable_tags: list[tuple[int, dict[str, Any]]] = []
    for tag in tags:
        sample = tag_sample_index(tag)
        if sample is None:
            stats["tags_missing_sample_index"] += 1
            continue
        sortable_tags.append((sample, tag))

    for sample, tag in sorted(sortable_tags, key=lambda item: item[0]):
        normalized = normalize_tag(tag.get("tag"))
        if normalized is None:
            stats["tags_ignored_unknown_name"] += 1
            continue

        if normalized in PAIR_BY_START_TAG:
            kind, _ = PAIR_BY_START_TAG[normalized]
            pending_starts[kind].append(sample)
            continue

        if normalized in PAIR_BY_END_TAG:
            kind, _ = PAIR_BY_END_TAG[normalized]
            if not pending_starts[kind]:
                stats[f"unmatched_{kind}_end"] += 1
                continue
            start_sample = pending_starts[kind].pop(0)
            if sample < start_sample:
                stats[f"reversed_{kind}_interval"] += 1
                start_sample, sample = sample, start_sample
            if sample == start_sample:
                stats[f"zero_length_{kind}_interval"] += 1
                continue
            intervals.append(Interval(kind=kind, start_sample=start_sample, end_sample=sample))

    for kind, starts in pending_starts.items():
        if starts:
            stats[f"unmatched_{kind}_start"] += len(starts)
    return intervals, stats


def clip_interval(interval: Interval, min_sample: int, max_sample: int) -> Interval | None:
    start_sample = max(min_sample, interval.start_sample)
    end_sample = min(max_sample, interval.end_sample)
    if start_sample > end_sample:
        return None
    return Interval(kind=interval.kind, start_sample=start_sample, end_sample=end_sample)


def range_overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def merge_sample_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []

    merged: list[tuple[int, int]] = []
    for start_sample, end_sample in sorted(ranges):
        if start_sample > end_sample:
            continue
        if not merged or start_sample > merged[-1][1] + 1:
            merged.append((start_sample, end_sample))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_sample))
    return merged


def complement_ranges(
    covered_ranges: list[tuple[int, int]],
    min_sample: int,
    max_sample: int,
) -> list[tuple[int, int]]:
    rest_ranges: list[tuple[int, int]] = []
    cursor = min_sample
    for start_sample, end_sample in merge_sample_ranges(covered_ranges):
        if cursor < start_sample:
            rest_ranges.append((cursor, start_sample - 1))
        cursor = max(cursor, end_sample + 1)
    if cursor <= max_sample:
        rest_ranges.append((cursor, max_sample))
    return rest_ranges


def positions_for_sample_range(samples: list[int], start_sample: int, end_sample: int) -> tuple[int, int] | None:
    start_position = bisect.bisect_left(samples, start_sample)
    end_position = bisect.bisect_right(samples, end_sample) - 1
    if start_position > end_position:
        return None
    return start_position, end_position


def add_window_start(starts: list[int], start_position: int) -> None:
    if not starts or starts[-1] != start_position:
        starts.append(start_position)


def window_starts_inside_positions(
    start_position: int,
    end_position: int,
    signal_length: int,
    window_size: int,
    step: int,
    allow_short_center: bool,
) -> list[int]:
    if signal_length < window_size:
        return []

    interval_length = end_position - start_position + 1
    starts: list[int] = []
    if interval_length >= window_size:
        last_start = end_position - window_size + 1
        for window_start in range(start_position, last_start + 1, step):
            add_window_start(starts, window_start)
        add_window_start(starts, last_start)
    elif allow_short_center:
        center_position = (start_position + end_position) // 2
        window_start = center_position - window_size // 2
        window_start = max(0, min(window_start, signal_length - window_size))
        add_window_start(starts, window_start)
    return starts


def window_sample_range(samples: list[int], start_position: int, window_size: int) -> tuple[int, int]:
    end_position = start_position + window_size - 1
    return samples[start_position], samples[end_position]


def conflicts_with_other_kinds(
    window_start_sample: int,
    window_end_sample: int,
    intervals: list[Interval],
    allowed_kind: str,
) -> bool:
    for interval in intervals:
        if interval.kind == allowed_kind:
            continue
        if range_overlaps(window_start_sample, window_end_sample, interval.start_sample, interval.end_sample):
            return True
    return False


def windows_for_labeled_interval(
    samples: list[int],
    values: list[float],
    session_id: str,
    interval: Interval,
    all_intervals: list[Interval],
    window_size: int,
    step: int,
) -> list[Window]:
    positions = positions_for_sample_range(samples, interval.start_sample, interval.end_sample)
    if positions is None:
        return []

    starts = window_starts_inside_positions(
        start_position=positions[0],
        end_position=positions[1],
        signal_length=len(values),
        window_size=window_size,
        step=step,
        allow_short_center=True,
    )

    windows: list[Window] = []
    label = LABEL_BY_KIND[interval.kind]
    for start_position in starts:
        window_start_sample, window_end_sample = window_sample_range(samples, start_position, window_size)
        if conflicts_with_other_kinds(window_start_sample, window_end_sample, all_intervals, interval.kind):
            continue
        window_values = tuple(values[start_position : start_position + window_size])
        if len(window_values) == window_size:
            windows.append(Window(session_id=session_id, label=label, values=window_values))
    return windows


def windows_for_rest_range(
    samples: list[int],
    values: list[float],
    session_id: str,
    rest_range: tuple[int, int],
    window_size: int,
    step: int,
) -> list[Window]:
    positions = positions_for_sample_range(samples, rest_range[0], rest_range[1])
    if positions is None:
        return []

    starts = window_starts_inside_positions(
        start_position=positions[0],
        end_position=positions[1],
        signal_length=len(values),
        window_size=window_size,
        step=step,
        allow_short_center=False,
    )

    windows: list[Window] = []
    for start_position in starts:
        window_values = tuple(values[start_position : start_position + window_size])
        if len(window_values) == window_size:
            windows.append(Window(session_id=session_id, label=LABEL_REST, values=window_values))
    return windows


def build_windows_for_trial(
    csv_path: Path,
    annotation: dict[str, Any],
    window_size: int,
    event_step: int,
    rest_step: int,
) -> tuple[list[Window], Counter[str]]:
    samples, values = load_raw_signal(csv_path)
    session_id = csv_path.parent.name
    min_sample = samples[0]
    max_sample = samples[-1]

    raw_intervals, stats = pair_tag_intervals(list(annotation.get("tags") or []))
    clipped_intervals = [
        clipped
        for interval in raw_intervals
        if (clipped := clip_interval(interval, min_sample=min_sample, max_sample=max_sample)) is not None
    ]
    if len(clipped_intervals) != len(raw_intervals):
        stats["intervals_outside_signal"] += len(raw_intervals) - len(clipped_intervals)

    windows: list[Window] = []
    for interval in clipped_intervals:
        if interval.kind in LABEL_BY_KIND:
            windows.extend(
                windows_for_labeled_interval(
                    samples=samples,
                    values=values,
                    session_id=session_id,
                    interval=interval,
                    all_intervals=clipped_intervals,
                    window_size=window_size,
                    step=event_step,
                )
            )

    covered_ranges = [(interval.start_sample, interval.end_sample) for interval in clipped_intervals]
    for rest_range in complement_ranges(covered_ranges, min_sample=min_sample, max_sample=max_sample):
        windows.extend(
            windows_for_rest_range(
                samples=samples,
                values=values,
                session_id=session_id,
                rest_range=rest_range,
                window_size=window_size,
                step=rest_step,
            )
        )

    stats.update({f"intervals_{kind}": sum(1 for interval in clipped_intervals if interval.kind == kind) for kind in LABEL_BY_KIND})
    return windows, stats


def write_dataset(windows: list[Window], output_path: Path, window_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["session_id", "label", *[f"p{index}" for index in range(window_size)]]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for window in windows:
            writer.writerow([window.session_id, window.label, *window.values])


def build_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    annotation_dir: Path = TAG_ANNOTATIONS_DIR,
    output_path: Path = REAL_DATA_CSV,
    window_size: int = WINDOW_SIZE,
    event_step: int = WINDOW_SIZE,
    rest_step: int = WINDOW_SIZE,
) -> dict[str, int]:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if event_step <= 0:
        raise ValueError("event_step must be positive.")
    if rest_step <= 0:
        raise ValueError("rest_step must be positive.")

    trial_csvs = iter_trial_csvs(raw_dir)
    all_windows: list[Window] = []
    stats: Counter[str] = Counter(scanned_trial_csv=len(trial_csvs))

    for csv_path in trial_csvs:
        session_id = csv_path.parent.name
        trial_id = csv_path.stem
        annotation_path = annotation_dir / session_id / f"{trial_id}.json"
        annotation = load_completed_annotation(annotation_path)
        if annotation is None:
            stats["skipped_missing_or_incomplete_annotation"] += 1
            continue

        stats["completed_annotations"] += 1
        try:
            trial_windows, trial_stats = build_windows_for_trial(
                csv_path=csv_path,
                annotation=annotation,
                window_size=window_size,
                event_step=event_step,
                rest_step=rest_step,
            )
        except Exception:
            stats["skipped_trial_errors"] += 1
            raise

        stats.update(trial_stats)
        all_windows.extend(trial_windows)

    if not all_windows:
        raise ValueError("No windows were generated. Check completed annotations and tag pairs.")

    write_dataset(all_windows, output_path=output_path, window_size=window_size)
    label_counts = Counter(window.label for window in all_windows)
    print(f"Scanned trial CSVs: {stats['scanned_trial_csv']}")
    print(f"Completed annotations used: {stats['completed_annotations']}")
    print(f"Skipped missing/incomplete annotations: {stats['skipped_missing_or_incomplete_annotation']}")
    print("Output windows:")
    for label in sorted(CLASS_NAMES):
        print(f"  {label} {CLASS_NAMES[label]}: {label_counts[label]}")
    for name in sorted(key for key, value in stats.items() if value and key.startswith(("unmatched_", "tags_", "intervals_outside"))):
        print(f"Warning {name}: {stats[name]}")
    print(f"Wrote dataset: {output_path}")
    return {
        "total": len(all_windows),
        **{str(label): int(label_counts[label]) for label in sorted(CLASS_NAMES)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build data/plant_real_dataset.csv from trial tag annotations.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR, help="Raw data root: data_raw")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=TAG_ANNOTATIONS_DIR,
        help="Tag annotation root: annotations/trial_tags",
    )
    parser.add_argument("--output", type=Path, default=REAL_DATA_CSV, help="Output CSV path.")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Samples per output window.")
    parser.add_argument("--event-step", type=int, default=WINDOW_SIZE, help="Step for Tap/HoldTouch intervals.")
    parser.add_argument("--rest-step", type=int, default=WINDOW_SIZE, help="Step for Rest intervals.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        raw_dir=args.raw_dir,
        annotation_dir=args.annotation_dir,
        output_path=args.output,
        window_size=args.window_size,
        event_step=args.event_step,
        rest_step=args.rest_step,
    )


if __name__ == "__main__":
    main()
