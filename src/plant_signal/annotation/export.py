"""Export completed Trial CSVs and tag JSONs into one training-friendly folder."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import stat
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from plant_signal.annotation.windows import (
    LABEL_BY_KIND,
    LABEL_HOLD_TOUCH,
    LABEL_REST,
    LABEL_TAP,
    TAG_ANNOTATIONS_DIR,
    clip_interval,
    iter_trial_csvs,
    load_completed_annotation,
    pair_tag_intervals,
)
from plant_signal.config import RAW_DATA_DIR, ROOT_DIR


DEFAULT_OUTPUT_DIR = ROOT_DIR / "data_labeled"
MANIFEST_ROW_NAME = "manifest_row.json"
DELETE_RETRY_DELAYS_S = (0.05, 0.1, 0.2, 0.4)
SIGNAL_COLUMNS = ("voltage_uV", "current_uA", "raw_value", "adc_raw")
LABEL_NAMES = {
    LABEL_REST: "\u9759\u606f",
    LABEL_TAP: "\u5f39\u51fb",
    LABEL_HOLD_TOUCH: "\u89e6\u6478",
    -1: "\u6392\u9664",
}
EXCLUDED_KIND_NAMES = {
    "noise": "\u566a\u58f0",
    "ignore": "\u5ffd\u7565",
}
KIND_LABEL_IDS = {
    "rest": LABEL_REST,
    "tap": LABEL_TAP,
    "touch": LABEL_HOLD_TOUCH,
    "noise": -1,
    "ignore": -1,
}


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def exported_trial_dir(output_dir: Path, session_id: str, trial_id: str) -> Path:
    return Path(output_dir) / session_id / trial_id


def ensure_path_within(path: Path, root: Path) -> None:
    path.resolve().relative_to(root.resolve())


def choose_signal_column(dataframe: pd.DataFrame) -> str:
    for column in SIGNAL_COLUMNS:
        if column in dataframe.columns and pd.to_numeric(dataframe[column], errors="coerce").notna().any():
            return column
    raise ValueError(f"CSV must contain at least one signal column: {', '.join(SIGNAL_COLUMNS)}")


def load_trial_dataframe(csv_path: Path) -> tuple[pd.DataFrame, str]:
    dataframe = pd.read_csv(csv_path)
    if "sample_index" not in dataframe.columns:
        raise ValueError(f"{csv_path} is missing required column: sample_index")

    dataframe = dataframe.copy()
    dataframe["sample_index"] = pd.to_numeric(dataframe["sample_index"], errors="raise").round().astype(int)
    signal_column = choose_signal_column(dataframe)
    dataframe[signal_column] = pd.to_numeric(dataframe[signal_column], errors="raise")
    return dataframe.sort_values("sample_index").reset_index(drop=True), signal_column


def apply_interval_labels(
    dataframe: pd.DataFrame,
    annotation: dict[str, Any],
) -> tuple[pd.DataFrame, Counter[str], list[dict[str, Any]]]:
    labeled = dataframe.copy()
    labeled["label_id"] = LABEL_REST
    labeled["label_name"] = LABEL_NAMES[LABEL_REST]
    labeled["interval_kind"] = "rest"
    labeled["train_use"] = 1

    if labeled.empty:
        return labeled, Counter(), []

    min_sample = int(labeled["sample_index"].iloc[0])
    max_sample = int(labeled["sample_index"].iloc[-1])
    raw_intervals, stats = pair_tag_intervals(list(annotation.get("tags") or []))
    intervals = [
        clipped
        for interval in raw_intervals
        if (clipped := clip_interval(interval, min_sample=min_sample, max_sample=max_sample)) is not None
    ]
    if len(intervals) != len(raw_intervals):
        stats["intervals_outside_signal"] += len(raw_intervals) - len(intervals)

    normalized_intervals: list[dict[str, Any]] = []
    for interval in intervals:
        label_id = KIND_LABEL_IDS.get(interval.kind)
        label_name = LABEL_NAMES.get(label_id, interval.kind) if label_id is not None else interval.kind
        if interval.kind in EXCLUDED_KIND_NAMES:
            label_name = EXCLUDED_KIND_NAMES[interval.kind]
        normalized_intervals.append(
            {
                "kind": interval.kind,
                "label_id": label_id,
                "label_name": label_name,
                "start_sample": int(interval.start_sample),
                "end_sample": int(interval.end_sample),
            }
        )

    # First assign trainable stimulus intervals.
    for interval in intervals:
        if interval.kind not in LABEL_BY_KIND:
            continue
        label_id = LABEL_BY_KIND[interval.kind]
        mask = labeled["sample_index"].between(interval.start_sample, interval.end_sample, inclusive="both")
        labeled.loc[mask, "label_id"] = label_id
        labeled.loc[mask, "label_name"] = LABEL_NAMES[label_id]
        labeled.loc[mask, "interval_kind"] = interval.kind
        labeled.loc[mask, "train_use"] = 1

    # Then make noise/ignore override everything else.
    for interval in intervals:
        if interval.kind not in EXCLUDED_KIND_NAMES:
            continue
        mask = labeled["sample_index"].between(interval.start_sample, interval.end_sample, inclusive="both")
        labeled.loc[mask, "label_id"] = -1
        labeled.loc[mask, "label_name"] = EXCLUDED_KIND_NAMES[interval.kind]
        labeled.loc[mask, "interval_kind"] = interval.kind
        labeled.loc[mask, "train_use"] = 0

    for interval in intervals:
        stats[f"intervals_{interval.kind}"] += 1
    return labeled, stats, normalized_intervals


def export_trial(csv_path: Path, annotation_path: Path, output_dir: Path) -> dict[str, Any]:
    annotation = load_completed_annotation(annotation_path)
    if annotation is None:
        raise ValueError(f"Annotation is missing or incomplete: {annotation_path}")

    dataframe, signal_column = load_trial_dataframe(csv_path)
    labeled, stats, normalized_intervals = apply_interval_labels(dataframe, annotation)

    session_id = csv_path.parent.name
    trial_id = csv_path.stem
    trial_output_dir = output_dir / session_id / trial_id
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    raw_output = trial_output_dir / "raw.csv"
    annotation_output = trial_output_dir / "annotation.json"
    labels_output = trial_output_dir / "labels.json"
    labeled_output = trial_output_dir / "samples_labeled.csv"
    metadata_output = trial_output_dir / "trial_metadata.json"
    manifest_row_output = trial_output_dir / MANIFEST_ROW_NAME

    shutil.copy2(csv_path, raw_output)
    shutil.copy2(annotation_path, annotation_output)
    metadata_source = csv_path.with_suffix(".json")
    if metadata_source.exists():
        shutil.copy2(metadata_source, metadata_output)

    labeled.to_csv(labeled_output, index=False, encoding="utf-8")
    labels_output.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "trial_id": trial_id,
                "source_csv": relative_path(csv_path),
                "source_annotation": relative_path(annotation_path),
                "completed": bool(annotation.get("completed") is True),
                "rest_all": bool(annotation.get("rest_all") is True),
                "default_label": LABEL_NAMES[LABEL_REST],
                "label_schema": {str(key): value for key, value in LABEL_NAMES.items()},
                "intervals": normalized_intervals,
                "note": "annotation.json is the original exported tag file; this labels.json is normalized for training.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    label_counts = Counter(int(value) for value in labeled["label_id"].tolist())
    train_use_count = int((labeled["train_use"] == 1).sum())
    row = {
        "session_id": session_id,
        "trial_id": trial_id,
        "source_csv": relative_path(csv_path),
        "source_annotation": relative_path(annotation_path),
        "export_dir": relative_path(trial_output_dir),
        "raw_csv": relative_path(raw_output),
        "annotation_json": relative_path(annotation_output),
        "labels_json": relative_path(labels_output),
        "samples_labeled_csv": relative_path(labeled_output),
        "trial_metadata_json": relative_path(metadata_output) if metadata_output.exists() else "",
        "signal_column": signal_column,
        "sample_count": int(len(labeled)),
        "train_sample_count": train_use_count,
        "rest_sample_count": int(label_counts[LABEL_REST]),
        "tap_sample_count": int(label_counts[LABEL_TAP]),
        "touch_sample_count": int(label_counts[LABEL_HOLD_TOUCH]),
        "excluded_sample_count": int(label_counts[-1]),
        "tag_count": len(annotation.get("tags") or []),
        "completed": bool(annotation.get("completed") is True),
        "rest_all": bool(annotation.get("rest_all") is True),
    }
    for key, value in sorted(stats.items()):
        row[key] = int(value)
    manifest_row_output.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def remove_exported_trial(output_dir: Path, session_id: str, trial_id: str) -> bool:
    output_dir = Path(output_dir)
    trial_output_dir = exported_trial_dir(output_dir, session_id, trial_id)
    ensure_path_within(trial_output_dir, output_dir)
    if not trial_output_dir.exists():
        return False
    if not trial_output_dir.is_dir():
        raise ValueError(f"Refusing to remove non-directory export path: {trial_output_dir}")

    last_error: Exception | None = None
    for index, delay_s in enumerate((0.0, *DELETE_RETRY_DELAYS_S)):
        if delay_s:
            time.sleep(delay_s)
        try:
            def make_writable_and_retry(function: Any, path_text: str, _: Any) -> None:
                try:
                    os.chmod(path_text, stat.S_IWRITE)
                    function(path_text)
                except Exception as exc:  # noqa: BLE001
                    raise exc

            shutil.rmtree(trial_output_dir, onerror=make_writable_and_retry)
            break
        except PermissionError as exc:
            last_error = exc
            gc.collect()
            if index >= len(DELETE_RETRY_DELAYS_S):
                raise
    else:
        if last_error is not None:
            raise last_error

    session_dir = trial_output_dir.parent
    try:
        if session_dir.exists() and session_dir.is_dir() and not any(session_dir.iterdir()):
            session_dir.rmdir()
    except OSError:
        pass
    return True


def rebuild_manifest_from_exports(output_dir: Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row_path in sorted(output_dir.glob(f"*/*/{MANIFEST_ROW_NAME}"), key=lambda item: str(item)):
        try:
            row = json.loads(row_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - keep a useful path in the error.
            raise ValueError(f"Could not read export manifest row: {row_path}") from exc
        if isinstance(row, dict):
            rows.append(row)

    manifest_path = output_dir / "manifest.csv"
    manifest = pd.DataFrame(rows).fillna("")
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    return manifest


def sync_labeled_trial_export(
    csv_path: Path,
    annotation_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    csv_path = Path(csv_path)
    annotation_path = Path(annotation_path)
    output_dir = Path(output_dir)
    session_id = csv_path.parent.name
    trial_id = csv_path.stem
    annotation = load_completed_annotation(annotation_path)
    if annotation is None:
        removed = remove_exported_trial(output_dir, session_id, trial_id)
        manifest = rebuild_manifest_from_exports(output_dir)
        return {
            "exported": False,
            "removed": removed,
            "manifest_rows": int(len(manifest)),
            "export_dir": relative_path(exported_trial_dir(output_dir, session_id, trial_id)),
        }

    row = export_trial(csv_path, annotation_path, output_dir)
    manifest = rebuild_manifest_from_exports(output_dir)
    return {
        "exported": True,
        "removed": False,
        "manifest_rows": int(len(manifest)),
        "export_dir": row.get("export_dir", relative_path(exported_trial_dir(output_dir, session_id, trial_id))),
        "row": row,
    }


def export_labeled_trials(
    raw_dir: Path = RAW_DATA_DIR,
    annotation_dir: Path = TAG_ANNOTATIONS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for csv_path in iter_trial_csvs(raw_dir):
        session_id = csv_path.parent.name
        trial_id = csv_path.stem
        annotation_path = annotation_dir / session_id / f"{trial_id}.json"
        if load_completed_annotation(annotation_path) is None:
            stats["skipped_missing_or_incomplete_annotation"] += 1
            try:
                remove_exported_trial(output_dir, session_id, trial_id)
            except PermissionError as exc:
                stats["stale_exports_locked"] += 1
                print(f"Warning: could not remove stale export {session_id}/{trial_id}: {exc}")
            continue
        try:
            rows.append(export_trial(csv_path, annotation_path, output_dir))
            stats["exported_trials"] += 1
        except Exception as exc:  # noqa: BLE001 - keep exporting other trials.
            stats["failed_trials"] += 1
            print(f"Failed: {csv_path} -> {exc}")

    if not rows:
        raise ValueError("No completed annotated trials were exported.")

    manifest = rebuild_manifest_from_exports(output_dir)
    manifest_path = output_dir / "manifest.csv"

    print(f"Exported trials: {stats['exported_trials']}")
    print(f"Skipped missing/incomplete annotations: {stats['skipped_missing_or_incomplete_annotation']}")
    if stats["failed_trials"]:
        print(f"Failed trials: {stats['failed_trials']}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote files under: {output_dir}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy completed raw CSVs and tag JSONs into data_labeled.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR, help="Raw data root.")
    parser.add_argument("--annotation-dir", type=Path, default=TAG_ANNOTATIONS_DIR, help="Trial tag JSON root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Export folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_labeled_trials(
        raw_dir=args.raw_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
