"""CSV loading, session splitting, and MCU-friendly preprocessing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plant_signal.config import (
    ARTIFACT_DIR,
    CLASS_NAMES,
    DATA_CSV,
    MIN_PREPROCESS_SCALE,
    NUM_CLASSES,
    PREPROCESS_PATH,
    SCALE_PERCENTILE,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    WINDOW_SIZE,
)


POINT_COLUMNS = [f"p{i}" for i in range(WINDOW_SIZE)]
EXPECTED_COLUMNS = ["session_id", "label", *POINT_COLUMNS]


@dataclass(frozen=True)
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_sessions: tuple[str, ...]
    val_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]
    preprocess: dict[str, Any]


def _require_existing_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}\n"
            "If you are running the demo pipeline, generate it first with:\n"
            "python run.py demo"
        )


def _validate_csv(dataframe: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    if dataframe.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    missing = [name for name in EXPECTED_COLUMNS if name not in dataframe.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if len(dataframe.columns) != len(EXPECTED_COLUMNS):
        extra = [name for name in dataframe.columns if name not in EXPECTED_COLUMNS]
        raise ValueError(
            f"CSV must contain exactly {len(EXPECTED_COLUMNS)} columns "
            f"(session_id, label, p0..p127). Extra columns: {extra}"
        )

    dataframe = dataframe[EXPECTED_COLUMNS].copy()

    if dataframe["session_id"].isna().any():
        raise ValueError("session_id contains empty values.")
    dataframe["session_id"] = dataframe["session_id"].astype(str)
    if (dataframe["session_id"].str.len() == 0).any():
        raise ValueError("session_id contains blank strings.")

    try:
        dataframe["label"] = pd.to_numeric(dataframe["label"], errors="raise").astype(int)
        dataframe[POINT_COLUMNS] = dataframe[POINT_COLUMNS].apply(pd.to_numeric, errors="raise")
    except Exception as exc:  # noqa: BLE001 - keep message friendly for CSV users.
        raise ValueError(f"CSV contains non-numeric label or sample values: {exc}") from exc

    invalid_labels = sorted(set(dataframe["label"].tolist()) - set(CLASS_NAMES.keys()))
    if invalid_labels:
        raise ValueError(f"Invalid label(s): {invalid_labels}. Expected labels: {sorted(CLASS_NAMES.keys())}")

    if dataframe[POINT_COLUMNS].isna().any().any():
        raise ValueError("CSV contains NaN values in p0..p127.")

    return dataframe


def load_signal_csv(csv_path: Path = DATA_CSV) -> pd.DataFrame:
    """Load and validate a CSV file shaped as session_id,label,p0..p127."""
    csv_path = Path(csv_path)
    _require_existing_file(csv_path, "Signal CSV")
    try:
        dataframe = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read CSV file {csv_path}: {exc}") from exc
    return _validate_csv(dataframe, csv_path)


def _split_sessions(dataframe: pd.DataFrame, seed: int = SEED) -> dict[str, tuple[str, ...]]:
    sessions = np.array(sorted(dataframe["session_id"].unique().tolist()), dtype=object)
    session_count = len(sessions)
    if session_count < 3:
        raise ValueError(
            "At least 3 independent sessions are required so train, validation, "
            "and test splits can each receive a session."
        )

    ratio_sum = TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)

    n_train = max(1, int(round(session_count * TRAIN_RATIO)))
    n_val = max(1, int(round(session_count * VALIDATION_RATIO)))
    n_train = min(n_train, session_count - 2)
    n_val = min(n_val, session_count - n_train - 1)
    n_test = session_count - n_train - n_val

    if n_test < 1:
        raise ValueError("Could not allocate a non-empty test split. Add more sessions.")

    train_sessions = tuple(str(item) for item in sessions[:n_train])
    val_sessions = tuple(str(item) for item in sessions[n_train : n_train + n_val])
    test_sessions = tuple(str(item) for item in sessions[n_train + n_val :])

    _assert_disjoint_sessions(train_sessions, val_sessions, test_sessions)
    return {
        "train": train_sessions,
        "val": val_sessions,
        "test": test_sessions,
    }


def _assert_disjoint_sessions(
    train_sessions: tuple[str, ...],
    val_sessions: tuple[str, ...],
    test_sessions: tuple[str, ...],
) -> None:
    train = set(train_sessions)
    val = set(val_sessions)
    test = set(test_sessions)
    if train & val or train & test or val & test:
        raise RuntimeError("A session appears in more than one split. This would leak data.")


def _raw_arrays_for_sessions(dataframe: pd.DataFrame, sessions: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    split = dataframe[dataframe["session_id"].isin(sessions)].copy()
    x = split[POINT_COLUMNS].to_numpy(dtype=np.float32)
    y = split["label"].to_numpy(dtype=np.int64)
    if x.shape[1] != WINDOW_SIZE:
        raise ValueError(f"Expected {WINDOW_SIZE} sample points per row, got {x.shape[1]}")
    return x, y


def _compute_preprocess(raw_train_x: np.ndarray) -> dict[str, Any]:
    if raw_train_x.size == 0:
        raise ValueError("Training split is empty; cannot compute preprocessing scale.")

    centered = raw_train_x - np.mean(raw_train_x, axis=1, keepdims=True)
    abs_centered = np.abs(centered)
    scale = float(np.percentile(abs_centered, SCALE_PERCENTILE))

    if not np.isfinite(scale) or scale < MIN_PREPROCESS_SCALE:
        fallback = float(np.max(abs_centered)) if abs_centered.size else 0.0
        scale = fallback if fallback >= MIN_PREPROCESS_SCALE else 1.0

    return {
        "window_size": WINDOW_SIZE,
        "subtract_window_mean": True,
        "scale": scale,
        "scale_method": f"train_abs_centered_percentile_{SCALE_PERCENTILE:g}",
        "classes": {str(key): value for key, value in CLASS_NAMES.items()},
    }


def save_preprocess(preprocess: dict[str, Any], path: Path = PREPROCESS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preprocess, ensure_ascii=False, indent=2), encoding="utf-8")


def load_preprocess(path: Path = PREPROCESS_PATH) -> dict[str, Any]:
    path = Path(path)
    _require_existing_file(path, "Preprocess JSON")
    preprocess = json.loads(path.read_text(encoding="utf-8"))

    if int(preprocess.get("window_size", -1)) != WINDOW_SIZE:
        raise ValueError(f"preprocess.json window_size must be {WINDOW_SIZE}.")
    if preprocess.get("subtract_window_mean") is not True:
        raise ValueError("preprocess.json must set subtract_window_mean to true.")

    scale = float(preprocess.get("scale", 0.0))
    if not np.isfinite(scale) or scale < MIN_PREPROCESS_SCALE:
        raise ValueError(f"Invalid preprocess scale: {scale}")
    preprocess["scale"] = scale
    return preprocess


def apply_preprocess(raw_x: np.ndarray, preprocess: dict[str, Any]) -> np.ndarray:
    """Subtract each window mean and divide by the fixed training-derived scale."""
    scale = float(preprocess["scale"])
    centered = raw_x.astype(np.float32) - np.mean(raw_x.astype(np.float32), axis=1, keepdims=True)
    normalized = centered / scale
    return normalized.astype(np.float32)[..., np.newaxis]


def _class_count_text(labels: np.ndarray) -> str:
    counts = np.bincount(labels.astype(np.int64), minlength=NUM_CLASSES)
    return ", ".join(f"{CLASS_NAMES[index]}={int(counts[index])}" for index in range(NUM_CLASSES))


def _print_split_summary(
    name: str,
    sessions: tuple[str, ...],
    labels: np.ndarray,
) -> None:
    counts = np.bincount(labels.astype(np.int64), minlength=NUM_CLASSES)
    print(f"{name}: sessions={len(sessions)}, samples={len(labels)}, class_counts=({_class_count_text(labels)})")

    nonzero_counts = counts[counts > 0]
    if len(nonzero_counts) < NUM_CLASSES:
        print(f"WARNING: {name} split is missing at least one class.")
    elif float(np.max(nonzero_counts)) / float(np.min(nonzero_counts)) >= 2.0:
        print(f"WARNING: {name} split has class imbalance.")


def prepare_datasets(
    csv_path: Path = DATA_CSV,
    preprocess_path: Path = PREPROCESS_PATH,
    fit_preprocess: bool = True,
    save_fitted_preprocess: bool = False,
    verbose: bool = True,
) -> DatasetBundle:
    """Load CSV, split by session, and return Keras-ready arrays."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = load_signal_csv(csv_path)
    splits = _split_sessions(dataframe)

    raw_train_x, y_train = _raw_arrays_for_sessions(dataframe, splits["train"])
    raw_val_x, y_val = _raw_arrays_for_sessions(dataframe, splits["val"])
    raw_test_x, y_test = _raw_arrays_for_sessions(dataframe, splits["test"])

    preprocess = _compute_preprocess(raw_train_x) if fit_preprocess else load_preprocess(preprocess_path)
    if save_fitted_preprocess:
        save_preprocess(preprocess, preprocess_path)

    x_train = apply_preprocess(raw_train_x, preprocess)
    x_val = apply_preprocess(raw_val_x, preprocess)
    x_test = apply_preprocess(raw_test_x, preprocess)

    _assert_disjoint_sessions(splits["train"], splits["val"], splits["test"])

    if verbose:
        print("Dataset split by session_id:")
        _print_split_summary("train", splits["train"], y_train)
        _print_split_summary("val", splits["val"], y_val)
        _print_split_summary("test", splits["test"], y_test)
        print(f"Preprocess scale: {preprocess['scale']:.8g}")
        print(f"Keras input shape: {x_train.shape[1:]}")

    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        train_sessions=splits["train"],
        val_sessions=splits["val"],
        test_sessions=splits["test"],
        preprocess=preprocess,
    )


def main() -> None:
    prepare_datasets(save_fitted_preprocess=True)


if __name__ == "__main__":
    main()
