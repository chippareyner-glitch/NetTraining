"""Generate synthetic demo data for the training pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plant_signal.config import (
    DATA_CSV,
    DATA_DIR,
    DEMO_SAMPLES_PER_CLASS_PER_SESSION,
    DEMO_SESSION_COUNT,
    SEED,
    WINDOW_SIZE,
)


def _base_signal(rng: np.random.Generator, session_offset: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, WINDOW_SIZE, dtype=np.float32)
    drift_slope = rng.normal(0.0, 8.0)
    slow_drift = drift_slope * (t - 0.5)
    mains_like = rng.normal(0.0, 2.0) * np.sin(2.0 * np.pi * rng.uniform(3.0, 7.0) * t + rng.uniform(0.0, 2.0 * np.pi))
    noise = rng.normal(0.0, rng.uniform(3.0, 7.0), size=WINDOW_SIZE)
    return 2048.0 + session_offset + slow_drift + mains_like + noise


def _rest_sample(rng: np.random.Generator, session_offset: float) -> np.ndarray:
    signal = _base_signal(rng, session_offset)
    t = np.linspace(0.0, 1.0, WINDOW_SIZE, dtype=np.float32)
    signal += rng.normal(0.0, 2.0) * np.sin(2.0 * np.pi * rng.uniform(0.4, 1.2) * t)
    return signal


def _tap_sample(rng: np.random.Generator, session_offset: float) -> np.ndarray:
    signal = _base_signal(rng, session_offset)
    idx = np.arange(WINDOW_SIZE, dtype=np.float32)
    center = rng.integers(28, 100)
    width = rng.uniform(1.8, 4.5)
    amplitude = rng.uniform(45.0, 120.0)
    spike = -amplitude * np.exp(-0.5 * ((idx - center) / width) ** 2)
    rebound = 0.22 * amplitude * np.exp(-0.5 * ((idx - center - rng.uniform(4.0, 9.0)) / (width * 1.7)) ** 2)
    signal += spike + rebound
    if rng.random() < 0.35:
        signal += rng.normal(0.0, 8.0, size=WINDOW_SIZE) * np.exp(-0.5 * ((idx - center) / 12.0) ** 2)
    return signal


def _hold_touch_sample(rng: np.random.Generator, session_offset: float) -> np.ndarray:
    signal = _base_signal(rng, session_offset)
    start = int(rng.integers(18, 52))
    end = int(rng.integers(78, 122))
    amplitude = rng.uniform(35.0, 100.0)
    plateau = np.zeros(WINDOW_SIZE, dtype=np.float32)
    plateau[start:end] = amplitude

    edge_width = int(rng.integers(3, 9))
    if start - edge_width >= 0:
        plateau[start - edge_width : start] = np.linspace(0.0, amplitude, edge_width, endpoint=False)
    if end + edge_width <= WINDOW_SIZE:
        plateau[end : end + edge_width] = np.linspace(amplitude, 0.0, edge_width, endpoint=False)

    plateau += rng.normal(0.0, 3.5, size=WINDOW_SIZE) * (plateau > 0.0)
    signal += plateau
    return signal


def generate_demo_dataframe(seed: int = SEED) -> pd.DataFrame:
    """Create a synthetic CSV-shaped dataframe with multiple independent sessions."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []

    for session_index in range(DEMO_SESSION_COUNT):
        session_id = f"demo_session_{session_index:02d}"
        session_offset = rng.normal(0.0, 45.0)

        generators = {
            0: _rest_sample,
            1: _tap_sample,
            2: _hold_touch_sample,
        }

        for label, make_sample in generators.items():
            for _ in range(DEMO_SAMPLES_PER_CLASS_PER_SESSION):
                sample = make_sample(rng, session_offset)
                row: dict[str, float | int | str] = {
                    "session_id": session_id,
                    "label": label,
                }
                for point_index, value in enumerate(sample):
                    row[f"p{point_index}"] = float(value)
                rows.append(row)

    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return dataframe


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = generate_demo_dataframe()
    dataframe.to_csv(DATA_CSV, index=False)

    print("这是synthetic demo data，仅用于测试程序，不能证明真实植物信号可以被正确分类。")
    print(f"Wrote {len(dataframe)} samples to: {DATA_CSV}")
    print(f"Sessions: {dataframe['session_id'].nunique()}")


if __name__ == "__main__":
    main()
