#!/usr/bin/env python3
"""Benchmark legacy vs vectorized spectral feature extraction."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.spectral_features import build_spectral_feature_matrix


def _make_timeseries(length: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2017-07-03", periods=length, freq="1s")
    return pd.DataFrame(
        {
            "flow_count": rng.poisson(5, length).astype(float),
            "total_bytes": rng.exponential(500, length),
            "total_packets": rng.exponential(20, length),
            "mean_duration": rng.exponential(1e5, length),
        },
        index=idx,
    )


def _time_build(ts_df: pd.DataFrame, force_legacy: bool) -> float:
    t0 = time.perf_counter()
    build_spectral_feature_matrix(ts_df, window_size=64, _force_legacy=force_legacy)
    return time.perf_counter() - t0


def main() -> None:
    for length in (1_000, 10_000):
        ts_df = _make_timeseries(length)
        legacy_s = _time_build(ts_df, force_legacy=True)
        fast_s = _time_build(ts_df, force_legacy=False)
        speedup = legacy_s / max(fast_s, 1e-9)
        print(
            f"n_bins={length:>6}  legacy={legacy_s:.3f}s  "
            f"vectorized={fast_s:.3f}s  speedup={speedup:.1f}x"
        )


if __name__ == "__main__":
    main()
