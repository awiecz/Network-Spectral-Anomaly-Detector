"""Equivalence and regression tests for spectral feature extraction."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.spectral_features import (
    apply_log_transform,
    build_spectral_feature_matrix,
    build_standard_timeseries,
    extract_spectral_features,
)


def _make_timeseries(length: int = 200, seed: int = 0) -> pd.DataFrame:
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


def _flows_df(n_benign: int, n_attack: int, attack_start: int) -> pd.DataFrame:
    rows = []
    for i in range(n_benign + n_attack):
        if attack_start <= i < attack_start + n_attack:
            rows.append({
                "Flow Duration": 5e6,
                "Total Fwd Packets": 50.0,
                "Total Backward Packets": 8.0,
                "Total Length of Fwd Packets": 2500.0,
                "Total Length of Bwd Packets": 400.0,
                "Label": "DDOS",
            })
        else:
            rows.append({
                "Flow Duration": 1e6,
                "Total Fwd Packets": 10.0,
                "Total Backward Packets": 8.0,
                "Total Length of Fwd Packets": 500.0,
                "Total Length of Bwd Packets": 400.0,
                "Label": "BENIGN",
            })
    base = pd.Timestamp("2017-07-03 08:00:00")
    df = pd.DataFrame(rows)
    df["Timestamp"] = [base + pd.Timedelta(seconds=i) for i in range(len(df))]
    df["day"] = "Monday"
    return df


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_vectorized_matches_legacy_matrix(seed: int):
    ts_df = _make_timeseries(length=256, seed=seed)
    cols = list(ts_df.columns)
    rolloff = [0.85, 0.95]
    kwargs = dict(
        timeseries_df=ts_df,
        window_size=64,
        overlap=0,
        feature_cols=cols,
        fs=1.0,
        window_fn="hann",
        rolloff_pcts=rolloff,
    )
    legacy, ts_l, names_l = build_spectral_feature_matrix(**kwargs, _force_legacy=True)
    fast, ts_f, names_f = build_spectral_feature_matrix(**kwargs)
    assert names_l == names_f
    np.testing.assert_array_equal(ts_l, ts_f)
    np.testing.assert_allclose(legacy, fast, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "series",
    [
        np.sin(np.linspace(0, 8 * np.pi, 64)),
        np.random.RandomState(0).randn(64),
        np.zeros(64),
        np.concatenate([np.ones(32) * 10, np.zeros(32)]),
        np.array([np.nan] * 32 + list(np.linspace(1, 5, 32))),
    ],
)
def test_extract_spectral_features_finite(series: np.ndarray):
    feats = extract_spectral_features(series)
    assert all(np.isfinite(v) for v in feats.values())


def test_build_from_standard_timeseries_parity():
    df = _flows_df(n_benign=500, n_attack=100, attack_start=200)
    ts_df = build_standard_timeseries(df)
    legacy, _, names = build_spectral_feature_matrix(
        ts_df, window_size=64, _force_legacy=True
    )
    fast, _, _ = build_spectral_feature_matrix(ts_df, window_size=64)
    assert legacy.shape == fast.shape
    assert legacy.shape[1] == len(names)
    np.testing.assert_allclose(legacy, fast, rtol=1e-10, atol=1e-12)


def test_delta_features_add_columns():
    ts_df = _make_timeseries(length=128, seed=7)
    base, _, names_base = build_spectral_feature_matrix(ts_df, window_size=64, delta_features=False)
    delta, _, names_delta = build_spectral_feature_matrix(ts_df, window_size=64, delta_features=True)
    assert delta.shape[1] == base.shape[1] + 8  # 4 channels * 2 delta stats
    assert all(n.endswith("__delta_mean") or n.endswith("__delta_std") for n in names_delta[-8:])


def test_log_transform_non_negative():
    ts_df = _make_timeseries(length=64, seed=3)
    out = apply_log_transform(ts_df, ["total_bytes"])
    assert (out["total_bytes"].to_numpy() >= 0).all()


def test_vectorized_large_series_under_time_budget():
    """Coarse regression guard: 10K bins should finish quickly."""
    ts_df = _make_timeseries(length=10_000, seed=99)
    t0 = time.perf_counter()
    build_spectral_feature_matrix(ts_df, window_size=64)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"vectorized path too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
