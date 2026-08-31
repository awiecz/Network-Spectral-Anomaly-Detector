"""Tests for flow-level ECOD detector and window aggregation."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.models.flow_ecod import FlowECODDetector, select_flow_feature_columns
from src.pipeline_helpers import apply_flow_ecod_fusion


def _flow_df(n: int, label: str = "BENIGN", scale: float = 1.0) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2017-07-07 08:00:00")
    for i in range(n):
        rows.append({
            "Flow Duration": 1e6 * scale,
            "Total Fwd Packets": 10.0 * scale,
            "Total Backward Packets": 8.0,
            "Total Length of Fwd Packets": 500.0 * scale,
            "Total Length of Bwd Packets": 400.0,
            "Label": label,
            "Timestamp": base + pd.Timedelta(seconds=i),
            "day": "Friday",
        })
    return pd.DataFrame(rows)


def test_select_flow_feature_columns_defaults():
    df = _flow_df(5)
    cols = select_flow_feature_columns(df)
    assert "Flow Duration" in cols
    assert len(cols) >= 5


def test_aggregate_to_windows_max_vs_mean():
    benign = _flow_df(30, "BENIGN", 1.0)
    attack = _flow_df(10, "DDOS", 50.0)
    attack["Timestamp"] = benign["Timestamp"].iloc[-1] + pd.to_timedelta(
        np.arange(10), unit="s"
    )
    df = pd.concat([benign, attack], ignore_index=True)

    det = FlowECODDetector(contamination=0.1)
    det.fit(benign)

    w_ts = np.array([benign["Timestamp"].iloc[0]])
    max_scores = det.aggregate_to_windows(
        df, w_ts, window_size=64, overlap=0, aggregation="max",
    )
    mean_scores = det.aggregate_to_windows(
        df, w_ts, window_size=64, overlap=0, aggregation="mean",
    )
    assert max_scores[0] >= mean_scores[0]
    assert len(max_scores) == 1


def test_flow_ecod_save_load_roundtrip():
    df = _flow_df(40)
    det = FlowECODDetector(contamination=0.1)
    det.fit(df)
    expected = det.score_flows(df)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "flow_ecod.pkl")
        det.save(path)
        loaded = FlowECODDetector.load(path)
        actual = loaded.score_flows(df)

    np.testing.assert_allclose(actual, expected)


def test_apply_flow_ecod_fusion_blend():
    df = pd.concat([
        _flow_df(50, "BENIGN").assign(day="Monday"),
        _flow_df(50, "DDOS", 20.0),
    ], ignore_index=True)

    cfg = {
        "data": {"train_day": "Monday"},
        "evaluation": {"eval_days": ["Friday"]},
        "spectral": {"window_size": 64, "overlap": 0},
        "models": {
            "flow_ecod": {
                "contamination": 0.1,
                "fusion_weight": 0.3,
                "aggregation": "max",
            },
        },
    }
    n = 10
    val_scores = {"ensemble": np.linspace(0.1, 0.9, n)}
    test_scores = {"ensemble": np.linspace(0.2, 0.8, n)}
    y_val = np.array([0] * 7 + [1] * 3)
    ts = pd.date_range("2017-07-07", periods=n, freq="64s")

    v_out, t_out, _ = apply_flow_ecod_fusion(
        df, cfg, val_scores, test_scores, y_val,
        ts.values, ts.values, flow_weight=0.3,
    )
    assert not np.allclose(v_out["ensemble"], val_scores["ensemble"])
    assert np.all((v_out["ensemble"] >= 0) & (v_out["ensemble"] <= 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
