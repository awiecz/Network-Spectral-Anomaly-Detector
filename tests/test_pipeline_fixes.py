"""Regression tests for split logic, preprocessing, and threshold selection."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    find_optimal_threshold,
    find_threshold_at_fpr,
    normalize_threshold_method,
)
from src.preprocessor import DataPreprocessor
from src.splits import build_cicids_day_splits, build_disjoint_window_splits


def _minimal_cfg() -> dict:
    return {
        "spectral": {
            "bin_size_seconds": 1,
            "window_size": 64,
            "overlap": 0,
            "window_function": "hann",
            "rolloff_percentages": [0.85, 0.95],
            "input_series": [
                "flow_count",
                "total_bytes",
                "total_packets",
                "mean_duration",
            ],
        },
        "evaluation": {
            "window_label_rule": "majority",
            "benign_test_frac": 0.30,
        },
    }


def _make_flow_row(label: str, scale: float = 1.0) -> dict:
    return {
        "Flow Duration": 1e6 * scale,
        "Total Fwd Packets": 10.0 * scale,
        "Total Backward Packets": 8.0,
        "Total Length of Fwd Packets": 500.0 * scale,
        "Total Length of Bwd Packets": 400.0,
        "Label": label,
    }


def _flows_df(n_benign: int, n_attack: int, attack_start: int) -> pd.DataFrame:
    """One flow per second; attack block is contiguous."""
    rows = []
    for i in range(n_benign + n_attack):
        if attack_start <= i < attack_start + n_attack:
            rows.append(_make_flow_row("DDOS", scale=5.0))
        else:
            rows.append(_make_flow_row("BENIGN"))
    base = pd.Timestamp("2017-07-03 08:00:00")
    df = pd.DataFrame(rows)
    df["Timestamp"] = [base + pd.Timedelta(seconds=i) for i in range(len(df))]
    df["day"] = "Monday"
    return df


def test_cicids_holdout_label_alignment():
    """Holdout fine-grained labels must align with the temporal tail of benign windows."""
    cfg = _minimal_cfg()
    # Train day: long benign run + short tail (still benign flows at end)
    train_df = _flows_df(n_benign=500, n_attack=0, attack_start=10_000)
    # Eval day: attack burst only
    test_df = _flows_df(n_benign=0, n_attack=300, attack_start=50)
    test_df["day"] = "Friday"

    splits = build_cicids_day_splits(
        train_df, test_df, cfg, np.random.RandomState(42)
    )

    n_benign = len(splits["X_train"]) + sum(
        1 for lab in splits["lab_val"] if lab == "BENIGN"
    ) + sum(1 for lab in splits["lab_test"] if lab == "BENIGN")
    assert n_benign > 0

    # All benign holdout labels in val/test should be BENIGN (not mis-indexed attacks)
    for lab in splits["lab_val"]:
        if str(lab).upper() != "DDOS":
            assert str(lab).upper() == "BENIGN"
    for lab in splits["lab_test"]:
        if str(lab).upper() != "DDOS":
            assert str(lab).upper() == "BENIGN"

    assert splits["y_test"].sum() > 0
    assert len(splits["feat_names"]) == splits["X_train"].shape[1]


def test_disjoint_splits_no_overlap():
    """Earliest benign windows train; holdout + attacks never overlap train indices."""
    rng = np.random.RandomState(0)
    n_windows = 30
    n_feat = 4
    X_all = np.arange(n_windows * n_feat, dtype=float).reshape(n_windows, n_feat)
    all_ts = np.arange(n_windows, dtype=float)
    y_all = np.array([0] * 20 + [1] * 10, dtype=int)
    y_label_all = np.array(["BENIGN"] * 20 + ["DDOS"] * 10, dtype=object)

    X_train, X_val, y_val, X_test, y_test, _, _ = build_disjoint_window_splits(
        X_all, all_ts, y_all, y_label_all, benign_holdout_frac=0.30, rng=rng,
    )

    train_rows = {tuple(row) for row in X_train}
    eval_rows = {tuple(row) for row in np.vstack([X_val, X_test])}
    assert train_rows.isdisjoint(eval_rows)
    assert y_val.sum() > 0 or y_test.sum() > 0


def test_threshold_single_class():
    """Single-class validation must return a finite fallback threshold."""
    y = np.zeros(20, dtype=int)
    scores = np.linspace(0.0, 1.0, 20)
    thr, metric = find_optimal_threshold(y, scores, method="f1")
    assert np.isfinite(thr)
    assert metric == 0.0


def test_threshold_method_aliases():
    """f1_optimal and percentile aliases resolve consistently."""
    y = np.array([0, 0, 0, 0, 1, 1], dtype=int)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    t1, _ = find_optimal_threshold(y, scores, method="f1")
    t2, _ = find_optimal_threshold(y, scores, method="f1_optimal")
    assert t1 == t2


def test_find_threshold_at_fpr():
    """FPR-targeted threshold should achieve FPR <= target on validation."""
    rng = np.random.RandomState(0)
    y = np.array([0] * 80 + [1] * 20, dtype=int)
    scores = np.concatenate([
        rng.uniform(0.0, 0.5, 80),
        rng.uniform(0.6, 1.0, 20),
    ])
    thr, _ = find_threshold_at_fpr(y, scores, max_fpr=0.05)
    preds = (scores >= thr).astype(int)
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fpr = fp / max(fp + tn, 1)
    assert fpr <= 0.05 + 1e-6
    assert normalize_threshold_method("f1_optimal") == "f1"
    t_fpr, _ = find_optimal_threshold(y, scores, method="fpr_0.05")
    assert t_fpr == thr


def test_preprocessor_roundtrip():
    """Saved preprocessor reproduces the same scaled features after load."""
    rng = np.random.RandomState(1)
    X = rng.randn(50, 8).astype(np.float64)
    names = [f"f{i}" for i in range(8)]

    pre = DataPreprocessor()
    expected = pre.fit_transform(X, feature_names=names)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preprocessor.pkl")
        pre.save(path)
        loaded = DataPreprocessor.load(path)
        actual = loaded.transform(X)

    np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
