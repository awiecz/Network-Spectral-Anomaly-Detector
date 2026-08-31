"""
scoring.py — Shared anomaly score calibration utilities.

Percentile scaling on validation benign scores avoids train-set min-max
saturation when attack scores exceed the training distribution.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def fit_percentile_bounds(
    raw_scores: np.ndarray,
    benign_mask: Optional[np.ndarray] = None,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
) -> Tuple[float, float]:
    """Fit lower/upper percentile bounds for score normalisation."""
    raw = np.asarray(raw_scores, dtype=float)
    if benign_mask is not None:
        ref = raw[np.asarray(benign_mask, dtype=bool)]
        if len(ref) == 0:
            ref = raw
    else:
        ref = raw
    p_low = float(np.percentile(ref, low_pct))
    p_high = float(np.percentile(ref, high_pct))
    if p_high <= p_low:
        p_low = float(ref.min())
        p_high = float(ref.max())
        if p_high <= p_low:
            p_high = p_low + 1e-6
    return p_low, p_high


def normalize_scores(
    raw_scores: np.ndarray,
    score_low: float,
    score_high: float,
) -> np.ndarray:
    """Map raw scores to [0, 1] using percentile bounds."""
    raw = np.asarray(raw_scores, dtype=float)
    span = score_high - score_low
    if span < 1e-12:
        return np.zeros(len(raw))
    return np.clip((raw - score_low) / span, 0.0, 1.0)


def calibrate_detector_scores(
    detector,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
) -> None:
    """Set percentile calibration bounds on a detector using val benign scores."""
    raw_fn = getattr(detector, "raw_score", None)
    if raw_fn is None:
        return
    raw = raw_fn(X_val)
    benign_mask = np.asarray(y_val, dtype=int) == 0
    p_low, p_high = fit_percentile_bounds(raw, benign_mask, low_pct, high_pct)
    detector.set_score_bounds(p_low, p_high)
