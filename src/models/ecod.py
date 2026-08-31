"""
ecod.py — ECOD (Empirical Cumulative Distribution Outlier Detection) wrapper.

ECOD is a parameter-free, deterministic outlier detector: it estimates the tail
probability of each feature independently via its empirical CDF and aggregates
the per-feature tail scores. It has no hyperparameters beyond *contamination*
(which only affects the built-in label threshold, not the raw scores used here).

This thin wrapper exposes the same ``fit`` / ``score`` (→ [0, 1]) / ``save`` /
``load`` interface as the other detectors so it can drop into the ensemble.

Reference: Li Z. et al. (2022). "ECOD: Unsupervised Outlier Detection Using
           Empirical Cumulative Distribution Functions." IEEE TKDE, 35(12).
"""

from __future__ import annotations

import logging
import os

import joblib
import numpy as np
from pyod.models.ecod import ECOD

from .scoring import normalize_scores

logger = logging.getLogger(__name__)


class ECODDetector:
    """pyod ECOD wrapped for the ensemble interface.

    Parameters
    ----------
    contamination:
        Assumed anomaly fraction (only used by pyod's internal labelling; the
        ensemble uses the normalised raw scores, so this does not affect AUROC).
    """

    def __init__(self, contamination: float = 0.1) -> None:
        self.contamination = contamination
        self.model = ECOD(contamination=contamination)
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._calibrated: bool = False

    # ------------------------------------------------------------------
    def raw_score(self, X: np.ndarray) -> np.ndarray:
        """Return unnormalised pyod decision scores."""
        return np.asarray(self.model.decision_function(X), dtype=float)

    def set_score_bounds(self, score_low: float, score_high: float) -> None:
        """Set percentile calibration bounds (validation benign)."""
        self._score_min = float(score_low)
        self._score_max = float(score_high)
        self._calibrated = True

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray) -> "ECODDetector":
        """Fit ECOD on *X_train* and record normalisation bounds."""
        self.model.fit(X_train)
        raw = np.asarray(self.model.decision_scores_, dtype=float)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        logger.info(
            "ECOD fitted on %d samples. Raw score range: [%.4f, %.4f]",
            len(X_train), self._score_min, self._score_max,
        )
        return self

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores normalised to [0, 1] (higher = more anomalous)."""
        return normalize_scores(self.raw_score(X), self._score_min, self._score_max)

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions (1 = anomaly) for a given *threshold*."""
        return (self.score(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the fitted model and bounds to *path* via joblib."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "contamination": self.contamination,
                "score_min": self._score_min,
                "score_max": self._score_max,
                "calibrated": self._calibrated,
            },
            path,
        )
        logger.info("ECODDetector saved to %s", path)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "ECODDetector":
        """Load a previously saved detector from *path*."""
        payload = joblib.load(path)
        inst = cls(contamination=payload.get("contamination", 0.1))
        inst.model = payload["model"]
        inst._score_min = payload.get("score_min", 0.0)
        inst._score_max = payload.get("score_max", 1.0)
        inst._calibrated = payload.get("calibrated", False)
        logger.info("ECODDetector loaded from %s", path)
        return inst
