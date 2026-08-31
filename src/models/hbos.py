"""HBOS wrapper for the ensemble interface."""

from __future__ import annotations

import logging
import os

import joblib
import numpy as np
from pyod.models.hbos import HBOS

from .scoring import normalize_scores

logger = logging.getLogger(__name__)


class HBOSDetector:
    def __init__(self, contamination: float = 0.1, n_bins: int = 10) -> None:
        self.contamination = contamination
        self.n_bins = n_bins
        self.model = HBOS(contamination=contamination, n_bins=n_bins)
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._calibrated: bool = False

    def raw_score(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.decision_function(X), dtype=float)

    def set_score_bounds(self, score_low: float, score_high: float) -> None:
        self._score_min = float(score_low)
        self._score_max = float(score_high)
        self._calibrated = True

    def fit(self, X_train: np.ndarray) -> "HBOSDetector":
        self.model.fit(X_train)
        raw = np.asarray(self.model.decision_scores_, dtype=float)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        logger.info(
            "HBOS fitted on %d samples. Raw score range: [%.4f, %.4f]",
            len(X_train), self._score_min, self._score_max,
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return normalize_scores(self.raw_score(X), self._score_min, self._score_max)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "contamination": self.contamination,
                "n_bins": self.n_bins,
                "score_min": self._score_min,
                "score_max": self._score_max,
                "calibrated": self._calibrated,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "HBOSDetector":
        payload = joblib.load(path)
        inst = cls(
            contamination=payload.get("contamination", 0.1),
            n_bins=payload.get("n_bins", 10),
        )
        inst.model = payload["model"]
        inst._score_min = payload.get("score_min", 0.0)
        inst._score_max = payload.get("score_max", 1.0)
        inst._calibrated = payload.get("calibrated", False)
        return inst
