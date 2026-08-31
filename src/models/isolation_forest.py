"""
isolation_forest.py — Isolation Forest wrapper for anomaly detection.

Isolation Forest isolates anomalies by randomly selecting a feature and a
split value between the observed min and max.  Anomalous samples require
fewer splits to isolate (shorter path lengths) and therefore receive higher
anomaly scores.

Reference: Liu F.T., Ting K.M. & Zhou Z-H. (2008). "Isolation Forest."
           ICDM 2008, pp. 413–422.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from .scoring import normalize_scores

logger = logging.getLogger(__name__)

np.random.seed(42)


class IsolationForestDetector:
    """Scikit-learn Isolation Forest wrapped for the ensemble interface.

    All anomaly scores are normalised to [0, 1] so they are directly
    comparable with VAE reconstruction errors in the ensemble weighted sum.

    Parameters
    ----------
    n_estimators:
        Number of isolation trees.
    max_samples:
        Number of samples drawn to train each tree.  ``'auto'`` uses
        min(256, n_samples).
    contamination:
        Assumed fraction of anomalies in the training set.  Use ``'auto'``
        when the training set is benign-only (CICIDS2017 Monday).
    n_jobs:
        Parallelism.  ``-1`` uses all available CPU cores.
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int | str = 256,
        contamination: float | str = "auto",
        max_features: float | int | str = 1.0,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.model = IsolationForest(
            n_estimators  = n_estimators,
            max_samples   = max_samples,
            contamination = contamination,
            max_features  = max_features,
            n_jobs        = n_jobs,
            random_state  = random_state,
        )

        # Score normalisation bounds — set in fit() or calibrate()
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._calibrated: bool = False

    # ------------------------------------------------------------------
    def raw_score(self, X: np.ndarray) -> np.ndarray:
        """Return inverted decision_function (higher = more anomalous)."""
        return -self.model.decision_function(X)

    def set_score_bounds(self, score_low: float, score_high: float) -> None:
        self._score_min = float(score_low)
        self._score_max = float(score_high)
        self._calibrated = True

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray) -> "IsolationForestDetector":
        """Fit the Isolation Forest on *X_train*.

        Parameters
        ----------
        X_train:
            2-D float array of training samples (ideally benign-only).
        """
        self.model.fit(X_train)

        # Isolation Forest's decision_function returns negative scores where
        # more negative → more anomalous.  We invert and normalise here.
        raw = self.raw_score(X_train)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        logger.info(
            "IsolationForest fitted on %d samples. Raw score range: [%.4f, %.4f]",
            len(X_train), self._score_min, self._score_max,
        )
        return self

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores normalised to [0, 1].

        Sklearn's ``decision_function`` returns higher values for normal
        samples.  We negate it so that higher values indicate anomalies,
        then min-max normalise to [0, 1] using bounds from the training set.

        Parameters
        ----------
        X:
            2-D float array of shape (n_samples, n_features).

        Returns
        -------
        1-D array of normalised anomaly scores in [0, 1].
        """
        raw = self.raw_score(X)
        return normalize_scores(raw, self._score_min, self._score_max)

    # ------------------------------------------------------------------
    def predict(
        self, X: np.ndarray, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Return binary predictions (0 = normal, 1 = anomaly).

        Parameters
        ----------
        X:
            2-D float array.
        threshold:
            Score threshold in [0, 1].  If ``None``, uses the model's
            built-in contamination-based threshold (sklearn predict ==  -1).

        Returns
        -------
        1-D int array.
        """
        if threshold is not None:
            return (self.score(X) >= threshold).astype(int)

        # sklearn returns 1 for normal, -1 for anomaly
        raw_pred = self.model.predict(X)
        return ((raw_pred == -1)).astype(int)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the model to *path* via joblib."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model":       self.model,
                "score_min":   self._score_min,
                "score_max":   self._score_max,
                "calibrated":  self._calibrated,
            },
            path,
        )
        logger.info("IsolationForestDetector saved to %s", path)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        """Load a previously saved detector from *path*."""
        payload = joblib.load(path)
        instance = cls()
        instance.model       = payload["model"]
        instance._score_min  = payload.get("score_min", 0.0)
        instance._score_max  = payload.get("score_max", 1.0)
        instance._calibrated = payload.get("calibrated", False)
        logger.info("IsolationForestDetector loaded from %s", path)
        return instance
