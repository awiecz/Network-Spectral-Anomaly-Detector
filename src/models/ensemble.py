"""
ensemble.py — Weighted ensemble of anomaly detectors.

Combines normalised scores from VAE, Isolation Forest, and optionally
ECOD/DeepSVDD using a configurable weighted sum.  Threshold selection
supports three strategies:
  - f1_optimal:   maximises F1 score on a labelled validation set
  - youden_j:     maximises Youden's J statistic (sensitivity + specificity - 1)
  - percentile:   uses a fixed percentile of the score distribution
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

from src.evaluation import find_optimal_threshold, normalize_threshold_method

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """Weighted ensemble of heterogeneous anomaly detectors.

    Each model in *models* must implement:
      - ``fit(X_train: np.ndarray) -> self``
      - ``score(X: np.ndarray) -> np.ndarray``  (scores in [0, 1])

    Parameters
    ----------
    models:
        Mapping of model name → detector instance.
    weights:
        Mapping of model name → scalar weight.  Need not sum to 1
        (normalisation is applied automatically).
    """

    def __init__(
        self,
        models: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.models = models

        if weights is None:
            # Equal weights if not specified
            weights = {name: 1.0 for name in models}
        self.weights = weights

        # Normalise weights to sum to 1
        total_w = sum(self.weights.get(name, 0.0) for name in self.models)
        if total_w <= 0:
            raise ValueError("Ensemble weights must sum to a positive value.")
        self._normalised_weights: Dict[str, float] = {
            name: self.weights.get(name, 0.0) / total_w
            for name in self.models
        }

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray) -> "EnsembleDetector":
        """Fit all component models on *X_train*.

        Parameters
        ----------
        X_train:
            2-D float array of training samples (benign-only recommended).
        """
        for name, model in self.models.items():
            logger.info("Fitting %s …", name)
            model.fit(X_train)
        return self

    # ------------------------------------------------------------------
    def score_models_separately(
        self, X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Return each model's normalised anomaly scores individually.

        Parameters
        ----------
        X:
            2-D float array of shape (n_samples, n_features).

        Returns
        -------
        Dict mapping model name → 1-D score array.
        """
        return {name: model.score(X) for name, model in self.models.items()}

    # ------------------------------------------------------------------
    def score_from(self, individual: Dict[str, np.ndarray]) -> np.ndarray:
        """Blend precomputed per-model scores with ensemble weights."""
        n = len(next(iter(individual.values())))
        combined = np.zeros(n, dtype=float)
        for name, scores in individual.items():
            w = self._normalised_weights.get(name, 0.0)
            combined += w * scores
        return np.clip(combined, 0.0, 1.0)

    # ------------------------------------------------------------------
    def score(
        self,
        X: np.ndarray,
        individual: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Return the weighted-sum ensemble anomaly score.

        Parameters
        ----------
        X:
            2-D float array of shape (n_samples, n_features).
        individual:
            Optional precomputed per-model scores from
            :meth:`score_models_separately`.  When provided, component
            models are not scored again.

        Returns
        -------
        1-D array of ensemble scores (not necessarily in [0, 1] before
        clipping, but practically so because each component is normalised).
        """
        if individual is None:
            individual = self.score_models_separately(X)
        return self.score_from(individual)

    # ------------------------------------------------------------------
    def predict(
        self, X: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """Return binary predictions (1 = anomaly) for a given *threshold*.

        Parameters
        ----------
        X:
            2-D float array.
        threshold:
            Decision boundary in [0, 1].

        Returns
        -------
        1-D int array.
        """
        return (self.score(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    @staticmethod
    def find_optimal_threshold(
        scores: np.ndarray,
        y_true: np.ndarray,
        method: str = "f1_optimal",
        percentile: float = 95.0,
    ) -> Tuple[float, float]:
        """Find the decision threshold that optimises a given criterion.

        Delegates to :func:`src.evaluation.find_optimal_threshold`.
        """
        internal = normalize_threshold_method(method)
        if internal == "percentile_95":
            return find_optimal_threshold(y_true, scores, method="percentile_95")
        return find_optimal_threshold(y_true, scores, method=internal)

    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        """Persist the ensemble to *directory*.

        Each component model is saved individually in a sub-directory.
        A ``meta.pkl`` file stores the weight configuration.
        """
        os.makedirs(directory, exist_ok=True)
        joblib.dump(
            {
                "weights":              self.weights,
                "normalised_weights":   self._normalised_weights,
                "model_names":          list(self.models.keys()),
            },
            os.path.join(directory, "meta.pkl"),
        )
        for name, model in self.models.items():
            model_path = os.path.join(directory, f"{name}.pkl")
            if hasattr(model, "save"):
                model.save(model_path)
            else:
                joblib.dump(model, model_path)
        logger.info("Ensemble saved to %s", directory)

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        directory: str,
        model_loaders: Optional[Dict[str, Any]] = None,
    ) -> "EnsembleDetector":
        """Load a previously saved ensemble from *directory*.

        Parameters
        ----------
        directory:
            Path saved by :meth:`save`.
        model_loaders:
            Optional mapping of model name → callable that loads a model
            from a path.  If not provided, ``joblib.load`` is used for all
            models.
        """
        meta = joblib.load(os.path.join(directory, "meta.pkl"))
        model_loaders = model_loaders or {}

        models: Dict[str, Any] = {}
        for name in meta["model_names"]:
            model_path = os.path.join(directory, f"{name}.pkl")
            if name in model_loaders:
                models[name] = model_loaders[name](model_path)
            else:
                models[name] = joblib.load(model_path)

        instance = cls(models=models, weights=meta["weights"])
        instance._normalised_weights = meta["normalised_weights"]
        logger.info("Ensemble loaded from %s", directory)
        return instance
