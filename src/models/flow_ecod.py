"""
flow_ecod.py — Flow-level ECOD for auxiliary stealthy-attack detection.

Scores individual CICFlowMeter flows on raw numeric features, then aggregates
to window-level scores for fusion with the spectral ensemble.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD

from .scoring import fit_percentile_bounds, normalize_scores

logger = logging.getLogger(__name__)

# Stable numeric CICFlowMeter columns used for flow-level scoring.
DEFAULT_FLOW_FEATURES: List[str] = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "ACK Flag Count",
]


def select_flow_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric flow columns present in *df*."""
    cols = [c for c in DEFAULT_FLOW_FEATURES if c in df.columns]
    if cols:
        return cols
    exclude = {
        "Flow ID", "Source IP", "Destination IP", "Timestamp", "Label",
        "label", "day", "Src IP", "Dst IP", "Src Port", "Dst Port",
    }
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ][:20]


def _clean_flow_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    X = df[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.to_numpy(dtype=float)


class FlowECODDetector:
    """ECOD on raw per-flow features with window aggregation."""

    def __init__(self, contamination: float = 0.1) -> None:
        self.contamination = contamination
        self.model = ECOD(contamination=contamination)
        self.feature_cols_: Optional[List[str]] = None
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._calibrated: bool = False
        self.aggregation: str = "max"

    def raw_score(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.decision_function(X), dtype=float)

    def set_score_bounds(self, score_low: float, score_high: float) -> None:
        self._score_min = float(score_low)
        self._score_max = float(score_high)
        self._calibrated = True

    def fit(self, df_benign: pd.DataFrame) -> "FlowECODDetector":
        self.feature_cols_ = select_flow_feature_columns(df_benign)
        X = _clean_flow_matrix(df_benign, self.feature_cols_)
        self.model.fit(X)
        raw = np.asarray(self.model.decision_scores_, dtype=float)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        logger.info(
            "FlowECOD fitted on %d flows, %d features",
            len(X), len(self.feature_cols_),
        )
        return self

    def score_flows(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_cols_ is None:
            raise RuntimeError("FlowECODDetector must be fit before scoring.")
        X = _clean_flow_matrix(df, self.feature_cols_)
        return normalize_scores(self.raw_score(X), self._score_min, self._score_max)

    def aggregate_to_windows(
        self,
        df: pd.DataFrame,
        window_timestamps: np.ndarray,
        window_size: int,
        overlap: int,
        timestamp_col: str = "Timestamp",
        aggregation: str = "max",
    ) -> np.ndarray:
        """Map flow scores to spectral window timestamps."""
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        flow_scores = self.score_flows(df)
        order = np.argsort(ts.values, kind="mergesort")
        ts_sorted = ts.iloc[order].values
        scores_sorted = flow_scores[order]

        step = window_size - overlap
        out = np.zeros(len(window_timestamps), dtype=float)
        for i, w_start in enumerate(window_timestamps):
            w_start = pd.Timestamp(w_start)
            w_end = w_start + pd.Timedelta(seconds=window_size)
            mask = (ts_sorted >= w_start.to_datetime64()) & (ts_sorted < w_end.to_datetime64())
            chunk = scores_sorted[mask]
            if len(chunk) == 0:
                out[i] = 0.0
            elif aggregation == "mean":
                out[i] = float(np.mean(chunk))
            else:
                out[i] = float(np.max(chunk))
        return out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "contamination": self.contamination,
                "feature_cols": self.feature_cols_,
                "score_min": self._score_min,
                "score_max": self._score_max,
                "calibrated": self._calibrated,
                "aggregation": self.aggregation,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "FlowECODDetector":
        payload = joblib.load(path)
        inst = cls(contamination=payload.get("contamination", 0.1))
        inst.model = payload["model"]
        inst.feature_cols_ = payload.get("feature_cols")
        inst._score_min = payload.get("score_min", 0.0)
        inst._score_max = payload.get("score_max", 1.0)
        inst._calibrated = payload.get("calibrated", False)
        inst.aggregation = payload.get("aggregation", "max")
        return inst


def calibrate_flow_ecod(detector: FlowECODDetector, df_val: pd.DataFrame, y_val_windows: np.ndarray,
                        window_timestamps: np.ndarray, window_size: int, overlap: int,
                        timestamp_col: str = "Timestamp") -> None:
    """Calibrate flow ECOD using benign validation windows."""
    benign_mask = np.asarray(y_val_windows, dtype=int) == 0
    if not np.any(benign_mask):
        return
    win_scores = detector.aggregate_to_windows(
        df_val, window_timestamps, window_size, overlap, timestamp_col,
    )
    p_low, p_high = fit_percentile_bounds(win_scores, benign_mask)
    detector.set_score_bounds(p_low, p_high)
