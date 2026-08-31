"""
preprocessor.py — Feature selection, scaling, and dataset splitting utilities.

Uses RobustScaler (median/IQR) instead of StandardScaler because IDS flow
features have extremely heavy-tailed distributions (a single DDoS burst can
add millions to Flow Bytes/s), and RobustScaler is insensitive to such
outliers.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column lists
# ---------------------------------------------------------------------------

# Core numeric features shared across most CICIDS2017 files.
_CICIDS_FEATURE_COLS: List[str] = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

_UNSW_FEATURE_COLS: List[str] = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "src_ttl",
    "dst_ttl",
    "src_loss",
    "dst_loss",
    "src_load",
    "dst_load",
    "src_pkts",
    "dst_pkts",
    "src_win",
    "dst_win",
    "tcprtt",
    "synack",
    "ackdat",
    "smeansz",
    "dmeansz",
    "trans_depth",
    "res_bdy_len",
    "ct_state_ttl",
    "ct_flw_http_mthd",
    "ct_ftp_cmd",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
]


def get_feature_columns(
    df: pd.DataFrame,
    dataset: str = "cicids2017",
) -> List[str]:
    """Return the list of numeric feature columns for *dataset*.

    Only columns that are actually present in *df* are returned, so this
    function is safe to call on partial or sample DataFrames.

    Parameters
    ----------
    df:
        The loaded DataFrame.
    dataset:
        ``'cicids2017'`` or ``'unsw_nb15'``.

    Returns
    -------
    List of column name strings.
    """
    candidates = _CICIDS_FEATURE_COLS if dataset == "cicids2017" else _UNSW_FEATURE_COLS
    present = [c for c in candidates if c in df.columns]
    if not present:
        # Fall back: use all numeric columns that aren't obvious meta-columns
        exclude = {"Label", "label", "attack_category", "day", "Timestamp", "Flow ID",
                   "Source IP", "Destination IP", "src_ip", "dst_ip"}
        present = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
    return present


# ---------------------------------------------------------------------------
# DataPreprocessor class
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """Fit-transform wrapper around :class:`sklearn.preprocessing.RobustScaler`.

    Attributes
    ----------
    scaler:
        The underlying :class:`RobustScaler` instance.
    feature_names_:
        Column names seen during :meth:`fit`, set after the first call.
    """

    def __init__(self) -> None:
        self.scaler = RobustScaler()
        self.feature_names_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None) -> "DataPreprocessor":
        """Fit the RobustScaler on *X_train*.

        Parameters
        ----------
        X_train:
            2-D array of shape (n_samples, n_features).
        feature_names:
            Optional list of column names for :meth:`get_feature_names`.
        """
        self.scaler.fit(X_train)
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted scaler to *X*.

        Returns
        -------
        np.ndarray of the same shape as *X*.
        """
        return self.scaler.transform(X)

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fit on *X_train* and return the scaled array."""
        self.fit(X_train, feature_names=feature_names)
        return self.transform(X_train)

    # ------------------------------------------------------------------
    def get_feature_names(self) -> List[str]:
        """Return feature names set during :meth:`fit`.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self.feature_names_ is None:
            raise RuntimeError("Call fit() with feature_names before get_feature_names().")
        return self.feature_names_

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the fitted scaler and feature names to *path* via joblib."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names_}, path)
        logger.info("Preprocessor saved to %s", path)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """Load a previously saved :class:`DataPreprocessor` from *path*."""
        payload = joblib.load(path)
        instance = cls()
        instance.scaler = payload["scaler"]
        instance.feature_names_ = payload.get("feature_names")
        logger.info("Preprocessor loaded from %s", path)
        return instance


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def get_train_test_split_cicids(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a CICIDS2017 DataFrame into train and test sets.

    Training set: **Monday** rows only (all benign — used for unsupervised
    model fitting without contamination).
    Test set: **Tuesday–Friday** rows (mixed benign + attack).

    Parameters
    ----------
    df:
        A DataFrame loaded by :func:`src.data_loader.load_cicids2017`.

    Returns
    -------
    (train_df, test_df) where train_df contains only Monday rows.
    """
    if "day" not in df.columns:
        raise ValueError("'day' column missing. Load data with load_cicids2017().")

    train_df = df[df["day"] == "Monday"].reset_index(drop=True)
    test_df  = df[df["day"] != "Monday"].reset_index(drop=True)

    logger.info(
        "Train (Monday): %d rows | Test (Tue–Fri): %d rows", len(train_df), len(test_df)
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Time-feature engineering
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, timestamp_col: str = "Timestamp") -> pd.DataFrame:
    """Extract ``hour`` and ``day_of_week`` from a parsed Timestamp column.

    The new columns are appended in-place and the original Timestamp column
    is preserved.

    Parameters
    ----------
    df:
        DataFrame with a ``Timestamp`` column of :class:`pandas.Timestamp` type.
    timestamp_col:
        Name of the datetime column.

    Returns
    -------
    The modified DataFrame (same object — no copy made).
    """
    if timestamp_col not in df.columns:
        logger.warning("Column '%s' not found; skipping time-feature extraction.", timestamp_col)
        return df

    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    df["hour"]        = ts.dt.hour.astype("Int16")
    df["day_of_week"] = ts.dt.dayofweek.astype("Int16")   # 0=Mon … 6=Sun
    return df
