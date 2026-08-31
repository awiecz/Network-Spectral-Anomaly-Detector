"""
splits.py — Train / validation / test splitting for spectral window features.

Shared by main.py and tune.py so both pipelines use identical CICIDS day-split
logic (temporal benign holdout + random attack val/test partition).
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data_loader import get_binary_labels
from src.spectral_features import (
    apply_log_transform,
    bin_labels_to_timeseries,
    build_multiscale_spectral_matrix,
    build_spectral_feature_matrix,
    build_standard_timeseries,
    build_window_labels,
)

logger = logging.getLogger(__name__)


def build_subset_features(
    df_subset: pd.DataFrame,
    cfg: dict,
    dataset: str = "cicids2017",
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None, np.ndarray | None]:
    """Build spectral features and aligned per-window labels for one subset.

    Returns ``(X, timestamps, feature_names, y_binary, y_label)`` where the
    label arrays are ``None`` when no ``Label`` column is available.
    """
    del dataset  # reserved for future UNSW channel mapping
    spectral_cfg = cfg["spectral"]
    bin_size = spectral_cfg["bin_size_seconds"]
    ts_col = "Timestamp" if "Timestamp" in df_subset.columns else "Stime"

    timeseries_df = build_standard_timeseries(
        df_subset, timestamp_col=ts_col, bin_size_seconds=bin_size
    )

    log_channels = spectral_cfg.get("log_transform_channels") or []
    if log_channels:
        timeseries_df = apply_log_transform(timeseries_df, log_channels)

    input_series = spectral_cfg.get("input_series", list(timeseries_df.columns))
    input_series = [s for s in input_series if s in timeseries_df.columns]
    if not input_series:
        input_series = list(timeseries_df.columns)

    multi_scale = spectral_cfg.get("multi_scale", {}).get("enabled", False)
    delta_features = bool(spectral_cfg.get("delta_features", False))

    if multi_scale:
        X, timestamps, feature_names = build_multiscale_spectral_matrix(
            timeseries_df, cfg,
        )
        label_window_size = spectral_cfg["window_size"]
        label_overlap = spectral_cfg.get("overlap", 0)
    else:
        X, timestamps, feature_names = build_spectral_feature_matrix(
            timeseries_df,
            window_size=spectral_cfg["window_size"],
            overlap=spectral_cfg["overlap"],
            feature_cols=input_series,
            fs=1.0 / bin_size,
            window_fn=spectral_cfg["window_function"],
            rolloff_pcts=spectral_cfg["rolloff_percentages"],
            delta_features=delta_features,
        )
        label_window_size = spectral_cfg["window_size"]
        label_overlap = spectral_cfg["overlap"]

    y_binary = None
    y_label = None
    if "Label" in df_subset.columns:
        biny = get_binary_labels(df_subset)
        bin_label_df = bin_labels_to_timeseries(
            df_subset,
            binary_labels=biny,
            timestamp_col=ts_col,
            attack_labels=df_subset["Label"].astype(str).to_numpy(),
            bin_size_seconds=bin_size,
            reference_index=timeseries_df.index,
        )
        rule = cfg.get("evaluation", {}).get("window_label_rule", "any")
        y_binary, y_label, _ = build_window_labels(
            bin_label_df,
            window_size=label_window_size,
            overlap=label_overlap,
            rule=rule,
        )

    return X, timestamps, feature_names, y_binary, y_label


def stack_eval_split(
    X_benign: np.ndarray,
    benign_ts: np.ndarray,
    lbl_benign: np.ndarray | None,
    X_attack: np.ndarray,
    attack_ts: np.ndarray,
    y_attack: np.ndarray | None,
    lbl_attack: np.ndarray | None,
    benign_idx: np.ndarray,
    attack_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine benign and attack windows into one labelled evaluation split."""
    Xb = X_benign[benign_idx]
    ts_b = benign_ts[benign_idx]
    n_neg = len(Xb)
    Xa = X_attack[attack_idx]
    ts_a = attack_ts[attack_idx]
    ya = (
        y_attack[attack_idx]
        if y_attack is not None
        else np.ones(len(attack_idx), dtype=int)
    )
    X = np.vstack([Xb, Xa])
    y = np.concatenate([np.zeros(n_neg, dtype=int), ya])
    ts = np.concatenate([ts_b, ts_a])
    if lbl_benign is not None and lbl_attack is not None:
        labels = np.concatenate([
            lbl_benign[benign_idx],
            lbl_attack[attack_idx],
        ])
    else:
        labels = np.concatenate([
            np.array(["BENIGN"] * n_neg, dtype=object),
            np.array(["ATTACK"] * len(attack_idx), dtype=object),
        ])
    return X, y, labels, ts


def make_val_test_splits(
    X_benign_holdout: np.ndarray,
    benign_ts: np.ndarray,
    lbl_benign: np.ndarray | None,
    X_attack: np.ndarray,
    attack_ts: np.ndarray,
    y_attack: np.ndarray | None,
    lbl_attack: np.ndarray | None,
    rng: np.random.RandomState,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Split benign holdout and attack windows 50/50 into val / test."""
    b_hold_idx = np.arange(len(X_benign_holdout))
    b_val_idx, b_test_idx = np.split(
        b_hold_idx,
        [max(1, len(b_hold_idx) // 2)],
    )

    n_attack = len(X_attack)
    if n_attack == 0:
        a_val_idx = np.array([], dtype=int)
        a_test_idx = np.array([], dtype=int)
    else:
        a_perm = rng.permutation(n_attack)
        a_half = max(1, n_attack // 2)
        a_val_idx, a_test_idx = a_perm[:a_half], a_perm[a_half:]

    X_val, y_val, lab_val, val_timestamps = stack_eval_split(
        X_benign_holdout, benign_ts, lbl_benign,
        X_attack, attack_ts, y_attack, lbl_attack,
        b_val_idx, a_val_idx,
    )
    X_test, y_test, y_label_test, test_timestamps = stack_eval_split(
        X_benign_holdout, benign_ts, lbl_benign,
        X_attack, attack_ts, y_attack, lbl_attack,
        b_test_idx, a_test_idx,
    )
    return X_val, y_val, lab_val, val_timestamps, X_test, y_test, y_label_test, test_timestamps


def build_cicids_day_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    rng: np.random.RandomState,
    dataset: str = "cicids2017",
) -> dict:
    """Build train / val / test splits for the CICIDS Monday + eval-days protocol.

    Benign train-day windows are filtered to genuinely-benign only, then split
    temporally: earliest ``(1 - benign_test_frac)`` for training, the tail for
    holdout negatives.  Attack-day windows are split 50/50 into val / test.

    Returns a dict with keys:
        X_train, X_val, y_val, lab_val, X_test, y_test, lab_test, feat_names
    """
    benign_test_frac = float(cfg.get("evaluation", {}).get("benign_test_frac", 0.30))

    parallel_days = os.environ.get("NSAD_PARALLEL_DAYS", "1") != "0"
    if parallel_days:
        results = Parallel(n_jobs=2, prefer="threads")(
            delayed(build_subset_features)(df, cfg, dataset=dataset)
            for df in (train_df, test_df)
        )
        X_benign, benign_ts, feature_names, y_benign, lbl_benign = results[0]
        X_attack, attack_ts, _, y_attack, lbl_attack = results[1]
    else:
        X_benign, benign_ts, feature_names, y_benign, lbl_benign = build_subset_features(
            train_df, cfg, dataset=dataset
        )
        X_attack, attack_ts, _, y_attack, lbl_attack = build_subset_features(
            test_df, cfg, dataset=dataset
        )

    if y_benign is not None:
        keep = y_benign == 0
        X_benign = X_benign[keep]
        benign_ts = benign_ts[keep]
        lbl_benign = lbl_benign[keep]

    n_benign = len(X_benign)
    n_holdout = max(2, int(benign_test_frac * n_benign))
    X_train = X_benign[: n_benign - n_holdout]
    X_benign_holdout = X_benign[n_benign - n_holdout:]
    ts_benign_holdout = benign_ts[n_benign - n_holdout:]
    lbl_benign_holdout = lbl_benign[n_benign - n_holdout:]

    X_val, y_val, lab_val, val_timestamps, X_test, y_test, lab_test, test_timestamps = make_val_test_splits(
        X_benign_holdout, ts_benign_holdout, lbl_benign_holdout,
        X_attack, attack_ts, y_attack, lbl_attack,
        rng,
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_val": y_val,
        "lab_val": lab_val,
        "val_timestamps": val_timestamps,
        "X_test": X_test,
        "y_test": y_test,
        "lab_test": lab_test,
        "test_timestamps": test_timestamps,
        "feat_names": feature_names,
    }


def build_disjoint_window_splits(
    X_all: np.ndarray,
    all_ts: np.ndarray,
    y_all: np.ndarray,
    y_label_all: np.ndarray,
    benign_holdout_frac: float,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Benign-only train + disjoint benign holdout / attack val-test (sample path)."""
    benign_idx = np.flatnonzero(y_all == 0)
    attack_idx = np.flatnonzero(y_all == 1)

    if len(benign_idx) == 0:
        logger.error("No benign windows available for training.")
        sys.exit(1)

    time_order = benign_idx[np.argsort(all_ts[benign_idx])]
    n_holdout = max(2, int(benign_holdout_frac * len(time_order)))
    n_train = len(time_order) - n_holdout
    if n_train < 1:
        n_train = max(1, len(time_order) - 2)
        n_holdout = len(time_order) - n_train

    train_idx = time_order[:n_train]
    holdout_benign_idx = time_order[n_train:]

    X_train = X_all[train_idx]
    X_benign_holdout = X_all[holdout_benign_idx]
    ts_benign_holdout = all_ts[holdout_benign_idx]
    lbl_benign_holdout = y_label_all[holdout_benign_idx]

    if len(attack_idx) > 0:
        X_attack = X_all[attack_idx]
        attack_ts = all_ts[attack_idx]
        y_attack = y_all[attack_idx]
        lbl_attack = y_label_all[attack_idx]
    else:
        n_feat = X_all.shape[1]
        X_attack = np.empty((0, n_feat), dtype=X_all.dtype)
        attack_ts = all_ts[:0]
        y_attack = y_all[:0]
        lbl_attack = y_label_all[:0]

    eval_idx = np.concatenate([holdout_benign_idx, attack_idx])
    if len(np.intersect1d(train_idx, eval_idx)) > 0:
        logger.error("Train and evaluation windows overlap — split logic bug.")
        sys.exit(1)

    logger.info(
        "Disjoint split — train: %d benign | eval holdout: %d benign + %d attack",
        len(train_idx), len(holdout_benign_idx), len(attack_idx),
    )

    if len(holdout_benign_idx) < 2 and len(attack_idx) == 0:
        logger.error(
            "Not enough benign holdout windows for validation and test. "
            "Use more data or lower evaluation.benign_test_frac."
        )
        sys.exit(1)

    X_val, y_val, _, val_ts, X_test, y_test, y_label_test, test_timestamps = make_val_test_splits(
        X_benign_holdout, ts_benign_holdout, lbl_benign_holdout,
        X_attack, attack_ts, y_attack, lbl_attack,
        rng,
    )
    del val_ts  # sample path does not expose val timestamps
    return X_train, X_val, y_val, X_test, y_test, y_label_test, test_timestamps
