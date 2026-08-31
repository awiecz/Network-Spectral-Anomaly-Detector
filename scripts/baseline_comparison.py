"""
baseline_comparison.py — Spectral vs raw CICFlowMeter window-feature baseline.

Trains ECOD and Isolation Forest on Monday benign windows only, selects
thresholds on validation, and reports test metrics for four configurations:

  spectral_ecod, spectral_if, raw_ecod, raw_if

Output: results/metrics/baseline_comparison.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import create_sample_dataset, load_cicids2017
from src.evaluation import compute_metrics, normalize_threshold_method
from src.feature_cache import cache_path, load_splits_cache, save_splits_cache
from src.models import ECODDetector, IsolationForestDetector
from src.models.flow_ecod import select_flow_feature_columns
from src.models.scoring import calibrate_detector_scores
from src.preprocessor import DataPreprocessor
from src.spectral_features import (
    bin_labels_to_timeseries,
    build_standard_timeseries,
    build_window_labels,
)
from src.splits import build_cicids_day_splits, build_subset_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("baseline")


def build_raw_window_features(
    df_subset: pd.DataFrame,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None, np.ndarray | None]:
    """Aggregate per-flow CICFlowMeter stats into window vectors (mean + max)."""
    spectral_cfg = cfg["spectral"]
    window_size = spectral_cfg["window_size"]
    overlap = spectral_cfg.get("overlap", 0)
    bin_size = spectral_cfg["bin_size_seconds"]
    ts_col = "Timestamp" if "Timestamp" in df_subset.columns else "Stime"

    timeseries_df = build_standard_timeseries(
        df_subset, timestamp_col=ts_col, bin_size_seconds=bin_size,
    )
    feature_cols = select_flow_feature_columns(df_subset)
    if not feature_cols:
        raise ValueError("No numeric flow feature columns found for raw baseline.")

    flow_ts = pd.to_datetime(df_subset[ts_col], errors="coerce")
    flow_mat = (
        df_subset[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    step = window_size - overlap
    t_bins = len(timeseries_df)
    if t_bins < window_size:
        n_feat = len(feature_cols) * 2
        return (
            np.empty((0, n_feat)),
            np.array([]),
            [f"{c}_mean" for c in feature_cols] + [f"{c}_max" for c in feature_cols],
            None,
            None,
        )

    n_windows = (t_bins - window_size) // step + 1
    n_feat = len(feature_cols) * 2
    X = np.zeros((n_windows, n_feat), dtype=float)
    starts = np.arange(0, n_windows * step, step)
    window_timestamps = timeseries_df.index[starts]
    bin_index = timeseries_df.index

    for i, start in enumerate(starts):
        end = start + window_size
        t_start = bin_index[start]
        t_end = bin_index[min(end - 1, t_bins - 1)]
        mask = (flow_ts >= t_start) & (flow_ts <= t_end)
        if mask.any():
            chunk = flow_mat[mask.to_numpy()]
            X[i, : len(feature_cols)] = chunk.mean(axis=0)
            X[i, len(feature_cols) :] = chunk.max(axis=0)

    feat_names = [f"{c}_mean" for c in feature_cols] + [f"{c}_max" for c in feature_cols]

    y_binary = None
    y_label = None
    if "Label" in df_subset.columns:
        from src.data_loader import get_binary_labels

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
            window_size=window_size,
            overlap=overlap,
            rule=rule,
        )

    return X, window_timestamps, feat_names, y_binary, y_label


def build_raw_day_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    rng: np.random.RandomState,
) -> dict:
    """Same split logic as build_cicids_day_splits but with raw window features."""
    import os
    from joblib import Parallel, delayed

    from src.splits import make_val_test_splits

    benign_test_frac = float(cfg.get("evaluation", {}).get("benign_test_frac", 0.30))
    parallel_days = os.environ.get("NSAD_PARALLEL_DAYS", "1") != "0"
    builder = build_raw_window_features

    if parallel_days:
        results = Parallel(n_jobs=2, prefer="threads")(
            delayed(builder)(df, cfg) for df in (train_df, test_df)
        )
        X_benign, benign_ts, feature_names, y_benign, lbl_benign = results[0]
        X_attack, attack_ts, _, y_attack, lbl_attack = results[1]
    else:
        X_benign, benign_ts, feature_names, y_benign, lbl_benign = builder(train_df, cfg)
        X_attack, attack_ts, _, y_attack, lbl_attack = builder(test_df, cfg)

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

    X_val, y_val, lab_val, val_timestamps, X_test, y_test, lab_test, test_timestamps = (
        make_val_test_splits(
            X_benign_holdout,
            ts_benign_holdout,
            lbl_benign_holdout,
            X_attack,
            attack_ts,
            y_attack,
            lbl_attack,
            rng,
        )
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


def _load_data(args: argparse.Namespace, cfg: dict) -> pd.DataFrame:
    if args.use_sample:
        sample_path = create_sample_dataset(
            n=5000, output_dir=cfg["data"]["sample_path"],
        )
        df = pd.read_csv(sample_path, low_memory=False)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(
                df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce",
            )
        df["day"] = cfg["data"].get("train_day", "Monday")
        return df

    train_day = cfg["data"].get("train_day", "Monday")
    eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
    processed_dir = cfg["data"].get("processed_path", "data/processed")
    return load_cicids2017(
        args.data_dir,
        days=[train_day, *eval_days],
        processed_dir=processed_dir,
    )


def _get_spectral_splits(
    cfg: dict,
    df: pd.DataFrame,
    use_cache: bool,
) -> dict:
    train_day = cfg["data"].get("train_day", "Monday")
    eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
    processed_dir = cfg["data"].get("processed_path", "data/processed")

    if use_cache:
        path = cache_path(cfg, train_day, eval_days, processed_dir)
        cached = load_splits_cache(path)
        if cached is not None:
            logger.info("Loaded spectral splits from cache: %s", path)
            return {k: cached[k] for k in cached}

    train_df = df[df["day"] == train_day].reset_index(drop=True)
    test_df = df[df["day"].isin(eval_days)].reset_index(drop=True)
    splits = build_cicids_day_splits(
        train_df, test_df, cfg, np.random.RandomState(42),
    )

    if use_cache:
        path = cache_path(cfg, train_day, eval_days, processed_dir)
        save_splits_cache(path, splits)
        logger.info("Wrote spectral splits cache: %s", path)

    return splits


def _run_detector(
    name: str,
    feature_type: str,
    model_type: str,
    splits: dict,
    cfg: dict,
    threshold_method: str,
) -> dict:
    pre = DataPreprocessor()
    X_tr = pre.fit_transform(splits["X_train"], feature_names=list(splits["feat_names"]))
    X_va = pre.transform(splits["X_val"])
    X_te = pre.transform(splits["X_test"])
    y_va, y_te = splits["y_val"], splits["y_test"]

    if model_type == "ecod":
        contamination = float(cfg.get("models", {}).get("ecod", {}).get("contamination", 0.1))
        det = ECODDetector(contamination=contamination)
    else:
        if_cfg = cfg.get("models", {}).get("isolation_forest", {})
        det = IsolationForestDetector(
            n_estimators=int(if_cfg.get("n_estimators", 200)),
            max_samples=if_cfg.get("max_samples", 256),
            contamination=if_cfg.get("contamination", "auto"),
            max_features=if_cfg.get("max_features", 1.0),
        )

    det.fit(X_tr)
    calibrate_detector_scores(det, X_va, y_va)
    va_scores = det.score(X_va)
    te_scores = det.score(X_te)

    from src.models.ensemble import EnsembleDetector

    thr, _ = EnsembleDetector.find_optimal_threshold(va_scores, y_va, method=threshold_method)
    metrics = compute_metrics(y_te, te_scores, threshold=thr)
    metrics["config"] = name
    metrics["feature_type"] = feature_type
    metrics["model"] = model_type
    metrics["threshold_method"] = normalize_threshold_method(threshold_method)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Spectral vs raw feature baseline comparison")
    ap.add_argument("--data-dir", default="data/raw/cicids2017")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--output", default="results/metrics/baseline_comparison.csv")
    ap.add_argument("--use-sample", action="store_true")
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    threshold_method = cfg.get("evaluation", {}).get("default_threshold_method", "f1_optimal")

    df = _load_data(args, cfg)
    if args.use_sample:
        logger.warning(
            "Sample mode: metrics are for smoke testing only. "
            "Run without --use-sample on full CICIDS2017 for published numbers."
        )
        from src.splits import build_disjoint_window_splits

        X, ts, feat_names, y_bin, y_lbl = build_subset_features(df, cfg)
        rng = np.random.RandomState(42)
        holdout = float(cfg.get("evaluation", {}).get("benign_test_frac", 0.30))
        X_tr, X_va, y_va, X_te, y_te, _, _ = build_disjoint_window_splits(
            X, ts, y_bin, y_lbl, holdout, rng,
        )
        spectral_splits = {
            "X_train": X_tr,
            "X_val": X_va,
            "y_val": y_va,
            "X_test": X_te,
            "y_test": y_te,
            "feat_names": feat_names,
        }
        X_raw, _, raw_names, _, _ = build_raw_window_features(df, cfg)
        X_tr_r, X_va_r, y_va_r, X_te_r, y_te_r, _, _ = build_disjoint_window_splits(
            X_raw, ts, y_bin, y_lbl, holdout, rng,
        )
        raw_splits = {
            "X_train": X_tr_r,
            "X_val": X_va_r,
            "y_val": y_va_r,
            "X_test": X_te_r,
            "y_test": y_te_r,
            "feat_names": raw_names,
        }
    else:
        train_day = cfg["data"].get("train_day", "Monday")
        eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
        train_df = df[df["day"] == train_day].reset_index(drop=True)
        test_df = df[df["day"].isin(eval_days)].reset_index(drop=True)
        spectral_splits = _get_spectral_splits(cfg, df, args.use_cache)
        raw_splits = build_raw_day_splits(
            train_df, test_df, cfg, np.random.RandomState(42),
        )

    configs = [
        ("spectral_ecod", "spectral", "ecod", spectral_splits),
        ("spectral_if", "spectral", "isolation_forest", spectral_splits),
        ("raw_ecod", "raw_flow", "ecod", raw_splits),
        ("raw_if", "raw_flow", "isolation_forest", raw_splits),
    ]

    rows = []
    for name, feat_type, model_type, splits in configs:
        logger.info("Evaluating %s …", name)
        rows.append(
            _run_detector(name, feat_type, model_type, splits, cfg, threshold_method),
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    col_order = [
        "config", "feature_type", "model", "auroc", "auprc", "f1",
        "precision", "recall", "fpr", "fnr", "threshold", "threshold_method",
    ]
    pd.DataFrame(rows)[col_order].to_csv(out, index=False)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
