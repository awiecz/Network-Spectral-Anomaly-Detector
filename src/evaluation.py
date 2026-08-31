"""
evaluation.py — Metrics computation and model comparison utilities.

All evaluation functions accept raw anomaly scores (not binary predictions)
to allow threshold-agnostic metrics (AUROC, AUPRC) alongside threshold-
dependent metrics (F1, precision, recall, etc.).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def normalize_threshold_method(method: str) -> str:
    """Map CLI/config threshold names to internal method ids."""
    if method in ("f1_optimal", "f1"):
        return "f1"
    if method in ("percentile", "percentile_95"):
        return "percentile_95"
    return method


def parse_fpr_target(method: str) -> Optional[float]:
    """Return max FPR from a method name like ``fpr_0.05``, or None."""
    if not method.startswith("fpr_"):
        return None
    try:
        return float(method.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def find_threshold_at_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    max_fpr: float,
) -> Tuple[float, float]:
    """Pick the highest threshold achieving FPR <= *max_fpr* on validation.

    Returns
    -------
    (threshold, recall_at_threshold)
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        logger.warning(
            "find_threshold_at_fpr: only one class in y_true; "
            "using 95th-percentile score fallback."
        )
        t = float(np.percentile(scores, 95)) if len(scores) else 0.5
        return t, 0.0

    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, scores)
    best_t = float(np.percentile(scores, 95))
    best_recall = 0.0
    for fpr, recall, t in zip(fpr_arr, tpr_arr, thresholds):
        if not np.isfinite(t):
            continue
        if fpr <= max_fpr:
            best_t = float(t)
            best_recall = float(recall)
    return best_t, best_recall


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute a comprehensive set of detection metrics.

    Parameters
    ----------
    y_true:
        1-D binary ground-truth array (0 = normal, 1 = anomaly/attack).
    scores:
        1-D anomaly score array (higher = more anomalous).
    threshold:
        Decision threshold.  If ``None``, the F1-optimal threshold is
        used automatically.

    Returns
    -------
    Dictionary with keys:
        auroc, auprc, f1, precision, recall, fpr, fnr, tp, tn, fp, fn,
        threshold.
    """
    y_true  = np.asarray(y_true, dtype=int)
    scores  = np.asarray(scores, dtype=float)

    # Threshold-agnostic metrics
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(y_true, scores))
    except ValueError:
        auprc = float("nan")

    # Determine threshold
    if threshold is None:
        threshold, _ = find_optimal_threshold(y_true, scores, method="f1")

    y_pred = (scores >= threshold).astype(int)

    f1_val    = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    return {
        "auroc":     auroc,
        "auprc":     auprc,
        "f1":        f1_val,
        "precision": precision,
        "recall":    recall,
        "fpr":       float(fpr),
        "fnr":       float(fnr),
        "tp":        int(tp),
        "tn":        int(tn),
        "fp":        int(fp),
        "fn":        int(fn),
        "threshold": float(threshold),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    method: str = "f1",
) -> Tuple[float, float]:
    """Find the score threshold that optimises *method*.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    scores:
        Anomaly scores.
    method:
        ``'f1'`` / ``'f1_optimal'`` — maximises F1 score.
        ``'youden_j'`` — maximises Youden's J (TPR - FPR).
        ``'percentile_95'`` / ``'percentile'`` — 95th percentile of scores.
        ``'fpr_0.05'`` etc. — highest threshold with FPR <= target.

    Returns
    -------
    (threshold, metric_value)
    """
    method = normalize_threshold_method(method)
    fpr_target = parse_fpr_target(method)
    if fpr_target is not None:
        return find_threshold_at_fpr(y_true, scores, fpr_target)

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        logger.warning(
            "find_optimal_threshold: only one class in y_true; "
            "using 95th-percentile score fallback."
        )
        t = float(np.percentile(scores, 95)) if len(scores) else 0.5
        return t, 0.0

    if method == "percentile_95":
        t = float(np.percentile(scores, 95))
        preds = (scores >= t).astype(int)
        return t, float(f1_score(y_true, preds, zero_division=0))

    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, scores)

    if method == "youden_j":
        j     = tpr_arr - fpr_arr
        idx   = int(np.argmax(j))
        t = float(thresholds[idx])
        return (t, float(j[idx])) if np.isfinite(t) else (float(np.percentile(scores, 95)), 0.0)

    # Default: f1_optimal
    best_f1 = -1.0
    best_t  = 0.5
    for t in thresholds:
        if not np.isfinite(t):
            continue
        preds  = (scores >= t).astype(int)
        f1_val = float(f1_score(y_true, preds, zero_division=0))
        if f1_val > best_f1:
            best_f1 = f1_val
            best_t  = float(t)
    if not np.isfinite(best_t):
        best_t = float(np.percentile(scores, 95))
    return best_t, best_f1


def evaluate_per_attack_type(
    y_label: np.ndarray,
    scores: np.ndarray,
    benign_label: str = "BENIGN",
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Compute AUROC, AUPRC and F1 for each attack type vs. benign.

    Parameters
    ----------
    y_label:
        1-D string array of class labels (e.g. ``['BENIGN', 'DDOS', ...]``).
    scores:
        1-D anomaly score array aligned with *y_label*.
    benign_label:
        The label string for normal/benign traffic.
    threshold:
        Optional global decision threshold.  When provided, F1/precision/
        recall/FPR use this threshold instead of per-attack F1 optimisation.

    Returns
    -------
    pd.DataFrame with columns [attack_type, n_samples, auroc, auprc, f1, ...].
    """
    y_label = np.asarray(y_label)
    benign_upper = benign_label.upper()
    attack_types = [
        t for t in np.unique(y_label) if str(t).upper() != benign_upper
    ]

    benign_mask = np.char.upper(y_label.astype(str)) == benign_upper
    rows: List[Dict] = []

    for attack in attack_types:
        attack_mask = y_label == attack
        mask        = benign_mask | attack_mask

        if mask.sum() < 10:
            continue

        y_sub  = (np.char.upper(y_label[mask].astype(str)) != benign_upper).astype(int)
        s_sub  = scores[mask]
        n_atk  = int(attack_mask.sum())

        try:
            auroc = float(roc_auc_score(y_sub, s_sub))
        except ValueError:
            auroc = float("nan")

        try:
            auprc = float(average_precision_score(y_sub, s_sub))
        except ValueError:
            auprc = float("nan")

        if threshold is not None:
            t_opt = float(threshold)
        else:
            t_opt, _ = find_optimal_threshold(y_sub, s_sub, method="f1")
        preds = (s_sub >= t_opt).astype(int)
        f1_val = float(f1_score(y_sub, preds, zero_division=0))

        row: Dict = {
            "attack_type": attack,
            "n_samples":   n_atk,
            "auroc":       auroc,
            "auprc":       auprc,
            "f1":          f1_val,
        }
        if threshold is not None:
            row["precision"] = float(precision_score(y_sub, preds, zero_division=0))
            row["recall"] = float(recall_score(y_sub, preds, zero_division=0))
            tn, fp, fn, tp = confusion_matrix(y_sub, preds, labels=[0, 1]).ravel()
            row["fpr"] = float(fp / max(fp + tn, 1))

        rows.append(row)

    if not rows:
        base_cols = ["attack_type", "n_samples", "auroc", "auprc", "f1"]
        if threshold is not None:
            base_cols += ["precision", "recall", "fpr"]
        return pd.DataFrame(columns=base_cols)
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def evaluate_all_models(
    model_scores: Dict[str, np.ndarray],
    y_true: np.ndarray,
    y_label: Optional[np.ndarray] = None,
    val_scores: Optional[Dict[str, np.ndarray]] = None,
    y_val: Optional[np.ndarray] = None,
    threshold_method: str = "f1",
) -> pd.DataFrame:
    """Build a comparison table across multiple models.

    Parameters
    ----------
    model_scores:
        Mapping of model name → 1-D anomaly score array (test set).
    y_true:
        Binary ground-truth labels for the test set.
    y_label:
        String labels (optional; used to check benign label presence).
    val_scores:
        Optional mapping of model name → validation scores.  When provided
        together with *y_val*, each model's decision threshold is chosen on
        the validation set and threshold-dependent metrics are reported on
        the test set.
    y_val:
        Binary ground-truth labels for the validation set.

    Returns
    -------
    pd.DataFrame with one row per model and metric columns.
    """
    rows: List[Dict] = []
    for name, scores in model_scores.items():
        if val_scores is not None and y_val is not None:
            thr, _ = find_optimal_threshold(
                y_val, val_scores[name],
                method=normalize_threshold_method(threshold_method),
            )
            m = compute_metrics(y_true, scores, threshold=thr)
        else:
            m = compute_metrics(y_true, scores)
        m["model"] = name
        m["threshold_method"] = threshold_method
        rows.append(m)

    df = pd.DataFrame(rows)
    col_order = ["model", "auroc", "auprc", "f1", "precision", "recall",
                 "fpr", "fnr", "threshold", "threshold_method"]
    col_order = [c for c in col_order if c in df.columns]
    return df[col_order].sort_values("auroc", ascending=False).reset_index(drop=True)


def evaluate_all_threshold_methods(
    model_scores: Dict[str, np.ndarray],
    y_true: np.ndarray,
    val_scores: Dict[str, np.ndarray],
    y_val: np.ndarray,
    methods: Optional[List[str]] = None,
    model_name: str = "ensemble",
) -> pd.DataFrame:
    """Report threshold-dependent metrics for each threshold selection method."""
    if methods is None:
        methods = ["f1", "youden_j", "percentile_95"]
    rows: List[Dict] = []
    scores = model_scores[model_name]
    val_s = val_scores[model_name]
    for method in methods:
        thr, _ = find_optimal_threshold(
            y_val, val_s, method=normalize_threshold_method(method),
        )
        m = compute_metrics(y_true, scores, threshold=thr)
        m["model"] = model_name
        m["threshold_method"] = method
        rows.append(m)
    df = pd.DataFrame(rows)
    col_order = ["model", "threshold_method", "auroc", "auprc", "f1",
                 "precision", "recall", "fpr", "fnr", "threshold"]
    return df[[c for c in col_order if c in df.columns]]


def evaluate_flow_level(
    y_flow_labels: np.ndarray,
    flow_scores: np.ndarray,
    benign_label: str = "BENIGN",
) -> pd.DataFrame:
    """Per-flow AUROC/AUPRC for each attack type vs. benign flows."""
    y_flow_labels = np.asarray(y_flow_labels)
    benign_upper = benign_label.upper()
    attack_types = [
        t for t in np.unique(y_flow_labels) if str(t).upper() != benign_upper
    ]
    benign_mask = np.char.upper(y_flow_labels.astype(str)) == benign_upper
    rows: List[Dict] = []

    for attack in attack_types:
        attack_mask = y_flow_labels == attack
        mask = benign_mask | attack_mask
        if mask.sum() < 10:
            continue

        y_sub = (
            np.char.upper(y_flow_labels[mask].astype(str)) != benign_upper
        ).astype(int)
        s_sub = flow_scores[mask]
        n_atk = int(attack_mask.sum())

        try:
            auroc = float(roc_auc_score(y_sub, s_sub))
        except ValueError:
            auroc = float("nan")
        try:
            auprc = float(average_precision_score(y_sub, s_sub))
        except ValueError:
            auprc = float("nan")

        t_opt, _ = find_optimal_threshold(y_sub, s_sub, method="f1")
        preds = (s_sub >= t_opt).astype(int)
        f1_val = float(f1_score(y_sub, preds, zero_division=0))

        rows.append({
            "attack_type": attack,
            "n_samples": n_atk,
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1_val,
        })

    if not rows:
        return pd.DataFrame(
            columns=["attack_type", "n_samples", "auroc", "auprc", "f1"]
        )
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def evaluate_per_attack_with_scores(
    y_label: np.ndarray,
    scores: np.ndarray,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Evaluate per-attack metrics and optionally save to CSV."""
    df = evaluate_per_attack_type(y_label, scores)
    if output_path and not df.empty:
        df.to_csv(output_path, index=False)
    return df


def print_metrics_table(results_df: pd.DataFrame) -> None:
    """Print a neatly formatted metrics comparison table to stdout.

    Parameters
    ----------
    results_df:
        DataFrame returned by :func:`evaluate_all_models`.
    """
    float_cols = ["auroc", "auprc", "f1", "precision", "recall", "fpr", "fnr", "threshold"]
    fmt = results_df.copy()
    for col in float_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda v: f"{v:.4f}" if not np.isnan(v) else "NaN")

    # Build header
    col_widths = {col: max(len(col), fmt[col].astype(str).str.len().max())
                  for col in fmt.columns}
    header = " | ".join(col.ljust(col_widths[col]) for col in fmt.columns)
    sep    = "-+-".join("-" * col_widths[col] for col in fmt.columns)

    print("\n" + header)
    print(sep)
    for _, row in fmt.iterrows():
        print(" | ".join(str(row[col]).ljust(col_widths[col]) for col in fmt.columns))
    print()
