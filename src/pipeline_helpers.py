"""
pipeline_helpers.py — Shared training, calibration, and scoring utilities.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.scoring import calibrate_detector_scores, fit_percentile_bounds


def _optimise_metric(y: np.ndarray, scores: np.ndarray, name: str) -> float:
    try:
        if name == "auprc":
            return float(average_precision_score(y, scores))
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")


def optimise_weights(
    val_scores: Dict[str, np.ndarray],
    y_val: np.ndarray,
    step: float = 0.05,
    min_auroc: float = 0.55,
    optimize_metric: str = "auroc",
) -> Tuple[Dict[str, float], float]:
    """Grid-search ensemble weights on validation scores (used by tune.py and notebooks)."""
    eligible = {
        n: s for n, s in val_scores.items()
        if _optimise_metric(y_val, s, "auroc") >= min_auroc
    }
    if not eligible:
        eligible = dict(val_scores)
    names = list(eligible.keys())
    levels = np.round(np.arange(0.0, 1.0 + 1e-9, step), 3)
    best_w, best_score = None, -1.0
    for combo in itertools.product(levels, repeat=len(names)):
        if abs(sum(combo) - 1.0) > 1e-6:
            continue
        blended = sum(w * eligible[n] for w, n in zip(combo, names))
        score = _optimise_metric(y_val, blended, optimize_metric)
        if score > best_score:
            best_score, best_w = score, dict(zip(names, [float(c) for c in combo]))
    full_w = {n: 0.0 for n in val_scores}
    full_w.update(best_w)
    return full_w, best_score


def calibrate_all_detectors(
    detectors: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    """Apply validation benign percentile calibration to all detectors."""
    for det in detectors.values():
        calibrate_detector_scores(det, X_val, y_val)


def _ensemble_mode(cfg: dict) -> str:
    return cfg.get("models", {}).get("ensemble", {}).get("mode", "tuned")


def build_ensemble_models(
    cfg: dict,
    input_dim: int,
    *,
    include_optional: bool = True,
) -> Dict[str, Any]:
    """Instantiate detector objects from config (untrained)."""
    from src.models import (
        COPODDetector,
        ECODDetector,
        HBOSDetector,
        IsolationForestDetector,
        VAETrainer,
    )

    mode = _ensemble_mode(cfg)
    if mode == "ecod_only":
        mcfg = cfg["models"]
        return {
            "ecod": ECODDetector(
                contamination=mcfg.get("ecod", {}).get("contamination", 0.1)
            ),
        }

    mcfg = cfg["models"]
    vcfg = mcfg["vae"]
    ifcfg = mcfg["isolation_forest"]

    models: Dict[str, Any] = {
        "vae": VAETrainer(
            input_dim=input_dim,
            encoder_dims=vcfg["encoder_dims"],
            latent_dim=vcfg["latent_dim"],
            decoder_dims=vcfg["decoder_dims"],
            dropout=vcfg.get("dropout", 0.2),
            beta=vcfg["beta"],
            learning_rate=vcfg["learning_rate"],
        ),
        "isolation_forest": IsolationForestDetector(
            n_estimators=ifcfg["n_estimators"],
            max_samples=ifcfg["max_samples"],
            max_features=ifcfg.get("max_features", 1.0),
            contamination=ifcfg["contamination"],
            n_jobs=ifcfg["n_jobs"],
            random_state=ifcfg["random_state"],
        ),
        "ecod": ECODDetector(contamination=mcfg.get("ecod", {}).get("contamination", 0.1)),
    }

    if include_optional and mode != "ecod_only":
        if mcfg.get("copod"):
            models["copod"] = COPODDetector(
                contamination=mcfg["copod"].get("contamination", 0.1)
            )
        if mcfg.get("hbos"):
            models["hbos"] = HBOSDetector(
                contamination=mcfg["hbos"].get("contamination", 0.1),
                n_bins=mcfg["hbos"].get("n_bins", 10),
            )
    return models


def train_detectors(
    models: Dict[str, Any],
    X_train: np.ndarray,
    cfg: dict,
    X_val: Optional[np.ndarray] = None,
) -> None:
    """Fit all detectors in *models*."""
    if "vae" in models:
        vcfg = cfg["models"]["vae"]
        models["vae"].fit(
            X_train,
            epochs=vcfg["epochs"],
            batch_size=vcfg["batch_size"],
            beta_warmup_epochs=vcfg["beta_warmup_epochs"],
            X_val=X_val,
            early_stopping_patience=vcfg.get("early_stopping_patience", 0),
        )
    if "isolation_forest" in models:
        models["isolation_forest"].fit(X_train)
    if "ecod" in models:
        models["ecod"].fit(X_train)
    if "copod" in models:
        models["copod"].fit(X_train)
    if "hbos" in models:
        models["hbos"].fit(X_train)


def score_all_models(
    models: Dict[str, Any],
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {name: m.score(X) for name, m in models.items()}


def ensemble_weights_from_config(cfg: dict) -> Dict[str, float]:
    """Return ensemble weights for models present in config."""
    w = cfg["models"]["ensemble"]["weights"]
    return {k: float(v) for k, v in w.items() if float(v) > 0}


def filter_splits_by_attack_labels(
    splits: dict,
    exclude_attacks: List[str],
) -> dict:
    """Drop windows whose fine-grained label is in *exclude_attacks*."""
    if not exclude_attacks:
        return splits
    drop = {a.strip().upper() for a in exclude_attacks if a.strip()}

    def _keep(lab: np.ndarray) -> np.ndarray:
        up = np.array([str(x).upper() for x in lab])
        return ~np.isin(up, list(drop))

    out = dict(splits)
    for split in ("val", "test"):
        key_lab = f"lab_{split}"
        key_y = f"y_{split}"
        key_x = f"X_{split}"
        if key_lab not in out:
            continue
        keep = _keep(out[key_lab])
        out[key_x] = out[key_x][keep]
        out[key_y] = out[key_y][keep]
        out[key_lab] = out[key_lab][keep]
        if split == "test" and "test_timestamps" in out:
            out["test_timestamps"] = out["test_timestamps"][keep]
        if split == "val" and "val_timestamps" in out:
            out["val_timestamps"] = out["val_timestamps"][keep]
    return out


def apply_flow_ecod_fusion(
    df: pd.DataFrame,
    cfg: dict,
    val_scores: Dict[str, np.ndarray],
    test_scores: Dict[str, np.ndarray],
    y_val: np.ndarray,
    val_timestamps: np.ndarray,
    test_timestamps: np.ndarray,
    *,
    flow_weight: Optional[float] = None,
    fit_detector: bool = True,
    flow_det: Optional[Any] = None,
    save_path: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[Any]]:
    """Fuse flow-level ECOD window scores into the spectral ensemble.

    Returns updated (val_scores, test_scores, flow_detector).
    """
    from src.models.flow_ecod import FlowECODDetector

    flow_cfg = cfg.get("models", {}).get("flow_ecod", {})
    w = float(flow_cfg.get("fusion_weight", 0.0) if flow_weight is None else flow_weight)
    if w <= 0.0:
        return val_scores, test_scores, flow_det

    train_day = cfg["data"].get("train_day", "Monday")
    eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
    benign_train = df[df["day"] == train_day]
    benign_train = benign_train[
        benign_train["Label"].astype(str).str.upper() == "BENIGN"
    ]
    eval_flows = df[df["day"].isin(list(dict.fromkeys([train_day, *eval_days])))]

    if flow_det is None:
        flow_det = FlowECODDetector(contamination=flow_cfg.get("contamination", 0.1))
        flow_det.aggregation = flow_cfg.get("aggregation", "max")

    if fit_detector:
        flow_det.fit(benign_train)

    spectral_cfg = cfg["spectral"]
    ws = spectral_cfg["window_size"]
    ov = spectral_cfg.get("overlap", 0)
    ts_col = "Timestamp" if "Timestamp" in df.columns else "Stime"
    aggregation = flow_cfg.get("aggregation", flow_det.aggregation)

    val_flow = flow_det.aggregate_to_windows(
        eval_flows, val_timestamps, ws, ov, ts_col, aggregation=aggregation,
    )
    test_flow = flow_det.aggregate_to_windows(
        eval_flows, test_timestamps, ws, ov, ts_col, aggregation=aggregation,
    )

    p_low_pct = float(flow_cfg.get("percentile_low", 5))
    p_high_pct = float(flow_cfg.get("percentile_high", 95))
    p_low, p_high = fit_percentile_bounds(
        val_flow, y_val == 0, low_pct=p_low_pct, high_pct=p_high_pct,
    )
    val_flow_n = np.clip((val_flow - p_low) / max(p_high - p_low, 1e-6), 0, 1)
    test_flow_n = np.clip((test_flow - p_low) / max(p_high - p_low, 1e-6), 0, 1)

    spectral_w = 1.0 - w
    val_out = dict(val_scores)
    test_out = dict(test_scores)
    val_out["ensemble"] = spectral_w * val_scores["ensemble"] + w * val_flow_n
    test_out["ensemble"] = spectral_w * test_scores["ensemble"] + w * test_flow_n

    if save_path:
        flow_det.save(save_path)

    return val_out, test_out, flow_det


def sweep_fusion_weight(
    df: pd.DataFrame,
    cfg: dict,
    val_scores: Dict[str, np.ndarray],
    test_scores: Dict[str, np.ndarray],
    y_val: np.ndarray,
    lab_val: np.ndarray,
    val_timestamps: np.ndarray,
    test_timestamps: np.ndarray,
    weights: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """Grid-search fusion_weight on validation AUPRC (BOT subset if present)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if weights is None:
        weights = [0.0, 0.1, 0.2, 0.3, 0.5]

    bot_mask = np.char.upper(lab_val.astype(str)) == "BOT"
    use_bot = int(bot_mask.sum()) >= 10

    best_w, best_score = 0.0, -1.0
    for w in weights:
        v_s, _, _ = apply_flow_ecod_fusion(
            df, cfg, val_scores, test_scores, y_val,
            val_timestamps, test_timestamps,
            flow_weight=w, fit_detector=True,
        )
        ens = v_s["ensemble"]
        if use_bot:
            benign = np.char.upper(lab_val.astype(str)) == "BENIGN"
            mask = benign | bot_mask
            y_sub = (np.char.upper(lab_val[mask].astype(str)) != "BENIGN").astype(int)
            try:
                score = float(average_precision_score(y_sub, ens[mask]))
            except ValueError:
                score = float("nan")
        else:
            try:
                score = float(average_precision_score(y_val, ens))
            except ValueError:
                score = float("nan")
        if np.isfinite(score) and score > best_score:
            best_score, best_w = score, w

    return best_w, best_score
