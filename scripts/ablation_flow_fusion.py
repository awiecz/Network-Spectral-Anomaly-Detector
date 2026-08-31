"""
ablation_flow_fusion.py — Compare spectral-only vs flow-fused ensemble.

Runs four configurations on the Friday CICIDS2017 split and writes
results/metrics/fusion_ablation.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data_loader import load_cicids2017
from src.evaluation import (
    compute_metrics,
    evaluate_per_attack_type,
    normalize_threshold_method,
)
from src.models import EnsembleDetector
from src.pipeline_helpers import (
    apply_flow_ecod_fusion,
    build_ensemble_models,
    calibrate_all_detectors,
    ensemble_weights_from_config,
    score_all_models,
    train_detectors,
)
from src.preprocessor import DataPreprocessor
from src.splits import build_cicids_day_splits
from src.tuning_config import apply_tuned_config, load_tuned_artifacts

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("ablation")


def _build_splits(cfg: dict, df: pd.DataFrame, label_rule: str) -> dict:
    run_cfg = deepcopy(cfg)
    run_cfg.setdefault("evaluation", {})["window_label_rule"] = label_rule
    train_day = run_cfg["data"].get("train_day", "Monday")
    train_df = df[df["day"] == train_day].reset_index(drop=True)
    eval_days = run_cfg.get("evaluation", {}).get("eval_days", ["Friday"])
    test_df = df[df["day"].isin(eval_days)].reset_index(drop=True)
    return build_cicids_day_splits(
        train_df, test_df, run_cfg, np.random.RandomState(42),
    )


def _score_pipeline(
    cfg: dict,
    df: pd.DataFrame,
    splits: dict,
    fusion_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = DataPreprocessor()
    X_tr = pre.fit_transform(splits["X_train"], feature_names=list(splits["feat_names"]))
    X_va = pre.transform(splits["X_val"])
    X_te = pre.transform(splits["X_test"])
    y_va, y_te = splits["y_val"], splits["y_test"]

    run_cfg = deepcopy(cfg)
    run_cfg.setdefault("models", {}).setdefault("flow_ecod", {})["fusion_weight"] = fusion_weight

    models = build_ensemble_models(run_cfg, X_tr.shape[1])
    train_detectors(models, X_tr, run_cfg, X_val=X_va)
    calibrate_all_detectors(models, X_va, y_va)

    weights = ensemble_weights_from_config(run_cfg)
    ens = EnsembleDetector(
        models={k: models[k] for k in weights if k in models},
        weights=weights,
    )
    va_ind = score_all_models(models, X_va)
    te_ind = score_all_models(models, X_te)
    va_scores = {**va_ind, "ensemble": ens.score_from(va_ind)}
    te_scores = {**te_ind, "ensemble": ens.score_from(te_ind)}

    val_ts = splits.get("val_timestamps", splits["test_timestamps"])
    va_scores, te_scores, _ = apply_flow_ecod_fusion(
        df, run_cfg, va_scores, te_scores, y_va,
        val_ts, splits["test_timestamps"],
        flow_weight=fusion_weight,
    )
    return va_scores["ensemble"], te_scores["ensemble"], y_te


def _run_config(
    name: str,
    cfg: dict,
    df: pd.DataFrame,
    label_rule: str,
    fusion_weight: float,
) -> list[dict]:
    splits = _build_splits(cfg, df, label_rule)
    va_s, te_s, y_te = _score_pipeline(cfg, df, splits, fusion_weight)
    thr_method = normalize_threshold_method(
        cfg.get("evaluation", {}).get("default_threshold_method", "f1_optimal"),
    )
    thr, _ = EnsembleDetector.find_optimal_threshold(va_s, splits["y_val"], method=thr_method)
    overall = compute_metrics(y_te, te_s, threshold=thr)
    per_attack = evaluate_per_attack_type(splits["lab_test"], te_s, threshold=thr)

    rows = [{
        "config": name,
        "label_rule": label_rule,
        "fusion_weight": fusion_weight,
        "scope": "overall",
        "attack_type": "ALL",
        "n_samples": int(y_te.sum()),
        "auroc": overall["auroc"],
        "auprc": overall["auprc"],
        "f1": overall["f1"],
        "fpr": overall["fpr"],
        "recall": overall["recall"],
    }]
    for _, row in per_attack.iterrows():
        rows.append({
            "config": name,
            "label_rule": label_rule,
            "fusion_weight": fusion_weight,
            "scope": "per_attack",
            "attack_type": row["attack_type"],
            "n_samples": row["n_samples"],
            "auroc": row["auroc"],
            "auprc": row["auprc"],
            "f1": row["f1"],
            "fpr": row.get("fpr", float("nan")),
            "recall": row.get("recall", float("nan")),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Flow fusion ablation study")
    ap.add_argument("--data-dir", default="data/raw/cicids2017")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--output", default="results/metrics/fusion_ablation.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tuned = load_tuned_artifacts()
    if tuned:
        cfg = apply_tuned_config(cfg, tuned)

    tuned_w = float(
        tuned.get("flow_ecod", {}).get("fusion_weight", 0.3) if tuned else
        cfg.get("models", {}).get("flow_ecod", {}).get("fusion_weight", 0.3)
    )

    train_day = cfg["data"].get("train_day", "Monday")
    eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
    processed_dir = cfg["data"].get("processed_path", "data/processed")
    df = load_cicids2017(
        args.data_dir, days=[train_day, *eval_days], processed_dir=processed_dir,
    )

    configs = [
        ("A_spectral_majority", "majority", 0.0),
        ("B_fused_majority", "majority", 0.3),
        ("C_spectral_any", "any", 0.0),
        ("D_fused_any_tuned", "any", tuned_w),
    ]

    all_rows: list[dict] = []
    for name, rule, fw in configs:
        logger.info("Running %s (rule=%s, fusion_weight=%.2f) …", name, rule, fw)
        all_rows.extend(_run_config(name, cfg, df, rule, fw))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)

    bot_rows = [r for r in all_rows if str(r.get("attack_type", "")).upper() == "BOT"]
    if bot_rows:
        baseline = next((r for r in bot_rows if r["config"] == "C_spectral_any"), None)
        fused = next((r for r in bot_rows if r["config"] == "D_fused_any_tuned"), None)
        if baseline and fused:
            delta = fused["auroc"] - baseline["auroc"]
            logger.info(
                "BOT AUROC delta (D vs C): %.4f (baseline=%.4f, fused=%.4f)",
                delta, baseline["auroc"], fused["auroc"],
            )
            if delta < 0.05:
                logger.warning(
                    "Flow fusion did not improve BOT AUROC by >= 0.05; "
                    "consider aggregation=mean or higher fusion_weight."
                )


if __name__ == "__main__":
    main()
