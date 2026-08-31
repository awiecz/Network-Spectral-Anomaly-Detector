"""
tune.py — Model-tuning harness for the Spectral Anomaly Detector.
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
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_loader import load_cicids2017
from src.feature_cache import cache_path, load_splits_cache, save_splits_cache
from src.preprocessor import DataPreprocessor
from src.evaluation import (
    compute_metrics,
    evaluate_all_models,
    evaluate_all_threshold_methods,
    evaluate_flow_level,
    evaluate_per_attack_type,
    normalize_threshold_method,
)
from src.models import (
    COPODDetector,
    ECODDetector,
    HBOSDetector,
    IsolationForestDetector,
    VAETrainer,
    EnsembleDetector,
)
from src.models.scoring import calibrate_detector_scores
from src.pipeline_helpers import (
    apply_flow_ecod_fusion,
    calibrate_all_detectors,
    filter_splits_by_attack_labels,
    optimise_weights,
    score_all_models,
    sweep_fusion_weight,
)
from src.splits import build_cicids_day_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tune")
logging.getLogger("src").setLevel(logging.WARNING)

RNG = np.random.RandomState(42)


def prepare_splits(args, cfg):
    train_day = cfg["data"].get("train_day", "Monday")
    eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
    cache_file = cache_path(cfg, train_day, eval_days)

    if args.use_cache:
        cached = load_splits_cache(cache_file)
        if cached is not None:
            logger.info("Loading feature cache from %s", cache_file)
            return cached, None

    logger.info("Loading CICIDS2017 (%s + %s) …", train_day, eval_days)
    processed_dir = cfg["data"].get("processed_path", "data/processed")
    df = load_cicids2017(args.data_dir, days=[train_day, *eval_days], processed_dir=processed_dir)
    train_df = df[df["day"] == train_day].reset_index(drop=True)
    test_df = df[df["day"].isin(eval_days)].reset_index(drop=True)

    logger.info("Building spectral features …")
    splits = build_cicids_day_splits(train_df, test_df, cfg, RNG)

    out = {
        "X_train": splits["X_train"],
        "X_val": splits["X_val"],
        "y_val": splits["y_val"],
        "lab_val": splits["lab_val"],
        "val_timestamps": splits.get("val_timestamps"),
        "X_test": splits["X_test"],
        "y_test": splits["y_test"],
        "lab_test": splits["lab_test"],
        "test_timestamps": splits["test_timestamps"],
        "feat_names": np.array(splits["feat_names"], dtype=object),
    }
    save_splits_cache(cache_file, out)
    logger.info("Feature cache written to %s", cache_file)
    return out, df


def _metric(y, s, name: str) -> float:
    try:
        if name == "auprc":
            return float(average_precision_score(y, s))
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def sweep_vae(Xtr, Xva, yva, input_dim, cfg):
    vcfg = cfg["models"]["vae"]
    grid = [
        {"latent_dim": ld, "beta": b, "dropout": dr, "epochs": ep}
        for ld in (16, 32)
        for b in (0.5, 1.0)
        for dr in (0.1, 0.2)
        for ep in (40, 60)
    ]
    best, best_score, rows = None, -1.0, []
    patience = vcfg.get("early_stopping_patience", 5)
    for g in grid:
        m = VAETrainer(
            input_dim=input_dim,
            encoder_dims=vcfg["encoder_dims"],
            latent_dim=g["latent_dim"],
            decoder_dims=vcfg["decoder_dims"],
            beta=g["beta"],
            dropout=g["dropout"],
            learning_rate=vcfg["learning_rate"],
        )
        m.fit(
            Xtr, epochs=g["epochs"], batch_size=vcfg["batch_size"],
            beta_warmup_epochs=vcfg["beta_warmup_epochs"],
            X_val=Xva, early_stopping_patience=patience,
        )
        calibrate_detector_scores(m, Xva, yva)
        a = _metric(yva, m.score(Xva), "auroc")
        rows.append({"model": "vae", **g, "val_auroc": a})
        logger.info("  VAE %s -> val AUROC %.4f", g, a)
        if a > best_score:
            best_score, best = a, (m, g)
    return best, rows


def sweep_if(Xtr, Xva, yva, cfg):
    grid = [
        {"n_estimators": n, "max_samples": ms, "max_features": mf}
        for n in (200, 400)
        for ms in (256, 512)
        for mf in (1.0, 0.8)
    ]
    best, best_score, rows = None, -1.0, []
    for g in grid:
        m = IsolationForestDetector(
            n_estimators=g["n_estimators"],
            max_samples=g["max_samples"],
            max_features=g["max_features"],
        )
        m.fit(Xtr)
        calibrate_detector_scores(m, Xva, yva)
        a = _metric(yva, m.score(Xva), "auroc")
        rows.append({"model": "isolation_forest", **g, "val_auroc": a})
        logger.info("  IF %s -> val AUROC %.4f", g, a)
        if a > best_score:
            best_score, best = a, (m, g)
    return best, rows


def sweep_ecod(Xtr, Xva, yva):
    best, best_score, rows = None, -1.0, []
    for c in (0.05, 0.1, 0.15):
        m = ECODDetector(contamination=c)
        m.fit(Xtr)
        calibrate_detector_scores(m, Xva, yva)
        a = _metric(yva, m.score(Xva), "auroc")
        rows.append({"model": "ecod", "contamination": c, "val_auroc": a})
        logger.info("  ECOD contamination=%s -> val AUROC %.4f", c, a)
        if a > best_score:
            best_score, best = a, (m, {"contamination": c})
    return best, rows


def sweep_copod(Xtr, Xva, yva):
    best, best_score, rows = None, -1.0, []
    for c in (0.05, 0.1, 0.15):
        m = COPODDetector(contamination=c)
        m.fit(Xtr)
        calibrate_detector_scores(m, Xva, yva)
        a = _metric(yva, m.score(Xva), "auroc")
        rows.append({"model": "copod", "contamination": c, "val_auroc": a})
        if a > best_score:
            best_score, best = a, (m, {"contamination": c})
    return best, rows


def sweep_hbos(Xtr, Xva, yva):
    best, best_score, rows = None, -1.0, []
    for c in (0.05, 0.1, 0.15):
        m = HBOSDetector(contamination=c)
        m.fit(Xtr)
        calibrate_detector_scores(m, Xva, yva)
        a = _metric(yva, m.score(Xva), "auroc")
        rows.append({"model": "hbos", "contamination": c, "val_auroc": a})
        if a > best_score:
            best_score, best = a, (m, {"contamination": c})
    return best, rows


def _run_dual_eval(args, cfg, results_dir, best_cfgs, vae_g, if_g, ecod_g, df):
    any_cfg = deepcopy(cfg)
    any_cfg["evaluation"]["window_label_rule"] = "any"
    any_args = argparse.Namespace(**{**vars(args), "use_cache": False})
    any_data, _ = prepare_splits(any_args, any_cfg)
    drop = {a.strip().upper() for a in args.exclude_attacks.split(",") if a.strip()}
    if drop:
        any_data = filter_splits_by_attack_labels(any_data, list(drop))

    pre2 = DataPreprocessor()
    X_tr2 = pre2.fit_transform(any_data["X_train"], feature_names=list(any_data["feat_names"]))
    X_va2, y_va2 = pre2.transform(any_data["X_val"]), any_data["y_val"]
    X_te2, lab2 = pre2.transform(any_data["X_test"]), any_data["lab_test"]
    m2 = {
        "vae": VAETrainer(
            input_dim=X_tr2.shape[1],
            encoder_dims=cfg["models"]["vae"]["encoder_dims"],
            decoder_dims=cfg["models"]["vae"]["decoder_dims"],
            latent_dim=vae_g["latent_dim"], beta=vae_g["beta"],
            dropout=vae_g.get("dropout", 0.2),
        ),
        "isolation_forest": IsolationForestDetector(**if_g),
        "ecod": ECODDetector(**ecod_g),
    }
    m2["vae"].fit(
        X_tr2, epochs=vae_g.get("epochs", 40), batch_size=256,
        beta_warmup_epochs=10, X_val=X_va2, early_stopping_patience=5,
    )
    m2["isolation_forest"].fit(X_tr2)
    m2["ecod"].fit(X_tr2)
    calibrate_all_detectors(m2, X_va2, y_va2)
    w2, _ = optimise_weights(
        score_all_models(m2, X_va2), y_va2,
        step=args.weight_step, optimize_metric=args.optimize_metric,
    )
    ens2 = EnsembleDetector(
        models={k: m2[k] for k in w2 if w2[k] > 0},
        weights={k: w2[k] for k in w2 if w2[k] > 0},
    )
    va2 = score_all_models(m2, X_va2)
    te2 = score_all_models(m2, X_te2)
    te2["ensemble"] = ens2.score_from(te2)
    va2["ensemble"] = ens2.score_from(va2)

    te_unfused = dict(te2)
    flow_w = float(cfg.get("models", {}).get("flow_ecod", {}).get("fusion_weight", 0.0))
    any_val_ts = any_data.get("val_timestamps", any_data["test_timestamps"])
    if df is not None and flow_w > 0:
        va2, te2, _ = apply_flow_ecod_fusion(
            df, any_cfg, va2, te2, y_va2,
            any_val_ts, any_data["test_timestamps"],
            flow_weight=flow_w,
        )

    per_unfused = evaluate_per_attack_type(lab2, te_unfused["ensemble"])
    if not per_unfused.empty:
        per_unfused.to_csv(results_dir / "per_attack_any_rule.csv", index=False)

    per_any = evaluate_per_attack_type(lab2, te2["ensemble"])
    if not per_any.empty:
        per_any.to_csv(results_dir / "per_attack_any_rule_fused.csv", index=False)

    thr_method = normalize_threshold_method(
        cfg.get("evaluation", {}).get("default_threshold_method", "f1_optimal"),
    )
    evaluate_all_models(
        te2, any_data["y_test"], val_scores=va2, y_val=y_va2,
        threshold_method=thr_method,
    ).to_csv(results_dir / "model_comparison_any_rule.csv", index=False)

    evaluate_all_threshold_methods(
        te2, any_data["y_test"], va2, y_va2,
        methods=cfg.get("evaluation", {}).get("threshold_methods"),
    ).to_csv(results_dir / "threshold_comparison_any_rule.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/cicids2017")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--label-rule", choices=["any", "majority"], default=None)
    ap.add_argument("--exclude-attacks", default="")
    ap.add_argument("--optimize-metric", choices=["auroc", "auprc"], default="auroc")
    ap.add_argument("--weight-step", type=float, default=0.05)
    ap.add_argument("--dual-eval", action="store_true")
    ap.add_argument("--skip-fusion-sweep", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.label_rule:
        cfg.setdefault("evaluation", {})["window_label_rule"] = args.label_rule
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)

    data, df = prepare_splits(args, cfg)
    if df is None:
        train_day = cfg["data"].get("train_day", "Monday")
        eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
        processed_dir = cfg["data"].get("processed_path", "data/processed")
        df = load_cicids2017(
            args.data_dir, days=[train_day, *eval_days], processed_dir=processed_dir,
        )

    X_val, y_val, lab_val = data["X_val"], data["y_val"], data["lab_val"]
    X_test, y_test, lab_test = data["X_test"], data["y_test"], data["lab_test"]
    val_ts = data.get("val_timestamps", data["test_timestamps"])
    test_ts = data["test_timestamps"]

    drop = {a.strip().upper() for a in args.exclude_attacks.split(",") if a.strip()}
    if drop:
        data = filter_splits_by_attack_labels(data, list(drop))
        X_val, y_val, lab_val = data["X_val"], data["y_val"], data["lab_val"]
        X_test, y_test, lab_test = data["X_test"], data["y_test"], data["lab_test"]
        val_ts = data.get("val_timestamps", data["test_timestamps"])
        test_ts = data["test_timestamps"]

    pre = DataPreprocessor()
    X_train_s = pre.fit_transform(data["X_train"], feature_names=list(data["feat_names"]))
    X_val_s = pre.transform(X_val)
    X_test_s = pre.transform(X_test)

    all_rows = []
    (vae, vae_g), r = sweep_vae(X_train_s, X_val_s, y_val, X_train_s.shape[1], cfg)
    all_rows += r
    (ifd, if_g), r = sweep_if(X_train_s, X_val_s, y_val, cfg)
    all_rows += r
    (ecod, ecod_g), r = sweep_ecod(X_train_s, X_val_s, y_val)
    all_rows += r
    (copod, copod_g), r = sweep_copod(X_train_s, X_val_s, y_val)
    all_rows += r
    (hbos, hbos_g), r = sweep_hbos(X_train_s, X_val_s, y_val)
    all_rows += r

    models = {"vae": vae, "isolation_forest": ifd, "ecod": ecod, "copod": copod, "hbos": hbos}
    best_cfgs = {"vae": vae_g, "isolation_forest": if_g, "ecod": ecod_g, "copod": copod_g, "hbos": hbos_g}

    val_scores = score_all_models(models, X_val_s)
    test_scores = score_all_models(models, X_test_s)
    weights, val_metric = optimise_weights(
        val_scores, y_val, step=args.weight_step, optimize_metric=args.optimize_metric,
    )
    ens = EnsembleDetector(
        models={k: models[k] for k in weights if weights[k] > 0},
        weights={k: weights[k] for k in weights if weights[k] > 0},
    )
    test_scores["ensemble"] = ens.score_from(test_scores)
    val_scores["ensemble"] = ens.score_from(val_scores)

    best_fusion_w = float(cfg.get("models", {}).get("flow_ecod", {}).get("fusion_weight", 0.0))
    val_fusion_metric = float("nan")
    if not args.skip_fusion_sweep and df is not None:
        best_fusion_w, val_fusion_metric = sweep_fusion_weight(
            df, cfg, val_scores, test_scores, y_val, lab_val, val_ts, test_ts,
        )
        cfg.setdefault("models", {}).setdefault("flow_ecod", {})["fusion_weight"] = best_fusion_w
        val_scores, test_scores, _ = apply_flow_ecod_fusion(
            df, cfg, val_scores, test_scores, y_val, val_ts, test_ts,
            flow_weight=best_fusion_w,
        )
        logger.info("Best fusion_weight=%.2f (val metric=%.4f)", best_fusion_w, val_fusion_metric)

    final_rows = []
    thr_method = normalize_threshold_method(
        cfg.get("evaluation", {}).get("default_threshold_method", "f1_optimal"),
    )
    for n, s_te in test_scores.items():
        thr, _ = EnsembleDetector.find_optimal_threshold(
            val_scores[n], y_val, method=thr_method,
        )
        m = compute_metrics(y_test, s_te, threshold=thr)
        m["model"] = n
        final_rows.append(m)

    cols = ["model", "auroc", "auprc", "f1", "precision", "recall", "fpr", "fnr", "threshold"]
    final_df = pd.DataFrame(final_rows)[cols].sort_values("auroc", ascending=False)
    print("\n=== Final TEST metrics ===")
    print(final_df.to_string(index=False))

    final_df.to_csv(results_dir / "model_comparison.csv", index=False)
    pd.DataFrame(all_rows).to_csv(results_dir / "tuning_results.csv", index=False)
    with open(results_dir / "ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "weights": weights,
                "best_configs": best_cfgs,
                f"val_ensemble_{args.optimize_metric}": val_metric,
                "flow_ecod": {
                    "fusion_weight": best_fusion_w,
                    "val_fusion_metric": val_fusion_metric,
                },
            },
            f, indent=2,
        )

    evaluate_all_threshold_methods(
        test_scores, y_test, val_scores, y_val,
        methods=cfg.get("evaluation", {}).get("threshold_methods"),
    ).to_csv(results_dir / "threshold_comparison.csv", index=False)

    opt_thr, _ = EnsembleDetector.find_optimal_threshold(
        val_scores["ensemble"], y_val, method=thr_method,
    )
    per_attack = evaluate_per_attack_type(
        lab_test, test_scores["ensemble"], threshold=opt_thr,
    )
    if not per_attack.empty:
        per_attack.to_csv(results_dir / "per_attack_metrics.csv", index=False)

    eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
    eval_flows = df[df["day"].isin(eval_days)] if df is not None else None
    if eval_flows is not None and "Label" in eval_flows.columns and best_fusion_w > 0:
        from src.models.flow_ecod import FlowECODDetector
        flow_det = FlowECODDetector(
            contamination=cfg["models"]["flow_ecod"].get("contamination", 0.1),
        )
        benign_train = df[
            (df["day"] == cfg["data"].get("train_day", "Monday"))
            & (df["Label"].astype(str).str.upper() == "BENIGN")
        ]
        flow_det.fit(benign_train)
        flow_scores = flow_det.score_flows(eval_flows)
        flow_df = evaluate_flow_level(eval_flows["Label"].values, flow_scores)
        if not flow_df.empty:
            flow_df.to_csv(results_dir / "per_attack_flow_level.csv", index=False)

    if args.dual_eval or cfg.get("evaluation", {}).get("dual_eval", False):
        _run_dual_eval(args, cfg, results_dir, best_cfgs, vae_g, if_g, ecod_g, df)

    logger.info("Tuning complete.")


if __name__ == "__main__":
    main()
