"""
Network Spectral Anomaly Detector — Main Pipeline
==================================================
End-to-end CLI for training and evaluating the spectral anomaly detector
on CICIDS2017 or UNSW-NB15.

Usage examples
--------------
# Full pipeline on CICIDS2017:
    python main.py --data-dir data/raw/cicids2017 --config config/config.yaml

# Quick demo on the synthetic sample (no dataset download required):
    python main.py --use-sample --config config/config.yaml

# Evaluate a previously saved ensemble without retraining:
    python main.py --use-sample --load-models results/models --skip-train
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Project modules
from src.data_loader import (
    create_sample_dataset,
    load_cicids2017,
)
from src.preprocessor import DataPreprocessor
from src.feature_cache import cache_path, load_splits_cache, save_splits_cache
from src.spectral_features import build_standard_timeseries
from src.splits import (
    build_cicids_day_splits,
    build_disjoint_window_splits,
    build_subset_features,
)
from src.evaluation import (
    evaluate_all_models,
    evaluate_all_threshold_methods,
    evaluate_flow_level,
    evaluate_per_attack_type,
    normalize_threshold_method,
    print_metrics_table,
)
from src.pipeline_helpers import (
    apply_flow_ecod_fusion,
    build_ensemble_models,
    calibrate_all_detectors,
    ensemble_weights_from_config,
    filter_splits_by_attack_labels,
    score_all_models,
    train_detectors,
)
from src.tuning_config import apply_tuned_config, load_tuned_artifacts
from src.models.ensemble import EnsembleDetector
from src.models.flow_ecod import FlowECODDetector
from src.visualization import (
    plot_anomaly_timeline,
    plot_confusion_matrix,
    plot_score_distributions,
    plot_spectrogram,
    plot_umap_latent_space,
    save_figure,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

np.random.seed(42)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Network Spectral Anomaly Detector — end-to-end pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to CICIDS2017 CSV directory.",
    )
    parser.add_argument(
        "--dataset",
        choices=["cicids2017", "unsw_nb15"],
        default="cicids2017",
        help="Which dataset to load (UNSW-NB15 is not yet supported for spectral features).",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Generate and use the synthetic sample dataset (no real data needed).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Root directory for all outputs (models, figures, CSVs).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training; load models from --load-models instead.",
    )
    parser.add_argument(
        "--load-models",
        default=None,
        help="Directory to load pre-trained models from (requires --skip-train).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse cached spectral feature splits (skips FFT rebuild).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable all visualisation output.",
    )
    parser.add_argument(
        "--tuned-weights",
        action="store_true",
        default=None,
        help="Load ensemble weights and model configs from results/metrics/ensemble_weights.json.",
    )
    parser.add_argument(
        "--no-tuned-weights",
        action="store_true",
        help="Do not load tuned weights even if ensemble_weights.json exists.",
    )
    parser.add_argument(
        "--threshold-method",
        default=None,
        choices=[
            "f1", "f1_optimal", "youden_j", "percentile_95",
            "fpr_0.01", "fpr_0.05", "fpr_0.10",
        ],
        help="Threshold selection method for primary metrics table.",
    )
    parser.add_argument(
        "--exclude-attacks",
        default="",
        help="Comma-separated attack labels to drop from val/test windows (e.g. BOT).",
    )
    return parser


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML configuration from *config_path*."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(args: argparse.Namespace, cfg: dict) -> pd.DataFrame:
    """Load the appropriate dataset based on CLI flags."""
    if args.use_sample:
        logger.info("Generating synthetic sample dataset …")
        sample_path = create_sample_dataset(
            n=5000, output_dir=cfg["data"]["sample_path"]
        )
        df = pd.read_csv(sample_path, low_memory=False)
        if "Timestamp" in df.columns:
            raw_ts = df["Timestamp"]
            df["Timestamp"] = pd.to_datetime(
                raw_ts, format="%d/%m/%Y %H:%M:%S", errors="coerce"
            )
            if df["Timestamp"].isna().all():
                df["Timestamp"] = pd.to_datetime(
                    raw_ts, format="%d/%m/%Y %H:%M", errors="coerce"
                )
        return df

    if args.data_dir is None:
        logger.error(
            "Provide --data-dir or use --use-sample for a quick demo."
        )
        sys.exit(1)

    if args.dataset == "unsw_nb15":
        logger.error(
            "UNSW-NB15 spectral pipeline is not yet implemented. "
            "Use --dataset cicids2017 with CICIDS2017 CSVs, or --use-sample for a demo."
        )
        sys.exit(1)

    if args.dataset == "cicids2017":
        train_day = cfg["data"].get("train_day", "Monday")
        eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
        wanted_days = list(dict.fromkeys([train_day, *eval_days]))
        processed_dir = cfg["data"].get("processed_path", "data/processed")
        return load_cicids2017(args.data_dir, days=wanted_days, processed_dir=processed_dir)

    logger.error("Unknown dataset: %s", args.dataset)
    sys.exit(1)


def _validate_preprocessor_features(
    preprocessor: DataPreprocessor,
    feature_names: list[str],
) -> None:
    """Ensure a loaded preprocessor matches the current feature matrix."""
    saved = preprocessor.feature_names_
    if saved is None:
        logger.error(
            "Loaded preprocessor has no feature_names_; retrain models instead."
        )
        sys.exit(1)
    if list(saved) != list(feature_names):
        logger.error(
            "Feature name mismatch between current data and saved preprocessor.\n"
            "  saved (%d): %s …\n"
            "  current (%d): %s …\n"
            "Re-run training or use the same config/data as when models were saved.",
            len(saved), saved[:3], len(feature_names), feature_names[:3],
        )
        sys.exit(1)


def _resolve_tuned_config(cfg: dict, args: argparse.Namespace, results_dir: Path) -> dict:
    """Apply tuned artifacts when requested or when file exists by default."""
    tuned_path = results_dir / "ensemble_weights.json"
    use_tuned = args.tuned_weights
    if use_tuned is None:
        use_tuned = not args.no_tuned_weights and tuned_path.is_file()
    if use_tuned:
        tuned = load_tuned_artifacts(tuned_path)
        if tuned:
            logger.info("Applying tuned weights from %s", tuned_path)
            return apply_tuned_config(cfg, tuned)
        logger.warning("Tuned weights requested but %s not found.", tuned_path)
    return cfg


def _run_feature_pipeline(
    args: argparse.Namespace,
    cfg: dict,
    df: pd.DataFrame,
) -> dict:
    """Build spectral splits (shared by primary and dual-eval paths)."""
    train_day = cfg["data"].get("train_day", "Monday")
    eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
    train_df = df[df["day"] == train_day].reset_index(drop=True)
    test_df = df[df["day"].isin(eval_days)].reset_index(drop=True)
    cache_file = cache_path(cfg, train_day, eval_days)
    splits = None
    if args.use_cache:
        splits = load_splits_cache(cache_file)
        if splits is not None:
            logger.info("Loaded feature cache from %s", cache_file)
    if splits is None:
        splits = build_cicids_day_splits(
            train_df, test_df, cfg,
            np.random.RandomState(42),
            dataset=args.dataset,
        )
        save_splits_cache(cache_file, {
            "X_train": splits["X_train"],
            "X_val": splits["X_val"],
            "y_val": splits["y_val"],
            "lab_val": splits["lab_val"],
            "val_timestamps": splits.get("val_timestamps", splits["test_timestamps"]),
            "X_test": splits["X_test"],
            "y_test": splits["y_test"],
            "lab_test": splits["lab_test"],
            "test_timestamps": splits["test_timestamps"],
            "feat_names": np.array(splits["feat_names"], dtype=object),
        })
        logger.info("Feature cache written to %s", cache_file)
    return splits


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    if args.skip_train and not args.load_models:
        logger.error("--skip-train requires --load-models pointing to saved model artifacts.")
        sys.exit(1)

    # ---- Load config --------------------------------------------------------
    cfg = load_config(args.config)
    output_dir  = Path(args.output_dir)
    models_dir  = output_dir / "models"
    vis_cfg = cfg["visualization"]
    figures_dir = Path(vis_cfg["figure_dir"])
    results_dir = output_dir / "metrics"

    for d in [models_dir, figures_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cfg = _resolve_tuned_config(cfg, args, results_dir)
    mcfg = cfg["models"]
    eval_cfg = cfg.get("evaluation", {})
    threshold_method = args.threshold_method or eval_cfg.get(
        "default_threshold_method", "f1_optimal"
    )
    threshold_method = normalize_threshold_method(threshold_method)

    exclude_attacks = [
        a.strip() for a in args.exclude_attacks.split(",") if a.strip()
    ]

    # ---- Load data ----------------------------------------------------------
    t0 = time.perf_counter()
    df = load_data(args, cfg)
    logger.info("Data loaded in %.1fs", time.perf_counter() - t0)

    # ---- Feature engineering + train/test split (real labels) ---------------
    t0 = time.perf_counter()
    preprocessor = DataPreprocessor()
    y_label_test: "np.ndarray | None" = None
    X_val: "np.ndarray | None" = None
    y_val: "np.ndarray | None" = None
    val_timestamps: "np.ndarray | None" = None

    use_day_split = (
        args.dataset == "cicids2017"
        and "day" in df.columns
        and "Label" in df.columns
        and not args.use_sample
    )

    if use_day_split:
        train_day = cfg["data"].get("train_day", "Monday")
        eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
        logger.info(
            "Day split — train=%s, eval=%s",
            train_day, eval_days,
        )
        splits = _run_feature_pipeline(args, cfg, df)
        if exclude_attacks:
            splits = filter_splits_by_attack_labels(splits, exclude_attacks)
        X_train = splits["X_train"]
        X_val = splits["X_val"]
        y_val = splits["y_val"]
        X_test = splits["X_test"]
        y_test = splits["y_test"]
        y_label_test = splits["lab_test"]
        feature_names = list(splits["feat_names"])
        test_timestamps = splits["test_timestamps"]
        val_timestamps = splits.get("val_timestamps", test_timestamps)
    else:
        # Sample / no-day-info path: benign-only train, disjoint holdout + attacks.
        benign_holdout_frac = float(
            cfg.get("evaluation", {}).get("benign_test_frac", 0.30)
        )
        X_all, all_ts, feature_names, y_all, y_label_all = build_subset_features(
            df, cfg, dataset=args.dataset
        )
        if y_all is None:
            y_all = np.zeros(len(X_all), dtype=int)
            y_label_all = np.array(["BENIGN"] * len(X_all), dtype=object)

        X_train, X_val, y_val, X_test, y_test, y_label_test, test_timestamps = (
            build_disjoint_window_splits(
                X_all, all_ts, y_all, y_label_all, benign_holdout_frac,
                np.random.RandomState(42),
            )
        )

    logger.info("Features built in %.1fs", time.perf_counter() - t0)

    if X_train.shape[0] == 0 or X_test.shape[0] == 0 or X_val is None or len(X_val) == 0:
        logger.error("Feature matrix is empty. Check that the dataset has "
                     "enough rows for the configured window size.")
        sys.exit(1)

    timestamps = test_timestamps

    n_val_attacks = int(y_val.sum())
    if n_val_attacks == 0 or n_val_attacks == len(y_val):
        logger.warning(
            "Validation set has only one class (%d/%d attack windows). "
            "AUROC and F1-optimal thresholds are not meaningful.",
            n_val_attacks, len(y_val),
        )

    # ---- Scale features (fit on benign-only training windows) ---------------
    if args.skip_train and args.load_models:
        preproc_path = os.path.join(args.load_models, "preprocessor.pkl")
        if not os.path.isfile(preproc_path):
            logger.error("Missing preprocessor at %s", preproc_path)
            sys.exit(1)
        preprocessor = DataPreprocessor.load(preproc_path)
        _validate_preprocessor_features(preprocessor, feature_names)
        X_train_scaled = preprocessor.transform(X_train)
        X_val_scaled   = preprocessor.transform(X_val)
        X_test_scaled  = preprocessor.transform(X_test)
    else:
        X_train_scaled = preprocessor.fit_transform(X_train, feature_names=feature_names)
        X_val_scaled   = preprocessor.transform(X_val)
        X_test_scaled  = preprocessor.transform(X_test)
        preprocessor.save(str(models_dir / "preprocessor.pkl"))

    logger.info(
        "Train: %d windows | Val: %d windows (%.1f%% atk) | "
        "Test: %d windows (%.1f%% atk)",
        len(X_train_scaled), len(X_val_scaled), 100 * y_val.mean(),
        len(X_test_scaled), 100 * y_test.mean() if len(y_test) else 0.0,
    )

    # ---- Train / load models ------------------------------------------------
    from src.models import VAETrainer, IsolationForestDetector, ECODDetector

    models = build_ensemble_models(cfg, X_train_scaled.shape[1])

    if args.skip_train and args.load_models:
        logger.info("Loading pre-trained models from %s …", args.load_models)
        models["vae"] = VAETrainer.load(os.path.join(args.load_models, "vae.pt"))
        models["isolation_forest"] = IsolationForestDetector.load(
            os.path.join(args.load_models, "if.pkl")
        )
        models["ecod"] = ECODDetector.load(os.path.join(args.load_models, "ecod.pkl"))
    else:
        train_detectors(models, X_train_scaled, cfg, X_val=X_val_scaled)
        models["vae"].save(str(models_dir / "vae.pt"))
        models["isolation_forest"].save(str(models_dir / "if.pkl"))
        models["ecod"].save(str(models_dir / "ecod.pkl"))
        if "copod" in models:
            models["copod"].save(str(models_dir / "copod.pkl"))
        if "hbos" in models:
            models["hbos"].save(str(models_dir / "hbos.pkl"))

    calibrate_all_detectors(models, X_val_scaled, y_val)

    weight_map = ensemble_weights_from_config(cfg)
    ensemble_models = {k: models[k] for k in weight_map if k in models}
    ensemble = EnsembleDetector(models=ensemble_models, weights=weight_map)

    # ---- Score validation and test sets -------------------------------------
    logger.info("Scoring validation and test sets …")
    val_individual = score_all_models(models, X_val_scaled)
    test_individual = score_all_models(models, X_test_scaled)
    val_scores = {**val_individual, "ensemble": ensemble.score_from(val_individual)}
    model_scores = {**test_individual, "ensemble": ensemble.score_from(test_individual)}

    # ---- Flow-level ECOD fusion (optional) ----------------------------------
    flow_cfg = mcfg.get("flow_ecod", {})
    flow_weight = float(flow_cfg.get("fusion_weight", 0.0))
    flow_det = None
    if flow_weight > 0 and use_day_split and not args.use_sample:
        flow_path = str(models_dir / "flow_ecod.pkl")
        fit_flow = not (args.skip_train and args.load_models)
        loaded_flow = None
        if args.skip_train and args.load_models and os.path.isfile(flow_path):
            loaded_flow = FlowECODDetector.load(flow_path)
        val_scores, model_scores, flow_det = apply_flow_ecod_fusion(
            df, cfg, val_scores, model_scores, y_val,
            val_timestamps, test_timestamps,
            fit_detector=fit_flow,
            flow_det=loaded_flow,
            save_path=flow_path if fit_flow else None,
        )
        logger.info("Fused flow-level ECOD scores (weight=%.2f)", flow_weight)

        # Flow-level per-attack metrics (individual flows, not windows)
        train_day = cfg["data"].get("train_day", "Monday")
        eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
        eval_flows = df[df["day"].isin(eval_days)]
        if flow_det is not None and "Label" in eval_flows.columns:
            flow_scores = flow_det.score_flows(eval_flows)
            flow_df = evaluate_flow_level(
                eval_flows["Label"].values, flow_scores,
            )
            if not flow_df.empty:
                flow_df.to_csv(
                    str(results_dir / "per_attack_flow_level.csv"), index=False,
                )
                logger.info(
                    "Flow-level per-attack metrics saved to %s",
                    results_dir / "per_attack_flow_level.csv",
                )

    # ---- Evaluate (thresholds chosen on validation, metrics on test) --------
    logger.info("\n=== Evaluation Results (test set, thresholds from validation) ===")
    eval_df = evaluate_all_models(
        model_scores, y_test, val_scores=val_scores, y_val=y_val,
        threshold_method=threshold_method,
    )
    print_metrics_table(eval_df)
    eval_df.to_csv(str(results_dir / "model_comparison.csv"), index=False)
    logger.info("Metrics saved to %s", results_dir / "model_comparison.csv")

    thr_methods = eval_cfg.get("threshold_methods", ["f1_optimal", "youden_j", "percentile_95"])
    thr_df = evaluate_all_threshold_methods(
        model_scores, y_test, val_scores, y_val, methods=thr_methods,
    )
    thr_df.to_csv(str(results_dir / "threshold_comparison.csv"), index=False)
    logger.info("Threshold comparison saved to %s", results_dir / "threshold_comparison.csv")

    # ---- Ensemble threshold (validation) & per-attack breakdown -------------
    ens_scores = model_scores["ensemble"]
    ens_val_scores = val_scores["ensemble"]
    opt_threshold, val_f1 = EnsembleDetector.find_optimal_threshold(
        ens_val_scores, y_val, method="f1_optimal"
    )
    logger.info(
        "Ensemble threshold (from validation): %.4f  (val F1=%.4f)",
        opt_threshold, val_f1,
    )

    # Per-attack-type metrics (using the propagated fine-grained window labels)
    if y_label_test is not None:
        per_attack_df = evaluate_per_attack_type(
            y_label_test, ens_scores, threshold=opt_threshold,
        )
        if not per_attack_df.empty:
            per_attack_df.to_csv(str(results_dir / "per_attack_metrics.csv"), index=False)
            logger.info("Per-attack metrics saved to %s",
                        results_dir / "per_attack_metrics.csv")
            print_metrics_table(per_attack_df.rename(columns={"attack_type": "model"}))

    # ---- Visualisations -----------------------------------------------------
    if not args.no_plots:
        logger.info("Generating figures …")

        # Anomaly timeline (timestamps already correspond to the test windows)
        if len(timestamps) == len(X_test_scaled):
            test_ts = timestamps
        else:
            test_ts = np.arange(len(X_test_scaled))

        fig = plot_anomaly_timeline(
            test_ts, ens_scores, y_test, threshold=opt_threshold
        )
        save_figure(fig, "anomaly_timeline", str(figures_dir), dpi=vis_cfg["dpi"])

        # Score distributions
        fig = plot_score_distributions(
            ens_scores, y_test,
            class_names=["BENIGN", "ATTACK"],
            threshold=opt_threshold,
        )
        save_figure(fig, "score_distributions", str(figures_dir), dpi=vis_cfg["dpi"])

        # Confusion matrix
        y_pred = (ens_scores >= opt_threshold).astype(int)
        fig = plot_confusion_matrix(y_test, y_pred)
        save_figure(fig, "confusion_matrix", str(figures_dir), dpi=vis_cfg["dpi"])

        # Spectrogram of the first spectral input series
        first_col = cfg["spectral"]["input_series"][0]
        try:
            ts_col_name = "Timestamp" if "Timestamp" in df.columns else "Stime"
            ts_df = build_standard_timeseries(
                df, timestamp_col=ts_col_name,
                bin_size_seconds=cfg["spectral"]["bin_size_seconds"],
            )
            if first_col in ts_df.columns:
                fig = plot_spectrogram(
                    ts_df[first_col].values,
                    fs=1.0 / cfg["spectral"]["bin_size_seconds"],
                    title=f"Spectrogram — {first_col}",
                )
                save_figure(fig, "spectrogram", str(figures_dir), dpi=vis_cfg["dpi"])
        except Exception as exc:
            logger.warning("Spectrogram generation failed: %s", exc)

        # UMAP of latent space
        try:
            latent = models["vae"].vae.get_latent(X_test_scaled)
            fig = plot_umap_latent_space(
                latent, y_test, class_names=["BENIGN", "ATTACK"]
            )
            save_figure(fig, "umap_latent", str(figures_dir), dpi=vis_cfg["dpi"])
        except Exception as exc:
            logger.warning("UMAP plot failed: %s", exc)

        logger.info("All figures saved to %s", figures_dir)

    # ---- Dual evaluation under 'any' label rule -----------------------------
    if eval_cfg.get("dual_eval", False) and use_day_split:
        logger.info("Running dual evaluation with window_label_rule='any' …")
        any_cfg = deepcopy(cfg)
        any_cfg.setdefault("evaluation", {})["window_label_rule"] = "any"
        any_splits = _run_feature_pipeline(args, any_cfg, df)
        if exclude_attacks:
            any_splits = filter_splits_by_attack_labels(any_splits, exclude_attacks)
        pre_any = DataPreprocessor()
        X_tr_a = pre_any.fit_transform(any_splits["X_train"], feature_names=list(any_splits["feat_names"]))
        X_va_a = pre_any.transform(any_splits["X_val"])
        X_te_a = pre_any.transform(any_splits["X_test"])
        any_models = build_ensemble_models(any_cfg, X_tr_a.shape[1])
        train_detectors(any_models, X_tr_a, any_cfg, X_val=X_va_a)
        calibrate_all_detectors(any_models, X_va_a, any_splits["y_val"])
        w_any = ensemble_weights_from_config(any_cfg)
        ens_any = EnsembleDetector(
            models={k: any_models[k] for k in w_any if k in any_models},
            weights=w_any,
        )
        te_scores = score_all_models(any_models, X_te_a)
        va_scores = score_all_models(any_models, X_va_a)
        te_scores["ensemble"] = ens_any.score_from(te_scores)
        va_scores["ensemble"] = ens_any.score_from(va_scores)

        any_val_ts = any_splits.get("val_timestamps", any_splits["test_timestamps"])
        te_unfused = dict(te_scores)
        va_unfused = dict(va_scores)
        if flow_weight > 0:
            va_scores, te_scores, _ = apply_flow_ecod_fusion(
                df, any_cfg, va_scores, te_scores, any_splits["y_val"],
                any_val_ts, any_splits["test_timestamps"],
                flow_weight=flow_weight,
                fit_detector=True,
            )

        per_any_unfused = evaluate_per_attack_type(
            any_splits["lab_test"], te_unfused["ensemble"],
        )
        if not per_any_unfused.empty:
            per_any_unfused.to_csv(
                str(results_dir / "per_attack_any_rule.csv"), index=False,
            )

        per_any = evaluate_per_attack_type(
            any_splits["lab_test"], te_scores["ensemble"],
        )
        if not per_any.empty:
            per_any.to_csv(
                str(results_dir / "per_attack_any_rule_fused.csv"), index=False,
            )
            logger.info(
                "Dual-eval per-attack metrics saved to %s and %s",
                results_dir / "per_attack_any_rule.csv",
                results_dir / "per_attack_any_rule_fused.csv",
            )
            print("\n=== Per-attack metrics (any label rule, fused) ===")
            print_metrics_table(per_any.rename(columns={"attack_type": "model"}))

        any_eval = evaluate_all_models(
            te_scores, any_splits["y_test"],
            val_scores=va_scores, y_val=any_splits["y_val"],
            threshold_method=threshold_method,
        )
        any_eval.to_csv(str(results_dir / "model_comparison_any_rule.csv"), index=False)
        any_thr = evaluate_all_threshold_methods(
            te_scores, any_splits["y_test"], va_scores, any_splits["y_val"],
            methods=thr_methods,
        )
        any_thr.to_csv(str(results_dir / "threshold_comparison_any_rule.csv"), index=False)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
