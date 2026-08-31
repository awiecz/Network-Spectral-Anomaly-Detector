"""Smoke-test the refreshed notebook pipeline (sample mode). Run from repo root."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import evaluate_all_models
from src.models import EnsembleDetector
from src.pipeline_helpers import (
    build_ensemble_models,
    calibrate_all_detectors,
    ensemble_weights_from_config,
    optimise_weights,
    score_all_models,
    train_detectors,
)
from nb_common import (
    NotebookSettings,
    build_splits,
    load_dataframe,
    load_project_config,
    save_notebook_artifacts,
    scale_splits,
    setup_notebook,
    split_summary,
)


def main() -> None:
    settings = NotebookSettings(use_sample=True, use_cache=True)
    cfg = load_project_config(settings)
    setup_notebook(cfg)

    print("=== 02 spectral splits ===")
    df = load_dataframe(settings, cfg)
    splits = build_splits(settings, cfg, df)
    print(split_summary(splits))
    assert splits["X_train"].shape[0] > 0
    assert splits["X_test"].shape[0] > 0

    print("=== 03 training ===")
    models_dir = settings.resolve_output_dir() / "models"
    X_train, X_val, X_test, _ = scale_splits(splits, models_dir)
    models = build_ensemble_models(cfg, X_train.shape[1])
    train_detectors(models, X_train, cfg, X_val=X_val)
    calibrate_all_detectors(models, X_val, splits["y_val"])
    val_scores = score_all_models(models, X_val)
    test_scores = score_all_models(models, X_test)
    save_notebook_artifacts({
        "val_scores": val_scores,
        "test_scores": test_scores,
        "y_val": splits["y_val"],
        "y_test": splits["y_test"],
        "lab_val": splits.get("lab_val"),
        "lab_test": splits["lab_test"],
        "test_timestamps": splits["test_timestamps"],
        "val_timestamps": splits.get("val_timestamps", splits["test_timestamps"]),
        "feat_names": splits["feat_names"],
        "settings": settings,
    })

    print("=== 04 ensemble ===")
    weight_map = ensemble_weights_from_config(cfg)
    ens = EnsembleDetector(models={k: None for k in weight_map}, weights=weight_map)
    val_scores["ensemble"] = ens.score_from({k: val_scores[k] for k in weight_map})
    test_scores["ensemble"] = ens.score_from({k: test_scores[k] for k in weight_map})
    opt_w, opt_score = optimise_weights(
        {k: v for k, v in val_scores.items() if k != "ensemble"},
        splits["y_val"],
    )
    print("optimise_weights:", {k: v for k, v in opt_w.items() if v > 0}, "score=", opt_score)

    eval_df = evaluate_all_models(
        test_scores, splits["y_test"], val_scores=val_scores, y_val=splits["y_val"],
    )
    ens_auroc = float(eval_df.loc[eval_df["model"] == "ensemble", "auroc"].iloc[0])
    print("ensemble AUROC:", ens_auroc)
    assert ens_auroc > 0.5, f"expected AUROC > 0.5, got {ens_auroc}"

    print("=== 05 figures (smoke) ===")
    from src.visualization import plot_score_distributions, save_figure

    fig = plot_score_distributions(
        test_scores["ensemble"], splits["y_test"], class_names=["BENIGN", "ATTACK"],
    )
    fig_dir = Path(cfg["visualization"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, "notebook_smoke_test", str(fig_dir), dpi=100)
    plt.close("all")
    print("OK — notebook pipeline smoke test passed")


if __name__ == "__main__":
    main()
