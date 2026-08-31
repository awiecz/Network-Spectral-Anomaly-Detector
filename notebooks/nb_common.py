"""
nb_common.py — Shared utilities for Jupyter notebooks.

Mirrors main.py / tune.py data loading, splitting, scaling, and artifact I/O
so notebooks stay aligned with the CLI pipeline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

# Project root (parent of notebooks/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import create_sample_dataset, load_cicids2017
from src.feature_cache import cache_path, load_splits_cache, save_splits_cache
from src.preprocessor import DataPreprocessor
from src.splits import (
    build_cicids_day_splits,
    build_subset_features,
    make_val_test_splits,
)
from src.tuning_config import apply_tuned_config, load_tuned_artifacts

RNG = np.random.RandomState(42)
NOTEBOOK_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "notebook_scores.joblib"
NOTEBOOK_SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "notebook_splits.joblib"


@dataclass
class NotebookSettings:
    """Runtime flags shared across notebooks."""

    use_sample: bool = True
    use_cache: bool = True
    apply_tuned: bool = True
    data_dir: str = "data/raw/cicids2017"
    config_path: str = "config/config.yaml"
    output_dir: str = "results"
    dataset: str = "cicids2017"

    def resolve_config_path(self) -> Path:
        p = Path(self.config_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def resolve_output_dir(self) -> Path:
        p = Path(self.output_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def resolve_data_dir(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


def setup_notebook(cfg: Optional[dict] = None) -> None:
    """Add project root to sys.path and apply matplotlib style from config."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    style = "seaborn-v0_8-whitegrid"
    if cfg:
        style = cfg.get("visualization", {}).get("style", style)
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")


def load_project_config(
    settings: NotebookSettings,
) -> dict:
    """Load config.yaml and optionally merge tuned ensemble_weights.json."""
    config_path = settings.resolve_config_path()
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if settings.apply_tuned:
        tuned_path = settings.resolve_output_dir() / "metrics" / "ensemble_weights.json"
        tuned = load_tuned_artifacts(tuned_path)
        if tuned:
            cfg = apply_tuned_config(cfg, tuned)
    return cfg


def load_dataframe(settings: NotebookSettings, cfg: dict) -> pd.DataFrame:
    """Load sample or CICIDS2017 data (same days as main.py)."""
    if settings.use_sample:
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

    train_day = cfg["data"].get("train_day", "Monday")
    eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
    wanted_days = list(dict.fromkeys([train_day, *eval_days]))
    processed_dir = cfg["data"].get("processed_path", "data/processed")
    return load_cicids2017(
        str(settings.resolve_data_dir()),
        days=wanted_days,
        processed_dir=processed_dir,
    )


def _use_day_split(settings: NotebookSettings, df: pd.DataFrame) -> bool:
    return (
        settings.dataset == "cicids2017"
        and "day" in df.columns
        and "Label" in df.columns
        and not settings.use_sample
    )


def _build_sample_splits_dict(
    df: pd.DataFrame,
    cfg: dict,
) -> dict:
    """Sample / no-day-info path with full lab_val and val_timestamps."""
    benign_holdout_frac = float(
        cfg.get("evaluation", {}).get("benign_test_frac", 0.30)
    )
    X_all, all_ts, feature_names, y_all, y_label_all = build_subset_features(
        df, cfg, dataset="cicids2017"
    )
    if y_all is None:
        y_all = np.zeros(len(X_all), dtype=int)
        y_label_all = np.array(["BENIGN"] * len(X_all), dtype=object)

    benign_idx = np.flatnonzero(y_all == 0)
    attack_idx = np.flatnonzero(y_all == 1)
    time_order = benign_idx[np.argsort(all_ts[benign_idx])]
    n_holdout = max(2, int(benign_holdout_frac * len(time_order)))
    n_train = max(1, len(time_order) - n_holdout)
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

    X_val, y_val, lab_val, val_timestamps, X_test, y_test, lab_test, test_timestamps = (
        make_val_test_splits(
            X_benign_holdout, ts_benign_holdout, lbl_benign_holdout,
            X_attack, attack_ts, y_attack, lbl_attack,
            RNG,
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


def build_splits(
    settings: NotebookSettings,
    cfg: dict,
    df: pd.DataFrame,
    *,
    cfg_override: Optional[dict] = None,
) -> dict:
    """Build train/val/test spectral window splits (mirrors main.py)."""
    active_cfg = cfg_override if cfg_override is not None else cfg

    if _use_day_split(settings, df):
        train_day = active_cfg["data"].get("train_day", "Monday")
        eval_days = active_cfg.get("evaluation", {}).get("eval_days", ["Friday"])
        processed_dir = active_cfg["data"].get("processed_path", "data/processed")
        cache_file = cache_path(active_cfg, train_day, eval_days, processed_dir)

        if settings.use_cache:
            cached = load_splits_cache(cache_file)
            if cached is not None:
                return _cache_to_splits(cached)

        train_df = df[df["day"] == train_day].reset_index(drop=True)
        test_df = df[df["day"].isin(eval_days)].reset_index(drop=True)
        splits = build_cicids_day_splits(
            train_df, test_df, active_cfg, RNG, dataset=settings.dataset
        )

        if settings.use_cache:
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
        return splits

    splits = _build_sample_splits_dict(df, active_cfg)
    if settings.use_cache:
        NOTEBOOK_SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(splits, NOTEBOOK_SPLITS_PATH)
    return splits


def _cache_to_splits(cached: dict) -> dict:
    feat_names = cached.get("feat_names")
    if feat_names is not None and hasattr(feat_names, "tolist"):
        feat_names = list(feat_names)
    return {
        "X_train": cached["X_train"],
        "X_val": cached["X_val"],
        "y_val": cached["y_val"],
        "lab_val": cached.get("lab_val"),
        "val_timestamps": cached.get("val_timestamps", cached.get("test_timestamps")),
        "X_test": cached["X_test"],
        "y_test": cached["y_test"],
        "lab_test": cached["lab_test"],
        "test_timestamps": cached["test_timestamps"],
        "feat_names": list(feat_names) if feat_names is not None else [],
    }


def load_cached_splits(settings: NotebookSettings, cfg: dict) -> Optional[dict]:
    """Try loading splits from NPZ cache (real data) or joblib (sample)."""
    if not settings.use_sample:
        train_day = cfg["data"].get("train_day", "Monday")
        eval_days = cfg.get("evaluation", {}).get("eval_days", ["Friday"])
        processed_dir = cfg["data"].get("processed_path", "data/processed")
        cache_file = cache_path(cfg, train_day, eval_days, processed_dir)
        cached = load_splits_cache(cache_file)
        if cached is not None:
            return _cache_to_splits(cached)
    if settings.use_sample and NOTEBOOK_SPLITS_PATH.is_file():
        return joblib.load(NOTEBOOK_SPLITS_PATH)
    return None


def scale_splits(
    splits: dict,
    models_dir: Path,
    *,
    load_existing: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataPreprocessor]:
    """Fit RobustScaler on train windows; transform val/test."""
    models_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(splits["feat_names"])
    preproc_path = models_dir / "preprocessor.pkl"

    if load_existing and preproc_path.is_file():
        preprocessor = DataPreprocessor.load(str(preproc_path))
        X_train_scaled = preprocessor.transform(splits["X_train"])
        X_val_scaled = preprocessor.transform(splits["X_val"])
        X_test_scaled = preprocessor.transform(splits["X_test"])
        return X_train_scaled, X_val_scaled, X_test_scaled, preprocessor

    preprocessor = DataPreprocessor()
    X_train_scaled = preprocessor.fit_transform(
        splits["X_train"], feature_names=feature_names
    )
    X_val_scaled = preprocessor.transform(splits["X_val"])
    X_test_scaled = preprocessor.transform(splits["X_test"])
    preprocessor.save(str(preproc_path))
    return X_train_scaled, X_val_scaled, X_test_scaled, preprocessor


def save_notebook_artifacts(
    payload: dict,
    path: Optional[Path] = None,
) -> Path:
    """Persist scores, labels, and metadata between notebooks."""
    out = path or NOTEBOOK_SCORES_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    return out


def load_notebook_artifacts(path: Optional[Path] = None) -> dict:
    """Load artifacts saved by save_notebook_artifacts."""
    p = path or NOTEBOOK_SCORES_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"Notebook artifacts not found at {p}. Run notebook 03 first."
        )
    return joblib.load(p)


def split_summary(splits: dict) -> str:
    """Human-readable summary of split sizes."""
    lines = [
        f"Train : {len(splits['X_train'])} windows (benign only)",
        f"Val   : {len(splits['X_val'])} windows "
        f"({100 * splits['y_val'].mean():.1f}% attack)",
        f"Test  : {len(splits['X_test'])} windows "
        f"({100 * splits['y_test'].mean():.1f}% attack)",
        f"Features: {len(splits['feat_names'])}",
    ]
    return "\n".join(lines)
