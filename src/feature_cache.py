"""
feature_cache.py — Persisted train/val/test spectral feature splits.

Cache files are keyed by spectral parameters, split logic, and dataset days
so stale artifacts are not reused after config or code changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Bump when split logic or spectral feature definitions change materially.
CACHE_VERSION = "v3"


def _cfg_fingerprint(cfg: dict, train_day: str, eval_days: List[str]) -> str:
    """Stable hash of all parameters that affect cached feature matrices."""
    spectral = cfg.get("spectral", {})
    evaluation = cfg.get("evaluation", {})
    payload = {
        "version": CACHE_VERSION,
        "train_day": train_day,
        "eval_days": sorted(eval_days),
        "bin_size_seconds": spectral.get("bin_size_seconds"),
        "window_size": spectral.get("window_size"),
        "overlap": spectral.get("overlap"),
        "window_function": spectral.get("window_function"),
        "rolloff_percentages": spectral.get("rolloff_percentages"),
        "input_series": spectral.get("input_series"),
        "delta_features": spectral.get("delta_features", False),
        "log_transform_channels": spectral.get("log_transform_channels"),
        "multi_scale": spectral.get("multi_scale"),
        "window_label_rule": evaluation.get("window_label_rule", "any"),
        "benign_test_frac": evaluation.get("benign_test_frac", 0.30),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cache_path(
    cfg: dict,
    train_day: Optional[str] = None,
    eval_days: Optional[List[str]] = None,
    processed_dir: str = "data/processed",
) -> Path:
    """Return the NPZ path for a given configuration."""
    train_day = train_day or cfg.get("data", {}).get("train_day", "Monday")
    eval_days = eval_days or cfg.get("evaluation", {}).get("eval_days", ["Friday"])
    fp = _cfg_fingerprint(cfg, train_day, list(eval_days))
    rule = cfg.get("evaluation", {}).get("window_label_rule", "any")
    return Path(processed_dir) / f"splits_cache_{CACHE_VERSION}_{rule}_{fp}.npz"


def load_splits_cache(path: Path) -> Optional[Dict[str, Any]]:
    """Load cached splits dict or return None if missing."""
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def save_splits_cache(path: Path, splits: Dict[str, Any]) -> None:
    """Write splits arrays to an NPZ cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **splits)
