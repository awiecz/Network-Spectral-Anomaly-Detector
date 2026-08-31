"""
tuning_config.py — Load tuned ensemble weights and model hyperparameters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_TUNED_PATH = Path("results/metrics/ensemble_weights.json")


def load_tuned_artifacts(path: Path | str | None = None) -> Optional[Dict[str, Any]]:
    """Load ensemble_weights.json if it exists."""
    p = Path(path) if path else DEFAULT_TUNED_PATH
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def apply_tuned_config(cfg: dict, tuned: Dict[str, Any]) -> dict:
    """Merge tuned weights and best_configs into a copy of *cfg*."""
    out = json.loads(json.dumps(cfg))  # deep copy via JSON
    weights = tuned.get("weights", {})
    if weights:
        out.setdefault("models", {}).setdefault("ensemble", {})["weights"] = weights

    best = tuned.get("best_configs", {})
    models = out.setdefault("models", {})
    if "vae" in best:
        models.setdefault("vae", {}).update(best["vae"])
    if "isolation_forest" in best:
        models.setdefault("isolation_forest", {}).update(best["isolation_forest"])
    if "ecod" in best:
        models.setdefault("ecod", {}).update(best["ecod"])
    if "copod" in best:
        models.setdefault("copod", {}).update(best["copod"])
    if "hbos" in best:
        models.setdefault("hbos", {}).update(best["hbos"])
    flow = tuned.get("flow_ecod", {})
    if flow:
        models.setdefault("flow_ecod", {}).update(flow)
    return out
