"""
visualization.py — Plotting utilities for the Network Spectral Anomaly Detector.

All public functions return a :class:`matplotlib.figure.Figure` object so
that callers control whether the figure is shown, saved, or embedded in a
report.  Call :func:`save_figure` to write both PNG and SVG outputs.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.signal import spectrogram as scipy_spectrogram
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)

# Use Agg backend when no display is available (headless servers / CI)
matplotlib.use("Agg")


def _apply_style() -> None:
    """Apply the project's default matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")


# ---------------------------------------------------------------------------
# Anomaly timeline
# ---------------------------------------------------------------------------

def plot_anomaly_timeline(
    timestamps: np.ndarray,
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    attack_periods: Optional[List[tuple]] = None,
    title: str = "Anomaly Score Timeline",
) -> Figure:
    """Plot anomaly scores over time with a threshold line and attack highlights.

    Parameters
    ----------
    timestamps:
        1-D array of timestamps (datetime or numeric).
    scores:
        1-D anomaly score array aligned with *timestamps*.
    y_true:
        1-D binary ground-truth (0 = normal, 1 = attack).
    threshold:
        Decision threshold drawn as a horizontal line.
    attack_periods:
        Optional list of (start, end) pairs for shading known attack windows.
    title:
        Figure title.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 4))

    # Shade ground-truth attack regions lightly
    in_attack = False
    start_idx = 0
    for i, label in enumerate(y_true):
        if label == 1 and not in_attack:
            in_attack  = True
            start_idx  = i
        elif label == 0 and in_attack:
            ax.axvspan(timestamps[start_idx], timestamps[i],
                       color="salmon", alpha=0.25, label="_nolegend_")
            in_attack = False
    if in_attack:
        ax.axvspan(timestamps[start_idx], timestamps[-1],
                   color="salmon", alpha=0.25, label="Attack (ground truth)")

    # Optional explicit attack period overlays
    if attack_periods:
        for (t_start, t_end) in attack_periods:
            ax.axvspan(t_start, t_end, color="red", alpha=0.15)

    ax.plot(timestamps, scores, linewidth=0.8, color="steelblue", label="Anomaly score")
    ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2, label=f"Threshold ({threshold:.3f})")

    ax.set_xlabel("Time")
    ax.set_ylabel("Anomaly Score")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------

def plot_spectrogram(
    timeseries: np.ndarray,
    fs: float = 1.0,
    title: str = "Network Traffic Spectrogram",
) -> Figure:
    """Plot a spectrogram (STFT magnitude in dB) using an inferno colormap.

    Parameters
    ----------
    timeseries:
        1-D univariate time series (e.g. flow count per second).
    fs:
        Sampling frequency in Hz.
    title:
        Figure title.
    """
    _apply_style()
    nperseg = min(64, len(timeseries) // 4)
    if nperseg < 4:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Insufficient data for spectrogram",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig

    freqs, times, Sxx = scipy_spectrogram(
        timeseries, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
    )
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 5))
    pcm = ax.pcolormesh(times, freqs, Sxx_dB, cmap="inferno", shading="gouraud")
    fig.colorbar(pcm, ax=ax, label="Power (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UMAP latent space
# ---------------------------------------------------------------------------

def plot_umap_latent_space(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str = "UMAP of Spectral Feature Space",
    random_state: int = 42,
) -> Figure:
    """Plot a 2-D UMAP embedding of *latent_vectors* coloured by class.

    Parameters
    ----------
    latent_vectors:
        2-D array of shape (n_samples, latent_dim).
    labels:
        1-D integer label array aligned with *latent_vectors*.
    class_names:
        List of class names indexed by integer label value.
    title:
        Figure title.
    random_state:
        UMAP random seed.
    """
    try:
        import umap  # type: ignore
    except ImportError:
        logger.warning("umap-learn is not installed. Falling back to PCA for 2-D projection.")
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        embedding = reducer.fit_transform(latent_vectors)
        proj_method = "PCA"
    else:
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        embedding = reducer.fit_transform(latent_vectors)
        proj_method = "UMAP"

    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")

    unique_labels = np.unique(labels)
    for idx in unique_labels:
        mask = labels == idx
        name = class_names[idx] if idx < len(class_names) else str(idx)
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=6, alpha=0.6, color=cmap(idx % 10), label=name,
        )

    ax.set_xlabel(f"{proj_method} 1")
    ax.set_ylabel(f"{proj_method} 2")
    ax.set_title(title)
    ax.legend(markerscale=2.5, fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Score distributions
# ---------------------------------------------------------------------------

def plot_score_distributions(
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    threshold: Optional[float] = None,
) -> Figure:
    """KDE plot of anomaly score distributions per class.

    Parameters
    ----------
    scores:
        1-D anomaly score array.
    labels:
        1-D integer label array.
    class_names:
        List of class names indexed by integer label.
    threshold:
        Optional threshold drawn as a vertical dashed line.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap    = plt.get_cmap("tab10")

    for idx in np.unique(labels):
        mask = labels == idx
        name = class_names[idx] if idx < len(class_names) else str(idx)
        sns.kdeplot(
            scores[mask], ax=ax, label=name, color=cmap(idx % 10),
            fill=True, alpha=0.3, linewidth=1.5,
        )

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--",
                   linewidth=1.5, label=f"Threshold = {threshold:.3f}")

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution by Class")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-attack ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_label: np.ndarray,
    scores_dict: Dict[str, np.ndarray],
    attack_types: Optional[List[str]] = None,
    benign_label: str = "BENIGN",
) -> Figure:
    """Plot per-attack ROC curves for each model in *scores_dict*.

    One subplot per attack type, one line per model.

    Parameters
    ----------
    y_label:
        1-D string label array.
    scores_dict:
        Mapping of model name → 1-D anomaly score array.
    attack_types:
        Attack type strings to include.  Defaults to all non-benign labels.
    benign_label:
        String denoting normal traffic.
    """
    _apply_style()
    y_label = np.asarray(y_label)

    if attack_types is None:
        attack_types = sorted([
            t for t in np.unique(y_label) if t.upper() != benign_label.upper()
        ])

    n_attacks = len(attack_types)
    if n_attacks == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No attack types found", ha="center", va="center")
        return fig

    ncols = min(3, n_attacks)
    nrows = (n_attacks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    cmap = plt.get_cmap("Set2")
    model_names = list(scores_dict.keys())

    for ax_idx, attack in enumerate(attack_types):
        ax = axes[ax_idx]
        mask = (y_label == benign_label) | (y_label == attack)
        y_sub = (y_label[mask] != benign_label).astype(int)

        for m_idx, m_name in enumerate(model_names):
            s_sub = scores_dict[m_name][mask]
            if len(np.unique(y_sub)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_sub, s_sub)
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=cmap(m_idx % 8),
                    linewidth=1.5, label=f"{m_name} (AUC={auc_val:.2f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title(attack, fontsize=9)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.legend(fontsize=7)

    # Hide unused subplots
    for ax_idx in range(n_attacks, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("Per-Attack ROC Curves", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SHAP summary
# ---------------------------------------------------------------------------

def plot_shap_summary(
    model,
    X: np.ndarray,
    feature_names: List[str],
    max_display: int = 20,
) -> Figure:
    """Generate a SHAP beeswarm summary plot.

    Parameters
    ----------
    model:
        A fitted sklearn-compatible model with ``predict`` or
        ``decision_function``.
    X:
        2-D float array of shape (n_samples, n_features).
    feature_names:
        Feature name strings (length must equal ``X.shape[1]``).
    max_display:
        Maximum number of features to display.
    """
    import shap  # type: ignore

    _apply_style()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception:
        explainer   = shap.KernelExplainer(
            model.predict if hasattr(model, "predict") else model.decision_function,
            shap.sample(X, min(100, len(X))),
        )
        shap_values = explainer.shap_values(X[:200])

    # shap_values can be a list for multi-class — take the anomaly class
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    fig = plt.figure(figsize=(10, max_display * 0.35 + 2))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
) -> Figure:
    """Plot a normalised confusion matrix heatmap.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.
    class_names:
        List of class names; defaults to ``['BENIGN', 'ATTACK']``.
    """
    if class_names is None:
        class_names = ["BENIGN", "ATTACK"]

    from sklearn.metrics import confusion_matrix as sk_cm

    _apply_style()
    cm = sk_cm(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar=True,
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix (Normalised)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

def save_figure(
    fig: Figure,
    filename: str,
    output_dir: str = "results/figures",
    dpi: int = 150,
) -> None:
    """Save *fig* as both PNG and SVG to *output_dir*.

    Parameters
    ----------
    fig:
        Matplotlib Figure to save.
    filename:
        Base filename without extension.
    output_dir:
        Directory in which to write files.
    dpi:
        Resolution for the PNG output.
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, filename)
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    logger.info("Figure saved: %s.png / %s.svg", stem, stem)
    plt.close(fig)
