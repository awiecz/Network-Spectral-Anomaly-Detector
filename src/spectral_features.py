"""
Spectral Feature Engineering for Network Flow Anomaly Detection
==============================================================
Converts temporal network flow statistics into frequency-domain
representations using FFT, enabling detection of periodic attack patterns.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import get_window

logger = logging.getLogger(__name__)

_EPSILON = 1e-10  # numerical guard against log(0) and division by zero
_N_ENERGY_BANDS = 3

# Window-function / FFT coefficient caches (keyed by n, fs, window_fn).
_WINDOW_CACHE: Dict[Tuple[int, float, str], np.ndarray] = {}
_FREQ_CACHE: Dict[Tuple[int, float], np.ndarray] = {}


# ---------------------------------------------------------------------------
# Time-series construction
# ---------------------------------------------------------------------------

def bin_flows_to_timeseries(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    feature_cols: Optional[List[str]] = None,
    bin_size_seconds: int = 1,
) -> pd.DataFrame:
    """Aggregate per-flow records into fixed-size time bins."""
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    if ts.isna().all():
        raise ValueError(
            f"Column '{timestamp_col}' could not be parsed as datetime. "
            "Ensure load_cicids2017() was called before bin_flows_to_timeseries()."
        )

    if feature_cols is None:
        exclude = {timestamp_col, "Label", "label", "day", "Flow ID",
                   "Source IP", "Destination IP", "src_ip", "dst_ip"}
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

    freq_str = f"{bin_size_seconds}s"
    df_work = df[feature_cols].copy()
    df_work.index = ts
    df_work = df_work.sort_index()

    flow_counts = df_work.resample(freq_str).size().rename("flow_count")
    aggregated = df_work.resample(freq_str).sum()
    aggregated["flow_count"] = flow_counts
    return aggregated.fillna(0)


_STD_CHANNEL_SOURCES = {
    "total_bytes":   ["Total Length of Fwd Packets", "Total Length of Bwd Packets"],
    "total_packets": ["Total Fwd Packets", "Total Backward Packets"],
    "mean_duration": ["Flow Duration"],
}


def build_standard_timeseries(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    bin_size_seconds: int = 1,
) -> pd.DataFrame:
    """Build the four semantic channels described in the README / ``config.yaml``."""
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    if ts.isna().all():
        raise ValueError(
            f"Column '{timestamp_col}' could not be parsed as datetime. "
            "Ensure load_cicids2017() was called before build_standard_timeseries()."
        )

    order = np.argsort(ts.values, kind="mergesort")
    ts_sorted = ts.iloc[order]
    df_sorted = df.iloc[order]
    n = len(df_sorted)

    def _sum_cols(cols: List[str]) -> np.ndarray:
        total = np.zeros(n, dtype=float)
        for c in cols:
            if c in df_sorted.columns:
                total = total + pd.to_numeric(df_sorted[c], errors="coerce").fillna(0.0).to_numpy()
        return total

    work = pd.DataFrame(index=ts_sorted.values)
    work["__bytes"] = _sum_cols(_STD_CHANNEL_SOURCES["total_bytes"])
    work["__packets"] = _sum_cols(_STD_CHANNEL_SOURCES["total_packets"])
    work["__duration"] = _sum_cols(_STD_CHANNEL_SOURCES["mean_duration"])

    freq_str = f"{bin_size_seconds}s"
    grouped = work.resample(freq_str)
    out = pd.DataFrame({
        "flow_count":    grouped.size(),
        "total_bytes":   grouped["__bytes"].sum(),
        "total_packets": grouped["__packets"].sum(),
        "mean_duration": grouped["__duration"].mean(),
    })
    return out.fillna(0.0)


def apply_log_transform(
    timeseries_df: pd.DataFrame,
    channels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply log1p to selected channels to compress dynamic range."""
    out = timeseries_df.copy()
    if not channels:
        return out
    for col in channels:
        if col in out.columns:
            out[col] = np.log1p(np.clip(out[col].to_numpy(dtype=float), 0.0, None))
    return out


def _delta_descriptor_names(series_names: List[str]) -> List[str]:
    names: List[str] = []
    for sname in series_names:
        names.append(f"{sname}__delta_mean")
        names.append(f"{sname}__delta_std")
    return names


def _compute_delta_features(segment: np.ndarray) -> Tuple[float, float]:
    """Mean and std of first differences within a window segment."""
    seg = np.asarray(segment, dtype=float)
    if len(seg) < 2:
        return 0.0, 0.0
    d = np.diff(seg)
    return float(np.mean(d)), float(np.std(d))


# ---------------------------------------------------------------------------
# Window / FFT helpers
# ---------------------------------------------------------------------------

def _get_window_coeffs(n: int, window_fn: str) -> np.ndarray:
    key = (n, window_fn)
    cached = _WINDOW_CACHE.get(key)
    if cached is not None:
        return cached
    if window_fn == "hann":
        coeffs = get_window("hann", n)
    elif window_fn in ("none", "rectangular"):
        coeffs = np.ones(n, dtype=float)
    else:
        try:
            coeffs = get_window(window_fn, n)
        except Exception:
            logger.warning("Unknown window '%s', falling back to Hann.", window_fn)
            coeffs = get_window("hann", n)
    _WINDOW_CACHE[key] = coeffs
    return coeffs


def _get_rfft_freqs(n: int, fs: float) -> np.ndarray:
    key = (n, fs)
    cached = _FREQ_CACHE.get(key)
    if cached is not None:
        return cached
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    _FREQ_CACHE[key] = freqs
    return freqs


def apply_hann_window(series: np.ndarray) -> np.ndarray:
    """Multiply *series* by a Hann window to reduce spectral leakage."""
    n = len(series)
    if n == 0:
        return series.copy()
    return series * _get_window_coeffs(n, "hann")


# ---------------------------------------------------------------------------
# Individual spectral descriptors (scalar)
# ---------------------------------------------------------------------------

def compute_spectral_entropy(psd: np.ndarray) -> float:
    psd = np.asarray(psd, dtype=float)
    total = psd.sum()
    if total <= _EPSILON:
        return 0.0
    p = np.clip(psd / total, _EPSILON, None)
    return float(-np.sum(p * np.log(p)))


def compute_spectral_centroid(freqs: np.ndarray, psd: np.ndarray) -> float:
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    total = psd.sum()
    if total <= _EPSILON:
        return 0.0
    return float(np.sum(freqs * psd) / total)


def compute_spectral_flatness(psd: np.ndarray) -> float:
    psd = np.asarray(psd, dtype=float)
    psd = np.clip(psd, _EPSILON, None)
    if len(psd) == 0:
        return 0.0
    log_mean = np.mean(np.log(psd))
    arithmetic_mean = np.mean(psd)
    if arithmetic_mean <= _EPSILON:
        return 0.0
    return float(np.exp(log_mean) / arithmetic_mean)


def compute_spectral_rolloff(
    freqs: np.ndarray,
    psd: np.ndarray,
    percentages: List[float],
) -> List[float]:
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    total = psd.sum()
    if total <= _EPSILON:
        return [0.0] * len(percentages)
    cumulative = np.cumsum(psd) / total
    rolloffs: List[float] = []
    for pct in percentages:
        idx = min(int(np.searchsorted(cumulative, pct)), len(freqs) - 1)
        rolloffs.append(float(freqs[idx]))
    return rolloffs


def compute_dominant_frequency(
    freqs: np.ndarray, psd: np.ndarray
) -> Tuple[float, float]:
    psd = np.asarray(psd, dtype=float)
    if psd.sum() <= _EPSILON:
        return 0.0, 0.0
    start_idx = 1 if len(psd) > 1 else 0
    peak_idx = start_idx + int(np.argmax(psd[start_idx:]))
    return float(freqs[peak_idx]), float(psd[peak_idx])


def compute_spectral_bandwidth(
    freqs: np.ndarray,
    psd: np.ndarray,
    centroid: Optional[float] = None,
) -> float:
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    total = psd.sum()
    if total <= _EPSILON:
        return 0.0
    if centroid is None:
        centroid = compute_spectral_centroid(freqs, psd)
    variance = np.sum(psd * (freqs - centroid) ** 2) / total
    return float(np.sqrt(variance))


def compute_energy_bands(psd: np.ndarray, n_bands: int = 3) -> List[float]:
    psd = np.asarray(psd, dtype=float)
    n = len(psd)
    total = psd.sum()
    if total <= _EPSILON or n == 0:
        return [0.0] * n_bands
    band_size = n / n_bands
    energies: List[float] = []
    for i in range(n_bands):
        start = int(round(i * band_size))
        end = int(round((i + 1) * band_size))
        energies.append(float(psd[start:end].sum() / total))
    return energies


def compute_dc_component(series: np.ndarray) -> float:
    series = np.asarray(series, dtype=float)
    if len(series) == 0:
        return 0.0
    return float(np.mean(series))


# ---------------------------------------------------------------------------
# Batched spectral descriptors (n_windows, n_freq) or (n_windows,)
# ---------------------------------------------------------------------------

def _batch_spectral_entropy(psd: np.ndarray) -> np.ndarray:
    total = psd.sum(axis=-1)
    out = np.zeros(psd.shape[0], dtype=float)
    valid = total > _EPSILON
    if not np.any(valid):
        return out
    p = np.clip(psd[valid] / total[valid, None], _EPSILON, None)
    out[valid] = -np.sum(p * np.log(p), axis=-1)
    return out


def _batch_spectral_centroid(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    total = psd.sum(axis=-1)
    out = np.zeros(psd.shape[0], dtype=float)
    valid = total > _EPSILON
    if np.any(valid):
        out[valid] = np.sum(freqs * psd[valid], axis=-1) / total[valid]
    return out


def _batch_spectral_flatness(psd: np.ndarray) -> np.ndarray:
    clipped = np.clip(psd, _EPSILON, None)
    log_mean = np.mean(np.log(clipped), axis=-1)
    arithmetic_mean = np.mean(clipped, axis=-1)
    out = np.exp(log_mean) / arithmetic_mean
    out[arithmetic_mean <= _EPSILON] = 0.0
    return out


def _batch_spectral_rolloff(
    freqs: np.ndarray, psd: np.ndarray, percentages: List[float]
) -> List[np.ndarray]:
    total = psd.sum(axis=-1)
    n = psd.shape[0]
    results: List[np.ndarray] = []
    cumulative = np.cumsum(psd, axis=-1)
    n_freq = len(freqs)
    for pct in percentages:
        rolloff = np.zeros(n, dtype=float)
        valid = total > _EPSILON
        if np.any(valid):
            normed = cumulative[valid] / total[valid, None]
            for row_i, row in enumerate(normed):
                idx = min(int(np.searchsorted(row, pct)), n_freq - 1)
                rolloff[np.flatnonzero(valid)[row_i]] = freqs[idx]
        results.append(rolloff)
    return results


def _batch_dominant_frequency(
    freqs: np.ndarray, psd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n = psd.shape[0]
    dom_freq = np.zeros(n, dtype=float)
    dom_amp = np.zeros(n, dtype=float)
    total = psd.sum(axis=-1)
    valid = total > _EPSILON
    if not np.any(valid):
        return dom_freq, dom_amp
    start_idx = 1 if psd.shape[-1] > 1 else 0
    sub = psd[valid, start_idx:]
    peak_local = np.argmax(sub, axis=-1)
    peak_idx = start_idx + peak_local
    dom_freq[valid] = freqs[peak_idx]
    dom_amp[valid] = psd[valid, peak_idx]
    return dom_freq, dom_amp


def _batch_spectral_bandwidth(
    freqs: np.ndarray, psd: np.ndarray, centroid: np.ndarray
) -> np.ndarray:
    total = psd.sum(axis=-1)
    out = np.zeros(psd.shape[0], dtype=float)
    valid = total > _EPSILON
    if np.any(valid):
        diff = freqs - centroid[valid, None]
        variance = np.sum(psd[valid] * diff ** 2, axis=-1) / total[valid]
        out[valid] = np.sqrt(variance)
    return out


def _batch_energy_bands(psd: np.ndarray, n_bands: int = 3) -> List[np.ndarray]:
    n_freq = psd.shape[-1]
    total = psd.sum(axis=-1)
    n_win = psd.shape[0]
    results = [np.zeros(n_win, dtype=float) for _ in range(n_bands)]
    valid = total > _EPSILON
    if not np.any(valid):
        return results
    band_size = n_freq / n_bands
    for i in range(n_bands):
        start = int(round(i * band_size))
        end = int(round((i + 1) * band_size))
        band_energy = psd[valid, start:end].sum(axis=-1) / total[valid]
        arr = np.zeros(n_win, dtype=float)
        arr[valid] = band_energy
        results[i] = arr
    return results


def _apply_one_sided_psd_scaling(psd: np.ndarray, n: int) -> None:
    if n % 2 == 0:
        psd[..., 1:-1] *= 2
    else:
        psd[..., 1:] *= 2


# ---------------------------------------------------------------------------
# Master feature extraction (single window)
# ---------------------------------------------------------------------------

def extract_spectral_features(
    series: np.ndarray,
    fs: float = 1.0,
    window_fn: str = "hann",
    rolloff_pcts: List[float] = None,
) -> Dict[str, float]:
    if rolloff_pcts is None:
        rolloff_pcts = [0.85, 0.95]

    series = np.asarray(series, dtype=float)
    n = len(series)

    if n == 0 or not np.isfinite(series).any():
        return _zero_features(rolloff_pcts, n_bands=_N_ENERGY_BANDS)

    series = np.where(np.isfinite(series), series, 0.0)
    dc = compute_dc_component(series)

    if window_fn == "hann":
        windowed = series * _get_window_coeffs(n, "hann")
    elif window_fn in ("none", "rectangular"):
        windowed = series.copy()
    else:
        windowed = series * _get_window_coeffs(n, window_fn)

    spectrum = np.fft.rfft(windowed)
    psd = (np.abs(spectrum) ** 2) / n
    _apply_one_sided_psd_scaling(psd, n)
    freqs = _get_rfft_freqs(n, fs)

    entropy = compute_spectral_entropy(psd)
    centroid = compute_spectral_centroid(freqs, psd)
    flatness = compute_spectral_flatness(psd)
    rolloffs = compute_spectral_rolloff(freqs, psd, rolloff_pcts)
    dom_freq, dom_amp = compute_dominant_frequency(freqs, psd)
    bandwidth = compute_spectral_bandwidth(freqs, psd, centroid=centroid)
    bands = compute_energy_bands(psd, n_bands=_N_ENERGY_BANDS)
    total_power = float(psd.sum())

    features: Dict[str, float] = {
        "spectral_entropy":    entropy,
        "spectral_centroid":   centroid,
        "spectral_flatness":   flatness,
        "spectral_bandwidth":  bandwidth,
        "dominant_frequency":  dom_freq,
        "dominant_amplitude":  dom_amp,
        "total_power":         total_power,
        "dc_component":        dc,
    }
    for i, pct in enumerate(rolloff_pcts):
        features[f"rolloff_{int(pct * 100)}"] = rolloffs[i]
    for i, energy in enumerate(bands):
        features[f"energy_band_{i}"] = energy
    return features


def _zero_features(rolloff_pcts: List[float], n_bands: int = 3) -> Dict[str, float]:
    features: Dict[str, float] = {
        "spectral_entropy":   0.0,
        "spectral_centroid":  0.0,
        "spectral_flatness":  0.0,
        "spectral_bandwidth": 0.0,
        "dominant_frequency": 0.0,
        "dominant_amplitude": 0.0,
        "total_power":        0.0,
        "dc_component":       0.0,
    }
    for pct in rolloff_pcts:
        features[f"rolloff_{int(pct * 100)}"] = 0.0
    for i in range(n_bands):
        features[f"energy_band_{i}"] = 0.0
    return features


def _canonical_keys(rolloff_pcts: List[float]) -> List[str]:
    keys = [
        "spectral_entropy",
        "spectral_centroid",
        "spectral_flatness",
        "spectral_bandwidth",
        "dominant_frequency",
        "dominant_amplitude",
        "total_power",
        "dc_component",
    ]
    for pct in rolloff_pcts:
        keys.append(f"rolloff_{int(pct * 100)}")
    for i in range(_N_ENERGY_BANDS):
        keys.append(f"energy_band_{i}")
    return keys


def _n_descriptors(rolloff_pcts: List[float]) -> int:
    return len(_canonical_keys(rolloff_pcts))


def _can_vectorize(window_fn: str) -> bool:
    return window_fn in ("hann", "none", "rectangular")


# ---------------------------------------------------------------------------
# Windowed feature matrix — legacy loop (reference implementation)
# ---------------------------------------------------------------------------

def _build_spectral_feature_matrix_legacy(
    timeseries_df: pd.DataFrame,
    window_size: int,
    overlap: int,
    feature_cols: List[str],
    fs: float,
    window_fn: str,
    rolloff_pcts: List[float],
    delta_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Row-by-row spectral extraction (reference / fallback path)."""
    step = window_size - overlap
    T = len(timeseries_df)
    feature_names = get_spectral_feature_names(feature_cols, rolloff_pcts)
    n_features = len(feature_names)
    n_desc = _n_descriptors(rolloff_pcts)
    canonical = _canonical_keys(rolloff_pcts)

    present_cols = [c for c in feature_cols if c in timeseries_df.columns]
    data = timeseries_df[present_cols].to_numpy(dtype=np.float64)
    col_index = {c: i for i, c in enumerate(present_cols)}

    n_windows = max(0, (T - window_size) // step + 1) if T >= window_size else 0
    if n_windows == 0:
        return np.empty((0, n_features)), np.array([]), feature_names

    feature_matrix = np.empty((n_windows, n_features), dtype=float)
    window_timestamps = np.empty(n_windows, dtype=object)

    win_idx = 0
    start = 0
    while start + window_size <= T:
        row_features: List[float] = []
        for col in feature_cols:
            if col not in col_index:
                row_features.extend([0.0] * n_desc)
                continue
            seg = data[start:start + window_size, col_index[col]]
            feats = extract_spectral_features(
                seg, fs=fs, window_fn=window_fn, rolloff_pcts=rolloff_pcts
            )
            row_features.extend(feats[k] for k in canonical)
        feature_matrix[win_idx] = row_features
        window_timestamps[win_idx] = timeseries_df.index[start]
        win_idx += 1
        start += step

    if delta_features:
        delta_names = _delta_descriptor_names(feature_cols)
        delta_matrix = np.empty((n_windows, len(delta_names)), dtype=float)
        start = 0
        for win_idx in range(n_windows):
            drow: List[float] = []
            for col in feature_cols:
                if col not in col_index:
                    drow.extend([0.0, 0.0])
                    continue
                seg = data[start:start + window_size, col_index[col]]
                drow.extend(_compute_delta_features(seg))
            delta_matrix[win_idx] = drow
            start += step
        feature_matrix = np.hstack([feature_matrix, delta_matrix])
        feature_names = feature_names + delta_names

    return feature_matrix, window_timestamps, feature_names


# ---------------------------------------------------------------------------
# Windowed feature matrix — vectorized path
# ---------------------------------------------------------------------------

def _extract_channel_features_batched(
    windows: np.ndarray,
    fs: float,
    window_fn: str,
    rolloff_pcts: List[float],
) -> np.ndarray:
    """Extract descriptors for all windows of one channel.

    Parameters
    ----------
    windows:
        Shape ``(n_windows, window_size)``.
    """
    n_win, n = windows.shape
    n_desc = _n_descriptors(rolloff_pcts)
    out = np.empty((n_win, n_desc), dtype=float)

    degenerate = np.zeros(n_win, dtype=bool)
    for i in range(n_win):
        if not np.isfinite(windows[i]).any():
            degenerate[i] = True

    clean = np.where(np.isfinite(windows), windows, 0.0)
    dc = clean.mean(axis=-1)
    coeffs = _get_window_coeffs(n, window_fn)
    windowed = clean * coeffs
    spectrum = np.fft.rfft(windowed, axis=-1)
    psd = (np.abs(spectrum) ** 2) / n
    _apply_one_sided_psd_scaling(psd, n)
    freqs = _get_rfft_freqs(n, fs)

    entropy = _batch_spectral_entropy(psd)
    centroid = _batch_spectral_centroid(freqs, psd)
    flatness = _batch_spectral_flatness(psd)
    rolloffs = _batch_spectral_rolloff(freqs, psd, rolloff_pcts)
    dom_freq, dom_amp = _batch_dominant_frequency(freqs, psd)
    bandwidth = _batch_spectral_bandwidth(freqs, psd, centroid)
    bands = _batch_energy_bands(psd, n_bands=_N_ENERGY_BANDS)
    total_power = psd.sum(axis=-1)

    cols = [
        entropy, centroid, flatness, bandwidth, dom_freq, dom_amp,
        total_power, dc,
    ]
    cols.extend(rolloffs)
    cols.extend(bands)
    out[:] = np.column_stack(cols)
    if np.any(degenerate):
        out[degenerate] = 0.0
    return out


def _build_spectral_feature_matrix_vectorized(
    timeseries_df: pd.DataFrame,
    window_size: int,
    overlap: int,
    feature_cols: List[str],
    fs: float,
    window_fn: str,
    rolloff_pcts: List[float],
    delta_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    step = window_size - overlap
    T = len(timeseries_df)
    feature_names = get_spectral_feature_names(feature_cols, rolloff_pcts)
    n_features = len(feature_names)
    n_desc = _n_descriptors(rolloff_pcts)

    present_cols = [c for c in feature_cols if c in timeseries_df.columns]
    if not present_cols:
        n_windows = max(0, (T - window_size) // step + 1) if T >= window_size else 0
        return np.zeros((n_windows, n_features)), np.array([]), feature_names

    data = timeseries_df[present_cols].to_numpy(dtype=np.float64)
    all_windows = sliding_window_view(data, window_shape=window_size, axis=0)
    # (n_all, C, window_size) -> stride along windows
    channel_windows = all_windows[::step]
    n_windows = channel_windows.shape[0]

    if n_windows == 0:
        return np.empty((0, n_features)), np.array([]), feature_names

    feature_matrix = np.empty((n_windows, n_features), dtype=float)
    col_offset = {c: i for i, c in enumerate(present_cols)}

    for out_col_idx, col in enumerate(feature_cols):
        base = out_col_idx * n_desc
        if col not in col_offset:
            feature_matrix[:, base:base + n_desc] = 0.0
            continue
        c_idx = col_offset[col]
        ch_feats = _extract_channel_features_batched(
            channel_windows[:, c_idx, :], fs, window_fn, rolloff_pcts
        )
        feature_matrix[:, base:base + n_desc] = ch_feats

    starts = np.arange(0, n_windows * step, step)
    window_timestamps = timeseries_df.index[starts]

    if delta_features:
        delta_names = _delta_descriptor_names(feature_cols)
        delta_matrix = np.empty((n_windows, len(delta_names)), dtype=float)
        for out_col_idx, col in enumerate(feature_cols):
            base = out_col_idx * 2
            if col not in col_offset:
                delta_matrix[:, base:base + 2] = 0.0
                continue
            c_idx = col_offset[col]
            segs = channel_windows[:, c_idx, :]
            diffs = np.diff(segs, axis=1)
            delta_matrix[:, base] = np.mean(diffs, axis=1)
            delta_matrix[:, base + 1] = np.std(diffs, axis=1)
        feature_matrix = np.hstack([feature_matrix, delta_matrix])
        feature_names = feature_names + delta_names

    return feature_matrix, window_timestamps, feature_names


def build_spectral_feature_matrix(
    timeseries_df: pd.DataFrame,
    window_size: int = 64,
    overlap: int = 0,
    feature_cols: Optional[List[str]] = None,
    fs: float = 1.0,
    window_fn: str = "hann",
    rolloff_pcts: Optional[List[float]] = None,
    delta_features: bool = False,
    *,
    _force_legacy: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Slide a window over *timeseries_df* and extract spectral features."""
    if rolloff_pcts is None:
        rolloff_pcts = [0.85, 0.95]
    if feature_cols is None:
        feature_cols = list(timeseries_df.columns)

    if window_size < 4:
        raise ValueError("window_size must be >= 4.")
    if window_size > len(timeseries_df):
        logger.warning(
            "window_size (%d) > timeseries length (%d); returning empty matrix.",
            window_size,
            len(timeseries_df),
        )
        feat_names = get_spectral_feature_names(feature_cols, rolloff_pcts)
        return np.empty((0, len(feat_names))), np.array([]), feat_names

    step = window_size - overlap
    if step <= 0:
        raise ValueError("overlap must be strictly less than window_size.")

    kwargs = dict(
        timeseries_df=timeseries_df,
        window_size=window_size,
        overlap=overlap,
        feature_cols=feature_cols,
        fs=fs,
        window_fn=window_fn,
        rolloff_pcts=rolloff_pcts,
        delta_features=delta_features,
    )

    if _force_legacy or not _can_vectorize(window_fn):
        return _build_spectral_feature_matrix_legacy(**kwargs)
    return _build_spectral_feature_matrix_vectorized(**kwargs)


def build_multiscale_spectral_matrix(
    timeseries_df: pd.DataFrame,
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Concatenate fast (64s) and slow (256s) spectral feature branches."""
    spectral_cfg = cfg.get("spectral", {})
    multi = spectral_cfg.get("multi_scale", {})
    fast_ws = int(spectral_cfg.get("window_size", 64))
    slow_ws = int(multi.get("slow_window_size", 256))
    slow_overlap = int(multi.get("slow_overlap", 0))
    fast_overlap = int(spectral_cfg.get("overlap", 0))
    input_series = spectral_cfg.get("input_series", list(timeseries_df.columns))
    input_series = [s for s in input_series if s in timeseries_df.columns]
    bin_size = spectral_cfg.get("bin_size_seconds", 1)
    fs = 1.0 / bin_size
    window_fn = spectral_cfg.get("window_function", "hann")
    rolloff_pcts = spectral_cfg.get("rolloff_percentages", [0.85, 0.95])
    delta_features = bool(spectral_cfg.get("delta_features", False))

    X_fast, ts_fast, names_fast = build_spectral_feature_matrix(
        timeseries_df,
        window_size=fast_ws,
        overlap=fast_overlap,
        feature_cols=input_series,
        fs=fs,
        window_fn=window_fn,
        rolloff_pcts=rolloff_pcts,
        delta_features=delta_features,
    )
    X_slow, ts_slow, names_slow = build_spectral_feature_matrix(
        timeseries_df,
        window_size=slow_ws,
        overlap=slow_overlap,
        feature_cols=input_series,
        fs=fs,
        window_fn=window_fn,
        rolloff_pcts=rolloff_pcts,
        delta_features=delta_features,
    )

    if len(X_fast) == 0 or len(X_slow) == 0:
        return X_fast, ts_fast, [f"fast__{n}" for n in names_fast]

    # Align slow windows to fast window timestamps (nearest at or before).
    slow_idx = np.searchsorted(ts_slow, ts_fast, side="right") - 1
    slow_idx = np.clip(slow_idx, 0, len(X_slow) - 1)
    X_slow_aligned = X_slow[slow_idx]

    names = [f"fast__{n}" for n in names_fast] + [f"slow__{n}" for n in names_slow]
    return np.hstack([X_fast, X_slow_aligned]), ts_fast, names


def get_spectral_feature_names(
    series_names: List[str],
    rolloff_pcts: Optional[List[float]] = None,
) -> List[str]:
    if rolloff_pcts is None:
        rolloff_pcts = [0.85, 0.95]
    canonical = _canonical_keys(rolloff_pcts)
    names: List[str] = []
    for sname in series_names:
        for key in canonical:
            names.append(f"{sname}__{key}")
    return names


# ---------------------------------------------------------------------------
# Label propagation
# ---------------------------------------------------------------------------

def _dominant_attack_per_bin(atk: pd.Series, freq_str: str, benign_label: str) -> pd.Series:
    """Mode of attack labels per time bin (attack flows only)."""
    def _mode_or_benign(s: pd.Series) -> str:
        if len(s) == 0:
            return benign_label
        counts = s.value_counts()
        return str(counts.index[0])

    return atk.resample(freq_str).agg(_mode_or_benign).rename("dominant_attack")


def bin_labels_to_timeseries(
    df: pd.DataFrame,
    binary_labels: np.ndarray,
    timestamp_col: str = "Timestamp",
    attack_labels: Optional[np.ndarray] = None,
    bin_size_seconds: int = 1,
    reference_index: Optional[pd.Index] = None,
    benign_label: str = "BENIGN",
) -> pd.DataFrame:
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    if ts.isna().all():
        raise ValueError(
            f"Column '{timestamp_col}' could not be parsed as datetime. "
            "Ensure load_cicids2017() was called before bin_labels_to_timeseries()."
        )

    order = np.argsort(ts.values, kind="mergesort")
    ts_sorted = ts.iloc[order]
    binary_sorted = np.asarray(binary_labels, dtype=float)[order]

    binary = pd.Series(binary_sorted, index=ts_sorted.values)
    freq_str = f"{bin_size_seconds}s"

    attack_count = binary.resample(freq_str).sum().rename("attack_count")
    flow_count = binary.resample(freq_str).size().rename("flow_count")
    attack_fraction = (
        attack_count / flow_count.replace(0, np.nan)
    ).fillna(0.0).rename("attack_fraction")

    out = pd.concat([attack_count, flow_count, attack_fraction], axis=1)

    if attack_labels is not None:
        lab_sorted = np.asarray(attack_labels, dtype=object)[order]
        lab = pd.Series(lab_sorted, index=ts_sorted.values)
        atk_mask = binary_sorted.astype(bool)
        atk = lab[atk_mask]
        if len(atk) > 0:
            dominant = _dominant_attack_per_bin(atk, freq_str, benign_label)
            out = out.join(dominant)

    if "dominant_attack" not in out.columns:
        out["dominant_attack"] = benign_label
    out["dominant_attack"] = out["dominant_attack"].fillna(benign_label)
    out = out.fillna({"attack_count": 0.0, "flow_count": 0.0, "attack_fraction": 0.0})

    if reference_index is not None:
        out = out.reindex(reference_index)
        out["attack_count"] = out["attack_count"].fillna(0.0)
        out["flow_count"] = out["flow_count"].fillna(0.0)
        out["attack_fraction"] = out["attack_fraction"].fillna(0.0)
        out["dominant_attack"] = out["dominant_attack"].fillna(benign_label)

    return out


def build_window_labels(
    bin_label_df: pd.DataFrame,
    window_size: int = 64,
    overlap: int = 0,
    rule: str = "any",
    benign_label: str = "BENIGN",
    attack_fraction_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rule not in ("any", "majority"):
        raise ValueError("rule must be 'any' or 'majority'.")

    step = window_size - overlap
    if step <= 0:
        raise ValueError("overlap must be strictly less than window_size.")

    T = len(bin_label_df)
    frac = bin_label_df["attack_fraction"].to_numpy(dtype=float)
    if "attack_count" in bin_label_df.columns:
        cnt = bin_label_df["attack_count"].to_numpy(dtype=float)
    else:
        cnt = (frac > 0).astype(float)
    if "dominant_attack" in bin_label_df.columns:
        dom = bin_label_df["dominant_attack"].to_numpy(dtype=object)
    else:
        dom = np.array([benign_label] * T, dtype=object)

    y_binary: List[int] = []
    y_label: List[str] = []
    timestamps: List = []

    start = 0
    while start + window_size <= T:
        end = start + window_size
        w_frac = frac[start:end]
        w_cnt = cnt[start:end]
        w_dom = dom[start:end]

        if rule == "any":
            is_attack = bool(np.any(w_cnt > 0) or np.any(w_frac > 0))
        else:
            is_attack = bool(np.mean(w_frac) >= attack_fraction_threshold)

        y_binary.append(int(is_attack))
        if is_attack:
            attack_bin = w_cnt > 0
            candidates = [
                lbl for lbl, m in zip(w_dom, attack_bin)
                if m and lbl != benign_label
            ]
            if candidates:
                vals, counts = np.unique(np.array(candidates, dtype=object),
                                         return_counts=True)
                y_label.append(str(vals[int(np.argmax(counts))]))
            else:
                y_label.append("ATTACK")
        else:
            y_label.append(benign_label)

        timestamps.append(bin_label_df.index[start])
        start += step

    return (
        np.array(y_binary, dtype=int),
        np.array(y_label, dtype=object),
        np.array(timestamps),
    )
