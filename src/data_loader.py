"""
data_loader.py — Dataset loading utilities for CICIDS2017 and UNSW-NB15.

Both datasets have well-known quality issues that are handled explicitly
below with inline comments explaining each fix.
"""

from __future__ import annotations

import os
import glob
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CICIDS 2017
# ---------------------------------------------------------------------------

# Columns that frequently contain Inf / -Inf in CICFlowMeter output.
_CICIDS_INF_COLS = ["Flow Bytes/s", "Flow Packets/s"]

# Minimal columns for the spectral pipeline (load only what we need).
_CICIDS_LOAD_COLS = [
    "Timestamp",
    "Label",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
]

# Day-of-week tag embedded in each CICIDS2017 filename.
_CICIDS_DAY_MAP = {
    "monday":    "Monday",
    "tuesday":   "Tuesday",
    "wednesday": "Wednesday",
    "thursday":  "Thursday",
    "friday":    "Friday",
}


def _clean_cicids_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all known CICIDS2017 data-quality fixes to *df* in-place.

    Fixes applied (each documented with the reason):

    1. Strip whitespace from column names — CICFlowMeter adds leading/
       trailing spaces to many column headers, causing KeyError surprises.
    2. Replace Inf values in throughput columns — division by very small
       durations produces IEEE 754 +Inf; replace with the finite column max.
    3. Fill remaining NaN with 0 — a small fraction of rows have NaN in
       numeric features; zero-fill is safe for rate/count features.
    4. Normalise the non-breaking space in one label — the label string
       'Web Attack \\xa0Brute Force' uses U+00A0 instead of a regular space.
    5. Uppercase all label strings — the dataset mixes 'BENIGN' and 'benign'.
    """

    # Fix 1: strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Fix 2: replace Inf/-Inf in throughput columns
    for col in _CICIDS_INF_COLS:
        if col in df.columns:
            finite_max = df[col].replace([np.inf, -np.inf], np.nan).max()
            finite_max = 0.0 if pd.isna(finite_max) else finite_max
            df[col] = df[col].replace([np.inf, -np.inf], finite_max)

    # Fix 3: fill NaN in numeric columns with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Fix 4: normalise non-breaking space in attack label
    if "Label" in df.columns:
        df["Label"] = df["Label"].str.replace("\xa0", " ", regex=False)

    # Fix 5: uppercase all labels for consistency
    if "Label" in df.columns:
        df["Label"] = df["Label"].str.strip().str.upper()

    return df


def _parquet_path_for_csv(csv_path: str, processed_dir: str) -> str:
    """Map a raw CSV path to a processed Parquet cache file."""
    base = os.path.splitext(os.path.basename(csv_path))[0]
    return os.path.join(processed_dir, f"{base}.parquet")


def _read_cicids_file(path: str, processed_dir: Optional[str] = None) -> pd.DataFrame:
    """Read one CICIDS CSV, optionally via a Parquet cache."""
    if processed_dir:
        pq_path = _parquet_path_for_csv(path, processed_dir)
        if os.path.isfile(pq_path) and os.path.getmtime(pq_path) >= os.path.getmtime(path):
            logger.info("Loading cached Parquet %s …", os.path.basename(pq_path))
            return pd.read_parquet(pq_path)

    header = pd.read_csv(path, nrows=0)
    strip_map = {c.strip(): c for c in header.columns}
    usecols = [strip_map[c] for c in _CICIDS_LOAD_COLS if c in strip_map]
    if not usecols:
        usecols = None
    df = pd.read_csv(path, low_memory=False, usecols=usecols)
    df = _clean_cicids_dataframe(df)

    if processed_dir:
        os.makedirs(processed_dir, exist_ok=True)
        pq_path = _parquet_path_for_csv(path, processed_dir)
        df.to_parquet(pq_path, index=False)
        logger.info("Wrote Parquet cache %s", os.path.basename(pq_path))
    return df


def load_cicids2017(
    data_dir: str,
    days: "List[str] | None" = None,
    processed_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load all CICIDS2017 CSV files from *data_dir* into a single DataFrame.

    Parameters
    ----------
    data_dir:
        Directory that contains the eight ``*.pcap_ISCX.csv`` files.
    days:
        Optional list of day names (e.g. ``["Monday", "Friday"]``) to restrict
        which files are read. Matching is done on the day tag embedded in each
        filename, so unneeded days are never loaded into memory.
    processed_dir:
        Optional directory for Parquet load caches (see ``_read_cicids_file``).

    Returns
    -------
    pd.DataFrame
        Concatenated, cleaned DataFrame with a parsed ``Timestamp`` column
        and a ``day`` column derived from the source filename.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            "See data/README.md for download instructions."
        )

    if days is not None:
        wanted = {d.lower() for d in days}
        filtered = [
            p for p in csv_files
            if any(key in os.path.basename(p).lower() and label.lower() in wanted
                   for key, label in _CICIDS_DAY_MAP.items())
        ]
        if filtered:
            csv_files = filtered
        else:
            logger.warning("No files matched days=%s; loading all files.", days)

    frames: List[pd.DataFrame] = []
    for path in csv_files:
        logger.info("Loading %s …", os.path.basename(path))
        df = _read_cicids_file(path, processed_dir=processed_dir)

        # Tag each row with the day of the week inferred from the filename.
        basename = os.path.basename(path).lower()
        day_tag = "unknown"
        for key, label in _CICIDS_DAY_MAP.items():
            if key in basename:
                day_tag = label
                break
        df["day"] = day_tag

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Parse Timestamp — CICIDS2017 uses dd/mm/yyyy HH:MM format
    if "Timestamp" in combined.columns:
        combined["Timestamp"] = pd.to_datetime(
            combined["Timestamp"], format="%d/%m/%Y %H:%M", errors="coerce"
        )

    # The "MachineLearningCVE" distribution of CICIDS2017 ships *without* a
    # Timestamp column. The spectral pipeline needs an ordered time grid, so we
    # synthesise one from the capture order (CICFlowMeter emits flows roughly in
    # chronological order). Flows are spaced one second apart, per source file,
    # so each day forms a contiguous, monotonically increasing sequence and
    # different days never overlap. This is an approximation: it preserves flow
    # ordering (and therefore burst/periodicity structure) but not the true
    # inter-arrival times, so the `flow_count` channel is roughly constant while
    # the byte/packet/duration channels carry the discriminative signal.
    if "Timestamp" not in combined.columns or combined["Timestamp"].isna().all():
        logger.warning(
            "No usable Timestamp column found; synthesising an ordered timeline "
            "(1 flow/second, per file) from capture order."
        )
        base = pd.Timestamp("2017-07-03 00:00:00")
        timestamps = np.empty(len(combined), dtype="datetime64[ns]")
        cursor = 0
        # Day order preserved from load order; offset each file's block by a day
        # so blocks stay separated and ordered within each day.
        day_offsets: Dict[str, int] = {}
        for day, group in combined.groupby("day", sort=False):
            offset_days = day_offsets.setdefault(day, len(day_offsets))
            start = base + pd.Timedelta(days=offset_days)
            idx = group.index.to_numpy()
            block = (start + pd.to_timedelta(np.arange(len(idx)), unit="s")).to_numpy()
            timestamps[idx] = block
            cursor += len(idx)
        combined["Timestamp"] = timestamps

    logger.info(
        "CICIDS2017 loaded: %d rows, %d columns, labels: %s",
        len(combined),
        len(combined.columns),
        list(combined["Label"].value_counts().head(5).index) if "Label" in combined.columns else "N/A",
    )
    return combined


# ---------------------------------------------------------------------------
# UNSW-NB15
# ---------------------------------------------------------------------------

_UNSW_COLUMN_RENAMES = {
    # Standardise common column names to lower-snake-case
    "srcip":       "src_ip",
    "sport":       "src_port",
    "dstip":       "dst_ip",
    "dsport":      "dst_port",
    "proto":       "protocol",
    "dur":         "duration",
    "sbytes":      "src_bytes",
    "dbytes":      "dst_bytes",
    "sttl":        "src_ttl",
    "dttl":        "dst_ttl",
    "sloss":       "src_loss",
    "dloss":       "dst_loss",
    "sload":       "src_load",
    "dload":       "dst_load",
    "spkts":       "src_pkts",
    "dpkts":       "dst_pkts",
    "swin":        "src_win",
    "dwin":        "dst_win",
    "attack_cat":  "attack_category",
    "label":       "Label",
}


def load_unsw_nb15(data_dir: str) -> pd.DataFrame:
    """Load all four UNSW-NB15 CSV files from *data_dir* into a single DataFrame.

    Parameters
    ----------
    data_dir:
        Directory containing ``UNSW-NB15_1.csv`` … ``UNSW-NB15_4.csv``.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with standardised column names.
        The ``Label`` column uses 0 (normal) / 1 (attack) integers.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "UNSW-NB15_*.csv")))
    # Exclude the features description file
    csv_files = [f for f in csv_files if "feature" not in os.path.basename(f).lower()]

    if not csv_files:
        raise FileNotFoundError(
            f"No UNSW-NB15 data CSV files found in '{data_dir}'. "
            "See data/README.md for download instructions."
        )

    frames: List[pd.DataFrame] = []
    for path in csv_files:
        logger.info("Loading %s …", os.path.basename(path))
        # UNSW-NB15 files have no header in some versions — detect automatically
        sample = pd.read_csv(path, nrows=1)
        has_header = not sample.columns[0].startswith("0") and not sample.columns[0].isdigit()
        df = pd.read_csv(path, low_memory=False, header=0 if has_header else None)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Standardise column names
    combined.columns = combined.columns.str.strip().str.lower()
    combined = combined.rename(columns=_UNSW_COLUMN_RENAMES)

    # Fill NaN in numeric columns
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].fillna(0)

    # Ensure Label is integer (0/1)
    if "Label" in combined.columns:
        combined["Label"] = pd.to_numeric(combined["Label"], errors="coerce").fillna(0).astype(int)

    logger.info(
        "UNSW-NB15 loaded: %d rows, %d columns", len(combined), len(combined.columns)
    )
    return combined


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def get_binary_labels(
    df: pd.DataFrame,
    label_col: str = "Label",
    benign_value: str = "BENIGN",
) -> np.ndarray:
    """Return a binary label array: 0 = benign, 1 = attack.

    Parameters
    ----------
    df:
        DataFrame containing *label_col*.
    label_col:
        Name of the label column.
    benign_value:
        The string (or integer 0 for UNSW-NB15) that represents benign traffic.

    Returns
    -------
    np.ndarray of shape (n,) with dtype int.
    """
    labels = df[label_col]
    if pd.api.types.is_numeric_dtype(labels):
        # Numeric labels (e.g. UNSW-NB15 already 0/1)
        binary = (labels != 0).astype(int)
    else:
        # String labels (CICIDS2017). Use a dtype-agnostic check: newer pandas
        # may infer a dedicated string dtype (not ``object``), so compare on the
        # normalised string value rather than on ``dtype == object``.
        binary = (
            labels.astype("string").str.upper().str.strip() != benign_value.upper()
        ).astype(int)
    return binary.values


def get_multiclass_labels(
    df: pd.DataFrame,
    label_col: str = "Label",
) -> Tuple[np.ndarray, List[str]]:
    """Return integer-encoded multi-class labels and the corresponding class names.

    Parameters
    ----------
    df:
        DataFrame containing *label_col*.
    label_col:
        Name of the label column.

    Returns
    -------
    (encoded, class_names):
        - ``encoded``: np.ndarray of shape (n,) with integer class indices.
        - ``class_names``: list of string class names ordered by index.
    """
    labels = df[label_col].astype(str).str.strip()
    class_names = sorted(labels.unique().tolist())
    mapping = {name: idx for idx, name in enumerate(class_names)}
    encoded = labels.map(mapping).values.astype(int)
    return encoded, class_names


# ---------------------------------------------------------------------------
# Day split (CICIDS2017 specific)
# ---------------------------------------------------------------------------

def split_by_day(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split a CICIDS2017 DataFrame into per-day sub-DataFrames.

    Requires that the ``day`` column was added by :func:`load_cicids2017`.

    Returns
    -------
    dict mapping day name (e.g. ``'Monday'``) to a DataFrame.
    """
    if "day" not in df.columns:
        raise ValueError(
            "'day' column not found. Load the data with load_cicids2017() first."
        )
    return {day: group.reset_index(drop=True) for day, group in df.groupby("day")}


# ---------------------------------------------------------------------------
# Synthetic sample dataset
# ---------------------------------------------------------------------------

def create_sample_dataset(n: int = 1000, output_dir: str = "data/sample") -> str:
    """Create a small synthetic dataset with CICIDS2017-like structure.

    Useful for rapid development and CI/CD pipelines when the full dataset
    is not available.  About 20 % of rows are assigned attack labels to
    match the approximate class imbalance in CICIDS2017.

    Parameters
    ----------
    n:
        Number of rows to generate.
    output_dir:
        Directory where ``sample_cicids2017.csv`` will be written.

    Returns
    -------
    str — path to the created CSV file.
    """
    np.random.seed(42)
    os.makedirs(output_dir, exist_ok=True)

    n_benign = int(n * 0.80)
    n_attack = n - n_benign

    attack_labels = [
        "DDOS", "PORTSCAN", "FTP-PATATOR", "SSH-PATATOR",
        "DOS SLOWLORIS", "BOT", "WEB ATTACK BRUTE FORCE",
    ]

    def _make_rows(count: int, is_attack: bool) -> pd.DataFrame:
        scale = 5.0 if is_attack else 1.0
        return pd.DataFrame({
            "Flow Duration":              np.random.exponential(1e6 * scale, count),
            "Total Fwd Packets":          np.random.poisson(10 * scale, count).astype(float),
            "Total Backward Packets":     np.random.poisson(8, count).astype(float),
            "Total Length of Fwd Packets": np.random.exponential(500 * scale, count),
            "Total Length of Bwd Packets": np.random.exponential(400, count),
            "Fwd Packet Length Max":      np.random.uniform(20, 1500, count),
            "Fwd Packet Length Min":      np.random.uniform(20, 100, count),
            "Fwd Packet Length Mean":     np.random.uniform(40, 800, count),
            "Fwd Packet Length Std":      np.random.uniform(0, 200, count),
            "Bwd Packet Length Max":      np.random.uniform(20, 1500, count),
            "Bwd Packet Length Min":      np.random.uniform(20, 100, count),
            "Bwd Packet Length Mean":     np.random.uniform(40, 800, count),
            "Bwd Packet Length Std":      np.random.uniform(0, 200, count),
            "Flow Bytes/s":               np.random.exponential(1e4 * scale, count),
            "Flow Packets/s":             np.random.exponential(50 * scale, count),
            "Flow IAT Mean":              np.random.exponential(1e4, count),
            "Flow IAT Std":               np.random.exponential(5e3, count),
            "Flow IAT Max":               np.random.exponential(1e5, count),
            "Flow IAT Min":               np.random.exponential(1e3, count),
            "Fwd IAT Total":              np.random.exponential(1e5, count),
            "Fwd IAT Mean":               np.random.exponential(1e4, count),
            "Fwd IAT Std":                np.random.exponential(5e3, count),
            "Fwd IAT Max":                np.random.exponential(1e5, count),
            "Fwd IAT Min":                np.random.exponential(1e3, count),
            "Bwd IAT Total":              np.random.exponential(1e5, count),
            "Bwd IAT Mean":               np.random.exponential(1e4, count),
            "Bwd IAT Std":                np.random.exponential(5e3, count),
            "Bwd IAT Max":                np.random.exponential(1e5, count),
            "Bwd IAT Min":                np.random.exponential(1e3, count),
            "Fwd PSH Flags":              np.random.randint(0, 2, count).astype(float),
            "Bwd PSH Flags":              np.random.randint(0, 2, count).astype(float),
            "Fwd URG Flags":              np.zeros(count),
            "Bwd URG Flags":              np.zeros(count),
            "Fwd Header Length":          np.random.randint(20, 60, count).astype(float),
            "Bwd Header Length":          np.random.randint(20, 60, count).astype(float),
            "Fwd Packets/s":              np.random.exponential(30 * scale, count),
            "Bwd Packets/s":              np.random.exponential(20, count),
            "Min Packet Length":          np.random.uniform(20, 60, count),
            "Max Packet Length":          np.random.uniform(100, 1500, count),
            "Packet Length Mean":         np.random.uniform(40, 800, count),
            "Packet Length Std":          np.random.uniform(0, 200, count),
            "Packet Length Variance":     np.random.uniform(0, 4e4, count),
            "FIN Flag Count":             np.random.randint(0, 2, count).astype(float),
            "SYN Flag Count":             np.random.randint(0, 3, count).astype(float),
            "RST Flag Count":             np.random.randint(0, 2, count).astype(float),
            "PSH Flag Count":             np.random.randint(0, 3, count).astype(float),
            "ACK Flag Count":             np.random.randint(0, 5, count).astype(float),
            "URG Flag Count":             np.zeros(count),
            "CWE Flag Count":             np.zeros(count),
            "ECE Flag Count":             np.zeros(count),
            "Down/Up Ratio":              np.random.uniform(0, 2, count),
            "Average Packet Size":        np.random.uniform(40, 800, count),
            "Avg Fwd Segment Size":       np.random.uniform(40, 800, count),
            "Avg Bwd Segment Size":       np.random.uniform(40, 800, count),
            "Fwd Avg Bytes/Bulk":         np.zeros(count),
            "Fwd Avg Packets/Bulk":       np.zeros(count),
            "Fwd Avg Bulk Rate":          np.zeros(count),
            "Bwd Avg Bytes/Bulk":         np.zeros(count),
            "Bwd Avg Packets/Bulk":       np.zeros(count),
            "Bwd Avg Bulk Rate":          np.zeros(count),
            "Subflow Fwd Packets":        np.random.poisson(5, count).astype(float),
            "Subflow Fwd Bytes":          np.random.exponential(300, count),
            "Subflow Bwd Packets":        np.random.poisson(4, count).astype(float),
            "Subflow Bwd Bytes":          np.random.exponential(250, count),
            "Init_Win_bytes_forward":     np.random.choice([8192, 16384, 65535], count).astype(float),
            "Init_Win_bytes_backward":    np.random.choice([8192, 16384, 65535], count).astype(float),
            "act_data_pkt_fwd":           np.random.poisson(3, count).astype(float),
            "min_seg_size_forward":       np.random.randint(20, 60, count).astype(float),
            "Active Mean":                np.random.exponential(1e5, count),
            "Active Std":                 np.random.exponential(1e4, count),
            "Active Max":                 np.random.exponential(2e5, count),
            "Active Min":                 np.random.exponential(5e4, count),
            "Idle Mean":                  np.random.exponential(1e6, count),
            "Idle Std":                   np.random.exponential(1e5, count),
            "Idle Max":                   np.random.exponential(2e6, count),
            "Idle Min":                   np.random.exponential(5e5, count),
        })

    benign_df = _make_rows(n_benign, is_attack=False)
    benign_df["Label"] = "BENIGN"

    attack_df = _make_rows(n_attack, is_attack=True)
    attack_df["Label"] = np.random.choice(attack_labels, n_attack)

    # Lay out a contiguous attack burst so spectral windows under the
    # majority label rule contain enough attack bins (mimics DDoS / scan bursts).
    attack_start = max(64, (n - n_attack) // 2)
    attack_end = min(n, attack_start + n_attack)
    attack_start = attack_end - n_attack

    benign_before = benign_df.iloc[:attack_start].reset_index(drop=True)
    benign_after = benign_df.iloc[attack_start:].reset_index(drop=True)
    combined = pd.concat(
        [benign_before, attack_df.reset_index(drop=True), benign_after],
        ignore_index=True,
    )
    combined = combined.iloc[:n].reset_index(drop=True)

    base_ts = pd.Timestamp("2017-07-03 08:00:00")
    combined["Timestamp"] = [
        (base_ts + pd.Timedelta(seconds=i)).strftime("%d/%m/%Y %H:%M:%S")
        for i in range(len(combined))
    ]
    combined["day"] = "Monday"

    output_path = os.path.join(output_dir, "sample_cicids2017.csv")
    combined.to_csv(output_path, index=False)
    logger.info("Sample dataset written to %s (%d rows)", output_path, len(combined))
    return output_path
