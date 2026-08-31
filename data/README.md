# Dataset Download Instructions

This project uses two public network intrusion detection datasets:
**CICIDS2017** and **UNSW-NB15**. Neither dataset is included in the
repository due to size; follow the steps below to download each one.

---

## 1. CICIDS2017

### Overview

The Canadian Institute for Cybersecurity Intrusion Detection System 2017 dataset
was generated over five days (Monday–Friday) at the University of New Brunswick.
Monday contains only benign traffic and is used as the training split. Tuesday
through Friday contain labelled attack traffic of 14 categories.

| Day        | Traffic types                                     | Approx. size |
|------------|---------------------------------------------------|--------------|
| Monday     | Benign only                                       | ~440 MB      |
| Tuesday    | Benign + FTP-Patator + SSH-Patator                | ~430 MB      |
| Wednesday  | Benign + DoS Slowloris/Slowhttptest/Hulk/GoldenEye| ~480 MB      |
| Thursday   | Benign + Web Attacks + Infiltration               | ~380 MB      |
| Friday     | Benign + Bot + PortScan + DDoS                    | ~530 MB      |

**Total uncompressed:** ~2.3 GB across 8 CSV files.

### Option A — Kaggle (recommended)

```bash
pip install kaggle
# Place your kaggle.json API key in ~/.kaggle/
kaggle datasets download -d cicdataset/cicids2017
unzip cicids2017.zip -d data/raw/cicids2017/
```

Kaggle page: <https://www.kaggle.com/datasets/cic-ids2017>

### Option B — Direct UNB download

1. Visit <https://www.unb.ca/cic/datasets/ids-2017.html>
2. Click "Download Dataset" and register with an institutional email
3. Extract all CSV files into `data/raw/cicids2017/`

### Expected file names after extraction

```
data/raw/cicids2017/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

### Known data quality issues (handled automatically in `src/data_loader.py`)

| Issue | Column(s) | Fix applied |
|-------|-----------|-------------|
| Leading/trailing whitespace in column names | All columns | `str.strip()` |
| `Inf` / `-Inf` values | `Flow Bytes/s`, `Flow Packets/s` | Replace with column max (non-inf) |
| `NaN` in numeric columns | Various | Fill with 0 |
| Non-breaking space in label | `'Web Attack \xa0Brute Force'` | Normalised to `'Web Attack Brute Force'` |
| Mixed-case `BENIGN` / `benign` | `Label` | Uppercased |
| Timestamp format | `Timestamp` | `'%d/%m/%Y %H:%M'` |

### CICFlowMeter column reference (key features)

| Column | Description |
|--------|-------------|
| `Flow ID` | 5-tuple identifier |
| `Source IP` / `Destination IP` | IP addresses |
| `Source Port` / `Destination Port` | TCP/UDP ports |
| `Protocol` | 6=TCP, 17=UDP, 0=ICMP |
| `Timestamp` | Flow start time (`dd/mm/yyyy HH:MM`) |
| `Flow Duration` | Duration in microseconds |
| `Total Fwd Packets` / `Total Backward Packets` | Packet counts |
| `Total Length of Fwd Packets` / `Total Length of Bwd Packets` | Byte counts |
| `Fwd Packet Length Max/Min/Mean/Std` | Payload length statistics (fwd) |
| `Bwd Packet Length Max/Min/Mean/Std` | Payload length statistics (bwd) |
| `Flow Bytes/s` | Throughput in bytes per second (**may contain Inf**) |
| `Flow Packets/s` | Throughput in packets per second (**may contain Inf**) |
| `Flow IAT Mean/Std/Max/Min` | Inter-arrival time statistics |
| `Fwd IAT Total/Mean/Std/Max/Min` | Fwd inter-arrival time |
| `Bwd IAT Total/Mean/Std/Max/Min` | Bwd inter-arrival time |
| `Fwd PSH Flags` / `Bwd PSH Flags` | TCP PSH flag counts |
| `Fwd URG Flags` / `Bwd URG Flags` | TCP URG flag counts |
| `Fwd Header Length` / `Bwd Header Length` | Header sizes |
| `Fwd Packets/s` / `Bwd Packets/s` | Directional packet rates |
| `Min Packet Length` / `Max Packet Length` | Overall packet length range |
| `Packet Length Mean/Std/Variance` | Packet length statistics |
| `FIN Flag Count` ... `URG Flag Count` | TCP flag occurrence counts |
| `Down/Up Ratio` | Download vs upload ratio |
| `Average Packet Size` | Mean packet size |
| `Avg Fwd Segment Size` / `Avg Bwd Segment Size` | Avg TCP segment sizes |
| `Subflow Fwd Packets` / `Subflow Bwd Packets` | Subflow packet counts |
| `Subflow Fwd Bytes` / `Subflow Bwd Bytes` | Subflow byte counts |
| `Init_Win_bytes_forward` / `Init_Win_bytes_backward` | TCP window size |
| `act_data_pkt_fwd` | Packets with payload (fwd) |
| `min_seg_size_forward` | Minimum forward segment size |
| `Active Mean/Std/Max/Min` | Active flow duration stats |
| `Idle Mean/Std/Max/Min` | Idle period stats |
| `Label` | Class label (`BENIGN` or attack name) |

---

## 2. UNSW-NB15

### Overview

The UNSW-NB15 dataset was created at the Australian Centre for Cyber Security
(ACCS) using the IXIA PerfectStorm tool. It contains nine attack categories
plus benign traffic, spread across four raw CSV files. Total size is
approximately 900 MB uncompressed.

### Option A — Kaggle (recommended)

```bash
kaggle datasets download -d mrwellsdavid/unsw-nb15
unzip unsw-nb15.zip -d data/raw/unsw_nb15/
```

Kaggle page: <https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15>

### Option B — UNSW official

1. Visit <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
2. Request access via the online form
3. Extract all CSV files into `data/raw/unsw_nb15/`

### Expected file names after extraction

```
data/raw/unsw_nb15/
├── UNSW-NB15_1.csv
├── UNSW-NB15_2.csv
├── UNSW-NB15_3.csv
├── UNSW-NB15_4.csv
└── UNSW-NB15_features.csv   # (optional, column metadata)
```

### UNSW-NB15 key columns

| Column | Description |
|--------|-------------|
| `srcip` | Source IP |
| `sport` | Source port |
| `dstip` | Destination IP |
| `dsport` | Destination port |
| `proto` | Protocol (tcp/udp/…) |
| `state` | Connection state (FIN/INT/…) |
| `dur` | Duration (seconds) |
| `sbytes` | Source-to-destination bytes |
| `dbytes` | Destination-to-source bytes |
| `sttl` / `dttl` | Source/destination TTL |
| `sloss` / `dloss` | Source/destination packet loss |
| `service` | HTTP/FTP/DNS/… |
| `sload` / `dload` | Source/destination bits per second |
| `spkts` / `dpkts` | Source/destination packet count |
| `swin` / `dwin` | TCP window size |
| `Sjit` / `Djit` | Source/destination jitter |
| `Stime` / `Ltime` | Start/end timestamps (Unix epoch) |
| `tcprtt` / `synack` / `ackdat` | TCP handshake timing |
| `attack_cat` | Attack category |
| `label` | Binary label (0=normal, 1=attack) |

---

## Quick verification

After downloading, run:

```bash
python - <<'EOF'
import pathlib
cicids = pathlib.Path("data/raw/cicids2017")
unsw   = pathlib.Path("data/raw/unsw_nb15")
for p in [cicids, unsw]:
    files = list(p.glob("*.csv"))
    print(f"{p}: {len(files)} CSV files found")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
EOF
```

Or run the pipeline with the built-in synthetic sample (no download needed):

```bash
python main.py --use-sample
```

---

## Processed caches

The pipeline may write artifacts under `data/processed/`:

| File pattern | Purpose |
|--------------|---------|
| `*.parquet` | Per-day Parquet load cache (faster than re-parsing CSV) |
| `splits_cache_v2_*.npz` | Precomputed spectral train/val/test splits (`--use-cache`) |

Delete these files after changing `config/config.yaml` spectral settings, split
logic in `src/splits.py`, or spectral feature code in `src/spectral_features.py`.
