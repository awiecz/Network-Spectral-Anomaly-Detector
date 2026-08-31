# Methodology: Network Spectral Anomaly Detector

## Overview

This document describes the technical methodology used in the Network Spectral Anomaly Detector. The system identifies malicious network flows by analysing their frequency-domain characteristics — an approach inspired by audio signal processing applied to network security.

---

## 1. Problem Formulation

Given a stream of unlabelled network flow records, identify flows that deviate from normal behaviour without any prior knowledge of attack signatures. This is an **unsupervised anomaly detection** problem.

**Formal definition:**  
Let `x_t` be the feature vector of a network flow at time `t`. We seek a scoring function `s(x) → ℝ` such that `s(x)` is significantly higher for malicious flows than for benign flows, trained only on benign examples.

---

## 2. Dataset

**CICIDS2017** (Canadian Institute for Cybersecurity, 2017):
- ~2.8 million bidirectional flow records across 5 days
- 83 features extracted by CICFlowMeter (flow duration, packet length stats, inter-arrival times, flag counts, etc.)
- Ground truth labels used **only for evaluation**, never for training
- Training set: Monday's traffic (100% benign, 674,308 flows)
- Test set: Friday by default (`eval_days: ["Friday"]` in config) — DDoS, PortScan, and Bot traffic. Tuesday–Thursday can be added to `eval_days` when those CSVs are available.

**Attack types on Friday:** DDoS LOIC, PortScan, Botnet ARES. Other CICIDS2017 attack types occur on other days.

---

## 3. Spectral Feature Engineering

### 3.1 Motivation

Standard machine learning applied to individual flow records treats each flow independently, missing the **temporal structure** of network attacks. Many attack types exhibit characteristic periodic patterns:

| Attack Type | Temporal Pattern | Spectral Signature |
|---|---|---|
| DDoS flood | Sustained high rate | High centroid, low entropy, high DC |
| Port scan | Regular sweeping interval | Sharp dominant frequency peak, low flatness |
| Botnet C&C | Periodic heartbeats (every 30–300s) | Peak at low frequency (0.003–0.033 Hz) |
| Slowloris | Slow, sustained connections | Low centroid, moderate entropy |
| Normal browsing | Irregular, bursty | Spectrally flat, high entropy |

By applying FFT to time series of flow statistics, these periodic signatures become directly measurable.

### 3.2 Time Binning

Network flows are grouped into 1-second time bins. For each bin `t`, we compute four aggregate statistics:

```
flow_count(t)     = number of flows in bin t
total_bytes(t)    = sum of bytes across all flows in bin t
total_packets(t)  = sum of packets across all flows in bin t
mean_duration(t)  = mean flow duration across all flows in bin t
```

This produces four parallel time series from the raw flow data.

### 3.3 Windowed FFT

We apply a **64-point tumbling window** (chosen as a power of 2 for FFT efficiency) over each time series. Before FFT, each window is multiplied by a **Hann window function**:

```
w(n) = 0.5 * (1 - cos(2*pi*n / (N-1)))
```

The Hann window reduces **spectral leakage** — the artificial spreading of energy to adjacent frequency bins that occurs when applying FFT to a finite, non-periodic segment.

The FFT is then applied to obtain complex coefficients `X[k]`, and the **Power Spectral Density (PSD)** is computed as:

```
PSD[k] = |X[k]|^2
```

### 3.4 Spectral Descriptors

From the PSD of each time series window, we extract 13 descriptors:

**1. Spectral Entropy (H)**  
Measures how uniformly energy is distributed across frequencies.
```
H = -sum_k p(k) log2 p(k),   where p(k) = PSD(k) / sum_j PSD(j)
```
Low H: energy concentrated at few frequencies (periodic, attack-like).  
High H: energy spread uniformly (irregular, normal-like).

**2. Spectral Centroid (C)**  
The centre of mass of the spectrum — weighted mean frequency.
```
C = sum_k f_k * PSD(k) / sum_k PSD(k)
```
High C: dominant high-frequency activity (DDoS floods).  
Low C: dominant low-frequency activity (slow/periodic traffic).

**3. Spectral Flatness (SF)**  
Ratio of geometric mean to arithmetic mean of PSD (Wiener entropy).
```
SF = geometric_mean(PSD) / arithmetic_mean(PSD)
```
SF near 0: tonal/periodic signal.  SF near 1: white-noise-like signal.

**4 & 5. Spectral Rolloff (R85, R95)**  
Frequency below which 85% (or 95%) of total spectral energy is contained.

**6. Dominant Frequency (f_peak)**  
The frequency bin with maximum PSD. A sharp peak signals periodic attack patterns.

**7. Peak Amplitude (A_peak)**  
The PSD value at f_peak, normalised by total PSD.

**8. Spectral Bandwidth (BW)**  
Weighted standard deviation of frequencies around the centroid.

**9. DC Component**  
FFT coefficient at f=0, proportional to the mean signal level.

**10, 11, 12. Energy Band Ratios (Low / Mid / High)**  
Fraction of total energy in each of three equal frequency bands.

**13. Total Power**  
Total spectral energy over the window.

### 3.5 Final Feature Vector

The default configuration (`config/config.yaml`) uses **three input channels**
(`total_bytes`, `total_packets`, `mean_duration`) with **50 % overlap** on
64-second windows.

Per channel, each window contributes **13 spectral descriptors**. With
`delta_features: true`, two within-window delta statistics (mean and std of
first differences) are appended per channel.

When `multi_scale.enabled` is true, a **256-second slow branch** (non-overlapping)
is concatenated with the fast 64-second branch, aligned to the same window
timestamps. The exact dimensionality depends on config; see
`get_spectral_feature_names()` in `src/spectral_features.py`.

**Typical fast-branch size:** 13 descriptors × 3 channels + 6 delta features = **45**
features per window before slow-scale concatenation.

---

## 4. Unsupervised Anomaly Detection

### 4.1 Training Protocol

Models are trained **exclusively on Monday's benign traffic**. No attack samples are seen during training. This mirrors real-world deployment: establish a normal baseline, then flag deviations.

**Preprocessing:** RobustScaler (uses median and IQR instead of mean/std) — preferred over StandardScaler because IDS features contain extreme outliers (e.g., DDoS packet rates can be 10^6 times normal rates).

### 4.2 Variational Autoencoder (VAE)

The VAE learns a compact 32-dimensional probabilistic latent representation of normal traffic.

**Architecture:**
```
Encoder: Input(64) -> Dense(256,ReLU) -> BN -> Dropout(0.2)
                   -> Dense(128,ReLU) -> BN -> Dropout(0.2)
                   -> [mu(32), log_sigma^2(32)]
Reparameterisation: z = mu + sigma * epsilon,  epsilon ~ N(0,I)
Decoder: z(32) -> Dense(128,ReLU) -> BN -> Dropout(0.2)
               -> Dense(256,ReLU) -> BN -> Dropout(0.2)
               -> Dense(64,Linear)
```

**Loss (ELBO with beta-annealing):**
```
L = MSE(x, x_hat) + beta * KL[N(mu, sigma^2) || N(0,I)]
```
beta is annealed from 0 to 1 over the first 10 epochs to prevent posterior collapse.

**Anomaly score:** Mean squared reconstruction error per sample.

### 4.3 Isolation Forest

Isolates anomalies using random binary partitions. Anomalies require fewer partitions to isolate (shorter average path length).

Configuration: 200 trees, max_samples=256, contamination='auto'.

### 4.4 ECOD (Empirical Cumulative Distribution-based Outlier Detection)

Identifies outliers in the **tails** of each feature's marginal distribution. Parameter-free and computationally efficient.

*Reference: Li et al. (2022). "ECOD." IEEE TKDE.*

### 4.5 Deep SVDD (Deep Support Vector Data Description)

Trains a neural network to map normal data into a minimum-radius hypersphere in latent space. Anomaly score = distance from the sphere centre.

*Reference: Ruff et al. (2018). ICML.*

### 4.6 Ensemble Score Fusion

Individual model scores are normalised to [0, 1], then combined as a weighted sum.
Weights and per-model hyperparameters are selected on the validation split by
`tune.py` (grid search with a minimum validation AUROC guard of 0.55 per model).

The tuned ensemble on the current CICIDS2017 Friday split (`ensemble_weights.json`):

```
s_ensemble = 0.75 * s_ECOD + 0.25 * s_IForest
```

β-VAE, COPOD, HBOS, and Deep SVDD are implemented but receive zero weight after
tuning (Deep SVDD is additionally excluded when validation AUROC < 0.55).

### 4.7 Flow-Level ECOD Fusion

Per-flow CICFlowMeter features are scored by ECOD on Monday benign traffic.
Flow scores are aggregated to spectral windows (`max` or `mean` pooling) and
linearly fused with the spectral ensemble:

```
s_final = (1 - w) * s_ensemble + w * s_flow
```

`w` is `models.flow_ecod.fusion_weight` (tuned by `tune.py`). This branch
targets stealthy attacks that do not dominate a window under the `majority`
labelling rule.

---

## 5. Evaluation

### 5.1 Metrics

- **AUROC** — threshold-free discrimination metric
- **AUPRC** — more informative than AUROC for imbalanced datasets (80% benign / 20% attacks)
- **F1 at optimal threshold** — F1-maximising threshold on validation
- **FPR-targeted thresholds** — `fpr_0.05`, `fpr_0.10` cap false-positive rate on validation
- **Per-attack-type AUROC** — reveals which attack patterns are detectable via spectral analysis
- **Per-flow metrics** — `evaluate_flow_level()` scores individual flows (not windows)

### 5.2 Baseline Comparison

Run `scripts/baseline_comparison.py` to compare **spectral FFT descriptors** against
**window-aggregated raw CICFlowMeter features** (per-flow mean + max pooled to each
spectral window). Both branches use the same Monday-benign training protocol and
val/test splits as `main.py`.

```bash
python scripts/baseline_comparison.py --data-dir data/raw/cicids2017 --use-cache
```

Results are written to `results/metrics/baseline_comparison.csv`. On the default
Friday split, ECOD on spectral windows reaches AUROC **0.954** vs **0.794** on raw
window aggregates — quantifying the value added by spectral engineering.
See also [docs/RESULTS.md](RESULTS.md).

---

## 6. Key References

1. Aldarwbi, M., Lashkari, A. H., & Ghorbani, A. A. (2022). "The Sound of Intrusion." *Computers & Electrical Engineering*, 103, 108306.

2. Zavrak, S., & Iskefiyeli, M. (2020). "Anomaly-based intrusion detection from network flow features using variational autoencoder." *IEEE Access*, 8, 108346-108358.

3. Ruff, L., et al. (2018). "Deep one-class classification." *ICML*, 4393-4402.

4. Li, Z., et al. (2022). "ECOD: Unsupervised outlier detection using empirical cumulative distribution functions." *IEEE TKDE*.

5. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). "Toward generating a new intrusion detection dataset." *ICISSP*, 108-116.

6. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation forest." *ICDM*, 413-422.

7. Moustafa, N., & Slay, J. (2015). "UNSW-NB15: A comprehensive data set for network intrusion detection systems." *MilCIS*. IEEE.
