"""
deep_svdd.py — Deep Support Vector Data Description (Deep SVDD) detector.

Deep SVDD learns a neural feature map φ(·; W) that pulls benign data into the
smallest possible hypersphere centred at a fixed point ``c``. Anomalies map far
from the centre, so the squared distance ‖φ(x) − c‖² is the anomaly score.

Training follows the two-stage recipe of Ruff et al. (2018):

1. **Autoencoder pretraining.** An encoder–decoder is trained to reconstruct the
   benign data (MSE). This gives the encoder a meaningful, information-preserving
   initialisation and is what makes Deep SVDD work in practice — without it the
   network tends to collapse to a trivial constant map.
2. **SVDD fine-tuning.** The pretrained encoder is detached from the decoder, the
   hypersphere centre ``c`` is fixed to the mean encoder output over the data,
   and the encoder is fine-tuned to minimise ‖φ(x) − c‖².

Collapse avoidance (per Ruff et al.): the encoder uses **bias-free** linear
layers and unbounded activations, and ``c`` is nudged away from 0.

Reference: Ruff L. et al. (2018). "Deep One-Class Classification."
           Proceedings of the 35th ICML, PMLR 80.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

torch.manual_seed(42)


class _SVDDEncoder(nn.Module):
    """Bias-free MLP feature map used by :class:`DeepSVDDDetector`."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SVDDAutoencoder(nn.Module):
    """Encoder + mirror decoder used for the pretraining stage.

    The decoder is symmetric to the encoder; both halves are bias-free to keep
    the same collapse-avoidance properties used during SVDD fine-tuning.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = _SVDDEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class DeepSVDDDetector:
    """One-class Deep SVDD anomaly detector with the ensemble interface.

    Parameters
    ----------
    hidden_dim:
        Width of the hidden layers.
    latent_dim:
        Dimensionality of the output feature space (the hypersphere lives here).
    epochs:
        Number of SVDD fine-tuning epochs.
    pretrain_epochs:
        Number of autoencoder pretraining epochs (set 0 to disable, which
        reproduces the old collapse-prone behaviour).
    batch_size:
        Mini-batch size.
    learning_rate:
        Adam learning rate.
    weight_decay:
        L2 regularisation (acts as the network weight decay in the SVDD objective).
    device:
        ``'auto'`` selects CUDA when available, else CPU.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        epochs: int = 30,
        pretrain_epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.input_dim: Optional[int] = None
        self.net: Optional[_SVDDEncoder] = None
        self.center: Optional[torch.Tensor] = None
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _init_center(self, loader: DataLoader, eps: float = 0.1) -> torch.Tensor:
        """Set the hypersphere centre to the mean encoder output (collapse-safe)."""
        assert self.net is not None
        self.net.eval()
        n = 0
        c = torch.zeros(self.latent_dim, device=self.device)
        for (batch,) in loader:
            batch = batch.to(self.device)
            out = self.net(batch)
            c += out.sum(dim=0)
            n += out.shape[0]
        c /= max(n, 1)
        # Nudge components that are too close to zero away from it.
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c >= 0)] = eps
        return c

    # ------------------------------------------------------------------
    def _pretrain_autoencoder(self, loader: DataLoader) -> _SVDDEncoder:
        """Pretrain an autoencoder on benign data and return its encoder."""
        assert self.input_dim is not None
        ae = _SVDDAutoencoder(self.input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(
            ae.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        ae.train()
        for epoch in range(1, self.pretrain_epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = ae(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "DeepSVDD[pretrain] epoch %3d/%d — recon MSE: %.6f",
                    epoch, self.pretrain_epochs, epoch_loss / max(n_batches, 1),
                )
        return ae.encoder

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray) -> "DeepSVDDDetector":
        """Train the feature map to enclose *X_train* in a tight hypersphere.

        Stage 1 pretrains an autoencoder (unless ``pretrain_epochs == 0``); the
        encoder is then fine-tuned in stage 2 to pull the data to the centre.
        """
        self.input_dim = int(X_train.shape[1])

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=len(X_train) > self.batch_size,
        )

        # Stage 1 — autoencoder pretraining (initialises the encoder weights).
        if self.pretrain_epochs > 0:
            self.net = self._pretrain_autoencoder(loader).to(self.device)
        else:
            self.net = _SVDDEncoder(self.input_dim, self.hidden_dim, self.latent_dim).to(self.device)

        # Centre is fixed from the (pretrained) encoder's outputs.
        self.center = self._init_center(loader)

        # Stage 2 — SVDD fine-tuning of the encoder.
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.net.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = self.net(batch)
                dist = torch.sum((out - self.center) ** 2, dim=1)
                loss = torch.mean(dist)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "DeepSVDD[svdd] epoch %3d/%d — loss: %.6f",
                    epoch, self.epochs, epoch_loss / max(n_batches, 1),
                )

        raw = self._raw_scores(X_train)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        logger.info(
            "DeepSVDD fitted on %d samples. Raw score range: [%.6f, %.6f]",
            len(X_train), self._score_min, self._score_max,
        )
        return self

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _raw_scores(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        assert self.net is not None and self.center is not None
        self.net.eval()
        device = self.device
        n = len(X)
        if n == 0:
            return np.array([], dtype=np.float32)
        if n <= batch_size:
            xt = torch.tensor(X, dtype=torch.float32).to(device)
            out = self.net(xt)
            return torch.sum((out - self.center) ** 2, dim=1).cpu().numpy()

        parts: List[np.ndarray] = []
        for start in range(0, n, batch_size):
            batch = X[start:start + batch_size]
            xt = torch.tensor(batch, dtype=torch.float32).to(device)
            out = self.net(xt)
            parts.append(torch.sum((out - self.center) ** 2, dim=1).cpu().numpy())
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores normalised to [0, 1] (higher = more anomalous)."""
        raw = self._raw_scores(X)
        score_range = self._score_max - self._score_min
        if score_range < 1e-12:
            return np.zeros(len(raw))
        return np.clip((raw - self._score_min) / score_range, 0.0, 1.0)

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions (1 = anomaly) for a given *threshold*."""
        return (self.score(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the network, centre and normalisation bounds to *path*."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        assert self.net is not None and self.center is not None
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "center": self.center.cpu(),
                "config": {
                    "input_dim":  self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "latent_dim": self.latent_dim,
                },
                "score_min": self._score_min,
                "score_max": self._score_max,
            },
            path,
        )
        logger.info("DeepSVDDDetector saved to %s", path)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str, device: str = "auto") -> "DeepSVDDDetector":
        """Load a previously saved detector from *path*."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["config"]
        inst = cls(hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], device=device)
        inst.input_dim = cfg["input_dim"]
        inst.net = _SVDDEncoder(cfg["input_dim"], cfg["hidden_dim"], cfg["latent_dim"]).to(inst.device)
        inst.net.load_state_dict(ckpt["state_dict"])
        inst.center = ckpt["center"].to(inst.device)
        inst._score_min = ckpt.get("score_min", 0.0)
        inst._score_max = ckpt.get("score_max", 1.0)
        logger.info("DeepSVDDDetector loaded from %s", path)
        return inst
