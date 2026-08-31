"""
vae.py — Variational Autoencoder for unsupervised anomaly detection.

Architecture:
    Encoder:  Input → [FC → BN → LeakyReLU → Dropout]* → μ, log σ²
    Decoder:  z     → [FC → BN → LeakyReLU → Dropout]* → Reconstruction

Anomaly score: per-sample mean squared reconstruction error.  Anomalous
samples (attacks) lie far from the learned benign manifold and therefore
have high reconstruction error.

Loss:  ELBO = E[log p(x|z)] - β · KL(q(z|x) || p(z))
       where β-annealing is used to stabilise early training.

Reference: Zavrak S. & Iskefiyeli M. (2020). "Anomaly-Based Intrusion
           Detection from Network Flow Features Using Variational Autoencoder."
           IEEE Access, Vol. 8, pp. 108346–108358.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .scoring import normalize_scores

logger = logging.getLogger(__name__)

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """MLP encoder that maps input *x* to the parameters of q(z|x).

    Parameters
    ----------
    input_dim:
        Dimensionality of the input feature vector.
    hidden_dims:
        Sequence of hidden layer widths (e.g. ``[256, 128]``).
    latent_dim:
        Dimensionality of the latent space **z**.
    dropout:
        Dropout rate applied after each hidden activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        self.shared = nn.Sequential(*layers)
        self.fc_mu      = nn.Linear(in_dim, latent_dim)
        self.fc_log_var = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (μ, log σ²) for the input batch."""
        h = self.shared(x)
        return self.fc_mu(h), self.fc_log_var(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """MLP decoder that maps a latent vector *z* to a reconstruction.

    Parameters
    ----------
    latent_dim:
        Dimensionality of the latent space.
    hidden_dims:
        Hidden layer widths in decoder order (e.g. ``[128, 256]``).
    output_dim:
        Dimensionality of the reconstruction (equals the input dimension).
    dropout:
        Dropout rate.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return the reconstructed input from latent *z*."""
        return self.net(z)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """Variational Autoencoder for spectral anomaly detection.

    Parameters
    ----------
    input_dim:
        Number of input features.
    encoder_dims:
        Hidden layer sizes for the encoder (e.g. ``[256, 128]``).
    latent_dim:
        Latent space dimensionality.
    decoder_dims:
        Hidden layer sizes for the decoder (e.g. ``[128, 256]``).
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int] = None,
        latent_dim: int = 32,
        decoder_dims: List[int] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if encoder_dims is None:
            encoder_dims = [256, 128]
        if decoder_dims is None:
            decoder_dims = [128, 256]

        self.input_dim  = input_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(input_dim, encoder_dims, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, decoder_dims, input_dim, dropout)

    # ------------------------------------------------------------------
    @staticmethod
    def reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: z = μ + ε · σ, ε ~ N(0, I).

        During inference (eval mode) this is equivalent to just returning μ.
        During training the stochastic sampling allows gradients to flow
        through the encoder via the deterministic path ε·σ.
        """
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode *x* and return (μ, log σ²)."""
        return self.encoder(x)

    # ------------------------------------------------------------------
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent *z* to a reconstruction."""
        return self.decoder(z)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns
        -------
        (reconstruction, μ, log σ²)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def get_anomaly_score(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Return the per-sample mean squared reconstruction error."""
        self.eval()
        device = next(self.parameters()).device
        n = len(x)
        if n == 0:
            return np.array([], dtype=np.float32)
        if n <= batch_size:
            xt = torch.tensor(x, dtype=torch.float32).to(device)
            recon, _, _ = self(xt)
            return torch.mean((xt - recon) ** 2, dim=1).cpu().numpy()

        parts: List[np.ndarray] = []
        for start in range(0, n, batch_size):
            batch = x[start:start + batch_size]
            xt = torch.tensor(batch, dtype=torch.float32).to(device)
            recon, _, _ = self(xt)
            parts.append(torch.mean((xt - recon) ** 2, dim=1).cpu().numpy())
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_latent(self, x: np.ndarray) -> np.ndarray:
        """Return the posterior mean μ as the latent representation.

        Suitable for UMAP / t-SNE visualisation of the latent space.

        Parameters
        ----------
        x:
            2-D float array of shape (n_samples, n_features).

        Returns
        -------
        2-D array of shape (n_samples, latent_dim).
        """
        self.eval()
        device = next(self.parameters()).device
        xt = torch.tensor(x, dtype=torch.float32).to(device)
        mu, _ = self.encode(xt)
        return mu.cpu().numpy()


# ---------------------------------------------------------------------------
# VAE Trainer
# ---------------------------------------------------------------------------

class VAETrainer:
    """Training wrapper for :class:`VAE` with β-annealing and score normalisation.

    Parameters
    ----------
    input_dim:
        Input feature dimensionality.
    encoder_dims, latent_dim, decoder_dims, dropout:
        Passed directly to :class:`VAE`.
    beta:
        Final β coefficient for the KL term in the ELBO loss.
    learning_rate:
        Adam learning rate.
    device:
        Torch device string.  ``'auto'`` selects CUDA if available.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: Optional[List[int]] = None,
        latent_dim: int = 32,
        decoder_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        beta: float = 1.0,
        learning_rate: float = 1e-3,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.vae = VAE(
            input_dim=input_dim,
            encoder_dims=encoder_dims or [256, 128],
            latent_dim=latent_dim,
            decoder_dims=decoder_dims or [128, 256],
            dropout=dropout,
        ).to(self.device)

        self.beta          = beta
        self.learning_rate = learning_rate
        self.history: dict = {"train_loss": [], "recon_loss": [], "kl_loss": []}

        # Score normalisation parameters — set after fit() or calibrate()
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._calibrated: bool = False

    # ------------------------------------------------------------------
    def raw_score(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample MSE reconstruction error."""
        return self.vae.get_anomaly_score(X)

    def set_score_bounds(self, score_low: float, score_high: float) -> None:
        self._score_min = float(score_low)
        self._score_max = float(score_high)
        self._calibrated = True

    # ------------------------------------------------------------------
    @staticmethod
    def vae_loss(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ELBO loss = reconstruction_loss + β · KL divergence.

        Parameters
        ----------
        recon_x:
            Decoder output of shape (batch, input_dim).
        x:
            Original input of shape (batch, input_dim).
        mu, log_var:
            Encoder outputs.
        beta:
            KL weight coefficient (β-VAE).

        Returns
        -------
        (total_loss, recon_loss, kl_loss) — all scalar tensors.
        """
        # Mean squared error per sample, then mean across batch
        recon_loss = torch.mean(torch.sum((x - recon_x) ** 2, dim=1))

        # Closed-form KL divergence: -0.5 · Σ(1 + log σ² - μ² - σ²)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )
        total = recon_loss + beta * kl_loss
        return total, recon_loss, kl_loss

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        beta_warmup_epochs: int = 10,
        X_val: Optional[np.ndarray] = None,
        early_stopping_patience: int = 0,
    ) -> "VAETrainer":
        """Train the VAE on *X_train*.

        β-annealing: β linearly ramps from 0 → self.beta over the first
        *beta_warmup_epochs* epochs to prevent posterior collapse.

        Parameters
        ----------
        X_train:
            2-D float array of benign training samples.
        epochs:
            Number of training epochs.
        batch_size:
            Mini-batch size.
        beta_warmup_epochs:
            Number of epochs over which β is annealed from 0 to self.beta.

        Returns
        -------
        self
        """
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        loader   = DataLoader(
            TensorDataset(X_tensor),
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(X_train) > batch_size,
        )

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.learning_rate)

        best_val_recon = float("inf")
        patience_left = early_stopping_patience if X_val is not None else 0
        best_state = None

        self.vae.train()
        for epoch in range(1, epochs + 1):
            # Linear β-annealing schedule
            beta_current = self.beta * min(1.0, epoch / max(1, beta_warmup_epochs))

            epoch_total = epoch_recon = epoch_kl = 0.0
            n_batches = 0

            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon, mu, log_var = self.vae(batch)
                loss, r_loss, kl_loss = self.vae_loss(
                    recon, batch, mu, log_var, beta=beta_current
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_total += loss.item()
                epoch_recon += r_loss.item()
                epoch_kl    += kl_loss.item()
                n_batches   += 1

            avg_total = epoch_total / max(n_batches, 1)
            avg_recon = epoch_recon / max(n_batches, 1)
            avg_kl    = epoch_kl    / max(n_batches, 1)

            self.history["train_loss"].append(avg_total)
            self.history["recon_loss"].append(avg_recon)
            self.history["kl_loss"].append(avg_kl)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d — total: %.4f  recon: %.4f  kl: %.4f  β: %.3f",
                    epoch, epochs, avg_total, avg_recon, avg_kl, beta_current,
                )

            if X_val is not None and early_stopping_patience > 0:
                self.vae.eval()
                with torch.no_grad():
                    val_recon = float(np.mean(self.vae.get_anomaly_score(X_val)))
                self.vae.train()
                if val_recon < best_val_recon - 1e-6:
                    best_val_recon = val_recon
                    patience_left = early_stopping_patience
                    best_state = {k: v.cpu().clone() for k, v in self.vae.state_dict().items()}
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        logger.info(
                            "Early stopping at epoch %d (best val recon %.4f)",
                            epoch, best_val_recon,
                        )
                        break

        if best_state is not None:
            self.vae.load_state_dict(best_state)

        # Default normalisation bounds from training scores (overridden by calibrate)
        train_scores = self.vae.get_anomaly_score(X_train)
        self._score_min = float(train_scores.min())
        self._score_max = float(train_scores.max())
        logger.info(
            "Training complete. Score range: [%.4f, %.4f]",
            self._score_min, self._score_max,
        )
        return self

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores normalised to [0, 1].

        Scores are clipped to ``[_score_min, _score_max]`` determined on
        the training set to prevent out-of-training-distribution values
        from dominating the ensemble.

        Parameters
        ----------
        X:
            2-D float array of shape (n_samples, n_features).

        Returns
        -------
        1-D array of normalised anomaly scores.
        """
        raw = self.raw_score(X)
        return normalize_scores(raw, self._score_min, self._score_max)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the VAE weights and training state to *path*."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.vae.state_dict(),
                "model_config": {
                    "input_dim":   self.vae.input_dim,
                    "latent_dim":  self.vae.latent_dim,
                    "encoder_dims": [
                        m.out_features
                        for m in self.vae.encoder.shared
                        if isinstance(m, nn.Linear)
                    ],
                    "decoder_dims": [
                        m.out_features
                        for m in self.vae.decoder.net
                        if isinstance(m, nn.Linear)
                    ][:-1],
                },
                "score_min":   self._score_min,
                "score_max":   self._score_max,
                "calibrated":  self._calibrated,
                "beta":        self.beta,
                "history":     self.history,
            },
            path,
        )
        logger.info("VAE saved to %s", path)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str, device: str = "auto") -> "VAETrainer":
        """Load a previously saved :class:`VAETrainer` from *path*."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint["model_config"]

        trainer = cls(
            input_dim    = cfg["input_dim"],
            encoder_dims = cfg.get("encoder_dims", [256, 128]),
            latent_dim   = cfg["latent_dim"],
            decoder_dims = cfg.get("decoder_dims", [128, 256]),
            beta         = checkpoint.get("beta", 1.0),
            device       = device,
        )
        trainer.vae.load_state_dict(checkpoint["model_state_dict"])
        trainer._score_min = checkpoint.get("score_min", 0.0)
        trainer._score_max = checkpoint.get("score_max", 1.0)
        trainer._calibrated = checkpoint.get("calibrated", False)
        trainer.history    = checkpoint.get("history", {})
        logger.info("VAE loaded from %s", path)
        return trainer
