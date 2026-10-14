"""LSTM Autoencoder loading and reconstruction-error scoring."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.artifact_loader import ArtifactError, ModelArtifacts


class LSTMAE(nn.Module):
    """Architecture must match the training notebook / inference.py export."""

    def __init__(self, n_features, hidden, latent, win_len, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, num_layers, batch_first=True)
        self.enc_to_latent = nn.Linear(hidden, latent)
        self.latent_to_hidden = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, num_layers, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = self.enc_to_latent(h[-1])
        hd = self.latent_to_hidden(z).unsqueeze(1).repeat(1, x.shape[1], 1)
        y, _ = self.decoder(hd)
        return self.out(y)


class DetectorError(RuntimeError):
    pass


class UAVDetector:
    def __init__(self, artifacts: ModelArtifacts, device: str = "cpu"):
        self.artifacts = artifacts
        self.device = torch.device(device)
        self.mu = np.asarray(artifacts.scaler["mean"], dtype=np.float32)
        self.sd = np.asarray(artifacts.scaler["std"], dtype=np.float32)
        if self.mu.shape[0] != artifacts.n_features:
            raise DetectorError("Scaler mean length != n_features")
        self.sd = np.where(self.sd < 1e-12, 1.0, self.sd).astype(np.float32)
        self.threshold = float(artifacts.threshold)
        self.features = list(artifacts.features)
        self.model = self._load_model()

    def _load_model(self) -> LSTMAE:
        try:
            ckpt = torch.load(
                self.artifacts.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            # Older torch without weights_only
            ckpt = torch.load(self.artifacts.checkpoint_path, map_location="cpu")
        except Exception as exc:
            raise DetectorError(f"Failed to load checkpoint: {exc}") from exc

        n_features = int(ckpt.get("n_features", self.artifacts.n_features))
        if n_features != self.artifacts.n_features:
            raise DetectorError(
                f"Checkpoint n_features={n_features} != config "
                f"{self.artifacts.n_features}."
            )
        hidden = int(ckpt.get("hidden_dim", self.artifacts.hidden_dim))
        latent = int(ckpt.get("latent_dim", self.artifacts.latent_dim))
        win = int(ckpt.get("win_samples", self.artifacts.win_samples))
        layers = int(ckpt.get("num_layers", self.artifacts.num_layers))

        model = LSTMAE(n_features, hidden, latent, win, layers)
        state = ckpt.get("state_dict", ckpt)
        try:
            model.load_state_dict(state)
        except Exception as exc:
            raise DetectorError(
                f"Checkpoint weights incompatible with LSTMAE architecture: {exc}"
            ) from exc
        model.to(self.device)
        model.eval()
        return model

    def scale(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd).astype(np.float32)

    def recon_error_per_window(
        self, X: np.ndarray, batch_size: int = 512
    ) -> np.ndarray:
        """Mean squared reconstruction error over time and features (training formula)."""
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        Xn = self.scale(X)
        errs = []
        with torch.no_grad():
            for i in range(0, len(Xn), batch_size):
                xb = torch.from_numpy(Xn[i : i + batch_size]).to(self.device)
                out = self.model(xb)
                e = ((out - xb) ** 2).mean(dim=(1, 2)).cpu().numpy()
                errs.append(e)
        return np.concatenate(errs).astype(np.float32)

    def recon_error_per_feature(
        self, X: np.ndarray, batch_size: int = 512
    ) -> np.ndarray:
        if len(X) == 0:
            return np.empty((0, X.shape[-1] if X.ndim == 3 else 0), dtype=np.float32)
        Xn = self.scale(X)
        errs = []
        with torch.no_grad():
            for i in range(0, len(Xn), batch_size):
                xb = torch.from_numpy(Xn[i : i + batch_size]).to(self.device)
                out = self.model(xb)
                e = ((out - xb) ** 2).mean(dim=1).cpu().numpy()
                errs.append(e)
        return np.concatenate(errs, axis=0).astype(np.float32)

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (per_window_scores, flags, per_feature_errors)."""
        scores = self.recon_error_per_window(X)
        flags = scores > self.threshold
        per_feat = self.recon_error_per_feature(X)
        return scores, flags, per_feat
