
"""Standalone CPU inference for the UAV anomaly detector."""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import preprocessing as prep

class LSTMAE(nn.Module):
    def __init__(self, n_features, hidden, latent, win_len, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, num_layers, batch_first=True)
        self.enc_to_latent = nn.Linear(hidden, latent)
        self.latent_to_hidden = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, num_layers, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        B, T, F = x.shape
        _, (h, _) = self.encoder(x)
        z = self.enc_to_latent(h[-1])
        hd = self.latent_to_hidden(z).unsqueeze(1).repeat(1, T, 1)
        y, _ = self.decoder(hd)
        return self.out(y)

class UAVAnomalyDetector:
    def __init__(self, model_dir: str):
        self.dir = Path(model_dir)
        with open(self.dir / "model_config.json") as f:
            self.cfg = json.load(f)
        with open(self.dir / "scaler.json") as f:
            s = json.load(f)
            self.mu = np.array(s["mean"], dtype=np.float32)
            self.sd = np.array(s["std"],  dtype=np.float32)
        with open(self.dir / "threshold_config.json") as f:
            self.thr = json.load(f)["lstm_ae"]

        ckpt = torch.load(self.dir / "lstm_autoencoder.pth",
                          map_location="cpu", weights_only=False)
        self.model = LSTMAE(ckpt["n_features"], ckpt["hidden_dim"],
                            ckpt["latent_dim"], ckpt["win_samples"],
                            ckpt["num_layers"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.features = self.cfg["features"]

    def _windows_from_flight(self, imu_csv, att_csv, baro_csv):
        imu  = prep.load_stream(Path(imu_csv),  prep.DEFAULT_SCHEMA["IMU"])
        att  = prep.load_stream(Path(att_csv),  prep.DEFAULT_SCHEMA["ATT"])
        baro = prep.load_stream(Path(baro_csv), prep.DEFAULT_SCHEMA["BARO"])
        df = prep.align_flight(imu, att, baro,
                               prep.DEFAULT_SCHEMA, prep.DEFAULT_ALIGN)
        X, meta = prep.make_windows(df, self.features,
                                    win_len=self.cfg["win_samples"],
                                    stride=self.cfg["stride_samples"],
                                    session_gap_s=prep.DEFAULT_ALIGN["session_gap_s"])
        return X, meta, df

    def _score(self, X):
        if len(X) == 0:
            return np.array([]), np.empty((0, X.shape[-1]))
        Xn = ((X - self.mu) / self.sd).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(Xn)
            out = self.model(t)
            per_win  = ((out - t) ** 2).mean(dim=(1, 2)).numpy()
            per_feat = ((out - t) ** 2).mean(dim=1).numpy()
        return per_win, per_feat

    def detect(self, imu_csv, att_csv, baro_csv, flight_id=""):
        X, meta, df = self._windows_from_flight(imu_csv, att_csv, baro_csv)
        if len(X) == 0:
            return {"flight_id": flight_id, "anomaly_detected": False,
                    "n_windows": 0, "anomalous_intervals": [],
                    "data_quality_warnings": ["no usable windows"]}
        errs, per_feat = self._score(X)
        flagged = errs > self.thr

        intervals = []
        i = 0
        while i < len(flagged):
            if flagged[i]:
                j = i
                while j + 1 < len(flagged) and flagged[j+1]:
                    j += 1
                top_feat_idx = np.argsort(per_feat[i:j+1].max(axis=0))[-3:][::-1]
                intervals.append({
                    "start_seconds": float(meta[i]["t_start"]),
                    "end_seconds":   float(meta[j]["t_end"]),
                    "score": float(errs[i:j+1].max()),
                    "affected_features": [self.features[k] for k in top_feat_idx],
                })
                i = j + 1
            else:
                i += 1

        warnings = []
        if "alt" in df.columns and (df["alt"] < -1000).any():
            warnings.append("BARO altitude contains unresolved extreme values")
        if "press" in df.columns and (df["press"] > 1.2e5).any():
            warnings.append("BARO pressure exceeds typical surface range")

        return {
            "flight_id": flight_id,
            "anomaly_detected": bool(flagged.any()),
            "n_windows": int(len(errs)),
            "threshold": float(self.thr),
            "anomalous_intervals": intervals,
            "data_quality_warnings": warnings,
            "score_summary": {
                "mean": float(errs.mean()),
                "p95":  float(np.quantile(errs, 0.95)),
                "max":  float(errs.max()),
            },
        }