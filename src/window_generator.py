"""Session-aware window generation for the deployed feature set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.artifact_loader import ModelArtifacts
from src.preprocessing_adapter import PreprocessingError, _import_prep


@dataclass
class WindowBatch:
    X: np.ndarray  # (N, T, F)
    meta: List[Dict[str, Any]]
    features: List[str]
    win_samples: int
    stride_samples: int

    @property
    def n_windows(self) -> int:
        return int(len(self.X))


def generate_windows(
    aligned: pd.DataFrame,
    artifacts: ModelArtifacts,
    model_dir: Optional[Any] = None,
) -> WindowBatch:
    """Build windows using only the deployed (conservative) features.

    Validity is checked on those features alone, so NaNs in excluded BARO
    columns do not discard otherwise-usable windows.
    """
    prep = _import_prep(model_dir or artifacts.model_dir)
    features = list(artifacts.features)
    missing = [f for f in features if f not in aligned.columns]
    if missing:
        raise PreprocessingError(
            f"Aligned data missing required model features: {', '.join(missing)}"
        )

    X, meta = prep.make_windows(
        aligned,
        features,
        win_len=artifacts.win_samples,
        stride=artifacts.stride_samples,
        session_gap_s=artifacts.feature_schema.get("alignment", {}).get(
            "session_gap_s", 1.0
        ),
    )

    if X.ndim != 3:
        raise PreprocessingError(f"Unexpected window tensor ndim={X.ndim}")
    if len(X) and (
        X.shape[1] != artifacts.win_samples or X.shape[2] != len(features)
    ):
        raise PreprocessingError(
            f"Window shape {X.shape} does not match "
            f"(N, {artifacts.win_samples}, {len(features)})."
        )
    if len(X) and not np.isfinite(X).all():
        raise PreprocessingError("Non-finite values present in generated windows.")

    return WindowBatch(
        X=X,
        meta=meta,
        features=features,
        win_samples=artifacts.win_samples,
        stride_samples=artifacts.stride_samples,
    )
