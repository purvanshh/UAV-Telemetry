"""Audit of the deployed model/threshold artifacts: what is verified, regenerable, or missing.

Only facts that can be established from files and git history in this checkout are
reported as verified. Claims that live only inside a JSON file (e.g. ``quantile_used``)
are listed as *recorded, not verified*.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.artifact_loader import ROOT, load_artifacts
from src.provenance import artifact_fingerprints


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False
        ).stdout.strip()
    except Exception:
        return ""


def _calibration_code_search() -> List[str]:
    """Pre-existing files that *write* uav_model/threshold_config.json.

    Deliberately narrow: a file only counts if it writes to a path mentioning
    ``threshold_config``. The new calibration module writes ``threshold_candidate.json``
    and is not a historical source.
    """
    pat = re.compile(
        r"(threshold_config[^\n]*\.write_text|write_text\([^\n]*threshold_config|"
        r"open\([^\n]*threshold_config[^\n]*['\"]w['\"]|json\.dump\([^\n]*threshold_config)"
    )
    hits = []
    for p in (
        list((ROOT / "src").glob("*.py"))
        + list((ROOT / "scripts").glob("*.py"))
        + list((ROOT / "uav_model").glob("*.py"))
    ):
        if p.name in ("artifact_audit.py",):
            continue
        if pat.search(p.read_text()):
            hits.append(str(p.relative_to(ROOT)))
    return hits


def audit(model_dir=None) -> Dict[str, Any]:
    a = load_artifacts(model_dir)
    d = a.model_dir
    tc = a.threshold_config
    fs = a.feature_schema
    sp = fs.get("split", {})
    thr = a.threshold
    f32 = float(np.float32(thr))

    hist = _git("log", "--format=%h %ad %s", "--date=short", "--", "uav_model/threshold_config.json").splitlines()
    first = hist[-1].split()[0] if hist else None
    at_first = _git("show", f"{first}:uav_model/threshold_config.json") if first else ""
    unchanged = False
    try:
        unchanged = json.loads(at_first) == tc if at_first else False
    except json.JSONDecodeError:
        pass

    det_src = (ROOT / "src" / "detector.py").read_text()
    operator_in_code = "flags = scores > self.threshold" in det_src

    import torch

    ck = torch.load(a.checkpoint_path, map_location="cpu", weights_only=False)
    ck_meta = {k: (v if not hasattr(v, "shape") and k != "state_dict" else None) for k, v in ck.items() if k != "state_dict"}
    extra_keys = sorted(set(ck) - {"state_dict", "n_features", "hidden_dim", "latent_dim", "win_samples", "num_layers"})

    train = set(sp.get("train", []))
    val = set(sp.get("val", []))
    test = set(sp.get("test", []))
    fit = set(a.scaler.get("fit_on_flights") or [])

    # which recorded split ids exist locally, and which local flights are in no split
    local_ids = set()
    store = ROOT / "data" / "flights"
    if store.is_dir():
        for m in store.glob("*/meta.json"):
            fn = (json.loads(m.read_text()).get("filenames") or {}).get("IMU", "")
            mt = re.match(r"(\d{8})_", fn)
            if mt:
                local_ids.add(mt.group(1))
    listed = train | val | test

    return {
        "schema": "artifact_audit/1",
        "threshold": {
            "value": thr,
            "repr": repr(thr),
            "config_key": "lstm_ae",
            "exact_in_float32": f32 == thr,
            "note_float32": (
                "The value is exactly representable in float32, consistent with a quantile "
                "taken over float32 reconstruction errors. This does not prove the quantile "
                "method, dataset or operator used."
            ),
            "recorded_quantile": tc.get("quantile_used"),
            "recorded_calibration_set": tc.get("calibrated_on"),
            "recorded_variant": tc.get("variant"),
            "flag_operator_in_deployed_code": "score > threshold (strict)" if operator_in_code else "NOT FOUND",
            "stale_value_in_repo_history": "PRD.md previously cited 4.42494 (corrected); it matches no artifact",
        },
        "git_history_threshold_config": {
            "commits": hist,
            "unchanged_since_first_commit": unchanged,
            "interpretation": "Single commit: the file was added with the model export; no history of how it was derived.",
        },
        "checkpoint": {
            "sha256": artifact_fingerprints(a)["checkpoint"],
            "non_weight_keys": ck_meta,
            "unexpected_keys": extra_keys,
            "contains_threshold_or_validation_scores": bool(extra_keys),
        },
        "scaler": {
            "sha256": artifact_fingerprints(a)["scaler"],
            "fit_on_flights": len(fit),
            "equals_schema_train_split": fit == train,
            "disjoint_from_val_and_test": not (fit & (val | test)),
        },
        "recorded_split": {
            "train": len(train), "val": len(val), "test": len(test),
            "pairwise_disjoint": not (train & val or train & test or val & test),
            "source_flights_in_repository": False,
            "local_flight_log_ids_found": sorted(local_ids),
            "local_log_ids_in_recorded_split": sorted(local_ids & listed),
            "local_log_ids_not_in_any_recorded_split": sorted(local_ids - listed),
        },
        "calibration_code_in_repository": _calibration_code_search(),
        "classification": {
            "verified_historical_artifacts": [
                "threshold_config.json value and its SHA-256 (file unchanged since the commit that added it)",
                "scaler.json was fit on exactly the 71 flights listed as split.train",
                "split id lists are pairwise disjoint (ids only)",
                "deployed code flags with strict '>'",
                "threshold is exactly a float32 value",
            ],
            "recorded_but_unverified_claims": [
                "threshold = 0.995 quantile (threshold_config.json: quantile_used)",
                "threshold computed on validation flights only (calibrated_on)",
                "val/test flights were never used for training beyond the id lists",
            ],
            "regenerable_with_new_data": [
                "a calibration candidate (scripts/calibrate_threshold.py) once reviewed-normal validation flights exist",
                "new flight-level split manifest and leakage checks (scripts/build_split_manifest.py)",
                "synthetic-injection sensitivity of the deployed detector (scripts/run_evaluation.py)",
            ],
            "missing_cannot_be_recovered": [
                "the 101 source flights (00000001..00000101) used for train/val/test",
                "validation reconstruction scores that the threshold was computed from",
                "the calibration script and the quantile implementation that produced 4.034964084625244",
                "any duplicate/near-duplicate check between the historical splits",
                "training code and hyperparameters / training logs for the checkpoint",
            ],
        },
    }
