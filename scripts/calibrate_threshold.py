"""Produce a CANDIDATE threshold calibration artifact (never promotes it).

    python scripts/calibrate_threshold.py [--manifest evaluation/manifests/evaluation_split.json]
                                          [--quantile 0.995] [--out-dir outputs/calibration]
                                          [--timestamp ISO8601]

Uses only flights with split == "val" AND calibration_eligible == true. Test flights
are never opened. If no such flight exists the artifact says so
(status "no_validation_data") and the exit code is 2; nothing is fabricated.
uav_model/threshold_config.json is never written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifact_loader import load_artifacts  # noqa: E402
from src.calibration import calibrate, write_candidate  # noqa: E402
from src.eval_corpus import resolve_flight  # noqa: E402
from src.detector import UAVDetector  # noqa: E402
from src.split_manifest import load_manifest, validate_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "evaluation" / "manifests" / "evaluation_split.json"))
    ap.add_argument("--quantile", type=float, default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "calibration"))
    ap.add_argument("--timestamp", default=None, help="fix the timestamp for byte-reproducible output")
    a = ap.parse_args()
    manifest = load_manifest(a.manifest)
    v = validate_manifest(manifest)
    if not v["ok"]:
        print("manifest invalid:", v["errors"])
        return 1
    art = load_artifacts()
    res = calibrate(manifest, artifacts=art, detector=UAVDetector(art), resolve=resolve_flight,
                    quantile=a.quantile, timestamp=a.timestamp)
    path = write_candidate(res, a.out_dir)
    summary = {k: res.get(k) for k in ("status", "reason", "threshold", "n_windows", "flight_ids")}
    print(json.dumps(summary, indent=2))
    cmp_ = res.get("comparison_with_deployed")
    if cmp_:
        print("deployed:", cmp_["deployed_threshold"], "candidate:", cmp_["candidate_threshold"],
              "reproduces_deployed:", cmp_["reproduces_deployed"])
    print(f"wrote {path}  (candidate only; uav_model/threshold_config.json untouched)")
    return 0 if res["status"] == "candidate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
