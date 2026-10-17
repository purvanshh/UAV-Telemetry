"""Write the split manifests under evaluation/manifests/.

    python scripts/build_split_manifest.py                # both manifests
    python scripts/build_split_manifest.py --seed 20261017

historical_split.json  – ids copied from uav_model/feature_schema.json (unverified)
evaluation_split.json  – new split over flights found in data/flights and testdata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_corpus import build_evaluation_manifest, discover_sources  # noqa: E402
from src.split_manifest import historical_manifest, save_manifest, validate_manifest  # noqa: E402

OUT = ROOT / "evaluation" / "manifests"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261017)
    a = ap.parse_args()
    fs = json.loads((ROOT / "uav_model" / "feature_schema.json").read_text())
    hist = historical_manifest(fs)
    hist["validation"] = validate_manifest(hist)
    save_manifest(hist, OUT / "historical_split.json")
    ev = build_evaluation_manifest(discover_sources(), seed=a.seed)
    save_manifest(ev, OUT / "evaluation_split.json")
    for name, m in (("historical_split", hist), ("evaluation_split", ev)):
        print(name, m["validation"]["counts"], "ok" if m["validation"]["ok"] else m["validation"]["errors"])
    return 0 if hist["validation"]["ok"] and ev["validation"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
