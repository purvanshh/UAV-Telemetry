"""Phase 3/4: compare alternative detectors with the aggregate baseline.

    python scripts/run_alt_detectors.py [--seed 20261017] [--timestamp ISO] [--reviewed path]

Writes outputs/evaluation/alt_detectors/report.{json,md}. Does not modify the model,
scaler or threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alt_eval import render_alt_markdown, run_alt_evaluation  # noqa: E402
from src.split_manifest import load_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "evaluation" / "manifests" / "evaluation_split.json"))
    ap.add_argument("--reviewed", default=str(ROOT / "evaluation" / "manifests" / "reviewed_intervals.json"))
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--timestamp", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "evaluation" / "alt_detectors"))
    a = ap.parse_args()
    rp = Path(a.reviewed)
    reviewed = json.loads(rp.read_text()) if rp.is_file() else None
    rep = run_alt_evaluation(load_manifest(a.manifest), seed=a.seed, reviewed=reviewed, timestamp=a.timestamp)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, sort_keys=True, allow_nan=False, default=float) + "\n")
    (out / "report.md").write_text(render_alt_markdown(rep))
    print(render_alt_markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
