"""Run the reproducible evaluation -> outputs/evaluation/evaluation_report.{json,md}

    python scripts/run_evaluation.py [--manifest ...] [--reviewed evaluation/manifests/reviewed_intervals.json]
                                     [--candidate outputs/calibration/threshold_candidate.json]
                                     [--seed 20261017] [--save-frames] [--timestamp ISO8601]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation_report import render_markdown, run_evaluation  # noqa: E402
from src.split_manifest import load_manifest  # noqa: E402

OUT = ROOT / "outputs" / "evaluation"


def _load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "evaluation" / "manifests" / "evaluation_split.json"))
    ap.add_argument("--reviewed", default=str(ROOT / "evaluation" / "manifests" / "reviewed_intervals.json"))
    ap.add_argument("--candidate", default=str(ROOT / "outputs" / "calibration" / "threshold_candidate.json"))
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--save-frames", action="store_true", help="write injected raw frames (large)")
    ap.add_argument("--timestamp", default=None)
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = run_evaluation(
        load_manifest(a.manifest), seed=a.seed, reviewed=_load(a.reviewed),
        candidate=_load(a.candidate), out_dir=out, save_frames=a.save_frames, timestamp=a.timestamp,
    )
    (out / "evaluation_report.json").write_text(json.dumps(rep, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "evaluation_report.md").write_text(render_markdown(rep))
    print(render_markdown(rep))
    print(f"wrote {out / 'evaluation_report.json'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
