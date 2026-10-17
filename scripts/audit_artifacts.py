"""Audit deployed artifacts -> outputs/evaluation/artifact_audit.json

    python scripts/audit_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifact_audit import audit  # noqa: E402

OUT = ROOT / "outputs" / "evaluation" / "artifact_audit.json"


def main() -> int:
    res = audit()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: res[k] for k in ("threshold", "scaler", "recorded_split", "calibration_code_in_repository")}, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
