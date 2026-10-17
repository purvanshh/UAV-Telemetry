"""Re-score stored flights and compare with the saved windows.parquet.

Reproducibility check for pipeline changes: with unchanged input data and model
artifacts, per-window scores, flags and window times must match what was stored
when the flight was first analysed. Never writes to the flight store.

    python scripts/verify_stored_flights.py [--data-dir data/flights] [--out report.json]
Exit code 1 if any stored flight does not reproduce.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.store import FlightStore  # noqa: E402
from src.inference_service import analyze_flight  # noqa: E402


def compare_flight(store: FlightStore, flight_id: str, atol: float = 1e-6) -> dict:
    stored = store.load_windows(flight_id)
    if stored is None:
        return {"flight_id": flight_id, "compared": False, "reason": "no stored windows"}
    meta = store.load_meta(flight_id)
    res = analyze_flight(
        store.load_frames(flight_id),
        flight_id=flight_id,
        upload_warnings=list(meta.get("warnings") or []),
    )
    new = res.windows
    out = {"flight_id": flight_id, "compared": True, "stored_windows": int(len(stored)),
           "new_windows": 0 if new is None else int(len(new))}
    if new is None or len(new) != len(stored):
        out["reproduced"] = False
        out["reason"] = "window count differs"
        return out
    ds = np.abs(new["anomaly_score"].to_numpy() - stored["anomaly_score"].to_numpy())
    dt = np.abs(new[["t_start", "t_end"]].to_numpy() - stored[["t_start", "t_end"]].to_numpy())
    out.update(
        max_abs_score_diff=float(ds.max()) if len(ds) else 0.0,
        max_abs_time_diff_s=float(dt.max()) if dt.size else 0.0,
        flags_identical=bool((new["is_anomaly"].to_numpy() == stored["is_anomaly"].to_numpy()).all()),
        sessions_identical=bool((new["session_id"].to_numpy() == stored["session_id"].to_numpy()).all()),
    )
    out["reproduced"] = bool(
        out["flags_identical"] and out["sessions_identical"]
        and out["max_abs_score_diff"] <= atol and out["max_abs_time_diff_s"] <= 1e-9
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "flights"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    store = FlightStore(a.data_dir)
    rows = [compare_flight(store, f["flight_id"]) for f in sorted(store.list_flights(), key=lambda f: f["flight_id"])]
    for r in rows:
        print(json.dumps(r))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
    ok = all(r.get("reproduced", True) for r in rows)
    print("ALL REPRODUCED" if ok and rows else ("NO FLIGHTS" if not rows else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
