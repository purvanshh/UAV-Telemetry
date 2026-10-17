"""Build the evaluation corpus manifest from flights available locally.

Every distinct recording is assigned to exactly one split by a seeded hash of its
content (``split_manifest.deterministic_assignment``), stratified so that real and
generated flights are split separately. Exact/content duplicates of an already-listed
recording are kept in the manifest as ``excluded`` with ``duplicate_of`` — they are
never used, so one recording cannot appear in two splits.

``calibration_eligible`` is always ``false`` here: it is a human declaration that a
flight is representative behaviour, and this builder cannot make it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.calibration import load_flight_frames
from src.flight_identity import STREAMS, classify, compute_identity
from src.split_manifest import (
    SCHEMA, deterministic_assignment, validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def discover_sources(root: Path = ROOT) -> List[Dict[str, Any]]:
    """Flights present in this checkout: the local flight store and testdata cases."""
    out: List[Dict[str, Any]] = []
    store = root / "data" / "flights"
    if store.is_dir():
        for d in sorted(store.iterdir()):
            meta = d / "meta.json"
            if not meta.is_file() or not all((d / f"{s}.parquet").is_file() for s in STREAMS):
                continue
            m = json.loads(meta.read_text())
            fn = (m.get("filenames") or {}).get("IMU", "")
            mt = re.match(r"(\d{8})_", fn)
            out.append({
                "path": d, "origin": "flight_store", "kind": "real_unreviewed",
                "eval_role": "synthetic_source",
                "log_id": mt.group(1) if mt else None, "store_id": d.name,
                "file_sha256": m.get("file_sha256") or None,
            })
    td = root / "testdata"
    if td.is_dir():
        for d in sorted(td.glob("case*")):
            if not d.is_dir():
                continue
            nominal = d.name.endswith("nominal")
            out.append({
                "path": d, "origin": "testdata", "kind": "synthetic_generated",
                "eval_role": "synthetic_source" if nominal else "smoke_test_known_injections",
                "log_id": None, "store_id": d.name, "file_sha256": None,
            })
    return out


def build_evaluation_manifest(
    sources: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    fractions: Mapping[str, float] = {"val": 0.4, "test": 0.6},
    created: Optional[str] = None,
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    used_ids: set = set()
    for s in sources:
        frames = load_flight_frames(s["path"])
        ident = compute_identity(frames, s.get("file_sha256"))
        fid = f"real-{s['log_id']}" if s.get("log_id") else str(s["store_id"])
        if fid in used_ids:
            fid = f"{fid}@{s['store_id']}"
        used_ids.add(fid)
        entries.append({
            "flight_id": fid,
            "kind": s["kind"],
            "eval_role": s["eval_role"],
            "source": {"origin": s["origin"], "path": _rel(Path(s["path"]))},
            "data_available": True,
            "calibration_eligible": False,
            "exact_hash": ident.get("exact_hash"),
            "content_hash": ident["content_hash"],
            "stream_content_sha256": ident["content_sha256"],
            "_ident": ident,
        })
    # duplicates: keep the lexicographically-first flight_id of each identical-content group
    primary: Dict[str, Dict[str, Any]] = {}
    for e in sorted(entries, key=lambda e: e["flight_id"]):
        p = primary.get(e["content_hash"])
        if p is None:
            primary[e["content_hash"]] = e
            e["duplicate_of"] = None
        else:
            e["duplicate_of"] = p["flight_id"]
            e["split"] = "excluded"
            e["notes"] = f"{classify(e['_ident'], p['_ident'])} of {p['flight_id']}; never used"
    for stratum in sorted({e["kind"] for e in entries}):
        group = [e for e in entries if e["kind"] == stratum and e["duplicate_of"] is None]
        assign = deterministic_assignment(group, fractions, seed)
        for e in group:
            e["split"] = assign[e["flight_id"]]
    for e in entries:
        e.pop("_ident")
    entries.sort(key=lambda e: e["flight_id"])
    man = {
        "schema": SCHEMA,
        "kind": "new_evaluation",
        "created": created,
        "seed": seed,
        "fractions": dict(fractions),
        "description": (
            "NEW evaluation split over the flights available in this checkout. It is not a "
            "reproduction of the historical train/val/test split (see historical_split.json). "
            "No flight here is reviewed as normal, so none is calibration_eligible."
        ),
        "flights": entries,
    }
    man["validation"] = validate_manifest(man)
    return man


def resolve_flight(flight: Mapping[str, Any]):
    """Load a manifest flight's frames and verify they match the recorded content hash."""
    frames = load_flight_frames(ROOT / flight["source"]["path"])
    want = flight.get("content_hash")
    if want and compute_identity(frames)["content_hash"] != want:
        raise ValueError(
            f"{flight['flight_id']}: data at {flight['source']['path']} do not match the manifest "
            "content hash (different recording than the one the split was built for)."
        )
    return frames
