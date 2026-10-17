"""Flight-level split manifests and leakage checks.

A manifest assigns whole flights to ``train`` / ``val`` / ``test`` (or
``excluded``). Windows therefore can never cross a split: every window belongs
to exactly one flight and a flight has exactly one split.

Leakage rules enforced by :func:`validate_manifest`:

1. a flight id appears once (in one split);
2. byte-identical recordings (``exact_hash``) and identical parsed content
   (``content_hash``) must not appear in different splits — duplicates inside the
   same split are allowed to be listed but are reported;
3. flights sharing the content of some stream (re-exports, e.g. same IMU log)
   across splits are reported as ``near_duplicate`` problems (errors when
   ``strict``).

``kind`` states what the manifest *is*: ``historical_unverified`` (ids copied from
``feature_schema.json``, data not available) or ``new_evaluation``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCHEMA = "split_manifest/1"
SPLITS = ("train", "val", "test", "excluded")


def historical_manifest(feature_schema: Mapping[str, Any]) -> Dict[str, Any]:
    """Ids recorded in ``feature_schema.json``. The flights themselves are absent."""
    sp = feature_schema.get("split") or {}
    flights = [
        {
            "flight_id": str(fid),
            "split": split,
            "data_available": False,
            "calibration_eligible": False,
            "notes": "id recorded in feature_schema.json; source recording not in repository",
        }
        for split in ("train", "val", "test")
        for fid in sp.get(split, [])
    ]
    return {
        "schema": SCHEMA,
        "kind": "historical_unverified",
        "description": (
            "Split ids as recorded in uav_model/feature_schema.json. Not verified: the source "
            "flights, the scores and the split code are not in the repository, so duplicate or "
            "near-duplicate leakage between these lists cannot be checked. Do not treat "
            "evaluation on a new manifest as a reproduction of historical results."
        ),
        "flights": flights,
    }


def deterministic_assignment(
    flights: Sequence[Mapping[str, Any]],
    fractions: Mapping[str, float],
    seed: int,
) -> Dict[str, str]:
    """Assign flight groups to splits by a seeded hash of their content hash.

    Flights with identical content form one group and always land together (the
    duplicates are then marked excluded by the caller, so a recording is only
    ever used once). Order of input does not matter.
    """
    names = [n for n in ("train", "val", "test") if fractions.get(n, 0) > 0]
    total = sum(fractions[n] for n in names)
    groups: Dict[str, List[str]] = {}
    for f in flights:
        key = f.get("content_hash") or f["flight_id"]
        groups.setdefault(key, []).append(f["flight_id"])
    ranked = sorted(groups, key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
    n = len(ranked)
    # largest-remainder allocation so small corpora still populate each split
    raw = {s: fractions[s] / total * n for s in names}
    counts = {s: int(raw[s]) for s in names}
    for s in sorted(names, key=lambda s: -(raw[s] - counts[s]))[: n - sum(counts.values())]:
        counts[s] += 1
    out: Dict[str, str] = {}
    i = 0
    for s in names:
        for k in ranked[i : i + counts[s]]:
            for fid in groups[k]:
                out[fid] = s
        i += counts[s]
    return out


def validate_manifest(manifest: Mapping[str, Any], *, strict: bool = False) -> Dict[str, Any]:
    flights = list(manifest.get("flights", []))
    errors: List[str] = []
    warnings: List[str] = []
    seen: Dict[str, str] = {}
    for f in flights:
        fid, sp = f.get("flight_id"), f.get("split")
        if not fid:
            errors.append("flight entry without flight_id")
            continue
        if sp not in SPLITS:
            errors.append(f"{fid}: invalid split {sp!r}")
        if fid in seen:
            errors.append(
                f"flight_id {fid} listed more than once"
                + (f" (splits {seen[fid]} and {sp})" if seen[fid] != sp else "")
            )
        seen[fid] = sp

    def _group(key: str) -> Dict[str, List[Mapping[str, Any]]]:
        g: Dict[str, List[Mapping[str, Any]]] = {}
        for f in flights:
            if f.get("split") == "excluded" or not f.get(key):
                continue
            g.setdefault(f[key], []).append(f)
        return g

    for key, label in (("exact_hash", "byte-identical"), ("content_hash", "identical-content")):
        for h, members in _group(key).items():
            splits = {m["split"] for m in members}
            if len(splits) > 1:
                errors.append(
                    f"leakage: {label} recordings {[m['flight_id'] for m in members]} "
                    f"are in different splits {sorted(splits)}"
                )
            elif len(members) > 1:
                warnings.append(
                    f"{label} duplicates in split {splits.pop()}: {[m['flight_id'] for m in members]} "
                    "(mark all but one as excluded)"
                )

    stream_owner: Dict[tuple, List[Mapping[str, Any]]] = {}
    for f in flights:
        if f.get("split") == "excluded":
            continue
        for s, h in (f.get("stream_content_sha256") or {}).items():
            stream_owner.setdefault((s, h), []).append(f)
    for (s, _h), members in stream_owner.items():
        ids = {m["flight_id"] for m in members}
        splits = {m["split"] for m in members}
        content = {m.get("content_hash") for m in members}
        if len(ids) > 1 and len(splits) > 1 and len(content) > 1:
            msg = (
                f"near-duplicate: flights {sorted(ids)} share identical {s} data "
                f"across splits {sorted(splits)}"
            )
            (errors if strict else warnings).append(msg)

    counts = {s: sum(1 for f in flights if f.get("split") == s) for s in SPLITS}
    return {"ok": not errors, "errors": errors, "warnings": warnings, "counts": counts}


def load_manifest(path: Path | str) -> Dict[str, Any]:
    m = json.loads(Path(path).read_text())
    if m.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported manifest schema {m.get('schema')!r}")
    return m


def save_manifest(manifest: Mapping[str, Any], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=False, allow_nan=False) + "\n")


def split_flight_ids(manifest: Mapping[str, Any], split: str) -> List[str]:
    return sorted(f["flight_id"] for f in manifest.get("flights", []) if f.get("split") == split)
