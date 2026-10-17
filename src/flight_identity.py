"""Deterministic flight identity for duplicate-upload detection.

Three levels, strongest first:

``exact_duplicate``    – every stream file is byte-identical (SHA-256 of the raw
                         CSV bytes). Packaging (ZIP vs three CSVs, file names)
                         does not matter because hashes are per stream file.
``content_duplicate``  – the parsed data are identical but the bytes differ
                         (line endings, column order, header spacing/case,
                         row order). Computed from a canonical serialisation.
``partial_overlap``    – some, not all, streams carry identical content (e.g. the
                         same IMU log re-exported with a different ATT).
``similar_metadata``   – same start time, duration and row counts but different
                         data. Informational only: distinct flights can look alike.

Only the first two are treated as duplicates; the last two never block an
upload and never merge records.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.timebase import find_timestamp_column

STREAMS = ("IMU", "ATT", "BARO")
EXACT = "exact_duplicate"
CONTENT = "content_duplicate"
PARTIAL = "partial_overlap"
SIMILAR = "similar_metadata"
DISTINCT = "distinct"
BLOCKING = (EXACT, CONTENT)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _norm_col(c: Any) -> str:
    return re.sub(r"[\s_]+", "", str(c)).lower()


def canonical_stream_hash(df: pd.DataFrame) -> str:
    """Hash of parsed content, independent of column order/case, line endings, row order."""
    cols = sorted(df.columns, key=lambda c: (_norm_col(c), str(c)))
    ts = find_timestamp_column(list(df.columns))
    work = df[cols]
    if ts is not None:
        order = np.argsort(pd.to_numeric(df[ts], errors="coerce").to_numpy(dtype="float64"), kind="stable")
        work = work.iloc[order]
    h = hashlib.sha256()
    h.update(f"{len(work)}x{len(cols)}".encode())
    for c in cols:
        h.update(b"\x00" + _norm_col(c).encode() + b"\x00")
        num = pd.to_numeric(work[c], errors="coerce")
        if num.notna().sum() == work[c].notna().sum():  # numeric column
            h.update(num.to_numpy(dtype="float64").tobytes())
        else:
            h.update("\x1f".join(map(str, work[c].tolist())).encode())
    return h.hexdigest()


def _combine(hashes: Mapping[str, str]) -> Optional[str]:
    if not all(s in hashes for s in STREAMS):
        return None
    return sha256_bytes("|".join(f"{s}:{hashes[s]}" for s in STREAMS).encode())


def _fingerprint(frames: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    fp: Dict[str, Any] = {"rows": {s: int(len(frames[s])) for s in STREAMS if s in frames}}
    ts = {}
    for s in STREAMS:
        if s not in frames:
            continue
        col = find_timestamp_column(list(frames[s].columns))
        if col is not None:
            v = pd.to_numeric(frames[s][col], errors="coerce").dropna()
            if len(v):
                ts[s] = [float(v.min()), float(v.max())]
    fp["time_range_raw"] = ts
    return fp


def compute_identity(
    frames: Mapping[str, pd.DataFrame],
    file_sha256: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    content = {s: canonical_stream_hash(frames[s]) for s in STREAMS if s in frames}
    ident: Dict[str, Any] = {
        "content_sha256": content,
        "content_hash": _combine(content),
        "fingerprint": _fingerprint(frames),
    }
    if file_sha256:
        ident["file_sha256"] = dict(file_sha256)
        ident["exact_hash"] = _combine(file_sha256)
    return ident


def classify(new: Mapping[str, Any], existing: Mapping[str, Any]) -> str:
    """Relationship of ``new`` to one already-stored flight identity."""
    ne, ee = new.get("exact_hash"), existing.get("exact_hash")
    if ne and ee and ne == ee:
        return EXACT
    nc, ec = new.get("content_hash"), existing.get("content_hash")
    if nc and ec and nc == ec:
        # Byte-identical files imply identical content; otherwise it is repackaged.
        return EXACT if (ne and ee and ne == ee) else CONTENT
    ncs, ecs = new.get("content_sha256") or {}, existing.get("content_sha256") or {}
    if any(s in ncs and ecs.get(s) == ncs[s] for s in STREAMS):
        return PARTIAL
    nf, ef = new.get("fingerprint") or {}, existing.get("fingerprint") or {}
    if nf and nf == ef:
        return SIMILAR
    return DISTINCT


def find_matches(
    new: Mapping[str, Any], existing: Iterable[tuple[str, Mapping[str, Any]]]
) -> List[Dict[str, str]]:
    """All non-distinct relations of ``new`` to ``existing`` (flight_id, identity) pairs."""
    out = []
    for fid, ident in existing:
        kind = classify(new, ident)
        if kind != DISTINCT:
            out.append({"flight_id": fid, "kind": kind})
    order = {EXACT: 0, CONTENT: 1, PARTIAL: 2, SIMILAR: 3}
    return sorted(out, key=lambda m: (order[m["kind"]], m["flight_id"]))


def duplicate_message(matches: List[Dict[str, str]]) -> Optional[str]:
    """User-facing message for a blocking duplicate, else None."""
    block = [m for m in matches if m["kind"] in BLOCKING]
    if not block:
        return None
    m = block[0]
    if m["kind"] == EXACT:
        return (
            f"Duplicate upload: these files are byte-identical to existing flight "
            f"'{m['flight_id']}'. Open that flight, or resubmit with allow_duplicate "
            "to store a second record (it will be marked as a duplicate and excluded "
            "from evaluation splits)."
        )
    return (
        f"Duplicate upload: the data match existing flight '{m['flight_id']}' "
        "(same content, different file packaging). Open that flight, or resubmit with "
        "allow_duplicate to store a second record (marked as a duplicate)."
    )
