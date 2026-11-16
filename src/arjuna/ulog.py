"""Open an ARJUNA flight (``.ulg`` or a zipped flight folder) the way the Logging Standard v1.1 requires.

Steps 1-4 of "What the app must do": import with pyulog (never a hand-written parser), verify the hash chain,
check compatibility (refuse unknown schema / frames / profile rather than guess), and build one time axis with
dropouts and missing sensors marked. Gaps are never filled.

Missing data is never zero: float NaN stays NaN, an integer at its type's maximum (e.g. 255) becomes NaN, and
fields listed in the ``unavailable_fields`` information key are dropped and reported.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyulog import ULog

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "arjuna"

KNOWN_SCHEMA_MAJOR = 1
KNOWN_SCHEMA_MINOR_MAX = 1
KNOWN_FRAMES = {"frame_nav": "ENU", "frame_body": "FLU"}
KNOWN_PROFILES = {"engineering", "standard"}
BITFLAG_FIELDS = {"prearm_failures", "mixer_saturated", "fault", "online"}
ZIP_MAX_MEMBERS = 64
ZIP_MAX_RATIO = 200.0


class ArjunaFormatError(ValueError):
    """The file is not an ARJUNA log this app can analyse (wrong format, unknown schema, damaged)."""


def load_json(name: str) -> Dict[str, Any]:
    return json.loads((CONFIG / name).read_text())


ENUMS = load_json("enums_provisional.json")


def enum_name(table: str, value: Any) -> str:
    try:
        key = str(int(value))
    except (TypeError, ValueError):
        return str(value)
    entry = ENUMS.get(table, {}).get(key)
    if entry is None:
        return f"{table.lower()}_{key}"
    return entry[0] if isinstance(entry, list) else str(entry)


@dataclass
class Topic:
    name: str
    instance: int
    df: pd.DataFrame  # column "t" = seconds since log start, then fields
    rate_hz: Optional[float]
    unavailable: List[str]

    @property
    def fields(self) -> List[str]:
        return [c for c in self.df.columns if c not in ("t", "timestamp")]


@dataclass
class ArjunaFlight:
    filename: str
    sha256: str
    size_bytes: int
    info: Dict[str, Any]
    params: Dict[str, Any]
    topics: Dict[Tuple[str, int], Topic]
    messages: List[Dict[str, Any]]
    dropouts: List[Dict[str, Any]]
    t0_us: int
    duration_s: float
    integrity: Dict[str, Any]
    compatibility: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    labels: Optional[Dict[str, Any]] = None
    signature: Optional[Dict[str, Any]] = None
    folder_files: List[str] = field(default_factory=list)
    _air: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # ---- access -------------------------------------------------------------------------------------------------------
    def get(self, name: str, instance: int = 0) -> Optional[pd.DataFrame]:
        tp = self.topics.get((name, instance))
        return tp.df if tp is not None else None

    def instances(self, name: str) -> List[int]:
        return sorted(i for (n, i) in self.topics if n == name)

    def has(self, name: str) -> bool:
        return any(n == name for (n, _) in self.topics)

    # ---- air time -----------------------------------------------------------------------------------------------------
    def _air_series(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._air is None:
            vs = self.get("arjuna_vehicle_status")
            if vs is not None and "landed_state" in vs:
                self._air = (vs["t"].to_numpy(float), np.isin(vs["landed_state"].to_numpy(), [1, 2, 3]))
            else:
                self._air = (np.array([0.0]), np.array([False]))
        return self._air

    def in_air(self, t: np.ndarray) -> np.ndarray:
        tv, air = self._air_series()
        idx = np.clip(np.searchsorted(tv, t, side="right") - 1, 0, len(tv) - 1)
        return air[idx]

    def air_intervals(self) -> List[Tuple[float, float]]:
        tv, air = self._air_series()
        out, start = [], None
        for i, a in enumerate(air):
            if a and start is None:
                start = tv[i]
            elif not a and start is not None:
                out.append((float(start), float(tv[i])))
                start = None
        if start is not None:
            out.append((float(start), float(tv[-1])))
        return out

    def airborne_s(self) -> float:
        tv, air = self._air_series()
        if len(tv) < 2:
            return 0.0
        dt = np.diff(tv, append=tv[-1])
        return float(np.sum(dt[air]))


# ---- input handling -------------------------------------------------------------------------------------------------

def _unzip_folder(raw: bytes, max_total: int) -> Tuple[bytes, str, Dict[str, bytes]]:
    """Return (ulg bytes, ulg name, sidecar files) from a zipped flight folder."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ArjunaFormatError("The ZIP archive could not be opened.") from exc
    members = [m for m in zf.infolist() if not m.is_dir() and not m.filename.startswith("__MACOSX")]
    if len(members) > ZIP_MAX_MEMBERS:
        raise ArjunaFormatError(f"The ZIP has {len(members)} files; a flight folder has at most {ZIP_MAX_MEMBERS}.")
    total = sum(m.file_size for m in members)
    if total > max_total or (len(raw) and total / len(raw) > ZIP_MAX_RATIO):
        raise ArjunaFormatError("The ZIP expands to more data than allowed.")
    ulgs = [m for m in members if m.filename.lower().endswith(".ulg")]
    if not ulgs:
        raise ArjunaFormatError("The ZIP does not contain a .ulg flight log.")
    if len({str(Path(m.filename).parent) for m in ulgs}) > 1:
        raise ArjunaFormatError("The ZIP contains more than one flight folder; upload one flight at a time.")
    ulg = next((m for m in ulgs if Path(m.filename).name == "flight.ulg"), ulgs[0])
    folder = str(Path(ulg.filename).parent)
    side = {}
    for m in members:
        if str(Path(m.filename).parent) == folder and Path(m.filename).name in (
                "metadata.json", "flight.ulg.sig", "labels.json", "README.txt"):
            side[Path(m.filename).name] = zf.read(m)
    return zf.read(ulg), Path(ulg.filename).name, side


def _json_or_none(b: Optional[bytes]) -> Optional[Dict[str, Any]]:
    if not b:
        return None
    try:
        v = json.loads(b.decode("utf-8"))
        return v if isinstance(v, dict) else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


# ---- integrity --------------------------------------------------------------------------------------------------------

def verify_chain(raw: bytes, ulog: ULog, t0_us: int, signature: Optional[Dict[str, Any]],
                 metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recompute the arjuna_log_hash chain: SHA-256(previous chain + block bytes), starting from 32 zero bytes."""
    hd = next((d for d in ulog.data_list if d.name == "arjuna_log_hash"), None)
    signed = bool(signature)
    converted = bool((metadata or {}).get("source")) or "source_format" in ulog.msg_info_dict
    base = {"hash_algorithm": "SHA-256", "signature_present": signed, "signature_checked": False,
            "converted_log": converted}
    if hd is None or len(hd.data["timestamp"]) == 0:
        return {**base, "result": "no_integrity_data", "label": "No hash chain in this log",
                "detail": "The log has no arjuna_log_hash topic, so it cannot be checked for changes."}
    ends = hd.data["block_end_offset"].astype(np.int64)
    hashes = np.vstack([hd.data[f"chain_hash[{i}]"] for i in range(32)]).T.astype(np.uint8)
    chain, start = b"\x00" * 32, 0
    for k, end in enumerate(ends):
        if end > len(raw) or end <= start:
            return {**base, "result": "modified", "label": "Modified",
                    "blocks_checked": k, "block_count": int(len(ends)), "bad_block": k,
                    "detail": f"Block {k} ends at byte {int(end)}, outside the file ({len(raw)} bytes): the file was cut or altered."}
        chain = hashlib.sha256(chain + raw[start:int(end)]).digest()
        if chain != bytes(hashes[k]):
            t = (int(hd.data["timestamp"][k]) - t0_us) / 1e6
            return {**base, "result": "modified", "label": "Modified", "blocks_checked": k, "bad_block": k,
                    "block_count": int(len(ends)), "bad_block_t_s": round(t, 1),
                    "detail": f"Block {k} (written at about t = {t:.1f} s) does not match its hash. Every finding must be "
                              "treated as coming from a modified log."}
        start = int(end)
    last_t = round((int(hd.data["timestamp"][-1]) - t0_us) / 1e6, 1)
    covered = int(ends[-1])
    out = {**base, "blocks_checked": int(len(ends)), "block_count": int(len(ends)), "covered_bytes": covered,
           "file_bytes": len(raw), "final_chain_hash": chain.hex(), "verified_up_to_t_s": last_t}
    sig_hash = str((signature or {}).get("final_chain_hash") or "").lower()
    if signed and re.fullmatch(r"[0-9a-f]{64}", sig_hash) and sig_hash != chain.hex():
        return {**out, "result": "modified", "label": "Modified",
                "detail": "The chain is internally consistent but its final value does not match flight.ulg.sig."}
    if signed:
        return {**out, "result": "chain_verified", "label": "Hash chain verified",
                "detail": f"All {len(ends)} blocks match. The signature file is present but was not checked: no ARJUNA "
                          "public key is configured in this app."}
    why = ("converted from another autopilot's log, so it is unsigned by design" if converted
           else "no signature file (the flight may have ended abnormally)")
    return {**out, "result": "verified_up_to", "label": f"Verified up to t = {last_t} s",
            "detail": f"Unsigned: {why}. The hash chain is intact for all {len(ends)} blocks, up to t = {last_t} s."}


# ---- compatibility ----------------------------------------------------------------------------------------------------

def check_compatibility(info: Dict[str, Any]) -> Dict[str, Any]:
    problems, notes = [], []
    sv = str(info.get("schema_version", ""))
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", sv)
    if not m:
        problems.append(f"schema_version is '{sv or 'missing'}', not an ARJUNA schema version.")
    else:
        major, minor = int(m.group(1)), int(m.group(2))
        if major != KNOWN_SCHEMA_MAJOR:
            problems.append(f"schema_version {sv} has major version {major}; this app reads major version {KNOWN_SCHEMA_MAJOR}.")
        elif minor > KNOWN_SCHEMA_MINOR_MAX:
            notes.append(f"schema_version {sv} is newer than 1.{KNOWN_SCHEMA_MINOR_MAX}; fields added since are ignored.")
    for key, want in KNOWN_FRAMES.items():
        if str(info.get(key, "")) != want:
            problems.append(f"{key} is '{info.get(key, 'missing')}', expected {want}.")
    prof = str(info.get("log_profile", ""))
    if prof not in KNOWN_PROFILES:
        problems.append(f"log_profile is '{prof or 'missing'}', expected engineering or standard.")
    if info.get("enum_table_version", "").startswith("provisional"):
        notes.append(f"Enum tables are provisional ({info['enum_table_version']}); mode and event names may change.")
    return {"ok": not problems, "problems": problems, "notes": notes,
            "schema_version": sv or None, "frame_nav": info.get("frame_nav"), "frame_body": info.get("frame_body"),
            "log_profile": prof or None}


# ---- topics -----------------------------------------------------------------------------------------------------------

def _unavailable(info: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for part in str(info.get("unavailable_fields", "")).split(";"):
        if ":" in part:
            topic, fields = part.split(":", 1)
            out[topic.strip()] = [f.strip() for f in fields.split(",") if f.strip()]
    return out


def _frame(d: Any, t0_us: int, unavailable: List[str]) -> pd.DataFrame:
    cols: Dict[str, np.ndarray] = {"t": (d.data["timestamp"].astype(np.int64) - t0_us) / 1e6}
    for name, arr in d.data.items():
        if name == "timestamp":
            continue
        base = name.split("[", 1)[0]
        if base in unavailable:
            continue
        a = np.asarray(arr)
        if a.dtype == np.bool_:
            cols[name] = a.astype(float)
        elif np.issubdtype(a.dtype, np.integer):
            f = a.astype(float)
            if base not in BITFLAG_FIELDS and a.dtype.itemsize < 8:
                f[a == np.iinfo(a.dtype).max] = np.nan
            cols[name] = f
        else:
            cols[name] = a.astype(float)
    return pd.DataFrame(cols)


def load_flight(raw: bytes, filename: str, *, max_unzipped: int = 2_000_000_000) -> ArjunaFlight:
    side: Dict[str, bytes] = {}
    name = filename
    if raw[:2] == b"PK":
        raw, inner, side = _unzip_folder(raw, max_unzipped)
        name = f"{filename} ({inner})"
    if raw[:7] != b"ULog\x01\x12\x35":
        raise ArjunaFormatError("This is not a ULog file (the first bytes are not the ULog magic).")
    try:
        u = ULog(io.BytesIO(raw), disable_str_exceptions=True)
    except Exception as exc:  # pyulog raises many types on damaged files
        raise ArjunaFormatError(f"pyulog could not read the file ({type(exc).__name__}).") from exc

    info = dict(u.msg_info_dict)
    compat = check_compatibility(info)
    t0 = int(u.start_timestamp)
    unav = _unavailable(info)
    topics: Dict[Tuple[str, int], Topic] = {}
    for d in u.data_list:
        df = _frame(d, t0, unav.get(d.name, []))
        n = len(df)
        span = float(df["t"].iloc[-1] - df["t"].iloc[0]) if n > 1 else 0.0
        topics[(d.name, int(d.multi_id))] = Topic(d.name, int(d.multi_id), df,
                                                    round((n - 1) / span, 2) if span > 0 else None,
                                                    unav.get(d.name, []))
    last = max((float(tp.df["t"].iloc[-1]) for tp in topics.values() if len(tp.df)), default=0.0)
    msgs = [{"t": round((int(m.timestamp) - t0) / 1e6, 3), "level": _level(m.log_level), "text": m.message}
            for m in u.logged_messages]
    drops = [{"t": round((int(x.timestamp) - t0) / 1e6, 3), "duration_ms": int(x.duration)} for x in u.dropouts]
    metadata = _json_or_none(side.get("metadata.json"))
    signature = _json_or_none(side.get("flight.ulg.sig"))
    flight = ArjunaFlight(
        filename=name, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw), info=info,
        params=dict(u.initial_parameters), topics=topics, messages=msgs, dropouts=drops, t0_us=t0,
        duration_s=round(last, 3), integrity=verify_chain(raw, u, t0, signature, metadata), compatibility=compat,
        metadata=metadata, labels=_json_or_none(side.get("labels.json")), signature=signature,
        folder_files=sorted(side),
    )
    return flight


_LEVELS = {"0": "emergency", "1": "alert", "2": "critical", "3": "error", "4": "warning", "5": "notice", "6": "info", "7": "debug"}


def _level(code: Any) -> str:
    c = chr(code) if isinstance(code, int) and code >= 48 else str(code)
    return _LEVELS.get(c, "info")


def is_ulog(raw: bytes) -> bool:
    if raw[:7] == b"ULog\x01\x12\x35":
        return True
    if raw[:2] == b"PK":
        try:
            return any(n.lower().endswith(".ulg") for n in zipfile.ZipFile(io.BytesIO(raw)).namelist())
        except zipfile.BadZipFile:
            return False
    return False
