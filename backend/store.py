"""Disk-backed flight store under ``data/flights/{flight_id}/``."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:  # POSIX advisory locks; absent on Windows (thread lock still applies)
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

import pandas as pd

from src.event_builder import AnomalyEvent
from src.flight_identity import compute_identity, find_matches
from src.limits import InvalidFlightId, validate_flight_id
from src.report_schema import FlightReport, validate_report
from src.session_state import new_flight_id

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "flights"

STREAMS = ("IMU", "ATT", "BARO")


class FlightStoreError(RuntimeError):
    pass


class FlightNotFound(FlightStoreError):
    pass


class InvalidFlightIdError(FlightNotFound):
    """Flight id is not a safe single path component (rejected, never touched)."""


_GEN_RE = re.compile(r"^gen-[0-9A-Za-z-]{1,64}$")
_CURRENT = "CURRENT"
_locks_guard = threading.Lock()
_thread_locks: Dict[str, threading.RLock] = {}


def _thread_lock(key: str) -> threading.RLock:
    with _locks_guard:
        return _thread_locks.setdefault(key, threading.RLock())


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a sibling temp file + fsync + ``os.replace`` (all-or-nothing)."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def resolve_data_dir(data_dir: Optional[Path | str] = None) -> Path:
    import os

    if data_dir is not None:
        return Path(data_dir)
    env = os.environ.get("DATA_DIR")
    return Path(env) if env else DEFAULT_DATA_DIR


class FlightStore:
    def __init__(self, data_dir: Optional[Path | str] = None):
        self.root = resolve_data_dir(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def flight_dir(self, flight_id: str) -> Path:
        try:
            validate_flight_id(flight_id)
        except InvalidFlightId as exc:
            raise InvalidFlightIdError(str(exc)) from exc
        d = self.root / flight_id
        root = self.root.resolve()
        if d.is_symlink() or not d.resolve().is_relative_to(root) or d.resolve() == root:
            raise InvalidFlightIdError("Flight id resolves outside the flight store.")
        return d

    @contextlib.contextmanager
    def _lock(self, flight_id: str) -> Iterator[None]:
        """Serialise writers of one flight (threads, and processes via flock)."""
        d = self.flight_dir(flight_id)
        with _thread_lock(str(d)):
            if fcntl is None or not d.is_dir():
                yield
                return
            with open(d / ".lock", "a") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)

    def _adir(self, d: Path) -> Path:
        """Directory holding the committed analysis files (legacy: the flight dir)."""
        ptr = d / _CURRENT
        if ptr.is_file():
            name = ptr.read_text().strip()
            if _GEN_RE.match(name) and (d / name).is_dir():
                return d / name
        return d

    def exists(self, flight_id: str) -> bool:
        return (self.flight_dir(flight_id) / "meta.json").is_file()

    def require(self, flight_id: str) -> Path:
        d = self.flight_dir(flight_id)
        if not (d / "meta.json").is_file():
            raise FlightNotFound(f"Flight not found: {flight_id}")
        return d

    def list_flights(self) -> List[Dict[str, Any]]:
        out = []
        for child in sorted(self.root.iterdir(), reverse=True):
            if not child.is_dir():
                continue
            meta_path = child / "meta.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
            out.append(
                {
                    "flight_id": meta.get("flight_id", child.name),
                    "status": meta.get("status", "unknown"),
                    "source": meta.get("source"),
                    "created_at": meta.get("created_at"),
                    "filenames": meta.get("filenames", {}),
                }
            )
        return out

    def create_from_upload(
        self,
        frames: Dict[str, pd.DataFrame],
        *,
        source: str,
        filenames: Dict[str, str],
        file_sizes: Dict[str, int],
        warnings: List[str],
        validation: Dict[str, Any],
        flight_id: Optional[str] = None,
        file_sha256: Optional[Dict[str, str]] = None,
        duplicate_of: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        context: Optional[pd.DataFrame] = None,
    ) -> str:
        try:
            fid = new_flight_id(flight_id)
        except InvalidFlightId as exc:
            raise InvalidFlightIdError(str(exc)) from exc
        d = self.flight_dir(fid)
        if d.exists():
            if self.exists(fid):
                raise FlightStoreError(
                    f"flight_id '{fid}' is already in use; refusing to overwrite an "
                    "existing flight. Choose another id or delete it first."
                )
            shutil.rmtree(d)  # orphan without meta.json; path validated above

        baro_absent = bool((extra_meta or {}).get("baro_absent"))
        for stream in STREAMS:
            if stream not in frames and not (stream == "BARO" and baro_absent):
                raise FlightStoreError(f"Missing frame: {stream}")

        meta = {
            "flight_id": fid,
            "status": "validated",
            "source": source,
            "filenames": filenames,
            "file_sizes": file_sizes,
            "file_sha256": dict(file_sha256 or {}),
            "identity": compute_identity(frames, file_sha256),
            "duplicate_of": duplicate_of,
            "warnings": warnings,
            "validation": validation,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "analyzed_at": None,
        }
        if extra_meta:
            meta.update(extra_meta)
        # Build the whole flight privately, then publish it with one rename.
        tmp = self.root / f".tmp-{uuid.uuid4().hex[:12]}"
        tmp.mkdir()
        try:
            for stream in STREAMS:
                if stream in frames:
                    frames[stream].to_parquet(tmp / f"{stream}.parquet", index=False)
            if context is not None and len(context):
                context.to_parquet(tmp / "CONTEXT.parquet", index=False)
            (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
            os.rename(tmp, d)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return fid

    def identities(self) -> List[tuple]:
        """(flight_id, identity) for every stored flight.

        Flights stored before identity tracking are back-filled from their parquet
        frames (content hash only — the original bytes are gone) and the result is
        persisted in meta.json.
        """
        out = []
        for child in sorted(self.root.iterdir()):
            if not (child / "meta.json").is_file():
                continue
            meta = json.loads((child / "meta.json").read_text())
            ident = meta.get("identity")
            if not ident:
                try:
                    ident = compute_identity(self.load_frames(child.name), meta.get("file_sha256"))
                except FlightStoreError:
                    continue
                meta["identity"] = ident
                _atomic_write_text(child / "meta.json", json.dumps(meta, indent=2))
            out.append((meta.get("flight_id", child.name), ident))
        return out

    def find_duplicates(self, identity: Dict[str, Any]) -> List[Dict[str, str]]:
        return find_matches(identity, self.identities())

    def delete(self, flight_id: str) -> None:
        d = self.require(flight_id)  # validated, inside the store, not a symlink
        with self._lock(flight_id):
            shutil.rmtree(d)

    def load_meta(self, flight_id: str) -> Dict[str, Any]:
        d = self.require(flight_id)
        return json.loads((d / "meta.json").read_text())

    def save_meta(self, flight_id: str, meta: Dict[str, Any]) -> None:
        d = self.require(flight_id)
        _atomic_write_text(d / "meta.json", json.dumps(meta, indent=2))

    def load_frames(self, flight_id: str) -> Dict[str, pd.DataFrame]:
        d = self.require(flight_id)
        frames = {}
        baro_absent = bool(json.loads((d / "meta.json").read_text()).get("baro_absent"))
        for stream in STREAMS:
            path = d / f"{stream}.parquet"
            if stream == "BARO" and baro_absent and not path.is_file():
                continue  # declared optional at upload; the adapter adds a timestamps-only scaffold
            if not path.is_file():
                raise FlightStoreError(f"Missing {stream} parquet for {flight_id}")
            frames[stream] = pd.read_parquet(path)
        return frames

    def load_context(self, flight_id: str) -> Optional[pd.DataFrame]:
        path = self.require(flight_id) / "CONTEXT.parquet"
        return pd.read_parquet(path) if path.is_file() else None

    def save_analysis(
        self,
        flight_id: str,
        *,
        summary: Dict[str, Any],
        windows: Optional[pd.DataFrame],
        events: List[AnomalyEvent],
        events_df: Optional[pd.DataFrame],
        aligned: Optional[pd.DataFrame],
        report: Optional[FlightReport],
        status: str,
    ) -> None:
        """Commit one analysis generation atomically.

        Every file is written into a fresh ``gen-*`` directory that nothing reads
        yet; the ``CURRENT`` pointer is then switched with one ``os.replace``. A
        failure before the switch leaves the previous generation untouched and
        removes the half-written one, so readers never see a mixed analysis.
        """
        d = self.require(flight_id)
        with self._lock(flight_id):
            gen = f"gen-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:6]}"
            g = d / gen
            g.mkdir()
            try:
                (g / "analysis_summary.json").write_text(
                    json.dumps(summary, indent=2, default=str)
                )
                if windows is not None:
                    windows.to_parquet(g / "windows.parquet", index=False)
                (g / "events.json").write_text(
                    json.dumps([e.to_dict() for e in events], indent=2)
                )
                if events_df is not None:
                    events_df.to_parquet(g / "events.parquet", index=False)
                if aligned is not None:
                    aligned.to_parquet(g / "aligned.parquet", index=False)
                if report is not None:
                    (g / "report.json").write_text(
                        json.dumps(report.to_json_dict(), indent=2, allow_nan=False)
                    )
                _atomic_write_text(d / _CURRENT, gen)  # commit point
            except BaseException:
                shutil.rmtree(g, ignore_errors=True)
                raise

            for old in d.iterdir():
                if old.is_dir() and _GEN_RE.match(old.name) and old.name != gen:
                    shutil.rmtree(old, ignore_errors=True)

            meta = self.load_meta(flight_id)
            meta["status"] = status
            meta["analyzed_at"] = datetime.now(timezone.utc).isoformat()
            self.save_meta(flight_id, meta)

    def load_windows(self, flight_id: str) -> Optional[pd.DataFrame]:
        path = self._adir(self.require(flight_id)) / "windows.parquet"
        if not path.is_file():
            return None
        return pd.read_parquet(path)

    def load_aligned(self, flight_id: str) -> Optional[pd.DataFrame]:
        path = self._adir(self.require(flight_id)) / "aligned.parquet"
        if not path.is_file():
            return None
        return pd.read_parquet(path)

    def load_events(self, flight_id: str) -> List[AnomalyEvent]:
        path = self._adir(self.require(flight_id)) / "events.json"
        if not path.is_file():
            return []
        raw = json.loads(path.read_text())
        events = []
        for item in raw:
            events.append(
                AnomalyEvent(
                    event_id=item["event_id"],
                    session_id=int(item["session_id"]),
                    start_time_s=float(item["start_time_s"]),
                    end_time_s=float(item["end_time_s"]),
                    window_indices=list(item.get("window_indices") or []),
                    scores=list(item.get("scores") or []),
                    per_feature_peak=dict(item.get("per_feature_peak") or {}),
                    per_feature_mean=dict(item.get("per_feature_mean") or {}),
                )
            )
        return events

    def load_events_df(self, flight_id: str) -> Optional[pd.DataFrame]:
        path = self._adir(self.require(flight_id)) / "events.parquet"
        if not path.is_file():
            return None
        return pd.read_parquet(path)

    def load_analysis_summary(self, flight_id: str) -> Optional[Dict[str, Any]]:
        path = self._adir(self.require(flight_id)) / "analysis_summary.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def load_report(self, flight_id: str) -> Optional[FlightReport]:
        path = self._adir(self.require(flight_id)) / "report.json"
        if not path.is_file():
            return None
        return validate_report(json.loads(path.read_text()))

    def save_report(self, flight_id: str, report: FlightReport) -> None:
        d = self.require(flight_id)
        with self._lock(flight_id):
            _atomic_write_text(
                self._adir(d) / "report.json",
                json.dumps(report.to_json_dict(), indent=2, allow_nan=False),
            )

    def save_ai_diagnostic(self, flight_id: str, diagnostic: Dict[str, Any]) -> None:
        d = self.require(flight_id)
        with self._lock(flight_id):
            _atomic_write_text(self._adir(d) / "ai_diagnostic.json", json.dumps(diagnostic, indent=2))

    def load_ai_diagnostic(self, flight_id: str) -> Optional[Dict[str, Any]]:
        path = self._adir(self.require(flight_id)) / "ai_diagnostic.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def save_explanation(self, flight_id: str, data: Dict[str, Any]) -> None:
        d = self.require(flight_id)
        with self._lock(flight_id):
            _atomic_write_text(self._adir(d) / "operator_explanation.json", json.dumps(data, indent=2))

    def load_explanation(self, flight_id: str) -> Optional[Dict[str, Any]]:
        path = self._adir(self.require(flight_id)) / "operator_explanation.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def save_chat(self, flight_id: str, chat: Dict[str, Any]) -> None:
        d = self.require(flight_id)
        with self._lock(flight_id):
            _atomic_write_text(self._adir(d) / "ai_chat.json", json.dumps(chat, indent=2))

    def load_chat(self, flight_id: str) -> Dict[str, Any]:
        path = self._adir(self.require(flight_id)) / "ai_chat.json"
        if path.is_file():
            return json.loads(path.read_text())
        return {"flight_id": flight_id, "messages": [], "selected_event_id": None}

    def clear_chat(self, flight_id: str) -> Dict[str, Any]:
        chat = {"flight_id": flight_id, "messages": [], "selected_event_id": None}
        self.save_chat(flight_id, chat)
        return chat
