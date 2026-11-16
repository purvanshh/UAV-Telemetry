"""Storage for Agentic Analyses.

Layout: ``<root>/<analysis_id>/{meta.json, summary.json, source.ulg|source.zip, original.*, report.json}``.
``source.*`` is the ARJUNA log the analysis read: the upload itself, or the ULog converted from a CSV (then the CSV is
kept unmodified as ``original.csv``). The agent and the charts re-read the same bytes.
``ANALYSIS_DATA_DIR`` overrides the root (default ``data/analyses``).
"""

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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "analyses"
_ID = re.compile(r"^an-[0-9A-Za-z-]{1,64}$")
_locks_guard = threading.Lock()
_locks: Dict[str, threading.RLock] = {}


class AnalysisNotFound(LookupError):
    pass


def _lock(key: str) -> threading.RLock:
    with _locks_guard:
        return _locks.setdefault(key, threading.RLock())


def _atomic_write(path: Path, data: bytes) -> None:
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


def resolve_root() -> Path:
    env = os.environ.get("ANALYSIS_DATA_DIR")
    return Path(env) if env else DEFAULT_ROOT


def new_id() -> str:
    return f"an-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"


class AnalysisStore:
    def __init__(self, root: Optional[Path | str] = None):
        self.root = Path(root) if root else resolve_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, aid: str) -> Path:
        if not _ID.match(aid or ""):
            raise AnalysisNotFound(f"Analysis not found: {aid}")
        d = self.root / aid
        if d.is_symlink() or not d.resolve().is_relative_to(self.root.resolve()):
            raise AnalysisNotFound(f"Analysis not found: {aid}")
        return d

    def require(self, aid: str) -> Path:
        d = self.dir(aid)
        if not (d / "meta.json").is_file():
            raise AnalysisNotFound(f"Analysis not found: {aid}")
        return d

    def create(self, *, meta: Dict[str, Any], summary: Dict[str, Any], source: bytes, source_name: str,
               extra: Optional[Dict[str, bytes]] = None) -> str:
        aid = new_id()
        tmp = self.root / f".tmp-{uuid.uuid4().hex[:12]}"
        tmp.mkdir()
        try:
            (tmp / source_name).write_bytes(source)
            for name, data in (extra or {}).items():
                (tmp / name).write_bytes(data)
            (tmp / "summary.json").write_text(json.dumps(summary, default=str))
            (tmp / "meta.json").write_text(json.dumps({**meta, "analysis_id": aid,
                                                       "created_at": datetime.now(timezone.utc).isoformat()},
                                                      indent=2, default=str))
            os.rename(tmp, self.dir(aid))
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return aid

    def load_meta(self, aid: str) -> Dict[str, Any]:
        return json.loads((self.require(aid) / "meta.json").read_text())

    def load_summary(self, aid: str) -> Dict[str, Any]:
        return json.loads((self.require(aid) / "summary.json").read_text())

    def load_source(self, aid: str) -> bytes:
        d = self.require(aid)
        for name in ("source.ulg", "source.zip"):
            if (d / name).is_file():
                return (d / name).read_bytes()
        raise AnalysisNotFound(f"Analysis {aid} has no stored flight log.")

    def load_file(self, aid: str, name: str) -> bytes:
        p = self.require(aid) / name
        if not p.is_file():
            raise AnalysisNotFound(f"Analysis {aid} has no {name}.")
        return p.read_bytes()

    def load_report(self, aid: str) -> Optional[Dict[str, Any]]:
        p = self.require(aid) / "report.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def save_report(self, aid: str, report: Dict[str, Any]) -> None:
        d = self.require(aid)
        with _lock(str(d)):
            _atomic_write(d / "report.json", json.dumps(report, indent=2, default=str).encode())

    def update_meta(self, aid: str, **fields: Any) -> None:
        d = self.require(aid)
        with _lock(str(d)):
            meta = json.loads((d / "meta.json").read_text())
            meta.update(fields)
            _atomic_write(d / "meta.json", json.dumps(meta, indent=2, default=str).encode())

    @contextlib.contextmanager
    def agent_lock(self, aid: str) -> Iterator[bool]:
        """One agent run per analysis at a time. Yields False if another run holds it."""
        lk = _lock(f"agent:{aid}")
        got = lk.acquire(blocking=False)
        try:
            yield got
        finally:
            if got:
                lk.release()

    def list(self) -> List[Dict[str, Any]]:
        out = []
        for child in sorted(self.root.iterdir(), reverse=True):
            mp = child / "meta.json"
            if child.is_dir() and not child.name.startswith(".") and mp.is_file():
                try:
                    out.append(json.loads(mp.read_text()))
                except json.JSONDecodeError:
                    continue
        return out

    def delete(self, aid: str) -> None:
        d = self.require(aid)
        with _lock(str(d)):
            shutil.rmtree(d)
