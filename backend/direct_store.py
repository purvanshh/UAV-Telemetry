"""Storage for Direct AI Telemetry Analyses, separate from the flight store.

Layout: ``<root>/<analysis_id>/{meta.json, summary.json, table.parquet, report.json}``. Nothing here reads
or writes ``data/flights`` and no function here knows about LSTM results, so the two cannot overwrite each
other. ``DIRECT_DATA_DIR`` overrides the root; by default it is ``<DATA_DIR>/../direct``.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.store import FlightNotFound, InvalidFlightIdError, _atomic_write_text, _thread_lock, resolve_data_dir
from src.limits import InvalidFlightId, validate_flight_id


def resolve_root() -> Path:
    env = os.environ.get("DIRECT_DATA_DIR")
    return Path(env) if env else resolve_data_dir().parent / "direct"


def new_id() -> str:
    return f"dir-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"


class DirectStore:
    def __init__(self, root: Optional[Path | str] = None):
        self.root = Path(root) if root else resolve_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, analysis_id: str) -> Path:
        try:
            validate_flight_id(analysis_id)
        except InvalidFlightId as exc:
            raise InvalidFlightIdError(str(exc)) from exc
        d = self.root / analysis_id
        if d.is_symlink() or not d.resolve().is_relative_to(self.root.resolve()):
            raise InvalidFlightIdError("Analysis id resolves outside the store.")
        return d

    def require(self, analysis_id: str) -> Path:
        d = self.dir(analysis_id)
        if not (d / "meta.json").is_file():
            raise FlightNotFound(f"Analysis not found: {analysis_id}")
        return d

    def create(self, *, meta: Dict[str, Any], summary: Dict[str, Any], table: pd.DataFrame) -> str:
        aid = new_id()
        d = self.dir(aid)
        tmp = self.root / f".tmp-{uuid.uuid4().hex[:12]}"
        tmp.mkdir()
        try:
            table.to_parquet(tmp / "table.parquet", index=False)
            (tmp / "summary.json").write_text(json.dumps(summary, default=str))
            (tmp / "meta.json").write_text(json.dumps({**meta, "analysis_id": aid,
                                                       "created_at": datetime.now(timezone.utc).isoformat()}, indent=2, default=str))
            os.rename(tmp, d)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return aid

    def load_meta(self, analysis_id: str) -> Dict[str, Any]:
        return json.loads((self.require(analysis_id) / "meta.json").read_text())

    def load_summary(self, analysis_id: str) -> Dict[str, Any]:
        return json.loads((self.require(analysis_id) / "summary.json").read_text())

    def load_table(self, analysis_id: str) -> pd.DataFrame:
        return pd.read_parquet(self.require(analysis_id) / "table.parquet")

    def load_report(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        p = self.require(analysis_id) / "report.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def save_report(self, analysis_id: str, report: Dict[str, Any]) -> None:
        d = self.require(analysis_id)
        with _thread_lock(str(d)):
            _atomic_write_text(d / "report.json", json.dumps(report, indent=2, default=str))

    def update_meta(self, analysis_id: str, **fields: Any) -> None:
        d = self.require(analysis_id)
        with _thread_lock(str(d)):
            meta = json.loads((d / "meta.json").read_text())
            meta.update(fields)
            _atomic_write_text(d / "meta.json", json.dumps(meta, indent=2, default=str))

    def list(self) -> List[Dict[str, Any]]:
        out = []
        for child in sorted(self.root.iterdir(), reverse=True):
            mp = child / "meta.json"
            if child.is_dir() and not child.name.startswith(".") and mp.is_file():
                m = json.loads(mp.read_text())
                rep = child / "report.json"
                m["has_ai_report"] = rep.is_file()
                out.append(m)
        return out

    def find_ai_report(self, input_fingerprint: str) -> Optional[str]:
        """Id of an earlier analysis of identical input that already has a usable AI report (avoids a repeat call)."""
        for m in self.list():
            if m.get("input_fingerprint") == input_fingerprint and m.get("has_ai_report"):
                rep = self.load_report(m["analysis_id"])
                if rep and rep.get("status") in ("ok", "partial"):
                    return m["analysis_id"]
        return None

    def delete(self, analysis_id: str) -> None:
        d = self.require(analysis_id)
        with _thread_lock(str(d)):
            shutil.rmtree(d)
