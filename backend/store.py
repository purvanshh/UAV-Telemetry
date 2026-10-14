"""Disk-backed flight store under ``data/flights/{flight_id}/``."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.event_builder import AnomalyEvent
from src.report_schema import FlightReport, validate_report
from src.session_state import new_flight_id

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "flights"

STREAMS = ("IMU", "ATT", "BARO")


class FlightStoreError(RuntimeError):
    pass


class FlightNotFound(FlightStoreError):
    pass


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
        return self.root / flight_id

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
    ) -> str:
        fid = new_flight_id(flight_id)
        d = self.flight_dir(fid)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

        for stream in STREAMS:
            if stream not in frames:
                raise FlightStoreError(f"Missing frame: {stream}")
            frames[stream].to_parquet(d / f"{stream}.parquet", index=False)

        meta = {
            "flight_id": fid,
            "status": "validated",
            "source": source,
            "filenames": filenames,
            "file_sizes": file_sizes,
            "warnings": warnings,
            "validation": validation,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "analyzed_at": None,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return fid

    def delete(self, flight_id: str) -> None:
        d = self.require(flight_id)
        shutil.rmtree(d)

    def load_meta(self, flight_id: str) -> Dict[str, Any]:
        d = self.require(flight_id)
        return json.loads((d / "meta.json").read_text())

    def save_meta(self, flight_id: str, meta: Dict[str, Any]) -> None:
        d = self.require(flight_id)
        (d / "meta.json").write_text(json.dumps(meta, indent=2))

    def load_frames(self, flight_id: str) -> Dict[str, pd.DataFrame]:
        d = self.require(flight_id)
        frames = {}
        for stream in STREAMS:
            path = d / f"{stream}.parquet"
            if not path.is_file():
                raise FlightStoreError(f"Missing {stream} parquet for {flight_id}")
            frames[stream] = pd.read_parquet(path)
        return frames

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
        d = self.require(flight_id)
        (d / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str))

        if windows is not None:
            windows.to_parquet(d / "windows.parquet", index=False)
        elif (d / "windows.parquet").exists():
            (d / "windows.parquet").unlink()

        events_payload = [e.to_dict() for e in events]
        (d / "events.json").write_text(json.dumps(events_payload, indent=2))

        if events_df is not None:
            events_df.to_parquet(d / "events.parquet", index=False)
        elif (d / "events.parquet").exists():
            (d / "events.parquet").unlink()

        if aligned is not None:
            aligned.to_parquet(d / "aligned.parquet", index=False)
        elif (d / "aligned.parquet").exists():
            (d / "aligned.parquet").unlink()

        if report is not None:
            (d / "report.json").write_text(
                json.dumps(report.to_json_dict(), indent=2, allow_nan=False)
            )
        elif (d / "report.json").exists():
            (d / "report.json").unlink()

        # Clear prior AI when re-analyzing
        for name in ("ai_diagnostic.json", "ai_chat.json"):
            p = d / name
            if p.exists():
                p.unlink()

        meta = self.load_meta(flight_id)
        meta["status"] = status
        meta["analyzed_at"] = datetime.now(timezone.utc).isoformat()
        self.save_meta(flight_id, meta)

    def load_windows(self, flight_id: str) -> Optional[pd.DataFrame]:
        path = self.require(flight_id) / "windows.parquet"
        if not path.is_file():
            return None
        return pd.read_parquet(path)

    def load_aligned(self, flight_id: str) -> Optional[pd.DataFrame]:
        path = self.require(flight_id) / "aligned.parquet"
        if not path.is_file():
            return None
        return pd.read_parquet(path)

    def load_events(self, flight_id: str) -> List[AnomalyEvent]:
        path = self.require(flight_id) / "events.json"
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
        path = self.require(flight_id) / "events.parquet"
        if not path.is_file():
            return None
        return pd.read_parquet(path)

    def load_analysis_summary(self, flight_id: str) -> Optional[Dict[str, Any]]:
        path = self.require(flight_id) / "analysis_summary.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def load_report(self, flight_id: str) -> Optional[FlightReport]:
        path = self.require(flight_id) / "report.json"
        if not path.is_file():
            return None
        return validate_report(json.loads(path.read_text()))

    def save_report(self, flight_id: str, report: FlightReport) -> None:
        d = self.require(flight_id)
        (d / "report.json").write_text(
            json.dumps(report.to_json_dict(), indent=2, allow_nan=False)
        )

    def save_ai_diagnostic(self, flight_id: str, diagnostic: Dict[str, Any]) -> None:
        d = self.require(flight_id)
        (d / "ai_diagnostic.json").write_text(json.dumps(diagnostic, indent=2))

    def load_ai_diagnostic(self, flight_id: str) -> Optional[Dict[str, Any]]:
        path = self.require(flight_id) / "ai_diagnostic.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def save_chat(self, flight_id: str, chat: Dict[str, Any]) -> None:
        d = self.require(flight_id)
        (d / "ai_chat.json").write_text(json.dumps(chat, indent=2))

    def load_chat(self, flight_id: str) -> Dict[str, Any]:
        path = self.require(flight_id) / "ai_chat.json"
        if path.is_file():
            return json.loads(path.read_text())
        return {"flight_id": flight_id, "messages": [], "selected_event_id": None}

    def clear_chat(self, flight_id: str) -> Dict[str, Any]:
        chat = {"flight_id": flight_id, "messages": [], "selected_event_id": None}
        self.save_chat(flight_id, chat)
        return chat
