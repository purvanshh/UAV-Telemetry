"""Streamlit session helpers for upload, analysis, and AI diagnostics."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.chat_service import ensure_chat_for_flight, reset_chat
from src.upload import UploadResult

# Keys stored in st.session_state
FLIGHT_KEY = "flight"
FLIGHT_ID_KEY = "flight_id"
UPLOAD_META_KEY = "upload_meta"
VALIDATION_KEY = "validation"
STATUS_KEY = "flight_status"
ANALYSIS_KEY = "analysis_result"
ANALYSIS_DF_KEY = "analysis_windows"
EVENTS_KEY = "analysis_events"
EVENTS_DF_KEY = "analysis_events_df"
ALIGNED_KEY = "analysis_aligned"
REPORT_KEY = "analysis_report"
AI_DIAGNOSTIC_KEY = "ai_diagnostic"
AI_CHAT_KEY = "ai_chat"
AI_ERROR_KEY = "ai_error"


def new_flight_id(user_id: Optional[str] = None) -> str:
    """Return a user-provided ID or a generated identifier."""
    if user_id and str(user_id).strip():
        return str(user_id).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"flight-{stamp}-{uuid.uuid4().hex[:6]}"


def _clear_analysis(session: Dict[str, Any]) -> None:
    for key in (
        ANALYSIS_KEY,
        ANALYSIS_DF_KEY,
        EVENTS_KEY,
        EVENTS_DF_KEY,
        ALIGNED_KEY,
        REPORT_KEY,
        AI_DIAGNOSTIC_KEY,
        AI_CHAT_KEY,
        AI_ERROR_KEY,
    ):
        session.pop(key, None)


def reset_flight(session: Dict[str, Any]) -> None:
    """Clear the current flight from session state."""
    for key in (
        FLIGHT_KEY,
        FLIGHT_ID_KEY,
        UPLOAD_META_KEY,
        VALIDATION_KEY,
        STATUS_KEY,
        ANALYSIS_KEY,
        ANALYSIS_DF_KEY,
        EVENTS_KEY,
        EVENTS_DF_KEY,
        ALIGNED_KEY,
        REPORT_KEY,
        AI_DIAGNOSTIC_KEY,
        AI_CHAT_KEY,
        AI_ERROR_KEY,
    ):
        if key in session:
            del session[key]
    session[STATUS_KEY] = "idle"


def has_flight(session: Dict[str, Any]) -> bool:
    flight = session.get(FLIGHT_KEY)
    return isinstance(flight, dict) and all(
        s in flight and isinstance(flight[s], pd.DataFrame) for s in ("IMU", "ATT", "BARO")
    )


def store_upload(
    session: Dict[str, Any],
    upload: UploadResult,
    flight_id: Optional[str] = None,
) -> str:
    """Persist a successful upload into session state (in memory only)."""
    if not upload.ok or upload.validation is None or not upload.validation.ok:
        raise ValueError("Cannot store an unsuccessful upload.")

    _clear_analysis(session)

    fid = new_flight_id(flight_id)
    session[FLIGHT_KEY] = {
        "IMU": upload.frames["IMU"],
        "ATT": upload.frames["ATT"],
        "BARO": upload.frames["BARO"],
    }
    session[FLIGHT_ID_KEY] = fid
    session[UPLOAD_META_KEY] = {
        "source": upload.source,
        "filenames": dict(upload.filenames),
        "file_sizes": dict(upload.file_sizes),
        "warnings": list(upload.all_warnings),
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }
    session[VALIDATION_KEY] = upload.validation.to_dict()
    session[STATUS_KEY] = "validated"
    return fid


def store_analysis(session: Dict[str, Any], result: Any) -> None:
    """Store analysis summary, windows, events, aligned grid and report.

    Clears prior AI diagnostic/chat so hypotheses cannot leak onto a new run.
    """
    session.pop(AI_DIAGNOSTIC_KEY, None)
    session.pop(AI_CHAT_KEY, None)
    session.pop(AI_ERROR_KEY, None)

    session[ANALYSIS_KEY] = result.summary_dict()
    session[ANALYSIS_DF_KEY] = result.windows
    session[EVENTS_KEY] = list(result.events or [])
    session[EVENTS_DF_KEY] = result.events_df
    session[ALIGNED_KEY] = result.aligned
    session[REPORT_KEY] = result.report
    if result.ok:
        session[STATUS_KEY] = "analyzed"
    elif result.status == "insufficient_data":
        session[STATUS_KEY] = "insufficient_data"
    else:
        session[STATUS_KEY] = "analysis_error"


def get_analysis(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return session.get(ANALYSIS_KEY)


def get_analysis_windows(session: Dict[str, Any]) -> Optional[pd.DataFrame]:
    df = session.get(ANALYSIS_DF_KEY)
    return df if isinstance(df, pd.DataFrame) else None


def get_events(session: Dict[str, Any]) -> List[Any]:
    return list(session.get(EVENTS_KEY) or [])


def get_events_df(session: Dict[str, Any]) -> Optional[pd.DataFrame]:
    df = session.get(EVENTS_DF_KEY)
    return df if isinstance(df, pd.DataFrame) else None


def get_aligned(session: Dict[str, Any]) -> Optional[pd.DataFrame]:
    df = session.get(ALIGNED_KEY)
    return df if isinstance(df, pd.DataFrame) else None


def get_report(session: Dict[str, Any]) -> Any:
    return session.get(REPORT_KEY)


def store_ai_diagnostic(session: Dict[str, Any], diagnostic: Any, report_with_ai: Any = None) -> None:
    session[AI_DIAGNOSTIC_KEY] = diagnostic
    session.pop(AI_ERROR_KEY, None)
    if report_with_ai is not None:
        session[REPORT_KEY] = report_with_ai


def get_ai_diagnostic(session: Dict[str, Any]) -> Any:
    return session.get(AI_DIAGNOSTIC_KEY)


def store_ai_error(session: Dict[str, Any], message: str) -> None:
    session[AI_ERROR_KEY] = message


def get_ai_error(session: Dict[str, Any]) -> Optional[str]:
    return session.get(AI_ERROR_KEY)


def get_or_create_chat(session: Dict[str, Any], flight_id: str) -> Dict[str, Any]:
    return ensure_chat_for_flight(session, flight_id, AI_CHAT_KEY)


def clear_chat(session: Dict[str, Any], flight_id: str) -> Dict[str, Any]:
    return reset_chat(session, flight_id, AI_CHAT_KEY)


def get_flight_summary(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not has_flight(session):
        return None
    meta = session.get(UPLOAD_META_KEY, {})
    validation = session.get(VALIDATION_KEY, {})
    return {
        "flight_id": session.get(FLIGHT_ID_KEY),
        "status": session.get(STATUS_KEY, "unknown"),
        "source": meta.get("source"),
        "filenames": meta.get("filenames", {}),
        "file_sizes": meta.get("file_sizes", {}),
        "warnings": meta.get("warnings", []),
        "validation": validation,
        "analysis": session.get(ANALYSIS_KEY),
        "has_ai_diagnostic": session.get(AI_DIAGNOSTIC_KEY) is not None,
    }
