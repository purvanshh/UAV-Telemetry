"""Direct AI Telemetry Analysis routes. Independent of the LSTM detector and the flight store."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from backend.direct_store import DirectStore
from backend.routers.flights import read_upload_capped
from backend.store import FlightNotFound
from src import direct_llm
from src.direct_analysis import ANALYSIS_VERSION, ProcessingLimit, analyze_table, event_context
from src.limits import get_limits

router = APIRouter(prefix="/api/direct", tags=["direct-ai-analysis"])


def get_direct_store() -> DirectStore:
    return DirectStore()


def _nf(exc: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def _run_ai(store: DirectStore, aid: str, consent: bool) -> Dict[str, Any]:
    """Generate (or reuse) the AI report. A failed attempt never replaces a good earlier report."""
    summary = store.load_summary(aid)
    existing = store.load_report(aid)
    current = lambda r: (r["provenance"].get("input_fingerprint") == summary["input_fingerprint"]  # noqa: E731
                         and r["provenance"].get("prompt_version") == direct_llm.PROMPT_VERSION)
    if consent and existing and existing.get("status") in ("ok", "partial") and current(existing):
        return {**existing, "reused": True}
    if consent:
        twin = store.find_ai_report(summary["input_fingerprint"])
        twin_rep = store.load_report(twin) if twin and twin != aid else None
        if twin_rep and current(twin_rep):
            rep = twin_rep
            rep = {**rep, "reused": True, "reused_from": twin}
            store.save_report(aid, rep)
            store.update_meta(aid, ai_attempt={"status": rep["status"], "at": datetime.now(timezone.utc).isoformat(), "reused_from": twin})
            return rep
    rep = direct_llm.generate(summary, consent=consent)
    fail = rep.get("failure") or {}
    store.update_meta(aid, ai_attempt={"status": rep["status"], "message": rep.get("message", ""),
                                       "reason": fail.get("reason"), "fields": (fail.get("fields") or [])[:6],
                                       "attempts": fail.get("attempts"), "at": datetime.now(timezone.utc).isoformat()})
    if rep["status"] in ("ok", "partial"):
        store.save_report(aid, rep)
    elif existing:
        return {**existing, "message": f"A new attempt failed ({rep['status']}: {rep.get('message', '')}); showing the earlier report.",
                "last_attempt": {"status": rep["status"]}}
    return rep


def _view(store: DirectStore, aid: str, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = store.load_meta(aid)
    summary = store.load_summary(aid)
    rep = report if report is not None else store.load_report(aid)
    return {"analysis_id": aid, "label": direct_llm.LABEL, "mode": direct_llm.MODE, "meta": meta, "summary": summary,
            "report": rep, "consent_text": direct_llm.CONSENT_TEXT,
            "stale_analysis": summary.get("analysis_version") != ANALYSIS_VERSION or "operator" not in summary,
            "separation_note": "This analysis is stored apart from flights and LSTM results and never changes them."}


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    llm_consent: bool = Form(False),
    store: DirectStore = Depends(get_direct_store),
) -> Any:
    raw = await read_upload_capped(file, get_limits().max_file_bytes)
    name = file.filename or "telemetry.csv"
    try:
        s = analyze_table(raw, name)
    except ProcessingLimit as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    aid = store.create(
        meta={"filename": name, "file_sha256": s.data["file_sha256"], "input_fingerprint": s.data["input_fingerprint"],
              "analysis_mode": direct_llm.MODE, "analysis_version": s.data["analysis_version"],
              "schema_profile": {k: s.data["profile"].get(k) for k in ("id", "version", "match")},
              "rows": s.data["data_quality"]["rows"], "columns": s.data["data_quality"]["columns"],
              "events": len(s.data["events"]), "ai_attempt": None},
        summary=s.data, table=s.table,
    )
    rep = _run_ai(store, aid, llm_consent) if llm_consent else None
    out = _view(store, aid, rep)
    out["stage"] = (rep or {}).get("status", "summary_only") if llm_consent else "summary_only"
    return out


@router.get("")
def list_analyses(store: DirectStore = Depends(get_direct_store)) -> List[Dict[str, Any]]:
    return store.list()


@router.get("/{analysis_id}")
def get_analysis(analysis_id: str, store: DirectStore = Depends(get_direct_store)) -> Any:
    try:
        return _view(store, analysis_id)
    except FlightNotFound as exc:
        raise _nf(exc) from exc


class AiRequest(BaseModel):
    consent: bool = False


@router.post("/{analysis_id}/ai")
def generate_ai(analysis_id: str, body: AiRequest, store: DirectStore = Depends(get_direct_store)) -> Any:
    if not body.consent:
        raise HTTPException(status_code=400, detail="Consent is required: " + direct_llm.CONSENT_TEXT)
    try:
        rep = _run_ai(store, analysis_id, True)
        return _view(store, analysis_id, rep)
    except FlightNotFound as exc:
        raise _nf(exc) from exc


@router.get("/{analysis_id}/events/{event_id}/context")
def event_ctx(analysis_id: str, event_id: str, store: DirectStore = Depends(get_direct_store)) -> Any:
    try:
        summary, table = store.load_summary(analysis_id), store.load_table(analysis_id)
    except FlightNotFound as exc:
        raise _nf(exc) from exc
    try:
        return event_context(summary, table, event_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown event: {event_id}")


@router.get("/{analysis_id}/rows")
def rows(analysis_id: str, start: int = 0, limit: int = 200, store: DirectStore = Depends(get_direct_store)) -> Any:
    try:
        t = store.load_table(analysis_id)
    except FlightNotFound as exc:
        raise _nf(exc) from exc
    limit = max(1, min(limit, 1000))
    sub = t.iloc[max(0, start) : max(0, start) + limit]
    return {"columns": list(sub.columns), "rows": sub.astype(object).where(sub.notna(), None).to_dict(orient="records"),
            "total_rows": int(len(t)), "returned": int(len(sub))}


@router.get("/{analysis_id}/export")
def export(analysis_id: str, store: DirectStore = Depends(get_direct_store)) -> Response:
    try:
        v = _view(store, analysis_id)
    except FlightNotFound as exc:
        raise _nf(exc) from exc
    import json

    v["summary"] = {k: val for k, val in v["summary"].items() if k != "series"}
    return Response(content=json.dumps(v, indent=2, default=str), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{analysis_id}_ai_analysis.json"'})


@router.delete("/{analysis_id}")
def delete(analysis_id: str, store: DirectStore = Depends(get_direct_store)) -> Dict[str, str]:
    try:
        store.delete(analysis_id)
    except FlightNotFound as exc:
        raise _nf(exc) from exc
    return {"status": "deleted", "analysis_id": analysis_id}
