"""Agentic Analysis API.

POST /api/analyses runs only code: a CSV is first converted into a standard ARJUNA ULog, then every upload is read,
checked and stored the same way, so the page can show findings at once. POST /api/analyses/{id}/agent then runs the AI investigation (consent required) and stores its
report. The agent never changes the code findings.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from backend.analysis_store import AnalysisNotFound, AnalysisStore
from src.agent import runner
from src.agent.sources import SignalError, ULogSource
from src.agent.symptoms import PHASES, SYMPTOMS, TIME_REFS
from src.arjuna.analysis import ANALYSIS_VERSION, analyse
from src.arjuna.convert import convert_csv
from src.arjuna.ulog import ArjunaFlight, ArjunaFormatError, is_ulog, load_flight
from src.limits import get_limits

router = APIRouter(prefix="/api/analyses", tags=["agentic-analysis"])

MAX_POINTS = 600
CODE_SIGNALS: Dict[str, List[tuple]] = {
    "VIB_HIGH": [("arjuna_vibration", f"accel_vibe[{i}]", 0) for i in range(3)] + [("arjuna_vibration", "clip_rate", 0)],
    "CRASH": [("arjuna_attitude", "roll", 0), ("arjuna_attitude", "pitch", 0), ("arjuna_position", "pos_u", 0),
              ("arjuna_actuator_outputs", "thrust_sp", 0)],
    "CTRL_OSC": [("arjuna_attitude", "roll", 0), ("arjuna_attitude", "pitch", 0), ("arjuna_setpoints", "roll_sp", 0),
                 ("arjuna_setpoints", "pitch_sp", 0)],
    "CTRL_SAT": [("arjuna_actuator_outputs", f"pwm_us[{i}]", 0) for i in range(4)],
    "MOTOR_ESC": [("arjuna_actuator_outputs", f"motor_cmd[{i}]", 0) for i in range(4)],
    "POWER": [("arjuna_battery", "voltage", 0), ("arjuna_battery", "current", 0)],
    "GPS_DEGRADED": [("arjuna_gps", "satellites", 0), ("arjuna_gps", "h_acc", 0)],
    "EST_DIVERGE": [("arjuna_estimator_innov", "test_ratio", i) for i in range(4)],
    "COMMS": [("arjuna_rc_input", "failsafe", 0), ("arjuna_rc_input", "rssi", 0)],
}


def store() -> AnalysisStore:
    return AnalysisStore()


async def read_upload_capped(upload: UploadFile, limit: int) -> bytes:
    chunks, total = [], 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"'{upload.filename or 'file'}' exceeds the {limit / 1e6:.0f} MB upload limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _nf(exc: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def _pilot(symptom: Optional[str], time_ref: Optional[str], time_value: Optional[str], note: Optional[str]) -> Optional[Dict[str, Any]]:
    if not symptom:
        return None
    if symptom not in SYMPTOMS:
        raise HTTPException(status_code=422, detail=f"Unknown symptom '{symptom}'. Choose one of: {', '.join(SYMPTOMS)}.")
    out: Dict[str, Any] = {"symptom": symptom, "note": (note or "")[:1000], "time_ref": None, "time_value": None}
    if time_ref:
        if time_ref not in TIME_REFS:
            raise HTTPException(status_code=422, detail=f"Unknown time reference '{time_ref}'.")
        out["time_ref"] = time_ref
        if time_ref == "phase":
            if time_value not in PHASES:
                raise HTTPException(status_code=422, detail=f"Phase must be one of: {', '.join(sorted(PHASES))}.")
            out["time_value"] = time_value
        elif time_value not in (None, ""):
            try:
                out["time_value"] = float(time_value)
            except ValueError:
                raise HTTPException(status_code=422, detail="The time must be a number of seconds.")
    return out


# ---- loading --------------------------------------------------------------------------------------------------------

def _flight(st: AnalysisStore, aid: str) -> ArjunaFlight:
    meta = st.load_meta(aid)
    return load_flight(st.load_source(aid), meta.get("filename", "flight.ulg"))


def _source(st: AnalysisStore, aid: str) -> ULogSource:
    return ULogSource(_flight(st, aid))


def _view(st: AnalysisStore, aid: str) -> Dict[str, Any]:
    meta, summary = st.load_meta(aid), st.load_summary(aid)
    s = dict(summary)
    s.pop("series", None)
    return {"analysis_id": aid, "meta": meta, "summary": s, "report": st.load_report(aid), "label": runner.LABEL,
            "consent_text": runner.consent_text(), "symptoms": {k: v["label"] for k, v in SYMPTOMS.items()},
            "current_version": ANALYSIS_VERSION}


# ---- routes ---------------------------------------------------------------------------------------------------------

@router.post("")
async def create(file: UploadFile = File(...), symptom: Optional[str] = Form(None), time_ref: Optional[str] = Form(None),
                 time_value: Optional[str] = Form(None), note: Optional[str] = Form(None)) -> Any:
    raw = await read_upload_capped(file, get_limits().max_file_bytes)
    name = file.filename or "upload"
    pilot = _pilot(symptom, time_ref, time_value, note)
    st = store()
    extra: Dict[str, bytes] = {}
    conversion = None
    try:
        if is_ulog(raw):
            ulg, source_name = raw, "source.zip" if raw[:2] == b"PK" else "source.ulg"
        else:
            ulg, conversion = convert_csv(raw, name)   # every CSV becomes a standard ARJUNA log first
            source_name, extra = "source.ulg", {"original.csv": raw}
        fl = load_flight(ulg, name if conversion is None else f"{name} (converted)")
    except ArjunaFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    summary = analyse(fl)
    if conversion:
        summary["conversion"] = conversion
    verdict = summary.get("verdict") or {}
    meta = {"filename": name, "kind": "ulog", "converted_from": "csv" if conversion else None,
            "source_profile": (conversion or {}).get("format", {}).get("id"), "analysis_version": ANALYSIS_VERSION,
            "sha256": fl.sha256, "flight_id": fl.info.get("flight_id"), "vehicle_id": fl.info.get("vehicle_id"),
            "refused": summary.get("refused", False), "integrity": (summary.get("integrity") or {}).get("result"),
            "classification": (summary.get("classification") or {}).get("kind"),
            "duration_s": fl.duration_s, "airborne_s": (summary.get("classification") or {}).get("airborne_s"),
            "worst": verdict.get("worst"), "critical": verdict.get("critical", 0), "warning": verdict.get("warning", 0),
            "codes": verdict.get("codes", []), "pilot_report": pilot, "agent": None}
    aid = st.create(meta=meta, summary=summary, source=ulg, source_name=source_name, extra=extra)
    return _view(st, aid)


@router.get("")
def list_analyses() -> List[Dict[str, Any]]:
    return store().list()


@router.get("/{aid}")
def get_analysis(aid: str) -> Any:
    try:
        return _view(store(), aid)
    except AnalysisNotFound as exc:
        raise _nf(exc) from exc


class AgentRequest(BaseModel):
    consent: bool = False
    symptom: Optional[str] = None
    time_ref: Optional[str] = None
    time_value: Optional[str] = None
    note: Optional[str] = None


@router.post("/{aid}/agent")
def run_agent(aid: str, body: AgentRequest) -> Any:
    if not body.consent:
        raise HTTPException(status_code=400, detail="Consent is required: " + runner.consent_text())
    st = store()
    try:
        meta, summary = st.load_meta(aid), st.load_summary(aid)
    except AnalysisNotFound as exc:
        raise _nf(exc) from exc
    if summary.get("refused"):
        raise HTTPException(status_code=422, detail="This log was refused by the compatibility check, so it cannot be investigated.")
    pilot = _pilot(body.symptom, body.time_ref, body.time_value, body.note) if body.symptom else meta.get("pilot_report")
    with st.agent_lock(aid) as got:
        if not got:
            raise HTTPException(status_code=409, detail="An investigation of this flight is already running.")
        src = _source(st, aid)
        findings = summary.get("findings", [])
        rep = runner.investigate(src, summary, findings, pilot_report=pilot, input_fingerprint=meta.get("sha256", ""))
        attempt = {"status": rep["status"], "at": datetime.now(timezone.utc).isoformat(), "message": rep.get("message", ""),
                   "model": rep["provenance"].get("model")}
        if rep["status"] in ("ok", "partial") or st.load_report(aid) is None:
            st.save_report(aid, rep)
        st.update_meta(aid, agent=attempt, pilot_report=pilot)
    view = _view(st, aid)
    if rep["status"] not in ("ok", "partial"):
        view["last_attempt"] = rep
    return view


def _decimate(t: np.ndarray, y: np.ndarray, n: int = MAX_POINTS) -> tuple:
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) <= n:
        return t, y
    keep = {0, len(t) - 1}
    for chunk in np.array_split(np.arange(len(t)), n // 2):
        if len(chunk):
            keep.add(int(chunk[np.argmin(y[chunk])]))
            keep.add(int(chunk[np.argmax(y[chunk])]))
    idx = np.array(sorted(keep))
    return t[idx], y[idx]


def _charts(src: Any, sigs: List[tuple], a: float, b: float) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for topic, field, inst in sigs:
        try:
            t, y, unit = src.series(topic, field, inst)
        except SignalError:
            continue
        m = (t >= a) & (t <= b)
        if not np.isfinite(y[m]).any():
            continue
        tt, yy = _decimate(t[m], y[m])
        key = f"{topic}|{unit}|{field.split('[')[0]}"
        g = groups.setdefault(key, {"title": f"{topic.replace('arjuna_', '')}.{field.split('[')[0]}", "unit": unit, "series": []})
        g["series"].append({"label": f"{field}" + (f" #{inst}" if inst else ""), "t": [round(float(v), 3) for v in tt],
                            "y": [round(float(v), 5) for v in yy]})
    return list(groups.values())[:6]


@router.get("/{aid}/findings/{fid}/context")
def finding_context(aid: str, fid: str) -> Any:
    st = store()
    try:
        summary = st.load_summary(aid)
        f = next((x for x in summary.get("findings", []) if x["id"] == fid), None)
        if f is None:
            raise HTTPException(status_code=404, detail=f"Unknown finding {fid}")
        src = _source(st, aid)
    except AnalysisNotFound as exc:
        raise _nf(exc) from exc
    lo, hi = src.bounds()
    s, e = float(f.get("t_start") or lo), float(f.get("t_end") or hi)
    if f.get("window") == "whole_flight":
        a, b = s, e
    else:
        pad = max(10.0, 0.5 * (e - s))
        a, b = max(lo, s - pad), min(hi, e + pad)
    sigs = [(x["topic"], x["field"], int(x.get("instance") or 0)) for x in f.get("evidence") or []
            if x.get("topic", "").startswith("arjuna_") and x.get("field") not in ("t", "event_code")]
    sigs += [s_ for s_ in CODE_SIGNALS.get(f["code"], []) if s_ not in sigs]
    return {"finding_id": fid, "window": [a, b], "highlight": [s, e], "time_basis": src.time_basis, "charts": _charts(src, sigs, a, b)}


@router.get("/{aid}/series")
def series(aid: str, sig: List[str] = Query(..., description="topic:field:instance"), t_start: Optional[float] = None,
           t_end: Optional[float] = None) -> Any:
    st = store()
    try:
        src = _source(st, aid)
    except AnalysisNotFound as exc:
        raise _nf(exc) from exc
    lo, hi = src.bounds()
    sigs = []
    for s_ in sig[:8]:
        parts = s_.split(":")
        sigs.append((parts[0], parts[1] if len(parts) > 1 else "", int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0))
    a, b = (lo if t_start is None else t_start), (hi if t_end is None else t_end)
    return {"window": [a, b], "time_basis": src.time_basis, "charts": _charts(src, sigs, a, b)}


@router.get("/{aid}/download/{what}")
def download(aid: str, what: str) -> Response:
    """The standard ARJUNA log the analysis read (``ulg``), or the CSV exactly as uploaded (``original``)."""
    st = store()
    try:
        meta = st.load_meta(aid)
        if what == "ulg":
            data, name, media = st.load_source(aid), "flight.ulg", "application/octet-stream"
            if data[:2] == b"PK":
                name, media = "flight_folder.zip", "application/zip"
        elif what == "original" and meta.get("converted_from") == "csv":
            data, name, media = st.load_file(aid, "original.csv"), meta.get("filename") or "original.csv", "text/csv"
        else:
            raise HTTPException(status_code=404, detail=f"Nothing to download as '{what}'.")
    except AnalysisNotFound as exc:
        raise _nf(exc) from exc
    stem = (meta.get("flight_id") or aid).replace('"', "")
    return Response(content=data, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{stem}_{name}"'})


@router.get("/{aid}/export")
def export(aid: str) -> Response:
    try:
        v = _view(store(), aid)
    except AnalysisNotFound as exc:
        raise _nf(exc) from exc
    return Response(content=json.dumps(v, indent=2, default=str), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{aid}_agentic_analysis.json"'})


@router.delete("/{aid}")
def delete(aid: str) -> Dict[str, str]:
    try:
        store().delete(aid)
    except AnalysisNotFound as exc:
        raise _nf(exc) from exc
    return {"status": "deleted", "analysis_id": aid}
