"""Flight upload, analysis, charts, exports, and AI routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from backend.deps import get_store
from backend.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatStateResponse,
    DiagnosticRequest,
    FlightDetail,
    FlightListItem,
    UploadResponse,
)
from backend.store import FlightNotFound, FlightStore
from src.chat_service import ask_followup, empty_chat_state
from src.diagnostic_schema import DiagnosticReport, validate_diagnostic
from src.diagnostic_trace import build_diagnostic_trace, format_diagnostic_trace
from src.evidence_builder import build_evidence_pack
from src.event_builder import DEFAULT_MAX_GAP_S
from src.inference_service import analyze_flight
from src.llm_service import LLMError, LLMService, api_configured, attach_ai_to_report
from src.report_export import events_to_csv_bytes, report_to_json_bytes, windows_to_csv_bytes
from src.upload import load_individual_csvs, load_zip_archive
from src.visualization import (
    baro_context_figure,
    per_feature_error_figure,
    score_timeline_figure,
    telemetry_around_event_figure,
)

router = APIRouter(prefix="/api/flights", tags=["flights"])


def _store_exc(exc: Exception) -> HTTPException:
    if isinstance(exc, FlightNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


@router.get("", response_model=List[FlightListItem])
def list_flights(store: FlightStore = Depends(get_store)) -> List[FlightListItem]:
    return [FlightListItem(**item) for item in store.list_flights()]


@router.post("/upload", response_model=UploadResponse)
async def upload_flight(
    imu: Optional[UploadFile] = File(None),
    att: Optional[UploadFile] = File(None),
    baro: Optional[UploadFile] = File(None),
    zip_file: Optional[UploadFile] = File(None),
    flight_id: Optional[str] = Form(None),
    store: FlightStore = Depends(get_store),
) -> UploadResponse:
    if zip_file is not None:
        raw = await zip_file.read()
        result = load_zip_archive(raw, archive_name=zip_file.filename or "upload.zip")
    elif imu is not None and att is not None and baro is not None:
        files = {
            "IMU": await imu.read(),
            "ATT": await att.read(),
            "BARO": await baro.read(),
        }
        filenames = {
            "IMU": imu.filename or "IMU.csv",
            "ATT": att.filename or "ATT.csv",
            "BARO": baro.filename or "BARO.csv",
        }
        result = load_individual_csvs(files, filenames=filenames)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either zip_file or all three of imu, att, and baro.",
        )

    if not result.ok:
        return UploadResponse(
            ok=False,
            source=result.source,
            filenames=dict(result.filenames),
            file_sizes=dict(result.file_sizes),
            warnings=list(result.all_warnings),
            errors=list(result.all_errors),
            validation=result.validation.to_dict() if result.validation else None,
        )

    validation = result.validation.to_dict() if result.validation else {}
    fid = store.create_from_upload(
        result.frames,
        source=result.source,
        filenames=dict(result.filenames),
        file_sizes=dict(result.file_sizes),
        warnings=list(result.all_warnings),
        validation=validation,
        flight_id=flight_id,
    )
    return UploadResponse(
        ok=True,
        flight_id=fid,
        status="validated",
        source=result.source,
        filenames=dict(result.filenames),
        file_sizes=dict(result.file_sizes),
        warnings=list(result.all_warnings),
        errors=[],
        validation=validation,
    )


@router.get("/{flight_id}", response_model=FlightDetail)
def get_flight(
    flight_id: str, store: FlightStore = Depends(get_store)
) -> FlightDetail:
    try:
        meta = store.load_meta(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc
    events = store.load_events(flight_id)
    return FlightDetail(
        flight_id=meta["flight_id"],
        status=meta.get("status", "unknown"),
        source=meta.get("source"),
        filenames=meta.get("filenames") or {},
        file_sizes=meta.get("file_sizes") or {},
        warnings=meta.get("warnings") or [],
        validation=meta.get("validation"),
        created_at=meta.get("created_at"),
        analyzed_at=meta.get("analyzed_at"),
        analysis=store.load_analysis_summary(flight_id),
        has_report=store.load_report(flight_id) is not None,
        has_ai_diagnostic=store.load_ai_diagnostic(flight_id) is not None,
        event_count=len(events),
    )


@router.delete("/{flight_id}")
def delete_flight(
    flight_id: str, store: FlightStore = Depends(get_store)
) -> Dict[str, str]:
    try:
        store.delete(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc
    return {"status": "deleted", "flight_id": flight_id}


@router.post("/{flight_id}/analyze", response_model=AnalyzeResponse)
def analyze(
    flight_id: str,
    body: AnalyzeRequest = AnalyzeRequest(),
    store: FlightStore = Depends(get_store),
) -> AnalyzeResponse:
    try:
        frames = store.load_frames(flight_id)
        meta = store.load_meta(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc

    result = analyze_flight(
        frames,
        flight_id=flight_id,
        upload_warnings=list(meta.get("warnings") or []),
        max_gap_s=float(body.max_gap_s or DEFAULT_MAX_GAP_S),
    )

    status = "analyzed" if result.ok else (
        "insufficient_data" if result.status == "insufficient_data" else "analysis_error"
    )
    if result.ok and result.status == "no_anomaly":
        status = "analyzed"

    store.save_analysis(
        flight_id,
        summary=result.summary_dict(),
        windows=result.windows,
        events=list(result.events or []),
        events_df=result.events_df,
        aligned=result.aligned,
        report=result.report,
        status=status if result.ok or result.status == "insufficient_data" else "analysis_error",
    )

    return AnalyzeResponse(
        ok=result.ok,
        flight_id=flight_id,
        status=result.status,
        errors=list(result.errors),
        warnings=list(result.warnings),
        n_windows=result.n_windows,
        n_flagged=result.n_flagged,
        flagged_rate=result.flagged_rate,
        n_events=result.n_events,
        threshold=result.threshold,
        variant=result.variant,
        features=list(result.features),
        score_mean=result.score_mean,
        score_p50=result.score_p50,
        score_p95=result.score_p95,
        score_max=result.score_max,
    )


@router.get("/{flight_id}/windows")
def get_windows(
    flight_id: str, store: FlightStore = Depends(get_store)
) -> Any:
    try:
        windows = store.load_windows(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc
    if windows is None:
        return []
    return windows.to_dict(orient="records")


@router.get("/{flight_id}/events")
def get_events(
    flight_id: str, store: FlightStore = Depends(get_store)
) -> Any:
    try:
        events = store.load_events(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc
    return [e.to_dict() for e in events]


@router.get("/{flight_id}/report")
def get_report(
    flight_id: str, store: FlightStore = Depends(get_store)
) -> Any:
    try:
        report = store.load_report(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc
    if report is None:
        raise HTTPException(status_code=404, detail="No report for this flight yet.")
    return report.to_json_dict()


@router.get("/{flight_id}/exports/{kind}")
def export_flight(
    flight_id: str,
    kind: str,
    store: FlightStore = Depends(get_store),
) -> Response:
    try:
        store.require(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc

    if kind == "json":
        report = store.load_report(flight_id)
        if report is None:
            raise HTTPException(status_code=404, detail="No report available.")
        return Response(
            content=report_to_json_bytes(report),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{flight_id}_report.json"'
            },
        )
    if kind == "windows.csv":
        windows = store.load_windows(flight_id)
        return Response(
            content=windows_to_csv_bytes(windows),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{flight_id}_windows.csv"'
            },
        )
    if kind == "events.csv":
        events_df = store.load_events_df(flight_id)
        return Response(
            content=events_to_csv_bytes(events_df),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{flight_id}_events.csv"'
            },
        )
    raise HTTPException(
        status_code=400,
        detail="kind must be one of: json, windows.csv, events.csv",
    )


@router.get("/{flight_id}/charts/{chart_name}")
def get_chart(
    flight_id: str,
    chart_name: str,
    event_id: Optional[str] = Query(None),
    store: FlightStore = Depends(get_store),
) -> Any:
    try:
        windows = store.load_windows(flight_id)
        events = store.load_events(flight_id)
        aligned = store.load_aligned(flight_id)
        summary = store.load_analysis_summary(flight_id) or {}
        report = store.load_report(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc

    threshold = float(
        (report.detector.threshold if report else None)
        or summary.get("threshold")
        or 0.0
    )
    features = list(
        (report.detector.features if report else None)
        or summary.get("features")
        or []
    )

    selected = None
    if event_id:
        selected = next((e for e in events if e.event_id == event_id), None)
        if selected is None:
            raise HTTPException(status_code=404, detail=f"Unknown event_id: {event_id}")

    if chart_name == "score_timeline":
        if windows is None:
            raise HTTPException(status_code=404, detail="No windows scored yet.")
        fig = score_timeline_figure(
            windows, threshold, events=events, selected_event_id=event_id
        )
    elif chart_name == "event_telemetry":
        if selected is None:
            raise HTTPException(status_code=400, detail="event_id is required.")
        if aligned is None:
            raise HTTPException(status_code=404, detail="No aligned telemetry.")
        fig = telemetry_around_event_figure(aligned, selected, features)
    elif chart_name == "per_feature":
        if selected is None:
            raise HTTPException(status_code=400, detail="event_id is required.")
        fig = per_feature_error_figure(selected)
    elif chart_name == "baro":
        if selected is None:
            raise HTTPException(status_code=400, detail="event_id is required.")
        if aligned is None:
            raise HTTPException(status_code=404, detail="No aligned telemetry.")
        fig = baro_context_figure(aligned, selected)
    else:
        raise HTTPException(
            status_code=400,
            detail="chart_name must be score_timeline|event_telemetry|per_feature|baro",
        )
    payload = fig.to_plotly_json()
    # Drop Plotly cloud config so the browser does not probe this API host
    if isinstance(payload, dict):
        payload.pop("config", None)
    return JSONResponse(content=payload)


@router.get("/{flight_id}/evidence-trace")
def evidence_trace(
    flight_id: str, store: FlightStore = Depends(get_store)
) -> Any:
    try:
        report = store.load_report(flight_id)
        windows = store.load_windows(flight_id)
        aligned = store.load_aligned(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc
    if report is None:
        raise HTTPException(status_code=404, detail="No report for this flight yet.")
    pack = build_evidence_pack(report, windows=windows, aligned=aligned)
    trace = build_diagnostic_trace(report, windows=windows, evidence_pack=pack)
    return {
        "trace": trace,
        "formatted": format_diagnostic_trace(trace),
        "evidence_approx_chars": pack.get("_approx_chars"),
    }


@router.post("/{flight_id}/ai/diagnostic")
def ai_diagnostic(
    flight_id: str,
    body: DiagnosticRequest = DiagnosticRequest(),
    store: FlightStore = Depends(get_store),
) -> Any:
    if not api_configured():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured. Detector results remain available.",
        )
    try:
        report = store.load_report(flight_id)
        windows = store.load_windows(flight_id)
        aligned = store.load_aligned(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc
    if report is None:
        raise HTTPException(status_code=400, detail="Analyze the flight before AI diagnostics.")

    try:
        service = LLMService()
        diagnostic = service.generate_diagnostic(
            report,
            windows=windows,
            aligned=aligned,
            selected_event_ids=body.selected_event_ids,
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    report_with_ai = attach_ai_to_report(report, diagnostic)
    store.save_report(flight_id, report_with_ai)
    diag_dict = diagnostic.to_json_dict()
    store.save_ai_diagnostic(flight_id, diag_dict)

    pack = service.last_evidence_pack
    trace = None
    if pack is not None:
        trace = build_diagnostic_trace(report, windows=windows, evidence_pack=pack)

    usage = None
    if service.last_usage:
        usage = {
            "latency_ms": service.last_usage.latency_ms,
            "model": service.last_usage.model,
            "prompt_tokens": service.last_usage.prompt_tokens,
            "completion_tokens": service.last_usage.completion_tokens,
        }
    return {"diagnostic": diag_dict, "trace": trace, "usage": usage}


@router.get("/{flight_id}/ai/diagnostic")
def get_ai_diagnostic(
    flight_id: str, store: FlightStore = Depends(get_store)
) -> Any:
    try:
        diag = store.load_ai_diagnostic(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc
    if diag is None:
        raise HTTPException(status_code=404, detail="No AI diagnostic yet.")
    return diag


@router.get("/{flight_id}/ai/chat", response_model=ChatStateResponse)
def get_chat(
    flight_id: str, store: FlightStore = Depends(get_store)
) -> ChatStateResponse:
    try:
        store.require(flight_id)
        chat = store.load_chat(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc
    return ChatStateResponse(
        flight_id=chat.get("flight_id", flight_id),
        messages=chat.get("messages") or [],
        selected_event_id=chat.get("selected_event_id"),
    )


@router.post("/{flight_id}/ai/chat")
def post_chat(
    flight_id: str,
    body: ChatRequest,
    store: FlightStore = Depends(get_store),
) -> Any:
    if not api_configured():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured.",
        )
    try:
        report = store.load_report(flight_id)
        windows = store.load_windows(flight_id)
        aligned = store.load_aligned(flight_id)
        chat = store.load_chat(flight_id)
        diag_raw = store.load_ai_diagnostic(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc

    if report is None:
        raise HTTPException(status_code=400, detail="Analyze the flight first.")
    if diag_raw is None:
        raise HTTPException(
            status_code=400,
            detail="Generate an AI diagnostic before chat.",
        )

    diagnostic = validate_diagnostic(diag_raw)
    if body.selected_event_id is not None:
        chat["selected_event_id"] = body.selected_event_id
    if chat.get("flight_id") != flight_id:
        chat = empty_chat_state(flight_id)

    try:
        service = LLMService()
        answer = ask_followup(
            service,
            report=report,
            chat_state=chat,
            user_message=body.message,
            diagnostic=diagnostic,
            windows=windows,
            aligned=aligned,
        )
    except (LLMError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    store.save_chat(flight_id, chat)
    return {
        "answer": answer.model_dump(mode="json"),
        "chat": chat,
    }


@router.delete("/{flight_id}/ai/chat", response_model=ChatStateResponse)
def clear_chat_route(
    flight_id: str, store: FlightStore = Depends(get_store)
) -> ChatStateResponse:
    try:
        chat = store.clear_chat(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc
    return ChatStateResponse(
        flight_id=chat["flight_id"],
        messages=[],
        selected_event_id=None,
    )
