"""Flight upload, analysis, charts, exports, and AI routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

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
from backend.store import FlightNotFound, FlightStore, FlightStoreError
from src.chat_service import ask_followup, empty_chat_state
from src.diagnostic_schema import DiagnosticReport, validate_diagnostic
from src.diagnostic_trace import build_diagnostic_trace, format_diagnostic_trace
from src.evidence_builder import build_evidence_pack
from src.event_builder import DEFAULT_MAX_GAP_S
from src.flight_identity import BLOCKING, PARTIAL, SIMILAR, compute_identity, duplicate_message
from src.inference_service import analyze_flight
from src.combined_ingest import load_combined, mapping_for_storage
from src.combined_preview import (
    analyze_combined,
    build_combined_preview,
    parse_assignments as _parse_assignments_raw,
    preview_from_analysis,
)
from src.explain_llm import DISCLOSURE as EXPLAIN_DISCLOSURE, generate_explanation
from src.limits import get_limits
from src.operator_view import (
    build_operator_report,
    build_overview,
    build_timeline,
    describe_event,
    event_context,
    verdict_from_summary,
)
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


def _parse_assignments(raw: Optional[str]) -> Dict[str, Any]:
    try:
        return _parse_assignments_raw(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def read_upload_capped(upload: UploadFile, limit: int) -> bytes:
    """Read an upload in chunks, stopping as soon as it exceeds ``limit`` bytes."""
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"'{upload.filename or 'file'}' exceeds the {limit / 1e6:.0f} MB "
                    "upload limit."
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _store_exc(exc: Exception) -> HTTPException:
    if isinstance(exc, FlightNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


@router.get("", response_model=List[FlightListItem])
def list_flights(store: FlightStore = Depends(get_store)) -> List[FlightListItem]:
    out = []
    for item in store.list_flights():
        try:
            v = verdict_from_summary(store.load_analysis_summary(item["flight_id"]))
            item = {**item, "verdict_state": v["state"], "verdict_headline": v["headline"]}
        except Exception:  # a damaged flight must not break the list
            pass
        out.append(FlightListItem(**item))
    return out


@router.post("/upload", response_model=UploadResponse)
async def upload_flight(
    imu: Optional[UploadFile] = File(None),
    att: Optional[UploadFile] = File(None),
    baro: Optional[UploadFile] = File(None),
    zip_file: Optional[UploadFile] = File(None),
    flight_id: Optional[str] = Form(None),
    allow_duplicate: bool = Form(False),
    store: FlightStore = Depends(get_store),
) -> UploadResponse:
    if zip_file is not None:
        raw = await read_upload_capped(zip_file, get_limits().max_zip_total_bytes)
        result = load_zip_archive(raw, archive_name=zip_file.filename or "upload.zip")
    elif imu is not None and att is not None and baro is not None:
        files = {
            "IMU": await read_upload_capped(imu, get_limits().max_file_bytes),
            "ATT": await read_upload_capped(att, get_limits().max_file_bytes),
            "BARO": await read_upload_capped(baro, get_limits().max_file_bytes),
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
    return _finish_upload(result, flight_id, allow_duplicate, store)


def _finish_upload(
    result,
    flight_id: Optional[str],
    allow_duplicate: bool,
    store: FlightStore,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> UploadResponse:
    """Duplicate check + persistence shared by the three-file, ZIP and combined uploads."""
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
    identity = compute_identity(result.frames, result.file_sha256)
    matches = store.find_duplicates(identity)
    dup_msg = duplicate_message(matches)
    blocking = [m for m in matches if m["kind"] in BLOCKING]
    if dup_msg and not allow_duplicate:
        return UploadResponse(
            ok=False,
            source=result.source,
            filenames=dict(result.filenames),
            file_sizes=dict(result.file_sizes),
            warnings=list(result.all_warnings),
            errors=[dup_msg],
            validation=validation,
            duplicate_of=blocking[0]["flight_id"],
            duplicate_kind=blocking[0]["kind"],
            matches=matches,
        )
    warnings = list(result.all_warnings)
    for m in matches:
        if m["kind"] == PARTIAL:
            warnings.append(
                f"Some streams carry data identical to existing flight '{m['flight_id']}' "
                "(partial overlap). It was stored as a separate flight."
            )
        elif m["kind"] == SIMILAR:
            warnings.append(
                f"Start time, duration and row counts match existing flight "
                f"'{m['flight_id']}' but the data differ. Stored as a distinct flight."
            )
    try:
        fid = store.create_from_upload(
            result.frames,
            source=result.source,
            filenames=dict(result.filenames),
            file_sizes=dict(result.file_sizes),
            warnings=warnings,
            validation=validation,
            flight_id=flight_id,
            file_sha256=dict(result.file_sha256),
            duplicate_of=blocking[0]["flight_id"] if blocking else None,
            extra_meta=extra_meta,
        )
    except FlightStoreError as exc:
        return UploadResponse(
            ok=False,
            source=result.source,
            filenames=dict(result.filenames),
            file_sizes=dict(result.file_sizes),
            warnings=warnings,
            errors=[str(exc)],
            validation=validation,
        )
    return UploadResponse(
        ok=True,
        flight_id=fid,
        status="validated",
        source=result.source,
        filenames=dict(result.filenames),
        file_sizes=dict(result.file_sizes),
        warnings=warnings,
        errors=[],
        validation=validation,
        duplicate_of=blocking[0]["flight_id"] if blocking else None,
        duplicate_kind=blocking[0]["kind"] if blocking else None,
        matches=matches,
    )




@router.post("/combined/preview")
async def combined_preview(
    file: UploadFile = File(...),
    assignments: Optional[str] = Form(None),
    llm_consent: bool = Form(False),
) -> Any:
    """Inspect a combined CSV and propose a validated column mapping. Stores nothing."""
    raw = await read_upload_capped(file, get_limits().max_file_bytes)
    return build_combined_preview(raw, file.filename or "combined.csv", _parse_assignments(assignments), llm_consent)


@router.post("/combined/upload", response_model=UploadResponse)
async def combined_upload(
    file: UploadFile = File(...),
    assignments: Optional[str] = Form(None),
    flight_id: Optional[str] = Form(None),
    allow_duplicate: bool = Form(False),
    store: FlightStore = Depends(get_store),
) -> UploadResponse:
    """Store a combined CSV once its mapping is fully valid.

    The mapping is rebuilt server-side from deterministic rules plus the user's explicit
    ``assignments``; AI suggestions are never trusted from the client.
    """
    raw = await read_upload_capped(file, get_limits().max_file_bytes)
    result, insp, mapping = load_combined(
        raw, file.filename or "combined.csv", assignments=_parse_assignments(assignments)
    )
    if not result.ok or insp is None or mapping is None:
        return UploadResponse(
            ok=False, source="combined", filenames=dict(result.filenames), file_sizes=dict(result.file_sizes),
            warnings=list(result.all_warnings), errors=list(result.all_errors),
            validation=result.validation.to_dict() if result.validation else None,
        )
    return _finish_upload(
        result, flight_id, allow_duplicate, store, extra_meta={"combined": mapping_for_storage(mapping, insp)}
    )


@router.post("/combined/auto")
async def combined_auto(
    file: UploadFile = File(...),
    assignments: Optional[str] = Form(None),
    llm_consent: bool = Form(False),
    flight_id: Optional[str] = Form(None),
    allow_duplicate: bool = Form(False),
    store: FlightStore = Depends(get_store),
) -> Any:
    """Upload → schema analysis → validation → (if fully compatible) store and analyse.

    ``stage`` is one of ``analyzed`` | ``analysis_failed`` | ``needs_input`` | ``incompatible`` |
    ``invalid`` | ``not_stored``. Anything other than ``analyzed``/``analysis_failed`` stores nothing
    and runs no inference: a file the deployed detector cannot use is never reported as analysed.
    """
    raw = await read_upload_capped(file, get_limits().max_file_bytes)
    name = file.filename or "combined.csv"
    a = analyze_combined(raw, name, _parse_assignments(assignments), llm_consent)
    out: Dict[str, Any] = preview_from_analysis(a, name)
    out["flight_id"] = None
    if not a.assessment["compatible"] or a.result is None or a.insp is None or a.mapping is None:
        out["stage"] = a.assessment["outcome"]
        out["stored"] = False
        return out

    up = _finish_upload(
        a.result, flight_id, allow_duplicate, store,
        extra_meta={"combined": mapping_for_storage(a.mapping, a.insp, a.llm.to_dict())},
    )
    if not up.ok or not up.flight_id:
        out.update(stage="not_stored", stored=False, upload_errors=up.errors, duplicate_of=up.duplicate_of,
                   duplicate_kind=up.duplicate_kind)
        return out
    out.update(stored=True, flight_id=up.flight_id, upload_warnings=up.warnings)
    analysis = _run_analysis(up.flight_id, store, DEFAULT_MAX_GAP_S)
    out["analysis"] = {"ok": analysis.ok, "status": analysis.status, "n_events": analysis.n_events,
                       "n_windows": analysis.n_windows, "errors": analysis.errors, "verdict": analysis.verdict}
    out["stage"] = "analyzed" if analysis.ok else "analysis_failed"
    return out


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
        verdict=verdict_from_summary(store.load_analysis_summary(flight_id)),
        has_ai_diagnostic=store.load_ai_diagnostic(flight_id) is not None,
        event_count=len(events),
        duplicate_of=meta.get("duplicate_of"),
        file_sha256=meta.get("file_sha256") or {},
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


def _run_analysis(flight_id: str, store: FlightStore, max_gap_s: float) -> AnalyzeResponse:
    try:
        frames = store.load_frames(flight_id)
        meta = store.load_meta(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc

    result = analyze_flight(
        frames,
        flight_id=flight_id,
        upload_warnings=list(meta.get("warnings") or []),
        max_gap_s=float(max_gap_s or DEFAULT_MAX_GAP_S),
        flight_meta=meta,
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
        coverage_status=(result.data_quality or {}).get("coverage_status"),
        data_quality=result.data_quality,
        verdict=verdict_from_summary(result.summary_dict()),
    )


@router.post("/{flight_id}/analyze", response_model=AnalyzeResponse)
def analyze(
    flight_id: str,
    body: AnalyzeRequest = AnalyzeRequest(),
    store: FlightStore = Depends(get_store),
) -> AnalyzeResponse:
    return _run_analysis(flight_id, store, body.max_gap_s)


def _operator_inputs(flight_id: str, store: FlightStore):
    try:
        meta = store.load_meta(flight_id)
        return (
            meta,
            store.load_report(flight_id),
            store.load_analysis_summary(flight_id),
            store.load_aligned(flight_id),
        )
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc


@router.get("/{flight_id}/overview")
def flight_overview(flight_id: str, store: FlightStore = Depends(get_store)) -> Any:
    """Operator summary: verdict, coverage, chronological events (read-only projection)."""
    meta, report, summary, aligned = _operator_inputs(flight_id, store)
    return build_overview(meta=meta, report=report, summary=summary, aligned=aligned)


@router.get("/{flight_id}/timeline")
def flight_timeline(flight_id: str, store: FlightStore = Depends(get_store)) -> Any:
    _meta, report, _summary, aligned = _operator_inputs(flight_id, store)
    if report is None:
        raise HTTPException(status_code=404, detail="No analysis for this flight yet.")
    return build_timeline(report, aligned)


@router.get("/{flight_id}/events/{event_id}/detail")
def event_detail(
    flight_id: str, event_id: str, store: FlightStore = Depends(get_store)
) -> Any:
    _meta, report, _summary, aligned = _operator_inputs(flight_id, store)
    if report is None:
        raise HTTPException(status_code=404, detail="No analysis for this flight yet.")
    try:
        return describe_event(report, event_id, aligned)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown event_id: {event_id}")


@router.get("/{flight_id}/events/{event_id}/context")
def event_context_route(
    flight_id: str, event_id: str, store: FlightStore = Depends(get_store)
) -> Any:
    """Channel values around one event (before / during / after) for the timeline chart."""
    _meta, report, _summary, aligned = _operator_inputs(flight_id, store)
    if report is None:
        raise HTTPException(status_code=404, detail="No analysis for this flight yet.")
    try:
        return event_context(report, event_id, aligned)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown event_id: {event_id}")


@router.get("/{flight_id}/telemetry")
def raw_telemetry(
    flight_id: str,
    start_s: Optional[float] = Query(None),
    end_s: Optional[float] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
    store: FlightStore = Depends(get_store),
) -> Any:
    """Aligned 10 Hz telemetry rows (advanced view). Capped; log-clock seconds."""
    try:
        aligned = store.load_aligned(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc
    if aligned is None:
        raise HTTPException(status_code=404, detail="No aligned telemetry yet.")
    df = aligned
    if start_s is not None:
        df = df[df["_t_s"] >= start_s]
    if end_s is not None:
        df = df[df["_t_s"] <= end_s]
    total = int(len(df))
    df = df.head(limit)
    rows = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    return {"columns": list(df.columns), "rows": rows, "total_rows": total, "returned": len(rows)}


@router.get("/{flight_id}/operator-report")
def operator_report(flight_id: str, store: FlightStore = Depends(get_store)) -> Any:
    """Plain-language report built only from stored, verified results (no LLM)."""
    meta, report, summary, aligned = _operator_inputs(flight_id, store)
    if report is None:
        raise HTTPException(status_code=404, detail="No analysis for this flight yet.")
    return build_operator_report(meta=meta, report=report, summary=summary, aligned=aligned)


class ExplainRequest(BaseModel):
    consent: bool = False


@router.get("/{flight_id}/explanation")
def get_explanation(flight_id: str, store: FlightStore = Depends(get_store)) -> Any:
    try:
        saved = store.load_explanation(flight_id)
    except FlightNotFound as exc:
        raise _store_exc(exc) from exc
    return saved or {"status": "not_generated", "events": [], "disclosure": EXPLAIN_DISCLOSURE,
                     "configured": api_configured()}


@router.post("/{flight_id}/explanation")
def create_explanation(
    flight_id: str, body: ExplainRequest, store: FlightStore = Depends(get_store)
) -> Any:
    """Optional AI wording for the verified events. Requires explicit consent; never alters results."""
    if not body.consent:
        raise HTTPException(status_code=400, detail="Consent is required: " + EXPLAIN_DISCLOSURE)
    meta, report, summary, aligned = _operator_inputs(flight_id, store)
    if report is None:
        raise HTTPException(status_code=400, detail="Analyze the flight first.")
    op = build_operator_report(meta=meta, report=report, summary=summary, aligned=aligned)
    result = generate_explanation(op)
    if result["status"] == "ok":
        store.save_explanation(flight_id, result)
    return result


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
