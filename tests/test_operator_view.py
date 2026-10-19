"""Operator verdicts, rule-based signal checks and the read-only operator API."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src import operator_view as ov
from src.operator_view import compute_verdict, signal_checks
from tests.conftest import _make_att, _make_baro, _make_imu


# ---- verdict: the five states ------------------------------------------------------

def v(**kw):
    base = dict(ok=True, status="ok", n_events=0, coverage_status="ok")
    base.update(kw)
    return compute_verdict(**base)


def test_five_states():
    assert v(n_events=2)["state"] == ov.ANOMALIES
    assert v()["state"] == ov.NO_ALERTS_OK
    assert v(coverage_status="low_coverage")["state"] == ov.NO_ALERTS_LOW
    assert v(ok=False, status="error", coverage_status="error")["state"] == ov.FAILED
    assert v(ok=False, status="insufficient_data", coverage_status="insufficient_data")["state"] == ov.INSUFFICIENT
    assert compute_verdict(ok=None, status=None, n_events=0, coverage_status=None)["state"] == ov.NOT_ANALYZED


def test_unknown_coverage_never_reads_as_adequate():
    r = v(coverage_status=None)
    assert r["state"] == ov.NO_ALERTS_LOW
    assert any("not recorded" in c for c in r["caveats"])


def test_no_alert_states_never_say_normal_or_safe():
    for r in (v(), v(coverage_status="low_coverage"), v(coverage_status=None)):
        text = " ".join([r["headline"], r["detail"]]).lower()
        assert "normal" not in text and "safe" not in text.replace("does not show that the aircraft was safe", "")
        assert any("does not show that the aircraft was safe" in c for c in r["caveats"])


def test_low_coverage_flags_still_note_unassessed_part():
    r = v(n_events=1, coverage_status="low_coverage")
    assert r["state"] == ov.ANOMALIES and r["caveats"]


# ---- rule-based signal checks on synthetic series ----------------------------------

def _aligned(y, t0=0.0):
    n = len(y)
    return pd.DataFrame({"_t_s": t0 + np.arange(n) * 0.1, "_session_id": 0, "gyro_x": y})


def _noise(n, seed=0, s=0.05):
    return np.random.default_rng(seed).normal(0, s, n)


def test_step_change_detected():
    y = _noise(600)
    y[400:] += 3.0  # step at t=40 s
    out = signal_checks(_aligned(y), "gyro_x", 39.5, 45.0, 0)
    kinds = {c["kind"] for c in out}
    assert "sudden_change" in kinds
    assert all("fault" not in c["text"].lower() for c in out)


def test_sustained_offset_detected():
    y = _noise(600)
    y[400:460] += 2.0
    kinds = {c["kind"] for c in signal_checks(_aligned(y), "gyro_x", 40.2, 45.5, 0)}
    assert "sustained_deviation" in kinds


def test_flatline_detected():
    y = _noise(600)
    y[400:440] = 0.123
    kinds = {c["kind"] for c in signal_checks(_aligned(y), "gyro_x", 40.0, 44.0, 0)}
    assert "flatline" in kinds


def test_quiet_signal_has_no_findings_and_short_context_is_skipped():
    y = _noise(600)
    assert signal_checks(_aligned(y), "gyro_x", 40.0, 44.0, 0) == []
    # event at the very start: no baseline -> no comparison-based findings, no crash
    assert signal_checks(_aligned(y), "gyro_x", 0.5, 3.0, 0) == []


def test_unimplemented_patterns_are_declared_not_reported():
    assert set(ov.NOT_IMPLEMENTED_CHECKS) == {"oscillation", "command_tracking"}


# ---- API ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch, fixture_model_dir):
    monkeypatch.setenv("UAV_MODEL_DIR", str(fixture_model_dir))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "flights"))
    from backend import deps
    from src import inference_service

    deps.get_store.cache_clear()
    inference_service.clear_caches()
    yield TestClient(app)
    inference_service.clear_caches()
    deps.get_store.cache_clear()


def _csv(df):
    b = io.BytesIO()
    df.to_csv(b, index=False)
    return b.getvalue()


def _upload(client, fid="op-flight"):
    r = client.post(
        "/api/flights/upload",
        files={
            "imu": ("IMU.csv", _csv(_make_imu(500)), "text/csv"),
            "att": ("ATT.csv", _csv(_make_att(200)), "text/csv"),
            "baro": ("BARO.csv", _csv(_make_baro(200)), "text/csv"),
        },
        data={"flight_id": fid},
    )
    assert r.json()["ok"], r.text
    return fid


def test_overview_before_analysis_is_not_analyzed(client):
    fid = _upload(client)
    o = client.get(f"/api/flights/{fid}/overview").json()
    assert o["verdict"]["state"] == ov.NOT_ANALYZED
    assert o["events"] == []


def test_overview_timeline_detail_report_after_analysis(client):
    fid = _upload(client)
    a = client.post(f"/api/flights/{fid}/analyze", json={}).json()
    assert a["verdict"]["state"] in set(ov.VERDICT_STATES)
    o = client.get(f"/api/flights/{fid}/overview").json()
    assert o["verdict"]["state"] == a["verdict"]["state"]
    assert len(o["events"]) == a["n_events"]
    starts = [e["start_rel_s"] for e in o["events"]]
    assert starts == sorted(starts)  # chronological

    t = client.get(f"/api/flights/{fid}/timeline").json()
    assert sum(1 for i in t["items"] if i["type"] == "event") == a["n_events"]

    rep = client.get(f"/api/flights/{fid}/operator-report").json()
    assert rep["disclaimer"] and "technical_appendix" in rep
    # numeric detector internals live only in the appendix, not in the plain events
    for e in rep["events"]:
        assert "peak_score" not in e and "threshold" not in e
    assert rep["technical_appendix"]["detector"]["threshold"] == pytest.approx(4.034964084625244)

    for e in o["events"]:
        d = client.get(f"/api/flights/{fid}/events/{e['event_id']}/detail").json()
        assert d["start_s"] <= d["end_s"]
        assert d["uncertain"] and d["suggested_action"] and d["what_observed"]
    assert client.get(f"/api/flights/{fid}/events/nope/detail").status_code == 404


def test_operator_views_do_not_modify_stored_results(client, tmp_path):
    fid = _upload(client)
    client.post(f"/api/flights/{fid}/analyze", json={})
    before = client.get(f"/api/flights/{fid}/report").json()
    for path in ("overview", "timeline", "operator-report"):
        client.get(f"/api/flights/{fid}/{path}")
    assert client.get(f"/api/flights/{fid}/report").json() == before


# ---- optional AI explanation (Task B) --------------------------------------------------

import json as _json
from types import SimpleNamespace as _NS

from src import explain_llm


def _op_report_with_event():
    return {
        "overview": {"verdict": {"headline": "Unusual telemetry detected"}, "coverage": {"status": "ok"}, "unassessed": []},
        "events": [{
            "event_id": "event-001", "when": "01:00 to 01:05", "duration_s": 5.0, "detector_note": "flagged",
            "channels": [{"label": "gyro Z", "unit": "rad/s", "during": {"max": 2.0}, "before": {"median": 0.0}}],
            "signal_checks": [], "uncertain": ["x"],
        }],
    }


class _Client:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.calls = content, exc, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return _NS(choices=[_NS(message=_NS(content=self.content))], usage=None)


def _good(**over):
    e = {"event_id": "event-001", "observed_facts": ["gyro Z reached 2.0 rad/s"], "interpretation": ["may reflect a rapid movement"],
         "uncertainty": ["cause unknown"], "suggested_review": ["compare with pilot notes"]}
    e.update(over)
    return _json.dumps({"events": [e]})


def test_explanation_ok_is_labelled_and_bounded():
    c = _Client(_good())
    r = explain_llm.generate_explanation(_op_report_with_event(), client=c)
    assert r["status"] == "ok" and "AI-generated" in r["label"] and r["events"][0]["event_id"] == "event-001"
    sent = c.calls[0]
    assert sent["response_format"]["json_schema"]["strict"] and sent["max_tokens"] <= 1800
    assert "peak_score" not in sent["messages"][1]["content"] and "threshold" not in sent["messages"][1]["content"]


@pytest.mark.parametrize("claim", ["indicates motor failure", "loss of control occurred", "the aircraft is safe to fly"])
def test_unsupported_diagnostic_claims_are_dropped(claim):
    r = explain_llm.generate_explanation(_op_report_with_event(), client=_Client(_good(interpretation=[claim])))
    assert r["status"] == "ok" and r["events"] == []


def test_unknown_event_ids_ignored_and_failures_degrade(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    r = explain_llm.generate_explanation(_op_report_with_event(), client=_Client(_good(event_id="event-999")))
    assert r["events"] == []
    assert explain_llm.generate_explanation(_op_report_with_event(), client=_Client("nope"))["status"] == "invalid_output"
    assert explain_llm.generate_explanation(_op_report_with_event(), client=_Client(exc=TimeoutError()))["status"] == "error"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert explain_llm.generate_explanation(_op_report_with_event())["status"] == "unavailable"


def test_explanation_api_needs_consent_and_never_changes_results(client, monkeypatch):
    fid = _upload(client, "ex-1")
    client.post(f"/api/flights/{fid}/analyze", json={})
    before = client.get(f"/api/flights/{fid}/report").json()
    assert client.post(f"/api/flights/{fid}/explanation", json={}).status_code == 400
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post(f"/api/flights/{fid}/explanation", json={"consent": True}).json()
    assert r["status"] in ("unavailable", "no_events")
    assert client.get(f"/api/flights/{fid}/report").json() == before
    assert client.get(f"/api/flights/{fid}/explanation").json()["status"] == "not_generated"


# ---- event context + raw telemetry ------------------------------------------------------

def test_event_context_has_before_during_after_and_stays_in_segment():
    from src.report_schema import AnomalyEventReport, DetectorInfo, FlightReport, FlightSummary

    y = _noise(1200)
    al = pd.DataFrame({"_t_s": np.arange(1200) * 0.1, "_session_id": 0, "gyro_x": y})
    al.loc[600:640, "_session_id"] = 1  # a different segment must not leak in
    ev = AnomalyEventReport(
        event_id="event-001", session_id=0, start_time_s=40.0, end_time_s=46.0, duration_s=6.0, flagged_window_count=1,
        threshold=4.0, window_indices=[1], affected_signals=["gyro_x"],
    )
    rep = FlightReport(
        flight_id="f", analysis_status="completed",
        detector=DetectorInfo(variant="v", feature_count=12, window_length=30, stride=15, threshold=4.0),
        summary=FlightSummary(total_windows=1, flagged_windows=1, flagged_rate=1.0, event_count=1), events=[ev],
    )
    c = ov.event_context(rep, "event-001", al)
    assert c["window_start_rel_s"] < c["start_rel_s"] < c["end_rel_s"] < c["window_end_rel_s"]
    assert min(c["time"]) < c["start_rel_s"] and max(c["time"]) > c["end_rel_s"]
    assert not any(60.0 <= x <= 64.0 for x in c["time"])  # other session excluded
    assert c["channels"][0]["signal"] == "gyro_x" and len(c["channels"][0]["values"]) == len(c["time"])
    with pytest.raises(KeyError):
        ov.event_context(rep, "nope", al)


def test_raw_telemetry_endpoint_is_capped(client):
    fid = _upload(client, "raw-1")
    client.post(f"/api/flights/{fid}/analyze", json={})
    r = client.get(f"/api/flights/{fid}/telemetry", params={"limit": 7}).json()
    assert r["returned"] == 7 and r["total_rows"] >= 7 and "_t_s" in r["columns"]
    assert client.get(f"/api/flights/{fid}/telemetry", params={"limit": 99999}).status_code == 422
