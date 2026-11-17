"""Agentic Analysis: tools, validation, the investigation loop (scripted fake model), physics fit and the API."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.agent import runner, symptoms
from src.agent.sources import ULogSource
from src.agent.tools import Toolbox
from src.agent.validate import Grounding, validate
from src.arjuna import physics as P
from src.arjuna.analysis import analyse
from src.arjuna.ulog import ArjunaFlight, Topic, load_flight
from tests.fixtures.arjuna_synth import flight


@pytest.fixture(scope="module")
def vib():
    fl = load_flight(flight(vibe_spike=(50, 60, 75.0), motor_bias={2: 0.18}), "f.ulg")
    return fl, analyse(fl)


# ---- tools ------------------------------------------------------------------------------------------------------------

def test_tools_compute_numbers_and_withhold_coordinates(vib):
    fl, a = vib
    tb = Toolbox(ULogSource(fl), a)
    _, s = tb.call("signal_stats", {"topic": "arjuna_vibration", "field": "accel_vibe[2]", "t_start": 50, "t_end": 60})
    assert s["max"] == pytest.approx(75.0, abs=1.5) and s["unit"] == "m/s^2" and s["ref"] == "T1"
    _, c = tb.call("compare_periods", {"topic": "arjuna_vibration", "field": "accel_vibe[2]", "a_start": 30, "a_end": 40,
                                       "b_start": 52, "b_end": 58})
    assert c["median_ratio_b_over_a"] > 5 and "median_percent_change" in c
    _, att = tb.call("signal_stats", {"topic": "arjuna_attitude", "field": "roll", "t_start": 20, "t_end": 30})
    assert att["unit"] == "deg" and 10 < att["max"] < 14  # radians converted by code, never by the model
    tb2 = Toolbox(ULogSource(fl), a)
    fl.topics[("arjuna_gps", 0)].df["latitude"] = 52.0
    _, g = tb2.call("signal_stats", {"topic": "arjuna_gps", "field": "latitude"})
    assert "withheld" in g["error"]
    _, bad = tb2.call("signal_stats", {"topic": "arjuna_nope", "field": "x"})
    assert "error" in bad and "arjuna_vibration" in bad["error"]


def test_static_results_are_fetched_once(vib):
    fl, a = vib
    tb = Toolbox(ULogSource(fl), a)
    ref, _ = tb.call("get_findings", {})
    again_ref, again = tb.call("get_findings", {"code": "VIB_HIGH"})
    assert again_ref == ref and again["duplicate"] and len(tb.calls) == 1


def test_sequence_of_departures_orders_what_changed_first(vib):
    fl, a = vib
    _, out = Toolbox(ULogSource(fl), a).call("sequence_of_departures", {"t_start": 45, "t_end": 65, "baseline_s": 30})
    first = out["departures_in_time_order"][0]
    assert first["signal"].startswith("vibration") and first["onset_t"] == pytest.approx(50.0, abs=0.2)


def test_pilot_report_ranks_by_symptom_time_and_confidence(vib):
    _, a = vib
    r = symptoms.rank({"symptom": "motor_sound", "time_ref": "since_takeoff", "time_value": 45}, a["findings"], 10.0, 110.0, 120.0)
    assert r["ranking"][0]["code"] in ("MOTOR_ESC", "VIB_HIGH")
    assert r["search_window"]["t_start"] == pytest.approx(25.0) and r["search_window"]["t_end"] == pytest.approx(85.0)


# ---- validation -------------------------------------------------------------------------------------------------------

def _report(**over):
    r = {"headline": "", "summary": "", "finding_reviews": [], "root_cause_chain": [], "additional_observations": [],
         "pilot_report_answer": "", "not_checked": [], "recommendations": []}
    r.update(over)
    return r


def test_grounding_accepts_rounding_percent_and_clock_but_not_invented_numbers():
    g = Grounding({"T1": {"max": 74.8312, "frac": 0.054, "t": 343.3, "at": "5:43.3"}})
    assert g.ungrounded("Vibration reached 74.8 m/s^2 (74.83) at 5:43.3, about 5:43, 5.4% of the time.") == []
    assert g.ungrounded("Vibration reached 91.2 m/s^2.") == ["91.2"]
    assert g.ungrounded("It happened at 7:10.") == ["7:10"]


def test_validation_removes_unsupported_text_and_keeps_every_finding(vib):
    _, a = vib
    f1, f2 = a["findings"][0]["id"], a["findings"][1]["id"]
    tools = {"T1": {"max": 75.0}}
    raw = _report(
        headline="Vibration reached 75 m/s^2. The aircraft is safe to fly.",
        summary="It was caused by a broken propeller.",
        finding_reviews=[{"finding_id": f1, "assessment": "likely_false_alarm", "explanation": "Peak was 75 m/s^2.",
                          "evidence_refs": [f1], "suggested_cause": "Definitely a chipped prop.", "check_next": "Inspect props."},
                         {"finding_id": "F99", "assessment": "confirmed", "explanation": "x", "evidence_refs": [],
                          "suggested_cause": "", "check_next": ""}],
        additional_observations=[{"title": "Odd", "detail": "Something at 123.4 units.", "evidence_refs": ["T9"],
                                  "t_start": None, "t_end": None, "suggested_code": "NOT_A_CODE"}])
    v = validate(raw, findings=a["findings"], tool_results=tools, analysis=a)
    assert v["headline"] == "Vibration reached 75 m/s^2."
    assert v["summary"] == ""  # certainty about the cause removed
    rv = {r["finding_id"]: r for r in v["finding_reviews"]}
    assert rv[f1]["assessment"] == "needs_review"  # a false alarm needs tool evidence
    assert rv[f1]["suggested_cause"] == ""
    assert rv[f2]["reviewed_by_agent"] is False and f2 in v["unreviewed_findings"]
    assert "F99" not in rv and v["additional_observations"] == []
    reasons = {r["reason"].split(":")[0] for r in v["rejected"]}
    assert {"safety_verdict", "cause_stated_as_certain", "false_alarm_without_tool_evidence"} <= reasons


# ---- the loop with a scripted model ----------------------------------------------------------------------------------

def _call(i, name, args):
    return SimpleNamespace(id=f"c{i}", type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class ScriptedModel:
    """Investigates each finding window, then submits. Records which tools were offered on each round."""

    def __init__(self, findings, submit_early=False):
        self.findings, self.round, self.offered, self.submit_early = findings, 0, [], submit_early

    def create(self, **kw):
        self.round += 1
        names = [t["function"]["name"] for t in kw["tools"]]
        self.offered.append(names)
        if "submit_report" in names and (self.round > 2 or self.submit_early):
            reviews = [{"finding_id": f["id"], "assessment": "confirmed", "explanation": f"Seen in {f['when']}.",
                        "evidence_refs": ["T5"], "suggested_cause": "Likely cause: a chipped propeller (suggestion).",
                        "check_next": "Inspect the propellers."} for f in self.findings]
            calls = [_call(99, "submit_report", _report(headline="Vibration and a hover imbalance.", summary="Two findings.",
                                                         finding_reviews=reviews))]
        else:
            calls = [_call(self.round * 10 + i, "signal_stats", {"topic": "arjuna_vibration", "field": "accel_vibe[2]",
                                                                  "t_start": max(0.0, (f["t_start"] or 0) - 5), "t_end": (f["t_end"] or 0) + 5})
                     for i, f in enumerate(self.findings)]
            calls += [_call(self.round * 10 + 8, "motor_outputs", {}), _call(self.round * 10 + 9, "get_messages", {})]
            calls += [_call(self.round * 10 + 7 - k, "signal_stats", {"topic": "arjuna_attitude", "field": f, "t_start": 0, "t_end": 120})
                      for k, f in enumerate(("roll", "pitch", "rates[0]", "rates[1]"))]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=calls))],
                               usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100))


def test_agent_must_investigate_before_it_can_submit(vib):
    fl, a = vib
    m = ScriptedModel(a["findings"], submit_early=True)
    r = runner.investigate(ULogSource(fl), a, a["findings"], client=m)
    assert "submit_report" not in m.offered[0]  # withheld until findings are examined and enough calls made
    assert r["status"] in ("ok", "partial") and r["report"]["headline"] == "Vibration and a hover imbalance."
    assert [t["tool"] for t in r["trace"][:3]] == ["get_flight_overview", "get_findings", "get_physics"]
    assert not r["report"]["unreviewed_findings"]


def test_agent_without_key_or_with_api_failure_degrades(vib, monkeypatch):
    fl, a = vib
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert runner.investigate(ULogSource(fl), a, a["findings"])["status"] == "unavailable"

    class Boom:
        def create(self, **kw):
            raise RuntimeError("invalid_api_key sk-should-not-leak")
    r = runner.investigate(ULogSource(fl), a, a["findings"], client=Boom())
    assert r["status"] == "error" and "sk-" not in r["message"] and r["report"] is None


def test_unverified_tracks_finding_windows(vib):
    _, a = vib
    vibf = next(f for f in a["findings"] if f["code"] == "VIB_HIGH")
    calls = [{"tool": "signal_stats", "ok": True, "args": {"topic": "arjuna_battery", "field": "voltage", "t_start": 0, "t_end": 5}}]
    assert vibf in runner.unverified([vibf], calls)
    calls.append({"tool": "signal_stats", "ok": True, "args": {"topic": "arjuna_imu", "field": "accel[2]", "t_start": 55, "t_end": 70}})
    assert runner.unverified([vibf], calls) == []


# ---- physics: per-motor effectiveness on a simulated quad -------------------------------------------------------------

def _simulate(eta, hz=200, T=40, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1 / hz
    t = np.arange(0, T, dt)
    n = len(t)
    x, y, s, _ = P._motors()
    B = np.vstack([P.AF["arm_projection_m"] * y, -P.AF["arm_projection_m"] * x, P.AF["yaw_torque_coeff_cq"] * s])
    c, I = 4.0, np.array([0.012, 0.013, 0.022])
    mix = np.linalg.pinv(np.vstack([B, np.ones(4)]))
    w_sp = np.vstack([np.convolve(rng.normal(0, 1, n), np.ones(40) / 40, "same") * 1.5 for _ in range(3)])
    w, Fl, integ = np.zeros(3), np.full(4, 0.35 ** 2 * c), np.zeros(3)
    a_ = dt / (P.AF["motor_time_constant_s"] + dt)
    U, W = np.zeros((4, n)), np.zeros((3, n))
    for k in range(n):
        err = w_sp[:, k] - w
        integ += err * dt
        u = np.sqrt(np.clip(mix @ np.concatenate([I * (8 * err + 4 * integ), [4 * 0.35 ** 2 * c]]) / c, 0.0025, 1.0))
        Fl = Fl + a_ * (u ** 2 * c - Fl)
        w = w + (B @ (eta * Fl) - np.cross(w, I * w)) / I * dt + rng.normal(0, 0.002, 3)
        U[:, k], W[:, k] = u, w
    frames = {"arjuna_vehicle_status": pd.DataFrame({"t": t, "landed_state": 2.0}),
              "arjuna_attitude": pd.DataFrame({"t": t, "roll": 0 * t, "pitch": 0 * t, **{f"rates[{i}]": W[i] for i in range(3)}}),
              "arjuna_actuator_outputs": pd.DataFrame({"t": t, "motor_count": 4.0, **{f"motor_cmd[{i}]": U[i] for i in range(4)}})}
    topics = {(k, 0): Topic(k, 0, df, hz, []) for k, df in frames.items()}
    return ArjunaFlight("sim", "", 0, {}, {}, topics, [], [], 0, T, {}, {"ok": True})


def test_eta_fit_finds_the_weak_motor_and_clears_a_healthy_one():
    weak = P.dynamics_fit(_simulate(np.array([1.0, 1.0, 0.8, 1.0])))
    assert weak["status"] == "computed" and weak["flag_motor"].startswith("M3")
    assert weak["relative_to_others"][2] == pytest.approx(0.8, abs=0.03)
    ok = P.dynamics_fit(_simulate(np.array([1.0, 1.0, 1.0, 1.0])))
    assert ok["flag_motor"] is None and max(abs(e - 1) for e in ok["eta"]) < 0.02


def test_eta_fit_refuses_low_rate_logs():
    low = P.dynamics_fit(load_flight(flight(), "x.ulg"))
    assert low["status"] == "not_computable" and "100 Hz" in low["reason"]


# ---- API ---------------------------------------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYSIS_DATA_DIR", str(tmp_path / "analyses"))
    from backend.main import app
    return TestClient(app)


def test_api_upload_ulog_then_agent_then_charts(client, monkeypatch):
    raw = flight(vibe_spike=(50, 60, 75.0))
    v = client.post("/api/analyses", files={"file": ("flight.ulg", raw, "application/octet-stream")},
                    data={"symptom": "wobble", "time_ref": "since_takeoff", "time_value": "45"}).json()
    aid = v["analysis_id"]
    assert v["meta"]["kind"] == "ulog" and v["meta"]["worst"] == "critical" and v["report"] is None
    assert v["meta"]["pilot_report"]["symptom"] == "wobble"
    assert client.post(f"/api/analyses/{aid}/agent", json={"consent": False}).status_code == 400

    findings = v["summary"]["findings"]
    real = runner.investigate
    monkeypatch.setattr(runner, "investigate", lambda *a, **k: real(*a, **{**k, "client": ScriptedModel(findings)}))
    out = client.post(f"/api/analyses/{aid}/agent", json={"consent": True}).json()
    assert out["report"]["status"] in ("ok", "partial") and out["meta"]["agent"]["status"] == out["report"]["status"]
    assert any(t["tool"] == "match_pilot_report" for t in out["report"]["trace"])

    ctx = client.get(f"/api/analyses/{aid}/findings/{findings[0]['id']}/context").json()
    assert ctx["charts"] and ctx["highlight"][0] <= 51 and ctx["window"][0] < ctx["highlight"][0]
    ser = client.get(f"/api/analyses/{aid}/series", params={"sig": ["arjuna_battery:voltage:0"]}).json()
    assert ser["charts"][0]["series"][0]["t"]
    assert client.get("/api/analyses").json()[0]["analysis_id"] == aid
    assert client.get(f"/api/analyses/{aid}/export").status_code == 200
    assert client.delete(f"/api/analyses/{aid}").json()["status"] == "deleted"
    assert client.get(f"/api/analyses/{aid}").status_code == 404


def test_api_csv_is_converted_then_analysed_like_any_log(client):
    from pathlib import Path
    raw = (Path(__file__).parent / "fixtures" / "data" / "FLT_0002.CSV").read_bytes()
    v = client.post("/api/analyses", files={"file": ("FLT_0002.CSV", raw, "text/csv")}).json()
    aid = v["analysis_id"]
    assert v["meta"]["converted_from"] == "csv" and v["meta"]["source_profile"] == "uav_flt_csv"
    assert v["summary"]["integrity"]["result"] == "verified_up_to" and v["summary"]["classification"]["kind"] == "flight"
    assert v["summary"]["conversion"]["format"]["id"] == "uav_flt_csv" and v["summary"]["identity"]["motor_layout"] == "unknown"
    ulg = client.get(f"/api/analyses/{aid}/download/ulg")
    assert ulg.status_code == 200 and ulg.content[:7] == b"ULog\x01\x12\x35"
    assert client.get(f"/api/analyses/{aid}/download/original").content == raw
    assert client.get(f"/api/analyses/{aid}/download/nope").status_code == 404
    ser = client.get(f"/api/analyses/{aid}/series", params={"sig": ["arjuna_attitude:roll:0"]}).json()
    assert ser["charts"][0]["unit"] == "deg"


def test_api_rejects_bad_inputs(client):
    assert client.post("/api/analyses", files={"file": ("x.ulg", b"ULog\x01\x12\x35garbage", "application/octet-stream")}).status_code == 422
    assert client.post("/api/analyses", files={"file": ("x.csv", b"a,b\n1,2\n3,4\n", "text/csv")}).status_code == 422
    ok_csv = b"time_s,roll\n0,0.1\n0.1,0.2\n0.2,0.1\n"
    assert client.post("/api/analyses", files={"file": ("x.csv", ok_csv, "text/csv")}, data={"symptom": "aliens"}).status_code == 422
    assert client.get("/api/analyses/../../etc").status_code == 404
    refused = client.post("/api/analyses", files={"file": ("x.ulg", flight(info={"frame_nav": "NED"}), "application/octet-stream")}).json()
    assert refused["summary"]["refused"] and client.post(f"/api/analyses/{refused['analysis_id']}/agent",
                                                          json={"consent": True}).status_code == 422


def test_agent_reports_each_step_for_the_progress_display(vib):
    fl, a = vib
    steps = []
    runner.investigate(ULogSource(fl), a, a["findings"], client=ScriptedModel(a["findings"]),
                       on_step=lambda s, d=None: steps.append((s, (d or {}).get("name"))))
    assert steps[0] == ("tool", "get_flight_overview") and ("thinking", None) in steps
    assert steps[-1] == ("validating", None) and any(s == ("tool", "signal_stats") for s in steps)


def test_progress_registry_and_endpoint(client):
    from src.agent import progress
    report = progress.start("an-x")
    report("tool", {"name": "signal_stats", "args": {"topic": "arjuna_vibration"}})
    p = progress.get("an-x")
    assert p["tool"] == "signal_stats" and p["args"]["topic"] == "arjuna_vibration" and p["calls"] == 1
    progress.finish("an-x")
    assert progress.get("an-x") is None
    assert client.get("/api/analyses/an-nothing/agent/progress").json() == {"running": False}
