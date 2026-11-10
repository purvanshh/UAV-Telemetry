"""Direct AI Telemetry Analysis: deterministic layer, model containment, storage separation."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src import direct_llm
from src.direct_analysis import ProcessingLimit, analyze_table, evidence_catalog, event_context, limits

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "fixtures" / "data"
FILES = ["FLT_0001.CSV", "FLT_0002.CSV"]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("UAV_PROFILE_CONFIRMATIONS", str(tmp_path / "none.json"))
    monkeypatch.setenv("UAV_PROFILES_FILE", str(ROOT / "config" / "dataset_profiles.json"))


def raw(name):
    return (DATA / name).read_bytes()


def csv(d):
    b = io.BytesIO()
    d.to_csv(b, index=False)
    return b.getvalue()


def summ(name):
    return analyze_table(raw(name), name)


# ---- deterministic layer ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", FILES)
def test_known_schema_is_recognised_and_described_without_unit_claims(name):
    d = summ(name).data
    df = pd.read_csv(DATA / name)
    assert d["profile"]["id"] == "uav_flt_csv" and d["profile"]["match"] == "exact"
    assert d["data_quality"]["rows"] == len(df) and d["data_quality"]["columns"] == 33
    assert d["timing"]["median_step_raw"] == 20.0 and d["timing"]["n_gaps"] == 0
    assert d["timing"]["time_basis"]["status"] == "supported"  # hypothesis, not verified
    assert "ASSUMING ms" in d["timing"]["duration_s"]["basis"] and d["timing"]["sampling_hz"]["value"] == 50.0
    ch = d["channels"]
    assert ch["phi_deg"]["group"] == "attitude" and ch["gx"]["group"] == "gyroscope" and ch["ax"]["group"] == "accelerometer"
    assert all(ch[c]["group"] == "pilot_command" for c in ("rc_roll", "rc_pitch", "rc_thr", "rc_yaw"))
    assert all(ch[m]["group"] == "motor_output" for m in ("m1", "m2", "m3", "m4"))
    assert all(c["unit"] is None for c in ch.values())  # no verified unit anywhere
    assert ch["phi_deg"]["unit_hypothesis"] == "deg" and ch["phi_deg"]["unit_status"] == "supported"
    text = " ".join(d["limitations"])
    assert "not verified" in text and "yaw" in text and "not validated by the deployed anomaly detector" in text
    assert d["stats"]["accel_magnitude_raw"]["note"].startswith("Euclidean norm of the three raw columns")


@pytest.mark.parametrize("name", FILES)
def test_every_event_and_peak_traces_to_source_rows(name):
    s = summ(name)
    d, df = s.data, pd.read_csv(DATA / name)
    assert d["events"], "the real files contain intervals of notable change"
    cat = evidence_catalog(d)
    for ev in d["events"]:
        assert df.loc[ev["start_row"], "t_ms"] == ev["start_raw"] and df.loc[ev["end_row"], "t_ms"] == ev["end_raw"]
        assert ev["start_row"] <= ev["end_row"] and ev["span_raw"] == ev["end_raw"] - ev["start_raw"]
        for m in ev["measurements"]:
            seg = df.loc[ev["start_row"]: ev["end_row"], m["channel"]]
            assert m["during_max"] == pytest.approx(seg.max(), abs=1e-6) and m["during_min"] == pytest.approx(seg.min(), abs=1e-6)
            for k, eid in m["evidence"].items():
                if m.get(k) is not None:
                    assert cat[eid] == m[k]
    for p in d["peaks"]:
        assert df.loc[p["row"], p["channel"]] == pytest.approx(p["value"], abs=1e-6)
        assert df.loc[p["row"], "t_ms"] == p["raw_time"]


def test_summary_is_deterministic_and_fingerprint_tracks_content():
    a, b = summ("FLT_0001.CSV").data, summ("FLT_0001.CSV").data
    assert a["input_fingerprint"] == b["input_fingerprint"]
    df = pd.read_csv(DATA / "FLT_0001.CSV")
    df.loc[300, "gx"] += 5
    assert analyze_table(csv(df), "x.csv").data["input_fingerprint"] != a["input_fingerprint"]


def test_unknown_columns_and_no_profile_do_not_break_analysis():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"tick": np.arange(300), "sensor_a": rng.normal(0, 1, 300), "sensor_b": np.cumsum(rng.normal(0, 1, 300)), "label": ["x"] * 300})
    out = analyze_table(csv(d), "weird.csv").data
    assert out["profile"]["id"] is None and out["profile"]["match"] == "none"
    assert out["channels"]["sensor_a"]["group"] == "unclassified"
    assert out["data_quality"]["non_numeric_columns"] == ["label"]
    assert out["timing"]["time_column"] is None and "no recognisable time column" in " ".join(out["limitations"])
    assert "duration_s" not in out["timing"]  # no time column -> no duration invented


def test_missing_optional_columns_still_use_the_profile():
    df = pd.read_csv(DATA / "FLT_0001.CSV").drop(columns=["z_set", "tof_h"])
    d = analyze_table(csv(df), "x.csv").data
    assert d["profile"]["match"] == "compatible" and d["channels"]["phi_deg"]["group"] == "attitude"


def synth(n=1000, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"t_ms": np.arange(n) * 20, "gx": rng.normal(0, 1, n), "gy": rng.normal(0, 1, n),
                         "phi_deg": np.cumsum(rng.normal(0, 0.05, n)), "armed": (np.arange(n) > 400).astype(int)})


def test_detectors_find_planted_spike_shift_flatline_and_status_change():
    d = synth()
    d.loc[600, "gx"] += 40                       # isolated spike
    d.loc[700:790, "gy"] += 25                   # sustained offset
    d.loc[300:360, "phi_deg"] = d.loc[300, "phi_deg"]  # flat-lined channel
    ev = analyze_table(csv(d), "x.csv").data["events"]
    kinds = {k for e in ev for k in e["kinds"]}
    assert {"spike", "sustained_shift", "flatline", "status_change"} <= kinds
    assert any(e["start_row"] <= 600 <= e["end_row"] and "gx" in e["channels"] for e in ev)
    assert any(e["start_row"] <= 750 <= e["end_row"] and "gy" in e["channels"] for e in ev)
    assert any(300 <= e["start_row"] <= 362 and "phi_deg" in e["channels"] for e in ev)


def test_quiet_stationary_noise_produces_no_events():
    rng = np.random.default_rng(5)
    d = pd.DataFrame({"t_ms": np.arange(1000) * 20, "gx": rng.normal(0, 1, 1000), "gy": rng.normal(0, 1, 1000), "az": rng.normal(9.8, 0.1, 1000)})
    assert analyze_table(csv(d), "q.csv").data["events"] == []


def test_toggling_two_value_channel_is_not_a_status_flag():
    d = synth()
    d["rc_roll"] = np.where(np.arange(len(d)) % 2, -2, -4)
    out = analyze_table(csv(d), "x.csv").data
    assert not out["stats"]["rc_roll"]["is_flag"] and out["stats"]["armed"]["is_flag"]
    assert not [t for t in out["flag_transitions"] if t["channel"] == "rc_roll"]


def test_missing_nonfinite_text_and_gaps_are_reported_not_hidden():
    d = synth()
    d.loc[100:130, "gx"] = np.nan
    d["bad"] = "text value"
    d.loc[50, "gy"] = np.inf
    d.loc[500:, "t_ms"] += 5000  # a gap of 250x the median step
    out = analyze_table(csv(d), "x.csv").data
    assert out["stats"]["gx"]["n_missing"] == 31 and out["stats"]["gy"]["n_non_finite"] == 1
    assert out["timing"]["n_gaps"] == 1 and out["timing"]["gaps"][0]["step_over_median"] > 100
    assert any("sampling_gap" in e["kinds"] for e in out["events"]) and any("missing_values" in e["kinds"] for e in out["events"])
    assert out["unassessed_periods"] and "bad" in out["data_quality"]["non_numeric_columns"]


def test_invalid_timestamps_are_handled_and_reported():
    d = synth(400)
    d.loc[10, "t_ms"] = d.loc[9, "t_ms"]              # repeated
    d.loc[50, "t_ms"] = np.nan                        # invalid
    d.iloc[100], d.iloc[101] = d.iloc[101].copy(), d.iloc[100].copy()  # out of order
    out = analyze_table(csv(d), "x.csv").data
    t = out["timing"]
    assert t["repeated_timestamps"] >= 1 and t["out_of_order_steps"] >= 1 and t["rows_without_valid_timestamp"] == 1
    assert out["data_quality"]["rows_analysed"] == 399
    assert any("out of order" in n for n in out["data_quality"]["notes"])
    d["t_ms"] = np.nan
    with pytest.raises(ProcessingLimit, match="no valid timestamps"):
        analyze_table(csv(d), "x.csv")


def test_resource_limits(monkeypatch):
    big = synth(2000)
    monkeypatch.setenv("UAV_DIRECT_MAX_ROWS", "1000")
    with pytest.raises(ProcessingLimit, match="rows"):
        analyze_table(csv(big), "big.csv")
    monkeypatch.delenv("UAV_DIRECT_MAX_ROWS")
    monkeypatch.setenv("UAV_DIRECT_MAX_COLUMNS", "3")
    with pytest.raises(ProcessingLimit, match="columns"):
        analyze_table(csv(big), "wide.csv")
    monkeypatch.delenv("UAV_DIRECT_MAX_COLUMNS")
    monkeypatch.setenv("UAV_DIRECT_MAX_CELLS", "100")
    with pytest.raises(ProcessingLimit, match="cells"):
        analyze_table(csv(big), "cells.csv")
    monkeypatch.delenv("UAV_DIRECT_MAX_CELLS")
    monkeypatch.setenv("UAV_DIRECT_TIME_BUDGET_S", "0")
    with pytest.raises(ProcessingLimit, match="processing budget"):
        analyze_table(csv(big), "slow.csv")
    assert limits()["max_rows"] == 500_000


def test_large_file_payload_stays_bounded_and_carries_no_raw_rows():
    d = synth(200_000)
    out = analyze_table(csv(d), "large.csv").data
    payload = direct_llm.build_payload(out)
    body = json.dumps(payload, default=str)
    assert len(body) <= direct_llm.MAX_PAYLOAD_CHARS
    assert out["series"]["bins"] <= 40 and all(len(v["min"]) <= 40 for v in out["series"]["channels"].values())
    assert len(out["events"]) <= 25


def test_event_context_for_charts_covers_before_during_after():
    s = summ("FLT_0001.CSV")
    ev = next(e for e in s.data["events"] if e["category"] == "statistical_irregularity" and e["n_channels_involved"] <= 3)
    c = event_context(s.data, s.table, ev["id"])
    r = c["regions"]
    assert c["window_start"] < r["event"]["start"] <= r["event"]["end"] < c["window_end"]
    assert r["baseline"]["available"] and r["post"]["available"] and "raw time ticks" in c["axis"] and "not verified" in c["axis"]
    assert c["channels"] and all(len(ch["values"]) == len(ch["time"]) for ch in c["channels"])
    assert all(ch["before"]["available"] and ch["during"]["available"] for ch in c["channels"])
    big = next(e for e in s.data["events"] if e["n_channels_involved"] > 10)    # ends close to the end of the recording
    cb = event_context(s.data, s.table, big["id"])
    assert cb["regions"]["post"]["end"] <= float(s.table["t_ms"].iloc[-1]) and len(cb["channels"]) == 8
    assert all({"available", "median", "min", "max", "n"} <= set(ch["after"]) for ch in cb["channels"])
    with pytest.raises(KeyError):
        event_context(s.data, s.table, "E99")


# ---- model containment ---------------------------------------------------------------------------------------------

class FakeClient:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.calls = content, exc, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))


def good_report(data, **over):
    ev = data["events"][0]
    m = ev["measurements"][0]
    r = {
        "executive_summary": f"The file has {data['data_quality']['rows']} rows. Assuming ms, it lasts about {data['timing']['duration_s']['value']} s.",
        "events": [{"event_ref": ev["id"], "title": f"Change in {m['channel']} around row {ev['start_row']}",
                    "summary": f"{m['channel']} ranged from {m['during_min']} to {m['during_max']} in rows {ev['start_row']} to {ev['end_row']}.",
                    "same_time": "", "conclusion": "The raw column departed from its earlier values.",
                    "unknown": "The file does not record the reason.", "review": f"Compare rows {ev['start_row']} to {ev['end_row']} with pilot notes or video."}],
        "data_quality": [{"issue": "Sampling", "detail": f"The median step is {data['timing']['median_step_raw']} raw ticks."}],
        "recommendations": [{"action": f"Compare rows {ev['start_row']} to {ev['end_row']} with pilot notes or video.",
                             "rationale": "The raw channels change there."}],
        "unknowns": ["Units are not verified."],
    }
    r.update(over)
    return json.dumps(r)


def gen(name="FLT_0001.CSV", content=None, exc=None, consent=True):
    data = summ(name).data
    c = FakeClient(content=content if content is not None else good_report(data), exc=exc)
    return data, c, direct_llm.generate(data, consent=consent, client=c)


def test_valid_report_is_accepted_labelled_and_fully_traceable():
    data, c, rep = gen()
    assert rep["status"] == "ok" and rep["label"] == "AI-generated telemetry analysis — not validated by the deployed anomaly detector."
    ev = rep["events"][0]
    d0 = data["events"][0]
    assert ev["event_id"] == d0["id"] and ev["source"]["start_row"] == d0["start_row"] and ev["source"]["end_raw"] == d0["end_raw"]
    assert ev["measurements"] and all(set(m) == {"id", "value"} for m in ev["measurements"])
    assert all(m["id"].startswith(d0["id"] + ".") for m in ev["measurements"])  # evidence is attached by the backend
    assert ev["category"] == d0["category"] and rep["unknowns"] == ["Units are not verified."]
    p = rep["provenance"]
    assert p["analysis_mode"] == "direct_ai_telemetry" and p["model"] and p["prompt_version"] == "direct-ai-3"
    assert p["input_fingerprint"] == data["input_fingerprint"] and p["schema_profile"]["id"] == "uav_flt_csv" and p["generated_at"]
    sent = c.calls[0]
    assert sent["response_format"]["json_schema"]["strict"] and sent["max_tokens"] <= 2800
    body = sent["messages"][1]["content"]
    assert len(body) <= direct_llm.MAX_PAYLOAD_CHARS and "FLT_0001" in body
    assert "anomaly_score" not in body and "threshold" not in body.lower().replace("known_limitations", "")


def test_report_never_mentions_or_imitates_the_detector():
    _d, _c, rep = gen()
    blob = json.dumps({k: v for k, v in rep.items() if k not in ("label", "limitations")}).lower()
    for w in ("lstm", "reconstruction", "anomaly score", "probability", "severity"):
        assert w not in blob


def test_consent_gates_every_external_call_and_missing_key_degrades(monkeypatch):
    data = summ("FLT_0001.CSV").data
    c = FakeClient(content=good_report(data))
    r = direct_llm.generate(data, consent=False, client=c)
    assert r["status"] == "not_requested" and c.calls == [] and r["events"] == []
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = direct_llm.generate(data, consent=True)
    assert r["status"] == "unavailable" and "unaffected" in r["message"]


@pytest.mark.parametrize("exc,fragment", [
    (TimeoutError("t"), "in time"),
    (type("RateLimitError", (Exception,), {})("429 slow down"), "rate"),
    (RuntimeError("boom"), "unavailable"),
])
def test_api_failures_degrade_without_leaking(monkeypatch, exc, fragment):
    monkeypatch.setattr("time.sleep", lambda s: None)
    _d, _c, rep = gen(exc=exc)
    assert rep["status"] == "error" and fragment in rep["message"].lower()
    assert "boom" not in rep["message"] and rep["events"] == []
    assert rep["limitations"]  # deterministic limitations are still present


@pytest.mark.parametrize("content", ["not json", "[]", '{"executive_summary": 1}', '{"events": []}',
                                     json.dumps({"executive_summary": "x", "events": [], "signal_observations": [], "data_quality": [],
                                                 "recommendations": [], "limitations": [], "extra": 1})])
def test_invalid_output_is_ignored(content):
    _d, _c, rep = gen(content=content)
    assert rep["status"] == "invalid_output" and rep["events"] == [] and rep["executive_summary"] == ""


BAD_SENTENCES = [
    ("The LSTM autoencoder flagged this window.", "detector"),
    ("The reconstruction error exceeded the threshold.", "detector"),
    ("This indicates a motor failure.", "fault"),
    ("The pilot suffered a loss of control.", "fault"),
    ("The jump was caused by wind.", "causation"),
    ("This happened because the throttle changed.", "causation"),
    ("The aircraft was safe to fly.", "safety"),
    ("The flight was unsafe.", "safety"),
    ("Yaw drifted during this interval.", "absent_signal"),
    ("Barometric altitude rose.", "absent_signal"),
    ("Roll changed by 4 degrees.", "unit_claim"),
    ("az was 9.8 m/s2 at rest.", "unit_claim"),
    ("The aircraft pitched up sharply.", "direction"),
    ("rc_roll shows the measured roll angle.", "command_as_attitude"),
    ("The channel reached 37.51 in this interval.", "ungrounded_number"),
    ("The flight lasted 9 seconds.", "time_unit_without_assumption"),
    ("There was a significant change in motion dynamics.", "overclaim"),
    ("This suggests the status flag has a meaning.", "inference"),
]


@pytest.mark.parametrize("sentence,reason", BAD_SENTENCES)
def test_unsupported_claims_are_removed_and_recorded(sentence, reason):
    data = summ("FLT_0001.CSV").data
    base = json.loads(good_report(data))
    base["executive_summary"] = "The file has 522 rows. " + sentence
    base["events"][0]["conclusion"] = sentence
    base["data_quality"][0]["detail"] = sentence + " " + base["data_quality"][0]["detail"]
    rep = direct_llm.generate(data, consent=True, client=FakeClient(content=json.dumps(base)))
    blob = json.dumps({k: v for k, v in rep.items() if k not in ("rejected", "provenance", "normalized")})
    assert sentence not in blob
    assert rep["summary_source"] == "deterministic" and rep["executive_summary"] == data["operator"]["headline"]  # no summary by omission
    assert any(reason in r["reason"] for r in rep["rejected"]), rep["rejected"]
    assert rep["status"] == "partial" and "removed" in rep["message"]


def test_clean_numbers_pass_including_rounding_and_ids():
    data = summ("FLT_0001.CSV").data
    g = direct_llm.Grounding(direct_llm.build_payload(data), evidence_catalog(data))
    dur = data["timing"]["duration_s"]["value"]
    assert g.ungrounded(f"Assuming ms it lasts {dur} s, about {round(dur, 1)} s; see E1.gx.during_max and stat.gx.max, 3 events.") == []
    assert g.ungrounded("The value was 8123.77.") == ["8123.77"]


def test_events_must_exist_and_positions_come_from_the_backend():
    data = summ("FLT_0001.CSV").data
    base = json.loads(good_report(data))
    fake = dict(base["events"][0], event_ref="E99", title="Invented event")
    lying = dict(base["events"][0], summary="It happened at row 9999 and time 123456.", title="Shifted")
    base["events"] = [fake, lying]
    rep = direct_llm.generate(data, consent=True, client=FakeClient(content=json.dumps(base)))
    assert all(e["event_id"] != "E99" for e in rep["events"])
    reasons = {r["reason"] for r in rep["rejected"]}
    assert "unknown_or_duplicate_event_ref" in reasons
    assert not any("9999" in json.dumps(e) for e in rep["events"])  # the shifted timestamp sentence is dropped
    for e in rep["events"]:
        d = next(x for x in data["events"] if x["id"] == e["event_id"])
        assert e["source"]["start_row"] == d["start_row"]


def test_mandatory_limitations_cannot_be_dropped_by_the_model():
    data = summ("FLT_0001.CSV").data
    base = json.loads(good_report(data))
    base["unknowns"] = []
    rep = direct_llm.generate(data, consent=True, client=FakeClient(content=json.dumps(base)))
    for lim in data["limitations"]:
        assert lim in rep["limitations"]
    assert any("not validated by the deployed anomaly detector" in x for x in rep["limitations"])


def test_unusable_model_summary_falls_back_to_a_deterministic_one():
    data = summ("FLT_0001.CSV").data
    base = json.loads(good_report(data, executive_summary="The motor failed because of vibration."))
    rep = direct_llm.generate(data, consent=True, client=FakeClient(content=json.dumps(base)))
    assert rep["summary_source"] == "deterministic" and rep["executive_summary"] == data["operator"]["headline"] and rep["status"] == "partial"


# ---- API: storage, separation, reuse ---------------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "flights"))
    monkeypatch.delenv("DIRECT_DATA_DIR", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("UAV_MODEL_DIR", str(ROOT / "uav_model"))
    from backend import deps

    deps.get_store.cache_clear()
    yield TestClient(app)
    deps.get_store.cache_clear()


_ORIG_GENERATE = direct_llm.generate  # captured once so repeated patching never chains fakes


def use_fake(monkeypatch, content=None, exc=None):
    c = FakeClient(content=content, exc=exc)
    orig = _ORIG_GENERATE

    def patched(data, *, consent, client=None):
        if c.content is None and exc is None:
            c.content = good_report(data)
        return orig(data, consent=consent, client=c)

    monkeypatch.setattr(direct_llm, "generate", patched)
    return c


def post(client, name="FLT_0001.CSV", **data):
    return client.post("/api/direct/analyze", files={"file": (name, raw(name), "text/csv")}, data=data)


def test_without_consent_only_the_deterministic_summary_is_stored(client, monkeypatch):
    called = use_fake(monkeypatch)
    r = post(client).json()
    assert r["stage"] == "summary_only" and r["report"] is None and called.calls == []
    assert r["label"].startswith("AI-generated telemetry analysis")
    assert r["summary"]["events"] and r["summary"]["profile"]["id"] == "uav_flt_csv"
    aid = r["analysis_id"]
    assert client.get(f"/api/direct/{aid}").json()["report"] is None


def test_acceptance_flt_0001_gets_an_evidence_linked_report(client, monkeypatch):
    use_fake(monkeypatch)
    r = post(client, llm_consent="true").json()
    rep = r["report"]
    assert r["stage"] == "ok" and rep["executive_summary"] and rep["events"]
    assert rep["events"][0]["source"]["start_row"] is not None and rep["events"][0]["measurements"]
    assert any("not validated by the deployed anomaly detector" in x for x in rep["limitations"])
    assert any("yaw" in x.lower() for x in rep["limitations"])  # missing signals are stated
    assert rep["label"] == "AI-generated telemetry analysis — not validated by the deployed anomaly detector."
    aid = r["analysis_id"]
    ctx = client.get(f"/api/direct/{aid}/events/{rep['events'][0]['event_id']}/context").json()
    assert ctx["time"] and ctx["channels"]
    assert client.get(f"/api/direct/{aid}/rows", params={"limit": 5}).json()["returned"] == 5
    ex = client.get(f"/api/direct/{aid}/export")
    assert ex.headers["content-disposition"].endswith('_ai_analysis.json"') and json.loads(ex.content)["report"]["status"] == "ok"
    assert client.get("/api/direct").json()[0]["analysis_id"] == aid


def test_direct_analysis_never_touches_the_lstm_workflow(client, monkeypatch, tmp_path):
    import backend.routers.flights as fl
    import src.inference_service as inf

    def boom(*a, **k):
        raise AssertionError("the LSTM detector must not run for direct analysis")

    monkeypatch.setattr(fl, "analyze_flight", boom)
    monkeypatch.setattr(inf, "analyze_flight", boom)
    monkeypatch.setattr(inf, "get_detector", boom)
    use_fake(monkeypatch)
    r = post(client, llm_consent="true").json()
    assert r["stage"] == "ok"
    assert client.get("/api/flights").json() == []  # no flight was created
    root = tmp_path / "direct" / r["analysis_id"]
    assert sorted(p.name for p in root.iterdir()) == ["meta.json", "report.json", "summary.json", "table.parquet"]
    assert not (tmp_path / "flights").exists() or not any((tmp_path / "flights").iterdir())
    blob = json.dumps(r).lower()
    assert "anomaly_score" not in blob and "is_anomaly" not in blob and "windows" not in r["summary"]


def test_existing_flight_results_are_never_overwritten(client, monkeypatch):
    from tests.conftest import _make_att, _make_baro, _make_imu

    fr = {"IMU": _make_imu(500), "ATT": _make_att(200), "BARO": _make_baro(200)}
    up = client.post("/api/flights/upload", files={k.lower(): (f"{k}.csv", v.to_csv(index=False).encode(), "text/csv") for k, v in fr.items()},
                     data={"flight_id": "keep-me"}).json()
    assert up["ok"]
    before = client.post("/api/flights/keep-me/analyze", json={}).json()
    rep_before = client.get("/api/flights/keep-me/report").json()
    use_fake(monkeypatch)
    post(client, llm_consent="true")
    assert client.get("/api/flights/keep-me/report").json() == rep_before
    assert client.get("/api/flights/keep-me").json()["analysis"]["n_windows"] == before["n_windows"]


def test_identical_input_reuses_the_report_instead_of_calling_again(client, monkeypatch):
    c = use_fake(monkeypatch)
    first = post(client, llm_consent="true").json()
    n = len(c.calls)
    second = post(client, llm_consent="true").json()
    assert len(c.calls) == n and second["report"]["reused_from"] == first["analysis_id"]
    assert second["analysis_id"] != first["analysis_id"]


def test_late_consent_generates_the_report_and_requires_it(client, monkeypatch):
    c = use_fake(monkeypatch)
    aid = post(client).json()["analysis_id"]
    assert client.post(f"/api/direct/{aid}/ai", json={"consent": False}).status_code == 400 and c.calls == []
    r = client.post(f"/api/direct/{aid}/ai", json={"consent": True}).json()
    assert r["report"]["status"] == "ok" and len(c.calls) == 1
    again = client.post(f"/api/direct/{aid}/ai", json={"consent": True}).json()
    assert again["report"]["reused"] is True and len(c.calls) == 1


def test_a_failed_attempt_never_replaces_a_good_report(client, monkeypatch, tmp_path):
    use_fake(monkeypatch)
    aid = post(client, llm_consent="true").json()["analysis_id"]
    from backend.direct_store import DirectStore
    from backend.routers.direct import _run_ai

    store = DirectStore(tmp_path / "direct")
    good = store.load_report(aid)
    stale = {**good, "provenance": {**good["provenance"], "input_fingerprint": "something-else"}}
    store.save_report(aid, stale)
    use_fake(monkeypatch, exc=RuntimeError("down"))
    monkeypatch.setattr("time.sleep", lambda s: None)
    out = _run_ai(store, aid, True)
    assert out["status"] == "ok" and "earlier report" in out["message"]
    assert store.load_report(aid)["provenance"]["input_fingerprint"] == "something-else"  # untouched
    assert store.load_meta(aid)["ai_attempt"]["status"] == "error"


def test_failure_modes_through_the_api(client, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    use_fake(monkeypatch, exc=TimeoutError("t"))
    r = post(client, llm_consent="true").json()
    assert r["stage"] == "error" and r["report"]["status"] == "error" and r["summary"]["events"]  # deterministic part intact
    use_fake(monkeypatch, content="not json")
    assert post(client, "FLT_0002.CSV", llm_consent="true").json()["stage"] == "invalid_output"


def test_limits_and_bad_ids_over_the_api(client, monkeypatch):
    monkeypatch.setenv("UAV_DIRECT_MAX_ROWS", "100")
    r = post(client)
    assert r.status_code == 422 and "rows" in r.json()["detail"]
    monkeypatch.delenv("UAV_DIRECT_MAX_ROWS")
    monkeypatch.setenv("UAV_MAX_UPLOAD_MB", "0.001")
    assert post(client).status_code == 413
    monkeypatch.delenv("UAV_MAX_UPLOAD_MB")
    assert client.get("/api/direct/..%2F..%2Fetc").status_code in (404, 400)
    assert client.get("/api/direct/does-not-exist").status_code == 404
    assert client.delete("/api/direct/does-not-exist").status_code == 404


def test_delete_removes_only_the_analysis(client, monkeypatch):
    use_fake(monkeypatch)
    aid = post(client, llm_consent="true").json()["analysis_id"]
    assert client.delete(f"/api/direct/{aid}").json()["status"] == "deleted"
    assert client.get(f"/api/direct/{aid}").status_code == 404 and client.get("/api/direct").json() == []
