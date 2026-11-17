"""Fix check, flight comparison and the pre-flight check."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.arjuna.analysis import analyse
from src.arjuna.compare import compare
from src.arjuna.preflight import preflight_check
from src.arjuna.ulog import load_flight
from tests.fixtures.arjuna_synth import flight


def run(**kw):
    fl = load_flight(flight(**kw), "x.ulg")
    return fl, analyse(fl)


# ---- fix check / compare ----------------------------------------------------------------------------------------------

def test_vibration_fix_is_reported_with_the_numbers():
    _, bad = run(vibe_spike=(40, 70, 75.0))
    _, good = run()
    r = compare(bad, good)
    assert r["status"] == "fixed" and r["status_text"] == "Fixed"
    vib = next(p for p in r["problems"] if p["code"] == "VIB_HIGH")
    assert vib["state"] == "fixed" and "down" in vib["evidence"] and "→" in vib["evidence"]
    assert any(m["key"] == "vibration_p95" and m["status"] == "better" for m in r["metrics"])


def test_going_the_other_way_is_new_problems_and_worse():
    _, bad = run(vibe_spike=(40, 70, 75.0))
    _, good = run()
    r = compare(good, bad)
    assert r["status"] == "worse" and r["problems"][0]["state"] == "new" and "New" in r["problems"][0]["text"]


def test_partly_fixed_and_not_fixed_and_no_problems():
    _, both = run(vibe_spike=(40, 70, 75.0), cell_min=2.9)   # vibration + low battery
    _, vib_only = run(vibe_spike=(40, 70, 75.0))
    _, healthy = run()
    part = compare(both, vib_only)
    assert part["status"] == "partly_fixed" and {p["code"]: p["state"] for p in part["problems"]}["POWER"] == "fixed"
    assert {p["code"]: p["state"] for p in part["problems"]}["VIB_HIGH"] == "still"
    assert compare(vib_only, vib_only)["status"] == "not_fixed"
    assert compare(healthy, healthy)["status"] == "no_problems"


def test_a_small_change_is_not_called_an_improvement():
    _, a = run(vibe=8.0)
    _, b = run(vibe=8.4)
    assert all(m["status"] == "same" for m in compare(a, b)["metrics"] if m["key"].startswith("vibration"))


def test_non_flights_cannot_be_compared():
    _, ground = run(takeoff_s=50, landing_s=55)
    _, good = run()
    r = compare(ground, good)
    assert r["status"] == "not_comparable" and "not a full flight" in r["status_detail"]


# ---- pre-flight check -------------------------------------------------------------------------------------------------

def test_preflight_ready_for_a_healthy_short_test():
    fl, a = run()
    r = preflight_check(fl, a)
    assert r["status"] == "ready" and r["status_text"] == "Ready for a mission"
    by = {c["id"]: c["status"] for c in r["checks"]}
    assert by["vibration"] == by["battery"] == by["gps"] == by["motors"] == "pass"
    assert by["link"] == "unknown"           # never counted as a pass when the log cannot tell
    assert "safe" not in (r["status_text"] + r["status_detail"]).lower() and "cannot promise" in r["caveat"]


def test_preflight_not_ready_when_the_test_found_a_serious_problem_or_vibration():
    r = preflight_check(*run(tilt_spike=75.0, crash_event_at=100.2))
    assert r["status"] == "not_ready" and any(c["id"] == "warnings" and c["status"] == "fail" for c in r["checks"])
    v = preflight_check(*run(vibe_spike=(20, 90, 75.0)))
    assert v["status"] == "not_ready" and next(c for c in v["checks"] if c["id"] == "vibration")["status"] == "fail"
    assert any("propeller" in x for x in v["actions"])


def test_preflight_bench_run_works_but_says_flight_behaviour_was_not_tested():
    fl, a = run(takeoff_s=50, landing_s=55)
    r = preflight_check(fl, a)
    by = {c["id"]: c for c in r["checks"]}
    assert by["flight"]["status"] == "unknown" and "did not fly" in by["flight"]["detail"]
    assert by["motors"]["status"] in ("unknown", "pass")


def test_preflight_low_battery_at_the_start_fails_and_is_not_counted_ready():
    fl, a = run(cell_min=3.7)
    ok = {c["id"]: c for c in preflight_check(fl, a)["checks"]}["battery"]
    assert ok["status"] == "pass" and ok["provisional"] is True   # the 3.7 V rest limit is provisional and says so


# ---- API --------------------------------------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYSIS_DATA_DIR", str(tmp_path / "analyses"))
    from backend.main import app
    return TestClient(app)


def test_api_preflight_upload_and_compare(client):
    up = lambda raw, **d: client.post("/api/analyses", files={"file": ("f.ulg", raw, "application/octet-stream")}, data=d).json()  # noqa: E731
    bad = up(flight(vibe_spike=(40, 70, 75.0)))
    good = up(flight())
    assert bad["meta"]["purpose"] == "flight" and "preflight" not in bad["summary"]
    pre = up(flight(), purpose="preflight")
    assert pre["meta"]["purpose"] == "preflight" and pre["summary"]["preflight"]["status"] == "ready"
    r = client.get("/api/analyses/compare", params={"before": bad["analysis_id"], "after": good["analysis_id"]}).json()
    assert r["result"]["status"] == "fixed" and r["before"]["analysis_id"] == bad["analysis_id"]
    assert client.get("/api/analyses/compare", params={"before": "an-nope", "after": good["analysis_id"]}).status_code == 404
    assert client.post("/api/analyses", files={"file": ("f.ulg", flight(), "application/octet-stream")}, data={"purpose": "x"}).status_code == 422
