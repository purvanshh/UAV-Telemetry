"""The pilot briefing must stay plain-language and must never overreach."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.direct_analysis import analyze_table
from src.operator_report import build_operator_report
from src.pilot_report import LIMITATION, REVIEW_LABEL, build_deep_pilot_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "fixtures" / "data"

BANNED = [
    "statistical irregularity", "median absolute deviation", "mad ", "controller-internal",
    "cross-channel", "sample-to-sample", "baseline spread", "reconstruction error",
    "feature-level", "excursion", "robust sigma", "channel",
]
OVERREACH = ["motor failure", "loss of control", "is safe", "is unsafe", "replace", "caused by", "proves"]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("UAV_PROFILE_CONFIRMATIONS", str(tmp_path / "none.json"))
    monkeypatch.setenv("UAV_PROFILES_FILE", str(ROOT / "config" / "dataset_profiles.json"))


def _report(name):
    s = analyze_table((DATA / name).read_bytes(), name)
    return s.data, build_operator_report(s.data)


@pytest.mark.parametrize("name", ["FLT_0001.CSV", "FLT_0002.CSV"])
def test_pilot_text_uses_plain_language_and_no_raw_channel_names(name):
    data, rep = _report(name)
    # technical_why is engineering-only and must stay inside the Technical Details section.
    events = [{k: v for k, v in e.items() if k != "technical_why"} for e in rep["pilot_events"].values()]
    text = json.dumps([rep["pilot_report"], events]).lower()
    for term in BANNED + OVERREACH:
        assert term not in text, term
    for col in data["channels"]:
        if len(col) > 2:
            assert not re.search(rf"\b{re.escape(col.lower())}\b", text), col


@pytest.mark.parametrize("name", ["FLT_0001.CSV", "FLT_0002.CSV"])
def test_briefing_structure_and_labels(name):
    _data, rep = _report(name)
    p = rep["pilot_report"]
    assert p["limitation"] == LIMITATION
    assert p["mode_label"] == "AI Analysis" and "not scored" in p["mode_note"]
    assert {a["review_label"] for a in p["attention"]} <= set(REVIEW_LABEL.values())
    assert 1 <= len(p["attention"]) <= 3
    for ev in rep["pilot_events"].values():
        assert ev["what_happened"] and ev["why_review"] and 1 <= len(ev["what_to_check"]) <= 2
        assert ev["when"]["display"]
        if ev["when"]["approximate"]:
            assert "not verified" in ev["when"]["approximate"] or "About" in ev["when"]["approximate"]


def test_deep_mode_is_labelled_as_deployed_model_findings():
    detail = {"event_id": "D1", "start_label": "00:05", "end_label": "00:07", "duration_s": 2.0,
              "channels": [{"pilot_label": "roll angle"}], "signal_checks": [], "flagged_window_count": 4}
    p = build_deep_pilot_report(report=None, details=[detail], coverage=None, verdict={}, analysed_s=10, flight_s=12)
    assert p["mode_label"] == "Deep Analysis" and "deployed" in p["mode_note"]
    assert "not confirmed faults" in p["summary"]
    assert p["attention"][0]["review_label"] == "Review recommended"
    assert p["limitation"] == LIMITATION
