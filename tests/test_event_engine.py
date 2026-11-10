"""Deterministic event engine and operator report: noise handling, persistence, merging, associations, wording."""

from __future__ import annotations

import inspect
import io
import json
import re

import numpy as np
import pandas as pd
import pytest

from src import event_engine, operator_report
from src.direct_analysis import analyze_table
from src.event_engine import BASELINE_MIN, SPIKE_MAX_SAMPLES, Z_THRESHOLD, channel_scale, detect_departures

from tests.test_direct_analysis import FILES, _env, summ  # noqa: F401  (autouse env fixture is re-exported for this module)


def csv(d):
    b = io.BytesIO()
    d.to_csv(b, index=False)
    return b.getvalue()


def base(n=600, seed=3, **cols):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"t_ms": np.arange(n) * 20})
    for k in ("c1", "c2", "c3"):
        d[k] = rng.normal(0, 1, n)
    for k, v in cols.items():
        d[k] = v
    return d


def run(d, name="x.csv"):
    return analyze_table(csv(d), name).data


def ev_for(data, channel, category=None):
    return [e for e in data["events"] if channel in e["channels"] and (category is None or e["category"] == category)]


# ---- constant and noisy signals ---------------------------------------------------------------------------------------

def test_constant_signal_is_skipped_without_division_errors():
    y = np.full(500, 7.0)
    assert channel_scale(y) is None and detect_departures(y, 50) == []
    d = base(k=np.full(600, 3.0))
    out = run(d)
    assert not ev_for(out, "k") and "k" in out["data_quality"]["constant_channels"]


def test_noise_aware_threshold_same_step_flags_quiet_channel_not_noisy_one():
    rng = np.random.default_rng(0)
    quiet, noisy = rng.normal(0, 0.1, 600), rng.normal(0, 2.0, 600)
    for y in (quiet, noisy):
        y[300:] += 3.0                                     # identical absolute step in both
    d = base(quiet=quiet, noisy=noisy)
    out = run(d)
    assert any("sustained_shift" in e["kinds"] for e in ev_for(out, "quiet", "statistical_irregularity"))
    assert not ev_for(out, "noisy", "statistical_irregularity")


def test_pure_noise_produces_no_events_even_with_heavy_tails():
    rng = np.random.default_rng(11)
    d = base(heavy=rng.standard_t(4, 600) * 0.5)
    assert run(d)["events"] == []


def test_quantised_signal_one_step_flicker_is_not_an_event():
    y = np.full(600, 10.0)
    y[::37] += 1.0                                         # one resolution step, occasionally
    y += np.random.default_rng(1).integers(0, 2, 600)      # integer-quantised jitter
    assert not ev_for(run(base(q=y)), "q")


# ---- spikes vs sustained changes -------------------------------------------------------------------------------------------

def test_isolated_spike_is_labelled_spike_and_brief():
    d = base()
    d.loc[300, "c1"] += 40
    e = ev_for(run(d), "c1", "statistical_irregularity")
    assert len(e) == 1 and e[0]["kinds"] == ["spike"]
    assert e[0]["measurements"][0]["persistence"] == "brief" and e[0]["n_samples"] <= SPIKE_MAX_SAMPLES + 1


def test_weak_two_sample_blip_is_not_reported_but_sustained_same_size_is():
    blip, held = base(seed=5), base(seed=5)
    blip.loc[300:301, "c1"] += 7.5                         # > Z threshold but far below the spike evidence bar
    held.loc[300:380, "c1"] += 7.5
    assert not ev_for(run(blip), "c1", "statistical_irregularity")
    h = ev_for(run(held), "c1", "statistical_irregularity")
    assert h and "sustained_shift" in h[0]["kinds"]


def test_returning_excursion_is_transient_and_level_that_stays_is_sustained():
    ret, stay = base(seed=7), base(seed=7)
    ret.loc[300:325, "c1"] += 15
    stay.loc[300:, "c1"] += 15
    r = ev_for(run(ret), "c1", "statistical_irregularity")[0]
    s = ev_for(run(stay), "c1", "statistical_irregularity")[0]
    assert "transient_excursion" in r["kinds"] and r["measurements"][0]["persistence"] == "returned"
    m = s["measurements"][0]
    assert "sustained_shift" in s["kinds"] and m["persistence"].startswith("sustained") and "end of the recording" in m["persistence"]
    assert m["before_median"] == pytest.approx(0, abs=0.5) and m["after_median"] is None or m["after_median"] > 10


def test_sustained_shift_that_returns_spans_the_whole_excursion():
    d = base()
    d.loc[300:390, "c1"] += 15
    e = ev_for(run(d), "c1", "statistical_irregularity")[0]
    assert e["start_row"] <= 301 and e["end_row"] >= 388


# ---- missing baselines / post windows ---------------------------------------------------------------------------------------

def test_event_near_start_has_no_baseline_and_near_end_no_post_window():
    d = base(flag=(np.arange(600) > 5).astype(int))        # status change at the very start
    d.loc[588:, "c1"] += 20                                # change that never returns, within the last 12 samples
    out = run(d)
    start = next(e for e in out["events"] if "flag" in e["channels"])
    m = start["measurements"][0]
    assert m["before"]["available"] is False and m["before_median"] is None and m["shift"] is None
    late = ev_for(out, "c1", "statistical_irregularity")[0]
    assert late["measurements"][0]["after"]["available"] is False and late["measurements"][0]["after_median"] is None
    text = operator_report.channel_line(m)
    assert "no earlier baseline" in text.lower() and "None" not in text and "nan" not in text.lower()
    assert operator_report.fmt(None) == "Unavailable" and operator_report.fmt(0) == "0"


# ---- merging and overlap -----------------------------------------------------------------------------------------------------

def test_nearby_detections_merge_and_distant_ones_do_not():
    d = base()
    d.loc[300:340, "c1"] += 15
    d.loc[305:345, "c2"] += 15                             # overlapping, same category -> one event
    d.loc[520:560, "c3"] += 15                             # far away -> its own event
    ev = [e for e in run(d)["events"] if e["category"] == "statistical_irregularity"]
    assert len(ev) == 2 and set(ev[0]["channels"]) == {"c1", "c2"} and ev[1]["channels"] == ["c3"]


def test_categories_are_never_merged_but_overlaps_are_linked():
    d = base(armed=(np.arange(600) > 300).astype(int))
    d.loc[300:340, "c1"] += 15
    out = run(d)
    irr = ev_for(out, "c1", "statistical_irregularity")[0]
    st = ev_for(out, "armed", "state_transition")[0]
    assert irr["id"] != st["id"]
    assert any(c["event_id"] == st["id"] and c["relation"] == "overlapping" for c in irr["concurrent_with"])
    assert any(c["event_id"] == irr["id"] for c in st["concurrent_with"])


def test_data_quality_events_are_separate_from_irregularities():
    d = base()
    d.loc[200:230, "c1"] = np.nan
    d.loc[400:, "t_ms"] += 9000
    out = run(d)
    dq = [e for e in out["events"] if e["category"] == "data_quality"]
    assert {"missing_values", "sampling_gap"} <= {k for e in dq for k in e["kinds"]}
    assert all(e["category"] == "data_quality" for e in out["events"] if set(e["kinds"]) & {"missing_values", "sampling_gap", "flatline"})


# ---- cross-channel association -------------------------------------------------------------------------------------------------

def motors(n=600, start=300, seed=2):
    rng = np.random.default_rng(seed)
    cols = {}
    for k in ("m1", "m2", "m3", "m4"):
        v = np.full(n, 1000.0) + rng.normal(0, 0.3, n).round()
        v[start:] += np.linspace(100, 500, n - start) + rng.normal(0, 3, n - start)
        cols[k] = v
    return cols


def test_all_four_motor_labelled_channels_rising_together_is_stated_only_when_measured():
    out = run(base(**motors()))
    e = ev_for(out, "m1")[0]
    g = next(a for a in e["associations"] if a["type"] == "group" and a["group"] == "motor_output") if any(
        a["type"] == "group" for a in e["associations"]) else None
    # these generic column names carry no profile, so the group claim must NOT be made...
    assert g is None
    assert any(a["type"] == "simultaneous" and {"m1", "m2", "m3", "m4"} <= set(a["channels"]) for a in e["associations"])
    assert all(a["physical_relationship"] == "not established" for a in e["associations"])


def test_group_statement_uses_registered_groups_on_the_fixture_file():
    out = summ("FLT_0001.CSV").data
    e = next(e for e in out["events"] if e["n_channels_involved"] > 10)
    g = next(a for a in e["associations"] if a["type"] == "group" and a["group"] == "motor_output")
    assert sorted(g["channels"]) == ["m1", "m2", "m3", "m4"] and g["n_in_file"] == 4 and g["direction"] == "higher"
    sim = [a for a in e["associations"] if a["type"] == "simultaneous"]
    assert any({"m1", "m2", "m3", "m4"} <= set(a["channels"]) for a in sim)
    texts = [s["text"] for s in operator_report.association_statements(e)]
    assert any(t.startswith("All four motor-labelled channels (m1, m2, m3, m4) had higher raw values") for t in texts)
    gyro = next(a for a in e["associations"] if a["type"] == "group" and a["group"] == "gyroscope")
    assert len(gyro["channels"]) == 3                       # "two gyroscope channels" would be wrong here: three changed
    assert out["operator"]["headline"].count("motor-labelled") == 1


def test_sequential_changes_are_distinguished_from_simultaneous_and_make_no_causal_claim():
    d = base()
    d.loc[300:360, "c1"] += 15
    d.loc[340:400, "c2"] += 15                             # starts 40 samples later
    d.loc[340:400, "c3"] += 15
    e = next(e for e in run(d)["events"] if e["category"] == "statistical_irregularity")
    kinds = [a["type"] for a in e["associations"]]
    assert "sequential" in kinds and "simultaneous" in kinds
    seq = next(a for a in e["associations"] if a["type"] == "sequential")
    assert seq["steps"][0]["channels"] == ["c1"] and seq["steps"][1]["lag_samples"] >= 30
    text = " ".join(s["text"] for s in operator_report.association_statements(e))
    assert "timing observation only" in text
    assert not re.search(r"\b(caused|because|due to|led to|triggered)\b", text)


def test_correlation_is_reported_as_statistical_not_physical():
    rng = np.random.default_rng(4)
    z = np.cumsum(rng.normal(0, 0.3, 600))
    d = base(a=z + rng.normal(0, 0.1, 600), b=2 * z + rng.normal(0, 0.1, 600))
    d.loc[300:340, "a"] += 30
    d.loc[300:340, "b"] += 60
    e = next(e for e in run(d)["events"] if {"a", "b"} <= set(e["channels"]))
    corr = [a for a in e["associations"] if a["type"] == "correlation"]
    assert corr and corr[0]["basis"].startswith("statistical correlation") and corr[0]["physical_relationship"] == "not established"
    stmt = next(s for s in operator_report.association_statements(e) if s["kind"] == "statistical correlation")
    assert "does not establish a physical relationship" in stmt["text"]


# ---- unknown semantics, timebase uncertainty, commands -------------------------------------------------------------------------

def test_unknown_columns_stay_raw_channels_with_unverified_meaning():
    out = run(base())
    out2 = run(base().assign(c1=lambda d: d.c1 + np.where(np.arange(600) > 300, 20, 0)))
    e = ev_for(out2, "c1")[0]
    m = e["measurements"][0]
    assert m["semantics"] == "raw channel: meaning unverified" and m["group"] in (None, "unclassified")
    n = out2["operator"]["events"][e["id"]]
    blob = json.dumps(n)
    assert "meaning unverified" in blob and "motor" not in blob and "gyro" not in blob.lower()
    assert out["events"] == []


def test_pilot_command_channels_are_state_transitions_not_irregularities():
    out = summ("FLT_0001.CSV").data
    assert all(e["category"] == "state_transition" for e in out["events"] if e["channels"][0] in ("rc_yaw", "rc_thr") and len(e["channels"]) <= 2)
    m = next(m for e in out["events"] for m in e["measurements"] if m["channel"] == "rc_yaw")
    assert "pilot command input" in m["semantics"]
    assert not any(e["category"] == "statistical_irregularity" and "rc_yaw" in e["channels"] for e in out["events"])


def test_time_unit_uncertainty_is_labelled_and_no_time_column_gives_rows_only():
    out = summ("FLT_0001.CSV").data
    e = out["events"][-2]
    w = out["operator"]["events"][e["id"]]["when"]
    assert "raw ticks" in w and "unit not verified" in w and "assuming ms" in w.lower()
    nt = run(base().drop(columns=["t_ms"]).assign(c1=lambda d: d.c1 + np.where(np.arange(600) > 300, 20, 0)))
    ev = nt["events"][0]
    assert ev["start_raw"] is None and ev["span_s_assumed"] is None
    assert "no time column" in nt["operator"]["events"][ev["id"]]["when"] and "s" not in re.findall(r"\bs\b", nt["operator"]["events"][ev["id"]]["when"])


# ---- wording rules for the deterministic report -----------------------------------------------------------------------------------

FORBIDDEN = re.compile(r"\b(fault\w*|fail\w*|malfunction\w*|crash\w*|loss of control|unsafe|safe to fly|airworthy|caused by|because|due to|"
                       r"degrees?|rad/s|m/s|nose[- ](up|down)|climbed|descended|lstm|anomaly score|severity|probability)\b", re.I)


@pytest.mark.parametrize("name", FILES)
def test_operator_report_makes_no_diagnosis_cause_unit_or_motion_claims(name):
    op = summ(name).data["operator"]
    text = json.dumps({k: v for k, v in op.items() if k != "essential_limitations"})
    hits = FORBIDDEN.findall(text)
    assert not hits, hits
    assert any("not validated by the deployed anomaly detector" in x for x in op["essential_limitations"])


def test_every_event_links_to_source_rows_and_has_a_selection_reason():
    for name in FILES:
        data = summ(name).data
        for e in data["events"]:
            assert e["start_row"] <= e["end_row"] and e["why_selected"] and e["category"] in event_engine.CATEGORIES
            assert e["priority"] in ("review_first", "review", "minor")
            for m in e["measurements"]:
                assert m["onset_row"] >= e["start_row"] - 1 and m["evidence"]


# ---- determinism and independence -----------------------------------------------------------------------------------------------

def test_engine_is_deterministic_and_independent_of_the_language_model():
    a, b = summ("FLT_0001.CSV").data, summ("FLT_0001.CSV").data
    assert json.dumps(a["events"], sort_keys=True) == json.dumps(b["events"], sort_keys=True) and a["input_fingerprint"] == b["input_fingerprint"]
    for mod in (event_engine, operator_report):
        src = inspect.getsource(mod)
        assert "llm_service" not in src and "openai" not in src.lower().replace("openai", "", 0) or "import openai" not in src
    assert "llm" not in " ".join(m for m in inspect.getsource(event_engine).splitlines() if m.startswith(("import", "from")))


def test_documented_constants_are_what_the_tests_assume():
    assert Z_THRESHOLD >= 5 and BASELINE_MIN >= 10 and SPIKE_MAX_SAMPLES >= 1
    assert event_engine.CATEGORIES == ("data_quality", "state_transition", "statistical_irregularity")
