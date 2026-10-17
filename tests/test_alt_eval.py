"""Phase 3/4 runner: protocol, baseline consistency with Phase 2, screen logic."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

import src.alt_eval as ae
import src.evaluation_report as er
from src.artifact_loader import load_artifacts
from src.eval_corpus import build_evaluation_manifest
from tests.test_evaluation_report import _write_flight

LEVELS = {t: lv[:1] for t, lv in er.DEFAULT_LEVELS.items()}


@pytest.fixture()
def setup(tmp_path, fixture_model_dir, monkeypatch):
    srcs = []
    for i in range(3):
        d = _write_flight(tmp_path / f"f{i}", seed=i + 1)
        srcs.append({"path": d, "origin": "tmp", "kind": "real_unreviewed", "eval_role": "synthetic_source",
                     "log_id": None, "store_id": f"f{i}", "file_sha256": None})
    man = build_evaluation_manifest(srcs, seed=3, fractions={"val": 0.34, "test": 0.66})
    art = dataclasses.replace(load_artifacts(fixture_model_dir), threshold=14.0)
    monkeypatch.setattr(er, "load_artifacts", lambda md=None: art)
    return man, art


def test_run_structure_and_protocol(setup):
    man, art = setup
    r = ae.run_alt_evaluation(man, seed=5, levels=LEVELS, reps=1, art=art, timestamp="T")
    assert r["status"] == "ok"
    p = r["protocol"]
    assert set(p["burden_set"]).isdisjoint(p["held_out"])  # nothing fitted on held-out flights
    assert r["thresholds"]["lstm_aggregate"] == 14.0
    base = r["detectors"]["lstm_aggregate"]
    assert "screen" not in base and "paired_vs_baseline_test" not in base
    for n, e in r["detectors"].items():
        if n != "lstm_aggregate":
            assert "screen" in e and "SYNTHETIC" in e["screen"]["scope"]
            assert e["paired_vs_baseline_test"]["net_gain"] == len(e["paired_vs_baseline_test"]["gained"]) - len(e["paired_vs_baseline_test"]["lost"])
    assert r["real_world"]["status"] == "no_reviewed_labels"
    assert "NOT demonstrated" in r["real_world"]["reason"]
    assert "Reproduce" in ae.render_alt_markdown(r)


def test_matched_burden_on_burden_set(setup):
    man, art = setup
    r = ae.run_alt_evaluation(man, seed=5, levels=LEVELS, reps=1, art=art, timestamp="T")
    tgt = r["burden_set"]["target_flagged_fraction"]
    for n, e in r["detectors"].items():
        bf = e["burden_val_in_sample"]["flagged_fraction"]
        if bf is not None and e["window_samples"] == 30:
            assert abs(bf - tgt) <= 0.01 + 1e-9, n  # matched up to ties/resolution


def test_baseline_detection_equals_phase2_on_same_injections(setup):
    """Same seeds/placement => the baseline row must reproduce Phase 2's held-out result."""
    man, art = setup
    r3 = ae.run_alt_evaluation(man, seed=5, levels=LEVELS, reps=1, art=art, timestamp="T")
    r2 = er.run_evaluation(man, seed=5, levels=LEVELS, reps=1, timestamp="T")
    s2 = r2["synthetic"]["by_split"]["test"]["deployed"]["overall"]
    s3 = r3["detectors"]["lstm_aggregate"]["synthetic_test"]["overall"]
    for k in ("n_injections", "n_eligible", "n_evaluable", "n_detected", "n_not_eligible", "n_confounded_by_baseline"):
        assert s2[k] == s3[k], k


def test_deterministic(setup):
    man, art = setup
    a = ae.run_alt_evaluation(man, seed=5, levels=LEVELS, reps=1, art=art, timestamp="T")
    b = ae.run_alt_evaluation(man, seed=5, levels=LEVELS, reps=1, art=art, timestamp="T")
    assert json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)


def test_unverified_convention_excludes_rate_detectors(setup, monkeypatch):
    man, art = setup
    monkeypatch.setattr(ae, "verify_rate_convention", lambda *a, **k: {"verified": False, "reason": "forced", "rows": []})
    r = ae.run_alt_evaluation(man, seed=5, levels=LEVELS, reps=1, art=art, timestamp="T")
    assert "rate_consistency_w30" in r["excluded_detectors"] and "fused_all" in r["excluded_detectors"]
    assert "rate_consistency_w30" not in r["detectors"]


def test_insufficient_corpus(setup):
    man, art = setup
    m = json.loads(json.dumps(man))
    for f in m["flights"]:
        if f["split"] == "test":
            f["split"] = "val"
    r = ae.run_alt_evaluation(m, seed=5, levels=LEVELS, reps=1, art=art)
    assert r["status"] == "insufficient_corpus"


def test_paired_and_status_logic():
    def rec(i, base, mx, elig=True):
        return {"injection_id": i, "eligible_for_scoring": elig, "baseline_max_overlap_score": base,
                "max_overlap_score": mx if elig else None}
    d = [rec("a", 0.1, 9), rec("b", 0.1, 1), rec("c", 0.1, 9), rec("n", 0.1, 0, elig=False), rec("x", 5.0, 9)]
    b = [rec("a", 0.1, 1), rec("b", 0.1, 9), rec("c", 0.1, 9), rec("n", 0.1, 0, elig=False), rec("x", 0.1, 9)]
    p = ae._paired(d, 4.0, b, 4.0)
    assert p["gained"] == ["a"] and p["lost"] == ["b"] and p["both_detected"] == 1
    assert p["n_paired"] == 3  # 'x' confounded for D, 'n' ineligible: not compared
    assert ae._status(d[4], 4.0) == "confounded" and ae._status(d[3], 4.0) == "not_eligible"
