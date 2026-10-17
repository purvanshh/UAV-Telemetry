"""End-to-end evaluation run on a tiny corpus: structure, separation of evidence, determinism."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import src.evaluation_report as er
from src.artifact_audit import audit
from src.eval_corpus import build_evaluation_manifest
from src.evaluation_report import render_markdown, run_evaluation
from tests.conftest import _make_att, _make_baro, _make_imu


def _write_flight(d: Path, seed: int, minutes: float = 3.0):
    d.mkdir(parents=True, exist_ok=True)
    n, m = int(minutes * 60 * 25), int(minutes * 60 * 10)
    imu = _make_imu(n=n)
    imu["GyrX"] = np.random.default_rng(seed).normal(0, 0.1, n)
    imu.to_parquet(d / "IMU.parquet")
    _make_att(n=m).to_parquet(d / "ATT.parquet")
    b = _make_baro(n=m)
    b["Temp"] = b["Temp"] + seed  # make flights distinct
    b.to_parquet(d / "BARO.parquet")
    return d


@pytest.fixture()
def corpus(tmp_path, fixture_model_dir, monkeypatch):
    srcs = []
    for i, kind in enumerate(["real_unreviewed", "real_unreviewed", "synthetic_generated"]):
        d = _write_flight(tmp_path / f"f{i}", seed=i + 1)
        srcs.append({"path": d, "origin": "tmp", "kind": kind, "eval_role": "synthetic_source",
                     "log_id": None, "store_id": f"f{i}", "file_sha256": None})
    # exact content duplicate of f0
    d = _write_flight(tmp_path / "dup", seed=1)
    srcs.append({"path": d, "origin": "tmp", "kind": "real_unreviewed", "eval_role": "synthetic_source",
                 "log_id": None, "store_id": "dup", "file_sha256": None})
    man = build_evaluation_manifest(srcs, seed=7)
    # raise threshold of the random-weight fixture model so clean placements exist
    real_load = er.load_artifacts
    monkeypatch.setattr(er, "load_artifacts",
                        lambda md=None: dataclasses.replace(real_load(fixture_model_dir), threshold=14.0))
    return man


FAST_LEVELS = {t: lv[:1] for t, lv in er.DEFAULT_LEVELS.items()}


def test_manifest_excludes_duplicates_and_validates(corpus):
    assert corpus["validation"]["ok"]
    dups = [f for f in corpus["flights"] if f["split"] == "excluded"]
    assert [f["flight_id"] for f in dups] == ["f0"] or [f["flight_id"] for f in dups] == ["dup"]
    assert all(f["duplicate_of"] for f in dups)
    assert all(f["calibration_eligible"] is False for f in corpus["flights"])


def test_run_structure_and_evidence_separation(corpus, tmp_path):
    rep = run_evaluation(corpus, seed=11, out_dir=tmp_path / "out", levels=FAST_LEVELS, reps=1, timestamp="T")
    assert rep["counts"]["flights_evaluated"] == 3
    assert rep["split_manifest"]["validation"]["ok"]
    # nothing reviewed -> real-world metrics absent, not zero
    assert rep["real_world"]["status"] == "no_reviewed_labels"
    for k in ("event_recall", "false_alarms_per_flight_hour", "precision", "specificity"):
        assert k not in rep["real_world"]
    assert rep["synthetic"]["scope"].startswith("SYNTHETIC")
    assert rep["synthetic"]["n_injections"] > 0
    assert "NOT real-world" in render_markdown(rep) or "not real-world" in render_markdown(rep)
    tc = rep["threshold_comparison"]
    assert tc["selection_performed"] is False and tc["baseline"] == "deployed"
    assert tc["validation_quantile_candidates"]["status"] == "not_computed"
    assert tc["thresholds"]["deployed"]["threshold"] == 14.0
    assert any("No human-reviewed" in x for x in rep["limitations"])
    inj = json.loads((tmp_path / "out" / "synthetic" / "injections.json").read_text())
    assert len(inj) == rep["synthetic"]["n_injections"]
    assert all("source_flight" in r and "seed" in r and "injected_raw_interval_s" in r for r in inj)
    # duplicates are listed as excluded, not evaluated
    assert any(e.get("duplicate_of") for e in rep["data_quality_exclusions"])


def test_run_is_deterministic(corpus, tmp_path):
    a = run_evaluation(corpus, seed=11, levels=FAST_LEVELS, reps=1, timestamp="T")
    b = run_evaluation(corpus, seed=11, levels=FAST_LEVELS, reps=1, timestamp="T")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = run_evaluation(corpus, seed=12, levels=FAST_LEVELS, reps=1, timestamp="T")
    assert json.dumps(a["synthetic"], sort_keys=True) != json.dumps(c["synthetic"], sort_keys=True)


def test_labels_activate_real_world_section(corpus):
    real = [f["flight_id"] for f in corpus["flights"] if f["kind"] == "real_unreviewed" and f["split"] != "excluded"][0]
    labels = {"schema": "reviewed_intervals/1", "reviewer": "tester", "reviewed_at": "2026-01-01",
              "intervals": [{"flight_id": real, "t_start_s": 10.0, "t_end_s": 100.0, "label": "expected"}]}
    rep = run_evaluation(corpus, seed=11, reviewed=labels, levels=FAST_LEVELS, reps=1, timestamp="T")
    assert rep["real_world"]["status"] == "ok"
    assert rep["real_world"]["expected_hours_reviewed"] == pytest.approx(90 / 3600)
    assert rep["real_world"]["event_recall"] is None  # no unusual intervals reviewed -> no recall
    assert rep["synthetic"]["scope"].startswith("SYNTHETIC")  # still separate


def test_invalid_manifest_is_refused(corpus):
    bad = json.loads(json.dumps(corpus))
    bad["flights"].append(dict(bad["flights"][0]))
    with pytest.raises(ValueError):
        run_evaluation(bad, seed=1, levels=FAST_LEVELS, reps=1)


def test_artifact_audit_on_production_artifacts():
    from src.artifact_loader import missing_artifacts

    if missing_artifacts():
        pytest.skip("production artifacts absent")
    a = audit()
    assert a["threshold"]["value"] == pytest.approx(4.034964084625244)
    assert a["threshold"]["flag_operator_in_deployed_code"].startswith("score > threshold")
    assert a["scaler"]["equals_schema_train_split"] is True
    assert a["recorded_split"]["source_flights_in_repository"] is False
    assert a["calibration_code_in_repository"] == []
    cl = a["classification"]
    assert cl["missing_cannot_be_recovered"] and cl["recorded_but_unverified_claims"]


def test_resolve_flight_rejects_data_that_differ_from_manifest(corpus, tmp_path):
    from src.eval_corpus import resolve_flight

    f = next(x for x in corpus["flights"] if x["split"] != "excluded")
    assert resolve_flight(f)  # matches
    tampered = dict(f, content_hash="0" * 64)
    with pytest.raises(ValueError, match="do not match"):
        resolve_flight(tampered)
