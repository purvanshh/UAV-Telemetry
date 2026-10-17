"""Flight-level splits and leakage detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.split_manifest import (
    deterministic_assignment, historical_manifest, split_flight_ids, validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _f(fid, split, content=None, exact=None, streams=None):
    return {"flight_id": fid, "split": split, "content_hash": content, "exact_hash": exact,
            "stream_content_sha256": streams or {}}


def test_clean_manifest_ok():
    m = {"flights": [_f("a", "train", "h1"), _f("b", "val", "h2"), _f("c", "test", "h3")]}
    v = validate_manifest(m)
    assert v["ok"] and v["counts"] == {"train": 1, "val": 1, "test": 1, "excluded": 0}


def test_flight_id_in_two_splits_is_error():
    v = validate_manifest({"flights": [_f("a", "train"), _f("a", "test")]})
    assert not v["ok"] and any("more than once" in e for e in v["errors"])


def test_exact_duplicate_across_splits_is_leakage():
    v = validate_manifest({"flights": [_f("a", "train", "h", "x"), _f("b", "test", "h2", "x")]})
    assert not v["ok"] and any("byte-identical" in e for e in v["errors"])


def test_content_duplicate_across_splits_is_leakage():
    v = validate_manifest({"flights": [_f("a", "val", "same"), _f("b", "test", "same")]})
    assert not v["ok"] and any("identical-content" in e for e in v["errors"])


def test_excluded_duplicate_does_not_leak():
    v = validate_manifest({"flights": [_f("a", "val", "same"), _f("b", "excluded", "same")]})
    assert v["ok"]


def test_duplicate_inside_one_split_warns():
    v = validate_manifest({"flights": [_f("a", "val", "same"), _f("b", "val", "same")]})
    assert v["ok"] and v["warnings"]


def test_near_duplicate_streams_across_splits():
    a = _f("a", "train", "c1", streams={"IMU": "s"})
    b = _f("b", "test", "c2", streams={"IMU": "s"})
    assert validate_manifest({"flights": [a, b]})["ok"]  # warning by default
    assert validate_manifest({"flights": [a, b]})["warnings"]
    assert not validate_manifest({"flights": [a, b]}, strict=True)["ok"]


def test_invalid_split_name():
    assert not validate_manifest({"flights": [_f("a", "dev")]})["ok"]


def test_assignment_deterministic_and_order_independent():
    flights = [_f(f"f{i}", None, f"h{i}") for i in range(20)]
    fr = {"train": 0.6, "val": 0.2, "test": 0.2}
    a = deterministic_assignment(flights, fr, seed=1)
    b = deterministic_assignment(list(reversed(flights)), fr, seed=1)
    assert a == b
    assert deterministic_assignment(flights, fr, seed=2) != a
    counts = {s: list(a.values()).count(s) for s in fr}
    assert counts == {"train": 12, "val": 4, "test": 4}


def test_assignment_keeps_identical_content_together():
    flights = [_f("a", None, "same"), _f("b", None, "same")] + [_f(f"x{i}", None, f"h{i}") for i in range(10)]
    a = deterministic_assignment(flights, {"train": 0.5, "test": 0.5}, seed=3)
    assert a["a"] == a["b"]


def test_small_corpus_populates_each_split():
    flights = [_f(f"f{i}", None, f"h{i}") for i in range(3)]
    a = deterministic_assignment(flights, {"val": 0.4, "test": 0.6}, seed=5)
    assert set(a.values()) == {"val", "test"}


def test_historical_manifest_matches_schema_and_is_unverified():
    fs = json.loads((ROOT / "uav_model" / "feature_schema.json").read_text())
    m = historical_manifest(fs)
    assert m["kind"] == "historical_unverified"
    assert len(split_flight_ids(m, "train")) == 71
    assert len(split_flight_ids(m, "val")) == 15 and len(split_flight_ids(m, "test")) == 15
    assert all(f["data_available"] is False and f["calibration_eligible"] is False for f in m["flights"])
    assert validate_manifest(m)["ok"]


def test_committed_manifests_are_valid():
    for name in ("historical_split.json", "evaluation_split.json"):
        p = ROOT / "evaluation" / "manifests" / name
        assert p.is_file(), name
        assert validate_manifest(json.loads(p.read_text()))["ok"]
