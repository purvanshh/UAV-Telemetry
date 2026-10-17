"""Duplicate-upload detection: exact, repackaged, partial and merely-similar flights."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.flight_identity import (
    CONTENT, DISTINCT, EXACT, PARTIAL, SIMILAR, canonical_stream_hash,
    classify, compute_identity, duplicate_message,
)
from src.upload import load_individual_csvs, load_zip_archive
from tests.conftest import _make_att, _make_baro, _make_imu


def _csv(df, **kw) -> bytes:
    b = io.BytesIO()
    df.to_csv(b, index=False, **kw)
    return b.getvalue()


def _flight(seed_shift=0.0, n=500):
    imu = _make_imu(n=n)
    imu["GyrX"] = imu["GyrX"] + seed_shift
    return {"IMU": imu, "ATT": _make_att(n=200), "BARO": _make_baro(n=200)}


def _identity(frames):
    csvs = {s: _csv(df) for s, df in frames.items()}
    res = load_individual_csvs(csvs, filenames={s: f"{s}.csv" for s in csvs})
    assert res.ok, res.all_errors
    return compute_identity(res.frames, res.file_sha256)


def test_hashes_independent_of_packaging():
    frames = _flight()
    csvs = {s: _csv(df) for s, df in frames.items()}
    ind = load_individual_csvs(csvs, filenames={"IMU": "a_IMU.csv", "ATT": "a_ATT.csv", "BARO": "a_BARO.csv"})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for s, raw in csvs.items():
            zf.writestr(f"{s.lower()}.csv", raw)
    zp = load_zip_archive(buf.getvalue())
    assert ind.file_sha256 == zp.file_sha256
    assert classify(compute_identity(ind.frames, ind.file_sha256),
                    compute_identity(zp.frames, zp.file_sha256)) == EXACT


def test_exact_duplicate():
    a = _identity(_flight())
    assert classify(_identity(_flight()), a) == EXACT


def test_content_duplicate_different_bytes():
    frames = _flight()
    a = _identity(frames)
    # different column order, CRLF line endings, shuffled rows -> different bytes, same data
    alt = {}
    for s, df in frames.items():
        d = df[list(reversed(df.columns))].sample(frac=1.0, random_state=1)
        alt[s] = d
    csvs = {s: _csv(d, lineterminator="\r\n") for s, d in alt.items()}
    res = load_individual_csvs(csvs, filenames={s: f"{s}.csv" for s in csvs})
    b = compute_identity(res.frames, res.file_sha256)
    assert b["exact_hash"] != a["exact_hash"]
    assert classify(b, a) == CONTENT
    assert "same content" in duplicate_message([{"flight_id": "f1", "kind": CONTENT}])


def test_distinct_flights_with_similar_metadata_are_not_duplicates():
    a = _identity(_flight())
    other = _flight(seed_shift=0.01)  # same timing, row counts and columns; different data
    b = _identity(other)
    kind = classify(b, a)
    assert kind in (PARTIAL, SIMILAR)  # ATT/BARO identical -> partial; never a blocking duplicate
    assert duplicate_message([{"flight_id": "x", "kind": kind}]) is None
    fully = {s: df.copy() for s, df in other.items()}
    for s in fully:
        fully[s].iloc[:, 1] = fully[s].iloc[:, 1] + 0.5
    assert classify(_identity(fully), a) in (SIMILAR, DISTINCT)


def test_canonical_hash_sensitive_to_single_value():
    df = _make_imu(n=100)
    h = canonical_stream_hash(df)
    df2 = df.copy()
    df2.loc[50, "GyrX"] += 1e-9
    assert canonical_stream_hash(df2) != h


@pytest.fixture()
def client(tmp_path, monkeypatch, fixture_model_dir):
    monkeypatch.setenv("UAV_MODEL_DIR", str(fixture_model_dir))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "flights"))
    from backend import deps
    from src import inference_service

    deps.get_store.cache_clear()
    inference_service.clear_caches()
    return TestClient(app)


def _post(client, frames, **data):
    return client.post(
        "/api/flights/upload",
        files={
            "imu": ("IMU.csv", _csv(frames["IMU"]), "text/csv"),
            "att": ("ATT.csv", _csv(frames["ATT"]), "text/csv"),
            "baro": ("BARO.csv", _csv(frames["BARO"]), "text/csv"),
        },
        data=data,
    ).json()


def test_api_rejects_exact_duplicate_then_allows_on_request(client):
    first = _post(client, _flight(), flight_id="f1")
    assert first["ok"]
    dup = _post(client, _flight())
    assert dup["ok"] is False
    assert dup["duplicate_of"] == "f1" and dup["duplicate_kind"] == EXACT
    assert "Duplicate upload" in dup["errors"][0]
    assert len(client.get("/api/flights").json()) == 1  # nothing stored

    forced = _post(client, _flight(), allow_duplicate="true")
    assert forced["ok"] and forced["duplicate_of"] == "f1"
    detail = client.get(f"/api/flights/{forced['flight_id']}").json()
    assert detail["duplicate_of"] == "f1" and detail["file_sha256"]["IMU"]


def test_api_stores_distinct_flight_with_similar_metadata(client):
    assert _post(client, _flight(), flight_id="f1")["ok"]
    other = _post(client, _flight(seed_shift=0.01), flight_id="f2")
    assert other["ok"] and other["duplicate_of"] is None


def test_api_refuses_to_overwrite_existing_flight_id(client):
    assert _post(client, _flight(), flight_id="same-id")["ok"]
    r = _post(client, _flight(seed_shift=0.02), flight_id="same-id")
    assert r["ok"] is False and "already in use" in r["errors"][0]


def test_legacy_flight_without_identity_is_backfilled(client, tmp_path):
    import json, os
    from backend import deps

    assert _post(client, _flight(), flight_id="old")["ok"]
    meta_path = tmp_path / "flights" / "old" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("identity"); meta.pop("file_sha256")
    meta_path.write_text(json.dumps(meta))
    dup = _post(client, _flight())
    assert dup["ok"] is False and dup["duplicate_kind"] == CONTENT  # bytes unknown -> content match
