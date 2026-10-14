"""FastAPI integration tests — upload → analyze → report."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.store import FlightStore
from tests.conftest import _make_att, _make_baro, _make_imu


@pytest.fixture()
def client(tmp_path, monkeypatch, fixture_model_dir):
    monkeypatch.setenv("UAV_MODEL_DIR", str(fixture_model_dir))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "flights"))
    # Clear cached store / artifacts
    from backend import deps
    from src import inference_service

    deps.get_store.cache_clear()
    inference_service.clear_caches()
    return TestClient(app)


def _csv_bytes(df) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_access_log_hides_opencode_vllm_probes():
    from backend.main import QuietLocalModelProbes

    filt = QuietLocalModelProbes()

    class _Rec:
        def __init__(self, message: str) -> None:
            self._message = message

        def getMessage(self) -> str:
            return self._message

    assert filt.filter(_Rec('127.0.0.1:1 - "GET /health HTTP/1.1" 404')) is False
    assert filt.filter(_Rec('127.0.0.1:1 - "GET /v1/models HTTP/1.1" 404')) is False
    assert filt.filter(_Rec('127.0.0.1:1 - "GET /api/health HTTP/1.1" 200')) is True


def test_artifacts_status(client, fixture_model_dir):
    r = client.get("/api/artifacts/status")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["n_features"] == 12
    assert Path(body["model_dir"]) == Path(fixture_model_dir)


def test_upload_analyze_report_flow(client):
    imu = _make_imu(n=500)
    att = _make_att(n=200)
    baro = _make_baro(n=200)

    upload = client.post(
        "/api/flights/upload",
        files={
            "imu": ("IMU.csv", _csv_bytes(imu), "text/csv"),
            "att": ("ATT.csv", _csv_bytes(att), "text/csv"),
            "baro": ("BARO.csv", _csv_bytes(baro), "text/csv"),
        },
        data={"flight_id": "api-test-flight"},
    )
    assert upload.status_code == 200, upload.text
    body = upload.json()
    assert body["ok"] is True
    fid = body["flight_id"]
    assert fid == "api-test-flight"

    detail = client.get(f"/api/flights/{fid}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "validated"

    analyzed = client.post(f"/api/flights/{fid}/analyze", json={"max_gap_s": 1.5})
    assert analyzed.status_code == 200, analyzed.text
    a = analyzed.json()
    assert a["ok"] is True
    assert a["n_windows"] > 0
    assert a["threshold"] > 0

    report = client.get(f"/api/flights/{fid}/report")
    assert report.status_code == 200
    assert report.json()["flight_id"] == fid
    assert report.json()["summary"]["total_windows"] == a["n_windows"]

    chart = client.get(f"/api/flights/{fid}/charts/score_timeline")
    assert chart.status_code == 200
    assert "data" in chart.json()

    export = client.get(f"/api/flights/{fid}/exports/json")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("application/json")

    trace = client.get(f"/api/flights/{fid}/evidence-trace")
    assert trace.status_code == 200
    assert "formatted" in trace.json()


def test_upload_validation_error(client):
    r = client.post(
        "/api/flights/upload",
        files={
            "imu": ("IMU.csv", b"not,a,valid\n1,2,3\n", "text/csv"),
            "att": ("ATT.csv", b"x\n1\n", "text/csv"),
            "baro": ("BARO.csv", b"y\n1\n", "text/csv"),
        },
    )
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert r.json()["errors"]
