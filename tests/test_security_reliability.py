"""Regression tests for the Stage 1 security / reliability fixes."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.store import FlightStore, FlightStoreError, InvalidFlightIdError
from src import upload as upload_mod
from src.artifact_loader import ArtifactError, ensure_checkpoint, load_artifacts
from src.detector import DetectorError, UAVDetector
from src.limits import InvalidFlightId, validate_flight_id
from src.validation import validate_stream
from tests.conftest import _make_att, _make_baro, _make_imu


def _csv(df) -> bytes:
    b = io.BytesIO()
    df.to_csv(b, index=False)
    return b.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch, fixture_model_dir):
    monkeypatch.setenv("UAV_MODEL_DIR", str(fixture_model_dir))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "flights"))
    from backend import deps
    from src import inference_service

    deps.get_store.cache_clear()
    inference_service.clear_caches()
    yield TestClient(app)
    inference_service.clear_caches()
    deps.get_store.cache_clear()


def _frames():
    return {"IMU": _make_imu(500), "ATT": _make_att(200), "BARO": _make_baro(200)}


def _store_with_flight(tmp_path, fid="f1") -> FlightStore:
    s = FlightStore(tmp_path / "flights")
    s.create_from_upload(
        _frames(), source="individual", filenames={}, file_sizes={},
        warnings=[], validation={}, flight_id=fid,
    )
    return s


# ---- 1. flight id / path traversal -------------------------------------------------

@pytest.mark.parametrize(
    "bad", ["../evil", "..", "a/b", "/abs", ".hidden", "a\\b", "", " x", "x" * 65, "a..b/c", "x\x00y"]
)
def test_validate_flight_id_rejects_unsafe(bad):
    with pytest.raises(InvalidFlightId):
        validate_flight_id(bad)


def test_store_rejects_traversal_and_never_deletes_outside(tmp_path):
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "meta.json").write_text("{}")
    (outside / "keep.txt").write_text("x")
    s = FlightStore(tmp_path / "flights")
    for bad in ("../precious", "..", "/" + str(outside)):
        with pytest.raises(InvalidFlightIdError):
            s.delete(bad)
        with pytest.raises(FlightStoreError):
            s.create_from_upload(
                _frames(), source="individual", filenames={}, file_sizes={},
                warnings=[], validation={}, flight_id=bad,
            )
    assert (outside / "keep.txt").exists()


def test_store_rejects_symlinked_flight_dir(tmp_path):
    s = FlightStore(tmp_path / "flights")
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "meta.json").write_text("{}")
    (s.root / "link").symlink_to(target, target_is_directory=True)
    with pytest.raises(InvalidFlightIdError):
        s.delete("link")
    assert target.exists()


def test_api_upload_with_traversal_id_is_rejected(client):
    r = client.post(
        "/api/flights/upload",
        files={
            "imu": ("IMU.csv", _csv(_make_imu(500)), "text/csv"),
            "att": ("ATT.csv", _csv(_make_att(200)), "text/csv"),
            "baro": ("BARO.csv", _csv(_make_baro(200)), "text/csv"),
        },
        data={"flight_id": "../../escape"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert "Flight id" in r.json()["errors"][0]


# ---- 2. upload / zip limits --------------------------------------------------------

def test_oversized_individual_upload_rejected(monkeypatch):
    monkeypatch.setenv("UAV_MAX_UPLOAD_MB", "0.001")  # ~1 KB
    res = upload_mod.load_individual_csvs(
        {"IMU": _csv(_make_imu(500)), "ATT": _csv(_make_att(20)), "BARO": _csv(_make_baro(20))}
    )
    assert not res.ok
    assert any("upload limit" in e for e in res.errors)


def test_api_oversized_upload_returns_413(client, monkeypatch):
    monkeypatch.setenv("UAV_MAX_UPLOAD_MB", "0.001")
    r = client.post(
        "/api/flights/upload",
        files={
            "imu": ("IMU.csv", _csv(_make_imu(500)), "text/csv"),
            "att": ("ATT.csv", _csv(_make_att(20)), "text/csv"),
            "baro": ("BARO.csv", _csv(_make_baro(20)), "text/csv"),
        },
    )
    assert r.status_code == 413


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for n, b in members.items():
            zf.writestr(n, b)
    return buf.getvalue()


def test_zip_bomb_rejected_by_ratio(monkeypatch):
    bomb = _zip({"IMU.csv": b"0" * 5_000_000, "ATT.csv": b"a", "BARO.csv": b"b"})
    res = upload_mod.load_zip_archive(bomb)
    assert not res.ok
    assert any("compression ratio" in e or "expand" in e for e in res.errors)


def test_zip_member_cap_enforced_even_if_ratio_ok(monkeypatch):
    monkeypatch.setenv("UAV_MAX_UPLOAD_MB", "0.01")  # ~10 KB per member
    rng = np.random.default_rng(1).bytes(50_000)  # incompressible
    res = upload_mod.load_zip_archive(_zip({"IMU.csv": rng, "ATT.csv": b"a", "BARO.csv": b"b"}))
    assert not res.ok
    assert any("limit" in e for e in res.errors)


def test_zip_too_many_members_rejected():
    members = {f"x{i}.csv": b"a,b\n1,2\n" for i in range(250)}
    res = upload_mod.load_zip_archive(_zip(members))
    assert not res.ok
    assert any("limit is" in e for e in res.errors)


def test_zip_read_is_capped_when_header_lies(monkeypatch):
    monkeypatch.setenv("UAV_MAX_UPLOAD_MB", "0.01")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("big.csv", np.random.default_rng(2).bytes(50_000))
    zf = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    with pytest.raises(Exception):
        upload_mod.read_zip_member(zf, "big.csv")


# ---- 3. atomic, generation-based analysis storage ---------------------------------

def _analysis_kwargs(tag: str):
    return dict(
        summary={"tag": tag}, windows=pd.DataFrame({"a": [1, 2]}), events=[],
        events_df=None, aligned=pd.DataFrame({"x": [1.0]}), report=None, status="analyzed",
    )


def test_failed_write_leaves_previous_generation_intact(tmp_path, monkeypatch):
    s = _store_with_flight(tmp_path)
    s.save_analysis("f1", **_analysis_kwargs("first"))
    before = s.load_analysis_summary("f1")
    meta_before = s.load_meta("f1")
    assert before["tag"] == "first"

    real = pd.DataFrame.to_parquet
    calls = {"n": 0}

    def flaky(self, path, *a, **k):
        calls["n"] += 1
        if "aligned" in str(path):
            raise OSError("disk full")
        return real(self, path, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", flaky)
    with pytest.raises(OSError):
        s.save_analysis("f1", **_analysis_kwargs("second"))
    monkeypatch.undo()

    assert s.load_analysis_summary("f1")["tag"] == "first"
    assert s.load_windows("f1") is not None
    assert [p.name for p in s.flight_dir("f1").iterdir() if p.name.startswith("gen-")] == [
        s._adir(s.flight_dir("f1")).name
    ]
    assert s.load_meta("f1")["analyzed_at"] == meta_before["analyzed_at"]  # not advanced


def test_new_generation_replaces_old_and_clears_ai(tmp_path):
    s = _store_with_flight(tmp_path)
    s.save_analysis("f1", **_analysis_kwargs("one"))
    s.save_ai_diagnostic("f1", {"x": 1})
    assert s.load_ai_diagnostic("f1") == {"x": 1}
    s.save_analysis("f1", **_analysis_kwargs("two"))
    assert s.load_analysis_summary("f1")["tag"] == "two"
    assert s.load_ai_diagnostic("f1") is None
    assert len([p for p in s.flight_dir("f1").iterdir() if p.name.startswith("gen-")]) == 1


def test_legacy_flat_layout_still_readable(tmp_path):
    s = _store_with_flight(tmp_path)
    d = s.flight_dir("f1")
    (d / "analysis_summary.json").write_text('{"tag": "legacy"}')
    assert s.load_analysis_summary("f1")["tag"] == "legacy"


def test_create_is_published_atomically(tmp_path, monkeypatch):
    s = FlightStore(tmp_path / "flights")
    real = pd.DataFrame.to_parquet

    def boom(self, path, *a, **k):
        if "ATT" in str(path):
            raise OSError("boom")
        return real(self, path, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        s.create_from_upload(_frames(), source="x", filenames={}, file_sizes={},
                             warnings=[], validation={}, flight_id="partial")
    monkeypatch.undo()
    assert not s.exists("partial")
    assert not (s.root / "partial").exists()
    assert not [p for p in s.root.iterdir() if p.name.startswith(".tmp-")]


def test_concurrent_saves_never_mix(tmp_path):
    import threading

    s = _store_with_flight(tmp_path)
    errs = []

    def run(tag):
        try:
            for _ in range(5):
                s.save_analysis("f1", **_analysis_kwargs(tag))
        except Exception as e:  # pragma: no cover
            errs.append(e)

    ts = [threading.Thread(target=run, args=(t,)) for t in ("a", "b", "c")]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    assert s.load_analysis_summary("f1")["tag"] in {"a", "b", "c"}
    assert len([p for p in s.flight_dir("f1").iterdir() if p.name.startswith("gen-")]) == 1


# ---- 4. non-finite scores ----------------------------------------------------------

def test_non_finite_scores_raise_instead_of_reading_as_normal(fixture_model_dir, monkeypatch):
    det = UAVDetector(load_artifacts(fixture_model_dir))
    X = np.zeros((3, 30, 12), dtype=np.float32)
    monkeypatch.setattr(
        UAVDetector, "recon_error_per_window",
        lambda self, X, batch_size=512: np.array([0.1, np.nan, np.inf], dtype=np.float32),
    )
    with pytest.raises(DetectorError, match="Non-finite"):
        det.score(X)


def test_non_finite_input_windows_rejected(fixture_model_dir):
    det = UAVDetector(load_artifacts(fixture_model_dir))
    X = np.zeros((1, 30, 12), dtype=np.float32)
    X[0, 3, 2] = np.nan
    with pytest.raises(DetectorError):
        det.score(X)


def test_analysis_reports_error_not_no_anomaly_on_invalid_scores(
    fixture_model_dir, long_flight_frames, monkeypatch
):
    from src import inference_service as inf

    inf.clear_caches()
    monkeypatch.setattr(
        UAVDetector, "recon_error_per_window",
        lambda self, X, batch_size=512: np.full(len(X), np.nan, dtype=np.float32),
    )
    try:
        res = inf.analyze_flight(long_flight_frames, flight_id="t", model_dir=str(fixture_model_dir))
    finally:
        inf.clear_caches()
    assert not res.ok and res.status == "error"
    assert res.n_flagged == 0 and "Inference failed" in res.errors[0]


# ---- 5. safe checkpoint loading / no artifact mutation -----------------------------

def test_pickle_checkpoint_not_executed(tmp_path, fixture_model_dir, monkeypatch):
    import shutil
    import torch

    dest = tmp_path / "m"
    shutil.copytree(fixture_model_dir, dest)
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (Path(str(marker)).write_text, ("x",))

    torch.save({"state_dict": Evil(), "n_features": 12}, dest / "lstm_autoencoder.pth")
    with pytest.raises(DetectorError):
        UAVDetector(load_artifacts(dest))
    assert not marker.exists()


def test_inference_never_rebuilds_or_modifies_checkpoint(tmp_path, fixture_model_dir):
    import shutil

    dest = tmp_path / "m"
    shutil.copytree(fixture_model_dir, dest)
    pth = dest / "lstm_autoencoder.pth"
    pth.write_bytes(b"not a zip " * 20)
    digest = hashlib.sha256(pth.read_bytes()).hexdigest()
    with pytest.raises(ArtifactError, match="not a valid Torch checkpoint"):
        load_artifacts(dest)
    with pytest.raises(ArtifactError):
        ensure_checkpoint(dest)
    assert hashlib.sha256(pth.read_bytes()).hexdigest() == digest


def test_deployed_artifacts_unchanged():
    """Pins the deployed model/scaler/threshold so UI/ingestion work cannot drift them."""
    root = Path(__file__).resolve().parents[1] / "uav_model"
    arts = load_artifacts(root)
    assert arts.threshold == 4.034964084625244
    assert arts.features == [
        "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z",
        "roll", "pitch", "yaw_sin", "yaw_cos", "roll_res", "pitch_res",
    ]
    assert (arts.win_samples, arts.stride_samples) == (30, 15)


# ---- 6. strict numeric validation --------------------------------------------------

def test_non_numeric_required_telemetry_rejected():
    imu = _make_imu(100).astype({"GyrX": object})
    imu.loc[:50, "GyrX"] = "N/A"
    res = validate_stream(imu, "IMU")
    assert not res.ok
    assert any("GyrX" in e and "non-numeric" in e for e in res.errors)


def test_all_text_required_column_rejected():
    imu = _make_imu(50).astype({"AccZ": object})
    imu["AccZ"] = "bad"
    res = validate_stream(imu, "IMU")
    assert not res.ok and any("no usable numeric" in e for e in res.errors)


def test_few_bad_cells_warn_but_pass():
    imu = _make_imu(500).astype({"GyrX": object})
    imu.loc[3, "GyrX"] = "oops"
    res = validate_stream(imu, "IMU")
    assert res.ok
    assert any("GyrX" in w and "non-numeric" in w for w in res.warnings)


def test_infinite_values_in_required_column_flagged():
    imu = _make_imu(100)
    imu.loc[:20, "GyrY"] = np.inf
    res = validate_stream(imu, "IMU")
    assert not res.ok
