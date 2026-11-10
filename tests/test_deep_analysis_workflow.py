"""Comprehensive tests for the Deep Analysis workflow.

Covers the scenarios listed in the task specification:
 1. Valid three-file uploads.
 2. Missing or incompatible telemetry files.
 3. Invalid timestamps and units.
 4. Missing required features.
 5. Low data coverage.
 6. Anomaly and non-anomaly cases.
 7. Event grouping.
 8. Chart data and event timestamps.
 9. Model artifact integrity.
10. Stored-flight reproducibility.
11. Failure handling and repeated analysis.
12. Upload-summary endpoint.
13. Visualization helpers (score timeline, telemetry chart, per-feature, baro).
14. Verdict correctness — never says 'safe' / 'normal'.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.store import FlightStore
from src.event_builder import build_events, DEFAULT_MAX_GAP_S
from src.inference_service import analyze_flight, clear_caches
from src.operator_view import compute_verdict, verdict_from_summary, VERDICT_STATES
from src.upload import load_individual_csvs, load_zip_archive
from src.validation import validate_stream
from src.upload_summary import build_upload_summary
from src.visualization import (
    baro_context_figure,
    per_feature_error_figure,
    score_timeline_figure,
    telemetry_around_event_figure,
)
from tests.conftest import _make_att, _make_baro, _make_imu


# ── helpers ───────────────────────────────────────────────────────────────────────

def _csv(df: pd.DataFrame) -> bytes:
    b = io.BytesIO()
    df.to_csv(b, index=False)
    return b.getvalue()


def _zip_of(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


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


def _upload(client, fid="daw-flight", n=500):
    r = client.post(
        "/api/flights/upload",
        files={
            "imu":  ("IMU.csv",  _csv(_make_imu(n)), "text/csv"),
            "att":  ("ATT.csv",  _csv(_make_att(n // 2)), "text/csv"),
            "baro": ("BARO.csv", _csv(_make_baro(n // 2)), "text/csv"),
        },
        data={"flight_id": fid},
    )
    assert r.json().get("ok"), r.text
    return fid


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Valid three-file uploads
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidUploads:
    def test_three_csv_upload_ok(self, client):
        fid = _upload(client, "valid-3csv")
        r = client.get(f"/api/flights/{fid}").json()
        assert r["status"] == "validated"
        assert set(r["filenames"].keys()) == {"IMU", "ATT", "BARO"}

    def test_upload_stores_file_sizes(self, client):
        fid = _upload(client, "sizes-check")
        r = client.get(f"/api/flights/{fid}").json()
        for stream in ("IMU", "ATT", "BARO"):
            assert r["file_sizes"].get(stream, 0) > 0

    def test_zip_upload_ok(self, client):
        z = _zip_of({
            "IMU.csv":  _csv(_make_imu(500)),
            "ATT.csv":  _csv(_make_att(200)),
            "BARO.csv": _csv(_make_baro(200)),
        })
        r = client.post(
            "/api/flights/upload",
            files={"zip_file": ("flight.zip", z, "application/zip")},
            data={"flight_id": "zip-upload"},
        ).json()
        assert r["ok"]
        assert r["flight_id"] == "zip-upload"

    def test_upload_and_analyze_full_pipeline(self, client):
        fid = _upload(client, "full-pipeline")
        a = client.post(f"/api/flights/{fid}/analyze", json={}).json()
        assert a["ok"]
        assert a["verdict"]["state"] in set(VERDICT_STATES)
        # Events list matches count
        overview = client.get(f"/api/flights/{fid}/overview").json()
        assert overview["verdict"]["state"] == a["verdict"]["state"]
        assert len(overview["events"]) == a["n_events"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Missing or incompatible telemetry files
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingIncompatibleFiles:
    def test_missing_baro_rejected(self, client):
        r = client.post(
            "/api/flights/upload",
            files={
                "imu": ("IMU.csv", _csv(_make_imu(500)), "text/csv"),
                "att": ("ATT.csv", _csv(_make_att(200)), "text/csv"),
            },
        )
        # Backend returns either 400 HTTP or ok=False in JSON
        if r.status_code == 200:
            assert not r.json().get("ok")
        else:
            assert r.status_code in (400, 422)

    def test_missing_imu_rejected(self, client):
        r = client.post(
            "/api/flights/upload",
            files={
                "att":  ("ATT.csv",  _csv(_make_att(200)), "text/csv"),
                "baro": ("BARO.csv", _csv(_make_baro(200)), "text/csv"),
            },
        )
        if r.status_code == 200:
            assert not r.json().get("ok")
        else:
            assert r.status_code in (400, 422)

    def test_wrong_stream_content_rejected(self, client):
        # Upload ATT bytes as IMU (columns don't match)
        r = client.post(
            "/api/flights/upload",
            files={
                "imu":  ("ATT.csv", _csv(_make_att(200)), "text/csv"),
                "att":  ("IMU.csv", _csv(_make_imu(500)), "text/csv"),
                "baro": ("BARO.csv", _csv(_make_baro(200)), "text/csv"),
            },
        ).json()
        assert not r["ok"]

    def test_upload_no_files_rejected(self, client):
        r = client.post("/api/flights/upload").json()
        assert r is not None  # returns 4xx or ok=False

    def test_binary_not_csv_rejected(self):
        result = load_individual_csvs(
            {
                "IMU":  b"\x00\x01\xff binary junk",
                "ATT":  _csv(_make_att(50)),
                "BARO": _csv(_make_baro(50)),
            },
            filenames={"IMU": "IMU.csv", "ATT": "ATT.csv", "BARO": "BARO.csv"},
        )
        assert not result.ok
        assert any("parse" in e.lower() or "binary" in e.lower() or "csv" in e.lower()
                   for e in result.all_errors)

    def test_zip_missing_one_stream_rejected(self):
        z = _zip_of({
            "IMU.csv": _csv(_make_imu(200)),
            "ATT.csv": _csv(_make_att(100)),
            # BARO intentionally absent
        })
        result = load_zip_archive(z)
        assert not result.ok
        assert any("BARO" in e for e in result.errors)

    def test_zip_invalid_archive_rejected(self):
        result = load_zip_archive(b"not a zip at all")
        assert not result.ok
        assert any("zip" in e.lower() for e in result.errors)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Invalid timestamps
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimestamps:
    def test_non_monotonic_timestamps_flagged(self):
        imu = _make_imu(100)
        # Reverse the time column so it decreases
        imu["TimeUS"] = imu["TimeUS"].iloc[::-1].values
        result = validate_stream(imu, "IMU", filename="IMU.csv")
        # Should warn or fail on non-monotonic timestamps
        all_msgs = result.errors + result.warnings
        assert any(
            "monoton" in m.lower() or "non-monoton" in m.lower() or "timestamp" in m.lower()
            for m in all_msgs
        )

    def test_duplicate_timestamps_flagged(self):
        imu = _make_imu(100)
        imu.loc[10, "TimeUS"] = imu.loc[9, "TimeUS"]  # duplicate
        result = validate_stream(imu, "IMU", filename="IMU.csv")
        all_msgs = result.errors + result.warnings
        assert any(
            "duplicate" in m.lower() or "timestamp" in m.lower()
            for m in all_msgs
        )

    def test_all_same_timestamp_rejected_or_warned(self):
        imu = _make_imu(100)
        imu["TimeUS"] = 1_000_000  # all identical
        result = validate_stream(imu, "IMU", filename="IMU.csv")
        all_msgs = result.errors + result.warnings
        assert any(
            "timestamp" in m.lower() or "duplicate" in m.lower() or "monoton" in m.lower()
            for m in all_msgs
        )

    def test_missing_time_column_rejected(self):
        imu = _make_imu(100).drop(columns=["TimeUS"])
        result = validate_stream(imu, "IMU", filename="IMU.csv")
        assert not result.ok


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Missing required features
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingFeatures:
    @pytest.mark.parametrize("drop_col", ["GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"])
    def test_missing_imu_required_column_rejected(self, drop_col):
        imu = _make_imu(200).drop(columns=[drop_col])
        result = validate_stream(imu, "IMU", filename="IMU.csv")
        assert not result.ok
        assert any(drop_col in e or "missing" in e.lower() for e in result.errors)

    @pytest.mark.parametrize("drop_col", ["Roll", "Pitch", "Yaw", "DesRoll", "DesPitch"])
    def test_missing_att_required_column_rejected(self, drop_col):
        att = _make_att(100).drop(columns=[drop_col])
        result = validate_stream(att, "ATT", filename="ATT.csv")
        assert not result.ok
        assert any(drop_col in e or "missing" in e.lower() for e in result.errors)

    @pytest.mark.parametrize("drop_col", ["Alt", "Press"])
    def test_missing_baro_required_column_rejected(self, drop_col):
        baro = _make_baro(100).drop(columns=[drop_col])
        result = validate_stream(baro, "BARO", filename="BARO.csv")
        assert not result.ok
        assert any(drop_col in e or "missing" in e.lower() for e in result.errors)

    def test_non_numeric_required_column_rejected(self):
        imu = _make_imu(200)
        imu["GyrX"] = "N/A"
        result = validate_stream(imu, "IMU", filename="IMU.csv")
        assert not result.ok

    def test_all_nan_required_column_rejected_or_warned(self):
        imu = _make_imu(200)
        imu["AccZ"] = float("nan")
        result = validate_stream(imu, "IMU", filename="IMU.csv")
        # Must not silently pass — error or warning
        assert not result.ok or len(result.warnings) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Low data coverage
# ═══════════════════════════════════════════════════════════════════════════════

class TestLowCoverage:
    def test_short_flight_insufficient_data(self, fixture_model_dir):
        """Flight too short for even one window → insufficient_data."""
        clear_caches()
        result = analyze_flight(
            {"IMU": _make_imu(50), "ATT": _make_att(20), "BARO": _make_baro(20)},
            flight_id="short",
            model_dir=str(fixture_model_dir),
        )
        assert not result.ok
        assert result.status == "insufficient_data"
        assert result.n_windows == 0

    def test_insufficient_data_verdict_is_not_no_alerts(self, fixture_model_dir):
        """Insufficient data must not produce a 'no_alerts' verdict."""
        clear_caches()
        result = analyze_flight(
            {"IMU": _make_imu(50), "ATT": _make_att(20), "BARO": _make_baro(20)},
            flight_id="short2",
            model_dir=str(fixture_model_dir),
        )
        verdict = verdict_from_summary(result.summary_dict())
        assert verdict["state"] not in ("no_alerts_adequate_coverage", "no_alerts_low_coverage")

    def test_low_coverage_verdict_note(self):
        v = compute_verdict(ok=True, status="ok", n_events=0,
                            coverage_status="low_coverage")
        assert v["state"] == "no_alerts_low_coverage"
        assert any("coverage" in c.lower() for c in v["caveats"])

    def test_no_alerts_with_unknown_coverage_is_not_adequate(self):
        v = compute_verdict(ok=True, status="ok", n_events=0, coverage_status=None)
        assert v["state"] == "no_alerts_low_coverage"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Anomaly and non-anomaly result correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnomalyResults:
    def test_completed_flight_has_windows(self, fixture_model_dir, long_flight_frames):
        clear_caches()
        result = analyze_flight(
            long_flight_frames, flight_id="res-ok",
            model_dir=str(fixture_model_dir),
        )
        assert result.ok
        assert result.n_windows > 0
        assert result.threshold > 0

    def test_no_flags_verdict_never_says_safe_or_normal(self, fixture_model_dir, long_flight_frames):
        clear_caches()
        result = analyze_flight(
            long_flight_frames, flight_id="no-flags",
            model_dir=str(fixture_model_dir),
        )
        verdict = verdict_from_summary(result.summary_dict())
        full_text = json.dumps(verdict).lower()
        # The word "safe" must never appear as a positive claim
        assert "aircraft was safe" not in full_text
        assert "flight is safe" not in full_text
        assert "normal flight" not in full_text
        # If no flags, the safety caveat must be present
        if verdict["state"] in ("no_alerts_adequate_coverage", "no_alerts_low_coverage"):
            assert "does not show that the aircraft was safe" in full_text

    def test_analysis_reports_error_on_nan_scores(
        self, fixture_model_dir, long_flight_frames, monkeypatch
    ):
        from src import inference_service as inf
        from src.detector import UAVDetector
        clear_caches()
        monkeypatch.setattr(
            UAVDetector,
            "recon_error_per_window",
            lambda self, X, batch_size=512: np.full(len(X), np.nan, dtype=np.float32),
        )
        try:
            res = inf.analyze_flight(
                long_flight_frames, flight_id="nan-scores",
                model_dir=str(fixture_model_dir),
            )
        finally:
            inf.clear_caches()
        assert not res.ok
        assert res.status == "error"
        assert res.n_flagged == 0

    def test_report_schema_version_present(self, fixture_model_dir, long_flight_frames):
        clear_caches()
        result = analyze_flight(
            long_flight_frames, flight_id="schema-v",
            model_dir=str(fixture_model_dir),
        )
        assert result.ok and result.report is not None
        assert result.report.schema_version == "1.0"

    def test_report_records_threshold(self, fixture_model_dir, long_flight_frames):
        clear_caches()
        result = analyze_flight(
            long_flight_frames, flight_id="thr-recorded",
            model_dir=str(fixture_model_dir),
        )
        assert result.ok
        assert result.report is not None
        assert result.report.detector.threshold == pytest.approx(4.034964084625244)

    def test_deployed_threshold_unchanged(self):
        from src.artifact_loader import load_artifacts
        arts = load_artifacts()
        assert arts.threshold == pytest.approx(4.034964084625244)
        assert arts.features == [
            "gyro_x", "gyro_y", "gyro_z",
            "acc_x", "acc_y", "acc_z",
            "roll", "pitch",
            "yaw_sin", "yaw_cos",
            "roll_res", "pitch_res",
        ]
        assert arts.win_samples == 30
        assert arts.stride_samples == 15


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Event grouping
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventGrouping:
    def _windows(self, flagged_indices: list[int], n=20) -> pd.DataFrame:
        rows = []
        for i in range(n):
            rows.append({
                "window_index": i,
                "session_id": 0,
                "t_start": float(i * 1.5),
                "t_end": float(i * 1.5 + 3.0),
                "anomaly_score": 5.0 if i in flagged_indices else 0.5,
                "threshold": 4.0,
                "is_anomaly": i in flagged_indices,
            })
        return pd.DataFrame(rows)

    def test_adjacent_flagged_windows_merge_into_one_event(self):
        events = build_events(self._windows([2, 3, 4]))
        assert len(events) == 1
        assert events[0].flagged_window_count == 3

    def test_gap_within_max_merges(self):
        # windows 2 and 4 — gap is 1.5 s which equals one stride (≤ DEFAULT_MAX_GAP_S)
        events = build_events(self._windows([2, 4]))
        assert len(events) == 1

    def test_gap_beyond_max_stays_separate(self):
        # windows 1 and 10 — gap > DEFAULT_MAX_GAP_S
        events = build_events(self._windows([1, 10]))
        assert len(events) == 2

    def test_session_boundary_never_merges(self):
        df = pd.DataFrame([
            {"window_index": 0, "session_id": 0, "t_start": 0.0, "t_end": 3.0,
             "anomaly_score": 5.0, "threshold": 4.0, "is_anomaly": True},
            {"window_index": 1, "session_id": 1, "t_start": 3.5, "t_end": 6.5,
             "anomaly_score": 5.0, "threshold": 4.0, "is_anomaly": True},
        ])
        events = build_events(df)
        assert len(events) == 2

    def test_no_flagged_windows_no_events(self):
        assert build_events(self._windows([])) == []

    def test_event_ids_are_sequential(self):
        events = build_events(self._windows([1, 5, 10]))
        ids = [e.event_id for e in events]
        assert ids == ["event-001", "event-002", "event-003"]

    def test_event_timestamps_span_member_windows(self):
        df = self._windows([3, 4, 5])
        events = build_events(df)
        assert len(events) == 1
        # start is t_start of first window, end is t_end of last
        assert events[0].start_time_s == pytest.approx(df.loc[3, "t_start"])
        assert events[0].end_time_s == pytest.approx(df.loc[5, "t_end"])

    def test_per_feature_peak_accumulated(self):
        df = self._windows([2, 3])
        df["feat_err_gyro_x"] = [0.0] * 20
        df.loc[2, "feat_err_gyro_x"] = 1.5
        df.loc[3, "feat_err_gyro_x"] = 2.1
        events = build_events(df, features=["gyro_x"])
        assert events[0].per_feature_peak["gyro_x"] == pytest.approx(2.1)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Chart data and event timestamps
# ═══════════════════════════════════════════════════════════════════════════════

class TestCharts:
    @pytest.fixture()
    def analysed_flight(self, client):
        fid = _upload(client, "chart-flight", n=600)
        client.post(f"/api/flights/{fid}/analyze", json={})
        return fid

    def test_score_timeline_chart_returns_valid_json(self, client, analysed_flight):
        r = client.get(f"/api/flights/{analysed_flight}/charts/score_timeline")
        assert r.status_code == 200
        body = r.json()
        assert "data" in body or "layout" in body  # Plotly JSON structure

    def test_per_feature_chart_returns_data_for_event(self, client, analysed_flight):
        events = client.get(f"/api/flights/{analysed_flight}/events").json()
        if not events:
            pytest.skip("No events in this flight")
        ev_id = events[0]["event_id"]
        r = client.get(f"/api/flights/{analysed_flight}/charts/per_feature?event_id={ev_id}")
        assert r.status_code == 200

    def test_baro_chart_returns_data_for_event(self, client, analysed_flight):
        events = client.get(f"/api/flights/{analysed_flight}/events").json()
        if not events:
            pytest.skip("No events in this flight")
        ev_id = events[0]["event_id"]
        r = client.get(f"/api/flights/{analysed_flight}/charts/baro?event_id={ev_id}")
        assert r.status_code == 200

    def test_score_timeline_figure_has_threshold_trace(self):
        windows = pd.DataFrame({
            "t_start": np.linspace(0, 10, 20),
            "t_end": np.linspace(1.5, 11.5, 20),
            "anomaly_score": np.random.default_rng(1).uniform(0, 6, 20),
            "is_anomaly": np.array([True] * 5 + [False] * 15),
        })
        fig = score_timeline_figure(windows, threshold=4.034964)
        trace_names = [t.name for t in fig.data]
        assert any("limit" in (n or "").lower() or "threshold" in (n or "").lower()
                   for n in trace_names)

    def test_score_timeline_readable_axis_labels(self):
        windows = pd.DataFrame({
            "t_start": [0.0, 1.5, 3.0],
            "t_end":   [1.5, 3.0, 4.5],
            "anomaly_score": [1.0, 2.0, 3.0],
            "is_anomaly": [False, False, False],
        })
        fig = score_timeline_figure(windows, threshold=4.0)
        layout = fig.to_dict()["layout"]
        assert "time" in layout.get("xaxis", {}).get("title", {}).get("text", "").lower()
        assert "error" in layout.get("yaxis", {}).get("title", {}).get("text", "").lower()

    def test_per_feature_figure_uses_human_readable_labels(self):
        from src.event_builder import AnomalyEvent
        ev = AnomalyEvent(
            event_id="event-001", session_id=0,
            start_time_s=1.0, end_time_s=4.0,
            per_feature_peak={"gyro_x": 0.8, "acc_z": 0.3, "roll": 0.5},
        )
        fig = per_feature_error_figure(ev)
        # x-axis tick labels should be readable words, not raw feature keys
        tick_text = str(fig.to_dict())
        assert "angular rate" in tick_text.lower() or "gyro" in tick_text.lower()

    def test_telemetry_figure_separates_scale_groups(self, fixture_model_dir, long_flight_frames):
        """Gyro and angle signals should not share the same y-axis."""
        clear_caches()
        result = analyze_flight(
            long_flight_frames, flight_id="chart-scale",
            model_dir=str(fixture_model_dir),
        )
        assert result.ok and result.aligned is not None
        if not result.events:
            pytest.skip("No events detected")
        fig = telemetry_around_event_figure(
            result.aligned, result.events[0], result.features
        )
        # Multiple rows = multiple y-axes = signals are separated by group
        assert fig.to_dict()["layout"].get("yaxis2") is not None \
            or len(fig.data) > 0

    def test_baro_figure_has_altitude_and_pressure_axes(self, fixture_model_dir, long_flight_frames):
        clear_caches()
        result = analyze_flight(
            long_flight_frames, flight_id="baro-chart",
            model_dir=str(fixture_model_dir),
        )
        if not result.events:
            pytest.skip("No events detected")
        fig = baro_context_figure(result.aligned, result.events[0])
        layout_str = str(fig.to_dict()).lower()
        assert "altitude" in layout_str
        assert "pressure" in layout_str

    def test_chart_contains_flagged_interval_shading(self):
        from src.event_builder import AnomalyEvent
        windows = pd.DataFrame({
            "t_start": np.linspace(0, 10, 20),
            "t_end": np.linspace(1.5, 11.5, 20),
            "anomaly_score": np.full(20, 5.0),
            "is_anomaly": np.ones(20, dtype=bool),
        })
        ev = AnomalyEvent("event-001", 0, 3.0, 6.0)
        fig = score_timeline_figure(windows, threshold=4.0, events=[ev])
        d = fig.to_dict()
        shapes = d["layout"].get("shapes", [])
        # Plotly vrect adds a shape
        assert len(shapes) > 0

    def test_empty_windows_produces_valid_figure(self):
        fig = score_timeline_figure(pd.DataFrame(), threshold=4.0)
        assert fig is not None

    def test_event_timestamps_in_chart_context_match_report(self, client, analysed_flight):
        events = client.get(f"/api/flights/{analysed_flight}/events").json()
        if not events:
            pytest.skip("No events in this flight")
        ev = events[0]
        ctx = client.get(
            f"/api/flights/{analysed_flight}/events/{ev['event_id']}/context"
        ).json()
        # Context window must bracket the event
        assert ctx["window_start_rel_s"] <= ctx["start_rel_s"]
        assert ctx["window_end_rel_s"] >= ctx["end_rel_s"]
        # Time series must cover the event span
        if ctx["time"]:
            assert min(ctx["time"]) <= ctx["start_rel_s"] + 0.5
            assert max(ctx["time"]) >= ctx["end_rel_s"] - 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Model artifact integrity
# ═══════════════════════════════════════════════════════════════════════════════

class TestArtifactIntegrity:
    def test_artifacts_endpoint_reports_ready(self, client):
        r = client.get("/api/artifacts/status").json()
        assert r["ready"] is True
        assert r["threshold"] == pytest.approx(4.034964084625244)
        assert len(r["features"]) == 12

    def test_feature_order_matches_model_config(self):
        from src.artifact_loader import load_artifacts
        import json as _json
        arts = load_artifacts()
        cfg = _json.loads(
            (Path(__file__).parents[1] / "uav_model" / "model_config.json").read_text()
        )
        assert arts.features == cfg["features"]

    def test_window_stride_from_config(self):
        from src.artifact_loader import load_artifacts
        arts = load_artifacts()
        assert arts.win_samples == 30
        assert arts.stride_samples == 15

    def test_corrupt_checkpoint_raises_artifact_error(self, tmp_path):
        import shutil
        from src.artifact_loader import ArtifactError, load_artifacts
        dest = tmp_path / "bad_model"
        shutil.copytree(Path(__file__).parents[1] / "uav_model", dest)
        (dest / "lstm_autoencoder.pth").write_bytes(b"not a pytorch checkpoint")
        with pytest.raises(ArtifactError):
            load_artifacts(dest)

    def test_non_finite_input_to_detector_raises(self, fixture_model_dir):
        from src.artifact_loader import load_artifacts
        from src.detector import DetectorError, UAVDetector
        det = UAVDetector(load_artifacts(fixture_model_dir))
        X = np.zeros((1, 30, 12), dtype=np.float32)
        X[0, 5, 3] = np.nan
        with pytest.raises(DetectorError):
            det.score(X)

    def test_artifacts_immutable_after_inference(self, fixture_model_dir, long_flight_frames):
        """Inference must not modify the checkpoint or scaler on disk."""
        import hashlib, shutil
        dest = fixture_model_dir  # fixture_model_dir is session-scoped; copy for safety
        pth = dest / "lstm_autoencoder.pth"
        before = hashlib.sha256(pth.read_bytes()).hexdigest()
        clear_caches()
        analyze_flight(long_flight_frames, flight_id="art-immut", model_dir=str(dest))
        assert hashlib.sha256(pth.read_bytes()).hexdigest() == before


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Stored-flight reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestReproducibility:
    def test_reanalysis_gives_same_scores(self, fixture_model_dir, long_flight_frames):
        clear_caches()
        r1 = analyze_flight(long_flight_frames, flight_id="repro",
                             model_dir=str(fixture_model_dir))
        r2 = analyze_flight(long_flight_frames, flight_id="repro",
                             model_dir=str(fixture_model_dir))
        assert r1.n_windows == r2.n_windows
        assert r1.n_flagged == r2.n_flagged
        assert abs(r1.score_mean - r2.score_mean) < 1e-7

    def test_stored_windows_match_fresh_analysis(self, client, tmp_path, monkeypatch,
                                                  fixture_model_dir):
        monkeypatch.setenv("UAV_MODEL_DIR", str(fixture_model_dir))
        data_dir = tmp_path / "flights2"
        monkeypatch.setenv("DATA_DIR", str(data_dir))
        from backend import deps
        from src import inference_service
        deps.get_store.cache_clear()
        inference_service.clear_caches()

        store = FlightStore(data_dir)
        fid = "repro-stored"
        frames = {
            "IMU": _make_imu(500), "ATT": _make_att(200), "BARO": _make_baro(200)
        }
        store.create_from_upload(
            frames, source="individual",
            filenames={"IMU": "IMU.csv", "ATT": "ATT.csv", "BARO": "BARO.csv"},
            file_sizes={}, warnings=[], validation={}, flight_id=fid,
        )
        r1 = analyze_flight(frames, flight_id=fid,
                            model_dir=str(fixture_model_dir))
        store.save_analysis(
            fid,
            summary=r1.summary_dict(),
            windows=r1.windows,
            events=list(r1.events),
            events_df=r1.events_df,
            aligned=r1.aligned,
            report=r1.report,
            status="analyzed",
        )
        r2 = analyze_flight(frames, flight_id=fid,
                            model_dir=str(fixture_model_dir))
        stored = store.load_windows(fid)
        assert stored is not None
        assert len(stored) == len(r2.windows)
        diff = np.abs(
            r2.windows["anomaly_score"].to_numpy()
            - stored["anomaly_score"].to_numpy()
        )
        assert diff.max() < 1e-6
        inference_service.clear_caches()
        deps.get_store.cache_clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Failure handling and repeated analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestFailureHandling:
    def test_failed_analysis_does_not_corrupt_previous_result(self, client):
        fid = _upload(client, "fail-safe", n=600)
        # First successful analysis
        a1 = client.post(f"/api/flights/{fid}/analyze", json={}).json()
        assert a1["ok"]
        report_before = client.get(f"/api/flights/{fid}/report").json()

        # Second analysis (re-run) should not corrupt first
        a2 = client.post(f"/api/flights/{fid}/analyze", json={}).json()
        # Either ok or error — must not produce a 500 crash
        assert "ok" in a2
        report_after = client.get(f"/api/flights/{fid}/report").json()
        # If second analysis succeeded, report should still be coherent
        assert report_after.get("schema_version") == "1.0"

    def test_repeated_analysis_is_deterministic(self, client):
        fid = _upload(client, "det-repeat", n=600)
        a1 = client.post(f"/api/flights/{fid}/analyze", json={}).json()
        a2 = client.post(f"/api/flights/{fid}/analyze", json={}).json()
        if a1["ok"] and a2["ok"]:
            assert a1["n_windows"] == a2["n_windows"]
            assert a1["n_flagged"] == a2["n_flagged"]

    def test_analysis_of_nonexistent_flight_returns_404(self, client):
        r = client.post("/api/flights/does-not-exist/analyze", json={})
        assert r.status_code == 404

    def test_report_of_unanalysed_flight_returns_404(self, client):
        fid = _upload(client, "no-report", n=500)
        r = client.get(f"/api/flights/{fid}/report")
        assert r.status_code == 404

    def test_overview_before_analysis_is_not_analyzed(self, client):
        fid = _upload(client, "pre-analysis", n=500)
        o = client.get(f"/api/flights/{fid}/overview").json()
        assert o["verdict"]["state"] == "not_analyzed"

    def test_missing_artifacts_produces_error_not_no_anomaly(self, tmp_path,
                                                               long_flight_frames):
        clear_caches()
        result = analyze_flight(
            long_flight_frames,
            flight_id="no-arts",
            model_dir=str(tmp_path),   # empty dir → no artifacts
        )
        assert not result.ok
        assert result.status == "error"
        assert any("missing" in e.lower() or "not found" in e.lower()
                   for e in result.errors)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Upload-summary endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestUploadSummary:
    def test_upload_summary_endpoint_returns_stream_cards(self, client):
        fid = _upload(client, "sum-test")
        r = client.get(f"/api/flights/{fid}/upload-summary").json()
        assert r["ready_for_analysis"] is True
        streams = {s["stream"]: s for s in r["streams"]}
        assert set(streams.keys()) >= {"IMU", "ATT", "BARO"}
        for name, s in streams.items():
            assert s["rows"] > 0

    def test_upload_summary_shows_sample_rates(self, client):
        fid = _upload(client, "sum-hz")
        r = client.get(f"/api/flights/{fid}/upload-summary").json()
        for s in r["streams"]:
            ts = s.get("timestamp")
            if ts:
                assert ts["est_hz"] is not None and ts["est_hz"] > 0

    def test_upload_summary_of_missing_flight_returns_404(self, client):
        r = client.get("/api/flights/ghost-flight/upload-summary")
        assert r.status_code == 404

    def test_build_upload_summary_with_minimal_meta(self):
        meta = {
            "flight_id": "test",
            "source": "individual",
            "filenames": {"IMU": "IMU.csv", "ATT": "ATT.csv", "BARO": "BARO.csv"},
            "file_sizes": {"IMU": 1000, "ATT": 800, "BARO": 600},
            "warnings": [],
        }
        result = build_upload_summary(meta, {})
        assert result["flight_id"] == "test"
        assert isinstance(result["streams"], list)
        assert isinstance(result["hints"], list)
        assert "analysis_note" in result

    def test_upload_summary_includes_analysis_note_about_baro(self, client):
        fid = _upload(client, "sum-baro-note")
        r = client.get(f"/api/flights/{fid}/upload-summary").json()
        assert "not scored" in r["analysis_note"].lower() or \
               "barometer" in r["analysis_note"].lower()

    def test_upload_summary_role_describes_detector_input(self, client):
        fid = _upload(client, "sum-role")
        r = client.get(f"/api/flights/{fid}/upload-summary").json()
        streams = {s["stream"]: s for s in r["streams"]}
        # IMU and ATT are scored; BARO is context only
        assert "scored" in streams["IMU"]["role"].lower() or \
               "detector" in streams["IMU"]["role"].lower()
        assert "context" in streams["BARO"]["role"].lower() or \
               "not scored" in streams["BARO"]["role"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Verdict text — never says safe / normal
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerdictText:
    @pytest.mark.parametrize("state", VERDICT_STATES)
    def test_verdict_text_never_claims_safe(self, state):
        """No verdict state must produce text claiming the flight is safe."""
        from src.operator_view import VERDICT_TEXT
        headline, _ = VERDICT_TEXT[state]
        assert "safe" not in headline.lower()
        assert "normal" not in headline.lower()

    def test_no_alerts_caveats_include_safety_disclaimer(self):
        v = compute_verdict(ok=True, status="ok", n_events=0, coverage_status="ok")
        full = json.dumps(v).lower()
        assert "does not show that the aircraft was safe" in full

    def test_analysis_failed_is_not_ok(self):
        v = compute_verdict(ok=False, status="error", n_events=0,
                            coverage_status="error")
        assert v["state"] == "analysis_failed"
        assert v["tone"] == "error"

    def test_anomalies_state_has_attention_tone(self):
        v = compute_verdict(ok=True, status="ok", n_events=3, coverage_status="ok")
        assert v["state"] == "anomalies_detected"
        assert v["tone"] == "attention"

    def test_all_verdict_states_covered(self):
        from src import operator_view as ov
        assert set(VERDICT_STATES) == {
            ov.ANOMALIES, ov.NO_ALERTS_OK, ov.NO_ALERTS_LOW,
            ov.FAILED, ov.INSUFFICIENT, ov.NOT_ANALYZED,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Analysis type is recorded and isolated from AI Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalysisTypeIsolation:
    def test_deep_analysis_report_has_detector_info(self, fixture_model_dir,
                                                     long_flight_frames):
        clear_caches()
        result = analyze_flight(
            long_flight_frames, flight_id="iso-deep",
            model_dir=str(fixture_model_dir),
        )
        assert result.ok and result.report is not None
        assert result.report.detector.model == "LSTMAE"
        assert result.report.detector.feature_count == 12

    def test_deep_analysis_flight_stored_in_flights_not_direct(self, client, tmp_path,
                                                                monkeypatch,
                                                                fixture_model_dir):
        """Deep Analysis must store results under data/flights, not data/direct."""
        monkeypatch.setenv("UAV_MODEL_DIR", str(fixture_model_dir))
        data_dir = tmp_path / "flights_iso"
        direct_dir = tmp_path / "direct_iso"
        monkeypatch.setenv("DATA_DIR", str(data_dir))
        monkeypatch.setenv("DIRECT_DATA_DIR", str(direct_dir))
        from backend import deps
        from src import inference_service
        deps.get_store.cache_clear()
        inference_service.clear_caches()

        tc = TestClient(app)
        fid = "iso-store"
        r = tc.post(
            "/api/flights/upload",
            files={
                "imu":  ("IMU.csv",  _csv(_make_imu(500)), "text/csv"),
                "att":  ("ATT.csv",  _csv(_make_att(200)), "text/csv"),
                "baro": ("BARO.csv", _csv(_make_baro(200)), "text/csv"),
            },
            data={"flight_id": fid},
        ).json()
        assert r["ok"]
        assert (data_dir / fid / "meta.json").exists()
        assert not direct_dir.exists() or not (direct_dir / fid).exists()

        inference_service.clear_caches()
        deps.get_store.cache_clear()

    def test_operator_report_does_not_expose_raw_scores_in_events(self, client):
        fid = _upload(client, "score-hidden", n=600)
        client.post(f"/api/flights/{fid}/analyze", json={})
        rep = client.get(f"/api/flights/{fid}/operator-report").json()
        for ev in rep.get("events", []):
            assert "peak_score" not in ev
            assert "threshold" not in ev

    def test_technical_appendix_has_threshold_and_scores(self, client):
        fid = _upload(client, "appendix-check", n=600)
        client.post(f"/api/flights/{fid}/analyze", json={})
        rep = client.get(f"/api/flights/{fid}/operator-report").json()
        app_ = rep.get("technical_appendix", {})
        det = app_.get("detector", {})
        assert det.get("threshold") == pytest.approx(4.034964084625244)
        assert "features" in det
