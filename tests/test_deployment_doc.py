"""docs/DEPLOYED_MODEL.md must match the deployed artifacts (no stale contradictions)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("describe", ROOT / "scripts" / "describe_deployment.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_doc_is_up_to_date():
    from src.artifact_loader import missing_artifacts

    if missing_artifacts():
        pytest.skip("production artifacts not present")
    m = _mod()
    assert m.DOC.is_file(), "run scripts/describe_deployment.py --write"
    assert m.DOC.read_text() == m.render(), "docs/DEPLOYED_MODEL.md is stale"


def test_doc_lists_every_deployed_feature_and_threshold():
    from src.artifact_loader import load_artifacts, missing_artifacts

    if missing_artifacts():
        pytest.skip("production artifacts not present")
    text = _mod().DOC.read_text()
    a = load_artifacts()
    for f in a.features:
        assert f"`{f}`" in text
    assert repr(a.threshold) in text
    assert "4.42494" not in text


def test_no_other_doc_states_the_stale_threshold():
    for name in ("README.md", "uav_model/ARTIFACTS.md"):
        body = (ROOT / name).read_text()
        assert "4.42494" not in body
