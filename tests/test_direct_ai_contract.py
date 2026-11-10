"""The GPT-4o mini report contract: root-cause regression, malformed/incomplete replies, API failures, retries, logging."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src import direct_llm
from tests.test_direct_analysis import _env, client, good_report, post, raw, summ  # noqa: F401  (fixtures)

SENTINEL = "SENTINEL-flight-value-87654321"
KEY = "sk-test-do-not-log-1234567890"


class SeqClient:
    """Returns / raises the queued items in order. Item: Exception | str content | (content, finish_reason[, refusal])."""

    def __init__(self, *items):
        self.items, self.calls = list(items), []

    def create(self, **kw):
        self.calls.append(kw)
        it = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(it, BaseException):
            raise it
        content, finish, refusal = (it, "stop", None) if isinstance(it, str) else (tuple(it) + (None,) * 3)[:3]
        msg = SimpleNamespace(content=content, refusal=refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("OPENAI_API_KEY", KEY)


@pytest.fixture(scope="module")
def data():
    return summ("FLT_0001.CSV").data


def gen(data, *items, consent=True):
    c = SeqClient(*items)
    return c, direct_llm.generate(data, consent=consent, client=c)


# ---- the original failure -------------------------------------------------------------------------------------------------------

def test_root_cause_over_long_but_valid_reply_is_no_longer_discarded(data):
    """Strict JSON-schema mode cannot enforce array/string length, so GPT-4o mini returned valid JSON whose lists exceeded the
    backend's length caps; the old pydantic ``max_length`` rules then rejected the *entire* reply as 'not in the required format'.
    Over-long text is now shortened at a sentence boundary and recorded; the structure contract is unchanged."""
    r = json.loads(good_report(data))
    long = "The raw channel departed from its earlier values. " * 30
    r["events"][0]["summary"] = long + f" Rows {data['events'][0]['start_row']} to {data['events'][0]['end_row']}."
    r["unknowns"] = [f"Unknown number {i}." for i in range(20)]
    r["data_quality"] = r["data_quality"] * 12
    c, rep = gen(data, json.dumps(r))
    assert rep["status"] in ("ok", "partial") and len(c.calls) == 1, rep["message"]
    assert len(rep["events"]) == 1 and len(rep["events"][0]["summary"]) <= direct_llm.LIMITS["field"] + 1
    assert len(rep["unknowns"]) == direct_llm.LIMITS["unknowns"] and len(rep["data_quality"]) == direct_llm.LIMITS["data_quality"]
    kinds = {(n["where"], n["action"]) for n in rep["normalized"]}
    assert ("unknowns", "list_truncated") in kinds and any(a == "shortened" for _w, a in kinds)


def test_request_schema_matches_the_response_model_and_forbids_extra_keys(data):
    sch = direct_llm.response_schema()["json_schema"]
    assert sch["strict"] is True
    top = sch["schema"]
    assert top["additionalProperties"] is False and set(top["required"]) == set(direct_llm.AIReport.model_fields)
    ev = top["properties"]["events"]["items"]
    assert set(ev["required"]) == set(direct_llm.EventNote.model_fields) and ev["additionalProperties"] is False
    # the model no longer produces evidence ids: those are attached by the backend
    assert "evidence_ids" not in json.dumps(sch)


def test_successful_structured_report_carries_backend_evidence_and_category(data):
    c, rep = gen(data, good_report(data))
    assert rep["status"] == "ok" and rep["failure"] is None and rep["message"] == ""
    e = rep["events"][0]
    d0 = data["events"][0]
    assert e["category"] == d0["category"] and e["source"]["start_row"] == d0["start_row"] and e["measurements"]
    assert c.calls[0]["response_format"]["json_schema"]["strict"] and "event_facts" in c.calls[0]["messages"][1]["content"]


# ---- malformed / incomplete replies ----------------------------------------------------------------------------------------------

def test_missing_field_is_a_named_schema_failure_with_one_corrective_retry(data):
    bad = json.loads(good_report(data))
    del bad["events"]
    c, rep = gen(data, json.dumps(bad), good_report(data))
    assert rep["status"] == "ok" and len(c.calls) == 2                     # retried once, then succeeded
    assert "events" in c.calls[1]["messages"][-1]["content"] and "missing" in c.calls[1]["messages"][-1]["content"]
    assert [a["outcome"] for a in rep["provenance"]["attempts"]] == ["schema", "ok"]


@pytest.mark.parametrize("content,reason", [
    ("not json at all", "not_json"), ("[1, 2]", "not_object"), ("", "empty"),
    ('{"executive_summary": 1, "events": [], "data_quality": [], "recommendations": [], "unknowns": []}', "schema"),
    (json.dumps({"executive_summary": "x", "events": [{"event_ref": "E1"}], "data_quality": [], "recommendations": [], "unknowns": []}), "schema"),
])
def test_unusable_replies_fail_with_a_reason_after_at_most_two_calls(data, content, reason):
    c, rep = gen(data, content)
    assert rep["status"] == "invalid_output" and rep["failure"]["reason"] == reason and len(c.calls) == 2   # bounded: 1 + 1 retry
    assert rep["events"] == [] and rep["executive_summary"] == "" and rep["limitations"]            # nothing from the model is used
    assert "computed summary" in rep["message"].lower() and "ignored" not in rep["message"]
    if reason == "schema":
        assert rep["failure"]["fields"]


def test_extra_unexpected_keys_are_still_rejected_not_tolerated(data):
    r = json.loads(good_report(data))
    r["evidence_ids"] = ["E1.x.y"]
    _c, rep = gen(data, json.dumps(r))
    assert rep["status"] == "invalid_output" and any("extra_forbidden" in f for f in rep["failure"]["fields"])


def test_truncated_reply_is_retried_with_a_larger_budget_then_reported(data):
    c, rep = gen(data, ('{"executive_summary": "cut', "length"), good_report(data))
    assert rep["status"] == "ok" and len(c.calls) == 2 and c.calls[1]["max_tokens"] > c.calls[0]["max_tokens"]
    c, rep = gen(data, ('{"executive_summary": "cut', "length"))
    assert rep["status"] == "invalid_output" and rep["failure"]["reason"] == "truncated" and "cut off" in rep["message"] and len(c.calls) == 2


def test_refusal_is_not_retried(data):
    c, rep = gen(data, ("", "stop", "I can't help with that."))
    assert rep["status"] == "invalid_output" and rep["failure"]["reason"] == "refusal" and len(c.calls) == 1
    assert "can't help" not in json.dumps(rep)


def test_no_choices_is_handled(data):
    class Odd:
        def create(self, **kw):
            return SimpleNamespace(choices=[], usage=None)

    rep = direct_llm.generate(data, consent=True, client=Odd())
    assert rep["status"] == "invalid_output" and rep["failure"]["reason"] == "empty"


def test_unsupported_claims_inside_a_valid_reply_are_removed_not_fatal(data):
    r = json.loads(good_report(data))
    r["events"][0]["conclusion"] = "This indicates the motor failed because of vibration."
    r["events"][0]["summary"] += " It was a significant change."
    _c, rep = gen(data, json.dumps(r))
    assert rep["status"] == "partial" and rep["events"]
    blob = json.dumps(rep["events"])
    assert "motor failed" not in blob and "significant" not in blob
    assert {x["reason"] for x in rep["rejected"]} >= {"fault", "overclaim"}


def test_unverified_time_unit_on_raw_ticks_is_repaired_and_recorded(data):
    e = data["events"][0]
    r = json.loads(good_report(data))
    r["events"][0]["summary"] = f"rc_yaw changed at raw time {e['start_raw']:.0f} ms."
    _c, rep = gen(data, json.dumps(r))
    s = rep["events"][0]["summary"]
    assert "raw ticks" in s and " ms" not in s
    assert any(n["action"] == "unverified_time_unit_replaced_with_raw_ticks" for n in rep["normalized"])


# ---- API failures ----------------------------------------------------------------------------------------------------------------------

def test_transient_api_errors_are_retried_once_then_reported_concisely(data):
    rate = type("RateLimitError", (Exception,), {})("429 secret-detail")
    c, rep = gen(data, rate, good_report(data))
    assert rep["status"] == "ok" and len(c.calls) == 2
    c, rep = gen(data, rate)
    assert rep["status"] == "error" and "rate-limiting" in rep["message"] and len(c.calls) == 2
    assert "secret-detail" not in json.dumps(rep) and rep["failure"]["reason"] == "api_rate_limit"
    c, rep = gen(data, TimeoutError("t"))
    assert rep["status"] == "error" and "in time" in rep["message"] and len(c.calls) == 2


def test_auth_errors_and_unknown_errors_are_not_retried(data):
    auth = type("AuthenticationError", (Exception,), {})("bad key " + KEY)
    c, rep = gen(data, auth)
    assert rep["status"] == "error" and len(c.calls) == 1 and "credentials" in rep["message"] and KEY not in json.dumps(rep)
    c, rep = gen(data, RuntimeError("boom"))
    assert rep["status"] == "error" and len(c.calls) == 1 and "boom" not in json.dumps(rep)


def test_no_consent_or_no_key_means_no_call(data, monkeypatch):
    c, rep = gen(data, good_report(data), consent=False)
    assert rep["status"] == "not_requested" and c.calls == []
    monkeypatch.delenv("OPENAI_API_KEY")
    assert direct_llm.generate(data, consent=True)["status"] == "unavailable"


# ---- logging ---------------------------------------------------------------------------------------------------------------------------

def test_logs_carry_reason_codes_but_no_key_reply_text_or_flight_values(data, caplog):
    caplog.set_level(logging.DEBUG)
    bad = json.dumps({"executive_summary": SENTINEL, "events": "nope"})
    gen(data, bad)
    gen(data, type("AuthenticationError", (Exception,), {})(f"{KEY} {SENTINEL}"))
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "reason=schema" in text and "kind=auth" in text
    assert SENTINEL not in text and KEY not in text
    for v in ("91501", "92541", "FLT_0001"):
        assert v not in text


# ---- deterministic results survive AI failure --------------------------------------------------------------------------------------------

def _use(monkeypatch, *items):
    c = SeqClient(*items)
    orig = direct_llm.generate
    monkeypatch.setattr(direct_llm, "generate", lambda d, *, consent, client=None: orig(d, consent=consent, client=c))
    return c


def test_ai_failure_leaves_computed_summary_events_and_operator_report_intact(client, monkeypatch):
    plain = post(client).json()
    _use(monkeypatch, "not json")
    failed = post(client, llm_consent="true").json()
    assert failed["stage"] == "invalid_output" and failed["report"]["failure"]["reason"] == "not_json"
    assert failed["summary"]["events"] == plain["summary"]["events"] and failed["summary"]["operator"] == plain["summary"]["operator"]
    assert failed["summary"]["input_fingerprint"] == plain["summary"]["input_fingerprint"]
    again = client.get(f"/api/direct/{failed['analysis_id']}").json()
    assert again["summary"]["operator"]["headline"] and again["meta"]["ai_attempt"]["reason"] == "not_json"
    assert again["report"] is None                                           # a failed attempt stores no report


def test_stale_analysis_is_flagged_not_crashed(client, tmp_path):
    from backend.direct_store import DirectStore

    aid = post(client).json()["analysis_id"]
    store = DirectStore(tmp_path / "direct")
    s = store.load_summary(aid)
    s.pop("operator"); s["analysis_version"] = "direct-1"
    (tmp_path / "direct" / aid / "summary.json").write_text(json.dumps(s))
    v = client.get(f"/api/direct/{aid}").json()
    assert v["stale_analysis"] is True


def test_model_cannot_change_deterministic_data(data):
    before = json.dumps(data, sort_keys=True, default=str)
    gen(data, good_report(data))
    gen(data, "garbage")
    assert json.dumps(data, sort_keys=True, default=str) == before


def test_report_from_an_older_prompt_version_is_not_reused(client, monkeypatch):
    c = _use(monkeypatch, good_report(summ("FLT_0001.CSV").data))
    first = post(client, llm_consent="true").json()
    n = len(c.calls)
    from backend.routers.direct import DirectStore  # noqa: F401
    import backend.routers.direct as d

    store = d.get_direct_store()
    rep = store.load_report(first["analysis_id"])
    rep["provenance"]["prompt_version"] = "direct-ai-1"
    store.save_report(first["analysis_id"], rep)
    again = post(client, llm_consent="true").json()
    assert len(c.calls) == n + 1 and "reused_from" not in again["report"]
