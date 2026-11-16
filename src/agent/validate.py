"""Check the agent's report against what the tools actually returned. Nothing the model writes is trusted.

* Every number in prose must match a number the tools or the deterministic analysis produced (rounding allowed,
  percentages of returned fractions allowed). Times written as m:ss must appear in a tool result or match a returned
  time. Sentences that fail are removed and listed.
* Finding ids and tool references must exist. Every detector finding gets a review; a finding the agent skipped is
  added back as "not reviewed by the agent", so nothing the code found can disappear.
* Causes must read as suggestions; safety verdicts and certainty language are removed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

import numpy as np

LIMITS = {"headline": 240, "summary": 1200, "text": 900, "item": 400, "reviews": 60, "chain": 40, "observations": 15,
          "recommendations": 10, "not_checked": 15}

MINOR = {"unknown_or_duplicate_finding", "unknown_finding", "unknown_reference"}  # dropped silently from the report

_NUM = re.compile(r"(?<![A-Za-z_\d.\[])[-+−]?\d+(?:[.,]\d+)?(?![A-Za-z_\d\]])")
_CLOCK = re.compile(r"\b(\d{1,3}):(\d{2}(?:\.\d+)?)\b")
_IDENT = re.compile(r"\b(?:[A-Za-z_]+\[\d+\]|[A-Za-z]+_\w+|[A-Z]{1,3}\d+\w*|M\d|E\d+|F\d+|T\d+)\b")
_SAFETY = re.compile(r"\b(safe to fly|is safe|was safe|unsafe|airworthy|unairworthy|cleared (for|to) fly|fit to fly|"
                     r"no (risk|danger))\b", re.I)
_CERTAIN = re.compile(r"\b(definitely|certainly|undoubtedly|without (a )?doubt|proves?|proven|confirmed (to be|as)|"
                      r"is the cause|was the cause|caused by)\b", re.I)


def _numbers(obj: Any, out: List[float], clocks: Set[str]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if np.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, str):
        for m in _CLOCK.finditer(obj):
            clocks.add(m.group(0))
        for m in _NUM.finditer(_IDENT.sub(" ", obj)):
            try:
                out.append(float(m.group(0).replace("−", "-").replace(",", ".")))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, out, clocks)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers(v, out, clocks)


class Grounding:
    def __init__(self, *sources: Any):
        pool: List[float] = []
        self.clocks: Set[str] = set()
        for s in sources:
            _numbers(s, pool, self.clocks)
        vals = np.abs(np.array(pool, dtype=float)) if pool else np.array([])
        self.pool = np.unique(np.concatenate([vals, vals * 100])) if len(vals) else vals  # fractions quoted as percent
        self.times = np.unique(np.array([v for v in pool if v >= 0], dtype=float)) if pool else np.array([])

    def _has(self, v: float) -> bool:
        if not len(self.pool):
            return False
        tol = max(0.011, 0.006 * v)  # rounding to 2-3 significant figures
        if np.any(np.abs(self.pool - v) <= tol):
            return True
        # rounded to whole number or one decimal place
        return bool(np.any(np.abs(np.round(self.pool) - v) < 1e-9) or np.any(np.abs(np.round(self.pool, 1) - v) < 1e-9))

    def ungrounded(self, text: str) -> List[str]:
        bad: List[str] = []
        body = text
        for m in _CLOCK.finditer(text):
            tok = m.group(0)
            secs = int(m.group(1)) * 60 + float(m.group(2))
            tol = 0.15 if "." in m.group(2) else 1.0  # "5:43" may be a rounded "5:43.3"
            if tok not in self.clocks and not (len(self.times) and np.any(np.abs(self.times - secs) <= tol)):
                bad.append(tok)
            body = body.replace(tok, " ")
        for tok in _NUM.findall(_IDENT.sub(" ", body)):
            v = abs(float(tok.replace("−", "-").replace(",", ".")))
            if v <= 12 and v == int(v):
                continue  # counts, ordinals, motor numbers, axis numbers
            if not self._has(v):
                bad.append(tok)
        return bad


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", (text or "").strip()) if s.strip()]


class Checker:
    def __init__(self, grounding: Grounding, refs: Set[str], finding_ids: Set[str]):
        self.g, self.refs, self.fids = grounding, refs, finding_ids
        self.rejected: List[Dict[str, str]] = []

    def text(self, value: str, where: str, limit: int, cause: bool = False) -> str:
        kept = []
        for s in _sentences((value or "")[: limit * 2]):
            reason = None
            if _SAFETY.search(s):
                reason = "safety_verdict"
            elif cause and _CERTAIN.search(s):
                reason = "cause_stated_as_certain"
            else:
                bad = self.g.ungrounded(s)
                if bad:
                    reason = "ungrounded_number:" + ",".join(bad[:4])
            if reason:
                self.rejected.append({"where": where, "reason": reason, "text": s[:220]})
            else:
                kept.append(s)
        out = " ".join(kept)
        return out if len(out) <= limit else out[:limit].rsplit(" ", 1)[0] + "…"

    def ref_list(self, refs: Iterable[str], where: str) -> List[str]:
        good, bad = [], []
        for r in refs or []:
            r = str(r).strip()
            (good if r in self.refs or r in self.fids else bad).append(r)
        if bad:
            self.rejected.append({"where": where, "reason": "unknown_reference", "text": ", ".join(bad[:6])})
        return list(dict.fromkeys(good))


def validate(raw: Dict[str, Any], *, findings: List[Dict[str, Any]], tool_results: Dict[str, Any],
             analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Return the cleaned report against the detector findings."""
    fid_key = "id"
    fmap = {f[fid_key]: f for f in findings}
    g = Grounding(tool_results, analysis)
    ck = Checker(g, set(tool_results), set(fmap))

    reviews, seen = [], set()
    for r in (raw.get("finding_reviews") or [])[: LIMITS["reviews"]]:
        fid = str(r.get("finding_id", "")).strip()
        if fid not in fmap or fid in seen:
            ck.rejected.append({"where": "finding_reviews", "reason": "unknown_or_duplicate_finding", "text": fid})
            continue
        seen.add(fid)
        w = f"finding_reviews.{fid}"
        assess = r.get("assessment") if r.get("assessment") in ("confirmed", "likely_false_alarm", "needs_review") else "needs_review"
        refs = ck.ref_list(r.get("evidence_refs"), w + ".evidence_refs")
        expl = ck.text(r.get("explanation", ""), w + ".explanation", LIMITS["text"])
        if assess == "likely_false_alarm" and (not expl or not any(x.startswith("T") for x in refs)):
            ck.rejected.append({"where": w, "reason": "false_alarm_without_tool_evidence", "text": fid})
            assess = "needs_review"
        reviews.append({"finding_id": fid, "assessment": assess, "explanation": expl, "evidence_refs": refs,
                        "suggested_cause": ck.text(r.get("suggested_cause", ""), w + ".suggested_cause", LIMITS["item"], cause=True),
                        "check_next": ck.text(r.get("check_next", ""), w + ".check_next", LIMITS["item"]),
                        "reviewed_by_agent": True})
    for fid, f in fmap.items():
        if fid not in seen:
            reviews.append({"finding_id": fid, "assessment": "needs_review", "explanation": "", "evidence_refs": [fid],
                            "suggested_cause": "", "check_next": "", "reviewed_by_agent": False})
    order = {fid: i for i, fid in enumerate(fmap)}
    reviews.sort(key=lambda r: order.get(r["finding_id"], 1e9))

    chain = []
    for c in (raw.get("root_cause_chain") or [])[: LIMITS["chain"]]:
        fid = str(c.get("finding_id", "")).strip()
        if fid not in fmap:
            ck.rejected.append({"where": "root_cause_chain", "reason": "unknown_finding", "text": fid})
            continue
        role = c.get("role") if c.get("role") in ("root_cause", "consequence", "independent") else "independent"
        chain.append({"finding_id": fid, "role": role, "link": ck.text(c.get("link", ""), f"root_cause_chain.{fid}", LIMITS["item"], cause=True)})

    obs = []
    for i, o in enumerate((raw.get("additional_observations") or [])[: LIMITS["observations"]]):
        w = f"additional_observations[{i}]"
        refs = ck.ref_list(o.get("evidence_refs"), w + ".evidence_refs")
        detail = ck.text(o.get("detail", ""), w + ".detail", LIMITS["text"])
        title = ck.text(o.get("title", ""), w + ".title", 160)
        if not refs or not detail:
            ck.rejected.append({"where": w, "reason": "observation_without_evidence", "text": (o.get("title") or "")[:160]})
            continue
        code = o.get("suggested_code")
        from src.arjuna.checks import FAULT_CODES
        ts, te = o.get("t_start"), o.get("t_end")
        obs.append({"title": title or "Observation", "detail": detail, "evidence_refs": refs,
                    "t_start": ts if isinstance(ts, (int, float)) and g._has(abs(float(ts))) else None,
                    "t_end": te if isinstance(te, (int, float)) and g._has(abs(float(te))) else None,
                    "suggested_code": code if code in FAULT_CODES else None})

    recs = []
    for i, r in enumerate((raw.get("recommendations") or [])[: LIMITS["recommendations"]]):
        action = ck.text(r.get("action", ""), f"recommendations[{i}].action", LIMITS["item"])
        if action:
            recs.append({"action": action, "reason": ck.text(r.get("reason", ""), f"recommendations[{i}].reason", LIMITS["item"], cause=True),
                         "finding_ids": [x for x in (r.get("finding_ids") or []) if x in fmap]})

    headline = ck.text(raw.get("headline", ""), "headline", LIMITS["headline"], cause=True)
    summary = ck.text(raw.get("summary", ""), "summary", LIMITS["summary"], cause=True)
    return {
        "headline": headline, "summary": summary, "finding_reviews": reviews, "root_cause_chain": chain,
        "additional_observations": obs,
        "pilot_report_answer": ck.text(raw.get("pilot_report_answer", ""), "pilot_report_answer", LIMITS["text"], cause=True),
        "not_checked": [t for t in (ck.text(x, "not_checked", LIMITS["item"]) for x in (raw.get("not_checked") or [])[: LIMITS["not_checked"]]) if t],
        "recommendations": recs, "rejected": [r for r in ck.rejected if r["reason"] not in MINOR],
        "normalized": [r for r in ck.rejected if r["reason"] in MINOR],
        "unreviewed_findings": [r["finding_id"] for r in reviews if not r["reviewed_by_agent"]],
    }
