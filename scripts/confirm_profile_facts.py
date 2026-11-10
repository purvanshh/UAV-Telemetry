"""List, confirm, revoke and audit owner confirmations of dataset-profile facts.

The profile (config/dataset_profiles.json) marks each unit / convention as supported (cross-checked in
the data) or unverified. The pipeline applies only facts the owner has confirmed here. Records go to
config/profile_confirmations.json, an APPEND-ONLY log; the profile file is never edited.

    python scripts/confirm_profile_facts.py --list
    python scripts/confirm_profile_facts.py --confirm time_unit=ms angle_unit=deg --by "Name" \\
        --basis "tilt test 2026-10-20 / firmware doc v1.2" --yes
    python scripts/confirm_profile_facts.py --revoke time_unit --by "Name" --basis "wrong" --yes
    python scripts/confirm_profile_facts.py --history

Rules: --by and --basis are mandatory; a unit fact must restate the profile's proposed unit; a choice
fact (roll_sign, pitch_sign, ...) takes one of its listed values; a record carries the digest of the
fact's definition and evidence, so editing the profile later invalidates it instead of transferring it;
re-confirming a different value needs --supersede (the old record stays in the log). Without --yes
nothing is written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_profiles import (  # noqa: E402
    DEFAULT_CONFIRMATIONS,
    apply_confirmations,
    fact_digest,
    load_base_profiles,
    load_confirmation_log,
)


def _facts(profile):
    """fact -> {kind, proposed|values, columns, evidence, effects}"""
    facts = {}
    if profile.time_column:
        facts["time_unit"] = {"kind": "unit", "proposed": profile.time_unit, "columns": [profile.time_column],
                              "level": profile.time_unit_status, "evidence": profile.time_evidence}
    for name, c in profile.columns.items():
        if c.fact:
            f = facts.setdefault(c.fact, {"kind": "unit", "proposed": c.unit or c.unit_hypothesis, "columns": [],
                                          "level": c.unit_status, "evidence": c.evidence})
            f["columns"].append(name)
    for fact, ch in profile.choices.items():
        eff = {}
        for v in ch["values"]:
            eff[v] = {}
            for name, c in profile.columns.items():
                if c.frame_expr and c.frame_expr.lstrip("-") == fact:
                    eff[v][name] = (-1 if c.frame_expr.startswith("-") else 1) * v
        cols = [n for n, c in profile.columns.items() if c.frame_expr and c.frame_expr.lstrip("-") == fact]
        facts[fact] = {"kind": "choice", "values": ch["values"], "meaning": ch.get("meaning", ""),
                       "how_to_check": ch.get("how_to_check", ""), "columns": cols, "effects": eff, "level": "owner choice",
                       "evidence": " ".join(profile.columns[c].evidence for c in cols[:1])}
    return facts


def _state(profile, log):
    applied = apply_confirmations(profile, log)
    return {a["fact"]: a for a in applied.applied_confirmations}, applied.confirmation_notes


def _write(path: Path, records: list) -> None:
    cur = json.loads(path.read_text()) if path.is_file() else {"confirmations": []}
    cur["confirmations"] += records
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=2) + "\n")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="uav_flt_csv")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--confirm", nargs="*", metavar="FACT=VALUE")
    ap.add_argument("--revoke", nargs="*", metavar="FACT")
    ap.add_argument("--by")
    ap.add_argument("--basis", default="")
    ap.add_argument("--supersede", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    prof = next((p for p in load_base_profiles() if p.id == a.profile), None)
    if prof is None:
        print(f"No profile '{a.profile}'.")
        return 2
    log = load_confirmation_log()
    facts = _facts(prof)
    done, notes = _state(prof, log)
    path = Path(os.environ.get("UAV_PROFILE_CONFIRMATIONS") or DEFAULT_CONFIRMATIONS)

    if a.history:
        for i, r in enumerate(log):
            print(f"#{i} {r.get('recorded_at','?')} {r.get('action','confirm')} {r.get('profile')} v{r.get('version')} "
                  f"{r.get('fact')}={r.get('value')} by {r.get('confirmed_by')} ({r.get('basis')})")
        return 0

    if a.list or (not a.confirm and not a.revoke):
        print(f"{prof.label}: {len(facts)} facts, {len(done)} confirmed\n")
        for k, f in sorted(facts.items()):
            tag = f"CONFIRMED {done[k]['value']} by {done[k]['confirmed_by']}" if k in done else "pending"
            if f["kind"] == "unit":
                print(f"  {k} = {f['proposed']}   [unit, {f['level']}, {tag}]   columns: {', '.join(sorted(set(f['columns'])))}")
            else:
                print(f"  {k} in {f['values']}   [owner choice, {tag}]   columns: {', '.join(f['columns']) or '-'}")
                print(f"      meaning: {f['meaning']}")
                print(f"      how to check: {f['how_to_check']}")
                for v, eff in f["effects"].items():
                    print(f"      if {v:+d}: " + ", ".join(f"{c} x{s:+d}" for c, s in eff.items()))
            if f["kind"] == "unit":
                print(f"      evidence: {f['evidence'][:900]}")
            print()
        for n in notes:
            print("  note:", n)
        return 0

    if not a.by or not a.basis.strip():
        print("--by NAME and a non-empty --basis are required: a confirmation records who confirmed it and on what grounds.")
        return 2
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = []
    for item in a.confirm or []:
        if "=" not in item:
            print(f"Bad item '{item}': use FACT=VALUE")
            return 2
        k, v = item.split("=", 1)
        if k not in facts:
            print(f"'{k}' is not a fact of {prof.label}. Facts: {', '.join(sorted(facts))}")
            return 2
        f = facts[k]
        if f["kind"] == "unit" and str(v) != str(f["proposed"]):
            print(f"Refusing {k}={v}: the profile proposes {f['proposed']}. Edit the profile (with evidence) to change a proposal.")
            return 2
        if f["kind"] == "choice" and v not in {str(x) for x in f["values"]}:
            print(f"Refusing {k}={v}: allowed values are {f['values']}.")
            return 2
        if k in done:
            if str(done[k]["value"]) == v:
                print(f"{k}={v} is already confirmed; nothing to record.")
                continue
            if not a.supersede:
                print(f"{k} is already confirmed as {done[k]['value']}. Use --supersede to record a change (the old record stays in the log).")
                return 2
        new.append({"action": "confirm", "profile": prof.id, "version": prof.version, "fact": k, "value": v,
                    "confirmed_by": a.by, "confirmed_on": date.today().isoformat(), "recorded_at": now, "basis": a.basis,
                    "fact_digest": fact_digest(prof, k), "evidence_snapshot": f["evidence"][:600],
                    "supersedes": next((i for i in range(len(log) - 1, -1, -1) if log[i].get("fact") == k and log[i].get("profile") == prof.id), None) if k in done else None,
                    "tool": "scripts/confirm_profile_facts.py"})
    for k in a.revoke or []:
        if k not in done:
            print(f"'{k}' is not currently confirmed; nothing to revoke.")
            return 2
        new.append({"action": "revoke", "profile": prof.id, "version": prof.version, "fact": k, "confirmed_by": a.by,
                    "confirmed_on": date.today().isoformat(), "recorded_at": now, "basis": a.basis, "tool": "scripts/confirm_profile_facts.py"})
    print(("Appending to " if a.yes else "Would append to ") + str(path))
    print(json.dumps(new, indent=2))
    if not a.yes:
        print("\nDry run. Re-run with --yes to record.")
        return 0
    if new:
        _write(path, new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
