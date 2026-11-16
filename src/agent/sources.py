"""The signal interface the agent's tools read. Every upload is an ARJUNA log by now (CSVs are converted first).

Signals are addressed as (topic, field, instance). Times are seconds since the start of the log. Angles are converted
from radians to degrees here, so the language model never converts units.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.arjuna import units
from src.arjuna.ulog import ArjunaFlight

# Withheld from the AI service: position coordinates are sensitive for defence customers (guide, "Privacy").
WITHHELD = {("arjuna_gps", "latitude"), ("arjuna_gps", "longitude")}


class SignalError(ValueError):
    """A tool asked for a signal or period that does not exist; the message goes back to the agent."""


class Source:
    kind = ""
    time_basis = "seconds since log start"

    def signals(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def series(self, topic: str, field: str, instance: int = 0) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        raise NotImplementedError

    def air_mask(self, t: np.ndarray) -> np.ndarray:
        return np.ones_like(t, dtype=bool)

    def bounds(self) -> Tuple[float, float]:
        raise NotImplementedError


class ULogSource(Source):
    kind = "ulog"

    def __init__(self, flight: ArjunaFlight):
        self.fl = flight

    def _controller_id(self, topic: str, instance: int) -> Optional[int]:
        if topic != "arjuna_controller":
            return None
        df = self.fl.get(topic, instance)
        return int(df["controller_id"].iloc[0]) if df is not None and len(df) and np.isfinite(df["controller_id"].iloc[0]) else None

    def signals(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for (name, inst), tp in sorted(self.fl.topics.items()):
            if topic and name != topic or name == "arjuna_log_hash":
                continue
            cid = self._controller_id(name, inst)
            fields = []
            for f in tp.fields:
                if (name, units.prefix(f)) in WITHHELD:
                    continue
                unit, _ = units.display(units.unit_of(name, f, cid))
                meaning = units.meaning_of(name, f)
                if name == "arjuna_rc_input" and f.startswith("channels["):
                    meaning = self._rc_meaning(int(f[9:-1]))
                fields.append({"field": f, "unit": unit, "meaning": meaning})
            row: Dict[str, Any] = {"topic": name, "instance": inst, "rate_hz": tp.rate_hz, "samples": int(len(tp.df)),
                                   "fields": fields}
            if tp.unavailable:
                row["never_recorded"] = tp.unavailable
            if name == "arjuna_controller" and len(tp.df):
                from src.arjuna.ulog import enum_name
                row["controller"] = enum_name("CONTROLLER_ID", tp.df["controller_id"].iloc[0])
                row["axis"] = enum_name("AXIS", tp.df["axis"].iloc[0])
            out.append(row)
        return out

    def _rc_meaning(self, i: int) -> str:
        """Stick role of RC channel i (0-based) from the RCMAP_* parameters when the log carries them."""
        p = self.fl.params
        roles = {int(p[k]): k[6:].lower() for k in ("RCMAP_ROLL", "RCMAP_PITCH", "RCMAP_THROTTLE", "RCMAP_YAW") if k in p}
        if roles:
            return f"raw RC channel {i + 1}" + (f" (pilot {roles[i + 1]} stick, from RCMAP)" if i + 1 in roles else "")
        return f"raw RC channel {i + 1} (stick mapping not in the log)"

    def series(self, topic: str, field: str, instance: int = 0) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        if (topic, units.prefix(field)) in WITHHELD:
            raise SignalError(f"{topic}.{field} is withheld from the AI service for privacy (position coordinates).")
        df = self.fl.get(topic, instance)
        if df is None:
            have = sorted({n for n, _ in self.fl.topics})
            raise SignalError(f"No topic {topic} instance {instance} in this log. Topics: {', '.join(have)}")
        if field not in df:
            raise SignalError(f"{topic}[{instance}] has no field '{field}'. Fields: {', '.join(c for c in df.columns if c != 't')}")
        unit, scale = units.display(units.unit_of(topic, field, self._controller_id(topic, instance)))
        return df["t"].to_numpy(float), df[field].to_numpy(float) * scale, unit

    def air_mask(self, t: np.ndarray) -> np.ndarray:
        return self.fl.in_air(t)

    def bounds(self) -> Tuple[float, float]:
        return 0.0, float(self.fl.duration_s)
