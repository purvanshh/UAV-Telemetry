"""What the agent is doing right now, so the app can say so while a review runs.

Each running review reports its steps here (a tool call with its arguments, thinking between calls, fact-checking the
report). The API returns the latest step; the frontend turns it into a plain sentence. In-process only: one entry per
analysis while its review runs, removed when it ends.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional

_lock = threading.Lock()
_runs: Dict[str, Dict[str, Any]] = {}


def start(key: str) -> Callable[[str, Optional[Dict[str, Any]]], None]:
    with _lock:
        _runs[key] = {"step": "starting", "tool": None, "args": {}, "calls": 0, "started": time.monotonic()}

    def report(step: str, detail: Optional[Dict[str, Any]] = None) -> None:
        with _lock:
            run = _runs.get(key)
            if run is None:
                return
            run["step"] = step
            if step == "tool" and detail:
                run["tool"], run["args"] = detail.get("name"), detail.get("args") or {}
                run["calls"] += 1
    return report


def finish(key: str) -> None:
    with _lock:
        _runs.pop(key, None)


def get(key: str) -> Optional[Dict[str, Any]]:
    with _lock:
        run = _runs.get(key)
        if run is None:
            return None
        return {"step": run["step"], "tool": run["tool"], "args": dict(run["args"]), "calls": run["calls"],
                "elapsed_s": round(time.monotonic() - run["started"], 1)}
