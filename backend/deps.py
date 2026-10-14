"""Shared FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache

from backend.store import FlightStore


@lru_cache(maxsize=1)
def get_store() -> FlightStore:
    return FlightStore()
