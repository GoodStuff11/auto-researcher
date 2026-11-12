"""Collision-safe identifiers for items and pending proposals (safe across independent writers)."""
from __future__ import annotations

import random
import string
from datetime import date

TYPE_PREFIX = {"question": "Q", "decision": "D", "action": "A"}
ALPHABET = string.ascii_lowercase + string.digits


def _suffix(rng, n: int) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(n))


def new_id(item_type: str, day: date, existing: set[str], rng=None) -> str:
    """Return `<T>-<YYYYMMDD>-<rand3>` not present in `existing`."""
    prefix = TYPE_PREFIX[item_type]
    rng = rng or random.Random()
    for _ in range(100):
        candidate = f"{prefix}-{day:%Y%m%d}-{_suffix(rng, 3)}"
        if candidate not in existing:
            return candidate
    raise RuntimeError("could not find a free item id after 100 attempts")


def new_pending_id(existing: set[str], rng=None) -> str:
    """Return `P-<rand4>` not present in `existing`."""
    rng = rng or random.Random()
    for _ in range(100):
        candidate = f"P-{_suffix(rng, 4)}"
        if candidate not in existing:
            return candidate
    raise RuntimeError("could not find a free pending id after 100 attempts")
