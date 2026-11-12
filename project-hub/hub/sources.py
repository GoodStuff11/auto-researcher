"""Idempotency: has a source id (circleback:<id>, clickup:<id>, commit:<ROOT>@<sha>, …) already been recorded?

Ingest runs by different people overlap; every skill checks a source id here (CLI: `hub seen`) before
writing anything derived from it.
"""
from __future__ import annotations

from pathlib import Path

from .pending import load_proposals
from .store import Store


def source_seen(root, store: Store, source: str) -> list[str]:
    """Where `source` already appears: item ids listing it in `sources`, `log:<YYYY-MM>` for monthly logs
    with an entry citing it, and pending proposal ids whose body or ops cite it. Empty list = new."""
    where = [it.id for it in store.all() if source in (it.meta.get("sources") or [])]
    log_dir = Path(root) / "Status" / "Log"
    if log_dir.is_dir():
        needle = f" · {source} — "
        for p in sorted(log_dir.glob("*.md")):
            if needle in p.read_text(encoding="utf-8"):
                where.append(f"log:{p.stem}")
    for pr in load_proposals(root):
        cited = {pr.body.get("source")}
        cited |= {op.get("source") for op in pr.body.get("ops") or [] if isinstance(op, dict)}
        if source in cited:
            where.append(pr.pid)
    return where
