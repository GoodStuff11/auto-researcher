"""Idempotency: has a source id (circleback:<id>, clickup:<id>, commit:<ROOT>@<sha>, …) already been recorded?

Ingest runs by different people overlap; every skill checks a source id here (CLI: `hub seen`) before
writing anything derived from it.
"""
from __future__ import annotations

from pathlib import Path

from .pending import load_proposals, load_unreadable
from .store import Store


def _rel(root, path) -> str:
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return str(path)


def source_seen(root, store: Store, source: str) -> list[str]:
    """Where `source` already appears: item ids listing it in `sources`, `log:<YYYY-MM>` for monthly logs
    with an entry citing it, and pending proposal ids whose body or ops cite it. Empty list = new.

    An `unreadable:<path>` entry means the answer may be incomplete; fix that file. Log files that cannot
    be decoded raise ValueError, since a silent "new" would defeat the idempotency check."""
    where = []
    for it in store.all():
        srcs = it.meta.get("sources")
        if isinstance(srcs, list) and source in srcs:
            where.append(it.id)
    where += [f"unreadable:{_rel(root, p)}" for p, _ in store.load_errors]
    log_dir = Path(root) / "Status" / "Log"
    if log_dir.is_dir():
        needle = f" · {source} — "
        for p in sorted(log_dir.glob("*.md")):
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ValueError(f"cannot read log {p}: {exc}") from exc
            if needle in text:
                where.append(f"log:{p.stem}")
    for pr in load_proposals(root):
        cited = {pr.body.get("source")}
        cited |= {op.get("source") for op in pr.body.get("ops") or [] if isinstance(op, dict)}
        if source in cited:
            where.append(pr.pid)
    where += [f"unreadable:{_rel(root, p)}" for p in load_unreadable(root)]
    return where
