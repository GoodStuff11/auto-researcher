# project-hub/hub/pending.py
"""Status/Pending/<pid>.md: staged judgement calls, each a list of store ops applied only on approval.

One file per proposal, so concurrent proposals from different clones never touch the same file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .ids import new_pending_id
from .store import Store, StoreError

FENCE = "`" * 3  # built, not literal, so this file nests inside Markdown fences
BLOCK_RE = re.compile(r"^## (P-[a-z0-9]{4}) · (.*?)\n" + FENCE + r"yaml\n(.*?)\n" + FENCE + r"\n", re.M | re.S)


class PendingError(Exception):
    """A proposal is unknown or its ops cannot be applied."""


@dataclass
class Proposal:
    pid: str
    title: str
    body: dict


def _dir(root) -> Path:
    return Path(root) / "Status" / "Pending"


def _file(root, pid: str) -> Path:
    return _dir(root) / f"{pid}.md"


def load_proposals(root) -> list[Proposal]:
    d = _dir(root)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("P-*.md")):
        m = BLOCK_RE.search(p.read_text())
        if m:
            out.append(Proposal(m[1], m[2], yaml.safe_load(m[3]) or {}))
    return out


def _write(root, proposal: Proposal) -> None:
    body = yaml.safe_dump(proposal.body, sort_keys=False, allow_unicode=True).rstrip("\n")
    _dir(root).mkdir(parents=True, exist_ok=True)
    _file(root, proposal.pid).write_text(f"## {proposal.pid} · {proposal.title}\n{FENCE}yaml\n{body}\n{FENCE}\n")


def _delete(root, pid: str) -> None:
    _file(root, pid).unlink(missing_ok=True)


def _check_author(author) -> None:
    if not isinstance(author, str) or not author or any(ch.isspace() for ch in author):
        raise PendingError(f"author must be a non-empty string with no whitespace, got {author!r}")


def add_proposal(root, title, ops, *, proposer, source, reason, day, rng=None) -> str:
    title = " ".join(str(title).split())  # one line, so BLOCK_RE always matches on reload
    existing = {p.stem for p in _dir(root).glob("P-*.md")} if _dir(root).is_dir() else set()
    pid = new_pending_id(existing, rng)
    _write(root, Proposal(pid, title, {"proposer": proposer, "date": day, "source": source,
                                       "reason": reason, "ops": list(ops)}))
    return pid


def _resolve(value, refs):
    if isinstance(value, str) and value.startswith("$"):
        if value not in refs:
            raise PendingError(f"unknown ref {value}")
        return refs[value]
    if isinstance(value, list):
        return [_resolve(v, refs) for v in value]
    return value


def _req(op: dict, key: str):
    if key not in op:
        raise PendingError(f"op missing required key '{key}'")
    return op[key]


def _run_ops(store: Store, ops, *, author, day) -> tuple[dict[str, str], list[str]]:
    refs: dict[str, str] = {}
    created: list[str] = []
    try:
        for op in ops:
            if not isinstance(op, dict):
                raise PendingError(f"op must be a mapping, got {op!r}")
            kind = op.get("op")
            if kind == "new":
                links = {k: _resolve(v, refs) for k, v in (op.get("links") or {}).items()}
                item = store.create(_req(op, "type"), _req(op, "title"), owner=op.get("owner"), author=author,
                                    day=day, source=op.get("source", "pending"), series=op.get("series") or [],
                                    priority=op.get("priority", 2), links=links,
                                    description=op.get("description", ""), fields=op.get("fields"))
                created.append(item.id)
                if op.get("ref"):
                    refs[op["ref"]] = item.id
            elif kind == "entry":
                store.entry(_resolve(_req(op, "id"), refs), _req(op, "text"), author=author,
                            source=op.get("source", "pending"), day=day)
            elif kind == "set":
                store.set_fields(_resolve(_req(op, "id"), refs), op.get("fields") or {})
            elif kind == "link":
                store.link(_resolve(_req(op, "id"), refs), _req(op, "rel"), _resolve(_req(op, "target"), refs))
            else:
                raise PendingError(f"unknown op '{kind}'")
    except PendingError:
        raise
    except (StoreError, KeyError, ValueError, AttributeError, TypeError) as exc:
        raise PendingError(f"op failed: {exc}") from exc
    return refs, created


def apply_ops(store: Store, ops, *, author, day) -> dict[str, str]:
    """Apply ops in order; return the ref -> created id map."""
    refs, _ = _run_ops(store, ops, author=author, day=day)
    return refs


def _take(root, pids) -> list[Proposal]:
    by_id = {p.pid: p for p in load_proposals(root)}
    missing = [pid for pid in pids if pid not in by_id]
    if missing:
        raise PendingError(f"unknown proposal(s): {', '.join(missing)}")
    return [by_id[pid] for pid in pids]


def apply_proposals(root, store: Store, pids, *, author, day) -> list[str]:
    _check_author(author)  # before any write, so the post-ops log call cannot fail after ops land
    chosen = _take(root, pids)
    created: list[str] = []
    for pr in chosen:
        _, made = _run_ops(store, pr.body.get("ops") or [], author=author, day=day)
        created += made
        store.log(f"Applied {pr.pid}: {pr.title}", author=author, source=f"pending:{pr.pid}", day=day)
        _delete(root, pr.pid)
    return created


def reject_proposals(root, store: Store, pids, *, author, day) -> None:
    _check_author(author)
    chosen = _take(root, pids)
    for pr in chosen:
        store.log(f"Rejected {pr.pid}: {pr.title}", author=author, source=f"pending:{pr.pid}", day=day)
        _delete(root, pr.pid)
