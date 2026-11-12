"""Status/Pending/<pid>.md: staged judgement calls, each a list of store ops applied only on approval.

One file per proposal, so concurrent proposals from different clones never touch the same file.
Ops are checked when staged and again before applying; applying a proposal is all-or-nothing (every
item file it touched is restored on failure). Items created by proposal P get ids drawn from a
generator seeded with "P:<op index>", so two clones applying the same proposal create the same ids.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .fsutil import atomic_write_text
from .ids import new_pending_id
from .items import LINK_KEYS, STATUSES, TYPES
from .store import PROTECTED_KEYS, Store, StoreError, check_field, coerce_value

FENCE = "`" * 3  # built, not literal, so this file nests inside Markdown fences
BLOCK_RE = re.compile(r"^## (P-[a-z0-9]{4}) · (.*?)\n" + FENCE + r"yaml\n(.*?)\n" + FENCE + r"\n", re.M | re.S)
OP_REQUIRED = {"new": ("type", "title"), "entry": ("id", "text"), "set": ("id", "fields"),
               "link": ("id", "rel", "target")}


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


def _parse(path: Path) -> Proposal | None:
    try:
        m = BLOCK_RE.search(path.read_text(encoding="utf-8").replace("\r\n", "\n"))
        if not m:
            return None
        body = yaml.safe_load(m[3]) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    return Proposal(m[1], m[2], body) if isinstance(body, dict) else None


def load_proposals(root) -> list[Proposal]:
    """Every readable proposal. Files that cannot be read are listed by load_unreadable()."""
    d = _dir(root)
    if not d.is_dir():
        return []
    return [pr for pr in (_parse(p) for p in sorted(d.glob("P-*.md"))) if pr is not None]


def load_unreadable(root) -> list[Path]:
    """Proposal files that load_proposals() cannot read (bad heading, fence or YAML)."""
    d = _dir(root)
    if not d.is_dir():
        return []
    return [p for p in sorted(d.glob("P-*.md")) if _parse(p) is None]


def _write(root, proposal: Proposal) -> None:
    body = yaml.safe_dump(proposal.body, sort_keys=False, allow_unicode=True).rstrip("\n")
    atomic_write_text(_file(root, proposal.pid),
                      f"## {proposal.pid} · {proposal.title}\n{FENCE}yaml\n{body}\n{FENCE}\n")


def _delete(root, pid: str) -> None:
    _file(root, pid).unlink(missing_ok=True)


def _token_ok(value) -> bool:
    return isinstance(value, str) and bool(value) and not any(ch.isspace() for ch in value)


def _check_author(author) -> None:
    if not _token_ok(author):
        raise PendingError(f"author must be a non-empty string with no whitespace, got {author!r}")


def check_ops(store: Store, ops) -> list[str]:
    """Every problem that would make `ops` fail to apply, as human-readable lines (empty = fine)."""
    if not isinstance(ops, list):
        return [f"ops must be a list, got {type(ops).__name__}"]
    if not ops:
        return ["proposal has no ops"]
    errs: list[str] = []
    existing = store.ids()
    refs: set[str] = set()
    ref_types: dict[str, str] = {}
    linked = {}  # clickup task id -> the action already carrying it
    for it in store.query(type="action"):
        if it.meta.get("clickup_id") not in (None, ""):
            linked.setdefault(str(it.meta["clickup_id"]), it.id)

    def type_of(value):
        if isinstance(value, str) and value.startswith("$"):
            return ref_types.get(value)
        try:
            return store.get(value).type
        except Exception:  # missing or unreadable: reported by target_ok / the load error
            return None

    def fields_ok(where, fields, item_type):
        for key, value in fields.items():
            if key in PROTECTED_KEYS:
                errs.append(f"{where}: cannot set '{key}' through fields")
                continue
            try:
                value = check_field(key, coerce_value(key, value))
            except StoreError as exc:
                errs.append(f"{where}: {exc}")
                continue
            if key == "status" and item_type in STATUSES and value not in STATUSES[item_type]:
                errs.append(f"{where}: status '{value}' invalid for {item_type}; "
                            f"expected one of {STATUSES[item_type]}")

    def target_ok(where, value):
        if isinstance(value, str) and value.startswith("$"):
            if value not in refs:
                errs.append(f"{where}: ref {value} is not defined by an earlier `new` op")
        elif not isinstance(value, str) or value not in existing:
            errs.append(f"{where}: item {value} does not exist")

    for i, op in enumerate(ops):
        where = f"op {i + 1}"
        if not isinstance(op, dict):
            errs.append(f"{where}: op must be a mapping, got {op!r}")
            continue
        kind = op.get("op")
        if kind not in OP_REQUIRED:
            errs.append(f"{where}: unknown op {kind!r} (expected one of {', '.join(OP_REQUIRED)})")
            continue
        where = f"{where} ({kind})"
        missing = [k for k in OP_REQUIRED[kind] if k not in op]
        if missing:
            errs.append(f"{where}: missing required key(s) {', '.join(repr(k) for k in missing)}")
            continue
        if "source" in op and not _token_ok(op["source"]):
            errs.append(f"{where}: source must be non-empty with no whitespace, got {op['source']!r}")
        if "author" in op and not _token_ok(op["author"]):
            errs.append(f"{where}: author must be non-empty with no whitespace, got {op['author']!r}")
        if "fields" in op and not isinstance(op["fields"], dict):
            errs.append(f"{where}: fields must be a mapping")
        if kind == "new":
            if op["type"] not in TYPES:
                errs.append(f"{where}: unknown type {op['type']!r} (expected one of {TYPES})")
            fields = op.get("fields") if isinstance(op.get("fields"), dict) else {}
            fields_ok(where, fields, op["type"])
            cid = fields.get("clickup_id")
            if cid not in (None, "") and str(cid) in linked:
                errs.append(f"{where}: ClickUp task {cid} already linked to {linked[str(cid)]}")
            if "series" in op and not isinstance(op["series"], list):
                errs.append(f"{where}: series must be a list")
            links = op.get("links") or {}
            if not isinstance(links, dict):
                errs.append(f"{where}: links must be a mapping of relation -> list of ids")
                links = {}
            for rel, targets in links.items():
                if rel not in LINK_KEYS:
                    errs.append(f"{where}: unknown link {rel!r} (expected one of {LINK_KEYS})")
                elif not isinstance(targets, list):
                    errs.append(f"{where}: links.{rel} must be a list")
                else:
                    for t in targets:
                        target_ok(f"{where} links.{rel}", t)
            if op.get("ref"):
                ref = op["ref"]
                if not isinstance(ref, str) or not ref.startswith("$"):
                    errs.append(f"{where}: ref must start with '$', got {ref!r}")
                else:
                    refs.add(ref)
                    ref_types[ref] = op["type"]
        else:
            target_ok(f"{where} id", op["id"])
            if kind == "set" and isinstance(op["fields"], dict):
                fields_ok(where, op["fields"], type_of(op["id"]))
            if kind == "link":
                if op["rel"] not in LINK_KEYS:
                    errs.append(f"{where}: unknown link {op['rel']!r} (expected one of {LINK_KEYS})")
                target_ok(f"{where} target", op["target"])
    return errs


def add_proposal(root, title, ops, *, proposer, source, reason, day, rng=None) -> str:
    """Stage a proposal and return its id; raises PendingError if check_ops finds any problem."""
    title = " ".join(str(title).split())  # one line, so BLOCK_RE always matches on reload
    errs = check_ops(Store(root), ops)
    if errs:
        raise PendingError("proposal rejected:\n  " + "\n  ".join(errs))
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


class _Undo:
    """Remembers each item file's bytes before its first change, and every file created."""

    def __init__(self, store: Store):
        self.store = store
        self.before: dict[Path, bytes | None] = {}

    def touch(self, item_id) -> None:
        try:
            path = self.store.path(item_id)
        except KeyError:
            return  # not a valid id: the store call will fail before writing
        if path not in self.before:
            self.before[path] = path.read_bytes() if path.is_file() else None

    def created(self, item_id) -> None:
        self.before.setdefault(self.store.path(item_id), None)

    def rollback(self) -> None:
        for path, data in self.before.items():
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(data)


def _run_ops(store: Store, ops, *, author, day, seed=None, undo: _Undo | None = None
             ) -> tuple[dict[str, str], list[str]]:
    refs: dict[str, str] = {}
    created: list[str] = []
    undo = undo or _Undo(store)
    try:
        for i, op in enumerate(ops):
            if not isinstance(op, dict):
                raise PendingError(f"op must be a mapping, got {op!r}")
            kind = op.get("op")
            if kind == "new":
                links = {k: _resolve(v, refs) for k, v in (op.get("links") or {}).items()}
                rng = random.Random(f"{seed}:{i}") if seed is not None else None
                item = store.create(_req(op, "type"), _req(op, "title"), owner=op.get("owner"), author=author,
                                    day=day, source=op.get("source", "pending"), series=op.get("series") or [],
                                    priority=op.get("priority", 2), links=links,
                                    description=op.get("description", ""), fields=op.get("fields"), rng=rng)
                undo.created(item.id)
                created.append(item.id)
                if op.get("ref"):
                    refs[op["ref"]] = item.id
            elif kind == "entry":
                item_id = _resolve(_req(op, "id"), refs)
                undo.touch(item_id)
                store.entry(item_id, _req(op, "text"), author=author, source=op.get("source", "pending"), day=day)
            elif kind == "set":
                item_id = _resolve(_req(op, "id"), refs)
                undo.touch(item_id)
                store.set_fields(item_id, op.get("fields") or {})
            elif kind == "link":
                item_id = _resolve(_req(op, "id"), refs)
                undo.touch(item_id)
                store.link(item_id, _req(op, "rel"), _resolve(_req(op, "target"), refs))
            else:
                raise PendingError(f"unknown op '{kind}'")
    except PendingError:
        raise
    except (StoreError, KeyError, ValueError, AttributeError, TypeError) as exc:
        raise PendingError(f"op failed: {exc}") from exc
    return refs, created


def apply_ops(store: Store, ops, *, author, day) -> dict[str, str]:
    """Apply ops in order; return the ref -> created id map. Not transactional; see apply_proposals."""
    refs, _ = _run_ops(store, ops, author=author, day=day)
    return refs


def _take(root, pids) -> list[Proposal]:
    by_id = {p.pid: p for p in load_proposals(root)}
    missing = [pid for pid in pids if pid not in by_id]
    if missing:
        unreadable = {p.stem for p in load_unreadable(root)} & set(missing)
        hint = f" (unreadable: {', '.join(sorted(unreadable))})" if unreadable else ""
        raise PendingError(f"unknown proposal(s): {', '.join(missing)}{hint}")
    return [by_id[pid] for pid in pids]


def apply_proposals(root, store: Store, pids, *, author, day, on_applied=None) -> list[str]:
    """Apply proposals in order and return every created id.

    Each proposal is all-or-nothing: its ops are checked first, and if any op fails every item file it
    touched is restored, files it created are removed, the proposal stays pending and PendingError is
    raised (proposals applied before it stay applied). `on_applied(pid, created_ids)` is called after
    each proposal is applied, logged and removed.
    """
    _check_author(author)  # before any write, so the post-ops log call cannot fail after ops land
    chosen = _take(root, pids)
    for pr in chosen:
        errs = check_ops(store, pr.body.get("ops") or [])
        if errs:
            raise PendingError(f"{pr.pid} cannot be applied:\n  " + "\n  ".join(errs))
    created: list[str] = []
    for pr in chosen:
        # Again just before applying: a proposal applied earlier in this call may have linked the same
        # ClickUp task or otherwise changed what these ops can do.
        errs = check_ops(store, pr.body.get("ops") or [])
        if errs:
            done = f" (already applied: {', '.join(p.pid for p in chosen[:chosen.index(pr)])})" \
                if chosen.index(pr) else ""
            raise PendingError(f"{pr.pid} cannot be applied:\n  " + "\n  ".join(errs) + done)
        undo = _Undo(store)
        try:
            _, made = _run_ops(store, pr.body.get("ops") or [], author=author, day=day, seed=pr.pid, undo=undo)
        except Exception as exc:
            undo.rollback()
            done = f" (already applied: {', '.join(p.pid for p in chosen[:chosen.index(pr)])})" \
                if chosen.index(pr) else ""
            msg = str(exc) if isinstance(exc, PendingError) else f"op failed: {exc}"
            raise PendingError(f"{pr.pid} not applied, nothing changed by it: {msg}{done}") from exc
        store.log(f"Applied {pr.pid}: {pr.title}", author=author, source=f"pending:{pr.pid}", day=day)
        _delete(root, pr.pid)
        created += made
        if on_applied:
            on_applied(pr.pid, made)
    return created


def reject_proposals(root, store: Store, pids, *, author, day) -> None:
    _check_author(author)
    chosen = _take(root, pids)
    for pr in chosen:
        store.log(f"Rejected {pr.pid}: {pr.title}", author=author, source=f"pending:{pr.pid}", day=day)
        _delete(root, pr.pid)
