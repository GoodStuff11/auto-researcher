"""`hub ingest batch FILE`: apply an agent-written ingest file — safe changes directly, judgement calls staged.

File schema (YAML), every op carrying a source id (no whitespace):
  watermark: {name: <sync-state key, letters/digits/_-.>, value: <str|int|date>}   # optional; set last
  safe:                                                  # applied directly; allowed keys only
    - {op: entry, id: <item id>, text: <non-empty str>, source: <id>}
    - {op: log, text: <non-empty str>, source: <id>}
    - {op: new, type: question|action, title: <non-empty str>, source: <id>,
       owner?: <str, a known person>, series?: [<str>], priority?: <int 1-3>,
       links?: {spawned_by: [<existing item id>]}, description?: <str>, fields?: {due: <YYYY-MM-DD>}}
  proposals:                                             # staged as Status/Pending/<P-id>.md; allowed keys only
    - {title: <non-empty str>, reason: <non-empty str, why a human decides>, source: <id>,
       ops: [<pending ops, see pending-format.md>]}
  flags: [<non-empty str>, ...]                          # added to Status/sync-state/flags.md (duplicates ignored)

Judgement calls are proposals: safe `new` creates only questions and actions, its links may only be
spawned_by, and its fields may only set due. Decisions, closures, answers and pivots are proposals.
Any problem fails the whole file with BatchError before anything is written. Re-running the same
file is a no-op: log lines, entries, new items and proposal titles are matched and skipped.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from .items import ENTRY_RE
from .pending import add_proposal, check_ops, load_proposals
from .store import Store
from .sync_state import NAME_RE, load_state, save_state, set_watermark

SAFE_REQUIRED = {"entry": ("id", "text", "source"), "log": ("text", "source"), "new": ("type", "title", "source")}
ALLOWED_KEYS = {
    "entry": ("op", "id", "text", "source"),
    "log": ("op", "text", "source"),
    "new": ("op", "type", "title", "source", "owner", "series", "priority", "links", "description", "fields"),
    "proposal": ("title", "reason", "source", "ops"),
}
SAFE_NEW_TYPES = ("question", "action")
SAFE_NEW_FIELDS = ("due",)
SAFE_LINK_RELS = ("spawned_by",)
TOP_KEYS = ("watermark", "safe", "proposals", "flags")
DECIDED_RE = re.compile(r" — (?:Applied|Rejected) P-[a-z0-9]{4}: (.*)$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class BatchError(ValueError):
    """The ingest file is malformed; nothing was written."""


def _norm(text) -> str:
    return " ".join(str(text).split())


def _key(text) -> str:
    return _norm(text).casefold()


def _is_str(value) -> bool:
    return isinstance(value, str)


def _nonempty(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _token(value) -> bool:
    return isinstance(value, str) and value != "" and not any(c.isspace() for c in value)


def _valid_due(value) -> bool:
    if isinstance(value, date):
        return True
    if not (isinstance(value, str) and DATE_RE.match(value)):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _check_new(op: dict, where: str, existing: set[str], people, errs: list[str]) -> None:
    if not _nonempty(op["title"]):
        errs.append(f"{where}: title must be a non-empty string")
    if not _is_str(op["type"]):
        errs.append(f"{where}: type must be a string")
    elif op["type"] not in SAFE_NEW_TYPES:
        errs.append(f"{where}: safe `new` may only create {' or '.join(SAFE_NEW_TYPES)}; decisions are proposals")
    owner = op.get("owner")
    if owner is not None:
        if not _is_str(owner):
            errs.append(f"{where}: owner must be a string")
        elif people is not None and owner not in people:
            errs.append(f"{where}: owner {owner} is not a known person")
    if "series" in op and not (isinstance(op["series"], list) and all(_is_str(s) for s in op["series"])):
        errs.append(f"{where}: series must be a list of strings")
    if "priority" in op:
        p = op["priority"]
        if isinstance(p, bool) or not isinstance(p, int) or not 1 <= p <= 3:
            errs.append(f"{where}: priority must be an integer 1-3, got {p!r}")
    if op.get("description") is not None and not _is_str(op["description"]):
        errs.append(f"{where}: description must be a string")
    fields = op.get("fields")
    if fields is not None:
        if not isinstance(fields, dict) or any(k not in SAFE_NEW_FIELDS for k in fields) \
                or ("due" in fields and not _valid_due(fields["due"])):
            errs.append(f"{where}: fields may only set due (a YYYY-MM-DD date); closures are proposals")
    links = op.get("links")
    if links is not None:
        if not isinstance(links, dict):
            errs.append(f"{where}: links must be a mapping of relation -> list of ids")
            return
        for rel, targets in links.items():
            if rel not in SAFE_LINK_RELS:
                errs.append(f"{where}: only spawned_by links are safe; {rel} is a proposal")
            elif not isinstance(targets, list):
                errs.append(f"{where}: links.{rel} must be a list")
            else:
                for t in targets:
                    if not _is_str(t) or t not in existing:
                        errs.append(f"{where}: links.{rel} -> {t} does not exist")


def _check_safe(store: Store, safe: list, people, errs: list[str]) -> None:
    existing = store.ids()
    for i, op in enumerate(safe):
        where = f"safe {i + 1}"
        if not isinstance(op, dict):
            errs.append(f"{where}: must be a mapping with op entry|log|new")
            continue
        kind = op.get("op")
        if not _is_str(kind) or kind not in SAFE_REQUIRED:
            errs.append(f"{where}: op must be one of entry|log|new")
            continue
        where = f"{where} ({kind})"
        unknown = [k for k in op if k not in ALLOWED_KEYS[kind]]
        if unknown:
            errs.append(f"{where}: unknown key(s) {', '.join(str(k) for k in unknown)}")
            continue
        missing = [k for k in SAFE_REQUIRED[kind] if k not in op or op[k] in (None, "")]
        if missing:
            errs.append(f"{where}: missing {', '.join(missing)}")
            continue
        if not _token(op["source"]):
            errs.append(f"{where}: source must be non-empty with no whitespace, got {op['source']!r}")
        if kind in ("entry", "log") and not _nonempty(op["text"]):
            errs.append(f"{where}: text must be a non-empty string")
        if kind == "entry":
            if not _is_str(op["id"]):
                errs.append(f"{where}: id must be a string")
            elif op["id"] not in existing:
                errs.append(f"{where}: item {op['id']} does not exist")
        elif kind == "new":
            _check_new(op, where, existing, people, errs)


def check_batch(store: Store, batch, people=None) -> list[str]:
    """Every problem with a batch, as readable lines (empty = fine). Nothing is written."""
    if not isinstance(batch, dict):
        return ["batch file must be a mapping"]
    errs = [f"unknown top-level key {k!r} (expected {', '.join(TOP_KEYS)})" for k in batch if k not in TOP_KEYS]
    for key in ("safe", "proposals", "flags"):
        if batch.get(key) is not None and not isinstance(batch[key], list):
            errs.append(f"{key} must be a list")
    if errs:
        return errs
    _check_safe(store, batch.get("safe") or [], people, errs)
    for i, pr in enumerate(batch.get("proposals") or []):
        where = f"proposal {i + 1}"
        if not isinstance(pr, dict):
            errs.append(f"{where}: must be a mapping")
            continue
        unknown = [k for k in pr if k not in ALLOWED_KEYS["proposal"]]
        if unknown:
            errs.append(f"{where}: unknown key(s) {', '.join(str(k) for k in unknown)}")
            continue
        if not all(pr.get(k) not in (None, "") for k in ("title", "reason", "source", "ops")):
            errs.append(f"{where}: needs title, reason, source and ops")
            continue
        if not _nonempty(pr["title"]):
            errs.append(f"{where}: title must be a non-empty string")
        if not _nonempty(pr["reason"]):
            errs.append(f"{where}: reason must be a non-empty string")
        if not _token(pr["source"]):
            errs.append(f"{where}: source must be non-empty with no whitespace, got {pr['source']!r}")
        ops = pr["ops"]
        if not isinstance(ops, list) or not all(isinstance(o, dict) and _is_str(o.get("op")) for o in ops):
            errs.append(f"{where}: ops must be a list of mappings, each with an op name")
            continue
        errs += [f"{where}: {e}" for e in check_ops(store, ops)]
        if people is not None:
            for j, o in enumerate(ops):
                if o.get("op") == "new" and o.get("owner") is not None:
                    if not _is_str(o["owner"]):
                        errs.append(f"{where} (op {j + 1}): owner must be a string")
                    elif o["owner"] not in people:
                        errs.append(f"{where} (op {j + 1}): owner {o['owner']} is not a known person")
    if not all(_nonempty(f) for f in batch.get("flags") or []):
        errs.append("flags must be non-empty strings")
    wm = batch.get("watermark")
    if wm is not None:
        if not isinstance(wm, dict) or "name" not in wm or "value" not in wm:
            errs.append("watermark must be {name: <sync-state key>, value: <string, int or date>}")
        elif not (_is_str(wm["name"]) and NAME_RE.match(wm["name"])):
            errs.append(f"watermark name must be letters, digits, '_', '-' or '.', got {wm['name']!r}")
        elif not (isinstance(wm["value"], (str, int, date)) and not isinstance(wm["value"], bool)):
            errs.append("watermark must be {name: <sync-state key>, value: <string, int or date>}")
    return errs


def _log_text(root) -> str:
    d = Path(root) / "Status" / "Log"
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(d.glob("*.md"))) if d.is_dir() else ""


def _log_pairs(root) -> set[tuple[str, str]]:
    pairs = set()
    for line in _log_text(root).splitlines():
        m = ENTRY_RE.match(line)
        if m:
            pairs.add((m[3], _norm(m[4])))
    return pairs


def run_batch(root, store: Store, batch, *, author, day, people=None) -> dict:
    errs = check_batch(store, batch, people=people)
    if errs:
        raise BatchError("nothing written; fix the batch file:\n  " + "\n  ".join(errs))
    report = {"applied": 0, "skipped": 0, "staged": [], "skipped_proposals": 0, "flags_added": 0,
              "watermark": None}
    logged = _log_pairs(root)
    items = store.all()
    for op in batch.get("safe") or []:
        kind, source = op["op"], op["source"]
        if kind == "entry":
            if any(e.source == source and _norm(e.text) == _norm(op["text"]) for e in store.get(op["id"]).entries):
                report["skipped"] += 1
                continue
            store.entry(op["id"], op["text"], author=author, source=source, day=day)
        elif kind == "log":
            key = (source, _norm(op["text"]))
            if key in logged:
                report["skipped"] += 1
                continue
            store.log(op["text"], author=author, source=source, day=day)
            logged.add(key)
        else:
            title = _key(op["title"])
            if any(i.type == op["type"] and _key(i.title) == title and source in (i.meta.get("sources") or [])
                   for i in items):
                report["skipped"] += 1
                continue
            fields = {"due": op["fields"]["due"]} if op.get("fields") else None
            items.append(store.create(
                op["type"], op["title"], owner=op.get("owner"), author=author, day=day, source=source,
                series=list(op.get("series") or []), priority=op.get("priority", 2),
                links={rel: list(t) for rel, t in (op.get("links") or {}).items()},
                description=op.get("description") or "", fields=fields))
        report["applied"] += 1
    known = {_key(p.title) for p in load_proposals(root)}
    known |= {_key(m[1]) for line in _log_text(root).splitlines() if (m := DECIDED_RE.search(line))}
    for pr in batch.get("proposals") or []:
        title = _key(pr["title"])
        if title in known:
            report["skipped_proposals"] += 1
            continue
        report["staged"].append(add_proposal(root, pr["title"], pr["ops"], proposer=author, source=pr["source"],
                                             reason=pr["reason"], day=day))
        known.add(title)
    flags = [_norm(f) for f in batch.get("flags") or []]
    if flags:
        state = load_state(root)
        for f in flags:
            if f not in state["flags"]:
                state["flags"].append(f)
                report["flags_added"] += 1
        save_state(root, state)
    wm = batch.get("watermark")
    if wm:
        set_watermark(root, str(wm["name"]), str(wm["value"]))
        report["watermark"] = f"{wm['name']}={wm['value']}"
    return report
