"""`hub ingest batch FILE`: apply an agent-written ingest file — safe changes directly, judgement calls staged.

File schema (YAML), every op carrying a source id (no whitespace):
  watermark: {name: <sync-state key>, value: <string>}   # optional; set only after everything succeeded
  safe:                                                  # applied directly
    - {op: entry, id: <item id>, text: <text>, source: <id>}
    - {op: log, text: <text>, source: <id>}
    - {op: new, type: question|action, title: <t>, source: <id>,
       owner?, series? (list), priority? (1-3), links? ({rel: [ids]}), description?, fields? (no status/closed)}
  proposals:                                             # staged as Status/Pending/<P-id>.md
    - {title: <t>, reason: <why a human decides>, source: <id>, ops: [<pending ops, see pending-format.md>]}
  flags: [<text>, ...]                                   # added to Status/sync-state/flags.md (duplicates ignored)
The whole file is checked before anything is written. Re-running the same file is a no-op.
"""
from __future__ import annotations

import re
from pathlib import Path

from .pending import add_proposal, check_ops, load_proposals
from .store import Store
from .sync_state import load_state, save_state, set_watermark

SAFE_REQUIRED = {"entry": ("id", "text", "source"), "log": ("text", "source"), "new": ("type", "title", "source")}
SAFE_NEW_TYPES = ("question", "action")
FORBIDDEN_NEW_FIELDS = ("status", "closed")
TOP_KEYS = ("watermark", "safe", "proposals", "flags")
DECIDED_RE = re.compile(r" — (?:Applied|Rejected) P-[a-z0-9]{4}: (.*)$")


class BatchError(ValueError):
    """The ingest file is malformed; nothing was written."""


def _norm(text) -> str:
    return " ".join(str(text).split())


def _token(value) -> bool:
    return isinstance(value, str) and value != "" and not any(c.isspace() for c in value)


def _check_safe(store: Store, safe: list, errs: list[str]) -> None:
    existing = store.ids()
    for i, op in enumerate(safe):
        where = f"safe {i + 1}"
        if not isinstance(op, dict) or op.get("op") not in SAFE_REQUIRED:
            errs.append(f"{where}: must be a mapping with op entry|log|new")
            continue
        missing = [k for k in SAFE_REQUIRED[op["op"]] if not op.get(k)]
        if missing:
            errs.append(f"{where} ({op['op']}): missing {', '.join(missing)}")
            continue
        if not _token(op["source"]):
            errs.append(f"{where}: source must be non-empty with no whitespace, got {op['source']!r}")
        if op["op"] == "entry" and op["id"] not in existing:
            errs.append(f"{where}: item {op['id']} does not exist")
        if op["op"] != "new":
            continue
        if op["type"] not in SAFE_NEW_TYPES:
            errs.append(f"{where}: safe `new` may only create {' or '.join(SAFE_NEW_TYPES)}; decisions are proposals")
        fields = op.get("fields") or {}
        if not isinstance(fields, dict) or any(k in fields for k in FORBIDDEN_NEW_FIELDS):
            errs.append(f"{where}: fields must be a mapping without status or closed (closures are proposals)")
        if "series" in op and not isinstance(op["series"], list):
            errs.append(f"{where}: series must be a list")
        links = op.get("links") or {}
        if not isinstance(links, dict):
            errs.append(f"{where}: links must be a mapping of relation -> list of ids")
            continue
        for rel, targets in links.items():
            if not isinstance(targets, list):
                errs.append(f"{where}: links.{rel} must be a list")
                continue
            for t in targets:
                if t not in existing:
                    errs.append(f"{where}: links.{rel} -> {t} does not exist")


def check_batch(store: Store, batch) -> list[str]:
    """Every problem with a batch, as readable lines (empty = fine). Nothing is written."""
    if not isinstance(batch, dict):
        return ["batch file must be a mapping"]
    errs = [f"unknown top-level key {k!r} (expected {', '.join(TOP_KEYS)})" for k in batch if k not in TOP_KEYS]
    for key in ("safe", "proposals", "flags"):
        if batch.get(key) is not None and not isinstance(batch[key], list):
            errs.append(f"{key} must be a list")
    if errs:
        return errs
    _check_safe(store, batch.get("safe") or [], errs)
    for i, pr in enumerate(batch.get("proposals") or []):
        where = f"proposal {i + 1}"
        if not isinstance(pr, dict) or not all(pr.get(k) for k in ("title", "reason", "source", "ops")):
            errs.append(f"{where}: needs title, reason, source and ops")
            continue
        if not _token(pr["source"]):
            errs.append(f"{where}: source must be non-empty with no whitespace, got {pr['source']!r}")
        errs += [f"{where}: {e}" for e in check_ops(store, pr["ops"])]
    if not all(isinstance(f, str) and f.strip() for f in batch.get("flags") or []):
        errs.append("flags must be non-empty strings")
    wm = batch.get("watermark")
    if wm is not None and not (isinstance(wm, dict) and _token(str(wm.get("name") or ""))
                               and wm.get("value") not in (None, "")):
        errs.append("watermark must be {name: <sync-state key>, value: <non-empty>}")
    return errs


def _log_text(root) -> str:
    d = Path(root) / "Status" / "Log"
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(d.glob("*.md"))) if d.is_dir() else ""


def run_batch(root, store: Store, batch, *, author, day) -> dict:
    errs = check_batch(store, batch)
    if errs:
        raise BatchError("nothing written; fix the batch file:\n  " + "\n  ".join(errs))
    report = {"applied": 0, "skipped": 0, "staged": [], "skipped_proposals": 0, "flags_added": 0,
              "watermark": None}
    logs = _log_text(root)
    items = store.all()
    for op in batch.get("safe") or []:
        kind, source = op["op"], op["source"]
        if kind == "entry":
            if any(e.source == source and _norm(e.text) == _norm(op["text"]) for e in store.get(op["id"]).entries):
                report["skipped"] += 1
                continue
            store.entry(op["id"], op["text"], author=author, source=source, day=day)
        elif kind == "log":
            line = f" · {source} — {_norm(op['text'])}"
            if line in logs:
                report["skipped"] += 1
                continue
            store.log(op["text"], author=author, source=source, day=day)
            logs += "\n" + line
        else:
            if any(i.type == op["type"] and i.title == op["title"] and source in (i.meta.get("sources") or [])
                   for i in items):
                report["skipped"] += 1
                continue
            items.append(store.create(
                op["type"], op["title"], owner=op.get("owner"), author=author, day=day, source=source,
                series=list(op.get("series") or []), priority=op.get("priority", 2),
                links={rel: list(t) for rel, t in (op.get("links") or {}).items()},
                description=op.get("description", ""), fields=op.get("fields")))
        report["applied"] += 1
    pending = {_norm(p.title) for p in load_proposals(root)}
    decided = {_norm(m[1]) for line in _log_text(root).splitlines() if (m := DECIDED_RE.search(line))}
    for pr in batch.get("proposals") or []:
        title = _norm(pr["title"])
        if title in pending or title in decided:
            report["skipped_proposals"] += 1
            continue
        report["staged"].append(add_proposal(root, title, pr["ops"], proposer=author, source=pr["source"],
                                             reason=pr["reason"], day=day))
        pending.add(title)
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
