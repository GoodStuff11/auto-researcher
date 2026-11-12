# project-hub/hub/validate.py
"""Structural rules every status store must satisfy. `hub validate` exits non-zero on any violation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from .config import HubConfig
from .ids import TYPE_PREFIX
from .items import CLOSING, LINK_KEYS, STATUSES, TYPES
from .store import Store


@dataclass
class Violation:
    item_id: str
    rule: str
    message: str


def _shape_problems(it) -> list[str]:
    """Frontmatter values whose type is wrong (e.g. hand-edited in Obsidian): reported, never fatal."""
    probs = []
    title = it.meta.get("title")
    if not isinstance(title, str) or not title.strip():
        probs.append("title must be a non-empty string")
    links = it.meta.get("links")
    if links is not None and not (isinstance(links, dict)
                                  and all(v is None or isinstance(v, list) for v in links.values())):
        probs.append("links must be a mapping of relation -> list of ids")
    for key in ("series", "sources"):
        value = it.meta.get(key)
        if value is not None and not isinstance(value, list):
            probs.append(f"{key} must be a list")
    return probs


def validate(store: Store, config: HubConfig) -> list[Violation]:
    items = {i.id: i for i in store.all()}
    out: list[Violation] = [Violation(path.stem, "format", f"cannot load {path.name}: {err}")
                            for path, err in store.load_errors]
    spawned_from: dict[str, list[str]] = defaultdict(list)
    for it in items.values():
        for t in it.links("spawned_by"):
            spawned_from[t].append(it.id)

    def bad(it, rule, msg):
        out.append(Violation(it.id, rule, msg))

    by_task: dict[str, list[str]] = defaultdict(list)
    for it in sorted(items.values(), key=lambda i: i.id):
        if it.type == "action" and it.meta.get("clickup_id") not in (None, ""):
            by_task[str(it.meta["clickup_id"])].append(it.id)
    for cid, ids in by_task.items():
        for extra in ids[1:]:
            out.append(Violation(extra, "duplicate-clickup-id",
                                 f"clickup_id {cid} is also on {ids[0]}; one task, one action"))

    for it in items.values():
        if it.type not in TYPES:
            bad(it, "type", f"unknown type '{it.type}'")
            continue
        if it.status not in STATUSES[it.type]:
            bad(it, "status", f"status '{it.status}' invalid for {it.type}")
        for msg in _shape_problems(it):
            bad(it, "shape", msg)
        if not it.id.startswith(TYPE_PREFIX[it.type] + "-"):
            bad(it, "id-prefix", f"id does not start with {TYPE_PREFIX[it.type]}-")
        for k in LINK_KEYS:
            for t in it.links(k):
                if t not in items:
                    bad(it, "link-target", f"{k} -> {t} does not exist")
        for e in it.entries:
            if e.author not in config.people:
                bad(it, "entry-author", f"entry author '{e.author}' not in project.yaml people")
                break
        if it.owner is not None and it.owner not in config.people:
            bad(it, "unknown-owner", f"owner '{it.owner}' not in project.yaml people")
        if it.status in CLOSING:
            closed = it.meta.get("closed")
            if not isinstance(closed, date):
                bad(it, "closed-date", f"status '{it.status}' requires a 'closed' date")
            elif not any(e.day >= closed for e in it.entries[1:]):
                bad(it, "closing-entry", "no entry dated on/after 'closed' explaining the closure")
        if it.status == "pivoted" and not it.links("pivoted_to"):
            bad(it, "pivot-target", "pivoted question needs a pivoted_to link")
        if it.type == "decision" and it.status == "made":
            if not it.links("answers"):
                bad(it, "decision-answers", "made decision answers no question")
            if not spawned_from[it.id] and not it.meta.get("no_action"):
                bad(it, "decision-actions", "made decision spawned no action and has no 'no_action' reason")
        if it.type == "action" and it.status == "open" and (it.owner is None or it.owner not in config.people):
            bad(it, "action-owner", "open action needs an owner from project.yaml people")
    return out
