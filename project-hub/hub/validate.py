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


def validate(store: Store, config: HubConfig) -> list[Violation]:
    items = {i.id: i for i in store.all()}
    out: list[Violation] = []
    spawned_from: dict[str, list[str]] = defaultdict(list)
    for it in items.values():
        for t in it.links("spawned_by"):
            spawned_from[t].append(it.id)

    def bad(it, rule, msg):
        out.append(Violation(it.id, rule, msg))

    for it in items.values():
        if it.type not in TYPES:
            bad(it, "type", f"unknown type '{it.type}'")
            continue
        if it.status not in STATUSES[it.type]:
            bad(it, "status", f"status '{it.status}' invalid for {it.type}")
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
            elif not any(e.day >= closed for e in it.entries):
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
