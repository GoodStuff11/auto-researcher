"""Reconcile a ClickUp list export with the hub's actions (ClickUp owns task status, assignee and due date).

Input: a JSON (or YAML) list the agent exported from the list with the ClickUp connector:
  [{id, name, status: <str | {status: str, type?: str}>, assignees: [<id | username | email | {id, username, email}>],
    due: YYYY-MM-DD | epoch-ms | null, due_date?: (same forms, used when `due` is absent)}]
Configuration: project.yaml `sources.clickup: {list_id, done_statuses?}`; `people.<key>.clickup` = user id.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import yaml

from .batch import DECIDED_RE
from .pending import add_proposal, check_ops, load_proposals
from .store import Store
from .sync_state import load_state, save_state

DEFAULT_DONE = ("complete", "completed", "closed", "done")
DONE_TYPES = ("closed", "done")
FLAG_PREFIX = "clickup: "


class ClickUpError(ValueError):
    """The task export is malformed."""


def _due(task_id, value):
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ClickUpError(f"task {task_id}: due must be YYYY-MM-DD, epoch milliseconds or null, got {value!r}")
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        try:
            return datetime.fromtimestamp(int(value) / 1000).date().isoformat()
        except (OverflowError, OSError, ValueError) as exc:
            raise ClickUpError(f"task {task_id}: due epoch milliseconds out of range: {value!r}") from exc
    text = value.isoformat() if isinstance(value, date) else str(value)
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise ClickUpError(f"task {task_id}: due must be YYYY-MM-DD or null, got {value!r}") from exc


def _key(text) -> str:
    return " ".join(str(text).split()).casefold()


def _decided_titles(root) -> set[str]:
    """Titles of proposals already applied or rejected, read from the monthly logs."""
    d = Path(root) / "Status" / "Log"
    if not d.is_dir():
        return set()
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(d.glob("*.md")))
    return {_key(m[1]) for line in text.splitlines() if (m := DECIDED_RE.search(line))}


def load_tasks(path) -> list[dict]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ClickUpError("the task export must be a list of tasks")
    tasks = []
    for i, t in enumerate(raw):
        if not isinstance(t, dict) or not t.get("id") or not t.get("name"):
            raise ClickUpError(f"task {i + 1}: needs id and name")
        tid = str(t["id"])
        if any(ch.isspace() for ch in tid):
            raise ClickUpError(f"task {tid!r}: id must not contain whitespace")
        status = t.get("status")
        status_type = ""
        if isinstance(status, dict):
            status_type = str(status.get("type") or "").casefold()
            status = status.get("status")
        status = "" if status is None else str(status)
        if not status.strip():
            raise ClickUpError(f"task {tid}: missing status")
        raw_assignees = t.get("assignees")
        if raw_assignees is None:
            raw_assignees = []
        if not isinstance(raw_assignees, list):
            raise ClickUpError(f"task {tid}: assignees must be a list, got {raw_assignees!r}")
        assignees = []
        for a in raw_assignees:
            if isinstance(a, dict):
                assignees.append([str(a[k]) for k in ("id", "username", "email") if a.get(k) not in (None, "")])
            else:
                assignees.append([str(a)])
        raw_due = t["due"] if "due" in t else t.get("due_date")
        tasks.append({"id": tid, "name": str(t["name"]), "status": status, "status_type": status_type,
                      "assignees": [al for al in assignees if al], "due": _due(tid, raw_due)})
    return tasks


def _people_index(people: dict) -> dict[str, str]:
    idx = {}
    for key, person in people.items():
        idx[str(key).casefold()] = key
        if isinstance(person, dict):
            for field in ("clickup", "email", "name"):
                if person.get(field) not in (None, ""):
                    idx[str(person[field]).casefold()] = key
    return idx


def _owner_of(aliases, idx):
    for alias in aliases:
        if alias.casefold() in idx:
            return idx[alias.casefold()]
    return None


def _iso(value):
    return value.isoformat() if isinstance(value, date) else None


def _is_done(t: dict, done: tuple) -> bool:
    return t["status"].casefold() in done or t["status_type"] in DONE_TYPES


def plan_clickup(store: Store, config, tasks: list[dict], *, today: date) -> dict:
    conf = (config.sources or {}).get("clickup") or {}
    done = tuple(str(s).casefold() for s in (conf.get("done_statuses") or DEFAULT_DONE))
    idx = _people_index(config.people)
    by_id = {t["id"]: t for t in tasks}
    plan: dict = {"set": [], "proposals": [], "flags": [], "push": []}
    linked: set[str] = set()
    sharing: dict[str, list[str]] = {}
    for a in store.query(type="action"):
        if a.meta.get("clickup_id") not in (None, ""):
            sharing.setdefault(str(a.meta["clickup_id"]), []).append(a.id)
    for cid, ids in sharing.items():
        if len(ids) > 1:
            plan["flags"].append(f"{FLAG_PREFIX}actions {', '.join(sorted(ids))} share task {cid}")
    for a in store.query(type="action"):
        cid = a.meta.get("clickup_id")
        if not cid:
            if a.status == "open" and a.owner:
                person = config.people.get(a.owner)
                plan["push"].append({"id": a.id, "title": a.title, "owner": a.owner,
                                     "assignee": person.get("clickup") if isinstance(person, dict) else None,
                                     "due": _iso(a.meta.get("due"))})
            continue
        cid = str(cid)
        linked.add(cid)
        t = by_id.get(cid)
        if t is None:
            plan["flags"].append(f"{FLAG_PREFIX}task {cid} for {a.id} ({a.title}) is not in the export "
                                 f"(deleted or moved?)")
            continue
        fields: dict = {}
        entry = None
        task_done = _is_done(t, done)
        if task_done and a.status == "open":
            fields.update(status="done", closed=today.isoformat())
            entry = f"Closed in ClickUp (status '{t['status']}')."
        elif not task_done and a.status in ("done", "dropped"):
            plan["flags"].append(f"{FLAG_PREFIX}{a.id} is {a.status} in the hub but task {cid} is "
                                 f"'{t['status']}' in ClickUp")
        if t["assignees"]:
            owner = _owner_of(t["assignees"][0], idx)
            if owner is None:
                plan["flags"].append(f"{FLAG_PREFIX}assignee '{t['assignees'][0][0]}' of task {cid} is not in "
                                     f"project.yaml people (add their ClickUp id as `clickup:`)")
            elif owner != a.owner:
                fields["owner"] = owner
        if t["due"] != _iso(a.meta.get("due")):
            fields["due"] = t["due"] or ""
        if fields:
            plan["set"].append({"id": a.id, "fields": fields, "entry": entry, "source": f"clickup:{cid}"})
    pending_sources = {p.body.get("source") for p in load_proposals(config.root)}
    decided = _decided_titles(config.root)
    for t in tasks:
        if t["id"] in linked or _is_done(t, done):
            continue
        title = f"New ClickUp task: {' '.join(t['name'].split())}"
        if _key(title) in decided:
            continue
        owner = None
        if t["assignees"]:
            owner = _owner_of(t["assignees"][0], idx)
            if owner is None:
                plan["flags"].append(f"{FLAG_PREFIX}assignee '{t['assignees'][0][0]}' of new task {t['id']} "
                                     f"is not in project.yaml people")
        if f"clickup:{t['id']}" in pending_sources:
            continue
        fields = {"clickup_id": t["id"]}
        if t["due"]:
            fields["due"] = t["due"]
        plan["proposals"].append({
            "title": title, "source": f"clickup:{t['id']}",
            "reason": "Task created directly in ClickUp; after applying, link it (spawned_by) to the decision "
                      "it serves.",
            "ops": [{"op": "new", "type": "action", "title": t["name"], "owner": owner,
                     "fields": fields, "source": f"clickup:{t['id']}"}]})
    return plan


def apply_clickup(root, store: Store, plan: dict, *, author, day) -> dict:
    problems = []
    ids = store.ids()
    for s in plan["set"]:
        if s["id"] not in ids:
            problems.append(f"set target {s['id']} does not exist")
        if any(ch.isspace() for ch in s["source"]):
            problems.append(f"source {s['source']!r} must not contain whitespace")
    for p in plan["proposals"]:
        if any(ch.isspace() for ch in p["source"]):
            problems.append(f"source {p['source']!r} must not contain whitespace")
        problems += [f"{p['title']}: {err}" for err in check_ops(store, p["ops"])]
    if problems:
        raise ClickUpError("nothing written:\n  " + "\n  ".join(problems))
    for s in plan["set"]:
        if s.get("entry"):
            item = store.get(s["id"])
            if not any(e.source == s["source"] and e.text == s["entry"] for e in item.entries):
                store.entry(s["id"], s["entry"], author=author, source=s["source"], day=day)
        store.set_fields(s["id"], s["fields"])
    staged = [add_proposal(root, p["title"], p["ops"], proposer=author, source=p["source"], reason=p["reason"],
                           day=day) for p in plan["proposals"]]
    state = load_state(root)
    state["flags"] = [f for f in state["flags"] if not f.startswith(FLAG_PREFIX)] + list(plan["flags"])
    save_state(root, state)
    return {"updated": [s["id"] for s in plan["set"]], "staged": staged, "flags": plan["flags"],
            "push": plan["push"]}
