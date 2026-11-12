"""Reconcile a ClickUp list export with the hub's actions (ClickUp owns task status, assignee and due date).

Input: a JSON (or YAML) list the agent exported from the list with the ClickUp connector:
  [{id, name, status: <str | {status: str}>, assignees: [<id | username | email | {id, username, email}>],
    due: YYYY-MM-DD | null}]
Configuration: project.yaml `sources.clickup: {list_id, done_statuses?}`; `people.<key>.clickup` = user id.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from .pending import add_proposal, load_proposals
from .store import Store
from .sync_state import load_state, save_state

DEFAULT_DONE = ("complete", "completed", "closed", "done")
FLAG_PREFIX = "clickup: "


class ClickUpError(ValueError):
    """The task export is malformed."""


def _due(task_id, value):
    if value in (None, ""):
        return None
    text = value.isoformat() if isinstance(value, date) else str(value)
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise ClickUpError(f"task {task_id}: due must be YYYY-MM-DD or null, got {value!r}") from exc


def load_tasks(path) -> list[dict]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ClickUpError("the task export must be a list of tasks")
    tasks = []
    for i, t in enumerate(raw):
        if not isinstance(t, dict) or not t.get("id") or not t.get("name"):
            raise ClickUpError(f"task {i + 1}: needs id and name")
        status = t.get("status")
        if isinstance(status, dict):
            status = status.get("status")
        assignees = []
        for a in t.get("assignees") or []:
            if isinstance(a, dict):
                assignees.append([str(a[k]) for k in ("id", "username", "email") if a.get(k) not in (None, "")])
            else:
                assignees.append([str(a)])
        tasks.append({"id": str(t["id"]), "name": str(t["name"]), "status": str(status or ""),
                      "assignees": [al for al in assignees if al], "due": _due(t["id"], t.get("due"))})
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


def plan_clickup(store: Store, config, tasks: list[dict], *, today: date) -> dict:
    conf = (config.sources or {}).get("clickup") or {}
    done = tuple(str(s).casefold() for s in (conf.get("done_statuses") or DEFAULT_DONE))
    idx = _people_index(config.people)
    by_id = {t["id"]: t for t in tasks}
    plan: dict = {"set": [], "proposals": [], "flags": [], "push": []}
    linked: set[str] = set()
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
        task_done = t["status"].casefold() in done
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
    pending = {p.title for p in load_proposals(config.root)}
    for t in tasks:
        if t["id"] in linked or t["status"].casefold() in done:
            continue
        title = f"New ClickUp task: {' '.join(t['name'].split())}"
        if title in pending:
            continue
        fields = {"clickup_id": t["id"]}
        if t["due"]:
            fields["due"] = t["due"]
        plan["proposals"].append({
            "title": title, "source": f"clickup:{t['id']}",
            "reason": "Task created directly in ClickUp; link it (spawned_by) to the decision it serves, then apply.",
            "ops": [{"op": "new", "type": "action", "title": t["name"],
                     "owner": _owner_of(t["assignees"][0], idx) if t["assignees"] else None,
                     "fields": fields, "source": f"clickup:{t['id']}"}]})
    return plan


def apply_clickup(root, store: Store, plan: dict, *, author, day) -> dict:
    for s in plan["set"]:
        store.set_fields(s["id"], s["fields"])
        if s.get("entry"):
            store.entry(s["id"], s["entry"], author=author, source=s["source"], day=day)
    staged = [add_proposal(root, p["title"], p["ops"], proposer=author, source=p["source"], reason=p["reason"],
                           day=day) for p in plan["proposals"]]
    state = load_state(root)
    state["flags"] = [f for f in state["flags"] if not f.startswith(FLAG_PREFIX)] + list(plan["flags"])
    save_state(root, state)
    return {"updated": [s["id"] for s in plan["set"]], "staged": staged, "flags": plan["flags"],
            "push": plan["push"]}
