# project-hub/hub/agenda/draft.py
"""Agenda draft: the content of one meeting's workbook, pre-filled from the store for the agent to finish."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from ..snapshot import last_meeting
from ..store import Store

SINCE_OUTCOMES = ("closed question", "new question", "pivot", "in progress")
MAX_DECISIONS = 7
REQUIRED = ("series", "meeting_date", "previous_meeting", "summary", "since_last", "questions", "decisions")


class DraftError(Exception):
    """The draft is missing or invalid."""


def _in_series(item, series) -> bool:
    s = item.meta.get("series") or []
    return not s or series in s


def _outcome(d, by_id: dict, items) -> str:
    qs = [by_id[q] for q in d.links("answers") if q in by_id]
    if any(q.status == "pivoted" for q in qs):
        return "pivot"
    if qs and all(q.status in ("answered", "dropped") for q in qs):
        return "closed question"
    if any(i.type == "question" and d.id in i.links("spawned_by") for i in items):
        return "new question"
    return "in progress"


def _raised(item, fallback: date) -> date:
    r = item.meta.get("raised")
    return r if isinstance(r, date) else fallback


def scaffold(store: Store, root, series: str, meeting_date: date) -> dict:
    items = store.all()
    by_id = {i.id: i for i in items}
    prev = last_meeting(root, series, before=meeting_date)
    since_last = []
    if prev:
        tag = f"{series}/{prev.isoformat()}"
        for d in items:
            if d.type == "decision" and d.status == "made" and d.meta.get("meeting") == tag:
                acts = [a for a in items if a.type == "action" and d.id in a.links("spawned_by")]
                since_last.append({
                    "id": d.id, "decision": d.meta.get("as_decided") or d.title,
                    "actions": "; ".join(f"{a.title} ({a.owner}, {a.status})" for a in acts),
                    "follow_up": "", "outcome": _outcome(d, by_id, items),
                    "affected": ", ".join(d.links("answers")),
                })
    proposed = [d for d in items if d.type == "decision" and d.status == "proposed" and _in_series(d, series)]
    proposed_for = {}
    for d in proposed:
        for q in d.links("answers"):
            proposed_for.setdefault(q, d.id)
    qs = [q for q in items if q.type == "question" and q.status == "open" and _in_series(q, series)]
    qs.sort(key=lambda q: (q.id not in proposed_for, q.meta.get("priority") or 9, _raised(q, meeting_date)))
    questions = [{"id": q.id, "title": q.title, "description": q.description, "owner": q.owner,
                  "age_days": (meeting_date - _raised(q, meeting_date)).days, "priority": q.meta.get("priority"),
                  "proposed_decision": proposed_for.get(q.id, "")} for q in qs]
    proposed.sort(key=lambda d: (d.meta.get("priority") or 9, _raised(d, meeting_date)))
    decisions = [{"id": d.id, "decide": d.title, "proposed_action": d.meta.get("proposed_action", ""),
                  "reason": d.meta.get("rationale", ""), "weigh_in": "", "minutes": 10, "link": ""}
                 for d in proposed[:MAX_DECISIONS]]
    return {"series": series, "meeting_date": meeting_date, "previous_meeting": prev, "summary": "",
            "since_last": since_last, "questions": questions, "decisions": decisions}


def _nonblank(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


def validate_draft(draft: dict) -> list[str]:
    if not isinstance(draft, dict):
        return ["draft is not a mapping"]
    errs = [f"missing key '{k}'" for k in REQUIRED if k not in draft]
    if errs:
        return errs
    if not _nonblank(draft["series"]):
        errs.append("series must be a non-empty string")
    if not isinstance(draft["meeting_date"], date):
        errs.append("meeting_date must be a date")
    if not _nonblank(draft["summary"]):
        errs.append("summary is empty")
    for key in ("since_last", "questions", "decisions"):
        if not isinstance(draft[key], list):
            errs.append(f"'{key}' must be a list")
    if isinstance(draft["since_last"], list):
        for i, row in enumerate(draft["since_last"]):
            if not isinstance(row, dict):
                errs.append(f"since_last[{i}] is not a mapping")
            elif row.get("outcome") not in SINCE_OUTCOMES:
                errs.append(f"since_last {row.get('id')}: outcome '{row.get('outcome')}' not in {SINCE_OUTCOMES}")
    if isinstance(draft["questions"], list):
        for i, row in enumerate(draft["questions"]):
            if not isinstance(row, dict):
                errs.append(f"questions[{i}] is not a mapping")
    if isinstance(draft["decisions"], list):
        if len(draft["decisions"]) > MAX_DECISIONS:
            errs.append(f"decisions: at most {MAX_DECISIONS} allowed, got {len(draft['decisions'])}")
        for i, row in enumerate(draft["decisions"]):
            if not isinstance(row, dict):
                errs.append(f"decisions[{i}] is not a mapping")
                continue
            if not _nonblank(row.get("decide")):
                errs.append(f"decisions[{i}]: decide must be a non-empty string")
            minutes = row.get("minutes")
            if not isinstance(minutes, int) or isinstance(minutes, bool):
                errs.append(f"decisions[{i}]: minutes must be an int")
    return errs


def draft_path(root, series: str) -> Path:
    return Path(root) / "Agenda" / f"{series}.draft.yaml"


def save_draft(root, draft: dict) -> Path:
    p = draft_path(root, draft["series"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(draft, sort_keys=False, allow_unicode=True))
    return p


def load_draft(root, series: str) -> dict:
    p = draft_path(root, series)
    if not p.is_file():
        raise DraftError(f"no draft at {p}; run `hub agenda scaffold {series} --date YYYY-MM-DD` first")
    try:
        data = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise DraftError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise DraftError(f"{p} does not contain a mapping")
    return data
