# project-hub/hub/agenda/lifecycle.py
"""Agenda lifecycle: build (with the 'previous meeting not read back' guard) and read back into the store."""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from ..config import HubConfig
from ..pending import add_proposal
from ..store import Store
from .build import build_workbook, set_read_back
from .draft import load_draft
from .read import parse_actions, read_workbook

VERB = {"approve": "Approved", "modify": "Modified"}
RAISED_STATUS = {"approve": "made", "modify": "made", "defer": "deferred", "reject": "rejected", None: "made"}


class AgendaError(Exception):
    """The agenda lifecycle would lose meeting edits or act on the wrong file."""


def live_path(root, series: str) -> Path:
    return Path(root) / "Agenda" / f"{series}.xlsx"


def build_agenda(root, series: str, *, force: bool = False) -> Path:
    draft = load_draft(root, series)
    live = live_path(root, series)
    if live.is_file():
        rb = read_workbook(live)
        if rb.series and rb.series != series:
            raise AgendaError(f"{live} belongs to series '{rb.series}', not '{series}'")
        if (not force and not rb.read_back and rb.meeting_date and rb.meeting_date < draft["meeting_date"]):
            raise AgendaError(f"the {rb.meeting_date} agenda has not been read back; run "
                              f"`hub agenda read {series}` first (or pass --force to discard its edits)")
    build_workbook(draft, live)
    return live


def _action_ops(text, parent, series, source):
    return [{"op": "new", "type": "action", "title": t, "owner": o, "series": [series],
             "links": {"spawned_by": [parent]}, "source": source} for t, o in parse_actions(text)]


def read_agenda(store: Store, config: HubConfig, series: str, *, author: str, today: date) -> dict:
    root = Path(config.root)
    live = live_path(root, series)
    if not live.is_file():
        raise AgendaError(f"no live agenda at {live}")
    rb = read_workbook(live)
    if rb.read_back:
        raise AgendaError(f"{live} was already read back")
    if rb.series and rb.series != series:
        raise AgendaError(f"{live} belongs to series '{rb.series}', not '{series}'")
    if rb.meeting_date is None:
        raise AgendaError(f"{live} has no meeting date (neither _meta nor sheet title)")
    tag = f"{series}/{rb.meeting_date.isoformat()}"
    source = f"agenda:{tag}"
    decisions = {i.id: i for i in store.all() if i.type == "decision"}
    by_title = {d.title: d.id for d in decisions.values() if d.status == "proposed"}
    applied, staged, not_reached, lines = [], [], [], []
    for row in rb.rows:
        did = row.id if row.planned and row.id in decisions else by_title.get(row.decide)
        if did is None:
            if row.outcome or row.as_decided or row.outcome_text:
                ops = [{"op": "new", "ref": "$d", "type": "decision", "title": row.decide, "owner": author,
                        "series": [series], "source": source,
                        "fields": {"status": RAISED_STATUS[row.outcome], "meeting": tag,
                                   "as_decided": row.as_decided}}]
                ops += _action_ops(row.new_actions, "$d", series, source)
                pid = add_proposal(root, f"Raised at {tag}: {row.decide}", ops, proposer=author, source=source,
                                   reason="Decision raised in meeting; link it to the question(s) it answers before applying.",
                                   day=today)
                staged.append((pid, f"Raised at {tag}: {row.decide}"))
            continue
        if row.outcome in ("approve", "modify"):
            fields = {"status": "made", "meeting": tag}
            if row.as_decided:
                fields["as_decided"] = row.as_decided
            store.set_fields(did, fields)
            store.entry(did, f"{VERB[row.outcome]} at {tag}: {row.as_decided or 'as proposed'}",
                        author=author, source=source, day=today)
            if row.new_actions:
                title = f"Actions from {tag}: {decisions[did].title}"
                pid = add_proposal(root, title, _action_ops(row.new_actions, did, series, source),
                                   proposer=author, source=source, reason="Free-text actions written in the meeting.",
                                   day=today)
                staged.append((pid, title))
        elif row.outcome == "defer":
            store.set_fields(did, {"status": "deferred"})
            store.entry(did, f"Deferred at {tag}.", author=author, source=source, day=today)
        elif row.outcome == "reject":
            store.set_fields(did, {"status": "rejected", "closed": today})
            store.entry(did, f"Rejected at {tag}: {row.as_decided or 'no reason recorded'}",
                        author=author, source=source, day=today)
        elif row.outcome_text:
            title = f"Unrecognised outcome at {tag}: {decisions[did].title}"
            ops = [{"op": "entry", "id": did, "source": source,
                    "text": f"Outcome typed in meeting: '{row.outcome_text}'; as decided: {row.as_decided or '—'}"}]
            pid = add_proposal(root, title, ops, proposer=author, source=source,
                               reason="Outcome cell not recognised; set the decision status by hand, then apply.",
                               day=today)
            staged.append((pid, title))
            continue
        else:
            store.entry(did, f"Not reached at {tag}.", author=author, source=source, day=today)
            not_reached.append(did)
            continue
        applied.append(did)
        lines.append(f"- [[{did}]] {row.outcome} — {row.as_decided or decisions[did].title}")
    attendees = ", ".join((config.series.get(series) or {}).get("attendees") or [])
    note = [f"# {series} — {rb.meeting_date.isoformat()}", f"- attendees: {attendees}", f"- source: {source}",
            "## Decisions", *(lines or ["None."]),
            "## Not reached", *([f"- [[{i}]]" for i in not_reached] or ["None."]),
            "## Staged for approval", *([f"- {p} · {t}" for p, t in staged] or ["None."])]
    note_path = root / "Status" / "Meetings" / series / f"{rb.meeting_date.isoformat()}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(note) + "\n")
    set_read_back(live, True)
    archive = root / "Agenda" / "archive" / series / f"{rb.meeting_date.isoformat()}.xlsx"
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(live, archive)
    store.log(f"Agenda read back: {len(applied)} applied, {len(staged)} staged", author=author,
              source=source, day=today)
    return {"applied": applied, "staged": [p for p, _ in staged], "not_reached": not_reached,
            "meeting_note": note_path, "archive": archive}
