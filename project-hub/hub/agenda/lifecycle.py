# project-hub/hub/agenda/lifecycle.py
"""Agenda lifecycle: build (with the 'previous meeting not read back' guard) and read back into the store."""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from ..config import HubConfig
from ..fsutil import atomic_write_text
from ..pending import add_proposal
from ..store import Store
from .build import build_workbook, set_read_back
from .draft import draft_path, load_draft
from .read import parse_actions, read_workbook

VERB = {"approve": "Approved", "modify": "Modified"}
RAISED_STATUS = {"approve": "made", "modify": "made", "defer": "deferred", "reject": "rejected", None: "proposed"}
UNTITLED_CHARS = 60


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
        if not force and not rb.read_back and rb.meeting_date is None:
            raise AgendaError("live agenda has no readable meeting date; pass --force to overwrite")
        if (not force and not rb.read_back and rb.meeting_date and rb.meeting_date < draft["meeting_date"]):
            raise AgendaError(f"the {rb.meeting_date} agenda has not been read back; run "
                              f"`hub agenda read {series}` first (or pass --force to discard its edits)")
        if not force and not rb.read_back and any(_has_input(r) for r in rb.rows):
            raise AgendaError(f"live agenda already contains meeting input; run `hub agenda read {series}` "
                              "or pass --force")
    build_workbook(draft, live)
    return live


def _has_input(row) -> bool:
    return bool(row.outcome_text or row.as_decided or row.new_actions or (not row.planned and row.decide))


def owner_key(raw, people: dict):
    """Map an owner typed in a meeting ("Bob", "ALICE", "Bob Smith") to a people key, case-insensitively,
    by key or by the person's `name`. Unmatched owners are returned unchanged (validate flags them)."""
    if not raw:
        return raw
    want = raw.strip().casefold()
    for key, person in people.items():
        name = person.get("name") if isinstance(person, dict) else None
        if str(key).casefold() == want or (isinstance(name, str) and name.strip().casefold() == want):
            return key
    return raw


def _action_ops(text, parent, series, source, people):
    return [{"op": "new", "type": "action", "title": t, "owner": owner_key(o, people), "series": [series],
             "links": {"spawned_by": [parent]}, "source": source} for t, o in parse_actions(text)]


def _same_date(value, day: date) -> bool:
    return (value.isoformat() if isinstance(value, date) else str(value)) == day.isoformat()


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
    if draft_path(root, series).is_file():
        draft = load_draft(root, series)
        if not _same_date(draft.get("meeting_date"), rb.meeting_date):
            raise AgendaError(f"workbook is for {rb.meeting_date.isoformat()}, "
                              f"draft is for {draft.get('meeting_date')}")
    tag = f"{series}/{rb.meeting_date.isoformat()}"
    source = f"agenda:{tag}"
    note_head = f"# {series} — {rb.meeting_date.isoformat()}"
    note_path = root / "Status" / "Meetings" / series / f"{rb.meeting_date.isoformat()}.md"
    if note_path.exists():
        raise AgendaError(f"meeting note {note_path} already exists — a previous read-back of this meeting started "
                          "(and may have stopped part-way); inspect the note and the items' entries, then remove "
                          "the note to retry")
    # Marker first: if anything below fails, the note's existence stops a blind retry from double-applying.
    atomic_write_text(note_path, f"{note_head}\n- status: read-back in progress\n")
    people = config.people
    decisions = {i.id: i for i in store.all() if i.type == "decision"}
    by_title = {d.title: d.id for d in decisions.values() if d.status in ("proposed", "deferred")}
    applied, staged, not_reached, lines = [], [], [], []
    for row in rb.rows:
        did = row.id if row.planned and row.id in decisions else (by_title.get(row.decide) if row.decide else None)
        if did is None:
            if row.outcome or row.as_decided or row.outcome_text or row.new_actions.strip():
                snippet = (row.as_decided or row.new_actions or row.outcome_text)[:UNTITLED_CHARS]
                title = row.decide or f"(untitled: {snippet})"
                ops = [{"op": "new", "ref": "$d", "type": "decision", "title": title, "owner": author,
                        "series": [series], "source": source,
                        "fields": {"status": RAISED_STATUS[row.outcome], "meeting": tag,
                                   "as_decided": row.as_decided}}]
                ops += _action_ops(row.new_actions, "$d", series, source, people)
                pid = add_proposal(root, f"Raised at {tag}: {title}", ops, proposer=author, source=source,
                                   reason="Decision raised in meeting; link it to the question(s) it answers before applying.",
                                   day=today)
                staged.append((pid, f"Raised at {tag}: {title}"))
            continue
        # A planned row whose "What to decide" was edited: keep the meeting's wording.
        as_decided, wording = row.as_decided, ""
        if row.decide and row.decide != decisions[did].title:
            if not as_decided and row.outcome in ("approve", "modify", "reject"):
                as_decided = row.decide
            else:
                wording = f' (wording in meeting: "{row.decide}")'
        if row.outcome and decisions[did].status not in ("proposed", "deferred"):
            title = f"Outcome for already-resolved decision at {tag}: {decisions[did].title}"
            ops = [{"op": "entry", "id": did, "source": source,
                    "text": f"Outcome typed in meeting: '{row.outcome_text}'; as decided: {as_decided or '—'}"
                            + (f"; new actions: {row.new_actions}" if row.new_actions else "") + wording}]
            pid = add_proposal(root, title, ops, proposer=author, source=source,
                               reason=f"Decision is already '{decisions[did].status}'; not changed by read-back.",
                               day=today)
            staged.append((pid, title))
            continue
        if row.outcome in ("approve", "modify"):
            fields = {"status": "made", "meeting": tag}
            if as_decided:
                fields["as_decided"] = as_decided
            store.set_fields(did, fields)
            store.entry(did, f"{VERB[row.outcome]} at {tag}: {as_decided or 'as proposed'}{wording}",
                        author=author, source=source, day=today)
            if row.new_actions:
                title = f"Actions from {tag}: {decisions[did].title}"
                pid = add_proposal(root, title, _action_ops(row.new_actions, did, series, source, people),
                                   proposer=author, source=source, reason="Free-text actions written in the meeting.",
                                   day=today)
                staged.append((pid, title))
        elif row.outcome == "defer":
            store.set_fields(did, {"status": "deferred"})
            store.entry(did, f"Deferred at {tag}.{wording}", author=author, source=source, day=today)
        elif row.outcome == "reject":
            store.set_fields(did, {"status": "rejected", "closed": today})
            store.entry(did, f"Rejected at {tag}: {as_decided or 'no reason recorded'}{wording}",
                        author=author, source=source, day=today)
        elif row.outcome_text:
            title = f"Unrecognised outcome at {tag}: {decisions[did].title}"
            ops = [{"op": "entry", "id": did, "source": source,
                    "text": f"Outcome typed in meeting: '{row.outcome_text}'; as decided: {as_decided or '—'}"
                            + wording}]
            pid = add_proposal(root, title, ops, proposer=author, source=source,
                               reason="Outcome cell not recognised; set the decision status by hand, then apply.",
                               day=today)
            staged.append((pid, title))
            continue
        else:
            if row.as_decided or row.new_actions.strip():
                title = f"Notes without outcome at {tag}: {decisions[did].title}"
                text = f"Notes typed in meeting without an outcome; as decided: {row.as_decided or '—'}"
                if row.new_actions.strip():
                    text += "; new actions: " + "; ".join(
                        f"{t} ({o})" if o else t for t, o in parse_actions(row.new_actions))
                pid = add_proposal(root, title, [{"op": "entry", "id": did, "text": text + wording,
                                                  "source": source}],
                                   proposer=author, source=source,
                                   reason="Meeting notes were written but no outcome was set.", day=today)
                staged.append((pid, title))
            store.entry(did, f"Not reached at {tag}.{wording}", author=author, source=source, day=today)
            not_reached.append(did)
            continue
        applied.append(did)
        lines.append(f"- [[{did}]] {row.outcome} — {as_decided or decisions[did].title}")
    attendees = ", ".join((config.series.get(series) or {}).get("attendees") or [])
    note = [note_head, f"- attendees: {attendees}", f"- source: {source}",
            "## Decisions", *(lines or ["None."]),
            "## Not reached", *([f"- [[{i}]]" for i in not_reached] or ["None."]),
            "## Staged for approval", *([f"- {p} · {t}" for p, t in staged] or ["None."])]
    atomic_write_text(note_path, "\n".join(note) + "\n")
    set_read_back(live, True)
    archive = root / "Agenda" / "archive" / series / f"{rb.meeting_date.isoformat()}.xlsx"
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(live, archive)
    store.log(f"Agenda read back: {len(applied)} applied, {len(staged)} staged", author=author,
              source=source, day=today)
    return {"applied": applied, "staged": [p for p, _ in staged], "not_reached": not_reached,
            "meeting_note": note_path, "archive": archive}
