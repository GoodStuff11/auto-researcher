# Meeting Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the regenerate-from-a-prompt React artifact with a local, SQLite-backed meeting dashboard that lives in the Overleaf repo beside `HANDOFF.md`, serves a three-section UI on localhost, and is refreshed by reconciling against `HANDOFF.md` §7 rather than re-authoring it.

**Architecture:** A Python standard-library package. `sources/` adapters parse `HANDOFF.md` §7 and a cached ClickUp snapshot into `Item` records. `sync.py` reconciles those into a SQLite store under ownership rules (tracker owns open/closed and descriptive fields; the UI owns resolutions, agenda and decisions). `export.py` keeps a committed, byte-deterministic `dashboard.json` in step with the database and can rebuild the database from it, which is what makes a fresh clone usable. `serve.py` exposes a JSON API and a static UI where every interaction writes through to the store.

**Tech Stack:** Python 3.12 standard library only — `sqlite3`, `http.server`, `json`, `re`, `argparse`, `dataclasses`. `pytest` 7.4 for tests. Plain HTML/CSS/JS for the UI, no build step, no npm, no pip install.

**Spec:** `docs/superpowers/specs/2026-09-21-meeting-dashboard-design.md` (in the `research` repo; read it alongside this plan)

## Global Constraints

- **Implementation repo is the Overleaf clone, not `research`.** All paths below are relative to `/home/jek354/research/68f6bef0eaaf5c0928a922c6/`. That directory is a git repo whose remote is Overleaf.
- **Never `git push` in that repo without asking the user.** Pushing publishes to Overleaf and changes what Eun-Ah and Tamra see. Commit locally; the user decides when it goes out.
- **Standard library only.** No third-party runtime imports anywhere under `dashboard/project_tracker/`. `pytest` is a test-time dependency and must not be imported by package code.
- **Python 3.12.7**, `sqlite3` 3.45.3. Do not use features newer than 3.12.
- **Item numbers are never minted locally.** A question's `id` is the `HANDOFF.md` §7 item number verbatim, as a string (`"14"`, `"14b"`). Closed items keep their numbers.
- **`grounds` on a decision is required.** Creating a decision without non-empty grounds raises `ValueError`. This is a group convention, not a nicety.
- **`dashboard.json` must be byte-deterministic** given store state: `json.dumps(..., indent=2, sort_keys=True, ensure_ascii=False)` plus a trailing newline, rows sorted by a stable key, and **no wall-clock timestamp in the document**. A no-op sync must produce a zero-line git diff.
- **Flags are listed, never auto-resolved.** Nothing in this codebase may mark a `§7.1` flag resolved on its own.
- **Run tests with:** `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/ -v`

---

## File Structure

| Path (under `dashboard/`) | Responsibility |
|---|---|
| `dashboard` | Executable CLI: `sync`, `serve`, `update`, `export`. Argument parsing and path resolution only. |
| `README.md` | The two commands a human needs, at the top, with no agent involved. |
| `.gitignore` | Ignores `data/dashboard.db` and nothing else. |
| `project_tracker/store.py` | SQLite schema plus the CRUD functions agents and the API call. Owns all writes and the changelog. |
| `project_tracker/export.py` | Store ↔ `dashboard.json`, and `ensure_db` bootstrap-from-JSON. |
| `project_tracker/sources/__init__.py` | The `Item` dataclass and the adapter contract. |
| `project_tracker/sources/handoff.py` | Parses `HANDOFF.md` §7 and §7.1. The only module that knows §7's prose shape. |
| `project_tracker/sources/clickup.py` | Reads/writes the cached ClickUp snapshot; merges status, priority, due date onto items. |
| `project_tracker/sync.py` | Reconciliation rules, difficulty estimation, agenda candidate proposal. |
| `project_tracker/report.py` | Renders `pending-changes.md` from the changelog. |
| `project_tracker/serve.py` | HTTP request routing and JSON API. No business logic. |
| `project_tracker/static/` | `index.html`, `app.js`, `style.css`. |
| `tests/` | One test module per source module. |
| `data/` | `dashboard.db` (ignored), `dashboard.json`, `pending-changes.md`, `clickup-cache.json`. |

---

### Task 1: Store — schema, CRUD, changelog

**Files:**
- Create: `dashboard/project_tracker/__init__.py`
- Create: `dashboard/project_tracker/store.py`
- Create: `dashboard/tests/__init__.py`
- Create: `dashboard/tests/test_store.py`
- Create: `dashboard/.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: `connect(db_path) -> sqlite3.Connection`; `upsert_question(conn, *, id, title, why, owner, priority, raised, state, clickup_id=None, clickup_status=None, due=None, actor="sync") -> None`; `get_question(conn, qid) -> dict | None`; `list_questions(conn, state=None) -> list[dict]`; `set_question_state(conn, qid, state, *, source, note=None, on=None, actor) -> None`; `set_question_difficulty(conn, qid, difficulty, *, source, actor) -> None`; `add_agenda(conn, *, title, detail="", origin="manual", proposed_by="", created=None, status="proposed", actor="ui") -> int`; `update_agenda(conn, aid, *, actor="ui", **fields) -> None`; `reorder_agenda(conn, ordered_ids, *, actor="ui") -> None`; `close_agenda(conn, aid, *, resolution, resolved_by, followups=None, resolved_on=None, actor="ui") -> int`; `list_agenda(conn, status=None) -> list[dict]`; `add_decision(conn, *, text, grounds, decided_on, decided_by, followups=None, from_agenda_id=None, source="ui", actor="ui") -> int`; `list_decisions(conn) -> list[dict]`; `upsert_flag(conn, key, text, first_seen, resolved=0) -> None`; `list_flags(conn) -> list[dict]`; `log(conn, entity, entity_id, field, old, new, actor) -> None`; `list_changes(conn, since_id=0) -> list[dict]`; `set_meta(conn, key, value) -> None`; `get_meta(conn, key, default=None)`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/tests/__init__.py` as an empty file, then `dashboard/tests/test_store.py`:

```python
import sqlite3

import pytest

from project_tracker.store import (
    add_agenda,
    add_decision,
    close_agenda,
    connect,
    get_meta,
    get_question,
    list_agenda,
    list_changes,
    list_decisions,
    list_flags,
    list_questions,
    reorder_agenda,
    set_meta,
    set_question_difficulty,
    set_question_state,
    update_agenda,
    upsert_flag,
    upsert_question,
)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "dashboard.db")
    yield c
    c.close()


def _q(conn, qid="14", **kw):
    base = dict(
        id=qid, title="HVA benchmark", why="Gates the venue.", owner="Tamra",
        priority="high", raised="2026-09-14", state="open",
    )
    base.update(kw)
    upsert_question(conn, **base)
    return qid


def test_upsert_question_then_get_round_trips(conn):
    _q(conn)
    row = get_question(conn, "14")
    assert row["id"] == "14"
    assert row["title"] == "HVA benchmark"
    assert row["state"] == "open"
    assert row["state_source"] == "tracker"


def test_upsert_question_overwrites_tracker_fields_but_keeps_ui_fields(conn):
    _q(conn)
    set_question_difficulty(conn, "14", "L", source="user", actor="ui")
    _q(conn, title="HVA benchmark, restated", owner="Jonathon")
    row = get_question(conn, "14")
    assert row["title"] == "HVA benchmark, restated"
    assert row["owner"] == "Jonathon"
    assert row["difficulty"] == "L"
    assert row["difficulty_source"] == "user"


def test_question_id_is_a_string_and_sub_items_are_preserved(conn):
    _q(conn, qid="14b")
    assert get_question(conn, "14b")["id"] == "14b"


def test_list_questions_filters_by_state(conn):
    _q(conn, qid="1")
    _q(conn, qid="2", state="closed")
    assert [r["id"] for r in list_questions(conn, state="open")] == ["1"]
    assert [r["id"] for r in list_questions(conn, state="closed")] == ["2"]


def test_set_question_state_records_source_note_and_date(conn):
    _q(conn)
    set_question_state(conn, "14", "closed", source="ui", note="settled in the meeting",
                       on="2026-09-21", actor="ui")
    row = get_question(conn, "14")
    assert row["state"] == "closed"
    assert row["state_source"] == "ui"
    assert row["closed_note"] == "settled in the meeting"
    assert row["closed_on"] == "2026-09-21"


def test_set_question_state_writes_a_changelog_entry(conn):
    _q(conn)
    set_question_state(conn, "14", "closed", source="ui", actor="ui")
    entries = [e for e in list_changes(conn) if e["entity"] == "question"]
    assert len(entries) == 1
    assert entries[0]["field"] == "state"
    assert entries[0]["old"] == "open"
    assert entries[0]["new"] == "closed"
    assert entries[0]["actor"] == "ui"


def test_set_question_state_on_unknown_id_raises(conn):
    with pytest.raises(KeyError):
        set_question_state(conn, "999", "closed", source="ui", actor="ui")


def test_add_agenda_returns_id_and_defaults_to_proposed(conn):
    aid = add_agenda(conn, title="Settle the cluster count", origin="sync:flag:1")
    row = list_agenda(conn)[0]
    assert row["id"] == aid
    assert row["status"] == "proposed"
    assert row["origin"] == "sync:flag:1"


def test_reorder_agenda_sets_sort_order_in_the_given_sequence(conn):
    a = add_agenda(conn, title="first")
    b = add_agenda(conn, title="second")
    reorder_agenda(conn, [b, a])
    assert [r["id"] for r in list_agenda(conn)] == [b, a]


def test_close_agenda_creates_one_decision_carrying_resolution_as_grounds(conn):
    aid = add_agenda(conn, title="Which loss for the 4x4 runs?", status="accepted")
    did = close_agenda(conn, aid, resolution="overlap, matching panels A and B",
                       resolved_by="Eun-Ah", followups=["Jonathon"], resolved_on="2026-09-21")
    decisions = list_decisions(conn)
    assert len(decisions) == 1
    assert decisions[0]["id"] == did
    assert decisions[0]["grounds"] == "overlap, matching panels A and B"
    assert decisions[0]["from_agenda_id"] == aid
    assert decisions[0]["followups"] == ["Jonathon"]
    assert list_agenda(conn, status="closed")[0]["id"] == aid


def test_close_agenda_without_a_resolution_raises(conn):
    aid = add_agenda(conn, title="something", status="accepted")
    with pytest.raises(ValueError):
        close_agenda(conn, aid, resolution="   ", resolved_by="Eun-Ah")


def test_add_decision_without_grounds_raises(conn):
    with pytest.raises(ValueError):
        add_decision(conn, text="We do X", grounds="", decided_on="2026-09-21",
                     decided_by="Eun-Ah")


def test_list_decisions_sorts_newest_first(conn):
    add_decision(conn, text="older", grounds="g", decided_on="2026-09-11", decided_by="E")
    add_decision(conn, text="newer", grounds="g", decided_on="2026-09-14", decided_by="E")
    assert [d["text"] for d in list_decisions(conn)] == ["newer", "older"]


def test_upsert_flag_is_idempotent_on_key_and_keeps_first_seen(conn):
    upsert_flag(conn, key="1", text="Ten clusters or fourteen?", first_seen="2026-09-21")
    upsert_flag(conn, key="1", text="Ten clusters or fourteen?", first_seen="2026-09-28")
    flags = list_flags(conn)
    assert len(flags) == 1
    assert flags[0]["first_seen"] == "2026-09-21"


def test_meta_round_trips_structured_values(conn):
    set_meta(conn, "meetings", [{"date": "2026-09-14", "label": "Frame and novelty"}])
    assert get_meta(conn, "meetings")[0]["date"] == "2026-09-14"
    assert get_meta(conn, "absent", default="fallback") == "fallback"


def test_update_agenda_rejects_unknown_columns(conn):
    aid = add_agenda(conn, title="x")
    with pytest.raises(ValueError):
        update_agenda(conn, aid, nonsense="boom")


def test_connect_is_idempotent_on_an_existing_database(tmp_path):
    path = tmp_path / "d.db"
    c1 = connect(path)
    upsert_question(c1, id="1", title="t", why="w", owner="o", priority="", raised="",
                    state="open")
    c1.close()
    c2 = connect(path)
    assert get_question(c2, "1")["title"] == "t"
    c2.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_store.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'project_tracker'`

- [ ] **Step 3: Write the store**

Create `dashboard/project_tracker/__init__.py` as an empty file. Create `dashboard/project_tracker/store.py`:

```python
"""SQLite store for the meeting dashboard.

Every mutation goes through a function here, and every mutation that changes a
field a person would care about appends to `changelog`. `report.py` turns that
log into the pending-changes file, so a write that skips the log is a write
that silently loses meeting work.
"""

import json
import sqlite3
from datetime import date
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id                TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    why               TEXT NOT NULL DEFAULT '',
    owner             TEXT NOT NULL DEFAULT '',
    priority          TEXT NOT NULL DEFAULT '',
    raised            TEXT NOT NULL DEFAULT '',
    clickup_id        TEXT,
    clickup_status    TEXT,
    due               TEXT,
    state             TEXT NOT NULL DEFAULT 'open',
    state_source      TEXT NOT NULL DEFAULT 'tracker',
    difficulty        TEXT,
    difficulty_source TEXT,
    closed_on         TEXT,
    closed_note       TEXT,
    last_seen_sync    TEXT,
    updated_at        TEXT
);
CREATE TABLE IF NOT EXISTS agenda (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT NOT NULL,
    detail       TEXT NOT NULL DEFAULT '',
    origin       TEXT NOT NULL DEFAULT 'manual',
    proposed_by  TEXT NOT NULL DEFAULT '',
    created      TEXT NOT NULL DEFAULT '',
    sort_order   INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'proposed',
    resolution   TEXT,
    followups    TEXT,
    resolved_on  TEXT,
    resolved_by  TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS agenda_origin_unique
    ON agenda(origin) WHERE origin != 'manual';
CREATE TABLE IF NOT EXISTS decisions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    text           TEXT NOT NULL,
    grounds        TEXT NOT NULL,
    decided_on     TEXT NOT NULL,
    decided_by     TEXT NOT NULL DEFAULT '',
    followups      TEXT,
    from_agenda_id INTEGER,
    source         TEXT NOT NULL DEFAULT 'ui'
);
CREATE TABLE IF NOT EXISTS flags (
    key        TEXT PRIMARY KEY,
    text       TEXT NOT NULL,
    first_seen TEXT NOT NULL DEFAULT '',
    resolved   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS changelog (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    entity    TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    field     TEXT NOT NULL,
    old       TEXT,
    new       TEXT,
    actor     TEXT NOT NULL DEFAULT 'ui'
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_TRACKER_FIELDS = ("title", "why", "owner", "priority", "raised",
                   "clickup_id", "clickup_status", "due", "state")
_AGENDA_FIELDS = ("title", "detail", "origin", "proposed_by", "created", "sort_order",
                  "status", "resolution", "followups", "resolved_on", "resolved_by")


def connect(db_path):
    """Open (creating if needed) the store at `db_path` and apply the schema."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    _set_meta_raw(conn, "schema_version", str(SCHEMA_VERSION))
    return conn


def _today():
    return date.today().isoformat()


def _now():
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def log(conn, entity, entity_id, field, old, new, actor):
    """Append a changelog entry. Values are stringified; None stays None."""
    conn.execute(
        "INSERT INTO changelog (ts, entity, entity_id, field, old, new, actor)"
        " VALUES (?,?,?,?,?,?,?)",
        (_now(), entity, str(entity_id), field,
         None if old is None else str(old), None if new is None else str(new), actor),
    )
    conn.commit()


def list_changes(conn, since_id=0):
    rows = conn.execute("SELECT * FROM changelog WHERE id > ? ORDER BY id", (since_id,))
    return [dict(r) for r in rows]


def _set_meta_raw(conn, key, value):
    conn.execute("INSERT INTO meta (key, value) VALUES (?,?)"
                 " ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    conn.commit()


def set_meta(conn, key, value):
    _set_meta_raw(conn, key, json.dumps(value, sort_keys=True, ensure_ascii=False))


def get_meta(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return row["value"]


def _question_row(row):
    d = dict(row)
    return d


def get_question(conn, qid):
    row = conn.execute("SELECT * FROM questions WHERE id = ?", (str(qid),)).fetchone()
    return None if row is None else _question_row(row)


def list_questions(conn, state=None):
    if state is None:
        rows = conn.execute("SELECT * FROM questions ORDER BY id")
    else:
        rows = conn.execute("SELECT * FROM questions WHERE state = ? ORDER BY id", (state,))
    return [_question_row(r) for r in rows]


def upsert_question(conn, *, id, title, why="", owner="", priority="", raised="",
                    state="open", clickup_id=None, clickup_status=None, due=None,
                    actor="sync"):
    """Insert or refresh a question's tracker-owned fields.

    UI-owned fields (difficulty when set by a user, closed_note, closed_on) are
    never touched here. `state` is tracker-owned by design: see sync.py.
    """
    qid = str(id)
    incoming = dict(title=title, why=why, owner=owner, priority=priority, raised=raised,
                    clickup_id=clickup_id, clickup_status=clickup_status, due=due,
                    state=state)
    existing = get_question(conn, qid)
    if existing is None:
        conn.execute(
            "INSERT INTO questions (id, title, why, owner, priority, raised, clickup_id,"
            " clickup_status, due, state, state_source, last_seen_sync, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,'tracker',?,?)",
            (qid, title, why, owner, priority, raised, clickup_id, clickup_status, due,
             state, _today(), _now()),
        )
        conn.commit()
        log(conn, "question", qid, "created", None, title, actor)
        return
    for field in _TRACKER_FIELDS:
        if existing[field] != incoming[field]:
            log(conn, "question", qid, field, existing[field], incoming[field], actor)
    conn.execute(
        "UPDATE questions SET title=?, why=?, owner=?, priority=?, raised=?, clickup_id=?,"
        " clickup_status=?, due=?, state=?, state_source='tracker', last_seen_sync=?,"
        " updated_at=? WHERE id=?",
        (title, why, owner, priority, raised, clickup_id, clickup_status, due, state,
         _today(), _now(), qid),
    )
    conn.commit()


def set_question_state(conn, qid, state, *, source, note=None, on=None, actor="ui"):
    qid = str(qid)
    existing = get_question(conn, qid)
    if existing is None:
        raise KeyError(f"no question {qid!r}")
    if state not in ("open", "closed"):
        raise ValueError(f"state must be 'open' or 'closed', got {state!r}")
    closed_on = (on or _today()) if state == "closed" else None
    closed_note = note if state == "closed" else None
    if existing["state"] != state:
        log(conn, "question", qid, "state", existing["state"], state, actor)
    conn.execute(
        "UPDATE questions SET state=?, state_source=?, closed_on=?, closed_note=?,"
        " updated_at=? WHERE id=?",
        (state, source, closed_on, closed_note, _now(), qid),
    )
    conn.commit()


def set_question_difficulty(conn, qid, difficulty, *, source, actor="ui"):
    qid = str(qid)
    existing = get_question(conn, qid)
    if existing is None:
        raise KeyError(f"no question {qid!r}")
    if difficulty not in ("S", "M", "L", None):
        raise ValueError(f"difficulty must be S, M, L or None, got {difficulty!r}")
    if existing["difficulty"] != difficulty:
        log(conn, "question", qid, "difficulty", existing["difficulty"], difficulty, actor)
    conn.execute("UPDATE questions SET difficulty=?, difficulty_source=?, updated_at=?"
                 " WHERE id=?", (difficulty, source, _now(), qid))
    conn.commit()


def _agenda_row(row):
    d = dict(row)
    d["followups"] = json.loads(d["followups"]) if d["followups"] else []
    return d


def list_agenda(conn, status=None):
    if status is None:
        rows = conn.execute("SELECT * FROM agenda ORDER BY sort_order, id")
    else:
        rows = conn.execute("SELECT * FROM agenda WHERE status = ? ORDER BY sort_order, id",
                            (status,))
    return [_agenda_row(r) for r in rows]


def get_agenda(conn, aid):
    row = conn.execute("SELECT * FROM agenda WHERE id = ?", (aid,)).fetchone()
    return None if row is None else _agenda_row(row)


def add_agenda(conn, *, title, detail="", origin="manual", proposed_by="", created=None,
               status="proposed", actor="ui"):
    nxt = conn.execute("SELECT COALESCE(MAX(sort_order), -1) + 1 AS n FROM agenda"
                       ).fetchone()["n"]
    cur = conn.execute(
        "INSERT INTO agenda (title, detail, origin, proposed_by, created, sort_order, status)"
        " VALUES (?,?,?,?,?,?,?)",
        (title, detail, origin, proposed_by, created or _today(), nxt, status),
    )
    conn.commit()
    aid = cur.lastrowid
    log(conn, "agenda", aid, "created", None, title, actor)
    return aid


def update_agenda(conn, aid, *, actor="ui", **fields):
    existing = get_agenda(conn, aid)
    if existing is None:
        raise KeyError(f"no agenda item {aid!r}")
    unknown = set(fields) - set(_AGENDA_FIELDS)
    if unknown:
        raise ValueError(f"unknown agenda fields: {sorted(unknown)}")
    if not fields:
        return
    values = dict(fields)
    if "followups" in values and not isinstance(values["followups"], str):
        values["followups"] = json.dumps(values["followups"] or [], ensure_ascii=False)
    for key, new in values.items():
        old = existing[key]
        if key == "followups":
            old = json.dumps(old, ensure_ascii=False)
        if old != new:
            log(conn, "agenda", aid, key, old, new, actor)
    assignments = ", ".join(f"{k}=?" for k in values)
    conn.execute(f"UPDATE agenda SET {assignments} WHERE id=?",
                 (*values.values(), aid))
    conn.commit()


def reorder_agenda(conn, ordered_ids, *, actor="ui"):
    for position, aid in enumerate(ordered_ids):
        conn.execute("UPDATE agenda SET sort_order=? WHERE id=?", (position, aid))
    conn.commit()
    log(conn, "agenda", "*", "sort_order", None, ",".join(str(i) for i in ordered_ids), actor)


def close_agenda(conn, aid, *, resolution, resolved_by, followups=None, resolved_on=None,
                 actor="ui"):
    """Close an agenda item and mint the decision it becomes. Returns the decision id."""
    existing = get_agenda(conn, aid)
    if existing is None:
        raise KeyError(f"no agenda item {aid!r}")
    if not (resolution or "").strip():
        raise ValueError("an agenda item cannot be closed without a resolution")
    on = resolved_on or _today()
    update_agenda(conn, aid, status="closed", resolution=resolution, resolved_by=resolved_by,
                  resolved_on=on, followups=followups or [], actor=actor)
    return add_decision(conn, text=existing["title"], grounds=resolution, decided_on=on,
                        decided_by=resolved_by, followups=followups, from_agenda_id=aid,
                        source="ui", actor=actor)


def _decision_row(row):
    d = dict(row)
    d["followups"] = json.loads(d["followups"]) if d["followups"] else []
    return d


def add_decision(conn, *, text, grounds, decided_on, decided_by="", followups=None,
                 from_agenda_id=None, source="ui", actor="ui"):
    if not (grounds or "").strip():
        raise ValueError("a decision requires grounds; the grounds field is not optional")
    cur = conn.execute(
        "INSERT INTO decisions (text, grounds, decided_on, decided_by, followups,"
        " from_agenda_id, source) VALUES (?,?,?,?,?,?,?)",
        (text, grounds, decided_on, decided_by,
         json.dumps(followups or [], ensure_ascii=False), from_agenda_id, source),
    )
    conn.commit()
    did = cur.lastrowid
    log(conn, "decision", did, "created", None, text, actor)
    return did


def list_decisions(conn):
    rows = conn.execute("SELECT * FROM decisions ORDER BY decided_on DESC, id DESC")
    return [_decision_row(r) for r in rows]


def upsert_flag(conn, *, key, text, first_seen, resolved=0):
    """Insert a flag, or refresh its text. `first_seen` is never overwritten."""
    conn.execute(
        "INSERT INTO flags (key, text, first_seen, resolved) VALUES (?,?,?,?)"
        " ON CONFLICT(key) DO UPDATE SET text=excluded.text, resolved=excluded.resolved",
        (str(key), text, first_seen, int(resolved)),
    )
    conn.commit()


def list_flags(conn):
    return [dict(r) for r in conn.execute("SELECT * FROM flags ORDER BY key")]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_store.py -v`
Expected: PASS, 17 tests

Note: the tests call `upsert_flag(conn, key=..., text=..., first_seen=...)` — keyword-only, matching the implementation.

- [ ] **Step 5: Add the gitignore and commit**

Create `dashboard/.gitignore` containing exactly:

```
data/dashboard.db
```

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6
git add dashboard/.gitignore dashboard/project_tracker/__init__.py dashboard/project_tracker/store.py dashboard/tests/__init__.py dashboard/tests/test_store.py
git commit -m "feat(dashboard): SQLite store with changelog-backed CRUD"
```

---

### Task 2: JSON export, import, and bootstrap

**Files:**
- Create: `dashboard/project_tracker/export.py`
- Create: `dashboard/tests/test_export.py`

**Interfaces:**
- Consumes: everything Task 1 produced.
- Produces: `export_json(conn, json_path) -> dict`; `import_json(conn, json_path) -> None`; `ensure_db(db_path, json_path) -> sqlite3.Connection`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/tests/test_export.py`:

```python
import json

from project_tracker.export import ensure_db, export_json, import_json
from project_tracker.store import (
    add_agenda,
    add_decision,
    close_agenda,
    connect,
    list_decisions,
    list_flags,
    list_questions,
    set_meta,
    set_question_difficulty,
    upsert_flag,
    upsert_question,
)


def _populate(conn):
    upsert_question(conn, id="14", title="HVA benchmark", why="Gates the venue.",
                    owner="Tamra", priority="high", raised="2026-09-14", state="open")
    upsert_question(conn, id="3", title="Reference state", why="", owner="Jonathon",
                    priority="", raised="2026-09-09", state="closed")
    set_question_difficulty(conn, "14", "L", source="user", actor="ui")
    aid = add_agenda(conn, title="Settle the cluster count", origin="sync:flag:1",
                     status="accepted")
    close_agenda(conn, aid, resolution="Ten, as in Figs. 2-4", resolved_by="Eun-Ah",
                 resolved_on="2026-09-21")
    add_decision(conn, text="PRR regular article", grounds="Body exceeds the Letter limit",
                 decided_on="2026-09-13", decided_by="Eun-Ah")
    upsert_flag(conn, key="1", text="Ten clusters or fourteen?", first_seen="2026-09-21")
    set_meta(conn, "meetings", [{"date": "2026-09-14", "label": "Frame and novelty"}])


def test_export_writes_every_table(tmp_path):
    conn = connect(tmp_path / "d.db")
    _populate(conn)
    path = tmp_path / "dashboard.json"
    doc = export_json(conn, path)
    on_disk = json.loads(path.read_text())
    assert on_disk == doc
    assert {"schema", "meta", "questions", "agenda", "decisions", "flags"} <= set(doc)
    assert [q["id"] for q in doc["questions"]] == ["14", "3"]


def test_export_is_byte_identical_when_nothing_changed(tmp_path):
    conn = connect(tmp_path / "d.db")
    _populate(conn)
    path = tmp_path / "dashboard.json"
    export_json(conn, path)
    first = path.read_bytes()
    export_json(conn, path)
    assert path.read_bytes() == first


def test_export_contains_no_wall_clock_timestamp(tmp_path):
    conn = connect(tmp_path / "d.db")
    _populate(conn)
    path = tmp_path / "dashboard.json"
    export_json(conn, path)
    assert "generated" not in json.loads(path.read_text())


def test_export_ends_with_a_single_trailing_newline(tmp_path):
    conn = connect(tmp_path / "d.db")
    _populate(conn)
    path = tmp_path / "dashboard.json"
    export_json(conn, path)
    text = path.read_text()
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_import_round_trips_losslessly(tmp_path):
    source = connect(tmp_path / "a.db")
    _populate(source)
    path = tmp_path / "dashboard.json"
    export_json(source, path)

    target = connect(tmp_path / "b.db")
    import_json(target, path)
    again = tmp_path / "again.json"
    export_json(target, again)
    assert again.read_bytes() == path.read_bytes()


def test_import_preserves_ui_owned_fields(tmp_path):
    source = connect(tmp_path / "a.db")
    _populate(source)
    path = tmp_path / "dashboard.json"
    export_json(source, path)

    target = connect(tmp_path / "b.db")
    import_json(target, path)
    q = [r for r in list_questions(target) if r["id"] == "14"][0]
    assert q["difficulty"] == "L"
    assert q["difficulty_source"] == "user"
    assert list_decisions(target)[0]["grounds"] == "Ten, as in Figs. 2-4"
    assert list_flags(target)[0]["first_seen"] == "2026-09-21"


def test_import_replaces_rather_than_appends(tmp_path):
    source = connect(tmp_path / "a.db")
    _populate(source)
    path = tmp_path / "dashboard.json"
    export_json(source, path)

    target = connect(tmp_path / "b.db")
    import_json(target, path)
    import_json(target, path)
    assert len(list_questions(target)) == 2
    assert len(list_decisions(target)) == 2


def test_ensure_db_builds_the_database_from_json_when_it_is_missing(tmp_path):
    source = connect(tmp_path / "a.db")
    _populate(source)
    path = tmp_path / "dashboard.json"
    export_json(source, path)

    db = tmp_path / "fresh.db"
    assert not db.exists()
    conn = ensure_db(db, path)
    assert len(list_questions(conn)) == 2


def test_ensure_db_rebuilds_when_json_is_newer_than_the_database(tmp_path):
    db = tmp_path / "d.db"
    path = tmp_path / "dashboard.json"
    conn = connect(db)
    _populate(conn)
    export_json(conn, path)
    conn.close()

    doc = json.loads(path.read_text())
    doc["questions"].append({
        "id": "99", "title": "Added by a collaborator", "why": "", "owner": "",
        "priority": "", "raised": "", "clickup_id": None, "clickup_status": None,
        "due": None, "state": "open", "state_source": "tracker", "difficulty": None,
        "difficulty_source": None, "closed_on": None, "closed_note": None,
        "last_seen_sync": None, "updated_at": None,
    })
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    import os, time
    os.utime(path, (time.time() + 10, time.time() + 10))

    conn = ensure_db(db, path)
    assert "99" in [q["id"] for q in list_questions(conn)]


def test_ensure_db_leaves_a_current_database_alone(tmp_path):
    db = tmp_path / "d.db"
    path = tmp_path / "dashboard.json"
    conn = connect(db)
    _populate(conn)
    export_json(conn, path)
    upsert_question(conn, id="55", title="local only", why="", owner="", priority="",
                    raised="", state="open")
    conn.close()

    conn = ensure_db(db, path)
    assert "55" in [q["id"] for q in list_questions(conn)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_export.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'project_tracker.export'`

- [ ] **Step 3: Write the exporter**

Create `dashboard/project_tracker/export.py`:

```python
"""Store <-> dashboard.json.

The JSON is the store of record: it is what git carries, so a collaborator who
clones the Overleaf repo has the database even though the .db file is ignored.
It must be byte-deterministic given store state, or every sync produces a noisy
diff and the file stops being reviewable.
"""

import json
from pathlib import Path

from . import store

SCHEMA = 1

_QUESTION_COLUMNS = ("id", "title", "why", "owner", "priority", "raised", "clickup_id",
                     "clickup_status", "due", "state", "state_source", "difficulty",
                     "difficulty_source", "closed_on", "closed_note", "last_seen_sync",
                     "updated_at")
_AGENDA_COLUMNS = ("id", "title", "detail", "origin", "proposed_by", "created",
                   "sort_order", "status", "resolution", "followups", "resolved_on",
                   "resolved_by")
_DECISION_COLUMNS = ("id", "text", "grounds", "decided_on", "decided_by", "followups",
                     "from_agenda_id", "source")
_FLAG_COLUMNS = ("key", "text", "first_seen", "resolved")
_META_KEYS = ("meetings", "last_sync", "last_clickup_pull", "workflow_ran",
              "last_reported_change")


def _pick(row, columns):
    return {c: row.get(c) for c in columns}


def build_document(conn):
    meta = {k: store.get_meta(conn, k) for k in _META_KEYS}
    meta = {k: v for k, v in meta.items() if v is not None}
    return {
        "schema": SCHEMA,
        "meta": meta,
        "questions": [_pick(r, _QUESTION_COLUMNS) for r in store.list_questions(conn)],
        "agenda": [_pick(r, _AGENDA_COLUMNS) for r in store.list_agenda(conn)],
        "decisions": [_pick(r, _DECISION_COLUMNS) for r in store.list_decisions(conn)],
        "flags": [_pick(r, _FLAG_COLUMNS) for r in store.list_flags(conn)],
    }


def export_json(conn, json_path):
    """Write the store to `json_path` deterministically. Returns the document."""
    doc = build_document(conn)
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return doc


def import_json(conn, json_path):
    """Replace the store's contents with the document at `json_path`.

    Replaces rather than merges: the JSON is authoritative when it is read.
    The changelog is deliberately not carried in the JSON, so it is left alone.
    """
    doc = json.loads(Path(json_path).read_text())
    conn.execute("DELETE FROM questions")
    conn.execute("DELETE FROM agenda")
    conn.execute("DELETE FROM decisions")
    conn.execute("DELETE FROM flags")
    for row in doc.get("questions", []):
        conn.execute(
            f"INSERT INTO questions ({','.join(_QUESTION_COLUMNS)})"
            f" VALUES ({','.join('?' * len(_QUESTION_COLUMNS))})",
            tuple(row.get(c) for c in _QUESTION_COLUMNS))
    for row in doc.get("agenda", []):
        values = dict(row)
        values["followups"] = json.dumps(values.get("followups") or [], ensure_ascii=False)
        conn.execute(
            f"INSERT INTO agenda ({','.join(_AGENDA_COLUMNS)})"
            f" VALUES ({','.join('?' * len(_AGENDA_COLUMNS))})",
            tuple(values.get(c) for c in _AGENDA_COLUMNS))
    for row in doc.get("decisions", []):
        values = dict(row)
        values["followups"] = json.dumps(values.get("followups") or [], ensure_ascii=False)
        conn.execute(
            f"INSERT INTO decisions ({','.join(_DECISION_COLUMNS)})"
            f" VALUES ({','.join('?' * len(_DECISION_COLUMNS))})",
            tuple(values.get(c) for c in _DECISION_COLUMNS))
    for row in doc.get("flags", []):
        conn.execute(
            f"INSERT INTO flags ({','.join(_FLAG_COLUMNS)})"
            f" VALUES ({','.join('?' * len(_FLAG_COLUMNS))})",
            tuple(row.get(c) for c in _FLAG_COLUMNS))
    conn.commit()
    for key, value in (doc.get("meta") or {}).items():
        store.set_meta(conn, key, value)


def ensure_db(db_path, json_path):
    """Open the store, rebuilding it from JSON if the JSON is newer or the DB is absent.

    This is what makes `git pull && ./dashboard serve` work on a machine that has
    never run a sync: the .db is gitignored, the .json is not.
    """
    db_path, json_path = Path(db_path), Path(json_path)
    stale = (not db_path.exists()) or (
        json_path.exists() and json_path.stat().st_mtime > db_path.stat().st_mtime)
    conn = store.connect(db_path)
    if stale and json_path.exists():
        import_json(conn, json_path)
        db_path.touch()
    return conn
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_export.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6
git add dashboard/project_tracker/export.py dashboard/tests/test_export.py
git commit -m "feat(dashboard): deterministic JSON export and bootstrap-from-JSON"
```

---

### Task 3: HANDOFF.md §7 parser

**Files:**
- Create: `dashboard/project_tracker/sources/__init__.py`
- Create: `dashboard/project_tracker/sources/handoff.py`
- Create: `dashboard/tests/test_handoff.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Item` dataclass with fields `id, title, why, owner, state, raised, clickup_id, priority`; `Flag` dataclass with fields `key, text, resolved`; `parse_handoff(text) -> tuple[list[Item], list[Flag]]`; `HandoffParseError(ValueError)`; `read_handoff(path) -> tuple[list[Item], list[Flag]]`.

The parser is the only module that knows §7's prose shape, and §7 is hand-written prose that will drift. It must **fail loudly** rather than silently drop an item.

Real format, confirmed against `HANDOFF.md` as of 2026-09-21:

- `## 7. Open items (state at YYYY-MM-DD)` opens the section; `## 8.` (or any `## ` heading that is not `### 7.1`) ends it.
- Owner sections are `### Open — Jonathon`, `### Open — Tamra`, `### Open — Eun-Ah`, and `### Closed`.
- An item begins at line start with `**<number>. <title>**`, where `<number>` is digits optionally followed by a lowercase letter (`14`, `14b`). Everything up to the next item header or heading is the body.
- The body may start with ` — ` or with nothing at all (item 21 has no dash).
- A ClickUp id appears as `` task `86akhca00` `` or `` Task `86akhca00` ``.
- A raised date appears as `raised YYYY-MM-DD` or `Asked YYYY-MM-DD`.
- `### 7.1 Flags ...` holds a numbered list; an entry whose text begins with `~~` is resolved.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/tests/test_handoff.py`:

```python
import pytest

from project_tracker.sources.handoff import HandoffParseError, parse_handoff

SAMPLE = """\
## 6. Do NOT re-add these

Nothing here should be parsed.

## 7. Open items (state at 2026-09-21)

Numbers are stable identifiers.

### Open — Jonathon

**1. Cost per gradient, exponential vs product form** — wall-clock per gradient for
path (a) vs (b). Lands in paragraph 17. *Asked 2026-09-10 in the channel; task `86akhc9zz`.*
**2. The small-dimension energy floor** — rerun at low dimension with tight tolerance.
*Task `86akhca00`.* **Answered 2026-09-21 but not yet written up.**

### Open — Tamra

**21. HVA benchmark: Cade's public code, or rebuilt?** Tamra · no task yet · raised
2026-09-14. Jonathon estimates about a day using their code.
**14b. The interim venue position** — a sub-item of 14. *Task `86akj18bz`.*

### Closed

**25. The certified data ledger is stale** — CLOSED 2026-09-21, no task was ever opened.
Regenerated the same day.
**23. Which loss do the runs use?** — CLOSED 2026-09-15, task `86akj1wtv`.

### 7.1 Flags — where this file, ClickUp and the code disagree (2026-09-21)

Listed, not resolved.

1. **Ten clusters or fourteen?** Item 7 asks for ten; the table covers 14.
2. ~~**Item 20 is closed but its standard is not met.**~~ RESOLVED 2026-09-21.
3. **Who owns updating this file?** The notes assign it to Eun-Ah; the task is Jonathon's.

## 8. Presentation choices worth keeping

Not parsed either.
"""


def test_parses_every_item_across_all_sections():
    items, _ = parse_handoff(SAMPLE)
    assert [i.id for i in items] == ["1", "2", "21", "14b", "25", "23"]


def test_ignores_content_outside_section_seven():
    items, _ = parse_handoff(SAMPLE)
    assert all("re-add" not in i.title for i in items)
    assert all("Presentation" not in i.title for i in items)


def test_assigns_owner_from_the_enclosing_heading():
    items, _ = parse_handoff(SAMPLE)
    by_id = {i.id: i for i in items}
    assert by_id["1"].owner == "Jonathon"
    assert by_id["21"].owner == "Tamra"
    assert by_id["25"].owner == ""


def test_state_follows_the_section():
    items, _ = parse_handoff(SAMPLE)
    by_id = {i.id: i for i in items}
    assert by_id["1"].state == "open"
    assert by_id["25"].state == "closed"


def test_sub_item_numbers_are_preserved_as_strings():
    items, _ = parse_handoff(SAMPLE)
    assert "14b" in {i.id for i in items}


def test_title_excludes_the_body():
    items, _ = parse_handoff(SAMPLE)
    by_id = {i.id: i for i in items}
    assert by_id["1"].title == "Cost per gradient, exponential vs product form"


def test_item_without_an_em_dash_still_parses_its_title():
    items, _ = parse_handoff(SAMPLE)
    by_id = {i.id: i for i in items}
    assert by_id["21"].title == "HVA benchmark: Cade's public code, or rebuilt?"


def test_extracts_clickup_task_ids_in_either_case():
    items, _ = parse_handoff(SAMPLE)
    by_id = {i.id: i for i in items}
    assert by_id["1"].clickup_id == "86akhc9zz"
    assert by_id["2"].clickup_id == "86akhca00"
    assert by_id["21"].clickup_id is None


def test_extracts_raised_dates_from_asked_or_raised():
    items, _ = parse_handoff(SAMPLE)
    by_id = {i.id: i for i in items}
    assert by_id["1"].raised == "2026-09-10"
    assert by_id["21"].raised == "2026-09-14"


def test_why_carries_the_body_text():
    items, _ = parse_handoff(SAMPLE)
    by_id = {i.id: i for i in items}
    assert "wall-clock per gradient" in by_id["1"].why


def test_parses_flags_and_marks_struck_through_ones_resolved():
    _, flags = parse_handoff(SAMPLE)
    assert [f.key for f in flags] == ["1", "2", "3"]
    assert flags[0].resolved is False
    assert flags[1].resolved is True
    assert "Ten clusters or fourteen?" in flags[0].text


def test_missing_section_seven_raises():
    with pytest.raises(HandoffParseError):
        parse_handoff("# A file\n\n## 1. Something else\n\nNo section seven here.\n")


def test_duplicate_item_numbers_raise():
    bad = SAMPLE.replace("**2. The small-dimension energy floor**",
                         "**1. The small-dimension energy floor**")
    with pytest.raises(HandoffParseError):
        parse_handoff(bad)


def test_a_section_with_no_items_raises():
    bad = """\
## 7. Open items (state at 2026-09-21)

### Open — Jonathon

Prose with no item headers at all.

## 8. Next
"""
    with pytest.raises(HandoffParseError):
        parse_handoff(bad)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_handoff.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'project_tracker.sources'`

- [ ] **Step 3: Write the parser**

Create `dashboard/project_tracker/sources/__init__.py`:

```python
"""Source adapters.

Each adapter turns some external record into `Item`s the sync layer can
reconcile. The contract is one function, `fetch()`, returning a list of `Item`.
Which source feeds the dashboard is still an open question, so keep adapters
free of anything specific to the store.
"""

from dataclasses import dataclass, field


@dataclass
class Item:
    id: str
    title: str
    why: str = ""
    owner: str = ""
    state: str = "open"
    raised: str = ""
    clickup_id: str | None = None
    priority: str = ""
    clickup_status: str | None = None
    due: str | None = None


@dataclass
class Flag:
    key: str
    text: str
    resolved: bool = False


__all__ = ["Item", "Flag"]
```

Create `dashboard/project_tracker/sources/handoff.py`:

```python
"""Parse `HANDOFF.md` section 7 — the ledger where item numbers are born.

Section 7 is hand-written prose and its shape will drift. Every failure mode
here is loud on purpose: a parser that silently drops an item produces a
dashboard that is quietly wrong, which is worse than one that will not build.
"""

import re
from pathlib import Path

from . import Flag, Item


class HandoffParseError(ValueError):
    """Raised when section 7 cannot be parsed into items."""


_SECTION_7 = re.compile(r"^##\s+7\.\s", re.M)
_SECTION_71 = re.compile(r"^###\s+7\.1\s", re.M)
_NEXT_H2 = re.compile(r"^##\s+(?!7\.)", re.M)
_OWNER_HEADING = re.compile(r"^###\s+Open\s+[—-]\s*(?P<owner>.+?)\s*$", re.M)
_CLOSED_HEADING = re.compile(r"^###\s+Closed\s*$", re.M)
_ITEM = re.compile(r"^\*\*(?P<num>\d+[a-z]?)\.\s+(?P<title>.+?)\*\*", re.M)
_CLICKUP = re.compile(r"[Tt]ask\s+`(?P<id>[0-9a-z]{6,})`")
_RAISED = re.compile(r"(?:raised|Asked|asked)\s+(?P<date>\d{4}-\d{2}-\d{2})")
_FLAG_ENTRY = re.compile(r"^(?P<key>\d+)\.\s+(?P<body>.*?)(?=^\d+\.\s|\Z)", re.M | re.S)


def _section_7(text):
    start = _SECTION_7.search(text)
    if start is None:
        raise HandoffParseError("no '## 7.' heading found in HANDOFF.md")
    rest = text[start.end():]
    end = _NEXT_H2.search(rest)
    body = rest if end is None else rest[:end.start()]
    return body


def _split_subsections(body):
    """Yield (owner, state, text) for each ### block inside section 7."""
    marks = []
    for m in _OWNER_HEADING.finditer(body):
        marks.append((m.start(), m.end(), m.group("owner").strip(), "open"))
    for m in _CLOSED_HEADING.finditer(body):
        marks.append((m.start(), m.end(), "", "closed"))
    marks.sort()
    if not marks:
        raise HandoffParseError(
            "section 7 has no '### Open — <name>' or '### Closed' subsections")
    flags_start = _SECTION_71.search(body)
    limit = flags_start.start() if flags_start else len(body)
    out = []
    for index, (start, content_start, owner, state) in enumerate(marks):
        if start >= limit:
            continue
        end = marks[index + 1][0] if index + 1 < len(marks) else limit
        out.append((owner, state, body[content_start:min(end, limit)]))
    return out


def _parse_items(body):
    items = []
    for owner, state, text in _split_subsections(body):
        matches = list(_ITEM.finditer(text))
        if not matches:
            raise HandoffParseError(
                f"subsection for owner {owner!r} (state {state}) contains no items; "
                "section 7's format has changed")
        for index, match in enumerate(matches):
            stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            chunk = text[match.end():stop]
            why = chunk.strip()
            if why.startswith("—") or why.startswith("--"):
                why = why.lstrip("—-").strip()
            clickup = _CLICKUP.search(chunk)
            raised = _RAISED.search(chunk)
            items.append(Item(
                id=match.group("num"),
                title=match.group("title").strip(),
                why=" ".join(why.split()),
                owner=owner,
                state=state,
                raised=raised.group("date") if raised else "",
                clickup_id=clickup.group("id") if clickup else None,
            ))
    seen = {}
    for item in items:
        if item.id in seen:
            raise HandoffParseError(
                f"item number {item.id!r} appears twice in section 7; "
                "numbers are unique identifiers and must not be reused")
        seen[item.id] = item
    return items


def _parse_flags(body):
    match = _SECTION_71.search(body)
    if match is None:
        return []
    text = body[match.end():]
    flags = []
    for entry in _FLAG_ENTRY.finditer(text):
        raw = " ".join(entry.group("body").split())
        flags.append(Flag(key=entry.group("key"), text=raw, resolved=raw.startswith("~~")))
    return flags


def parse_handoff(text):
    """Return (items, flags) parsed from section 7 of a HANDOFF.md document."""
    body = _section_7(text)
    return _parse_items(body), _parse_flags(body)


def read_handoff(path):
    return parse_handoff(Path(path).read_text())


def fetch(path):
    """Adapter entry point: items only, for callers that do not want flags."""
    return read_handoff(path)[0]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_handoff.py -v`
Expected: PASS, 13 tests

- [ ] **Step 5: Verify against the real HANDOFF.md**

Run:

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard
python3 -c "
from project_tracker.sources.handoff import read_handoff
items, flags = read_handoff('../HANDOFF.md')
print(len(items), 'items;', len(flags), 'flags')
print(sorted((i.id for i in items), key=lambda s: (int(''.join(c for c in s if c.isdigit())), s)))
print([ (f.key, f.resolved) for f in flags ])
"
```

Expected, verified against `HANDOFF.md` as of 2026-09-21: **30 items and 7 flags**, with flag 2 resolved, `14b` present, and both `open` and `closed` states represented. The item ids are:

```
1 2 4 5 6 7 10 12 14 26 27 28 29 15 21 16 17 25 23 22 20 19 14b 24 3 8 9 11 13 18
```

If the parser raises, or returns a different count, fix the parser — do not loosen it into silence. A count below 30 means items are being dropped, which is the exact failure this design is built to prevent.

- [ ] **Step 6: Pin the real-file result as a regression test**

Append to `dashboard/tests/test_handoff.py`. If `HANDOFF.md` has gained or lost items since 2026-09-21, update these two constants to the counts from Step 5 and say so in the commit message:

```python
from pathlib import Path

REAL = Path(__file__).resolve().parents[2] / "HANDOFF.md"
EXPECTED_ITEMS = 30
EXPECTED_FLAGS = 7


@pytest.mark.skipif(not REAL.exists(), reason="HANDOFF.md not present")
def test_parses_the_real_handoff_without_losing_items():
    items, flags = parse_handoff(REAL.read_text())
    assert len(items) == EXPECTED_ITEMS
    assert len(flags) == EXPECTED_FLAGS
    assert "14b" in {i.id for i in items}
    assert {i.state for i in items} == {"open", "closed"}
    assert any(i.clickup_id for i in items)
```

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_handoff.py -v`
Expected: PASS, 14 tests

- [ ] **Step 7: Commit**

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6
git add dashboard/project_tracker/sources/ dashboard/tests/test_handoff.py
git commit -m "feat(dashboard): HANDOFF.md section 7 parser with loud failure modes"
```

---

### Task 4: ClickUp cache adapter

**Files:**
- Create: `dashboard/project_tracker/sources/clickup.py`
- Create: `dashboard/tests/test_clickup.py`

**Interfaces:**
- Consumes: `Item` from Task 3's `sources/__init__.py`.
- Produces: `load_cache(path) -> dict[str, dict]`; `save_cache(path, tasks) -> None`; `merge(items, cache) -> list[Item]`; `cache_age_days(path, today=None) -> int | None`.

ClickUp is never called over the network from this package — pulling is expensive and needs a connector the server does not have. An agent with the connector writes the snapshot; this module only reads it.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/tests/test_clickup.py`:

```python
import json

from project_tracker.sources import Item
from project_tracker.sources.clickup import cache_age_days, load_cache, merge, save_cache


def _items():
    return [
        Item(id="1", title="Cost per gradient", clickup_id="86akhc9zz"),
        Item(id="21", title="HVA benchmark", clickup_id=None),
    ]


def test_save_and_load_cache_round_trip(tmp_path):
    path = tmp_path / "clickup-cache.json"
    save_cache(path, {"86akhc9zz": {"status": "in progress", "priority": "high",
                                    "due": "2026-09-30"}}, pulled_on="2026-09-21")
    cache = load_cache(path)
    assert cache["tasks"]["86akhc9zz"]["priority"] == "high"
    assert cache["pulled_on"] == "2026-09-21"


def test_load_cache_returns_an_empty_shape_when_absent(tmp_path):
    cache = load_cache(tmp_path / "missing.json")
    assert cache == {"pulled_on": None, "tasks": {}}


def test_merge_applies_status_priority_and_due(tmp_path):
    cache = {"pulled_on": "2026-09-21",
             "tasks": {"86akhc9zz": {"status": "in progress", "priority": "high",
                                     "due": "2026-09-30"}}}
    merged = {i.id: i for i in merge(_items(), cache)}
    assert merged["1"].clickup_status == "in progress"
    assert merged["1"].priority == "high"
    assert merged["1"].due == "2026-09-30"


def test_merge_leaves_items_without_a_task_untouched(tmp_path):
    cache = {"pulled_on": "2026-09-21", "tasks": {}}
    merged = {i.id: i for i in merge(_items(), cache)}
    assert merged["21"].clickup_status is None
    assert merged["21"].priority == ""


def test_merge_does_not_mutate_the_input_items():
    items = _items()
    cache = {"pulled_on": "x", "tasks": {"86akhc9zz": {"priority": "high"}}}
    merge(items, cache)
    assert items[0].priority == ""


def test_save_cache_is_deterministic(tmp_path):
    path = tmp_path / "c.json"
    tasks = {"b": {"priority": "low"}, "a": {"priority": "high"}}
    save_cache(path, tasks, pulled_on="2026-09-21")
    first = path.read_bytes()
    save_cache(path, tasks, pulled_on="2026-09-21")
    assert path.read_bytes() == first
    assert list(json.loads(path.read_text())["tasks"]) == ["a", "b"]


def test_cache_age_days_counts_from_pulled_on(tmp_path):
    path = tmp_path / "c.json"
    save_cache(path, {}, pulled_on="2026-09-14")
    assert cache_age_days(path, today="2026-09-21") == 7


def test_cache_age_days_is_none_without_a_cache(tmp_path):
    assert cache_age_days(tmp_path / "missing.json", today="2026-09-21") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_clickup.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'project_tracker.sources.clickup'`

- [ ] **Step 3: Write the adapter**

Create `dashboard/project_tracker/sources/clickup.py`:

```python
"""Cached ClickUp snapshot.

Nothing here touches the network. Pulling from ClickUp is expensive and needs a
connector this process does not have, so an agent that *does* have one writes
the snapshot with `save_cache`, and every later sync reads it for free.
"""

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

EMPTY = {"pulled_on": None, "tasks": {}}


def load_cache(path):
    path = Path(path)
    if not path.exists():
        return dict(EMPTY)
    doc = json.loads(path.read_text())
    return {"pulled_on": doc.get("pulled_on"), "tasks": doc.get("tasks") or {}}


def save_cache(path, tasks, *, pulled_on=None):
    """Write a snapshot. `tasks` maps ClickUp task id -> {status, priority, due}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"pulled_on": pulled_on or date.today().isoformat(), "tasks": tasks}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def merge(items, cache):
    """Return new Items with ClickUp status, priority and due date applied."""
    tasks = cache.get("tasks") or {}
    out = []
    for item in items:
        task = tasks.get(item.clickup_id) if item.clickup_id else None
        if not task:
            out.append(replace(item))
            continue
        out.append(replace(
            item,
            clickup_status=task.get("status"),
            priority=task.get("priority") or item.priority,
            due=task.get("due"),
        ))
    return out


def cache_age_days(path, today=None):
    cache = load_cache(path)
    if not cache["pulled_on"]:
        return None
    then = date.fromisoformat(cache["pulled_on"])
    now = date.fromisoformat(today) if today else date.today()
    return (now - then).days
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_clickup.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6
git add dashboard/project_tracker/sources/clickup.py dashboard/tests/test_clickup.py
git commit -m "feat(dashboard): cached ClickUp snapshot adapter"
```

---

### Task 5: Sync — reconciliation, difficulty, agenda proposals

**Files:**
- Create: `dashboard/project_tracker/sync.py`
- Create: `dashboard/tests/test_sync.py`

**Interfaces:**
- Consumes: `store` (Task 1), `export` (Task 2), `sources.handoff` (Task 3), `sources.clickup` (Task 4).
- Produces: `estimate_difficulty(text) -> str`; `sync(conn, *, handoff_path, clickup_cache_path=None, today=None) -> SyncResult`; `SyncResult` dataclass with fields `added, updated, reverted, stale, flags, agenda_proposed`.

The ownership rules, restated because this is the module that enforces them:

| Field | Written by sync? |
|---|---|
| `title`, `why`, `owner`, `priority`, `raised`, `clickup_*`, `due` | Yes, unconditionally |
| `state` | Yes, unconditionally — the tracker dictates open/closed |
| `difficulty` | Only where `difficulty_source != 'user'` |
| `closed_note`, `closed_on` | Never |
| agenda rows that are `accepted` or `closed` | Never |
| decisions | Never |
| flags | Inserted and text-refreshed; `resolved` is taken from the tracker's strikethrough and never inferred |

A question the UI closed that §7 still lists open is **reverted to open** and recorded in `reverted`, which Task 6 turns into an owed §7 edit.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/tests/test_sync.py`:

```python
import pytest

from project_tracker.store import (
    add_agenda,
    close_agenda,
    connect,
    get_question,
    list_agenda,
    list_decisions,
    list_flags,
    set_question_difficulty,
    set_question_state,
    update_agenda,
)
from project_tracker.sync import estimate_difficulty, sync

HANDOFF = """\
## 7. Open items (state at 2026-09-21)

### Open — Jonathon

**1. Cost per gradient** — wall-clock per gradient for path (a) vs (b).
*Asked 2026-09-10; task `86akhc9zz`.*
**2. Write the answer into paragraph 17** — a writing task, nothing to run.
*Task `86akhca00`.*

### Open — Tamra

**21. Implement the HVA ansatz from scratch** — rebuild Cade's circuit builder in our
codebase. raised 2026-09-14.

### Closed

**25. The certified data ledger is stale** — CLOSED 2026-09-21.

### 7.1 Flags — disagreements (2026-09-21)

1. **Ten clusters or fourteen?** Someone has to say which set the paper shows.
2. ~~**Item 20's standard is not met.**~~ RESOLVED 2026-09-21.

## 8. Next
"""


@pytest.fixture
def handoff(tmp_path):
    path = tmp_path / "HANDOFF.md"
    path.write_text(HANDOFF)
    return path


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "d.db")
    yield c
    c.close()


def test_first_sync_adds_every_item(conn, handoff):
    result = sync(conn, handoff_path=handoff, today="2026-09-21")
    assert sorted(result.added) == ["1", "2", "21", "25"]
    assert get_question(conn, "25")["state"] == "closed"
    assert get_question(conn, "1")["owner"] == "Jonathon"


def test_second_sync_adds_nothing_and_is_idempotent(conn, handoff):
    sync(conn, handoff_path=handoff, today="2026-09-21")
    result = sync(conn, handoff_path=handoff, today="2026-09-21")
    assert result.added == []


def test_tracker_text_overwrites_the_stored_description(conn, handoff, tmp_path):
    sync(conn, handoff_path=handoff, today="2026-09-21")
    handoff.write_text(HANDOFF.replace("**1. Cost per gradient**",
                                       "**1. Cost per gradient, restated**"))
    sync(conn, handoff_path=handoff, today="2026-09-21")
    assert get_question(conn, "1")["title"] == "Cost per gradient, restated"


def test_ui_closure_not_in_section_seven_is_reverted_and_reported(conn, handoff):
    sync(conn, handoff_path=handoff, today="2026-09-21")
    set_question_state(conn, "1", "closed", source="ui", note="settled in the meeting",
                       on="2026-09-21", actor="ui")
    result = sync(conn, handoff_path=handoff, today="2026-09-21")
    assert get_question(conn, "1")["state"] == "open"
    assert len(result.reverted) == 1
    assert result.reverted[0]["id"] == "1"
    assert result.reverted[0]["note"] == "settled in the meeting"


def test_a_user_set_difficulty_is_never_overwritten(conn, handoff):
    sync(conn, handoff_path=handoff, today="2026-09-21")
    set_question_difficulty(conn, "21", "S", source="user", actor="ui")
    sync(conn, handoff_path=handoff, today="2026-09-21")
    row = get_question(conn, "21")
    assert row["difficulty"] == "S"
    assert row["difficulty_source"] == "user"


def test_agent_estimates_difficulty_for_items_it_has_not_been_told_about(conn, handoff):
    sync(conn, handoff_path=handoff, today="2026-09-21")
    assert get_question(conn, "21")["difficulty"] == "L"
    assert get_question(conn, "21")["difficulty_source"] == "agent"
    assert get_question(conn, "2")["difficulty"] == "S"


def test_item_dropped_from_section_seven_is_flagged_not_deleted(conn, handoff):
    sync(conn, handoff_path=handoff, today="2026-09-21")
    handoff.write_text(HANDOFF.replace(
        "**21. Implement the HVA ansatz from scratch** — rebuild Cade's circuit builder in our\n"
        "codebase. raised 2026-09-14.\n", ""))
    result = sync(conn, handoff_path=handoff, today="2026-09-21")
    assert result.stale == ["21"]
    assert get_question(conn, "21") is not None
    assert any("21" in f["text"] for f in list_flags(conn))


def test_flags_are_imported_with_their_resolved_state(conn, handoff):
    sync(conn, handoff_path=handoff, today="2026-09-21")
    flags = {f["key"]: f for f in list_flags(conn)}
    assert flags["1"]["resolved"] == 0
    assert flags["2"]["resolved"] == 1


def test_sync_proposes_agenda_items_for_unresolved_flags(conn, handoff):
    result = sync(conn, handoff_path=handoff, today="2026-09-21")
    origins = {a["origin"] for a in list_agenda(conn)}
    assert "sync:flag:1" in origins
    assert "sync:flag:2" not in origins
    assert "sync:flag:1" in result.agenda_proposed


def test_agenda_proposals_are_not_duplicated_on_a_second_sync(conn, handoff):
    sync(conn, handoff_path=handoff, today="2026-09-21")
    sync(conn, handoff_path=handoff, today="2026-09-21")
    origins = [a["origin"] for a in list_agenda(conn)]
    assert origins.count("sync:flag:1") == 1


def test_sync_never_touches_an_accepted_agenda_item(conn, handoff):
    sync(conn, handoff_path=handoff, today="2026-09-21")
    aid = [a["id"] for a in list_agenda(conn) if a["origin"] == "sync:flag:1"][0]
    update_agenda(conn, aid, status="accepted", title="Curated title")
    sync(conn, handoff_path=handoff, today="2026-09-21")
    row = [a for a in list_agenda(conn) if a["id"] == aid][0]
    assert row["status"] == "accepted"
    assert row["title"] == "Curated title"


def test_sync_never_removes_a_decision(conn, handoff):
    sync(conn, handoff_path=handoff, today="2026-09-21")
    aid = add_agenda(conn, title="Settle it", status="accepted")
    close_agenda(conn, aid, resolution="Settled", resolved_by="Eun-Ah",
                 resolved_on="2026-09-21")
    sync(conn, handoff_path=handoff, today="2026-09-21")
    assert len(list_decisions(conn)) == 1


def test_sync_writes_the_json_export_when_given_a_path(conn, handoff, tmp_path):
    out = tmp_path / "dashboard.json"
    sync(conn, handoff_path=handoff, json_path=out, today="2026-09-21")
    assert out.exists()
    assert '"questions"' in out.read_text()


def test_estimate_difficulty_reads_the_verbs():
    assert estimate_difficulty("rerun at low dimension with a tighter tolerance") == "S"
    assert estimate_difficulty("write the answer into paragraph 17") == "S"
    assert estimate_difficulty("implement the HVA ansatz from scratch") == "L"
    assert estimate_difficulty("rebuild Cade's circuit builder in our codebase") == "L"
    assert estimate_difficulty("decide whether the caption is right") == "M"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_sync.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'project_tracker.sync'`

- [ ] **Step 3: Write sync**

Create `dashboard/project_tracker/sync.py`:

```python
"""Reconcile the store against the tracker.

The whole point of this module is that regenerating the dashboard is a
reconciliation, not a generation: no prose is re-authored, and the only things
that change are the fields the tracker owns. What the UI owns — resolutions,
decisions, curated agenda, hand-set difficulty — survives untouched.
"""

import re
from dataclasses import dataclass, field
from datetime import date

from . import store
from .export import export_json
from .sources import clickup as clickup_source
from .sources.handoff import read_handoff

_SMALL = re.compile(
    r"\b(rerun|re-run|regenerate|regenerated|relabel|rename|write|written|writing|"
    r"caption|one line|note|record|cite|check|confirm)\b", re.I)
_LARGE = re.compile(
    r"\b(implement|implementation|rebuild|build|from scratch|new appendix|benchmark|"
    r"port|derive|a week|refactor)\b", re.I)


@dataclass
class SyncResult:
    added: list = field(default_factory=list)
    updated: list = field(default_factory=list)
    reverted: list = field(default_factory=list)
    stale: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    agenda_proposed: list = field(default_factory=list)


def estimate_difficulty(text):
    """Guess S/M/L from how the item describes the work. Always overridable."""
    text = text or ""
    if _LARGE.search(text):
        return "L"
    if _SMALL.search(text):
        return "S"
    return "M"


def _flag_title(text):
    """A flag's first bolded phrase makes a better agenda title than its whole body."""
    stripped = text.strip().lstrip("~").strip()
    bold = re.match(r"\*\*(?P<t>.+?)\*\*", stripped)
    if bold:
        return bold.group("t").strip()
    return " ".join(stripped.split()[:12])


def sync(conn, *, handoff_path, clickup_cache_path=None, json_path=None, today=None):
    """Reconcile the store against HANDOFF.md section 7, plus a ClickUp cache if given."""
    today = today or date.today().isoformat()
    items, flags = read_handoff(handoff_path)
    if clickup_cache_path is not None:
        cache = clickup_source.load_cache(clickup_cache_path)
        items = clickup_source.merge(items, cache)
        store.set_meta(conn, "last_clickup_pull", cache.get("pulled_on"))

    result = SyncResult()
    known = {q["id"]: q for q in store.list_questions(conn)}
    seen = set()

    for item in items:
        seen.add(item.id)
        before = known.get(item.id)
        if before is None:
            result.added.append(item.id)
        elif before["state"] == "closed" and before["state_source"] == "ui" \
                and item.state == "open":
            # The UI closed it in a meeting; section 7 has not caught up. Hand it back
            # rather than dropping it, then let the tracker win.
            result.reverted.append({
                "id": item.id,
                "title": before["title"],
                "note": before["closed_note"] or "",
                "closed_on": before["closed_on"] or "",
            })
        elif any(before[f] != getattr(item, f) for f in ("title", "why", "owner", "state")):
            result.updated.append(item.id)

        store.upsert_question(
            conn, id=item.id, title=item.title, why=item.why, owner=item.owner,
            priority=item.priority, raised=item.raised, state=item.state,
            clickup_id=item.clickup_id, clickup_status=item.clickup_status, due=item.due,
            actor="sync")

        row = store.get_question(conn, item.id)
        if row["difficulty_source"] != "user":
            store.set_question_difficulty(
                conn, item.id, estimate_difficulty(f"{item.title} {item.why}"),
                source="agent", actor="sync")

    for qid, row in known.items():
        if qid not in seen:
            result.stale.append(qid)
            store.upsert_flag(
                conn, key=f"stale:{qid}",
                text=(f"Item {qid} ({row['title']}) is in the dashboard but no longer in "
                      "HANDOFF.md section 7. Listed, not resolved."),
                first_seen=today)

    for flag in flags:
        result.flags.append(flag.key)
        store.upsert_flag(conn, key=flag.key, text=flag.text, first_seen=today,
                          resolved=1 if flag.resolved else 0)

    existing_origins = {a["origin"] for a in store.list_agenda(conn)}
    for flag in flags:
        if flag.resolved:
            continue
        origin = f"sync:flag:{flag.key}"
        if origin in existing_origins:
            continue
        store.add_agenda(conn, title=_flag_title(flag.text), detail=flag.text,
                         origin=origin, proposed_by="sync", created=today, actor="sync")
        result.agenda_proposed.append(origin)

    for row in store.list_questions(conn, state="open"):
        if row["priority"] != "high" and not (row["due"] and row["due"] <= today):
            continue
        origin = f"sync:item:{row['id']}"
        if origin in existing_origins:
            continue
        store.add_agenda(conn, title=f"Item {row['id']} — {row['title']}",
                         detail=row["why"], origin=origin, proposed_by="sync",
                         created=today, actor="sync")
        result.agenda_proposed.append(origin)

    store.set_meta(conn, "last_sync", today)
    if json_path is not None:
        export_json(conn, json_path)
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_sync.py -v`
Expected: PASS, 14 tests

- [ ] **Step 5: Commit**

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6
git add dashboard/project_tracker/sync.py dashboard/tests/test_sync.py
git commit -m "feat(dashboard): reconciliation with tracker-wins state and UI-owned fields"
```

---

### Task 6: pending-changes.md

**Files:**
- Create: `dashboard/project_tracker/report.py`
- Create: `dashboard/tests/test_report.py`

**Interfaces:**
- Consumes: `store` (Task 1), `SyncResult` (Task 5).
- Produces: `render_pending(conn, result, *, today=None) -> str`; `write_pending(conn, result, path, *, today=None) -> str`.

This file is the whole write-back mechanism. The dashboard never edits `HANDOFF.md`; it writes what it owes §7 here, for a person or an agent to fold in.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/tests/test_report.py`:

```python
from project_tracker.report import render_pending, write_pending
from project_tracker.store import (
    add_agenda,
    close_agenda,
    connect,
    upsert_question,
)
from project_tracker.sync import SyncResult


def _conn(tmp_path):
    conn = connect(tmp_path / "d.db")
    upsert_question(conn, id="1", title="Cost per gradient", why="", owner="Jonathon",
                    priority="high", raised="2026-09-10", state="open")
    return conn


def test_reverted_closures_are_listed_as_owed_section_seven_edits(tmp_path):
    conn = _conn(tmp_path)
    result = SyncResult(reverted=[{"id": "1", "title": "Cost per gradient",
                                   "note": "settled in the meeting",
                                   "closed_on": "2026-09-21"}])
    text = render_pending(conn, result, today="2026-09-21")
    assert "Owed to `HANDOFF.md` §7" in text
    assert "item 1" in text.lower()
    assert "settled in the meeting" in text


def test_decisions_taken_in_the_ui_are_listed(tmp_path):
    conn = _conn(tmp_path)
    aid = add_agenda(conn, title="Settle the cluster count", status="accepted")
    close_agenda(conn, aid, resolution="Ten, as in Figs. 2-4", resolved_by="Eun-Ah",
                 resolved_on="2026-09-21")
    text = render_pending(conn, SyncResult(), today="2026-09-21")
    assert "Ten, as in Figs. 2-4" in text
    assert "Eun-Ah" in text


def test_stale_items_are_listed(tmp_path):
    conn = _conn(tmp_path)
    text = render_pending(conn, SyncResult(stale=["21"]), today="2026-09-21")
    assert "21" in text
    assert "no longer in" in text.lower()


def test_a_clean_sync_says_so_rather_than_emitting_an_empty_file(tmp_path):
    conn = _conn(tmp_path)
    text = render_pending(conn, SyncResult(), today="2026-09-21")
    assert "Nothing owed" in text


def test_write_pending_creates_the_file_and_returns_its_text(tmp_path):
    conn = _conn(tmp_path)
    path = tmp_path / "pending-changes.md"
    text = write_pending(conn, SyncResult(), path, today="2026-09-21")
    assert path.read_text() == text
    assert path.read_text().startswith("# Pending changes")


def test_rendering_is_stable_across_repeated_calls(tmp_path):
    conn = _conn(tmp_path)
    result = SyncResult(stale=["21"])
    assert render_pending(conn, result, today="2026-09-21") == \
        render_pending(conn, result, today="2026-09-21")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_report.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'project_tracker.report'`

- [ ] **Step 3: Write the report**

Create `dashboard/project_tracker/report.py`:

```python
"""Render what the dashboard owes HANDOFF.md section 7.

The dashboard never edits section 7 — item numbers are born there and an agent
rewriting that ledger unreviewed is how it gets corrupted. Instead every state
change the UI made that the tracker does not know about lands here, in prose a
person can fold in.
"""

from datetime import date
from pathlib import Path

from . import store


def render_pending(conn, result, *, today=None):
    today = today or date.today().isoformat()
    lines = [f"# Pending changes — {today}", "",
             "Written by the meeting dashboard. Nothing here has been applied to",
             "`HANDOFF.md`; §7 is where items are born and closed, so these are edits",
             "owed to it.", ""]
    body = []

    if result.reverted:
        body.append("## Owed to `HANDOFF.md` §7 — closed in the UI, still open in §7")
        body.append("")
        for row in result.reverted:
            note = row.get("note") or "(no note recorded)"
            body.append(f"- **item {row['id']} — {row['title']}** closed "
                        f"{row.get('closed_on') or 'in the UI'}: {note}")
        body.append("")
        body.append("These were reverted to open, because §7 dictates state. Close them in")
        body.append("§7 to make the closure stick.")
        body.append("")

    decisions = store.list_decisions(conn)
    if decisions:
        body.append("## Decisions taken in the UI")
        body.append("")
        for d in decisions:
            who = d["decided_by"] or "unattributed"
            body.append(f"- **{d['decided_on']} — {d['text']}** ({who}). "
                        f"Grounds: {d['grounds']}")
            if d["followups"]:
                body.append(f"  - Follow-up: {', '.join(d['followups'])}")
        body.append("")

    if result.stale:
        body.append("## In the dashboard, no longer in §7")
        body.append("")
        for qid in result.stale:
            row = store.get_question(conn, qid)
            title = row["title"] if row else ""
            body.append(f"- **item {qid} — {title}**. Kept and flagged, not deleted.")
        body.append("")
        body.append("Either it was removed from §7 by mistake, or it should be retired here.")
        body.append("Listed, not resolved.")
        body.append("")

    if not body:
        body = ["Nothing owed. The dashboard and §7 agree.", ""]

    return "\n".join(lines + body).rstrip() + "\n"


def write_pending(conn, result, path, *, today=None):
    text = render_pending(conn, result, today=today)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_report.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6
git add dashboard/project_tracker/report.py dashboard/tests/test_report.py
git commit -m "feat(dashboard): pending-changes report as the write-back path to section 7"
```

---

### Task 7: HTTP server and JSON API

**Files:**
- Create: `dashboard/project_tracker/serve.py`
- Create: `dashboard/tests/test_serve.py`

**Interfaces:**
- Consumes: `store` (Task 1), `export` (Task 2).
- Produces: `make_server(conn, json_path, *, host="127.0.0.1", port=8765, static_dir=None) -> http.server.ThreadingHTTPServer`; `build_state(conn) -> dict`; `run(conn, json_path, *, host, port, static_dir)`.

Every mutating endpoint calls the matching `store` function and then `export_json`, so `dashboard.json` is never behind the database.

Endpoints:

| Method | Path | Body | Effect |
|---|---|---|---|
| GET | `/` | — | `index.html` |
| GET | `/static/<name>` | — | a file from `static/` |
| GET | `/api/state` | — | `build_state(conn)` |
| POST | `/api/questions/<id>/state` | `{"state": "closed", "note": "..."}` | `set_question_state(source="ui")` |
| POST | `/api/questions/<id>/difficulty` | `{"difficulty": "M"}` | `set_question_difficulty(source="user")` |
| POST | `/api/agenda` | `{"title": "...", "detail": "..."}` | `add_agenda(origin="manual", status="accepted")` |
| PATCH | `/api/agenda/<id>` | any of the agenda fields | `update_agenda` |
| POST | `/api/agenda/<id>/close` | `{"resolution": "...", "resolved_by": "...", "followups": [...]}` | `close_agenda` |
| POST | `/api/agenda/reorder` | `{"ids": [3, 1, 2]}` | `reorder_agenda` |
| PATCH | `/api/decisions/<id>` | `{"text": "...", "grounds": "..."}` | update a decision |

`build_state` adds one derived field per open question, `staleness`: the number of recorded meetings that happened on or after the item was raised. This is the heat counter from `dashboard_v7`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/tests/test_serve.py`:

```python
import json
import threading
import urllib.error
import urllib.request

import pytest

from project_tracker.export import export_json
from project_tracker.serve import build_state, make_server
from project_tracker.store import (
    add_agenda,
    connect,
    get_question,
    list_agenda,
    list_decisions,
    set_meta,
    upsert_question,
)


@pytest.fixture
def server(tmp_path):
    conn = connect(tmp_path / "d.db")
    upsert_question(conn, id="1", title="Cost per gradient", why="w", owner="Jonathon",
                    priority="high", raised="2026-09-10", state="open")
    upsert_question(conn, id="25", title="Stale ledger", why="", owner="", priority="",
                    raised="2026-09-20", state="closed")
    set_meta(conn, "meetings", [{"date": "2026-09-11", "label": "one"},
                                {"date": "2026-09-14", "label": "two"}])
    json_path = tmp_path / "dashboard.json"
    export_json(conn, json_path)
    httpd = make_server(conn, json_path, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, conn, json_path
    httpd.shutdown()
    httpd.server_close()
    conn.close()


def _get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return r.status, json.loads(r.read())


def _send(base, path, payload, method="POST"):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method=method)
    with urllib.request.urlopen(req) as r:
        return r.status, json.loads(r.read())


def test_state_endpoint_returns_every_section(server):
    base, _, _ = server
    status, doc = _get(base, "/api/state")
    assert status == 200
    assert {"questions", "agenda", "decisions", "flags", "meta"} <= set(doc)


def test_root_serves_the_ui(server):
    base, _, _ = server
    with urllib.request.urlopen(base + "/") as r:
        assert r.status == 200
        assert b"<html" in r.read().lower()


def test_build_state_counts_staleness_from_meetings(server):
    _, conn, _ = server
    state = build_state(conn)
    q = [x for x in state["questions"] if x["id"] == "1"][0]
    assert q["staleness"] == 2


def test_closing_a_question_persists_and_rewrites_the_json(server):
    base, conn, json_path = server
    status, _ = _send(base, "/api/questions/1/state",
                      {"state": "closed", "note": "settled"})
    assert status == 200
    row = get_question(conn, "1")
    assert row["state"] == "closed"
    assert row["state_source"] == "ui"
    assert row["closed_note"] == "settled"
    on_disk = json.loads(json_path.read_text())
    assert [q for q in on_disk["questions"] if q["id"] == "1"][0]["state"] == "closed"


def test_reopening_a_closed_question_works(server):
    base, conn, _ = server
    _send(base, "/api/questions/25/state", {"state": "open"})
    assert get_question(conn, "25")["state"] == "open"


def test_setting_difficulty_marks_it_user_owned(server):
    base, conn, _ = server
    _send(base, "/api/questions/1/difficulty", {"difficulty": "L"})
    row = get_question(conn, "1")
    assert row["difficulty"] == "L"
    assert row["difficulty_source"] == "user"


def test_adding_an_agenda_item_returns_its_id(server):
    base, conn, _ = server
    status, body = _send(base, "/api/agenda", {"title": "Ask about the venue"})
    assert status == 200
    assert any(a["id"] == body["id"] for a in list_agenda(conn))


def test_closing_an_agenda_item_creates_a_decision_and_removes_it_from_the_agenda(server):
    base, conn, _ = server
    aid = add_agenda(conn, title="Settle the cluster count", status="accepted")
    status, body = _send(base, f"/api/agenda/{aid}/close",
                         {"resolution": "Ten, as in Figs. 2-4", "resolved_by": "Eun-Ah",
                          "followups": ["Jonathon"]})
    assert status == 200
    assert body["decision_id"] == list_decisions(conn)[0]["id"]
    assert [a for a in list_agenda(conn) if a["id"] == aid][0]["status"] == "closed"
    assert list_decisions(conn)[0]["followups"] == ["Jonathon"]


def test_closing_an_agenda_item_without_a_resolution_is_a_400(server):
    base, conn, _ = server
    aid = add_agenda(conn, title="x", status="accepted")
    with pytest.raises(urllib.error.HTTPError) as exc:
        _send(base, f"/api/agenda/{aid}/close", {"resolution": "", "resolved_by": "E"})
    assert exc.value.code == 400


def test_patching_an_agenda_item_updates_it(server):
    base, conn, _ = server
    aid = add_agenda(conn, title="x")
    _send(base, f"/api/agenda/{aid}", {"status": "accepted", "title": "y"},
          method="PATCH")
    row = [a for a in list_agenda(conn) if a["id"] == aid][0]
    assert row["status"] == "accepted"
    assert row["title"] == "y"


def test_reordering_the_agenda_persists(server):
    base, conn, _ = server
    a = add_agenda(conn, title="first", status="accepted")
    b = add_agenda(conn, title="second", status="accepted")
    _send(base, "/api/agenda/reorder", {"ids": [b, a]})
    ordered = [x["id"] for x in list_agenda(conn, status="accepted")]
    assert ordered == [b, a]


def test_unknown_question_is_a_404(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _send(base, "/api/questions/999/state", {"state": "closed"})
    assert exc.value.code == 404


def test_unknown_route_is_a_404(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(base, "/api/nonsense")
    assert exc.value.code == 404


def test_server_binds_to_loopback_only(server):
    base, _, _ = server
    assert base.startswith("http://127.0.0.1:")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_serve.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'project_tracker.serve'`

- [ ] **Step 3: Write the server**

Create `dashboard/project_tracker/serve.py`:

```python
"""HTTP surface for the dashboard.

Routing and serialisation only — every rule lives in store.py. Each mutation
rewrites dashboard.json before replying, so what git carries is never behind
what the browser has just done, and there is no save button to forget.
"""

import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import store
from .export import export_json

STATIC = Path(__file__).parent / "static"

_Q_STATE = re.compile(r"^/api/questions/(?P<id>[^/]+)/state$")
_Q_DIFF = re.compile(r"^/api/questions/(?P<id>[^/]+)/difficulty$")
_A_CLOSE = re.compile(r"^/api/agenda/(?P<id>\d+)/close$")
_A_ONE = re.compile(r"^/api/agenda/(?P<id>\d+)$")
_D_ONE = re.compile(r"^/api/decisions/(?P<id>\d+)$")


class ApiError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def build_state(conn):
    """The whole board, plus the derived staleness counter the UI colours by."""
    meetings = store.get_meta(conn, "meetings", default=[]) or []
    dates = sorted(m["date"] for m in meetings if m.get("date"))

    questions = []
    for row in store.list_questions(conn):
        row = dict(row)
        raised = row.get("raised") or ""
        row["staleness"] = sum(1 for d in dates if raised and d >= raised)
        questions.append(row)

    return {
        "meta": {
            "meetings": meetings,
            "last_sync": store.get_meta(conn, "last_sync"),
            "last_clickup_pull": store.get_meta(conn, "last_clickup_pull"),
        },
        "questions": questions,
        "agenda": store.list_agenda(conn),
        "decisions": store.list_decisions(conn),
        "flags": store.list_flags(conn),
    }


def _handle_api(conn, json_path, method, path, payload):
    if method == "GET" and path == "/api/state":
        return build_state(conn)

    if method == "POST" and (m := _Q_STATE.match(path)):
        state = payload.get("state")
        if state not in ("open", "closed"):
            raise ApiError(400, "state must be 'open' or 'closed'")
        try:
            store.set_question_state(conn, m.group("id"), state, source="ui",
                                     note=payload.get("note"), on=payload.get("on"),
                                     actor="ui")
        except KeyError:
            raise ApiError(404, f"no question {m.group('id')}")
        return {"ok": True}

    if method == "POST" and (m := _Q_DIFF.match(path)):
        try:
            store.set_question_difficulty(conn, m.group("id"), payload.get("difficulty"),
                                          source="user", actor="ui")
        except KeyError:
            raise ApiError(404, f"no question {m.group('id')}")
        except ValueError as exc:
            raise ApiError(400, str(exc))
        return {"ok": True}

    if method == "POST" and path == "/api/agenda":
        title = (payload.get("title") or "").strip()
        if not title:
            raise ApiError(400, "an agenda item needs a title")
        aid = store.add_agenda(conn, title=title, detail=payload.get("detail", ""),
                               origin="manual", proposed_by=payload.get("proposed_by", ""),
                               status=payload.get("status", "accepted"), actor="ui")
        return {"id": aid}

    if method == "POST" and path == "/api/agenda/reorder":
        store.reorder_agenda(conn, payload.get("ids") or [], actor="ui")
        return {"ok": True}

    if method == "POST" and (m := _A_CLOSE.match(path)):
        try:
            did = store.close_agenda(
                conn, int(m.group("id")), resolution=payload.get("resolution", ""),
                resolved_by=payload.get("resolved_by", ""),
                followups=payload.get("followups"), resolved_on=payload.get("resolved_on"),
                actor="ui")
        except KeyError:
            raise ApiError(404, f"no agenda item {m.group('id')}")
        except ValueError as exc:
            raise ApiError(400, str(exc))
        return {"decision_id": did}

    if method == "PATCH" and (m := _A_ONE.match(path)):
        try:
            store.update_agenda(conn, int(m.group("id")), actor="ui", **payload)
        except KeyError:
            raise ApiError(404, f"no agenda item {m.group('id')}")
        except ValueError as exc:
            raise ApiError(400, str(exc))
        return {"ok": True}

    if method == "PATCH" and (m := _D_ONE.match(path)):
        text, grounds = payload.get("text"), payload.get("grounds")
        if grounds is not None and not grounds.strip():
            raise ApiError(400, "a decision requires grounds")
        sets, values = [], []
        for column, value in (("text", text), ("grounds", grounds),
                              ("decided_by", payload.get("decided_by"))):
            if value is not None:
                sets.append(f"{column}=?")
                values.append(value)
        if sets:
            conn.execute(f"UPDATE decisions SET {','.join(sets)} WHERE id=?",
                         (*values, int(m.group("id"))))
            conn.commit()
        return {"ok": True}

    raise ApiError(404, f"no route for {method} {path}")


def _make_handler(conn, json_path, static_dir):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MeetingDashboard/1"

        def log_message(self, fmt, *args):  # quieter than the default
            pass

        def _reply(self, code, body, content_type="application/json"):
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _static(self, name):
            target = (static_dir / name).resolve()
            if static_dir.resolve() not in target.parents or not target.is_file():
                self._reply(404, {"error": "not found"})
                return
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._reply(200, target.read_bytes(), ctype)

        def _dispatch(self, method):
            path = self.path.split("?", 1)[0]
            if method == "GET" and path in ("/", "/index.html"):
                self._static("index.html")
                return
            if method == "GET" and path.startswith("/static/"):
                self._static(path[len("/static/"):])
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._reply(400, {"error": "body is not valid JSON"})
                return
            try:
                result = _handle_api(conn, json_path, method, path, payload)
            except ApiError as exc:
                self._reply(exc.code, {"error": exc.message})
                return
            if method != "GET":
                export_json(conn, json_path)
            self._reply(200, result)

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_PATCH(self):
            self._dispatch("PATCH")

    return Handler


def make_server(conn, json_path, *, host="127.0.0.1", port=8765, static_dir=None):
    static_dir = Path(static_dir) if static_dir else STATIC
    return ThreadingHTTPServer((host, port), _make_handler(conn, json_path, static_dir))


def run(conn, json_path, *, host="127.0.0.1", port=8765, static_dir=None):
    httpd = make_server(conn, json_path, host=host, port=port, static_dir=static_dir)
    bound = httpd.server_address[1]
    print(f"Meeting dashboard: http://{host}:{bound}")
    print("  VS Code Remote-SSH forwards this port automatically — the link is clickable.")
    print(f"  On a plain terminal: ssh -L {bound}:localhost:{bound} <this-host>")
    print("  Ctrl-C to stop. Every change is saved as you make it.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
```

- [ ] **Step 4: Create a placeholder index.html so the static tests pass**

Create `dashboard/project_tracker/static/index.html`:

```html
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Meeting dashboard</title></head>
<body><div id="app">Loading…</div></body></html>
```

Task 8 replaces this file entirely.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_serve.py -v`
Expected: PASS, 14 tests

- [ ] **Step 6: Commit**

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6
git add dashboard/project_tracker/serve.py dashboard/project_tracker/static/index.html dashboard/tests/test_serve.py
git commit -m "feat(dashboard): localhost JSON API with write-through to dashboard.json"
```

---

### Task 8: The three-section UI

**Files:**
- Modify (replace entirely): `dashboard/project_tracker/static/index.html`
- Create: `dashboard/project_tracker/static/style.css`
- Create: `dashboard/project_tracker/static/app.js`

**Interfaces:**
- Consumes: `GET /api/state` and the mutating endpoints from Task 7.
- Produces: no Python interface.

Palette, carried over from `dashboard_v7.jsx` so the board still looks like the one the group knows:

```
bg #EEF1F4   surface #FFFFFF   ink #14202A   muted #5C6B78
faint #8A98A4   rule #D3DBE2   accent #1F5673   accentSoft #E4EDF2
staleness heat: 0 #6E8494 · 1 #B98A2A · 2 #B0562A · 3+ #97231C
```

- [ ] **Step 1: Write the stylesheet**

Create `dashboard/project_tracker/static/style.css`:

```css
:root {
  --bg: #EEF1F4; --surface: #FFFFFF; --ink: #14202A; --muted: #5C6B78;
  --faint: #8A98A4; --rule: #D3DBE2; --accent: #1F5673; --accent-soft: #E4EDF2;
  --heat-0: #6E8494; --heat-1: #B98A2A; --heat-2: #B0562A; --heat-3: #97231C;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
header { padding: 20px 24px 8px; }
header h1 { margin: 0; font-size: 20px; font-weight: 600; }
header .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
nav { display: flex; gap: 4px; padding: 0 24px; border-bottom: 1px solid var(--rule); }
nav button {
  background: none; border: 0; border-bottom: 2px solid transparent; cursor: pointer;
  padding: 10px 14px; font: inherit; color: var(--muted);
}
nav button[aria-selected="true"] { color: var(--accent); border-bottom-color: var(--accent); }
main { padding: 20px 24px 60px; max-width: 980px; }
section[hidden] { display: none; }
h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 24px 0 10px; }
.card {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 12px 14px; margin-bottom: 8px;
}
.card.closed { opacity: .72; }
.card .row { display: flex; gap: 10px; align-items: baseline; }
.card .num { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--accent); }
.card .title { font-weight: 600; flex: 1; }
.card .why { color: var(--muted); font-size: 13px; margin-top: 6px; }
.meta { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; font-size: 12px; color: var(--faint); }
.tag { background: var(--accent-soft); color: var(--accent); border-radius: 3px; padding: 1px 6px; }
.heat { color: #fff; border-radius: 3px; padding: 1px 6px; }
button.act {
  font: inherit; font-size: 13px; cursor: pointer; background: var(--surface);
  border: 1px solid var(--rule); border-radius: 4px; padding: 4px 10px; color: var(--ink);
}
button.act:hover { border-color: var(--accent); color: var(--accent); }
textarea, input[type=text] {
  width: 100%; font: inherit; font-size: 13px; padding: 6px 8px; margin-top: 8px;
  border: 1px solid var(--rule); border-radius: 4px; background: var(--surface); color: var(--ink);
}
.grounds { font-size: 13px; color: var(--muted); margin-top: 4px; }
.daygroup { margin-top: 18px; }
.daygroup > .date { font-weight: 600; font-size: 13px; color: var(--accent); margin-bottom: 6px; }
.empty { color: var(--faint); font-size: 13px; padding: 8px 0; }
.error { background: #97231C; color: #fff; padding: 8px 24px; font-size: 13px; }
select { font: inherit; font-size: 12px; }
```

- [ ] **Step 2: Write the page**

Replace `dashboard/project_tracker/static/index.html` entirely:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Meeting dashboard</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <div id="error" class="error" hidden></div>
  <header>
    <h1>State of project</h1>
    <div class="sub" id="subtitle"></div>
  </header>
  <nav>
    <button id="tab-questions" aria-selected="true" data-panel="questions">Open questions</button>
    <button id="tab-agenda" aria-selected="false" data-panel="agenda">Meeting agenda</button>
    <button id="tab-decisions" aria-selected="false" data-panel="decisions">Decisions</button>
  </nav>
  <main>
    <section id="questions"></section>
    <section id="agenda" hidden></section>
    <section id="decisions" hidden></section>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Write the client**

Create `dashboard/project_tracker/static/app.js`:

```javascript
"use strict";

const PRIORITY_RANK = { high: 0, urgent: 0, normal: 1, medium: 1, "": 2, low: 3 };
const DIFFICULTY_RANK = { S: 0, M: 1, L: 2, null: 3 };
const HEAT = ["var(--heat-0)", "var(--heat-1)", "var(--heat-2)", "var(--heat-3)"];

let state = null;

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) node.appendChild(c);
  return node;
}

function showError(message) {
  const bar = document.getElementById("error");
  bar.textContent = message;
  bar.hidden = !message;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `${response.status} on ${path}`);
  return body;
}

async function mutate(path, payload, method = "POST") {
  try {
    showError("");
    await api(path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await load();
  } catch (err) {
    showError(err.message);
  }
}

function sortQuestions(questions) {
  return questions.slice().sort((a, b) => {
    const p = (PRIORITY_RANK[a.priority] ?? 2) - (PRIORITY_RANK[b.priority] ?? 2);
    if (p !== 0) return p;
    const d = (DIFFICULTY_RANK[a.difficulty] ?? 3) - (DIFFICULTY_RANK[b.difficulty] ?? 3);
    if (d !== 0) return d;
    return (b.staleness || 0) - (a.staleness || 0);
  });
}

function difficultyPicker(question) {
  const select = el("select", {
    onchange: (e) =>
      mutate(`/api/questions/${question.id}/difficulty`, { difficulty: e.target.value || null }),
  });
  for (const value of ["", "S", "M", "L"]) {
    const option = el("option", { value, text: value || "difficulty?" });
    if ((question.difficulty || "") === value) option.selected = true;
    select.appendChild(option);
  }
  return select;
}

function questionCard(question) {
  const closed = question.state === "closed";
  const heat = el("span", {
    class: "heat",
    text: `${question.staleness || 0} mtg`,
    style: `background:${HEAT[Math.min(question.staleness || 0, 3)]}`,
  });
  const meta = el("div", { class: "meta" }, [
    question.owner ? el("span", { class: "tag", text: question.owner }) : null,
    question.priority ? el("span", { class: "tag", text: question.priority }) : null,
    closed ? null : heat,
    question.clickup_status ? el("span", { text: `ClickUp: ${question.clickup_status}` }) : null,
    question.due ? el("span", { text: `due ${question.due}` }) : null,
    question.raised ? el("span", { text: `raised ${question.raised}` }) : null,
    closed && question.closed_on ? el("span", { text: `closed ${question.closed_on}` }) : null,
  ]);
  if (!closed) meta.appendChild(difficultyPicker(question));
  else if (question.difficulty) meta.appendChild(el("span", { text: question.difficulty }));

  const toggle = el("button", {
    class: "act",
    text: closed ? "Reopen" : "Close",
    onclick: () => {
      if (closed) return mutate(`/api/questions/${question.id}/state`, { state: "open" });
      const note = window.prompt("How did it close? (recorded, and owed to HANDOFF §7)");
      if (note === null) return;
      return mutate(`/api/questions/${question.id}/state`, { state: "closed", note });
    },
  });

  return el("div", { class: closed ? "card closed" : "card" }, [
    el("div", { class: "row" }, [
      el("span", { class: "num", text: question.id }),
      el("span", { class: "title", text: question.title }),
      toggle,
    ]),
    question.why ? el("div", { class: "why", text: question.why }) : null,
    meta,
    closed && question.closed_note
      ? el("div", { class: "grounds", text: question.closed_note })
      : null,
  ]);
}

function renderQuestions() {
  const root = document.getElementById("questions");
  root.replaceChildren();
  const open = sortQuestions(state.questions.filter((q) => q.state === "open"));
  const closed = state.questions
    .filter((q) => q.state === "closed")
    .sort((a, b) => (b.closed_on || "").localeCompare(a.closed_on || ""));

  root.appendChild(el("h2", { text: `Open — ${open.length}` }));
  if (!open.length) root.appendChild(el("div", { class: "empty", text: "Nothing open." }));
  for (const q of open) root.appendChild(questionCard(q));

  root.appendChild(el("h2", { text: `Closed — ${closed.length}` }));
  if (!closed.length) root.appendChild(el("div", { class: "empty", text: "Nothing closed yet." }));
  for (const q of closed) root.appendChild(questionCard(q));
}

function agendaCard(item) {
  const resolution = el("textarea", {
    rows: 2,
    placeholder: "How was it resolved? (becomes the decision's grounds)",
  });
  resolution.value = item.resolution || "";
  const followups = el("input", {
    type: "text",
    placeholder: "Follow-up people, comma separated (optional)",
  });
  followups.value = (item.followups || []).join(", ");
  const by = el("input", { type: "text", placeholder: "Resolved by" });

  const close = el("button", {
    class: "act",
    text: "Close → Decisions",
    onclick: () => {
      if (!resolution.value.trim()) {
        showError("An agenda item needs a resolution before it can be closed.");
        return;
      }
      return mutate(`/api/agenda/${item.id}/close`, {
        resolution: resolution.value,
        resolved_by: by.value,
        followups: followups.value.split(",").map((s) => s.trim()).filter(Boolean),
      });
    },
  });

  return el("div", { class: "card" }, [
    el("div", { class: "row" }, [
      el("span", { class: "title", text: item.title }),
      close,
    ]),
    item.detail ? el("div", { class: "why", text: item.detail }) : null,
    resolution,
    followups,
    by,
  ]);
}

function proposedCard(item) {
  return el("div", { class: "card" }, [
    el("div", { class: "row" }, [
      el("span", { class: "title", text: item.title }),
      el("button", {
        class: "act",
        text: "Accept",
        onclick: () => mutate(`/api/agenda/${item.id}`, { status: "accepted" }, "PATCH"),
      }),
      el("button", {
        class: "act",
        text: "Dismiss",
        onclick: () => mutate(`/api/agenda/${item.id}`, { status: "dismissed" }, "PATCH"),
      }),
    ]),
    item.detail ? el("div", { class: "why", text: item.detail }) : null,
    el("div", { class: "meta" }, [el("span", { class: "tag", text: item.origin })]),
  ]);
}

function renderAgenda() {
  const root = document.getElementById("agenda");
  root.replaceChildren();
  const accepted = state.agenda.filter((a) => a.status === "accepted");
  const proposed = state.agenda.filter((a) => a.status === "proposed");

  root.appendChild(el("h2", { text: "Agenda" }));
  if (!accepted.length) {
    root.appendChild(el("div", { class: "empty", text: "Nothing on the agenda." }));
  }
  for (const item of accepted) root.appendChild(agendaCard(item));

  const title = el("input", { type: "text", placeholder: "Add an action item for the meeting" });
  root.appendChild(
    el("div", { class: "card" }, [
      title,
      el("button", {
        class: "act",
        text: "Add",
        onclick: () => {
          if (!title.value.trim()) return;
          return mutate("/api/agenda", { title: title.value, status: "accepted" });
        },
      }),
    ])
  );

  if (proposed.length) {
    root.appendChild(el("h2", { text: `Proposed by sync — ${proposed.length}` }));
    for (const item of proposed) root.appendChild(proposedCard(item));
  }
}

function renderDecisions() {
  const root = document.getElementById("decisions");
  root.replaceChildren();
  if (!state.decisions.length) {
    root.appendChild(el("div", { class: "empty", text: "No decisions recorded yet." }));
    return;
  }
  const byDate = new Map();
  for (const d of state.decisions) {
    if (!byDate.has(d.decided_on)) byDate.set(d.decided_on, []);
    byDate.get(d.decided_on).push(d);
  }
  for (const [day, decisions] of byDate) {
    const group = el("div", { class: "daygroup" }, [el("div", { class: "date", text: day })]);
    for (const d of decisions) {
      group.appendChild(
        el("div", { class: "card" }, [
          el("div", { class: "row" }, [el("span", { class: "title", text: d.text })]),
          el("div", { class: "grounds", text: `Grounds: ${d.grounds}` }),
          el("div", { class: "meta" }, [
            d.decided_by ? el("span", { class: "tag", text: d.decided_by }) : null,
            (d.followups || []).length
              ? el("span", { text: `follow-up: ${d.followups.join(", ")}` })
              : null,
          ]),
        ])
      );
    }
    root.appendChild(group);
  }
}

function renderSubtitle() {
  const parts = [];
  if (state.meta.last_sync) parts.push(`synced ${state.meta.last_sync}`);
  if (state.meta.last_clickup_pull) parts.push(`ClickUp ${state.meta.last_clickup_pull}`);
  const unresolved = state.flags.filter((f) => !f.resolved).length;
  if (unresolved) parts.push(`${unresolved} open flag${unresolved === 1 ? "" : "s"}`);
  document.getElementById("subtitle").textContent = parts.join(" · ");
}

function render() {
  renderSubtitle();
  renderQuestions();
  renderAgenda();
  renderDecisions();
}

async function load() {
  try {
    state = await api("/api/state");
    render();
  } catch (err) {
    showError(err.message);
  }
}

for (const button of document.querySelectorAll("nav button")) {
  button.addEventListener("click", () => {
    for (const other of document.querySelectorAll("nav button")) {
      const selected = other === button;
      other.setAttribute("aria-selected", String(selected));
      document.getElementById(other.dataset.panel).hidden = !selected;
    }
  });
}

load();
```

- [ ] **Step 4: Verify the server still serves and the API tests still pass**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/ -v`
Expected: PASS, all tests

- [ ] **Step 5: Commit**

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6
git add dashboard/project_tracker/static/
git commit -m "feat(dashboard): three-section UI writing through to the store"
```

---

### Task 9: CLI, README, and the research-repo pointer

**Files:**
- Create: `dashboard/dashboard` (executable)
- Create: `dashboard/README.md`
- Create: `dashboard/tests/test_cli.py`
- Modify: `/home/jek354/research/project-tracker/WHERE-THINGS-LIVE.md` (in the **research** repo)

**Interfaces:**
- Consumes: `sync` (Task 5), `report` (Task 6), `serve` (Task 7), `export` (Task 2).
- Produces: `main(argv=None) -> int`, importable as `project_tracker.cli`.

Commands:

| Command | Effect |
|---|---|
| `./dashboard sync` | Reconcile, write `dashboard.json` and `pending-changes.md`, print a summary. Does not serve. |
| `./dashboard serve` | Bootstrap the DB from JSON if needed, then serve. No sync. This is the command a human runs. |
| `./dashboard update` | `sync` then `serve`. This is the command an agent runs. |
| `./dashboard export` | Rewrite `dashboard.json` from the DB and exit. |

Flags: `--clickup` (use the ClickUp cache), `--port N`, `--host H`, `--handoff PATH`, `--data DIR`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/tests/test_cli.py`:

```python
import json

import pytest

from project_tracker.cli import main

HANDOFF = """\
## 7. Open items (state at 2026-09-21)

### Open — Jonathon

**1. Cost per gradient** — wall-clock per gradient. *Asked 2026-09-10; task `86akhc9zz`.*

### Closed

**25. Stale ledger** — CLOSED 2026-09-21.

### 7.1 Flags — disagreements (2026-09-21)

1. **Ten clusters or fourteen?** Someone has to say which.

## 8. Next
"""


@pytest.fixture
def workspace(tmp_path):
    handoff = tmp_path / "HANDOFF.md"
    handoff.write_text(HANDOFF)
    data = tmp_path / "data"
    return handoff, data


def test_sync_creates_the_json_and_pending_files(workspace, capsys):
    handoff, data = workspace
    code = main(["sync", "--handoff", str(handoff), "--data", str(data)])
    assert code == 0
    doc = json.loads((data / "dashboard.json").read_text())
    assert {q["id"] for q in doc["questions"]} == {"1", "25"}
    assert (data / "pending-changes.md").exists()


def test_sync_prints_a_summary(workspace, capsys):
    handoff, data = workspace
    main(["sync", "--handoff", str(handoff), "--data", str(data)])
    out = capsys.readouterr().out
    assert "added" in out.lower()
    assert "2" in out


def test_second_sync_leaves_the_json_byte_identical(workspace):
    handoff, data = workspace
    main(["sync", "--handoff", str(handoff), "--data", str(data)])
    first = (data / "dashboard.json").read_bytes()
    main(["sync", "--handoff", str(handoff), "--data", str(data)])
    assert (data / "dashboard.json").read_bytes() == first


def test_export_rewrites_the_json_without_syncing(workspace):
    handoff, data = workspace
    main(["sync", "--handoff", str(handoff), "--data", str(data)])
    (data / "dashboard.json").unlink()
    assert main(["export", "--data", str(data)]) == 0
    assert (data / "dashboard.json").exists()


def test_a_missing_handoff_file_fails_loudly(tmp_path, capsys):
    code = main(["sync", "--handoff", str(tmp_path / "nope.md"),
                 "--data", str(tmp_path / "data")])
    assert code != 0
    assert "nope.md" in capsys.readouterr().err


def test_unparseable_handoff_fails_loudly(tmp_path, capsys):
    handoff = tmp_path / "HANDOFF.md"
    handoff.write_text("# Nothing resembling section seven\n")
    code = main(["sync", "--handoff", str(handoff), "--data", str(tmp_path / "data")])
    assert code != 0
    assert "section 7" in capsys.readouterr().err.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_cli.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'project_tracker.cli'`

- [ ] **Step 3: Write the CLI module**

Create `dashboard/project_tracker/cli.py`:

```python
"""Command line entry point.

Four verbs, and only two of them matter day to day: `serve` for a person and
`update` for an agent. Everything else is plumbing.
"""

import argparse
import sys
from pathlib import Path

from . import serve as serve_module
from .export import ensure_db, export_json
from .report import write_pending
from .sources.handoff import HandoffParseError
from .sync import sync as run_sync

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data"
DEFAULT_HANDOFF = Path(__file__).resolve().parents[2] / "HANDOFF.md"


def _paths(args):
    data = Path(args.data) if args.data else DEFAULT_DATA
    return {
        "db": data / "dashboard.db",
        "json": data / "dashboard.json",
        "pending": data / "pending-changes.md",
        "clickup": data / "clickup-cache.json",
    }


def _do_sync(conn, args, paths):
    handoff = Path(args.handoff) if args.handoff else DEFAULT_HANDOFF
    if not handoff.exists():
        print(f"error: no HANDOFF.md at {handoff}", file=sys.stderr)
        return None
    try:
        result = run_sync(
            conn, handoff_path=handoff,
            clickup_cache_path=paths["clickup"] if args.clickup else None,
            json_path=paths["json"])
    except HandoffParseError as exc:
        print(f"error: could not parse section 7 of {handoff}: {exc}", file=sys.stderr)
        return None
    write_pending(conn, result, paths["pending"])
    print(f"added {len(result.added)} · updated {len(result.updated)} · "
          f"reverted {len(result.reverted)} · stale {len(result.stale)} · "
          f"flags {len(result.flags)} · agenda proposed {len(result.agenda_proposed)}")
    if result.reverted:
        print(f"  {len(result.reverted)} UI closure(s) owed to HANDOFF §7 — "
              f"see {paths['pending']}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dashboard",
                                     description="The Kim group meeting dashboard.")
    parser.add_argument("command", choices=["sync", "serve", "update", "export"])
    parser.add_argument("--handoff", help="path to HANDOFF.md")
    parser.add_argument("--data", help="directory holding the database and JSON")
    parser.add_argument("--clickup", action="store_true",
                        help="merge the cached ClickUp snapshot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    paths = _paths(args)
    conn = ensure_db(paths["db"], paths["json"])

    if args.command in ("sync", "update"):
        if _do_sync(conn, args, paths) is None:
            return 1
        if args.command == "sync":
            return 0

    if args.command == "export":
        export_json(conn, paths["json"])
        print(f"wrote {paths['json']}")
        return 0

    serve_module.run(conn, paths["json"], host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Write the executable wrapper**

Create `dashboard/dashboard`:

```python
#!/usr/bin/env python3
"""Entry point: ./dashboard serve | sync | update | export"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from project_tracker.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
```

Then: `chmod +x /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard/dashboard`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/test_cli.py -v`
Expected: PASS, 6 tests

- [ ] **Step 6: Write the README**

Create `dashboard/README.md`:

```markdown
# Meeting dashboard

The state-of-project board: open questions, the meeting agenda, and the decisions
those agenda items became. It reads `HANDOFF.md` §7 and writes its own database;
it never edits `HANDOFF.md`.

## Running it

```bash
cd 68f6bef0eaaf5c0928a922c6/dashboard
./dashboard serve
```

That serves the current database at <http://127.0.0.1:8765> and prints the link.
Under VS Code Remote-SSH the port is forwarded automatically and the link is
clickable. On a plain terminal, run the `ssh -L` line the server prints.

Every change you make in the browser is saved as you make it. There is no save
button and nothing is lost when you close the tab.

## Refreshing it from HANDOFF.md

```bash
./dashboard update            # reconcile against §7, then serve
./dashboard update --clickup  # also merge the cached ClickUp snapshot
```

`sync` does the reconciliation without serving. `export` rewrites `dashboard.json`
from the database.

## What lives where

| File | |
|---|---|
| `data/dashboard.json` | The board. **Committed** — this is what a clone carries. |
| `data/dashboard.db` | Local SQLite cache. **Gitignored**, rebuilt from the JSON automatically. |
| `data/pending-changes.md` | What the dashboard owes `HANDOFF.md` §7. **Committed.** |
| `data/clickup-cache.json` | Last ClickUp pull. **Committed**, refreshed only on request. |

## The rules it follows

- **Item numbers come from `HANDOFF.md` §7 and nowhere else.** Closed items keep theirs.
- **§7 dictates open and closed.** If you close a question here and §7 still lists it
  open, the next sync reopens it — and records the closure in `pending-changes.md`, so
  it is owed to §7 rather than lost.
- **The UI owns what the UI creates**: resolutions, decisions, follow-up people, the
  curated agenda, and any difficulty you set by hand. Sync never overwrites them.
- **A decision cannot be recorded without grounds.**
- **Flags are listed, never resolved automatically.**

## Getting a colleague set up

`git pull`, then `./dashboard serve`. The database rebuilds itself from the committed
JSON. Nothing to install: it is Python standard library only.
```

- [ ] **Step 7: Run the whole suite**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/ -v`
Expected: PASS, all tests

- [ ] **Step 8: Commit the dashboard**

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6
git add dashboard/dashboard dashboard/README.md dashboard/project_tracker/cli.py dashboard/tests/test_cli.py
git commit -m "feat(dashboard): CLI entry point and README"
```

- [ ] **Step 9: Point the research repo at the dashboard's real home**

In the **research** repo, edit `project-tracker/WHERE-THINGS-LIVE.md`. Replace the URL line under `## The state-of-project dashboard` — currently `https://claude.ai/artifacts/latest/4df53bc3-a0d4-49b9-bc8f-21c79b86411b` — with:

```markdown
Run it locally: `cd 68f6bef0eaaf5c0928a922c6/dashboard && ./dashboard serve`

It lives in the Overleaf repo, so a `git pull` gives you both the code and the data.
See `dashboard/README.md`. There is no shared URL any more, and no rebuild that mints
a new one: the board is a local server over a database that travels in git.
```

Then delete the paragraph beginning **Read-only.** and the paragraph beginning **This file holds the link**, since neither is true any more. Leave the rest of the file alone.

```bash
cd /home/jek354/research
git add project-tracker/WHERE-THINGS-LIVE.md
git commit -m "docs: point WHERE-THINGS-LIVE at the local dashboard"
```

---

### Task 10: End-to-end verification against the real ledger

**Files:**
- Create: `dashboard/tests/test_end_to_end.py`

**Interfaces:**
- Consumes: everything.
- Produces: nothing.

This task is the one that decides whether the thing works. It runs against the real `HANDOFF.md`, and it ends with the user opening the page themselves.

- [ ] **Step 1: Write the end-to-end test**

Create `dashboard/tests/test_end_to_end.py`:

```python
"""One pass through the whole loop, against the real HANDOFF.md."""

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from project_tracker.export import ensure_db, export_json
from project_tracker.report import write_pending
from project_tracker.serve import make_server
from project_tracker.store import connect, get_question, list_decisions
from project_tracker.sync import sync

REAL = Path(__file__).resolve().parents[2] / "HANDOFF.md"
pytestmark = pytest.mark.skipif(not REAL.exists(), reason="HANDOFF.md not present")


def _post(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_sync_then_edit_then_resync_keeps_ui_work_and_reports_it(tmp_path):
    db, js = tmp_path / "d.db", tmp_path / "dashboard.json"
    conn = connect(db)

    first = sync(conn, handoff_path=REAL, json_path=js)
    assert len(first.added) > 10, "the real section 7 should yield a substantial board"

    httpd = make_server(conn, js, host="127.0.0.1", port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        open_id = [q["id"] for q in json.loads(js.read_text())["questions"]
                   if q["state"] == "open"][0]
        _post(base, f"/api/questions/{open_id}/state",
              {"state": "closed", "note": "closed live in the meeting"})
        assert get_question(conn, open_id)["state"] == "closed"

        aid = _post(base, "/api/agenda",
                    {"title": "Confirm the venue fork", "status": "accepted"})["id"]
        _post(base, f"/api/agenda/{aid}/close",
              {"resolution": "PRR now, PRX Quantum on a win", "resolved_by": "Eun-Ah"})
        assert len(list_decisions(conn)) == 1
    finally:
        httpd.shutdown()
        httpd.server_close()

    second = sync(conn, handoff_path=REAL, json_path=js)
    assert get_question(conn, open_id)["state"] == "open", "the tracker dictates state"
    assert any(r["id"] == open_id for r in second.reverted)
    assert len(list_decisions(conn)) == 1, "sync must never remove a decision"

    pending = write_pending(conn, second, tmp_path / "pending-changes.md")
    assert "closed live in the meeting" in pending
    assert "PRR now, PRX Quantum on a win" in pending


def test_a_fresh_clone_can_serve_from_the_committed_json_alone(tmp_path):
    db, js = tmp_path / "d.db", tmp_path / "dashboard.json"
    conn = connect(db)
    sync(conn, handoff_path=REAL, json_path=js)
    export_json(conn, js)
    conn.close()

    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "dashboard.json").write_text(js.read_text())

    fresh = ensure_db(clone / "dashboard.db", clone / "dashboard.json")
    httpd = make_server(fresh, clone / "dashboard.json", host="127.0.0.1", port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/state") as r:
            state = json.loads(r.read())
        assert len(state["questions"]) > 10
    finally:
        httpd.shutdown()
        httpd.server_close()
```

- [ ] **Step 2: Run the whole suite**

Run: `cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard && python3 -m pytest tests/ -v`
Expected: PASS, every test in every module

- [ ] **Step 3: Run a real sync and inspect what it produced**

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard
./dashboard sync
head -40 data/dashboard.json
cat data/pending-changes.md
```

Expected: a summary line with a non-zero `added` count; `dashboard.json` containing every §7 item; `pending-changes.md` saying nothing is owed on a first run.

- [ ] **Step 4: Confirm a second sync is a no-op in git**

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard
./dashboard sync
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6
git status --short dashboard/data/
git diff --stat dashboard/data/dashboard.json
```

Expected: `git diff` on `dashboard.json` is empty. If it is not, the export is not deterministic — fix that before going further, because a noisy diff makes the committed state unreviewable.

- [ ] **Step 5: Confirm the git ignore rule is right**

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6
git check-ignore -v dashboard/data/dashboard.db
git status --short dashboard/data/
```

Expected: `dashboard.db` is ignored; `dashboard.json`, `pending-changes.md` and `clickup-cache.json` show as untracked or modified.

- [ ] **Step 6: Start the server and confirm it responds locally**

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6/dashboard
./dashboard serve &
sleep 2
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8765/
curl -s http://127.0.0.1:8765/api/state | head -c 300
```

Expected: `200`, then a JSON document beginning with the board's contents.

- [ ] **Step 7: Hand the link to the user — this is a blocking gate**

Tell the user: the dashboard is running at <http://127.0.0.1:8765>, VS Code should have forwarded the port, and ask them to open it and confirm all three sections render, a question closes and reopens, and an agenda item closes into Decisions.

**Do not mark this task complete, and do not claim the dashboard works, until the user says they have seen it in their own browser.** `curl` proves the server answers; it does not prove the page reaches them over SSH, and that is exactly the thing this design has not yet been tested on.

- [ ] **Step 8: Commit the data files and stop**

```bash
cd /home/jek354/research/68f6bef0eaaf5c0928a922c6
git add dashboard/data/dashboard.json dashboard/data/pending-changes.md dashboard/tests/test_end_to_end.py
git commit -m "feat(dashboard): first synced board from HANDOFF.md section 7"
```

**Do not push.** The remote is Overleaf, and pushing changes what Eun-Ah and Tamra see. Tell the user the commits are ready and let them decide.

---

## Self-Review

**Spec coverage.** Every section of the spec maps to a task: *Storage* and *Portability* → Tasks 1–2; *Data model* → Task 1; *Sync semantics* → Task 5; write-back → Task 6; *Server and API* → Task 7; *UI* → Task 8; *Commands* and *Layout* → Task 9; *Testing* and the manual gate → Tasks 3, 5, 10. The *Risks* section's loud-parser requirement is Task 3 Steps 3–6; its deterministic-JSON requirement is Task 2 plus the Task 10 Step 4 git check.

**Type consistency.** `Item` is defined once in `sources/__init__.py` and used unchanged by `handoff.py`, `clickup.py` and `sync.py`. Question ids are strings everywhere, agenda and decision ids are integers everywhere. `store.upsert_flag` is keyword-only (`key=`, `text=`, `first_seen=`, `resolved=`) at its definition and at all three call sites (Task 1 tests, Task 5 sync). `SyncResult`'s six fields are set in `sync` and read in `report.render_pending` and `cli._do_sync` under the same names.

**Known gap, deliberately left.** `update_agenda` accepts `status="dismissed"` from the UI's Dismiss button, and `list_agenda` will return those rows; neither the Agenda nor the Proposed view renders them, so a dismissed item simply disappears. That is the intended behaviour and needs no extra column.
