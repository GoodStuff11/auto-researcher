# Meeting dashboard — design

*2026-09-21. Replaces the regenerate-from-a-prompt React artifact described in
`.claude/skills/project-tracker/references.md/regenerate-dashboard.md`.*

## Why

The current dashboard is a React artifact (`dashboard_v7.jsx`) whose entire state is a hardcoded
`SEED` object. Rebuilding it means pasting a long prompt into a Claude project chat and having a
model retype the whole dataset. Three consequences make it unusable in practice:

- **No storage.** Nothing typed in during a meeting survives. Shared storage was tried and removed
  because writes reported success while staying invisible to other sessions.
- **Bulky to regenerate.** Every rebuild is a full model-authored restatement of every item, which
  is expensive, non-reproducible, and drifts in tone. Generation quality is not stable across runs.
- **A new URL every time.** Each rebuild mints a new artifact UUID, kills the previous link, and
  forces a manual edit of `WHERE-THINGS-LIVE.md` plus an unshare of the old build.

## What this replaces it with

A local, self-contained dashboard that lives in the Overleaf repository alongside `HANDOFF.md`:
a SQLite store, a sync command that reconciles it against `HANDOFF.md` §7 (and optionally
ClickUp), a stdlib HTTP server, and a three-section browser UI whose every interaction writes
straight through to the store.

Regenerating stops being a generation task and becomes a reconciliation task. The agent's job
shrinks to "make the database agree with the tracker" — no prose is re-authored, no link changes,
and meeting work persists.

## Non-goals

- Not a replacement for `HANDOFF.md` §7. Item numbers are still born there and nowhere else.
- Not a record of what was said in a meeting. Circleback keeps doing that.
- Not a multi-user live-editing surface. Single writer — see the *Reach* row under
  *Decisions taken*. Others read the committed JSON, or run their own copy from a clone.

## Decisions taken

| Question | Decision |
|---|---|
| Input source | `HANDOFF.md` §7 always; ClickUp opt-in per run. Adapters are pluggable — the split of responsibility between `HANDOFF.md` and `project_tracker.md` is still under review, and a change there is a new adapter, not a rewrite. |
| Storage | SQLite (gitignored) + a committed JSON export, both in the Overleaf repo beside `HANDOFF.md`. |
| Portability | The **JSON is the store of record**; SQLite is a local cache rebuilt from it. A fresh clone has the database. |
| Code location | Ships inside the Overleaf repo, so one clone gives any group member code and data together. |
| Reach | Localhost only, bound to `127.0.0.1`. Author connects via VS Code Remote-SSH, which auto-forwards the port. |
| Sync conflicts | The tracker dictates open/closed and the descriptive fields. The UI owns resolutions, follow-ups, decisions, agenda, and any hand-set difficulty. |
| Write-back | The UI never edits §7. It records changes to `pending-changes.md` for a person or agent to fold into §7. |
| Agenda origin | Sync proposes candidates; the user curates in the UI. |
| Difficulty | Agent estimates at sync; a user-set value is never overwritten. |

## Layout

Everything lives in `68f6bef0eaaf5c0928a922c6/dashboard/` (the Overleaf clone):

```
dashboard/
  dashboard                  # executable entry point: sync | serve | update | export
  README.md                  # the commands, for a human with no agent
  .gitignore                 # ignores data/dashboard.db only
  project_tracker/
    __init__.py
    store.py                 # schema + the CRUD API agents call
    sources/__init__.py      # Adapter protocol: fetch() -> list[Item]
    sources/handoff.py       # parse HANDOFF.md §7 and §7.1
    sources/clickup.py       # cached ClickUp snapshot -> status/priority/due/task-id
    sync.py                  # reconciliation + pending-changes emission
    export.py                # store <-> dashboard.json (bidirectional)
    serve.py                 # HTTP + JSON API
    static/index.html, app.js, style.css
  data/
    dashboard.db             # gitignored, local cache
    dashboard.json           # committed, store of record
    pending-changes.md       # committed, what the UI owes §7
    clickup-cache.json       # committed, last ClickUp pull
```

In the research repo, `project-tracker/` keeps `project_tracker.md` and `WHERE-THINGS-LIVE.md` and
gains a one-line pointer at the dashboard's real home. No second copy of the code.

## Stack

Python 3 standard library only — `http.server`, `sqlite3`, `json`, `re`, `argparse`. No npm, no
build step, no pip install. The UI is plain HTML/CSS/JS served by the same process, using
`dashboard_v7`'s palette so the board still looks familiar to the group.

The reason for stdlib-only is longevity: this runs on a shared cluster account and gets touched
every few weeks. Anything with a dependency tree will be broken the third time someone needs it.

## Data model

### `questions`

`id` is the §7 item number verbatim (`"14"`, `"14b"`). Never minted locally — §7 is where numbers
are born, and closed items keep theirs.

| Column | Owner | Notes |
|---|---|---|
| `id` | tracker | §7 item number, primary key |
| `title`, `why` | tracker | Overwritten every sync |
| `owner`, `priority`, `raised` | tracker | Overwritten every sync |
| `clickup_id`, `clickup_status`, `due` | tracker | Only refreshed on a `--clickup` run |
| `state` | tracker | `open` \| `closed` |
| `state_source` | — | `tracker` \| `ui`; drives the write-back report |
| `difficulty` | either | `S` \| `M` \| `L` |
| `difficulty_source` | — | `agent` \| `user`; agent estimates skip `user` rows |
| `closed_on`, `closed_note` | ui | |
| `last_seen_sync`, `updated_at` | — | `last_seen_sync` detects items dropped from §7 |

### `agenda`

Action items to be **discussed and resolved during the meeting** — not work owed afterwards.

`id` (local autoincrement), `title`, `detail`, `origin` (`sync:flag` \| `sync:item:<n>` \|
`manual`), `proposed_by`, `created`, `sort_order`, `status` (`proposed` \| `accepted` \| `closed`),
`resolution` (free text: how it was resolved), `followups` (JSON list of people, optional),
`resolved_on`, `resolved_by`.

### `decisions`

`id`, `text`, `grounds` (**required** — the group convention is that the grounds field is not
optional), `decided_on`, `decided_by`, `followups`, `from_agenda_id`, `source` (`ui` \| `tracker`).

Closing an agenda item creates exactly one decision row, carrying its resolution as `grounds` and
its follow-ups forward.

### `flags`

`id`, `text`, `first_seen`, `resolved`. Populated from §7.1. Disagreements are listed, never
auto-resolved — this mirrors the existing convention and is deliberate.

### `changelog`

`id`, `ts`, `entity`, `entity_id`, `field`, `old`, `new`, `actor`. Append-only. This is the sole
input to `pending-changes.md`.

### `meta`

Key/value: `last_sync`, `last_clickup_pull`, `meetings` (the list driving the staleness counter),
`workflow_ran`.

## Sync semantics

`sync` parses §7, optionally refreshes from ClickUp, and reconciles:

1. **Upsert by item number.** Tracker-owned fields are overwritten unconditionally.
2. **State.** §7's open/closed wins, always.
3. **The collision that matters.** An item closed in the UI that §7 still lists as open is
   reverted to open *and* written to `pending-changes.md` as an owed §7 edit — with its closing
   note — so in-meeting work is handed back rather than silently dropped.
4. **Difficulty.** Estimated from the §7 text (a rerun vs. a new implementation vs. a writing task)
   only where `difficulty_source != 'user'`.
5. **Disappearances.** An item in the store but absent from §7 is flagged, never deleted.
6. **Agenda.** New `proposed` candidates are inserted from §7.1 flags, high-priority open items,
   and anything due before the next meeting — keyed by `origin` so they are not duplicated. Rows
   already `accepted` or `closed` are never touched.
7. **Decisions.** Never overwritten by sync.
8. **Export.** `dashboard.json` is rewritten at the end of every sync.

ClickUp is opt-in (`--clickup`) because the pull is expensive. Without the flag the cached
snapshot is reused and the run costs nothing.

## Server and API

`serve.py` runs a `ThreadingHTTPServer` on `127.0.0.1:8765` (`--port`, `--host` available). On
startup, if `dashboard.db` is missing or older than `dashboard.json`, it rebuilds the DB from the
JSON — which is what makes a fresh clone usable.

- `GET /` — the UI
- `GET /api/state` — the full board as one JSON document
- `POST /api/questions/<id>/state` — open/close, with an optional note
- `POST /api/questions/<id>/difficulty`
- `POST /api/agenda`, `PATCH /api/agenda/<id>`, `POST /api/agenda/<id>/close`, `POST /api/agenda/reorder`
- `PATCH /api/decisions/<id>`

Every mutation runs in a single SQLite transaction that also appends to `changelog` and rewrites
`dashboard.json`. There is no save button and no unsaved state.

## UI

Three sections, always present.

**Open questions.** Two groups, open and closed. Open questions sort by priority, then difficulty,
and carry the staleness heat counter from v7 (how many meetings an item has survived). Each shows
item number, title, why it matters, owner, priority, difficulty, and ClickUp status where known.
One control closes an open question; one reopens a closed one.

**Meeting agenda.** Accepted items in order, each with a resolution textbox and an optional
follow-up-people field. Proposed items from sync sit in a separate tray to be accepted or
dismissed. A Close button on an item requires a resolution, then moves it out of the agenda.

**Decisions.** Grouped and sorted by date resolved, newest first. Each shows its text, grounds,
who decided, and any follow-ups.

## Commands

For the agent, one line:

```
./dashboard update            # sync from HANDOFF.md, reconcile, export, serve, print the URL
./dashboard update --clickup  # same, paying for a fresh ClickUp pull
```

For a human with no agent involved, documented at the top of `README.md`:

```
./dashboard serve             # serve the current database, no sync
```

Under VS Code Remote-SSH the port is forwarded automatically and the printed `http://127.0.0.1:8765`
is clickable. The server also prints the equivalent `ssh -L` line for anyone on a plain terminal.

## Testing

Test-driven, `pytest`, matching the existing `auto-researcher/tests` conventions.

- **Parser** against the real §7 as of 2026-09-21: items across the Jonathon/Tamra/Eun-Ah/Closed
  sections, sub-items like `14b`, and the §7.1 flag block.
- **Reconciliation**, case by case: tracker-wins on state, UI-wins on resolution, the
  revert-and-report collision, stale-item flagging, agenda de-duplication by `origin`.
- **Store** CRUD and transaction atomicity.
- **Export** round-trip: `store -> json -> store` is lossless, and the JSON is byte-deterministic so
  git diffs are meaningful.
- **API** round-trip through a live server on an ephemeral port.
- **Manual gate**: the server is started, curled locally to confirm it responds, and the forwarded
  link confirmed in the author's own browser before the work is called done.

## Risks

- **The Overleaf file tree grows a `dashboard/` folder.** Accepted deliberately: the point is that
  a clone carries both the code and the data.
- **The §7 parser is coupled to prose formatting.** §7 is hand-written Markdown with bold item
  headers. The parser must fail loudly on an unparseable item rather than silently dropping it, and
  the item count is asserted after every parse.
- **Two people syncing independently will produce conflicting `dashboard.json` commits.** Single
  writer is the assumption; the JSON is line-oriented so a conflict is at least resolvable by hand.
