# Project hub — design
*2026-11-12. Replaces the SQLite meeting dashboard (`docs/superpowers/specs/2026-09-21-meeting-dashboard-design.md`) and the `project-tracker` skill. `career-update` is unaffected.*

## Why
Research and meeting preparation currently run through one long prose file per project (`HANDOFF.md`, 2,077 lines for the UCC paper) plus a dashboard regenerated from it. That file mixes four kinds of content with different update rates — a snapshot, a changelog, stable project knowledge, and an item ledger — so it goes stale in parts (three disagreeing "as of" dates), every read costs the whole history, item state lives in prose ("CLOSED … everything below is superseded"), and questions, decisions and actions are not distinguished, so the chain *question → decision → action → outcome* cannot be followed. Catch-up passes after quiet weeks are expensive because nothing records what has already been ingested.

The hub splits this into a shared, structured, mergeable store that agents maintain through a CLI, and a per-meeting Excel agenda generated from it and read back after the meeting.

## Decisions taken
| Question | Decision |
|---|---|
| Where the one-stop-shop lives | Obsidian. **One shared vault per project**, its own git repo (`<project>-hub`), shared with every project member. The owner's private vault stays private. |
| Old systems | Replaced: SQLite dashboard (left in place, unmaintained) and `project-tracker` (retired / redirected). |
| Who runs the agents | Anyone in the project, from their own clone. Multi-writer from day one. |
| Agendas | One per **meeting series** (e.g. full group vs. a working meeting), defined in `project.yaml`. |
| Agent autonomy | Safe changes apply directly; judgement calls are staged in `Status/Pending/` and applied after one yes/no from whoever ran the agent. |
| Store representation | Markdown files with YAML frontmatter are the store of record. No database. |
| Agenda format | `.xlsx`, built and read back by a script; the agent writes content only. Target viewers: Excel, LibreOffice, VS Code *Office Viewer* (`cweijan.vscode-office`). |

## Non-goals
- Not a record of what was said in meetings — Circleback keeps that; the hub cites it.
- Not a task tracker UI — ClickUp is where humans edit tasks.
- Not a place for personal assessments, career or 1:1 material.
- No live multi-user editing of a single file; concurrency is handled by file granularity and git.

## Vault layout
```
<project>-hub/
  project.yaml                       # sources, people, meeting series, named path roots
  README.md                          # collaborator setup: Obsidian ≥1.9, uv, connectors
  .obsidian/                         # committed: obsidian-git auto pull/push, Bases
  .claude/skills/                    # hub-init, hub-vision, hub-status, hub-agenda, hub-figures
  tools/hub/                         # Python package, `uv run hub <cmd>`
  Vision.md
  Locator.md
  Figures/
    Gallery.md                       # generated
    <fig-id>/figure.png              # copied from the source repo
    <fig-id>/caption.md              # human-owned, never overwritten
    <fig-id>/meta.yaml               # source script, data hash, status, what it licenses
  Status/
    Items/<ID>.md                    # one file per question / decision / action
    Meetings/<series>/<date>.md
    Log/<YYYY-MM>.md                 # append-only journal
    Knowledge/                       # stable findings, what the code does, rejected claims
    Snapshot.md                      # generated, ≤ 1 page
    Pending/<P-id>.md                # staged judgement calls, one file per proposal
    sync-state/                      # one small file per key, so clones never touch the same file
      watermark.<source>.yaml        #   `value: <watermark>`, `updated: <ISO UTC>` per ingest source
      next_meeting.<series>.yaml     #   `value: <YYYY-MM-DD>`, `updated: <ISO UTC>` per series
      flags.md                       #   one `- <flag>` line per unresolved sync flag (merge=union)
    Items.base                       # Bases views
  Agenda/
    <series>.xlsx
    <series>.draft.yaml              # agent-written content the xlsx is rendered from
    archive/<series>/<date>.xlsx
```

### Ownership
| File | Written by |
|---|---|
| `Items/`, `Log/`, `Meetings/` | Agent via the CLI (safe changes) |
| `Pending/` | Agent proposes; human approves via `hub pending apply` |
| `Snapshot.md`, `Gallery.md`, `*.xlsx` | Generated only; never hand-merged — regenerate after pull |
| `caption.md`, `Vision.md` prose, `Locator.md` | Humans (agent assists on request) |
| Task status / assignee / due date | ClickUp; mirrored into action items |
| Outcome / "as decided" cells in the live xlsx | Humans during the meeting |

## `project.yaml`
```yaml
name: fermion-unitary
people:
  jonathon: {name: <full name>, clickup: <user-id>, role: student}
  tamra:    {name: <full name>, clickup: <user-id>, role: student}
  eunah:    {name: <full name>, clickup: <user-id>, role: PI}
roots:                       # named roots; Locator and meta.yaml use $NAME/…, each user may override in hub.local.yaml (gitignored)
  ML_REPO: /home/jek354/research/ML-signproblem
  PAPER: /home/jek354/research/68f6bef0eaaf5c0928a922c6
sources:
  clickup: {list_id: "901307841677"}
  circleback: {folder: "Fermion Unitary", terms: ["UCC", "HVA"]}
  repos:
    - {root: ML_REPO, branches: [main], remote: https://github.com/…}
  overleaf: {root: PAPER}
  calendar: {}
  notes: []                  # opt-in per person, in hub.local.yaml
  gmail: {enabled: false, terms: []}
  slack: {enabled: false, channels: []}
series:
  group: {attendees: [jonathon, tamra, eunah], calendar_match: "Fermion Unitary"}
```
`hub.local.yaml` (gitignored) holds per-user overrides: root paths on this machine, opt-in private note folders, the user's own people key (`me`).

## Status store
### Item IDs
`<T>-<YYYYMMDD>-<rand3>` with `T ∈ {Q, D, A}`, e.g. `Q-20261112-a3f`; `rand3` is three lowercase base-36 characters. Collision-safe across writers without coordination; `hub new` retries on the (local) collision it can see.

### Item file
```markdown
---
id: Q-20261112-a3f
type: question
title: Does the HVA beat pruned UCC at matched fidelity?
status: open
owner: jonathon
series: [group]
raised: 2026-11-12
closed:
priority: 2
links:
  answers: []
  spawned_by: []
  pivoted_to: []
clickup_id:
sources: [circleback:123456]
---
One-paragraph description: what is asked and why it matters to the vision.
## Entries
- 2026-11-12 · jonathon · circleback:123456 — Raised at group meeting; …
```
Status values by type:
- question: `open | answered | dropped | pivoted`
- decision: `proposed | made | rejected | deferred`; extra fields `rationale:`, `meeting:` (series/date)
- action: `open | done | dropped`; extra field `due:`

Links:
- `answers` (decision → questions it resolves)
- `spawned_by` (action → decision; question → the item that raised it)
- `pivoted_to` (question → replacement question)

The CLI writes the reverse direction implicitly by query, not by storing back-links, so a link is edited in exactly one file.

Rules (enforced by `hub validate`):
- every entry has date · author · source;
- a closing status (`answered | dropped | pivoted | done | rejected`) requires a closing entry stating why and `closed:` set;
- `pivoted` requires `pivoted_to`;
- a `made` decision must have ≥1 `answers` link and ≥1 action spawned from it (or an entry saying why none);
- open actions must have an owner who is in `project.yaml`;
- all link targets exist.

### Meetings, Log, Knowledge, Snapshot
- `Meetings/<series>/<date>.md`: attendees, Circleback ID, decisions made (IDs), items opened/closed, unplanned decisions. Written by `agenda read` + `hub-status`.
- `Log/<YYYY-MM>.md`: append-only bullets `date · author · source — text` for events not tied to one item (runs finished, drafts pushed, ingest summaries). Appends from different writers merge via the `union` merge driver.
- `Knowledge/`: one note per topic (what the code does, what each figure licenses, rejected claims / "do not re-add", conventions). Edited only through `Pending/`.
- `Snapshot.md`: regenerated by `hub snapshot` — where the project stands, last and next meeting per series, open items by type and owner, flags (sync conflicts), stale figures. Never hand-edited.

### Pending changes
`Status/Pending/` holds one file per proposal, `<P-id>.md` (P-ids are random, not numbered), each a structured block (ops, reason, source, proposer). `hub pending apply P-k3x9 P-m2q7` / `reject P-k3x9`. Judgement calls that must go through it:
- closing a question;
- recording a decision as made from any source other than the agenda read-back (e.g. a Circleback transcript or a Slack thread);
- a pivot;
- any `Vision.md` or `Knowledge/` change;
- new actions parsed from free text;
- items seeded during init or import;
- actions created directly in ClickUp.

Safe changes applied directly:
- log entries and item entries with a source;
- new items with a source;
- mirroring ClickUp status/assignee/due;
- regenerated files.

### CLI (`uv run hub …`)
| Command | Purpose |
|---|---|
| `new {q,d,a} TITLE [--owner --series --priority --link REL:ID --source S]` | Create item, print ID |
| `entry ID TEXT --source S` | Append dated entry (author from `hub.local.yaml`) |
| `set ID KEY=VALUE…` · `link ID REL ID2` | Validated frontmatter edits |
| `list [--type --status --owner --series --since] [--json]` | Queries; `--owner me` for todos |
| `show ID [--chain]` | Item, optionally its whole Q→D→A tree |
| `since DATE [--series]` | Everything changed since DATE |
| `validate` | Rules above; non-zero exit on violation |
| `snapshot` | Regenerate `Snapshot.md` |
| `pending {add,list,apply,reject}` | Staged changes |
| `agenda scaffold SERIES --date D [--force]` | Write `Agenda/<series>.draft.yaml` pre-filled from the store |
| `agenda {build,read} SERIES [--force]` | Workbook (below) |
| `figures {sync,add,gallery}` | Figure registry |
| `check-links` | Resolve every `$ROOT/…` path and URL in `Locator.md` and `meta.yaml` |
| `sync-state {get,set} SOURCE` | Watermarks |
| `import-handoff PATH` | One-off migration, stages everything in `Status/Pending/` |
| `commit MSG` | commit → pull --rebase → regenerate generated files → push, retry ×3 on rejection; stops and reports on a real conflict |

Agents read only through `list`/`show`/`since`/`snapshot` and write only through the CLI.

## Agenda workbook
`hub agenda build SERIES` renders `Agenda/<series>.draft.yaml` (written by the agent) into `Agenda/<series>.xlsx` with fixed styling: frozen headers, wrapped text, set widths, status fills, print setup, hyperlinks, a hidden `id` column on every sheet. A meeting-date cell and a `read_back: false` marker live in a hidden `_meta` sheet.

**Sheet 1 — Since last meeting**
- A summary block of 3–5 sentences, framed against last meeting's decisions.
- One row per decision made at the previous meeting of this series, with columns:
  - decision
  - spawned actions · owners
  - follow-up done (links)
  - outcome: `closed question` / `new question` / `pivot` / `in progress`
  - affected item(s)

**Sheet 2 — Open questions**
- Columns: title, description, owner, age (days), priority, proposed decision (link into Sheet 3).
- Sorted: questions with a proposed decision first, then by priority, then age.

**Sheet 3 — Decisions for this meeting**
- Columns:
  - what to decide
  - proposed action
  - reason
  - who weighs in
  - time budget (min)
  - figure / pre-read link
  - **Outcome**: dropdown `Approve / Modify / Defer / Reject`
  - **As decided**: free text
  - **New actions · owners**: free text
- Capped at 5–7 rows, followed by 5 blank "Raised in meeting" rows.

**Lifecycle**
1. **Prep:** `build` refuses if the live xlsx's `_meta.read_back` is false for a past meeting date, or if it already holds meeting input (`--force` overrides).
2. **Meeting:** edit in any viewer.
3. **Read-back:** `read` matches rows by hidden ID, falling back to the title. It accepts Outcome as dropdown or typed text, matched as exact case-insensitive tokens after trimming trailing `.`/`!` (`approve`/`approved`/`a`/`yes`/`y`/`✓`/`✔`, `modify`/`modified`/`m`, `defer`/`deferred`/`d`, `reject`/`rejected`/`r`/`no`/`n`/`x`/`✗`/`✘`); any other non-empty text is staged for review, never guessed.
   - Approve/Modify set the decision to `made`, copy "as decided" into the decision, and append entries. These are safe changes, because they record what the meeting itself decided.
   - Defer/Reject set the status and append an entry.
   - Free-text actions and "Raised in meeting" rows go to `Status/Pending/`.
   - The file is then copied to `archive/`, re-styled, and stamped `read_back: true`.
4. **Circleback cross-check:** decisions or actions in the transcript but absent from the sheet are staged in `Status/Pending/`.

**Spike first:** before building the rest, confirm what Office Viewer preserves when it saves a file `build` produced (data validation, hidden column, hidden sheet, styles). If it drops the hidden `_meta` sheet, move meeting metadata to a visible header row on Sheet 1.

## Skills
All live in the hub's `.claude/skills/` and are thin procedures over `hub` commands.
- **`hub-init`** — the full initialization pipeline:
  1. create the repo from the skeleton (`.obsidian`, skills, tools, `uv`);
  2. interview for `project.yaml`;
  3. run `hub-vision`;
  4. seed the status store from existing material (a HANDOFF-style file via `import-handoff`, ClickUp tasks, recent Circleback meetings), all into `Status/Pending/`;
  5. register figures, build `Locator.md`;
  6. build the first agenda per series (Sheet 1 says "first meeting");
  7. `hub commit`, then print collaborator setup.
- **`hub-vision`** — Heilmeier-style interview covering:
  - the vision, and how we approach it;
  - the community's problem statement, and how we answer it;
  - our unfair advantage over competitors;
  - the difference we'll make in the landscape;
  - risks.

  Advantage and novelty claims are checked with `research-question`. Each revision appends a dated entry to a *Revisions* section, linked to the decision that caused it. Any decision with `pivot` outcome makes `hub-status` propose a vision check.
- **`hub-status`** — three modes:
  - *update* — ingest each source since its watermark (table below), apply the safe changes, stage the rest, then `validate`, `snapshot`, `commit`. It ends with a source report: reached / not reachable / disabled, so a partial run is never mistaken for a complete one.
  - *todos* — `hub list --type action --owner me --status open`, sorted by due date then priority.
  - *post-meeting* — `agenda read`, then the Circleback cross-check, then *update*.
- **`hub-agenda`** — for a series:
  1. check the last meeting was read back;
  2. run *update* if any watermark is older than the last meeting;
  3. choose questions and proposed decisions, capped and ranked by priority, age and whether they block other items;
  4. write `draft.yaml`, `build`;
  5. report what was chosen and why, so the user can prune before the meeting.
- **`hub-figures`** — `figures sync`: copy each figure from its source (`meta.yaml` path under a root), recompute the data hash, flag stale figures, regenerate `Gallery.md` (embed + `caption.md` transclusion). `figures add` registers a new figure. It never writes `caption.md` after creation.

### Ingest sources
| Source | Pulled | Watermark | Lands as |
|---|---|---|---|
| ClickUp | status/assignee/due changes; new tasks in the list | last sync time | mirrored fields (safe); new tasks → Pending |
| Circleback | meetings in the project folder: decisions, action items, transcript as needed | last meeting ID | entries + Meetings note; decisions/actions → Pending |
| Repos | commits on configured branches, grouped by topic | last SHA per repo | Log entries; item entries where a commit clearly serves an item |
| Overleaf | co-author commits to the paper | last SHA | Log entries |
| Calendar | next meeting per series | — | Snapshot |
| Private notes | user's opted-in folders, summarized, never copied verbatim | last mtime | entries, attributed to that user |
| Gmail / Slack | threads matching project terms (opt-in) | last thread date | entries |

Every ingested item carries its source ID; re-ingesting a source ID already present is a no-op, so overlapping runs by different people are idempotent.

### ClickUp mapping
Only actions sync. Creating an action with an owner creates the task (description links back to the item path) and stores `clickup_id`. ClickUp owns status, assignee and due date; the hub mirrors them. Hub-side description changes are not pushed. Disagreements a mirror cannot settle become flags in `Snapshot.md`, including:
- a task deleted in ClickUp;
- an action closed in the hub but open in ClickUp;
- an assignee who is not in `project.yaml`.

Questions and decisions never become tasks.

## Privacy and sharing
Everything in the hub is readable by every project member. Skills instruct the agent to record decisions, stated positions and actions with their source, and never personal assessments of people, career or 1:1 material, or verbatim content from a private vault. `hub.local.yaml` and anything under a user's private roots stay out of the repo.

## Concurrency
Writers do not lock. Mergeability comes from:
- one file per item;
- append-only logs, merged with the `union` merge driver;
- one file per pending proposal, and one file per sync-state key;
- item files merged by the `hub-item` git merge driver (`python -m hub.merge_item %O %A %B`, registered by `hub commit` in `.gitattributes` and `.git/config`): entries are unioned (de-duplicated, sorted by date), list fields (`sources`, `series`, `links.*`) are unioned, and every other field and the description merge three-way — one side changed it, that side wins;
- sync-state key files (watermarks, next meeting) merged by the `hub-watermark` driver (`python -m hub.merge_watermark %O %A %B`): the side with the newer `updated` stamp wins, so two same-day ingests of one source never conflict;
- items created by a pending proposal get ids seeded from the proposal id and op index, so two people applying the same proposal create the same item;
- generated files rebuilt after every pull, never merged;
- source-ID idempotency.

`hub commit` rebases and retries; a genuine conflict — both sides changed the same field of one item to different values — is reported to the user running the agent, naming the field and both values, and not auto-resolved. The live `.xlsx` is binary: if two people edit it during a meeting, the later push loses. The skill tells the meeting to have one editor, and `read` refuses a file whose `_meta` meeting date doesn't match the expected one.

## Testing
- `tools/hub` unit tests (pytest):
  - ID generation;
  - frontmatter round-trip preserves body and unknown keys;
  - every `validate` rule, positive and negative;
  - `since` and `list` filters;
  - `pending` apply/reject;
  - `agenda build` → `read` round-trip on a fixture, including a typed-text Outcome and a dropped ID column;
  - `import-handoff` on a trimmed copy of the UCC `HANDOFF.md`.
- A two-clone test: two working copies each create items and log entries, push in either order, and both end clean with `validate` passing.
- Manual: the Office Viewer spike; open the fixture vault in Obsidian and check `Items.base` views and `Gallery.md` embeds.
- First real use: migrate the UCC paper (`HANDOFF.md`, ClickUp list `901307841677`, its Circleback meetings) into a `fermion-unitary-hub`, and prepare one group-meeting agenda from it.

## Build order
1. Office Viewer spike (throwaway).
2. `tools/hub` core: item model, CLI read/write, `validate`, `snapshot`, `pending`, `commit`.
3. Agenda build/read.
4. Skills: `hub-status` (update, todos, post-meeting) with the ClickUp and Circleback ingesters first, then repos and Overleaf, then the opt-in sources.
5. `hub-agenda`.
6. `hub-figures`, `Locator`, `check-links`.
7. `hub-vision`.
8. `hub-init` and the vault skeleton; `import-handoff`; the UCC migration as acceptance test.
9. Retire `project-tracker` (redirect note) and mark the SQLite dashboard unmaintained in `WHERE-THINGS-LIVE.md`.
