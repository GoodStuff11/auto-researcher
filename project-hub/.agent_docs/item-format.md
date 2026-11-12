# Item files (`Status/Items/<ID>.md`)
Code: parse/render in `hub/items.py` (`parse_item`, `render_item`, `Entry`, `Item`, `default_meta`); storage and queries in `hub/store.py` (`Store`); rules in `hub/validate.py` (`validate`); ids in `hub/ids.py` (`new_id`). One file per question, decision or action. Questions and decisions are never tasks.
## Example (a made decision, as written by the tool)
```
---
id: D-20261112-igo
type: decision
title: Ship it
status: made
owner: alice
series:
- group
raised: 2026-11-12
closed:
priority: 2
links:
  answers:
  - Q-20261112-m8h
  spawned_by: []
  pivoted_to: []
sources:
- manual
- agenda:group/2026-11-19
rationale: ''
meeting: group/2026-11-19
as_decided: ''
---
<description paragraph, may be empty>
## Entries
- 2026-11-12 · alice · manual — Raised.
- 2026-11-19 · alice · agenda:group/2026-11-19 — Approved at group/2026-11-19: as proposed
```
File structure: line 1 `---`; YAML mapping; a line `---`; free-text description (everything up to the line `## Entries`); the line `## Entries`; one entry line per event. A file without `## Entries` parses with no entries. YAML is dumped with `sort_keys=False`, no line wrapping, empty value for null. A file that does not follow this raises `ItemFormatError` (missing/unclosed frontmatter, invalid YAML, non-mapping frontmatter, missing `id`/`type`/`title`/`status`, malformed entry line or entry date).
## Frontmatter keys
Common to all types (written by `default_meta`, `hub/items.py:86`):
| Key | Type | Set by | Meaning / allowed values |
|---|---|---|---|
| `id` | string | `Store.create` | Immutable id, regex `^[QDA]-\d{8}-[a-z0-9]{3}$`. Not settable. |
| `type` | string | `Store.create` | `question`, `decision` or `action`. Immutable. |
| `title` | string | create / `set` | One line, non-empty (empty or whitespace raises `StoreError`). |
| `status` | string | create (first value of the type's list) / `set` | question: `open`, `answered`, `dropped`, `pivoted`. decision: `proposed`, `made`, `rejected`, `deferred`. action: `open`, `done`, `dropped`. `set` rejects values invalid for the type. |
| `owner` | string or null | create (`--owner`, default `me`) / `set` | A key of `project.yaml` `people`. May be cleared; `validate` flags an unknown owner and an open action without one. |
| `series` | list of strings | create / `set` | Meeting series the item may appear on; empty list means every series. Must be a list. |
| `raised` | date | create | Day the item was created (`HUB_TODAY`/system date). |
| `closed` | date or null | `set` | Day of closure. Required (`validate`) when status is `answered`, `dropped`, `pivoted`, `done` or `rejected`. |
| `priority` | int or null | create / `set` | 1 highest, 2 default, 3 lowest. `set` converts digits to int. |
| `links` | mapping | `hub link` / create `--link` | Keys `answers`, `spawned_by`, `pivoted_to`, each a list of item ids. Not settable through `set`. |
| `sources` | list of strings | create, `entry` | Every source id that has touched the item; `Store.entry` appends the entry's source if new. Must be a list. |
Decision only (not required to be filled): `rationale` (string, why), `meeting` (string `<series>/<YYYY-MM-DD>`, set by the read-back when approved/modified), `as_decided` (string, wording of what was decided), `no_action` (string, reason a made decision spawns no action; absent unless set), `proposed_action` (string, read by the agenda draft; absent unless set).
Action only: `clickup_id` (string or null, ClickUp task id), `due` (date or null).
`set` coercion (`coerce_value`, `hub/store.py:23`): empty string becomes null; `raised`, `closed`, `due` parse `YYYY-MM-DD`; `priority` parses an int; a bad value raises `StoreError`. `id`, `type`, `links` are protected.
## Links
Stored one way only, on the item that points; reverse lookups use `Store.children(id)` (`hub/store.py`), which returns each descendant once and skips missing pivot targets (raises `UnknownItem` for an unknown id). `hub show ID --chain` prints the tree.
- `answers`: on a decision, the question(s) it answers.
- `spawned_by`: on an action or question, the decision or item it came out of.
- `pivoted_to`: on a question, the question that replaces it (children() also returns those targets).
Link targets must exist and match the id regex (`StoreError` otherwise).
## Entries (history)
Line format, regex `ENTRY_RE` (`hub/items.py:20`): `^- (\d{4}-\d{2}-\d{2}) · (\S+) · (\S+) — (.*)$`, that is `- <date> · <author> · <source> — <text>`. Author is a people key; both author and source must be non-empty with no whitespace (`Entry.render` raises `ValueError`). Text is collapsed to one line. The first entry of every item is `Raised.` (the creation entry). Appended by `Store.entry` / `hub entry`.
Source id conventions: `circleback:<meeting id>`, `commit:<root>@<short sha>`, `clickup:<task id>`, `agenda:<series>/<date>`, `pending:<P-id>`, `manual`; the default source of ops in proposals is `pending`.
## IDs (`hub/ids.py`)
Items: `<T>-<YYYYMMDD>-<rand3>` with T = `Q`, `D`, `A`, the creation day, and 3 characters of `a-z0-9`; retried until unused (100 attempts). Random, so independent clones never collide; items created by a pending proposal use a generator seeded by `<P-id>:<op index>` so every clone applying it gets the same id (pending-format.md). Proposals: `P-<rand4>`. A missing or malformed id passed to `Store.get/path` raises `UnknownItem` (a `StoreError` and a `KeyError`), which the CLI prints as `error: no such item <id>` (exit 1).
## Validation rules (`validate`, `hub/validate.py`; `hub validate` prints `<id>  <rule>: <message>` per violation then `<n> violation(s)`, exit 1 if any)
| Rule | Condition that raises it |
|---|---|
| `format` | the file `<id>.md` could not be parsed (item id = file stem); the item is skipped by every reader, the other items are still checked |
| `type` | `type` not question/decision/action (other rules skipped for that item) |
| `status` | status not valid for the type |
| `id-prefix` | id does not start with Q-/D-/A- matching the type |
| `link-target` | an id in any link list does not exist as an item |
| `entry-author` | any entry author is not a `people` key (reported once per item) |
| `unknown-owner` | `owner` set but not a `people` key |
| `closed-date` | status in `answered`, `dropped`, `pivoted`, `done`, `rejected` and `closed` is not a date |
| `closing-entry` | closing status with a valid `closed` date but no entry among `entries[1:]` (the creation entry is ignored) dated on/after `closed` |
| `pivot-target` | status `pivoted` and `links.pivoted_to` empty |
| `decision-answers` | decision `made` with empty `links.answers` |
| `decision-actions` | decision `made`, no item has it in `spawned_by`, and `no_action` is empty |
| `action-owner` | action `open` and owner null or not a `people` key |
Practical consequences: when closing, add a closing entry (`hub entry`) on or after the `closed` date; when making a decision, link it to its question and either create an action spawned by it or set `no_action=<reason>`.
## CLI errors
`StoreError` (bad field/status/link/type), `ItemFormatError`, `UnknownItem` (unknown id) all print `error: ...` and exit 1 (see git-sync.md for exit codes).
## Robustness
- `parse_item` normalises CRLF line endings to LF before parsing; `raised`, `closed` and `due` written as datetimes (e.g. `2026-11-12 10:00:00`) are read as dates.
- `Store.all()` skips files that fail to parse and records them in `store.load_errors` (list of `(path, error)`, reset on each call): one malformed file never blocks `list`, `since`, `validate`, the Snapshot or `hub commit`. `hub validate` reports each as rule `format` with the file stem as item id; `list`/`since` print `warning: cannot load <path>: <error>` to stderr; the Snapshot lists them under `## Flags`.
- `Store.get`/`Store.path` raise `UnknownItem` (a subclass of both `StoreError` and `KeyError`) for a missing id or one not of the form `T-YYYYMMDD-xxx`; the CLI prints `error: no such item <id>`. A file that exists but does not parse raises `ItemFormatError` from `get`.
- Item files are written atomically (temp file + `os.replace`).
- Concurrent edits of one item from two clones are merged by the `hub-item` git driver (git-sync.md).
