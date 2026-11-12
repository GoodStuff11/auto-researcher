# Ingest: reading sources into the vault
Code: `hub/sources.py`, `hub/ingest_git.py`, `hub/batch.py`, `hub/clickup.py`, CLI wiring in `hub/cli_ext.py`. The agent (skill) reads a source with its connector, then writes through these commands; the CLI never talks to Circleback, ClickUp, Gmail or Slack.
## Source ids and watermarks
Every derived record carries a source id (no whitespace). `hub seen SOURCE` says whether it was recorded already.
| source id | meaning |
|---|---|
| `circleback:<meeting id>` | a Circleback meeting |
| `clickup:<task id>` | a ClickUp task |
| `commit:<ROOT>@<7-char sha>` | a commit of the repository `ROOT` (a key of `roots`) |
| `agenda:<series>/<YYYY-MM-DD>` | agenda read-back of a meeting |
| `notes:<people key>@<YYYY-MM-DD>` | a private note of that person on that day |
| `gmail:<thread id>` | a Gmail thread (opt-in source) |
| `slack:<channel id>-<ts>` | a Slack message (opt-in source) |
| `manual` | typed by a person |
Watermarks live in `Status/sync-state/watermark.<name>.yaml` (vault-layout.md; name matches `[A-Za-z0-9_-][A-Za-z0-9_.-]*`, `hub/sync_state.py:18`). Names used: `clickup`, `circleback`, `git.<ROOT>` (full sha of the newest ingested commit; also for the Overleaf root), `gmail`, `slack`, `notes.<people key>`. Set by `hub sync-state set` or the `watermark` key of a batch.
## hub seen SOURCE
`source_seen` (`hub/sources.py:21`). Prints `new` when SOURCE appears nowhere. Otherwise `seen` plus space-separated places: an item id (SOURCE is in its `sources` frontmatter list), `log:<YYYY-MM>` (a monthly log line has ` · SOURCE — `), a pending id `P-xxxx` (the proposal's `source` or one of its ops' `source`). Example: `seen S-20261101-ab12 log:2026-11`. Exit 0.
An `unreadable:<path>` entry (item or proposal file that failed to load) means the answer may be incomplete; fix that file. A monthly log that cannot be decoded is an error (exit 1) rather than a silent `new`.
## hub ingest git ROOT [--since YYYY-MM-DD] [--advance] [--limit N]
`ingest_git` (`hub/ingest_git.py:102`).
- Configuration: ROOT is a key of `roots` (project.yaml, local path overridden in hub.local.yaml); unknown root or a path without `.git` is an error (exit 1). Branch: `branch` of the `sources.repos[]` entry (or `sources.overleaf`) with that `root`; default `HEAD`; a missing branch is an error.
- Listing: non-merge commits after the watermark `git.<ROOT>`, oldest first. Without a watermark (or with `--since`, which ignores the watermark) the commits since `--since`, default the last 30 days. A watermark sha that does not exist, or is not an ancestor of the branch, gives a warning and the date fallback (`hub/ingest_git.py:58`).
- `--limit N` (default 500): at most the N oldest unlisted commits; the warning says how many newer ones remain ("run again after --advance"), so `--advance` never skips a commit.
- Output (stdout, JSON): `{root, path, branch, watermark_before, commits: [{sha, short, date, author, subject, files}], warning, advanced}`. A non-null `warning` is also printed on stderr as `warning: ...`.
- `--advance` moves `git.<ROOT>` to the newest listed commit (`advanced: true`, only if any were listed). Use it only after the commits are recorded.
## hub ingest batch FILE
`run_batch` (`hub/batch.py:219`), validation `check_batch` (`hub/batch.py:158`). YAML mapping with these top-level keys only (`watermark`, `safe`, `proposals`, `flags`); each op's keys must be in its allow-list, an unknown key is an error.
```yaml
watermark: {name: circleback, value: "2026-11-19"}   # optional; name as above; value str, int or date; set last
safe:                                                # applied directly
  - {op: entry, id: <item id>, text: <str>, source: <id>}
  - {op: log, text: <str>, source: <id>}
  - {op: new, type: question|action, title: <str>, source: <id>,
     owner: <people key>, series: [<str>], priority: 1-3, description: <str>,
     links: {spawned_by: [<existing item id>]}, fields: {due: YYYY-MM-DD}}
proposals:                                           # staged as Status/Pending/<P-id>.md
  - {title: <str>, reason: <why a human decides>, source: <id>, ops: [<pending ops, pending-format.md>]}
flags: [<str>, ...]                                  # added to Status/sync-state/flags.md
```
Rules (`hub/batch.py:83-203`):
- `source` is required on every op and proposal, non-empty with no whitespace. `text`, `title`, `reason` and flags are non-empty strings; `priority` is an integer 1-3; `series` a list of strings; `description` a string.
- `entry.id` must exist. Safe `new` creates only questions and actions (decisions are proposals); `owner` must be a key of `people` (else error `owner X is not a known person`); `links` may only be `spawned_by` to existing ids; `fields` may only set `due`. Closures, answers, pivots and decisions go through `proposals`.
- Proposal ops are checked with the pending-format.md rules (`check_ops`).
- The whole file is checked before anything is written. Any problem raises `BatchError`: stderr `error: nothing written; fix the batch file:` plus one line per problem, exit 1, vault untouched.
Idempotency (re-running the same file is a no-op, counted as skipped):
- `entry`: skipped when the item already has an entry with the same source and the same text (whitespace-normalised).
- `log`: skipped when a log line with the same source and text (parsed, not substring) exists.
- `new`: skipped when an item of that type has the same title (whitespace and case-insensitive) and lists the source in `sources`.
- proposal: skipped (counted under `skipped proposals`) when a pending proposal has the same title, or the monthly log records `Applied P-xxxx: <title>` or `Rejected P-xxxx: <title>` (a decided proposal is not re-staged).
- flags already present are ignored.
Output (stdout): `applied N, skipped N, staged N (P-id P-id), skipped proposals N, flags added N`, plus `, watermark name=value` when set.
## hub clickup plan|apply --tasks FILE
`load_tasks` (`hub/clickup.py:59`), `plan_clickup` (`hub/clickup.py:116`), `apply_clickup` (`hub/clickup.py:186`). ClickUp owns task status, assignee and due date; the hub mirrors them.
Export (JSON or YAML list the agent writes from the list `sources.clickup.list_id`; `id` and `name` required, ids without whitespace):
```json
[{"id": "t1", "name": "Rerun", "status": "complete" or {"status": "complete", "type": "closed"},
  "assignees": ["<user id | username | email>" or {"id": 1, "username": "x", "email": "x@y"}],
  "due": "2026-11-26", "due_date": 1795737600000}]
```
`status` is required and non-empty (a string, or `{status, type}`). `due` is `YYYY-MM-DD`, an ISO datetime, epoch milliseconds or null; `due_date` (epoch milliseconds, or an ISO date) is read only when `due` is absent. Only the first assignee counts.
Configuration: `sources.clickup: {list_id, done_statuses?}` (default done statuses `complete, completed, closed, done`, case-insensitive); `people.<key>.clickup` = ClickUp user id, matched (with the people key, `email` and `name`, case-insensitively) against the assignee ids, usernames and emails.
`plan` prints JSON `{set, proposals, flags, push}` and writes nothing; `apply` writes, then prints `{updated, staged, flags, push}`.
| situation | result |
|---|---|
| action has `clickup_id`, task is done (status in done_statuses, or status type `closed`/`done`) and action is open | set `status: done`, `closed: <today>`, with entry `Closed in ClickUp (status '<s>').` (source `clickup:<id>`) |
| task not done but action is done or dropped | flag `clickup: <id> is <status> in the hub but task ... is '<s>' in ClickUp` |
| assignee maps to a person other than the owner | set `owner` |
| assignee maps to nobody | flag `clickup: assignee '<a>' of task ... is not in project.yaml people` |
| task has no assignee | owner unchanged |
| assignee of a new (not yet linked) task maps to nobody | flag `clickup: assignee '<a>' of new task ... is not in project.yaml people`; the proposal is still staged, with no owner |
| `due` differs | set `due` (empty clears it) |
| action's `clickup_id` missing from the export | flag `... is not in the export (deleted or moved?)` |
| open owned action without `clickup_id` | listed in `push` (`{id, title, owner, assignee, due}`; create the ClickUp task, then `hub set ID clickup_id=<id>`) |
| open task with no action carrying its id | proposal `New ClickUp task: <name>` with one `new` action op (owner mapped from the assignee, `fields: {clickup_id, due}`), to be linked `spawned_by` by a human and applied; done tasks are not proposed |
New-task dedup: skipped when a pending proposal has source `clickup:<id>`, or when the log records `Applied`/`Rejected` for the same title. Apply is retry-safe: the whole plan is pre-checked (nothing written on a problem, `ClickUpError`, exit 1); the closing entry is written before the status change and an identical existing entry is not repeated; `apply` replaces all flags starting `clickup: ` in `flags.md` with the plan's flags.
## hub meeting next SERIES YYYY-MM-DD
Stores the date in `Status/sync-state/next_meeting.<SERIES>.yaml`; prints `next <SERIES> meeting: <date>`. SERIES must be a key of `series` in project.yaml, else exit 1 (`unknown series 'x' (project.yaml series: ...)`).
## hub meeting decided SERIES YYYY-MM-DD [--json]
Decisions whose `meeting` frontmatter is `SERIES/DATE` or that have an entry with source `agenda:SERIES/DATE` (`_decided`, `hub/cli_ext.py:121`). Text: one line `<id>  <status> <title> — <as_decided or "as proposed">  actions: <ids or none>`. `--json`: list of `{id, title, status, as_decided, actions}`; `actions` are action ids with a `spawned_by` link to the decision.
## hub sources [--json]
`_source_rows` (`hub/cli_ext.py:96`). One row per configured source, in order: `clickup` (`sources.clickup`), `circleback` (`sources.circleback`: `folder`, `terms`), `git.<ROOT>` for each `sources.repos[]` and `sources.overleaf`, `calendar` (when the key is present; no watermark), `gmail` and `slack` (only when `enabled: true`). Text: `<source padded 20> <watermark or —, padded 28> <detail>`. `--json`: list of `{source, detail, watermark}` (watermark null when never ingested).
