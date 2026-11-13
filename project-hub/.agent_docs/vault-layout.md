# Vault layout, configuration and small state files
Code: `hub/config.py` (`find_root`, `load_config`, `HubConfig`), `hub/sync_state.py`, `hub/snapshot.py`, `hub/fsutil.py` (`atomic_write_text`: temp file in the same directory + `os.replace`, used for items, proposals, sync-state, Snapshot, the agenda draft and meeting notes; the monthly log is appended in place).
## Directory tree (relative to the vault root, the directory holding `project.yaml`)
- `project.yaml` — shared configuration (committed)
- `hub.local.yaml` — per-user configuration (gitignored)
- `.gitattributes`, `.gitignore` — maintained by `hub commit` (see git-sync.md)
- `Vision.md` — the project vision (section skeleton from `hub init`); changed only through a `note` proposal op or by a human
- `Locator.md` — where code, data, documents and links live, as `$ROOT/relative/path` references checked by `hub check-links`. Put each `$ROOT/...` path alone in backticks; backtick any path with spaces; keep commands with flags out of backticks (a backticked command is read as one path) or put them in a fenced block (init.md)
- `Knowledge/*.md` — stable project knowledge, one note per topic; changed only through a `note` proposal op or by a human
- `Figures/<id>/{figure.<ext>, caption.md, meta.yaml}` and `Figures/Gallery.md` — figure registry (figures.md); `caption.md` is written once and never by the tools, `Gallery.md` and `figure.*` are generated
- `Status/Items.base` — Obsidian Bases views over `Status/Items/` (open questions, open actions, decisions)
- `tools/` — vendored `project-hub` copy, `setup.sh`, and the gitignored `tools/.venv` (init.md)
- `.claude/skills/hub-*` — the hub skills, refreshed by `hub init` (init.md)
- `Status/Items/<ID>.md` — one file per item (item-format.md)
- `Status/Pending/<P-id>.md` — one file per staged proposal (pending-format.md)
- `Status/Meetings/<series>/<YYYY-MM-DD>.md` — meeting note written by `hub agenda read`
- `Status/Log/<YYYY-MM>.md` — append-only journal
- `Status/Snapshot.md` — generated, never hand-edited
- `Status/sync-state/` — `watermark.<source>.yaml`, `next_meeting.<series>.yaml`, `flags.md` (one file per key)
- `Agenda/<series>.draft.yaml`, `Agenda/<series>.xlsx`, `Agenda/archive/<series>/<YYYY-MM-DD>.xlsx` (agenda-workbook.md)
## project.yaml
Parsed by `load_config` in `hub/config.py`. Required top-level keys: `name`, `people`, `series`; a missing one raises `ConfigError`.
- `name` (string) — project name, shown in the Snapshot title
- `people` (mapping) — key = people key (the identifier used as item owner and entry author), value = mapping (e.g. `{name: Alice}`); the people key must contain no whitespace. Optional per person: `clickup` (ClickUp user id) and `email`, used to map ClickUp assignees to people (ingest.md)
- `series` (mapping) — key = meeting series name; value = mapping; keys read: `attendees` (list of people keys, printed in the meeting note) and `calendar_match` (text the calendar skill matches against event titles to find the series' next meeting)
- `sources` (mapping, optional) — `HubConfig.sources`; read by the ingest code and `hub sources` (ingest.md). Keys:
  - `clickup: {list_id, done_statuses?}` — the ClickUp list and the status names meaning done
  - `circleback: {folder, terms}` — Circleback folder and search terms for this project's meetings
  - `repos: [{root, branch}]` — code repositories (`root` is a key of `roots`; `branch` optional, default HEAD)
  - `overleaf: {root, branch}` — Overleaf git clone, same shape as one `repos` entry
  - `calendar: {}` — presence enables calendar lookups for next meeting dates
  - `gmail: {enabled, terms}` and `slack: {enabled, channels}` — opt-in sources, ignored unless `enabled: true`
- `roots` (mapping, optional) — name to path; merged with `roots` from `hub.local.yaml` (local wins)
## hub.local.yaml
- `me` (string) — your people key; must be a key of `people`, else `ConfigError`. Overridden by the environment variable `HUB_ME`. Required by every command that writes (entry author).
- `roots` (mapping, optional) — local paths, merged over `project.yaml` `roots`
- `notes` (list, optional) — folders of private notes; opt-in, read only by the skill, never by the CLI
## Status/Log/<YYYY-MM>.md
First line `# Log YYYY-MM`, then one line per event in the item entry format `- YYYY-MM-DD · <author> · <source> — <text>` (see item-format.md). Written only by `Store.log` (`hub/store.py`, `hub log`, apply/reject of proposals, agenda read-back). Merges with git's union driver.
## Status/sync-state/
One small file per key, so clones updating different sources never touch the same file (`hub/sync_state.py`). Key names (source, series) must match `[A-Za-z0-9_-][A-Za-z0-9_.-]*`, else `ValueError`. Writes are atomic.
- `watermark.<source>.yaml` — `value: <opaque string>` (typically an ISO timestamp or last meeting id) and `updated: <ISO datetime, UTC>` (when the value last changed; absent in files from older versions, which count as oldest). `get_watermark`/`set_watermark` (CLI `hub sync-state get SOURCE` / `set SOURCE VALUE`) read/write only this one file; a write of the value already stored leaves the file untouched.
- `next_meeting.<series>.yaml` — `value: <YYYY-MM-DD>` and `updated:`; shown in the Snapshot meetings table.
Both kinds merge with the `hub-watermark` driver (git-sync.md): the newer `updated` wins.
- `flags.md` — one `- <text>` line per unresolved sync problem; `merge=union` in `.gitattributes`. Listed under `## Flags` in the Snapshot.
`load_state(root)` assembles `{"watermarks": {source: value}, "next_meeting": {series: date}, "flags": [text]}` (empty when absent); `save_state(root, state)` writes that dict back out (only keys whose value changed get a new `updated`), deleting key files no longer in it.
## Status/Meetings/<series>/<YYYY-MM-DD>.md
Written by `read_agenda` (`hub/agenda/lifecycle.py`). Lines: `# <series> — <date>`, `- attendees: <keys>`, `- source: agenda:<series>/<date>`, then sections `## Decisions` (`- [[<id>]] <outcome> — <text>`), `## Not reached` (`- [[<id>]]`), `## Staged for approval` (`- <P-id> · <title>`); an empty section holds `None.`. Before changing anything, `hub agenda read` writes the note as a marker (`# <series> — <date>` then `- status: read-back in progress`) and rewrites it in full at the end, so the note's existence means a previous read-back started: a second read-back is refused while it exists (a note still holding the marker means that run stopped part-way). `last_meeting(root, series, before=)` (`hub/snapshot.py`) takes the latest file stem that parses as a date.
## Status/Snapshot.md
Generated by `hub snapshot` (`render_snapshot` in `hub/snapshot.py`). Sections in order: `# Snapshot — <name>`, `## Meetings` (table Series, Last meeting, Next meeting), `## Open questions (n)` (open questions sorted by priority then raised date; columns ID, Title, Owner, Age (d), Priority; a missing `raised` counts as today, a missing priority prints `—`), `## Proposed decisions (n)` and `## Deferred decisions (n)` (columns ID, Title, Answers, Owner), `## Open actions (n)` (grouped under `### <owner>` or `### unassigned`, sorted by due date), `## Flags` (sync-state flags, then one `- cannot load <path>: <error>` per item file that failed to parse), `## Validation` (`<n> issues.`). Regenerated by `hub commit`; on merge the upstream copy wins and is then regenerated.
