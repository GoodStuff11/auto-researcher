# project-hub agent docs
Read this first. Each file below is the complete truth for its domain; do not read the code to learn a format. Everything lives in the package `hub/` (`project-hub/hub/`); the CLI is `hub/cli.py` (`hub --help`, `hub <command> --help`).
- [item-format.md](item-format.md) — item files under `Status/Items/`: every frontmatter key, statuses, links, entries, IDs, validation rules (`hub/items.py`, `hub/store.py`, `hub/validate.py`, `hub/ids.py`)
- [pending-format.md](pending-format.md) — `Status/Pending/<P-id>.md` proposals (one file per proposal) and the op schema (`hub/pending.py`)
- [agenda-workbook.md](agenda-workbook.md) — draft YAML keys, the sheets and their columns, `_meta`, read-back mapping, build/read guards (`hub/agenda/draft.py`, `build.py`, `read.py`, `lifecycle.py`)
- [git-sync.md](git-sync.md) — `.gitattributes` merge drivers (incl. the `hub-item` item merge driver), `hub commit` sequence, best-effort pull, conflict handling, exit codes (`hub/gitsync.py`, `hub/merge_item.py`, `hub/cli.py`)
- [vault-layout.md](vault-layout.md) — `project.yaml`, `hub.local.yaml`, vault directory tree, `Status/Log/<YYYY-MM>.md`, `Status/sync-state/`, `Status/Snapshot.md`, `Status/Meetings/` (`hub/config.py`, `hub/sync_state.py`, `hub/snapshot.py`)
- [ingest.md](ingest.md) — source ids and watermarks, `hub seen`, `ingest git|batch` (batch file schema and rules), `clickup plan|apply` (export format, rule table), `meeting next|decided`, `sources` (`hub/sources.py`, `hub/ingest_git.py`, `hub/batch.py`, `hub/clickup.py`, `hub/cli_ext.py`)
- [office-viewer-spike.md](office-viewer-spike.md) — what VS Code Office Viewer keeps when it saves an agenda xlsx: `_meta` and the id column survive but become visible, the dropdown is not shown, styling survives (`project-hub/spike/`)

Rules of thumb for agents:
- Safe changes go through the CLI (`hub new`, `entry`, `set`, `link`, `log`). Judgement calls (closing a question, recording a decision as made outside the agenda, anything inferred from free text) go through `hub pending add`.
- Never hand-edit `Status/Snapshot.md`, `Agenda/*.xlsx` or files under `Agenda/archive/`.
- "Today" is the system date unless `HUB_TODAY=YYYY-MM-DD` is set. Your people key comes from `HUB_ME` or `me:` in `hub.local.yaml`.
