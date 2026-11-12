# Git sync (`hub commit`) and CLI exit codes
Code: `hub/gitsync.py` (`commit`, `ensure_repo_files`, `GITATTRIBUTES_LINES`, `GITIGNORE_LINES`, `SyncError`, `SyncConflict`), CLI dispatch in `hub/cli.py` (`main`).
## `.gitattributes` lines (added if missing by every `hub commit`)
- `Status/Log/*.md merge=union` — logs are append-only; git's built-in union driver keeps both sides' appended lines instead of conflicting.
- `Status/Snapshot.md merge=ours` — generated file; the driver `merge.ours.driver=true` is set in the repo's local git config on every commit, so the upstream copy wins and the file is regenerated after the pull.
- `*.xlsx binary` — workbooks are never text-merged.
There is no line for `Status/Pending/`: every proposal is its own file, so there is nothing to merge. Item files (one file per item) and the draft YAML merge as normal text; a genuine conflict on one surfaces as `SyncConflict`.
## `.gitignore` lines (added if missing)
`hub.local.yaml`, `.venv/`, `__pycache__/`, `*.egg-info/`, `.obsidian/workspace*.json`.
## `hub commit MESSAGE` sequence
1. `ensure_repo_files`: add the lines above and set `merge.ours.driver`.
2. `git add -A -- .` (scoped to the vault directory) and commit with MESSAGE if anything is staged.
3. No remote configured: regenerate `Status/Snapshot.md`, commit it as `Regenerate generated files` if changed, return HEAD sha (done).
4. Remote but no upstream branch: `SyncError` "no upstream branch configured; run `git push -u origin <branch>` once".
5. Up to 3 attempts of: `git pull --rebase`; regenerate Snapshot and commit it; `git push`.
   - pull fails and unmerged files exist: abort the rebase and raise `SyncConflict` (lists the files; nothing lost).
   - pull fails with no unmerged files: abort any rebase in progress and raise `SyncError`.
   - push fails with non-fast-forward / fetch first / rejected: retry from `git pull --rebase`; any other push failure raises `SyncError` immediately.
   - after 3 rejected pushes: `SyncError` "push rejected 3 times".
6. Prints the HEAD sha.
## Resolving a `SyncConflict` (human)
`git pull --rebase`; open the conflicted item file(s) (both versions are in the file, frontmatter and `## Entries`); keep every entry line from both sides and merge frontmatter by hand; `git add` them and `git rebase --continue`; `hub validate`; `hub commit "<message>"`. Never hand-merge `Status/Snapshot.md` or an `.xlsx`: take either side and run `hub snapshot` / rebuild the workbook.
Warning: the live `Agenda/<series>.xlsx` must have one editor per meeting; two edited copies cannot be merged and the later push loses. Pull before the meeting, push after `hub agenda read`.
## CLI exit codes (`hub/cli.py`)
- 0 success
- 1 user error: `UserError`, `ConfigError`, `StoreError`, `PendingError`, `DraftError`, `AgendaError`, `SyncError`, `ItemFormatError`, `ValueError`, `OSError`, YAML errors, unknown item id (`KeyError`), or `hub validate` found violations; message `error: ...` on stderr
- 2 bad arguments (argparse)
- 3 `SyncConflict`
Per-command help: `hub --help`, `hub <command> --help`, `hub pending <sub> --help`, `hub agenda <sub> --help`.
