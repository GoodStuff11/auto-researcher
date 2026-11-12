# Git sync (`hub commit`) and CLI exit codes
Code: `hub/gitsync.py` (`commit`, `ensure_repo_files`, `pull_if_possible`, `GITATTRIBUTES_LINES`, `GITIGNORE_LINES`, `SyncError`, `SyncConflict`), item merge driver `hub/merge_item.py`, CLI dispatch in `hub/cli.py` (`main`).
## `.gitattributes` lines (added if missing by every `hub commit`)
- `Status/Log/*.md merge=union` — logs are append-only; git's built-in union driver keeps both sides' appended lines instead of conflicting.
- `Status/Snapshot.md merge=ours` — generated file; the driver `merge.ours.driver=true` is set in the repo's local git config, so the upstream copy wins and the file is regenerated after the pull.
- `Status/Items/*.md merge=hub-item` — item files merge with the hub item driver (below).
- `Status/sync-state/flags.md merge=union` — one flag per line; both sides' flags are kept.
- `*.xlsx binary` — workbooks are never text-merged.
- `*.md text eol=lf`, `*.yaml text eol=lf` — checked-in text uses LF, so Windows clones do not rewrite every line (the parsers also accept CRLF).
There is no line for `Status/Pending/` or the per-key `Status/sync-state/*.yaml`: each proposal and each watermark is its own file, so there is nothing to merge.
## The `hub-item` merge driver (`hub/merge_item.py`)
Registered in `.git/config` by `ensure_repo_files` (and by `pull_if_possible`): `merge.hub-item.name = hub item merge`, `merge.hub-item.driver = <python running hub> -m hub.merge_item %O %A %B`. Git calls it whenever both sides changed the same item file (O = common base, A = ours, B = theirs; an add/add of the same id has an empty base). It parses all three with `parse_item` and merges:
- entries: union of A and B, duplicates (identical rendered line) kept once, stable-sorted by date (same date: A's order, then B's);
- `sources`, `series` and every `links.*` list: union, A's order first;
- every other frontmatter key, and the description: three-way. Only one side changed it from O → take that side; both changed it to the same value → take it; both changed it differently → a real conflict.
Success: the merged item is written to A, exit 0. Real conflict: A is left as it was, one line per key `hub-item conflict in <id>: <key>: A=<A value> B=<B value>` goes to stderr, exit 1 (git marks the file conflicted). Any input that fails to parse: exit 1. During `git pull --rebase`, A is the upstream version and B is your local commit. Clones that never ran `hub` have no driver configured and fall back to a normal text merge.
## `.gitignore` lines (added if missing)
`hub.local.yaml`, `.venv/`, `__pycache__/`, `*.egg-info/`, `.obsidian/workspace*.json`, `~$*` and `.~lock.*#` (Office / LibreOffice lock files next to an open workbook), `.*.tmp` (temp files of interrupted atomic writes).
## `hub commit MESSAGE` sequence (commit → pull --rebase → regenerate → push)
1. `ensure_repo_files`: add the lines above and set `merge.ours.driver` and the `hub-item` driver.
2. `git add -A -- .` (scoped to the vault directory) and commit with MESSAGE if anything is staged.
3. No remote configured: regenerate `Status/Snapshot.md`, commit it as `Regenerate generated files` if changed, return HEAD sha (done).
4. Remote but no upstream branch: `SyncError` "no upstream branch configured; run `git push -u origin <branch>` once".
5. Up to 3 attempts of: `git pull --rebase`; regenerate Snapshot and commit it; `git push`.
   - pull fails and unmerged files exist: abort the rebase and raise `SyncConflict`. Its message lists the files, the pull output (which carries the driver's per-key `hub-item conflict …` lines) and the guidance line "re-run after fixing: `git pull --rebase`, fix the file, `git add` it, `git rebase --continue`, then `hub validate` and `hub commit`". Nothing is lost: your commit is still local.
   - pull fails with no unmerged files: abort any rebase in progress and raise `SyncError`.
   - regenerating fails (e.g. the snapshot code raises): print `warning: could not regenerate generated files: <err>` to stderr and push anyway.
   - push fails with non-fast-forward / fetch first / rejected: retry from `git pull --rebase`; any other push failure raises `SyncError` immediately.
   - after 3 rejected pushes: `SyncError` "push rejected 3 times".
6. Prints the HEAD sha.
## Best-effort pull before acting (`pull_if_possible(root) -> str | None`)
`hub pending apply` and `hub agenda read` first call it so they act on the latest shared state. It does nothing (returns None) when the vault is not a git work tree or the branch has no upstream. Otherwise it registers the merge drivers and runs `git pull --rebase`; on failure (e.g. uncommitted local changes, unreachable remote) it aborts any rebase and returns `warning: could not pull before acting (continuing with local state): <git message>`, which the CLI prints to stderr before carrying on.
## Resolving a `SyncConflict` (human)
A conflict now means both sides changed the same field of one item to different values (the message names the field and both values). `git pull --rebase`; open the conflicted item file (git leaves A's side; read the stderr lines for B); set the field to the right value; `git add` the file and `git rebase --continue`; `hub validate`; `hub commit "<message>"`. Never hand-merge `Status/Snapshot.md` or an `.xlsx`: take either side and run `hub snapshot` / rebuild the workbook.
Warning: the live `Agenda/<series>.xlsx` must have one editor per meeting; two edited copies cannot be merged and the later push loses. Pull before the meeting, push after `hub agenda read`.
## CLI exit codes (`hub/cli.py`)
- 0 success
- 1 user error: `UserError`, `ConfigError`, `StoreError`, `PendingError`, `DraftError`, `AgendaError`, `SyncError`, `ItemFormatError`, `ValueError`, `OSError`, YAML errors, unknown item id (`UnknownItem`, printed `error: no such item <id>`), or `hub validate` found violations; message `error: ...` on stderr. Any other `KeyError` is a bug and is not caught.
- 2 bad arguments (argparse)
- 3 `SyncConflict`
Per-command help: `hub --help`, `hub <command> --help`, `hub pending <sub> --help`, `hub agenda <sub> --help`.
