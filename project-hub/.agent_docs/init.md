# Creating and updating a vault, and checking links
Code: `hub/init.py` (`init_vault`, `update_tools`, `SETUP_SH`, templates), `hub/links.py` (`check_links`), CLI in `hub/cli_ext.py` (`run_init`).
## hub init DIR --name NAME [--force]
Works outside any vault (it is handled before the vault lookup). `DIR` is created if missing. Prints each created or refreshed path, then `next: cd DIR && bash tools/setup.sh` (DIR shell-quoted).
Refusals (`error: …`, exit 1, nothing written): NAME is empty or holds a newline or other control character; `DIR` is non-empty and has no `project.yaml` (a folder holding only `.git` counts as empty) unless `--force`; `tools/setup.sh` exists without the marker line `# hub-setup: managed by hub init — do not edit`; the hub skills cannot be found next to the package.
### Created (only when absent)
`project.yaml` (commented template; the `name:` line is written by `yaml.safe_dump`, so names such as `Demo: project`, `#x` or `yes` read back as the same string), `README.md` (setup steps for collaborators), `Vision.md` (section skeleton), `Locator.md` (describes the path format as `<ROOT>/relative/path` so a fresh vault has no reference for `hub check-links` to flag; no template holds a `$NAME/` token), `Knowledge/README.md`, `Status/Items.base` (Obsidian Bases views of open questions, open actions, decisions), `Figures/Gallery.md`. `.gitattributes` and `.gitignore` get the hub lines (merged into the repository's via `ensure_repo_files` when DIR is a git repository, appended otherwise).
### Never overwritten
Existing notes and templates are kept, whatever their content (`--force` only lifts the non-empty-folder refusal). A foreign `tools/setup.sh` is refused, not replaced.
### Always replaced (this is how tools are updated)
- `tools/project-hub/` — vendored copy of `pyproject.toml`, `README.md`, `USER-GUIDE.md`, `hub/`, `skills/`, `.agent_docs/` (each item is deleted and copied afresh; caches and `.venv` excluded). Skipped when init runs from the vendored copy itself.
- `.claude/skills/hub-*` — each skill folder of the package is replaced, and a `hub-*` folder the package no longer ships is deleted (printed as `.claude/skills/<name> (removed)`); skill folders with other names are kept.
- `tools/setup.sh` — rewritten (mode 755).
## tools/setup.sh (run once per machine: `bash tools/setup.sh`)
1. Needs Python 3.10+ (`PYTHON=python3.12 bash tools/setup.sh` if `python3` is older); native Windows is not supported (use WSL).
2. Creates `tools/.venv` (gitignored) and installs the vendored package into it (network: PyYAML, openpyxl).
3. Links `~/.local/bin/hub` to `tools/.venv/bin/hub`; the link points at the vault whose setup.sh ran last.
4. If the vault is a git repository, registers the merge drivers and `.gitattributes`/`.gitignore` lines (`ensure_repo_files`). If it is not, prints a one-line note (not a repository; skipping merge-driver setup; initialise or clone the shared repo, then re-run `bash tools/setup.sh`) and still exits 0.
5. Prints a PATH hint when `~/.local/bin` is not on `PATH`.
## Updating a vault's tools
From a newer project-hub checkout: `<newer project-hub>/.venv/bin/hub init <vault> --name <name>`, then `bash tools/setup.sh` if the dependencies changed, then review and commit (`hub commit "update hub tools"`). `--name` is only used when `project.yaml` is new.
## hub check-links [--online] [--json]
Checks every `$ROOT/path` in `Locator.md` and in each `Figures/*/meta.yaml` (`source`, `script`, `data`) exists on this machine, and lists URLs (with `--online`, requests each: HEAD then GET, short timeout). Never aborts: an unreadable file or failing request is a row with `ok: false`.
Text output: one line per reference, `ok`, `BROKEN` or `listed` (URL not checked), then file, ref and problem. `--json`: list of `{file, ref, kind (path|url), ok (true|false|null), problem}`. Exit 1 if any `ok` is false, else 0.
How references are read from `Locator.md` (write it so they are found):
- Put each `$ROOT/...` path alone in backticks, e.g. `` `$SRC/results/run1` ``.
- A backticked bare `$ROOT` (nothing after it) is the root itself: it must exist and be a directory (`not a directory: <path>` otherwise). Unbackticked, a bare `$ROOT` is not a reference.
- A path with spaces must be in backticks, else it ends at the first space.
- A backticked command with flags is read as one path, so keep commands out of the backticks or put them in a fenced block.
- A `$` not followed by `NAME/` is never a path, so LaTeX `$x$` is safe; `$a/b$` maths is skipped.
- Trailing `.,;:` is dropped from a path; trailing punctuation and an unmatched `)` from a URL.
