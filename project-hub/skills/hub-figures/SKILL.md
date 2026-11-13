---
name: hub-figures
description: Use when asked to register, update, sync or list the figures of a project hub ("add this figure", "sync the figures", "which figures are stale", "build the gallery") in a vault that has project.yaml and a Figures/ folder.
---
# Hub figures
Registers figures with the script that makes them and the data they come from, so the hub can tell when a figure is out of date. `hub <command> --help` documents every option. `Figures/Gallery.md` and `Figures/<id>/figure.*` are generated; `Figures/<id>/caption.md` belongs to the user.
## Before starting
1. Work from the vault root (the directory with `project.yaml`); `git pull --rebase` (skip in a dry run). If it fails, stop and show the error.
2. **Dry run.** If the user says "dry run", run only `hub validate`, `hub check-links` and read files; describe the registrations and syncs without running `hub figures add|sync|gallery` or `hub commit`.
## Add
For each figure the user names, ask for:
1. a short id (lowercase, `-`);
2. the figure file under a root (`$ROOT/...`, e.g. `$SRC/plots/overview.png`);
3. the script that makes it (`$ROOT/...`);
4. its data files or folders (`$ROOT/...`, repeatable): the summary files the figure is drawn from, not a large run tree (every sync hashes every file under them);
5. a title;
6. `licenses`: one sentence on the claim the figure supports.

Draft a caption (findings first, short) and show it; the user accepts or rewrites it. Then:
```
hub figures add <id> '$SRC/plots/overview.png' --title "<title>" --script '$SRC/plots/overview.py' --data '$DATA/summary.csv' --licenses "<claim>" --caption "<caption>"
```
Quote `$ROOT` paths in single quotes so the shell leaves them alone. Then `hub commit "figures: add <id>"`. The caption is the user's from now on.
## Sync
After figures are regenerated (or whenever asked): `hub figures sync` (it rewrites the gallery itself; it exits 1 when any figure has a PROBLEM, which is expected: report it and continue), then `hub commit "figures: sync (<YYYY-MM-DD>)"`.
- **STALE**: the data changed after the figure was made. Regenerate the figure with its script (ask the user to run it, or run it if they say so), then sync again.
- **SKIPPED: not available on this machine (`$ROOT/...`)**: the figure file or one of its data entries is not on this machine (or its root is not set in `hub.local.yaml`). Nothing was recorded; a collaborator with the files will sync it. Tell the user; if they expected the files here, check the root in `hub.local.yaml`.
- **PROBLEM**: something in the vault is wrong: `meta.yaml` unreadable or holding a bad `source`/`data` value, the source is a folder, the figure folder is a symbolic link, or the vault copy could not be written. Show the line to the user. (The `script` path is not checked by sync; `hub check-links` checks it.)
## Editing
`title`, `status` (draft or final), `licenses`, `script` and `data` may be edited by hand in `Figures/<id>/meta.yaml`; then run `hub figures sync`. Never edit `caption.md` for the user unless they ask; it is theirs.
## Links
`hub check-links` reports paths (in `Locator.md` and the figures' `meta.yaml`) that are missing on this machine.
