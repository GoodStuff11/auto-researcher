---
name: hub-init
description: Use when asked to set up a new project hub, create the vault for a project, bring existing notes into a hub, or update an existing vault's tools ("set up a hub for demo", "initialise the vault", "import our old notes", "update the hub tools").
---
# Hub init
Creates one project hub vault (an Obsidian vault that is a git repository shared by the project), fills its configuration by interview, imports existing material and runs the first ingest. Work step by step and confirm with the user before every outward-facing action. `hub <command> --help` documents every option.
## Dry run
If the user says "dry run", only describe each step and show the files you would write. There is no vault yet, so these cannot or must not run: `hub init`, `git` writes, `setup.sh`, `hub validate`, `hub sources`, `hub check-links`, `hub figures add|sync|gallery`, `hub pending add|apply`, `hub ingest batch`, `hub commit`, and connector writes. The sub-skills (hub-vision, hub-figures, hub-status) and the legacy import also run in dry-run mode: describe them, and tell each so. Read-only connector calls (`list_events`, `ListFolders`, `clickup_get_list`, `clickup_get_workspace_members`) are allowed. Draft files go in the scratch directory.
## Scratch files
`SCRATCH=$(mktemp -d)` outside the vault; every temporary file (ops files, batch files, drafts) goes there, because `hub commit` stages everything in the vault. `rm -rf "$SCRATCH"` at the end.
## 0. Before starting
Ask the user for: the project name, where the vault folder should live, and whether a remote repository already exists. Creating a GitHub repository is outward-facing: ask the user to create it, or get an explicit yes before running `gh repo create --private`. Never make it public. For an existing remote, check it is private (`gh repo view --json visibility` when `gh` is available, otherwise ask the user); if it is public, stop until the user confirms.
## 1. Vault
1. Clone the empty remote, or `git init` the folder. Do this before `tools/setup.sh`: setup skips the merge-driver step when the folder is not yet a repository.
2. `hub init <path> --name <name>`, using the user's installed `hub`. If `hub` is not installed, point to USER-GUIDE Step 1 (the project-hub checkout's `USER-GUIDE.md`). Pass `--force` only when the folder already has files but is not a vault; existing files are kept.
3. `bash <path>/tools/setup.sh` (installs the vendored tools and links `hub`).
## 2. `project.yaml` interview
Work from the vault root. Ask about each key in turn, one question at a time, and fill it in:
- **people:** a key (lowercase) per person, with name and email; ClickUp ids from `clickup_get_workspace_members`.
- **series:** each recurring meeting, its attendees (people keys) and `calendar_match`, checked against real event titles with `list_events` so it matches.
- **sources:**
  - ClickUp: find the list by name with `clickup_get_list`; record `list_id`.
  - Circleback: `ListFolders` to find the folder. Recommend creating a project folder and sharing it with every collaborator.
  - repositories: a root plus a branch each.
  - Overleaf: the git clone as a root plus branch.
  - calendar.
  - Gmail and Slack: opt-in only; enable only if the user asks.
- **roots:** names for code and data folders (e.g. `SRC`, `DATA`) with their local paths.

Write `project.yaml` (shared) and the user's `hub.local.yaml` (`me: <their people key>` and their local `roots`). Then run `hub validate` and `hub sources`, and fix what they report with the user.
## 3. First commit
Ask the user before the first push; `hub commit` pushes as soon as an upstream exists. Then `hub commit "Set up project hub"`. If it reports no upstream, run `git push -u origin <branch>` once, then continue.
## 4. Vision
Run the hub-vision skill.
## 5. Existing material
For each document the user names, follow `sources/legacy-import.md` (next to this file).
## 6. Figures
Run the hub-figures skill.
## 7. Locator
1. Draft `Locator.md`: code and data roots as `$ROOT/...` paths, documents, external links. Put each `$ROOT/...` path alone in backticks (a path with spaces must be in backticks; keep commands out of backticks).
2. Show it to the user.
3. Stage it as a proposal: an ops file in the scratch directory with one `note` op, then `hub pending add --title "Locator" --ops-file <file> --source manual --reason "Initial locator"`:
```yaml
- op: note
  path: Locator.md
  mode: replace
  text: |
    # Locator
    ...
```
4. After the user's yes, `hub pending apply <P-id>`, then `hub check-links`; fix or explain every BROKEN line. `hub commit "Locator"`.
## 8. First ingest
Run the hub-status skill in update mode, first as a dry run, then for real once the user agrees.
## 9. Collaborators
Tell the user what to send each collaborator:
1. Add them to `people` in `project.yaml` first, and commit.
2. The repository invitation (the user sends it on GitHub).
3. They clone, run `bash tools/setup.sh`, and write their own `hub.local.yaml` (`me` and local `roots`).
4. Share the Circleback folder with them.
5. They connect their own services (ClickUp, Circleback, calendar, optional Gmail/Slack) and open the vault in Claude Code.
## Updating an existing vault's tools
From a newer project-hub checkout: `<newer project-hub>/.venv/bin/hub init <vault> --name <name>` (existing notes are kept; tools and skills are replaced), run `bash <vault>/tools/setup.sh` if dependencies changed, review `git diff --stat`, then `hub commit "Update hub tools"`.
## Privacy
The vault is read by everyone on the project: no personal assessments, private notes, career material or credentials in any file you write.
