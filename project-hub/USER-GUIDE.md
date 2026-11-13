# Project hub — user guide
How to set up and use the project hub. A project hub is a shared Obsidian vault, one per research project. Every open question, decision and action lives in it as a small Markdown file, kept up to date mostly by Claude agents. Each meeting is run from an Excel sheet that the agents prepare and read back.

This guide covers everything that is built today: Plans 1 and 2, the first two of the three build stages (Plan 1 is the `hub` command and the agenda sheet, Plan 2 the Claude skills and the source syncing; Plan 3 is not built yet). Parts that are designed but not built yet are listed in [What is not built yet](#what-is-not-built-yet); nothing else in this guide depends on them. Words in **bold** on first use are explained in the [Glossary](#glossary).

---

## Contents
1. [The idea in two minutes](#the-idea-in-two-minutes)
2. [Quick reference card](#quick-reference-card)
3. [What exists today](#what-exists-today)
4. [One-time setup (project owner)](#one-time-setup-project-owner)
5. [Joining a project (every collaborator)](#joining-a-project-every-collaborator)
6. [Obsidian and Obsidian Git](#obsidian-and-obsidian-git)
7. [Everyday use with Claude](#everyday-use-with-claude)
8. [Running a meeting with the agenda sheet](#running-a-meeting-with-the-agenda-sheet)
9. [How the status store works](#how-the-status-store-works)
10. [Using the `hub` command yourself](#using-the-hub-command-yourself)
11. [Working together safely](#working-together-safely)
12. [Troubleshooting](#troubleshooting)
13. [Known limitations](#known-limitations)
14. [What is not built yet](#what-is-not-built-yet)
15. [Glossary](#glossary)
16. [Where to look for more](#where-to-look-for-more)

---

## The idea in two minutes
Every project gets **one vault**: a folder that is both an Obsidian vault and a git repository shared by everyone on the project. Inside it, the project's state is kept as three kinds of **items**, one Markdown file each:
- **Questions** (`Q-…`): things we do not know yet ("Does the HVA beat pruned UCC at matched fidelity?").
- **Decisions** (`D-…`): choices that answer questions ("Compare only on the fermionic register").
- **Actions** (`A-…`): work someone owns ("Rerun 4×3 at L=12", owner `tamra`).

They link into a chain: a decision **answers** a question; an action is **spawned by** a decision; a question can **pivot to** a new question. Each item keeps a dated, sourced history ("2026-11-15 · tamra · commit:ML_REPO@ab12cd3 — run finished, F = 0.97").

Claude agents keep this up to date by reading ClickUp, Circleback meeting notes, your code repositories, an Overleaf clone and the calendar. They apply plain facts directly. Anything that is a **judgement call** (closing a question, recording a decision, a new action found in notes) is **staged as a proposal** that a person approves with one yes/no.

Before each meeting, an agent builds an **agenda workbook** (`.xlsx`) with three sheets: what happened since the last meeting, the open questions, and up to 7 decisions to take. In the meeting one person types the outcomes into the sheet. Afterwards an agent reads the sheet back and records everything.

You rarely touch files by hand. You ask Claude things like *"what are my todos"*, *"update the status"*, *"prep the agenda for group"* and *"we just met"*.

---

## Quick reference card
| You want to… | Say to Claude (started in the vault folder) | Or run yourself |
|---|---|---|
| see your open actions | *"what are my todos"* | `hub list --type action --status open --owner me` |
| bring the hub up to date | *"update the status"* | (agents only) |
| check first, change nothing | *"dry run: update the status"* | — |
| prepare the next meeting | *"prep the agenda for group"* | [Agenda by hand](#agenda-by-hand) |
| record a meeting | *"we just met"* (the sheet's editor, on their machine) | `hub agenda read group` |
| see proposals waiting | *"show the pending proposals"* | `hub pending list` |
| check the hub's rules | — | `hub validate` |
| share your changes | — | `hub commit "message"` |

The meeting cycle in one line: **prep** (anyone, a day or two before; it commits the sheet) → **editor pulls** → **meeting** (one editor types) → **editor says "we just met"** → **next prep** (answer *yes* when the agent asks to overwrite the old draft).

When `hub commit` exits with code 3, follow [If `hub commit` exits with code 3](#if-hub-commit-exits-with-code-3) exactly.

---

## What exists today
| Piece | What it is | Where |
|---|---|---|
| `hub` command | Python command-line tool that reads and writes the vault; every agent write goes through it | `project-hub/` in the research repository |
| `hub-status` skill | Claude skill: todos, update (read all sources), post-meeting | `project-hub/skills/hub-status/` |
| `hub-agenda` skill | Claude skill: prepare the meeting workbook | `project-hub/skills/hub-agenda/` |
| Agent docs | Exact file formats and rules, written for agents but readable by people | `project-hub/.agent_docs/` |
| Design spec | Why it is built this way (also describes parts not built yet) | `docs/superpowers/specs/2026-11-12-project-hub-design.md` |

Supported sources: **ClickUp** (one list per project), **Circleback** (meeting notes), **git repositories** (code), an **Overleaf** git clone, **Google Calendar** (next meeting date), your own **private notes folders** (opt-in), **Gmail** and **Slack** (opt-in per project). The `hub` command never talks to these services itself: the agent reads them through Claude's connectors and writes the results through `hub`.

---

## One-time setup (project owner)
Do this once per project. It takes about 30 minutes. A future `hub-init` skill will automate it ([not built yet](#what-is-not-built-yet)).
### Step 1 — Get and install the `hub` command
On every machine where you (or your agent) will work. You need **Python 3.10 or newer**: check with `python3 --version`.
```bash
cd ~/research/project-hub          # the project-hub folder of the research repository (owner: your checkout; collaborators: Joining step 3)
python3 -m venv .venv              # or: python3.12 -m venv .venv, if python3 is older than 3.10
.venv/bin/pip install -e '.[test]'
mkdir -p ~/.local/bin
ln -sf ~/research/project-hub/.venv/bin/hub ~/.local/bin/hub
hub --help                         # should print the command list
```
- If you keep the research repository somewhere other than `~/research`, replace `~/research` with your path here and in Step 6. Collaborators get the folder as described in [Joining](#joining-a-project-every-collaborator).
- If `hub --help` says *command not found*, add `~/.local/bin` to your `PATH`: `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc` (on macOS, whose default shell is zsh, use `~/.zshrc`), then open a new shell. **Restart Claude Code afterwards**, so the shell it starts also finds `hub`.
- macOS: the system `python3` is 3.9, which fails at `pip install`. Install a newer one (e.g. `brew install python@3.12`) and create the venv with `python3.12 -m venv .venv`.
- Windows: run these steps in Git Bash (Claude Code uses it too). The command is `.venv/Scripts/hub.exe` instead of `.venv/bin/hub`.
- Check the install: `cd ~/research/project-hub && .venv/bin/python -m pytest -q` should end with `passed`.
- The merge drivers that `hub commit` registers (Step 5) point at this venv's Python by its full path. If you later move, delete or recreate `.venv`, run `hub commit "…"` once in every vault on that machine to register them again.
### Step 2 — Create the vault repository
1. Create an **empty private** git repository for the project on GitHub (e.g. `<you>/fermion-unitary-hub`; no README, no licence) and invite every collaborator (Settings → Collaborators).
2. Clone it where you keep vaults: `git clone git@github.com:<you>/fermion-unitary-hub.git ~/vaults/fermion-unitary-hub`. (git warns that you cloned an empty repository; that is expected.)
3. Keep it **separate from your private Obsidian vault**: everything in this vault is read by the whole project.
4. Agents run `git pull` and `git push` on the vault without you watching, so git must never ask for a password. Use an SSH key for GitHub (the `git@github.com:` form above).
### Step 3 — Write `project.yaml` (shared configuration)
Create `project.yaml` at the root of the vault. A complete example with every key the tools read (replace the placeholder ids, emails and paths with real ones):
```yaml
name: fermion-unitary              # shown in the Snapshot title
people:                            # key = "people key", used everywhere as owner/author; no spaces
  jonathon: {name: Jonathon Kambulow, clickup: "1111111", email: jonathon@example.edu}
  tamra:    {name: Tamra Nebabu,      clickup: "2222222", email: tamra@example.edu}
  eunah:    {name: Eun-Ah Kim,        clickup: "3333333", email: eunah@example.edu}
series:                            # one entry per kind of recurring meeting
  group:
    attendees: [jonathon, tamra, eunah]   # printed in each meeting note
    calendar_match: "Fermion Sign weekly" # words that appear in that meeting's calendar title
roots:                             # named paths; each person sets their own copy in hub.local.yaml
  ML_REPO: /path/on/the/owners/machine/ML-signproblem
  PAPER:   /path/on/the/owners/machine/overleaf-clone
sources:                           # every key is optional; leave out what you do not use
  clickup:
    list_id: "900000000000"        # the project's ClickUp list: the number after /li/ in the list's URL
                                   # (…/v/li/901307841677), or ask Claude "get the ClickUp list named <name>"
    # done_statuses: [complete, closed]   # optional. Default: complete, completed, closed, done
    #                                     # (any case). Statuses of type closed/done always count.
  circleback:
    folder: "Fermion Unitary"      # Circleback folder the project's meetings are shared into
    terms: [UCC, HVA]              # extra search words (each searched separately)
  repos:
    - {root: ML_REPO, branch: main}
  overleaf: {root: PAPER, branch: master}
  calendar: {}                     # presence switches on next-meeting lookup
  gmail: {enabled: false, terms: []}
  slack: {enabled: false, channels: []}
```
How to fill it:
- **People and ClickUp ids.** Ask Claude: *"list the ClickUp workspace members with their ids"* (it uses the ClickUp connector), then copy each id as a quoted string. `email` helps match Circleback attendees and ClickUp assignees. The `name` matters too: in the meeting sheet an owner can be typed as the people key or as this full name ([During the meeting](#during-the-meeting)).
- **People keys** are short, lower-case, with no spaces (`hub` does not check this, so keep to it). Add every collaborator here **before** they set up their own clone ([Joining](#joining-a-project-every-collaborator), step 2).
- **Series.** Use one series per kind of meeting (e.g. `group` for the full group, `sub` for a smaller working meeting). Each series gets its own agenda workbook. `calendar_match` must match words in that meeting's real calendar title; if no event matches, the next-meeting date is simply not found. `attendees` should be people keys (not checked).
- **Circleback folder.** In Circleback, create a folder for the project, **share it with every collaborator**, and file the project's meetings into it. Search terms alone are unreliable: they must appear in the meeting name or notes.
- **Roots.** `roots` names local paths (code repository, Overleaf clone). The value in `project.yaml` is only a default; each person's real path goes in their `hub.local.yaml`. During an update the agent runs `git fetch` in each repository and fast-forwards the configured branch when the clone is clean, and runs `git pull` in the Overleaf clone. Point `roots` at clones where that is fine.
- **Overleaf.** Overleaf projects can be cloned with git (Overleaf → Menu → Git; this needs an Overleaf plan with git integration). Overleaf asks for a git authentication token (Account settings → Git integration). Store it with a credential helper (`git config --global credential.helper store`, or the macOS keychain helper) so that `git pull` in the clone runs without a prompt. Clone it once on your machine and point `PAPER` at it.

Which keys are checked:
- A missing `name`, `people` or `series` is an error for every command.
- Unknown top-level keys and unknown per-person keys (e.g. `role`) are ignored.
- Inside a `repos` or `overleaf` entry only `root`, `branch` (or `branches: [name]`, but not both) and `remote` are allowed. A misspelt key there is an error: `hub sources` shows that entry as `config error: …`.
### Step 4 — Write your `hub.local.yaml` (personal, never shared)
At the vault root:
```yaml
me: jonathon                       # your people key from project.yaml
roots:                             # your machine's paths (override project.yaml)
  ML_REPO: /home/<you>/research/ML-signproblem
  PAPER: /home/<you>/overleaf/fermion-unitary
notes:                             # optional, opt-in: your private note folders the agent may summarise
  - /home/<you>/obsidian/personal/research
```
- `me` must be a key of `people` in `project.yaml`. If it is not, **every** `hub` command fails (`me='x' is not a key of project.yaml people`), reads included.
- Instead of `me:` you can set the environment variable `HUB_ME=jonathon`.
- This file is git-ignored by the `.gitignore` that `hub commit` writes in Step 5. Do not run `git add` or `git commit` yourself before Step 5.
### Step 5 — First commit (sets up git merging)
From the vault root:
```bash
hub validate                        # prints "0 violation(s)" on an empty vault
hub commit "Set up project hub"     # on a new, empty GitHub repository this stops with:
                                    #   error: no upstream branch configured; run `git push -u origin <branch>` once
git push -u origin HEAD             # pushes the commit just made, whatever your branch is called
hub commit "Set up project hub"     # prints a 40-character commit id: that is success
```
- The first `hub commit` has already made the commit locally; it only could not push. The second one pushes and creates `Status/Snapshot.md`. The skills only recognise a vault that has a `Status/` folder, so do not skip it.
- Every successful `hub commit` prints just the commit id.
- `hub commit` writes two files and registers three **merge drivers** in this clone's `.git/config` (they let several people edit the vault at once; see [Working together safely](#working-together-safely)):
  - `.gitignore`: `hub.local.yaml`, `.venv/`, `__pycache__/`, `*.egg-info/`, `.obsidian/workspace*.json`, `~$*` and `.~lock.*#` (Office and LibreOffice lock files), `.*.tmp`.
  - `.gitattributes`: which merge rule applies to which files (items, logs, Snapshot, sync state), `*.xlsx binary`, and LF line endings for `*.md` and `*.yaml`.
- The rest of `.obsidian/` (your Obsidian settings and installed plugins, including Obsidian Git's settings) **is** committed and shared with the project.
- Line endings: nothing to configure on Windows or macOS. `.gitattributes` keeps text files LF and `hub` also reads CRLF. Do not change `core.autocrlf` for the vault.
### Step 6 — Make the skills visible to Claude Code
The skills live in the research repository. Link them into your personal Claude skills folder (once per machine):
```bash
mkdir -p ~/.claude/skills
ln -sfn ~/research/project-hub/skills/hub-status ~/.claude/skills/hub-status
ln -sfn ~/research/project-hub/skills/hub-agenda ~/.claude/skills/hub-agenda
```
- Windows without symlink rights: copy the two folders instead (and copy them again after each update), or use `mklink /D` in an administrator prompt.
- Do not put the links inside the shared vault: paths differ between machines.
- Start Claude Code **from the vault folder** (`cd ~/vaults/fermion-unitary-hub && claude`). Ask *"what skills do you have?"*: `hub-status` and `hub-agenda` should be listed.
- An update runs dozens of `hub` and `git` commands, and Claude Code asks permission for each one unless you allow them. Optional: add `"Bash(hub:*)"` to `permissions.allow` in `~/.claude/settings.json` (and, if you are comfortable with it, the `git` commands it uses).
### Step 7 — Connect the services Claude reads
- In claude.ai → **Settings → Connectors**, connect what the project uses: **ClickUp**, **Circleback**, **Google Calendar**, and Gmail/Slack only if they are switched on in `sources`.
- Claude Code sees these connectors only when it is logged in with that same claude.ai account (`/login` in Claude Code). A login with an API key has no connectors.
- Each person connects their own account. An agent can only see what that person can see (e.g. Circleback meetings shared with them).
### Step 8 — Open it in Obsidian
1. Obsidian → *Open folder as vault* → the vault folder.
2. Optional: install **Obsidian Git** with the settings in [Obsidian and Obsidian Git](#obsidian-and-obsidian-git). Read that section first: the wrong settings can disturb a running agent.
3. Browse `Status/Snapshot.md` for the overview; item ids in it are links to the item files.
4. **Do not edit files under `Status/` by hand.** Use Claude or the `hub` command. Hand edits to item files can break their format (`hub validate` will report it).
### Step 9 — First ingest (dry run first)
In Claude Code, from the vault folder:
1. *"dry run: update the status"* reads ClickUp, Circleback, the calendar and your repositories without writing anything, and reports what it would do. Check that every source shows as *reached*.
2. *"update the status"* is the real run. On a new project with an existing ClickUp list, expect one **"New ClickUp task: …" proposal per open task** (done tasks are never proposed). Approve the ones that belong in the hub. You can also skip a proposal and decide later. **Rejecting is permanent for that task name**: it is never proposed again ([recipe](#recipes) to add it later by hand). Proposals with no owner need an owner after approval (the agent asks).

A first run only looks back a limited time: **28 days** for Circleback and **30 days** for repositories. To ingest older history, before the first run set `hub sync-state set circleback 2026-06-01` and ask the agent to read the repositories *"since 2026-06-01"* (it runs `hub ingest git <ROOT> --since 2026-06-01`).

---

## Joining a project (every collaborator)
Do these in order. Steps 1 and 2 need the owner.
1. **Get access.** Accept the GitHub invitation to the vault repository. You also need read access to the research repository that holds `project-hub/`; ask the owner to grant it.
2. **Be added to `people`.** Send the owner the people key you want (short, no spaces), your full name, your ClickUp user id and your email. The owner adds you to `people` in `project.yaml` (and to the `attendees` of your series) and runs `hub commit "Add <name>"`. Until that is pushed, you cannot set `me` (every command would fail).
3. **Get `project-hub/`.** If you do not have the research repository yet, take only the folder you need with a sparse clone:
   ```bash
   git clone --filter=blob:none --sparse git@github.com:GoodStuff11/auto-researcher.git ~/research
   git -C ~/research sparse-checkout set project-hub docs/superpowers
   ```
   (`docs/superpowers` holds the design spec and plans that [Where to look for more](#where-to-look-for-more) points to.)
   This gives `~/research/project-hub`, the path the commands in this guide use. If you put it elsewhere, use your path in Steps 1 and 6. To update later, see [Updating `hub` and the skills](#updating-hub-and-the-skills).
4. **Install `hub`** ([Step 1](#step-1--get-and-install-the-hub-command)): Python 3.10 or newer.
5. **Clone the vault:** `git clone git@github.com:<owner>/<project>-hub.git ~/vaults/<project>-hub` (SSH, so agents never meet a password prompt). Check that `project.yaml` lists you under `people`; if not, `git pull` until the owner's change arrives.
6. **Write your `hub.local.yaml`** ([Step 4](#step-4--write-your-hublocalyaml-personal-never-shared)) with your own people key and your own paths.
7. **Register the merge drivers:** from the vault folder run `hub validate`, then `hub commit "Join <your name>"`. It prints a commit id. This must run once in **each** clone you make: the drivers live in `.git/config`, which git does not share. Your `hub.local.yaml` is not committed.
8. Link the skills ([Step 6](#step-6--make-the-skills-visible-to-claude-code)), connect your services and log Claude Code in to your claude.ai account ([Step 7](#step-7--connect-the-services-claude-reads)), open the vault in Obsidian ([Step 8](#step-8--open-it-in-obsidian)).
9. Make sure the project's Circleback folder is shared with you, otherwise your agent will report Circleback as *not reachable*.
10. Check everything with *"dry run: update the status"*.
### Updating `hub` and the skills
`git -C ~/research pull`. The install is editable and the skills are symlinked, so both pick up the change. Re-run `.venv/bin/pip install -e '.[test]'` only if `pyproject.toml` changed. If you copied the skill folders instead of linking them, copy them again.

---

## Obsidian and Obsidian Git
Obsidian is only a viewer here. The agents do all the git work themselves. Obsidian Git is useful for pulling others' changes into the vault you are reading, and for sharing notes you add by hand. Used carelessly, though, it commits and pushes at the same moments an agent does.

**The risks.**
- *Auto commit-and-sync* commits **everything** in the vault. That includes an agent's half-finished run: what one source has written before the agent's own `hub commit`, or the meeting sheet while it is being edited.
- It can run `git pull` or `git push` at the same moment as the agent's `hub commit`. The agent's command then fails (e.g. `index.lock` exists) and has to be re-run, or extra commits appear.
- Its default pull method is *merge*. `hub` always uses *rebase*.
- **Obsidian mobile** uses a built-in git library that does not run `hub`'s merge drivers. Two people's edits to one item then conflict as plain text.
- The plugin's settings live in `.obsidian/`, which is committed. Whatever one person sets is shared with everyone, so agree on the settings as a project.

**Safe settings** (Obsidian → Settings → Obsidian Git; names differ slightly between plugin versions):

| Setting | Value |
|---|---|
| Pull on startup | **On** |
| Merge strategy / Sync method | **Rebase** |
| Auto commit-and-sync interval | **0 (off)**. Use the *Commit-and-sync* command by hand when you added notes. If you want it on, use a long interval (60 minutes or more). |
| While an agent is running | Pause auto commit-and-sync, or close Obsidian. |
| Auto pull interval | Optional, e.g. 10 minutes (pulling is harmless) |
| Mobile | No auto commit-and-sync, and no edits under `Status/` |

The meeting sheet's editor should not let Obsidian Git push the sheet half-way through the meeting. Nothing breaks if it does, but [the sheet workflow](#who-does-what) assumes the editor decides when the sheet is shared.

---

## Everyday use with Claude
Always start Claude Code **from the vault folder**. Things to say:

| You say | What happens |
|---|---|
| *"what are my todos"* | Lists your open actions sorted by due date, then priority. Read-only. Works for others too: *"what are Tamra's todos"*. Mentions waiting proposals. |
| *"update the status"* | Reads every configured source, one at a time, committing and pushing after each. Then asks you once to approve the proposals it staged, closes the gaps (see below), creates ClickUp tasks for new owned actions, and reports per source. |
| *"dry run: update the status"* | Same, but read-only: no hub writes, no commits, no `git fetch`/pull, no ClickUp changes. Shows what it would write. Use it whenever you are unsure. |
| *"prep the agenda for group"* | Builds `Agenda/group.xlsx` for the next `group` meeting (see [Running a meeting](#running-a-meeting-with-the-agenda-sheet)). |
| *"dry run: prep the agenda for group"* | Shows the draft text it would write and which sources are stale; writes nothing. |
| *"we just met"* / *"process the meeting"* | Reads the meeting sheet back, cross-checks the Circleback notes, reads the other sources, asks for approvals, commits. Run it on the machine where the sheet was edited ([who does what](#who-does-what)). |
| *"dry run: we just met"* | Reports what it would do; reads only. |
| *"show the pending proposals"* | Lists proposals still waiting, from anyone, and lets you apply or reject them. |
### What an update does, source by source
- **ClickUp.** ClickUp is the authority for task **status, assignee and due date**. Those are copied onto the linked hub actions automatically, so an owner or due date you change in the hub is overwritten at the next update. A task created directly in ClickUp becomes a *"New ClickUp task: …"* proposal. After approvals, every open owned action without a task gets a ClickUp task created from **your** ClickUp account: assigned to the owner's `clickup` id (unassigned if they have none), with the due date and the description *"Hub item A-…"*. The hub never changes a task's status, assignee or due date, and never deletes tasks. Mismatches (e.g. done in the hub but open in ClickUp) are shown as **flags** in the Snapshot for a person to resolve. ClickUp is always reconciled as a whole list, so it has no backlog to catch up on.
- **Circleback.** New meetings in the project folder or matching the terms are read: a short log line per meeting, entries on items that were discussed, and new open questions directly. **Decisions, actions and closures become proposals.** A meeting of a configured series whose sheet has not been read back yet is left for the post-meeting run.
- **Repositories and Overleaf.** The agent fetches each repository (fast-forwarding the configured branch when the clone is clean) and pulls the Overleaf clone. New commits are grouped by topic into log lines ("Trotter optimizer: warm starts resume in order (3 commits)"). Entries go on items a commit clearly serves. A commit that seems to finish an action or answer a question becomes a proposal to close it.
- **Calendar.** Records the next meeting date of each series (it searches the next 7 days, then up to 5 weeks ahead).
- **Private notes** (only folders you listed under `notes:`). Project facts are summarised in the agent's own words, never copied; anything that decides or closes is a proposal.
- **Gmail / Slack.** Only if `enabled: true`; same rules as Circleback.
### The approval prompt
At the end of an update the agent lists the proposals staged in this run (who proposed each, and what applying it does) and asks once: **apply all, apply some, or reject some**. Things it will point out:
- *"New ClickUp task"* proposals whose task is not really active (e.g. status "paused"): they would become open actions.
- Proposals with **no owner**.
- If an open action with the same title already exists, it links that action to the task and rejects the duplicate proposal instead.

**Rejecting is permanent for that title.** A proposal whose title was once applied or rejected is never staged again, and the same goes for a ClickUp task name. If in doubt, skip it: proposals you neither apply nor reject stay pending in `Status/Pending/`. `hub pending list` shows them, and *"what are my todos"* mentions that some are waiting. Later say *"show the pending proposals"*, or run `hub pending apply P-…` / `hub pending reject P-…`.

Proposals staged by other people are mentioned separately and left alone unless you say otherwise. Anyone *can* apply anyone's proposal; by convention, leave others' proposals to them unless they ask.

After applying, the agent runs `hub validate` and **closes the gaps** ([Validation rules](#validation-rules-what-hub-validate-reports)). It asks you:
- who owns each new action that has no owner;
- which question a made decision answers (it creates the question if none exists);
- which action follows from a made decision, or why none does;
- which decision a new ClickUp-task action serves.

A decision approved on the meeting sheet with no linked question or no action leads to the same questions.
### What is applied directly vs proposed
| Applied directly | Staged for your approval |
|---|---|
| a dated entry on an existing item, with its source | closing a question (answered, dropped, pivoted) |
| a log line, with its source | recording a decision as made (except outcomes of planned decisions on the meeting sheet) |
| a new open question raised somewhere | new actions from ClickUp or from the meeting sheet's *New actions* column |
| ClickUp status/assignee/due copied onto linked actions | every action found in notes, Circleback, Gmail or Slack |
| outcomes typed for the **planned** decisions on the meeting sheet (Approve/Modify/Defer/Reject) | decisions raised in the meeting (yellow rows), unrecognised outcome text, and anything else that needs judgement |
### Privacy
The vault is read by everyone on the project. Agents record decisions, stated positions ("Eun-Ah prefers framing B") and actions, each with its source. They never record personal assessments of people, career or 1:1 material, health, private Circleback notes, or verbatim text from anyone's private notes. Your private notes are only read if you list their folders under `notes:` in **your** `hub.local.yaml`.

---

## Running a meeting with the agenda sheet
### Who does what
The sheet is a binary file: two edited copies cannot be merged, and the later push wins. So **one person, the editor, types into it**, and the edits stay on the editor's machine until they are read back or pushed.

| When | Who | What |
|---|---|---|
| A day or two before | anyone (usually the owner) | *"prep the agenda for group"*. The agent builds `Agenda/group.xlsx` and commits and pushes it. |
| Just before the meeting | the editor | **Pull**, so you have the new sheet: Obsidian Git *Pull*, or `git pull --rebase` in the vault. Then open `Agenda/group.xlsx`. |
| During | the editor only | Type outcomes on the **Decisions** sheet ([below](#during-the-meeting)); save. Everyone else only looks (screen share), never saves. |
| Right after | the editor | Save, then say *"we just met"* in Claude Code **on the same machine**. |
| …or, if someone else processes the meeting | the editor first | Run `hub commit "agenda: group <date> meeting edits"` to push the sheet, then tell that person. They say *"we just met"* on their machine. |
| Before the next meeting | whoever preps | *"prep the agenda for group"*. The agent says the old draft exists and asks before overwriting it (`--force`): answer **yes**. That draft belongs to the meeting already read back. Answer no only if you edited the draft for the coming meeting by hand. |

If you are not the editor and you saved the sheet by accident, throw your copy away before pulling or committing: `git checkout -- Agenda/group.xlsx`.
### Before the meeting
Ask Claude *"prep the agenda for group"* (a day or two before). It:
1. Finds the meeting date (from you, or the calendar).
2. Checks the **previous meeting was read back**; if not, it offers to do that first.
3. Checks every source is fresh; if not, it runs an update first.
4. Pre-fills `Agenda/group.draft.yaml` from the store (asking before it overwrites an old draft) and writes the summary and follow-ups.
5. Builds `Agenda/group.xlsx` and shows you which decisions it chose, why, which it left off, and the total minutes against the meeting length.
6. Validates and commits (`agenda: group <date>`), so the editor can pull the sheet.

You can ask it to drop, add or reorder decisions; it rebuilds. To **add** a decision that is not in the hub yet, say so. The agent creates it (and its question) in the hub first, then re-scaffolds the draft, keeping its text, and rebuilds, so the sheet can be read back.

**How decisions get onto the sheet.** Only decisions with status `proposed` or `deferred` can appear (a new `hub new d` is `proposed`). The sheet takes the `proposed` decisions of this series first, plus those with no series: priority 1 first, then the oldest. Then it adds `deferred` ones while room remains, up to **7**. Deferred and not-reached decisions come back automatically.
- To push a decision up: `hub set D-… priority=1`.
- To pre-fill its *Proposed action* and *Reason* cells: `hub set D-… proposed_action="…" rationale="…"`.
- Priority also orders todos, the Snapshot and the *Open questions* sheet.
### The workbook
| Sheet | Contents | Do you edit it? |
|---|---|---|
| **Since last meeting** | A short summary on top, then one row per decision taken last time: the actions it spawned, what was done, and the outcome (closed question / new question / pivot / in progress, coloured) | No |
| **Open questions** | Every open question of the series (and those with no series), with owner, age and priority | No |
| **Decisions** | Up to 7 decisions to take: what to decide, proposed action, reason, who weighs in, minutes, link to a figure or pre-read; then 5 **yellow rows** for decisions raised in the meeting | **Yes**: the last three columns (grey), and the yellow rows |

Open it in Excel, LibreOffice, or VS Code with the **Office Viewer** extension (all work; see below).
### During the meeting
On the **Decisions** sheet, for each decision discussed:

| Column | What to type |
|---|---|
| **Outcome** | `Approve`, `Modify`, `Defer` or `Reject`. Short forms work: `a`, `m`, `d`, `r`, `yes`/`y`, `no`/`n`, `x`, `✓`, `✗`; a trailing `.` or `!` is fine. Leave it empty if the decision was not reached. |
| **As decided** | The decision as actually taken, in one sentence (especially for Modify; for Reject, the reason). |
| **New actions · owners** | `title (owner); title (owner)`, e.g. `Rerun 4x3 at L=12 (tamra); Redraw Fig. 2 (jonathon)`. One action per `;` or per line. |
- **Owners** are matched against `people` in any case. Use the people key (`tamra`) or the person's full `name` exactly as in `project.yaml` (`Tamra Nebabu`). Nicknames and partial names (`Eun-Ah`, `Tamra N.`) match nobody. The bracket then stays in the title and the action has no owner; after approval the agent asks who owns it. Anything in brackets that is not a person stays in the title on purpose, e.g. `Review draft (due Friday)`.
- **A decision nobody planned**: use a **yellow row**. Write it under *What to decide* and fill the same three columns. It becomes a proposal (with its actions); after approval, the agent asks which question it answers. If a yellow row's text equals the title of an existing proposed or deferred decision, it updates that decision instead.
- **Rewording a planned decision**: you may edit *What to decide*, but the stored title does not change. The meeting's wording is kept in *as decided* (when that cell is empty) or in the entry.
- **Anything else typed into *Outcome*** (e.g. "approve, but only 4x3") is not guessed at: it is staged for review. Applying that proposal only records the text; then set the status yourself (`hub set D-… status=made`, or ask Claude).

**Office Viewer (VS Code) note:** it shows no dropdown in the Outcome column, and it shows the normally hidden column A (item ids) and the `_meta` tab. That is fine: type the outcome, and do not edit column A or `_meta`. Save normally.
### After the meeting
The editor saves the workbook, then tells Claude *"we just met"* on the same machine (or pushes the sheet first, see [Who does what](#who-does-what)). The agent:
1. Pulls with `--autostash`, which keeps your uncommitted sheet edits safe while it pulls. It stops if git could not put them back: your edits are then in `git stash list`, and nothing has been read.
2. Runs `hub agenda read group`. That command prints `warning: could not pull before acting … unstaged changes`. **This is expected** when the edited sheet is not committed: the agent pulled just before, and the read still works.
3. Reads the sheet back, row by row (table below).
4. Writes a meeting note `Status/Meetings/group/<date>.md` (decisions, not reached, staged), and archives the workbook to `Agenda/archive/group/<date>.xlsx`. The note lists the series' configured `attendees`, not who actually came. The agent then commits.
5. Cross-checks the Circleback notes: anything decided or assigned aloud but not on the sheet becomes a proposal.
6. Reads the other sources, asks for approvals, closes the gaps, creates ClickUp tasks for new actions, commits.

How each row of the Decisions sheet is read back:

| Outcome typed | What is recorded |
|---|---|
| Approve / Modify | decision set to `made`, with the meeting and *as decided*; its New actions become a proposal |
| Defer | decision set to `deferred`; it comes back on the next agenda |
| Reject | decision set to `rejected`, with `closed` set to today |
| (empty) | status unchanged and a *"Not reached"* entry; it comes back on the next agenda. Text typed in *As decided* / *New actions* without an outcome is staged for review. |
| unrecognised text | staged for review ([above](#during-the-meeting)) |
| on a decision already resolved earlier | staged for review; the decision is not changed |
| yellow row | a proposal for a new decision plus its actions |

---

## How the status store works
### Vault layout
| Path | What | Written by |
|---|---|---|
| `project.yaml` | Shared configuration | You |
| `hub.local.yaml` | Your personal configuration (git-ignored) | You |
| `.gitignore`, `.gitattributes` | Ignore list and merge rules | `hub commit` (adds missing lines) |
| `Status/Items/<ID>.md` | One file per question, decision, action | `hub` only |
| `Status/Pending/<P-id>.md` | One file per proposal awaiting approval | `hub` only |
| `Status/Meetings/<series>/<date>.md` | What a meeting decided | `hub agenda read` |
| `Status/Log/<YYYY-MM>.md` | Monthly journal of events (runs finished, drafts pushed, meetings, applied/rejected proposals) | `hub` only |
| `Status/Snapshot.md` | One-page overview, regenerated on every commit | `hub` only; never edit |
| `Status/sync-state/` | One small file per source watermark and per series' next meeting date, plus `flags.md` | `hub` only |
| `Agenda/<series>.draft.yaml` | The agent's content for the next workbook | the agenda agent |
| `Agenda/<series>.xlsx` | The live meeting workbook | `hub agenda build`; the editor in the meeting |
| `Agenda/archive/<series>/<date>.xlsx` | Past workbooks | `hub agenda read` |
### An item file
```markdown
---
id: Q-20261112-a3f
type: question
title: Does the HVA beat pruned UCC at matched fidelity?
status: open
owner: jonathon
series:
- group
raised: 2026-11-12
closed:
priority: 2
links:
  answers: []
  spawned_by: []
  pivoted_to: []
sources:
- circleback:abc123
---
Why it matters, in one paragraph.
## Entries
- 2026-11-12 · jonathon · circleback:abc123 — Raised.
- 2026-11-15 · tamra · commit:ML_REPO@ab12cd3 — 4×3 run finished: F = 0.97.
```
- **Ids** are `Q-`/`D-`/`A-` + creation date + 3 random characters, so two people creating items at the same time never collide.
- **Statuses**: question `open | answered | dropped | pivoted`; decision `proposed | made | rejected | deferred`; action `open | done | dropped`. Every status except open/proposed/deferred/made is a *closed* status and needs a `closed` date.
- **Series**: the meeting series the item may appear on. An item with **no** series appears on **every** series' agenda.
- **Links** are stored on one side only: `answers` (on a decision → the questions it answers), `spawned_by` (on an action or question → where it came from), `pivoted_to` (on a question → the question that replaced it).
- **Entries** are append-only: date · author (people key) · source · text. The first entry is always `Raised.`.
- **Priority** is 1 (highest), 2 (default) or 3.
- Extra frontmatter:
  - decisions: `rationale` (why), `proposed_action` (shown on the agenda), `meeting` (`<series>/<date>` when decided in a meeting), `as_decided`, optional `no_action` (why no action follows);
  - actions: `due` and `clickup_id`.
### Sources and watermarks
Every fact cites where it came from:

| Source id | Meaning |
|---|---|
| `circleback:<meeting id>` | a Circleback meeting |
| `clickup:<task id>` | a ClickUp task |
| `commit:<ROOT>@<short sha>` | a commit in a repository or the Overleaf clone |
| `notes:<people key>@<date>-<file>` | one of your private notes |
| `gmail:<thread id>`, `slack:<channel>-<ts>` | mail and chat |
| `agenda:<series>/<date>` | the meeting sheet read-back |
| `pending:<P-id>` | written when a proposal is applied (its log line) |
| `manual` | typed by a person |

A **watermark** records how far each source has been read (`hub sources` shows them). Agents only read what is new since the watermark. `hub seen <source id>` tells them whether something was already recorded, so running an update twice does not duplicate anything.

| Watermark | Value it holds |
|---|---|
| `clickup` | time of the last sync (ClickUp is always read whole; this only records when) |
| `circleback` | date (`YYYY-MM-DD`) of the newest meeting ingested |
| `git.<ROOT>` | full commit id (sha) of the newest commit ingested, not a date |
| `notes.<people key>`, `gmail`, `slack` | date (`YYYY-MM-DD`) |
### The Snapshot
`Status/Snapshot.md` is the page to read in Obsidian. Sections:
- meetings (last and next per series);
- open questions;
- proposed and deferred decisions;
- open actions grouped by owner;
- **flags**: problems a person should look at, e.g. ClickUp mismatches or unreadable files;
- the validation count.

---

## Using the `hub` command yourself
You never have to, but everything the agents do you can do by hand. Run it anywhere inside the vault, or pass `--root <vault>` **before** the command: `hub --root ~/vaults/x list`. Every command has full help: `hub --help`, `hub <command> --help`, `hub agenda <sub> --help`.

Commands that need your people key (`me`): `new`, `entry`, `log`, `list --owner me`, `pending apply|reject|add`, `agenda read`, `ingest batch`, `clickup apply`. Other commands run without it.

"Today" is the system date. To back-date (e.g. record last Friday's decision), prefix the command: `HUB_TODAY=2026-11-14 hub entry …`.
### Reading
| Command | Does |
|---|---|
| `hub list --type action --status open --owner me` | your open actions (`--json` adds due, priority, clickup_id, last activity) |
| `hub list --type question --series group` | questions that name the series `group` (add `--status open`). Items with **no** series are not listed here, although they do appear on the agenda: run it without `--series` to see them. |
| `hub list --since 2026-11-01` / `hub since 2026-11-01 [--series group]` | items with activity on or after a date |
| `hub show <ID>` / `hub show <ID> --chain` | one item / with everything that descends from it |
| `hub pending list` | proposals awaiting approval |
| `hub sources` (`--json`) | configured sources and their watermarks |
| `hub seen <source id>` | `new`, or `seen` and where it is recorded (item ids, `log:<month>`, proposal ids) |
| `hub meeting decided group 2026-11-19 --json` | what a meeting recorded |
| `hub validate` | check every rule (exit 1 if any violation) |
### Writing
| Command | Does |
|---|---|
| `hub new q "Title" --source manual --series group` | new question (`d` decision, `a` action); prints the id. Options: `--owner` (default: you), `--priority 1-3`, `--link answers:<Q>`, `--link spawned_by:<D>`, `--description`. `--series` and `--link` may be repeated; no `--series` means every series. |
| `hub entry <ID> "text" --source manual` | add a dated history entry |
| `hub set <ID> status=answered closed=2026-11-20` | change fields (dates `YYYY-MM-DD`; `field=` clears). Keep `priority` to 1–3 and `owner` to a people key: `set` does not check them, `validate` reports a bad owner. |
| `hub link <D> answers <Q>` | add a link |
| `hub log "text" --source manual` | add a line to the monthly log |
| `hub pending apply P-xxxx …` / `hub pending reject P-xxxx …` | approve or reject proposals (apply pulls first) |
| `hub snapshot` | regenerate the Snapshot |
| `hub commit "message"` | add **everything** in the vault, commit, pull, regenerate the Snapshot, push; prints the commit id |

`entry`, `set`, `link`, `log` and `pending reject` print nothing on success.
### Commands agents use (you rarely need them)
| Command | Does |
|---|---|
| `hub meeting next group 2026-11-19` | set the next meeting date by hand (when the calendar has none) |
| `hub sync-state get <source>` / `set <source> <value>` | read / move a watermark (value format per source in [the watermark table](#sources-and-watermarks); `set` does not check it) |
| `hub ingest git <ROOT> [--since D] [--limit N]` | list new commits of a repository as JSON (first run: last 30 days; at most N, default 500). `--advance [--to SHA]` moves the watermark; use it only after the commits are recorded |
| `hub ingest batch F.yaml` | apply an agent-written batch (all-or-nothing, safe to re-run; format in `.agent_docs/ingest.md`) |
| `hub clickup plan --tasks F.json` / `apply` | reconcile a ClickUp list export (the agent makes the export with the connector; `plan` writes nothing) |
| `hub pending add --title T --ops-file F.yaml --source S --reason R` | stage your own proposal (format in `.agent_docs/pending-format.md`; owners are not checked) |
| `hub agenda scaffold … --force` / `hub agenda build … --force` | overwrite an existing draft / build over an unread workbook (discards its meeting edits) |
### Recipes
- **Close a question:** `hub entry Q-… "Answered: …" --source manual` then `hub set Q-… status=answered closed=<today>`. A closed item needs an entry dated on or after its `closed` date.
- **Record a decision by hand:** `hub new d "Compare on the fermionic register" --link answers:Q-… --source manual`, then `hub set D-… status=made`, then either `hub new a "…" --owner tamra --link spawned_by:D-… --source manual` or `hub set D-… no_action="framing choice only"`.
- **Finish, reassign or re-date an action linked to ClickUp:** do it **in ClickUp**; the next update copies it over. For an unlinked action: `hub set A-… owner=tamra`, `hub set A-… due=2026-12-01`, or close it with an entry and `hub set A-… status=done closed=<today>`.
- **Clear a "done in the hub but open in ClickUp" flag:** fix the cause. Close the task in ClickUp, or reopen the action (`hub set A-… status=open closed=`). The next update rewrites all `clickup:` flags.
- **Add a ClickUp task whose proposal you rejected:** `hub new a "<title>" --owner <key> --source clickup:<task id>`, then `hub set A-… clickup_id=<task id>`, and `hub link A-… spawned_by D-…` for the decision it serves.
- **Re-read a source from an earlier date:**
  - Circleback, notes, Gmail, Slack: `hub sync-state set circleback 2026-06-01` (a date), then update.
  - A repository: its watermark is a commit, not a date. Ask the agent to read it *"since 2026-06-01"* (`hub ingest git ML_REPO --since 2026-06-01`). Or set the watermark to an older commit's full sha: `git -C <repo> log -1 --format=%H --until=2026-06-01` prints one.
  - ClickUp is always read whole.

  Re-reading is safe: anything already recorded is skipped.
- **Commit:** `hub commit "…"` commits everything in the vault folder (`git add -A`). Keep scratch files, exports and downloads **outside** the vault.
### Agenda by hand
What the agenda skill does, step by step:
```bash
git pull --rebase
# first: every decision to take must exist in the hub as a proposed decision
#   hub new d "What to decide" --link answers:Q-… --series group --source manual
# (the Decisions sheet is filled only from proposed/deferred decisions; see the notes below)
hub agenda scaffold group --date 2026-11-19    # from the second meeting on, add --force (the old draft is no longer needed)
# edit Agenda/group.draft.yaml (see below)
hub agenda build group
hub validate
hub commit "agenda: group 2026-11-19"           # so the editor can pull the sheet
# ... the meeting: the editor fills in the sheet and saves ...
git pull --rebase --autostash                   # on the editor's machine; the sheet is uncommitted
hub agenda read group                           # the "could not pull before acting" warning is expected here
hub commit "status: agenda read-back group 2026-11-19"
hub pending list                                # then hub pending apply P-… (or reject)
hub validate                                    # fix what it reports (Troubleshooting table)
hub commit "status: approvals group 2026-11-19"
```
In the draft:
- `summary` must not be empty.
- In each of the at most 7 `decisions`, `weigh_in` is one string (`"eunah, tamra"`, not a YAML list) and `minutes` a whole number.
- `since_last[].follow_up` is free text.
- Only decisions with status `proposed` or `deferred` are scaffolded, so creating decisions (`hub new d …`) is how you fill the Decisions sheet. Scaffold again (`--force`) after creating one; copy your draft text out of the vault first, because `--force` replaces it.

**Do not run `hub agenda build` again after the read-back** for the same meeting. It rebuilds the old sheet as "not read back", and the next prep then refuses. Scaffold the next date first. (If it happened: `hub agenda build group --force` after scaffolding the next date.)

Exit codes: `0` ok, `1` error or validation violations (message starts `error:`), `2` bad arguments, `3` sync conflict.

---

## Working together safely
Several people (and their agents) write to the same vault. What makes that safe:
- **One file per item and per proposal**, random ids, append-only logs: two people rarely touch the same file.
- **Merge drivers** (registered in each clone by `hub commit`):
  - Two people changing the same item merge automatically: entries, sources, series and links from both sides are kept. Only two *different* new values for the same field (title, status, owner, description, …) conflict.
  - Watermark and next-meeting files keep the side written most recently (its `updated:` time), even if its value is older. At worst that re-reads work, which `hub seen` then skips; it never skips anything.
  - Logs and the flags file keep both sides' lines.
  - `Snapshot.md` never conflicts: the incoming copy is taken and then rebuilt.
- **Commit after each source**: an update commits and pushes after every source, so a colleague's run a few minutes later sees your work and does not record the same meeting or task again.
- **ClickUp duplicates are caught**: a proposal cannot create a second action for a task that is already linked. If two actions still end up on one task (a hand `hub set`, or two clones), `hub validate` reports `duplicate-clickup-id` and the Snapshot shows a flag.

Rules for people:
- Run `hub commit` once in every new clone (registers the drivers), and again after moving or recreating `hub`'s `.venv`.
- One editor per meeting workbook ([Who does what](#who-does-what)).
- Never hand-merge `Status/Snapshot.md` or an `.xlsx`: take either side and regenerate (`hub snapshot`, or rebuild the workbook).
- Never commit `hub.local.yaml`. `hub commit` commits everything in the vault folder, so keep scratch files and downloads outside it.
- Use Obsidian Git only with the [safe settings](#obsidian-and-obsidian-git).
### If `hub commit` exits with code 3
A real conflict: both sides changed the **same field of one item** to different values. The message is long (git's own output is in it). Look for the line:
```
hub-item conflict in Q-20261113-e6h: title: A=Title from Tamra B=Title from Jon
```
`A=` is the version already on GitHub, `B=` is yours. Your commit is still local, and `hub commit` has already undone its own pull, so nothing has changed in your files yet.

When you pull again, the item file already holds both sides' changes except the field(s) named in the message, which show the GitHub value. Your value is only in the message: copy it before you edit. Replace `<ID>` with the item id from the message.
```bash
git pull --rebase
# edit the named field(s) in Status/Items/<ID>.md
git add Status/Items/<ID>.md
GIT_EDITOR=true git rebase --continue
hub validate
hub commit "Resolve conflict on <ID>"
```
- In the edit step, set each named field to the right value (agree with the other person if unsure). Every entry, source, link and other field from both sides is already in the file; do not delete them.
- `GIT_EDITOR=true` keeps your commit message. Without it, an editor opens for the message: save and close it.
- **Several files** listed after `merge conflict in:` → edit and `git add` each one before `git rebase --continue`. If the rebase stops again on another of your commits, repeat from `git pull --rebase`.
- A conflicted **`Status/sync-state/*.yaml`** file → compare the two with `git show REBASE_HEAD:<file>` and keep the side with the newer `updated:`.
- If `git pull --rebase` succeeds without a conflict (the remote changed again), just run `hub commit` again.
- To back out at any point: `git rebase --abort`. You are then back where `hub commit` left you, with your commit still local.
- Agents never resolve a conflict themselves: they stop and show you the message.

---

## Troubleshooting
### Validation rules (what `hub validate` reports)
| Rule | Meaning | Fix |
|---|---|---|
| `decision-answers` | a made decision answers no question | `hub link <D> answers <Q>` (create the question first with `hub new q` if needed) |
| `decision-actions` | a made decision has no action and no reason | `hub new a "…" --owner <key> --link spawned_by:<D> --source manual`, or `hub set <D> no_action="<reason>"` |
| `action-owner` | an open action has no owner, or one not in `people` | `hub set <A> owner=<key>` |
| `unknown-owner` | an owner is not a key of `people`. On an open action this comes together with `action-owner`; one `hub set` fixes both. | add the person to `project.yaml`, or `hub set <ID> owner=<key>` |
| `closed-date` | a closed item has no valid `closed` date | `hub set <ID> closed=YYYY-MM-DD` |
| `closing-entry` | a closed item has no entry on/after its `closed` date | `hub entry <ID> "why it closed" --source manual` |
| `pivot-target` | a pivoted question does not say what replaced it | `hub link <Q> pivoted_to <new Q>` |
| `link-target` | a link points at an item that does not exist | `hub show <ID>`, then fix the link (ask Claude) |
| `duplicate-clickup-id` | two actions are linked to the same ClickUp task | decide which to keep; clear the other with `hub set <A> clickup_id=` |
| `entry-author` | an entry's author is not in `people` | add the person to `project.yaml` |
| `format`, `shape`, `type`, `status`, `id-prefix` | an item file was hand-edited into an invalid state | open the file, compare with the [item file](#an-item-file) above, fix or ask Claude; `git log -p <file>` shows what changed, `git checkout -- <file>` restores the last committed version |
### Common messages
| Message | What to do |
|---|---|
| `writes need your people key` (also from `list --owner me`) | add `me: <key>` to `hub.local.yaml` (or set `HUB_ME`) |
| `me='x' is not a key of project.yaml people` | fix `me`, or have the owner add you to `people` (and pull). Until then remove `me:`. |
| `no project.yaml found` | run from inside the vault, or pass `--root <vault>` before the command |
| `unrecognized arguments: --root …` | put `--root` before the command: `hub --root <vault> list` |
| `no upstream branch configured` | `git push -u origin HEAD` once, then `hub commit` again |
| `merge conflict in: …` with `hub-item conflict in <ID>: <field>: A=… B=…`, exit 3 | follow [If `hub commit` exits with code 3](#if-hub-commit-exits-with-code-3) |
| `… index.lock …` from `hub commit` | another git process (often Obsidian Git) was running at the same moment; run the command again |
| `could not pull before acting … unstaged changes` from `hub agenda read` | **expected** when the edited meeting sheet is not committed; the read still works (pull with `git pull --rebase --autostash` first) |
| `could not pull before acting … unstaged changes` from `hub pending apply` | something was left uncommitted; `hub commit "…"` first |
| `the <date> agenda has not been read back` (agenda build) | run the post-meeting step first (*"we just met"*). If that meeting was in fact read back, you rebuilt it afterwards: `--force` is safe then. Otherwise use `--force` only if the old sheet's edits can be discarded. |
| `live agenda already contains meeting input` (agenda build) | the sheet was already filled in: read it back first |
| `draft … already exists; pass --force` (agenda scaffold) | expected at every prep after the first; the agent asks you. By hand: add `--force` (copy the draft out first if you edited it for the coming meeting) |
| `summary is empty` (agenda build) | write the `summary` in `Agenda/<series>.draft.yaml` |
| `workbook is for <date1>, draft is for <date2>` (agenda read) | the next meeting was scaffolded before this one was read back. Move the new draft out of the vault (`mv Agenda/group.draft.yaml ~/`), run `hub agenda read group`, then move it back and build. |
| `meeting note … already exists` (agenda read) | that meeting was already read back. If the note says `status: read-back in progress`, a read-back stopped half-way: check the decisions' entries for that date, finish or undo by hand, delete the note, and re-run |
| `was already read back` | nothing to do |
| `unknown proposal(s): P-…` | the proposal was already applied or rejected (by you or someone else; pull), or the id is mistyped: `hub pending list` |
| `no such item <ID>` | mistyped id, or the item is not pulled yet |
| `nothing written; fix the batch file` (ingest batch) | an agent wrote a malformed batch; it lists every problem and the agent fixes and re-runs |
| `root 'ML_REPO' (…) is not a git repository on this machine` | set that root's path in your `hub.local.yaml` `roots` |
| `warning: cannot load …/Status/Items/<ID>.md: …` (e.g. `missing frontmatter opening '---'`) | that file is broken; every read command warns, `hub show` on it fails, `hub validate` reports `format`. Restore it with `git checkout -- <file>` or fix it. |
| `seen … unreadable:<path>` | a vault file is broken; fix that file (as above) |
| `warning: … N newer commit(s) not listed yet` | a large backlog of commits; the agent runs git ingest again until the warning is gone |
| `clickup: …` flags in the Snapshot | a ClickUp mismatch for a person: a task deleted or moved, done in the hub but open in ClickUp, an assignee who is not in `people` (add their `clickup:` id), or two actions sharing a task. See the [recipes](#recipes). |
### Other situations
- **An agent reports a source as "not reachable".** It says why: the connector is not connected or Claude Code is not logged in to your claude.ai account (Step 7), the Circleback folder is not shared with you, or a repository path is not set in your `hub.local.yaml`.
- **Circleback finds nothing.** Use a `folder:` and file the meetings into it; search terms must appear in the meeting name or notes.
- **No next meeting date.** `calendar_match` does not match the calendar title. Fix it in `project.yaml`, tell the agent the date, or run `hub meeting next group <date>`.
- **The agent says to run `hub-init`.** That skill is not built. Follow Steps 2–5 of this guide.
- **The skills do not trigger.** Start Claude Code from the vault folder, check the links from Step 6, and check that `Status/` exists (the second `hub commit` of Step 5 creates it).
- **Every agent command asks for permission.** See the allow rule in Step 6.
- **Merges suddenly conflict on every item.** The merge drivers are missing or point at a deleted Python: run `hub commit "…"` once in that clone.
- **You edited a `Status/` file by hand and broke it.** `hub validate` names it; `git checkout -- <file>` restores the last committed version.

---

## Known limitations
From `docs/superpowers/plans/2026-11-12-project-hub-skills-ingest-followups.md`, `…-core-followups.md`, and testing of this guide:
- **Rejected titles are never re-staged.** A proposal (or ClickUp task name) whose title was once applied or rejected is skipped forever, even for a different task with the same name.
- Two people processing the **same source in the same few minutes** can still create a question twice and log a line twice (ClickUp tasks are fully guarded).
- A clone where `hub commit` has never run has no merge drivers (and Obsidian mobile never runs them): concurrent edits to one item conflict as text there.
- `hub agenda read` always warns *could not pull before acting* while the edited sheet is uncommitted (expected).
- `hub agenda scaffold` drops decisions past 7 without a message; the agent lists them, by hand check the Decisions sheet.
- Rebuilding a workbook after its read-back makes it look unread again (see [Agenda by hand](#agenda-by-hand)).
- `hub list --series X` and `hub since --series X` leave out items with no series, although those appear on X's agenda. An unknown series name gives an empty list, not an error.
- `hub meeting next` rewrites the whole sync state and can lose a concurrent change to another watermark made at the same moment in the same clone.
- `hub set` does not check `priority` (e.g. `priority=9` is stored) or owners; `hub new --owner` and `hub pending add` do not check owners either (validate reports unknown owners).
- `hub validate` does not check people keys for spaces or series `attendees` against `people`.
- `hub sync-state set` does not check the value's format (a date for `git.<ROOT>` is accepted, and `hub ingest git` then falls back to the last 30 days with a warning).
- ClickUp owner and due mirroring still applies to done and dropped actions; due dates sent as epoch milliseconds are converted in the machine's time zone.
- The meeting note does not record the Circleback meeting id. Circleback meetings outside a configured series get no meeting note (only a log line and items). The note lists configured attendees, not who came.
- `hub seen manual` lists every item ever typed by hand: `manual` is not a useful id to check.

---

## What is not built yet
Designed in the spec but not implemented (Plan 3). Nothing in this guide depends on them.
- **`hub-init`** skill: automatic vault creation (Steps 2–8 above), installing the skills and `hub` inside the vault, registering merge drivers on clone. The hub-status skill may still mention it; follow Steps 2–5 instead.
- **Vault skeleton**: a vault `README.md`, a committed `.obsidian/` with Obsidian Git preconfigured, `.claude/skills/` inside the vault, and running `hub` as `uv run hub`. Today you set these up by hand as in Steps 1–8.
- **`hub-vision`** skill and **`Vision.md`**: the vision document (problem statement, approach, unfair advantage, impact), and the "vision check" proposed after a pivot.
- **`Status/Knowledge/`**: notes of stable findings. No such files exist yet. The skill's rule that changes to Vision.md or Knowledge/ are proposals has nothing to act on.
- **Figures gallery** (`hub-figures`, captions), the **Locator** document and its link checker.
- **CLI commands** `hub figures {sync,add,gallery}`, `hub check-links` and `hub import-handoff` do not exist (`hub --help` lists what does).
- **`import-handoff`**: converting an existing HANDOFF.md into items.
- **Obsidian Bases views** of the items (`Items.base`: tables of open questions, my actions).
- **Meeting note extras**: the Circleback meeting id and the items opened and closed in the meeting. Re-styling of the archived workbook is also not done.

---

## Glossary
| Term | Meaning |
|---|---|
| **vault** | The project's shared folder: an Obsidian vault that is also a git repository. |
| **item** | One question, decision or action: one Markdown file in `Status/Items/`. |
| **people key** | Your short name in `project.yaml` `people` (e.g. `tamra`), used as owner and author everywhere. |
| **`me`** | Your people key, set in your `hub.local.yaml` or with `HUB_ME`. |
| **series** | A kind of recurring meeting (e.g. `group`), with its own agenda workbook. |
| **entry** | One dated, sourced line in an item's history. |
| **source id** | Where a fact came from, e.g. `circleback:abc123` or `manual`. |
| **proposal** (pending) | A staged change that waits for a person's yes/no, in `Status/Pending/`. |
| **apply / reject** | Approve a proposal (its changes are made) / drop it. Both are logged and permanent for that title. |
| **closing the gaps** | After approvals, fixing what `hub validate` reports (missing owner, question, action). |
| **watermark** | How far a source has been read (a date, a time or a commit id). |
| **flag** | A problem for a person to look at, listed in the Snapshot. |
| **Snapshot** | `Status/Snapshot.md`, the generated one-page overview. |
| **draft** | `Agenda/<series>.draft.yaml`: the agent's text for the next sheet. |
| **workbook / sheet** | `Agenda/<series>.xlsx`, the Excel file used in the meeting. |
| **editor** | The one person who types into the sheet during a meeting. |
| **yellow row** | A row on the Decisions sheet for a decision nobody planned. |
| **read-back** | `hub agenda read`: turning the filled-in sheet into hub changes. |
| **dry run** | Ask for it by saying *"dry run: …"*: the agent reads and reports, but writes nothing. |
| **merge driver** | A rule that tells git how to combine two people's versions of a file; registered per clone by `hub commit`. |
| **pull / push / rebase** | Get others' commits / send yours / replay your commits on top of theirs (what `hub commit` does). |
| **upstream** | The GitHub branch your local branch pushes to; set once with `git push -u origin HEAD`. |
| **connector** | Claude's link to an outside service (ClickUp, Circleback, Calendar, …), set up in claude.ai. |
| **skill** | Instructions Claude Code loads for a task (`hub-status`, `hub-agenda`). |
| **ROOT** | A named local path in `roots` (e.g. `ML_REPO`), set per machine. |

---

## Where to look for more
- `hub --help` and `hub <command> --help`: every option.
- `project-hub/.agent_docs/`: exact formats.
  - `item-format.md`: items and validation;
  - `pending-format.md`: proposals;
  - `agenda-workbook.md`: draft and sheets;
  - `ingest.md`: sources, batch files, ClickUp rules;
  - `git-sync.md`: merging and conflicts;
  - `vault-layout.md`: configuration keys.
- `project-hub/skills/hub-status/SKILL.md` and `sources/*.md`, `project-hub/skills/hub-agenda/SKILL.md`: exactly what the agents do.
- `docs/superpowers/specs/2026-11-12-project-hub-design.md`: the design and its reasons (including the parts not built yet).
