---
name: hub-status
description: Use when asked to update a project hub's status, list someone's todos, or process a meeting that just happened ("update the status", "what are my todos", "we just met", "after the meeting") in a vault that has project.yaml and a Status/ folder.
---
# Hub status
Maintains the shared status store of one project hub vault (an Obsidian vault that is a git repo shared by the whole project). Three modes: **todos**, **update**, **post-meeting**. Every write goes through the `hub` command; never edit files under `Status/` by hand. `hub <command> --help` documents every option.
## Before any mode
1. Work from the vault root (the directory with `project.yaml`). Check `hub.local.yaml` has `me: <your people key>`; if not, ask the user which key in `project.yaml` people is them and add it.
2. Run `git pull --rebase`. If it fails, stop and show the error.
3. Run `hub validate` and note existing violations (do not fix unrelated ones silently).
4. Make a scratch directory **outside the vault**: `SCRATCH=$(mktemp -d)`. Every temporary file (tasks.json, batch files, proposal files) goes there, never inside the vault, because `hub commit` stages everything in the vault. Delete it at the end (`rm -rf "$SCRATCH"`).
5. **Dry run.** If the user says "dry run", run only this allow-list: `hub validate`, `hub list`, `hub show`, `hub since`, `hub seen`, `hub sources`, `hub pending list`, `hub clickup plan`, `hub meeting decided`, `hub sync-state get`, `hub ingest git` without `--advance`, and read-only connector calls (search, read, list, get). Everything else is a write and is skipped, in particular: `hub new|entry|set|link|log`, `hub ingest batch`, `hub clickup apply`, `hub pending add|apply|reject`, `hub agenda read|scaffold|build`, `hub snapshot`, `hub meeting next`, `hub sync-state set`, `hub commit`, editing `hub.local.yaml`, `git fetch`/`git pull`/fast-forwards, and connector write tools (ClickUp create/update). Write the batch files to the scratch directory and show them instead; tell the user which steps were skipped.
## Mode: todos
`hub list --type action --status open --owner me --json` (or `--owner <key>` for someone else). Sort by the `due` key (rows without one last), then by `priority`. Present a short list: title, id, due date, the decision it came from (`hub show <id> --chain` on request). Mention `hub pending list` if proposals are waiting for approval. No writes.
## Mode: update
1. `hub sources` lists what is configured and when each source was last ingested.
2. For each configured source, follow its reference file in `sources/` (next to this file): ClickUp → `sources/clickup.md`, Circleback → `sources/circleback.md`, each repository and Overleaf → `sources/git.md`, calendar → `sources/calendar.md`, opted-in personal notes → `sources/notes.md`, Gmail/Slack → `sources/mail-chat.md`. Each ends with either `hub ingest batch <file>` or `hub clickup apply`. If `hub ingest batch` rejects the file, fix every listed problem and re-run; nothing was written.
3. Classify every fact with this table — the batch format enforces the left column, the right column always goes in `proposals`:

| Apply directly (`safe`) | Stage as a proposal |
|---|---|
| an entry on an existing item, with its source | closing a question (answered/dropped/pivoted) |
| a log line, with its source | recording a decision as made (except via `hub agenda read`) |
| a new open question raised, with its source | a pivot (question replaced by another) |
| a new action from a tracker (ClickUp), mirrored with `hub clickup apply`, or from the agenda read-back (`hub agenda read`) | every action found in notes, Circleback, Gmail or Slack, even with a named owner |
| mirroring ClickUp status/assignee/due (`hub clickup apply`) | any change to Vision.md or Knowledge/ |

4. Before writing anything derived from a source id, `hub seen <source>`; skip what is already recorded unless it adds a new fact. An `unreadable:<path>` entry means a vault file is broken and the answer may be incomplete: report it to the user and do not treat it as "already ingested". If a log cannot be read, `hub seen` exits 1.
5. **Approval batch.** If proposals were staged, list them with `hub pending list`: for each, its proposer, and one line on what applying does. Ask the user once about the proposals staged in this run (apply all, apply some, or reject some); mention proposals staged by other people separately and leave them alone unless the user says otherwise. Then `hub pending apply <P-ids>` / `hub pending reject <P-ids>`.
6. **Validate.** `hub validate`; fix violations you caused (or report them) before going on.
7. **ClickUp push.** If ClickUp is a source: re-run `hub clickup plan --tasks <file>` (the applied proposals may have created owned actions) and do the *Push* step of `sources/clickup.md` for the new `push` entries.
8. `hub snapshot`, then ONE `hub commit "status: ingest <sources> (<YYYY-MM-DD>)"`. Exit code 3 means a real conflict: stop, show the message, and do not resolve it by guessing. Delete the scratch directory.
9. **Report** to the user: a table with one row per source — reached / not reachable (say why: connector missing, folder not shared, path not on this machine) / disabled / not configured — and what changed (entries, new items, flags, proposals applied or rejected). Never present a partial run as complete.
## Mode: post-meeting
1. Identify the series and date (the user, or the series whose `Agenda/<series>.xlsx` has not been read back; ask if ambiguous).
2. `hub agenda read <series>` applies the decisions recorded in the workbook and stages free-text items. If it refuses, show its message (it explains what to do).
3. **Circleback cross-check** (`sources/circleback.md`, section *Cross-check*): compare what was said with `hub meeting decided <series> <date> --json`; anything decided or assigned aloud but not recorded becomes a proposal.
4. Run steps 1–4 of **update** for the other sources (ClickUp especially), not a full update, so approvals and the commit happen once in step 5.
5. Then steps 5-9 of update, in that order: approval batch → `hub validate` → ClickUp push (new actions need tasks, including the ones from the proposals just applied) → one `hub commit` → report.
## Writing entries
One factual sentence each, with numbers when there are numbers ("4×3 open-boundary run finished: F = 0.97 at 1092 parameters"). Name the item the fact serves. Cite the source id in `--source` (formats in each source file).
## Privacy (the vault is read by everyone on the project)
Record decisions, stated positions ("Eun-Ah prefers framing B") and actions with their source. Never record personal assessments of people, career or 1:1 material, health, or anything from someone's private notes verbatim. Ignore the "private notes" field that Circleback's `ReadMeetings` returns. When unsure whether something belongs, leave it out and mention it to the user.
