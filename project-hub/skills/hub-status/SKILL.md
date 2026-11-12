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
4. If the user says **dry run**: never call a write tool of a connector (ClickUp create/update), never run `hub` commands that write (`new`, `entry`, `set`, `link`, `log`, `ingest batch`, `clickup apply`, `pending apply`, `commit`, …). Write batch files to a scratch directory and show them instead. `hub clickup plan`, `hub ingest git` without `--advance`, `hub seen`, `hub sources`, `hub list` and `hub meeting decided` are read-only and allowed.
## Mode: todos
`hub list --type action --status open --owner me` (or `--owner <key>` for someone else). Sort by due date, then priority. Present a short list: title, id, due date, the decision it came from (`hub show <id> --chain` on request). Mention `hub pending list` if proposals are waiting for approval. No writes.
## Mode: update
1. `hub sources` lists what is configured and when each source was last ingested.
2. For each configured source, follow its reference file in `sources/` (next to this file): ClickUp → `sources/clickup.md`, Circleback → `sources/circleback.md`, each repository and Overleaf → `sources/git.md`, calendar → `sources/calendar.md`, opted-in personal notes → `sources/notes.md`, Gmail/Slack → `sources/mail-chat.md`. Each ends with either `hub ingest batch <file>` or `hub clickup apply`. If `hub ingest batch` rejects the file, fix every listed problem and re-run; nothing was written.
3. Classify every fact with this table — the batch format enforces the left column, the right column always goes in `proposals`:

| Apply directly (`safe`) | Stage as a proposal |
|---|---|
| an entry on an existing item, with its source | closing a question (answered/dropped) |
| a log line, with its source | recording a decision as made (except via `hub agenda read`) |
| a new open question raised, with its source | a pivot (question replaced by another) |
| a new action that is unambiguous (named owner, from a tracker or explicit notes) | an action parsed from free text or a transcript |
| mirroring ClickUp status/assignee/due (`hub clickup apply`) | any change to Vision.md or Knowledge/ |

4. Before writing anything derived from a source id, `hub seen <source>`; skip what is already recorded unless it adds a new fact. An `unreadable:<path>` entry means a vault file is broken and the answer may be incomplete: report it to the user and do not treat it as "already ingested". If a log cannot be read, `hub seen` exits 1.
5. When all sources are done: `hub validate`, `hub snapshot`, then `hub commit "status: ingest <sources> (<YYYY-MM-DD>)"`. Exit code 3 means a real conflict: stop, show the message, and do not resolve it by guessing.
6. **Report** to the user: a table with one row per source — reached / not reachable (say why: connector missing, folder not shared, path not on this machine) / disabled / not configured — and what changed (entries, new items, flags). Never present a partial run as complete.
7. **Approval batch:** if proposals were staged, list them (`hub pending list`) with one line each on what applying does, and ask once: apply all, apply some, or reject some. Then `hub pending apply <P-ids>` / `hub pending reject <P-ids>` and `hub commit "status: approvals (<date>)"`.
## Mode: post-meeting
1. Identify the series and date (the user, or the series whose `Agenda/<series>.xlsx` has not been read back; ask if ambiguous).
2. `hub agenda read <series>` applies the decisions recorded in the workbook and stages free-text items. If it refuses, show its message (it explains what to do).
3. **Circleback cross-check** (`sources/circleback.md`, section *Cross-check*): compare what was said with `hub meeting decided <series> <date> --json`; anything decided or assigned aloud but not recorded becomes a proposal.
4. Run **update** for the other sources (ClickUp especially: new actions need tasks — see the *Push* step in `sources/clickup.md`, done after their proposals are applied).
5. Approval batch and commit as in update.
## Writing entries
One factual sentence each, with numbers when there are numbers ("4×3 open-boundary run finished: F = 0.97 at 1092 parameters"). Name the item the fact serves. Cite the source id in `--source` (formats in each source file).
## Privacy (the vault is read by everyone on the project)
Record decisions, stated positions ("Eun-Ah prefers framing B") and actions with their source. Never record personal assessments of people, career or 1:1 material, health, or anything from someone's private notes verbatim. When unsure whether something belongs, leave it out and mention it to the user.
