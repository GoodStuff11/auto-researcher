---
name: hub-agenda
description: Use when asked to prepare, prep or build the agenda for an upcoming meeting of a project hub ("prep the agenda for group", "get ready for Monday's meeting", "build the meeting sheet") in a vault that has project.yaml.
---
# Hub agenda
Prepares `Agenda/<series>.xlsx` for one meeting: Sheet 1 recaps last meeting's decisions and their follow-up, Sheet 2 lists open questions, Sheet 3 holds at most 7 decisions to take. The script owns layout; you write content into `Agenda/<series>.draft.yaml` only. `hub <command> --help` documents every option.
## Before starting
1. Work from the vault root (the directory with `project.yaml`), with `me` set in `hub.local.yaml`. Run `git pull --rebase`; if it fails, stop and show the error.
2. Make a scratch directory **outside the vault**: `SCRATCH=$(mktemp -d)`. Every temporary file goes there, never inside the vault, because `hub commit` stages everything. Delete it at the end (`rm -rf "$SCRATCH"`).
3. **Dry run.** If the user asks for a dry run, do not run `hub agenda scaffold`, `hub agenda build`, `hub new`, `hub set`, `hub snapshot`, `hub meeting next` or `hub commit`. Do the read-only steps, then show the text you would put into the draft (summary, follow-ups, decisions) and say which steps were skipped.
## Steps
1. **Series and date.** From the user; otherwise the Snapshot's *Next meeting*: run `hub snapshot`, then read `Status/Snapshot.md` (or run `sources/calendar.md` of the hub-status skill, which sets `hub meeting next`). Confirm both with the user if they were not stated.
2. **Last meeting read back?** If `Agenda/<series>.xlsx` exists for an earlier date and was not read back, run the hub-status skill's post-meeting mode first (ask the user — it applies that meeting's decisions).
3. **Fresh status.** `hub sources`: if any watermark is older than the last meeting of this series, run the hub-status update mode first.
4. **Scaffold.** `hub agenda scaffold <series> --date <YYYY-MM-DD>`. If it refuses because a draft exists, ask before passing `--force` (it discards the old draft's text).
5. **Fill the draft** (`Agenda/<series>.draft.yaml`). Edit only:
   - `summary` — 3–5 sentences, findings first, framed against last meeting's decisions (what was decided → what happened → what that means now).
   - `since_last[].follow_up` — what was done for each decision, with evidence (commit short shas, figure paths, ClickUp links). Correct `outcome` (closed question / new question / pivot / in progress) only if the store-derived value is wrong, and say why in `follow_up`.
   - `decisions[]` — keep at most 7, ordered by what unblocks most work first. For each: `proposed_action` (concrete, with an owner), `reason` (one sentence, cite evidence), `weigh_in` (people keys whose input is needed), `minutes` (realistic), `link` (figure or document to look at).
   - A decision that is needed but not in the store yet: create it first, then re-scaffold so it gets an id the read-back can match. If no question exists for it, create the question first: `hub new q "<question>" --series <series> --source manual`. Then `hub new d "<what to decide>" --link answers:<question id> --series <series> --source manual` (add `--owner <key>` when someone other than you owns it; the default owner is `me`). Before `hub agenda scaffold <series> --date <date> --force`, copy the draft to a temp file outside the vault (`cp Agenda/<series>.draft.yaml "$SCRATCH/draft.yaml"`), because `--force` discards its text; after re-scaffolding, paste your text back into the new draft.
   - Do not invent questions; do not delete open questions from Sheet 2 (reorder only by editing priorities in the store with `hub set <id> priority=<1|2|3>`).
6. **Build.** `hub agenda build <series>`. If it refuses (previous meeting unread, or the live workbook already holds meeting input), show the message; never pass `--force` without the user's explicit yes.
7. **Show the user** the decisions chosen and why, the total minutes against the meeting length, and open questions left off; invite pruning and rebuild after edits.
8. **Validate and commit.** `hub validate` and report any violations to the user (fix those you caused). Then `hub commit "agenda: <series> <date>"`. Exit code 3 means a real conflict: stop and show the message. Delete the scratch directory.
## Tell the meeting (once, when sharing the file)
One person edits the sheet. In **Outcome** type Approve, Modify, Defer or Reject (Office Viewer in VS Code shows no dropdown — typing works). Put the decision as actually taken in **As decided**; new actions in **New actions** as `title (owner); title (owner)`. Unplanned decisions go in the yellow rows. After the meeting, run the hub-status skill's post-meeting mode.
