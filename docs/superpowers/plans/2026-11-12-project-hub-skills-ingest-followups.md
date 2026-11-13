# Project hub Plan 2 — follow-ups
*From the subagent-driven execution ledger (`.superpowers/sdd/2026-11-12-project-hub-skills-ingest/progress.md`) and the final-review fix wave, 2026-11-12. None block Plan 2 use.*
## Kept deferred at the final review
- Meeting notes (`Status/Meetings/…`) do not record the Circleback meeting id of the cross-check.
- Circleback meetings outside a configured series get no Meetings note (only a log line and items).
- `store.link` crashes on a hand-edited item whose `links` is a list (validate reports `shape`; linking that item fails until it is fixed by hand).
- ClickUp new-task dedup by decided title is coarse (Task 5): two different tasks with the same name, or a task renamed after its proposal was rejected, are not re-proposed.
- `hub meeting next` rewrites the whole sync state (`save_state`), so a concurrent change to another key in the same clone can be lost (Task 6). Unchanged values keep their file and `updated` stamp since the fix wave, which narrows it.
- Commit source ids use git's `%h`, which can exceed 7 characters in large repositories (Task 7).
- Dry-run wording (Task 8): the `mktemp` scratch directory is still created in a dry run (outside the vault); the hub-agenda step 2 STOP is not explicitly dry-run-qualified (it is read-only anyway).
## Residual risks after the fix wave
- Two people ingesting the *same source within the same few minutes* (between one's per-source pull and commit) can still both write it: a safe `new` question is then created twice, and two log lines appear. ClickUp tasks are hard-guarded (`check_ops` refuses a second action for a linked task; `duplicate-clickup-id`); other sources rely on the per-source pull/commit of the hub-status skill.
- The `hub-watermark` driver keeps the side with the newer `updated`, even if its value is behind (e.g. a `git.<ROOT>` sha older than the other side's). This errs towards re-listing already recorded work, which `hub seen` then skips; it never skips unrecorded work.
- A clone that never ran `hub` has no `hub-item` / `hub-watermark` driver configured and falls back to a text merge (conflict) for those files.
- In post-meeting mode, `hub agenda read` always prints "could not pull before acting" because the edited workbook is uncommitted; the skill pulls with `--autostash` just before, so the warning is expected. `pull_if_possible` could autostash itself.
- `hub pending add` does not check owners against `people` (the skills never use it; batch proposals are checked).
## Deferred minors (per task, from the ledger)
- Task 1: `(bob)` with an empty title yields " (raw)"; by_title matches raw multi-line "What to decide" text; planned-row as_decided/wording keep raw newlines.
- Task 2: every `hub seen` call re-reads all logs (a per-run index later).
- Task 3: branch names starting with '-'; an empty repository says "branch 'HEAD' not found"; bare clones are rejected; the first run parses the whole 30-day window before truncating (rev-list later); file names with quotes/backslashes stay git-quoted (use -z later).
- Task 4: the same action recorded under two different sources still duplicates (by design); `flags_added` undercounts merged flags; an empty or newline watermark value is allowed; `series` values are not checked against project.yaml. (Fixed in the fix wave: newline titles — `Store.create` collapses whitespace; `ops: []` proposals — rejected.)
- Task 5: assignee flags and the owner mirror are still applied to done/dropped actions; identical-entry skip on a reopen/close cycle; an all-digit due string is read as epoch ms; epoch → date uses the machine's time zone.
- Task 6: ingest.md rule-table wording gaps (due formats); no CLI test for a missing `me` on writes; a YAML parse error keeps only its first line (no position); ingest.md has an "Exit 0." line above the exit-1 note.
- Task 7: test_skills checks command names only (not options); `hub pending add` owner check (see above).
## From the live dry run (Task 9, 2026-11-13)
- `hub ingest git` could not be exercised from a worktree-isolated Claude session (its sandbox refuses commands naming git outside the worktree); run it from a normal session to confirm against ML-signproblem.
- `test_plan_clickup_flags_shared_task_and_validate_reports_it` failed once in one run; 25/25 isolated and 6/6 full-suite runs pass since — watch for a flake.
- clickup.md step 5 "every linked task must be present" should carve out linked tasks whose `clickup_get_task` errors (step 4 leaves them out on purpose so the plan flags them).
- The post-meeting `git pull --rebase --autostash` has no explicit dry-run skip note (covered by the general dry-run rule).
- With no linked actions, the ClickUp export is open tasks only; with many done linked actions it costs one `clickup_get_task` per task.
- Circleback for this project: terms UCC/HVA matched no meetings in the last 4 weeks — set `sources.circleback.folder` (share the project's meetings into one folder) before relying on it.
- No calendar event matched "Fermion" — set `series.group.calendar_match` to the real meeting title.
