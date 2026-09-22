# Where things live

For the three of us. Revised 2026-09-15. One page, so nobody has to ask which document is
authoritative. It follows the source ranking already set in `HANDOFF.md`.

## The state-of-project dashboard

Run it with `cd 68f6bef0eaaf5c0928a922c6/dashboard && ./dashboard serve`. It opens at
`127.0.0.1:8765`; under VS Code Remote-SSH the port is forwarded automatically.

Open questions with their staleness, decisions with their grounds, next meeting's agenda, and the
workflow.

It lives inside the Overleaf repo, so `git pull` gives you both the code and the data —
`dashboard/README.md` has the details. It is no longer read-only: changes made in the UI are saved
to a database that travels in git.

There is no shared URL any more, and no rebuild that mints a new one. The old warning about
updating this file after every rebuild, and about unsharing the previous build, no longer applies.

`HANDOFF.md` §7 remains authoritative for what is open and closed. Closing a question in the
dashboard does not edit §7 — it is recorded in `dashboard/data/pending-changes.md` as an edit owed
to §7.

## What is authoritative for what

| Question | Where the answer is |
|---|---|
| What does the code actually do? | `2026/UCC_Hubbard` and Tamra's `ED_UCC_workflow_walkthrough.md` — the technical truth |
| What may we claim? | Figs. 2–4, and `HANDOFF.md` §3 for what each figure does and does not license |
| Which data made a figure? | The certified CSV folder in GitHub (item 20, in progress) |
| Why is the paper shaped this way? | `HANDOFF.md` §5 (novelty), `SKELETON.md` (the arc → paragraph map) |
| What are we arguing? | `main.tex` in Overleaf, at `5f54ce7`, paused until the open items close |
| How did it get here? | `HANDOFF.md` §1.1 (dated log), `DEBRIEF-2026-09-15.md` |
| What is still owed, by whom? | `HANDOFF.md` §7, and the ClickUp list *Fermion Sign Tasks* (`901307841677`) |
| What did we decide, and what is still open? | The dashboard above |
| What was said in a meeting? | Circleback |
| How do I rebuild the dashboard? | `REGENERATE-DASHBOARD.md` |

## Links

| | |
|---|---|
| Overleaf (`main.tex`, `HANDOFF.md`) | *(add URL)* |
| GitHub (`2026/UCC_Hubbard`) | *(add URL)* |
| ClickUp — Fermion Sign Tasks | list `901307841677` |
| Figure set (PowerPoint) | *(add URL)* |
| Google Slides | *(add URL)* |
| Circleback | https://circleback.ai |

## Two conventions

**Item numbers are assigned in `HANDOFF.md` §7 and carried by everything else.** The ClickUp task
and the dashboard cite the number; neither mints its own. An item keeps its number when it closes.
Cite the number, not the description.

**No sentence enters `main.tex` until its source has been read** — a figure, a line of the
walkthrough, or a quoted paper. This applies to all three of us and to anything the AI drafts.

## The loop, as it ran on 2026-09-14

1. **Circleback → `HANDOFF.md`.** The meeting notes go into `HANDOFF.md` in the local Overleaf
   repo, then get pushed to Overleaf. §7 is where an item is first written down and numbered.
2. **`HANDOFF.md` → ClickUp.** Tasks created from §7 unless one already exists, with the item
   number in the title.
3. **`HANDOFF.md` → project knowledge**, so the ledger and the assistant see the same state.
4. **`HANDOFF.md` + ClickUp → the dashboard**, regenerated before the next meeting by sweeping
   ClickUp status and adding to the agenda.

Worth knowing: Circleback's extracted action items are a starting point, not the whole set. On
2026-09-14 they missed the Fig. 2C regeneration entirely and filed item 24 under the wrong person.
Read the transcript alongside them.
