---
name: hard-problem-brainstorm
description: >
  Use when a user wants to find, formalize, and tackle high-impact open problems in science, mathematics,
  or computational engineering. Decomposes problems into structural invariants, enforces novelty audits
  against existing literature, persists in generating and mutating ideas (not in spending compute) until
  viable routes survive, supports parallel subagent fan-out across all leaf subproblems, and falsifies
  routes at the cheapest decisive tier. Every cluster job goes through compute_gate.py, which refuses
  jobs whose result is predictable, whose verdict would not change a decision, or that exceed the budget.
  Documents are kept small and read by id (bs.py), so runs of any length fit in context.
---

# Hard Problem Brainstorming Procedure

A procedure for finding and attacking high-impact open problems without semantic drift, evaluation collusion, early surrender, regurgitated known solutions, or wasted compute. This file is the core: the rules in short, the procedure at a glance, and where the detail lives. **Read a reference file only when you reach the step it covers.**

## Files

| File | Read it when |
| :--- | :--- |
| `ref/rules.md` | a rule below needs its full wording |
| `ref/setup.md` | Step 0 (Invariants, budget), Stage 1 (decomposition, Final Tree), Stage 1.5 (routing), Stage 2 (barriers, Analogy Map) |
| `ref/generation.md` | Stage 3: generating routes, Gates 0–4, mutation (Operations A and B), idea saturation |
| `ref/evidence.md` | Stage 3.5.1–3.5.5 and Stage 4: evidence tiers T0–T3, the route portfolio, submit-and-switch, pre-flight dry run, power, early stops, the decisive test |
| `ref/compute.md` | before any job: the Compute Value Gate (3.5.6), the auditor (3.5.7), diminishing returns and calibration (3.5.8), job dispatch |
| `ref/records.md` | writing anything: the document layout and size caps, route cards, kill codes, the Kill Index, the findings library, leaf files, Leaf Status Summary rules (3a) |
| `ref/states.md` | a leaf's state changes: entering and leaving DORMANT, the reopening ledger, exits (3b) |
| `ref/orchestration.md` | Mode B: pipelined orchestration, worker dispatch, fan-in, cross-pollination, the front-page Summary |
| `ref/review.md` | linting and scheduling critics |
| `ref/uptime.md` | waiting, resuming after a context loss, crashes, finishing the run (3.5.9) |
| `briefs/worker.md`, `critic.md`, `auditor.md`, `compactor.md` | spawning that sub-agent: the brief replaces this file for it |
| `bs.py` | reading or creating anything in a run: `state`, `context`, `show`, `grep`, `sizes`, `new-leaf`, `new-card`, `migrate` |
| `compute_gate.py` | every job: `init`, `check`, `submit`, `record`, `refresh`, `cancel`, `reconcile`, `report`, `set-budget`, `digest` |
| `check_brainstorm.py` | after every edit: lints Markdown, record fields, size caps and the Final Tree question |
| `templates/spec_invariants_template.md` | Step 0 |

Each tool documents itself with `--help`. Tests for the tools are in `tests/`.

## Context discipline (runs grow without limit; context does not)

* **Working files are small; archives are only searched.** Each leaf has `summary.md` (2500 tokens), `problem.md` (4000) and one file per route card (900). The front page `brainstorm-<slug>.md` is capped at 5000. History goes to `leaves/<id>/log/` and `history.md`, which are never read whole. Layout and caps: ref/records.md, "Documents and context".
* **Read by id, not by file.** The orchestrator reads `bs.py state` and the front page. A sub-agent's context is the output of `bs.py context <wd> <leaf> --role <worker|critic|compactor>`. Anything else is fetched with `bs.py show <R-id | L-id | P-id | job id>` or found with `bs.py grep`.
* **Sub-agents get a brief, not this file.** Use `briefs/<role>.md`. Workers return at most 15 lines and write everything else to files.
* **Every fact has one home**: current state in `summary.md`, an idea and its kill on its card, a lesson in `library.md`, what happened in a log. Elsewhere, point to it by id.
* **Compact, don't accumulate.** When `bs.py sizes --over` lists a file, or at the end of each round, a compactor (`briefs/compactor.md`) moves history out of working files. The linter warns over cap, and rule 9 allows no warnings.

## Rules (full wording: ref/rules.md)

0. **Compute is spent only when it changes a decision.** Persist in ideas, never in jobs. Every `sbatch` goes through `compute_gate.py submit`. Above 10 CPU-hours an independent auditor approves. The cheapest evidence that could kill a route comes first. The budget is set once, and the auditor may move it between leaves within the total. Only jobs above 1000 CPU-hours and changes to the total go to the user, and neither blocks other work.
1. **Anti-drift.** The Invariants (Hard Core, Unacceptable Outcome, foil, budget) are frozen at Step 0.
2. **No early surrender, and no spending past usefulness.** A batch of kills triggers mutation and new generation. A leaf ends only in a defined state (ref/states.md).
3. **Novelty and adoption.** A known idea must name why it is not standard practice and fix that barrier (Gate 0).
4. **Every idea and every kill is explained, and lessons are kept.** Each route gets a self-contained card: what the idea is; for a kill, its code (`G0`–`G4`, `T0`–`T3`, `CRITIC`), rule, evidence, inference, what it leaves open, and the strongest case against it. Lessons that reach beyond one route go into `library.md`.
5. **No subjective scores.** Binary gates and pre-registered rules decide. The value-of-information note's probability is the only number of belief, and the gate checks its calibration.
6. **Decisive termination.** The next step for a route is the cheapest test that could kill it. Caps are per job in total CPU-hours, never per task.
7. **Cluster awareness.** Never run heavy work on login nodes. Submit only through the gate. The queue review cancels jobs that can no longer change a decision.
8. **Mathematical rigor.** LaTeX for all mathematics; every symbol is defined at first use.
9. **Lint clean.** Every Markdown file passes `check_brainstorm.py <wd>` with zero errors and zero warnings, including record fields and size caps.
10. **Route portfolio, cheapest falsifier first.** Keep several live routes per open leaf. A running job blocks only its own route (submit-and-switch).

## The procedure at a glance

1. **Step 0, Invariants** (ref/setup.md). Freeze the Hard Core, the Unacceptable Outcome, the Minimal Contrast Foil, the existing solutions and the compute budget.
2. **Stage 1, decomposition** (ref/setup.md). Wide fan-out over four dimensions, then a bottleneck audit, down to 3–5 core problems and a Final Tree of leaves. Each leaf's question is written once, there, and every later answer refers to it word for word.
3. **Stage 1.5, routing** (ref/setup.md). Mode A (one leaf) or Mode B (all leaves, in parallel), and the budget if not given, in one question to the user; skip it if the request already says. Then `compute_gate.py init ... --leaves ...` and `bs.py new-leaf` for each leaf.
4. **Stage 2, per leaf** (ref/setup.md). The formulation and barriers, and the **Analogy Map**: each barrier in field-neutral terms, with at least three other fields that faced it, what they did, and the translation gap. It goes into `problem.md`.
5. **Stage 3, routes** (ref/generation.md). Generate after reading the library index and searching the Kill Index. Include a cross-field translation in every batch. Pass every route through Gates 0–4, and give every route a card (ref/records.md). Mutate kills (Operation A, including translation repair), or replace them (Operation B). Two batches killed in full, cross-field routes included, means idea saturation: the leaf goes DORMANT.
6. **Stage 3.5, evidence** (ref/evidence.md). Routes climb T0 (argument), T1 (toy, interactive), T2 (pilot) and T3 (decisive), each tier with a pre-registered kill rule, keeping at least 3 live routes. Submit and switch; pre-flight before any T2 or T3; power both ways; early stops; no T2 whose T3 cannot be afforded.
7. **Every job** (ref/compute.md). Write the value-of-information note, run `compute_gate.py check`, get the auditor above 10 CPU-hours, then `submit`. After each result, run `refresh` and `record`, and update the card, the Kill Index, the library and the summary.
8. **Leaf states** (ref/states.md). OPEN ⇄ DORMANT → ANSWERED, PROVEN-IMPOSSIBLE, EXHAUSTED at this scale, or EXHAUSTED at this budget. Every way into DORMANT seeds a reopening ledger, and the exits are checked whenever a row changes.
9. **Critics** (ref/review.md, `briefs/critic.md`). A per-leaf critic as each leaf returns, and a cross-node critic each round, with the library in its scope. Critics check the kills as well as the survivors, and they check the answers.
10. **Finishing** (ref/uptime.md). When every leaf is terminal, or DORMANT with only user-blocked rows: the final critic pass, the front-page Summary, a lint, and a report. Do not idle, and do not spend leftover budget.

## Every leaf answers its question

Each leaf's `summary.md` opens its Leaf Status Summary with four plain-language fields (rules and an example: ref/records.md, 3a):

* **Question**: the leaf's Final Tree question, word for word (the linter checks it).
* **Answer**: a reply to that question *as asked*: **Yes.**, **No.**, **Partly.** or **Not known.** (or the value, for a "how much" question), then at most two sentences of reason. Yes or No only when the evidence settles the question in the generality in which it is asked. Results that hold only for the cases tried go in the next field, and the answer is then "Not known." or "Partly.".
* **What we showed**: the strongest findings, each with its scope.
* **What it would take**: what would settle the question as asked, or what would overturn a settled answer.

The front page's Summary answers the whole problem first, leaf by leaf, from these fields (ref/orchestration.md).

## The orchestrator's loop (Mode B; ref/orchestration.md and ref/uptime.md)

* **Act on every arrival in the same turn.** A worker report means: lint, update the front page's leaf row, spawn the leaf's critic. A critic report means: apply the objections, mutate or fix. A ready test means: gate, audit, submit. A result means: record it, update the records, check the leaf-state exits, cross-pollinate.
* **Two queues.** The compute queue holds jobs. The thinking queue (T0 and T1 work, gating, critics, audits, Analogy Map extension, pre-planning, compaction) never empties while jobs run. Auditors, critics and user requests run in the background and never block other work.
* **Wait only when the thinking queue is truly empty**, with one written line per item saying why. Wait with a background watcher, never by sleeping in the foreground.
* **Resume from disk**: `compute_gate.py reconcile` and `refresh`, re-arm the watcher, `bs.py state`, respawn the agents the front page shows in flight, and rebuild the thinking queue.
