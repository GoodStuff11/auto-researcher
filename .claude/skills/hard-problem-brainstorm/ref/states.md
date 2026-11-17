# Leaf States

<!-- Part of the hard-problem-brainstorm skill. Read when SKILL.md points here; never needed whole. -->

#### 3b. The Leaf State Machine: Entering and Leaving DORMANT

DORMANT means "no route is currently worth compute, but the leaf is not settled". It is a working state with a fixed to-do list, not a parking place: every DORMANT leaf eventually leaves by one of the exits below, and the exits are checked every time the leaf is touched.

**Entering DORMANT.** A leaf becomes DORMANT when any of the following happens, and the Leaf Status Summary records which under `Entered because`:
* a Saturation Diagnostic is adopted (Stage 3.3, Stage 3.5.3);
* **idea saturation**: the leaf's last two refill batches (at least 3 routes each, from Operation A or B, including at least one cross-field translation that also went through translation repair) were all killed at the gates, T0 or T1. Write the Saturation Diagnostic from their kill reasons (the shared mechanism they hit, and the ingredients they lacked) instead of generating a third batch;
* a closure certificate is attempted but one of its items cannot be filled honestly (3a);
* the compute gate's no-change streak fires (G6, Stage 3.5.8);
* the leaf reaches its CPU-hour budget (G5, Stage 3.5.8).

**The reopening ledger.** The Leaf Status Summary of a DORMANT leaf carries one table that lists everything still owed, with one row per item. Every row ends in one of the outcomes shown:

| Item | Source | Route or test | Outcome |
| :--- | :--- | :--- | :--- |
| each named missing ingredient or untried regime | the Saturation Diagnostic | at least one route, taken through the gates, T0 and T1 | `KILLED(tier)`, `SURVIVED-T1` or `GATE-REFUSED(reason)` |
| each named falsifier | the diagnostic and the critics | the falsifying test | `DIAGNOSTIC-HOLDS`, `DIAGNOSTIC-FALSIFIED` or `GATE-REFUSED(reason)` |
| the step that triggered a G5 or G6 dormancy, or a job awaiting the user's approval above 1000 CPU-hours | the gate | a cheaper version, a reopen justification (G6), or a budget move by the auditor (3.5.8) | `APPROVED`, `SURVIVED-T1`, `KILLED(tier)` or `GATE-REFUSED(budget / auditor / awaiting user)` |

* **Every entry seeds the ledger.** A Saturation Diagnostic seeds one row per named ingredient and falsifier. A G5 or G6 dormancy also seeds a row for the step the gate refused. A leaf can never sit in DORMANT with an empty ledger, so exit 2 is never reached vacuously.
* Rows that fit T0 or T1 are thinking-queue work (Stage 3.5.3, step 3) and run while other leaves compute.
* A falsifier or route step larger than T1 is an ordinary job: a value-of-information note whose `decision` line is "reopen the leaf or not", `compute_gate.py`, and the auditor above 10 CPU-hours. Any refusal is recorded as `GATE-REFUSED` with its reason, including an auditor's REFUSED verdict on a reopen justification; it is an outcome, not an open item.
* For a G5 (budget) row, the agent first tries a cheaper version, then asks the auditor whether to move unspent budget from another leaf within the unchanged total (`set-budget --approved-by`, 3.5.8). Only if both fail does the row become `GATE-REFUSED(budget)`. Raising the total is the user's decision.

**Exits.** Check them in this order whenever a row's outcome changes:
1. **DORMANT → OPEN** when a falsifier returns `DIAGNOSTIC-FALSIFIED`; or a route returns `SURVIVED-T1` and its next step needs no compute or passes the gate (with the auditor's approval where required); or the auditor approves a G6 reopen justification. The leaf reopens with only those routes. The 3-route portfolio refill (3.5.2) resumes only after a reopened route passes T2, so a reopening does not restart the whole portfolio. If every reopening route dies before T2, the leaf returns to DORMANT at once: those rows become `KILLED(tier)`, and the exits are checked again. In the same turn as either transition:
   * rewrite the Leaf Status Summary (`State: OPEN`, `Updated` naming the result, and the diagnostic marked falsified or narrowed);
   * update the master's "Current leaf state" row;
   * add an entry to the Revision & Critic Log in `history.md`.
2. **DORMANT → EXHAUSTED at this scale** when every row is `KILLED` or `DIAGNOSTIC-HOLDS`, with the closure certificate (3a).
3. **DORMANT → EXHAUSTED at this budget** when every row is closed and at least one is `GATE-REFUSED`: a route survived T1, or a falsifier is named, but its next step was refused as unaffordable (G5), not worth its cost (G2, or the auditor), or is still awaiting the user's approval above 1000 CPU-hours. The closure certificate lists each refused step: what it would test, its capped cost, why it was refused, and the budget or new evidence that would reopen it. This is terminal for the agent and is stated in the master Summary without asking the user. It reopens only if the user raises the budget, or if new evidence makes a refused step cheap or decisive.
4. **DORMANT → ANSWERED or PROVEN-IMPOSSIBLE** with the full closure certificate (3a).

**No silent dormancy.** A DORMANT leaf whose reopening ledger has no open row must take one of the exits in the same turn. Before any idle wait (Stage 3.5.3, step 5), each DORMANT leaf gets one line: its open rows, or the exit it just took.
