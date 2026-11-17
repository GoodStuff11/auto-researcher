# Mode B Orchestration

<!-- Part of the hard-problem-brainstorm skill. Read when SKILL.md points here; never needed whole. -->

### Protocol for Mode B (Parallel Fan-Out)

#### 0. Pipelined Orchestration (No Global Barriers)
Mode B is dependency-driven, not phase-driven. Each leaf advances through its own pipeline as soon as its own inputs are ready. The orchestrator never waits for "all workers" (or "all critics", or "all tests") before moving a leaf that is not blocked.
- **Per-leaf state machine.** Track every leaf in the front page's "Current leaf state" table (`brainstorm-<slug>.md`):
  `DISPATCHED → RETURNED → LINTED → CRITIC → OPEN (routes climb T0 → T1 → T2 → T3 independently) ⇄ DORMANT → ANSWERED | PROVEN-IMPOSSIBLE | EXHAUSTED at this scale | EXHAUSTED at this budget` (transitions in section 3b)
  Record the current state, the number of live routes, the routes in flight (job ids), the agents in flight (worker, critic or auditor, with their task), and what, if anything, the leaf is blocked on. This table is what a resumed orchestrator respawns from (3.5.9). Route-level state lives in each leaf's Route Portfolio table (Stage 3.5.2). A leaf with a job in flight is **not** blocked while it has another live route below T3.
- **Act on each arrival immediately.** When any agent or job for leaf $P_i$ finishes, advance $P_i$ in the same turn, without waiting for its siblings:
  1. When a worker returns: lint its artifact (rule 9), and fix it or send it back. Add or update its row in the synthesis matrix. Spawn that leaf's critic right away (see the Evaluation Protocol: one critic per leaf, or per small batch of leaves that return together).
  2. When a critic returns: apply its accepted objections. A route that is killed or downgraded goes straight back to a mutation worker for that leaf (Stage 3.3). A route that survives with a fixable test defect gets a fix worker, or the orchestrator fixes it.
  3. When a test is ready and critic-clean: write its value-of-information note, run `compute_gate.py check`, get the compute auditor's approval if the cap is above 10 CPU-hours, then `compute_gate.py submit` right away (fire-and-forget on Slurm). A refusal is a result: act on the prediction, find a cheaper version, or close the route.
  4. When a result lands: record it, apply the Stage 4 Autopsy & Re-Loop rule for that leaf (route card, Kill Index row, library entry if the reason generalises), and run the cross-pollination check against the leaves still running (step 5).
- **Explicit dependencies are the only barriers.** Keep a `Blocked on` column. A leaf waits only if its premise, instance, metric or baseline depends on a result that is not yet available, for example "the premise of P1.3 depends on the verdict of P1.1". When a result arrives, unblock exactly the leaves that list it. A critical-path test is run first, but it does **not** hold back leaves that do not depend on it. If only part of a leaf depends on a pending result, start the independent part now and record the pending part as a follow-up.
- **Broadcast, don't restart.** When a result, correction or new anti-pattern affects leaves that are still running, send it to those workers as a message (e.g. "the competitor you were told about is in another symmetry sector"). Do not wait for them to finish and then redo the work.
- **Keep agents busy (submit-and-switch, Stage 3.5.3).** Run up to the concurrency limit (the session's workflow-size guideline, or the user's instruction). When an agent finishes and work is queued, start the next queued item (mutation, critic, fix, test build) in the same turn. Every Slurm submission is followed, in the same turn, by dispatching T0 or T1 work on another live route of that leaf or of another leaf. Auditors, critics and pending user approvals are background work too: dispatch them and switch, never wait on them. Idle waiting is acceptable only under the stricter test of Stage 3.5.3, step 5, and then only in the way Stage 3.5.9 describes.
- **Two queues, one scheduler.** Keep a *compute queue* (T2 and T3 jobs, limited by the cluster) and a *thinking queue* (T0 analysis, T1 toy runs, pre-flight dry runs, gating of new routes, Gate 0 checks, critics, decision-tree pre-planning, falsifiers of DORMANT and terminal leaves, the cross-node critic and master synthesis; full list in Stage 3.5.3, step 3). The thinking queue is never allowed to empty while the compute queue is busy, unless every leaf is in a terminal state (ANSWERED, PROVEN-IMPOSSIBLE, EXHAUSTED at this scale or at this budget) or is DORMANT with no open reopening-ledger row that fits T0 or T1. "Every live route is in flight" does not empty the thinking queue (Stage 3.5.3, step 5).
- **Incremental synthesis.** The Master Leaf Synthesis Matrix, the Kill Index, the findings library and the Cross-Pollination section are updated as leaves and results arrive, not compiled once at the end. A final whole-document lint and a cross-node critic pass (Evaluation Protocol) close each round, but they never hold back per-leaf progress.

#### 1. Directory Setup
For each leaf, run `bs.py new-leaf <wd> <id> --title "..." --parent P<n> --question "<its Final Tree question, word for word>"`. This creates `leaves/<id>/` with `summary.md`, `problem.md`, `cards/` and `log/` (layout: ref/records.md, "Documents and context").

#### 2. Worker Subagent Dispatch
For each leaf, spawn a worker with `briefs/worker.md`, filled in with the leaf id, the working directory, the skill directory and the task for this round. The brief tells the worker to load its context with `bs.py context <wd> <leaf> --role worker` and to return at most 15 lines. Never paste SKILL.md or whole files into a worker's prompt.

#### 3. Leaf Files
The files each worker writes, and the Leaf Status Summary rules, are in `ref/records.md` (sections 3 and 3a).

#### 4. The Fan-In Synthesis (Incremental Master Compilation)
As each worker returns (not after all of them), the orchestrator lints its files with `python3 check_brainstorm.py leaves/<id>`. Files with errors are fixed, or sent back to the worker. The orchestrator then reads the worker's report and `bs.py show P<id>` (the summary, not the whole leaf) and adds or updates its row in the **Master Leaf Synthesis Matrix** in `brainstorm-<slug>.md`, together with its pipeline state and `Blocked on` entry (step 0):

| Node | Subproblem Name | Status | Top Surviving Route | Sharpest Gap | Next Decisive Test (tier, CPU-h cap) | Cross-Node Dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P1.1** | ... | SURVIVED | ... | ... | ... | Unblocks P1.2 |
| **P1.2** | ... | SURVIVED (via Mutation) | ... | ... | ... | Unblocks P2.1 |
| **P2.1** | ... | SURVIVED | ... | ... | ... | Independent |

**The master Summary.** The master's `## Summary` answers the problem, in plain words, before it reports anything else. It is kept current with every leaf update and has this order:
1. **The question.** One or two sentences stating what the whole brainstorm asks, in the words of the Final Tree's top-level nodes.
2. **The answers, one per leaf.** For every leaf, in Final Tree order: the leaf's Final Tree question (shortened only by dropping symbols, never by changing what is asked), its state in plain words ("settled", "closed without a final answer because the next test was too expensive", "still open"), and its `Answer` copied from the Leaf Status Summary. The same rules apply as in 3a: lead with the direct reply (Yes, No, Partly, Not known yet, or the value asked for); no route codes, job ids, arm names, verdict labels or undefined acronyms.
3. **What it adds up to.** At most five short bullets answering each top-level node of the Final Tree (P1, P2, ...) from its leaves' answers, in plain words.
4. **Then the details** for specialists: open work, the cross-leaf picture, running jobs, the findings library and the Kill Index. Technical shorthand is allowed here, but each item still says what it means for the answer.

A Summary that leads with state bookkeeping ("after the four gated jobs, ...") or with route names is rewritten. The test is the same as for a leaf: a reader who stops after part 3 knows what the brainstorm found about each question it set out to answer.

#### 5. Cross-Pollination Analysis (Run on Every Arrival)
Each time a leaf returns or a result lands, the orchestrator checks it against every other leaf, whether finished or still running:
- **Mechanism Transfer**: Can a mechanism that survived in $P_1$ resolve the sharpest gap in $P_3$? If $P_3$ is still running, message its worker now.
- **Cascade Refinement**: If a mutation or result in $P_1$ changed the interface to $P_2$ (a premise, instance, metric or baseline), harmonize the boundary conditions. Update `Blocked on` for $P_2$, or broadcast the change if $P_2$ is running.
- **Domino Invalidation**: If a result kills a premise, mark every leaf that depends on it, and send those leaves to mutation right away.
- **Critical Path Recommendation**: Select the highest-leverage test that passes the Compute Value Gate and submit it first, the one that unblocks the most downstream work per CPU-hour. Only the leaves that list it in `Blocked on` wait for it; every other leaf keeps moving.

---
