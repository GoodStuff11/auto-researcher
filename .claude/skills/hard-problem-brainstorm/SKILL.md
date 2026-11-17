---
name: hard-problem-brainstorm
description: >
  Use when a user wants to find, formalize, and tackle high-impact open problems in science, mathematics,
  or computational engineering. Decomposes problems into structural invariants, enforces novelty audits
  against existing literature, persists in generating and mutating ideas (not in spending compute) until
  viable routes survive, supports parallel subagent fan-out across all leaf subproblems, and falsifies
  routes at the cheapest decisive tier. Every cluster job goes through compute_gate.py, which refuses
  jobs whose result is predictable, whose verdict would not change a decision, or that exceed the budget.
---

# Hard Problem Brainstorming Procedure

A structured procedure for finding and attacking high-impact open problems without semantic drift, evaluation collusion, early surrender, or regurgitating known solutions.

Files in this skill's base directory:
- `SKILL.md` (this procedure)
- `check_brainstorm.py` (mechanical syntax, invariant and Markdown-structure linter; accepts files or directories)
- `compute_gate.py` (the only sanctioned way to submit cluster jobs: enforces the Compute Value Gate, keeps the compute ledger, renders it as Markdown; `python3 compute_gate.py --help`)
- `tests/test_compute_gate.py` (tests for the gate; output in `tests/test_compute_gate_output.txt`)
- `templates/spec_invariants_template.md` (immutable problem anchor template)

---

## Non-Negotiable Operational Rules

0. **Compute Is Spent Only When It Changes a Decision (overrides every rule below).** The goal is the conclusion a careful PI would reach with the least compute, not the most tests run. Persistence applies to *ideas*: generating, mutating, gating and T0/T1 work cost thinking, not cluster time, and they never stop while a leaf is open. Persistence never applies to *jobs*. A job runs only if it passes the Compute Value Gate (Stage 3.5.6), and the agent does not grade its own gate:
   - **Every `sbatch` goes through `python3 <skill-dir>/compute_gate.py submit`**, including resumes, extensions, extra seeds and re-runs. Calling `sbatch` directly is a procedure violation. The tool refuses a job whose value-of-information note is incomplete, whose predicted verdict would not change the decision, whose cost exceeds its tier or the budget, or whose leaf is in a run of jobs that changed nothing.
   - **Above 10 CPU-hours, an independent compute auditor (Stage 3.5.7) must approve the job.** It is a fresh agent with no stake in the route. It plays the role the user would play, so the user is not interrupted.
   - **The cheapest evidence that could kill a route comes first, always.** A T0 argument (scaling, operation count, small lemma) that would have killed a route after a large job has run is the most expensive mistake this procedure can make.
   - **The budget is set once at Step 0.** Leaves draw from it, and a leaf whose spend stops changing decisions is closed as DORMANT, not funded further. Only jobs above 1000 CPU-hours, and changes to the budget, go to the user.
   - **When no job passes the gate, keep thinking, not computing.** Refill routes at T0 and T1, or close the leaf with a Saturation Diagnostic. "Not worth running" is a valid and valuable outcome; record the prediction instead of the job.
1. **Anti-Drift Anchoring**: Step 0 (Invariants) is immutable once set. No subsequent phase, subagent worker, or critic pass may soften, redefine, or bypass the Hard Core or Unacceptable Outcome.
2. **Persistence Mandate in Ideas, Not in Jobs (Zero Early Surrender)**: The agent is strictly forbidden from terminating with "no viable routes found" or giving up when a batch of candidate ideas is killed. Falsification is a diagnostic signal to trigger branching, mutation, or fresh generation, not a stopping condition. Persistence means more routes generated and gated at T0 and T1; it never justifies a job that fails the Compute Value Gate (rule 0). A leaf ends honestly in one of the states of the Leaf Status Summary (ANSWERED, PROVEN-IMPOSSIBLE, EXHAUSTED at this scale, EXHAUSTED at this budget, DORMANT; transitions in Mode B section 3b), never by surrender and never by spending past the point where results change decisions.
3. **The Novelty & Adoption Mandate**: The goal is strictly novel directions. If a proposed idea already exists in the literature:
   - It cannot be presented as a novel solution.
   - You must answer: "Why is this not the universal standard in widespread use?"
   - What hidden bottleneck, scaling limit, instability, or data barrier prevents practitioners from adopting it?
   - The route is disqualified unless the core of the proposal is a concrete mechanism that bypasses that exact adoption bottleneck.
4. **Problem-Specific Anti-Patterns as Mutation Fuel**: Failures are stored locally within the working file (`brainstorm-<slug>.md`). Killed ideas are not dead ends; their specific failure mechanisms form the boundary conditions for the next generation of mutated ideas.
5. **Zero Subjective Scoring**: Never assign numeric ratings, promise multipliers, or arbitrary tractability scores. Ideas survive or die based on explicit binary gates.
6. **Decisive Termination**: For every live route, the next step is the cheapest analytical derivation or numerical micro-step that could kill it (Stage 3.5 tiers). A multi-seed decisive test (T3) is reserved for routes that already passed the cheaper tiers. Caps are per *job* in total CPU-hours (Stage 3.5.6 cost tiers), never per task: an 85-task array spent 8,906 CPU-hours in the first run without breaking any per-task cap.
7. **Cluster & Compute Environment Awareness**: Before running any diagnostic scripts or benchmarks, detect if the environment is an HPC cluster (check for `sbatch`, Slurm environment variables, or module systems). If on a cluster, never run heavy workloads interactively or directly on login nodes. Generate Slurm scripts with a measured wall time, and submit them only through `compute_gate.py submit` (rule 0, Stage 3.5.6). The standing queue review cancels, without asking, any job that can no longer change a decision (`compute_gate.py cancel`).
8. **Mathematical Rigor**: Write all variables, operators, inequalities, and expressions in explicit LaTeX ($...$ inline, $$...$$ display). Every symbol must be defined at first use.
9. **Markdown Hygiene (Every `.md` Must Lint Clean)**: Every Markdown file this procedure writes or edits must pass `check_brainstorm.py` with **zero errors and zero warnings** before it is handed back or synthesized. That includes the master `brainstorm-<slug>.md`, every `.subproblems/P{i}_*.md` leaf artifact, and any brief or note. Lint the whole working directory in one call with `python3 check_brainstorm.py <working-dir>`. The linter detects each file's role (master, leaf or other). Rules that are easy to break when pasting logs or appending sections:
   - Put a blank line before every opening code fence and after every closing fence, even inside a list item. Otherwise Obsidian merges the fence into the preceding list item or paragraph, pairs every later fence off by one, and leaves an extra code block that never closes (M2, M3).
   - Never continue a list with text indented by 4 or more spaces after a code fence or a table. That text renders as a code block (M4). Start a new top-level list instead.
   - Put a blank line before and after every table. Without the blank line after, the next line is swallowed into the table (M5).
   - Close every inline code span and every inline math span on the same line or paragraph (M6, M7).
10. **Route Portfolio, Cheapest Falsifier First (Never Wait on One Route)**: Each leaf keeps a *portfolio* of several live routes, not one "top route". Evidence is gathered in tiers (Stage 3.5): analysis first, then toy computation, then a short pilot, and a multi-seed decisive test last. A running job blocks only the route it tests, never the leaf. The moment a job is submitted, work switches to the next route in the portfolio. While any job runs, every free agent slot is filled with tier-0 or tier-1 work on other routes (rule: *submit-and-switch*).

---

## Output Document Architecture

The procedure supports two execution paths:
- **Mode A (Focused Deep-Dive)**: Writes and edits a single file `brainstorm-<slug>.md`.
- **Mode B (Parallel Subagent Fan-Out)**: Uses `brainstorm-<slug>.md` as the master index and orchestrator, dispatching worker subagents that write into `.subproblems/P{i}_<slug>.md`, then synthesizing them into a master Leaf Comparison Matrix.

---

## Step 0 — Freeze Problem Invariants

Populate the `Invariants` section in `brainstorm-<slug>.md` using `templates/spec_invariants_template.md`. 
Define and freeze:
1. **The Hard Core**: The irreducible mathematical, structural, or physical phenomenon that cannot be approximated or simplified away.
2. **The Unacceptable Outcome**: The exact condition that makes a proposed solution worthless in practice (e.g., bounds energy but misses off-diagonal order; scales exponentially with system volume; requires tuning to measure zero).
3. **The Minimal Contrast Foil**: The simplest existing baseline, status-quo method, or standard ansatz that any credible advance must beat under identical evaluation constraints.
4. **Saturated / Existing Solutions**: List existing methods already known to address parts of the problem, along with why they are insufficient.
5. **Compute Budget**: the total CPU-hours for the whole brainstorm and the default per leaf, set once and recorded in the Invariants. Use the user's figure if they gave one. Otherwise ask once, together with the Stage 1.5 routing question (one interruption, not a recurring checkpoint); if there is still no answer, use 2000 CPU-hours in all and 300 per leaf, and say so in the Summary. Then run `python3 <skill-dir>/compute_gate.py init <working-dir> --slug <slug> --total-cpuh X --per-leaf-cpuh Y`. Only the user changes the budget afterwards.

---

## Stage 1 — Survey & Two-Pass Branch-and-Prune Decomposition

To eliminate stochastic drift and ensure run-to-run reproducibility, problem decomposition must follow a strict two-pass protocol: wide structural branching first, followed by audited bottleneck pruning. Do NOT generate a final 3–5 subproblem list in a single unconstrained pass.

### Pass 1: Domain-Agnostic Structural Fan-Out (Wide Mapping)
Before selecting the top subproblems, map the problem landscape exhaustively across four universal dimensions:
1. **Mathematical & Formal Formulation**: (Ambiguities in what "solved" means, boundary conditions, unproven existence/uniqueness, domain of validity).
2. **Algorithmic & Computational Tractability**: (Asymptotic scaling bottlenecks, hardness bounds, resource ceilings, numerical instability, non-convex landscapes).
3. **Observation, Measurement & Validation**: (Ground truth unobservability, gauge dependencies, signal-to-noise ratios, sample complexity, proxy gaming).
4. **Asymptotic & Regime Transition**: (Thermodynamic limits, finite-size effects, breakdown of approximations across parameter regimes).

Generate an unpruned candidate pool of $\ge 8\text{--}12$ candidate subproblems across these universal dimensions based on field-backed evidence (review articles, roadmaps, workshop reports, stated open questions in seminal papers).

### Pass 2: Bottleneck Audit & Dependency Pruning
Subject every candidate subproblem to a 3-point centrality audit to filter the candidate pool down to the 3–5 most critical obstacles:
1. **The Blocking Test**: Does this obstacle directly prevent progress on other candidate branches? (If resolving it unblocks multiple downstream branches, it is a primary bottleneck. If resolving it leaves the main barrier untouched, it is secondary).
2. **Root Cause vs. Symptom**: Is this subproblem a fundamental root cause, or merely a downstream consequence of another obstacle in the tree? Merge symptoms into their root causes.
3. **Field Consensus Evidence**: Verify that the obstacle is explicitly named by leading literature as a binding constraint. Mark any unverified or inferred importance as `[unverified]`.

### Final Tree Structure
Assemble the surviving 3–5 core problems and recurse into their subproblems until reaching actionable leaf nodes. Every node carries:
- **Statement**: One sentence.
- **Why the field cares**: What capability unlocks if solved, and what this node blocks in its parent.
- **Evidence of importance**: Specific citations, reviews, or open-question declarations.
- **Current direction and why**: The dominant attack vector and the specific reason the field is moving there.
- **Feasibility**: Tractable, Open and contested, or Believed infeasible (must cite the exact no-go theorem or complexity class).

Stop recursing when a node reaches a concrete question that a single researcher can compute or prove. Mark as **[leaf]**.

---

## Stage 1.5 — Execution Routing: Single vs. Parallel Fan-Out

Present the leaf summary table to the user and prompt for the routing mode, and in the same prompt for the compute budget if Step 0 did not get one (Step 0, item 5). This is the only routine question to the user; after it, the procedure runs on its own, and the compute auditor (Stage 3.5.7) stands in for the user on spending decisions up to 1000 CPU-hours per job.
- **Option 1 (Mode A: Focused Deep-Dive)**: User selects 1 leaf node. Continue synchronously through Stages 2–4 in `brainstorm-<slug>.md`.
- **Option 2 (Mode B: Parallel Fan-Out)**: Tackle all leaf subproblems simultaneously via worker subagents.

---

### Protocol for Mode B (Parallel Fan-Out)

#### 0. Pipelined Orchestration (No Global Barriers)
Mode B is dependency-driven, not phase-driven. Each leaf advances through its own pipeline as soon as its own inputs are ready. The orchestrator never waits for "all workers" (or "all critics", or "all tests") before moving a leaf that is not blocked.
- **Per-leaf state machine.** Track every leaf in a status table at the top of `## Master Leaf Synthesis Matrix`:
  `DISPATCHED → RETURNED → LINTED → CRITIC → PORTFOLIO-ACTIVE → (routes climb T0 → T1 → T2 → T3 independently) → CLOSED`
  Record the current state, the number of live routes, the routes in flight (job ids), and what, if anything, the leaf is blocked on. Route-level state lives in each leaf's Route Portfolio table (Stage 3.5.2). A leaf with a job in flight is **not** blocked while it has another live route below T3.
- **Act on each arrival immediately.** When any agent or job for leaf $P_i$ finishes, advance $P_i$ in the same turn, without waiting for its siblings:
  1. When a worker returns: lint its artifact (rule 9), and fix it or send it back. Add or update its row in the synthesis matrix. Spawn that leaf's critic right away (see the Evaluation Protocol: one critic per leaf, or per small batch of leaves that return together).
  2. When a critic returns: apply its accepted objections. A route that is killed or downgraded goes straight back to a mutation worker for that leaf (Stage 3.3). A route that survives with a fixable test defect gets a fix worker, or the orchestrator fixes it.
  3. When a test is ready and critic-clean: write its value-of-information note, run `compute_gate.py check`, get the compute auditor's approval if the cap is above 10 CPU-hours, then `compute_gate.py submit` right away (fire-and-forget on Slurm). A refusal is a result: act on the prediction, find a cheaper version, or close the route.
  4. When a result lands: record it, apply the Stage 4 Autopsy & Re-Loop rule for that leaf, and run the cross-pollination check against the leaves still running (step 5).
- **Explicit dependencies are the only barriers.** Keep a `Blocked on` column. A leaf waits only if its premise, instance, metric or baseline depends on a result that is not yet available, for example "the premise of P1.3 depends on the verdict of P1.1". When a result arrives, unblock exactly the leaves that list it. A critical-path test is run first, but it does **not** hold back leaves that do not depend on it. If only part of a leaf depends on a pending result, start the independent part now and record the pending part as a follow-up.
- **Broadcast, don't restart.** When a result, correction or new anti-pattern affects leaves that are still running, send it to those workers as a message (e.g. "the competitor you were told about is in another symmetry sector"). Do not wait for them to finish and then redo the work.
- **Keep agents busy (submit-and-switch, Stage 3.5.3).** Run up to the concurrency limit (the session's workflow-size guideline, or the user's instruction). When an agent finishes and work is queued, start the next queued item (mutation, critic, fix, test build) in the same turn. Every Slurm submission is followed, in the same turn, by dispatching T0 or T1 work on another live route of that leaf or of another leaf. Idle waiting is acceptable only when every live route of every leaf is in flight or dependency-blocked (Stage 3.5.3, item 5). In that case wait for notifications.
- **Two queues, one scheduler.** Keep a *compute queue* (T2 and T3 jobs, limited by the cluster) and a *thinking queue* (T0 analysis, T1 toy runs, pre-flight dry runs, gating of new routes, Gate 0 checks, critics, decision-tree pre-planning, falsifiers of closed leaves, the cross-node critic and master synthesis; full list in Stage 3.5.3, step 3). The thinking queue is never allowed to empty while the compute queue is busy, unless every leaf is closed *and* every closed leaf's cheap falsifiers have run. "Every live route is in flight" does not empty the thinking queue (Stage 3.5.3, step 5).
- **Incremental synthesis.** The Master Leaf Synthesis Matrix and the Cross-Pollination section are updated row by row as leaves arrive, not compiled once at the end. A final whole-document lint and a cross-node critic pass (Evaluation Protocol) close each round, but they never hold back per-leaf progress.

#### 1. Directory Setup
Create `.subproblems/` inside the working directory.

#### 2. Worker Subagent Dispatch
For each surviving leaf node $P_i$, spawn an isolated subagent. 
Provide each worker with:
- The frozen `Invariants` section from Step 0.
- The specific node definition (statement, unlock, parent blockage, cited obstacles).
- The prompt mandate:
  > "You are an expert computational and theoretical researcher analyzing subproblem $P_i$. 
  > Execute Stage 2 (Barrier Isolation), Stage 3 (Candidate Routes with Gates 0–4, Mutation Loops, and Persistence Rules) and Stage 3.5 (Evidence Tiers) strictly for this node.
  > Persist in ideas, not in jobs (rule 0). If candidate routes fail, apply the Stage 3.3 Mutation and Replenishment protocols at T0 and T1. Try the cheapest argument that could kill each route before proposing any computation.
  > Do not submit Slurm jobs yourself. For each test you propose, write its script with a measured wall time and a value-of-information note `tests/<dir>/voi.md` in the format of `compute_gate.py --help`, and run `python3 <skill-dir>/compute_gate.py check <working-dir> <script> --voi <voi.md>`. Report the check output; the orchestrator handles auditing and submission. A job the gate refuses is replaced by its recorded prediction or a cheaper version, not argued past the gate.
  > Output your result to `.subproblems/P{i}_{slug}.md` matching the standard Leaf Artifact Schema. Do not soften the frozen invariants.
  > Before returning, run `python3 <skill-dir>/check_brainstorm.py .subproblems/P{i}_{slug}.md` and fix every error and warning (Markdown Hygiene, rule 9). Paste-in logs need a blank line before and after their code fence."

#### 3. Leaf Artifact Schema (Written by Workers to `.subproblems/P{i}_{slug}.md`)
```markdown
# Subproblem: [P{i} Title]
**Parent Chain**: [P{parent} -> P{i}]  **Status**: OPEN | DORMANT | EXHAUSTED at this scale | EXHAUSTED at this budget | ANSWERED | PROVEN-IMPOSSIBLE (must match the Leaf Status Summary)

### 1. Mathematical Abstraction & Barriers
* **Formulation**: Explicit inputs, outputs, and what "solved" means.
* **The Node Barrier**: Where existing methods halt against the Hard Core.

### 2. Candidate Routes, Gates & Mutation Cycles
* **Generation Batch 1**:
  - Route 1.0: [Mechanism] -> Killed at Gate X [Reason]
  - Route 1.1 (Mutated from 1.0): [Modified Mechanism addressing Gate X] -> SURVIVED
* **Generation Batch 2 (if needed)**: ...

### 3. Route Portfolio & Sharpest Gaps
| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| R1 | T1 | IN-FLIGHT(job id) | — | ... | PASS → ...; FAIL → ... |

* For every live route: **Mechanism**, **Sharpest Gap** (the missing lemma, unproven scaling, or unverified assumption), and its **Gate 4 claim** (cost versus $N$; extensive gain or finite-size bonus).
* **Tier log**: each T0 argument, T1 toy result and pre-flight dry-run output, with its kill rule and verdict.

### 4. Tier-3 Decisive Test & Compute Execution (only for routes that passed T0–T2)
* **Minimal Instance**: Smallest non-trivial model retaining the Hard Core.
* **Scalar Metric $M$**: Exact mathematical definition.
* **Execution Plan**: Shell command or Slurm job script (`run_test.sbatch`) if on an HPC cluster.
* **Thresholds**: PASS (value), FAIL (value), AMBIGUOUS (action).

### 5. Local Autopsy Log
* Log of all killed parent ideas and why they were mutated or discarded.

## Leaf Status Summary
* **State**: OPEN | DORMANT | EXHAUSTED at this scale | EXHAUSTED at this budget | ANSWERED | PROVEN-IMPOSSIBLE
* **Updated**: YYYY-MM-DD (after which result)
* **Conclusion**: two or three sentences a reader can trust without reading the leaf.
* (state-specific fields, see "Leaf Status Summary rules" below)
```

#### 3a. Leaf Status Summary Rules (Mandatory, Always the Last Section)

Every leaf artifact ends with `## Leaf Status Summary`. It is the **last** heading of the file: rounds, results and notes are inserted *above* it, never after it, and the summary is rewritten (not appended to) whenever the leaf's state changes or a result lands. It is written for a reader who reads nothing else in the leaf.

**Self-contained argument (the standard every summary must meet).** After reading only the summary, the user must be able to explain the idea or the closure to a colleague and defend it against obvious objections. Field lists are not enough. The summary therefore contains, in prose (about 400–900 words; longer only if the argument needs it):
* **The question in plain terms**: what problem this leaf attacks and why it matters for NQS of fermions (or the domain at hand), with every symbol and acronym defined at first use. No route labels ("7.1", "R2", "frdK") without saying what they are.
* **The idea or mechanism**: what the approach does and *why it should work*, explained physically or mathematically, not by name.
* **The chain of evidence**: premise → test → result → inference, step by step. For each step: what was compared with what (the arm and its foil), on which instance and size, the key numbers side by side with their uncertainties, and why that comparison is fair (matched parameters, matched wall time, pre-registered rule).
* **Why the alternatives fail or are worse**: the strongest competing routes or baselines and the evidence against them.
* **The strongest objection and the answer to it**: the best counterargument a skeptical referee would raise (from the critics, the cross-node critic, or a confound found later), and how the evidence answers it, or an honest statement that it is not answered.
* **Limits**: what the conclusion does not cover (sizes, instances, sectors, optimisation regime, sampled versus exact arithmetic).

Write it so that each claim is traceable: cite the section or job id where the number comes from in parentheses, but make the argument readable without following those pointers.

**Common fields (all states):** `State`, `Updated` (date and the result that triggered the update), `Conclusion`, `Scope` (instances, sizes, sectors and compute regime the conclusion covers), and `Compute spent` (CPU-hours, from the master compute ledger).

**Leaf still live: `State: OPEN` or `DORMANT`.** Add:
* `Most promising direction`: the single best live route, with its mechanism, the strongest evidence for it so far and the evidence against it.
* `Next step`: the concrete next test (tier, cost, pre-registered rule in one line), or the job in flight and what each verdict will trigger.
* `What would close it`: the result that would move the leaf to EXHAUSTED, ANSWERED or PROVEN-IMPOSSIBLE.
* DORMANT only: `Entered because` (which trigger of section 3b) and the `Reopening ledger` table of section 3b.

**Leaf truly complete: `State: ANSWERED`, `PROVEN-IMPOSSIBLE`, `EXHAUSTED at this scale` or `EXHAUSTED at this budget`.** These states are allowed only with a **closure certificate** containing all of:
1. `Claim`: the exact statement that is now settled, with its scope.
2. `Decisive evidence`: the results that settle it (job ids or derivations, verdicts under pre-registered rules, and the critic that reviewed them). For PROVEN-IMPOSSIBLE, the proof or its location and the model it holds in.
3. `Why no doubt remains`: every critic objection and confound raised in the leaf, each with how it was resolved, plus the cross-node critic's objections that touch this leaf.
4. `Falsifiers run`: every falsifier named by the leaf's diagnostics or critics, with its result. A named falsifier that has not run blocks these states, except EXHAUSTED at this budget, where a falsifier refused by the compute gate is listed with its refusal (section 3b, exit 3).
5. `Missing ingredients exhausted` (both EXHAUSTED states): every ingredient named by the Saturation Diagnostic, the route generated from it, and the tier at which it was killed, or, for EXHAUSTED at this budget only, the gate refusal that stopped it. An ingredient with neither blocks EXHAUSTED; the leaf stays DORMANT.
6. `What would reopen it`: the specific new evidence, scale or method that would invalidate the claim. For EXHAUSTED at this budget, also each refused step's capped cost and the budget that would fund it.

If any certificate item cannot be filled honestly, the leaf is DORMANT, not closed. The per-leaf critic signs off on every certificate before the state changes, and the master's "Current leaf state" table copies each leaf's `State` and `Conclusion` verbatim.

#### 3b. The Leaf State Machine: Entering and Leaving DORMANT

DORMANT means "no route is currently worth compute, but the leaf is not settled". It is a working state with a fixed to-do list, not a parking place: every DORMANT leaf eventually leaves by one of the exits below, and the exits are checked every time the leaf is touched.

**Entering DORMANT.** A leaf becomes DORMANT when any of the following happens, and the Leaf Status Summary records which under `Entered because`:
* a Saturation Diagnostic is adopted (Stage 3.3, Stage 3.5.3);
* a closure certificate is attempted but one of its items cannot be filled honestly (3a);
* the compute gate's no-change streak fires (G6, Stage 3.5.8);
* the leaf reaches its CPU-hour budget (G5, Stage 3.5.8).

**The reopening ledger.** The Leaf Status Summary of a DORMANT leaf carries one table that lists everything still owed, with one row per item. Every row ends in one of the outcomes shown:

| Item | Source | Route or test | Outcome |
| :--- | :--- | :--- | :--- |
| each named missing ingredient or untried regime | the Saturation Diagnostic | at least one route, taken through the gates, T0 and T1 | `KILLED(tier)`, `SURVIVED-T1` or `GATE-REFUSED(reason)` |
| each named falsifier | the diagnostic and the critics | the falsifying test | `DIAGNOSTIC-HOLDS`, `DIAGNOSTIC-FALSIFIED` or `GATE-REFUSED(reason)` |
| the reopen justification (G6 dormancy only) | a new ingredient, a different question, or a cheaper decisive test | auditor review (3.5.7) | `APPROVED` or `REFUSED` |

* Rows that fit T0 or T1 are thinking-queue work (Stage 3.5.3, step 3) and run while other leaves compute.
* A falsifier or route step larger than T1 is an ordinary job: a value-of-information note whose `decision` line is "reopen the leaf or not", `compute_gate.py`, and the auditor above 10 CPU-hours. A refusal by the gate or the auditor is recorded as `GATE-REFUSED` with its reason; it is an outcome, not an open item.
* A budget (G5) dormancy has no exit through the agent other than a cheaper version of the refused step. Raising a leaf's budget is the user's decision.

**Exits.** Check them in this order whenever a row's outcome changes:
1. **DORMANT → OPEN** when a falsifier returns `DIAGNOSTIC-FALSIFIED`; or a route returns `SURVIVED-T1` and its next step needs no compute or passes the gate (with the auditor's approval where required); or the auditor approves a G6 reopen justification. The leaf reopens with only those routes. The 3-route portfolio refill (3.5.2) resumes only after a reopened route passes T2, so a reopening does not restart the whole portfolio. In the same turn:
   * rewrite the Leaf Status Summary (`State: OPEN`, `Updated` naming the result, and the diagnostic marked falsified or narrowed);
   * update the master's "Current leaf state" row;
   * add an entry to the Revision & Critic Log.
2. **DORMANT → EXHAUSTED at this scale** when every row is `KILLED` or `DIAGNOSTIC-HOLDS`, with the closure certificate (3a).
3. **DORMANT → EXHAUSTED at this budget** when every row is closed and at least one is `GATE-REFUSED`: a route survived T1, or a falsifier is named, but its next step was refused as unaffordable (G5) or not worth its cost (G2, or the auditor). The closure certificate lists each refused step: what it would test, its capped cost, why it was refused, and the budget or new evidence that would reopen it. This is terminal for the agent and is stated in the master Summary without asking the user. It reopens only if the user raises the budget, or if new evidence makes a refused step cheap or decisive.
4. **DORMANT → ANSWERED or PROVEN-IMPOSSIBLE** with the full closure certificate (3a).

**No silent dormancy.** A DORMANT leaf whose reopening ledger has no open row must take one of the exits in the same turn. Before any idle wait (Stage 3.5.3, step 5), each DORMANT leaf gets one line: its open rows, or the exit it just took.

#### 4. The Fan-In Synthesis (Incremental Master Compilation)
As each worker returns (not after all of them), the orchestrator lints that artifact with `python3 check_brainstorm.py .subproblems/P{i}_{slug}.md`. An artifact with Markdown errors is fixed, or sent back to its worker. The orchestrator then reads it and adds or updates its row in the **Master Leaf Synthesis Matrix** in `brainstorm-<slug>.md`, together with its pipeline state and `Blocked on` entry (step 0):

| Node | Subproblem Name | Status | Top Surviving Route | Sharpest Gap | Next Decisive Test (tier, CPU-h cap) | Cross-Node Dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P1.1** | ... | SURVIVED | ... | ... | ... | Unblocks P1.2 |
| **P1.2** | ... | SURVIVED (via Mutation) | ... | ... | ... | Unblocks P2.1 |
| **P2.1** | ... | SURVIVED | ... | ... | ... | Independent |

#### 5. Cross-Pollination Analysis (Run on Every Arrival)
Each time a leaf returns or a result lands, the orchestrator checks it against every other leaf, whether finished or still running:
- **Mechanism Transfer**: Can a mechanism that survived in $P_1$ resolve the sharpest gap in $P_3$? If $P_3$ is still running, message its worker now.
- **Cascade Refinement**: If a mutation or result in $P_1$ changed the interface to $P_2$ (a premise, instance, metric or baseline), harmonize the boundary conditions. Update `Blocked on` for $P_2$, or broadcast the change if $P_2$ is running.
- **Domino Invalidation**: If a result kills a premise, mark every leaf that depends on it, and send those leaves to mutation right away.
- **Critical Path Recommendation**: Select the highest-leverage test that passes the Compute Value Gate and submit it first, the one that unblocks the most downstream work per CPU-hour. Only the leaves that list it in `Blocked on` wait for it; every other leaf keeps moving.

---

## Stage 2 — Abstraction & Barrier Mapping (Mode A Execution)

1. **Mathematical Abstraction**: Formulate the chosen problem into explicit mathematical inputs, outputs, and exact criteria for what counts as solved.
2. **Barrier Isolation**:
   - For every major existing approach, identify the exact point where it halts against the Hard Core.
   - Separate **fundamental barriers** (theorems, complexity limits, symmetries) from **practical barriers** (representation bottlenecks, optimization landscapes).
   - Formulate the **Anti-Solution Space**: Explicitly list what an acceptable solution cannot look like based on established no-go results.

---

## Stage 3 — Dynamic Generation, Falsification & Continuous Mutation

**Core Objective**: Identify and refine candidate routes until at least 1–2 robust proposals pass all elimination gates. You must not surrender if initial ideas fail.

### 3.1 Initial Generation Batch
Generate an initial pool of 3–4 candidate routes. Ingest the local "Problem Anti-Patterns & Killed Routes" section first to avoid re-treading dead ends.

### 3.2 Binary Falsification Gates
Subject each candidate route to five binary elimination gates:

* **Gate 0: Novelty & Adoption Reality Check**
  - Is this approach already known or essentially identical to an existing baseline?
  - If known: Why is it not in widespread use today?
    - If the field tried it and moved on, what was the fatal flaw (e.g., sample complexity, sign instability, optimization traps)?
    - Does the proposal present a concrete, novel mechanism to fix that exact flaw?
    - If no novel mechanism is proposed to fix the known adoption barrier, the route is FAIL (Not Novel / Stale Baseline).

* **Gate 1: Conservation & Asymptotic Consistency**
  - Does the route violate known conservation laws, gauge symmetries, or asymptotic complexity bounds?
  - Does it claim poly-time convergence over an NP-hard or QMA-hard space without proving a restrictive structural property of the instance distribution?
  - Verdict: PASS / FAIL.

* **Gate 2: Baseline Gameability**
  - Could an uninformative, cheap baseline (e.g., mean-field state, product state, random phase sampling, greedy lookup) achieve the target metric within stated error bars?
  - If the target metric cannot distinguish the proposed mechanism from the Minimal Contrast Foil, it is gamed.
  - Verdict: PASS / FAIL.

* **Gate 3: The Hard-Core Shield**
  - Does the mechanism secretly simplify away the Hard Core identified in Step 0 (e.g., vanishing sign problem only in bipartite lattices; bounded entanglement only in 1D; non-interacting limits)?
  - Verdict: PASS / FAIL.

* **Gate 4: Scaling & Relevance**
  - State the asymptotic claim explicitly: the cost per amplitude, sample or step as a function of system size $N$, and whether the claimed gain should hold *per site* (extensive) or only as a finite-size bonus.
  - Is there a standard argument that the gain vanishes with $N$ (e.g., a fixed number of mean-field components captures non-extensive correlation; a symmetry restoration is a finite-size effect; an exact-sum or full-vector operation is exponential)?
  - If the claim survives on paper, it must be checked at tier 1 on at least three toy sizes *before* any tier-3 investment (Stage 3.5). A method whose benefit exists only on small clusters fails here, not after the decisive test.
  - Verdict: PASS / FAIL.

---

### 3.3 The Closed-Loop Mutation & Replenishment Engine

If a route fails a gate, or if **ALL** candidate routes in a batch fail, do **NOT** stop or report failure. Immediately execute the following branching operations:

#### Operation A: Direct Route Mutation (Branching from Failure)
Do not throw away the whole idea if only one part triggered a gate failure. Mutate the route by specifically modifying the sub-component that failed:
1. **If killed by Gate 0 (Not Novel / Adoption Wall)**:
   - Isolate the exact adoption wall (e.g., "requires storing $\mathcal{O}(N!)$ basis states" or "gradients vanish exponentially").
   - Branch Route $X \to X'$: Keep the core physical intuition, but swap the representation or optimization engine to one specifically designed to bypass that adoption wall.
2. **If killed by Gate 1 (Asymptotic / Conservation Violation)**:
   - Branch Route $X \to X'$: Enforce the violated conservation law explicitly into the ansatz by construction (e.g., project into the physical gauge subspace, reparameterize via Lie algebra generators) and recalculate the scaling.
3. **If killed by Gate 2 (Gameable by Trivial Baselines)**:
   - Branch Route $X \to X'$: Identify what non-trivial higher-order correlation, non-local entanglement, or non-linear response separates the true solution from the foil. Retarget the route's objective specifically to that higher-order witness.
4. **If killed by Gate 3 (Evaded the Hard Core)**:
   - Branch Route $X \to X'$: Introduce a perturbation parameter $\lambda \in [0, 1]$ connecting the tractable simplified regime to the full Hard Core. Formulate the route as an analytical continuation or controlled adiabatic deformation starting from the known limit.

#### Operation B: Orthogonal Replenishment (New Seed Generation)
If an idea is completely irrecoverable (e.g., fundamentally violates a no-go theorem), log it to `## Problem Anti-Patterns & Killed Routes` and generate a fresh replacement route using one of three forced orthogonal directions:
1. **Duality / Frame Transformation**: If current ideas operate in real space, formulate the candidate in dual momentum, graph, or phase space.
2. **Active / Interactive Probing**: If passive representations hit complexity walls, formulate a route based on active, adaptive query algorithms or localized projective measurements.
3. **Cross-Field Translation**: Port a concrete mathematical technique from an adjacent field that conquered an analogous barrier, modifying it to respect the Hard Core.

**Termination Condition for Stage 3**: Continue the generate $\to$ gate $\to$ mutate loop until **at least 3 distinct candidate routes survive all 5 gates** (the starting Route Portfolio of Stage 3.5). Generation does not stop when the first route survives: the portfolio is refilled whenever it drops below 3 live routes. A Saturation Diagnostic does not end this loop: it moves the leaf to DORMANT, and its named missing ingredients become route sources (Stage 3.5.3, step 3).

For each surviving route, state the **Sharpest Gap**: the missing lemma, unproven scaling behavior, or unverified physical assumption required for the route to function.

---

## Stage 3.5 — Evidence Tiers & the Route Portfolio (Submit-and-Switch)

**Core Objective**: Kill weak routes in minutes, not days, and never let one long job idle a leaf. Every gate-surviving route climbs a ladder of increasingly expensive evidence. A route is promoted only after it passes the cheaper tier, and while it waits on a job, the other routes keep moving.

### 3.5.1 The Evidence Tiers

| Tier | What it is | Budget (default cap) | Typical kills |
| :--- | :--- | :--- | :--- |
| **T0 Analysis** | Lemmas, counting and dimension arguments, small linear-algebra models (e.g. a $2\times2$ effective model), operation counts, asymptotic cost and extensivity estimates, known results | minutes of agent time, no job | mechanisms impossible in principle; cost models that cannot win; Gate 4 failures |
| **T1 Toy computation** | Exact computation on a toy instance that keeps the Hard Core (Hilbert dimension up to about $10^4$), plus the mandatory pre-flight dry run (3.5.4) | under 10 CPU-minutes; a short interactive allocation or a tiny job | structural failures, invalid instances, broken controls, the three-size Gate 4 trend |
| **T2 Pilot** | One seed at the decisive instance size, with checkpointed early-stop rules (3.5.5) | under 10 CPU-hours per job by default | weak effects, wrong regime, premise failures |
| **T3 Decisive** | The pre-registered test of Stage 4, sized by power analysis (3.5.5), seeds released one at a time | per job, by the cost tiers of 3.5.6 | only routes that passed T0–T2 |

* **Each tier has its own pre-registered kill rule**, written before that tier runs. A T0 or T1 kill is as final as a T3 kill when its argument or computation is checked by the per-leaf critic.
* **Budget caps need a written exception.** A job above its tier (3.5.6), or an instance above about $10^6$ states, requires a stated reason why no smaller instance keeps the Hard Core. The default instance is the smallest one that does (e.g. $4\times2$, $3\times3$, $2\times L$ ladders), not the largest one exact diagonalization allows.
* **A premise shared by several routes is tested once, first.** If routes $X$, $Y$, $Z$ all assume the same premise (an instance has a target, a regime exists, a reference is valid), test that premise at the cheapest tier that decides it, and let its result prune all three.

### 3.5.2 The Route Portfolio (Per Leaf)

Each leaf maintains a portfolio table (in its artifact, section "Route Portfolio") with one row per live route:

| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |

* **Route states**: `GATED → T0 → T1 → T2 → T3 → VERDICT`, plus `IN-FLIGHT(job id)` while a job runs and `KILLED(tier, reason)`.
* **Portfolio size.** Keep at least 3 live routes per leaf whenever the leaf is not yet closed. When the count drops below 3, refill immediately with Operation A (mutation of the latest kill) or Operation B (orthogonal replenishment) from Stage 3.3, gated as usual.
* **Breadth before depth.** All live routes are taken through T0 and T1 together, in parallel. At most one or two routes per leaf are promoted to T3 at the same time, chosen by these ordering rules (binary comparisons, no scores, rule 5):
  1. a route that has passed a higher tier goes before one that has not;
  2. among equals, a route whose test would decide a premise shared by other routes goes first;
  3. among equals, the route with the cheaper next tier goes first.
* **A route in flight does not block its leaf.** A leaf is blocked only when *every* live route is `IN-FLIGHT` or waiting on a stated cross-leaf dependency.

### 3.5.3 Submit-and-Switch Scheduling

The orchestrator (and any worker that owns a leaf) follows this loop. It replaces "design one test, submit, wait":

1. **Before submitting a T2 or T3 job**, write the decision tree: for each pre-registered verdict (PASS, FAIL, AMBIGUOUS, INCOMPLETE), the next action and, where it is a test, its design. When the result lands, the next step starts in the same turn.
2. **Submit, then switch immediately.** In the same turn as the `compute_gate.py submit`, take the next route in the portfolio and advance it one step at the cheapest tier it still needs (T0 analysis or a T1 toy computation).
3. **Fill every idle slot with non-compute work**, in this order:
   * T0 analysis of live routes that have not passed T0 (proofs of the sharpest gap, Gate 4 cost and extensivity arguments);
   * T1 toy computations and pre-flight dry runs for routes about to be promoted;
   * generation and gating of replacement routes when a portfolio is below 3;
   * deeper Gate 0 literature checks on routes approaching T3;
   * per-leaf critics on routes that have not yet been tested;
   * decision-tree pre-planning for jobs in flight;
   * **refill when every live route of a leaf is in flight.** For scheduling, a leaf whose live routes are all `IN-FLIGHT` has an empty portfolio: generate and gate at least one new route (Operation A on the latest kill, or Operation B), and take it through T0 and T1 while the jobs run;
   * **untested variants and merged routes** that a critic kept alive but nobody has run (e.g. a variant merged into another route);
   * **open gaps and caveats named by critics or results** that a cheap T0 or T1 check can settle (e.g. "the sampled force is untested", "the foil may be confounded", "the toy did not test X");
   * **falsifiers of closed leaves.** Every Saturation Diagnostic names the experiment that would falsify it. Run the ones that fit T0 or T1 caps; larger ones go through the compute gate as ordinary jobs. Record each outcome in the leaf's reopening ledger; a falsified diagnostic reopens its leaf with that route only (Mode B section 3b, exit 1);
   * **routes from Saturation Diagnostics.** A Saturation Diagnostic is not a terminal state: it says which routes failed at this scale and names the missing ingredients. A leaf closed with a diagnostic is **DORMANT**, not finished. Each named missing ingredient and each untried regime is a route source: generate at least one route from each, and take it through the gates, T0 and T1, recording each in the reopening ledger. The leaf then leaves DORMANT by the exits of Mode B section 3b: back to OPEN if a route survives and its next step passes the gate, EXHAUSTED at this scale if every ingredient's route was killed, or EXHAUSTED at this budget if a surviving route's next step was refused by the gate. Report leaves as ANSWERED (a diagnostic question with an answer), PROVEN-IMPOSSIBLE (within a stated model, with the proof), EXHAUSTED at this scale, EXHAUSTED at this budget, or DORMANT — never just "closed";
   * **cross-node critic and master synthesis.** Run the cross-node critic whenever two or more leaves have changed status (closed, promoted, killed) since the last one, and keep the master Summary and Cross-Pollination current with every closure;
   * **cleanup of the compute queue.** Cancel any queued or running job whose verdict can no longer change a decision (e.g. a pre-registered rule that is already decided from data on disk), and record why (Stage 3.5.6 queue review; no confirmation needed).
4. **On every job checkpoint or completion**, apply the early-stop rule (3.5.5) or the verdict rule, update the portfolio row, and launch the pre-planned next action.
5. **Idle waiting is legitimate only when every item of the list in step 3 is empty**, not merely when every live route is `IN-FLIGHT` or dependency-blocked. Before waiting, write one line per item stating why it is empty (for example "P2.1 closed; its compiled-update falsifier already ran"). If any line cannot be written, that item is the next action. A long wait is the most common way this procedure stalls: treat "all routes are in flight" as a prompt to check step 3, never as permission to stop.

### 3.5.4 Mandatory Pre-Flight Dry Run (Before Any T2 or T3 Submission)

Run the complete pipeline (train, aggregate, verdict) on a toy instance with a deliberately short budget, and record its output in the leaf artifact:

* the instance parameters pass explicit validity asserts (e.g. an active-space size not exceeding the orbital count; the target symmetry sector exists and is the one the reference uses);
* the positive control returns its intended verdict, and the negative control returns its intended verdict, *in every case the rule will see* (a positive control must be a perfect partner regardless of which branch the trial state is in);
* the aggregate script parses the outputs and prints a verdict from synthetic results that cover every rule branch;
* the reference method (ED, DMRG, ...) matches an independent check on the toy instance in the same sector the arms use.

### 3.5.5 Decisive by Design: Power Sizing and Early Stops

* **Power sizing.** From the T2 pilot's seed-to-seed and sampling spread, choose seeds, samples and steps so that the PASS and FAIL thresholds are separated by at least $2\sigma$ of the scored metric. A T3 test that is likely to return AMBIGUOUS is redesigned before submission, not resolved afterwards with "add more seeds".
* **Checkpointed early stops.** T2 and T3 jobs write checkpoints and per-checkpoint metrics. The pre-registered rule includes futility and early-kill clauses at fixed fractions of the budget (e.g. 25% and 50%): if the metric at a checkpoint is already outside the PASS region by more than its projected remaining improvement (measured on the pilot's trajectory), cancel the job and record the verdict at that checkpoint.
* **Anytime reporting.** Partial checkpoint metrics are read and logged when they appear. They may *kill* early under the pre-registered rule, but never *promote* early.

### 3.5.6 Compute Value Gate (Every Job Must Earn Its Cost)

Compute is a scarce, shared resource. A job is submitted only if what it can teach is worth its cost. "The rule says to test it" is not a justification; the decision the result changes is. The gate is enforced by `compute_gate.py`, not by the agent's own judgement: every submission, resume, extension or re-run goes through `compute_gate.py submit`, which refuses on any failed rule (G1–G7 in its `--help`).

**The waste this gate exists to stop.** In the first run of this procedure about half of 28,000 CPU-hours went to jobs whose verdict changed no decision. The patterns were:
* a large array run before the T0 argument that later killed the route anyway (an 8,900 CPU-hour job returned AMBIGUOUS, then a scaling argument killed it for free);
* runs that could not converge in their budget, resumed three times and each time ended UNCLEAR;
* extra seeds after the verdict was already decided from stored seeds;
* overpowered tests: 18 tasks to measure an effect $10^5$ times the threshold, where one would have decided it;
* repeat pilots that returned the same answer, and wall-time misses followed by resumes.

**Value-of-information note (`tests/<dir>/voi.md`, linked from the leaf, written before every job of any tier).** One `key: value` line per field; `compute_gate.py --help` gives the exact keys. A job without every field is refused (G1):
1. **Decision.** For each pre-registered verdict, the concrete next action it triggers. If two verdicts trigger the same action, they are one verdict for this purpose.
2. **Prediction and p.** The verdict expected from the T0 analysis and T1 data, with its probability, and whether that predicted verdict changes the decision. If $p\ge0.8$ and it does not, the gate refuses the job (G2): record the prediction, act on it, and name the cheap evidence that would overturn it. The threshold moves with the agent's track record (3.5.8).
3. **Cheapest kill tried.** The T0 argument most likely to kill the route (operation count, scaling or extensivity argument, small effective-model lemma, known result) and its outcome. "None exists" must say why.
4. **Cost.** The capped CPU-hours (array tasks × tasks × CPUs × wall time). The gate computes it from the `#SBATCH` lines and refuses if it exceeds the declared figure (G3), the tier (G4), or the leaf or brainstorm budget (G5).
5. **Cheapest decisive version.** Why no smaller instance, fewer seeds, fewer steps, fewer arms, or a single rung decides the same question. Arms that are "reported only" are justified separately or dropped.
6. **Convergence.** Evidence from a pilot trajectory that the run ends converged within its budget. A design that can end "unconverged" or "unclear" is redesigned, not run. A resume is a new submission with its own note.
7. **Wall-time basis.** Measured time per step × steps, plus about 30%. Guessed wall times cause timeouts and resumes.
8. **Power, both ways.** With the measured run-to-run spread (not the per-run error bar), the probability that the job returns AMBIGUOUS or INCOMPLETE; above about 0.3, redesign. And the opposite check: if the expected margin is many spreads beyond the threshold, one seed on one instance decides it, so request one.

**Cost tiers (CPU-hours per job, capped; enforced by G4).**

| Tier | Requirement before submission |
| :--- | :--- |
| at most 10 | value-of-information note |
| 10–100 | note, plus `auditor: APPROVED <date>` from the independent compute auditor (3.5.7) |
| 100–1000 | note and auditor, plus `passed-t2:` naming the T2 the route already passed; the leaf's cumulative spend is checked against its budget |
| above 1000 | **never automatic: ask the user**, with the note, the cost in plain numbers, and the cheaper alternative that was rejected and why; record `user-approval:` |

**Release compute one unit at a time.** Seeds, instances, arms and ladder rungs are submitted sequentially at every cost tier: the unit that can kill first, each later unit only if the earlier ones leave the question open. A pre-registered rule may span several units, but compute is released unit by unit, and the rule is applied after each. A single large array for a whole ladder or seed sweep is not allowed above 100 CPU-hours.

**Compute ledger.** `compute/ledger.tsv` is written by the tool, one row per job, matched to `sacct` by a unique job name (`hpb-<slug>-<leaf>-<route>-<NNN>`) so reused job ids are never miscounted. After each result, run `compute_gate.py refresh` and `compute_gate.py record <job> --verdict ... --changed yes|no --hit yes|no|na`. The master document's `## Compute Ledger` section is the output of `compute_gate.py report`, pasted in with every synthesis.

**Queue review (cancel without asking).** Every time a result lands, and before any wait, re-check every queued and running job against its value-of-information note. Cancel immediately with `compute_gate.py cancel <job> --reason ...`, without asking, any job whose result can no longer change a decision: its verdict is already decided from data on disk, its leaf is closed, its question has moved to another job, or its outcome is now predicted with probability at least the skip threshold by cheaper evidence. Partially needed array jobs are trimmed to the tasks that still matter (`scancel` by task index, then record it).

### 3.5.7 The Compute Auditor (Stands In for the User)

The agent that designed a route wants to see it tested; it should not also judge whether the test is worth its cost. For every job above 10 CPU-hours, spawn a **fresh** agent with no access to the route's history beyond what is listed here. Give it only the value-of-information note, the `#SBATCH` script, the output of `compute_gate.py check` and `compute_gate.py report`, and the leaf's T0 and T1 evidence for this route. Its mandate:

> "You are the PI who pays for this compute and wants the right conclusion at the least cost. Approve this job only if all of the following hold, and otherwise say which fails and what to do instead (act on the prediction, a cheaper version, a T0 argument to try first, or close the route):
> 1. Its verdicts lead to different next actions, and the predicted verdict is not already near-certain from the T0 and T1 evidence.
> 2. The cheapest argument that could kill the route has been tried, and no T0 or T1 step could decide the question instead.
> 3. It is the smallest version that decides the question: one instance, one seed, one rung, the fewest arms, the shortest run. Nothing is overpowered.
> 4. The pilot evidence shows it will converge within its wall time, and the wall time is measured, not guessed.
> 5. The leaf's recent jobs have been changing decisions; if not, this job has a concrete reason to break the pattern.
> 6. Spending this on this leaf is a better use of the remaining budget than any other open leaf's next step.
> Answer APPROVED or REFUSED, with one sentence per failed item. Be stingy: a refusal costs minutes, a wasted job costs hours."

On APPROVED, write `auditor: APPROVED <date>` into the note; the auditor's reasons go in the leaf next to the note. On REFUSED, do what it says. Do not re-ask a fresh auditor with the same note; change the design first.

### 3.5.8 Diminishing Returns and Prediction Calibration

* **No-change streak (G6).** When a leaf's last 3 finished jobs all changed no decision, the gate refuses further jobs for that leaf. The leaf moves to DORMANT with a Saturation Diagnostic unless a `reopen-justification` (a new ingredient, a different question, a cheaper decisive test) is approved by the auditor. Thinking-queue work on the leaf continues, and it leaves DORMANT only by the exits of Mode B section 3b. The streak resets once a reopened job changes a decision.
* **Leaf budget (G5).** A leaf that reaches its budget is closed or made cheaper, not topped up; only the user moves budget between leaves or raises the total.
* **Calibration.** Every job's prediction and probability go in the ledger, and `record --hit` scores them. When at least 5 predictions with $p\ge0.8$ are scored, the gate adjusts the skip threshold of G2: to 0.7 if at least 90% hit (the agent's confident predictions are reliable, so skip more jobs), and to 0.9 if fewer than 75% hit (they are not, so test more). `compute_gate.py report` prints the current calibration.
* **Report.** With every synthesis, read the report's flags (streaks, wall-time misses, calibration, budget left) and act on each before the next submission.

---

## Stage 4 — The Decisive Test (Tier 3) & Compute Execution

Take a route that has passed T0–T2 (Stage 3.5) and design the minimal decisive calculation that can falsify it. Respect the pre-flight dry run (3.5.4), power sizing (3.5.5), the Compute Value Gate (3.5.6) and the compute auditor (3.5.7). A T3 whose outcome the T0 and T1 evidence already predicts with high confidence is not run; the prediction is recorded instead.

### 1. Test Specification
1. **Target Instance**: The minimal non-trivial instance preserving the Hard Core (e.g., 2x2 cluster, N=2 particle sector, smallest non-Abelian group).
2. **Scalar Metric $M$**: A single mathematically defined quantity to compute or derive.
3. **Falsification Threshold**:
   - **PASS**: $M$ satisfies the theoretical bound; proceed to full-scale derivation.
   - **FAIL**: $M$ violates the bound or matches the Minimal Contrast Foil; route is killed.
   - **AMBIGUOUS**: Noise or finite-size effects dominate; state the pre-committed resolution action.

### 2. Compute Environment Check & Job Dispatch
Before running any code or generating test scripts:
1. **Detect Cluster Environment**:
   - Check if `which sbatch` succeeds or if environment variables such as `SLURM_JOB_ID`, `SLURM_CLUSTER_NAME`, or cluster module commands (`module avail`) exist.
2. **Local vs. Cluster Execution**:
   - **Local Machine**: Generate a direct runnable Python/Julia/C++ script (e.g., `python3 micro_test.py`) with an execution timeout guard.
   - **HPC Cluster**: **NEVER run non-trivial computations on login nodes.** Always generate:
     - The calculation code: `micro_test.py`
     - A submission script: `submit_test.sbatch` specifying:

       ```bash
       #!/bin/bash
       # --job-name is set by compute_gate.py (unique per job); do not rely on it here
       #SBATCH --output=micro_test_%x_%A_%a.log
       #SBATCH --error=micro_test_%x_%A_%a.err
       #SBATCH --time=01:00:00   # measured time per step x steps x 1.3 (wall-time-basis in voi.md)
       #SBATCH --nodes=1
       #SBATCH --ntasks=1
       #SBATCH --cpus-per-task=4
       #SBATCH --mem=8G
       # (Include GPU directives #SBATCH --gpus=... only if explicitly required)

       python3 micro_test.py
       ```

     - A value-of-information note `voi.md` next to the script (3.5.6).
     - Check with `python3 <skill-dir>/compute_gate.py check <working-dir> submit_test.sbatch --voi voi.md`, get the auditor's approval above 10 CPU-hours (3.5.7), then submit with `compute_gate.py submit` (same arguments). Never call `sbatch` directly. Monitor via `squeue -u $USER`; after it finishes, `compute_gate.py refresh` and `compute_gate.py record`.
3. **Autopsy & Re-Loop Rule**: If the computation yields a **FAIL**, log the failure mechanism into `## Problem Anti-Patterns & Killed Routes`, mark the route `KILLED` in the leaf's Route Portfolio, launch the pre-planned FAIL action from its decision tree, and refill the portfolio (Stage 3.3) if it drops below 3 live routes. The other live routes were already advancing while this job ran (Stage 3.5.3), so a kill never restarts the leaf from scratch. Do not stop.

---

## Evaluation Protocol (Adversarial Critic Loop)

After drafting, lint every Markdown file in the working directory (master, leaf artifacts and notes):

```bash
python3 check_brainstorm.py <working-dir>
```

Resolve all syntax and formatting failures flagged by the linter until the summary line reports `0 with errors` and no file prints a warning. Re-run it after every edit pass, including after the critic loop and after appending test results. A single file can still be linted with `python3 check_brainstorm.py <file>.md`.

**Scheduling (Mode B)**: critics are pipelined too (Mode B step 0).
- Spawn a **per-leaf critic** as soon as that leaf's artifact returns and lints clean, or a critic for a small batch of leaves that return together. Do not wait for the whole fan-out.
- Point the per-leaf critic at that leaf's artifact and test code, plus the master Invariants and Anti-Patterns. Have it read the test code itself, not only the markdown.
- Each critic's objections go straight to that leaf's fix or mutation step.
- A single **cross-node critic** closes each round, once the last leaf of the round has passed its own critic. It checks only claims that span leaves: shared baselines, shared instances, corrections that must propagate, and global anti-patterns. It never holds back per-leaf progress.

Spawn a fresh adversarial critic agent with this mandate:
> "Act as a hostile referee whose sole objective is to break the surviving routes in this document. 
> 1. Verify Persistence in ideas: Check if the generator attempted only one round of ideas and stopped. If all ideas were dismissed without mutation or replacement at T0 and T1, flag this as a critical failure. Do not demand more compute as a remedy: persistence is about routes, not jobs.
> 2. Challenge Novelty: Identify if this route is an existing published method under new terminology. If it exists, why is it not standard practice? Name the unstated adoption barrier.
> 3. Find the implicit assumption that sweeps the Hard Core under the rug.
> 4. Determine whether the decisive test can be trivially satisfied by the Minimal Contrast Foil.
> 5. Verify whether compute scripts conform to the environment (Slurm on a cluster) and were submitted only through `compute_gate.py` (every job id appears in `compute/ledger.tsv`).
> 6. Check whether any proposed step repeats a mechanism logged in the Problem Anti-Patterns section.
> 7. Check tier discipline: could this route have been killed at a cheaper tier (a T0 argument such as a small effective-model lemma, an operation count, or an extensivity argument; or a T1 toy run)? Is a T2 or T3 job requested without a passed T0 and T1, without a pre-flight dry run whose controls behave, without power sizing, or above the budget cap without a written exception?
> 8. Check Gate 4: does the claimed gain survive with system size, or is it a finite-size bonus or an exponential-cost operation in disguise?
> 9. Check compute value (Stage 3.5.6–3.5.8): is every requested job the cheapest decisive version (one seed, one rung, released sequentially, not overpowered), is its outcome already predictable from cheaper evidence, would any verdict leave the decision unchanged, was a T0 kill argument tried first, will it converge within a measured wall time, and has the leaf's recent spend been changing decisions? Name every past job in the ledger whose verdict a cheaper argument could have given.
> Return only a numbered list of concrete, fatal objections categorized as Major or Minor. Do not provide praise or vague commentary."

Address all major critiques in place. Log every action in the `Revision & Critic Log`.