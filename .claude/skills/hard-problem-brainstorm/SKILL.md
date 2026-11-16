---
name: hard-problem-brainstorm
description: >
  Use when a user wants to find, formalize, and tackle high-impact open problems in science, mathematics,
  or computational engineering. Decomposes problems into structural invariants, enforces novelty audits
  against existing literature, executes persistent mutation/replenishment loops until viable routes survive,
  supports parallel subagent fan-out across all leaf subproblems, detects cluster environments to dispatch
  computations via sbatch, and terminates at concrete, 24-hour falsification tests.
---

# Hard Problem Brainstorming Procedure

A structured procedure for finding and attacking high-impact open problems without semantic drift, evaluation collusion, early surrender, or regurgitating known solutions.

Files in this skill's base directory:
- `SKILL.md` (this procedure)
- `check_brainstorm.py` (mechanical syntax, invariant and Markdown-structure linter; accepts files or directories)
- `templates/spec_invariants_template.md` (immutable problem anchor template)

---

## Non-Negotiable Operational Rules

1. **Anti-Drift Anchoring**: Step 0 (Invariants) is immutable once set. No subsequent phase, subagent worker, or critic pass may soften, redefine, or bypass the Hard Core or Unacceptable Outcome.
2. **Persistence Mandate (Zero Early Surrender)**: The agent is strictly forbidden from terminating with "no viable routes found" or giving up when a batch of candidate ideas is killed. Falsification is a diagnostic signal to trigger branching, mutation, or fresh generation, not a stopping condition. Every run must persist until at least 1–2 genuinely viable routes survive all gates or reach a decisive 24-hour test.
3. **The Novelty & Adoption Mandate**: The goal is strictly novel directions. If a proposed idea already exists in the literature:
   - It cannot be presented as a novel solution.
   - You must answer: "Why is this not the universal standard in widespread use?"
   - What hidden bottleneck, scaling limit, instability, or data barrier prevents practitioners from adopting it?
   - The route is disqualified unless the core of the proposal is a concrete mechanism that bypasses that exact adoption bottleneck.
4. **Problem-Specific Anti-Patterns as Mutation Fuel**: Failures are stored locally within the working file (`brainstorm-<slug>.md`). Killed ideas are not dead ends; their specific failure mechanisms form the boundary conditions for the next generation of mutated ideas.
5. **Zero Subjective Scoring**: Never assign numeric ratings, promise multipliers, or arbitrary tractability scores. Ideas survive or die based on explicit binary gates.
6. **Decisive Termination**: For every live route, the next step is the cheapest analytical derivation or numerical micro-step that could kill it (Stage 3.5 tiers). A multi-seed decisive test (T3, at most 24 CPU-hours per task by default) is reserved for routes that already passed the cheaper tiers.
7. **Cluster & Compute Environment Awareness**: Before running any diagnostic scripts or benchmarks, detect if the environment is an HPC cluster (check for `sbatch`, Slurm environment variables, or module systems). If on a cluster, never run heavy workloads interactively or directly on login nodes—generate and submit jobs via Slurm `sbatch` scripts with appropriate partition, walltime, and resource allocations. **Every job must pass the Compute Value Gate (Stage 3.5.6)**: a written value-of-information note, cost tiers with escalating justification, incremental ladders, a compute ledger, and a standing queue review that cancels, without asking, any job that can no longer change a decision. Jobs above 1000 CPU-hours always need the user's approval.
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

Present the leaf summary table to the user and prompt for the routing mode:
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
  3. When a test is ready and critic-clean: submit its job right away (fire-and-forget on Slurm).
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
  > Execute Stage 2 (Barrier Isolation), Stage 3 (Candidate Routes with Gates 0–3, Mutation Loops, and Persistence Rules), and Stage 4 (24-Hour Decisive Test) strictly for this node.
  > You are strictly forbidden from terminating without at least one surviving route. If candidate routes fail, apply the Stage 3.3 Mutation and Replenishment protocols until a viable route survives.
  > If generating or executing code, check if the system is an HPC cluster with Slurm; if so, wrap jobs in `sbatch`.
  > Output your result to `.subproblems/P{i}_{slug}.md` matching the standard Leaf Artifact Schema. Do not soften the frozen invariants.
  > Before returning, run `python3 <skill-dir>/check_brainstorm.py .subproblems/P{i}_{slug}.md` and fix every error and warning (Markdown Hygiene, rule 9). Paste-in logs need a blank line before and after their code fence."

#### 3. Leaf Artifact Schema (Written by Workers to `.subproblems/P{i}_{slug}.md`)
```markdown
# Subproblem: [P{i} Title]
**Parent Chain**: [P{parent} -> P{i}]  **Status**: OPEN | DORMANT | EXHAUSTED at this scale | ANSWERED | PROVEN-IMPOSSIBLE (must match the Leaf Status Summary)

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
* **State**: OPEN | DORMANT | EXHAUSTED at this scale | ANSWERED | PROVEN-IMPOSSIBLE
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

**Leaf truly complete: `State: ANSWERED`, `PROVEN-IMPOSSIBLE` or `EXHAUSTED at this scale`.** These states are allowed only with a **closure certificate** containing all of:
1. `Claim`: the exact statement that is now settled, with its scope.
2. `Decisive evidence`: the results that settle it (job ids or derivations, verdicts under pre-registered rules, and the critic that reviewed them). For PROVEN-IMPOSSIBLE, the proof or its location and the model it holds in.
3. `Why no doubt remains`: every critic objection and confound raised in the leaf, each with how it was resolved, plus the cross-node critic's objections that touch this leaf.
4. `Falsifiers run`: every falsifier named by the leaf's diagnostics or critics, with its result. A named falsifier that has not run blocks these states.
5. `Missing ingredients exhausted` (EXHAUSTED only): every ingredient named by the Saturation Diagnostic, the route generated from it, and the tier at which it was killed. An ingredient without a killed route blocks EXHAUSTED; the leaf stays DORMANT.
6. `What would reopen it`: the specific new evidence, scale or method that would invalidate the claim.

If any certificate item cannot be filled honestly, the leaf is DORMANT, not closed. The per-leaf critic signs off on every certificate before the state changes, and the master's "Current leaf state" table copies each leaf's `State` and `Conclusion` verbatim.

#### 4. The Fan-In Synthesis (Incremental Master Compilation)
As each worker returns (not after all of them), the orchestrator lints that artifact with `python3 check_brainstorm.py .subproblems/P{i}_{slug}.md`. An artifact with Markdown errors is fixed, or sent back to its worker. The orchestrator then reads it and adds or updates its row in the **Master Leaf Synthesis Matrix** in `brainstorm-<slug>.md`, together with its pipeline state and `Blocked on` entry (step 0):

| Node | Subproblem Name | Status | Top Surviving Route | Sharpest Gap | 24-Hour Decisive Micro-Test | Cross-Node Dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P1.1** | ... | SURVIVED | ... | ... | ... | Unblocks P1.2 |
| **P1.2** | ... | SURVIVED (via Mutation) | ... | ... | ... | Unblocks P2.1 |
| **P2.1** | ... | SURVIVED | ... | ... | ... | Independent |

#### 5. Cross-Pollination Analysis (Run on Every Arrival)
Each time a leaf returns or a result lands, the orchestrator checks it against every other leaf, whether finished or still running:
- **Mechanism Transfer**: Can a mechanism that survived in $P_1$ resolve the sharpest gap in $P_3$? If $P_3$ is still running, message its worker now.
- **Cascade Refinement**: If a mutation or result in $P_1$ changed the interface to $P_2$ (a premise, instance, metric or baseline), harmonize the boundary conditions. Update `Blocked on` for $P_2$, or broadcast the change if $P_2$ is running.
- **Domino Invalidation**: If a result kills a premise, mark every leaf that depends on it, and send those leaves to mutation right away.
- **Critical Path Recommendation**: Select the highest-leverage 24-hour test and submit it first, the one that unblocks the most downstream work. Only the leaves that list it in `Blocked on` wait for it; every other leaf keeps moving.

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
| **T2 Pilot** | One seed at the decisive instance size, with checkpointed early-stop rules (3.5.5) | under 2 CPU-hours per task by default | weak effects, wrong regime, premise failures |
| **T3 Decisive** | The multi-seed, pre-registered test of Stage 4, sized by power analysis (3.5.5) | under 24 CPU-hours per task by default | only routes that passed T0–T2 |

* **Each tier has its own pre-registered kill rule**, written before that tier runs. A T0 or T1 kill is as final as a T3 kill when its argument or computation is checked by the per-leaf critic.
* **Budget caps need a written exception.** A T3 task above the cap, or an instance above about $10^6$ states, requires a stated reason why no smaller instance keeps the Hard Core. The default instance is the smallest one that does (e.g. $4\times2$, $3\times3$, $2\times L$ ladders), not the largest one exact diagonalization allows.
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
2. **Submit, then switch immediately.** In the same turn as the `sbatch`, take the next route in the portfolio and advance it one step at the cheapest tier it still needs (T0 analysis or a T1 toy computation).
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
   * **falsifiers of closed leaves.** Every Saturation Diagnostic names the experiment that would falsify it. Run the ones that fit T0 or T1 caps; a falsified diagnostic reopens its leaf with that route only;
   * **routes from Saturation Diagnostics.** A Saturation Diagnostic is not a terminal state: it says which routes failed at this scale and names the missing ingredients. A leaf closed with a diagnostic is **DORMANT**, not finished. Each named missing ingredient and each untried regime is a route source: generate at least one route from each, and take it through the gates, T0 and T1. Only when every named ingredient has a route killed at T0 or T1 (with the kill recorded) is the leaf **EXHAUSTED at this scale**. Report leaves as ANSWERED (a diagnostic question with an answer), PROVEN-IMPOSSIBLE (within a stated model, with the proof), EXHAUSTED at this scale, or DORMANT — never just "closed";
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

Compute is a scarce, shared resource. A job is submitted only if what it can teach is worth its cost. "The rule says to test it" is not a justification; the decision the result changes is.

**Value-of-information note (written in the leaf before every T2 or T3 submission).** It has five lines, and a job without them is not submitted:
1. **Decision.** For each pre-registered verdict, the concrete next action it triggers. If two verdicts trigger the same action, they are one verdict for this purpose.
2. **Prediction.** The outcome expected from the T0 analysis and T1 data, with a rough probability. If one outcome is predicted with probability at least 0.8 *and* that outcome leaves the leaf's decision unchanged, **do not run the job**: record the prediction and act on it, naming the cheap evidence that would overturn it.
3. **Cost.** Expected and capped CPU-hours (CPUs per task × wall time × tasks), and the leaf's running total from the compute ledger.
4. **Cheapest decisive version.** Why no smaller instance, fewer seeds, fewer steps, fewer arms, or a single rung of a ladder decides the same question. Arms that are "reported only" must be justified separately or dropped.
5. **Power.** With the measured run-to-run spread (not the per-run error bar), the probability that the job returns AMBIGUOUS or INCOMPLETE. If it is above about 0.3, redesign before submitting.

**Cost tiers (CPU-hours per job, capped).**

| Tier | Requirement before submission |
| :--- | :--- |
| at most 10 | value-of-information note |
| 10–100 | note, plus the per-leaf critic agrees in writing that the job is the cheapest decisive version |
| 100–1000 | note and critic, plus the route has already passed T2, and the leaf's cumulative spend is stated; ladders and seed sweeps must be incremental (below) |
| above 1000 | **never automatic: ask the user**, with the note, the cost in plain numbers, and the cheaper alternative that was rejected and why |

**Incremental ladders and sweeps.** Size ladders, seed sweeps and multi-arm grids are submitted one rung at a time: the smallest rung that can kill first, each later rung only if the earlier rung leaves the question open. A pre-registered rule may span several rungs, but the compute is released rung by rung. A single large array for a whole ladder is not allowed above 100 CPU-hours.

**Compute ledger.** The master document keeps a table with one row per submitted job: leaf, job id, tier, CPU-hours requested, CPU-hours used, verdict, and whether the verdict changed a decision. Report the running totals per leaf and overall with every synthesis. A leaf whose spend is high and whose verdicts rarely changed a decision is a signal to close it with a Saturation Diagnostic.

**Queue review (cancel without asking).** Every time a result lands, and before any wait, re-check every queued and running job against its value-of-information note. Cancel immediately, without asking, any job whose result can no longer change a decision: its verdict is already decided from data on disk, its leaf is closed, its question has moved to another job, or its outcome is now predicted with probability at least 0.8 by cheaper evidence. Record the cancellation and the reason in the leaf and the ledger. Partially needed array jobs are trimmed to the tasks that still matter (`scancel` by task index).

---

## Stage 4 — The Decisive Test (Tier 3) & Compute Execution

Take a route that has passed T0–T2 (Stage 3.5) and design the minimal decisive calculation that can falsify it. Respect the T3 budget cap, the pre-flight dry run (3.5.4), power sizing (3.5.5) and the Compute Value Gate (3.5.6). A T3 whose outcome the T0 and T1 evidence already predicts with high confidence is not run.

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
       #SBATCH --job-name=micro_test
       #SBATCH --output=micro_test_%j.log
       #SBATCH --error=micro_test_%j.err
       #SBATCH --time=01:00:00
       #SBATCH --nodes=1
       #SBATCH --ntasks=1
       #SBATCH --cpus-per-task=4
       #SBATCH --mem=8G
       # (Include GPU directives #SBATCH --gpus=... only if explicitly required)

       python3 micro_test.py
       ```

     - Submit via `sbatch submit_test.sbatch` and monitor via `squeue -u $USER`.
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
> 1. Verify Persistence: Check if the generator attempted only one round of ideas and stopped. If all ideas were dismissed without mutation or replacement, flag this as a critical failure.
> 2. Challenge Novelty: Identify if this route is an existing published method under new terminology. If it exists, why is it not standard practice? Name the unstated adoption barrier.
> 3. Find the implicit assumption that sweeps the Hard Core under the rug.
> 4. Determine whether the 24-hour test can be trivially satisfied by the Minimal Contrast Foil.
> 5. Verify whether compute scripts conform to the environment (Slurm/sbatch if on a cluster).
> 6. Check whether any proposed step repeats a mechanism logged in the Problem Anti-Patterns section.
> 7. Check tier discipline: could this route have been killed at a cheaper tier (a T0 argument such as a small effective-model lemma, an operation count, or an extensivity argument; or a T1 toy run)? Is a T2 or T3 job requested without a passed T0 and T1, without a pre-flight dry run whose controls behave, without power sizing, or above the budget cap without a written exception?
> 8. Check Gate 4: does the claimed gain survive with system size, or is it a finite-size bonus or an exponential-cost operation in disguise?
> 9. Check compute value (Stage 3.5.6): is every requested job the cheapest decisive version, is its outcome already predictable from cheaper evidence, would any verdict leave the decision unchanged, and is its cost tier justified?
> Return only a numbered list of concrete, fatal objections categorized as Major or Minor. Do not provide praise or vague commentary."

Address all major critiques in place. Log every action in the `Revision & Critic Log`.