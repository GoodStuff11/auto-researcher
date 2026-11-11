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
6. **Decisive Termination**: The terminal output is the single smallest analytical derivation or numerical micro-step capable of killing the route within 24 hours.
7. **Cluster & Compute Environment Awareness**: Before running any diagnostic scripts or benchmarks, detect if the environment is an HPC cluster (check for `sbatch`, Slurm environment variables, or module systems). If on a cluster, never run heavy workloads interactively or directly on login nodes—generate and submit jobs via Slurm `sbatch` scripts with appropriate partition, walltime, and resource allocations.
8. **Mathematical Rigor**: Write all variables, operators, inequalities, and expressions in explicit LaTeX ($...$ inline, $$...$$ display). Every symbol must be defined at first use.
9. **Markdown Hygiene (Every `.md` Must Lint Clean)**: Every Markdown file this procedure writes or edits must pass `check_brainstorm.py` with **zero errors and zero warnings** before it is handed back or synthesized. That includes the master `brainstorm-<slug>.md`, every `.subproblems/P{i}_*.md` leaf artifact, and any brief or note. Lint the whole working directory in one call with `python3 check_brainstorm.py <working-dir>`. The linter detects each file's role (master, leaf or other). Rules that are easy to break when pasting logs or appending sections:
   - Put a blank line before every opening code fence and after every closing fence, even inside a list item. Otherwise Obsidian merges the fence into the preceding list item or paragraph, pairs every later fence off by one, and leaves an extra code block that never closes (M2, M3).
   - Never continue a list with text indented by 4 or more spaces after a code fence or a table. That text renders as a code block (M4). Start a new top-level list instead.
   - Put a blank line before and after every table. Without the blank line after, the next line is swallowed into the table (M5).
   - Close every inline code span and every inline math span on the same line or paragraph (M6, M7).

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
**Parent Chain**: [P{parent} -> P{i}]  **Status**: Completed | Under Mutation

### 1. Mathematical Abstraction & Barriers
* **Formulation**: Explicit inputs, outputs, and what "solved" means.
* **The Node Barrier**: Where existing methods halt against the Hard Core.

### 2. Candidate Routes, Gates & Mutation Cycles
* **Generation Batch 1**:
  - Route 1.0: [Mechanism] -> Killed at Gate X [Reason]
  - Route 1.1 (Mutated from 1.0): [Modified Mechanism addressing Gate X] -> SURVIVED
* **Generation Batch 2 (if needed)**: ...

### 3. Top Surviving Route & Sharpest Gap
* **Mechanism**: ...
* **Sharpest Gap**: The missing lemma, unproven scaling, or unverified physical assumption.

### 4. 24-Hour Decisive Test & Compute Execution
* **Minimal Instance**: Smallest non-trivial model retaining the Hard Core.
* **Scalar Metric $M$**: Exact mathematical definition.
* **Execution Plan**: Shell command or Slurm job script (`run_test.sbatch`) if on an HPC cluster.
* **Thresholds**: PASS (value), FAIL (value), AMBIGUOUS (action).

### 5. Local Autopsy Log
* Log of all killed parent ideas and why they were mutated or discarded.
```

#### 4. The Fan-In Synthesis (Master Compilation)
Once all subagent workers complete, the orchestrator first runs `python3 check_brainstorm.py <working-dir>`. Any leaf artifact with Markdown errors is fixed, or sent back to its worker, before synthesis. The orchestrator then reads each `.subproblems/P{i}_{slug}.md` and generates the **Master Leaf Synthesis Matrix** in `brainstorm-<slug>.md`:

| Node | Subproblem Name | Status | Top Surviving Route | Sharpest Gap | 24-Hour Decisive Micro-Test | Cross-Node Dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P1.1** | ... | SURVIVED | ... | ... | ... | Unblocks P1.2 |
| **P1.2** | ... | SURVIVED (via Mutation) | ... | ... | ... | Unblocks P2.1 |
| **P2.1** | ... | SURVIVED | ... | ... | ... | Independent |

#### 5. Cross-Pollination Analysis
The orchestrator checks for global interactions across the subproblems:
- **Mechanism Transfer**: Can a mechanism that survived in $P_1$ resolve the sharpest gap in $P_3$?
- **Cascade Refinement**: If a mutation in $P_1$ altered the interface to $P_2$, harmonize the boundary conditions.
- **Critical Path Recommendation**: Select the single highest-leverage 24-hour test to execute first (the one that unblocks the most downstream work).

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
Subject each candidate route to four binary elimination gates:

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

**Termination Condition for Stage 3**: Continue the generate $\to$ gate $\to$ mutate loop until **at least 1–2 distinct, fully qualified candidate routes survive all 4 gates**.

For each surviving route, state the **Sharpest Gap**: the missing lemma, unproven scaling behavior, or unverified physical assumption required for the route to function.

---

## Stage 4 — The 24-Hour Decisive Test & Compute Execution

Take the top surviving route and design the minimal diagnostic calculation or numerical check to attempt to falsify it immediately.

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
3. **Autopsy & Re-Loop Rule**: If the computation yields a **FAIL**, log the failure mechanism into `## Problem Anti-Patterns & Killed Routes`, then return to Stage 3.3 to mutate the route or activate the secondary surviving route. Do not stop.

---

## Evaluation Protocol (Adversarial Critic Loop)

After drafting, lint every Markdown file in the working directory (master, leaf artifacts and notes):

```bash
python3 check_brainstorm.py <working-dir>
```

Resolve all syntax and formatting failures flagged by the linter until the summary line reports `0 with errors` and no file prints a warning. Re-run it after every edit pass, including after the critic loop and after appending test results. A single file can still be linted with `python3 check_brainstorm.py <file>.md`.

Spawn a fresh adversarial critic agent with this mandate:
> "Act as a hostile referee whose sole objective is to break the surviving routes in this document. 
> 1. Verify Persistence: Check if the generator attempted only one round of ideas and stopped. If all ideas were dismissed without mutation or replacement, flag this as a critical failure.
> 2. Challenge Novelty: Identify if this route is an existing published method under new terminology. If it exists, why is it not standard practice? Name the unstated adoption barrier.
> 3. Find the implicit assumption that sweeps the Hard Core under the rug.
> 4. Determine whether the 24-hour test can be trivially satisfied by the Minimal Contrast Foil.
> 5. Verify whether compute scripts conform to the environment (Slurm/sbatch if on a cluster).
> 6. Check whether any proposed step repeats a mechanism logged in the Problem Anti-Patterns section.
> Return only a numbered list of concrete, fatal objections categorized as Major or Minor. Do not provide praise or vague commentary."

Address all major critiques in place. Log every action in the `Revision & Critic Log`.