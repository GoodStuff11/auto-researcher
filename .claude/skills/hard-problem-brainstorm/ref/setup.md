# Setup: Invariants, Decomposition, Routing and the Analogy Map

<!-- Part of the hard-problem-brainstorm skill. Read when SKILL.md points here; never needed whole. -->

## Step 0 — Freeze Problem Invariants

Populate the `Invariants` section in `brainstorm-<slug>.md` using `templates/spec_invariants_template.md`. 
Define and freeze:
1. **The Hard Core**: The irreducible mathematical, structural, or physical phenomenon that cannot be approximated or simplified away.
2. **The Unacceptable Outcome**: The exact condition that makes a proposed solution worthless in practice (e.g., bounds energy but misses off-diagonal order; scales exponentially with system volume; requires tuning to measure zero).
3. **The Minimal Contrast Foil**: The simplest existing baseline, status-quo method, or standard ansatz that any credible advance must beat under identical evaluation constraints.
4. **Saturated / Existing Solutions**: List existing methods already known to address parts of the problem, along with why they are insufficient.
5. **Compute Budget**: the total CPU-hours for the whole brainstorm and the default per leaf, set once and recorded in the Invariants. Use the user's figure if they gave one. Otherwise ask once, together with the Stage 1.5 routing question (one interruption, not a recurring checkpoint); if there is still no answer, use 2000 CPU-hours in all and 300 per leaf, and say so in the Summary. Once the leaves are fixed (end of Stage 1.5), run `python3 <skill-dir>/compute_gate.py init <working-dir> --slug <slug> --total-cpuh X --per-leaf-cpuh Y --leaves P1.1,P1.2,...`; the gate refuses any job naming an unregistered leaf, and leaves added later are registered with `set-budget --add-leaf`. Only the user changes the total afterwards (`set-budget --total X --user-approval ...`).

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

Present the leaf summary table to the user and prompt for the routing mode, and in the same prompt for the compute budget if Step 0 did not get one (Step 0, item 5). If the user's request already fixes the mode (for example "fan out over all leaves") and the budget, do not ask: record them and continue. This is the only routine question to the user; after it, the procedure runs on its own, and the compute auditor (Stage 3.5.7) stands in for the user on spending decisions up to 1000 CPU-hours per job.
- **Option 1 (Mode A: Focused Deep-Dive)**: User selects 1 leaf node. Continue synchronously through Stages 2–4 in `brainstorm-<slug>.md`.
- **Option 2 (Mode B: Parallel Fan-Out)**: Tackle all leaf subproblems simultaneously via worker subagents.

---

## Stage 2 — Abstraction & Barrier Mapping (Mode A Execution)

1. **Mathematical Abstraction**: Formulate the chosen problem into explicit mathematical inputs, outputs, and exact criteria for what counts as solved.
2. **Barrier Isolation**:
   - For every major existing approach, identify the exact point where it halts against the Hard Core.
   - Separate **fundamental barriers** (theorems, complexity limits, symmetries) from **practical barriers** (representation bottlenecks, optimization landscapes).
   - Formulate the **Anti-Solution Space**: Explicitly list what an acceptable solution cannot look like based on established no-go results.
3. **Analogy Map (how other fields attacked the same structure).** Many barriers have been met before in another field under another name. Before generating routes, look there on purpose:
   - **Restate each barrier in field-neutral terms**: the structure, not the physics. Examples: "estimate a sum of $N$ terms of both signs whose magnitudes span orders of magnitude, from few evaluations"; "learn a function with exponentially many sign changes from samples of its magnitude"; "choose between two near-degenerate hypotheses when each estimate is biased toward its own starting point".
   - **For each restated barrier, find at least three fields that faced it**: numerical analysis, statistics, signal processing, machine-learning theory, combinatorial optimisation, control, cryptography and coding theory, other branches of physics or chemistry. Use a few quick targeted literature searches per barrier, not an exhaustive review.
   - **For each source, record:**
     - the source problem;
     - a correspondence table (what in their problem plays the role of each object here);
     - the technique, and *why it worked there*: which property of their problem it exploited;
     - the **translation gap**: which of those properties may fail here, checked against the Hard Core and the Anti-Solution Space;
     - a citation.
   - The map is T0 work. It lives in the leaf's section 1b (Mode B) or in this stage's section of the single document (Mode A), and it is the main source of cross-field routes (3.1, Operation A item 5, Operation B item 3).

---
