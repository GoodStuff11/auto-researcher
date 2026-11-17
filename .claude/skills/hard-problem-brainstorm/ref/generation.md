# Generating, Gating and Mutating Routes

<!-- Part of the hard-problem-brainstorm skill. Read when SKILL.md points here; never needed whole. -->

## Stage 3 — Dynamic Generation, Falsification & Continuous Mutation

**Core Objective**: Identify and refine candidate routes until at least 1–2 robust proposals pass all elimination gates. You must not surrender if initial ideas fail.

### 3.1 Initial Generation Batch
Generate an initial pool of 3–4 candidate routes. Read the findings library's index (`bs.py context` includes it; fetch entries with `bs.py show L<n>`) and search `kill_index.md` for the mechanism you are about to propose (`bs.py grep`) first, so no new route repeats a mechanism that already failed. At least one route in the initial pool, and at least one in every later refill batch while the Analogy Map has untried entries, is a **cross-field translation** taken from the Analogy Map (Stage 2, item 3). Ideas from the field's own playbook tend to share its blind spots, and a borrowed technique that dies quickly is still worth a targeted repair (Operation A, item 5). Write a route card for each route as it is generated.

### 3.2 Binary Falsification Gates
Subject each candidate route to five binary elimination gates:

* **Gate 0: Novelty & Adoption Reality Check**
  - Is this approach already known or essentially identical to an existing baseline?
  - If known: Why is it not in widespread use today?
    - If the field tried it and moved on, what was the fatal flaw (e.g., sample complexity, sign instability, optimization traps)?
    - Does the proposal present a concrete, novel mechanism to fix that exact flaw?
    - If no novel mechanism is proposed to fix the known adoption barrier, the route is FAIL (Not Novel / Stale Baseline).
  - For a cross-field translation, check both literatures. If the technique has already been ported to this field, the questions above apply as usual. If it has not, the port itself is the novelty, and the question becomes why nobody here has used it; the answer is usually the translation gap, which the route must address.

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
5. **If a cross-field translation is killed (any gate, T0 or T1): translation repair.** Borrowed techniques often die because one property of the source problem does not carry over, while the rest of the idea is sound. Before abandoning the route, mutate it once:
   - Read from the kill which property of the source problem it exposed as missing here: the translation gap that mattered, which may differ from the one predicted in the Analogy Map.
   - Branch Route $X \to X'$ in one of two ways:
     - **restore the property**, through a change of variables, basis, representation or instance class that recreates it here;
     - **replace the component** of the technique that relied on it with one that does not.
   - The repaired route goes through the gates like any other and gets its own card, with lineage "translation repair of R<x>".
   - If the repair also dies, both cards record why the analogy fails. If the analogy is natural enough that others would try it, add a findings-library entry under "About methods", so the next project knows the gap without paying for it.

#### Operation B: Orthogonal Replenishment (New Seed Generation)
If an idea is completely irrecoverable (e.g., fundamentally violates a no-go theorem), complete its route card (KILLED, with the gate code, the evidence and the strongest case against the kill), add its Kill Index row and, if the reason generalises, a library entry, and generate a fresh replacement route using one of three forced orthogonal directions:
1. **Duality / Frame Transformation**: If current ideas operate in real space, formulate the candidate in dual momentum, graph, or phase space.
2. **Active / Interactive Probing**: If passive representations hit complexity walls, formulate a route based on active, adaptive query algorithms or localized projective measurements.
3. **Cross-Field Translation**: Port a concrete technique from another field that conquered an analogous barrier, modifying it to respect the Hard Core. Take it from an untried entry of the Analogy Map; if the map has no untried entry, extend the map first (Stage 2, item 3).

**Termination Condition for Stage 3**: Continue the generate $\to$ gate $\to$ mutate loop until **at least 3 distinct candidate routes survive all 5 gates** (the starting Route Portfolio of Stage 3.5). Generation does not stop when the first route survives: the portfolio is refilled whenever it drops below 3 live routes. The loop is bounded: when two consecutive refill batches are killed in full at the gates, T0 or T1, the leaf stops generating blind and goes DORMANT with a Saturation Diagnostic built from those kills (idea saturation, section 3b). Saturation counts only if the batches included at least one cross-field translation that also went through translation repair; otherwise the next batch is drawn from the Analogy Map first. Its named missing ingredients then become targeted route sources (Stage 3.5.3, step 3), which is cheaper than a third untargeted batch.

For each surviving route, state the **Sharpest Gap**: the missing lemma, unproven scaling behavior, or unverified physical assumption required for the route to function.

---
