# Evidence Tiers, the Portfolio and the Decisive Test

<!-- Part of the hard-problem-brainstorm skill. Read when SKILL.md points here; never needed whole. -->

## Stage 3.5 — Evidence Tiers & the Route Portfolio (Submit-and-Switch)

**Core Objective**: Kill weak routes in minutes, not days, and never let one long job idle a leaf. Every gate-surviving route climbs a ladder of increasingly expensive evidence. A route is promoted only after it passes the cheaper tier, and while it waits on a job, the other routes keep moving.

### 3.5.1 The Evidence Tiers

| Tier | What it is | Budget (default cap) | Typical kills |
| :--- | :--- | :--- | :--- |
| **T0 Analysis** | Lemmas, counting and dimension arguments, small linear-algebra models (e.g. a $2\times2$ effective model), operation counts, asymptotic cost and extensivity estimates, known results | minutes of agent time, no job | mechanisms impossible in principle; cost models that cannot win; Gate 4 failures |
| **T1 Toy computation** | Exact computation on a toy instance that keeps the Hard Core (Hilbert dimension up to about $10^4$), plus the mandatory pre-flight dry run (3.5.4) | under 10 CPU-minutes; interactive (local or `srun`) without the gate, or a batch job of at most 1 CPU-hour with the quick note | structural failures, invalid instances, broken controls, the three-size Gate 4 trend |
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
* **Portfolio size.** Keep at least 3 live routes per OPEN leaf. When the count drops below 3, refill immediately with Operation A (mutation of the latest kill) or Operation B (orthogonal replenishment) from Stage 3.3, gated as usual. Two exceptions: a DORMANT leaf is not refilled (it works through its reopening ledger instead, section 3b), and a leaf just reopened from DORMANT keeps only its reopening routes until one of them passes T2. Two consecutive refill batches killed in full send the leaf to DORMANT (idea saturation, section 3b).
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
   * **pre-work for in-flight routes.** When every live route of a leaf is `IN-FLIGHT`, prepare what each verdict will need (the decision tree's next test, its pre-flight dry run, its value-of-information note) instead of generating new routes. New routes are generated only when the leaf has fewer than 3 live routes, counting in-flight ones (3.5.2), so a leaf with jobs in flight does not grow its portfolio without bound;
   * **untested variants and merged routes** that a critic kept alive but nobody has run (e.g. a variant merged into another route);
   * **open gaps and caveats named by critics or results** that a cheap T0 or T1 check can settle (e.g. "the sampled force is untested", "the foil may be confounded", "the toy did not test X");
   * **falsifiers of DORMANT and terminal leaves.** Every Saturation Diagnostic names the experiment that would falsify it. Run the ones that fit T0 or T1 caps; larger ones go through the compute gate as ordinary jobs. Record each outcome in the leaf's reopening ledger; a falsified diagnostic reopens its leaf with that route only (Mode B section 3b, exit 1);
   * **routes from Saturation Diagnostics.** A Saturation Diagnostic is not a terminal state: it says which routes failed at this scale and names the missing ingredients. A leaf closed with a diagnostic is **DORMANT**, not finished. Each named missing ingredient and each untried regime is a route source: generate at least one route from each, and take it through the gates, T0 and T1, recording each in the reopening ledger. The leaf then leaves DORMANT by the exits of Mode B section 3b: back to OPEN if a route survives and its next step passes the gate, EXHAUSTED at this scale if every ingredient's route was killed, or EXHAUSTED at this budget if a surviving route's next step was refused by the gate. Report leaves as ANSWERED (a diagnostic question with an answer), PROVEN-IMPOSSIBLE (within a stated model, with the proof), EXHAUSTED at this scale, EXHAUSTED at this budget, or DORMANT — never just "closed";
   * **cross-node critic and master synthesis.** Run the cross-node critic whenever two or more leaves have changed status (closed, promoted, killed) since the last one, and keep the master Summary and Cross-Pollination current with every closure;
   * **cleanup of the compute queue.** Cancel any queued or running job whose verdict can no longer change a decision (e.g. a pre-registered rule that is already decided from data on disk), and record why (Stage 3.5.6 queue review; no confirmation needed).
4. **On every job checkpoint or completion**, apply the early-stop rule (3.5.5) or the verdict rule, update the portfolio row, and launch the pre-planned next action.
5. **Idle waiting is legitimate only when every item of the list in step 3 is empty**, not merely when every live route is `IN-FLIGHT` or dependency-blocked. Before waiting, write one line per item stating why it is empty (for example "P2.1 EXHAUSTED at this scale; its compiled-update falsifier already ran"). If any line cannot be written, that item is the next action. A long wait is the most common way this procedure stalls: treat "all routes are in flight" as a prompt to check step 3, never as permission to stop.

### 3.5.4 Mandatory Pre-Flight Dry Run (Before Any T2 or T3 Submission)

Run the complete pipeline (train, aggregate, verdict) on a toy instance with a deliberately short budget, and record its output in the leaf's current log (`leaves/<id>/log/round-NN.md`):

* the instance parameters pass explicit validity asserts (e.g. an active-space size not exceeding the orbital count; the target symmetry sector exists and is the one the reference uses);
* the positive control returns its intended verdict, and the negative control returns its intended verdict, *in every case the rule will see* (a positive control must be a perfect partner regardless of which branch the trial state is in);
* the aggregate script parses the outputs and prints a verdict from synthetic results that cover every rule branch;
* the reference method (ED, DMRG, ...) matches an independent check on the toy instance in the same sector the arms use.

### 3.5.5 Decisive by Design: Power Sizing and Early Stops

* **Power sizing.** From the T2 pilot's seed-to-seed and sampling spread, choose seeds, samples and steps so that the PASS and FAIL thresholds are separated by at least $2\sigma$ of the scored metric. A T3 test that is likely to return AMBIGUOUS is redesigned before submission, not resolved afterwards with "add more seeds".
* **Checkpointed early stops.** T2 and T3 jobs write checkpoints and per-checkpoint metrics. The pre-registered rule includes futility and early-kill clauses at fixed fractions of the budget (e.g. 25% and 50%): if the metric at a checkpoint is already outside the PASS region by more than its projected remaining improvement (measured on the pilot's trajectory), cancel the job (`compute_gate.py cancel`, which keeps a deliberate cancel out of the no-change streak) and record the verdict at that checkpoint with `record`.
* **Afford the T3 before running its T2.** Before submitting a T2 pilot, project the cost of the T3 it would lead to. If that exceeds the leaf's remaining budget, first try to make the T3 cheaper, or get the auditor to move budget from another leaf (3.5.8). If neither works, do not run the T2: record the route as `GATE-REFUSED(budget)` with the projected cost. A pilot for a test that cannot be afforded is wasted compute.
* **Anytime reporting.** Partial checkpoint metrics are read and logged when they appear. They may *kill* early under the pre-registered rule, but never *promote* early.

## Stage 4 — The Decisive Test (Tier 3) & Compute Execution

Take a route that has passed T0–T2 (Stage 3.5) and design the minimal decisive calculation that can falsify it. Respect the pre-flight dry run (3.5.4), power sizing (3.5.5), the Compute Value Gate (3.5.6) and the compute auditor (3.5.7). A T3 whose outcome the T0 and T1 evidence already predicts with high confidence is not run; the prediction is recorded instead.

### 1. Test Specification
1. **Target Instance**: The minimal non-trivial instance preserving the Hard Core (e.g., 2x2 cluster, N=2 particle sector, smallest non-Abelian group).
2. **Scalar Metric $M$**: A single mathematically defined quantity to compute or derive.
3. **Falsification Threshold**:
   - **PASS**: $M$ satisfies the theoretical bound; proceed to full-scale derivation.
   - **FAIL**: $M$ violates the bound or matches the Minimal Contrast Foil; route is killed.
   - **AMBIGUOUS**: Noise or finite-size effects dominate; state the pre-committed resolution action.
