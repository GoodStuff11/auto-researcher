# Compute: the Value Gate, the Auditor and Job Dispatch

<!-- Part of the hard-problem-brainstorm skill. Read when SKILL.md points here; never needed whole. -->

### 3.5.6 Compute Value Gate (Every Job Must Earn Its Cost)

Compute is a scarce, shared resource. A job is submitted only if what it can teach is worth its cost. "The rule says to test it" is not a justification; the decision the result changes is. The gate is enforced by `compute_gate.py`, not by the agent's own judgement: every submission, resume, extension or re-run goes through `compute_gate.py submit`, which refuses on any failed rule (G1–G7 in its `--help`).

**The waste this gate exists to stop.** In the first run of this procedure about half of 28,000 CPU-hours went to jobs whose verdict changed no decision. The patterns were:
* a large array run before the T0 argument that later killed the route anyway (an 8,900 CPU-hour job returned AMBIGUOUS, then a scaling argument killed it for free);
* runs that could not converge in their budget, resumed three times and each time ended UNCLEAR;
* extra seeds after the verdict was already decided from stored seeds;
* overpowered tests: 18 tasks to measure an effect $10^5$ times the threshold, where one would have decided it;
* repeat pilots that returned the same answer, and wall-time misses followed by resumes.

**Value-of-information note (`tests/<dir>/voi.md`, linked from the leaf, written before every job of any tier).** One `key: value` line per field; `compute_gate.py --help` gives the exact keys. A job without every field is refused (G1). A job whose cap is at most 1 CPU-hour needs only the quick note (fields 1 and 2 plus leaf, route and tier), because writing the rest would cost more than the job:
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
| at most 1 | quick note (decision, prediction and p) |
| 1–10 | full value-of-information note |
| 10–100 | note, plus `auditor: APPROVED <date> <digest>` from the independent compute auditor (3.5.7) |
| 100–1000 | note and auditor, plus `passed-t2:` naming the T2 the route already passed; the leaf's cumulative spend is checked against its budget |
| above 1000 | **never automatic: ask the user**, with the note, the cost in plain numbers, and the cheaper alternative that was rejected and why; record `user-approval:` |

**Release compute one unit at a time.** Seeds, instances, arms and ladder rungs are submitted sequentially at every cost tier: the unit that can kill first, each later unit only if the earlier ones leave the question open. A pre-registered rule may span several units, but compute is released unit by unit, and the rule is applied after each. A single large array for a whole ladder or seed sweep is not allowed above 100 CPU-hours.

**Compute ledger.** `compute/ledger.tsv` is written by the tool, one row per job, matched to `sacct` by a unique job name (`hpb-<slug>-<leaf>-<route>-<NNN>`) so reused job ids are never miscounted. After each result, run `compute_gate.py refresh` and `compute_gate.py record <job> --verdict ... --changed yes|no --hit yes|no|na`. The master document's `## Compute Ledger` section is the output of `compute_gate.py report`, pasted in with every synthesis.

**Queue review (cancel without asking).** Every time a result lands, and before any wait, re-check every queued and running job against its value-of-information note. Cancel immediately with `compute_gate.py cancel <job> --reason ...`, without asking, any job whose result can no longer change a decision: its verdict is already decided from data on disk, its leaf is terminal or the route is dropped, its question has moved to another job, or its outcome is now predicted with probability at least the skip threshold by cheaper evidence. Partially needed array jobs are trimmed to the tasks that still matter (`scancel` by task index, then record it).

### 3.5.7 The Compute Auditor (Stands In for the User)

The agent that designed a route wants to see it tested; it should not also judge whether the test is worth its cost. For every job above 10 CPU-hours, spawn a **fresh** agent with no access to the route's history beyond what is listed here. Give it only the value-of-information note, the `#SBATCH` script, the output of `compute_gate.py check` and `compute_gate.py report`, and the leaf's T0 and T1 evidence for this route. Its mandate:

> The full mandate is in `briefs/auditor.md`; give the auditor that brief and only the inputs listed above.

Give the auditor the note's digest (`compute_gate.py digest voi.md`). On APPROVED, write `auditor: APPROVED <date> <digest>` into the note; the gate checks that the digest still matches, so any later edit to the note needs a new approval. The auditor's reasons go in the leaf next to the note. On REFUSED, do what it says, and record the refusal as `GATE-REFUSED(auditor: reason)` wherever the job was a reopening-ledger row. Do not re-ask a fresh auditor with the same note; change the design first.

**Keep the auditor off the critical path.** Spawn it in the background and switch to other work at once (submit-and-switch, 3.5.3); act on its verdict when it returns. When several jobs above 10 CPU-hours are ready at about the same time, give them to one auditor in a single call, with one verdict and digest per note: one agent comparing them can also apply item 6 across leaves.

**Jobs above 1000 CPU-hours.** Get the auditor's approval first, then send one request to the user (with a push notification if available) that bundles everything the job needs: the note, the cost in plain numbers, the cheaper alternative that was rejected, and the budget change (`set-budget --total` or a leaf cap) without which G5 would still refuse it. Keep working; the job stays prepared and blocks only its own route. The `user-approval:` line does not change the note's digest, so the auditor's approval stands. If the user has not answered by the time nothing else is left to do, record the route as `GATE-REFUSED(awaiting user)` and finish the run (3.5.9). Never wait idle for the answer.

### 3.5.8 Diminishing Returns and Prediction Calibration

* **No-change streak (G6).** When a leaf's last 3 recorded jobs all changed no decision, the gate refuses that leaf's jobs above 1 CPU-hour, and the leaf moves to DORMANT with a Saturation Diagnostic. Crashes and deliberate cancels do not count toward the streak. A job needs a `reopen-justification` (a new ingredient, a different question, a cheaper decisive test); above 10 CPU-hours the auditor's approval of the note covers it. Quick jobs of at most 1 CPU-hour stay exempt, so cheap falsifiers keep running. Thinking-queue work on the leaf continues, and it leaves DORMANT only by the exits of Mode B section 3b. The streak resets once a job changes a decision.
* **Leaf budget (G5).** A leaf that reaches its budget goes DORMANT (section 3b). Its next step is made cheaper; or the auditor approves moving unspent budget from another leaf, within the unchanged total (`compute_gate.py set-budget --leaf P=X --approved-by "<verdict>"`, logged); or the step is recorded as `GATE-REFUSED(budget)`. Only the user raises the total.
* **Calibration.** Every job's prediction and probability go in the ledger, and `record --hit` scores them. When at least 5 predictions with $p\ge0.8$ are scored, the gate adjusts the skip threshold of G2: to 0.7 if at least 90% hit (the agent's confident predictions are reliable, so skip more jobs), and to 0.9 if fewer than 75% hit (they are not, so test more). `compute_gate.py report` prints the current calibration.
* **Report.** With every synthesis, read the report's flags (streaks, crashes, wall-time misses, calibration, budget left) and act on each before the next submission.

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
3. **Autopsy & Re-Loop Rule**: If the computation yields a **FAIL**, complete the route's card (`KILLED at T<k>`, the pre-registered rule, the numbers against the foil with their uncertainties, the inference, what the kill does not rule out, and the strongest case against it). Add its Kill Index row, and write or revise a library entry if the failure mechanism generalises. Mark the route `KILLED` in the leaf's Route Portfolio, launch the pre-planned FAIL action from its decision tree, and refill the portfolio (Stage 3.3) if it drops below 3 live routes. The other live routes were already advancing while this job ran (Stage 3.5.3), so a kill never restarts the leaf from scratch. Do not stop.

---
