# VOI note: P4.1 r20, sampled-LM viability T1 (`submit_r20.sbatch`)

* leaf: P4.1
* route: R18'' (sampled truncated-metric linear method, the scalable form of the P4.1 polish optimiser)
* tier: T1
* decision: The test runs `che` at 12 sites, 4 steps per arm. f = R(4) / 0.5392, where 0.5392 is the exact LM's 4-step gain (RULE.md).
  * VIABLE: some arm has f >= 0.5 with no exact-energy rise. A sampled LM realises the gain. The 24-site sampled rung 2 may be prepared at the passing arm's samples-per-parameter ratio, with the critic's agreement.
  * MARGINAL: the best arm without a rise has 0.2 <= f < 0.5. No 24-site rung runs. At most one further 12-site T1 (4x samples or 3 seeds) is allowed, with the critic's agreement; otherwise this verdict is treated as NOT-VIABLE.
  * NOT-VIABLE: every arm has a rise or f < 0.2. No 24-site sampled-LM rung is ever run. The leaf's scalable-optimiser question goes to another route (an SR / minSR variant or a reduced block).
* prediction: MARGINAL.
  * P(MARGINAL) about 0.45, P(NOT-VIABLE) about 0.45, P(VIABLE) about 0.10.
  * The 64k probe's step 1 (same seed as the job) had an oracle-best exact R of +0.080 per step. Its best validation candidate was at 1.8 sigma, so it was rejected.
  * Reaching f >= 0.5 (R >= 0.27) would need about 3 to 4 accepted steps of that size. Reaching 0.2 (R >= 0.108) needs about 2.
  * The 8k probe found no lowering candidate at all, even for the oracle, so `tu8k` is predicted at f about 0.
  * `naive64k` is predicted to rise (symmetrised H; r19 rose 4.9 gaps at 8k), with P(rise) about 0.75.
* p: 0.45
* prediction-changes-decision: yes. Today the leaf's sampled-optimiser status is "SAMPLED-LOSES at 8k, cause undiagnosed", and the next action is undecided.
  * MARGINAL keeps the sampled LM alive for one more critic-gated 12-site T1.
  * NOT-VIABLE closes the sampled LM and moves the optimiser question to another route.
  * VIABLE opens rung 2.
  * The probes already show that the non-symmetric estimator removes r19's catastrophic rise. Whether any arm can turn that into a net gain is not predictable from one step.
* cheapest-kill-tried: T0 argument. In the 8k probe the oracle found no lowering candidate at step 1, so at about 3 samples per parameter the sampled tangent is too noisy to contain a descent step.
  * That kills `tu8k` but not `tu64k`, whose oracle-best step lowers E (R +0.080).
  * Local T1 probes: one step each of `tu8k` and `tu64k` (about 12 CPU-min together, logs/probe_*). They diagnose r19's failure as the symmetrised H estimator (|z|_true / |z| goes from >1 to about 1). They cannot fix the 4-step f, because acceptance at 1.8 sigma against a 2 sigma bar depends on the step-2 to step-4 samples.
* cost-cpuh: 0.85
* cheapest-version: one seed, `che` only (the control), 12 sites, 4 steps, 3 arms. Each arm answers a different branch of the decision:
  * `tu64k` is the decisive arm.
  * `tu8k` sets the samples-per-parameter floor if both TU arms pass.
  * `naive64k` asks whether sample count alone suffices.
  * The total is about 37 CPU-min, above the 10 CPU-min interactive allowance, so it cannot run as a local T1.
* convergence: the exact LM reaches R = 0.539 in 4 steps (r19 seq_che), and the rule scores 4 steps. Rejected steps count toward the 4.
* wall-time-basis: measured on 1 thread.
  * `tu64k`: 2.4 CPU-min per step plus 1.6 min set-up (logs/probe_tu64k_1step_1thread.txt, 11.8 GB peak RSS).
  * `tu8k`: at most 2.4 min per step, plus 1.5 min for the exact solver check.
  * `naive64k`: about 2.5 min per step (r19's 1.9 min at 8k, plus the 64k S/H build).
  * Total: 4 x 7.3 + 7.3 set-up and checks = about 37 min. x1.3 = 48 min, so --time 0:50:00 on 1 CPU, with --mem=20G.
* power: the run is deterministic given its fixed seeds (sampling uses fixed seeds; the energies are exact).
  * Seed-to-seed spread is unmeasured. P(the best f within 0.1 of a bar) is about 0.3. A second seed is the next release, not a re-read.
  * P(INCOMPLETE) is about 0.05.
