# P4.1 rung 1: pre-registered rule

Written 2026-11-17, before any output of this run existed. Earlier draft: `tests/P4.1_cheap_su2_r12/polish.py`. It printed a rule but never ran, and its `results/` is empty. This rule replaces that draft and uses the leaf's recorded rung-1 thresholds (Leaf Status Summary, "Next step").

## System and arms

* Open 4x3 Hubbard strip, N_e = 10 (5+5), U = 8, t' = -0.25.
* Sector: (+,+) point-group irrep through P_G, with exact S = 0 (closed-form projection).
* Every energy is an exact sum over all 627,264 configurations, using the orbit reduction (157,216 representatives).
* E0 = -7.827899034271753, and eps = (E - E0)/|E0|.
* Arms:
  * `che`: SR-trained pilot. eps0 = 2.86e-3.
  * `pcg`: R2, SR-trained T2a, H = 15. eps0 = 1.28e-3.
* Each arm's gap is g = E(theta0) - E0, and R = (E(theta0) - E)/g.

## Protocol

The protocol is the same for both arms. It is the Round 11 variant that polished `che`.

* **Step.** Block: all parameters. Truncate the tangent metric (orthogonalised against Psi) at a relative eigenvalue cutoff of 1e-5. Mode `lm`: the lowest root of the linear method, level-shifted only until the metric norm |z| is at most 0.05. No diagonal shift.
* **Line search.** Try s in {1, 1/2, 1/4} on the exact energy and take the first s that lowers it. If none does, the sequence stops, and the stop is final.
* **Rebuild.** S and H are rebuilt exactly after every accepted step.

### Part (a): exact polish

* Run up to 8 steps per arm.
* `che` resumes from Round 11's step-4 parameters, with steps 5 to 8 added. `pcg` starts from its T2a parameters.
* Define rho_pol = eps_pcg / eps_che, both after at most 8 steps (N8).
* rho at N4 is reported but decides nothing.

### Part (b): sampled polish

* Start each arm from its SR-trained theta0. Run 4 steps.
* Each step draws 8192 fresh i.i.d. samples r with p(r) proportional to |orbit(r)| Psi(r)^2. This is exact sampling, with no Markov chain, and the seed equals the step number.
* S and H come from the standard sampled LM estimator, which includes the local-energy-derivative term, normalised by the sampled S_00. The step uses the same cutoff, tau and mode as part (a).
* The line search uses only the sampled energy: correlated reweighting of the same samples.
* Exact energies are used for scoring only.
* f_arm = R_sampled(4 steps) / R_exact(4 steps), where R_exact is that arm's own exact 4-step R from part (a). For `che`, R_exact = 0.5392.
* Sampled `pcg` runs only if the exact verdict is KEEP-LIVE or CONFOUNDED. When the verdict is RETRACT, R2 needs no 24-site polish, so its sampled arm decides nothing.

## Verdicts

### Exact verdict, at N8

| Verdict | Condition |
| :--- | :--- |
| RETRACT | rho_pol >= 0.85 |
| KEEP-LIVE | rho_pol <= 0.60 |
| CONFOUNDED | 0.60 < rho_pol < 0.85 |
| INCOMPLETE | a sequence was cut by the wall-time or CPU cap before 8 steps (an early stop on a rejected step is complete) |

### Sampled verdict

The required arms are `che`, plus `pcg` when the exact verdict is KEEP-LIVE or CONFOUNDED.

| Verdict | Condition |
| :--- | :--- |
| SAMPLED-KEEPS | f >= 0.5 for every required arm |
| SAMPLED-LOSES | f < 0.5 for any required arm |
| INCOMPLETE | a required arm's sequence was cut by the cap |

### INVALID

The run is INVALID, and no verdict is read, if any of these holds:

* an arm's E(theta0) differs from its stored exact energy by more than 1e-8;
* the energy recomputed at Round 11's `che` step-4 parameters differs from the recorded value by more than 1e-10;
* the autograd S and H at `che` theta0 differ from Round 11's hand-assembled `S_H_theta0.npz` by more than 1e-8 (relative max-norm);
* a sampled arm's step-1 sampled energy is more than 6 standard errors from the exact energy.

## What each verdict makes the leaf do next

* **RETRACT.**
  * R2's T2a PASS is retracted as optimiser slack.
  * The 24-site T2b design can no longer be read as frame evidence.
  * R2 is killed at 12 sites, polished. P4.1 turns to an optimiser route, the sampled truncated linear method.
  * Together with SAMPLED-LOSES, and with R16 and R17 dying at the polished parent (a separate 7 CPU-minute T1), the leaf goes to EXHAUSTED at this scale.
* **KEEP-LIVE.**
  * The frame advantage survives polishing, and R2 stays live.
  * Any 24-site comparison must use polished arms. With SAMPLED-KEEPS, rung 2 is prepared: 24-site warm-started sampled polish of `che` and `pcg`, about 16 to 32 CPU-hours, which needs the critic's written agreement.
  * With SAMPLED-LOSES, no 24-site rung runs. The next step is a 12-site T1 on the sampled optimiser itself (sample count and cutoff).
* **CONFOUNDED.**
  * R2 stays at T2a, flagged as optimiser-confounded.
  * Rung 2 is prepared only with SAMPLED-KEEPS.
* **SAMPLED-LOSES.**
  * No 24-site rung is run, because a sampled optimiser cannot realise the gain at about 3 samples per parameter.
  * The 12-site exact result stays a diagnosis, not a method.
* **SAMPLED-KEEPS.**
  * The 24-site plateau can be tested as an optimiser limit (rung 2).
* **INCOMPLETE or INVALID.**
  * No verdict is read. Fix and rerun only the missing part. The script resumes from results/*.json.

## Not in this run

The R16 and R17 tangent rule at the polished `che` (about 7 CPU-minutes) belongs to the same session in the leaf's plan. It is a separate T1 and is not scripted here.

## Amendment 1 (2026-11-17, after the local smoke test, before any job)

### What the smoke tests showed

Two local smoke tests ran on 1 thread. Their outputs are in `results_probe/`, and they are not part of the run.

* **Exact `pcg`, step 1** (`logs/probe_pcg_1step_1thread.txt`): R = 0.240.
* **Sampled `che`, step 1** (`logs/probe_sampled_che_1step_1thread.txt`, seed 1, which is identical to the job's step 1):
  * The INVALID checks passed. Autograd S and H agree with Round 11 to 2.4e-16 and 2.7e-16. The sampled energy is within 1.3 standard errors of the exact energy.
  * The step itself raised the exact energy by 4.9 gaps (R = -4.906, s = 1/2).
  * Over the same step, the in-sample reweighted energy claimed a drop from -7.7995 to -7.9303.

### Change to the run

* Sampled `pcg` is skipped when sampled `che` has completed with f < 0.5. In that case SAMPLED-LOSES is already fixed by the rule above ("f < 0.5 for any required arm"), so skipping `pcg` cannot change any verdict.
* No threshold, arm, step count or sample count is changed.
* The job's sampled `che` step 1 must reproduce R = -4.906, because the seed is the same. A mismatch is reported as a reproducibility failure.
