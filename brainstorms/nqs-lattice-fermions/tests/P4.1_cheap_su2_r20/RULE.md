# P4.1 sampled-LM viability T1 (r20): pre-registered rule

Written 2026-11-17, before any output of this run existed. It follows r19's SAMPLED-LOSES (job 731462). The 8192-sample truncated-metric LM on `che` raised the exact energy at every step (R = -4.9, -14.1, -24.0, -42.4). Over the same steps, its in-sample reweighted line search claimed drops. r19 RULE.md says the next step for SAMPLED-LOSES is a 12-site T1 on the sampled optimiser itself. This rule is that T1.

## Question

Can a sampled truncated-metric linear method realise the exact LM's gain at 12 sites? It must do so without ever raising the exact energy. This must be settled before any 24-site sampled rung.

## What r19 already shows (diagnosis, not a verdict)

* The exact LM's step 1 at `che` theta0 had |z| = 0.040, so the cap did not bind, and |dtheta| = 1.01.
* The sampled step 1 hit the cap (|z| = 0.050 in the *sampled* metric), with |dtheta| = 3.21. At s = 1 the exact energy rose by 58 gaps, and the sampled estimate saw a rise too. At s = 1/2 the sampled estimate claimed a drop while the exact energy rose.
* So the step runs along directions where the sampled metric is too small. A |z| cap measured in the noisy metric cannot stop that, and an in-sample line search cannot detect it.

## Literature prescription tested

The stochastic linear method of Toulouse and Umrigar (JCP 126, 084102 (2007); Umrigar et al., PRL 98, 110201 (2007)) has three ingredients that r19 lacked:

1. **Non-symmetric H estimator.** H_ij = <O_i E_loc[O_j]> is used as sampled. It obeys the strong zero-variance principle (Nightingale and Melik-Alaverdian, PRL 87, 043401 (2001)). r19 symmetrised it, which throws that property away.
2. **a_diag stabilisation.** a_diag is added to the diagonal of H in raw-parameter space, so it penalises a_diag |dtheta|^2. This penalty does not depend on the noisy S, and it damps exactly the badly sampled directions.
3. **Choosing a_diag on independent samples.** a_diag is chosen by correlated-sampling energy differences on an independent sample set.

The cases of Neuscamman's blocked LM (Zhao and Neuscamman, JCTC 2017) and Sabzevari et al. (accelerated LM, 2020) are cost and memory at large P, which are not the issue at P = 2637. They also keep the shift and independent-sample selection. Neither is tested separately here.

## System (identical to r19 part b)

* Open 4x3 Hubbard strip, N_e = 10, U = 8, t' = -0.25. The sector is (+,+) with exact S = 0.
* E0 = -7.827899034271753.
* Arm `che`: SR-trained pilot, P = 2637, theta0 = the r19 / Round 11 starting point. The gap is g = 0.0223728.
* R = (E(theta0) - E) / g. Every energy used for scoring is an exact sum. Exact energies never enter a decision inside the optimiser.
* The reference is the exact LM 4-step gain at the same cutoff and cap, R_exact(4) = 0.5392 (r19 `results/seq_che.json`).
* Sampling is i.i.d. from p(r) proportional to |orbit(r)| Psi(r)^2, as in r19.

## Arms (`che` only, 4 steps each, from theta0)

| arm | H estimator | stabilisation | step choice / acceptance | N per step |
| :--- | :--- | :--- | :--- | :--- |
| `tu8k` | non-symmetric | a_diag in {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2} x w_max(S_T) | a_diag = argmin of the correlated dE on a fresh, independent validation set (N_v = N); accept only if dE < -2 sigma(dE) | 8192 (3.1 per parameter) |
| `tu64k` | non-symmetric | same | same | 65536 (24.9 per parameter) |
| `naive64k` | symmetrised (r19) | none | r19 in-sample line search, s in {1, 1/2, 1/4} | 65536 |

The common settings for every arm are a metric cutoff of 1e-5 relative, the |z| <= 0.05 level-shift cap, and orthogonalisation against Psi. These are the r19 and exact-LM settings, so R_exact = 0.5392 is the right reference.

A rejected step leaves theta unchanged. It still counts as one of the 4 steps, and the next step draws fresh samples.

Seeds are fixed. The training seed equals the step number, which makes `tu8k` step 1 draw r19's step-1 sample set. The validation seed is 1000 + the step number.

The free diagnostics are recorded but decide nothing:
* the exact energy of every a_diag candidate (oracle);
* each step's norm in the exact metric, |z|_true;
* the a_diag the oracle would have chosen.

## Per-arm quantities

* f = R(after 4 steps) / 0.5392.
* An exact rise is any accepted step whose exact energy is higher than the exact energy before it, by more than 1e-10.

## Verdict

| Verdict | Condition |
| :--- | :--- |
| VIABLE | at least one arm has f >= 0.5 and no exact rise |
| NOT-VIABLE | every arm has an exact rise or f < 0.2 |
| MARGINAL | anything else (the best arm without a rise has 0.2 <= f < 0.5) |
| INCOMPLETE | an arm was cut by the CPU cap before 4 steps; no verdict is read, and the run resumes |

### INVALID (no verdict is read)

The run is INVALID if any of these holds:
* E(theta0) differs from the stored value by more than 1e-8;
* an arm's step-1 sampled energy is more than 6 standard errors from the exact energy;
* the non-symmetric solver, given the exact S and H at theta0 with a_diag = 0, does not reproduce r19's exact-LM step-1 dtheta to a relative error of 1e-6. This checks the new solver against the symmetric secular-equation solver.

## What each verdict does to P4.1

* **VIABLE.** A sampled LM can realise the 12-site gain.
  * The passing arm with the fewest samples sets the 24-site protocol and its samples-per-parameter ratio.
  * Rung 2 (the 24-site warm-started sampled polish of `che` and `pcg`) may then be prepared. Its cost is re-estimated from that ratio, and it needs the critic's written agreement.
  * If only `naive64k` passes, the fix is sample count alone. If only `tu64k` passes, both the TU prescription and about 25 samples per parameter are needed.
* **MARGINAL.** No 24-site rung runs.
  * At most one further 12-site T1 is allowed: the best arm with 4x samples, or 3 seeds. It needs the critic's agreement.
  * Without that agreement, MARGINAL is treated as NOT-VIABLE.
* **NOT-VIABLE.** No 24-site sampled-LM rung is run.
  * The 12-site exact polish stays a diagnosis, not a method.
  * The leaf's question of a scalable optimiser for the polished arms goes to another route (for example an SR / minSR variant, or a reduced parameter block).
  * R2 stays at T2a, flagged as optimiser-confounded (r19).
* **INCOMPLETE or INVALID.** No verdict is read. Fix and rerun only the missing part; the script resumes from results/*.json.

## Amendment 1 (2026-11-17, after the local timing probes, before any job)

### What the probes showed

Two one-step probes ran. Their outputs are in `results_probe/` and are not part of the run.

* **`tu8k` step 1** (`logs/probe_tu8k_1step_1thread.txt`, seeds identical to the job's).
  * BLAS was not pinned to 1 thread in this probe: 9.2 CPU-min for 125 s wall. It used the original 6-value grid.
  * The solver check passed: rel 1.1e-13. Sampled E is 1.3 se from exact.
  * With the non-symmetric estimator, |z|_true is about equal to |z| (0.047 against 0.044). For r19's symmetrised step the exact rise was 4.9 gaps. Here every candidate raises the exact energy by 0.16 to 0.78 gaps.
  * The validation set scored every candidate as a rise (dE > 0), so the step was REJECTED. The selection worked, but nothing was available to select.
* **`tu64k` step 1** (`logs/probe_tu64k_1step_1thread.txt`, 1 thread, 2.4 CPU-min per step).
  * The best validation candidate was a_diag = 1e-3: dE = -1.81e-3 +- 1.0e-3, which is 1.8 sigma, so the step was REJECTED.
  * The oracle's best exact R is +0.080, also at a_diag = 1e-3. The exact LM's step 1 gives +0.269.
  * At a_diag = 1e-1 the secular equation found no root, and that candidate is skipped.

### Changes to the run

* The a_diag grid gains 1e-1 and 1. At 8192 samples, a_diag = 1e-2 still raised the exact energy, so the grid had not reached the damped end. Taking the argmin over the validation set already covers the larger grid, so no threshold changes.
* BLAS threads are now pinned to 1, as they are in r19.
* No arm, threshold, step count, sample count, seed or acceptance bar is changed.

### Reproducibility checks

* The job's `tu64k` step 1 must reproduce the probe: REJECT, with every candidate's validation dE equal to the probe's to 1e-6 relative.
* The job's `tu8k` step 1 must reproduce the probe's 6 shared candidates to 1e-6 relative. That probe was multithreaded, so its rounding differs.
* A mismatch is reported as a reproducibility failure.
