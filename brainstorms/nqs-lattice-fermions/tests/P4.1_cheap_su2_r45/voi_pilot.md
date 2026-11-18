# VOI note: P4.1 R45 from-init pilot (`submit_pilot.sbatch`)

* leaf: P4.1
* route: R45 (`sup-fit`); pilot required by the compute auditor (CHANGES REQUIRED on T2 digest 08d370f1663e)
* tier: T1
* decision: RULE_pilot.md (pre-registered). Power-law extrapolation of each arm's best-so-far 1-F (evaluations 50-150) to n = 2000. PASS (both <= 3e-4) -> T2 re-gated as prepared (maxfun 2000). FAIL -> T2 maxfun raised to n* (<= 6000) or warm-started, then re-gated and re-audited. INVALID -> diagnose before any T2.
* prediction: FAIL p 0.6. From-init fits start at 1-F 0.42 (logs/check_local.txt) and must fall about three decades; T1 showed fast progress only from an energy-converged start (0.25% per evaluation near 2.6e-4). PASS p 0.35, INVALID 0.05.
* p: 0.6
* prediction-changes-decision: yes. The auditor will not approve the T2 until it is known whether 2000 evaluations can reach the capacity regime; the pilot sets the T2's maxfun (and so its cost, 18 CPU-h now, up to about 50 at n* = 6000) or redirects it to a warm start. Without it the T2 risks a predictable progress-limited UNSETTLED.
* cheapest-kill-tried: (1) T0: no derivation gives the from-init decay rate. (2) Interactive: 150 evaluations of both arms cost about 27 CPU-min, above the 10 CPU-minute interactive allowance; one arm alone would not fit either. (3) Cheapest version: 150 evaluations (the minimum for a 100-point fit window after the initial transient), one seed.
* cost-cpuh: 1.5
* cheapest-version: 2 array tasks x 0.75 h x 1 CPU = 1.5 CPU-h cap; expected 0.85 CPU-h.
* fail-fast: the code path is the T2's, already verified (FD gradients 4e-10 / 2.5e-10, target symmetry 2e-13) and smoke-tested through train/resume/verdict; progress logged every 50 evaluations.
* convergence: not sought; this is a rate measurement.
* wall-time-basis: measured 2026-11-17, logs/check_local.txt: che 0.117, nnbf 0.065 CPU-min per evaluation, setup 0.3; x 1.8 in-job, x 1.3 margin -> 00:45:00. Peak RSS 1.66 GB (logs/rss_local.txt) -> 5G.
* power: deterministic exact sums; the extrapolation's uncertainty (power law vs plateau) is stated in RULE_pilot.md; the T2's NOT-PLATEAUED clause remains the convergence test.
* auditor: not needed (cap 1.5 <= 10 CPU-h)
