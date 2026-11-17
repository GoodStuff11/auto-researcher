# VOI note: P4.1 round 21, R21 `trSR` T1 (`submit_r24.sbatch`), as amended (RULE_amend1.md)

* leaf: P4.1
* route: R21
* tier: T1
* decision: 15 sampled SR steps from `che` theta0 at 12 sites, 16,384 i.i.d. samples per step. There are two main arms, `trsr` (relative cutoff 1e-4) and `shift` (absolute 1e-3). Each chooses its metric step length per step from {0.005, 0.01, 0.02, 0.05} on a held-out sample set, and every step is scored by the exact energy. The verdict uses the better of the two arms (RULE_amend1.md).
  * VIABLE: some main arm has R(15) >= 0.20 and no step raises the exact energy by more than 0.02 of the gap. This sets the 24-site protocol, with the critic's agreement.
  * NOT-VIABLE: both main arms have R(15) < 0.08. This closes sampled SR polishing at about 6 samples per parameter.
  * MARGINAL: anything else. No 24-site rung runs.
  * R_trsr - R_shift and the `ladder` arm are reported only.
* prediction: MARGINAL.
  * Probabilities: MARGINAL 0.5, VIABLE 0.25, NOT-VIABLE 0.25.
  * Basis is the amended seed-1 probe (logs/probe_amend1_trsr_1step_1thread.txt). The held-out set chose length 0.02, and that step gained R = +0.024. The oracle's best was +0.037, at length 0.01. The exact-sum step at 0.02 gains 0.227.
  * So sampling noise at 6 samples per parameter keeps only about 10 to 15% of the exact step's gain, and per-step gains usually shrink. 15 steps at 0.01 to 0.025 each gives R between about 0.1 and 0.3, which straddles the VIABLE bar.
  * The `shift` arm's exact step 1 is better (0.288 at length 0.02), which raises P(VIABLE) slightly.
* p: 0.5
* prediction-changes-decision: yes. R21 is LIVE and untested, and its next action depends on the verdict. MARGINAL would hold off any 24-site sampled rung, VIABLE would open it, and NOT-VIABLE would close sampled polishing.
* cheapest-kill-tried: the exact-sum step 1 checks (logs/exact0_output.txt).
  * Length 0.05 overshoots even with exact sums, which led to the amendment.
  * At length 0.02 the exact shifted step beats the truncated one (0.288 against 0.227), so the card's "truncation fixes the shift" premise fails at step 1.
  * The amended seed-1 probe shows a positive but small sampled step. Nothing cheaper fixes R(15).
* cost-cpuh: 0.45
* cheapest-version: one seed, `che` only, two main arms plus the reported-only cutoff ladder. Total about 15.5 CPU-min, above the 10 CPU-min interactive allowance (about 10 CPU-min without the ladder arm).
* convergence: no convergence claim. The rule scores a fixed 15 steps.
* wall-time-basis: measured on 1 thread. Set-up is 0.25 CPU-min per arm. A `trsr` or `shift` step takes 0.30 CPU-min (logs/probe_amend1_trsr_1step_1thread.txt) and a `ladder` step 0.38. 3 x 0.25 + 15 x 0.98 = 15.5 min, x1.3 = 20 min, so --time 0:25:00 on 1 CPU with --mem=4G.
* power: deterministic given the fixed seeds, and the energies are exact. Seed-to-seed spread is unmeasured. P(the best R(15) within 0.03 of 0.20 or 0.08) is about 0.25. P(INCOMPLETE) is below 0.05.
