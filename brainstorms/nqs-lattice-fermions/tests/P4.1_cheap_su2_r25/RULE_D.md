# P4.1 round 22: R38 `guided` T1b — does guided sampled SR reach the exactly polished 12-site `che`? (pre-registered)

Written 2026-11-17 after RULE_C's run (GUIDED-WINS, R(15) = 0.364, one seed) and before any longer run.

Protocol: `guided.py 60 <seed0> results_guided_s<seed0>` for seed0 in {6000, 7000, 8000}: 60 steps from `che`'s
SR-trained theta0, N = 16,384, otherwise identical to RULE_C. Exact energies scored every step, never used to choose.
Reference: the exactly ladder-polished `che` (job 742206) has eps 7.217e-4; with che's SR-trained eps 2.858e-3 that is
R_pol = 1 - 7.217e-4/2.858e-3 = 0.7475.

Rule on the median over the three replicas of R(60):
* REACHES-POLISHED if median R(60) >= 0.7475 and no replica has a step raising the exact energy by more than 0.02 g:
  guided SR is the leaf's scalable optimiser candidate; next, the same test from the `pcg` start, then the
  24-site protocol (critic).
* CLOSE if 0.60 <= median R(60) < 0.7475: report the per-step gain ratio over steps 41-60 and the fitted tail
  (RULE_amend2 method); the leaf decides with the critic whether a longer run is worth it.
* FAILS if median R(60) < 0.60: guided SR plateaus above the polished state; R38 stays a step-efficiency tool only.
Reported only: replica spread; steps to reach R21's 0.195; sample cost against the exact ladder's CPU-minutes.
