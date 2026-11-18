# P5.1 round 6: R51.8 next spends after unit 1a (prepared, NOT submitted)
Written 2026-11-17 by the P5.1 worker, after round-06-scoring.md and before any new data. Files in
`tests/P5.1_size_scaling_r04/`. Summary, problem and cards were not edited.

## T0: is s ~ 0.04 at 5x2 a floor? (`t0_floor_r06.py` -> `results/t0_floor_r06.json`, read-only on results)
Units: G_ref = 1.615e-3 per step; s(t) = (E_arm(t) - E_plat(exact)) / G_ref.

| arm | last-40 slope | last-20 slope | s, 20-step windows (1-20 ... 101-120) |
|---|---|---|---|
| exact | -3.8e-4 +- 1.2e-3 | -5.6e-3 +- 3.7e-3 | (reference) |
| guided 65,536 | -7.5e-4 +- 3.2e-4 | -1.5e-3 +- 1.0e-3 | 0.102, 0.100, 0.069, 0.067, 0.067, 0.054 |
| guided 262,144 | -8.0e-4 +- 5.9e-4 | -1.0e-3 +- 1.9e-3 | 0.045, 0.082, 0.052, 0.053, 0.060, 0.040 |
| guided 16,384 | -1.8e-3 +- 2.0e-4 | -1.3e-3 +- 5.7e-4 | 0.262, 0.261, 0.196, 0.163, 0.149, 0.107 |

* Finding: the 262k arm is a plateau, not a descent. Its window means are flat at 0.05 +- 0.015 from step 1 on, and its
  last-40 slope is not significant (1.4 sigma). If it continued, 40 more steps would remove about 0.03. The 65k arm
  is still descending slowly (2.3 sigma), and 16k is clearly descending.
* Mechanism: the exact arm is not converged either. It sits in a period-2 limit cycle (the fixed metric length
  tau = 0.005, always taken, overshoots near the minimum). Successive steps alternate between (E_start - E)/G = 0.78
  and 1.11 around steps 95-105, damping to 1.01/1.06 by step 120, with residual sd 0.09 G over the last 20 steps.
  The 262k arm shows the same alternation (s = -0.01 vs 0.11 on successive steps). The S10 start is very close to E0
  (G_ref is 0.45 of the gap; eps 5.5e-4), so tau is large compared with the gap that remains, and both arms bounce.
  The "shortfall" is therefore a difference between two limit-cycle averages, and its floor is set by tau against
  the start's distance to E0. It is not a sampling floor, and it is not universal: at 4x3 (eps 3.0e-3) the shortfall
  still fell as N^-0.87.
* What 5% means here: the threshold lies inside the plateau scatter. Changing only the averaging window moves
  N_needed(S10, 5%) from 94,150 (last 20 steps) to 258,000 (steps 81-120) to censored above 262,144 (steps 61-120).
  The 5% statistic is ill-conditioned at S10. It will be ill-conditioned at any size whose start is this close to E0
  (6x2 has eps 1.1e-3).

## PROPOSAL for the auditor and critic (NOT adopted; written before any new data)
Re-register R51.8's statistic before unit 1c data exist:
* P1 (preferred): s is the mean over steps 61-120 (60 steps; an even count cancels the period-2 alternation) against
  the exact arm's mean over the same steps, and the threshold is s = 0.10. Post hoc at S10, so this is illustrative
  only: N_needed = 29,530 (18,830 with last-20, 25,750 with 81-120; spread factor 1.6, against 2.7 or more at 5%).
  The bracketing rungs are 16,384 and 65,536, and the local exponent is 0.49-0.58.
* P2 (alternative): N_needed against each size's own large-N plateau. s* = s / s(262,144) - 1 crosses a fixed value.
  This is rejected as primary, because the reference then carries the top rung's noise.
* Conditioning check (either statistic): a size's N_needed is scored only if the bracketing rungs' s differ by more
  than twice the 20-step-window sd of s at the upper rung. Otherwise that size is UNRESOLVABLE at that threshold.
* The verdict references are unchanged (g3 = 1.728, gH1 = 3.52). The born comparator is unchanged.

## Prepared jobs (gate `check --pre-audit`, all three PASS)
| job | script / VOI | cap CPU-h | digest | gate |
|---|---|---|---|---|
| (a) 6x2 ladder: exact + guided 5 rungs + born 5 rungs | `submit_r06_s12p_ladder.sbatch` / `voi_r06_s12p_ladder.md` | 16.7 (2 x 8.33 h) | 4ae6306df3e8 | PASS PENDING AUDIT |
| (b) 4x3 exact arm | `submit_r06_ref12.sbatch` / `voi_r06_ref12.md` | 8.0 | df2e8fb635e6 | PASS |
| (c) 5x2 second seeds, guided 65k, 262k (+16k) | `submit_r06_s10_seed2.sbatch` / `voi_r06_s10_seed2.md` | 0.7 | c68271bbad05 | PASS |

* Wall times are sized for the job, not the interactive session: interactive pre-flight x 1.8 x 1.3. Job 763968
  measured an in-job/interactive ratio of 1.54-1.60, so the factor 1.8 is conservative.
* (a) is an array: task 0 runs the exact arm; task 1 runs guided and then born, in sequence, so a short wall time
  truncates born first.
* Signal path: (a) and (c) use B:USR1 plus a trap that forwards the signal to ns_run.py and stops later arms (tested
  locally with a mock). (b) uses exec. The old scripts had `--signal=USR1` without B:, which does not reach
  python children of the batch shell.
* (c) seed0 is 19000.
* Decision value given the fragility flag:
  * (a) is the primary job and changes the next spend: the S14 ladder, a top rung, or variance reduction. Under the
    5% rule it is likely to land AMBIGUOUS from conditioning alone. Fix the statistic first, then run it.
  * (b) changes no spend by itself; it changes only the geometry scope of the Answer. Its own statistic is well
    conditioned, but its ratio inherits the S10 denominator. Lowest priority; it can wait for (a).
  * (c) is needed under the current rule: the seed spread decides whether 5% is scorable at all. If P1 is adopted
    first, the 65k and 262k seeds become nearly moot and only the 16k seed matters. (c) costs 0.7 CPU-h either way,
    so keep it as written, or merge it into (a)'s task 1. It is cheaper to run it first: it would settle the
    re-registration empirically before (a) is scored.
