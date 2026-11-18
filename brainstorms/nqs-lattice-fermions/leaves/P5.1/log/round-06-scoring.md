# P5.1 round 6: scoring of R51.8 unit 1a (job 763968) and R51.10 kappa at 6x2 (job 763902)
Scored 2026-11-17 by `tests/P5.1_size_scaling_r04/score_r06.py` -> `results/score_r06.json`, under RULE.md rev 1,
voi_unit1a.md (staged decision) and RULE_kappa.md rev 1. No rule was amended.

## R51.8 unit 1a: 5x2 (S10) ladder, all 11 arms DONE at 120 steps
Exact arm: E_start -6.547116, E_plat -6.548731, G_ref 1.615e-3 (R_plat = 0.451 of the gap to E0_pg; the 4x3 exact
plateau, 0.7475 at the polished reference, is not comparable: S10 starts 5x closer to E0).

| N | guided s | born s | N < P=1832 |
|---|---|---|---|
| 1024 | 0.926 | 1.057 (E above start) | yes (MinSR regime) |
| 4096 | 0.392 | 0.800 | no |
| 16384 | 0.107 | 0.415 | no |
| 65536 | 0.054 | 0.159 | no |
| 262144 | 0.040 | 0.072 | no |

* N_needed(S10, guided) = 94,150 (ln-ln interpolation between 65,536 and 262,144). N_needed(S10, born): censored
  above 262,144 (> 2.8x guided).
* Staged decision (voi_unit1a.md): 16,850 < 94,150 < 151,700 -> IN-BETWEEN (as predicted, p 0.65). Against the 4x3
  bound (65,536, 262,144] the cross-geometry r lies in (0.70, 2.78]: undecided between POLY-CONSISTENT and AMBIGUOUS.
  That bound is valid only if the 4x3 exact arm's R_plat lies in [0.732, 0.770]; this is still unmeasured, and only
  the 4x3 exact arm can check it.
* Release condition for the 4x3 exact arm: geometry flag is RAISED (kappa(6x2)/kappa(4x3) = 0.61 < 1/1.5, below), so
  under RULE.md the arm is released, subject to audit. The second condition (r' vs r disagreement) waits on unit 1c.
* Fragility (flag, not scored): the guided tail flattens. The s ratio per 4x in N goes 2.36, 3.66, 1.98, 1.35, giving
  a local exponent of 0.22 between the bracketing rungs against 0.87 at 4x3. A rise of 0.010 in s(262,144), which is
  about 0.0045 in R and smaller than R38's run-to-run spread of 0.015, would censor N_needed above 262,144
  (POLY-CONSISTENT branch). Both bracketing rungs lie within 0.3 ln-units of 0.05 (0.077 and 0.22), which triggers
  RULE.md's second-seed release. A shortfall floor of about 0.04 for guided would make a 5% threshold ill-conditioned
  at every size.
* Comparator at S10 (90% of G_ref): T_tot(guided) = 16,384 x 105 = 1.72M; T_tot(born) = 262,144 x 62 = 16.25M.
  The ratio is 9.4 (>= 2), so S10 is GUIDED-FEWER; the verdict needs every primary size. At equal N, guided's s is lower at every rung, and
  born needs about 4x the samples to match it. MinSR (N < P) applies only at N = 1024 for both arms: neither
  reaches 90% there, and born ends above E_start. So MinSR = SR at every other rung, and the S10 comparator is effectively
  guided vs plain SR.

## R51.10 kappa (rev 1) at 6x2; 7x2 (job 763903) still RUNNING
* kappa_guided: S10 1.081e5, S12' (6x2) 7.61e4, S12 (4x3) 1.243e5. k1 = kappa(S12')/kappa(S10) = 0.70 <= 1.728
  (gH1 = 2750/781 = 3.52), so step 1 is on the KAPPA-POLY side. The KAPPA-POLY / -SUPPORT / -AMBIGUOUS class
  needs k2 (S14), which waits on 763903.
* Geometry flag: kappa(6x2)/kappa(4x3) = 0.61, outside [0.667, 1.5], so it is RAISED.
* Prediction recorded before the S12' ladder: R51.8 step S10 -> S12' is POLY-CONSISTENT (r' <= 1.728). A naive
  proportional estimate is N_needed(S12') ~ 0.70 x 94k ~ 66k. If R51.8 lands SUPPORT-LIKE at that step, R51.10 is
  KILLED as a predictor.
* Reported only: kappa_born 2.04e8 (S10), 2.32e10 (S12'), 1.48e9 (4x3). Born grows 113x from S10 to S12' while guided
  falls, so the heavy-tail barrier grows fast on the ladder. d_eff 526 / 886 / 999. gH for the S12' trained start is
  3.52 >= 1.5 g3 = 2.59, so r' is resolvable.

## Ambiguities resolved (flagged, rules not amended)
1. The 4x3 validity window cannot be checked without the 4x3 exact arm. I applied the staged branch as written
   (IN-BETWEEN) and treated the r interval as conditional on the window.
2. "Geometry matters" is read as either release trigger being met; the kappa trigger alone fires now.
3. The second-seed clause ("an arm whose s lies within 0.3 ln-units of 0.05 at a bracketing rung") is read as
   applying to both bracketing guided rungs (65,536 and 262,144). It is a release, not part of the verdict.
4. T_tot uses the first step at which the exact energy is <= E_start - 0.9 G_ref (not the plateau mean).
