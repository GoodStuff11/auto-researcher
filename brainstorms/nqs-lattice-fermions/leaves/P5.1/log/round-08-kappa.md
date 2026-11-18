# P5.1 round 8: scoring of R51.10 `kappa` (job 763903, 7x2) under RULE_kappa.md rev 1
Scored 2026-11-18 from `tests/P5.1_size_scaling_r04/results/kappa_S{5x2,6x2,7x2,4x3}_start.json` and
`logs/hpb-nqs-fermions-P5.1-R51.10-002_763903.{out,err}` (err empty; 7x2 start trained 1000 SR steps, final E
-8.546149 +- 0.0020; exact kappa two-pass, 154.3 CPU-min; ledger used 3.28 of 3.25 CPU-h cap). No rule amended.
Scored before any R51.8 ladder at 6x2 (769376 RUNNING) or 7x2 returned, so the class below is a recorded prediction.

## Inputs (exact sums at each size's SR-trained start, guided q, shift 1e-3)
| size | P | reps | kappa_guided | kappa_born | d_eff | d_eff/P | exp(H[p]) |
|---|---|---|---|---|---|---|---|
| S10 5x2 | 1832 | 11,100 | 1.081e5 | 2.04e8 | 526 | 0.287 | 781 |
| S12' 6x2 | 2637 | 156,816 | 7.61e4 | 2.32e10 | 886 | 0.336 | 2750 |
| S14 7x2 | 3590 | 2,255,421 | 1.989e5 | 5.99e10 | 1271 | 0.354 | 17,370 |
| S12 4x3 (geometry) | 2637 | 157,216 | 1.243e5 | 1.48e9 | 999 | 0.379 | 3035 |

## Verdict
* k1 = kappa(6x2)/kappa(5x2) = 0.704 $\le$ g3 = 1.728; gH1 = 3.52.
* k2 = kappa(7x2)/kappa(5x2) = 1.840 $\le$ 1.4^3 = 2.744; gH2 = 17,370/781 = 22.2.
* Class: **KAPPA-POLY** (both conditions hold; far below KAPPA-SUPPORT on both steps). Ledger prediction
  "KAPPA-POLY, k2 about 1.3-2" (p 0.55): hit.
* Geometry flag: kappa(6x2)/kappa(4x3) = 0.612 $<$ 1/1.5: RAISED (unchanged from round 6; feeds RULE.md's 4x3 exact-arm
  release together with the r' vs r comparison).
* Reported only: the series is not monotone (0.70 then x2.61 from 6x2 to 7x2; the second step alone exceeds
  (7/6)^3 = 1.59, so the 3-size POLY class rests on the dip at 6x2). Born kappa x113 then x2.6; Born/guided 1.9e3,
  3.0e5, 3.0e5 (heavy-tail barrier B2 grows, guiding absorbs it). d_eff/P rises slowly (0.29 -> 0.35).

## Prediction recorded for R51.8 (before its ladders return)
* 5x2 -> 6x2 step (769376): POLY-CONSISTENT, i.e. N_needed(6x2) $\le$ 54,050 (RULE.md rev 2 rescaled threshold). If
  N_needed tracked kappa linearly, N_needed(6x2) ~ 0.70 x 31,280 ~ 22,000.
* 5x2 -> 7x2 (later S14 ladder): POLY-CONSISTENT-3, N_needed(7x2)/N_needed(5x2) $\le$ 2.744; naive ~ 1.84 x 31,280 ~ 57,600.
* Rule: a class disagreement at a step where R51.8 is not AMBIGUOUS (or UNRESOLVABLE) KILLS R51.10 as a predictor;
  agreement at both steps PASSES it for a sampled kappa at 24 sites. Outcome stays LIVE (T1 complete, T2 = R51.8's arms).
* Decision effect: none now (KAPPA-SUPPORT would have redirected the leaf before the S14 ladder; POLY leaves the plan
  as it was). Ledger: changed no, hit yes.

## R-scale caveat (for later scoring; does not touch R51.10)
kappa is a one-state ratio, independent of any gap normalisation, so R51.10 is unaffected. For R51.16 and R51.11:
R51.1's gap is 0.022373 whereas `score_common.py` uses 0.023111 (3.3% larger); any R or s compared across the two
must be put on one scale before scoring.
