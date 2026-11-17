# P4.1 round 26 — R41 `anderson-raw` (ledger row 8, R30's translation repair)

Task (orchestrator): T0 then, if alive, a capped interactive T1 of Anderson mixing on the raw LM direction with a
fixed cutoff. summary.md not edited (orchestrator's instruction). Files: `tests/P4.1_cheap_su2_r41/` — `RULE.md`
(written before the run), `anderson_raw.py`, `logs/anderson_raw_che.txt`, `results/anderson_raw_che.json`,
`results/theta_anderson_raw.npy`.

## T0
1. Smoothness from R30's logs. R30 did not save its iterates, but its Anderson coefficients and step norms fix
   all inner products of its accepted steps F_0, F_1, F_2 (m=1 at k=1: one equation; m=1 and m=2 at k=2: three;
   unknowns |F_0|^2, <F_0,F_1>, <F_0,F_2>, <F_1,F_2>; the m=2 step norm is reproduced to 1e-16 as a check).
   |F_0| = 0.468, |F_1| = 0.327, |F_2| = 0.402; cos(F_0,F_2) = 0.905 (both cutoff 1e-7, s = 1/4);
   cos(F_0,F_1) = 0.171, cos(F_1,F_2) = 0.090 (F_1 had cutoff 1e-6, s = 1). Raw norms 1.87, 0.33, 1.61.
   Reading: R30's non-smoothness came from the cutoff switch; at fixed cutoff the raw direction is correlated.
   Not a kill.
2. R28 bound. It bounds schedules of gradient calls between exact builds (best 1.8×). Anderson keeps one build per
   step and changes the gain per build, which the bound does not cover. Cost per build can fall from the ladder's
   ~2.9 to ~2.2 CPU-min (one cutoff, 2-5 energies), so PASS (3×) needs about 2.3× the plain gain per build.
   Not a kill. -> T1.

## T1 (1 thread, OMP/OPENBLAS/MKL_NUM_THREADS=1, ulimit -t 600; set-up 0.06 CPU-min)
| build | eps(x_k) | rank | |d| | cos(d_k,d_{k-1}) | plain eps | Anderson candidates eps (gamma) | accepted | build CPU |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 0 | 7.21736e-4 | 2435 | 1.872 | — | 7.03466e-4 | — | plain | 2.16 |
| 1 | 7.03466e-4 | 2434 | 1.471 | 0.946 | 6.91146e-4 | m1 7.03151e-4 (−0.964) | plain | 2.21 |
| 2 | 6.91146e-4 | 2432 | 1.192 | 0.965 | 6.79427e-4 | m1 6.89228e-4 (−1.360); m2 6.88153e-4 | plain | 2.13 |
| 3 | 6.79427e-4 | 2430 | 0.997 | 0.964 | 6.68138e-4 | m1 6.66865e-4 (−1.243); m2 6.73028e-4; m3 6.69021e-4 | AA m=1 | 2.18 |

Build 4 was killed by the ulimit (exit 137) after results were written for build 3.
gamma = (7.21736 − 6.66865)e-4 / 9.05 CPU-min = 6.07e-6 = 0.83 gamma_LM (builds 1-3: 0.73). VERDICT: KILL.
Build 0 reproduced R30's build-0 step exactly (same cutoff and factor), a consistency check.

Reading: the repair worked as a smoothing (cos $\approx$ 0.95-0.97), and Anderson still failed. Nearly parallel residuals
whose norm shrinks ~20% per build give gamma $\approx$ −1 to −1.4, i.e. the extrapolation roughly doubles the step
(step norm 2.0-2.4× plain), and the full raw step is known to overshoot (beta = 1/4 is the line search's choice).
The step is curvature-limited, not contraction-limited; Anderson accelerates contractions. One win (build 3,
1.3e-6, a tenth of a plain gain). Fixed-cutoff plain gains (1.23, 1.17, 1.13e-5) are also below the ladder's
(1.57-1.70e-5), so fixing the cutoff costs gain per build.

Ledger row 8 outcome for the orchestrator: KILLED(T1): 0.83 gamma_LM, Anderson lost 5 of 6 times on a smooth map.
Proposed library entry: Anderson/DIIS accelerates a slowly contracting map, not a step already cut back by a line
search because of curvature; at a ladder-polished endpoint it extrapolates past the minimum along the step.
Compute: 9.2 CPU-min interactive.
