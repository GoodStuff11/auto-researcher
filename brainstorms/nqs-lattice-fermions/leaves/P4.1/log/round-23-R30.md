# P4.1 round 23 — R30 `anderson` cheapest falsifier

Task (coordinator): the cheapest falsifier of the parked R30 (Anderson/DIIS acceleration of the exact ladder-LM
steps), at most 10 CPU-min interactive. summary.md not edited (coordinator's instruction).

Files: `tests/P4.1_cheap_su2_r27/` — `RULE_R30.md` (written before the run), `anderson.py`,
`logs/anderson_che.txt`, `results/anderson_che.json`, `results/theta_anderson.npy`. (`r26` was already in use by
another worker, so the next free directory was taken.)

Design: `che` from the job 742206 endpoint (start eps reproduced exactly, 7.217364e-4). Map G = one ladder step
copied from `P4.1_cheap_su2_r22/ladder.py` (one exact S, H build reused across cutoffs 1e-5..1e-9, $|z|\le 0.05$,
line search s in {1, 1/2, 1/4}). Safeguarded type-II Anderson, beta = 1, depths m = 1..min(k, 4), each candidate
scored by the exact energy (about 0.045 CPU-min), kept only if below E(G(x_k)). Three builds fit the 10-min limit,
so only $m\le 2$ was reachable (depths 3-5 need 4-6 builds, 12-17 CPU-min; stated in the rule).

Result (1 thread; set-up 0.07, total 8.86 CPU-min):

| build | eps(x_k) | plain G: eps (cut, s) | Anderson candidates eps (gamma coeffs) | accepted | build CPU |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 0 | 7.21736e-4 | 7.03466e-4 (1e-7, 1/4) | none (no history) | plain | 2.98 |
| 1 | 7.03466e-4 | 6.87717e-4 (1e-6, 1) | m1 6.88900e-4 (0.295) | plain | 2.84 |
| 2 | 6.87717e-4 | 6.70673e-4 (1e-7, 1/4) | m1 6.79978e-4 (0.612); m2 6.78051e-4 (-0.304, 0.315) | plain | 2.86 |

gamma_AA = (7.21736 - 6.70673)e-4 / 8.80 CPU-min = 5.80e-6 = 0.80 gamma_LM (builds 1-2 only: 0.77). Gains per
build 1.83, 1.57, 1.70e-5; r = (d2/d0)^{1/2} = 0.966, tail/eps_end = 0.72 -> tail does not shrink.
VERDICT (RULE_R30.md): KILL (< 1.5 gamma_LM). Comparison: R28 `plbfgs` AMBIGUOUS at 1.12.

Reading: zero of three Anderson candidates beat the plain step from the same point, so the verdict does not depend
on the baseline (this probe's plain steps ran at about 0.83 gamma_LM, within the 742206 step-to-step scatter of
1.68-2.67e-5 per step). The accepted line-search factor alternated 1/4, 1, 1/4: the map's step length jumps
discretely, so the residual history is not a smooth linear contraction and the secant extrapolation points the
wrong way. This is the translation gap the card predicted (G2 note at T0).

Untested variants (card, "does not rule out"): m = 3-5; Anderson on the raw LM direction before the line search;
`pcg`. Not proposed: each needs a gated job, and R27, R29, R30 now all fail to beat the ladder at the polished
endpoint, consistent with the round-22 proposed library entry (ill-conditioning, not the algorithm).

Card updates: R30 LIVE -> KILLED at T1; kill_index row added.
