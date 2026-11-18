# P5.1 round 5, R51.13 multi-step T1: STORM recursion at 10 sites (pre-registered)
Written 2026-11-17 before `storm.py` ran beyond a 3-step a = 1 identity check (which must reproduce the stored
plain guided N = 4096 arm's first three energies exactly; otherwise the code is fixed before any scoring).

Instance: R51.8's S10 start (open 5x2, 4+4, U = 8, t' = -0.25, `che`, P = 1832), tau 0.005, 120 steps, seeds
9000 + n, guided draws, shift 1e-3. R = (E_start - E)/(E_start - E0_pg); R_plat = mean of the last 20 steps.
Foils (stored, same code, same seeds, finished): plain guided N = 4096 (equal CPU to STORM at N = 2048, since each
STORM draw is evaluated at two parameter sets) and plain guided N = 16384 (4x the samples, 2x STORM's CPU), from
`../P5.1_size_scaling_r04/results/S5x2_N8_guided_N{4096,16384}_tau0.005_s9000/run.json`. Only their R_plat is read
here; R51.8's own statistics and verdicts are not computed.
Arm: STORM N = 2048, a = 0.1 (one seed).

Let G = R_plat(plain 16k) - R_plat(plain 4k) (the gain from 4x samples).
* KILL if R_plat(STORM 2k) <= R_plat(plain 4k) (no better than spending the second evaluation on fresh draws).
* PASS if R_plat(STORM 2k) - R_plat(plain 4k) >= 0.5 G → prepare a T2 at 4x3 against R51.1's arms (equal CPU).
* MARGINAL otherwise. If G < 0.01 the test cannot resolve and is reported UNRESOLVED (S10 then too easy at 4096).

## Addendum (written after STORM-x scored KILL, before the repair ran): R51.18 translation repair
STORM-x (recursion on the solved direction) scored R_plat 0.2112 against plain 4k 0.2740 (G = 0.1286) → KILL.
Repair: the recursion is applied to the linear estimators S and f (each a plain sample mean, as SGD gradients are),
and the step solves (S_rec + lam)^-1 f_rec, with the metric length taken in S_rec (`storm.py --mode sf`).
Same arm (N = 2048, a = 0.1, seeds), same foils, same KILL / PASS / MARGINAL thresholds as above.
