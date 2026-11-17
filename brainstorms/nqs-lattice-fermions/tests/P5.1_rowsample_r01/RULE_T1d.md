# P5.1 round 2, R51.3 `levrow` T1d: like-for-like rate against the exact ladder from the same point (pre-registered)
Written 2026-11-17 after T1c (MARGINAL, 2.40 gamma_LM; that verdict stands and is not rescored) and before
`t1d_ladder.py` or `t1d_cont.py` ran. Removes T1c's comparator caveat: gamma_LM was measured at eps 8.3-7.2e-4,
T1c ran at 7.2-6.1e-4.

Start point for both arms: T1c's endpoint, `results/t1c_theta.npy` (eps 6.11115e-4). 12-site `che`, one thread,
exact sums.

Baseline arm (`t1d_ladder.py`, interactive, `ulimit -t 600`): the exact cutoff ladder exactly as
`../P4.1_cheap_su2_r22/ladder.py` run_arm (one exact S, H build and tangent eigendecomposition per step, rungs
1e-5..1e-9, LM candidate |z| <= 0.05, line search s in {1, 1/2, 1/4} first lowering eps by > 1e-7, lowest-energy
rung accepted, STOP-NOGAIN if none) for as many steps as fit under a 9.4 CPU-min process cap (a step starts only if
elapsed + last step's cost fits). Setup excluded from its CPU, as in 742206. Result: eps_L after k_L >= 1 steps and
gamma_L = (6.11115e-4 - eps_L) / (ladder step CPU). If no step completes or the first step is STOP-NOGAIN, T1d is
VOID (no verdict; reported).

Comparison arm (`t1d_cont.py`, a gated quick job: it cannot also fit in the 10 CPU-min interactive cap): T1c's
protocol unchanged (same m = 8P, q = 0.5 Born + 0.5 ell/sum(ell), cutoffs {1e-5, 1e-6, 1e-7}, first-accept lengths
{0.002, 0.001, 0.0005}, exact gradient and energies, seeds continue the step index 11, 12, ...), reusing T1c's
scores (`results/lev0.npy`, built at the 742206 endpoint) without refresh. Steps run until eps <= eps_L, at most 20
(30 steps in all on one score build). Score cost, fixed now: the amortised 30-step reuse, 1.896/30 = 0.0632
CPU-min charged per continuation step. gamma_R = (6.11115e-4 - eps_k) / (sum of step CPU + 0.0632 k) at the first
step k with eps_k <= eps_L (or at step 20 if never reached, flagged).

Ratio rho = gamma_R / gamma_L.
* PASS if rho >= 3.
* KILLED if rho < 1.5.
* MARGINAL otherwise.
Reported only: rho with the score build charged in full to the continuation (1.896 CPU-min, T1c-style); per-step
gains of both arms.
