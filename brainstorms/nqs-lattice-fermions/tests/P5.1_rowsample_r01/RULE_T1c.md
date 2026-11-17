# P5.1 round 2, R51.3 `levrow` T1c: 10 polishing steps with a leverage-row metric, rate per CPU-minute (pre-registered)
Written 2026-11-17 after T1 (phi = 0.998 oracle) and T1b (phi(near) = 0.990, phi(far) = 0.948) and before `t1c.py`
was run. Fixes the details the R51.3 card left open; the bar is the card's.

Instance: 12-site `che` (2637 parameters, 157,216 orbit representatives), start = the ladder-polished endpoint of job
742206 (`../P4.1_cheap_su2_r22/results/theta_che.npy`, eps 7.2174e-4). One thread, `ulimit -t 600`.

Leverage scores (computed ONCE, at the start; no refresh during the 10 steps; T1b showed scores 20 steps stale keep
phi = 0.95): full Jacobian Y (all rows), S = Y^T Y, eigh, ell_r = y_r^T (S + lam I)^{-1} y_r, lam = 1e-7 w_max.
Their whole cost (energy + Jacobian + Gram + eigh + the leverage product) is charged to the run, i.e. amortised
over the 10 steps. Saved to results/lev0.npy so a resumed run does not recompute (cost still charged once).

Per step k = 1..10 at the current theta:
1. Exact energy and exact gradient G (`../P4.1_cheap_su2_r25/exgrad.py` energy_grad); force f = G/2.
2. q = 0.5 Born(current) + 0.5 ell/sum(ell) (Born from the current amplitudes, free in the exact setting as with
   samples in VMC); draw m = 8P = 21,096 rows i.i.d. (seed = k); compute the Jacobian only for the distinct drawn
   rows; S~ = sum of y_r y_r^T/(m q_r), normalised by S~_00 and centred (as T1).
3. One eigh of S~_T. Cutoff ladder c in {1e-5, 1e-6, 1e-7}: d_c = -Pm Pm^T f/|Pm^T f|, Pm = U_k/sqrt(w_k) over
   eigenvalues > c w_max. At each rung a backtracking line search over metric lengths tau in {0.002, 0.001, 0.0005}:
   the first tau whose exact energy lowers eps by > 1e-7 (the ladder's MIN_GAIN) is the rung's candidate. The rung
   with the lowest exact energy is accepted. (Mirrors the exact ladder: first-accept s in {1, 1/2, 1/4} per rung,
   best rung kept; the tau values are T1's amended grid without 0.004, which never gained.)
4. No rung qualifies -> STOP-NOGAIN (nothing applied; the step's cost is still charged).
CPU accounting: process time (user + sys) from just after setup (orbit reduction and model load excluded, as the
ladder's per-step times exclude them) to the end of the last step. The validity check below is excluded.

Validity check (INVALID if it fails): at the start, f from step 1 equals Y^T(H psi)/S_00 - E s from the full
Jacobian, relative max difference < 1e-6.

Rate gamma = (eps_start - eps_end) / CPU-min. Comparator gamma_LM = 7.27e-6 (exact ladder, last 5 steps of 742206).
* PASS if gamma >= 3 gamma_LM = 2.18e-5 after 10 steps (or after STOP-NOGAIN, over the steps run).
* KILLED if gamma < 1.5 gamma_LM = 1.09e-5.
* MARGINAL otherwise.
If the 10-CPU-min cap stops the run before step 10, the verdict on the steps completed is reported as PRELIMINARY
and the remaining steps go to a gated quick job (resume from results/t1c_state.json, theta, lev0.npy).
Reported only: per-step eps gain and CPU time, accepted rung, number of distinct rows, the rate without the score
cost, and the rate of the second half (steps 6-10).
