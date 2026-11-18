# RULE — P5.1 round 12 T1: force variance-reduction screen (R51.19 ZV, R51.21 strat, R51.23 RB, R51.25 Stein)

Written 2026-11-18 before any number below was computed. Script: `vr_t1.py` (exact sums, interactive, threads=1, `ulimit -t 600` per process).

**States (4).** 5x2 (N_e 8) and 6x2 (N_e 10) R51.8 ladders, `che`, U=8, t'=-0.25, (+,+): the SR-trained start
(`../P5.1_size_scaling_r04/starts/*_theta.npy`) and the exact-arm endpoint after 120 fixed steps
(`../P5.1_size_scaling_r04/results/S*_exact_tau0.005_s9000/theta.npy`; the late window where R51.8's need is set).
**Primary** = endpoint states; starts reported only.

**Quantity.** First-order step-error constant of the sampled SR step (R51.10/R51.11 machinery, lambda 1e-3):
kappa_X = sum_x p (p/q) h_X(x) / x^T S x, h_X = (g_X - <g_X>_p)^T M (g_X - <g_X>_p), M = A S A, q = guided
(0.8 p + 0.2 p_flat). Baseline g = O_c r, r = e - O_c^T x. rho_X = kappa_X / kappa_guided at the same state.
Known limit (L67, R51.10 kill): a one-step constant; it screens (a route with rho ~ 1 is dead), it does not predict the
multi-step need.

**Variants.**
* ZV (R51.19): g + c D, D_k = d_k E_loc = (H d_k psi)/psi - E_loc O_k (mean zero under p); c in {0.5, 1, c*}, c* the
  scalar minimiser of the guided-weighted metric-norm variance. Cost factor (assumed, per sample vs guided) 3.
* Stein (R51.25): g - C Z, Z_j = (H(psi v_j))/psi - E_loc v_j for orbit-symmetrised occupation features v_j (site
  density, doublon, NN density-density, NN Sz-Sz); C the (oracle, infinite-sample) weighted regression. Cost factor 1.
* Strat (R51.21): Neyman allocation over 40 quantile bins of log p, kappa = (sum_b sqrt(P_b W_b))^2/xSx with W_b the
  within-bin metric-norm second moment about the bin mean; also reported against R51.11's binned-IS optimum
  kappa_amp40 (same bins, about the global mean). Cost factor 1.
* RB (R51.23): Kg with the Metropolis kernel K(x,x') = A_xx'/D_max min(1, p'/p) on the symmetrised hop graph of the
  representatives (reversible w.r.t. p). Cost factor 1 + E_q[degree] (measured).

**Verdicts per route** (eff = rho x cost factor; 5 = 5x2 endpoint, 6 = 6x2 endpoint):
* PASS (release a T2 multi-step ladder design, gated, not run here): eff_6 <= 0.5 and rho_6 / rho_5 <= 1.
* KILL: eff_6 >= 0.8 and (eff_5 >= 0.8 or rho_6 / rho_5 > 1).
* GROWS flag (reported with any verdict): rho_6 / rho_5 <= 1/1.5 (gain growing with size); a GROWS route that is
  otherwise MARGINAL is re-costed before any kill.
* MARGINAL otherwise.
* Strat additionally: kappa_strat >= 0.8 kappa_amp40 at both endpoints -> MERGED into R51.11 (adds nothing over its
  binned optimum), whatever its rho.

**Sanity checks (abort on failure, no verdict):** |<D>_p| and |<Z>_p| <= 1e-6 x their p-rms; <Kg>_p = <g>_p to 1e-6
relative; kappa_guided reproduces `kappa_S5x2_start.json` / `kappa_S6x2_start.json` to 1% at the starts.
**R51.22 (antithetic) check:** spin-flip invariance of |psi| and of O on 256 random configurations (T0 claim).
