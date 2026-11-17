# P2.2 R7.4 — T1b: excess of the path-gadget compensator on REAL trained states

Written 2026-11-17, before any run of real_error.py.

## Why
The path gadget Y = D^-1 L D^-1 (D = diag psi0) is certified with excess delta^2 tr L under RELATIVE noise
psi0 (1 + delta xi) (H8 f_rob = 0.060), but under ADDITIVE noise of equal norm its excess is 0.66 tau (H6) to
~125 tau (H8, eps 2.24e-3), because Y_ii reaches 7e4-5e5 on small-psi0 determinants
(logs/additive_noise_check.log). Which noise model do trained NQS states obey?

## State availability (checked before writing this rule)
No trained NQS parameters or amplitude vectors are stored anywhere under tests/P2.2_*/ (only JSON summaries;
the r5 NNBF-J states were for distorted H6 R=3.6, 6-31G** active K=12, a different instance). They are therefore
REGENERATED here, on the exact R7.4 instance, with the leaf's own NNBF-J code (r5 keyx_core.Ansatz, arm NNBFJ,
hidden 64, hj 32, full-sum L-BFGS as in keyx_core Part A, no sampling noise).

## Instance
Identical to path_gadget.py: linear H_n, STO-3G, R = 3.0 bohr, Loewdin (LOW) basis, full S_z = 0 sector,
psi0 exact, Y = the path gadget build_L at eps, tau = 0.2 mHa.
H6 (400 dets): seeds 0, 1, 2.  H8 (4900 dets): seeds 0, 1 (seed 2 if CPU allows).  L-BFGS maxiter 3000 (H6),
1500 (H8), or until the CPU cap, whichever first.

## Quantities (per state psi_NN, normalized, sign aligned so psi0 . psi_NN > 0)
* excess(eps) = psi_NN^T Y(eps) psi_NN = u^T L u, u = psi_NN / psi0 (on psi0 != 0 dets)
  = the certified upper bound minus the true dropped-term bias, i.e. what the certificate adds.
* also: true SCREEN bias psi_NN^T H_drop psi_NN, dE = E[psi_NN] - E0.
* decomposition: r = psi_NN / psi0 - 1; psi0^2-weighted RMS of r (= effective relative delta);
  |r| and |psi_NN - psi0| on the dets carrying the largest Y_ii (top 1% and Y_ii > 1e3);
  relative-model prediction delta_eff^2 tr L, additive-model prediction ||psi_NN - psi0||^2 / n_nz * sum Y_ii.
* eps_S = 2.2387e-3 (f_S*-matched screening point: f = 0.306 H6, 0.144 H8) and the grid end eps = 0.1.

## Rule (decided at eps_S; grid end reported)
Statistic: median over seeds of excess(eps_S), per instance.  Admissibility: a state counts only if
dE < 1.6 mHa (the leaf's NNBF-J premise); if no seed of an instance meets it, that instance is reported as
"premise not met" and does not vote.
* KILL R7.4 if the statistic > tau on any voting instance.
* PASS-T1b if the statistic <= 0.3 tau on every voting instance.
* AMBIGUOUS otherwise.
Caveat stated in advance: full-sum training has no sampling noise; VMC-trained states see few samples on
small-psi0 dets and are expected to be worse there, so PASS here is optimistic and KILL here is robust.

## Budget
<= 10 CPU-minutes interactive total (RLIMIT_CPU 600 s), BLAS threads = 1, no sbatch.
