# P2.2 R7.4: measured delta and compensator excess for a cheap Jastrow-HF reference (H8)

Written 2026-11-17, after the run. It answers the critic's note that the R7.4 T0 cost-kill numbers (delta ~ 0.3-1,
excess ~ 3-35 tau) were estimates, not measurements. Script: `jhf_reference.py`. Log: `logs/jhf_reference_H8.log`.
Data: `results/jhf_reference_H8.json`. CPU: 112 s on one thread, interactive.

## Instance and construction
* Same instance as `path_gadget.py` / `real_error.py`: linear H8, STO-3G, R = 3.0 bohr, Loewdin basis, full
  S_z = 0 space (n = 4900), E0 = -3.94474801, tau = 0.2 mHa, eps_S = 2.2387e-3, grid end eps = 0.1.
* RHF determinant: the SCF occupied MOs, written in the Loewdin basis; amplitude = det_alpha * det_beta (minors).
  Check: the energy of the vector equals the SCF energy (-3.57234732).
* psi_ref = D_RHF(x) * exp(sum_k theta_k F_k(x)). theta is fixed by full-sum exact energy minimisation
  (L-BFGS with an analytic gradient; the orbitals are not re-optimised):
  * **G**: Gutzwiller only, F = sum_i n_iu n_id (1 parameter).
  * **JD**: Gutzwiller plus a distance Jastrow sum_{|i-j|=d} n_i n_j, d = 1..7 (8 parameters).
  * **JF**: site-resolved on-site g_i plus every pair v_ij n_i n_j (36 parameters). This is the most generous
    "cheap" two-body Jastrow-HF.
* Compensator: Y_ref = D_ref^-1 L_ref D_ref^-1. It is the same path-gadget `build_L` with psi_ref in place of psi0
  (pair_data on psi_ref, same dropped mask |h_ij| <= eps). By construction it is PSD, with Y = -h on dropped pairs
  and psi_ref^T Y_ref psi_ref = 0 (checked: <= 1e-17).
* **Certified excess** on a state phi = phi^T Y_ref phi = u^T L_ref u, u = phi / psi_ref. This is the amount the
  certificate adds above the true dropped-term bias. It is evaluated on the exact psi0, and on an NNBF-J
  state retrained here: A1 architecture h4/hj4, seed 0, full-sum L-BFGS, 1500 iterations, 37 CPU-s. Two snapshots
  were taken: the first iterate with dE <= 0.74 mHa (dE = 0.583 mHa, the leaf's accuracy regime) and the final
  state (dE = 0.036 mHa).
* **delta (definition used)**: delta = psi_ref^2-weighted RMS of (psi0/psi_ref - 1). This is the relative-noise
  model psi0 = psi_ref (1 + delta xi), which gives E[excess] = delta^2 tr L_ref. Both states are normalised and
  sign-aligned. This delta is identically equal to the psi0^2-weighted RMS of (psi_ref/psi0 - 1), the d_eff_rel of
  `real_error.py`, and to ||psi0 - psi_ref||. Both columns in the log agree.
* Needed delta = sqrt(tau / tr L_ref), computed with the reference's own tr L.

## Results (excess in units of tau = 0.2 mHa)

| ref | P | dE_ref (mHa) | 1 - overlap^2 | delta | need delta eps_S / end | excess on psi0 eps_S | eps_S on NNBF-J 0.58 / 0.04 mHa | excess on psi0 grid end | end on NNBF-J 0.58 / 0.04 |
|---|---|---|---|---|---|---|---|---|---|
| RHF only | 0 | 372.4 | 0.535 | 0.798 | 0.124 / 0.034 | 25.7 | 26.0 / 25.8 | 207 | 208 / 207 |
| G | 1 | 46.8 | 0.130 | 0.367 | 0.155 / 0.040 | **9.5** | 9.7 / 9.6 | **64.5** | 65.8 / 64.4 |
| JD | 8 | 22.4 | 0.099 | 0.319 | 0.176 / 0.047 | **5.4** | 5.5 / 5.4 | **35.1** | 35.2 / 35.0 |
| JF | 36 | 7.6 | 0.064 | 0.256 | 0.176 / 0.047 | **3.1** | 3.2 / 3.0 | **16.5** | 16.7 / 16.2 |

tr L_ref: RHF 1.29e-2 / 1.76e-1, G 8.36e-3 / 1.24e-1, JD 6.45e-3 / 9.16e-2, JF 6.46e-3 / 9.14e-2 (eps_S / end).
For comparison, tr L for psi0 is 7.1e-3 / 9.9e-2. The relative-model prediction delta^2 tr L_ref is 5.6 / 3.3 / 2.1 tau at eps_S
(the measured values are 1.5-1.7x higher) and 83 / 47 / 30 tau at the grid end (the measured values are 0.55-0.8x of that).

## Reading
* The measured delta for a cheap Jastrow-HF is **0.26-0.37** (0.80 for bare RHF). This is inside the claimed 0.3-1
  (JF is slightly below 0.3). It is above the needed 0.18 at eps_S and is 5-8x above the needed 0.045 at the grid end.
* The measured certified excess on psi0 is **3.1-9.5 tau at eps_S** and **16-65 tau at the grid end**. Every
  variant exceeds tau at both eps. The T0 estimate of 3-35 tau is confirmed at eps_S, and at the grid end it is an
  underestimate for G.
* The excess on the trained NNBF-J states is within 5% of the excess on psi0. The bias is set by the reference
  error, not by the state's own error.
* So the R7.4 kill premise holds as measured: a Jastrow-HF reference with up to 36 energy-optimised parameters
  leaves an excess > tau.
* Caveats:
  * The RHF orbitals are not re-optimised in the presence of the Jastrow. At R = 3.0 a UHF or optimised-orbital
    reference could do better; this was not tested.
  * JF has 92 determinants with |psi_ref| < 1e-9. psi0 has weight 2e-14 there; dropped pairs touching those
    determinants are skipped by `build_L`, a negligible omission.
