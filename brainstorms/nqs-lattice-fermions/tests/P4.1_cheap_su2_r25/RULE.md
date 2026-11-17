# P4.1 round 22, sub-question A: exact-sum optimisers that converge the 12-site arms (pre-registered)

Written 2026-11-17 before any optimiser run. Only `probe_timing.py` has been run (timing and a gradient check:
che E+gradient 0.121 CPU-min, energy alone 0.045; pcg 0.552 and 0.173; directional derivative equals the central
finite difference to 7 digits on both arms).

## Start points and reference rate
* Start: the job 742206 endpoints `../P4.1_cheap_su2_r22/results/theta_{che,pcg}.npy`
  (che eps 7.217e-4 after 12 ladder steps; pcg eps 4.230e-4 after 9).
* Reference rate gamma_LM (relative-energy units per CPU-minute) = mean of the last 5 accepted ladder gains
  divided by the mean CPU-minutes per ladder step, from `state_<arm>.json`:
  che (2.17+2.07+1.93+2.67+1.68)e-5/5 / 2.86 = 0.73e-5 per CPU-min;
  pcg (5.57+4.56+1.02+7.37+2.73)e-5/5 / about 7.6 = 0.56e-5 per CPU-min (recomputed by `run.py` from the json).

## Arms (each 1 thread, CPU budget per arm in brackets; every energy is exact)
* R27 `lbfgs` : scipy L-BFGS-B, memory 30, exact energy and gradient, default line search. [che 8, pcg 9 CPU-min]
* R29 `ncg`   : scipy trust-ncg (Steihaug truncated Newton), Hessian-vector product by a one-sided difference of
  exact gradients, h = 1e-4 along the unit direction. [che 8 CPU-min]
* R28 `plbfgs`: one exact metric build at the start (polish.build), whitening theta = theta0 + M z with
  M = U (w + 1e-7 w_max)^{-1/2}, then L-BFGS in z as `lbfgs`. Build time counts. [che 9 CPU-min]

## Rule (per arm and route)
gamma = (eps_start - eps_end) / CPU-min over the whole budget; gamma_late = the same over the second half of the
budget (by CPU time).
* PASS-T1 if gamma >= 3 gamma_LM **and** gamma_late >= 2 gamma_LM: the route becomes the 12-site convergence
  optimiser candidate (next: a convergence run under the compute gate with the RULE_amend2 tail fit).
* KILL (T1) if gamma <= gamma_LM.
* AMBIGUOUS otherwise.
A route that PASSes on che but has no pcg run is PASS-T1 for che only. Reported only: eps trajectory, number of
energy/gradient calls, final gradient norm.
