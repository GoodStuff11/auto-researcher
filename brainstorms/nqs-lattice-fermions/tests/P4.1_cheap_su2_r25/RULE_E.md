# P4.1 round 24: R39 `js-force` one-step screen (pre-registered)

Written 2026-11-17 before `shrink_screen.py` was run. Protocol = RULE_B.md exactly (`che` SR-trained theta0 of R21,
16,384 i.i.d. draws from p = size * pr^2, seeds 3001-3003 with the same draw order as noise_split.py, so V1 and V11
reproduce noise_split's base 0.016 and guided 0.086; absolute shift 1e-3; direction rescaled to exact-metric length
tau in {0.01, 0.02, 0.03}; R_best = max over tau of the exact-energy gap fraction; Rbar = mean over the 3 seeds).
Reference values from RULE_B's run: base Rbar 0.016 +- 0.022, exact force (V3) 0.171 +- 0.015.

## Estimators (force shrinkage only; the metric is the sampled S_T)
Eigenbasis (w_k, u_k) of the sampled S_T. Force components ft_k = u_k . f. Their sampling variances are estimated
from the same samples: with normalised weights om_i (1/N for p-draws, p/q self-normalised for guided draws),
z_i = (O_i - Obar)(E_i - Ebar), var_k = sum_i om_i^2 (u_k . z_i - ft_k)^2. Whitened components g_k = ft_k/sqrt(w_k+1e-3).
* J1 mode-wise positive-part James-Stein (empirical-Bayes Wiener filter): ft_k <- ft_k max(0, 1 - var_k/ft_k^2).
* J2 grouped positive-part James-Stein: modes sorted by w_k into 32 contiguous groups; in each group b,
  g <- g max(0, 1 - ((n_b - 2)/n_b) sum_b var_k/(w_k+1e-3) / sum_b g_k^2) (shrinkage toward zero, per group).
Direction d = -sum_k u_k ft'_k/(w_k + 1e-3). (Whole-vector James-Stein toward zero is a scalar rescaling, which the
fixed-length scoring removes, so it is not tested: T0 kill of that variant.)
Reported only (not in the rule): J1 and J2 applied to the guided (V11) estimator, run only if CPU allows (cap 10 CPU-min).

## Rule (on Rbar of J1 and J2, from p-draws)
* PASS-T1 if max(Rbar(J1), Rbar(J2)) >= Rbar(V1) + 0.05 (as RULE_B): R39 goes to a multi-step test on top of
  guided SR (60 steps, as RULE_D), as a gated job.
* KILL (T1) if max(Rbar(J1), Rbar(J2)) <= Rbar(V1) + 0.02 (this also covers "less than twice the base",
  0.032 < 0.036): force shrinkage does not recover the force-noise headroom; ledger row 3 closes KILLED(T1).
* AMBIGUOUS otherwise: the row stays open with the numbers; a second start point decides.
