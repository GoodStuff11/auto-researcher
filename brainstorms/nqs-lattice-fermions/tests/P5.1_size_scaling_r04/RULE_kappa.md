# P5.1 round 4, R51.10 `kappa`: does the ridge-regression noise-to-signal ratio predict the sample need? (pre-registered)
Written 2026-11-17 after `kappa.py` ran at the 12-site start only (`results/kappa_S4x3_start.json`: kappa_guided =
1.24e5, kappa_born = 1.48e9, d_eff = 999 at shift 1e-3, exp(H[p]) = 3035) and before it ran at any other size or
state, and before any R51.8 arm ran.

Predictor. kappa_q(V) = tr[A C_q A S] / (x^T S x) (`kappa.py` docstring) is the number of i.i.d. draws from q at
which the sampled SR step's expected squared error in the metric norm equals the exact step's squared length. For
a ridge regression this is the standard sample-complexity constant (effective dimension times residual variance
over signal). Computed exactly at each size's SR-trained start, for the guided q (R51.1's sampler).

Test (T1, interactive; needs R51.8 unit 1's N_needed at S10 and S12). Calibrate c = N_needed(12) / kappa_guided(12)
and predict N_pred(10) = c kappa_guided(10); ratio m = N_needed(10) / N_pred(10) (equivalently the measured growth
r over the predicted growth kappa(12)/kappa(10)).
* PASS if 1/1.5 <= m <= 1.5: the one-state exact predictor tracks the multi-step sample need across sizes; next,
  kappa at S14 (one exact metric build, a quick job) and a sampled kappa at 24 sites replace multi-step ladders.
* KILLED if m < 1/2.5 or m > 2.5: the predictor misses the size trend; multi-step ladders stay the only measurement.
* MARGINAL otherwise: computed again at each size's exact-arm state after 60 steps (stored theta) before deciding.
* VOID if either N_needed is censored (no calibration possible).
Reported only: kappa_born (its value is dominated by rare near-node configurations: 1.2e4 times kappa_guided at
12 sites, the heavy-tail barrier B2 in one number), d_eff, exp(H[p]).

## REVISION 1 (2026-11-17, compute auditor; before kappa ran at S12' or S14 and before any R51.8 arm ran)
Why: as written above, m = r / 1.15 is a function of R51.8's own ratio r, so R51.10 was not an independent check of
R51.8, only a restatement of it against a fixed number. The test above is withdrawn as a verdict and kept only as a
reported comparison.

Recast test (same-geometry 2-leg series, exact kappa_guided at each trained start; `submit_quick_s12p.sbatch`,
`submit_kappa_s14.sbatch`, chunked two-pass `kappa.py`): k1 = kappa(S12') / kappa(S10), k2 = kappa(S14) / kappa(S10),
against g3 = 1.728, 2.744 and the measured support growths gH1, gH2 (exp(H[p]) at the same starts).
* KAPPA-POLY if k1 <= 1.728 and k2 <= 2.744: the exact one-state noise-to-signal ratio grows no faster than V^3.
* KAPPA-SUPPORT if k1 >= gH1 and k2 >= gH2.
* KAPPA-AMBIGUOUS otherwise.
* Geometry flag (feeds RULE.md's geometry check): kappa(S12', 6x2) / kappa(S12, 4x3) outside [1/1.5, 1.5].
Use: a prediction for R51.8's ladders, recorded before they run. Afterwards, per size step, the kappa class is
compared with R51.8's measured class: a disagreement at a step where R51.8 is not AMBIGUOUS KILLS R51.10 as a
predictor; agreement at both steps PASSES it for use at 24 sites (sampled kappa). Reported: kappa_born, d_eff, exp(H).
