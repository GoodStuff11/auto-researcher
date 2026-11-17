# P4.1 R21 `trSR` T1 (round 21, dir r24): pre-registered rule

Written 2026-11-17, before any output of this run (including the `exact0` diagnostic and the timing probe) existed.

## Card rule, verbatim (`bs.py show <wd> R21`, Lineage field)

> Cheapest falsifier (T1, about 12 CPU-minutes, so a quick-note batch job of at most 1 CPU-hour): from `che`'s trained parameters, 15 SR steps with 16,384 i.i.d. exact samples per step, relative cutoff $10^{-4}$, metric step length 0.05, against a control arm with the training's absolute shift $10^{-3}$; every step scored by the exact energy, no in-sample line search (L52). Pre-registered: VIABLE if $R_{\mathrm{trSR}}(15)\ge0.20$, at least 0.15 above the control, and no step raises the exact energy by more than 0.02 of the gap; NOT-VIABLE if $R_{\mathrm{trSR}}(15)<0.08$; MARGINAL otherwise. VIABLE sets the 24-site protocol (with the critic's agreement); NOT-VIABLE closes sampled polishing at about 6 samples per parameter, and R2 can then be decided only at 12 sites (R19, R20).

## System (identical to r19 / r20)

* Open 4x3 Hubbard strip, N_e = 10, U = 8, t' = -0.25, sector (+,+), exact S = 0. E0 = -7.827899034271753.
* `che` SR-trained pilot, P = 2637, theta0 = its trained parameters; gap g = E(theta0) - E0 = 0.0223728.
* R(n) = (E(theta0) - E(theta_n)) / g, E always the exact sum over all 627,264 configurations.
* Sampling: i.i.d. representatives from p(r) proportional to |orbit(r)| Psi(r)^2 (as r19/r20), N = 16,384 per step
  (6.2 per parameter), seed = step number (common random numbers at step 1 across arms).

## Interpretation of the card's terms (fixed here, before any output)

* SR quantities from the samples: O_i = d ln Psi / d theta_i on the drawn rows, S_T = cov(O, O),
  f = cov(O, E_loc) (standard SR force; no H-matrix estimator needed).
* `trsr`: eigen-decompose S_T, keep eigenvalues > 1e-4 x w_max, d = -0.05 Pm g~ / |g~| with Pm = U_k / sqrt(w_k),
  g~ = Pm^T f. So the step's length in the sampled metric, sqrt(d^T S_T d), is exactly 0.05 ("metric step length").
* `shift` (control): d proportional to -(S_T + 1e-3 I)^{-1} f, rescaled to the same metric length 0.05 in the
  sampled S_T. Only the regulariser differs between the arms.
* Every step is taken: no acceptance test, no line search of any kind. "No step raises the exact energy by more
  than 0.02 of the gap" means max over steps of (E(theta_n) - E(theta_{n-1})) / g <= 0.02.

## Verdict (trsr vs shift)

| Verdict | Condition |
| :--- | :--- |
| VIABLE | R_trsr(15) >= 0.20, and R_trsr(15) - R_shift(15) >= 0.15, and no trsr step raises E by more than 0.02 g |
| NOT-VIABLE | R_trsr(15) < 0.08 |
| MARGINAL | anything else |
| INCOMPLETE | either arm stopped by the CPU cap before 15 steps; no verdict; resume from results/*.json |
| INVALID | E(theta0) differs from the stored value by > 1e-8, or a step-1 sampled E is > 6 se from exact; no verdict |

What each verdict does is the card's: VIABLE sets the 24-site protocol (critic's agreement); NOT-VIABLE closes sampled
polishing at about 6 samples per parameter, and R2 can then be decided only at 12 sites (R19, R20).

## Reported-only third arm (not part of the verdict)

* `ladder`: the same sampled SR step at metric length 0.05, with the relative cutoff chosen per step from
  {1e-4, 1e-5, 1e-6, 1e-7, 1e-8} by the correlated-sampling energy change on a fresh held-out set
  (N = 16,384, seed 1000 + step), never by the exact energy. The argmin candidate is taken if its held-out dE < 0;
  otherwise the step is skipped (theta unchanged; it still counts as one of the 15).
* Reason: L29 (fixed relative cutoffs can freeze steps; P4.2 r10, P4.1 r21 R19 probe).
* The exact energy of every candidate is logged as an oracle diagnostic; it never enters the choice.
* Its R(15) and maximum rise are reported beside the verdict. It changes no verdict and no threshold.

## Free diagnostic (decides nothing)

* `exact0`: from Round 11's exact S and H at theta0 (`tests/P4.1_cheap_su2_r11/results/S_H_theta0.npz`), the
  infinite-sample step 1 of both arms at metric length 0.05 (and 0.02, for reference), scored by the exact energy.
