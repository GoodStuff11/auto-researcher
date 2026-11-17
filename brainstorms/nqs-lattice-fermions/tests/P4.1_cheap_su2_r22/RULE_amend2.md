# P4.1 r22: pre-registered amendment 2 (convergence guard only)

Written 2026-11-17, after RULE_amend1.md and **before any further gated polish step**. It replaces the RULE.md
convergence guard (two-gain geometric tail). Everything else (ladder protocol, thresholds, RULE_amend1 stopping
rule and 30-step hard cap, AMBIGUOUS rule) is unchanged. Applied identically to all three arms.

## Disclosure

Written after seeing: the provisional 742206 ratios (rho_pol 0.586 / 0.610, rho_abl 0.550), and one throwaway che
step 13 run on a deleted scratch copy (7.217e-4 -> 7.035e-4; RULE_amend1 disclosure). Motive: the two-gain guard
reads an unbounded tail whenever the last gain exceeds the previous one, and the arms' gains alternate (che's rose at
steps 8, 11 and 13), so about 0.4 of capped che endpoints would read UNSETTLED from one step of noise. The residual
threshold ln 2 was fixed after computing the fit on the 742206 trajectories below (pcg's RMS is 0.682, just under
it); the 5-gain window and the log-linear form were set by the coordinator before any fit was computed.

## New guard (arms ending STOP-CAP; STOP-RULE and STOP-NOGAIN arms keep tail 0)

1. Take the last 5 accepted gains d_k (energy units) of the arm. Least-squares fit ln d_k = a + b k (k = step index).
2. r = exp(b). Fitted tail = d_hat_n r / (1 - r), with d_hat_n = exp(a + b n) the fitted last gain.
   eps_inf = eps_end - tail / |E0|.
3. The arm's extrapolation is **UNSETTLED** if any of: fewer than 5 positive gains; r >= 1; RMS of the log
   residuals > ln 2 (gains scatter about the fit by more than a factor 2); eps_inf <= 0 (fitted tail exceeds the
   remaining error, so the geometric model is inconsistent with eps >= 0).
4. A ratio is UNSETTLED if either of its arms is UNSETTLED under 3, or if its class at eps_end differs from its class
   at eps_inf (unchanged rule). At the 30-step hard cap an UNSETTLED ratio is AMBIGUOUS (RULE_amend1).

## Fit on the 742206 trajectories (code test, verdict mode on a scratch copy with all arms marked capped)

| arm | last 5 gains (1e-5 eps) | r | RMS | eps_end | eps_inf |
| :-- | :-- | :-- | :-- | :-- | :-- |
| pcg (9 steps) | 5.57 4.56 1.02 7.37 2.73 | 0.910 | 0.682 | 4.230e-4 | 1.304e-4 |
| rank5 (8 steps) | 10.67 23.38 6.16 9.06 4.45 | 0.764 | 0.413 | 7.693e-4 | 5.981e-4 |
| che (12 steps) | 2.17 2.07 1.93 2.67 1.68 | 0.975 | 0.147 | 7.217e-4 | <= 0 -> UNSETTLED |
| che + scratch step 13 | 2.07 1.93 2.67 1.68 1.83 | 0.962 | 0.148 | 7.035e-4 | 2.314e-4 |

## Effect on the 0.4 UNSETTLED risk

The single-up-tick failure is gone: che's alternation now gives a smooth fit (RMS 0.15), and one larger gain moves r
by about 0.01, not to infinity. But the honest fit shows the arms are far from geometric convergence (r 0.91-0.975),
so the fitted tails are large: 69% of pcg's eps, 22% of rank5's, 67-100% of che's. The UNSETTLED risk therefore
moves from "noise in one step" to "real slow convergence": a capped che at step 30 with r about 0.96 still carries a
tail comparable to its eps, and the end and tail classes of rho_pol can differ. Estimated P(rho_pol UNSETTLED ->
AMBIGUOUS) about 0.45 (was about 0.43 under the old guard, from a different cause); P(rho_abl UNSETTLED) about 0.25
(rank5 likely reaches STOP-RULE, so only pcg's tail matters there).
