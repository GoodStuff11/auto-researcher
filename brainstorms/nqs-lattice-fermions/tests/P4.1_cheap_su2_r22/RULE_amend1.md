# P4.1 r22: pre-registered amendment 1 (stopping rule only)

Written 2026-11-17, after job 742206 returned INCOMPLETE and **before any further polish step**. It amends
`RULE.md` Protocol item 5 (step cap) and the UNSETTLED follow-up only. Everything else in RULE.md stands unchanged.

## Disclosure

This amendment was written **after seeing the provisional 742206 ratios**: rho_pol = 0.586 at the endpoints
(0.610 against che's tail-extrapolated eps_inf 6.930e-4), rho_abl = 0.550. Arms at that point: pcg 9 steps
(eps 4.230e-4, stop null, CPU cap), rank5 8 steps (7.693e-4, stop null, CPU cap), che 12 steps (7.217e-4, STOP-CAP).
It changes only the stopping rule and applies it symmetrically to all three arms. Its motive is that under the
fixed 12-step cap no arm reaches the predicted-gain stop (decay in the 742206 logs needs about 15 more steps for pcg,
10 for rank5, 13 for che), and an UNSETTLED read under the guard would have no follow-up because the cap leaves
nothing to resume.

## Amendment

1. **Stopping rule.** The 12-step cap (RULE.md Protocol item 5, STOP-CAP at 12) is replaced, for **all three arms
   symmetrically** (`pcg`, `che`, `rank5`), by: run each arm with the unchanged per-step ladder protocol until the
   pre-registered predicted-gain stop (accepted step's predicted gain < 1e-3 g; STOP-RULE), STOP-NOGAIN, or a
   **hard cap of 30 total steps per arm** (STOP-CAP at 30; steps already taken in 742206 count). `che`, which ended
   STOP-CAP at 12, is resumed from its saved state like the others.
2. **Verdicts.** Read on those endpoints with the **unchanged** RULE.md thresholds (rho_pol: KEEP-LIVE <= 0.60,
   RETRACT >= 0.85; rho_abl: PAIRING-KEPT <= 0.60, PAIRING-RETRACTED >= 0.85) and the **unchanged**
   geometric-tail convergence guard (applied to arms ending STOP-CAP). If a ratio is still UNSETTLED at the hard
   cap, the result is recorded as **AMBIGUOUS**: no change to R2 or the leaf, and **no further compute** on this
   question at 12 sites.
3. **INVALID / INCOMPLETE** are unchanged (an arm cut by wall time or CPU cap before STOP-RULE, STOP-NOGAIN or the
   30-step cap is INCOMPLETE and resumable).

## Code change (made with this amendment, before any run)

`ladder.py`: an arm whose saved stop is STOP-CAP with fewer steps than `--steps` is reopened (STOP-RULE,
STOP-NOGAIN and INVALID stay final); `--mode verdict` labels an UNSETTLED joint read AMBIGUOUS when every STOP-CAP
arm has reached 30 steps. The resume runs with `--steps 30`. No change to the ladder, line search, stop fraction,
thresholds or guard.

Code-test disclosure (after this amendment was written): the reopen path was tested once on a scratch copy of
`results/` (`--arm che --steps 13`, deleted afterwards; `results/` untouched). That one step took che from 7.217e-4 to
7.035e-4 (predicted gain 7.5e-3 g, absolute gain larger than step 12's, so a guard tail at that point would be
unbounded). It is not a result; the gated job recomputes it from the saved state.
