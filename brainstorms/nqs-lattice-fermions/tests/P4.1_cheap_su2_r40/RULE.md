# P4.1 R40: pre-registered rule (charge-only vs spin-reading backflow, both exactly projected, 12 sites)

Written 2026-11-17, round 25, after `r40.py --mode check` / `--mode time` (code checks and timings only) and
**before any pretraining or polish step**. Route card: `leaves/P4.1/cards/R40.md`.

## Question decided

The leaf asks whether restricting the backflow to charges (needed for cheap exact S) sacrifices expressivity
against a spin-reading backflow. Every polished comparison so far was charge-only against charge-only (pcg, che,
rank5). This run compares the two inputs on one architecture, at one parameter count, with both arms exactly
projected onto S = 0 and polished by the same optimiser to the same stopping rule.

## Data before this run (disclosed; all L-BFGS 2000 iterations from the common reference, 3 seeds, job 451886/508440)

| arm | input | P | eps median (seed range) |
| :-- | :-- | :-- | :-- |
| `che` | charges (d, h) | 2637 | 3.43e-4 (3.26-4.26e-4) |
| `nnbf_SG` | spins (n_up, n_dn) | 2637 | 2.07e-4 (1.89-2.31e-4) |
| `che_k8` | charges, 8 components | 2614 | 1.35e-4 (1.12-1.46e-4) |

So the single charge-only network was 1.66 times worse (seed 0: 1.49); only the 8-component charge-only mixture
(R1, killed at G4 because rank gains do not grow with size) beat the spin-reading one. The `che` seed-0 run continued
to 4000 iterations reaches 3.0585e-4.

## Arms (seed 0 only; the cheapest version)

| arm | model | start | P |
| :-- | :-- | :-- | :-- |
| `che` | `V.Model('che', 16)`, Che closed-form exact P_S | r5 `che_drift4000_seed0` theta (seed-0 common reference, 4000 L-BFGS iterations), eps 3.0585e-4 | 2637 |
| `nnbf` (= `nnbf_SG`) | `V.Model('nnbf', 16)`, full-vector exact P_S (block projector) | `--mode pretrain`: same seed-0 init (theta identical to the legacy code's), same L-BFGS-B options (maxcor 30, ftol 1e-15, gtol 1e-10), 4000 iterations, best theta | 2637 |

Why `che` and not `che_k8` or `pcg`: `che` and `nnbf_SG` are the same network with only the input swapped, so their
ratio is the cost of the restriction itself. `che_k8` adds rank, a finite-size gain (R1, L10). `pcg` adds a pairing
frame and was polished from a different (SR) start, so its ratio to `nnbf_SG` would mix three differences; its
r22 endpoint (4.23e-4) is reported next to the result, but it decides nothing.

## Protocol

1. `nnbf` pretraining (above). INVALID if: the init energy differs from the legacy code's value at the same theta
   (-6.2824656105) by more than 1e-8; the run is cut by its CPU cap (360 CPU-min); or eps at iteration 2000 is outside
   [0.8 x 1.885e-4, 1.2 x 2.306e-4] (the legacy 3-seed range, widened by 20%).
2. Both arms: cutoff-ladder polish, the protocol of `../P4.1_cheap_su2_r22/ladder.py` unchanged (ladder 1e-5..1e-9,
   LM |z| <= 0.05, line search {1, 1/2, 1/4} on the exact energy with minimum gain 1e-7 in eps, best rung accepted).
   Stops: STOP-RULE (accepted step's predicted gain < 1e-3 g, with **one common g = 2.5e-4 |E0|** for both arms, so
   the absolute stop is identical), STOP-NOGAIN, or the **hard cap of 30 steps** (STOP-CAP).
3. Convergence guard: the r22 RULE_amend2 guard unchanged (log-linear fit of the last 5 accepted gains; the arm is
   UNSETTLED if r >= 1, RMS log-residual > ln 2, or eps_inf <= 0).

## Verdict: rho = eps_che / eps_nnbf, classified at the endpoints and tail-extrapolated

| verdict | condition | what it means |
| :-- | :-- | :-- |
| NO-LOSS | rho <= 1.1 | no measurable expressivity loss from the charge-only restriction at 12 sites |
| INTERMEDIATE | 1.1 < rho < 1.5 | a loss under 1.5 times, not resolvable from none on one seed |
| LOSS | rho >= 1.5 | the restriction costs at least 1.5 times the energy error at 12 sites |

The class must agree at the endpoints and tail-extrapolated, with neither arm UNSETTLED; otherwise UNSETTLED, and at
the 30-step cap AMBIGUOUS: no change to the leaf and no further compute on this question at 12 sites.
INVALID (any check above, or a start energy off its record by more than 1e-8) and INCOMPLETE (an arm cut by the CPU
cap before a stop) change nothing.

## What each verdict does

* **LOSS**: the Answer's accuracy clause becomes "yes, at 12 sites: after polishing, the charge-only network has
  rho times the error of the same network reading spins; only a non-scalable 8-component mixture made up for it".
  The leaf's Question is answered No at 12 sites for the single-Gaussian class (the cheap-S class).
* **INTERMEDIATE**: Answer: "a loss of 10-50% at 12 sites, one seed". Seeds 1-2 (a new gated job, about 2/3 of this
  cost each) are funded only if rho lies within 0.1 of a threshold.
* **NO-LOSS**: Answer: "no measurable loss at 12 sites"; the 451886 1.66x gap was the optimiser (L33).
* **AMBIGUOUS / INVALID**: no change; the Answer keeps "not known", citing this run.

## What a 12-site result cannot say

P = 2637 against an S = 0, (+,+) sector of about 42,500 states (169,884 / 4), so P/dim is about 0.06, below the
sector-fitting regime of L36. But 12 sites is an exact-sum cluster (L39): spin correlations longer than the cluster,
which a spin-reading backflow might exploit at larger sizes, do not exist here. A LOSS here is evidence the
restriction costs accuracy; NO-LOSS does not transfer to 24+ sites. The comparison is at matched parameters, not
matched cost: `nnbf_SG`'s exact projection is exponential in the number of singly occupied sites (L35: this gate
scores parameter efficiency only, which is what the Question asks).
