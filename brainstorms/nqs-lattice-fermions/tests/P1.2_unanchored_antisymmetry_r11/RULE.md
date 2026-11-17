# P1.2 r11: t1c training run of the linear phase carrier on span S (reopening ledger row 4)

Written 2026-11-17 BEFORE `t1i_phaseS.py` was run. This is the follow-up that the r10 rule
(`../P1.2_unanchored_antisymmetry_r10/RULE.md`) prescribes after its FAMILY-OPEN: "a linear phase on S,
warm-started along the lambda(S) eigenvector", trained and scored under the t1c rule. T1, interactive if the
measured cost is at most 10 CPU-minutes (the run is single-thread; RLIMIT_CPU per invocation).

## Carrier and arms (sizes 1 = 6x2 5+5 and 2 = 4x3 5+5; foil seeds 11 and 22; unprojected, as R9.1)
psi(n) = psi_F(n) exp(i phi(n)), psi_F the frozen trained t1c foil (NNBF width 6), exactly as RPC (R9.1,
`t1f_phase.py`), with the energy `t1f_phase.e_and_dphi` and the sign read-out s(n) = sgn(psi_F(n) cos phi(n)).
S is built by the r10 code (`t1h_curv2.py`), unchanged.

* `S` (the route): phi = B c, with B the psi^2-centred, psi^2-whitened basis of span S (rank 1464 at 6x2,
  1760 at 4x3). Start: c0 = +-eps v, v the lambda(S) eigenvector, eps such that the psi^2-weighted rms of phi is
  0.1; L-BFGS-B on the exact energy, maxiter 300; the lower-ENERGY of the two signs is kept (no oracle).
* `rand` (capacity control): the same on an equal-rank random +-1 span (the r10 control), started along its own
  lambda eigenvector.
* `Sb` (sign-capacity ablation arm): the same span and start, but phi = (pi/2)(1 - 1e-6) tanh(B c), so
  |phi| < pi/2 and cos phi > 0 everywhere: the carrier keeps all its phase capacity but cannot flip a sign.
  Its dEs equals the foil's by construction; what it reports is how much of `S`'s complex-energy gain needs no
  flip at all.
* `S-nn` (flips-removed read-out of the trained `S`): the `S` read-out with every flip reverted on near-node
  configurations, N = {n : max_e |R_e(n)| >= 10} (R_e the exact foil edge ratio of r10; threshold fixed here).
  This is the leaf's own lesson (R5.1, "credit a sign-layer gain only after a flips-off read-out"), applied to
  the post-hoc r10 finding that the lambda(S) eigenvector sits on 12 to 86 near-node configurations.

Comparator: the stored t1e `wid` arm (foil widened to 1255 parameters, +145, 300 more SR steps;
`../P1.2_unanchored_antisymmetry_r6/results/t1e/A_<size>_wid_<seed>.json`), as for R9.0 and R9.1.

Matched parameters. S is NOT weight-shared: `S` has 1464 / 1760 free coefficients against `wid`'s 145 added
parameters, so the comparison is biased toward the route. S contains every weight-shared carrier that is linear
in its columns, and a weight-shared carrier at <= 145 parameters (the matched arm) can only do worse than the
optimum over S. Hence a KILL of `S` is a matched-parameter KILL a fortiori (up to the optimiser reaching the
optimum on a superset), while a PASS-W of `S` is not yet a matched-parameter pass.

## RULE (the t1c rule of the leaf, as used for R8.3 to R9.1; median over seeds 11 and 22; counted sizes are
those whose foil median dEs >= 0.005, i.e. sizes 1 and 2; the largest is size 2)
r = dEs[S] / dEs[wid].
* KILL if r >= 0.9 at both counted sizes (">= 2 counted sizes including the largest").
* PASS-W if r <= 0.7 at both counted sizes and r(2) <= r(1) + 0.1.
* UNDECIDED otherwise.
Credit, applied only after a PASS-W (all three needed for SIGN-CREDITED; otherwise PASS-W NOT CREDITED):
* (a) non-specific capacity: dEs[S] <= 0.9 dEs[rand] at both counted sizes;
* (b) flips-removed: r_nn = dEs[S-nn] / dEs[wid] <= 0.7 at both counted sizes (the gain does not live on
  near-node spikes);
* (c) sign capacity used: dE_complex[S] < dE_complex[Sb] at both counted sizes (`S` reaches an energy that a
  flip-free phase of the same span cannot).
Validity (else INVALID): CHECK A of `t1f_phase.setup` (rel < 1e-10); E[phi = 0] equals the foil energy
(< 1e-12); the analytic gradient of the `S` objective matches central finite differences (rel < 1e-5) on 5
random coordinates at size 1, seed 11.

## What each verdict does
* KILL: ledger row 4 becomes KILLED(T1): every linear phase carrier on S (and so every weight-shared linear
  carrier on these columns) is killed on the frozen foil. Row 9 (V > 12) needs a T1 PASS-W first, so it closes
  as GATE-REFUSED (precondition never met). All rows are then closed, item 5 of the closure certificate can be
  filled, and the leaf goes from DORMANT to EXHAUSTED at this budget. Lesson: a negative lambda(S) carried by
  near-node spikes does not train into a sign gain; the screen needs a sup-normalised q before it opens a
  family.
* UNDECIDED: row 4 is recorded UNDECIDED(T1). If r(1) > 0.7 or r(2) > 0.7 with the other >= 0.9, PASS-W is
  out of reach of the only remaining lever (more foil seeds; no other seed has a saved foil), so row 4 closes as
  GATE-REFUSED(G2, PASS-W unreachable), as R9.2 did, and the leaf goes to EXHAUSTED. Otherwise (both in
  (0.7, 0.9)) one extra foil seed pair is the only follow-up, through the compute gate.
* PASS-W NOT CREDITED: row 4 stays open with a note (the gain is capacity, spikes, or flip-free phase);
  leaf stays DORMANT; no T2.
* PASS-W SIGN-CREDITED: row 4 stays open and moves to its matched-parameter step: the weight-shared restriction
  of S at <= 145 parameters under this same rule, then the I2 attribution control (features O). The leaf goes
  to ACTIVE. No T2 before both.

## Prediction (recorded before the run)
KILL with p ~ 0.85 (UNDECIDED 0.13, PASS-W 0.02). Reasons: S is a subspace of the unrestricted phase, whose
L-BFGS optimum reached only r_free 1.17 at 6x2 and 0.77 at 4x3 (R9.1); the lambda(S) eigenvector's realisable
gain q/max|g|^2 is 30 to 300 times below the free phase's (r10 `t1h_conc`), and S's captured part of the free
direction ascends at 4x3 (r10 `t1h_proj`). This KILL changes the decision (row 4 closes, leaf EXHAUSTED).

## Post-run (added after the verdict; does not change it)
Run interactively (measured projection 5.3 CPU-min, `t1i_time_output.txt`; actual about 3 CPU-min including checks
and timing), no sbatch, no compute-gate entry. Checks pass (`t1i_checks_output.txt`: CHECK A 0, E[phi=0] 0,
gradient FD 4e-8, lambda(S) reproduces r10's -9.7317). Verdict by the rule: **KILL** (`t1i_verdict_output.txt`).

| size | foil | wid | S | r_S | S-nn | r_S-nn | rand | Sb |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 (6x2) | 0.01107 | 0.00850 | 0.00918 | 1.08 | 0.01070 | 1.26 | 0.01107 | 0.01107 |
| 2 (4x3) | 0.04962 | 0.04267 | 0.04327 | 1.01 | 0.04822 | 1.13 | 0.04962 | 0.04962 |

* Unlike RPC, `S` does leave phi = 0 and flips signs (321 to 665 configurations, psi0 weight 0.002 to 0.009), and
  lowers dEs below the foil by 17% (6x2) and 13% (4x3), but less than width does at 10x fewer parameters.
* Most of that gain sits on near-node flips (75 to 90% of the flipped psi0 weight is in N); reverting them returns
  dEs to within 3% (6x2) and 3% (4x3) of the foil.
* `rand` never leaves the foil (no flips); `Sb` reaches nearly the same complex energy as `S` without any flip
  (dE_complex 0.0716 vs 0.0714 at 6x2, 0.1614 vs 0.1606 at 4x3): S's descent is mostly flip-free phase.
* L-BFGS stopped at about 40 iterations by its relative-f criterion; a post-hoc retrain at size 2 with ftol 1e-15,
  gtol 1e-12 (`t1i_tol_posthoc_output.txt`) converges to the same point (dEs 0.04303 and 0.04355).
