# P4.1 round 22, sub-question B: which sampled-step estimator recovers the exact SR step (pre-registered)

Written 2026-11-17 before `noise_split.py` was run. Instance and start point are R21's: `che` trained parameters
(ctx_for("che").th0, the SR-trained 12-site state), where the exact-sum SR step 1 with absolute shift 1e-3 at metric
length 0.02 realises R = 0.288 of the gap and the sampled one (16,384 samples) about 0.02-0.04 (R21 probes).

## Method (one step, at theta0, exact energies)
Full orbit-weighted tangents Y and H Y over all 157,216 representatives are built once (polish.build). A sampled
estimator is a row subset with i.i.d. draws r ~ p (size * pr^2), N = 16,384, seeds 3001-3003 (disjoint from R21's
1-15 and 1001-1015). From a subset: S_T = <O O> - <O><O>, f = <O E_loc> - <O><E_loc> (R21's estimators).
Every direction d = -(S' + 1e-3 I)^{-1} f' for the variant's S', f', then scaled to exact-metric length
tau in {0.01, 0.02, 0.03} (sqrt(d^T S_T,exact d) = tau), and scored by the exact energy: R = (E(theta0) - E)/gap.
R_best = max over tau (oracle length; a direction-quality score, not an optimiser). Rbar = mean over the 3 seeds.

Variants: V1 base; V2 exact S + sampled f; V3 sampled S + exact f; V4 S from 4N + f from N (bound for R30, reuse
of stale rows at zero staleness); V5 plain 4N (bound for R29, reweighted fresh-sample reuse); V6 Ledoit-Wolf linear
shrinkage of S_T toward its diagonal; V7 NERCOME split-sample eigenvalue correction; V8 top-256 eigenpairs plus
diagonal residual (factor model); V9a block-diagonal S over 21 blocks (P0, jv, W1, b1, b2, W2 per hidden unit);
V9b two-stage blocked step (V9a block directions recombined by a 21x21 sampled solve, Zhao-Neuscamman style);
V10 control-variate force f_beta = (1-beta) f + beta f_alt, f_alt = <(H O psi)/psi> - <O><E_loc>, beta 0.5 and 1;
V11 guided sampling q = 0.8 p + 0.2 p_flat (p_flat ~ size*|pr|), self-normalised weights.
Reported only: trunc (cutoff 1e-4) for V1 and V2; cosine to the exact direction in the exact metric.

## Rule
* Class bounds. If Rbar(V2) < Rbar(V1) + 0.05, a perfect metric does not help: every metric-side route (R30, R31,
  R27-blocked) is KILLED (T1). If Rbar(V3) < Rbar(V1) + 0.05, every force-side route (R33, R34) is KILLED (T1).
* Each variant V: PASS-T1 if Rbar(V) >= Rbar(V1) + 0.05; KILL if Rbar(V) <= Rbar(V1) + 0.02; AMBIGUOUS otherwise.
  Extra conditions: V5 (R29) and V4 (R30) bound their routes from above (perfect reuse), so a KILL there kills the
  route; a PASS only promotes it to a multi-step staleness test. V10 (R33) must also reach a variance reduction of
  the force (metric norm (S_T,exact + 1e-3)^{-1}, 16 seeds, no energies) of at least K, the mean number of Hamiltonian
  connections per sampled row (its cost multiplier), or it is KILLED at equal cost.

## Erratum (2026-11-17, per-leaf critic round 23, objection 10; appended, rule text above unchanged)

The route numbers above are the round-22 working numbers, renumbered before the cards were written. Read them as:
old R30 (V4, stale-metric reuse) = card R35 `rls-S`; old R29 (V5, fresh-sample reuse) = card R34 `reuse`;
old R33 (V10, control variate) = card R37 `cv`; old R27-blocked (V9a/V9b) = card R32 `bsr`; old R31 (metric side,
V6-V8 shrinkage) = card R36 `shrink` (by variant). Old R34 in the force-side class bound has no card mapping from
the critic; read it by variant in log `leaves/P4.1/log/round-22.md`. The class bounds are no longer used as kill
reasons: same-sample S and f errors partly cancel, so exact S with sampled f is not an upper bound (cards R32, R36).
