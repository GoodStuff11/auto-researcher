# P4.1 round 23: R16 (`sq`) and R17 (`nj`) T1 tangent rule at the ladder-polished parents

Written 2026-11-17 before any run output of this directory existed.

## What is run

The Round 10 T1 code, unchanged: `../P4.1_cheap_su2_r10/tangent.py` (its "Pre-registered rule" block, families
C1, C3, SQ, W, NJ, RF, validation checks and controls), driven by `t1_polished.py`, which only swaps the parent
parameters and the stored energy:

* `che`: V.Model('che', H=16), theta = `../P4.1_cheap_su2_r22/results/theta_che.npy` (job 742206, 12 ladder steps,
  STOP-CAP), stored E = state_che.json last step `E_new` = -7.822249354501926 (eps 7.217e-4).
* `pcg`: T2.DetModel('pcg', H=15), theta = `../P4.1_cheap_su2_r22/results/theta_pcg.npy` (9 steps, CPU cap),
  stored E = state_pcg.json last step `E_new` = -7.824587605391688 (eps 4.230e-4).

## Rule (the card's, thresholds unchanged)

Gains in units of the parent's exact gap g = E(Psi) - E0:
G_SQ = [Delta(C1+C3+SQ) - Delta(C1+C3)]/g, G_W = [Delta(C1+C3+W) - Delta(C1+C3)]/g,
G_NJ = [Delta(C1+NJ) - Delta(C1)]/g, G_RF = [Delta(C1+RF) - Delta(C1)]/g.
Per route over che and pcg: KILL if G_route <= G_foil on both; PASS if G_route >= 2 G_foil and G_route >= 0.05 on
both; AMBIGUOUS otherwise. INVALID if any tangent.py validation check or control fails (P_G overlap < 1 - 1e-10,
|E(Psi) - stored| > 1e-8, tangent <S^2> > 1e-10, positive control off by > 1e-8, negative control G >= 0.05).

## Effect of the unconverged parent (stated before running)

The rule is a first-order test: it asks whether the new directions lower the exact energy beyond a matched foil
in span{Psi, T}. At a stationary parent, gains from the parent's own directions would be zero and all gain from
SQ/NJ would be genuinely new. Both parents are not stationary (742206 fitted tails: che predicted gain still
4.4e-3 g per step, decaying x0.89; pcg 0.025 g per step). Consequences:

1. Any family that overlaps the parent's own unexhausted gradient picks up residual descent that is not a property
   of the route. The rule already guards against this by subtracting the C1+C3 (parent Jastrow + orbital shift)
   baseline and comparing against a parameter-matched foil (W, RF) that is equally able to pick up the residual.
   The comparison route-vs-foil is therefore still the meaningful quantity; the absolute G values are inflated
   by residual descent for both route and foil alike.
2. The residual is reported as Delta(C1+C3)/g; it is not used in the verdict.
3. Direction of bias: residual descent adds a common term to G_route and G_foil, which compresses their ratio
   toward 1. This makes PASS (needs 2x) harder and KILL (needs route <= foil) neither easier nor harder. A KILL is
   therefore read as a valid kill; a PASS is read as conservative; AMBIGUOUS carries the caveat that a converged
   parent could separate the families further.
4. Thresholds are kept as on the card. No new parent will be polished in this round.

## Budget and decision order (HARD LIMIT 10 CPU-minutes, interactive)

Round 10 timings (4 threads): che 6.2 CPU-min, pcg 7.5 CPU-min (total 13.7 > 10). Run with 1 thread to reduce
thread overhead. Order: che first (measured by /usr/bin/time). Decision logic for what pcg can change:

* If che gives, for a route, G_route > G_foil but not (G_route >= 2 G_foil and G_route >= 0.05), that route is
  AMBIGUOUS whatever pcg gives (KILL and PASS both need che): pcg is not needed for that route.
* Otherwise pcg is needed for that route. pcg is run only if che's measured CPU plus pcg's predicted CPU
  (che CPU x 7.5/6.2) fits in the 10 CPU-minute total; else its cost is reported and the route stays
  AMBIGUOUS-pending (no verdict).
