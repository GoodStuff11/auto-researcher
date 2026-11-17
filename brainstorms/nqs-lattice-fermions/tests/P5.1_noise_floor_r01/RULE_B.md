# P5.1 round 1, R1 `nfloor` T2: does the plateau of sampled SR fall with samples per step? (pre-registered)
Written 2026-11-17 after RULE_A's T1a (stored data) and the 4-step pre-flight, before any arm below ran.

Background (RULE_A, T1a): in R38's guided runs (job 752885, 16,384 samples, held-out step-length choice) the
plateau at R about 0.51 was mostly the step-length choice: at steps 41-60 the smallest candidate (metric length
0.005) never raised the exact energy (0 of 60 steps) and gained 0.0078 +- 0.0006 of the gap per step, while the
held-out choice realised 0.0020 (it picked 0.01, which raised the energy in 68% of those steps). So the test below
fixes the metric length and always takes the step (no selection), and asks whether what remains is a sampling
floor. R = fraction of the gap (SR-trained `che` energy minus exact ground energy) removed, scored exactly.
R_pol = 0.7475 is the exactly ladder-polished `che` (job 742206).

Arms (`nf_run.py`, start = SR-trained `che`, guided draws, shifted SR direction at fixed metric length tau):
* A: N = 16,384, tau = 0.005, 120 steps.     * B: N = 65,536 (4N), tau = 0.005, 120 steps.
* C: N = 262,144 (16N), tau = 0.005, 120 steps.     * D: N = 16,384, tau = 0.0025, 240 steps.
Release order: unit 1 = A and B (seed0 9000); unit 2 = C and D, only if unit 1 does not decide REACHES or
NOT-NOISE below; unit 3 = a second seed (9500) of the arms whose verdict margin is within 2 replica spreads
(spread from R38: sd of R(60) = 0.015).
Plateau value R_plat(arm) = mean R over its last 20 steps; deficit Delta = R_pol - R_plat; tail slope = mean dR
over the last 20 steps. Rules, applied after each unit:
* REACHES at N if R_plat(A) >= R_pol (unit 1): fixed-length guided SR at 6 samples per parameter already reaches
  the polished baseline; the plateau of R38 was the step-length choice. Next: total samples to R_pol against the
  exact ladder's CPU cost and against plain SR; then the 24-site protocol.
* NOISE-FLOOR if Delta(B)/Delta(A) <= 0.62 and (after unit 2) Delta(C)/Delta(B) <= 0.62 or Delta(C) <= 0: the
  plateau falls at least as N^-0.35. Report the fitted exponent a from A, B, C. A scalable optimiser then needs
  samples per step growing as (target deficit)^(-1/a), or a variance reduction that buys the same factor; routes
  that change the conditioning only are deprioritised.
* NOT-NOISE if Delta(B)/Delta(A) >= 0.85 and (after unit 2) Delta(C)/Delta(A) >= 0.75, with Delta(C) > 0.05: 16
  times the samples moves the plateau by less than a quarter, so the floor is not sampling noise but the landscape
  (ill-conditioning, L60) or the fixed length; routes that change conditioning are prioritised.
* STEP-NOISE (from D, reported with the above): if Delta(D) <= 0.75 Delta(A) while D's tail slope is >= 0, the
  floor scales with the step length, i.e. it is the noise ball of a constant step; R7 (decaying step length) is
  promoted to a T1 run.
* AMBIGUOUS otherwise; INCOMPLETE if an arm stops before its last step.
An arm whose tail slope over the last 20 steps exceeds 0.002 per step is reported as "not yet plateaued", and its
R_plat is a lower bound of the plateau; NOT-NOISE is then not declared from it.
