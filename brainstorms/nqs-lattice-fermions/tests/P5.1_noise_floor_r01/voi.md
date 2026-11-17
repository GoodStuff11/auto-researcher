# VOI note: P5.1 round 1, R1 `nfloor` unit 1 (`submit_unit1.sbatch`; rule RULE_B.md)

* leaf: P5.1
* route: R1
* tier: T2
* decision: arms A (16,384 samples per step) and B (65,536) of fixed-length guided SR from the SR-trained 12-site `che`, 120 steps each, exact scoring. REACHES (arm A plateau at or above the polished R 0.7475) makes fixed-length guided SR the leaf's sampled optimiser and moves the leaf to a total-samples comparison with plain SR and then a 24-site run; NOISE-FLOOR candidate (deficit ratio B/A <= 0.62) releases unit 2 (16N and half step length) to fit the exponent and steers the leaf to variance-reduction routes; NOT-NOISE candidate (ratio >= 0.85) releases only unit 2's 16N arm to confirm, and steers the leaf to conditioning routes (R3 and successors); AMBIGUOUS releases unit 2 as planned.
* prediction: NOISE-FLOOR candidate (deficit ratio B/A <= 0.62).
  * Probabilities: NOISE-FLOOR 0.40, AMBIGUOUS 0.25, REACHES 0.15, NOT-NOISE 0.20.
  * Basis: RULE_A's T1a shows the fixed length 0.005 still descends at the R38 plateau (0.0078 +- 0.0006 of the gap per step, never a rise), so arm A should pass R 0.51; whether it reaches 0.7475 in 120 steps depends on how fast that gain decays (0.023, 0.011, 0.008 per step over steps 1-20, 21-40, 41-60). Pre-flight from R 0.5355: two steps at 4N gained 0.022 and 0.010 against 0.010 and 0.007 at N, i.e. 4N roughly doubles the per-step gain there.
* p: 0.40
* prediction-changes-decision: yes. The leaf has no measured answer to whether its sampled plateau is noise or landscape; each verdict sends the next routes to a different family (variance reduction versus conditioning), and REACHES would change the leaf's question to sample cost.
* cheapest-kill-tried: RULE_A's free analysis of stored candidates (job 752885), which already removed the step-length selection as the main cause, and the replica-average energy (R(mean theta) 0.548 against 0.536 best replica: only a weak noise ball). A T0 argument cannot separate sampling noise from landscape at a fixed length: both give a decaying per-step gain.
* cost-cpuh: 2.4 (cap 2 tasks x 1.167 h = 2.33; expected use 1.5)
* cheapest-version: one seed per arm and only two of the four arms; the 16N and half-length arms (unit 2, about 4 CPU-h) and second seeds (unit 3) are released only if this unit leaves the verdict open. Fewer steps (60) would compare unplateaued trajectories, which RULE_B forbids as a NOT-NOISE basis. No smaller instance: the 12-site `che` is the only state with an exactly polished reference and stored guided runs.
* convergence: not guaranteed: arm A may still be descending at step 120 (gain at steps 41-60 of R38 was 0.008 per step with the fixed length). RULE_B handles it: an arm with tail slope above 0.002 per step is reported as not plateaued and cannot support NOT-NOISE, while the B/A deficit ratio at equal steps still measures whether 4 times the samples buys progress.
* wall-time-basis: measured on 1 thread (logs/preflight.txt): 0.221 CPU-min per step at N, 0.441 at 4N, set-up 0.3-0.6. Arm B 120 x 0.441 + 0.6 = 53.5 min, x1.3 = 69.5 -> --time 01:10:00; array of 2 tasks.
* power: R38's replica spread of R(60) is 0.015 (sd over 3 seeds); with deficits near 0.15-0.25 the ratio's sd is about 0.1, so P(ratio within 0.1 of 0.62 or 0.85, i.e. AMBIGUOUS-adjacent) is about 0.3; unit 3 adds a second seed only for arms near a threshold.
