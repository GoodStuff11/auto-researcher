# P4.1 round 24: ledger row 2, R28 hybrid (metric refresh every k gradient steps) — KILLED at T0 (worker, 2026-11-17)

Task (coordinator): prepare, not submit, the test of ledger row 2 (R28's hybrid: L-BFGS whitened by an exact S
build, metric refreshed every k gradient steps), which must reach $\ge$ 3 gamma_LM (gamma_LM = 7.27e-6 relative energy
per CPU-min, `che`) to reopen anything; first try the cheapest kill at T0 from round 22's R28 data. The T0 kill
holds, so no RULE.md, script, sbatch or voi.md was written and no gate check was run. No compute beyond a few
seconds of arithmetic on `tests/P4.1_cheap_su2_r25/results/che_plbfgs.json`.

## Inputs (all measured, round 22, one thread, `che` from the 742206 endpoint, eps0 = 7.217e-4)
* Exact S build B = 1.874 CPU-min (first entry of the R28 trajectory, after setup; same `polish.build` the ladder
  uses; the ladder log `tests/P4.1_cheap_su2_r22/logs/arm_che_output.txt` times only whole steps, 2.68–2.93
  CPU-min, so B $\le$ 2.86 is consistent).
* Energy+gradient call c = 0.1327 CPU-min (mean over 54 calls; probe_timing 0.121).
* Ladder: last-5 accepted gains 2.10e-5 in eps per step at 2.86 CPU-min per step, gamma_LM = 7.27e-6 /CPU-min;
  target 3 gamma_LM = 2.18e-5 /CPU-min.
* R28 trajectory: 55 calls, eps 7.217e-4 → 6.805e-4 (half) → 6.479e-4; calls 3, 5, 7 were line-search overshoots
  (eps 5.09, 8.4e-3, 1.04e-3).

## Derivation
A hybrid with refresh period k spends B + k c per cycle and gains at most G(k) per cycle, so its rate is
G(k) / (B + k c). Three bounds on G(k), each more generous than the last:

1. **Period repeat** (each refresh reproduces the first cycle after the round-22 build, from the same eps0):
   G(k) = eps0 − min eps over the first k calls. max_k rate / gamma_LM = **1.31** (k about 16; 1.13 at k = 10,
   1.30 at k = 20–25, 1.15 at k = 50).
2. **Best window anywhere** (G(k) = the largest drop over any k consecutive calls in the whole 55-call run, i.e.
   each cycle gets the run's best stretch, ignoring that gains shrink as eps falls): **1.35** (k = 15); with the
   three overshoot calls deleted (a warm-started curvature scale that never overshoots): **1.45** (k = 11).
3. **Ladder step plus k whitened L-BFGS steps on the same build** (the build is shared: one ladder LM step,
   gain 2.10e-5 at 2.86 CPU-min including its build, then k L-BFGS calls in that step's whitening, each cycle
   credited the best window): (2.10e-5 + G_best(k)) / (2.86 + k c) / gamma_LM = **1.69** (k = 13), **1.79**
   (k = 11) without overshoots.

Even with no build at all, the best 10-call window is 2.97 gamma_LM (5-call 3.59, 3-call 4.26), so 3× needs the
build to cost at most 0.33 CPU-min (0.48 without overshoots), 4–6 times cheaper than the exact build; that is a
different route (a cheap or sketched metric), not R28's hybrid.

Why these are upper bounds: (a) the gains per cycle are taken from the best stretches of a run that started at a
higher eps, while ladder and R28 gains fall as eps falls (ladder 2.6e-5 → 1.3e-5 per step over its last 5 steps);
(b) a fresh whitening restarts the L-BFGS memory, and round 22's first cycle shows that restart costs calls 1–7
(rate 1.08 gamma_LM at k = 7); (c) bound 3 assumes the ladder step and the L-BFGS calls do not compete for the same
low-curvature directions, while both act on the same whitened metric.

## Verdict
The most optimistic hybrid bound (1.79×) is below the 3× reopening threshold, and every bound that keeps the
hybrid's own structure (1, 2) is below the 1.5× KILL line of the round-22/23 rules. The test's best possible outcome
is AMBIGUOUS, which reopens nothing. **R28 (hybrid) KILLED at T0; ledger row 2 → KILLED(T0).** No compute was
spent, so the 12-site convergence resume (ledger row 4, 13.0 CPU-h) has no affordable optimiser from this family.

Not ruled out: a metric whose build costs under about 0.3 CPU-min at 12 sites (sketched or low-rank S updated by
gradients), the `pcg` arm (its build is about 7 CPU-min, so the build-to-gradient ratio is worse: 7 / 0.55 = 13
against 1.87 / 0.13 = 14, essentially the same, so no reason to expect better), and unpolished starts, where both
methods gain more and the comparison is not the one the ledger asks about.

## Proposed library entry
* **Refreshing a metric cannot beat the method that pays for it.** When one exact metric build costs about 14
  gradient calls and whitened quasi-Newton gains at most about 3 times the ladder rate per call in its best
  stretch, any refresh schedule is bounded by G(k)/(B + kc) at most 1.8 times the ladder; a decisive speed-up needs a
  metric about 5 times cheaper, not a better schedule. Source: P4.1 R28 (round 24), supports the round-22 entry on
  ill-conditioning.
