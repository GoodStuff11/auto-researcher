# Amendment 1 to RULE.md (2026-11-17, ordered by the orchestrator)

**Disclosure:** I wrote this amendment after the exact-sum step-1 checks (`logs/exact0_output.txt`) and the seed-1 timing probes (`logs/probe_*_1step_1thread.txt`, outputs in `results_probe/`), and before any scored run. `results/` is empty.

## Why

With exact sums, the first step at the card's metric length of 0.05 already raises the energy. The truncated step (`trsr`) gives R = -0.562 and the shifted control (`shift`) gives R = -0.533. The seed-1 sampled `trsr` step gives R = -0.365. Under the card's rule a verdict would therefore measure the step length, not sampled SR.

The same checks undercut the card's premise. At length 0.02 the exact shifted step gains more than the truncated one, 0.288 against 0.227. Truncating instead of shifting does not help at step 1 with exact sums.

## Changes (they supersede RULE.md where the two differ)

### 1. Step length chosen per step

* `trsr` (relative cutoff 1e-4) and `shift` (absolute 1e-3) each choose the metric step length per step from {0.005, 0.01, 0.02, 0.05}.
* The choice is the candidate with the lowest correlated-sampling energy change dE on a fresh held-out set (N = 16,384, seed 1000 + step number).
* The step is taken if that dE < 0. Otherwise it is skipped: theta is unchanged, and the step still counts as one of the 15.
* The exact energy of every candidate is logged as an oracle diagnostic and never enters the choice.

### 2. The `ladder` arm

* The `ladder` arm (relative cutoff from 1e-4 to 1e-8 at length 0.05, chosen on the held-out set) stays reported only.

### 3. Verdict on the better main arm

The verdict is read on the best of `trsr` and `shift`, because the 24-site rung needs any sampled optimiser that reaches the polished baseline.

| Verdict | Condition |
| :--- | :--- |
| VIABLE | some main arm has R(15) >= 0.20 and no step that raises the exact energy by more than 0.02 g |
| NOT-VIABLE | R(15) < 0.08 for both main arms |
| MARGINAL | anything else |

* INCOMPLETE and INVALID are as in RULE.md.
* R_trsr(15) - R_shift(15) is reported separately. It is no longer a VIABLE condition.

### What the verdicts trigger

* VIABLE sets the 24-site protocol (the passing arm, at about 6 samples per parameter), with the critic's agreement.
* NOT-VIABLE closes sampled SR polishing at about 6 samples per parameter.
* MARGINAL runs no 24-site rung.
