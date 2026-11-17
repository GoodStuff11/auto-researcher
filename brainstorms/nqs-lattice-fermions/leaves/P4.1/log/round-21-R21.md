# P4.1 round 21: T1 preparation for R21 `trSR` (worker, 2026-11-17)

Task: prepare R21's T1 exactly as its card pre-registers it, plus a reported-only per-step cutoff-ladder arm (L29). No job submitted. Summary and cards were not edited (a compactor is rewriting them).

## Files (`tests/P4.1_cheap_su2_r24/`)

* `RULE.md`: written before any output. It has the card rule verbatim, the interpretation fixed in advance, the verdict table and the reported-only `ladder` arm.
  * Interpretation: SR uses S_T = cov(O, O) and f = cov(O, E_loc) from the samples. `trsr` keeps eigenvalues above 1e-4 w_max, with the step at metric length exactly 0.05 in the sampled S_T. `shift` uses (S_T + 1e-3 I)^{-1} f, rescaled to the same metric length. Every step is taken.
* `trsr.py`: modes `exact0` (free diagnostic), `run --arm {trsr,shift,ladder}` and `verdict`.
  * It reuses r19 `polish.py` (ctx_for, logder) read only, and r20's i.i.d. draw.
  * O rows are computed only on the drawn representatives, so a step costs 0.18 CPU-min against r20's 2.4.
* `submit_r24.sbatch`: partition kim, 1 CPU, 4G, 0:20:00. Measured 11.9 CPU-min, times 1.3.
* `voi.md`, and the logs `exact0_output.txt`, `probe_trsr_1step_1thread.txt` and `probe_ladder_1step_1thread.txt`. Probe outputs are in `results_probe/`, not `results/`.

## Free diagnostics (decide nothing under the rule)

`logs/exact0_output.txt` gives the infinite-sample step 1 from Round 11's exact S and H at theta0 (`r11/results/S_H_theta0.npz`). R is the change in units of the gap, positive meaning the energy fell.

| arm | metric length 0.05 | metric length 0.02 |
| :--- | :--- | :--- |
| `trsr` (cutoff 1e-4, rank 1018) | R = -0.562 | R = +0.227 (reproduces Round 11's `ng` 0.2274) |
| `shift` (abs 1e-3) | R = -0.533 | R = +0.288 |

* S_T has w_max = 8.06. 1680 of its 2637 eigenvalues lie below the 1e-3 shift.
* **The card's metric length 0.05 overshoots even with exact sums.** Round 11's screen already showed this: the `ng` predicted gain at tau 0.05 is -0.24. The 0.227 the card cites came from tau 0.02.
* **The card's premise is contradicted at step 1.** At length 0.02 the exact Tikhonov-shifted step beats the truncated one, 0.288 against 0.227. The absolute shift is not what limited the trained state's first step.

Sampled probe at 16,384 samples, seed 1, identical to the job's step 1 (`logs/probe_trsr_1step_1thread.txt`):
* `trsr` step 1 gives R = -0.365 with |dtheta| = 0.565, rank 833. The sampled E is 0.8 se from the exact energy.
* So the job's `trsr` step 1 will raise E by 0.365 g. That is above the 0.02 g ceiling, so **VIABLE is already excluded under the card's rule**.

Ladder probe, step 1 (`logs/probe_ladder_1step_1thread.txt`):
* Every cutoff from 1e-4 to 1e-8 raised both the held-out and the exact energy, with exact R from -0.36 down to -879. The step was skipped.
* The held-out ranking matched the exact ranking at every cutoff.

## Gate

`compute_gate.py check <wd> tests/P4.1_cheap_su2_r24/submit_r24.sbatch --voi tests/P4.1_cheap_su2_r24/voi.md --pre-audit` returned:
* cap 0.3 CPU-h (quick note); leaf committed 3.7 / 300; digest a964aaa15dba;
* GATE: PASS.

It was not submitted. The whole run is 11.9 CPU-min, above the 10 CPU-min interactive allowance.

## Prediction

* Card prior: NOT-VIABLE 0.5, MARGINAL 0.3, VIABLE 0.2.
* Updated after the diagnostics: NOT-VIABLE 0.8, MARGINAL 0.2, VIABLE 0.

## Recommendation to the orchestrator and critic

As specified, a NOT-VIABLE from this test is confounded by step length, because exact SR at 0.05 also rises. It would not cleanly "close sampled polishing".

Options, before any data:
1. Amend the metric length to 0.02, where exact `trsr` step 1 gives +0.227.
2. Add a held-out step-length ladder (for example {0.005, 0.01, 0.02, 0.05}) to `trsr` and `shift`.

The arms then cost about the same. The VIABLE margin (0.15 above `shift`) looks unlikely in either case, since `shift` wins step 1 exactly.

## Proposed library entry

Before pre-registering a fixed natural-gradient or SR step length for a sampled test, check its infinite-sample step 1 on the exact S and f. A length that overshoots with exact sums makes the sampled verdict a step-length verdict. (P4.1 R21: exact step 1 at metric length 0.05 gave R = -0.56, and at 0.02 gave +0.23.)

## Amendment 1 (orchestrator-ordered, 2026-11-17; `tests/P4.1_cheap_su2_r24/RULE_amend1.md`)

Written after the exact-sum step-1 checks and the seed-1 timing probes, before any scored run (`results/` is empty).

### Changes

* **Step length.** `trsr` and `shift` each choose the metric step length per step from {0.005, 0.01, 0.02, 0.05}. The choice is the candidate with the lowest correlated-sampling dE on a held-out set (same N, seed 1000 + step). The step is taken if that dE < 0, otherwise skipped. The exact energy is logged per candidate but never used to choose.
* **Verdict.** It is read on the better of `trsr` and `shift`.
  * VIABLE: $R(15)\ge0.20$ and no step raises the exact energy by more than 0.02 g.
  * NOT-VIABLE: both arms have R(15) < 0.08.
  * MARGINAL: anything else.
* **Reported only.** R_trsr - R_shift, and the cutoff-ladder arm.
* **Code and job.** `trsr.py` is updated, and the sbatch is now 0:25:00.

### Card premise

The card's premise, that the absolute shift is what stalls SR, is already undercut by the exact step 1. At length 0.02 the shifted step gives 0.288 and the truncated step 0.227.

### Amended seed-1 probe

Source: `logs/probe_amend1_trsr_1step_1thread.txt`.

| length | held-out dE | exact R |
| :--- | :--- | :--- |
| 0.005 | -7.1e-4 | +0.025 |
| 0.01 | -1.21e-3 | +0.037 |
| 0.02 | -1.53e-3 (chosen) | +0.024 |
| 0.05 | +3.8e-3 | -0.365 |

* The chosen step at length 0.02 realises about 10% of the exact-sum gain at the same length (0.024 against 0.227).
* A step costs 0.30 CPU-min. The run totals 15.5 CPU-min, which is above 10, so it was not run interactively.

### Gate (amended)

`compute_gate.py check --pre-audit` returned **GATE: PASS**: cap 0.4 CPU-h (quick note), leaf committed 6.4 / 300, digest a719714997b3. Not submitted.

### Prediction (amended)

MARGINAL 0.5, VIABLE 0.25, NOT-VIABLE 0.25.

### Library entry (proposed, repeated here for the orchestrator)

**L-new.** Before pre-registering a fixed SR or natural-gradient step length for a sampled test, check its infinite-sample step 1 on the exact S and f. A length that already overshoots with exact sums turns the sampled verdict into a step-length verdict. Evidence: P4.1 R21, where the exact step 1 gave R = -0.56 at metric length 0.05 and +0.23 at 0.02.
