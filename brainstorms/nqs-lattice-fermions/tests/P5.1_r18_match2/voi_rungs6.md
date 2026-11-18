# VOI note: P5.1 R51.33 `match2`, job 2 of 2 (`submit_r18_rungs6.sbatch`; `RULE.md`)
* leaf: P5.1
* route: R51.33
* tier: T2
* decision: As `voi.md`. This job runs the two 6×2 guided rungs that job 1 placed in `results/rungs6.json`, and these rungs decide $r'_{mm}$.
  - POLY-CONSISTENT → the Answer keeps about 1.2 at two matched regimes, drops "the 12-site step was not shortened to match", and releases the third size.
  - AMBIGUOUS → the growth is reported as regime-dependent.
  - SUPPORT-LIKE → the 1.21 is demoted to one-sided matching.
  - Anything else gives numbers only.
* prediction: POLY-CONSISTENT. Prior as `voi.md`, to be refreshed before submission from job 1's outputs: the 6×2 $m$, $N_5^*$, and $G_{\rm ref}(6\times2,0.0025)/0.0070107$. If the 6×2 reference deepened by more than the 5×2 analog's −1.2%, lower p(POLY).
* p: 0.55
* prediction-changes-decision: yes. POLY rewrites the Answer's growth sentence (two-sided matching) and releases the third-size spend.
* cheapest-kill-tried: Two rungs placed 0.13–0.14 ln-units outside the lines, so either censored outcome decides a class with one run. No stored 6×2 arm is at τ 0.0025.
* cost-cpuh: 6.0 (1 task × 6 h × 1 CPU; expected 3.4 at the predicted rungs ≈ 335k and 900k).
* cheapest-version: Two rungs, one seed. The second seed is released only by the L72 clause (job 3), and no rung is run below $N_P$.
* convergence: Fixed 120 steps, window 61–120, as R51.8. The 6×2 exact arm from job 1 is the reference.
* wall-time-basis: Round-18 interactive 6×2 guided rates were 0.315, 0.469 and 0.665 CPU-min per step at 147k, 393k and 1M, times 1.54 in-job (6×2 262k: 0.653/0.425). At the predicted rungs that is 82 + 119 = 201 min. The worst case, with $N_S$ at the 2,097,152 cap, is 281 min; × 1.25 gives 6 h.
* power: R51.8's 6×2 $sd_w$ was 0.0022–0.0024 and its conditioning 26.5 sd_w. The placed rungs keep censored outcomes outside the near-line band, and the predicted margin of about 0.36 ln-units allows one seed.
* auditor: APPROVED 2026-11-18 556530334848
