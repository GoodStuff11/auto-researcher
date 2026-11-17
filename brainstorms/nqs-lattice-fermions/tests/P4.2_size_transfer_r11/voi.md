# VOI note: P4.2 R9.5 via R5.1 on the 4x4 torus (`submit_torus_pa.sbatch`)

* leaf: P4.2
* route: R9.5 (R5.1 anchored copy, frame change to periodic targets)
* tier: T2
* decision: verdict of `torus_pa.py --aggregate` under RULE.md (the leaf's R5.1 rule restated, three declared adaptations).
  * PASS (S >= 2 at both counted budgets B = 20 and 50, source cost charged; b-clause; beats the unanchored copy): reopen P4.2 on tori, unshelve R9.5, release seeds 1-2 and the `scr+` control one at a time.
  * FAIL (S < 1.25 at both, or eps_pa(50) no better than 90% of the control's excess): close R9.5; with the optimiser row holding, P4.2 is EXHAUSTED at this scale.
  * AMBIGUOUS: close R9.5 at this budget (the leaf reopens only on a PASS).
  * The 47 CPU-h MFNG-O torus capacity pilot is no longer needed: job 451914 already measured the torus capacity of the MFNG family (all three seeds <= 2.13% relative error at 50 steps, eps_best 1.96%), so this job replaces it.
* prediction: FAIL
* p: 0.55
  * Marginals: FAIL about 0.55, AMBIGUOUS about 0.30, PASS about 0.15.
  * For FAIL: the unanchored copy from the same sources (451914 `tr`) never reached the 50-step control level and had S(20) = 0.7 or 0 per seed; the offset-free clock gives a static head start S = 1, and the source charge is about 1 target step.
  * Against: the anchored source fit stays close to its MF init (|theta - theta_init| = 0.73 at the selected rung 1e-2, against |theta| = 3.0 for the R3.1 copy; `source_ladder_preview.txt`), so the copy should start near the MF prior rather than 2-6x above it.
* prediction-changes-decision: yes. The predicted FAIL closes the R9.5 row of the reopening ledger, which is one of the two open rows keeping P4.2 DORMANT; without this job that row stays open behind a 47 CPU-h pilot.
* cheapest-kill-tried: T0 Gate 2 (offset-free head start): a copy that only starts lower on the control's own curve scores S = 1, so only a trajectory that descends faster can pass; no T0 argument fixes the descent rate. T1 done locally: the full source ladder (53 CPU-s, rung 1e-2 qualifies at source relerr 7.3e-4 / 1.8e-5, so anchor_ok holds), a timing pass, and an end-to-end smoke on the r3 smoke instance (`smoke_output.txt`). The zero-shot target energy needs the 13070-dim sector isometry of the 1.68M-state space (~148 s at 8 threads, about 20 CPU-min), above the 10 CPU-min interactive cap, and it would not decide the rule anyway.
* cost-cpuh: 9.6
* cheapest-version: one seed, one arm (`pa`), 50 steps, one instance. The control (`scr`, 3 seeds) and the ablation (`cp` = 451914 `tr`) are reused from job 451914 at zero cost; the counted budgets and levels are fixed from it already (RULE.md). 50 steps is the largest counted budget the control has. No smaller torus exists: a 4x3 torus is in the sector-fitting regime (sector about 10^2 against 122-426 parameters). Seeds 1-2 are released only after a PASS.
* convergence: the rule scores steps-to-level at B <= 50, not a plateau, so the run need not converge. Control trajectories (451914 `scr`): eps(50) = 0.01858 / 0.01707 / 0.01828, last-5-step drops 2.0e-4 / 3e-5 / 1.3e-4 t/site; both B = 20 and B = 50 pass the plateau check against the frozen x.
* wall-time-basis: 52.6 s per projected SR step (451914 seed 0, slowest node; 36.6-37.7 s on the other two) x 50 = 2630 s, plus setup 148 s, MF-init check ~15 s, source ladder <= 185 s (one rung measured 53 s), four records <= 240 s: about 3220 s = 0.89 h; x1.3 = 1.16 h, so --time 01:12:00 at 8 CPUs (cap 9.6 CPU-h). A stepping guard at 1.05 h records the partial trajectory before the limit.
* power: one seed. Control seed-to-seed spread in steps-to-level is 16-20 at B = 20 and 25 to never at B = 50, so S for a single pa seed is uncertain by about +-25%. P(AMBIGUOUS or INCOMPLETE) about 0.3. One seed suffices for FAIL when S < 1.25 by a margin, and is never enough to declare the leaf ANSWERED: a PASS only releases seeds 1-2.
