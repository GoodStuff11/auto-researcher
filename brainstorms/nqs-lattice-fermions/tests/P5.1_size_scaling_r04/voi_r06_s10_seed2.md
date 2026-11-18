# VOI note (quick): P5.1 R51.8 S10 second seeds (`submit_r06_s10_seed2.sbatch`; RULE.md rev. 1)

* leaf: P5.1
* route: R51.8
* tier: T1
* decision: RULE.md's second-seed clause released both bracketing S10 guided rungs (65,536: s 0.054; 262,144: s 0.040; both within 0.3 ln-units of 0.05). With seed0 19000: if both seeds keep 65k above and 262k at or below 0.05 (mean of the two seeds per rung), N_needed(S10) = interpolation of the seed means and the 94k denominator stands; if the 262k seed mean rises above 0.05, N_needed(S10) is censored above 262,144 and the staged branch moves to POLY-CONSISTENT (r <= 1.728 vs 4x3, the 4x3 arm loses its staged purpose); if the seed-to-seed difference at either rung exceeds 0.01 (the 0.0045-in-R rise that flips the bracket), the 5% threshold is declared unresolvable at S10 and the round-6 PROPOSED 10% / 61-120 statistic is the only scorable one. The third arm, 16,384, is NOT part of the RULE release: it is the upper bracketing rung (s 0.107) of the proposed 10% statistic, added so that whichever statistic the auditor fixes has a second seed at its bracket.
* prediction: the seed spread at 262k is comparable to the plateau's window scatter (about 0.01-0.015 in s), so the 5% bracket is not robust (p 0.6); the 10% bracket (16k above, 65k below 0.10) holds in both seeds (p 0.85).
* p: 0.6
* prediction-changes-decision: yes. A spread above 0.01 settles that the 5% statistic cannot be scored at S10 (forcing the re-registration before unit 1c is scored); a small spread keeps the RULE statistic and the 94k denominator.
* cheapest-kill-tried: T0 on the existing trajectory (t0_floor_r06.py): the 20-step-window scatter of s within the 262k arm (0.045-0.082) already suggests the answer, but within-run windows are serially correlated (period-2 limit cycle) and the exact reference is shared, so they do not measure seed-to-seed spread of the last-20 mean; only a second seed does.
* cost-cpuh: 0.7 (1 task x 0.67 h x 1 CPU)
* cheapest-version: two RULE rungs plus one 6-minute rung serving the proposed statistic; no exact arm (deterministic).
* convergence: same 120-step statistic as seed 1 (job 763968, all arms DONE at 120 steps).
* wall-time-basis: interactive pre-flights 0.042 + 0.055 + 0.032 CPU-min/step x 120 + 0.6 = 16.1 min; x 1.8 x 1.3 = 38 min -> 00:40:00 (in-job 763968 used 25.9 min for the same three arms).
* power: one extra seed per rung gives a two-point spread, enough to detect a difference above 0.01 against the 0.0011 last-20 R sd seen at 4x3's 262k arm, not to estimate a variance; a third seed is not bought.
auditor: APPROVED 2026-11-17 c68271bbad05
