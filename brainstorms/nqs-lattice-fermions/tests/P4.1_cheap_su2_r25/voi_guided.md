# VOI note: P4.1 round 22, R38 `guided` T1b (`submit_guided.sbatch`)

* leaf: P4.1
* route: R38
* tier: T1
* decision: three replicas of 60 guided sampled SR steps from `che`'s SR-trained start at 12 sites (RULE_D.md). REACHES-POLISHED (median R(60) >= 0.7475, the exactly ladder-polished che of job 742206) makes guided SR the leaf's scalable optimiser candidate and opens the `pcg` test and then a 24-site protocol; CLOSE (0.60-0.7475) goes to the critic with a tail fit; FAILS (< 0.60) leaves R38 a step-efficiency tool and keeps "no sampled optimiser reaches the polished baseline" as the leaf's limit.
* prediction: CLOSE.
  * Probabilities: CLOSE 0.45, FAILS 0.35, REACHES-POLISHED 0.20.
  * Basis: RULE_C's single-seed run reached R 0.364 at step 15 with step gains falling from 0.11 to about 0.005 (steps 10, 12, 13 skipped by the held-out rule). A geometric extrapolation of steps 9-15 (gain ratio about 0.85-0.9) gives a tail of 0.03-0.05, so R(60) near 0.40-0.45 on that trend, i.e. FAILS; but the held-out ladder picked tau 0.005-0.01 late, and R21's unguided arm was still rising linearly at step 15, so linear-in-steps progress at about 0.005 per step would give about 0.6 (CLOSE). Exact LM from the same start needed about 20 exact steps to reach 0.7475.
* p: 0.45
* prediction-changes-decision: yes. The leaf's 24-site question is blocked on "no scalable optimiser reaches the polished baseline" (summary, What it would take). Any of the three outcomes changes the leaf's next step: REACHES opens a 24-site rung, FAILS closes the sampled-optimiser sub-question at this sample budget, CLOSE asks for a tail decision.
* cheapest-kill-tried: the one-step noise-split screen (RULE_B, 9.6 CPU-min; guided Rbar 0.086 vs 0.016) and the 15-step RULE_C run (GUIDED-WINS, R(15) 0.364 vs R21's 0.195). A single 60-step replica would cost a third as much but cannot give a median; one replica interactively exceeds the 10 CPU-min allowance (25 CPU-min).
* cost-cpuh: 2.0 (cap; expected use 1.25)
* cheapest-version: one replica (0.42 CPU-h); refused here because RULE_C already is a one-seed result and the decision needs seed spread.
* convergence: no convergence claim; the rule scores R(60) and the CLOSE branch reports a tail fit.
* wall-time-basis: measured on 1 thread from R21's shift arm (identical per-step work): 0.41 CPU-min per step, 0.25 set-up; 60 steps = 24.9 CPU-min, x1.3 = 32 min, --time 0:40:00 per array task, 3 tasks.
* power: exact energies; replica-to-replica spread unknown (one-step screen sd 0.011 in R). P(median within 0.03 of a threshold) about 0.25.
