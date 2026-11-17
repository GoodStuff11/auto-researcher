# P5.1 round 1, T1a (free: stored data plus 4 exact energies): what kind of plateau did guided SR hit? (pre-registered)
Written 2026-11-17 before `floor_diag.py` was run. Data: `../P4.1_cheap_su2_r25/results_guided_s{6000,7000,8000}/shift.json`
(job 752885; 60 guided SR steps, 16,384 samples, held-out metric-length ladder {0.005, 0.01, 0.02, 0.05}; every
candidate's exact energy was logged but never used to choose). Window W = steps 41-60, pooled over the 3 replicas.
Per step, gains are in units of the gap g (R units): realised dR; oracle-best gain = best candidate's exact gain
(clipped at 0, since skipping is always allowed); rule-2sig gain = gain of the held-out argmin taken only if
dE_val < -2 sig_val, else 0.
* A1 (floor type). SELECTION-FLOOR if mean oracle-best gain over W >= 0.005 and mean realised gain <= 0.002:
  the sampled directions still descend and the held-out choice wastes them. DIRECTION-FLOOR if mean oracle-best
  gain < 0.002: the sampled directions themselves stopped descending. Otherwise MIXED.
* A2 (noise ball). NOISE-BALL if R(theta_bar) >= max_k R(theta_k) + 0.05, theta_bar the mean of the three final
  parameter vectors (exact energy); NO-BALL if R(theta_bar) <= max_k R(theta_k); otherwise WEAK.
* A3 (R6 acceptance rule, counterfactual at the stored states). R6 passes T1a if the mean rule-2sig gain over W is
  >= 0.004 and >= 2x the mean realised gain; KILLED at T1a if it is <= the mean realised gain; else MARGINAL.
Caveat written in advance: A3 scores the rule at states produced by the old rule (a one-step counterfactual), so
a PASS earns R6 a real multi-step run, not adoption.
