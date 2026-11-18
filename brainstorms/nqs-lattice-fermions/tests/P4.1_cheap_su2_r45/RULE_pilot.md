# R45 from-init pilot rule (written 2026-11-17, before the pilot; auditor CHANGES REQUIRED on T2 digest 08d370f1663e)

Question: from the T2's own start (V.Model(arm).init(0)), can each arm's fidelity fit reach the level energy training
reached (1-F about 3e-4; `che` drift4000 has 3.18e-4) within the T2's 2000 evaluations?

* Run: `r45.py --mode train --arm {che,nnbf} --seed 0 --maxfun 150` (same code, init, optimiser as the T2), results in
  `results_pilot/`; scored by `pilot_score.py`.
* Fit: for each arm, least squares of log(1-F_best-so-far) against log(n) over evaluations n = 50..150 (power law
  1-F = a n^-b); extrapolate to n = 2000.
* PASS: both arms' extrapolated 1-F(2000) <= 3e-4. The T2 then runs as prepared (maxfun 2000).
* FAIL: either arm above 3e-4. Then n* = the evaluation count where the slower arm's extrapolation reaches 3e-4:
  if n* <= 6000, the T2's maxfun is raised to n* (cap recomputed); otherwise the T2 warm-starts both arms from an
  equal-effort energy pretraining, or is withdrawn. Either way the T2 is re-gated and re-audited.
* INVALID: an arm stops before 150 evaluations, or 1-F is not decreasing over n = 50..150 (b <= 0).
* Note: a power law over-predicts late progress if the fit plateaus early, and under-predicts if L-BFGS speeds up after
  its memory fills; the pilot is a feasibility screen, the T2's NOT-PLATEAUED clause remains the convergence test.
* Prediction: FAIL p about 0.6 (from-init fits start at 1-F 0.42; reaching 3e-4 needs a decade more than T1 showed
  from a converged start); `nnbf` faster than `che` per evaluation.
