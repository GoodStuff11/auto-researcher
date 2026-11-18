# P5.1 round 5, R51.13 `svrg` T1 (pre-registered, written before `kq.py` ran)

Question. Does an SVRG-type control variate (the same draws evaluated at an anchor k steps old, importance-weighted,
mean known exactly at the anchor) remove enough of the step's first-order error to pay for evaluating rows and local
energies at two parameter sets per sample?

Statistic: rho(k) = remaining fraction of the guided-draw step error (S norm, optimal scalar coefficient), anchor =
theta minus the exact step of metric length k x 0.005; exact anchor means (an upper bound: a real anchor's means
come from a large sample). Per-CPU gain G(k) = 1 / (2 rho(k)), anchor cost ignored (optimistic).

* KILL if rho(1) > 0.25 at both sizes (even a one-step-old exact anchor, i.e. the SARAH/STORM recursion's best
  case, cannot double the samples per CPU).
* PASS if rho(5) <= 0.25 at both sizes → prepare a multi-step T2 with a sampled anchor refreshed every 5 steps.
* MARGINAL otherwise.
Sizes: S10 (k = 1, 5, 20) and S12 4x3 (k = 1, 5).
