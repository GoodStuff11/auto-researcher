# P4.2 r10 zero-step validity check (pre-registered 2026-11-17, before any check run)
State: results/sr3000_c8h32.json (relerr 1.0572%, exact sums). Question: is polish.py's |z|=0 real stationarity or an artefact?
Checks (all exact sums, interactive, <=10 CPU-min total): (1) energy gradient g=2Re<O_k(H-E)> norm and its fraction
inside vs outside the kept rank-37 metric subspace (cutoff 1e-5); (2) plain gradient descent with line search and a few
SR steps (training diag shift) from theta_SR; (3) LM at cutoffs 1e-6, 1e-7 with |z|<=0.05 cap and line search;
(4) code read of polish.py for anything forcing |z|=0.
ARTEFACT if any method lowers relerr below 1.02% (then U3 verdict for c8h32 is not HOLDS; polish.py must be fixed and rerun).
GENUINE if no method lowers relerr by more than 0.02 percentage points (i.e. all stay >= 1.0372%).
UNCLEAR otherwise.

## Result (appended after running; check_zero_step.py, check_zero_step_iter.py, polish_v2.py; ~4.7 CPU-min total)
- |g| = 0.338 (equals 2 T^T H p0 to 3e-11). Metric max eig 1.24e6; ranks 37/55/78/103/176 at 1e-5/-6/-7/-8/-11.
  98.2% of |g|^2 lies INSIDE the kept rank-37 space, but its natural-gradient content there is only 1.5e-5 (2.6e-2 at 1e-8).
- v1 LM at 1e-5: |z| = 3.3e-4 (printed 0.000), gain 1e-5 pp. At 1e-6: -0.0002 pp; at 1e-7: -0.0022 pp (single step).
- Euclidean GD, 15 line-searched steps: no change at 5 digits. SR (lr 0.05, shift 0.01), 40 steps: 1.05722 -> 1.05543%.
- Iterated capped LM at 1e-8: 1.0572 -> 1.0281% (plateau after ~10 steps). At 1e-9: first step does not lower E.
- polish_v2 (cutoff scan 1e-5..1e-9 per step, min-gain acceptance): 1.0572 -> 1.0279% in 16 steps, 65 CPU-s.
- Diagonally rescaled metric: every capped step blew up (60-76% relerr); dropped.
VERDICT (pre-registered): UNCLEAR. Best 1.0279% is a 0.029 pp drop (> 0.02 pp) but stays > 1.02%.
No code bug forces z = 0; the 1e-5 relative cutoff against a 1.2e6 top eigenvalue keeps only stiff directions.
