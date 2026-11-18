# P5.1 round 5, R51.11 T2: |psi| sampling against guided SR in total samples at 4x3 (pre-registered)
Written 2026-11-17 after RULE_qopt_T1b.md scored PASS at S10 (R_plat at N = 4096: q ~ |psi| 0.397, b = 0.5 0.350,
guided 0.274, guided 16,384 0.403), before any 4x3 arm below ran (only 3- and 2-step timing pre-flights,
`logs/preflight_q1_*.txt`).

REVISION 1 (2026-11-17, before any arm ran; compute auditor): own job (`submit_qopt.sbatch`, `voi_qopt.md`) and own
verdict; first passage is the sustained crossing of `score_common.py` (trailing 20-step mean of R >= R*), and the
budgets are re-sized to R51.1 arm B's sustained T_B = 7.41M (RULE_cmp.md revision 1, item c).

Arms (`ns_q.py --qmix 1.0`, i.e. ns_run.py's step with q ~ |psi|; 4x3 start, tau 0.005, seeds 9000 + n):
* q1 N 16,384, 362 steps; q1 N 65,536, 91 steps (5.93M draws each = 0.8 T_B; beyond 0.8 T_guided the verdict is
  NO-GAIN whatever happens, and T_guided <= T_B).
Statistic: T = N x t* (sustained crossing of R* = 0.90 R_pol = 0.6728), censored at 5.93M; T_q = min over the two
arms; T_guided = min(T_B, T(R51.1 A + its continuation)) from `submit_cmp.sbatch` task 0 (scored only after it ends).
* Q-FEWER if T_q <= 0.5 T_guided → R51.11 becomes the leaf's optimiser candidate; its size dependence is then
  measured on R51.8's ladder protocol (5x2 → 6x2) as a later unit.
* NO-GAIN if T_q >= 0.8 T_guided (or both q arms censored) → the S10 gain was an easy-instance effect (L40);
  R51.11 is killed as an optimiser and kept as a kappa diagnostic.
* MIXED otherwise.
Secondary (reported): the same at 0.80 R_pol; unique draws per step (the cost of flatter draws).
