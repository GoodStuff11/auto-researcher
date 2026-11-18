# P5.1 round 5, R51.11 T2: |psi| sampling against guided SR in total samples at 4x3 (pre-registered)
Written 2026-11-17 after RULE_qopt_T1b.md scored PASS at S10 (R_plat at N = 4096: q ~ |psi| 0.397, b = 0.5 0.350,
guided 0.274, guided 16,384 0.403), before any 4x3 arm below ran (only 3- and 2-step timing pre-flights,
`logs/preflight_q1_*.txt`). Runs as task 2 of `submit_cmp.sbatch`, sharing R51.16's foils and statistic.

Arms (`ns_q.py --qmix 1.0`, i.e. ns_run.py's step with q ~ |psi|; 4x3 start, tau 0.005, seeds 9000 + n):
* q1 N 16,384, 412 steps (6.75M draws)     * q1 N 65,536, 103 steps (6.75M draws)
Statistic (as RULE_cmp.md): T = draws to first R >= 0.90 R_pol (R* = 0.6728), censored at 6.75M; T_q = min over
the two arms; T_guided = min(6.75M (R51.1 B), T(guided 16k continued), from task 1).
* Q-FEWER if T_q <= 0.5 T_guided → R51.11 becomes the leaf's optimiser candidate; its size dependence is then
  measured on R51.8's ladder protocol (5x2 → 6x2) as a later unit.
* NO-GAIN if T_q >= 0.8 T_guided (or both q arms censored) → the S10 gain was an easy-instance effect (L40);
  R51.11 is killed as an optimiser and kept as a kappa diagnostic.
* MIXED otherwise.
Secondary (reported): the same at 0.80 and 0.95 R_pol (0.95 censored for most arms); unique draws per step.
