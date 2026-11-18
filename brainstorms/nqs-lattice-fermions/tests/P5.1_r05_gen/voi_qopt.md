# VOI note: P5.1 R51.11 `qopt` T2 (`submit_qopt.sbatch`; rule RULE_qopt_T2.md, revision 1)

* leaf: P5.1
* route: R51.11
* tier: T2
* decision: whether sampling from |psi| (an amplitude-only distribution, same per-sample cost as guided) becomes the leaf's optimiser candidate. Q-FEWER → it replaces guided SR as the optimiser whose size dependence is measured next (R51.8's ladder protocol, 5x2 → 6x2); NO-GAIN → the 10-site gain was an easy-instance effect (L40), R51.11 is killed as an optimiser and kept as a kappa diagnostic; MIXED → the size test is run for both samplers.
* prediction: MIXED or Q-FEWER. Basis: at 5x2 (T1b) q ~ |psi| at N = 4096 reached R_plat 0.397, matching guided at 16,384 (0.403) — a 4x sample saving — and exact kappa is 4.7x below guided at both 5x2 and 4x3. Against: the amplitude-only headroom falls from 48x to 23x between the sizes, weights p/q grow with the support, and 5x2 is the easier instance; first-passage gains are usually smaller than plateau gains.
  * Probabilities: Q-FEWER 0.45, MIXED 0.35, NO-GAIN 0.2.
* p: 0.45
* prediction-changes-decision: yes. Each verdict chooses a different optimiser for the leaf's size test.
* cheapest-kill-tried: exact kappa at both sizes (T1, `kq.py`) and a 120-step run at 5x2 (T1b, 3.6 CPU-min interactive); no cheaper multi-step evidence at 4x3 exists (one-step proxies rejected, R51.9).
* cost-cpuh: 5.75 (1 task x 5.75 h x 1 CPU; expected use 4.3)
* cheapest-version: one seed; two arms (16k many steps, 65k fewer steps), both stopped at 5.93M draws = 0.8 T_B, beyond which the verdict is NO-GAIN regardless (T_guided <= T_B); T_guided comes from submit_cmp.sbatch task 0, not re-run here.
* convergence: ns_run.py's step (resumable each step) with only the draw distribution changed (`ns_q.py`; at qmix 0.2 it is ns_run's guided sampler line for line); sustained first passage, censored at budget.
* wall-time-basis: no in-job q ~ |psi| rate exists; interactive pre-flights 0.20 (16k) and 0.39 (65k) CPU-min/step x 2.4 (R51.1 arm A in-job 0.36 over guided 16k interactive 0.15) = 0.48 and 0.94; 362 x 0.48 + 91 x 0.94 = 260 min x 1.3 = 338 min; per-run --cpu-cap 226 + 112; --time 05:45:00. Signal handling as submit_cmp.sbatch (tested, `sigtest.sh`).
* power: as voi_cmp.md: passage uncertainty about 10 steps (0.16M draws at 16k, 0.65M at 65k), under 10% of T_B; thresholds 0.5x and 0.8x of T_guided are separated by 2.2M draws.
