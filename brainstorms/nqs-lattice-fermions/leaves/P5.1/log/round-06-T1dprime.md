# P5.1 round 6: R51.3 `levrow` T1d′, same-machine re-timing of the comparison arm
Run 2026-11-17 22:36 under the T1d′ rule pre-registered on the R51.3 card (before any data): rerun R51.3's 5
continuation steps on the ladder's machine; $\rho'$ as in `tests/P5.1_rowsample_r01/RULE_T1d.md`; PASS if
$\rho'\ge3$, KILL if $<1.5$, MARGINAL between. No rule was amended.

## Setup
* Machine: kim-compute-03 (Claude's interactive Slurm allocation 451618), the node where `t1d_ladder.py` and T1c
  ran interactively. One thread (`OMP/OPENBLAS/MKL_NUM_THREADS=1`), `ulimit -t 600`.
* Code: `t1d_prime.py` is `t1d_cont.py` unchanged except the output file (`results/t1d_prime.json`, so job 760095's
  `results/t1d_cont.json` is kept) and a host field. Same start point (`results/t1c_theta.npy`), same scores
  (`results/lev0.npy`), seeds 11..15, same amortised score charge 0.0632 CPU-min per step, same $\gamma_L$.
* Only the cheaper arm was re-timed: the ladder needs 8.2 CPU-min and both arms would not fit the 10 CPU-min cap.
  The ladder was already timed on this node (interactive, 17:35), so the comparison is now same-machine.

## Result (`logs/t1d_prime.txt`)
| step | cut | tau | eps after | CPU-min here | CPU-min job 760095 |
|---|---|---|---|---|---|
| 11 | 1e-7 | 0.001 | 6.05520e-4 | 0.47 | 0.76 |
| 12 | 1e-6 | 0.002 | 5.98489e-4 | 0.49 | 0.76 |
| 13 | 1e-7 | 0.001 | 5.92017e-4 | 0.48 | 0.75 |
| 14 | 1e-7 | 0.001 | 5.86737e-4 | 0.47 | 0.72 |
| 15 | 1e-6 | 0.002 | 5.80417e-4 | 0.47 | 0.71 |

* The eps path matches job 760095 to all printed digits (deterministic seeds, exact sums), so only the timing
  changed. eps_L 5.84794e-4 reached at step 15.
* Step CPU 2.38 min (0.48 per step, against 0.44 in T1c and 0.74 on kim-cpu-02).
* $\gamma_R = 1.138\times10^{-5}$ per CPU-min; $\gamma_L = 3.225\times10^{-6}$; $\rho' = 3.53$ → **PASS**.
* Reported only: $\rho'$ with the 1.896 CPU-min score build charged in full to these 5 steps: 2.22.
* Process: 2.50 CPU-min (131.6 user + 18.3 sys s), peak RSS 2.5 GB.

## Caveats
* The ladder was not re-timed in this session. Node load was about 30 on 20 cores during the rerun; the ladder's
  load at 17:35 was not recorded. Per-step CPU here (0.48) is close to T1c's (0.44) on the same node, so load
  appears to shift timing by under 10%. If the ladder's steps were 10% cheaper, $\rho'$ would be 3.2, still PASS.
* Off-question: R51.3 lowers the cost of exact-sum (unscalable) polishing, not the sample count, so this PASS
  matters only for P4.1-style 12-site comparisons. Sharpest Gap (ii), the sampled-VMC version, is untested.
* The PASS uses the pre-registered 30-step amortisation of the score build. Charged in full over 5 steps it would
  be MARGINAL (2.22).

Compute: 2.50 CPU-min interactive. No job submitted.
