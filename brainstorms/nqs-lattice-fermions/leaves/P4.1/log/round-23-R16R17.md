# P4.1 round 23: R16 (`sq`) and R17 (`nj`) T1 tangent rule rerun at the ladder-polished parent

2026-11-17. Worker, interactive (no Slurm). Hard limit 10 CPU-minutes. Directory: tests/P4.1_cheap_su2_r26.

## Set-up
* Rule: the Round 10 pre-registered T1 rule in tests/P4.1_cheap_su2_r10/tangent.py, unchanged (families C1 parent
  Jastrow features, C3 parent orbital shifts, SQ squeezed-frame rows (R16) vs W one random hidden unit x free orbital
  vector (foil), NJ 81 plaquette charge-pattern counts (R17) vs RF 81 random tanh charge features (foil); KILL if
  route $\le$ foil on che and pcg; PASS if route $\ge$ 2 foil and $\ge$ 0.05 g on both; else AMBIGUOUS).
* RULE.md (written before running) states the unconverged-parent effect: residual descent of the parent adds a common
  term to route and foil, compressing their ratio toward 1, so PASS is harder, KILL unaffected; residual reported as
  Delta(C1+C3)/g, not used in the verdict; thresholds kept. Decision order: che first; pcg only if needed and if it
  fits the cap.
* Driver t1_polished.py swaps in tests/P4.1_cheap_su2_r22/results/theta_{che,pcg}.npy with stored E from
  state_<arm>.json last step E_new (che -7.822249354501926, eps 7.217e-4; pcg -7.824587605391688, eps 4.230e-4).

## che result (logs/run_che_output.txt, results/tangent_che.json)
All validation checks and controls pass: P_G overlap 1 - 2e-16, |E - stored| 3.6e-15, tangent <S^2> max 5.3e-15,
positive control Delta/g = 1 to 7e-15, negative control G 0.0033 (< 0.05). Gap g = 5.650e-3.

| set | Delta/g | rank |
|---|---|---|
| C1 | 0.0011 | 74/78 |
| C1+C3 | 0.0023 | 194/198 |
| C1+C3+SQ | 0.0087 | 284/318 |
| C1+C3+W | 0.0090 | 314/318 |
| C1+NJ | 0.0084 | 144/159 |
| C1+RF | 0.0052 | 155/159 |

* R16: G_SQ 0.0064 vs G_W 0.0067 (route $\le$ foil on che; Round 10 che 0.0183 vs 0.0162, pcg 0.0263 vs 0.0271).
* R17: G_NJ 0.0073 vs G_RF 0.0041 (route > foil, but 1.8x foil < 2x and 0.0073 < 0.05): AMBIGUOUS regardless of pcg.
  (Round 10: che 0.0052 vs 0.0075, pcg 0.0124 vs 0.0073.)
* Residual of the parent's own C1+C3 directions is small (0.0023 g), so the unconverged-parent compression of the
  route/foil ratio is minor at first order here.

## Budget overrun (disclosed)
The che run used 1128 s user CPU (18.8 CPU-minutes, 988% CPU, 1 min 55 s wall), above the 10 CPU-minute hard limit.
Cause: the driver imported numpy before tangent.py set OMP/MKL/OPENBLAS_NUM_THREADS, so BLAS ran on all 20 cores
(torch was 1-thread). Fixed in t1_polished.py (env set before the numpy import); not re-tested. No further compute
was run after the overrun was seen.

## Not run: pcg half
* Needed only for R16 (KILL needs route $\le$ foil on pcg too). Estimated cost: Round 10 pcg 7.5 CPU-min at 4 threads,
  2 min 25 s wall; at a true single thread about 3-8 CPU-minutes, about 10 GB RAM.
  Command: `~/.conda/envs/lit_env/bin/python t1_polished.py --mode run --parent pcg --threads 1`, then `--mode report`.
* Prediction: R16 KILL likely (Round 10 pcg already route < foil, and che now route < foil), p ~0.65.

## Verdicts
* R16: undecided (che route $\le$ foil; pcg pending). Stays LIVE, parked.
* R17: AMBIGUOUS (decided on che alone). Second AMBIGUOUS, first-order gain under 1% of the gap at every parent;
  no promotion under the card's rule (T2 only after a written reason; none written here).
