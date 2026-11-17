# P4.1 round 22: optimiser batch (worker, 2026-11-17)

Task: generate and gate 5-7 routes for a scalable (sampled) optimiser that reaches the exactly polished 12-site baseline; mid-round the coordinator added sub-question A (an exact-sum optimiser that converges both 12-site arms affordably, since job 742206's ladder LM left fitted tails of 22-100% of eps) and R21's result (job 748762 MARGINAL: shift arm R(15) = 0.195, about 0.013 R per 0.41 CPU-min step). summary.md not edited (coordinator's instruction). Code and rules: `tests/P4.1_cheap_su2_r25/`.

## Sub-question A: exact-sum convergence of the 12-site arms (RULE.md, pre-registered)

Tool: `exgrad.py`, the exact energy gradient by one chunked backward through all 627,264 amplitudes (checked: directional derivative equals the central difference to 7 digits on both arms). Cost: `che` E 0.045, E+grad 0.121 CPU-min; `pcg` 0.173 and 0.552. Ladder reference rate from `state_{arm}.json` (last 5 accepted gains / mean CPU per step): gamma_LM = 7.27e-6 (`che`), 5.51e-6 (`pcg`) relative energy per CPU-min. Rule: PASS-T1 if gamma ≥ 3 gamma_LM and second-half rate ≥ 2 gamma_LM; KILL if gamma ≤ gamma_LM.

| route | arm | eps start → half → end | CPU-min | calls | gamma/gamma_LM (late) | verdict |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| R27 `lbfgs` (m 30) | che | 7.217e-4 → 7.176e-4 → 7.118e-4 | 8.1 | 61 | 0.17 (0.19) | KILL |
| R27 `lbfgs` | pcg | 4.230e-4 → 4.179e-4 → 4.162e-4 | 9.5 | 15 | 0.13 (0.06); line search stalled | KILL |
| R29 `ncg` (trust-ncg, FD HVP) | che | 7.217e-4 → 7.187e-4 → 7.187e-4 | 8.0 | 60 | 0.05 (0.00) | KILL |
| R28 `plbfgs` (one exact S build, whitened, L-BFGS) | che | 7.217e-4 → 6.805e-4 → 6.479e-4 | 9.0 (build 1.9) | 55 | 1.12 (0.99) | AMBIGUOUS |

R30 `anderson` (DIIS on ladder iterates; cross-field SCF) gated at T0 PASS, parked (its T1 needs 6+ builds, about 20 CPU-min; R27/R29 tested the multisecant idea with cheap maps and lost). R31 `krylov-lm` KILLED at G2 T0 (the build, not the solve, sets the cost).

Ranking by gain per CPU-minute at the polished endpoints: R28 matches the ladder LM (1.1x), then R27 (0.15x), then R29 (0.05x). Conclusion: no exact-sum optimiser tested beats the ladder by the factor needed; the slow convergence is the landscape's ill-conditioning (only metric-aware steps make progress), so 12-site arm comparisons must either accept the tail fit (RULE_amend2) or be decided elsewhere. Note: rates are measured at the polished endpoints; R21's 0.013 R per step is from the SR-trained start, where gains are far larger, so the two are not comparable per CPU-minute.

## Sub-question B: one-step noise split (RULE_B.md, pre-registered)

`noise_split.py`, che SR-trained theta0 (R21's start), full Y and HY built once (1.5 CPU-min), N = 16,384, seeds 3001-3003, every direction a shifted (1e-3) solve, scaled to exact-metric length tau ∈ {0.01, 0.02, 0.03}, R_best = best tau (oracle length). Exact shifted step: R 0.199 / 0.288 / 0.233.

| variant | Rbar ± sd | cos to exact step | verdict |
| :-- | :-- | :-- | :-- |
| V1 base | 0.016 ± 0.022 | 0.27 | (reference) |
| V1t truncated 1e-4 | 0.005 ± 0.020 | 0.24 | reported |
| V2 exact S, sampled f | −0.024 ± 0.011 | 0.11 | class bound FAILS → metric-side class KILLED |
| V3 sampled S, exact f | 0.171 ± 0.015 | 0.84 | force side open |
| V4 S from 4N, f from N (R35 bound) | 0.003 ± 0.011 | 0.22 | KILL |
| V5 plain 4N (R34 bound) | 0.095 ± 0.012 | 0.60 | PASS (bound only) |
| V6 Ledoit-Wolf (alpha 0.02-0.03) | 0.016 ± 0.020 | 0.28 | KILL |
| V7 NERCOME | 0.021 ± 0.019 | 0.28 | KILL |
| V8 top-256 + diagonal | −0.011 ± 0.016 | 0.14 | KILL |
| V9a block-diagonal (21 blocks) | −0.002 ± 0.008 | 0.14 | KILL |
| V9b two-stage blocked | −0.007 ± 0.005 | 0.16 | KILL |
| V10 CV beta 0.5 / 1 | −0.005 / −0.008 | 0.20 / 0.19 | KILL (force VR 3.9 / 6.7 vs K = 32.6) |
| V11 guided q = 0.8p + 0.2p_flat | 0.086 ± 0.011 | 0.56 | PASS-T1 |

Findings: (1) at about 6 samples per parameter the force noise, not the metric, limits the sampled step: the exact metric with a sampled force is worse than fully sampled, the exact force with a sampled metric recovers 60% of the exact step. (2) Errors of S and f from the same samples partly cancel (the SR step is a ridge regression of E_loc on O), so improving one ingredient alone (V2, V4, V10) hurts. (3) A flatter sampling distribution at the same N matches 4N plain sampling, consistent with heavy-tailed force estimators near nodes.

## R38 multi-step (RULE_C.md, pre-registered after the screen)

`guided.py`: R21's shift arm (held-out tau ladder, same seeds for held-out sets) with guided S, f draws. R by step: 0.114, 0.176, 0.226, 0.255, 0.283, 0.304, 0.313, 0.331, 0.343, (skip), 0.355, (skip), (skip), 0.359, 0.364. No exact rise. R(15) = 0.364 ≥ 0.295 → GUIDED-WINS; R21's 0.195 reached at step 3. CPU 12.0 min for 247 s wall (BLAS threads were not pinned in this first version; fixed). Late gains about 0.005 per step, R_pol = 0.7475 still far.

Next (proposed, not submitted): T1b, RULE_D.md, three 60-step replicas; `compute_gate.py check ... submit_guided.sbatch --voi voi_guided.md --pre-audit` → first REFUSED (G3: declared 1.25 below cap 2.0), corrected the declared cost: cap 2.0 CPU-h, leaf committed 6.7/300, digest 549b163a0f92, GATE: PASS. Prediction CLOSE 0.45, FAILS 0.35, REACHES 0.20.

## Routes this round

R27 KILLED T1; R28 LIVE (T1 AMBIGUOUS); R29 KILLED T1; R30 LIVE (T0, parked); R31 KILLED G2; R32 KILLED T1; R33 KILLED G2 (minSR = SR by the push-through identity when N exceeds P); R34 LIVE (bound PASS, parked on the sampling/row cost ratio at 24 sites); R35 KILLED T1 (cross-field, signal processing); R36 KILLED T1 (cross-field, statistics; translation repair recorded: force shrinkage, untried); R37 KILLED T1; R38 LIVE T1 PASS (Operation B). Analogy Map Barrier C extended (statistics, signal processing, SCF, QMC guiding functions).

## Compute

All interactive, one thread unless noted. A first batch of probes (about 35 CPU-min) was killed by a session interrupt before finishing and was rerun from scratch; rerun: A probes 34.6 CPU-min, noise split 9.6, guided 12.0 (multi-threaded BLAS), timing 1.0. Total about 1.5 CPU-h including the lost batch; no Slurm job submitted.

## Proposed library entries

* **Force noise limits sampled SR.** At about 6 samples per parameter on a 12-site exact-S state, replacing the sampled metric by the exact one made the step worse (Rbar −0.024 vs 0.016), while the exact force recovered 0.171 of an exact 0.288; metric estimators (shrinkage, blocking, running averages) cannot help, and a flatter sampling distribution helps (0.086). Sources: P4.1 R32, R35-R38, RULE_B.
* **Ill-conditioning, not the LM algorithm, slows 12-site polishing.** At ladder-polished endpoints exact-gradient L-BFGS and truncated Newton gained 0.05-0.17x the ladder LM per CPU-minute; metric-whitened L-BFGS matched it (1.1x). Sources: P4.1 R27-R29.
* (repeated from round 21) check a pre-registered SR step length against the exact step 1 first.
