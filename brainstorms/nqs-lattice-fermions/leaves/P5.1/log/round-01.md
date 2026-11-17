# P5.1 round 1 (2026-11-17): setup, novelty audit, first batch R51.1–R51.7

## Stage 2
`problem.md` written: formulation (representable minimum, sampled cost per step, total samples to accuracy), barriers B1–B4, anti-solution space, novelty audit, Analogy Map (4 restated barriers, 3–4 source fields each).

## Literature searches (WebSearch, standard mode)
1. "neural quantum states optimization stochastic reconfiguration sample efficiency 2025": MinSR (Chen & Heyl, arXiv:2302.01941), large-scale SR (Rende et al., arXiv:2310.05715), time evolution with KFAC/MinSR (Zhang et al., CPC 2025), SR as statistical filter (Hur, NeurIPS 2026 poster), fermionic NQS loss landscape (Dai & Soljačić, arXiv:2601.06939). None measures total samples to the representable minimum.
2. "SPRING Kaczmarz ...": Goldshlager, Abrahamsen & Lin, arXiv:2401.10190 (JCP 2024). Already killed here at 0.1 samples/parameter (P4.1 R15).
3. "leverage score sampling variational Monte Carlo quantum geometric tensor sketching": no NQS/VMC use found; block-diagonal QGT (Shokry et al., arXiv:2510.08430).
4. "variance reduction SVRG stochastic reconfiguration": none found; SPRING momentum convergence with sampling floor (Wang & Liu, arXiv:2604.18357).
5. "randomized Nyström preconditioning natural gradient neural quantum states": NyströmNGD for PINNs (arXiv:2505.11638); Nyström-PCG for neural Gross–Pitaevskii VMC (arXiv:2512.11339).
6. "KFAC neural quantum states lattice ...": KFAC standard in continuum (DeepQMC, FermiNet); second-order NQS strategies (arXiv:2401.17550); no lattice KFAC study.
7. "linear method neural quantum states ...": Frank & Kastoryano arXiv:2104.11011; Zhao & Neuscamman blocked LM arXiv:1702.01481; accelerated LM arXiv:1908.04423.
8. "stochastic reconfiguration convergence sampling error noise floor 2604.18357": Wang & Liu bound residual O(1/N_s); a TFI experiment fits $\sqrt{b/N_s+c/N_s^2}$.
9. "semistochastic ... Petruzielo Umrigar": semistochastic FCIQMC (PRL 109, 230201, 2012; Blunt et al. 2015); not applied to the VMC metric.
10. "Polyak averaging ... variational Monte Carlo": no explicit source returned; parameter averaging in VMC cited from Becca & Sorella (CUP 2017) from memory.

## Generation batch 1 (7 routes; cross-field: R51.2, R51.3, R51.4, R51.5, R51.6, R51.7)
| Route | Name | Gate 0 | Outcome |
| :--- | :--- | :--- | :--- |
| R51.1 | noise-floor scaling of fixed-length guided SR (seed a) | measurement of a known phenomenon, unmeasured here | LIVE; T1a done; T2 unit 1 gate PASS (not submitted) |
| R51.2 | sketched low-rank Nyström metric (seed b) | Nyström NGD known in PINNs, not NQS | KILLED T1 (timing) |
| R51.3 | leverage-score row sampling of the metric | not found in NQS/VMC | T1 PASS (oracle); T1b below |
| R51.4 | semistochastic metric | semistochastic known in FCIQMC | KILLED G4 |
| R51.5 | median-of-means block SR steps | robust stats known; not in VMC | KILLED T0 |
| R51.6 | stochastic-trust-region acceptance | STORM known; not in VMC | KILLED T1a |
| R51.7 | Robbins–Monro decay + Polyak–Ruppert averaging | averaging known in VMC (adopted tool) | LIVE T0, gated on R51.1 |

## T1a for R51.1 / R51.6 / R51.7 (`tests/P5.1_noise_floor_r01/RULE_A.md`, `floor_diag.py`, 1.0 CPU-min)
Stored candidates of job 752885, steps 41–60, 3 replicas (60 steps pooled). Window means (units of the gap per step): realised 0.00202, oracle-best 0.00795, rule-2sigma 0.000. Per replica (realised / oracle): s6000 0.0021/0.0073, s7000 0.0008/0.0062, s8000 0.0032/0.0103. 13 rises, mean 0.0119. Median held-out sigma 4.65e-4; median selected dE_val -2.34e-4. Verdicts: A1 MIXED (realised 0.00202 just over the 0.002 bound), A3 KILLED (R51.6). A2: R(replicas) 0.5355, 0.5121, 0.5080; R(mean theta) 0.5483; mean distance to mean 0.328 → WEAK.
Reported (post hoc, not pre-registered) per-length candidate gains, mean ± se, fraction negative:
* steps 1–20: tau 0.005 +0.0233 ± 0.0016 (0%), 0.01 +0.0282 (12%), 0.02 −0.020 (63%), 0.05 −0.75 (98%)
* steps 21–40: 0.005 +0.0114 ± 0.0009 (0%), 0.01 +0.0046 (43%), 0.02 −0.066 (87%), 0.05 −0.86 (100%)
* steps 41–60: 0.005 +0.0078 ± 0.0006 (0%), 0.01 −0.0036 (68%), 0.02 −0.087 (100%), 0.05 −0.94 (100%)
Reading: the shortest length always descended; the held-out choice picked 0.01 often enough to cancel it. R38's plateau is mostly a selection floor; RULE_B therefore fixes the length.

## Pre-flight for R51.1 T2 (`nf_run.py`, from R38 replica s6000's endpoint, R 0.5355; logs/preflight.txt)
N 16384: step 1 dR +0.0104, step 2 +0.0066; 6243–6272 unique configurations; 0.22 CPU-min per step.
N 65536: step 1 dR +0.0220, step 2 +0.0096; 14,890–14,915 unique; 0.44 CPU-min per step.
`submit_unit1.sbatch` (arms A, B; 2 tasks x 1:10:00) with `voi.md`. Gate check:

```
cap = 2 array x 1.17 h x (1 tasks x 1 cpus) = 2.3 CPU-h
leaf P5.1: committed 0.0 / 300 CPU-h; brainstorm committed 20.5 / 2000.0 CPU-h
skip threshold: base 0.8 (0 scored predictions); leaf no-change streak: 0; script crash run: 0; VOI digest: 583d363c11c7
GATE: PASS
```

## T1 for R51.2 / R51.3 (`tests/P5.1_rowsample_r01/RULE.md`, `rowsample.py`)
Run 1 (`--part timing`, 1.55 CPU-min): energy 0.046, Jacobian 1.121, Gram 0.233, eigh 0.043 CPU-min; R2 KILLED.
Run 2 (`--part all`, first grid): exact metric step gained 0 at lengths {0.005, 0.01, 0.02}; crashed on division by zero before any sampled metric was scored. Amendment 2 (lengths {0.0005, 0.001, 0.002, 0.004}) written then.
Run 3 (5.75 CPU-min, 8.0 GB peak): timing 0.043 / 1.138 / 0.231 / 0.042. Exact metric best gain 1.255e-4 (cutoff 1e-7, length 0.001); candidates c1e-5: +1.72e-5, +2.86e-5, +3.46e-5, −2.22e-5; c1e-6: +3.01e-5, +5.51e-5, +8.45e-5, −6.46e-5; c1e-7: +7.23e-5, +1.25e-4, +2.54e-5, −3.39e-3. Effective dimension Σℓ 2162.9 of 2638; leverage share on below-median Born rows 0.111. phi at m = 8P = 21,096 rows (13.4% of 157,216): Born 0.235, 0.218; guided 0.731, 0.720; leverage 1.037, 0.958. R3 (R51.3) PASS (0.998 vs 0.226).

## T1b for R51.3 (`tests/P5.1_rowsample_r01/RULE_stale.md`, `stale.py`, 8.05 CPU-min, 11.8 GB peak)
Target exact gain 1.255e-4 (same as T1). near (theta_che_lbfgs, eps 7.12e-4): phi 0.985, 0.996; top-10% leverage overlap 0.994. far (SR-trained start, eps 2.86e-3; one draw, reported only): phi 0.948; overlap 0.927. Verdict PASS (phi(near) 0.990).

## Cards, kill index
Cards R51.1–R51.7 written; Kill Index rows for R51.2 (T1), R51.4 (G4), R51.5 (T0), R51.6 (T1).

## Proposed library entries (orchestrator owns library.md)
1. At a sampled-SR plateau, energy-based step-length selection on held-out samples is noisier than one step's gain, so the selection itself can make the floor: in R38's last 20 steps the shortest length gained 0.0078 of the gap per step and never rose, the held-out choice realised 0.0020 (R51.1 T1a, R51.6).
2. In exact-sum polishing of NQS the metric's cost is the per-configuration Jacobian (1.14 of 1.41 CPU-min at 12 sites), not its products, so product sketches (Nyström, range finders) cannot make it cheaper (R51.2).
3. Leverage-score row sampling reproduces the exact metric's polishing step from 13% of the configurations (phi 1.00 against 0.23 for Born-weight rows), and the scores are stable along the optimisation (phi 0.95 with scores 20 steps old): the small-eigenvalue directions are carried by rare configurations that Born sampling misses (R51.3; extends L61's guided-sampling result).

## Compute
Interactive only: floor_diag 1.0, preflight 1.6, rowsample runs 1.55 + ~2.9 (crashed) + 5.75, stale 8.05 CPU-min; total about 21 CPU-min = 0.35 CPU-h. No job submitted.
