# P5.1 round 16 — refill batch R51.26–R51.31, gated (worker, 2026-11-18)
Nothing submitted, no commits. Jobs 773500 and 773528 not touched (their results and logs not opened). Cards R51.7, R51.8, R51.11, R51.16 and library.md not edited; summary.md Answer / What we showed not edited (orchestrator rewriting). New tests in `tests/P5.1_r16_gen/`; all runs interactive, single thread, `ulimit -t 600`, about 17 CPU-min in all (check 0.3, snr arms 2.9 + 3.9 + 2.9, Wiener 2.7, MinSR 4.0).

## T0 facts used across the batch
* Start states (`tests/P5.1_size_scaling_r04/starts/*.json`): 1000 exact-sum SR iterations at each size, same hyperparameters; relative error 5.5e-4 (5×2), 1.1e-3 (6×2).
* Stage mismatch in R51.8: the 5×2 noise-free arm is a limit cycle over steps 61–120 (z −1.1), $G_{\rm ref}$/gap-to-$E_0$ 0.447; the 6×2 one still descends (z −3.7), 0.806. Endpoint exact SR step norm $x^{\mathsf T}Sx$ 3.7e-3 (5×2) vs 2.4e-4 (6×2); endpoint $\kappa_{\rm guided}$ 10.5k vs 1.89M (×180; `tests/P5.1_vr_r12/results/vr_S*_end.json`); start $\kappa$ ×0.70; measured need ×3.79.
* Guided draws at the need: 5×2 $N_{\rm needed}$ 31,280 about 40 $e^{H}$ (781), 6×2 118,600 about 43 $e^{H}$ (2750) (info files).
* Plain-SR length ratio (R51.18 critic replay, 5×2, 2048 draws): $\hat x^{\mathsf T}S_{\rm ex}\hat x/\hat x^{\mathsf T}S_N\hat x$ median 0.67, 10–90% 0.50–0.77.
* Push-through: MinSR with shift ≡ SR with shift at all $N$; `snr_run.py --mode check` reproduces ns_run's step to 2–3e-13 and R51.8's 4096-arm step-3 energy to $<10^{-12}$.

## Routes
| Route | Direction | Code | One line |
| :--- | :--- | :--- | :--- |
| R51.26 `snrcut` | (a)/(c) cross-field: inverse problems, noise-level spectral cut | KILLED T1 | $g$ 0.75 / 0.76 ($\rho_c$=1 at 4096 / 16,384), 0.44 ($\rho_c$=2); Wiener repair 0.85 (KILL); late gain in low-SNR modes |
| R51.27 `tilewarm` | (b) size transfer | KILLED T0 | counted draws all after an SR-trained start; `che` does not tile; transfer heads used up early (R15c, R9.6, L43) |
| R51.28 `blockrank` | (a) blocked/K-FAC metric | KILLED T0 | need tracks neither start (×0.70) nor end (×180) $\kappa$; blocking lost at one step (R32); force noise dominates (L59) |
| R51.29 `startstage` | (d) start state | MERGED (i) into R51.7's 773528 CONT reading; (ii) dismissed T0 | stage-later need = CONT $N$ per size from `score_r15_decay.py`; from-init only postpones the tail |
| R51.30 `kltrust` | (c) cross-field: RL trust regions | KILLED T0 | sampled length a near-uniform 0.71–0.88 of $\tau$ → a $\tau$ change; repair (PPO reuse) = R51.14/R51.17 |
| R51.31 `minsr` | comparator, MinSR half | LIVE T1 GUIDED-AHEAD | Born $N$=1024, $\tau$ 0.00125: $R_{20}$ 0.124 at 819k draws vs guided 0.691, Born SR 16k 0.383 |

Considered and not generated (map onto killed mechanisms): sketch-and-project across steps (a recursion on the direction, R51.13/R51.18, L70); curriculum in $N$ (R51.15), in size (R51.27), in $U$ (L18); FCIQMC semistochastic split (R51.4, G4).

## R51.26 T1 (`RULE_snrcut.md`, scored by `score_r16.py` → `results/score_r16.json`)
Instance R51.8 5×2, guided, $\tau$ 0.005, shift 1e-3, seeds 9000+n, 120 steps, $B$=16 position batches for per-mode sd. Plain seed-9000 $s$: 0.9408, 0.4449, 0.1406, 0.0631, 0.0514 (1k–262k).

| Arm | $s$ | $N_{\rm eq}$ | $g$ | median kept modes | $R_{\rm last20}$ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| $\rho_c$=1, 4096 | 0.520 | 3068 | 0.75 | 327 | 0.534 |
| $\rho_c$=1, 16,384 | 0.176 | 12,529 | 0.76 | 621 | 0.864 |
| $\rho_c$=2, 4096 (reported) | 0.693 | 1803 | 0.44 | 63 | 0.363 |
| Wiener, 4096 (repair addendum) | 0.485 | 3490 | 0.85 | — | 0.572 |

Verdict KILL ($g\le1.25$ at both); prediction was MARGINAL 0.45 / KILL 0.35. Repair addendum written after the KILL and before the Wiener arm; repair KILL (prediction KILL 0.75). Translation gap that mattered: no source condition — the noise-free path's late gain lies in modes with per-step SNR $<1$.

## R51.31 T1 (`RULE_minsr.md`)
Born, $N$=1024, $\tau$=0.00125, 800 steps (819,200 draws), sample-space solve: final trailing-20 $R$ 0.124; references at the same draws (R51.8 16,384 arms, step 50): guided 0.691, Born SR 0.383; no sustained 90%. GUIDED-AHEAD (prediction 0.8). 12-site repeat (about 3.7 CPU-h to guided's 4.51M) not proposed: GUIDED-AHEAD p about 0.9 changes no decision; no voi.md written, no gate check run (no job proposed).

## Translation repairs owed (critic round 15, Major 4) — recorded on the dead cards
* R51.10: trajectory $\kappa$ repair dismissed at T0 (end $\kappa$ ×180, start ×0.70, need ×3.79; any weighting fitted after the result, L51).
* R51.15: repair = step-length schedule (R51.7, running) or a size where small $N$ stalls (R51.8, shelved); no separate route.
* R51.23: energy-only conditioning gains nothing ($E_{\rm loc}$ is already the hop sum); one random neighbour is a longer chain; batched rows keep cost ∝ $V$.

## Proposed library entries (orchestrator)
1. Noise-level spectral filters of the SR step (SNR cut or Wiener shrinkage) raised the sample need ×1.2–2.3 at 10 sites: the late gain sits in modes whose per-step force SNR is below 1 (R51.26).
2. In fixed-length SR with small steps the sampled metric length is a near-uniform 0.7–0.9 rescaling of $\tau$, so trust-region length checks reduce to a step-length change (R51.30).
3. Extension of L67: neither the start nor the end one-state $\kappa$ tracks the multi-step need's size growth (×0.70, ×180 vs ×3.79), and R51.8's two sizes were measured at different stages (plateau vs descending) (R51.28, R51.29).
4. MinSR with a Tikhonov shift is SR with the same shift at every $N$; at 10 sites Born MinSR at $N<P$ is behind Born SR at $N\ge P$ at equal draws (R51.31).

## Summary prose moved by compactor (2026-11-18)
Moved verbatim from `summary.md` to keep it under cap.

Portfolio footnote as it stood:

Killed: R51.2, R51.6 (T1), R51.4 (G4) in round 1; R51.9 (T0) in round 4; R51.12, R51.14, R51.17 (T0), R51.13 (T1, $a$=0.1 only) in round 5; R51.18 (T1) in round 7; R51.5 (T1) in round 8; R51.15 (T1) in round 10; R51.10 (T2 as a predictor) in round 11; R51.19, R51.23, R51.25 (T1), R51.20, R51.22, R51.24 (T0) in round 12; R51.26 (T1), R51.27, R51.28, R51.30 (T0) in round 16. Merged: R51.21 into R51.11 (round 12); R51.29 into R51.7's running test (round 16). Shelved: R51.7 (trigger met at 10 sites only; test `RULE_r14_decay.md` running as job 773528, cap 6; by Addendum 1 only DECAY-HELPS changes its state); R51.8 (round 14, 7×2 rejected by the auditor; 10 → 12 result kept). R51.1 decided: NOISE-FLOOR. See `kill_index.md`.

#### The question and the argument
A fermionic neural quantum state (NQS) is trained by variational Monte Carlo: each step draws $N$ configurations, estimates the energy gradient (force $f$) and the metric $S$ (covariance of the log-derivatives), and moves along $S^{-1}f$ (SR; MinSR solves the same step in sample space, identical when $N\ge P$, the parameter count). On the 12-site state exact-sum optimisation reaches 75% of the gap to the exact energy (L61). "Guided" means a flattened distribution, a published sampler (arXiv:2507.05352); the Question's SR draws from $|\psi|^2$.

*Fixed step, sampling-limited (R51.1).* With the step fixed, the 120-step shortfall fell as $N^{-0.87}$ and barely responded to halving the step, so noise slows the path.

*The size question (R51.8, R51.10; R51.9 killed).* Polynomial cost is decided by how the needed $N$ grows with size against each size's noise-free run; one-step proxies missed it by 2–4× (R51.9). The statistic was re-registered (rev. 2) for the fixed step's bounce before any 12-site data; the 10-site need is 31,280 samples per step and the 12-site need 118,600: ×3.79, above the occupied-support growth ×3.52 and far above $V^3$ (×1.73), so this step is support-like (one seed; margin 1.08×; provisional: R51.8's second-seed clause fired at the 12-site 65k rung and that seed was not run). The 14-site third size (about 160 CPU-h) was rejected by the compute auditor: its likeliest result (AMBIGUOUS, p 0.49) changes nothing, and a support-like one would need a second seed beyond the budget (R51.8 SHELVED). Ridge-regression theory ($\kappa$, R51.10) predicted about 22k and is killed as a predictor.

*Alternatives.* Sampling $\propto|\psi|$ has 4.7× lower $\kappa$ than guided (R51.11) and at 12 sites needed 0.43× guided's total samples (Q-FEWER), but its reweighting variance grows with size (effective samples per draw 45% → 31% → 22% at 10, 12, 14 sites; shape unresolved). A growing samples-per-step schedule (R51.15) was killed (its gain was censoring); recursion on $S$ and $f$ (R51.18) failed under its length rule: energy rose on 82 of 120 steps from oversized steps (cause not isolated; no positive-semidefinite-projected control was run). Direction recursion (R51.13), reuse, reshuffling and sketched leverage (R51.14, R51.17, R51.12) lost. Median-of-means (R51.5) was killed (block steps too often uphill); step decay (R51.7) is shelved, its trigger met at 10 sites. Force variance reduction (round 12, 10 and 12 sites): zero-variance control variates (R51.19) and discrete Stein controls (R51.25) keep ≥ 94% of the variance, because the SR force $O(E_{\rm loc}-E)$ is already zero-variance at an eigenstate; Rao–Blackwell averaging (R51.23) cuts variance 7–9× but costs about a full sample's work per neighbour, a net loss; antithetic pairs (R51.22) and multilevel estimates (R51.24) died at T0, gradient-optimal sampling (R51.20) is dominated by R51.11, and stratification (R51.21) merged into it.

*Strongest objections.* (1) Born-SR comparisons have one Born seed per arm; MinSR's $N<P$ regime is untested. (2) 10–14 sites can exclude support-like growth, not prove a polynomial; conversely the 10 → 12 step is one seed over a size factor 1.2, where ×3.8 is also $V^{7.3}$: far above $V^3$, not proven exponential.

*Limits.* Exact i.i.d. draws, one seed per arm, one network family, nothing at 24 sites.
