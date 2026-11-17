# P4.1 round 24: ledger rows 1 (R34) and 3 (force shrinkage, R39) (worker, 2026-11-17)

Both tasks run interactively on kim-compute-03 with OMP/OPENBLAS/MKL/NUMBA threads = 1 set before Python and
`ulimit -t 600`. CPU used: 2.0 CPU-min (row 1) + 4.7 CPU-min (row 3) = 6.7 CPU-min, each under its 10 CPU-min cap.
Row 2 (R28 hybrid) was closed in parallel by another worker (log `round-24-R28`).

## 1. Row 1, R34 `reuse`: 24-site cost ratio → KILLED(G4)

Pre-registered (round 22, card and ledger): reuse saves only the Markov-chain cost of a configuration; speed-up per
step at most (sampling + row)/(row); below 2 → KILLED(G4).

What a reused row needs at the new parameters: amplitude ratio (for the weight), local energy E_loc (the force is
cov(O, E_loc) and E_loc depends on theta), and log-derivatives O. So row = E_loc + O (+ amplitude); sampling = the
Metropolis proposals (0.5 sweeps x N_e = 10 proposals per chain per step; `t_mcmc` also includes the amplitude
recomputation at step start, which reuse needs too, so the ratios below are upper bounds).

Measurements, open 8x3, N_e = 20, U = 8, t' = -0.25, P_G (+,+), 1024 chains, seconds per SR step
(sampling / local energy / log-derivatives):

| source | arm | sampling | E_loc | O | (s+row)/row | (s+O)/O only |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| job 699824, 4 threads, median of logged steps (61 / 34) | che | 0.27 | 3.12 | 0.31 | 1.079 | 1.87 |
| same | pcg | 0.79 | 9.42 | 0.66 | 1.078 | 2.20 |
| this round, `pfretime/retime.py --mode timing --sizes 8 --n-chains 1024 --threads 1 --repeats 3` | che | 0.45 | 6.61 | 0.79 | 1.061 | 1.57 |
| same | pcg | 0.97 | 17.56 | 1.09 | 1.052 | 1.89 |

Output: `tests/P4.1_cheap_su2_r34cost/logs/retime_L24_t1.txt`, JSON in the same directory (2.0 CPU-min, 123 s wall).
(12-site, job 699824: 1.15-1.20.)

Verdict: ratio 1.05-1.08 < 2 → KILLED(G4). The log-derivative-only reading (the orchestrator's shorthand) is
1.57-2.20, but it omits E_loc, which a reweighted row cannot skip; even that reading is below 2 at one thread for both
arms. No T1 of reuse on top of guided SR is run or prepared (the rule closes the route before it). Card R34 rewritten
(KILLED, all fields), kill-index row added.

## 2. Row 3, R39 `js-force`: James–Stein shrinkage of the force (translation repair of R36) → KILLED(T1)

Route card written first (R39), then gates at T0:
* G0 narrow PASS: SNR-based regularisation of SR force modes exists in time-dependent VMC (Schmitt & Heyl, PRL 125,
  100503, 2020, arXiv:1912.08828); adopted as the leaf's optimiser, not claimed as new (as R21, R38).
* G1 PASS, G3 PASS: estimator change only; no symmetry or Hard-Core change.
* G2: not predictable from L19 (shrinkage is biased at finite N and changes the flow).
* G4 PASS on paper: O(N P^2) per step, or O(N^2 P) in sample space when P > N (24 sites).
* T0 kill of a sub-variant: whole-vector James–Stein toward zero only rescales the step, and the fixed-length scoring
  removes scale, so it cannot change R; not tested.
Prediction before the run: KILL p about 0.6 (the exact-force headroom is spread over many low-SNR modes; R21 found
eigenvalue truncation no better than a shift).

Rule `tests/P4.1_cheap_su2_r25/RULE_E.md` (written before the run): RULE_B protocol; J1 mode-wise positive-part JS
(empirical-Bayes Wiener factor max(0, 1 - var_k/ft_k^2) in the sampled-S eigenbasis), J2 grouped positive-part JS
(32 eigenvalue groups, whitened coordinates); PASS-T1 if max Rbar $\ge$ base + 0.05, KILL if $\le$ base + 0.02 (covers the
orchestrator's "< 2 x base"), else AMBIGUOUS. Guided-draw variants reported only.

Script `tests/P4.1_cheap_su2_r25/shrink_screen.py` (same draw stream as noise_split.py; base reproduced to 4 digits,
seed 3001 R +0.0454/-0.0082/-0.1717 as in round 22). Log `logs/shrink_screen.txt`, JSON `results/shrink_screen.json`.
4.7 CPU-min (user 235 s + sys 48 s), 283 s wall.

| variant | seed 3001 | 3002 | 3003 | Rbar ± sd |
| :--- | ---: | ---: | ---: | ---: |
| V1 base | +0.0454 | -0.0063 | +0.0100 | 0.016 ± 0.022 |
| J1 mode JS | +0.0228 | -0.0127 | +0.0059 | 0.005 ± 0.015 |
| J2 group JS | +0.0098 | -0.0094 | -0.0229 | -0.008 ± 0.013 |
| guided base (reported) | +0.0851 | +0.0731 | +0.0989 | 0.086 ± 0.011 |
| guided + J1 (reported) | +0.0600 | +0.0533 | +0.0731 | 0.062 ± 0.008 |
| guided + J2 (reported) | +0.0574 | +0.0414 | +0.0704 | 0.056 ± 0.012 |

Diagnostics: J1 kept 33-34% of modes, mean factor 0.16; many J2 groups shrunk to exactly 0. Most force modes have
estimated SNR below one, so shrinking toward zero removes signal with noise.

Verdict: best shrink 0.005 $\le$ 0.016 + 0.02 → KILL (T1). Ledger row 3 → KILLED(T1). Card R39 written (KILLED, all
fields), kill-index row added. Translation gap: James–Stein pays when signal is concentrated relative to noise (or a
good non-zero target exists); the SR force here is spread over many modes, each below its noise.

## 3. Ledger state after this round

Rows 1, 2, 3 KILLED; rows 4, 5 GATE-REFUSED(G2); row 6 R16 KILLED, R17 GATE-REFUSED(G2); row 7 R21 MARGINAL
(superseded), R38 KILLED(T1). No row is open; at least one is GATE-REFUSED, so exit 3 (EXHAUSTED at this budget)
applies per ref/states.md; the orchestrator decides and writes the closure certificate.

## 4. Proposed library entries

* Force shrinkage cannot replace samples in sampled SR: mode-wise and grouped James–Stein shrinkage of the force
  lowered the one-step gain (0.016 → 0.005 / -0.008; on guided draws 0.086 → 0.062 / 0.056) while the exact force
  gives 0.171; with R36 (metric shrinkage) the shrinkage analogy from statistics fails at both ends (P4.1 R36, R39).
* In this VMC code the local energy is 85-90% of a sampled SR step at 12-24 sites, so sample reuse or any scheme
  that saves only drawing gains at most about 8% (P4.1 R34).
