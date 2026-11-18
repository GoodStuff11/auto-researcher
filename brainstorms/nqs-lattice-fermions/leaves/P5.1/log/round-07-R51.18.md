# P5.1 round 7: R51.18 pre-registered rerun (storm-sf, metric length in S_n)

Rule (card, written 2026-11-17 before rerun data): `storm.py --mode sf` with one change, $d=-\tau v/\sqrt{v^{\mathsf T}S_nv}$,
v = (S_rec + lam)^-1 f_rec unchanged; 5x2 `che` start (P = 1832), guided draws, N = 2048, a = 0.1, tau = 0.005,
120 steps, seeds 9000+n. Foils: plain guided N = 4096 R_plat 0.2740, N = 16384 0.4026, G = 0.1286.
KILL if $R_{\rm plat}\le0.2740$ or divergence; PASS if $\ge0.3383$; MARGINAL otherwise.

Code: `tests/P5.1_r05_gen/storm_v2.py` (copy of storm.py; the only arm change is dropping `S = Sr` so the length uses
the fresh S_n; output suffix `_v2`; an optional `--diag` flag logs diagnostics and writes run_diag.json, arm unchanged).
Run under OMP/OPENBLAS/MKL_NUM_THREADS=1, `ulimit -t 600`.

* Identity check (a = 1, N = 4096, 3 steps; `logs/storm_sf_v2_identity_a1.txt`): energies -6.5470710256681155,
  -6.547085436573092, -6.547100374965894 vs stored plain guided N = 4096 -6.547071025668115, -6.547085436573096,
  -6.547100374965895: match to ~1e-15.
* Arm (`logs/storm_sf_v2_N2048.txt`, `results/storm_N2048_a0.1_sf_v2/run.json`): R at steps 10..120 =
  -0.81, -1.42, -2.16, -2.96, -3.13, -3.36, -3.55, -3.22, -3.11, -3.21, -3.35, -3.66; R_plat = -3.4686. All energies
  finite (no NaN). 1.92 CPU-min.
* Diagnostic (40 steps, trajectory identical to the arm; `logs/storm_sf_v2_diag40.txt`): min eig(S_rec + lam) < 0 at
  every step 2-40 (median -0.0366, min -0.1459, lam = 1e-3); cos(f_n, v) steps 1-5 = 0.226, 0.088, 0.032, -0.018,
  -0.004, median over 2-40 = 0.003, f_n . v < 0 on 31% of steps (uphill). 0.94 CPU-min.
* Total interactive cost: about 2.9 CPU-min (cap 10).

Verdict: $R_{\rm plat}=-3.47\le0.2740$ → **KILL** (T1). The first T1's NaN blow-up was the length; with a sane length the
direction itself fails, because the recursed metric is indefinite from step 2 (about 920 unique draws, fewer than P) and the
solve amplifies its negative directions. Card, kill_index and summary updated; R51.13's kill_index library note
updated (its "until R51.18's rerun is scored" condition is now met). Per-leaf critic review of the kill owed.
