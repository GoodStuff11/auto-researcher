# P5.1 round 8: R51.5 `mom` premise arm prepared and gated (not submitted)

Rule: card R51.5, pre-registered 2026-11-17 before any data. Premise (T1): fixed-step guided SR, 4x3 `che`, R51.1's
start, N = 4096/step, tau 0.005, seed0 9000, 240 steps (0.98M draws). KILLED if R(240) < 0.247 or > 80 of 240 steps
raise the exact energy. T2 (K = 4 median arm) not prepared; released only by SURVIVES.

Files (`tests/P5.1_r51_5/`): `submit_premise.sbatch` (ns_run.py, `--signal=B:USR1@600`, python backgrounded, trap
forwards USR1, resumable), `score_premise.py`, `voi.md` (full set; cap 1.6 CPU-h).

* Reproduction: ns_run.py guided 16k seed 9000 step 1 gives E -7.806660361834169, 6161 unique = R51.1 arm A's step 1
  (`logs/preflight_N16384_repro.txt`).
* Preflight 4096 (3 steps, login node): R 0.021, 0.027, 0.049; 0.102 CPU-min/step (16k: 0.142). In-job estimate
  0.102 x (0.36/0.142) = 0.26 CPU-min/step (not the linear 0.09; exact scoring per step is a fixed cost);
  240 x 0.26 x 1.3 = 81 min -> --cpu-cap 81, --time 01:35:00.
* R scale: the card's 0.494 is arm A's stored R (gap 0.022373 from nf_run). `../P5.1_r05_gen/score_common.py` uses E0
  from info_S4x3_N10.json (gap 0.023111), on which arm A's step-60 R is 0.478. score_premise.py uses R51.1's own gap
  so the 0.247 line is on the card's scale. (Flag: R_pol = 0.7475 and Δ in R51.16/R51.11 scoring may mix the scales.)
* Gate (`compute_gate.py check ... --pre-audit`): cap 1.6 CPU-h; leaf committed 45.7/300; GATE: PASS (VOI digest
  fe2575b37e39).
* Prediction SURVIVES, p 0.65: R criterion safe (5x2 4096 matched 16k at equal draws); rises the risk (arm A 18/120,
  5x2 about 0.39 at every N; expected 55-75 against 80, binomial sd about 7).

## Scoring of 769669 (2026-11-18)
`score_premise.py` -> `results/score_premise.json`: 240 steps, 983,040 draws; R(240) 0.4948 (line 0.247, passes;
16k arm A 0.494 at equal draws); R last-20 mean 0.4844; rises 83/240 = 0.346 (line: > 80) -> kill_rises true ->
KILLED at T1 (rise clause). Binomial sd at p = 1/3, n = 240: 7.3; margin 3 rises, about 0.4 sd. Prediction
(SURVIVES, p 0.65; expected 55-75 rises) missed: rises above the predicted range. Arm A (16k) rose 18/120 (0.15);
5x2 fixed-step SR about 0.39 at every N, so rises may be generic to fixed-step SR rather than small-block specific
(recorded as the strongest case against). T2 (K = 4 median) not prepared, not run. Card R51.5 rewritten to KILLED;
kill_index row added; summary updated. Per-leaf critic review owed.

Former-kill paragraph moved off card R51.5 (cap), verbatim:
* **Former kill (T0, round 1)** and why it no longer holds: it rested on a one-step, Born-sampled, held-out probe (L59: 6.2 samples per parameter recovered $0.016\pm0.022$ of an exact 0.288 gain), on P4.1 R15 (0.1 samples per parameter) and on L55 (a different method). R51.1 later showed fixed-step guided SR at 6.2 samples per parameter descends steadily (arm A: 18 energy rises in 120 steps), and at equal total samples smaller batches won at every budget (0.98M draws: 16,384 per step 0.494 > 65,536 0.426 > 262,144 0.290), which points to blocks of about 2 samples per parameter still carrying signal. Near-node forces have infinite variance (Trail, Phys. Rev. E 77, 016703, 2008), so heavy tails persist at block level, median-of-means' best case.
