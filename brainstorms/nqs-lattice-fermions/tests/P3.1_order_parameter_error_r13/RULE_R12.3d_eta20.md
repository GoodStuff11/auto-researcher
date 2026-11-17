# R12.3d T2 rerun at the measured trained noise level, ETA = (2.0, 1.2) — pre-registered rule

Written 2026-11-17, before the job is submitted. The rule is the original R12.3d T2 rule (leaf
`.subproblems/P3.1_order_parameter_error.md`, Round 12, "T2 premise run for 12.3d", unchanged and used for job 731418),
copied verbatim; only the noise amplitude and the actions differ.

Why the rerun: the per-leaf critic refused P3.1's ANSWERED close because job 731418 (DAV-R 8/8, median width 0.994)
used grade-R synthetic noise ETA = (0.3 in-window, 0.5 junk) on the Davidson vectors, while trained Davidson vectors
were measured at eta_low ~ 0.90, eta ~ 0.85 (T3 job 705951). This run uses ETA = (2.0, 1.2) (coordinator revision, 2026-11-17, replacing the first proposal ETA = (0.9, 0.5)).

Choice of ETA (written before any run of the T2 rerun; `probe_eta_map.py`, `logs/probe_eta_map.log` and
`logs/probe_eta_map2.log`). The measured eta = 1 - cos^2(u, y_exact) and eta_low (the same on the window levels,
t3_scrd.py lines 463-465) are infidelities, not amplitudes. On the 4x3 OBC 4+4 T3 instance (DAV sources, 4 trial
states x 3 sources x 3 draws, pass 1), grade-R noise gives median (eta, eta_low):
  (0.3, 0.5) 0.254 / 0.865;  (0.9, 0.5) 0.514 / 0.882;  (1.5, 1.5) 0.818 / 0.922;  (1.8, 1.2) 0.823 / 0.931;
  (2.0, 1.0) 0.832 / 0.933;  (2.0, 1.2) 0.844 / 0.933;  (2.2, 1.0) 0.852 / 0.935.
The exact Davidson responses have almost no window weight (median 0.002), so eta_low is about 0.87-0.94 for every
amplitude and cannot be set independently: no value hits both 0.90 and 0.85. Chosen: ETA = (2.0, 1.2), total
eta = 0.844 (measured 0.85) and in-window eta_low = 0.933 (measured 0.90, i.e. 0.03 noisier, the conservative side).
ETA = (0.9, 0.5) matched eta_low but gave total eta 0.51, so it would not have answered the critic.

Run: `t2_block_premise_eta.py --instance main --n-low 31 --window-all --cpu-budget 0 --eta 2.0 1.2`, seeds 11, 22, 33
all recomputed (no `--reuse`: the reused seed-11 arms were computed at ETA = (0.3, 0.5); the script refuses the
combination). Per-seed noise stream rng 50000 + seed, as in the original.

**Pre-registered rule** (8 scored items of job 698163; covered: $a_0\in[\lambda_{\min}-0.1|\delta|,\lambda_{\max}+0.1|\delta|]$; width ratio $(\lambda_{\max}-\lambda_{\min})/w_{\mathrm{ex}}$, as at T1):
* INVALID if $\delta$ does not reproduce job 698163 within $10^{-3}$ relative.
* KILL (Gate 2) if the L1R block covers at least 7 of 8 with median width ratio $\le1.5$.
* PASS if DAV-R covers at least 7 of 8 with median width ratio $\le1.5$.
* KILL if DAV-R covers fewer than 5 of 8, or its median width ratio exceeds 3.
* AMBIGUOUS otherwise. Early stop: KILL as soon as 5 covered items become unreachable.

Note: the L1R block carries no noise, so it reproduces job 731418 exactly (6 of 8, median width 1.83); KILL (Gate 2)
cannot fire.

**Actions (set by the orchestrator, 2026-11-17)**
* KILL → R12.3d killed; the leaf runs its Saturation Diagnostic.
* PASS → the claim is re-scoped to synthetic noise at the measured trained level (ETA = (2.0, 1.2): total eta 0.844, in-window eta_low 0.933 on the 4x3 T3 instance); the leaf → EXHAUSTED
  at this budget, with trained 4×4 Davidson vectors (~2400 CPU-h) as the refused step.
* AMBIGUOUS (6 of 8 covered, or 7–8 of 8 with median width ratio in (1.5, 3]) → R12.3d SHELVED with coverage and width
  reported as found; the interval is not claimed at the trained noise level; the leaf proceeds as on KILL.

**Interaction with R12.4 T1b (run 2026-11-17 in this directory, verdict PASS, `logs/t1b_verdict.log`)** — flagged for
the orchestrator to confirm, not a change of the actions above. R12.4's card says it is unshelved if "a later critic
rejects the interval as the leaf's small-gap answer". On KILL or AMBIGUOUS here, that trigger fires, and R12.4's
pre-registered next step (T2 on the real states with 12 exact responses) precedes the Saturation Diagnostic. On PASS,
the interval stands and R12.4 stays SHELVED (G2), so the EXHAUSTED action is unchanged.
