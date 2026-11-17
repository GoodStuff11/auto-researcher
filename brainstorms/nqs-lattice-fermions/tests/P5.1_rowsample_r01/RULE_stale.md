# P5.1 round 1, R51.3 `levrow` T1b: do stale leverage scores keep the step? (pre-registered)
Written 2026-11-17 after RULE.md's PASS (oracle leverage rows at m = 8P recovered phi = 0.998 of the exact step's
gain, Born rows 0.226, guided rows 0.726) and before `stale.py` was run. Same instance, target point, step, grid
(cutoffs {1e-5, 1e-6, 1e-7} x metric lengths {0.0005, 0.001, 0.002, 0.004}), exact force, m = 8P = 21,096 rows,
2 draws (seeds 1, 2), q = 0.5 Born(target) + 0.5 ℓ_stale / Σℓ_stale, with ℓ_stale the ridge leverage scores of
the Jacobian rows computed at a different parameter point, λ = 1e-7 w_max there. Born weights are always the
target's (in VMC they come free with the samples).
* near: theta from P4.1's exact-gradient L-BFGS run started at the target (`../P4.1_cheap_su2_r25/results/
  theta_che_lbfgs.npy`, eps 7.12e-4 against the target's 7.22e-4): a point about one polishing step away.
* far: the SR-trained start of `che` (eps 2.86e-3), about 20 exact steps away.
Rule on phi(near), mean over the 2 draws:
* PASS if phi(near) >= 0.8: scores refreshed once per few steps suffice; next, a multi-step polish run of the
  row-sampled ladder against the exact ladder per CPU-minute (T1c, design written then).
* KILLED if phi(near) < 0.5: the scores must be fresh, which costs the full Jacobian, so no saving is possible
  (translation repair: cheap leverage estimates from a small row sketch).
* MARGINAL otherwise.
Reported only (one draw, to fit the 10 CPU-min cap): phi(far), which sets how often scores must be refreshed; the overlap of the top-10% leverage sets.
