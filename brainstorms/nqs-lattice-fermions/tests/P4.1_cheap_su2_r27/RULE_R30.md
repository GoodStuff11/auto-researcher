# P4.1 round 23, R30 `anderson`: cheapest falsifier (pre-registered)

Written 2026-11-17 before any run of `anderson.py`. Question: does Anderson mixing over the last ladder-LM steps
raise the energy gain per CPU-minute enough that the 12-site arms can be converged affordably?

## Setup
* Arm `che` only (cheapest map: about 2.9 CPU-min per ladder step), 1 thread, start = job 742206 endpoint
  `../P4.1_cheap_su2_r22/results/theta_che.npy` (eps 7.217e-4; start energy must reproduce it to 1e-9 relative,
  else INVALID).
* Map G(x): one ladder-LM step exactly as `../P4.1_cheap_su2_r22/ladder.py` (one exact S, H build and one tangent
  eigendecomposition reused for every cutoff 1e-5..1e-9; LM candidate |z| <= 0.05; line search s in {1, 1/2, 1/4};
  lowest exact energy over rungs). If no rung lowers eps by > 1e-7, G(x) = x.
* Safeguarded type-II Anderson (Walker & Ni 2011), mixing beta = 1: after build k (k = 0, 1, 2), residual
  f_k = G(x_k) - x_k; for each history depth m = 1..min(k, 4), gamma = least squares of f_k on the last m residual
  differences, candidate x_AA(m) = G(x_k) - DeltaG gamma. Each candidate's exact energy is evaluated (about
  0.045 CPU-min each). x_{k+1} = the lowest-energy candidate if it is below E(G(x_k)), else G(x_k). Every energy is
  exact; all candidate costs count.
* Budget: HARD LIMIT 10 CPU-min total including set-up. Three builds (k = 0, 1, 2) are expected to fit, so the
  probe reaches history depth m <= 2 only; depths 3-5 need 4-6 builds (12-17 CPU-min) and are outside this
  probe. Before each build the script stops if elapsed CPU + last build's CPU would exceed 10 min; if fewer than
  3 builds complete, the outcome is TOO-COSTLY (cost reported, no verdict).

## Baseline and rule
* gamma_LM = 7.27e-6 relative energy per CPU-min (742206 `che`: mean of the last 5 accepted ladder gains / mean
  CPU per step; round 22). Plain-ladder per-step gain at the start: about 2.1e-5.
* gamma_AA = (eps_start - eps_end) / CPU-minutes from the start of build 0 to the end of the last candidate
  evaluation (set-up excluded, as in gamma_LM's per-step CPU).
* Tail: d_k = eps decrease of the accepted iterate at build k (k = 0, 1, 2); r = (d_2/d_0)^{1/2};
  tail_AA = d_2 r/(1-r) (inf if r >= 1). The plain ladder's RULE_amend2 fit at the 742206 endpoint gives
  r_LM = 0.975 and tail 7.7e-4 = 1.07 eps (unsettled). "Tail shrinks" = r < 0.975 and tail_AA / eps_end < 0.5.
* KILL (T1) if gamma_AA < 1.5 gamma_LM.
* PASS-T1 if gamma_AA >= 3 gamma_LM **and** the tail shrinks.
* AMBIGUOUS otherwise.
* Since build 0 has no history, x_1 = G(x_0) always; the Anderson effect is carried by builds 1 and 2. Reported
  also: gamma over builds 1-2 only, and how often an Anderson candidate was accepted.
* Comparison recorded: R28 `plbfgs` T1 AMBIGUOUS, gamma/gamma_LM = 1.12 (second half 0.99) (RULE.md in
  `../P4.1_cheap_su2_r25/`).
* Scope of a KILL: it kills R30 at depth m <= 2 within a 3-build window from the polished endpoint; deeper
  histories are untested here and would need a gated job.
