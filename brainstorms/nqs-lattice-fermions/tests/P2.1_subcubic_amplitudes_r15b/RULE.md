# P2.1 R15.1 T1 rule (pre-registered before any run, 2026-11-17)

Source: route card R15.1 in the Round 15 route-generator output (`P2.1_round15.md`). This file restates its rule
before `polish.py` is run in polish mode. R15.1 is a revival of R12.3 (killed by job 731409, rho(15) = 1.915) in
the sense of L51: it has its own pre-registered test and does not overturn the R12.3 verdict.

## Question

Is R12.3's V = 15 loss (`half_F16` dE_sign 2.83e-3 against `full_F4` 1.48e-3) an optimiser artefact of scoring
the best point of a noisy SR trajectory, or a capacity limit?

## Protocol (both arms, identically; a polish of one arm only is excluded)

Instance 25 (5x3 OBC, 7+7, V = 15, D = 51480), t-t'-J with t' = -0.25, J = 0.4.

1. **Start state = the scored state of the original run.** `full_F4`: `best_x` of
   `../P2.1_subcubic_amplitudes_r12b/results/ckpt/full_inst25_s11.json` (its best_E equals the scored best,
   step 482). `half_F16`: the checkpoint `../P2.1_subcubic_amplitudes_r12d/results/half_F16/ckpt/half_inst25_s11.json`
   was written at step 500, before the scored best (step 526, E - E0 = 0.023879); its `best_x` (E - E0 = 0.026089)
   is therefore not the scored state. The scored state is recovered by resuming the unchanged SR loop from that
   checkpoint (exact resume: saved x and RNG state) to step 526, and every replayed energy must equal the stored
   trajectory to 1e-9 t (else INCONCLUSIVE: replay mismatch).
2. **Replay score.** The unchanged budget (`p11_core.budget`, same RNG stream: calibration draw first) on each start
   state must reproduce the stored dE_sign to relative 1e-6 (else INCONCLUSIVE: replay mismatch).
3. **Polish.** Exact-gradient L-BFGS (scipy L-BFGS-B, no bounds, maxcor 10, ftol 1e-14, gtol 1e-10) of the exact
   energy <psi|H|psi>/<psi|psi> (full sum over D, same ansatz code, no projection; masked entries of `half` held at
   zero by optimising only unmasked coordinates), at most 300 iterations or 480 CPU-s per arm, single thread.
   The polished state is the lowest-energy point evaluated. No metric and no cutoff is used, so the P4.2 caveat
   about a fixed relative metric cutoff does not apply.
4. **Score.** The unchanged budget on each polished state (fresh RNG stream as in step 2) gives dE_sign_pol.

rho_pol(15) = dE_sign_pol(`half_F16`) / dE_sign_pol(`full_F4`).

## Guards (any one makes the result INCONCLUSIVE)

* A polish does not lower its arm's exact energy.
* An arm stopped at the iteration or CPU cap whose energy fell by more than 2e-4 t over its last 50 iterations.
* Replay mismatch (steps 1-2).
* The Round 12 counting guards on the polished budgets: calibration error > 1e-3 x dE_tot, budget spread >= 0.05,
  or dE_sign_pol(`full_F4`) < 1e-3 t (the premise: below it the size cannot test sign capacity).

## Verdict

* **KILL** if rho_pol(15) > 1.25: R15.1 is killed; the V = 15 loss of R12.3 stands as a capacity result after
  polishing, and the R12.3 kill is confirmed.
* **PASS (follow-up licensed)** if rho_pol(15) <= 1.15: polish V = 12 and V = 14 the same way (both arms, from their
  saved checkpoints) and apply the Round 12d rule (`t1_sub.decide`, unchanged) to the four polished ratios
  (V = 8, 12, 14, 15). That is a new pre-registered test, gated separately.
* **AMBIGUOUS (NOT RESOLVED)** if 1.15 < rho_pol(15) <= 1.25: sent to the per-leaf critic. Action: no further
  compute on R15.1 until the critic rules; the critic chooses between closing R15.1 (treating the polished V = 15
  ratio as still above the trend bar 1.15 of the Round 12d rule, which would fire the trend clause again) and
  licensing the V = 12 and 14 polish.

## Recorded alongside

Per arm: start and polished E - E0, dE_tot, dE_sign, iterations, function evaluations, stop reason, CPU-s, energy
drop over the last 50 iterations. Polished ratio of dE_tot (reported only).
