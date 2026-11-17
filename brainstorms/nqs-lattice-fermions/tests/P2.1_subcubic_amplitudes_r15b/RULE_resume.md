# P2.1 R15.1 T1 resume rule (pre-registered before the resume runs, 2026-11-17)

Parent rule: RULE.md (job 742029). This file only adds the resume of one arm; every threshold and guard of RULE.md
is unchanged. RULE.md and voi.md prescribe nothing after INCONCLUSIVE beyond "no resubmission without a new note";
this file and voi_resume.md are that new note.

## Why a resume

Job 742029 returned INCONCLUSIVE for one reason only: `half_F16` hit its 480 CPU-s cap still falling (energy drop
3.7e-4 t over its last 50 iterations > the 2e-4 guard). Every other guard passed (replays exact, spread <= 2.6e-12,
calibration 2.1e-14, polished `full_F4` dE_sign 1.495e-3 >= 1e-3). `full_F4` met the guard (2.7e-5 over 50 it).
The saved `half_F16` history shows a linear tail (7.2e-6 to 7.5e-6 t per iteration from iteration 40 to 188, no
deceleration), so no geometric or power-law fit converges and the energy at convergence cannot be extrapolated
(T0 in voi_resume.md).

## Protocol

1. Start state: `x_pol` of results/polish_half_F16_inst25.json (the lowest-energy point of the 742029 polish,
   full flat vector). Replay checks: exact E - E0 equals the stored 0.021171000533 to 1e-9 t, and the unchanged
   budget (fresh RNG stream) reproduces the stored polished dE_sign 2.529666e-3 to relative 1e-6.
   (Measured in the local probe, resume_probe_output.txt: energy |diff| 0.0.)
2. Resume: exact-gradient L-BFGS-B, options as RULE.md (maxcor 10, ftol 1e-14, gtol 1e-10, unmasked coordinates
   only), empty curvature memory (the 742029 memory was not saved). Stop at the first of: the convergence guard
   (energy drop <= 2e-4 t over the last 50 iterations of the resume, checked from resume iteration 50), the L-BFGS
   tolerances, 3000 iterations, 7200 CPU-s of the L-BFGS loop.
   The stop criterion is the RULE.md definition of a finished polish, which `full_F4` already satisfies at its
   saved state; both arms are therefore stopped by the same criterion and `full_F4` is not re-run.
3. Score: the unchanged budget on the lowest-energy evaluated point -> dE_sign_pol(`half_F16`).
   rho_pol(15) = dE_sign_pol(`half_F16`, resumed) / 1.495182e-3 (`full_F4`, job 742029, unchanged).

## Guards (RULE.md, unchanged)

A resume that does not lower the energy; stopped at the iteration or CPU cap with a last-50 drop > 2e-4 t; replay
mismatch (step 1); calibration error > 1e-3 x dE_tot or budget spread >= 0.05.

## Verdict (thresholds of RULE.md, unchanged)

* **KILL** if rho_pol(15) > 1.25: R15.1 is killed; the R12.3 kill is confirmed as a capacity result after
  polishing. R15.1 is the last live route of P2.1 (R15.6 killed 2026-11-17), so the leaf meets the two-batch
  saturation condition and goes DORMANT with the Saturation Diagnostic.
* **PASS** if rho_pol(15) <= 1.15: as RULE.md (V = 12, 14 polish of both arms as a new pre-registered test, gated
  separately; that test must stop every arm by this same guard).
* **AMBIGUOUS (NOT RESOLVED)** if 1.15 < rho_pol(15) <= 1.25, **or if any guard fails again (INCONCLUSIVE ->
  AMBIGUOUS)**: no more compute on R15.1; to the per-leaf critic as in RULE.md.

## Recorded alongside

Start and resumed E - E0, dE_tot, dE_sign, iterations, function evaluations, stop reason, CPU-s, last-50 drop, the
full energy history, x_pol. Ratio of dE_tot (reported only).
