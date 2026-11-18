# R45 T2 rule (written 2026-11-17, before any training; card leaves/P4.1/cards/R45.md)

Question: at 12 sites, does reading only charges cost *capacity* (best achievable fidelity to the exact ground state)
against the same network reading spins, both exactly projected (S = 0, (+,+))?  The fit removes the energy landscape;
fitting the same target from the same init recipe for the same number of exact-gradient evaluations limits the
"how far did each optimiser get" confound as far as an equal-effort fit can (L33).

* System: open 4x3, 5+5, U = 8, t' = -0.25; arms `che` and `nnbf` (H = 16, P = 2637 each; R40's pair), init
  V.Model(arm).init(0); L-BFGS-B (maxcor 30) on -log F; each arm 2000 function evaluations (equal evaluations). The CPU
  caps (485 / 270 CPU-min) are safety stops only; an arm stopped by its cap before 1600 evaluations is INVALID.
* Score: kappa = (1-F)_che / (1-F)_nnbf, best-so-far over the whole run ("full") and over the first half of each arm's
  CPU ("half"). Stage 1 is one seed, so single-seed margins: NO-LOSS kappa <= 1.0; LOSS kappa >= 1.65; else
  INTERMEDIATE. Verdict = the full class if the half class agrees; otherwise UNSETTLED.
* NOT-PLATEAUED flag (appended, does not change the class): an arm's 1-F fell more than 10% over its last quarter.
* Reported, unscored: eps of each best-fidelity state and their ratio (to set beside the 1.66 L-BFGS energy ratio).
* Actions: LOSS -> the Answer's expressivity clause becomes "at 12 sites the charge-only restriction costs capacity
  (fit infidelity ratio kappa), not only optimiser progress"; R47 (24-site distillation) is prepared as the size test.
  NO-LOSS -> the 1.66-2.10 energy ratios are retracted as capacity evidence; the Answer says no capacity cost was found
  at 12 sites; R47's prior on LOSS drops and its VOI is recomputed before preparation. INTERMEDIATE -> the number
  replaces the 1.66 in the Answer. UNSETTLED -> numbers only; seed 1 (a second gated job, cap 18 CPU-h) only if the
  orchestrator judges the Answer needs it. Every outcome: the leaf stays on exit 3 unless R47 runs.
* Prediction: LOSS 0.40, INTERMEDIATE 0.25, NO-LOSS 0.15, UNSETTLED 0.20; NOT-PLATEAUED likely (0.6) on at least one arm.
