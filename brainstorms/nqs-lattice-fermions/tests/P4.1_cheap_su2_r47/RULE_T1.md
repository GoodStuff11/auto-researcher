# R47 T1 rule (design written 2026-11-18, round 33, after R45 T2 = LOSS; before any R47 code or data; card leaves/P4.1/cards/R47.md)

Status: DESIGN ONLY. No script, sbatch or voi.md yet; nothing submitted. Next: write `r47_t1.py` (reusing r45.py's Fit/SpinProj
and r43.py's FlipModel), a local gradient/estimator check, `voi.md` (full note: cap 1-10 CPU-h tier), `compute_gate.py check`.

## Why a T1, and what it must decide

R45 T2 (job 769413) found, at 12 sites with exact projection on both sides, kappa = (1-F)_che / (1-F)_nnbf = 2.23 (half
budget 1.85; `che` PLATEAUED). R47 asks the same question at 24 sites, where `nnbf` cannot be projected exactly, by
fitting *unprojected* spin-reading students to samples of a DMRG teacher. Its LOSS inference is one-sided: for an exactly
S = 0 teacher, F(P_S psi) = F(psi)/<P_S>_psi >= F(psi), so an unprojected `nnbf` fit that beats `che` shows a loss.
The T1 answers, at 12 sites where everything can be checked exactly: (1) is the sampled fidelity estimator accurate
enough at the planned sample count; (2) does the one-sided test have power, i.e. does an *unprojected* spin-reading
student still beat `che` by the LOSS margin at 12 sites, where the projected comparison is known to give 2.23. If the
unprojected students cannot reproduce a known loss at 12 sites, a 24-site null would be uninformative and a 24-site
win unlikely, and R47's 26 CPU-h T2 is not worth preparing.

## Design

* System: open 4x3, 5+5, U = 8, t' = -0.25, (+,+) via P_G as in R45. Teacher g = exact ground state (R.Ctx().gs).
* Data: N_tr = 32,768 configurations drawn i.i.d. from |g|^2 (seed 101), fixed for the whole fit; held-out set
  N_ho = 16,384 (seed 202). Estimator on a set X: F_hat = |mean_X[psi/g]|^2 / mean_X[|psi/g|^2]; loss -log F_hat.
* Students (all H = 16, P = 2637, V.Model(arm).init(0), L-BFGS-B maxcor 30, 2000 function evaluations, as R45):
  `che` (exact S, as R45); `nnbf_u` (`nnbf` with P_G but **no** P_S); `nnbf_even` (`nnbf_u` symmetrised under global
  spin flip, r43.py FlipModel). One seed. Every 50 evaluations the exact F (full 627,264 sum) and exact <S^2> are logged
  (reported only; never used in the loss).
* Teacher-correction pipeline check (requirement (a) of the card, rehearsed at 12 sites): rerun the stored 4x3 DMRG
  (`P4.1_cheap_su2_scaling/dmrg_ref.jl`, same settings) with the MPS dumped; compute <S^2> as an MPO, w_T <= <S^2>/2
  (smallest nonzero S(S+1) is 2), and |<MPS|g>|^2 against the dense exact vector; sample 1,024 configurations from the
  MPS and check their amplitudes against the dense MPS vector.

## Score (pre-registered; all numbers from the exact F, with the held-out F_hat reported beside it)

* INVALID (fix and rerun once, no verdict): any student's held-out 1-F_hat differs from its exact 1-F by more than 20%
  at the final state, or train 1-F_hat < 0.8 x held-out 1-F_hat (overfitting the sample); or the DMRG pipeline check
  fails (MPO <S^2> vs dense <S^2> differ by > 1e-6, or sampled amplitudes by > 1e-8 relative).
* kappa_u = (1-F)_che / (1-F)_best, best = the lower-infidelity of `nnbf_u`, `nnbf_even`; full run and half CPU.
* **PASS** (T2 prepared at 24 sites, with the teacher <S^2> correction and the R45 plateau/half-budget conditions):
  kappa_u >= 1.65 at full and half budget, `che` PLATEAUED (last-quarter drop <= 10%), valid.
* **KILL** (KILLED(T1)): valid and kappa_u < 1.65 at full budget: the unprojected students cannot show, at 12 sites, the
  loss that exact projection shows (2.23), so the one-sided 24-site test lacks power at its margin.
* **UNSETTLED**: full and half classes disagree, or `che` NOT-PLATEAUED with kappa_u >= 1.65: numbers only; R47 stays
  LIVE at T1; the orchestrator decides on a second seed (cap as this job).
* Reported, unscored: exact <S^2> and 1 - <P_S> of `nnbf_u` / `nnbf_even` (implicit contamination suppression by the
  fit), and F(P_S psi) >= F(psi) checked numerically; `che` sampled vs R45's exact-fit 1-F (2.891e-4); energy errors.

## Cost (estimate, to be replaced by the local check's per-evaluation timing)

Each sample needs 4 point-group images (P_G); 4 x 32,768 = 131k amplitudes per evaluation against R45's 627k: about
0.2-0.25 of R45's per-evaluation cost (`che` 0.117, `nnbf` 0.065 CPU-min; `nnbf_even` twice `nnbf_u`), plus exact-F
logging every 50 evaluations (+2%). 2000 evaluations: `che` about 60 min, `nnbf_u` about 30, `nnbf_even` about 60;
DMRG 4x3 rerun under 15 min. Planned cap: 3-task array x 2.5 h x 1 core = 7.5 CPU-h (tier 1-10: full voi note, no
auditor). Leaf spend after: about 45 of 300.

## Prediction

PASS 0.35, KILL 0.40, UNSETTLED 0.15, INVALID 0.10. For KILL: unprojected `nnbf` in R42 carried <S^2> 1.1-1.2 and
spin-flip symmetrisation bought no energy (R43), and unprojected weight outside S = 0 enters 1-F directly
(1-F(psi) >= 1-<P_S>), so a 1.3e-4 projected infidelity can be swamped. For PASS: the fit target is exactly S = 0, so
the fidelity loss itself suppresses contamination, and the projected margin (2.23) leaves room.

## Amendment (per-leaf critic, 2026-11-18, before any T1 data)

Objections 3 and 6 of the per-leaf critic (R45/R47 design review), binding; written before any R47 code ran or any T1 data
existed. Where this section and the text above differ, this section governs.
* (3b) **Fourth arm `nnbf_p`**: `nnbf` with P_G **and exact P_S**, fitted to the same sampled data set (same N_tr,
  seeds, estimator, optimiser, maxfun). P_S is block diagonal in the charge configuration (r45.py SpinProj), so psi(x) at a
  sample needs phi on the charge blocks of its 4 point-group images only. It separates "the missing projection" from
  "sampling": KILL with `nnbf_p` still beating `che` by >= 1.65 attributes the lost power to the absent projection;
  KILL with `nnbf_p` also < 1.65 attributes it to the sampled fit itself. `nnbf_p` does not enter kappa_u (best stays the
  better of `nnbf_u`, `nnbf_even`: the 24-site test cannot project); kappa_p = (1-F)_che / (1-F)_nnbf_p is reported beside it.
* (3a) **KILL** additionally requires the best spin-reading student (the arm that defines kappa_u at full budget) to be
  PLATEAUED (last-quarter drop of exact 1-F <= 10%, as for `che`) and the half-budget class of kappa_u to agree (also
  < 1.65). Otherwise UNSETTLED (progress-limited).
* (6) **PASS** requires `che` PLATEAUED as written above; R45's `che` cleared the 10% line by about one point at 2000
  evaluations and was still falling (12% over the last third, 21% over the last half). Therefore **maxfun is raised from
  2000 to 4000** for all four arms (the sampled evaluation is about 4-5x cheaper than R45's exact one, so the budget stays
  in tier 1-10). The full/half classes are read at 4000/2000 evaluations (CPU-time halves, as R45). kappa_u at 2000
  evaluations is reported (unscored) for comparison with R45's 2.23.
* (3c) **KILL's action**: KILL concerns the 24-site test's power only; the 12-site result and R45 are unchanged.
* Prediction re-derived (voi.md): PASS 0.30, KILL 0.25, UNSETTLED 0.35, INVALID 0.10 (UNSETTLED raised: two plateau
  conditions and a half-budget agreement now gate both verdicts).

## Amendment 2 (orchestrator, 2026-11-18, before any T1 data; motivated by a pre-flight overfitting defect, not by any result)

The fixed-data fit (32,768 samples) overfits within 100 evaluations on every arm (train/held-out 1-F_hat 0.31-0.49 at
100, 0.17-0.20 by 300-400; exact 1-F improves with more data), so the INVALID clause would fire by construction.
Changes; every other clause stands (lagging-arm PLATEAUED, half-budget agreement, fourth arm `nnbf_p`, KILL concerns
the 24-site power only, `che` PLATEAUED for PASS, margins, maxfun 4000).
* **Online distillation, fresh batch per L-BFGS restart**: each L-BFGS-B run (maxcor 30) is 50 evaluations on a fresh
  batch of N_tr = 32,768 exact-teacher samples (batch k: seed 101 + 7919 k; batch 0 = seed 101), then restarts from the
  carried theta (best batch loss of that run) on the next batch. Chosen by a pre-flight on `nnbf_u`, 400 evaluations
  (logs/pf_*.txt): per-evaluation resampling inside one L-BFGS run is unstable (exact 1-F jumped 0.031 -> 0.64 at
  evaluation 400) and slower (0.031 at 350); restart every 20 reached 1.15e-2, restart every 50 reached 9.9e-3 (fixed
  data 1.21e-2 at 300). Held-out vs exact stays within 0.92-1.02 throughout.
* **N_ho = 131,072** (seed 202), scored through the full vector.
* **Overfitting clause, re-expressed for online data**: "train 1-F_hat" is the logged theta's 1-F_hat on the next,
  not-yet-seen batch (`infid_fresh`); INVALID if it is < 0.8 x held-out 1-F_hat at the final state. The in-batch
  1-F_hat is biased low by construction within a 50-evaluation run (0.35-0.49 x held-out in the pre-flight) and is
  reported, not scored. The held-out-vs-exact 20% clause is unchanged.
* Prediction re-derived from the pre-flight (voi.md): KILL 0.40, UNSETTLED 0.30, INVALID 0.25, PASS 0.05.
