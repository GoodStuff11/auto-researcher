# Pre-registered rule: R5.1 (PA) on the 4x4 torus, seed 0 (`torus_pa.py --aggregate`)

Written 2026-11-17, before any target step of the `pa` arm. Restates the leaf's existing R5.1 rule
(`tests/P4.2_size_transfer_r5/obc_transfer.py` docstring, "Decision"), with three declared adaptations.

## Instance and arms
* Target: 4x4 torus, 6+6, t'=-0.25, J=0.4, irrep Tx=Ty=Mx=My=+1, D=-1 (n_irrep 13070, E0=-13.93410177).
* Family: periodic MFNG, 122 parameters (the R3.1 family). Projected exact-sum SR, lr 0.05, shift 0.01.
* `pa` (new, seed 0): L2-SP anchored source fit on 4x2 torus + t8 (3+3), ladder {1e-2,1e-3,1e-4}, largest
  rung with max source relerr <= 1e-2 (else INVALID); copy; 50 target steps.
* Control C = `scr` of job 451914, 3 seeds (MF init + 50 steps), read from
  `../P4.2_size_transfer_r5/results_close31/close31_s{0,1,2}.json`.
* Ablation `cp` = `tr` of 451914 (unanchored min-norm copy from the same sources), 3 seeds.

## Rule (m = median over seeds; `pa` has one seed, so m eps_pa = eps_pa)
* x = 0.017071481 t/site, eps_best of job 451914 over all arms, steps and seeds, FROZEN now
  (1.96% relative error; capacity condition eps V/|E0| <= 3% is therefore met).
* Budgets B in {20, 50}. B is COUNTED if m eps_C(B) - x >= 1e-3 and m[eps_C(B-5) - eps_C(B)] >= 0.02 (m eps_C(B) - x).
  With the frozen x, both B = 20 and B = 50 are counted (computed from 451914 before this run).
* S(B), offset-free step-count speed-up with source cost charged: l0 = min(m eps_C(0), eps_pa(0)),
  l_B = m eps_C(B); spans from first step <= l0 to first step <= l_B; pa's span is charged
  (source ED + every ladder rung wall time) / (pa's mean target step wall time).
  S(B) = m span_C / span_pa; S = 0 if pa never reaches l_B.
* b = 50.
* **PASS**: S(B) >= 2 at both counted B, eps_pa(b) - x <= 0.5 (m eps_C(b) - x), and eps_pa(b) < m eps_cp(b).
* **FAIL**: S(B) < 1.25 at every counted B, or eps_pa(b) - x >= 0.9 (m eps_C(b) - x).
* **AMBIGUOUS**: otherwise. **INVALID**: anchor_ok false, or the seed-0 MF-init check against 451914 fails (1e-8).
  **INCOMPLETE**: fewer than 50 steps.

## Actions
* PASS -> reopen P4.2 on tori (unshelve R9.5); next release seeds 1-2 and the `scr+` control, one at a time.
* FAIL -> close R9.5; with the optimiser row holding, P4.2 is EXHAUSTED at this scale.
* AMBIGUOUS -> close R9.5 at this budget (the leaf reopens only on an R5.1 PASS).
* INVALID / INCOMPLETE -> no decision; fix and rerun.

## Declared adaptations of the leaf's R5.1 rule
1. Family: MFNG (122, periodic grid kernels), not MFNG-O (426, 8x8 quadrature, OBC-only code). This is the family
   whose torus capacity 451914 measured (all seeds <= 2.13%).
2. x: frozen eps_best of 451914 instead of a 600-step capacity-pilot extrapolation eps_inf (no torus pilot exists).
   eps_best >= eps_inf, so the b-clauses are more lenient than with eps_inf; the x = 0 ratio is printed as a diagnostic.
3. Control: 451914's `scr` (MF init), not `scr+`; S in steps (not wall time), because the control's per-step
   clock is not stored and its nodes differ (36.6-52.6 s/step). Same code and family, so a step costs the same.
   The leaf's VOID-vs-NNBF clause is replaced by the 3% capacity condition (no torus NNBF number exists).
