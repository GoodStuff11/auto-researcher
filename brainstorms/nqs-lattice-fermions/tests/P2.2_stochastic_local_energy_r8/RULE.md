# P2.2 R7.4 (C3') — T1: best possible compensator on H6, localized orbitals

Written 2026-11-17, before any run.

## Question
R7.4's diagonal compensator pays at least sum_dropped(|e| - e) (Statement 4). Any cheap compensator C can
have off-diagonal entries only on pairs that are already kept (any other entry needs extra forward passes).
What is the best certified kept fraction any such compensator can reach?

## Instance (identical to the reopening-review T1, `../P2.2_reopening_review/reopen_t1.py`)
Linear H6, STO-3G, R = 3.0 bohr, Loewdin (LOW) basis, full S_z = 0 sector (400 determinants), exact psi0.
Same integrals, Hamiltonian builder, pair list (|h| <= 1e-12 symmetry zeros are not terms), same 41-point
log grid eps in [1e-7, 1e-1] Ha, tau = 0.2 mHa, same f = n_kept / n_EX(CAN) (expected kept count under
psi0^2, canonical-basis denominator).

## Optimum compensator (per eps)
Dropped set D(eps) = {pairs with |h| <= eps}, kept set K(eps) = {pairs with |h| > eps}.

    b_opt(eps) = min_C psi0^T (C - H_drop) psi0   s.t.  Y = C - H_drop >= 0 (PSD),
                 C symmetric, supported on the diagonal + K(eps).

Equivalently Y_ij = -h_ij fixed on D, free on the diagonal and on K, zero elsewhere.
b_opt is non-decreasing in eps (larger eps: more fixed entries, fewer free ones), so the largest admissible
eps is found by bisection on the grid. The primal solution, repaired to exact PSD (diagonal shift by
-lambda_min if negative), gives a rigorous upper bound on b_opt; the solver's dual value gives a lower bound.
Both are reported. f_opt = f at the largest grid eps with b_opt <= tau; the PASS side uses the repaired
primal, the KILL side uses the dual lower bound (an eps counts as inadmissible only if dual > tau).

## Comparators (reopening-review T1, recomputed here as a check)
f_S* (uncertified LOW SCREEN, H6; better of both bases) = 0.306;  f_diag (diagonal VPS1, LOW, H6) = 1.003
(ratio 3.3).

## Rule
ratio = f_opt / f_S*(6).
* KILL R7.4 if ratio >= 3: even the optimum compensator restricted to kept pairs leaves the certified kept
  fraction at >= 3x uncertified screening at H6; no practical compensator can beat the optimum, and the
  diagonal ratio already grows with N (3.3, 4.7, 6.1).
* PASS-T1 if ratio < 3: propose a <= 1 CPU-h H8 job (4900 determinants); voi.md drafted, not submitted.

Caveat stated in advance: the diagonal ratio at H6 is already only 3.3, so a PASS at H6 is weak evidence;
the H8 job is what decides whether the ratio still grows.

## Budget
<= 10 CPU-minutes interactive (RLIMIT_CPU 600 s); no sbatch.
