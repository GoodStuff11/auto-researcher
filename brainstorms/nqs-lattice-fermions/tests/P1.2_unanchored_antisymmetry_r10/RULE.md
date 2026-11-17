# P1.2 r10: curvature screen of a cross-site / two-hop phase span (reopening ledger row 4)

Written 2026-11-17 BEFORE `t1h_curv2.py` was run. T1, interactive, cap 10 CPU-minutes, no sbatch.

## Why
Ledger row 4 (restated ingredient): a carrier whose function class contains the configuration-specific
phase-descent directions, with a parameter count independent of V. R9.1's screen
(`../P1.2_unanchored_antisymmetry_r9/t1f_curv.py`) showed every per-site span (ratio, ratio2, occ) has
lambda > 0, while the free phase has lambda ~ -1.7. Reviewer's argument: q(g) = -(1/2N) sum psi_n psi_m H_nm (g_n - g_m)^2
descends only through bonds with psi psi H > 0, so g must vary across pairs of neighbouring edges, which
per-site spans cannot do.

## Quantity (unchanged from t1f_curv.py)
lambda(S) = min over g in span(S) of q(g) / var_{psi^2}(g), at the stored t1c foils (NNBF width 6), with the same
`lam_min` code, at sizes 1 (6x2) and 2 (4x3), foil seeds 11 and 22, median over seeds.

## Span S
S is NOT weight-shared, so it contains every weight-shared carrier that is linear in these columns; a kill
therefore also covers weight-shared carriers of this form.
* `ratio2` of R9.1 (per-site f_a(i) and same-site products; 15V columns), included so S contains the old span.
* Cross-site products f_a(i) f_b(j), with a, b over the 5 per-site features (x_up, x_dn, f_0, f_1, f_2), for
  every Hamiltonian bond (i, j) (nearest and t' diagonal neighbours): 25 columns per bond.
* Single edge ratios g(R_e), and two-hop products g(R_e) g(R_e') for every pair of distinct edges sharing a
  site. Edges e: hop of spin up and of spin down on every bond with T_ij != 0, and spin exchange on every nn bond.
  R_e(n) = H_{m_e n} psi_F(m_e) / psi_F(n) (exact foil ratios; 0 when the move is not allowed), and
  g(R) = R / (1 + |R|) (bounded, sign-preserving; fixed before the run).

Control `rand`: random +-1 columns, as many as the numerical rank of S (equal dimension).

Reported with no decision weight: lambda of the cross-site part alone (ratio2 + site products), of the edge part
alone, of R9.1's `ratio2`, of `free_dir`, and the psi^2-weighted fraction of the centred free-phase direction
captured by each span, ||P_S g||^2 / ||g||^2.

## RULE (restated from t1f_curv.py, applied to S; counted sizes 1 and 2; median over seeds 11 and 22)
* FAMILY-KILL if lambda(S) >= 0 at both counted sizes. Ledger row 4 for this span: KILLED at T1.
* FAMILY-OPEN if lambda(S) < 0 and lambda(S) < lambda(rand) at both counted sizes. A <= 1 CPU-h t1c training job
  (a linear phase on S, warm-started along the lambda(S) eigenvector) would follow.
* AMBIGUOUS otherwise.

## Post-run (added after the verdict; does not change it)
Verdict by the rule: FAMILY-OPEN (t1h_curv2_output.txt). Three context checks written after the verdict
(t1h_global.py, t1h_conc.py, t1h_proj.py) show that near-node configurations dominate the lambda metric:
* the unrestricted lambda_min is -2.5e4 to -6.5e5 and sits on a single configuration;
* the lambda(S) eigenvector has participation 12 to 86 configurations, and its realisable gain q/max|g|^2 is about
  1e-6, roughly 100x below the free phase (2e-4 to 9e-4);
* the part of the free direction that S captures is ascending at 4x3 (q > 0).

Recommendation: amend the screen (sup-normalised q, or q of the projected free direction) before the t1c job.
