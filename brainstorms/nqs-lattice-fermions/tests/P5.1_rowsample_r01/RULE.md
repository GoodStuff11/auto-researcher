# P5.1 round 1, T1 for R2 (sketched low-rank metric) and R3 (leverage-score row sampling of the metric) (pre-registered)
Written 2026-11-17 before `rowsample.py` was run. Instance: 12-site `che` (open 4x3 Hubbard, 5+5, U=8, t'=-0.25,
2637 parameters, 157,216 orbit representatives), at the exactly ladder-polished parameters of job 742206
(`../P4.1_cheap_su2_r22/results/theta_che.npy`). One thread. Target from R28's T0 kill: an exact-sum polishing step
whose metric costs <= 0.33 CPU-min (the exact S build is 1.874 CPU-min).

Part 1 (timing, decides R2). Measure on one thread: J = per-representative log-derivative (Jacobian) rows for all
representatives; G = Gram Y^T Y; D = eigh of the 2637 x 2637 metric; e = one exact energy.
* R2 (any sketch of the Gram from the full Jacobian: randomized Nystrom, range finder, CG on stored rows) is KILLED at
  T1 if J >= 0.33 CPU-min, since every such method computes all Jacobian rows first. If J < 0.33, R2 goes on to a
  step-quality test (not run here).

Part 2 (step quality, decides R3). Exact centred metric S_T and exact force f at the polished point. A step uses a
metric M (exact S_T, or S~ from m rows drawn i.i.d. with probabilities q, unbiased weights 1/(m q), normalised by
the sampled S~_00 and centred) and the EXACT force: d = -tau * Pm Pm^T f / |Pm^T f|, Pm = U_k / sqrt(w_k) over the
eigenvalues above c * w_max, for c in {1e-5, 1e-6, 1e-7} and metric length tau in {0.005, 0.01, 0.02}; gain = the
best exact energy drop over these 9 candidates (0 if none drops). phi = gain / gain(exact S_T).
Row distributions: Born q ∝ size*psi^2; guided q = 0.8 Born + 0.2 (∝ size*|psi|); leverage q = 0.5 Born + 0.5 ℓ/Σℓ,
ℓ_r = y_r^T (S + λ I)^{-1} y_r the ridge leverage of row r of Y (uncentred, with the psi column), λ = 1e-7 w_max(S).
Row counts m = 3P (7911) and 8P (21,096); 2 draws each (seeds 1, 2); phi averaged over draws.
* R3 PASSES T1 if phi(leverage, 8P) >= 0.8 and phi(leverage, 8P) >= phi(Born, 8P) + 0.15.
* R3 is KILLED at T1 if phi(leverage, 8P) < 0.5, or phi(leverage, 8P) <= phi(Born, 8P) + 0.05.
* Otherwise MARGINAL.
Caveat written in advance: the leverage scores here come from the exact Y (an oracle). A PASS is a class bound that
earns the stale-score version (scores from the previous build) a test; it is not a cost saving yet. Reported only:
the effective dimension Σℓ, the share of leverage on rows with Born weight below the median, and phi at 3P.

Amendment (2026-11-17, after Part 1 timing and before Part 2 ran): Part 1 measured 0.046 CPU-min per exact energy, so
the reported-only m = 3P arm (54 energies) is dropped to stay under the 10 CPU-min interactive cap. The verdict
rules use only m = 8P and are unchanged.

Amendment 2 (2026-11-17, after the exact-metric step and before any sampled metric was scored): the exact metric gained
nothing at metric lengths {0.005, 0.01, 0.02} from the polished point (every candidate raised the energy; the
ladder polish of job 742206 took steps of |z| about 0.004 and smaller). The length grid becomes
{0.0005, 0.001, 0.002, 0.004} for every metric (L62: check the exact step at the pre-registered length). Verdict
rules unchanged.
