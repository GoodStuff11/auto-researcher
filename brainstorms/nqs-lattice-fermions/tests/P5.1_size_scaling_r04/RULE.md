# P5.1 round 4, R51.8 `nscale`: how do the samples per step that fixed-length guided SR needs grow with size? (pre-registered)
Written 2026-11-17, after R51.1 unit 2 (NOISE-FLOOR at 12 sites) and the pre-flights in `logs/`, before any arm below ran.

Question. R51.1 showed on the 12-site `che` (open 4x3 Hubbard, 5+5, U = 8, t' = -0.25, (+,+), P = 2637) that
guided SR at a fixed metric length tau = 0.005, always taken, removes 0.732 of the gap at 262,144 samples per step
(120 steps) against 0.7475 for exact polishing, and that its shortfall after 120 steps falls about as N^-0.87 with
the samples per step N. The leaf asks whether the sampled cost per step stays polynomial in the system size V. The
statistic is the samples per step this optimiser needs to keep pace with its own noise-free version, measured at
several V where the noise-free version is computable by exact sums.

REVISION 1 (2026-11-17, before any arm ran; compute auditor, CHANGES REQUIRED): the primary comparison is now
same-geometry (2-leg ladders S10 -> S12' -> S14); the 4x3 S12 is demoted to a geometry check. The original
unit-1 design (S10 vs 4x3 S12) is kept below only where it is now the geometry check.

Instances (open strips, U = 8, t' = -0.25, irrep (+,+), `che` width 16, exact S = 0; each start = `vmc.py --mode
train` with the 12-site start's protocol: 1000 SR iterations, 1024 chains, lr 0.05, decay 0.1, last-25% average,
shift 1e-3, seed 0, `starts/`):
* S10: 5x2, N_e = 8 (4+4), P = 1832, 11,100 orbit representatives (start trained, 9.9 CPU-min).
* S12': 6x2, N_e = 10 (5+5), P = 2637, 156,816 representatives (start: quick job `submit_quick_s12p.sbatch`).
* S14: 7x2, N_e = 12 (6+6), P = 3590, about 2.25 million representatives (start and exact kappa: gated job
  `submit_kappa_s14.sbatch`; a full ladder at S14 is a later unit, costed after its kappa).
* Geometry check only: S12 = 4x3, N_e = 10, R51.1's start; guided rungs = R51.1's A, B, C (seed0 9000).

Arms (`ns_run.py`, tau = 0.005, 120 steps, seed0 9000, exact i.i.d. draws, duplicate draws aggregated exactly;
resumable state saved atomically every step):
* exact (N = infinity): exact S_T and f over all representatives (the reference), at every primary size.
* guided N in {1024, 4096, 16384, 65536, 262144} (the 1,048,576 rung is dropped: a need censored above 262,144 at
  S10 already gives r < 1 against any larger size's bracket); larger rungs only where a bracket requires them.
* born (plain SR; at N < P MinSR's regime) N in {1024, 4096, 16384, 65536, 262144} at S10 and S12'.

Statistics. E_plat(arm) = mean exact energy over the arm's last 20 steps; G_ref(V) = E_start - E_plat(exact);
shortfall s(arm) = (E_plat(arm) - E_plat(exact)) / G_ref. N_needed(V) = the N at which s = 0.05 for the guided
sampler, by linear interpolation of ln s against ln N between the smallest rung with s <= 0.05 and the rung below
it; censored if no bracket exists. Growth references at the starts (`ns_run.py --info`): g3 = (V2/V1)^3 and
gH = exp(H[p])(V2) / exp(H[p])(V1).

Primary verdicts, r' = N_needed(S12') / N_needed(S10), g3 = 1.728, gH from the two starts (S12' probe start:
exp(H) 3864; recomputed at the trained start before scoring), UNRESOLVABLE if gH < 1.5 g3:
* POLY-CONSISTENT if r' <= g3; SUPPORT-LIKE if r' >= gH; AMBIGUOUS otherwise or if censoring leaves r' undecided;
  INCOMPLETE if a scored arm stops before step 120.
* Three-size series (after S14's ladder): POLY-CONSISTENT-3 if N_needed(S14)/N_needed(S10) <= 1.4^3 = 2.744 and r'
  was POLY-CONSISTENT; SUPPORT-LIKE-3 if both steps are SUPPORT-LIKE; AMBIGUOUS-3 otherwise.
Next actions: POLY-CONSISTENT -> S14 ladder, then the 24-site protocol; SUPPORT-LIKE -> force variance-reduction /
resampling batch first, the Answer reports the trend; AMBIGUOUS -> S14 ladder.

Geometry check (secondary), r = N_needed(S12, 4x3) / N_needed(S10). R51.1's arms bound N_needed(S12) in
(65,536, 262,144] provided the 4x3 exact arm's R_plat lies in [0.732, 0.770]: then B has s >= 0.071 and C has
s <= 0.05 (C's s is 0.013-0.021 if R_plat <= 0.7475). Caveat: an exact-limit fit through A, B, C extrapolates past
R_pol (fitted Delta_inf = -0.038, i.e. R_ref about 0.785), in which case C's s = 0.067 and the bound becomes
(262,144, ...); the staged thresholds below assume the bound and are re-checked against the exact arm if it is bought.
So S10's ladder alone decides r when N_needed(S10) >= 151,700 (r <= 1.728) or <= 16,850 (r >= 3.89); otherwise
the 4x3 exact arm (`submit_unit1_ref12.sbatch`) is bought only if kappa or the ladders show that geometry matters
(kappa(S12', 2-leg) and kappa(S12, 4x3) differ by more than 1.5x, or r' and the bounded r disagree in verdict).

Staged release: (1) quick T1s: S12' start + exact kappa (quick job), S14 start + exact kappa (gated job);
(2) unit 1a = S10 ladder (`submit_unit1_s10.sbatch`); (3) unit 1c = S12' ladder (exact + guided + born, prepared
after its trained start exists, pre-flight 1.69 CPU-min per exact step measured on the probe start); (4) S14 ladder.

Comparator (the leaf question's "fewer total samples than SR or MinSR", scored per primary size from the same arms):
T_tot(sampler, V) = min over the sampler's rungs of N x (first step at which E <= E_start - 0.9 G_ref), censored
if no rung reaches it in 120 steps. GUIDED-FEWER if T_tot(born) / T_tot(guided) >= 2 at every size scored;
NO-GAIN if it is <= 1.25 at any size; MIXED otherwise.

What a verdict does and does not show. POLY-CONSISTENT over 10 -> 12 (later 10 -> 14) sites is evidence against a
sample need that tracks the Hilbert space or the effective support, not a proof of polynomial scaling: over a factor
1.2-1.4 in V any polynomial and a slow exponential agree. Confounds named now: geometry (removed from the primary ratio by
revision 1; checked against 4x3), doping (1/5 vs 1/6), P growing as V^2 for this network at fixed width, exact i.i.d.
draws (no Markov-chain autocorrelation), and one seed per arm (a second seed is released only for an arm whose s
lies within 0.3 ln-units of 0.05 at a bracketing rung). The cost per step with duplicate aggregation is reported
(unique draws / N) but not scored: it is a small-cluster saving, predicted at T0 to vanish once exp(H[p]) >> N.
Representativeness (L40), measured before any arm ran (`results/info_S5x2_N8.json`, `info_S4x3_N10.json`): the S10
start is 5 times closer to the lowest (+,+) level in relative terms (eps 5.5e-4 against 3.0e-3) and its P / (orbit
representatives) is 0.165 against 0.017, so S10 is the easier landscape; a SUPPORT-LIKE verdict could partly reflect
that, which the S12' start's own eps (probe start, not yet trained: 3.5e-2) will show. Measured growth references:
exp(H[p]) 781 (S10) and 3035 (S12), gH = 3.89 >= 1.5 g3 = 2.59, so the 10 -> 12 comparison is resolvable.
