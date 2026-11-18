# P5.1 round 4, R51.8 `nscale`: how do the samples per step that fixed-length guided SR needs grow with size? (pre-registered)
Written 2026-11-17, after R51.1 unit 2 (NOISE-FLOOR at 12 sites) and the pre-flights in `logs/`, before any arm below ran.

Question. R51.1 showed on the 12-site `che` (open 4x3 Hubbard, 5+5, U = 8, t' = -0.25, (+,+), P = 2637) that
guided SR at a fixed metric length tau = 0.005, always taken, removes 0.732 of the gap at 262,144 samples per step
(120 steps) against 0.7475 for exact polishing, and that its shortfall after 120 steps falls about as N^-0.87 with
the samples per step N. The leaf asks whether the sampled cost per step stays polynomial in the system size V. The
statistic is the samples per step this optimiser needs to keep pace with its own noise-free version, measured at
several V where the noise-free version is computable by exact sums.

Instances (open strips, U = 8, t' = -0.25, irrep (+,+), `che` width 16, exact S = 0):
* S10: 5x2, N_e = 8 (4+4), P = 1832, start = `vmc.py --mode train` with the 12-site start's protocol (1000 SR
  iterations, 1024 chains, lr 0.05, decay 0.1, last-25% average, shift 1e-3), seed 0 (`starts/`).
* S12: 4x3, N_e = 10, P = 2637, R51.1's start; its guided arms 16,384 / 65,536 / 262,144 are R51.1's A, B, C
  (seed0 9000; identical step algebra, `ns_run.py` reproduces E(start) to 1e-15).
* Later units only: S14 = 7x2, N_e = 12, and S12' = 6x2, N_e = 10 (same geometry family as S10, controls the
  2-leg/3-leg change).

Arms (`ns_run.py`, tau = 0.005, 120 steps, seed0 9000, exact i.i.d. draws, duplicate draws aggregated exactly):
* exact (N = infinity): exact S_T and f over all representatives (the reference). S10 and S12.
* guided N in {1024, 4096, 16384, 65536, 262144, 1048576} at S10; at S12 R51.1's three, plus 1,048,576 only if
  needed to bracket (unit 1b).
* born (plain SR draws from |psi|^2; at N < P this is MinSR's regime, identical to SR with the same shift in exact
  arithmetic) N in {1024, 4096, 16384, 65536, 262144} at S10; at S12 {16384, 65536, 262144} in unit 1b.

Statistics. E_plat(arm) = mean exact energy over the arm's last 20 steps; G_ref(V) = E_start - E_plat(exact);
shortfall s(arm) = (E_plat(arm) - E_plat(exact)) / G_ref. N_needed(V) = the N at which s = 0.05 for the guided
sampler, by linear interpolation of ln s against ln N between the smallest rung with s <= 0.05 and the rung below
it; censored ("< lowest rung" or "> highest rung") if no bracket exists. Growth references (computed at the starts,
`ns_run.py --info`): g3 = (V2/V1)^3 (polynomial bound) and gH = exp(H[p])(V2) / exp(H[p])(V1), the growth of the
effective support of |psi|^2 (the scale an estimator whose noise tracks the support would follow).

Verdicts after unit 1 (S10 complete, S12 exact reference), r = N_needed(12) / N_needed(10), g3 = 1.728:
* UNRESOLVABLE if gH < 1.5 g3 (the two references are too close at 10 -> 12 to separate): no verdict; unit 2.
* POLY-CONSISTENT if r <= g3: between 10 and 12 sites the samples per step grow no faster than V^3.
* SUPPORT-LIKE if r >= gH: they grow at least as fast as the state's effective support.
* AMBIGUOUS otherwise, or if either N_needed is censored on the side that matters (r's bound does not decide).
* INCOMPLETE if any scored arm stops before step 120.
Next actions: POLY-CONSISTENT -> unit 2 (S14 and S12', the third size and the geometry control that Gate 4 needs)
and, if it holds there, the 24-site protocol of the leaf's closing test; SUPPORT-LIKE -> variance-reduction or
resampling routes for the force first (refill batch), unit 2 deferred, the leaf's Answer reports the 10 -> 12 trend;
AMBIGUOUS / UNRESOLVABLE -> unit 2.

Comparator (the leaf question's "fewer total samples than SR or MinSR", scored per size from the same arms):
T_tot(sampler, V) = min over the sampler's rungs of N x (first step at which E <= E_start - 0.9 G_ref), censored
if no rung reaches it in 120 steps. GUIDED-FEWER if T_tot(born) / T_tot(guided) >= 2 at every size scored;
NO-GAIN if it is <= 1.25 at any size; MIXED otherwise.

What a verdict does and does not show. POLY-CONSISTENT over 10 -> 12 (later 10 -> 14) sites is evidence against a
sample need that tracks the Hilbert space or the effective support, not a proof of polynomial scaling: over a factor
1.2-1.4 in V any polynomial and a slow exponential agree. Confounds named now: geometry (2-leg S10 vs 3-leg S12;
S12' controls it in unit 2), doping (1/5 vs 1/6), P growing as V^2 for this network at fixed width, exact i.i.d.
draws (no Markov-chain autocorrelation), and one seed per arm (a second seed is released only for an arm whose s
lies within 0.3 ln-units of 0.05 at a bracketing rung). The cost per step with duplicate aggregation is reported
(unique draws / N) but not scored: it is a small-cluster saving, predicted at T0 to vanish once exp(H[p]) >> N.
Representativeness (L40), measured before any arm ran (`results/info_S5x2_N8.json`, `info_S4x3_N10.json`): the S10
start is 5 times closer to the lowest (+,+) level in relative terms (eps 5.5e-4 against 3.0e-3) and its P / (orbit
representatives) is 0.165 against 0.017, so S10 is the easier landscape; a SUPPORT-LIKE verdict could partly reflect
that, which S12' (same sector size as S12, same geometry as S10) separates in unit 2. Measured growth references:
exp(H[p]) 781 (S10) and 3035 (S12), gH = 3.89 >= 1.5 g3 = 2.59, so the 10 -> 12 comparison is resolvable.
