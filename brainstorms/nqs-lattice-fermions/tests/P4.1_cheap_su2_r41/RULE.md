# P4.1 round 26, R41 `anderson-raw`: cheapest falsifier (pre-registered)

Written 2026-11-17 before any run of `anderson_raw.py`. Question: does Anderson mixing applied to the raw
fixed-cutoff linear-method (LM) direction, before any line search or cutoff selection, raise the energy gain
per CPU-minute enough to converge the 12-site arms affordably (ledger row 8, R30's translation repair)?

## T0 result that licenses the run (from R30's logs, no compute)
R30 saved only the Anderson coefficients and step norms, which fix every inner product of its three accepted
steps F_k = G(x_k) - x_k (round-26 log). cos(F_0, F_2) = 0.905 (both cutoff 1e-7, s = 1/4); cos(F_0, F_1) = 0.17,
cos(F_1, F_2) = 0.09 (F_1 had cutoff 1e-6, s = 1). Raw-direction norms 1.87, 0.33, 1.61. So at a fixed cutoff
the raw direction is strongly correlated across builds; R30's non-smoothness came from the cutoff switch.
R28's bound (no gradient-call schedule between builds beats the ladder by > 1.8x) does not bound Anderson,
which changes the gain per build, not the calls between builds; cost per build can fall at most from 2.9 to
about 2.1 CPU-min (one cutoff, 2-4 energies instead of up to 15), so PASS needs about 2x the plain gain per build.

## Setup
* Arm `che`, 1 thread (OMP/OPENBLAS/MKL_NUM_THREADS=1 exported before python), `ulimit -t 600` (hard 10
  CPU-min incl. set-up); start `../P4.1_cheap_su2_r22/results/theta_che.npy` (eps 7.217364e-4; must reproduce
  to 1e-9, else INVALID).
* Map: one exact S, H build; tangent at the fixed cutoff 1e-7 (R30's modal accepted cutoff); raw LM direction
  d(x) with |z| <= 0.05; fixed mixing beta = 1/4 (R30's accepted factor at this cutoff, both times):
  G(x) = x + beta d(x), f_k = beta d(x_k). No line search, no cutoff choice.
* Type-II Anderson (Walker & Ni 2011), mixing 1 on f: for m = 1..min(k, 3), gamma = lstsq(DeltaF, f_k) over the
  last m differences, x_AA(m) = x_k + f_k - (DeltaX + DeltaF) gamma. Safeguard: exact energies of G(x_k) and
  every x_AA(m); x_{k+1} = lowest. If none lowers eps by > 1e-7, x_{k+1} = x_k + f_k/2 if that lowers it,
  else stop (STALLED).
* Builds until the CPU cap; results written after every build (the ulimit kill must not lose data). Fewer than
  3 completed builds -> TOO-COSTLY (cost reported, no verdict).

## Baseline and rule
* gamma_LM = 7.27e-6 relative energy per CPU-min (742206 `che`, round 22).
* gamma = (eps_start - eps_end) / CPU-min from the start of build 0 to the end of the last energy evaluation
  (set-up excluded, as for gamma_LM). Also reported: gamma over builds 1+ (where Anderson can act), the
  cosine between successive raw directions, and how often an Anderson candidate was accepted.
* KILL (T1) if gamma < 1.5 gamma_LM.
* PASS-T1 if gamma >= 3 gamma_LM.
* AMBIGUOUS otherwise.
* Scope of a KILL: depth m <= 3 within the builds that fit 10 CPU-min, `che` arm, cutoff 1e-7, beta = 1/4,
  from the polished 742206 endpoint.
