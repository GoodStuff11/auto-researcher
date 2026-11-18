# P5.1 round 12 — force variance-reduction batch (released by R51.8 SUPPORT-LIKE)

Date 2026-11-18. Worker. Task: 5–7 routes attacking the growth of force noise with support size; gate each; cheapest falsifiers.
R51.8's second-seed job (`tests/P5.1_size_scaling_r04/`) was not touched; `summary.md` not edited (changes returned).

## Library read first
L59 (force noise dominates at ~6 samples/parameter), L61/L73 (guided and $|\psi|$ sampling), L65, L67 (one-step proxies
mislead by 2–4×), L70 (SGD recursions failed), L66 (leverage rows), L25 (cheap surrogate CVs ≤4%), L12.

## Routes (R51.19–R51.25)
| Route | Field | Gate / tier | Verdict |
| :-- | :-- | :-- | :-- |
| R51.19 ZV force control $c\,\partial_kE_{\rm loc}$ | continuum QMC (cross-field) | G0–G3 PASS; T1 | KILLED T1 |
| R51.20 gradient-optimal $q$ | VMC importance sampling | T0 | KILLED T0 (dominated by R51.11's oracle) |
| R51.21 Neyman strata on $\log p$ + RQMC | survey sampling / QMC (cross-field) | T1; QMC G4 | MERGED into R51.11 (strata); QMC KILLED G4 |
| R51.22 symmetry antithetic pairs | MC integration | T0 (+ numerical check) | KILLED T0 |
| R51.23 Rao–Blackwell over a Metropolis move | MCMC statistics (cross-field) | T1 | KILLED T1 |
| R51.24 MLMC across sizes / widths | SPDE numerics (cross-field) | T0 | KILLED T0 |
| R51.25 discrete Stein control functionals | comp. statistics (cross-field), Operation B replacing R51.24; translation repair of R51.19 | T1 | KILLED T1 |

## G0 searches (2026-11-18, quick WebSearch)
* ZVZB: Assaraf & Caffarel (PRL 1999; JCP 2003), Tiihonen et al. 2021 (arXiv:2103.12782), NQS forces Linteau, Moroni,
  Carleo & Holzmann (arXiv:2603.14521), FermiNet forces (arXiv:2207.07810); all nuclear (Hellmann–Feynman) forces; the
  stated barrier is per-iteration cost (arXiv:2603.25381). No lattice SR parameter-gradient use found.
* Rao–Blackwell / waste recycling in VMC/NQS: none found (closest: importance sampling arXiv:2507.05352).
* QMC / stratified for NQS: none found (autoregressive conditional sampling, arXiv:2003.01358; review 2608.21291).
* MLMC for VMC/NQS: none found.
* Discrete Stein CVs: ML gradient estimation (arXiv:2202.09497), lattice-field CVs (arXiv:2501.14614); not in VMC.

## T1 (rule `tests/P5.1_vr_r12/RULE.md`, written before any number; script `vr_t1.py`)
Exact sums over orbit representatives, threads=1, `ulimit -t 600` per process. States: SR-trained start and the
step-120 exact-arm endpoint of R51.8's 5×2 and 6×2 ladders. $\rho=\kappa_X/\kappa_{\rm guided}$ (first-order
step-error constant under guided draws). CPU: 0.16 + 0.16 (5×2), 3.28 (6×2 end), 3.50 (6×2 start; a first attempt was killed for memory and the script was re-blocked by columns, re-verified identical at 5×2 end) CPU-min.

Sanity: $|\langle D\rangle_p|$, $|\langle Z\rangle_p|\le4\times10^{-16}$ of rms; RB mean shift and reversibility
$\le10^{-19}$; $\kappa_{\rm guided}$ at the starts 108,115 (5×2; stored 108,115) and 76,082 (6×2; stored 76,082). PASS.
Spin-flip check (5×2 start, 256 configurations): $\max|\Delta\ln|\phi||$ 1.83e-12, sign ratio +1, $\max|\Delta O|/\max|O|$
1.81e-12.

| quantity | 5×2 start | 5×2 end | 6×2 start | 6×2 end |
| :-- | --: | --: | --: | --: |
| $\kappa_{\rm guided}$ | 1.081e5 | 1.054e4 | 7.608e4 | 1.893e6 |
| $x^TSx$ | 2.80e-3 | 3.71e-3 | 1.81e-2 | 2.38e-4 |
| ZV $\rho(c{=}0.5)$ / $\rho(c{=}1)$ | 24.3 / 96.5 | 209 / 833 | 6.80 / 26.8 | 24.9 / 98.6 |
| ZV $c^\*$ / $\rho^\*$ | 0.0112 / 0.988 | 0.00118 / 0.9988 | 0.0456 / 0.941 | 0.0100 / 0.990 |
| tail (1% smallest $p$) share: base → $c{=}1$ | 0.307 → 4.5e-4 | 0.276 → 2.7e-5 | 0.158 → 9.5e-3 | 0.209 → 1.4e-6 |
| Stein $\rho$ ($J$) | 0.99991 (46) | 0.99995 (46) | 0.99993 (56) | 0.99996 (56) |
| strat $\rho$ / amp40 $\rho$ / strat÷amp40 | 0.0202 / 0.0203 / 0.992 | 0.0157 / 0.0158 / 0.993 | 0.0576 / 0.0577 / 0.9994 | 0.0444 / 0.0444 / 0.9995 |
| RB $\rho$ / cost (1+deg) / eff | 0.132 / 24.3 / 3.2 | 0.108 / 24.3 / 2.6 | 0.157 / 30.8 / 4.9 | 0.139 / 30.9 / 4.3 |

Verdicts by the rule (endpoints): ZV eff 3.0 / 3.0 → KILL; Stein eff 1.0 / 1.0 → KILL; RB eff 2.6 / 4.3, $\rho_6/\rho_5$ 1.28
→ KILL; strat ÷ amp40 ≥ 0.8 at both → MERGED into R51.11. No GROWS flag for any route ($\rho_6/\rho_5$: ZV 0.99, Stein
1.0, RB 1.28, strat 2.8).

## Readings
* The SR force term $O_c(E_{\rm loc}-E)$ is already the zero-variance form for parameter gradients; the
  Assaraf–Caffarel term removes the near-node tail (share 21–31% → ≤0.05%) but adds a bulk term that does not vanish at
  an eigenstate, 100–800× larger. Optimal mixing $c^\*\approx0.01$.
* Stein features from free hop ratios are orthogonal to the force terms (<0.01% removed).
* Rao–Blackwell cuts variance 7–9× but costs a sample per neighbour; degree 23 → 30 from 10 to 12 sites.
* For R51.11 (side result): the best free amplitude-only (40-bin) distribution's gain over guided at the endpoints
  shrinks from 63× (5×2) to 22.5× (6×2); at the starts 49× (5×2) → 17.3× (6×2). One-step constants (L67) — reported, not scored.
* $\kappa_{\rm guided}$ at the endpoints grew 180× from 5×2 to 6×2 (signal $x^TSx$ 16× smaller at 6×2, whose exact arm is
  still descending); one-step, not a need prediction (L67, R51.10).

## Proposed library entry
Hamiltonian-generated zero-mean controls (ZV / discrete Stein) cannot reduce the noise of an SR (parameter) force,
because $O_c(E_{\rm loc}-E)$ is already zero-variance at an eigenstate; the ZVZB gain is specific to Hellmann–Feynman
(nuclear) forces. Evidence R51.19, R51.25 (5×2, 6×2, start and endpoint, $\rho\ge0.988$).

## No job proposed
No route survived to a T2; nothing for `compute_gate.py`.

## Summary changes returned to the orchestrator (not applied here)
* Killed line: add "R51.19, R51.23, R51.25 (T1), R51.20, R51.22, R51.24 (T0) in round 12; R51.21 merged into R51.11".
* Most promising direction / Next step: the force variance-reduction batch is spent (none survive T1); remaining
  levers are R51.11's amplitude-only distribution (whose free binned optimum's gain over guided shrinks 63× → 22.5× from
  10 to 12 sites at the endpoints, 49× → 17.3× at the starts) and the R51.8 second seed / 7×2 ladder.
* What would close it: the ZV clause of the NO condition is now met (ZV force control variates killed at T1,
  $\rho^\*\ge0.94$ at 10 and 12 sites); the NO answer still needs SUPPORT-LIKE at a second size step and R51.11, R51.7
  resolved.
* Updated: 2026-11-18 (round 12: force variance-reduction batch, 7 routes, none survive; `log/round-12.md`).
