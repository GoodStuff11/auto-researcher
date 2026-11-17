# P4.1 r22: pre-registered rule (R19 full ladder polish + R20 step 2)

Written 2026-11-17, before `ladder.py` existed and before any r22 output. Sources: the R19 and R20 cards
(`/home/jek354/.claude/jobs/91298e5c/tmp/P4.1_round20.md`, copied into `../P4.1_cheap_su2_r21/RULE.md`), the r19
rung-1 rule (`../P4.1_cheap_su2_r19/RULE.md`), and the r21 results (`../P4.1_cheap_su2_r21/results/`).

## Why this run exists (critic-relevant)

The r19 rung-1 read, rho_pol = 0.761 (CONFOUNDED), was taken on a **frozen polish**. The r21 one-step probe
(`probe_pcg.json`, `probe_che.json`) found both N8 endpoints FROZEN-ARTEFACT under the R19 rule:

* At the fixed relative cutoff 1e-5, 99.97% of `pcg`'s natural gradient (92.7% of `che`'s) lay outside the kept
  metric subspace. The control step lowered eps by 0.01% (`pcg`) and 1.01% (`che`).
* One ladder step lowered eps by 13.0% (`pcg`, cutoff 1e-6) and 7.1% (`che`, cutoff 1e-7). With those single
  steps rho moves 0.761 -> 0.71.

So r19's thresholds were applied to endpoints that were not optimiser plateaus. This run **re-applies the same
r19 rule, unchanged** (<= 0.60 KEEP-LIVE, >= 0.85 RETRACT, else CONFOUNDED), to properly polished endpoints. It is
a re-read of the same pre-registered rule, not a new threshold chosen after seeing 0.761 or 0.71. The r19 verdict
is superseded by the r22 verdict; neither is averaged with the other.

R20 step 1 (r21 `truncate.json`) gave eps_trunc(5)/eps_pcg = 5.49 > 1.10, PASS to step 2. Step 2 is the third
arm here.

## System

Open 4x3 Hubbard strip, N_e = 10 (5+5), U = 8, t' = -0.25, (+,+) via P_G, exact S = 0. Exact sums over all
627,264 configurations (157,216 orbit representatives). E0 = -7.827899034271753, eps = (E - E0)/|E0|.

## Arms (all three start from fixed, already-computed points)

| arm | model | start | start eps | reference gap g for the stopping rule |
| :-- | :-- | :-- | :-- | :-- |
| `pcg` | R2 `DetModel('pcg', H=15)`, P = 2696 | r19 N8 `theta_pcg.npy` | 9.262e-4 | g_pcg = 0.0100523 (r19 SR gap) |
| `che` | `V.Model('che', 16)`, P = 2637 | r19 N8 `theta_che.npy` | 1.217e-3 | g_che = 0.0223728 (r19 SR gap) |
| `rank5` | `pcg` with f0 = U_f D_f^T, U_f, D_f in R^{12x5}, P = 2672 | balanced SVD factors of r19 N8 `pcg` f0 (U sqrt(s), V sqrt(s), top 5) | 5.081e-3 | g_pcg (the R20 card: "the same stopping rule as `pcg`") |

`rank5` keeps `pcg`'s backflow, Jastrow, P_G, P_S and quadrature; only f0 is reparametrised. Its 25-dimensional
gauge (U_f G, D_f G^{-T}) gives zero metric eigenvalues, which the relative cutoff removes.

## Protocol (identical for all three arms)

Per step, at the current theta:

1. Build exact S, H once (r19 `polish.build`, `exact_SH`); one eigendecomposition of the tangent metric (r11 `Tan`).
2. For each relative cutoff c in the ladder {1e-5, 1e-6, 1e-7, 1e-8, 1e-9} (all reuse step 1's S, H and
   eigendecomposition): LM candidate with |z| <= 0.05 (level shift raised only until the cap holds); line search
   s in {1, 1/2, 1/4} on the exact energy, taking the first s whose energy lowers eps by more than 1e-7
   (P4.2 r10 `polish_v2` min-gain). A cutoff whose kept rank equals the previous cutoff's rank is skipped (it gives
   the identical step).
3. Accept the rung with the lowest exact energy. If no rung qualifies, the arm stops (final): **STOP-NOGAIN**.
4. Predicted gain of the accepted step = E - E_LM(s z), the linear-method model energy at the accepted rung and s.
   If it is below 1e-3 g (g from the table), the step is still applied and the arm then stops: **STOP-RULE**.
5. **Step cap: 12 steps per arm** (STOP-CAP). Justification: from r21, a ladder step costs about 4.8 CPU-min
   (`pcg`, `rank5`) and 2.9 (`che`), so 12 steps keep each array task near 1 CPU-h. Even at `che`'s slow r19 tail
   rate (gains shrinking by 0.75 per step) and a first ladder gain of 13% of eps, the untaken tail after 12
   steps is about 0.13 x 0.75^12 / 0.25, i.e. about 2% of eps, which moves rho by about 0.02. The guard below
   catches the cases where that matters.

## Convergence guard (applied before reading any verdict)

For an arm that ended on STOP-CAP, extrapolate a geometric tail from its last two realised gains d_{n-1}, d_n:
r = d_n / d_{n-1}; tail = d_n r / (1 - r) if r < 1, else unbounded. eps_inf = eps_end - tail/|E0|. Arms that ended
on STOP-RULE or STOP-NOGAIN have tail 0. Each ratio is classified twice, with eps_end and with eps_inf. If the two
classes differ (or a tail is unbounded and the class could change), that ratio is **UNSETTLED**: it is reported,
decides nothing, and the next action is a resume (a new gated job), not a re-read.

## Verdicts (read at the endpoints of all three arms)

### rho_pol = eps_pcg / eps_che (r19 thresholds, re-applied)

| verdict | condition |
| :-- | :-- |
| KEEP-LIVE | rho_pol <= 0.60 |
| RETRACT | rho_pol >= 0.85 |
| CONFOUNDED | 0.60 < rho_pol < 0.85 |

### rho_abl = eps_pcg / eps_rank5 (R20 card, step 2)

| verdict | condition |
| :-- | :-- |
| PAIRING-KEPT (R2's pairing mechanism attributed) | rho_abl <= 0.60 |
| PAIRING-RETRACTED (pairing adds nothing beyond the backflow form) | rho_abl >= 0.85 |
| ABL-CONFOUNDED | 0.60 < rho_abl < 0.85 |

### INVALID (no verdict is read)

* `pcg` or `che` start energy differs from r19's recorded N8 energy by more than 1e-8.
* `rank5` start energy (reparametrised model) differs from r21 `truncate.json` trunc[5].E (-7.788124403494811)
  by more than 1e-10.
* The rank-12 reparametrisation (all singular triplets) does not reproduce `pcg`'s N8 energy to 1e-10.
* The `rank5` log-derivatives differ from the chain rule applied to `pcg`'s log-derivatives at the truncated point
  (d/dU_f = G D_f, d/dD_f = G^T U_f, other parameters equal) by more than 1e-9 relative, on 64 configurations.

### INCOMPLETE

An arm cut by the wall-time limit before STOP-RULE, STOP-NOGAIN or STOP-CAP. The script resumes from
`results/state_<arm>.json`; the missing arm is resumed by a new gated job.

## What each joint outcome does to R2 and to the leaf

rho_pol answers "is R2 better than the control after a fair polish?"; rho_abl answers "is that because of the
pairing frame?". `pcg` enters both ratios, so they share its endpoint.

| rho_pol \ rho_abl | PAIRING-KEPT (<= 0.60) | ABL-CONFOUNDED | PAIRING-RETRACTED (>= 0.85) |
| :-- | :-- | :-- | :-- |
| **KEEP-LIVE** | R2 live with its mechanism attributed at 12 sites. R2 moves to the 24-site question; that rung still needs R21 `trSR` VIABLE (scalable optimiser) and the critic. Leaf stays OPEN, R2 is its lead route. | R2 live (beats control) but the gain is not attributed to pairing. R2 stays live at T2a-polished with flag "mechanism unattributed"; any 24-site rung must carry the `rank5` arm. | R2's advantage is the backflow form B1 f0 B2^T, not pairing. R2's pairing claim is retracted; a new route "charge-backflowed UHF in `pcg` form" is opened with its own pre-registered test (L51). R2 as stated is killed at 12 sites. |
| **CONFOUNDED** | Pairing helps inside `pcg`'s code, but `pcg` does not clearly beat `che`. R2 stays at T2a with flag "pairing effect real, cross-architecture ratio confounded"; the 12-site question is closed (no further 12-site polish), the 24-site question needs R21. | R2 stays at T2a, flagged optimiser- and ablation-confounded. 12-site evidence is exhausted for R2: no further 12-site rung on R2 is funded; P4.1 closes R2 at 12 sites as "unresolved at this budget". | Pairing adds nothing and `pcg` does not clearly beat `che`. R2 is killed (mechanism retracted, no clean advantage). |
| **RETRACT** | `che` matches or beats `pcg` although pairing helps within `pcg`'s form: the backflow form is worse than `che`'s. R2 is killed as a competitive ansatz; the result is recorded as a library entry (pairing helps a weaker backflow). | R2 is killed (RETRACT on the r19 rule). | R2 is killed on both counts. With R16/R17 already dead at the polished parent, Q1 of the leaf has no live route and P4.1 goes to EXHAUSTED at this scale unless Q2 (R22 `spin-tree`) is live. |

Any UNSETTLED ratio replaces its column or row by "resume": no change to R2 or the leaf until the resumed arm
gives a settled class. INVALID and INCOMPLETE change nothing.

## Not in this run

R21 `trSR` (sampled optimiser) and R22 `spin-tree` falsifier 2 are separate jobs.

## Erratum (2026-11-17, after `ladder.py --mode check`, before any polish step)

The parameter counts in the arm table were first written from the card's "about 2600". Instantiating the models
gave `pcg` P = 2696 and `rank5` P = 2672 (24 fewer pairing parameters, 120 against 144). The table now carries
these. No rule, threshold, arm or step cap changed. The `rank5` INVALID checks passed locally
(`logs/probe_check_rank5.log`): rank-12 identity 8.9e-16, start energy vs `truncate.json` 1.8e-15, chain-rule
log-derivatives 0.0 relative.
