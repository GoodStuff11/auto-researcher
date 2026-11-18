# P4.1 round 31: R43 (`flip-even-24`, ledger row 10) read and applied (editor, 2026-11-18)

No jobs submitted, no commits. Job 769413 (R45 T2) running; `tests/P4.1_cheap_su2_r45/` not touched.

## 1. R43 result (job 764949, `tests/P4.1_cheap_su2_r43/RULE.md`, results/nnbf_even_seed{0,1}.json)

Instance and harness as R42 (open 8x3, N_e = 20, U = 8, t' = -0.25, (+,+), H = 16, P = 9711, guided SR, 180 CPU-min one core, 1024 chains). Arm `nnbf_even` = $\psi(n_\uparrow,n_\downarrow)+\psi(n_\downarrow,n_\uparrow)$. `che` not rerun: R42's job 759907 values.

| seed | E_even (131,072 samples) | E_che (R42) | E_che - E_even | E_even - E_nnbf (R42) | rho'_s @ -16.62058 | rho'_s @ -16.58860 | half-budget rho'_s | <S^2> even | steps even / che |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 0 | -16.2660 +- 0.0039 | -16.4314 +- 0.0033 | -0.165 | +0.017 | 0.533 | 0.487 | 0.487-0.490 / 0.457-0.460 (range over seeds, at the two ends) | 0.676 +- 0.019 | 591 / 714 |
| 1 | -16.2746 +- 0.0038 | -16.4295 +- 0.0033 | -0.155 | 0.000 | 0.552 | 0.507 | (as above) | 0.608 +- 0.019 | 610 / 719 |

Half-budget nnbf_even energies (raw parameters, 32,768 samples): -16.058 +- 0.010, -16.032 +- 0.010; <S^2> 0.56, 0.64.

* **Verdict: CHE-STILL-WINS**: every $\rho'_s\le0.80$ at both reference ends (largest 0.552). Half-budget class the same (not BUDGET-DEPENDENT). Stage-2 clause (seed 2) not triggered: no rho'_s within 0.05 of 0.80 or 1.00, classes agree across seeds. Stage-1 verdict final.
* **Validity**: median guided mass fraction F 0.80, median ESS about 907 ($\ge512$), no non-finite step energies, no INVALID-EARLY or INVALID file; final Polyak energies far below half-budget (no divergence); <S^2> estimator finite.
* **Comparability check** (reported, not scored): in-job CPU-min per step 180.17/591 = 0.305 and 180.15/610 = 0.295, against 2 x R42 `nnbf`'s 0.143 = 0.286: ratios 1.07, 1.03, inside [0.8, 1.25]. `nnbf_even` took 0.83-0.85x `che`'s steps (it was expected to take about the same).
* "Energy at `che`'s final step count": not applicable, `nnbf_even` stopped before `che`'s 714/719 steps.
* **Reading**: flip symmetry removed S = 1, 3 and lowered <S^2> from 1.11-1.18 to 0.61-0.68 (with only even S present, weight outside $S=0$ $\le\langle S^2\rangle/6$ = 0.10-0.11), but bought no energy at equal CPU (+0.017, 0.000 against bare `nnbf`): its half-cost steps offset whatever the symmetry gave. So R42's margin is not explained by spin-1 contamination (L1). Not tested: a spin penalty (R46 stays killed: its reopen trigger was MARGINAL or CONTAMINATION-EXPLAINS), S = 2, 4 removal, other sizes, budgets, fillings.
* Prediction (voi.md): CHE-STILL-WINS most likely (p 0.45 without budget dependence, "by less than R42's margin"): class hit; the margin was not smaller (rho' 0.49-0.55 vs R42's rho 0.51-0.56).
* Compute: 7.71 of 10.5 CPU-h (ledger row 21).

## 2. What was changed (RULE.md "What each verdict does", CHE-STILL-WINS)

* Card R43: Outcome LIVE, T2 complete, CHE-STILL-WINS; Evidence with the numbers above; R42 survives.
* Card R42: Outcome notes it survives R43; Strongest case against gains (i) L1, answered by R43; Sharpest Gap points to R43.
* Card R46: Strongest case against answered by R43; Outcome notes not reopened.
* Card R45: Outcome brought up to date from the compute ledger (pilot 769112 PASS; T2 approved decbc3e60d64, running as 769413). Summary portfolio row likewise (was IN-FLIGHT(769112, pilot)).
* Summary: R43 row removed from the portfolio (complete); ledger row 10 closed (CHE-STILL-WINS, R42 stands); Updated, Entered because, Answer's 24-site sentence (rule's permitted wording, "with or without spin-flip symmetry imposed", keeping "at 3 CPU-h per run, one core" and the one-strip, two-seed scope; symmetrised baseline's total spin squared 0.6-0.7 stated), What we showed, What it would take (row-10 item removed; R45 running, R47 after it; overturn clause), Conclusion, chain item (6), Strongest objection (i) answered, Limits, Closure certificate line, Compute spent (35.4 CPU-h in nine completed gated jobs plus 769413 running).
* Certificate draft (log `round-29-critic` section 2): header, preamble, items 1, 3, 4 and 6 updated. Still not adopted: R45 running, R47 gated on it; state stays DORMANT.
* Library (not edited here): proposal to the orchestrator to update L64 from "R43 pending" to the CHE-STILL-WINS result.
* Not edited (orchestrator's files): front page P4.1 row (still says row 10 being prepared), history.md.

Old Answer sentence (replaced): "... the charge-only network ended with about half the energy error of the same network reading spins with no spin symmetry imposed, whose state drifted away from the ground state's expected total spin."
