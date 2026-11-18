# P4.1 round 33: R45 T2 (`sup-fit`, job 769413) scored and applied; R47 released, T1 designed (worker, 2026-11-18)

No jobs submitted, no commits.

## 1. Scoring (job 769413, `tests/P4.1_cheap_su2_r45/RULE_T2.md` with both amendments)

Scored with `r45.py --mode verdict` (output `results/verdict.json`). The code is the amended version: single-seed margins
1.0 / 1.65; UNSETTLED if the half-budget class differs; lagging-arm NOT-PLATEAUED (`che` for LOSS, `nnbf` for NO-LOSS)
turns the verdict into UNSETTLED (progress-limited); INTERMEDIATE gets the flag appended.

Instance: open 4x3, 5+5, U = 8, t' = -0.25, (+,+), S = 0; H = 16, P = 2637; V.Model(arm).init(0); L-BFGS-B maxcor 30 on
-log F; maxfun 2000. Both arms `done`, 2001 evaluations, neither CPU-capped (che 349.7 of 485 CPU-min, nnbf 213.6 of 270).

| arm | best 1-F (full) | eps at best | best 1-F (half CPU) | eval at half | last-quarter drop | plateau |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| che | 2.8905e-4 | 5.077e-4 | 3.658e-4 | 976 | 9.0% | PLATEAUED ($\le$ 10%) |
| nnbf | 1.2980e-4 | 2.429e-4 | 1.979e-4 | 900 | 14.3% | NOT-PLATEAUED |

* kappa (full) = 2.227 -> LOSS ($\ge$ 1.65). kappa (half) = 1.848 -> LOSS: half-budget class agrees.
* Lagging arm for LOSS is `che`: PLATEAUED, so no progress-limited downgrade. `nnbf` (leading) NOT-PLATEAUED does not
  enter the rule; its further progress could only raise kappa.
* **Verdict: LOSS** (not UNSETTLED). Unscored: eps ratio of best-fidelity states 2.09 (half 1.75), beside the 1.66 energy
  ratio from L-BFGS energy training.
* Prediction (voi.md): LOSS likeliest, p 0.40: hit. Ledger: `compute_gate.py refresh`, `record 769413` (verdict LOSS,
  changed yes, hit yes); 9.42 CPU-h used of cap 18. P4.1 gated total 44.86 CPU-h in ten jobs.
* Caveats kept on the record: one seed; `che`'s 9.0% last-quarter drop is close to the 10% line (still falling slowly,
  3.17e-4 at evaluation 1500, 2.89e-4 at 2000); kappa grew with budget (1.85 -> 2.23).

## 2. What the LOSS action changed (RULE_T2 "Actions")

* Card R45: Outcome LIVE, T2 complete, LOSS, with the table's numbers; reading and limits.
* Summary: R45 row removed from the portfolio (complete); R47 row now T1 designed; Answer's expressivity sentence rewritten
  (old, replaced: "Whether the restriction costs expressivity is not settled: in unconverged exact-gradient training the
  charge-only network had 1.7 to 2.1 times the energy error on the two 12-site fillings tried, against the spin-reading
  network projected exactly at exponential cost."); What we showed: pairing sentence replaced by the capacity sentence
  (pairing stays in Conclusion and chain item 3); What it would take; Conclusion; chain item (7); Strongest objection
  (ii); Limits; certificate line; Compute spent; ledger row 11 (R47) added so the dormancy has an open row.
* Certificate draft (`round-29-critic` §2) item 4: R45 result noted; not adopted (R47 released).

## 3. R47 decision

Card condition: prepared only after a LOSS that is not UNSETTLED, with (a) the teacher $\langle S^2\rangle$ correction and (b) `che`
PLATEAUED and an agreeing half-budget class. R45 meets (b)'s pattern at 12 sites (LOSS, not UNSETTLED; `che`
PLATEAUED; half agrees), so R47 is **released**. T1 designed in `tests/P4.1_cheap_su2_r47/RULE_T1.md`: 12 sites, exact
teacher sampled (32,768 train / 16,384 held out), students `che`, `nnbf_u` (no P_S), `nnbf_even`; INVALID if held-out
1-F_hat is off exact 1-F by more than 20%; PASS if (1-F)_che / (1-F)_best-unprojected $\ge$ 1.65 at full and half budget with `che`
plateaued; KILL if < 1.65 (no power for the one-sided 24-site test); requirement (a) rehearsed on a dumped 4x3 DMRG MPS
($\langle S^2\rangle$ MPO, $w_T\le\langle S^2\rangle/2$). Cap about 7.5 CPU-h (tier 1-10, full voi note, no auditor). Prediction PASS 0.35, KILL
0.40, UNSETTLED 0.15, INVALID 0.10. Not yet written: script, sbatch, voi.md, local check.

## 4. Exits (ref/states.md)

Ledger rows 1-10 closed; row 11 (R47) open. R47 released, so the EXHAUSTED-at-this-budget certificate is not adopted.
State stays DORMANT. Not edited (orchestrator's files): front page P4.1 row, history.md, library (proposal: L33/L39
entries can cite R45's kappa 2.23 as the 12-site capacity datum).

## Correction (2026-11-18, per-leaf critic, log `round-34-critic`)

§1's "che ... PLATEAUED" and "its further progress could only raise kappa" are corrected: `che` 9.0% is under the
pre-registered 10% line (12% over the last third, 21% over the last half; still falling), not plateaued in the plain
sense; `nnbf`'s further progress raises kappa only if `che` does not improve faster, which the trajectory shows: kappa
1.55, 1.64, 1.95, 2.13, 2.23 at 250, 500, 1000, 1500, 2000 evaluations; 2.62 at equal CPU (`che` at 1204 evaluations).
That trajectory, not the plateau label, is the defence of LOSS. Verdict LOSS unchanged.
