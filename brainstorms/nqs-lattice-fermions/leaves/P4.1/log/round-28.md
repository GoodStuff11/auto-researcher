# P4.1 round 28 (worker): results of jobs 759446 (row 4) and 759907 (R42); exit check

Date 2026-11-17. Task from the orchestrator: write the two results into the leaf, close the ledger rows they decide,
check the DORMANT exits, and write the closure certificate (ANSWERED, scoped to the practical question, if it can
be filled honestly; otherwise EXHAUSTED at this budget), marked "per-leaf critic sign-off owed".

## 1. Job 759446, ledger row 4 (R2 `pcg` against its rank-5 ablation R20, RULE_amend3)

Read from `tests/P4.1_cheap_su2_r22row4/results/verdict_row4.json` and `logs/ladder_pcg.txt`.

* rank-5 arm (`pcg` with its 12x12 pairing matrix truncated to rank 5, i.e. a charge-backflowed UHF in `pcg`'s own
  parametrisation): STOP-RULE after 29 ladder steps, eps 4.4438e-4 (tail 0).
* `pcg`: resumed from 742206's 9 steps; **STOP-RULE at step 24** (accepted step's predicted gain 3.70e-4 g, below 1e-3 g;
  last step dropped eps 2.7737e-4 -> 2.7669e-4, 0.24%). eps_end 2.7669e-4 against the PAIRING-KEPT dominance
  target 2.6663e-4 (= 0.60 x 4.4438e-4): 3.6% short.
* rho_abl = 2.7669/4.4438 = 0.6226; end and extrapolated classes agree (both tails 0): **ABL-CONFOUNDED**
  (KEPT at most 0.60, RETRACTED at least 0.85). Wall 276 min, 4.61 CPU-h ledgered (cap 7.0).
* Correction to the orchestrator's job digest ("pcg still falling at the time limit") and to the compute-ledger
  verdict text ("pcg unconverged at time limit"): `pcg` was not cut by the time limit; it reached its
  pre-registered predicted-gain stop, so the verdict is read at a finished arm, as RULE_amend3 item 4 requires. It
  was still dropping 0.24% per step, so the stop is a step-gain threshold, not proven convergence. The compute
  ledger row was not edited by this worker (owned by compute_gate.py); flagged to the orchestrator.
* Predicted PAIRING-KEPT p 0.70 -> miss.
* Per rule: R2 flagged "mechanism unattributed at 12 sites" (its general pairing matrix gives 38% less error than
  the same code cut to a product of up and down orbitals, below the 40% credit line); row 4 closes. R2 stays
  SHELVED (no 24-site optimiser reaches polished states); R20 step 2 complete, card set to its final outcome.
* Side number (not scored, `che` unconverged, STOP-CAP at 7.22e-4): `pcg` 2.77e-4 is 0.38 of that `che` value;
  not used anywhere because `che` was not resumed (the rho_pol half of row 4 was closed as AMBIGUOUS-predicted).

## 2. Job 759907, R42 practitioner comparison at 24 sites (RULE.md)

Open 8x3, N_e = 20, U = 8, t' = -0.25, P_G(+,+), P = 9711 each; guided SR, 180 CPU-min on one core per run; 4 tasks,
3.4-3.8 h wall, 14.37 CPU-h ledgered (cap 18.0, auditor APPROVED digest 4ed16de4e3f2).

| seed | E_che (Polyak, 131,072 samples) | E_nnbf | E_che - E_nnbf | rho vs -16.62058 | rho vs -16.58860 | half-budget rho (two ends) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 0 | -16.4314 +- 0.0033 | -16.2830 +- 0.0038 | -0.149 | 0.560 | 0.514 | 0.556 / 0.525 |
| 1 | -16.4295 +- 0.0033 | -16.2746 +- 0.0036 | -0.155 | 0.552 | 0.507 | 0.581 / 0.552 |

* rho (mean of two seeds) 0.556 and 0.510 at the two DMRG bracket ends: **CHE-NO-WORSE** (at most 1.10, every seed
  at most 1.30) at both ends, both seeds; half budget same class -> not BUDGET-DEPENDENT; stage-2 clause not triggered
  (every rho_s at least 0.5 from 1.10).
* Steps: `che` 714/719, `nnbf` 1257/1290 (1.8x cheaper per step). Per-step comparison (unscored, computed this
  round from `energy_per_step`): `nnbf`'s step energy averaged over steps 664-764 (around `che`'s final step
  count) is -16.176 (s0) / -16.189 (s1), against `che`'s last-100-step mean -16.435 / -16.432, so per step `che` is
  further ahead than per CPU-minute.
* `nnbf` <S^2> = 1.181 +- 0.017 (s0), 1.112 +- 0.015 (s1), 32,768 samples; the ground state is assumed singlet
  (S = 0; DMRG with S^z = 0 does not fix S). `che` max |S^2_loc| 7.8e-10.
* Validity: tail-median guided mass fraction F 0.80, ESS 900-912, no non-finite energy, no INVALID-EARLY, final
  Polyak energy below the half-budget energy for every task. Valid.
* Gaps to the lower bracket end: `che` 0.189 (1.1% of |E|), `nnbf` 0.338 (2.0%).
* Predicted CHE-NO-WORSE p 0.45 -> hit.
* Per RULE.md: the Answer gains the 24-site equal-compute statement with "at 3 CPU-h per run, one core"; the
  "What it would take" item "like-for-like comparison at 24+ sites" is met for practice, expressivity stays open;
  ANSWERED may be proposed for the practical question.

## 3. Reopening ledger and exit check

| row | outcome (confirmed from summary.md and cards) |
| :-- | :-- |
| 1 | KILLED(G4) R34 |
| 2 | KILLED(T0) R28 |
| 3 | KILLED(T1) R39 |
| 4 | ABL-CONFOUNDED, closed by its rule (this round) |
| 5 | GATE-REFUSED(G2) R22 seeds 1-2 |
| 6 | R16 KILLED(T1); R17 GATE-REFUSED(G2) |
| 7 | R21 KILLED(G2); R38 KILLED(T1) |
| 8 | KILLED(T1) R41 (card R41; it had no kill_index row, added this round) |
| 9 | GATE-REFUSED(auditor) R40; recast as R42 at 24 sites: CHE-NO-WORSE |

No open row, so an exit must be taken this turn (no silent dormancy). Exit order (ref/states.md): exit 1 (OPEN)
does not apply (no route SURVIVED-T1 with a funded next step). Exit 2 (EXHAUSTED at this scale) does not apply:
rows 5, 6 (R17) and 9 are GATE-REFUSED. Exit 3 (EXHAUSTED at this budget) applies to the expressivity part. Exit 4
(ANSWERED) is what R42's RULE allows for the practical question.

Can the ANSWERED certificate be filled honestly? Yes, if and only if the Claim is exactly what was shown: (a) at 24
sites (open 8x3, one filling, two seeds), trained by the same guided SR for 3 CPU-h per run on one core, the
exactly projected charge-only network had a 0.51-0.56 energy-gap ratio to the unprojected spin-reading network,
which was spin-contaminated; (b) at 12 sites, the cost and accuracy facts already in the summary (cost ratio
1.3-3.6 at 12-48 sites; 1.66x restriction cost in unconverged exact-gradient training; row 4 unattributed). The
GATE-REFUSED rows (5, 6-R17, 9) test the expressivity part and the pairing form, not the practical claim; they are
listed in the certificate's "What would reopen it" with their refusals, as for an EXHAUSTED-at-this-budget part.
The Answer stays "Partly." because the Question asks about expressivity, which is not settled.

Decision: State ANSWERED (practical question, scoped), per-leaf critic sign-off owed. Note the skill says the
per-leaf critic signs off before the state changes; the orchestrator instructed setting it now, so the State line
carries "sign-off owed" and the critic can revert it to DORMANT.

## 4. Edits this round

summary.md rewritten (state change, closure certificate, ledger compacted); cards R2, R20 (row 4), R42 (result);
kill_index rows R2, R20 updated and an R41 row added. Library entries proposed to the orchestrator, not written.

Proposed library entries:
* (problem) At 24 sites with equal CPU and the same sampled optimiser, an exactly spin-projected charge-only
  backflow determinant reached about half the energy gap of the unprojected spin-reading backflow (gap ratio
  0.51-0.56, two seeds, half budget agrees), while the latter carried <S^2> 1.1-1.2 in an S = 0 ground state.
  PROVISIONAL (one size, one filling, one budget). Cuts against reading L33-type 12-site restriction costs as
  practical losses.
* (testing) A 12-site, exact-gradient, matched-parameter restriction cost (1.66x) did not predict the 24-site
  equal-compute ranking, which reversed (0.5x): restriction costs measured at exact sums and fixed parameters do not
  transfer to practice when the unrestricted arm loses its symmetry under sampled training. Extends L35/L39.
* (testing) A "predicted-gain" ladder stop (gain below 1e-3 of the gap) can end an arm while it still drops 0.24% per
  step; a dominance target 3.6% away is then unreachable by rule, so ablation verdicts near a line read the stop
  rule as much as the ansatz (row 4, rho_abl 0.623 vs 0.60).

Lint: see the end of the run (0 errors, 0 warnings required).
