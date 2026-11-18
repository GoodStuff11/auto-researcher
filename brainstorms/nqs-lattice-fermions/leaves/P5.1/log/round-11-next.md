# Round 11 — next R51.8 spends after SUPPORT-LIKE at 6×2 (worker, 2026-11-18)

No submission, no Slurm. P4.1 untouched. New files in `tests/P5.1_size_scaling_r04/`:
`submit_r11_s12p_seed2.sbatch`, `voi_r11_s12p_seed2.md`, `sens_r11_seed2.py` (→ `logs/sens_r11_seed2.txt`),
`cost_r11_s14.py` (→ `logs/cost_r11_s14.txt`).

## 1. Seed-clause job (prepared; the gate refuses it, as it should)
* Clause scope. RULE.md names "an arm whose s lies within 0.3 ln-units of [0.10] at a bracketing rung". Only
  the 65,536 rung qualifies (0.28). The 262,144 rung (0.38) is outside the clause and is added only as the
  matched bracket, as at S10. 16,384 is not a bracketing rung. The exact arm is reused (769376).
* Job: seed0 19000, guided 65k then 262k. The rates are in-job measurements from 769376: 53.1 + 78.4 CPU-min
  per 120 steps, plus 2 min setup, × 1.3 = 174 min (`--time 02:55:00`). Memory 3.28 GB MaxRSS → 6G. The job is
  resumable through state.npz, with USR1/TERM forwarded by the `B:` trap.
* Sensitivity (`sens_r11_seed2.py`). The margin is 0.0744 ln-units. ∂lnN/∂ln s is 1.20 at 65k and 0.90 at 262k.
  The pooled S10 single-seed sd is 0.042 in ln s (0.0039 absolute). The two-seed mean moves ln N by 0.036 (65k
  only) or 0.044 (both rungs), so z = 1.7–2.1. P(flip to AMBIGUOUS) is 0.03–0.10 with t at 3 dof, and 0.12–0.23
  if 6×2 seed noise were double S10's. The 6×2 detrended sd_w (0.0022–0.0024) is not larger than S10's, so
  doubling is unlikely.
* Prediction: SUPPORT-LIKE is unchanged, p 0.88. The predicted verdict is the one already acted on, so
  prediction-changes-decision is no.
  `compute_gate.py check … --pre-audit`: cap 2.9 CPU-h; **GATE: REFUSED — G2** (p 0.88 ≥ 0.8, no decision
  change). Recommendation: record the prediction (SUPPORT-LIKE holds under a second seed, p 0.88) and act on it.
  The seed becomes decision-relevant only once 7×2 is bought, because SUPPORT-LIKE-3 needs step 1 to be
  SUPPORT-LIKE. At that point fold it into the 7×2 package, where it is about 1% of the cost.

## 2. Cost of the 7×2 ladder (not prepared)
The model is in-job CPU-min/step = a + b·unique, fitted on the 769376 guided and exact arms (a 0.145,
b 1.75e-5). It is scaled to 7×2 by the measured energy-pass and per-row times (`info_S7x2_N12.json`). For
the exact arm it is cross-checked against 763903's in-job 7×2 κ passes.
* Exact arm: 59–69 CPU-min/step. Over 120 steps that is 118–137 CPU-h expected, a cap of 153–178 CPU-h, and
  5–6 days of wall time on one CPU. Memory is 12.8 GB MaxRSS per pass (763903), so request 32G.
* Guided arms per 120 steps: 16k 9.6, 65k 10.6, 262k 12.8, 1M 18.0 CPU-h. The exact energy pass sets a
  floor of about 9 CPU-h per arm. Born: 262k 11.7, 1M 14.4.
* Thresholds. SUPPORT-LIKE at the second step needs N(7×2) ≥ 696k (against S10, gH 22.2) or ≥ 749k (against
  S12', gH 6.32). POLY-3 needs ≤ 85.8k. A need 3–4× 6×2's (356k–475k) would be **AMBIGUOUS-3**. Only growth
  that tracks the support (about 6.3×) gives SUPPORT-LIKE-3. RULE.md does not say whether the step-2 ratio is
  taken against S10 or S12'; that must be registered before any 7×2 data.
* Minimal decisive ladder: exact + guided 262k + 1M. If s(1M) > 0.10 (censored above 1.05M), the step is
  SUPPORT-LIKE at either reference. If s(262k) ≤ 0.10, the result is AMBIGUOUS-3. No 4M rung is needed. Cost
  is about 158 CPU-h expected, cap about 206–218. Adding guided 65k and Born 262k/1M for the comparator gives
  195 expected, cap 253.
* Gates: > 10 CPU-h needs the auditor; > 100 needs passed-t2 (769376 is a finished T2 of this route). The leaf
  has 300 − 31.8 = 268.2 CPU-h left. The minimal ladder fits, leaving about 50. The comparator version leaves
  about 15. Either way the 7×2 ladder would consume most of what the variance-reduction batch needs, and the
  SUPPORT-LIKE action ranks that batch first. The total brainstorm budget is not binding (82.2 / 2000).
* Also note: the 7×2 lowest (+,+) level is not computed (`E0_pg` nan), so the L40 representativeness check
  (G_ref / gap) is unavailable there without a 9M-dimensional Lanczos run.

## 3. Exact reference: re-register before 7×2?
The 6×2 exact arm is long enough under rev 2 as registered. Rev 2 defines G_ref as the exact mean over steps
61–120 at a fixed 120-step horizon. It only flags a significant slope, it does not score it, and 769376's
exact arm is DONE at 120. The flag does not threaten the SUPPORT-LIKE verdict: a lower late-time reference
raises every guided s and can only push N_needed up. A longer exact arm is not needed before 7×2, and it would
be the wrong fix. Lengthening the horizon at one size breaks the same-statistic comparison across sizes, so
5×2 and 6×2 would have to be rerun too. At 7×2 every extra exact step costs about 1.1 CPU-h, so a 240-step arm
(≥ 300 CPU-h with the cap) is over the leaf's budget. What should be registered before any 7×2 data (rev-2
addendum, zero compute):
* the step-2 reference (S10 or S12');
* the minimal rung set;
* a reported-only sensitivity: N_needed with E_win(exact) shifted by the arm's own fitted 61–120 slope × 60
  steps;
* a statement that a "still descending" flag at 7×2, which is expected and larger because more modes are slow,
  cannot by itself move a SUPPORT-LIKE verdict, but would make a POLY-side verdict at 7×2 provisional.

## Needed summary.md changes (not made; returned to the orchestrator)
* R51.8 row / next: second seed prepared (`submit_r11_s12p_seed2.sbatch`, 2.9 CPU-h) and **G2-refused** (p 0.88,
  no decision change). Prediction to record: SUPPORT-LIKE holds under seed0 19000.
* 7×2 ladder costed: minimal decisive (exact + 262k + 1M) about 158 CPU-h expected, cap about 210. It needs the
  auditor and passed-t2 and fits the leaf's 268 remaining, but it competes with the variance-reduction batch,
  which ranks first. A 3–4× need would score AMBIGUOUS-3. Step-2 reference to be registered first.

## Orchestrator decisions (2026-11-18)
1. 6×2 second seed (submit_r11_s12p_seed2.sbatch, 2.9 CPU-h cap): GATE-REFUSED (G2) — predicted SUPPORT-LIKE holds under a second seed, p = 0.88 (sens_r11_seed2.py: second seed moves ln N by 0.036–0.044 against a 0.074 margin), changes no decision. Prediction recorded; the seed is folded into any 7×2 package.
2. 7×2 ladder (minimal: exact + guided 262k + 1M, ~158 CPU-h expected, cap ~206–218; 5–6 days wall, 32 GB): HELD. RULE.md ranks the force variance-reduction batch first after SUPPORT-LIKE (round 12 generation in progress), and the ladder would take most of the leaf's remaining 268 CPU-h. Before any 7×2 data: register a zero-compute addendum (step-2 reference — against 5×2 or 6×2; minimal rungs; reported-only slope-shifted reference; a 7×2 "still descending" flag makes only a POLY-side verdict provisional). Revisit after round 12's survivors are tested.
