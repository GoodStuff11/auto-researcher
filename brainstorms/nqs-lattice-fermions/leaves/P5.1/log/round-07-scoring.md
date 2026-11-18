# P5.1 round 7: scoring of R51.8 at 5x2 (S10) under RULE.md revision 2, with second seeds
Scored 2026-11-17 by `tests/P5.1_size_scaling_r04/score_r07.py` -> `results/score_r07.json`. Data: job 763968
(seed0 9000: exact arm, guided and born, 5 rungs each) and job 769098 (seed0 19000: guided 16,384, 65,536, 262,144).
All 14 arms are DONE at 120 steps, so INCOMPLETE does not apply. No rule was amended. The critic's rev-2 addendum
(RULE.md section "Rev 2 addendum") was appended before this scoring, while 769098 was running and before any of its
output was scored, as RULE.md states (wording corrected 2026-11-17, per-leaf critic re-check Minor 6). Its
quantities are reported separately below.

## Rev-2 primary statistic (E_win over steps 61-120, threshold s = 0.10, seed means)
Exact arm: E_start -6.547116, E_win -6.548717, G_ref 1.6011e-3.

| N | guided s (s9000 / s19000 / mean) | born s |
|---|---|---|
| 1024 | 0.941 | 1.086 |
| 4096 | 0.445 | 0.830 |
| 16384 | 0.141 / 0.148 / 0.1445 | 0.455 |
| 65536 | 0.063 / 0.068 / 0.0656 | 0.194 |
| 262144 | 0.051 / 0.053 / 0.0524 | 0.085 |

* **N_needed(S10) = 31,280** (ln s vs ln N between 16,384 and 65,536).
* Conditioning: gap = 0.1445 - 0.0656 = 0.0789. sd_w = max(sd of the 65,536 rung's window means 0.0213; seed
  difference / sqrt 2 = 0.0037) = 0.0213. The gap is 3.69 sd_w > 2, so the result is **RESOLVABLE**.
  The two estimates of the window sd agree (per-seed mean 0.02135; seed-averaged trajectory 0.02132). The window
  sd mostly measures the downward trend: the window means are 0.089, 0.060 and 0.048, not scatter.
* Consequence (rev 2): N_needed(5x2) = 31,280 becomes the denominator of the 6x2 ladder. The auditor's thresholds
  (51,030 and 103,900) are 1.728 and 3.52 times a provisional 29,530, so they must be recomputed. From the scored
  value they are POLY-CONSISTENT if $N_{\rm needed}(6\times2)\le54{,}050$, SUPPORT-LIKE if $\ge110{,}100$, and AMBIGUOUS in between.

## Reported, not scored
* The 5% / last-20 statistic with seed means at the bracketing rungs gives s(65,536) = 0.0558 and s(262,144) =
  0.0409, so N_needed = 107,200. With one seed it was 94,150. The 5% statistic still depends strongly on the window.
* Exact-arm slope over steps 61-120 is -1.22e-6 per step (SE 1.11e-6, z = -1.10), so it is not flagged. Over 60
  steps that slope amounts to 0.046 G_ref, which is comparable to s at the top two rungs.
* Guided vs Born, total draws to a sustained 90% (the trailing 20-step mean of the gain first reaches 0.9 G_ref,
  scored as in RULE_cmp):
  - Born, seed 9000: 262,144 x 90 = 23.59M. No other Born rung reaches 90%.
  - Guided, seed 9000: 16,384 x 120 = 1.97M. It crosses at the last step, so this result is marginal.
  - Guided, seed 19000: 16,384 never crosses, so the minimum is 65,536 x 76 = 4.98M.
  - Ratio Born/guided: 12.0 with seed 9000 and 4.7 with seed 19000. Both are $\ge2$.
  - Against the last-20 gain (rev-1 G_ref) the ratios are 5.0 and 4.85.

## Rev-2 addendum quantities (reported separately from the rev-2 verdict)
* sd_w from detrended window means (linear fit over steps 61-120 removed) at the 65,536 rung: 0.0051. Then
  sd_w = max(0.0051, 0.0037) = 0.0051, and the gap is 15.4 sd_w, so the result is RESOLVABLE under the addendum too.
  The addendum's
  UNRESOLVABLE actions (i) and (ii) are therefore not triggered at 5x2.
* Item (i) quantities, given for the record (Born, seed 9000: 23.59M). Total guided draws to a sustained 90% of 5x2's own exact gain: 1.97M with seed 9000 (marginal, step 120) and
  4.98M with seed 19000. The minimising rung differs between the two seeds, so this statistic is fragile
  at 16,384.
