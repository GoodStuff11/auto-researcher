# VOI note: P5.1 R51.33 `match2`, job 1 of 2 (`submit_r18_match2.sbatch`; `RULE.md`)
* leaf: P5.1
* route: R51.33
* tier: T2
* decision: The package (job 1 + job 2; job 3 only on the L72 clause) decides $r'_{mm}$. This is the 10 → 12 need ratio with 6×2 at τ 0.0025, scored against its own noise-free arm at that τ, and 5×2 at the step whose realised window regime $m$ equals 6×2's.
  - POLY-CONSISTENT: the Answer's "about 1.2" holds at two matched regimes, and "the 12-site step was not shortened to match" is dropped. Next spend: the regime-matched third size (7×2 plan, round-18 log).
  - AMBIGUOUS: the Answer reports 1.2 and X as regime-dependent and drops "less than the cube" as a general reading.
  - SUPPORT-LIKE: the 1.21 is demoted to one-sided matching; the Answer reports X with "at least as fast as the occupied configurations"; first regime-matched support-like step toward the "no" branch.
  - NOT-MATCHED, REGIME-SENSITIVE, PROVISIONAL, UNRESOLVABLE or INCOMPLETE: numbers only.

  Job 1 alone decides two things:
  - whether job 2 runs. NOT-MATCHED or a failed reference ends the package, and the Answer keeps "the 12-site step was not shortened to match".
  - where job 2's two rungs go. Its outputs are also reported: the 12-site reference-depth change the critic predicted (>1), and the two-seed 5×2 matching curve.
* prediction: POLY-CONSISTENT for the package.
  - Basis: the measured 5×2 analog of the 12-site halving. 6×2 at τ 0.005 is in the regime of 5×2 at τ 0.0025 ($m$ 0.0485 vs 0.051; bounce-envelope gap 2.5% vs 2.6%). At 5×2, halving τ 0.0025 → 0.00125 (round-18 pilot) raised the need ×2.23, cut $m$ ×0.236 and changed $G_{\rm ref}$ by −1.2%.
  - If 6×2 responds the same: $N(6\times2,0.0025)\approx$ 264k at $m\approx0.0115$. The 5×2 need there is about 225k, so $r'_{mm}\approx1.2$.
  - POLY holds as long as the 12-site τ effect is below about ×3.3 (vs ×2.23 at 10 sites). SUPPORT needs ×6.7 or more.
  - The critic's deeper-reference mechanism moved the 10-site need ×3.15 only where it removed a 7.5% bounce; at 6×2 the bounce part is 2.5%.

  Prior: POLY-CONSISTENT 0.55, AMBIGUOUS 0.25, SUPPORT-LIKE 0.05, PROVISIONAL / REGIME-SENSITIVE / NOT-MATCHED / INCOMPLETE 0.15.
* p: 0.55
* prediction-changes-decision: yes. The predicted POLY rewrites the Answer's growth sentence: it drops the one-sided-matching caveat the critic raised, and it releases the third-size spend. Today the 1.2 is caveated as one-sided with one 12-site seed.
* cheapest-kill-tried: T0 demoted the drafted same-τ pair; no argument fixes the 12-site τ effect, so it is measured. Detail:
  - T0 (RULE.md): the same-τ pair the orchestrator drafted is predictable from 5×2 data alone. It would re-measure the regime offset (≈ 2.7, AMBIGUOUS by construction), so it was demoted to reported, and the deciding pair was moved to $m$-matching.
  - The stored-envelope argument bounds the 12-site bounce bias at about a third of the 10-site one, but no argument fixes the 12-site τ effect: 5×2 gives ×2.23 for the analogous halving, and nothing transfers it across size. That transfer is what is measured.
  - The 5×2 grid was bought interactively (pilots, 37 CPU-min) instead of in the job.
  - No stored 6×2 arm is at a τ other than 0.005.
* cost-cpuh: 9.5 (1 task × 9.5 h × 1 CPU; expected 7.6). Package:
  - job 2 cap 6.0 (expected 3.4), gated after job 1 with its own note `voi_rungs6.md`;
  - job 3 only on the L72 clause (≤ 3.5);
  - expected 11.0 CPU-h for jobs 1–2, about 4% of the leaf's remaining 268 CPU-h. Each job is under 10, so no auditor.
* cheapest-version: one 6×2 exact arm, two placed 6×2 rungs, 5×2 second seeds at two grid points only. Detail:
  - The 6×2 exact arm (348 CPU-min) is the reference without which no 6×2 rung is scored. It cannot be shortened: G_ref is its 61–120 mean.
  - Only two 6×2 rungs, placed by formula so that either censored outcome decides a class.
  - 5×2 second seeds only at the two grid points that bracket the 6×2 $m$.
  - No Born arms; no τ 0.00375 job (pilot); no second 6×2 seed unless L72 triggers.
  - Job 1 is split from job 2 because the rung placement needs job 1's $m$ and keeps each job under 10 CPU-h.
* convergence: fixed 120 steps, window 61–120, as R51.8/R51.32.
  - The 6×2 exact arm at τ 0.005 reached 90% of its window gain by step 32. At half the step the analogous 5×2 arm (0.00125 vs 0.0025) reached a window G within 1.2%, but still descending (z −30).
  - The slope flag is reported; each size is scored against its own same-τ arm at the same step count.
* wall-time-basis: in-job rates.
  - 6×2 exact: 2.90 CPU-min/step (R51.8 job; round-18 probe at τ 0.0025, 1.67 interactive = τ 0.005 probe 1.69) → 348.
  - 5×2 (R51.32 job): exact 14.2 per 120 steps; 4k 4.2, 16k 5.8, 65k 8.2, 262k 10.7; 1M 13.4. Stage A0 ≤ 57, stage A2 ≤ 48.
  - Setups 4. Total ≤ 457 min = 7.6 h; × 1.25 → 9.5 h.
* power: one 12-site seed suffices at the predicted margin (about 5 seed sd); the L72 clause covers a near-line result. Detail:
  - Seed spread at 5×2: $s$ differences 0.002–0.005 between seeds at the bracketing rungs (R51.32: 0.0021 and 0.0017), ln $N$ sd about 0.03–0.07. At 6×2 one seed; R51.8's $sd_w$ was 0.0022–0.0024.
  - The rung placement keeps censored outcomes 0.13–0.14 ln-units outside the lines.
  - The predicted $r'_{mm}$ ≈ 1.2 is 0.36 ln-units below 1.728, about 5 seed sd, so one 12-site seed suffices unless the result lands near a line (L72 clause → job 3).
  - P(UNRESOLVABLE or INCOMPLETE) about 0.05 (conditioning gaps at 5×2 and 6×2 have been 25–29 sd_w).
