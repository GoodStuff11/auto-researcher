# RULE: R51.33 `match2`, the 10 → 12 size ratio with both sizes at a shortened step (round 18, written before any 6×2 arm)
Written 2026-11-18 by the round-18 worker. Pre-audit draft; nothing submitted. Not opened: `../P5.1_size_scaling_r04/results/r14_decay/` (job 773528 running).

**Question.** R51.32 (job 773759) shortened only the 10-site step. 5×2 at τ 0.0025 needs $N_{\rm needed}$ = 98,387, against the stored 12-site need of 118,626 at τ 0.005, so $r'_m$ = 1.21, POLY-CONSISTENT. The round-17 per-leaf critic found three gaps:
* The matching was one-sided.
* The 12-site reference at τ 0.005 is still descending (z −3.7). The mechanism that raised the 10-site need (a shorter step deepens the noise-free reference) can act at 12 sites too and raise $N(6\times2)$.
* The 12-site need rests on one seed.

Does the 10 → 12 ratio stay below 1.728 when the 12-site step is also shortened (τ 0.0025, its noise-free reference rerun at that τ) and paired with the 10-site step that puts the 10-site arm in the same regime?

**Matched quantity (T0, revised after the 5×2 pilots below).** Four options:
* (a) Same τ = 0.0025 at both sizes. The orchestrator's draft made this the deciding pair. The pilots show it is not regime-matched and its outcome is largely predictable. At 5×2, halving τ from 0.0025 to 0.00125 raised the need ×2.23, and the realised $m$ fell ×0.236. 6×2 at τ 0.005 already sits in the regime of 5×2 at 0.0025 ($m$ 0.0485 against 0.051; envelope gap 2.5% against 2.6%). So 6×2 at 0.0025 is expected in the regime of 5×2 at about 0.00125. If the matched growth is the R51.32 value, the same-τ ratio is then about 1.2 × 2.23 ≈ 2.7, AMBIGUOUS by construction: it re-measures the step-regime offset (L78), not size. Reported with its class, not deciding.
* (b) Equal realised $m$ (chosen, deciding). $m$ is the mean $|\Delta E|/G_{\rm ref}$ over steps 61–120 of each size's exact arm, the window statistic the critic recommended (round-17 Major 1). 6×2 runs at τ 0.0025. The 5×2 need at the 6×2 $m$ comes from ln $N$ against ln $m$, interpolated between the two 5×2 grid points that bracket it. The grid is τ ∈ {0.005, 0.00375, 0.0025, 0.00125, 0.000625}, every point with its own exact arm. Both sizes are then off τ 0.005, both references are rerun at the shorter steps, and the regimes are equal on the measured window statistic. This is two-sided matching at a second regime level ($m\approx0.012$, against R51.32's 0.05).
* (c) Equal $g_1$ (window first-order gain from the xsx log). Same construction. Reported, and used in the regime clause below.
* (d) 6×2 at a longer step matched to 5×2: rejected, because 6×2 already bounces at τ 0.005.

**Zero-cost checks and pilots done first (disclosed).**
* *Stored-data envelope check.* Take the lower envelope of the period-2 pairs over steps 61–120 of the τ 0.005 exact arm. At 5×2 it is 7.5% deeper than the window mean, and the τ 0.0025 arm's window mean landed on it (0.0017216 against 0.0017202). At 6×2 it is 2.5% deeper (G 0.007183 against 0.007011), the same as 5×2 at τ 0.0025 (2.6%). There, halving τ to 0.00125 changed G by −1.2%.
* *Common reference at 5×2* (critic Major 2). Against the τ 0.0025 reference, the τ 0.005 rungs give $s$ 0.204, 0.130 and 0.118 at 16k, 65k and 262k.
* *5×2 pilots (interactive, before this file was finished).* Guided rungs 4k–262k at τ 0.00375 and 0.00125, seed 9000, 120 steps, scored against R51.32's exact arms at the same τ (`logs/pilot_*.txt`, `results/S5x2_*`, `logs/scorer_checks_round18.txt`). These are the same code and seeds as the job, so job 1 skips them as DONE.
  - τ 0.00375: $N_{\rm needed}$ 39,080 ($m$ 0.115, $g_1$ 0.163).
  - τ 0.00125: $N_{\rm needed}$ 219,118 ($m$ 0.0121, $g_1$ 0.0179).
  - With the stored points (0.005: 31,284; 0.0025: 98,387), the 5×2 need rises monotonically as the step shortens.
  - For R51.32's one-sided pair they give 1.17 matched on $m$ and 1.19 on $g_1$. At the edges of the 12-site $g_1$ band they give 2.08 (top, AMBIGUOUS; the critic's interpolation gave 1.8) and 0.61 (bottom). One seed; reported only.
* *6×2 cost probes* (2–3 steps each, scratch directory, energies unused): exact τ 0.0025 through the xsx wrapper, 1.67 CPU-min per step; guided 147,456, 0.315; 393,216, 0.469; 1,048,576, 0.665.

**Arms.** R51.8's starts and `ns_run.py`, unchanged; exact i.i.d. guided draws; 120 steps.
* Job 1 (`submit_r18_match2.sbatch`):
  - A0: 5×2 exact arm at τ 0.000625 (xsx log) and guided rungs 4k–262k at seed 9000, plus 1,048,576 if $s$(262k) > 0.10. This extends the curve below $m$ 0.012, since the predicted 6×2 $m$ is about 0.0115.
  - A: 6×2 exact arm at τ 0.0025 (xsx log). Release check: DONE, 120 steps, finite energies, $G_{\rm ref}>0$; otherwise the job stops (INCOMPLETE).
  - A2: if the 6×2 $m$ lies outside the 5×2 range, NOT-MATCHED and stop. Otherwise run seed 19000 at the defining rungs of the two bracketing 5×2 points, so the matched denominator has two seeds, as R51.32's did.
  - A3: place the 6×2 rungs (below) in `results/rungs6.json`.
* Job 2 (`submit_r18_rungs6.sbatch`, released after job 1): 6×2 guided rungs $N_P$ and $N_S$ at τ 0.0025, seed 9000. Placement uses $N_5^*$, the 5×2 need interpolated at the 6×2 $m$ (geometric middle if it is an interval):
  - $N_P=4096\lfloor 1.728\,e^{-0.14}N_5^*/4096\rfloor$ and $N_S=4096\lceil 3.52\,e^{0.13}N_5^*/4096\rceil$, with $N_S$ capped at 2,097,152.
  - Each censored outcome then decides a class outside the near-line band. $s(N_P)\le0.10$ gives $r'_{\rm hi}\approx1.50$ (POLY); $s(N_S)>0.10$ gives $r'_{\rm lo}\approx4.0$ (SUPPORT). Between the two, the need is interpolated.
* Job 3 (gated after scoring): seed 19000 at the 6×2 rungs `score_r18_match2.py owed` lists. Nothing else.
* Not bought: 5×2 τ 0.00375 rungs as a job, since the pilot replaced the interpolation; 6×2 rungs below $N_P$; a 6×2 window at τ 0.005.

**Statistic: R51.8 rev. 2, as in R51.32, unchanged except τ and rungs.**
* Window and gap: $E_{\rm win}$ is the mean exact energy over steps 61–120 of each size's exact arm at its τ, and $G_{\rm ref}=E_{\rm start}-E_{\rm win}$, with $E_{\rm start}(6\times2)$ = −7.605891062 and $E_{\rm start}(5\times2)$ = −6.547115994.
* Shortfall: $s=(E_{\rm win}({\rm arm})-E_{\rm win}({\rm exact}))/G_{\rm ref}$, the seed mean where two seeds exist.
* Need: $N_{\rm needed}$ at $s$ = 0.10, by ln $s$–ln $N$ interpolation.
* Censoring: as `score_r16_qladder.py`.
* Conditioning on a bracket: the gap must exceed $2\,sd_w$, where $sd_w$ is the detrended window sd of the upper rung, or |seed diff|/√2 if larger; otherwise UNRESOLVABLE.
* Matching: $m$ as above, for each exact arm. $N_{5\times2}(m)$ is piecewise linear in ln–ln between the bracketing grid points; if either point is censored it becomes an interval.
* Code: `score_r18_match2.py`. The `identity` mode reproduces R51.32 through the same code path (98,387.30; 118,626.20; 1.2057; $m$(6×2, τ 0.005) 0.0485). The `test` mode exercises every branch on synthetic arms (PASS, `logs/scorer_checks_round18.txt`).

**Verdict.** The deciding ratio is $r'_{mm}=N_{\rm needed}(6\times2,0.0025)/N_{5\times2}(m_{6\times2})$, as an interval.
* Classes: POLY-CONSISTENT if $r'_{mm,\rm hi}\le1.728$; SUPPORT-LIKE if $r'_{mm,\rm lo}\ge3.52$; AMBIGUOUS if strictly between. A line inside the interval gives UNRESOLVABLE (CENSORED-UNDECIDED).
* Precedence: INCOMPLETE > NOT-MATCHED > UNRESOLVABLE > class.
* Near-line clause (L72): PROVISIONAL if $|\ln r'_{mm}-\ln{\rm line}|\le0.1$ for either line. The class is reported, and the Answer waits for job 3.
* Second-seed clause (L72, and R51.8's rung clause): seed 19000 is owed at every single-seed defining 6×2 rung if the verdict is PROVISIONAL, or if that rung's $s$ lies within 0.3 ln-units of 0.10 (job 3). Otherwise the verdict is final and labelled "one 12-site seed".
* Regime clause: if the class is not among those the $g_1$-matched interval allows, the verdict carries REGIME-SENSITIVE. The Answer then reports both numbers and no class.
* Reported only:
  - the same-τ ratio $N_{\rm needed}(6\times2,0.0025)/98{,}387$ and its class (predicted ≈ 2.7 if matched growth is 1.2);
  - $G_{\rm ref}(6\times2,0.0025)/0.0070107$, the depth change whose sign the critic predicted to be > 1;
  - $N_{\rm needed}(6\times2,0.0025)/118{,}626$, the 12-site τ effect, against ×2.23 at 10 sites for the matching halving;
  - the stored τ 0.005 6×2 rungs' $s$ against the τ 0.0025 reference;
  - the slope flags.

**What each verdict does to the Answer.** The Answer now says the 10 → 12 need grew about 1.2 times with the 10-site step matched to the 12-site regime, that the 12-site step was not shortened to match, and that the growth was not as fast as the occupied configurations for any matching the data allow (up to about 1.8).
* POLY-CONSISTENT (not provisional, not regime-sensitive): the growth sentence becomes "about 1.2 times at two matched step regimes, the second with both sizes' steps shortened and both noise-free references rerun", and "the 12-site step was not shortened to match" is dropped. The 10 → 12 step is then poly-consistent from both sides, and the next spend is the third size (the critic's 7×2 plan, round-18 log).
* AMBIGUOUS: the Answer says the growth was about 1.2 times at the 10-site step's regime and X times at the shorter regime, so the polynomial reading depends on the regime. "Less than the cube of the size" is kept only for the first regime.
* SUPPORT-LIKE: the 1.21 was a one-sided-matching artefact. The Answer says the need grew X times with both steps shortened and regime-matched, at least as fast as the occupied configurations, and about 1.2 only with the 10-site step alone shortened. R51.32's POLY is demoted to "one-sided matching". This is the first regime-matched support-like size step toward the "no" branch, which needs a second.
* NOT-MATCHED, REGIME-SENSITIVE, PROVISIONAL, UNRESOLVABLE or INCOMPLETE: numbers only. "The 12-site step was not shortened to match" becomes "shortening it gives X (class not settled)".
* In every case the band edge "up to about 1.8" becomes the pilot-based 2.1, which the orchestrator may adopt now (one 5×2 seed).

**Scope and confounds named now.**
* One size step, a factor of 1.2 in size: no verdict separates a polynomial from an exponential.
* One 12-site seed unless job 3 runs.
* At short τ both exact arms may still descend over steps 61–120 (5×2 at 0.00125: z −30). Each size is scored against its own noise-free arm at the same step count, as in R51.8, and the slope flags are reported.
* The $m$-matched denominator is an interpolation between two τ grid points.
* The 5×2 pilots were run before this file was written. They are single-seed until job 1's stage A2, and enter only the denominator and the rung placement.
