# P5.1 round 18 (worker): records after the R51.32 critic; R51.33 two-sided regime-matched firm-up designed — 2026-11-18

Compute: about 43 CPU-min interactive. That is 37 CPU-min of 5×2 pilots (8 arms) and 6 CPU-min of 6×2 cost probes (in the scratch directory, energies unused), plus seconds of scorer checks. No sbatch, no commits. Not opened: `tests/P5.1_size_scaling_r04/results/r14_decay/` (job 773528 running).

## Part 1: records (critic Major 3, Minor 5)
* **Card R51.32**: the Outcome now reads POLY-CONSISTENT (job 773759, 1.5 CPU-h; τ* 0.0025; $N_{\rm needed}$ 98,387; $r'_m$ 1.21, not provisional).
  - The critic's band sensitivity is recorded: ~1.14 matched on realised $m$, ~1.8 at the band top (interpolated), support-like only at $g_1\approx0.27$.
  - Also recorded: the missed prediction (p 0.40 SUPPORT; POLY was the 0.12 class), the mechanism (a shorter step removed the 10-site bounce and deepened the reference 7%), and the sign of the remaining one-sided bias (a deeper 12-site reference raises $N(6\times2)$), along with the common-reference check.
* **Card R51.8**: SUPPORT-LIKE demoted to "at one fixed step length (τ 0.005)", with a pointer to R51.32 and L78.
* **Card R51.11**: the ladder 773500 is recorded.
  - Same-size gains are 3.78 at 10 sites and 4.20 at 12 sites.
  - $r'_q$ = 3.42 is provisional AMBIGUOUS (0.03 ln-units from 3.52), at one fixed step, and read per Addendum 2.
  - The 12-site 16k second seed of the near-line clause is not owed, because under Addendum 2 the size class decides nothing and the 12-site gain (4.2) is far from its lines.
* **Summary**:
  - `Conclusion`: R51.8 is now "SUPPORT-LIKE at one fixed step length only", with R51.32 and the 773500 ladder added.
  - Portfolio: the R51.32 row is updated and an R51.33 row added.
  - `Updated`, `Compute spent` and `Next step` are updated.
  - Answer and What we showed are untouched, as instructed; one number for the orchestrator is noted below.
* **Library L78**: index line and entry now record that R51.32 tested the lesson. The fixed-τ ratio 3.79 fell to 1.21 when matched: mostly step regime and reference depth. The 10-site need rose 3.15× because the long-step reference was 7% shallow, and the matching was one-sided.

## Part 2: R51.33 `match2` (card `cards/R51.33.md`, `tests/P5.1_r18_match2/`)
**Pilots changed the deciding pair.** The brief proposed both sizes at τ 0.0025 as the deciding pair. Before writing the rule I ran the 5×2 rungs at τ 0.00375 and 0.00125 interactively (candidate stage (c), plus its 0.00125 twin; 4k–262k, seed 9000, 3–7 CPU-min each).

| 5×2 τ | $m$ | $g_1$ | $N_{\rm needed}$ | seeds |
| :--- | :--- | :--- | :--- | :--- |
| 0.005 | 0.153 | 0.301 | 31,284 | 2 (R51.8) |
| 0.00375 | 0.115 | 0.163 | 39,080 | 1 (pilot) |
| 0.0025 | 0.051 | 0.070 | 98,387 | 2 (R51.32) |
| 0.00125 | 0.0121 | 0.0179 | 219,118 | 1 (pilot) |

* The 10-site need rises monotonically as the step shortens. Halving τ from 0.0025 to 0.00125 raised it ×2.23, cut $m$ ×0.236 and changed $G_{\rm ref}$ by −1.2%.
* 6×2 at τ 0.005 is already in 5×2's τ 0.0025 regime: $m$ 0.0485 against 0.051, and the period-2 envelope lies 2.5% below the window mean against 2.6% (7.5% for 5×2 at τ 0.005).
* So 6×2 at τ 0.0025 should sit near 5×2's τ 0.00125 regime. If matched growth stays 1.2, the same-τ ratio is about 1.2 × 2.23 about 2.7, which is AMBIGUOUS from regime mismatch alone. That is the L78 confound again, and largely predictable.
* I therefore made the deciding pair $m$-matched: 6×2 at τ 0.0025 against $N_{5\times2}$ interpolated (ln–ln) at the 6×2 $m$ on the measured τ curve. The same-τ ratio is reported with its class, and the $g_1$-matched ratio is a regime clause (REGIME-SENSITIVE → no class).
* This deviates from the brief's "verdict lines on both at τ 0.0025". The reason is that the same-τ verdict is predicted AMBIGUOUS by construction.
* The predicted 6×2 $m$ (about 0.0115) lies just below the lowest pilot point (0.0121), so job 1 adds a 5×2 τ 0.000625 grid point.

**Reported-only numbers from the pilots (one seed).** These use R51.32's one-sided pair, with the stored 6×2 τ 0.005 need as numerator.
* $m$-matched 1.17; $g_1$-matched 1.19.
* At the 12-site band edges: 2.08 at the top ($g_1$ 0.116, AMBIGUOUS; the critic's interpolation gave 1.79) and 0.61 at the bottom.
* For the orchestrator: the Answer's "up to about 1.8" is, on data, "up to about 2.1".

**Stage judgement (brief's candidates).**
* (a) 6×2 exact τ 0.0025: needed, and it is the bulk of the cost (348 CPU-min in-job).
* (b) 6×2 rungs: two, placed from $N_5^*$ after (a) at $4096\lfloor1.728e^{-0.14}N_5^*/4096\rfloor$ and $4096\lceil3.52e^{0.13}N_5^*/4096\rceil$ (cap 2,097,152). Either censored outcome then decides a class 0.13–0.14 ln-units outside a line.
* (c) 5×2 τ 0.00375: done as a pilot (39,080), so no job.
* (d) 6×2 second seed: only on the L72 clauses (PROVISIONAL, or a defining rung's $s$ within 0.3 ln of 0.10), as job 3. At the predicted margin (0.36 ln below 1.728, about 5 seed sd) one seed suffices.
* Added: a 5×2 τ 0.000625 point, and 5×2 second seeds at the two grid points bracketing the 6×2 $m$, so the matched denominator has two seeds as in R51.32.

**Jobs.** Total expected 11.0 CPU-h for jobs 1–2, split to keep each under 10, so no auditor is needed.
* Job 1 (`submit_r18_match2.sbatch`), cap 9.5 CPU-h, expected 7.6:
  - A0: 5×2 τ 0.000625 exact arm and ladder; the pilots are skipped as DONE (same code and seeds).
  - A: 6×2 τ 0.0025 exact arm through `xsx_run.py`. On resume, the xsx log is trimmed to state.npz's step count instead of restarting a 5.8 h arm.
  - Release check.
  - A2: bracket and 5×2 second seeds.
  - A3: place the 6×2 rungs in `results/rungs6.json`.
* Job 2 (`submit_r18_rungs6.sbatch`), cap 6.0, expected 3.4: the two placed 6×2 rungs.
* Gate:
  - job 1: `compute_gate.py check … submit_r18_match2.sbatch --voi voi.md --pre-audit` → cap 9.5, GATE: PASS (digest e23cc90c5418);
  - job 2: `… submit_r18_rungs6.sbatch --voi voi_rungs6.md --pre-audit` → cap 6.0, GATE: PASS. Its p is to be refreshed from job 1's $N_5^*$ and 6×2 reference depth.
* Wall-time basis:
  - 6×2 exact probe at τ 0.0025: 1.67 CPU-min per step interactive, the same work as the τ 0.005 probe (1.69); in-job 2.90 per step (R51.8).
  - 6×2 guided interactive: 0.315, 0.469 and 0.665 CPU-min per step at 147k, 393k and 1M, × 1.54 in-job (262k: 0.653/0.425).
  - 5×2 rates from R51.32's job.

**Scorer** (`score_r18_match2.py`). It reuses `score_r16_qladder.py`'s need, classify and censoring code and R51.32's loaders.
* `identity` reproduces R51.32 through the same code path: 98,387.30 (29.4 sd_w), 118,626.20 (26.5 sd_w), ratio 1.2057, $m$(6×2, τ 0.005) 0.0485 → PASS.
* `test` exercises synthetic arms through every branch → SELFTEST PASS: censored POLY and SUPPORT with and without the rung clause, interpolated AMBIGUOUS, conditioning UNRESOLVABLE, provisional near 3.52, REGIME-SENSITIVE, second seed present, NOT-MATCHED, missing rung.
* Outputs are in `tests/P5.1_r18_match2/logs/scorer_checks_round18.txt`; flake8 is clean at 120 columns.

**Prediction and p (voi.md).** The 5×2 analog transferred to 12 sites gives $N(6\times2,0.0025)\approx$ 264k at $m\approx0.0115$, with $N_5^*\approx$ 225k, so $r'_{mm}\approx1.2$.
* POLY needs a 12-site τ effect below ×3.3; SUPPORT needs ×6.7 or more.
* The critic's mechanism, a deeper 12-site reference, raised the 10-site need ×3.15 only where it removed a 7.5% bounce; at 6×2 the bounce part is 2.5%.
* Prior: POLY 0.55, AMBIGUOUS 0.25, SUPPORT 0.05, no class 0.15.

**What each verdict does to the Answer.** See RULE.md.
* POLY: drops "the 12-site step was not shortened to match" and states 1.2 at two matched regimes; the third size is next.
* AMBIGUOUS: growth reported as regime-dependent (1.2 and X).
* SUPPORT: 1.21 demoted to one-sided matching; the Answer reports X with "at least as fast as the occupied configurations".
* Other: numbers only.

## The critic's 7×2 plan (Minor 6), costed roughly; the later step, after R51.33 decides
1. **Pre-flight of the 7×2 exact step.** 5 steps. The rate is still inferred from κ passes: 68.6 CPU-min per step in-job, so about 6 CPU-h. It cannot run interactively (about 39 CPU-min per step even interactively).
2. **One 7×2 exact arm at τ 0.005 with the xsx log.** 120 × 68.6 about 137 CPU-h, about 5.7 days on one CPU.
3. **Matching the cheap sizes down to the 7×2 $m$ and $g_1$.** These fall with size at fixed τ (5×2 0.153 → 6×2 0.0485). R51.33's 6×2 τ 0.0025 arms and the 5×2 τ grid to 0.000625 may already bracket the 7×2 value. Otherwise one more 6×2 τ costs about 5.8 CPU-h for the exact arm plus about 3 for rungs, and a 5×2 τ about 0.5.
4. **A single 7×2 guided rung at the N dividing POLY from SUPPORT** (about 2.4 × the matched $N(6\times2)$, about 0.6M if R51.33 gives about 264k): about 13–15 CPU-h. A second seed only under L72: about 15 more.

Total: about 160 CPU-h, or about 190 with the seed and an extra 6×2 τ. That needs the auditor (> 10) and passed-t2 (R51.8) (> 100). The leaf has about 268 CPU-h, less 773528's 6 and R51.33's ≤ 15.5. The exact arm is about 85% of the cost, so a cheaper 7×2 reference is the lever, as R51.8's card already says. Three sizes still cannot separate a polynomial from an exponential; the 7×2 step can only rule out support-like growth a second time (the "no" branch).

## Proposed for the orchestrator
* The Answer's "up to about 1.8" could read "up to about 2.1" (pilot data, one 5×2 seed). This is not edited here, per the brief.
* Library: a candidate lesson, not added. "At a fixed size the sampled-SR need rises monotonically as the step shortens (5×2: 31k → 39k → 98k → 219k over τ 0.005 → 0.00125), so a same-τ size ratio compares different regimes unless both sizes sit at equal realised $m$; read size ratios on equal $m$." This extends L78. It is one size and one seed per new point.
