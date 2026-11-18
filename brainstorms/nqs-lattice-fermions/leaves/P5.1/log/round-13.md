# P5.1 round 13 — R51.8 S14 (7×2) ladder prepared for the compute auditor (worker, 2026-11-18)

Nothing was submitted and no Slurm job was started. `summary.md` and the cards were not edited. P4.1 was not touched.
Precondition: RULE.md ranks the force variance-reduction batch first after SUPPORT-LIKE. Round 12 spent that batch
with no survivor, so the S14 ladder is the next spend.

## 1. RULE.md addendum (written first, zero compute, before any 7×2 ladder data)
This is the section "Addendum (2026-11-18, before any 7×2 data)" in `tests/P5.1_size_scaling_r04/RULE.md`. It
registers the following.
* **Step-2 reference: S10.** The step-2 ratio is r'' = N(S14)/N(S10), compared with g3'' = 2.744 and gH'' = 22.24.
  SUPPORT-LIKE needs N(S14) ≥ 695,900, and the POLY line is N(S14) ≤ 85,840. Three reasons:
  * The verdict table's only stated three-size ratio is N(S14)/N(S10) ≤ 2.744.
  * The S10 denominator has two seeds at both rungs, while S12' has one seed and a 1.08× margin. Dividing by S12'
    would anti-correlate the noise of the two steps.
  * The two possible thresholds differ by only 7.7% (695.9k vs 749.3k). The S12' ratio is reported only, and a
    result between the two thresholds is labelled "vs S10 only".
* **Rungs:** exact, guided 262k and guided 1M, all seed 9000. A 262k rung at s ≤ 0.10 is censored below and makes
  the result AMBIGUOUS-3. A 1M rung at s > 0.10 is censored above and makes it SUPPORT-LIKE. The rule needs a
  rev-2 gap of more than 2 sd_w, or the step is UNRESOLVABLE. S14 has no comparator.
* **6×2 second seed (65k rung, seed0 19000):** it runs first. s(65k) becomes the two-seed mean, and the seed term
  enters sd_w. If r' is no longer SUPPORT-LIKE, the job writes `results/STOP_r13` and runs no S14 arm.
* **Reported-only check:** N_needed is recomputed with the exact reference shifted by its own 61–120 slope × 60.
* **Still-descending flag at S14:** it makes only a POLY-side result provisional (r'' < 22.24, scored
  AMBIGUOUS-3). A SUPPORT-side result stands.
* **What each outcome does to the Answer:**
  * SUPPORT-LIKE-3: the Answer becomes "No, for SR-type optimisers with these samplers at this scale". It is
    marked ANSWERED only once R51.7 and R51.11 are also resolved.
  * AMBIGUOUS-3: numbers only.
  * POLY-3: unreachable, because r' is not POLY-CONSISTENT. With the seed it could get there only through a
    shift of about 24 sd. A ratio r'' ≤ 2.744 is scored AMBIGUOUS-3 with no polynomial claim.
  * UNRESOLVABLE and INCOMPLETE are handled as in rev 2.

## 2. Job (prepared, not submitted)
New files in `tests/P5.1_size_scaling_r04/`: `submit_r13_s14_ladder.sbatch`, `voi_r13_s14_ladder.md`,
`score_r13_step1.py` (step-1 rescorer; `--dry-run` reproduces score_r08: N 118,626, r' 3.79) and `mock_r13_worker.sh`.
* **Partition:** `kim` has MaxTime UNLIMITED (sinfo `%l` = infinite). The gate charges array × time × cpus, so a
  single array whose exact task runs for 7 days would be charged 4 × 178 h. Instead the job runs 5 identical
  48-h workers (`--array=0-4%1`, one at a time).
* **Each worker:** it walks the four items in order: S12' 65k seed 19000, then the scorer and STOP rule, then S14
  exact, then guided 1M, then guided 262k. It skips DONE items. It runs the first unfinished item with
  `--cpu-cap` set to its remaining wall time minus 20 min. ns_run stops before a step that would cross the cap
  and saves state.npz, and the next worker resumes from there.
* **Signals:** `B:USR1@1200` and SIGTERM are forwarded to ns_run. `--requeue` is set because the partition's
  PreemptMode is REQUEUE.
* **Memory:** 32G.
* **Mock test** (fake ns_run, no physics): passed chunked resume across 3 workers, with later workers exiting at
  once. It also passed the STOP path, and USR1 forwarding (TRUNCATED, exit 0, no later item started).
* **Cost:** in-job rates × 1.3. Exact 178.4 h, guided 1M 23.4 h, guided 262k 16.7 h, seed 1.2 h, setups 7.6 h and
  tail waste about 5 h, for a total of 232 h, which is at most the 240 h cap. Expected use is 156–175 CPU-h and
  about 6.5–7.3 days of wall time.
* **Prediction:** AMBIGUOUS-3, p 0.49. The other outcomes are SUPPORT-LIKE-3 0.34, step-1 STOP 0.07, UNRESOLVABLE
  0.05 and INCOMPLETE 0.05. Under support-tracking, r'' is about 24–26 against a line of 22.24. Under the fixed
  V^7.3 power seen at step 1, r'' is about 11.7. Two sizes cannot weight these two readings.
* **Gate:** `compute_gate.py check … --pre-audit` → cap 240.0 CPU-h. **GATE: PASS PENDING AUDIT**, digest
  **7c2fdfcdf155**. It needs the auditor (cap > 10) and has passed-t2 769376 (cap > 100). The leaf has 31.8 of
  300 committed, so 28.2 would remain uncommitted after the cap.

## Needed summary.md changes (not made; returned to the orchestrator)
* R51.8 next step: the S14 ladder is prepared (`submit_r13_s14_ladder.sbatch`, cap 240, expected 156–175 CPU-h,
  about 7 days) and is pending audit. The step-2 reference (S10, 695,900) and the outcome-to-Answer map are
  registered in the RULE.md addendum of 2026-11-18. POLY-3 is unreachable. The 6×2 second seed is folded in and
  can stop the job.
