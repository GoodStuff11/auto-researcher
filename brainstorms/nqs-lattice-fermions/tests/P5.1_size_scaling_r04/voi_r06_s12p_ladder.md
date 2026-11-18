# VOI note: P5.1 R51.8 `nscale` unit 1c, the S12' (6x2) ladder (`submit_r06_s12p_ladder.sbatch`; RULE.md rev. 2 + addendum)

* leaf: P5.1
* route: R51.8
* tier: T2
* decision: the only primary statistic is RULE.md rev. 2: s is the mean over steps 61-120 against the exact arm's mean over the same steps, and N_needed(V) is where guided s = 0.10, interpolated in ln s vs ln N. The S10 denominator was scored in round 7: N_needed(5x2) = 31,280, RESOLVABLE (results/score_r07.json). So r' = N_needed(6x2)/31,280 gives four branches:
  * POLY-CONSISTENT if N_needed(6x2) <= 54,050. Next: cost and prepare the S14 (7x2) ladder under rev 2, then the 24-site protocol.
  * SUPPORT-LIKE if N_needed(6x2) >= 110,100. Next: a batch of force variance-reduction routes first, and R51.10 is KILLED as a predictor.
  * AMBIGUOUS in between, or if censoring leaves r' undecided. Next: the S14 ladder.
  * UNRESOLVABLE if s(lower rung) - s(upper rung) <= 2 sd_w, with sd_w from detrended window means (critic addendum). Next, in the addendum's order: (i) the zero-compute rescore, total draws to a sustained 90% of each size's own exact gain; (ii) R51.7's step-decay plus Polyak-Ruppert test from the S10 limit-cycle state, then a rev 3.
  The Born rungs feed only the comparator: GUIDED-FEWER needs a Born/guided ratio >= 2 at every scored size. At S10 the ratio is 4.7-12.0. The exact arm is the reference without which no rung can be scored. The 5% / last-20 statistic is reported, not scored.
* prediction: POLY-CONSISTENT. kappa_guided falls 0.70x from S10 to S12', so the naive N_needed(6x2) at s = 0.10 is about 22k. Comparator: GUIDED-FEWER (kappa_born grows 113x while kappa_guided falls).
  * Probabilities: POLY-CONSISTENT 0.50, AMBIGUOUS 0.20, SUPPORT-LIKE 0.10, UNRESOLVABLE 0.15, INCOMPLETE 0.05. Comparator: GUIDED-FEWER 0.8, MIXED 0.15, NO-GAIN 0.05.
* p: 0.50
* prediction-changes-decision: yes. POLY releases the S14 ladder. SUPPORT redirects the leaf to variance reduction and kills R51.10. UNRESOLVABLE triggers the addendum's rescore and step redesign before any further size. Nothing else measures N_needed at a second size of the same geometry.
* cheapest-kill-tried: (i) R51.10's exact kappa (job 763902) predicts the POLY side (k1 = 0.70). It is the hypothesis under test, and this ladder is its KILL condition. (ii) Round-6 T0 (t0_floor_r06.py) found that at S10 the 5% statistic sat inside a fixed-tau period-2 bounce. This led to rev 2, which is well conditioned at S10: the gap is 15.4 detrended sd_w (score_r07.json). Whether the S12' start (eps 1.1e-3) bounces the same way is measured here by the exact arm. Its slope over steps 61-120 is reported and flagged if above 2 sigma. (iii) Trimmed per the auditor. Guided 1,024 is dropped: s is about 0.94 at S10, far above 0.10. Born 1k, 4k and 16k are dropped: at S10 none reached a sustained 90%, so they cannot set T_tot.
* cost-cpuh: 14.3 (2 array tasks x 7.17 h x 1 CPU; expected use about 9.5)
* cheapest-version: guided 4,096 is kept as the lower guard rung. The bracket for s = 0.10 at 6x2 is predicted near 16k-65k, but kappa's 2-4x error (R51.9) allows N_needed below 16,384, and 4,096 costs 0.107 CPU-min/step. Guided 16k, 65k and 262k cover the predicted bracket and its upper guard. Born is cut to 65k and 262k, the only rungs that can set T_tot. All arms use seed0 9000, one seed each. RULE.md releases a second seed only after scoring, for a bracketing rung within 0.3 ln-units of s = 0.10, as its own gated job.
* convergence: rev 2's statistic is the 61-120 mean at 120 steps against the exact arm over the same window. It is not a convergence criterion. At S10 all 14 arms (jobs 763968 and 769098) ran 120 steps, and the exact arm's 61-120 slope was not significant (z = -1.10). State is resumable every step; a TRUNCATED arm is resumed by rerunning the script.
* wall-time-basis: interactive 6x2 pre-flights give exact 1.688 CPU-min/step and guided 262k 0.425. The other rungs use a + b x unique, with a = 0.077 and b = 1.06e-5, giving guided 4k 0.107, 16k 0.142, 64k 0.230; born 64k 0.179, 256k 0.269. In-job rates multiply these by the in-job / interactive ratios that job 763968 measured at S10: 1.60 for exact and up to 1.75 for sampled arms.
  * Task 0: 204 min x 1.60 x 1.3 = 424 min.
  * Task 1: 164 min x 1.75 x 1.3 = 373 min.
  * --time 07:10:00 for both tasks; the exact arm has --cpu-cap 420.
* power: at S10 under rev 2 the bracket gap was 0.079 against a detrended sd_w of 0.0051, and the seed term was 0.0037. If S12' behaves like S10, its local exponent is 0.58, and an s error of about 0.01 moves ln N_needed by about 0.15, far inside the factor 2.04 between g3 and gH1. UNRESOLVABLE (p 0.15) covers an S12' start that bounces harder. One seed per arm suffices for the first scoring; the RULE clause releases more seeds.
auditor: APPROVED 2026-11-17 7d1639fae94a
