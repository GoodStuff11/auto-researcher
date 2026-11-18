# P5.1 round 4, R51.8 `nscale`: how do the samples per step that fixed-length guided SR needs grow with size? (pre-registered)
Written 2026-11-17, after R51.1 unit 2 (NOISE-FLOOR at 12 sites) and the pre-flights in `logs/`, before any arm below ran.

Question. R51.1 showed on the 12-site `che` (open 4x3 Hubbard, 5+5, U = 8, t' = -0.25, (+,+), P = 2637) that
guided SR at a fixed metric length tau = 0.005, always taken, removes 0.732 of the gap at 262,144 samples per step
(120 steps) against 0.7475 for exact polishing, and that its shortfall after 120 steps falls about as N^-0.87 with
the samples per step N. The leaf asks whether the sampled cost per step stays polynomial in the system size V. The
statistic is the samples per step this optimiser needs to keep pace with its own noise-free version, measured at
several V where the noise-free version is computable by exact sums.

REVISION 1 (2026-11-17, before any arm ran; compute auditor, CHANGES REQUIRED): the primary comparison is now
same-geometry (2-leg ladders S10 -> S12' -> S14); the 4x3 S12 is demoted to a geometry check. The original
unit-1 design (S10 vs 4x3 S12) is kept below only where it is now the geometry check.

Instances (open strips, U = 8, t' = -0.25, irrep (+,+), `che` width 16, exact S = 0; each start = `vmc.py --mode
train` with the 12-site start's protocol: 1000 SR iterations, 1024 chains, lr 0.05, decay 0.1, last-25% average,
shift 1e-3, seed 0, `starts/`):
* S10: 5x2, N_e = 8 (4+4), P = 1832, 11,100 orbit representatives (start trained, 9.9 CPU-min).
* S12': 6x2, N_e = 10 (5+5), P = 2637, 156,816 representatives (start: quick job `submit_quick_s12p.sbatch`).
* S14: 7x2, N_e = 12 (6+6), P = 3590, about 2.25 million representatives (start and exact kappa: gated job
  `submit_kappa_s14.sbatch`; a full ladder at S14 is a later unit, costed after its kappa).
* Geometry check only: S12 = 4x3, N_e = 10, R51.1's start; guided rungs = R51.1's A, B, C (seed0 9000).

Arms (`ns_run.py`, tau = 0.005, 120 steps, seed0 9000, exact i.i.d. draws, duplicate draws aggregated exactly;
resumable state saved atomically every step):
* exact (N = infinity): exact S_T and f over all representatives (the reference), at every primary size.
* guided N in {1024, 4096, 16384, 65536, 262144} (the 1,048,576 rung is dropped: a need censored above 262,144 at
  S10 already gives r < 1 against any larger size's bracket); larger rungs only where a bracket requires them.
* born (plain SR; at N < P MinSR's regime) N in {1024, 4096, 16384, 65536, 262144} at S10 and S12'.

Statistics. E_plat(arm) = mean exact energy over the arm's last 20 steps; G_ref(V) = E_start - E_plat(exact);
shortfall s(arm) = (E_plat(arm) - E_plat(exact)) / G_ref. N_needed(V) = the N at which s = 0.05 for the guided
sampler, by linear interpolation of ln s against ln N between the smallest rung with s <= 0.05 and the rung below
it; censored if no bracket exists. Growth references at the starts (`ns_run.py --info`): g3 = (V2/V1)^3 and
gH = exp(H[p])(V2) / exp(H[p])(V1).

Primary verdicts, r' = N_needed(S12') / N_needed(S10), g3 = 1.728, gH from the two starts (S12' probe start:
exp(H) 3864; recomputed at the trained start before scoring), UNRESOLVABLE if gH < 1.5 g3:
* POLY-CONSISTENT if r' <= g3; SUPPORT-LIKE if r' >= gH; AMBIGUOUS otherwise or if censoring leaves r' undecided;
  INCOMPLETE if a scored arm stops before step 120.
* Three-size series (after S14's ladder): POLY-CONSISTENT-3 if N_needed(S14)/N_needed(S10) <= 1.4^3 = 2.744 and r'
  was POLY-CONSISTENT; SUPPORT-LIKE-3 if both steps are SUPPORT-LIKE; AMBIGUOUS-3 otherwise.
Next actions: POLY-CONSISTENT -> S14 ladder, then the 24-site protocol; SUPPORT-LIKE -> force variance-reduction /
resampling batch first, the Answer reports the trend; AMBIGUOUS -> S14 ladder.

Geometry check (secondary), r = N_needed(S12, 4x3) / N_needed(S10). R51.1's arms bound N_needed(S12) in
(65,536, 262,144] provided the 4x3 exact arm's R_plat lies in [0.732, 0.770]: then B has s >= 0.071 and C has
s <= 0.05 (C's s is 0.013-0.021 if R_plat <= 0.7475). Caveat: an exact-limit fit through A, B, C extrapolates past
R_pol (fitted Delta_inf = -0.038, i.e. R_ref about 0.785), in which case C's s = 0.067 and the bound becomes
(262,144, ...); the staged thresholds below assume the bound and are re-checked against the exact arm if it is bought.
So S10's ladder alone decides r when N_needed(S10) >= 151,700 (r <= 1.728) or <= 16,850 (r >= 3.89); otherwise
the 4x3 exact arm (`submit_unit1_ref12.sbatch`) is bought only if kappa or the ladders show that geometry matters
(kappa(S12', 2-leg) and kappa(S12, 4x3) differ by more than 1.5x, or r' and the bounded r disagree in verdict).

Staged release: (1) quick T1s: S12' start + exact kappa (quick job), S14 start + exact kappa (gated job);
(2) unit 1a = S10 ladder (`submit_unit1_s10.sbatch`); (3) unit 1c = S12' ladder (exact + guided + born, prepared
after its trained start exists, pre-flight 1.69 CPU-min per exact step measured on the probe start); (4) S14 ladder.

Comparator (the leaf question's "fewer total samples than SR or MinSR", scored per primary size from the same arms):
T_tot(sampler, V) = min over the sampler's rungs of N x (first step at which E <= E_start - 0.9 G_ref), censored
if no rung reaches it in 120 steps. GUIDED-FEWER if T_tot(born) / T_tot(guided) >= 2 at every size scored;
NO-GAIN if it is <= 1.25 at any size; MIXED otherwise.

What a verdict does and does not show. POLY-CONSISTENT over 10 -> 12 (later 10 -> 14) sites is evidence against a
sample need that tracks the Hilbert space or the effective support, not a proof of polynomial scaling: over a factor
1.2-1.4 in V any polynomial and a slow exponential agree. Confounds named now: geometry (removed from the primary ratio by
revision 1; checked against 4x3), doping (1/5 vs 1/6), P growing as V^2 for this network at fixed width, exact i.i.d.
draws (no Markov-chain autocorrelation), and one seed per arm (a second seed is released only for an arm whose s
lies within 0.3 ln-units of 0.05 at a bracketing rung). The cost per step with duplicate aggregation is reported
(unique draws / N) but not scored: it is a small-cluster saving, predicted at T0 to vanish once exp(H[p]) >> N.
Representativeness (L40), measured before any arm ran (`results/info_S5x2_N8.json`, `info_S4x3_N10.json`): the S10
start is 5 times closer to the lowest (+,+) level in relative terms (eps 5.5e-4 against 3.0e-3) and its P / (orbit
representatives) is 0.165 against 0.017, so S10 is the easier landscape; a SUPPORT-LIKE verdict could partly reflect
that, which the S12' start's own eps (probe start, not yet trained: 3.5e-2) will show. Measured growth references:
exp(H[p]) 781 (S10) and 3035 (S12), gH = 3.89 >= 1.5 g3 = 2.59, so the 10 -> 12 comparison is resolvable.

## Revision 2 (auditor, 2026-11-17, adopted before any unit 1c data)

Primary statistic (replaces the 5% / last-20 rule): E_win(arm) = mean exact energy over steps 61-120; G_ref(V) = E_start - E_win(exact); s = (E_win(arm) - E_win(exact)) / G_ref; N_needed(V) = the N at which guided s = 0.10, by ln s vs ln N interpolation between the bracketing rungs; censored if no bracket exists. Where two seeds exist, s is the seed mean. Conditioning: N_needed(V) is scored only if s(lower rung) - s(upper rung) > 2 sd_w, where sd_w = max(sd of the upper rung's three 20-step window means over steps 61-80, 81-100, 101-120; |seed difference| / sqrt(2)). Otherwise V is UNRESOLVABLE and any ratio using it is UNRESOLVABLE (not AMBIGUOUS); next action: redesign the step (decaying or line-searched tau) before buying another size. Verdict references (g3 = 1.728, gH1 = 3.52), the Born T_tot comparator and the INCOMPLETE rule are unchanged. The 5% / last-20 N_needed is reported, not scored. Each size's exact-arm slope over steps 61-120 is reported; above 2 sigma it is flagged "exact still descending" (not scored).

## Rev 2 addendum (per-leaf critic, 2026-11-17, before any unit 1c data)

Appended, not replacing anything above. Written while job 769098 (S10 second seeds) is running and before any of its output is scored; no unit 1c (S12') data exist.
* Disclosure: when revision 2 was registered, the S10 denominator under the new statistic (N_needed(S10) about 29,530 at s = 0.10 from unit 1a) was already known. The revision's motivation (the 5% need moved 94,150 -> 258,000 -> censored with the averaging window, from the period-2 bounce) was established before any S12' data.
* sd_w is computed from detrended window means: fit a least-squares line to the upper rung's per-step exact energy over steps 61-120, subtract it, then take the sd of the three 20-step window means (61-80, 81-100, 101-120); the seed term |seed difference| / sqrt(2) is unchanged. Reason: a still-descending upper rung (S10's 65k rung descends at 2.3 sigma) inflates the raw window sd.
* UNRESOLVABLE action (replaces "redesign the step before buying another size"), in order:
  (i) zero-compute rescore from the same runs: for each size and sampler, total draws until the trailing 20-step mean of E first reaches E_start - 0.9 G_ref(V), i.e. a sustained 90% of that size's own exact gain (the comparator's statistic), reported per size and used for the size ratio;
  (ii) R51.7's pre-registered unshelve test: the S10 exact arm is a fixed-step limit cycle (last-40 slope about -1.7e-4 in R, inside R51.7's trigger of |slope| < 2e-4), so run step decay with Polyak-Ruppert averaging from that state against fixed-step continuation at equal samples; then rerun the S10 ladder under the new step, with a revision 3 of this rule registered before any of that ladder's data.

## Addendum (per-leaf critic re-check, 2026-11-17, before job 769376 is scored)

Appended, not replacing anything above. Written while job 769376 (S12' ladder) is running and before any of its output is scored. Earlier text, including `voi_*.md` files whose digests are approved, is left as written; the corrections below take precedence over them.
* Primary comparator statistic (critic Major 2). The comparator is scored with this file's own first-crossing statistic: T_tot(sampler, V) = min over that sampler's rungs of N x (first step at which E <= E_start - 0.9 G_ref), G_ref as defined under revision 2 (exact-arm mean over steps 61-120). The sustained (trailing 20-step mean) statistic of RULE_cmp.md and of the rev-2 addendum's item (i) is reported only; it decides nothing in the comparator. Correction: the S10 figure "4.7-12.0" in `voi_r06_s12p_ladder.md` is the sustained statistic. Under the registered statistic S10 gives guided 1.72M (seed0 9000, 16,384 x 105) and 1.75M (seed0 19000, 16,384 x 107) against Born 16.25M (one seed, 262,144 x 62), i.e. 9.4x and 9.3x (two guided seeds, one Born seed, 120-step horizon). The last-20 G_ref gives the same draw counts.
* Censored Born (critic Minor 4). A Born T_tot that no rung reaches within 120 steps counts as T_tot(born) >= (largest Born rung's N) x 120, as RULE_cmp.md counts censoring (at S12': >= 262,144 x 120 = 31.46M). The size counts toward GUIDED-FEWER only if that lower bound is >= 2 T_tot(guided); a censored Born result cannot satisfy NO-GAIN at that size. If the guided T_tot is also censored, the size is not scored on the comparator.
* MinSR (critic Minor 5). Every arm, including the Born arms, uses the SR solve in parameter space; no arm runs MinSR. MinSR equals SR at N >= P, so the Born arms are also MinSR there; the phrase "at N < P MinSR's regime" above names a regime, not a test. The MinSR half of the comparator (N < P) is untested by this rule.
* e^H convention (critic Minor 7). exp(H[p]) here is the Shannon entropy of p over orbit representatives, each carrying its orbit's total weight (`ns_run.py --info`): 781 at S10, 3,035 at S12. Counting each configuration of an orbit separately (as in R51.11's T0 log, `tests/P5.1_r05_gen/t0_weightvar_R51.11.py`) gives 3,053 at S10, about 3.9 times larger (point group of order 4). gH uses the representative convention at every size.

## Addendum (2026-11-18, before any 7×2 data)

Appended by the round-13 worker, not replacing anything above. Zero compute. Written before any S14 ladder arm or the
S12' seed0 19000 arm has run. The only S14 numbers that exist are the trained start, `results/info_S7x2_N12.json`
(exp H 17,370; E0_pg not computed) and the exact kappa (job 763903). The two earlier sizes are scored:
N_needed(S10) = 31,284 (`score_r07.json`, two seeds at both bracketing rungs) and N_needed(S12') = 118,626
(`score_r08.json`, one seed, r' 3.79, SUPPORT-LIKE, margin 1.08x). Round 12's force variance-reduction batch is
spent: all seven routes were killed or merged. So the rev-1 SUPPORT-LIKE action ("variance-reduction batch first")
is done, and the S14 ladder is the next spend.

* **Step-2 reference: S10.** The second step is scored as r'' = N_needed(S14) / N_needed(S10). It is compared with
  g3'' = (14/10)^3 = 2.744 and gH'' = exp H(S14) / exp H(S10) = 17,370 / 781 = 22.24 (representative convention, as
  gH1). In absolute terms, SUPPORT-LIKE at step 2 needs N_needed(S14) >= 695,900, and the POLY line is <= 85,840.
  There are three reasons for this choice.
  (a) The verdict table above states exactly one three-size ratio, "N_needed(S14)/N_needed(S10) <= 1.4^3 = 2.744".
  Scoring the SUPPORT side on that same ratio, against the gH of that same ratio, scores one number against both
  lines. The table never defines an S12'-based step-2 ratio.
  (b) The S10 denominator rests on two seeds at both bracketing rungs. The S12' denominator rests on one seed (two
  at 65k after this job) with a 1.08x margin. Dividing by S12' would carry step 1's noise into step 2 with the
  opposite sign, so a low draw at 6×2 would help step 1 fail and step 2 pass.
  (c) The two thresholds differ by only 7.7% (695,900 against S10, 749,300 against S12' with gH 6.32 and
  g3 1.588). The S12'-based ratio is reported, not scored. A result in [695,900, 749,300) is labelled
  "SUPPORT-LIKE vs S10 only".
* **Minimal rungs at S14:** exact, guided 262,144, and guided 1,048,576. All use seed0 9000, tau 0.005, 120 steps,
  the rev-2 statistic (steps 61–120, s = 0.10) and the rev-2-addendum conditioning (a gap of more than 2 detrended
  sd_w of the upper rung).
  * If s(262k) <= 0.10, N_needed(S14) <= 262,144 (censored below). r'' is then at most 8.38, which is below gH'',
    so step 2 is not SUPPORT-LIKE.
  * If s(1M) > 0.10, N_needed(S14) > 1,048,576 (censored above). r'' is then at least 33.5, so step 2 is
    SUPPORT-LIKE.
  * Otherwise the two rungs bracket 0.10 and the rule interpolates as in rev 2.
  * The two-rung gap s(262k) − s(1M) must exceed 2 sd_w(1M) in every case, or S14 is UNRESOLVABLE.
  * There is no guided 65k and no Born arm at S14. The comparator is not scored at S14; GUIDED-FEWER rests on
    S10 and S12' only, and the Answer must say so.
* **Second seed of step 1, folded in:** the S12' guided 65,536 rung with seed0 9000 → 19000 is run first in the
  same job, at about 1% of the cost. It is the rung the seed clause names (0.28 ln-units from 0.10).
  * s(65k) becomes the two-seed mean. The 262k rung stays at one seed.
  * The seed term |seed difference at 65k| / sqrt(2) enters sd_w for step 1's conditioning (max with the
    detrended window sd). It is the only measured seed term at S12'.
  * r' is re-scored against 1.728 and 3.52 by `score_r13_step1.py`.
  * If r' is no longer SUPPORT-LIKE (AMBIGUOUS or UNRESOLVABLE), SUPPORT-LIKE-3 is unreachable, because it needs
    both steps. POLY-CONSISTENT-3 was already unreachable, so the S14 arms cannot change the three-size class. The
    job then writes `results/STOP_r13` and stops before any S14 arm. The class is reported as "AMBIGUOUS (step 1,
    two seeds); S14 not bought". Buying S14 after that needs a new VOI note.
* **Reported-only check: reference shifted by its own slope.** For each size (S10, S12', S14), fit the exact arm's
  61–120 slope c, which is already reported. Then:
  * E_win'(exact) = E_win(exact) + 60c (one more window of descent);
  * G_ref' = E_start − E_win';
  * s' = (E_win(arm) − E_win') / G_ref';
  * N'_needed, r'_shifted and r''_shifted.
  These are reported next to the scored values and decide nothing.
* **Still-descending flag at S14:** the flag (slope beyond 2 sigma) is expected at S14 and is not scored. A lower
  true late-time reference raises every guided s, so the flag biases N_needed(S14) low, that is, toward POLY.
  * A step-2 result on the SUPPORT side (r'' >= 22.24) stands as scored.
  * A result on the POLY side of the SUPPORT line (r'' < 22.24, scored AMBIGUOUS-3) is marked
    "provisional: exact still descending". If r''_shifted >= 22.24, the marking also says that the 120-step
    horizon leaves the SUPPORT/AMBIGUOUS split unresolved.
  * The flag never changes a class by itself.
* **What each outcome does to the leaf's Answer** (the editor applies it; this rule only registers it):
  * SUPPORT-LIKE-3 (step 1 still SUPPORT-LIKE with two seeds at 65k, and r'' >= 22.24). This meets the R51.8
    clause of "What would close it" (SUPPORT-LIKE at two size steps). Of the "none removes it" list, guided draws
    (the sampler measured), R51.5 (killed) and zero-variance force control variates (R51.19 and R51.25, killed in
    round 12) are settled. The Answer becomes "No, for SR-type optimisers with these samplers at this scale",
    with the three needs and both ratios. The leaf is marked ANSWERED (no) only once R51.7 (shelved) and R51.11
    (shape unresolved) are also resolved. Until then the Answer carries "pending R51.7, R51.11".
  * AMBIGUOUS-3 (step 1 SUPPORT-LIKE, r'' in (2.744, 22.24)). The Answer reports numbers only: N_needed at 10, 12
    and 14 sites, r', r'', and r'' against both lines. It makes no trend or growth-law claim and stays "Not
    known". It is provisional if S14's exact arm is flagged.
  * POLY-3 is **unreachable** under this rule. POLY-CONSISTENT-3 requires r' POLY-CONSISTENT. r' is 3.79, and with
    the second seed it can fall to 1.728 only if the 65k two-seed s drops to about 0.086 (bracket [16k, 65k]). That
    needs a seed-2 value of about 0.040, a shift of about 24 pooled seed-sd (0.0039). A POLY-side step 2 (r'' <= 2.744, N_needed(S14) <= 85,840; impossible to score below 262k
    with these rungs except as "<= 262,144") is therefore scored AMBIGUOUS-3. It is reported as "need grew 3.8x
    from 10 to 12 sites but no faster than V^3 from 10 to 14 (one seed)". It is provisional if the exact arm is
    flagged, and the Answer makes no polynomial claim.
  * UNRESOLVABLE at S14 (the two-rung gap is at most 2 sd_w(1M)). r'' is UNRESOLVABLE. The Answer reports
    numbers only. Next action: the rev-2 addendum's zero-compute rescore (i).
  * INCOMPLETE (a scored S14 arm stops before step 120). This is not scored. It is resumable from state.npz under
    a new gated submission.
  * Step 1 leaves SUPPORT-LIKE with the second seed: the job stops before S14 (above). The Answer's 10→12 sentence
    is updated to the two-seed r', and the size question stays "Not known".
