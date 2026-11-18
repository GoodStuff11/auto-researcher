# Per-leaf critic, P5.1 round 17: R51.32 (job 773759) — 2026-11-18

Checked: registration holds. RULE.md and score_r17_regime.py were committed at 07:01:16 (51471de), job 773759 started at 07:01:38 (sacct), and neither file changed afterwards; only voi.md's p went from DRAFT to 0.40. τ* = 0.0025 came from `select` inside the job (log 07:44:22). τ 0.00375 had g1 0.163 and τ 0.00125 had g1 0.018, both outside the band, so no other grid value was eligible. No L67 repeat: g1 only chooses τ and is not used to predict the need. L33 and L78 are respected, since both exact references drift by about the same share of G over steps 61–120 (4.2% at 5×2 τ 0.0025, 4.3% at 6×2).

## Major
1. **The verdict depends on where the target sits in the band, and the rule's near-line clause does not test this.** At 5×2, N_needed falls steeply as g1 rises (elasticity about 0.79 between τ 0.005 and 0.0025: 31,284 → 98,387 while g1 goes 0.301 → 0.0703). Interpolating from those two points:
   - g1 = 0.116 (top of the band): N about 66,400, r'_m about 1.79, AMBIGUOUS.
   - g1 = 0.022 (bottom of the band): N about 246,000, r'_m about 0.48.
   So the band admits r'_m from about 0.5 to 1.8. It crosses the 1.728 line, and the bottom end has the need falling with size, which no size mechanism produces. L72's clause covers seed noise (0.1 ln-units), but here the target uncertainty is about ±0.6 ln-units. Every target inside the band gives "not SUPPORT-LIKE": SUPPORT-LIKE needs g1 about 0.27, which is the τ 0.005 regime. The POLY label does not survive the whole band, though.
   - Fix: base the matching on the window statistic both sizes already have. m (mean |ΔE|/G over steps 61–120) is 0.051 at 5×2 τ 0.0025 and 0.0485 at 6×2; matching on m gives r'_m about 1.14.
   - Report the result as "not support-like at any in-band target; ~1.2 when matched on g1 or m".
   - Measuring the τ 0.00375 rungs at 5×2 (about 0.5 CPU-h) would replace the interpolation with data.
2. **The need is defined against each arm's own noise-free reference, and that definition alone moves the 10-site number by more than 8×.** Recomputed from stored arms, against the deepest 5×2 reference (τ 0.0025, G 0.001720 vs 0.001601 at τ 0.005):
   - The τ 0.005 guided rungs give s_common 0.204, 0.130 and 0.118 at 16k, 65k and 262k, so they never reach 0.10 (censored above 262k).
   - The τ 0.0025 rungs reach 0.10 at about 98k.
   So for a fixed accuracy, τ 0.0025 is also the cheaper choice at 10 sites. This supports R51.32, and the fixed-τ ×3.79 was partly a shallow-reference artefact.
   - The same check cannot be made at 12 sites. Its only reference is τ 0.005, still descending, and a shorter τ there might go deeper and raise N(6×2).
   - The Answer should say the needs are "within 10% of each size's own noise-free run at that step length", not of the representable minimum the Question names. It should also add this common-reference check as independent support.
3. **Stale records.** The card R51.32 Outcome and the Route Portfolio row still say "LIVE (gated, not submitted)" with the prior. The summary's `Conclusion` field still states R51.8 as SUPPORT-LIKE and does not mention R51.32. L78 has no line recording that R51.32 tested its lesson. Update all of them to POLY-CONSISTENT (job 773759, 1.5 CPU-h) and add the in-band range from objection 1.

## Minor
4. **Wording in the Answer.**
   - "one step makes the same fraction of progress at both sizes": g1 is a first-order prediction, and the 6×2 value is the mean of two states that differ ×5. Say "the same predicted gain per step (0.070 vs 0.069; realised mean |ΔE|/G 0.051 vs 0.049)".
   - "a fixed step length that is too long for the smaller cluster": back it with objection 2's number (its reference is 7% shallower) or drop it.
   - Add "over one size step of 10 → 12 sites, one 12-site seed". The current "less than the cube of the size" invites a scaling reading that a factor of 1.2 cannot support. r' 1.21 is also V^1.05 or any exponential with a small rate, and the Answer should say so in one clause.
5. **The prediction was off.** p(SUPPORT-LIKE) was 0.40 and POLY was the 0.12 class. VOI's two-opposing-effects analysis named effect (ii) (a shorter step removes the bounce and deepens the reference), and that effect dominated. The same effect predicts the 12-site sign: a shorter 6×2 τ would deepen its reference and raise N(6×2). Record this as the sign of the remaining one-sided-matching bias. The fixed-τ result understated the 10-site need, and nothing yet bounds the 12-site understatement.
6. **Next spend: a regime-matched 7×2 is right in kind, but the R51.8 design is the wrong version.**
   - g1 falls with size at fixed τ (0.30 → 0.069 from 10 to 12 sites), so matching 7×2 to 6×2 would need a longer 7×2 step, which risks divergence, or several 7×2 exact arms at about 120 CPU-h each.
   - Cheapest decisive version:
     1. A 5-step pre-flight that measures the 7×2 exact-step rate (it is still inferred).
     2. One 7×2 exact arm at τ 0.005 with the xsx log.
     3. Match the cheap sizes down to the 7×2 m and g1 by shortening τ: 5×2 costs about 0.3 CPU-h per τ, and 6×2 about 5.8 CPU-h for the exact arm plus a ladder.
     4. At 7×2, a single guided rung at the N that divides POLY from SUPPORT (about 2.4 × N(6×2, matched)), plus a second seed only if it lands within L72's band. A censored interval decides the class.
   - Before that: the owed 6×2 second seed (2.9 CPU-h) and the 5×2 τ 0.00375 rung pair, because the 1.21 currently rests on one 12-site seed.
   - Even three sizes up to 14 sites cannot separate a polynomial from an exponential. The 7×2 step can only rule out support-like growth twice, which is the "no" branch of the closing criterion.
