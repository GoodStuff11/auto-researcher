# Round 9 — per-leaf critic on the R51.5 kill (2026-11-18) applied

Editor pass; no jobs, no commits; `tests/` untouched (769376, 769213, 769211 running). Critic verdict: R51.5 kill SIGNED (upheld under the pre-registered rule).

5. R51.5 Strongest case against the kill answered with the matched-progress rise table (4096 arm vs R51.1 arm A, 16,384 per step): $R<0.3$: 9/51 vs 0/13; 0.3–0.4: 24/70 vs 0/15; 0.4–0.5: 50/119 vs 4/33; overall below $R$ 0.495: 83/240 = 0.35 vs 2/59 = 0.03, Fisher $p=10^{-7}$. The extra rises come from small blocks, not from fixed-step SR near the floor. "Why this kills it": the majority-vote model (two of four uphill with probability 0.43) replaced by the measured fact that one 4096-sample block step went uphill on 35% of steps at matched progress against 3% for the 16,384-sample average of four such blocks, so averaging, not a median, removed the noise.
6. Library: new L72 (PROVISIONAL, source R51.5): a count threshold within about 1 sd of its predicted count needs a stage-2 seed clause (as R43's rule had); otherwise the verdict can turn on a coin flip. R51.5's Lesson field points to it.
7. Summary Next step: each scorer of R51.16 and R51.11 must state its gap explicitly (0.022373 vs `score_common.py` 0.023111). The R51.5 card's Evidence now states its gap (0.022373).

Checked by (R51.5): "per-leaf critic 2026-11-18: upheld under the pre-registered rule; matched-R rise rates 0.35 vs 0.03 support it; objections 6, 7 applied". Card compacted to its 900-token cap (Source analogy and Evidence wording); summary compacted to its cap (Updated, Limits).
