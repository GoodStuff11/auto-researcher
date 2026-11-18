# P4.1 round 36: closure critic SIGNED WITH CHANGES, all applied (editor, 2026-11-18)

No jobs, no commits. Source: per-leaf closure critic, 2026-11-18, on the EXHAUSTED-at-this-budget certificate adopted in `round-35`. Verdict: SIGNED WITH CHANGES. State stays EXHAUSTED at this budget; sign-off recorded as "per-leaf critic, 2026-11-18: SIGNED WITH CHANGES (applied)".

## Changes applied
1+2. Answer replaced with the critic's text (cost half settled; expressivity not settled; 12-site fit about 2 times worse at equal effort, unconverged, one start; half the energy error at equal computer time on 24 sites). The other numbers (0.49–0.56, total spin squared 0.6–1.2, $2.9\times10^{-4}$ vs $1.3\times10^{-4}$, 1.6 at 500 steps) moved to What we showed. Front page P4.1 Answer copied verbatim; P4 bullet in "What it adds up to" rewritten to agree; matrix row updated.
   *Mandated phrase.* R42's `tests/P4.1_cheap_su2_r42/RULE.md` (CHE-NO-WORSE, the class R42 returned) says "Any Answer wording for this class must contain 'at 3 CPU-h per run, one core'". The critic's 3a form rule (plain words, no rule-specific qualifiers in the Answer) conflicts with it. The pre-registered rule binds, so the Answer keeps the phrase verbatim as a parenthetical: "...half the energy error at equal computer time (at 3 CPU-h per run, one core) on 24 sites". The phrase is also in What we showed's R42 sentence.
3. R47 card, What it would take, kill_index: "loses on spin contamination alone" → "loses mostly on spin contamination (40–70% of its error) early in fitting (100–400 of a planned 4,000 steps)" (4,000 = RULE_T1 Amendment 1 maxfun); $\kappa_p\approx1.8$ marked as read at 100 evaluations only (pre-flight smoke, log `round-34-R47`); refusal stands: PASS needs $\kappa_u$ to rise ninefold (0.18 → 1.65); KILL and refusal both close the leaf.
4. Cost sentence in What we showed replaced with the critic's text (1.3–3.6 from R2's per-step ratios 1.25–3.59 at 12–48 sites).
5. Certificate item 5 retitled "Missing ingredients exhausted" with the map (a) commutant lemma and R5.7; (b) R45 run, R40 and R47 refused; (c) Lemma B and R5.5. Refused steps kept, with costs, in item 5 and in What it would take.
6. R47 label in card, kill_index row and certificate items 4–5: "refused by the orchestrator on G2 grounds after the gate passed (KILL's action equals refusal's)".
7. R40 card, kill_index row and certificate: its refusal odds predate R45 (stale); the refusal stands on its unshelve term (the result would change only the Answer's wording; L39 non-transfer).
8. R17's 2.5–3 CPU-h marked "(post-hoc estimate)" in the certificate, What it would take and kill_index (the card already said so).
9. "Neither had converged" replaced by "the charge-only arm met the plateau rule (9.0% last-quarter drop); the spin-reading network was still improving" in What we showed and the R45 card; Conclusion, certificate item 1 and Limits made consistent.
10. What it would take: the converged 24-site comparison is unestimated, because no sampled optimiser converges there; P5.1's R51.8 found the sample need grew like the occupied configurations from 10 to 12 sites (cited by code in the Conclusion; in plain words in What it would take).

## Ledger row 11 (supersedes the label in `round-35`'s final table)
| Row | Ingredient | Step | Outcome |
| :--- | :--- | :--- | :--- |
| 11. 12-site capacity loss at 24 sites | R45 LOSS (job 769413) | R47 T1 (7.7 CPU-h), then T2 (~26) | refused by the orchestrator on G2 grounds after the gate passed (KILL's action equals refusal's), 2026-11-18 |

## Caps and lint
Summary, cards R40/R45/R47 compacted to stay under their caps (R40: dropped duplicate detail on local checks and the 30-step cap; R45: dropped the digest and the 1500-evaluation point and the unscored absolute energies, ratio kept). Lint of `leaves/P4.1`, `kill_index.md` and the front page: 0 errors, 0 warnings.
