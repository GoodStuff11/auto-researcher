# Round 32 — per-leaf critic (2026-11-18) applied

Editor pass; no jobs, no commits; `tests/` untouched (R45 job 769413 running). Critic verdict: P4.1 SIGNED, conditional on objection 1 (applied below).

1. Answer: "the same optimiser" → "the same optimiser, tuned on the charge-only network"; "at 3 CPU-h per run, one core, i.e. three CPU-hours" → "at 3 CPU-h per run, one core". The exact phrase "at 3 CPU-h per run, one core" is kept because R42's `tests/P4.1_cheap_su2_r42/RULE.md` (line 95) requires it in any Answer wording for this class; only the redundant "i.e. three CPU-hours" was dropped. Copied verbatim to the front page's P4.1 Summary bullet (the Current leaf state table carries State and Waiting on only, no Answer).
2. What we showed: "(also at half the time)" → "(0.53–0.58 at half the time)".
3. L64 and R43: the causal clause "bought no energy because each step costs twice as much" replaced by step-matched evidence. Averaged over steps 561–580, `nnbf_even` beats bare `nnbf` by 0.13–0.16 (seed 0: −16.279 vs −16.146; seed 1: −16.288 vs −16.131), while `che` at the same steps is lower still (−16.415, −16.406), so `che`'s lead holds per step too. Caveat: the learning-rate schedule runs in CPU fraction, so step-matched comparisons mix schedule phase. Equal-CPU statement (+0.017, 0.000) unchanged. Also reflected in R42 and R46 "Strongest case" lines and summary chain item 6.
4. R43, seed 1: $E_{\rm even}-E_{\rm nnbf}=2.6\times10^{-7}$ with $\sigma\approx0.005$ — coincidence checked: distinct parameters (relative difference 0.54); dE, $\langle S^2\rangle$ and half-budget energies differ.

Checked-by lines: R43 "per-leaf critic 2026-11-18: upheld (ρ′, validity and pre-registration verified); objections 4, 5 applied"; R42 "upheld; survives R43"; R46 "reopen trigger not met (verified)".

Compaction to caps: R43 (idea, lineage, outcome, evidence wording), R42 (two phrases), summary (Limits, Strongest objection (ii), Updated) and L64 (Evidence, Counter-evidence) shortened without dropping ids, numbers or required fields.
