# P5.1 round 14 — 7×2 ladder rejected, R51.8 SHELVED; next spend costed (worker, 2026-11-18)
Nothing was submitted and no Slurm job was started. No commits.
## 1. Compute auditor REJECTED the 7×2 ladder (digest 7c2fdfcdf155)
R51.8 closes as SHELVED at this budget (route-card rules: the next step was refused by the auditor; not a kill, the size question is untested beyond 12 sites).
* Why shelved: modal outcome AMBIGUOUS-3 (p 0.49) changes nothing (numbers only, Answer stays "Not known"). SUPPORT-LIKE-3 (p 0.34) would likely land near the line and trigger the RULE.md seed clause (~31 CPU-h), beyond the 28 CPU-h left after the 240 cap, and would still be pending R51.7 and R51.11. One seed, 14 sites. The exact arm is ~75% of the cost and cannot be shortened (G_ref is its 61–120 mean; the 7×2 lowest level is not computed). The 7×2 exact-step rate (68.6 CPU-min) was inferred from κ passes (763903 × the 6×2 ratio), not measured directly.
* Recorded prediction: AMBIGUOUS-3 0.49, SUPPORT-LIKE-3 0.34, STOP 0.07, UNRESOLVABLE 0.05, INCOMPLETE 0.05.
* Unshelve: a cheaper exact reference at 7×2 or a larger budget (cap plus seed clause, ~270 CPU-h).
* Kept result: N_needed 31,280 → 118,600 (r′ 3.79 vs support 3.52, one seed at 6×2, margin 1.08×, exact arm still descending); comparator GUIDED-FEWER at 10 and 12 sites.
* Edits: card R51.8 (SHELVED; Why shelved, What would unshelve it), kill_index.md (SHELVED row), summary.md (portfolio row removed, Shelved line, Updated, Answer, What it would take, size paragraph, Most promising direction, Next step, What would close it). The RULE.md step-2 addendum stays registered and unused; `submit_r13_s14_ladder.sbatch` is not to be submitted.
* Answer now: the need grew like the occupied configurations over the one size step measured (10 → 12); the third size needed to tell that from a steep power was not bought.
## 2. Candidates for the remaining 268 CPU-h (pre-audit gate checks; nothing submitted)
### (a) R51.7 step decay with Polyak–Ruppert averaging, 10 and 12 sites
* Rule (written first): `tests/P5.1_size_scaling_r04/RULE_r14_decay.md`. 60-step continuations from the step-120 states of R51.8's arms (S10 exact, S10 guided 4k/16k/65k, S12' guided 16k/65k/262k; 18 arms), CONT (fixed 0.005) vs DEC ($\tau_k=0.005(1+k/10)^{-0.75}$ plus running average from step 121), same draws. $s$ on steps 151–180 against R51.8's registered references; $\rho=N^{\rm DEC}/N^{\rm CONT}$; DECAY-HELPS if ≤ 0.5 at both sizes, NO-GAIN if ≥ 0.8 at both, MIXED otherwise; bounce amplitude at S10 exact and $r'_{\rm DEC}$ reported.
* Cost: cap 6.0 CPU-h (expected 4–5.1) plus ~0.1 interactive. Needs a new driver `ns_decay.py` with an identity test; sbatch is a DRAFT that exits 1.
* Prediction: MIXED 0.45 (helps at 10 sites where the shortfall is a noise ball; slows progress at 12 where every arm still descends), NO-GAIN 0.25, HELPS 0.20, UNRESOLVABLE/INCOMPLETE 0.10.
* Changes the Answer only on DECAY-HELPS with $r'_{\rm DEC}<3.52$ (the growth sentence would then be partly a fixed-step artefact). Beyond small clusters: no; and its 180-step statistic is not the registered N_needed, so it cannot re-score r′.
* Gate: `compute_gate.py check --pre-audit` → cap 6.0, **GATE: PASS**, VOI digest 6ba5a3fb0bcd (`voi_r14_decay.md`).
### (b) R51.11: fourth volume and/or size ladder
* Fourth volume (sampled 8×2 $E_q[w^2]$): needs an 8×2 training job (7×2 took 0.61 h for 1000 iterations; 8×2 about 1 h) plus a sampling pass, about 2–3 CPU-h. Not re-proposed: round 10's decision stands ($V^2$ and exponential predictions 10% apart at 8×2, predicted SHAPE-UNRESOLVED p ~0.7, changes no decision), and it only forecasts the need.
* Size ladder (recommended form). Rule (written first): `tests/P5.1_r05_gen/RULE_r14_qladder.md`. R51.8 rev. 2 protocol unchanged except $q\propto|\psi|$ (`ns_q.py --qmix 1.0`), against R51.8's exact references at 10 and 12 sites; $r'_q$ vs 1.728 / 3.52. Rungs: S10 4k s9000 reused (`results/storm_N4096_a1.0_q1.0`; $s$ = 0.154, computed before the rule and disclosed), new S10 4k s19000, 16k ×2 seeds, 65k guard; S12' 4k, 16k, 65k. No new code.
* Cost: cap 4.0 CPU-h (expected 2.8–3.6), one task.
* Prediction: SUPPORT-LIKE 0.50 ($N_q(10)\approx6.9$k from the reused rung, so the line is ~24k at 12; a held 3.6–4.7× gain gives 25–33k, and $E_q[w^2]$ rising ×1.43 pushes up), AMBIGUOUS 0.37, POLY 0.03, UNRESOLVABLE 0.05, INCOMPLETE 0.05.
* Changes the Answer: SUPPORT-LIKE extends the growth sentence to both samplers and settles R51.11's "none removes it" item for this step; POLY-CONSISTENT would make the growth specific to the flattened sampler. Beyond small clusters: no (same one size step).
* Gate: cap 4.0, **GATE: PASS** (no auditor needed, cap ≤ 10), VOI digest 84711a59d932 (`voi_r14_qladder.md`). Needs the orchestrator to release `RULE_R51.11_vol3.md`'s ladder hold (round 10's reopen trigger, R51.8 showing faster-than-polynomial growth, is met by step 1).
## 3. Recommendation
(b) the R51.11 size ladder: cheapest (4 CPU-h), no new code, ready to submit, uses R51.8's registered statistic and references, so its $r'_q$ is directly comparable to 3.79, and every outcome settles or changes something. (a) is the natural follow-on (about 6 CPU-h after the driver is written); neither says anything beyond 12 sites.
## Lint
`check_brainstorm.py leaves/P5.1 kill_index.md`: 0 errors, 0 warnings (summary 2497/2500 tokens).
