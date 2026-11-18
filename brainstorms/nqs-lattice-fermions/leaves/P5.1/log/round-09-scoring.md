# Round 9 — scoring of 769213 (R51.16 `cmp`) and 769211 (R51.11 `qopt` T2), 2026-11-18

Worker pass; no commits, no Slurm. Running jobs untouched: 769376 (6x2 ladder), 769413 (P4.1). Both jobs finished
cleanly (all five arms `status: DONE`; 769213 used 6.27 CPU-h of 14.33, 769211 3.16 of 5.75).

## 1. Gap scale (resolved before scoring)
Two reference energies exist for the 4x3 `che` start ($E_{\rm start}=-7.805526$):
* g1 = 0.022373: $E_0=-7.827899$ (P4.1 r19 `polish.py`, exact $S=0$ level; R51.1's `nf_run.py` stores $R$ on it).
* g2 = 0.023111: $E_0=-7.828637$ (`E0_pg`, lowest (+,+) level, `P5.1_size_scaling_r04/results/info_S4x3_N10.json`),
  used by `score_common.py` for every `ns_run.py` arm.

What the rules pre-registered. RULE_cmp.md rev. 1 and RULE_qopt_T2.md rev. 1 define $R$ only in words ("fraction of the
start's gap to $E_0$") but fix three numbers, all on g1: $R_{\rm pol}=0.7475$ (P4.1 r19/r22 exact ladder polish, gap
0.0223728, `P4.1_cheap_su2_r22/RULE.md`), $R^\*=0.6728$ (= 0.90 x that), and $T_B$ = 7.41M (R51.1 arm B re-scored from
its stored $R$, i.e. g1). **Pre-registered scale: g1 = 0.022373.**

Bug in `score_common.py` (as registered): `guided_T` concatenates arm A's stored $R$ (g1) with the continuation's $R$
recomputed on g2, and the born / q arms are on g2, all against the g1 threshold 0.6728. Arm A's step-60 $R$ is 0.494 on
g1, 0.478 on g2. The rule's scripts were not edited; `tests/P5.1_r05_gen/score_scales.py` (new, read-only on data)
recomputes every arm from absolute energies on one gap and applies the same numeric $R^\*$ and the same sustained rule
(`score_common.tstar`, trailing 20-step mean). Output `results/score_scales.json`. Rescaling both $R$ and $R^\*$
consistently changes nothing (a pure relabelling); the only real alternative is g2 with the numeric $R^\*=0.6728$ kept
(stricter: equivalent to $R_{g1}\ge0.695$). That is what is reported as "other scale".

## 2. R51.16 `cmp` (RULE_cmp.md rev. 1) — **GUIDED-FEWER** (gap 0.022373)

| 0.90 $R_{\rm pol}$ ($R^\*$ 0.6728) | g1 0.022373 (pre-reg.) | g2 0.023111 | as coded (mixed) |
| :--- | :--- | :--- | :--- |
| $T_B$ (R51.1 arm B, 65,536) | 7.41M | censored | 7.41M |
| $T$(A + continuation, 16,384) | 4.51M ($t^\*$ 275) | 5.85M ($t^\*$ 357) | 5.85M |
| $T_{\rm guided}$ | 4.51M | 5.85M | 5.85M |
| born 16,384, $\tau$ 0.005 (904 steps) | censored (best trailing mean 0.652, $R$ end 0.657) | censored (0.632) | censored |
| born 65,536, $\tau$ 0.010 (226 steps) | 14.68M (step 224 of 226; best trailing 0.674) | censored (0.653) | censored |
| ratio $\min(T_{\rm born},14.8{\rm M})/T_{\rm guided}$ | **3.26** → GUIDED-FEWER | 2.53 → GUIDED-FEWER | 2.53 → GUIDED-FEWER |

Secondary (0.80 $R_{\rm pol}$, $R^\*$ 0.598), g1: $T_{\rm guided}$ 2.31M (A+cont; $T_B$ 4.33M), $T_{\rm born}$ 6.42M (born
65k; born 16k 7.98M) → 2.78; g2: 2.99. Verdict identical on every scale; the born 65k crossing on g1 sits 2 steps before
its budget, but censoring would give the same verdict.
Prediction GUIDED-FEWER (p 0.75): hit. Next action per rule: the comparator leg of "What would close it" is met at 12
sites for plain (Born-sampled) SR, credited to a published sampler (guided draws, arXiv:2507.05352); MinSR at $N<P$
still owed (no route). Ledger: changed yes, hit yes.

## 3. R51.11 `qopt` T2 (RULE_qopt_T2.md rev. 1) — **Q-FEWER** (gap 0.022373)
$T_{\rm guided}$ from cmp task 0 (above).

| 0.90 $R_{\rm pol}$ | g1 0.022373 (pre-reg.) | g2 0.023111 / as coded |
| :--- | :--- | :--- |
| q1 16,384 (362 steps) | 1.92M ($t^\*$ 117; $R$ end 0.766) | 2.39M ($R$ end 0.742) |
| q1 65,536 (91 steps) | 4.33M ($R$ end 0.714) | 5.31M |
| $T_q / T_{\rm guided}$ | 1.92/4.51 = **0.43** → Q-FEWER | 2.39/5.85 = 0.41 → Q-FEWER |

Secondary: 0.80 $R_{\rm pol}$ g1 1.10M / 2.31M = 0.48 (g2 0.47), Q-FEWER but within 0.02 of the line. Unique draws per
step (mean): q1 16k 11,550, q1 65k 29,795, guided 16k 6,293, born 16k 4,859, born 65k 10,257 — flatter draws repeat less.
Not registered, reported only: against the best Born arm $T_{\rm born}/T_q$ = 14.68/1.92 = 7.7 (g1); q1 16k's final $R$
0.766 exceeds $R_{\rm pol}$ 0.7475 (exact-sum ladder polish; L61's 75% is not an upper bound on sampled SR).
Prediction "MIXED or Q-FEWER": hit. Next action per rule: R51.11 becomes the leaf's optimiser candidate; its size ladder
(R51.8 protocol, 5x2 → 6x2) waits on the third-volume weight test (Gate 4 FAIL at T0: exact 7x2, or sampled 8x2 / 10x2).
Ledger: changed yes, hit yes.

## 4. R51.15 `adaptN` read-out from task 0 (not a registered rule; flagged for the orchestrator)
The portfolio row's rule: "long small-$N$ run as cheap → censoring artefact; else T2". Guided 16k continued (A + 332
steps), g1, first crossing (the envelope's statistic): 0.90 $R_{\rm pol}$ at 4.26M (envelope 4.8M, best fixed 6.75M),
0.95 at 6.67M (envelope 12.1M, best fixed 22.0M); sustained: 4.51M and 6.93M. The fixed small-$N$ arm run longer beats
the oracle schedule's bound at both targets, so the T1 PASS was a censoring artefact. Card updated; KILL left to the
orchestrator / critic (the row's rule is not a RULE file).

## Library proposals (orchestrator owns library.md)
* At 12 sites, at equal total samples, fixed-step SR with flattened (guided) draws reached a sustained 90% of the
  exactly polished gain with at least 3.3x fewer samples than Born-sampled SR at two step lengths (one seed each).
* $|\psi|$ sampling with reweighting needed 0.43x guided's samples (7.7x fewer than Born) at 12 sites and overshot the
  exact ladder polish ($R$ 0.766 vs 0.7475) — the T1b 10-site gain did not shrink at 12 sites in total samples.
* A gain from an oracle samples-per-step envelope built on runs stopped at a fixed step count can be a censoring artefact:
  continue the smallest-$N$ arm before crediting the schedule (R51.15).
* Two scripts scoring against one threshold must share one gap; a rule should state $E_0$ numerically, not only $R_{\rm pol}$.
