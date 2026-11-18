# Round 10 — R51.15 kill, R51.11 third-volume weight test (2026-11-18)

Leaf worker; no jobs submitted, no commits; files of 769376 and 769413 untouched.

## 1. R51.15 `adaptN`: KILLED at T1 (orchestrator decision)
* **Criterion provenance, checked in git** (`git log -p` on the card and on `summary.md`): the censoring criterion was on
  the card from a2fd002 (2026-11-17 22:40, "a small-$N$ arm run longer may reach 0.95 as cheaply. R51.16's guided 16k
  continuation … measures exactly this") and in the portfolio row from 5380e13 (22:53, "the long small-$N$ run reaches 95%
  as cheaply → the schedule's gain was censoring; otherwise build the schedule run (T2)"). Job 769213 was submitted in
  b560aec (22:56) and its data entered the card in b8c0de8 (2026-11-18 02:37). The criterion predates the data, so the
  kill stands as KILLED (not SHELVED). The rule lived on the card and the portfolio row, not in a RULE file.
* **Evidence** (from `log/round-09-scoring.md` §4): fixed guided 16,384 continued 332 steps first reached 0.90 / 0.95
  $R_{\rm pol}$ at 4.26M / 6.67M draws (sustained 4.51M / 6.93M), against the T1 oracle envelope's 4.8M / 12.1M.
* Card rewritten with the KILLED fields; kill_index row added; portfolio row removed, Killed list and Alternatives updated.

## 2. R51.11 `qopt`: Gate 4 third-volume weight test
* **Choice of test.** Exact 7x2 (V = 14, ladder geometry). The 7x2 start of R51.10 (job 763903,
  `starts/S7x2_N12_..._theta.npy`) was reused; amplitudes only, no Hamiltonian, so the 16 GB of 763903's `--info` (mostly
  Hr) was not needed. Sampled 8x2 / 10x2 was not possible cheaply: no trained states exist at those sizes.
* **Rule first**: `tests/P5.1_r05_gen/RULE_R51.11_vol3.md` written before any 7x2 number (card's T0 rule on three ladder
  volumes, plus a registered shape diagnostic: exponential at 0.180/site predicts 4.60, $V^{2.0}$ predicts 4.37).
* **Run**: `tests/P5.1_r05_gen/t0_weightvar_vol3_R51.11.py`, interactive, one thread, `ulimit -t 600`; 53.7 CPU-s, peak
  RSS 5.8 GB; log `tests/P5.1_r05_gen/logs/R51.11_vol3_weightvar.txt`. Built-in check reproduced round 7 (5x2 2.244,
  6x2 3.212).

| size | V | D | e^H | E_q[w²] q∝\|ψ\| | ESS frac | E[w²] guided |
|---|---|---|---|---|---|---|
| 5x2 | 10 | 44,052 | 3,053 | 2.244 | 0.446 | 1.048 |
| 6x2 | 12 | 627,264 | 10,999 | 3.212 | 0.311 | 1.070 |
| 7x2 | 14 | 9,017,409 | 68,945 | 4.448 | 0.225 | 1.080 |

* Fit over V = 10, 12, 14: slope of ln E_q[w²] 0.171 per site (factor 1.19) against e^H's 0.779 (factor 2.18).
  Local: 0.180 (10 → 12), 0.163 (12 → 14); log-log exponent 2.00, 2.11.
* **Verdict under the rule: Gate 4 FAIL** (slope clause, 0.171 > 0.1; the e^H clause does not fire). **Shape:
  SHAPE-UNRESOLVED** (4.448 between 4.37 and 4.60). Caveat: the per-representative convention of `kq.py` gives 4.387 at
  7x2 (it equals the configuration-level value at 6x2, and was 2.195 against 2.244 at 5x2), still inside
  the window, 0.4% above the POLY line; the registered statistic is the configuration-level one.
* **Next action per rule**: R51.11 stays LIVE with Gate 4 FAIL; its size ladder is not released until a fourth volume
  (sampled 8x2, E_p[|ψ|]/E_q[|ψ|]) exists. That needs a trained 8x2 state (sector $C(16,7)^2\approx1.3\times10^8$, too big for exact
  sums), i.e. a VMC training job plus a sampling pass; not prepared this round (orchestrator decides whether the fourth
  volume is worth it, given that three volumes leave 4.37 vs 4.60 only 5% apart, and 8x2 would separate
  exponential 6.4 from $V^2$ 5.8 by about 10%).
* Compute: about 1 CPU-min interactive (not a ledger job).

## Orchestrator decision (2026-11-18): R51.11 fourth-volume (8×2) test declined
The sampled 8×2 estimate would need a new 8×2 training job, and the V² and exponential predictions differ by only about 10% there (6.4 vs 5.8), within what a sampled estimate of $E_q[w^2]$ is likely to resolve; either outcome leaves the leaf's scaling question to the guided sampler (weights bounded by construction, $E_q[w^2]\le1.08$ at 10–14 sites), whose size dependence R51.8 measures directly. Decision: no 8×2 job now (predicted SHAPE-UNRESOLVED or a 10% call, p about 0.7, changes no decision); R51.11 stays LIVE with its size ladder held. Reopen if R51.8 shows guided SR's sample need growing faster than polynomially.
