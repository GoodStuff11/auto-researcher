# Round 8 — per-leaf critic re-check (round 7 report) applied, 2026-11-17

Editor pass; no commits, no Slurm jobs. Running jobs untouched: 769376, 769213, 769211, 763903. The critic's report is at `/home/jek354/.claude/jobs/91298e5c/tmp/critics/p51_r7.md` (NOT SIGNED on Majors 1–3); its substance is restated item by item below. The critic independently confirmed $N_{\rm needed}(5\times2)$ = 31,284, the 6×2 thresholds 54,052 / 110,106, and RESOLVABLE both ways (3.69 raw, 15.4 detrended sd_w).

## 1. (Major) R51.18: kill stands, stated reason did not — length-ratio check run
Critic: energy rose on 82 of 120 steps (not every step from step 2); median $|\Delta E|$ 2.3e-4 against 1.5–4.5e-5 for plain-SR identity steps; the sample linear term pointed downhill on 69% of steps; hypothesis: with about 900 unique draws $<P$=1832, $v^{\mathsf T}S_nv$ may understate the true length. Check asked: $r=v^{\mathsf T}S_{\rm exact}v/v^{\mathsf T}S_nv$ on the 40-step diagnostic trajectory.

Check (interactive, OMP/OPENBLAS/MKL_NUM_THREADS=1, `ulimit -t 600`; `tests/P5.1_r05_gen/lenratio_R51.18.py`, log `logs/storm_sf_v2_lenratio40.txt`, output `results/storm_N2048_a0.1_sf_v2/lenratio.json`; 2.1 CPU-min). The script replays `storm_v2.py --mode sf` step for step and asserts every energy matches `run_diag.json` to 1e-10 (it did, all 40). $S_{\rm exact}$ = exact metric over all 11,100 orbit representatives with weights $|\psi|^2$ at the current parameters; only $v^{\mathsf T}S_{\rm exact}v=\sum_x p_x(O_c v)_x^2$ is formed. Sign of the exact gradient checked by central finite difference at the start: $dE/d\theta\cdot\hat f$ = 0.0207753 = 2 × 0.0103876 ($=2f_{\rm exact}\cdot\hat f$).
* 82 of 120 steps rose (recount from `run.json`, confirmed); median $|\Delta E|$ 2.26e-4. Steps 1–40: 36 rose.
* $r(v)$: median 1.135 (steps 2–40: 1.140), min 0.342 (step 2), max 1.527 (step 20). Steps 1–5: 0.511, 0.342, 0.711, 0.769, 1.08. Exact metric length $\tau\sqrt r$ therefore lies in $[0.59\tau, 1.24\tau]$.
* Reference, fresh plain direction $\hat x_n=(S_n+\lambda)^{-1}f_n$ on the same draws: $r$ median 0.669 (0.340–0.823).
* Exact first-order change $2f_{\rm exact}\cdot d$: downhill on 24 of 40 steps (60%); of those 24, 20 still rose. Median second-order part ($\Delta E$ minus first-order) 3.41e-4.
* Reading: the critic's length hypothesis is not supported — the steps were not oversized in the exact metric (at most 1.24τ, median 1.07τ). The failure is the direction: through the indefinite $S_{\rm rec}$ the solve loads $v$ onto directions of steep upward energy curvature per unit metric length.

Applied: R51.18 card rewritten (Evidence: "rose on 82 of 120 steps", R defined as the fraction of the gap to the exact ground energy $E_0=-6.5507$, the ratio numbers; Why this kills it; Strongest case: length understatement answered, PSD projection and $a$=0.5 unresolved; scope "this length rule"; Lesson → L70). kill_index row and L70 (Evidence, lesson sentence, scope, status, index line) narrowed to "this length rule", R definition fixed. Summary Conclusion and Alternatives updated.

## 2. (Major) Comparator statistic
RULE.md's registered comparator is first crossing of $E\le E_{\rm start}-0.9G_{\rm ref}$. Recomputed from the run files: guided 16,384 × 105 = 1.72M (seed0 9000), 16,384 × 107 = 1.75M (seed0 19000); Born 262,144 × 62 = 16.25M (one seed) → 9.4× and 9.3× (same with rev-1 or rev-2 $G_{\rm ref}$). The 4.7–12× figure is RULE_cmp's sustained statistic. Applied: What we showed, Conclusion, Strongest objection (1), R51.8 Outcome and Rule; `voi_r06_s12p_ladder.md` NOT edited (approved digest), correction stated in RULE.md's addendum, which also registers first crossing as primary and the sustained statistic as reported only.

## 3. (Major) R51.11 Gate 4 FAIL yet LIVE
Applied: FAIL kept as registered; "Why not killed" line (V² fits as well as an exponential; $\kappa_{\rm guided}/\kappa_{|\psi|}$ 4.76 at 10 sites, 4.71 at 12; R51.4 died on an $e^{-cV}$ argument); Q-FEWER's next action gated on a third-volume test (exact 7×2 or sampled 8×2 or 10×2) on the card and in the portfolio; T0 script cited (`tests/P5.1_r05_gen/t0_weightvar_R51.11.py`).

## Minors
4. Censored Born counts as $\ge$ (largest Born rung's $N$ × 120) — in RULE.md's addendum, with what it can and cannot decide.
5. problem.md Analogy Map R51.18 now killed; R51.16 idea no longer says no arm measured Born SR (R51.8's 5×2 Born arms did); R51.8 idea says MinSR half untested; RULE.md addendum says no arm runs MinSR.
6. round-07-scoring.md wording corrected: the rev-2 addendum was appended before that scoring, while 769098 ran.
7. e^H convention named: 781 = over orbit representatives (RULE.md), 3,053 = per configuration (T0 log), ratio about 3.9; RULE.md addendum and a note in `round-07-R51.11-T0.md`.

RULE.md: appended "Addendum (per-leaf critic re-check, 2026-11-17, before job 769376 is scored)" covering Major 2, Minors 4, 5, 7. No voi_*.md edited.

Checked-by lines set per the critic on R51.18, R51.11, R51.8, R51.16, R51.7.
