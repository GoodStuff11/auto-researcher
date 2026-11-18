# P5.1 round 17 (2026-11-18): worker response to per-leaf critic round 16
Compute: about 22 CPU-min interactive (5×2 exact rerun 8.9, 6×2 two-state check ≈6, MinSR τ 0.0025 4.0, snrP 2.8). No sbatch, no commits. Not opened: `results/r14_decay/`, any output of jobs 773500/773528.
Also: trimmed R51.15, R51.10 (critic Checked-by overflow) and L61 to cap (wording only); repaired escape-mangled LaTeX in L76/L77 (`\t`, `\r` had become tab/newline).

## A. Major 4 verified independently (and partly contradicted)
* Drift over steps 61–120, linear fit, in units of each size's $G_{\rm ref}$ (run.json, score_r07/r08): 5×2 slope −1.22e-6/step → 4.57% of G; 6×2 −5.02e-6/step → 4.29%. Matches the critic (4.6%, 4.3%).
* Both exact arms bounce with period 2 over 61–120: energy rises on exactly 30 of 60 steps at each size; mean |ΔE|/G per step 0.153 (5×2), 0.048 (6×2). So 6×2 overshoots too; it is not "descending without overshoot".
* First-order gain of one fixed step: deterministic rerun of R51.8's 5×2 exact arm (`xsx_rerun.py`, energies identical to the stored run, max diff 0) logging $\sqrt{x^{\mathsf T}Sx}$ and $f\cdot d$ per step. $2\tau\sqrt{x^{\mathsf T}Sx}/G$ over steps 61–120: mean 0.301, median 0.297, range 0.009–0.619 (exact first-order $-2f\cdot d/G$: mean 0.317). Step-120 end state: 0.380 (matches critic).
* 6×2 (`six.py`, ~2.9 CPU-min per exact step): step-120 end state 0.0220 (matches critic's 0.022), and one exact step later 0.116 (energy rose 3.4e-4 = 0.049 G on that step). The end state is the low phase of the bounce.
* **Difference from the critic**: the 17× ratio (0.38 vs 0.022) compares a high-phase 5×2 state with a low-phase 6×2 state. With the 5×2 window mean 0.30 and the two 6×2 states, the ratio is 3–14× (≈4.4× against the 6×2 two-state mean 0.069). And both sizes overshoot (period-2 bounce); the difference is degree, not kind. The critic's conclusion (×3.79 mixes size with step regime; CONT is "later", not stage-matched) stands; "overshoots on the smaller cluster but not the larger" does not. Files: `tests/P5.1_size_scaling_r04/results/r17_gain/`.
* Applied: R51.29 card rewritten (stage claim withdrawn; regime numbers; CONT rule removed: CONT bears on the step-length confound only with the DECAY arms); L78 rewritten and index line updated. The summary's Answer/What we showed contain "overshoots on the 10-site cluster and not on the 12-site one" / "2% at 12, so it overshoots only on the smaller cluster": not edited (orchestrator-owned); flagged.

## B. Major 1: MinSR step lengths
* RULE_minsr.md addendum written before the arm (τ/N premise withdrawn; critic arms recorded; prediction R 0.08–0.20, GUIDED-AHEAD 0.9). Critic outputs copied to `tests/P5.1_r16_gen/results/critic_tau/` (run.json ×2, t1.log, t2.log).
* τ 0.0025 arm (Born, N=1024, seeds 9000+n, 800 steps, 4.0 CPU-min): trailing-20 R 0.175 (0.127 at step 600; still rising; R max 0.189). All four (`score_r17_minsr_tau.json`): 0.0003125 → 0.046, 0.000625 → 0.081, 0.00125 → 0.124, 0.0025 → 0.175; 0.005 diverged (R51.8). Monotone in τ up to 0.0025; 0.0025–0.005 untried. Verdict GUIDED-AHEAD at every τ (best: 0.691 − 0.175 = 0.52); behind Born SR at N≥P (0.383) by 0.21. Prediction met.
* Card R51.31 and L79 updated (all arms; τ/N sentence removed; best of four; learning speed at one budget, before target).

## C. Major 5: R51.26 mechanism test
* RULE_snrcut.md addendum 2 written before the run: `snr_plainnorm.py`, cut ρ_c=1 at 4096 with the plain step's normalisation τ/√(xᵀS_N x) (x unfiltered), so no kept mode is lengthened. g ≥ 0.95 refutes the low-SNR mechanism (LIVE only if g > 1.05); g < 0.95 keeps it. Prediction: stands 0.6.
* Result (2.8 CPU-min, `score_r17_snrP.json`): s 0.503, N_eq 3265, **g 0.80** → mechanism stands. The unlengthened step is 0.84 of τ (10–90%: 0.79–0.86), median 325 modes kept. Residual caveat: a 16% shorter step, of the order that moved R51.1 by 14%.
* Applied: R51.26 Evidence/Why/Strongest case/Checked by/Lesson; Analogy Map B5 "Tried" line in problem.md; L76 title, lesson, evidence and index.

## D. Major 2 and Minors 7–9
* R51.31 T0 and L79: cutoff MinSR = truncated-SVD SR by SVD; MinSR equals SR with the same regulariser at every N; only cost differs. Summary's What would close it already says so (orchestrator).
* R51.27 (Minor 7): Why now rests on (ii) (`che` untileable; P4.2's transferred start worse); (i) holds only on exact clusters; at 24 sites the transient is paid in samples.
* R51.30 (Minor 8): idea text sign fixed (S_N over-estimates along the step); lenratio states are from R51.18's STORM trajectory, not plain SR.
* R51.28 (Minor 9): L59 cited as one start, one step; untested at R51.8's late stage.
* Minor 6 (commit RULE files): not done (no commits this round, per brief); orchestrator to commit `tests/P5.1_r16_gen/`.

## Lessons proposed
* L78 extension (applied): read the first-order gain per step over a window, not one state; within a period-2 bounce it alternates ×5 or more.
