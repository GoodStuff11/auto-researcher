# RULE_snrcut — T1 of R51.26 (noise-level spectral filter of the SR step), 5×2 (written 2026-11-18, before any arm ran)

**Instance and foil.** R51.8's 5×2 instance and start (`starts/S5x2_N8_..._theta.npy`, $P$=1832), guided draws, seeds 9000 + n, metric length $\tau$=0.005, shift $10^{-3}$, 120 steps: the foil is R51.8's plain guided arms (same draws, same step family). The noise-free reference of the filtered step is R51.8's exact arm itself (with $N=\infty$ every mode has infinite SNR, so the filter is the identity): $E_{\rm win}^{\rm exact}=-6.548717127$, $G_{\rm ref}=0.0016011$ (score_r07.json; $E_0$ (+,+) $=-6.550697$, gap from start 0.003581).

**Driver.** `snr_run.py --mode snr` (identity check `--mode check` passed: $\rho_c=0$ and $\rho_c=10^{-9}$ reproduce ns_run's step to $3\times10^{-13}$, and the step-3 energy of R51.8's 4096 arm to $10^{-12}$). Per-mode SNR from $B=16$ position batches of the same draws; soft cut $1/(1+(\rho_c/\rho_k)^6)$.

**Arms.** Deciding: $\rho_c=1$ at $N$=4096 and $N$=16,384. Reported only: $\rho_c=2$ at $N$=4096.

**Statistic.** $s=(\overline{E}_{61\text{–}120}-E_{\rm win}^{\rm exact})/G_{\rm ref}$ (R51.8 rev. 2). Plain seed-9000 curve: $s$ = 0.9408, 0.4449, 0.1406, 0.0631, 0.0514 at $N$ = 1k, 4k, 16k, 65k, 262k. $N_{\rm eq}$ = plain $N$ with the same $s$ (ln $s$–ln $N$ interpolation; below 0.0514 censored as $>262$k); gain $g=N_{\rm eq}/N$.

**Verdict.** PASS if $g\ge2$ at both $N$; KILL if $g\le1.25$ at both; MARGINAL otherwise. Seed noise: plain seeds differ by 0.008 in $s$ at 16k (≈ a 1.1× change in $N_{\rm eq}$), so the 1.25 and 2 lines are outside it.

**Why it decides.** PASS earns a 6×2 T2 (does $g$ grow with size, i.e. does the filter remove the support-like part?); KILL ends the route at this size (no gain where it is cheapest to have one).

**Prediction.** MARGINAL 0.45, KILL 0.35, PASS 0.20 (the force noise dominates (L59), but a 4096-draw metric already has rank $\le$ unique draws, and the low-SNR modes may carry the noise-free path's late gain).

**Cost.** about 1 CPU-s per step: 3 arms × 120 steps ≈ 6–8 CPU-min, interactive, `ulimit -t 600` each.

## Addendum (2026-11-18, after the deciding arms were scored KILL, before the repair arm ran): translation repair
The KILL ($g$ 0.75, 0.76 at $\rho_c$=1; 0.44 at $\rho_c$=2) exposed the missing property: the noise-free step's gain is not concentrated in high-SNR modes (no source condition), so a cut discards gain. Repair by replacing the cut with the per-mode minimum-error linear shrinkage (Wiener / James–Stein form) $\rho_k^2/(1+\rho_k^2)$, which never discards a mode (`--wiener --rho 1`). One arm: guided $N$=4096, seeds 9000+n, 120 steps. Same statistic; repair PASS if $g\ge2$, KILL if $g\le1.25$, MARGINAL otherwise. Prediction KILL 0.75 (the trend from $\rho_c$=2 to 1 to 0 is monotone toward plain SR).
