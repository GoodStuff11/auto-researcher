# RULE_minsr — T1 of R51.31 (Born MinSR at N < P against guided SR at equal draws), 5×2 (written 2026-11-18, before the arm ran)

**T0 identity (checked).** MinSR with a Tikhonov shift, $x=A^{\mathsf T}(AA^{\mathsf T}+\lambda)^{-1}b$, equals SR with the same shift, $(A^{\mathsf T}A+\lambda)^{-1}A^{\mathsf T}b$, at every $N$ (push-through); `snr_run.py --mode check` reproduces ns_run's step to $3\times10^{-13}$. So "MinSR at $N<P$" in its shift form (NetKet's SRt, Rende et al. arXiv:2310.05715) is SR at $N<P$ solved in sample space; only the pseudo-inverse-cutoff variant of Chen & Heyl differs (not tested here).

**Arm.** Born draws, $N$=1024 ($<P$=1832), $\tau$=0.00125 (one quarter of R51.8's 0.005: R51.8's Born 1024 arm at 0.005 rose above its start, $s$=1.09, so the practitioner's choice at small $N$ is a shorter step; shorter than R51.8's, so no L62 overshoot), seeds 9000 + n, 800 steps = 819,200 draws.

**Statistic.** $R_{20}$ = trailing 20-step mean of $(E_{\rm start}-E)/G_{\rm ref}$ ($G_{\rm ref}$=0.0016011) at 819,200 draws. References at the same draws (step 50 of R51.8's $N$=16,384 arms): guided 0.691, Born SR 0.383. Also reported: first sustained crossing of 0.9 (guided 16,384 crosses at 1.97M draws).

**Verdict.** GUIDED-AHEAD if $R_{20}^{\rm guided}-R_{20}^{\rm MinSR}\ge0.10$; MINSR-AHEAD if $R_{20}^{\rm MinSR}-R_{20}^{\rm guided}\ge0.10$; TIE otherwise. One $\tau$, one seed: GUIDED-AHEAD says only that this practitioner setting of Born MinSR at $N<P$ is behind guided SR at equal draws at 10 sites (L33: not polished arms; an equal-draw comparison at 0.82M draws).

**Prediction.** GUIDED-AHEAD 0.8 (Born's $\kappa$ is $1.9\times10^3$ times guided's at this start; linear scaling of the noise ball, $\propto\tau/N$, makes the $N$/$\tau$ split matter little), TIE 0.15, MINSR-AHEAD 0.05.

**Cost.** about 0.5 CPU-s per step, 800 steps ≈ 6–8 CPU-min, interactive, `ulimit -t 600`.

**Addendum (2026-11-18, round 17, before the τ 0.0025 arm ran).** The Prediction's "noise ball ∝ τ/N, so the N/τ split matters little" is withdrawn: the round-16 critic's arms (same start, Born, N=1024, seeds 9000+n, 800 steps; `results/critic_tau/`) give trailing-20 R 0.046 at τ 0.0003125 and 0.081 at τ 0.000625, against 0.124 at 0.00125, so τ/N scaling fails at N<P.
One further arm, τ 0.0025 (same start, Born, N=1024, 800 steps, seeds 9000+n), same statistic and verdict lines; the card reports the best of all step lengths tried (0.005 diverged in R51.8). Prediction: R between 0.08 and 0.20, GUIDED-AHEAD 0.9.
