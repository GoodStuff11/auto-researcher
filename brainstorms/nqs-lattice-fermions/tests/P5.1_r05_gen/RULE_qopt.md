# P5.1 round 5, R51.11 `qopt` T1 (pre-registered, written before `kq.py` ran)

Question. How much of the sampled SR step's first-order error (kappa, R51.10's statistic) can a better sampling
distribution remove beyond guided draws, (i) within the amplitude-only class (costs what guided costs), (ii) at all
(oracle q* ~ p sqrt(h), needing rows and local energies per proposal)?

Instances: S10 (5x2, N_e 8) and S12 (4x3, N_e 10) SR-trained starts (R51.8 / R51.1), exact sums.
Statistics: r_amp = kappa_guided / min(kappa_amp_only_10bins, kappa_amp_only_40bins); r_orc = kappa_guided / kappa_oracle.

* KILL (whole route) if r_orc < 2 at both sizes.
* KILL amplitude-only arm if r_amp < 2 at both sizes (guided is already near the best cheap sampler); the route
  then survives only as an oracle-headroom statement whose sharpest gap is the per-proposal cost.
* PASS amplitude-only arm if r_amp >= 3 at both sizes and r_amp does not fall from S10 to S12 → prepare a multi-step
  T2 (the learned bin weights, fixed tau, 4x3) against guided arm A/B.
* MARGINAL otherwise.
Caveat recorded in advance: kappa is a second moment and over-states heavy-tailed errors (R51.10 Sharpest Gap); a
kappa gain is necessary, not sufficient, for a multi-step gain. The 40-bin optimum is fitted with the exact h, an
upper bound on any amplitude-only rule learned from samples.
