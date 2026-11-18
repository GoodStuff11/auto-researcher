# VOI note: P5.1 S14 start + exact kappa (`submit_kappa_s14.sbatch`; RULE_kappa.md rev. 1, RULE.md rev. 1)

* leaf: P5.1
* route: R51.10
* tier: T1
* decision: kappa(S14)/kappa(S10) = k2 completes R51.10's three-size trend. KAPPA-POLY (k1 <= 1.728, k2 <= 2.744) -> the S14 ladder of R51.8 is costed and released after S10/S12' ladders agree, with a recorded prediction; KAPPA-SUPPORT (k2 >= gH2, the support growth from 781 at S10) -> the leaf's next generation targets force variance reduction before any S14 ladder, and R51.8's S14 ladder is predicted SUPPORT; KAPPA-AMBIGUOUS -> S14 ladder needed without a prediction. The S14 start is also a prerequisite of any S14 ladder.
* prediction: KAPPA-POLY (k2 about 1.3-2), from kappa 1.08e5 (S10) and 1.24e5 (4x3 S12) at nearly constant d_eff/P.
* p: 0.55
* prediction-changes-decision: yes. KAPPA-SUPPORT would redirect the leaf before the most expensive exact-sum ladder (S14, tens of CPU-h) is bought; KAPPA-POLY gives that ladder a recorded prediction.
* cheapest-kill-tried: (i) the T0 noise model (R51.1: shortfall ~ N^-0.87; stationary-ball model off 2.7x) makes the need a direction-fidelity question, which ridge theory reduces to kappa; (ii) kappa at the two existing starts (0.13 and 2.7 CPU-min interactive) grows 1.15x; (iii) S14 cannot be done interactively: one exact pass is 23 CPU-min and the sector needs 16.3 GB (this session's 20 GB limit was hit when a second process ran), so it is a gated job; a sampled kappa at S14 would not be exact and would need the very heavy-tail estimate it is meant to validate.
* cost-cpuh: 3.25 (1 task x 3.25 h x 1 CPU; expected use 2.5)
* cheapest-version: one start (seed 0, the protocol of every other start) and one state; the start is needed by any later S14 work anyway; kappa's second pass recomputes rows instead of storing 2.25e6 x 3590 doubles (65 GB). No exact arm or ladder at S14 in this job.
* convergence: training is 1000 fixed iterations (as every start); kappa is a finite exact sum. The chunked two-pass kappa reproduces the one-pass value at S10 to 1e-11 (kappa_born 204426537.6348 vs .6353, guided 108115.48360 vs .48360).
* wall-time-basis: interactive probes: training 1.7 s/iteration at 7x2 (20 iterations, 48 CPU-s) -> 29 min; S14 --info 59 CPU-s with one energy 1.24 CPU-min, rows 1.02e-5 CPU-min each (23.0 min per pass); kappa 51 min; sum 82 min x 1.8 in-job factor x 1.3 = 192 min -> --time 03:15:00.
* power: kappa is exact for the state; the uncertainty is the start (one seed). P(KAPPA-AMBIGUOUS) about 0.3 from the 1.73-to-gH gap, not from noise.
