# VOI note (quick): P5.1 S12' start + exact kappa (`submit_quick_s12p.sbatch`; RULE.md rev. 1, RULE_kappa.md rev. 1)

* leaf: P5.1
* route: R51.10
* tier: T1
* decision: the S12' (6x2) start is required by every primary R51.8 arm at 12 sites (no S12' ladder can be prepared or costed without it); its exact kappa gives k1 = kappa(S12')/kappa(S10) for R51.10's recast trend and, against kappa(4x3) = 1.24e5, the geometry flag that decides whether the 4x3 exact arm is ever bought (ratio outside [1/1.5, 1.5] -> geometry matters -> the 4x3 reference becomes a candidate; inside -> it is dropped). k1 <= 1.728 predicts POLY for R51.8's first step; k1 >= gH1 predicts SUPPORT.
* prediction: k1 about 1.2 (KAPPA-POLY side for the first step) and no geometry flag (kappa(6x2) within 1.5x of 1.24e5).
* p: 0.6
* prediction-changes-decision: yes. The geometry flag decides whether the 8 CPU-h 4x3 exact arm is ever run, and the start is a prerequisite for unit 1c.
