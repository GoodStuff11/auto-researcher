# P5.1 round 5, R51.11 T1b: does a kappa-better amplitude-only sampler train better? (pre-registered)
Written 2026-11-17 after RULE_qopt.md scored MARGINAL (r_amp 48.6 at S10, 23.0 at 4x3: >= 3 at both but falling),
before these arms ran.

The best amplitude-only q needs the exact h; the realisable members of the class are mixtures (1 - b) p + b p_flat.
kappa (exact, starts): q ~ |psi| (b = 1) is 4.7x below guided at both sizes (S10 22,519 vs 106,804; 4x3 26,270 vs
124,296); b = 0.5 is 2.4x below.
Arms (`storm.py --a 1.0`, i.e. plain fixed-step SR, identity-checked against ns_run.py): S10 start, N = 4096,
tau 0.005, 120 steps, seeds 9000 + n, sampler b = 1.0 and b = 0.5. Foils: stored guided (b = 0.2) N = 4096 and 16384
arms (R_plat 0.2740 and 0.4026, G = 0.1286).
* KILL (amplitude-only arm of R51.11) if R_plat(b = 1.0) <= 0.2740 and R_plat(b = 0.5) <= 0.2740.
* PASS if max over b of R_plat >= 0.2740 + 0.5 G = 0.3383 → prepare a 4x3 multi-step T2 against R51.1's A/B.
* MARGINAL otherwise.
The kill also tests the kappa → multi-step link: a 4.7x kappa gain with no training gain would mean kappa's
tail-dominated second moment does not predict training (relevant to R51.10).
