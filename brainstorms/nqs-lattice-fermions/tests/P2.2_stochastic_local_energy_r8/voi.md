# VOI note (DRAFT, not submitted): P2.2 R7.4 (C3'), H8 kept-pattern compensator off psi0

- leaf: P2.2
- route: R7.4
- tier: T1
- decision: H8 linear chain, STO-3G, R = 3.0, Loewdin basis, 4900 determinants, exact psi0. The H6 T1 (this folder, RULE.md) showed that on exact psi0 the best kept-pattern compensator has zero excess whenever the kept graph is connected (component SDP; full 400x400 primal attains 2.3e-11 Ha at the SCREEN eps*, free entries <= 30 max|h|). So the exact-psi0 optimum is decided and is not rerun. The open question is robustness: a compensator built from a reference state psi_ref = psi0 (1 + delta xi) and evaluated on psi0 (and vice versa), at delta = 1e-2 (a realistic NNBF amplitude error). Metric: f_rob = kept fraction (CAN denominator, as in reopen_t1.py) at the largest grid eps with median excess <= tau = 0.2 mHa over 5 noise draws. KILL R7.4 if f_rob(8) >= 3 f_S*(8) = 3 x 0.144 = 0.433 (no better than the diagonal certificate's failing margin) -> R7.4 killed, leaf goes to the EXHAUSTED check (R5.2 remains). PASS if f_rob(8) < 0.433 and f_rob(8)/f_S*(8) <= f_rob(6)/f_S*(6) (ratio not growing) -> T2 design with a trained NNBF and a stale-reference compensator, including the O(N^2) cost of the compensator's diagonal. AMBIGUOUS otherwise -> H10 would be needed and is gated separately.
- prediction: KILL (f_rob(8) >= 0.433): at H6 the zero-excess optimum already costs 1.84e-4 Ha (0.92 tau) under 1% amplitude noise at f = 0.306, i.e. only just admissible at ratio 1.0, and both the number of dropped pairs and the compensator norm grow with N.
- p: 0.6
- prediction-changes-decision: yes (KILL removes the last certified route of F_C; PASS opens a T2 that does not exist yet)
- cheapest-kill-tried: T0 in this folder. Exact-psi0 optimum is zero (cannot kill). Explicit triangle gadget (gadget.py) is incomplete: 5924 of 20244 dropped pairs at eps = 2.24e-3 have no common kept neighbour, so a scalable complete construction (path or chordal-completion gadgets, Grone et al. 1984) must be written before the job; the full SCS primal does not scale to 4900 (eigendecomposition of a 4900x4900 matrix per iteration, thousands of iterations, estimated well above 1 CPU-h). If no complete sparse construction exists, the job is not run and R7.4 is reported AMBIGUOUS-blocked.
- cost-cpuh: 1.0
- cheapest-version: one size (H8), one instance, one noise level (1e-2), 5 draws; H6 is on disk; H10 only if AMBIGUOUS.
- convergence: no iterative training; ED of 4900 determinants is seconds; the construction is closed form.
- wall-time-basis: H6 component SDP + gadget took 8 CPU-s; H8 has ~12x the determinants and ~20x the pairs; with a sparse construction and 41 grid points x 5 draws, estimate < 0.3 CPU-h; --time=01:00:00, 1 CPU.
- power: deterministic given the noise seed; spread over the 5 draws at H6 was small (median reported). A single run suffices unless f_rob(8) lands within one grid step of 0.433.
