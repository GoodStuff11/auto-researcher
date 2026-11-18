# P5.1 R51.11 `qopt`: Gate 4 third-volume weight test (pre-registered)
Written 2026-11-18, before any 7x2 weight number was computed. Extends `log/round-07-R51.11-T0.md` (two volumes).

**Question.** Does the second moment of the reweighting weights of q ∝ |ψ| grow exponentially with size (Gate 4 FAIL
stands on three volumes), or does it stay bounded or polynomial?

**Cheapest decisive version (chosen).** Exact sums at 7x2, N_e 12 (V = 14), the third rung of the 5x2 → 6x2 ladder
(same geometry, so no 4x3-vs-6x2 geometry wobble). No new state is trained: R51.10's 7x2 start
(`P5.1_size_scaling_r04/starts/S7x2_N12_U8_tp-0.25_pg11_che_seed0_theta.npy`, 1000 VMC iterations, job 763903) is used.
At 10 and 12 sites start, trained and ground states agreed to 1%, so the value is set by size, not state; the 7x2
ground state (eigsh on 9.0M configurations) is not computed. Amplitudes only (no Hamiltonian is built), so memory is the
sector plus symmetry maps (a few GB) and CPU is about one exact amplitude pass (≲ 2 CPU-min; 763903's probe: one exact
energy 1.24 CPU-min including the matvec). Sampled 8x2 / 10x2 was not chosen: no trained states exist at those sizes, and
the estimate E_p[|ψ|]/E_q[|ψ|] would carry sampling error that the exact sum does not.

**Statistic.** E_q[w²] = Z₁ Σ|ψ|³ / Z₂² over the full (+,+) sector, configuration-level weights (each orbit member
counted; the `stats` function of `t0_weightvar_R51.11.py`, unchanged), and e^H of p in the same convention. Ladder values
already measured: E(10) = 2.244, E(12) = 3.212 (start states, round 7).

**Rule (the card's Gate 4 T0 rule, applied to three ladder volumes).** s = least-squares slope of ln E_q[w²] against V
over V = 10, 12, 14 (5x2, 6x2, 7x2 start states); per-site factor e^s, compared with e^H's least-squares per-site factor
over the same three states.
* **FAIL** (Gate 4 FAIL confirmed on three volumes) if s > 0.1 per site, or if e^s exceeds e^H's per-site factor.
* **PASS** if s ≤ 0.1 per site and e^s is below e^H's per-site factor.
* No UNCLEAR branch remains: the two clauses cover every case.

**Shape diagnostic (registered, not the verdict).** The 0.1 per-site line cannot separate V² from an exponential at
V ≈ 12 (V² has local slope 2/V ≈ 0.15 per site), so the shape is read separately from E(14):
* an exponential at the measured 10 → 12 rate (0.180 per site) predicts E(14) = 4.60; E(14) ≥ 4.60 → EXP-OR-FASTER;
* V^2.0 (the 10 → 12 log-log exponent) predicts E(14) = 4.37; E(14) ≤ 4.37 → POLY (no faster than V²);
* between → SHAPE-UNRESOLVED (a 5% window; 4x3 and 6x2 differ by 3.5% at equal V, which sets the resolution).

**Next action on each outcome.** PASS → R51.11's size ladder (R51.8 protocol) is released. FAIL with POLY → the card's
"why not killed" stands (reweighting cost grows polynomially, not as a constant factor); the size ladder may proceed with
the Gate 4 claim restated as polynomial. FAIL with EXP-OR-FASTER → R51.11 is KILLED at G4 (diagnostic only, as the
portfolio row says). FAIL with SHAPE-UNRESOLVED → stays LIVE with the Gate 4 FAIL standing; the size ladder is not
released until a fourth volume (sampled 8x2) is built.

## Hold released (orchestrator, 2026-11-18, before any ladder data)
The size-ladder hold above is released by the reopen trigger recorded in leaves/P5.1/log/round-10.md ("reopen if R51.8 shows guided SR's sample need growing faster than polynomially"), which R51.8's 6x2 verdict (SUPPORT-LIKE, r' 3.79) meets. The ladder runs under RULE_r14_qladder.md.
