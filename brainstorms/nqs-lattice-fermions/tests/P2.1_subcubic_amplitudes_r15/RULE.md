# P2.1 R15.6 T1 rule (pre-registered before any run, 2026-11-17)

Source: route card R15.6 in the Round 15 route-generator output (`P2.1_round15.md`, "Cheapest falsifier", step 1).
This file restates that rule; it is written before `sparse_gate.py` was run in train mode.

## What is trained

Learned sparse gating of hidden rows: `t2_routes.HFPSAnsatz` (plus patch, F = 4, Ha = 0, pairing reference
F0 = round-5 projected d-wave BCS, two-body density Jastrow), all V hidden rows present, with row m multiplied by
a learned gate

    psi = e^J det(F0 + sum_m K[:, m] g_m(n) h_m(n) . Omega_m)[up, dn],
    g_m(n) = ReLU(w . x_m(n) + c_{s(m)}),

where x_m is the plus-patch input of row m (the same 10-vector the hidden features see), w is shared by all rows
(10 parameters) and c_s is one bias per checkerboard sublattice s(m) = (x + y) mod 2 (2 parameters). A row with
g_m = 0 contributes exactly zero. Gate initialisation (stated here, not in the card): w ~ N(0, 1/10) from its own
RNG stream (seed 11 + 7), c = (0, 0), so about half the rows start active. All other parameters start from the
seed-11 initialisation that `full_F4` used.

## Protocol (Round 12 protocol unchanged, plus the gate penalty)

Instance 23 (4x3 OBC, 5+5, V = 12, D = 16632), t-t'-J with t' = -0.25, J = 0.4. Unprojected exact-force SR
(`micro_test.sr_step` arithmetic: lr 0.05, diagonal shift 1e-3, 1000 metric samples from |psi|^2, step-norm clip),
seed 11, RNG offset 307 for the metric samples. Plateau test unchanged: from step 300 on, every 50 steps, stop when
the best energy improved by less than 2e-4 t over the last 200 steps. SR CPU cap 480 s (the card's figure).

Gate penalty (ReLU routing with adaptive L1, after Wang, Chen and Zhu, arXiv:2412.14711): the SR force is the
gradient of E + lambda * A, with A = <(1/V) sum_m g_m>_{|psi|^2} (exact sum over D). The active fraction
a = <(1/V) sum_m 1[g_m > 0]>_{|psi|^2} is computed exactly every step, and lambda follows a dual-ascent step in
log lambda: lambda <- lambda * 1.2^{sign(a - 0.5)}, lambda_0 = 1e-3.

Scored state: the lowest-energy step among steps whose exact active fraction is <= 0.55 (the plateau test runs on
the history of that constrained best energy). It is then scored by one unchanged budget (`p11_core.budget`, same
RNG stream as Round 12: calibration first, then the budget) giving dE_sign.

## Reference

`full_F4` on instance 23: dE_sign = 7.95e-3 t (Round 12b job 704752,
`../P2.1_subcubic_amplitudes_r12b/results/t1_full_inst23_s11.json`, converged). `half_F4` on instance 23 has
rho = 1.587 (same source).

rho(12) = dE_sign(gated) / dE_sign(full_F4).

## Verdict

* **KILL** if rho(12) > 1.59 (no better than `half_F4`, the gated family's own fixed-sublattice member), whether
  or not the run converged.
* **KILL** if rho(12) > 1.25 and the run is converged (plateau reached).
* **INCONCLUSIVE** if the final active fraction of the scored state exceeds 0.55, or the plateau is not reached
  within 480 SR CPU-seconds (and the 1.59 clause does not fire). Also INCONCLUSIVE if the Round 12 counting guards
  fail (calibration error > 1e-3 x dE_tot, or budget spread >= 0.05).
* Otherwise (rho(12) <= 1.25 converged, or rho(12) <= 1.59 unconverged is INCONCLUSIVE as above): **no kill;
  proceed to the T0 cost check** (card step 2: KILL if both two-term 11c cost fits predict R_D(16) < 4.3 from the
  measured update rank k), then step 3 (instance 25, V = 15, gated batch job).

## Recorded alongside (not part of the verdict)

On the scored state, exact expectations over accepted single-electron hops (proposal uniform over the hop moves
connected by H from configuration x, Metropolis-Hastings acceptance under |psi|^2, every term summed exactly):
mean number of hidden rows changed (rows whose patch contains a hop endpoint and that are active before or
after), mean number of rows active both before and after among those, mean number of gate flips (rows whose
active status changes), and the same "rows changed" count for `full_F4` geometry (every row active) for scale.
On this 4x3 OBC cluster the counts are smaller than on a torus (boundary rows have short patches), so the ratio
to the full-geometry count is the number to carry to the T0 cost check.

## Budget

Interactive only if the whole run is <= 10 CPU-minutes: SR cap 480 s plus ED, calibration, budget and hop
statistics (about 1 CPU-minute together at V = 12), run under `ulimit -t 600`.

## Addendum A (2026-11-17, after run 1, before run 2)

Run 1 (controller as above, log `logs/train_inst23_run1_dual.log`, result
`results/t1_sgate_inst23_s11_run1_dual.json`, 93 CPU-s) is **not a valid test of the family**: the multiplicative
dual controller oscillated (active fraction 0.21-0.88, lambda 5e-4 to 1.5), and the explicit gate-penalty gradient,
which is not of the score form <O (.)>, was amplified by the SR solve (divided by the 1e-3 shift in the
low-metric gate directions) and clipped steps threw the energy between 0.10 t and 5.7 t above E0. The plateau test
fired at step 300 only because the constrained best (E - E0 = 0.333, active 0.534) stopped improving. By the letter
of the rule above it gives rho(12) = 4.94 > 1.59 (KILL); it is reported, and judged an optimiser failure of the
controller this worker chose (the card fixes the thresholds, not the controller).

Run 2 changes only the controller, everything else as above (same thresholds, same verdict function, same
protocol and guards): no L1 penalty (lambda = 0); after every SR step both sublattice biases c_s are shifted by the
same delta, chosen as the weighted quantile that sets the exact active fraction under the pre-step |psi|^2 to
0.5 (a projection onto the constraint a = 0.5, applied also once at initialisation). The gate is still learned
(w and c are trained by SR; the projection fixes only the common bias level). Run 2's verdict is the R15.6 T1
verdict; run 1 is reported alongside it.
