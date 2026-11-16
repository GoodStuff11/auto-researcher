#!/usr/bin/env python3
"""
P2.1 Round 11b: re-count of the Round 11 operation model for v_h < 1 with the X-table term as implemented.
No options. Usage: OMP_NUM_THREADS=1 python3 ops_recount.py > ops_recount_output.txt (single thread, < 1 min).

The Round 11 model (../P2.1_subcubic_amplitudes_r11/crossover_model.py, imported read-only) counts the X
update as V * V_h * k. In accept_update.HFPSAcc the X table is V x V (physical up row x physical down
column) for any V_h, and its stage-1 update is -Gr[:, R] (Z allcols) + Gr[:, R] W, i.e. 2 V^2 k; the
stage-2 term is 2 V^2. This script prints R_ops(dense) by L and v_h for both counts (b scaled by v_h,
as in Round 11; the sublattice b is measured in sub_hfps runs, not here).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r11"))
import crossover_model as CM  # noqa: E402  (read-only)

orig = CM.route_ops


def route_ops_impl(m, b, N, V, Vh):
    r, a = orig(m, b, N, V, Vh)
    up = any(s == 0 for (s, _, _) in m)
    k = b + (1 if up else 0)
    dn = any(s == 1 for (s, _, _) in m)
    a += -V * Vh * k + 2 * V * V * k + ((2 * V * V - 2 * V * Vh) if dn else 0)
    return r, a


LS = [8, 12, 16, 24]
for label, fn in (("Round 11 count (X as V V_h k)", orig), ("implemented X (2 V^2 k)", route_ops_impl)):
    CM.route_ops = fn
    for vh in (1.0, 0.5, 0.25):
        rng = np.random.default_rng(11)
        rr = [CM.rops(CM.sweep_ops(L, "plus", vh, rng), "dense") for L in LS]
        print(f"{label:32s} v_h={vh:4.2f}: " + " ".join(f"L{L}:{x:.2f}" for L, x in zip(LS, rr)), flush=True)
