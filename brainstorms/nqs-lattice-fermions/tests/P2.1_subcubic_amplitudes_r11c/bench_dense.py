#!/usr/bin/env python3
"""
P2.1 Round 11c fairness check of the dense baseline (no options; single thread).
For L = 16 and 24 (doping 1/8): time per compiled dense ratio (delayed.d_ratio: F N^2 gather + LAPACK LU)
and of dense2 (carried gather) against a bare LAPACK LU of the same N x N matrix (numpy slogdet in a loop), and the implied GFLOP/s of
the LU (2/3 N^3). If the per-ratio time is close to the bare LU time, the compiled dense arm is LU-bound
and has no avoidable overhead. Also prints the route arms' time per sweep split into proposals+updates
and local energy at L = 16 (do_eloc False / True), to locate their cost.
"""
import time

import numpy as np

import delayed as DL
import t1_r7 as T1

for L in (16, 24):
    lat = DL.lattice(L, L, "pbc")
    n = T1.filling(L * L)
    d = DL.make("dense", lat[0], L, L, "pbc", n, 0, 11 + L)
    d.setup(*T1.rand_cfg(L * L, n, d.rng))
    DL.seed_nb(3)
    DL.do_sweep(d, lat, 4 * L * L, False)
    mv = []
    while len(mv) < 200:
        m = DL.propose(d.ou, d.od, lat[3], lat[4], lat[5])
        if m[0] != -2:
            mv.append(m)
    DL.ratio_c(d, mv[0])
    t0 = time.perf_counter()
    for m in mv:
        DL.ratio_c(d, m)
    tr = (time.perf_counter() - t0) / len(mv)
    A = d.A.copy()
    np.linalg.slogdet(A)
    t0 = time.perf_counter()
    for _ in range(200):
        np.linalg.slogdet(A)
    tl = (time.perf_counter() - t0) / 200
    d2 = DL.make("dense2", lat[0], L, L, "pbc", n, 0, 11 + L)
    d2.setup(d.ou, d.od, d.ru, d.rd)
    DL.ratio_c(d2, mv[0])
    t0 = time.perf_counter()
    for m in mv:
        DL.ratio_c(d2, m)
    tr2 = (time.perf_counter() - t0) / len(mv)
    print(f"L={L} N={n}: dense2 (carried gather) ratio {tr2 * 1e6:.0f} us = {tr2 / tl:.2f} x bare LU")
    print(f"L={L} N={n}: compiled dense ratio {tr * 1e6:.0f} us; bare LU (numpy slogdet) {tl * 1e6:.0f} us "
          f"({2 / 3 * n ** 3 / tl / 1e9:.1f} GFLOP/s); ratio/LU = {tr / tl:.2f}", flush=True)

L = 16
lat = DL.lattice(L, L, "pbc")
n = T1.filling(L * L)
for arm, rmax in (("full", 32), ("half", 16), ("quarter", 16), ("dense", 0), ("dense2", 0)):
    o = DL.make(arm, lat[0], L, L, "pbc", n, rmax, 11 + L)
    o.setup(*T1.rand_cfg(L * L, n, o.rng))
    DL.seed_nb(4)
    DL.do_sweep(o, lat, 4 * L * L)
    t0 = time.perf_counter()
    for _ in range(5):
        DL.do_sweep(o, lat, 4 * L * L, False)
    tp = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for _ in range(5):
        DL.eloc_c(o, lat)
    te = (time.perf_counter() - t0) / 5
    print(f"L=16 {arm:7s}: proposals+updates {tp:.4f}s per sweep, local energy {te:.4f}s", flush=True)
