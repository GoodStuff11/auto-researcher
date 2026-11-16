#!/usr/bin/env python3
"""
P2.1 Round 11c: which BLAS/LAPACK is slower, and for whom? (no options; single thread)
numba's np.dot / np.linalg go through scipy's bundled OpenBLAS; numpy uses its own scipy-openblas64.
Times, per call: LU (slogdet) at N = 112 and 252 [dense arm's kernel], and GEMM of the route's shapes at
L = 16 (k x n x n with k = 10, n = 368; flush n x 32 x n) [route arm's kernels], in numpy and in numba.
"""
import time

import numpy as np
from numba import njit


@njit(cache=True)
def nb_slogdet_loop(A, reps):
    s = 0.0
    for _ in range(reps):
        sg, la = np.linalg.slogdet(A)
        s += la
    return s


@njit(cache=True)
def nb_dot_loop(A, B, reps):
    s = 0.0
    for _ in range(reps):
        C = np.dot(A, B)
        s += C[0, 0]
    return s


def per_call(f, reps):
    f(1)
    t0 = time.perf_counter()
    f(reps)
    return (time.perf_counter() - t0) / reps


rng = np.random.default_rng(0)
for N in (112, 252):
    A = rng.standard_normal((N, N))
    tn = per_call(lambda r: [np.linalg.slogdet(A) for _ in range(r)], 200)
    tb = per_call(lambda r: nb_slogdet_loop(A, r), 200)
    print(f"LU N={N}: numpy {tn * 1e6:.0f} us ({2 / 3 * N ** 3 / tn / 1e9:.1f} GFLOP/s), numba {tb * 1e6:.0f} us "
          f"({2 / 3 * N ** 3 / tb / 1e9:.1f} GFLOP/s); numba/numpy {tb / tn:.2f}", flush=True)
for (m, k, n) in ((10, 368, 368), (368, 32, 368), (6, 240, 240), (240, 16, 240)):
    A = rng.standard_normal((m, k)); B = rng.standard_normal((k, n))
    tn = per_call(lambda r: [A @ B for _ in range(r)], 300)
    tb = per_call(lambda r: nb_dot_loop(A, B, r), 300)
    print(f"GEMM {m}x{k}x{n}: numpy {tn * 1e6:.0f} us ({2 * m * k * n / tn / 1e9:.1f} GFLOP/s), numba {tb * 1e6:.0f} us "
          f"({2 * m * k * n / tb / 1e9:.1f} GFLOP/s); numba/numpy {tb / tn:.2f}", flush=True)
