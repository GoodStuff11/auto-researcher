#!/usr/bin/env python3
"""Round 3 check: nf_run.guided_stats(dedup=True) equals dedup=False (S_T, f, sampled E, step d) at the SR start,
N = 65536, seed 9001 (arm B's step-1 draw), and times both paths.  Output: logs/dedup_check.txt."""
import os, sys, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
sys.argv = [sys.argv[0]]
import numpy as np
import nf_run as R
ctx = R.P.ctx_for("che"); th = ctx.th0.clone()
_, pr, phi0 = ctx.energy(th)
out = {}
for dd in (False, True):
    c0 = R.P.cpu_min()
    ST, f, Es, nu = R.guided_stats(ctx, th, phi0, pr, 65536, 9001, dd)
    w, U = np.linalg.eigh(ST); d = R.T.d_shift(w, U, f, 0.005)[0]
    out[dd] = (ST, f, Es, d, R.P.cpu_min() - c0)
a, b = out[False], out[True]
rel = lambda x, y: float(np.abs(x - y).max() / np.abs(x).max())
print(f"N 65536 seed 9001: unique {nu}; max rel diff S_T {rel(a[0], b[0]):.2e}, f {rel(a[1], b[1]):.2e}, "
      f"E_samp {abs(a[2]-b[2]):.2e}, step d {rel(a[3], b[3]):.2e}")
print(f"CPU-min for stats+eigh: plain {a[4]:.3f}, dedup {b[4]:.3f}")
print("PASS" if max(rel(a[0], b[0]), rel(a[1], b[1]), rel(a[3], b[3])) < 1e-8 else "FAIL")
