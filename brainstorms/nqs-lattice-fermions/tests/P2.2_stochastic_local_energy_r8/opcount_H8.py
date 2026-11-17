#!/usr/bin/env python3
"""P2.2 R7.4 Q-B: per-sample operation counts on H8 LOW (psi0^2-weighted): connected terms, kept terms,
dropped neighbours, kept-graph 2/3-balls, and the work of a local length-2 (triangle) gadget evaluation."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import sys  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import path_gadget as pg  # noqa: E402
rt = pg.rt
H, codes, occ, K, nocc = rt.build(8, 3.0, "LOW")
w, v = rt.lowest(H, 1)
psi = v[:, 0]
p = psi ** 2
n = H.shape[0]
Ho = (H - sp.diags(H.diagonal())).tocsr()
Ho.data[np.abs(Ho.data) <= 1e-12] = 0
Ho.eliminate_zeros()
deg = np.diff(Ho.indptr)
for eps in (2.2387211385683377e-3, 0.1):
    Kp = Ho.copy()
    Kp.data[np.abs(Kp.data) <= eps] = 0
    Kp.eliminate_zeros()
    A = (Kp != 0).astype(np.int32).tocsr()
    dk = np.diff(A.indptr)
    A2 = (A @ A).tocsr()
    A2.data[:] = 1
    ball2 = np.diff(((A + A2) > 0).tocsr().indptr)
    A3 = (A2 @ A).tocsr()
    A3.data[:] = 1
    ball3 = np.diff(((A + A2 + A3) > 0).tocsr().indptr)
    # local triangle gadget at sample x: dropped nbrs of x (ref evals) + pairs of kept nbrs of x (H_ij evals,
    # x as path middle) + kept nbrs of each kept y (paths through edge x-y)
    tri = (deg - dk) + dk * (dk - 1) / 2 + np.asarray(A @ dk).ravel()
    print(f"eps={eps:.2e}: <conn>={p @ deg:.1f} <kept>={p @ dk:.1f} <dropped>={p @ (deg - dk):.1f} "
          f"<kept 2-ball>={p @ ball2:.0f} <kept 3-ball>={p @ ball3:.0f} of n={n} "
          f"<local triangle-gadget work>={p @ tri:.0f}", flush=True)
print(f"cpu {pg.cpu():.0f}s")
