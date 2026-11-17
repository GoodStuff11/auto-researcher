#!/usr/bin/env python3
"""Context only (no decision weight; written AFTER the t1h_curv2 verdict): the global minimum of q(g)/var(g) over ALL
phase patterns g at each foil (sparse generalized eigsh), so lambda(S) can be read against the unrestricted bound,
and the psi^2-weighted participation number of that eigenvector. Sizes 1, 2; seeds 11, 22."""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "P1.2_unanchored_antisymmetry_r9"))
import t1f_phase as P  # noqa: E402

resource.setrlimit(resource.RLIMIT_CPU, (240, 250))
for k in (1, 2):
    for seed in (11, 22):
        d = P.setup(k, seed, "G")
        H, psi = d["H"], d["psiF"]
        A = sp.diags(psi) @ H @ sp.diags(psi)
        A = A - sp.diags(A.diagonal())
        L = -(sp.diags(np.asarray(A.sum(1)).ravel()) - A)   # N q(g) = g^T L g
        M = sp.diags(psi ** 2)
        w, U = eigsh(L.tocsc(), k=2, M=M.tocsc(), which="SA")
        i = int(np.argmin(w))
        g = U[:, i]
        p = psi ** 2 * g ** 2
        p /= p.sum()
        P.log(f"size {k} seed {seed}: global lambda_min {w[i]:+.4f}; participation of eigvec "
              f"{1 / np.sum(p ** 2):.0f} of D {len(psi)}; CPU {time.process_time():.0f} s")
