#!/usr/bin/env python3
"""Side check for path_gadget.py: mean excess of the same Y under ADDITIVE noise psi' = psi0 + delta eta / sqrt(n)
(eta ~ N(0, I)), = delta^2 / n * sum_i Y_ii (Y psi0 = 0), restricted to psi0 != 0 dets. Not part of the voi.md metric."""
import os
import sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import path_gadget as pg  # noqa: E402
rt = pg.rt
for nat in (6, 8):
    H, codes, occ, K, nocc = rt.build(nat, 3.0, "LOW")
    w, v = rt.lowest(H, 1)
    psi = v[:, 0]
    n = H.shape[0]
    P = rt.pair_data(H, psi, occ, K)
    nz = np.abs(psi) > 1e-9
    for eps in (2.2387211385683377e-3, 0.1):
        L, st = pg.build_L(P, np.abs(P["h"]) <= eps, nz, n)
        Yd = L.diagonal()[nz] / psi[nz] ** 2
        for d in (1e-3, 1e-2):
            print(f"H{nat} eps={eps:.2e} delta={d}: additive mean excess = {d*d*Yd.sum()/nz.sum():.3e} Ha "
                  f"(relative-noise formula {d*d*L.diagonal().sum():.3e}); max Y_ii={Yd.max():.3e}", flush=True)
