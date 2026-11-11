"""Pre-check driver: lowest K=0, Mx=My=+1 singlet levels on 4x3 PBC.

Usage: python3 run_sector.py NUP U TP1,TP2,...
  NUP  N_up = N_dn
  U    on-site repulsion
  TP*  comma-separated list of t' values
"""
import sys
import time

import explore_sectors as es

import numpy as np


def struct(model, psi):
    """Max over q != 0 of spin S(q) and charge N(q) structure factors, with the q index."""
    iu, idd = model.iu_id()
    nu = model.occ_u[iu].astype(float)
    nd = model.occ_d[idd].astype(float)
    p = psi ** 2 / (psi @ psi)
    ph = np.exp(1j * model.coords @ np.array(
        [[2 * np.pi * kx / model.Lx, 2 * np.pi * ky / model.Ly]
         for ky in range(model.Ly) for kx in range(model.Lx)]).T)  # (V, Q)
    sz = (nu - nd) @ ph / 2
    rho = (nu + nd) @ ph
    Ss = (p @ np.abs(sz) ** 2) / model.V
    Nc = (p @ np.abs(rho) ** 2) / model.V
    Ss[0] = Nc[0] = 0.0
    return int(np.argmax(Ss)), Ss.max(), int(np.argmax(Nc)), Nc.max()


nup, U = int(sys.argv[1]), float(sys.argv[2])
for tp in [float(s) for s in sys.argv[3].split(",")]:
    t0 = time.time()
    model, lv = es.sector_levels(tp, U, 3, nup)
    parts = []
    for E, s, p, psi in lv:
        qs, Ss, qc, Nc = struct(model, psi)
        parts.append(f"E={E:.5f} S2={s:.2f} Pd={p:+.2e} Ss[{qs}]={Ss:.3f} Nc[{qc}]={Nc:.3f}")
    print(f"n={nup} U={U} tp={tp:+.3f} " + " | ".join(parts)
          + f"  gap={lv[1][0] - lv[0][0]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
