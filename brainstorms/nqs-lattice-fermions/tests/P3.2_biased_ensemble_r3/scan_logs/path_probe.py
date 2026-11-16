"""Probe 1-D paths in t' at fixed t_y near the U=8 conical intersection (4x3 torus, 4+4).
Prints, for several t_y offsets, the minimum in-irrep gap along t' and the order character
c = [S(qB)+S(-qB)]/2 - [S(qA)+S(qA')]/2 (qB=(pi/2,0), qA=(pi,+-2pi/3)) of E0 and E1."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import minimize_scalar
import ed_scan as E

cl = E.Cluster("4x3pp", 4, 4)
chi = np.ones(cl.G)
cl.build_irrep(chi, 0, 7000)
m = cl.m


def cchar(psi):
    st = E.structure(m, psi)
    return 0.5 * (st[(1, 0)][0] + st[(3, 0)][0]) - 0.5 * (st[(2, 1)][0] + st[(2, 2)][0])


TPC, TYC = -0.515243, 1.143944
cpl = lambda tp, ty: {"tx": 1.0, "ty": ty, "tp": tp, "U": 8.0}
for dty in [float(s) for s in sys.argv[1].split(",")]:
    ty = TYC + dty
    r = minimize_scalar(lambda tp: np.diff(cl.sector_eigs(cpl(tp, ty), 2)[0])[0],
                        bounds=(TPC - 0.1, TPC + 0.1), method="bounded", options=dict(xatol=1e-7))
    print(f"dty={dty:+.4f} ty={ty:.5f}: min gap {r.fun:.5f} ({r.fun/12:.5f}/site) at tp={r.x:+.5f}")
    for dtp in (-0.04, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.04):
        tp = r.x + dtp
        w, v = cl.sector_eigs(cpl(tp, ty), 3)
        c0, c1 = cchar(cl.full_vec(v[:, 0])), cchar(cl.full_vec(v[:, 1]))
        print(f"   tp={tp:+.5f} gap01={w[1]-w[0]:.5f} gap12={w[2]-w[1]:.4f} c0={c0:+.4f} c1={c1:+.4f}")
