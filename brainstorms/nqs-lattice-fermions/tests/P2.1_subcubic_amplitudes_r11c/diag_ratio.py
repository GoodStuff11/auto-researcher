#!/usr/bin/env python3
"""
P2.1 Round 11c diagnostic of gate (i): where the per-connection relative ratio error comes from.
No options. For the 8x8 full, half and quarter arms (same seeds and moves as delayed.py --mode check),
prints, per configuration, the worst connection: |rho_ref|, |rho_fast - rho_ref| and the relative error,
and recomputes that connection's reference in 50-digit arithmetic (mpmath determinant of the same float64
matrices det(F0 + K C)[sorted rows]) to say whether the fast engine or the float64 brute force is off.
Also prints the max relative error restricted to |rho_ref| >= 1e-2 and the max error normalised by the
largest |rho| of that configuration. Threads: 1.
"""
import numpy as np
import mpmath as mp

import delayed as DL
import t1_r7 as T1
from lh_geom import patch_input

mp.mp.dps = 50


def mpdet(A):
    return mp.det(mp.matrix(A.tolist()))


def explicit_matrix(py, ou, od):
    H = np.tanh(patch_input(ou, od, py.loc.P) @ py.loc.W + py.loc.b)
    C = np.einsum("mf,mfv->mv", H, py.Om)
    A = py.F0 + py.K @ C
    return A[np.ix_(np.nonzero(ou)[0], np.nonzero(od)[0])]


for arm in ("full", "half", "quarter"):
    Lx, Ly, bc, n = DL.parse_geom("8")
    lat = DL.lattice(Lx, Ly, bc)
    obj = DL.make(arm, lat[0], Lx, Ly, bc, n, 32, 21)
    obj.setup(*T1.rand_cfg(Lx * Ly, n, obj.rng))
    DL.seed_nb(5)
    for rep in range(3):
        DL.do_sweep(obj, lat, 2 * Lx * Ly, False)
        mv, ref = DL.brute_ratios(obj, lat)
        fast = np.array([DL.ratio_c(obj, tuple(int(v) for v in m)) for m in mv])
        rel = np.abs(fast - ref) / np.abs(ref)
        w = int(np.argmax(rel))
        big = np.abs(ref) >= 1e-2
        xu, yu, xd, yd = (int(v) for v in mv[w])
        ou, od = obj.ou.copy(), obj.od.copy()
        u2, d2 = ou.copy(), od.copy()
        sg = 1
        if xu >= 0:
            sg *= (-1) ** int(ou[min(xu, yu) + 1:max(xu, yu)].sum()); u2[xu] = 0; u2[yu] = 1
        if xd >= 0:
            sg *= (-1) ** int(od[min(xd, yd) + 1:max(xd, yd)].sum()); d2[xd] = 0; d2[yd] = 1
        A0, A1 = explicit_matrix(obj.py, ou, od), explicit_matrix(obj.py, u2, d2)
        rmp = float(sg * mpdet(A1) / mpdet(A0))
        print(f"8x8 {arm:7s} cfg{rep}: worst |rho_ref|={abs(ref[w]):.2e} abs err={abs(fast[w] - ref[w]):.1e} "
              f"rel={rel[w]:.1e} | vs 50-digit: fast {abs(fast[w] - rmp) / abs(rmp):.1e}, float64 brute "
              f"{abs(ref[w] - rmp) / abs(rmp):.1e} | max rel (|rho|>=1e-2) {rel[big].max():.1e} | "
              f"max |err|/max|rho| {np.max(np.abs(fast - ref)) / np.max(np.abs(ref)):.1e}", flush=True)


# ---- part 2: is the ~1e-13 absolute ratio error intrinsic to the updated-inverse ratio formula?
# Same configurations; compare (a) the compiled engine as carried (pending rank r > 0), (b) the compiled
# engine rebuilt fresh (r = 0, exact inverse), (c) the round-7 Python engine (SubHFPS.ratio, tables Gr,
# Gc, X) rebuilt fresh on the same slots, all against the float64 brute ratios; and cond(M).
print("part 2: max |err|/max|rho| / max rel err (all connections) per engine; cond(M)")
for arm in ("full", "half", "quarter"):
    Lx, Ly, bc, n = DL.parse_geom("8")
    lat = DL.lattice(Lx, Ly, bc)
    obj = DL.make(arm, lat[0], Lx, Ly, bc, n, 32, 21)
    obj.setup(*T1.rand_cfg(Lx * Ly, n, obj.rng))
    DL.seed_nb(5)
    for rep in range(3):
        DL.do_sweep(obj, lat, 2 * Lx * Ly, False)
        mv, ref = DL.brute_ratios(obj, lat)
        fr = DL.make(arm, lat[0], Lx, Ly, bc, n, 32, 21)
        fr.setup(obj.ou, obj.od, obj.ru, obj.rd)
        obj.py.setup(obj.ou, obj.od, obj.ru, obj.rd)
        res = []
        for name in ("carried", "fresh", "python r7"):
            vals = []
            for m in mv:
                xu, yu, xd, yd = (int(v) for v in m)
                if name == "carried":
                    vals.append(DL.ratio_c(obj, (xu, yu, xd, yd)))
                elif name == "fresh":
                    vals.append(DL.ratio_c(fr, (xu, yu, xd, yd)))
                else:
                    mm = ([(0, xu, yu)] if xu >= 0 else []) + ([(1, xd, yd)] if xd >= 0 else [])
                    vals.append(obj.py.ratio(mm))
            vals = np.array(vals)
            res.append(f"{name}: {np.max(np.abs(vals - ref)) / np.max(np.abs(ref)):.1e} / "
                       f"{np.max(np.abs(vals - ref) / np.abs(ref)):.1e}")
        M, _, _ = fr.build(obj.ou, obj.od, obj.ru, obj.rd)
        print(f"8x8 {arm:7s} cfg{rep}: " + "  ".join(res) + f"  cond(M)={np.linalg.cond(M):.1e}", flush=True)
