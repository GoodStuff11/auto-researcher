#!/usr/bin/env python3
"""
probe_instances.py -- ED probe of the candidate P2.1 round-3 t-t'-J instances
==============================================================================
For each instance (Lx, Ly, N_up, N_dn, t', J) it builds the t-J Hamiltonian (tj_core.TJ),
runs Lanczos for the lowest --k levels, and prints: dimension D, build/ED times, levels, the
ground-state degeneracy, <S^2> of psi_0, and psi_0's characters under the space-group
generators (the full group with the diagonal mirror when Lx == Ly; falls back to the group
without it if psi_0 is not in a 1-D irrep of the full group). Also prints, for each level,
its <S^2> and its overlap with psi_0's irrep (weight ||P_chi v||^2), so that a same-sector
competitor (same irrep and same S) is visible.

Usage
-----
  python3 probe_instances.py [--which LIST] [--k INT] [--tp FLOAT] [--J FLOAT]
Options
-------
  --which LIST  comma-separated labels from {I0, I1, I2, I3}: I0 = 4x3 PBC 5+5 (2 holes),
                I1 = 4x4 PBC 7+7 (2 holes), I2 = 4x4 cylinder (PBC x, OBC y) 7+7 (2 holes),
                I3 = 4x4 PBC 6+6 (4 holes). Default I0,I1.
  --k INT       number of Lanczos levels (default 6).
  --tp FLOAT    next-nearest hopping t' (default -0.25). --J FLOAT exchange J (default 0.4).
Output: printed to stdout only.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tj_core import TJ, SpaceGroup  # noqa: E402

CASES = {"I0": (4, 3, 5, 5, "pbc", "pbc"), "I1": (4, 4, 7, 7, "pbc", "pbc"),
         "I2": (4, 4, 7, 7, "pbc", "obc"), "I3": (4, 4, 6, 6, "pbc", "pbc")}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="I0,I1")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--tp", type=float, default=-0.25)
    ap.add_argument("--J", type=float, default=0.4)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    for lab in a.which.split(","):
        Lx, Ly, nu, nd, bx, by = CASES[lab]
        t0 = time.time()
        m = TJ(Lx, Ly, bx, by, nu, nd, 1.0, a.tp, a.J)
        m.spin_setup()
        t1 = time.time()
        w, v = eigsh(m.H, k=a.k, which="SA", tol=1e-12, v0=np.random.default_rng(0).standard_normal(m.D))
        o = np.argsort(w); w, v = w[o], v[:, o]
        t2 = time.time()
        print(f"{lab}: {Lx}x{Ly} N=({nu},{nd}) t'={a.tp} J={a.J} D={m.D} build {t1 - t0:.1f}s ED {t2 - t1:.1f}s")
        sg = None
        for diag in (True, False):
            try:
                sg = SpaceGroup(m, diag=diag)
                gen = sg.set_from(v[:, 0])
                print(f"  group |G|={sg.G} (diag={diag}) psi_0 characters {gen}")
                break
            except RuntimeError as e:
                print(f"  group diag={diag}: {e}")
                sg = None
        for k in range(a.k):
            s2 = m.S2_expect(v[:, k])
            wt = float(np.linalg.norm(sg.project(v[:, k])) ** 2) if sg is not None else float("nan")
            print(f"  level {k}: E={w[k]:.8f}  E-E0={w[k] - w[0]:.5f}  <S2>={s2:.4f}  irrep weight={wt:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
