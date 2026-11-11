#!/usr/bin/env python3
"""
sector_dims.py -- exact projected-sector dimensions of candidate instances (instance selection)
================================================================================================
For each candidate t-t'-J cluster: ED (Lanczos, 4 levels), psi_0 resolved into a 1-D irrep
(micro_test.resolve_ground / build_symmetry), and the exact dimensions dim range(P_chi) and
dim range(P_S P_chi) (micro_test.sector_dims: a character sum over the group, minus the
S_z = 1 sector for S = 0). Also prints the parameter count of NNBF at several widths, so that the
ratio dim range(P_S P_chi) / n_params can be read off (the orchestrator's broadcast: a projected
control leaves a measurable error only when this ratio is large).

Usage
-----
  python3 sector_dims.py [--which LIST] [--widths LIST]
Options
-------
  --which LIST   comma-separated labels from CANDIDATES below (default all):
                 A = 4x3 PBC 5+5, B = 4x4 PBC 7+7, C = 4x4 cylinder (PBC x, OBC y) 7+7,
                 D = 4x4 OBC 7+7, E = 4x4 cylinder 6+6 (4 holes), F = 4x4 OBC 6+6.
                 All t = 1, t' = -0.25, J = 0.4.
  --widths LIST  NNBF widths whose parameter counts are printed (default 2,4,8,16,32,128).
Output: stdout only.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from micro_test import build_symmetry, references, resolve_ground, sector_dims  # noqa: E402
from p11_core import solve_ed  # noqa: E402
from tj_core import TJ, LRAnsatz  # noqa: E402

CANDIDATES = {"A": (4, 3, "pbc", "pbc", 5), "B": (4, 4, "pbc", "pbc", 7), "C": (4, 4, "pbc", "obc", 7),
              "D": (4, 4, "obc", "obc", 7), "E": (4, 4, "pbc", "obc", 6), "F": (4, 4, "obc", "obc", 6)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default=",".join(CANDIDATES))
    ap.add_argument("--widths", default="2,4,8,16,32,128")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    ws = [int(w) for w in a.widths.split(",")]
    for lab in a.which.split(","):
        Lx, Ly, bx, by, n = CANDIDATES[lab]
        t0 = time.time()
        m = TJ(Lx, Ly, bx, by, n, n, 1.0, -0.25, 0.4)
        m.spin_setup()
        W, Vv, deg = solve_ed(m, 4, 1e-8, 1234)
        psi0 = resolve_ground(m, Vv, deg)
        sg, gen = build_symmetry(m, psi0)
        S2 = m.S2_expect(psi0)
        S0 = float(round(0.5 * (np.sqrt(1 + 4 * S2) - 1)))
        d0, dS = sector_dims(m, sg, S0)
        refs = references(m)
        npar = {w: LRAnsatz(m, refs[0], "full", K=1, Hs=w).n_params() for w in ws}
        print(f"{lab}: {Lx}x{Ly} {bx}/{by} {n}+{n} D={m.D} deg={deg} |G|={sg.G} chi={gen} S={S0:g} "
              f"E0={W[0]:.6f} gap(levels)={np.round(W - W[0], 4).tolist()} dim(P_chi)={d0:.0f} "
              f"dim(P_S P_chi)={dS:.0f} ({time.time() - t0:.0f} s)")
        print("   NNBF params / ratio dim(P_S P_chi)/params: " +
              "  ".join(f"W={w}: {npar[w]} ({dS / npar[w]:.2f})" for w in ws))
    return 0


if __name__ == "__main__":
    sys.exit(main())
