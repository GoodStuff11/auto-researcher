#!/usr/bin/env python3
"""
screen_candidates.py -- cheap screen for a second qualifying P2.1 instance (round 5)
====================================================================================
The round-3 control pilot (job 410988 + 451706) found exactly one instance on which projected
NNBF at width 6 keeps dE_sign >= 0.01 t: the 4x4 OBC 6+6 cluster (sector 30220, P = 1734,
ratio 17.4, lowest ED gap 0.049). The cylinder (ratio 8.8, gap 0.120) gave 0.0038 t and the
OBC 7+7 cluster (ratio 3.3, gap 0.135) gave 0.0028 t. This script computes, for each candidate
t-t'-J cluster, the quantities that separate them: the exact projected-sector dimension
dim range(P_S P_chi) (character sum, micro_test.sector_dims), the parameter count P of NNBF at
width 6 (the control's budget), the ratio dim/P, and the lowest ED levels (the gap to the
first excited level and how many levels lie within 0.1 t). No training is done.

Usage
-----
  python3 screen_candidates.py [--which LIST] [--width INT] [--k INT]
Options
-------
  --which LIST  comma-separated labels from CANDIDATES (default all).
  --width INT   NNBF width whose parameter count is the budget (default 6).
  --k INT       Lanczos levels (default 6).
Output: stdout only (one line per candidate); redirect to screen_output.txt.
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

# label: (Lx, Ly, bcx, bcy, n_up = n_dn, t', J)
CANDIDATES = {
    "ref0": (4, 4, "obc", "obc", 6, -0.25, 0.4),   # instance 0 (qualifies), for calibration
    "J03": (4, 4, "obc", "obc", 6, -0.25, 0.3),
    "J05": (4, 4, "obc", "obc", 6, -0.25, 0.5),
    "tp35": (4, 4, "obc", "obc", 6, -0.35, 0.4),
    "tp15": (4, 4, "obc", "obc", 6, -0.15, 0.4),
    "tpp25": (4, 4, "obc", "obc", 6, +0.25, 0.4),
    "h6": (4, 4, "obc", "obc", 5, -0.25, 0.4),     # 6 holes, D = 2018016
    "r53n6": (5, 3, "obc", "obc", 6, -0.25, 0.4),  # 5x3 OBC, 3 holes
    "r53n5": (5, 3, "obc", "obc", 5, -0.25, 0.4),  # 5x3 OBC, 5 holes
    "r63n8": (6, 3, "obc", "obc", 8, -0.25, 0.4),  # 6x3 OBC, 2 holes, D = 1969110
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default=",".join(CANDIDATES))
    ap.add_argument("--width", type=int, default=6)
    ap.add_argument("--k", type=int, default=6)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    for lab in a.which.split(","):
        Lx, Ly, bx, by, n, tp, J = CANDIDATES[lab]
        t0 = time.time()
        m = TJ(Lx, Ly, bx, by, n, n, 1.0, tp, J)
        m.spin_setup()
        W, Vv, deg = solve_ed(m, a.k, 1e-8, 1234)
        psi0 = resolve_ground(m, Vv, deg)
        sg, gen = build_symmetry(m, psi0)
        S2 = m.S2_expect(psi0)
        S0 = float(round(0.5 * (np.sqrt(1 + 4 * S2) - 1)))
        d0, dS = sector_dims(m, sg, S0)
        refs = references(m)
        P = LRAnsatz(m, refs[0], "full", K=1, Hs=a.width).n_params()
        gaps = W - W[0]
        print(f"{lab}: {Lx}x{Ly} {bx}/{by} {n}+{n} t'={tp} J={J} D={m.D} deg={deg} |G|={sg.G} chi={gen} "
              f"S={S0:g} E0={W[0]:.6f} gaps={np.round(gaps, 4).tolist()} n(<0.1t)={int(np.sum(gaps[1:] < 0.1))} "
              f"dim(P_chi)={d0:.0f} dim(P_S P_chi)={dS:.0f} P(W={a.width})={P} ratio={dS / P:.1f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
