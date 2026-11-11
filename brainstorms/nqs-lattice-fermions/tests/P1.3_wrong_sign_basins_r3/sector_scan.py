#!/usr/bin/env python3
"""
P1.3 round 3: ED scan for the smallest SAME-SECTOR gap on 4x3 PBC Hubbard clusters
==================================================================================

Purpose
-------
Round-3 brief rule 3: a test that claims the competing-order half of the Hard Core must
use an instance whose two lowest states lie in ONE symmetry sector (same space-group irrep
chi AND same total spin S), confirmed in code, with an in-sector gap as small as an
ED-feasible cluster allows. This script scans (filling, t', U) on the 4x3 torus and, for
every point, reports:
  E0            ground energy (full fixed-(N_up, N_dn) Lanczos);
  deg           ground degeneracy (points with deg > 1 or psi_0 outside a real 1-D irrep
                are skipped: the irrep projector of the test is undefined there);
  chi, S        psi_0's irrep (Tx, Ty, Mx, My eigenvalues) and total spin;
  E1_in, E2_in  the next two levels INSIDE psi_0's sector, by Lanczos on H P with
                P = P_S P_chi (symm.sector_lowest);
  gap_in        E1_in - E0, and gap_in / V (V = 12 sites);
  check         for E1_in's vector: <Tx>,<Ty>,<Mx>,<My> and <S^2> are recomputed from the
                vector and must equal chi and S(S+1) (printed as "same-sector OK").

The selected instance for the decisive test is the scanned point with the smallest gap_in
among points with a non-degenerate psi_0 in a real 1-D irrep, restricted to the fillings
in --fillings (pre-committed rule; the scan output is saved next to the script).

Usage
-----
    python3 sector_scan.py [options]

Options
-------
  --fillings LIST   comma-separated N_up=N_dn values on the 4x3 torus (V = 12); valid
                    values 2..6. D = C(12,n)^2: 3 -> 48400, 4 -> 245025, 5 -> 627264.
                    Default "3".
  --tps LIST        comma-separated t' values (default "-0.5,-0.45,...,0.5", 21 values).
  --Us LIST         comma-separated U values (default "4,6,8,10,12").
  --k INT           in-sector levels computed (default 3, at least 2).
  --seed INT        Lanczos start-vector seed (default 7).
  --out PATH        output text file (default sector_scan_n<fillings>.txt next to this
                    script); every row is also printed.
"""
import argparse
import os
import sys
import time

import numpy as np

import p11_core as P
import symm

HERE = os.path.dirname(os.path.abspath(__file__))


def scan_point(n, tp, U, k, seed):
    m = P.Hubbard(4, 3, "pbc", "pbc", n, n, 1.0, tp, U)
    W, Vv, deg = P.solve_ed(m, 2, 1e-8, seed)
    sg = symm.SpaceGroup(m)
    spn = symm.Spin(m)
    if deg != 1:
        return dict(n=n, tp=tp, U=U, D=m.D, E0=W[0], deg=deg, skip="degenerate psi_0")
    try:
        lam = sg.irrep_of(Vv[:, 0])
    except RuntimeError:
        return dict(n=n, tp=tp, U=U, D=m.D, E0=W[0], deg=deg, skip="psi_0 not in a real 1-D irrep")
    s2 = spn.expect(Vv[:, 0])
    S = float(round(0.5 * (np.sqrt(1 + 4 * s2) - 1)))
    proj = symm.SectorProjector(m, sg, spn, lam, S)
    w, Uv = symm.sector_lowest(m, proj, k, seed)
    ok0 = abs(w[0] - W[0]) < 1e-7
    ge = sg.gen_expect(Uv[:, 1])
    s21 = spn.expect(Uv[:, 1])
    same = all(abs(ge[g] - lam[g]) < 1e-6 for g in lam) and abs(s21 - S * (S + 1)) < 1e-6
    return dict(n=n, tp=tp, U=U, D=m.D, E0=W[0], E1_full=W[1], deg=deg, chi=lam, S=S,
                E_in=w.tolist(), gap_in=w[1] - w[0], gap_full=W[1] - W[0],
                e0_match=ok0, same_sector=same, gen1=ge, S2_1=s21)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="same-sector gap scan (see module docstring)")
    ap.add_argument("--fillings", default="3")
    ap.add_argument("--tps", default=",".join(f"{x:.2f}" for x in np.arange(-0.5, 0.5001, 0.05)))
    ap.add_argument("--Us", default="4,6,8,10,12")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    a.fill = [int(x) for x in a.fillings.split(",")]
    a.tp = [float(x) for x in a.tps.split(",")]
    a.U = [float(x) for x in a.Us.split(",")]
    if any(not 2 <= n <= 6 for n in a.fill):
        ap.error("fillings must be in 2..6")
    if a.k < 2:
        ap.error("--k must be >= 2")
    if not a.out:
        a.out = os.path.join(HERE, f"sector_scan_n{a.fillings.replace(',', '-')}.txt")
    return a


def main(argv=None):
    a = parse_args(argv)
    rows = []
    with open(a.out, "w") as fh:
        def emit(s):
            print(s, flush=True)
            fh.write(s + "\n"); fh.flush()
        emit("# n tp U D E0 gap_full gap_in gap_in/V chi S same_sector e0_match  (or skip reason)")
        for n in a.fill:
            for U in a.U:
                for tp in a.tp:
                    t0 = time.time()
                    r = scan_point(n, tp, U, a.k, a.seed)
                    if "skip" in r:
                        emit(f"{n} {tp:+.3f} {U:4.1f} {r['D']} {r['E0']:.8f} SKIP {r['skip']}")
                        continue
                    rows.append(r)
                    c = r["chi"]
                    emit(f"{n} {tp:+.3f} {U:4.1f} {r['D']} {r['E0']:.8f} {r['gap_full']:.5f} "
                         f"{r['gap_in']:.5f} {r['gap_in'] / 12:.5f} "
                         f"Tx{c['Tx']:+d}Ty{c['Ty']:+d}Mx{c['Mx']:+d}My{c['My']:+d} S={r['S']:g} "
                         f"{'same-sector OK' if r['same_sector'] else 'SECTOR MISMATCH'} "
                         f"{'E0 OK' if r['e0_match'] else 'E0 MISMATCH'} ({time.time() - t0:.1f}s)")
        good = [r for r in rows if r["same_sector"] and r["e0_match"]]
        if good:
            b = min(good, key=lambda r: r["gap_in"])
            emit(f"# SELECTED (min gap_in): n={b['n']} tp={b['tp']:+.3f} U={b['U']:g} "
                 f"gap_in={b['gap_in']:.5f} ({b['gap_in'] / 12:.5f} per site) chi={b['chi']} S={b['S']:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
