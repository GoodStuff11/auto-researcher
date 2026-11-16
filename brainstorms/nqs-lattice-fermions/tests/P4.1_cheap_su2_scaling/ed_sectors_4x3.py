#!/usr/bin/env python3
"""ED check for the sector-resolved DMRG: lowest eigenvalue of H (no S^2 term, S^z = 0) of the 4x3 open,
5+5, U=8, t'=-0.25 Hubbard model restricted to each point-group irrep (Mx, My), via the r3_micro_test.SG
projector. Usage: python3 ed_sectors_4x3.py   (no options). Prints E for (+,+), (+,-), (-,+), (-,-) and the
lowest energy with My = +1 and My = -1 (what dmrg_ref.jl --sector=+1/-1 targets)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import r2_micro_test as R
import r3_micro_test as M3
sec = M3.make_sector(4, 3, 10, 8.0, -0.25, "obc", "obc")
sg = M3.SG(sec)
assert [e[2:] for e, _ in sg.el] == [(0, 0), (0, 1), (1, 0), (1, 1)]
out = {}
for cx in (1, -1):
    for cy in (1, -1):
        sg.chi = np.array([1.0, cy, cx, cx * cy])
        shp = sec.shape
        mv = lambda x: sg.project(sec.H(sg.project(x.reshape(shp)))).ravel() + 50.0 * (x - sg.project(x.reshape(shp)).ravel())
        op = R.LinearOperator((sec.dim, sec.dim), matvec=mv, dtype=float)
        w = R.eigsh(op, k=1, which="SA", v0=sg.project(np.random.default_rng(1).standard_normal(shp)).ravel(), tol=1e-11)[0]
        out[(cx, cy)] = float(w[0])
        print(f"(Mx, My) = ({cx:+d}, {cy:+d}): E = {w[0]:.10f}", flush=True)
print(f"lowest My=+1: {min(out[(1, 1)], out[(-1, 1)]):.10f}   lowest My=-1: {min(out[(1, -1)], out[(-1, -1)]):.10f}")
