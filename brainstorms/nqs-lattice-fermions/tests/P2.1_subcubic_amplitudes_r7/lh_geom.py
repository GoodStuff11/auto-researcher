"""
lh_geom.py -- lattice geometry and patch indexing for the P2.1 round-7 local-hidden-row families
================================================================================================
(no command line; imported by t1_r7.py and lh_train.py)

Round-6 critic (Major 5 / pre-flight item): the round-6 T1 code built patches with periodic
wrap-around (site index mod L), which is wrong on open-boundary clusters such as instance 4
(5x3 OBC). Here a patch position that falls outside an open boundary is marked -1 and the
corresponding network input is 0 (distinct from the occupied/empty inputs +1/-1).

Functions
---------
geometry(Lx, Ly, bc, tp)  -> T (V x V hopping, H_kin = -sum T_ij c^dag_i c_j; t = 1 on NN bonds,
                              tp on diagonals), nn (sorted NN bond list (i, j), i < j)
                              bc in {"pbc", "obc"} applies to both directions.
patch_index(Lx, Ly, bc, kind) -> (V, p) int array of patch sites, -1 outside an open boundary;
                              bc is "pbc", "obc" or a (bcx, bcy) pair.
                              kind: "site" (p = 1: the site itself), "plus" (p = 5: site and its
                              4 nearest neighbours), "3x3" (p = 9).
neighbours4(Lx, Ly, bc)   -> (V, 4) int array of the +x, -x, +y, -y neighbours, -1 if absent
                              (used for the hidden-hidden stencil of SH-HFDS).
patch_input(ou, od, P)    -> network input for occupations ou, od (0/1, shape (..., V)) on patch
                              index P (shape (..., p)): concatenation [2 n_up - 1, 2 n_dn - 1]
                              over patch positions, 0 where P == -1. Shape (..., 2p).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r5"))
from p11_core import build_lattice  # noqa: E402

OFFSETS = {"site": [(0, 0)],
           "plus": [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
           "3x3": [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]}


def geometry(Lx, Ly, bc, tp):
    T, coords, bonds = build_lattice(Lx, Ly, bc, bc, 1.0, tp)
    nn = sorted({(min(i, j), max(i, j)) for (i, j, _) in bonds})
    return T, nn


def _site(Lx, Ly, bc, x, y):
    bcx, bcy = (bc, bc) if isinstance(bc, str) else bc
    if bcx == "pbc":
        x %= Lx
    if bcy == "pbc":
        y %= Ly
    if 0 <= x < Lx and 0 <= y < Ly:
        return x + Lx * y
    return -1


def patch_index(Lx, Ly, bc, kind):
    off = OFFSETS[kind]
    P = np.array([[_site(Lx, Ly, bc, x + dx, y + dy) for (dx, dy) in off]
                  for y in range(Ly) for x in range(Lx)], dtype=np.int64)
    return P


def neighbours4(Lx, Ly, bc):
    return np.array([[_site(Lx, Ly, bc, x + dx, y + dy) for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1))]
                     for y in range(Ly) for x in range(Lx)], dtype=np.int64)


def patch_input(ou, od, P):
    ou = np.asarray(ou); od = np.asarray(od)
    Pc = np.where(P >= 0, P, 0)
    valid = (P >= 0).astype(float)
    xu = (2.0 * ou[..., Pc] - 1.0) * valid
    xd = (2.0 * od[..., Pc] - 1.0) * valid
    return np.concatenate([xu, xd], -1)


def check_geometry():
    """Validity asserts for the OBC indexing (used by the pre-flight)."""
    P = patch_index(5, 3, "obc", "plus")
    assert P.shape == (15, 5)
    # corner (0,0): only itself, +x and +y exist
    assert list(P[0]) == [0, 1, -1, 5, -1], P[0]
    # centre (2,1) = site 7: all five present
    assert list(P[7]) == [7, 8, 6, 12, 2], P[7]
    # every present patch site is within Manhattan distance 1 without wrap-around
    for m in range(15):
        x, y = m % 5, m // 5
        for s in P[m]:
            if s >= 0:
                assert abs(s % 5 - x) + abs(s // 5 - y) <= 1
    Pp = patch_index(5, 3, "pbc", "plus")
    assert list(Pp[0]) == [0, 1, 4, 5, 10]
    # batched and single inputs agree
    rng = np.random.default_rng(0)
    ou = rng.integers(0, 2, (3, 15)); od = rng.integers(0, 2, (3, 15))
    xb = patch_input(ou, od, P)
    for k in range(3):
        assert np.array_equal(xb[k], patch_input(ou[k], od[k], P))
    assert np.all(patch_input(ou[0], od[0], P)[0, [2, 4, 7, 9]] == 0)
    return True


if __name__ == "__main__":
    print("check_geometry:", check_geometry())
