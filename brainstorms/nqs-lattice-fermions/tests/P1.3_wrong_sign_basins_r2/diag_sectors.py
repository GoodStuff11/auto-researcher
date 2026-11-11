#!/usr/bin/env python3
"""
P1.3 round-2 pre-diagnostic: lattice-symmetry content of the low-lying ED states and
of the UHF reference on a P1.1 instance (does the competitor share the sector?).

What it does
------------
Builds the P1.1 instance (code in p11_core.py, a verbatim copy of the P1.1 test),
computes the lowest --k eigenpairs by Lanczos, and for each eigenvector and for the
UHF Slater determinant used by P1.1 as the NNBF reference, reports
  * <T_x>, <T_y>  expectation values of the unit translations (fermionic signs included),
  * <M_x>, <M_y>  expectation values of the mirrors x -> -x, y -> -y,
  * overlaps |<E_k|UHF>|^2 with the ED states.
For a non-degenerate eigenvector these are its eigenvalues (momentum / parity).

Usage
-----
    python3 diag_sectors.py [--instance K] [--k INT] [--seed INT]

Options
-------
  --instance K   P1.1 instance index 0, 1, 2 or 3 (default 1: 4x3 PBC, 4+4, U=8,
                 t'=-0.25). See p11_core.INSTANCES.
  --k INT        number of lowest eigenpairs (default 6, at least 2).
  --seed INT     seed of the UHF symmetry-breaking noise and Lanczos start (default 1234,
                 the P1.1 production seed).
"""
import argparse
import sys
import time

import numpy as np

import p11_core as P


def site_perm(model, op):
    """Return the site permutation p (site i -> p[i]) for op in {Tx, Ty, Mx, My}."""
    Lx, Ly = model.Lx, model.Ly
    p = np.empty(model.V, dtype=np.int64)
    for y in range(Ly):
        for x in range(Lx):
            if op == "Tx":
                x2, y2 = (x + 1) % Lx, y
            elif op == "Ty":
                x2, y2 = x, (y + 1) % Ly
            elif op == "Mx":
                x2, y2 = (-x) % Lx, y
            elif op == "My":
                x2, y2 = x, (-y) % Ly
            p[x + Lx * y] = x2 + Lx * y2
    return p


def species_perm(states, rows, lookup_V, p):
    """Index map and fermionic sign for one species under site permutation p."""
    newidx = np.empty(len(states), dtype=np.int64)
    sgn = np.empty(len(states))
    for k, occ in enumerate(rows):
        img = p[occ]
        # c^dag_{occ0} c^dag_{occ1} ... -> c^dag_{img0} ...; sign = parity of sorting img
        order = np.argsort(img, kind="stable")
        perm = order.copy()
        # parity of permutation
        seen = np.zeros(len(perm), bool); par = 0
        for i in range(len(perm)):
            if not seen[i]:
                j, L = i, 0
                while not seen[j]:
                    seen[j] = True; j = perm[j]; L += 1
                par += L - 1
        b = 0
        for i in img:
            b |= 1 << int(i)
        newidx[k] = lookup_V[b]
        sgn[k] = -1.0 if par % 2 else 1.0
    return newidx, sgn


def apply_op(model, psi, maps):
    (iu_map, su), (id_map, sd) = maps
    Psi = psi.reshape(model.Du, model.Dd)
    out = np.zeros_like(Psi)
    out[np.ix_(iu_map, id_map)] = Psi * su[:, None] * sd[None, :]
    return out.ravel()


def build_maps(model, op):
    p = site_perm(model, op)
    res = []
    for N in (model.nup, model.ndn):
        st, lk, _, rows = P.species_basis(model.V, N)
        res.append(species_perm(st, rows, lk, p))
    return res


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="symmetry content of ED states and UHF reference")
    ap.add_argument("--instance", type=int, default=1, choices=[0, 1, 2, 3])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    c = P.INSTANCES[a.instance]
    m = P.Hubbard(c["Lx"], c["Ly"], c["bcx"], c["bcy"], c["nup"], c["ndn"], 1.0, c["tp"], c["U"])
    t0 = time.time()
    W, Vv, deg = P.solve_ed(m, max(2, a.k), 1e-8, a.seed)
    print(f"instance {a.instance} {c}: D={m.D}  ED {time.time() - t0:.1f}s")
    ops = {op: build_maps(m, op) for op in ("Tx", "Ty", "Mx", "My")}
    # check H commutes with each op
    rng = np.random.default_rng(0)
    v = rng.standard_normal(m.D)
    for op, mp in ops.items():
        err = np.linalg.norm(apply_op(m, m.H @ v, mp) - m.H @ apply_op(m, v, mp)) / np.linalg.norm(m.H @ v)
        print(f"  [H,{op}] relative residual {err:.1e}")
    Pu, Pd = P.uhf_orbitals(m, a.seed)
    uhf = P.Ansatz(m, Pu, Pd, "slater").amplitudes(65536)
    uhf /= np.linalg.norm(uhf)
    print(f"  E_UHF = {P.energy(m.H, uhf)[0]:.6f}")
    rows = [(f"E{k}={W[k]:.6f}", Vv[:, k]) for k in range(len(W))] + [("UHF", uhf)]
    for name, psi in rows:
        vals = {op: float(psi @ apply_op(m, psi, mp)) for op, mp in ops.items()}
        s = "  ".join(f"<{op}>={x:+.4f}" for op, x in vals.items())
        extra = ""
        if name == "UHF":
            extra = "  |<E_k|UHF>|^2 = " + " ".join(f"{(Vv[:, k] @ psi) ** 2:.3e}" for k in range(len(W)))
        print(f"  {name:>14s}: {s}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
