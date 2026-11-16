#!/usr/bin/env python3
"""Exploratory pre-check (not the decisive test): symmetry labels of the lowest ED levels.

Usage: python3 explore_sectors.py TP U K [NUP]
  TP   next-nearest hopping t' (float)
  U    on-site repulsion (float)
  K    number of lowest eigenpairs of the fixed-(N_up, N_dn) sector (int)
  NUP  N_up = N_dn (int, default 4)
Prints, per level: energy, <T_x>, <T_y>, <M_x>, <M_y> (expectations; a sector eigenstate
gives a unit-modulus value, a degenerate multiplet gives a fraction), <S^2>, and the d-wave
long-range pair correlation Pd of P1.1. 4x3 PBC lattice only.
"""
import sys
import time

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/P1.1_sign_vs_amplitude")
import micro_test as m  # noqa: E402


def site_perm_species(states, lookup, V, perm):
    D = len(states)
    idx = np.empty(D, np.int64)
    sg = np.empty(D)
    for k, b in enumerate(states):
        occ = [i for i in range(V) if (b >> i) & 1]
        img = [perm[i] for i in occ]
        inv = sum(1 for a in range(len(img)) for c in range(a + 1, len(img)) if img[a] > img[c])
        nb = 0
        for i in img:
            nb |= 1 << i
        idx[k] = lookup[nb]
        sg[k] = (-1) ** inv
    return idx, sg


def op_apply(model, pu, pd, psi):
    Psi = psi.reshape(model.Du, model.Dd)
    out = np.zeros_like(Psi)
    out[np.ix_(pu[0], pd[0])] = Psi * pu[1][:, None] * pd[1][None, :]
    return out.ravel()


def perms(Lx, Ly):
    idx = lambda x, y: (x % Lx) + Lx * (y % Ly)
    return {
        "Tx": [idx(x + 1, y) for y in range(Ly) for x in range(Lx)],
        "Ty": [idx(x, y + 1) for y in range(Ly) for x in range(Lx)],
        "Mx": [idx(-x, y) for y in range(Ly) for x in range(Lx)],
        "My": [idx(x, -y) for y in range(Ly) for x in range(Lx)],
    }


def S2(model, psi):
    V = model.V
    Psi = psi.reshape(model.Du, model.Dd)
    acc = None
    for i in range(V):
        cu = m.species_annihilate(V, model.nup + 1, i).T
        ad = m.species_annihilate(V, model.ndn, i)
        t = cu @ (ad @ Psi.T).T
        acc = t if acc is None else acc + t
    return float(np.sum(np.asarray(acc) ** 2) / (psi @ psi))


def main():
    tp, U, k = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    nup = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    t0 = time.time()
    model = m.Hubbard(4, 3, "pbc", "pbc", nup, nup, 1.0, tp, U)
    model.pair_setup(1.5)
    su, lu, _, _ = m.species_basis(12, nup)
    ops = {n: site_perm_species(su, lu, 12, p) for n, p in perms(4, 3).items()}
    w, v = eigsh(model.H, k=k, which="SA", tol=1e-10,
                 v0=np.random.default_rng(0).standard_normal(model.D))
    o = np.argsort(w)
    w, v = w[o], v[:, o]
    print(f"# tp={tp} U={U} n={nup} D={model.D} t={time.time() - t0:.1f}s")
    for j in range(k):
        psi = v[:, j]
        q = {n: psi @ op_apply(model, a, a, psi) for n, a in ops.items()}
        print(f"{j} E={w[j]:.6f} dE={w[j] - w[0]:.4f} Tx={q['Tx']:+.3f} Ty={q['Ty']:+.3f} "
              f"Mx={q['Mx']:+.3f} My={q['My']:+.3f} S2={S2(model, psi):.3f} "
              f"Pd={model.dwave(psi):+.2e}")


def group_ops(nup):
    """All 48 elements of translations x {1, Mx, My, MxMy} as (idx, sign) species maps."""
    Lx, Ly = 4, 3
    su, lu, _, _ = m.species_basis(12, nup)
    els = []
    for a in range(Lx):
        for b in range(Ly):
            for sx in (1, -1):
                for sy in (1, -1):
                    p = [((sx * x + a) % Lx) + Lx * ((sy * y + b) % Ly)
                         for y in range(Ly) for x in range(Lx)]
                    els.append(site_perm_species(su, lu, 12, p))
    return els


def sector_levels(tp, U, k, nup=4, lam=1.0):
    """Lowest k levels in the K=0, Mx=My=+1, S=0 sector (projected Lanczos with an
    S^2 penalty lam*S^2 and a shift 50 on the projected-out complement)."""
    from scipy.sparse.linalg import LinearOperator
    model = m.Hubbard(4, 3, "pbc", "pbc", nup, nup, 1.0, tp, U)
    model.pair_setup(1.5)
    els = group_ops(nup)
    V = model.V
    Sp = [(m.species_annihilate(V, nup + 1, i).T.tocsr(), m.species_annihilate(V, nup, i).tocsr())
          for i in range(V)]

    def proj(x):
        acc = np.zeros_like(x)
        for e in els:
            acc += op_apply(model, e, e, x)
        return acc / len(els)

    def s2(x):
        X = x.reshape(model.Du, model.Dd)
        Y = None
        for cu, ad in Sp:
            t = cu @ (ad @ X.T).T
            Y = t if Y is None else Y + t
        Y = np.asarray(Y)
        Z = None
        for cu, ad in Sp:
            t = cu.T @ (ad.T @ Y.T).T
            Z = t if Z is None else Z + t
        return np.asarray(Z).ravel()

    def mv(x):
        x = np.asarray(x).ravel()
        px = proj(x)
        y = model.H @ px + lam * s2(px)
        return proj(y) + 50.0 * (x - px)

    A = LinearOperator((model.D, model.D), matvec=mv, dtype=float)
    v0 = proj(np.random.default_rng(1).standard_normal(model.D))
    w, v = eigsh(A, k=k, which="SA", tol=1e-9, v0=v0)
    out = []
    for j in np.argsort(w):
        psi = proj(v[:, j])
        E = psi @ (model.H @ psi) / (psi @ psi)
        out.append((E, S2(model, psi), model.dwave(psi), psi / np.linalg.norm(psi)))
    return model, out


if __name__ == "__main__":
    main()
