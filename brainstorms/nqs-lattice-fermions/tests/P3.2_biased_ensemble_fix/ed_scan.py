#!/usr/bin/env python3
"""
P3.2 round 3, step 1: search for a near-degenerate SAME-IRREP avoided crossing (ED only).
=========================================================================================

Builds, for one cluster and filling, the exact symmetry-adapted basis of ONE irrep
(1-D real irrep chi of the full space group x total spin S), forms the Hamiltonian pieces
    H = t_x H_tx + t_y H_ty + t' H_t' + U H_U
inside it, and scans a 2-D plane of two of the couplings (t_y, t', U; t_x = 1 is the unit)
at the cost of one small eigensolve per point. H restricted to one irrep is real symmetric,
so exact degeneracies inside one irrep have codimension 2 (von Neumann-Wigner): in a 2-D
plane they are isolated conical intersections (CIs). The scan locates the minimum in-irrep
gap E1 - E0 between the two lowest spin-S levels; `--refine 1` minimises it with
Nelder-Mead inside the scanned box and then runs a codimension test (gap at distance 1e-3
and 2e-3 in six directions: an isolated CI gives a gap growing linearly in EVERY direction;
a hidden symmetry would give a seam, i.e. a direction with zero gap).

Usage
-----
  python3 ed_scan.py --geom 4x3pp --nup 4 --ndn 4 --x tp,-0.5,-0.1,21 --y ty,0.6,1.4,17 \
                     --fix U=8 [options]

Options
-------
  --geom STR      Cluster: "4x3pp" (4x3 torus), "6x2po" (6x2 ladder, periodic legs, open
                  rungs), "4x3po" (4x3 cylinder, periodic x, open y), "3x4pp", "3x3pp",
                  "4x2po". Default 4x3pp.
  --nup, --ndn    Electron numbers (default 4, 4).
  --x, --y        Scan axes "name,start,stop,npoints" with name in {tp, ty, U}
                  (defaults "tp,-0.6,0,25" and "U,2,16,15").
  --fix STR       Values of the third coupling(s), e.g. "ty=1" or "U=8" (default "ty=1,U=8,tp=0";
                  scan axes override).
  --irrep STR     Space-group irrep: "gs" (default; read from the global ground state at the
                  centre of the scan box) or "Tx=1,Ty=1,Mx=1,My=1" (generator characters).
  --S INT         Total spin of the irrep (default 0).
  --nlev INT      Spin-S levels to report per point (default 3).
  --dense-max INT Space-group irrep dimension up to which the spin-S subspace is built
                  densely (exact) (default 7000). Above it, each point uses sparse eigsh on
                  the space-group irrep and keeps the lowest levels with <S^2> = S(S+1).
  --refine {0,1}  1: Nelder-Mead minimisation of the gap from the best grid point (default 1).
  --char {0,1}    1: characterise the two lowest irrep states (spin/charge structure factor
                  maxima, d-wave pair correlation Pd, <S^2>) at the refined point and at
                  +-0.03 along x, and compare the irrep E0 with the global ground state.
  --out PATH      Optional .npz with the gap grid.

Self-checks printed: sector closure ||H B y - B H_s y|| for a random sector vector y, and the
irrep E0 versus the global (fixed N_up, N_dn) E0 at the box centre.
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p11_core as P  # noqa: E402

GEOMS = {"4x3pp": (4, 3, "pbc", "pbc"), "3x4pp": (3, 4, "pbc", "pbc"),
         "6x2po": (6, 2, "pbc", "obc"), "4x3po": (4, 3, "pbc", "obc"),
         "3x3pp": (3, 3, "pbc", "pbc"), "4x2po": (4, 2, "pbc", "obc")}
NAMES = ("tx", "ty", "tp", "U")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ lattice pieces
def hop_mats(Lx, Ly, bcx, bcy):
    """Hopping matrices (H_kin = -sum T_ij c^dag_i c_j) for x bonds, y bonds, diagonals."""
    V = Lx * Ly
    idx = lambda x, y: x + Lx * y
    T = {k: np.zeros((V, V)) for k in ("tx", "ty", "tp")}

    def add(k, x, y, dx, dy):
        x2, y2 = x + dx, y + dy
        if not 0 <= x2 < Lx:
            if bcx == "obc":
                return
            x2 %= Lx
        if not 0 <= y2 < Ly:
            if bcy == "obc":
                return
            y2 %= Ly
        i, j = idx(x, y), idx(x2, y2)
        if i != j:
            T[k][i, j] += 1.0
            T[k][j, i] += 1.0

    for y in range(Ly):
        for x in range(Lx):
            add("tx", x, y, 1, 0)
            add("ty", x, y, 0, 1)
            add("tp", x, y, 1, 1)
            add("tp", x, y, 1, -1)
    return T


def site_group(Lx, Ly, bcx, bcy, T_test):
    """Site permutations (optional x<->y swap) o (mirror, then translate) that leave every
    matrix in T_test invariant."""
    out = []
    for sw in ((0, 1) if (Lx == Ly and bcx == bcy) else (0,)):
        for cx in (0, 1):
            for cy in (0, 1):
                for a in (range(Lx) if bcx == "pbc" else [0]):
                    for b in (range(Ly) if bcy == "pbc" else [0]):
                        p = np.empty(Lx * Ly, dtype=np.int64)
                        for y in range(Ly):
                            for x in range(Lx):
                                x2 = ((-x) % Lx if bcx == "pbc" else Lx - 1 - x) if cx else x
                                y2 = ((-y) % Ly if bcy == "pbc" else Ly - 1 - y) if cy else y
                                x3, y3 = (x2 + a) % Lx, (y2 + b) % Ly
                                if sw:
                                    x3, y3 = y3, x3
                                p[x + Lx * y] = x3 + Lx * y3
                        if all(np.allclose(T[np.ix_(p, p)], T) for T in T_test):
                            out.append(((a, b, cx, cy, sw), p))
    return out


def species_perm(states, rows, lookup, p):
    """Image index and fermionic sign of every species basis state under site perm p."""
    newidx = np.empty(len(states), dtype=np.int64)
    sgn = np.empty(len(states))
    for k, occ in enumerate(rows):
        img = p[occ]
        perm = np.argsort(img, kind="stable")
        seen = np.zeros(len(perm), bool)
        par = 0
        for i in range(len(perm)):
            if not seen[i]:
                j, L = i, 0
                while not seen[j]:
                    seen[j] = True
                    j = perm[j]
                    L += 1
                par += L - 1
        b = 0
        for i in img:
            b |= 1 << int(i)
        newidx[k] = lookup[b]
        sgn[k] = -1.0 if par % 2 else 1.0
    return newidx, sgn


class SpinOp:
    """S^+ on the fixed-(N_up, N_dn) basis (logic copied from tests/P1.1P_projected Spin)."""

    def __init__(self, m):
        V, nu, nd = m.V, m.nup, m.ndn
        self.Sz = 0.5 * (nu - nd)
        su, sd = m.su, m.sd
        st_u2, lk_u2, _, _ = P.species_basis(V, nu + 1)
        st_d2, lk_d2, _, _ = P.species_basis(V, nd - 1)
        Dd2 = len(st_d2)
        pc = lambda s, i: np.array([bin(int(x) & ((1 << i) - 1)).count("1") for x in s])
        r, c, v = [], [], []
        for i in range(V):
            au = np.nonzero(((su >> i) & 1) == 0)[0]
            ad = np.nonzero(((sd >> i) & 1) == 1)[0]
            if len(au) == 0 or len(ad) == 0:
                continue
            newu = lk_u2[su[au] | (1 << i)]
            newd = lk_d2[sd[ad] ^ (1 << i)]
            r.append((newu[:, None] * Dd2 + newd[None, :]).ravel())
            c.append((au[:, None] * m.Dd + ad[None, :]).ravel())
            v.append(((-1.0) ** pc(su[au], i))[:, None] * ((-1.0) ** pc(sd[ad], i))[None, :])
        self.Sp = sp.csr_matrix((np.concatenate(v).ravel(),
                                 (np.concatenate(r), np.concatenate(c))),
                                shape=(len(st_u2) * Dd2, m.D))

    def S2exp(self, psi):
        x = self.Sp @ psi
        return float((x @ x + (self.Sz ** 2 + self.Sz) * (psi @ psi)) / (psi @ psi))


class Cluster:
    """Hubbard pieces, space group, spin, and symmetry-adapted irrep basis."""

    def __init__(self, geom, nup, ndn):
        Lx, Ly, bcx, bcy = GEOMS[geom]
        self.geom = geom
        m = P.Hubbard(Lx, Ly, bcx, bcy, nup, ndn, 1.0, 0.0, 0.0)
        self.m = m
        self.V, self.D = m.V, m.D
        T = hop_mats(Lx, Ly, bcx, bcy)
        self.Tm = T
        self.parts_full = {}
        for k in ("tx", "ty", "tp"):
            Hu = P.species_hop(m.V, nup, T[k])
            Hd = P.species_hop(m.V, ndn, T[k])
            self.parts_full[k] = (sp.kron(Hu, sp.identity(m.Dd))
                                  + sp.kron(sp.identity(m.Du), Hd)).tocsr()
        self.parts_full["U"] = sp.diags(m.dbl).tocsr()
        elems = site_group(Lx, Ly, bcx, bcy, [T["tx"], T["ty"], T["tp"]])
        st_u, lk_u, _, rows_u = P.species_basis(m.V, nup)
        st_d, lk_d, _, rows_d = P.species_basis(m.V, ndn)
        self.elems, self.maps = [], []
        for e, p in elems:
            mu, su = species_perm(st_u, rows_u, lk_u, p)
            md, sd = species_perm(st_d, rows_d, lk_d, p)
            self.elems.append(e)
            self.maps.append(((mu[:, None] * m.Dd + md[None, :]).ravel(),
                              (su[:, None] * sd[None, :]).ravel()))
        self.G = len(self.elems)
        self.spin = SpinOp(m)

    def H(self, c):
        return sum(c[k] * self.parts_full[k] for k in NAMES if c.get(k, 0.0) != 0.0)

    def Ug(self, k, psi):
        img, sgn = self.maps[k]
        out = np.zeros_like(psi)
        out[img] = sgn * psi
        return out

    def chi_of(self, psi):
        psi = psi / np.linalg.norm(psi)
        return np.array([psi @ self.Ug(k, psi) for k in range(self.G)])

    def build_irrep(self, chi, S, dense_max):
        D, G = self.D, self.G
        rows = np.concatenate([self.maps[k][0] for k in range(G)])
        cols = np.tile(np.arange(D), G)
        vals = np.concatenate([chi[k] * self.maps[k][1] for k in range(G)]) / G
        Pm = sp.csr_matrix((vals, (rows, cols)), shape=(D, D))
        reps = np.unique(np.stack([self.maps[k][0] for k in range(G)]).min(0))
        B = Pm[:, reps].tocsc()
        nrm = np.sqrt(np.asarray(B.multiply(B).sum(0)).ravel())
        keep = nrm > 1e-12
        self.B = (B[:, keep] @ sp.diags(1.0 / nrm[keep])).tocsr()
        self.dB = self.B.shape[1]
        self.S = S
        self.SpB = (self.spin.Sp @ self.B).tocsr()
        sec = {k: (self.B.T @ (M @ self.B)).tocsr() for k, M in self.parts_full.items()}
        self.dense = self.dB <= dense_max
        if self.dense:
            S2 = (self.SpB.T @ self.SpB).toarray() + (self.spin.Sz ** 2 + self.spin.Sz) * np.eye(self.dB)
            w, v = eigh(S2, driver="evd")
            self.Q = np.linalg.qr(v[:, np.abs(w - S * (S + 1)) < 1e-6])[0]
            self.parts = {k: self.Q.T @ (M @ self.Q) for k, M in sec.items()}
            self.dS = self.Q.shape[1]
        else:
            self.Q = None
            self.parts = sec
            self.dS = -1
        return self.dB, self.dS

    def sector_eigs(self, c, k):
        """Lowest k spin-S levels of the irrep: energies and coefficient vectors."""
        Hs = sum(c[n] * self.parts[n] for n in NAMES if c.get(n, 0.0) != 0.0)
        if self.dense:
            return eigh(Hs, subset_by_index=[0, k - 1])
        kk = k + 6
        while True:
            w, v = eigsh(Hs, k=kk, which="SA", tol=1e-11,
                         v0=np.random.default_rng(0).standard_normal(self.dB))
            o = np.argsort(w)
            w, v = w[o], v[:, o]
            s2 = np.array([np.sum((self.SpB @ v[:, j]) ** 2) for j in range(kk)])
            s2 += self.spin.Sz ** 2 + self.spin.Sz
            sel = np.nonzero(np.abs(s2 - self.S * (self.S + 1)) < 1e-4)[0]
            if len(sel) >= k or kk >= 60:
                sel = sel[:k]
                return w[sel], v[:, sel]
            kk *= 2

    def full_vec(self, v):
        return self.B @ (self.Q @ v if self.dense else v)


# ------------------------------------------------------------------ characterisation
def structure(m, psi):
    iu, idd = m.iu_id()
    nu = m.occ_u[iu].astype(float)
    nd = m.occ_d[idd].astype(float)
    p = psi ** 2 / (psi @ psi)
    qs = [(kx, ky) for ky in range(m.Ly) for kx in range(m.Lx)]
    Q = np.array([[2 * np.pi * kx / m.Lx, 2 * np.pi * ky / m.Ly] for kx, ky in qs])
    ph = np.exp(1j * m.coords @ Q.T)
    sz = (nu - nd) @ ph / 2
    rho = (nu + nd) @ ph
    Ss = (p @ np.abs(sz) ** 2) / m.V
    Nc = (p @ np.abs(rho) ** 2) / m.V
    return {q: (Ss[i], Nc[i]) for i, q in enumerate(qs)}


def describe(cl, psi):
    m = cl.m
    st = structure(m, psi)
    qs = max((q for q in st if q != (0, 0)), key=lambda q: st[q][0])
    qc = max((q for q in st if q != (0, 0)), key=lambda q: st[q][1])
    if not hasattr(m, "pair_ops"):
        m.pair_setup(1.5)
    return (f"Ss_max q={qs} {st[qs][0]:.3f} | N_max q={qc} {st[qc][1]:.3f} | "
            f"Pd={m.dwave(psi):+.2e} | S2={cl.spin.S2exp(psi):.3f}")


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="4x3pp", choices=sorted(GEOMS))
    ap.add_argument("--nup", type=int, default=4)
    ap.add_argument("--ndn", type=int, default=4)
    ap.add_argument("--x", default="tp,-0.6,0,25")
    ap.add_argument("--y", default="U,2,16,15")
    ap.add_argument("--fix", default="ty=1,U=8,tp=0")
    ap.add_argument("--irrep", default="gs")
    ap.add_argument("--S", type=int, default=0)
    ap.add_argument("--nlev", type=int, default=3)
    ap.add_argument("--dense-max", type=int, default=7000)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--char", type=int, default=1)
    ap.add_argument("--out", default="")
    return ap.parse_args(argv)


def axis(s):
    n, a, b, k = s.split(",")
    return n, np.linspace(float(a), float(b), int(k))


def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    cl = Cluster(a.geom, a.nup, a.ndn)
    base = {"tx": 1.0, "ty": 1.0, "tp": 0.0, "U": 8.0}
    for kv in a.fix.split(","):
        k, v = kv.split("=")
        base[k] = float(v)
    xn, xs = axis(a.x)
    yn, ys = axis(a.y)
    coup = lambda x, y: {**base, xn: x, yn: y}
    log(f"geom={a.geom} nup={a.nup} ndn={a.ndn} D={cl.D} |G|={cl.G} base={base} "
        f"x={xn} y={yn} ({time.time() - t0:.0f}s)")
    c0 = coup(xs[len(xs) // 2], ys[len(ys) // 2])
    w0, v0 = eigsh(cl.H(c0), k=4, which="SA", tol=1e-10,
                   v0=np.random.default_rng(0).standard_normal(cl.D))
    o = np.argsort(w0)
    w0, v0 = w0[o], v0[:, o]
    if a.irrep == "gs":
        chi = cl.chi_of(v0[:, 0])
        if np.any(np.abs(np.abs(chi) - 1) > 1e-6):
            log(f"global ground state at {c0} is not in a 1-D irrep; E0..3={w0}")
            return
        chi = np.round(chi)
    else:
        lam = {k: float(v) for k, v in (kv.split("=") for kv in a.irrep.split(","))}
        chi = np.array([lam.get("Tx", 1) ** e[0] * lam.get("Ty", 1) ** e[1]
                        * lam.get("Mx", 1) ** e[2] * lam.get("My", 1) ** e[3]
                        * lam.get("Sw", 1) ** e[4] for e in cl.elems])
    gens = {e: c for e, c in zip(cl.elems, chi) if sum(e) == 1}
    log(f"irrep generators chi={gens}; global E0..3 at centre {np.round(w0, 6)} "
        f"S2(gs)={cl.spin.S2exp(v0[:, 0]):.3f}")
    dB, dS = cl.build_irrep(chi, a.S, a.dense_max)
    ws, vs = cl.sector_eigs(c0, 1)
    y = vs[:, 0]
    clos = np.linalg.norm(cl.H(c0) @ cl.full_vec(y) - ws[0] * cl.full_vec(y))
    log(f"irrep basis: space-group dim {dB}, spin-S dim {dS} (dense={cl.dense}); "
        f"residual ||H v - E v||={clos:.1e}; irrep E0={ws[0]:.8f} vs global E0={w0[0]:.8f} "
        f"({time.time() - t0:.0f}s)")
    gap = np.full((len(xs), len(ys)), np.nan)
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            w, _ = cl.sector_eigs(coup(xv, yv), a.nlev)
            gap[i, j] = w[1] - w[0]
            print(f"{xn}={xv:+.4f} {yn}={yv:7.4f} E=" + " ".join(f"{e:.6f}" for e in w)
                  + f" gap={gap[i, j]:.5f} gap/V={gap[i, j] / cl.V:.5f}", flush=True)
    i, j = np.unravel_index(np.nanargmin(gap), gap.shape)
    best = np.array([xs[i], ys[j]])
    bg = gap[i, j]
    log(f"grid minimum gap {bg:.5f} ({bg / cl.V:.5f}/site) at {xn}={best[0]:+.4f} {yn}={best[1]:.4f}")
    if a.refine:
        lo, hi = np.array([xs.min(), ys.min()]), np.array([xs.max(), ys.max()])

        def f(z):
            zc = np.clip(z, lo, hi)
            pen = np.sum(np.abs(z - zc))
            return np.diff(cl.sector_eigs(coup(*zc), 2)[0])[0] + 10.0 * pen

        r = minimize(f, x0=best, method="Nelder-Mead",
                     options=dict(xatol=1e-7, fatol=1e-10, maxiter=800))
        best, bg = np.clip(r.x, lo, hi), r.fun
        log(f"refined minimum gap {bg:.3e} ({bg / cl.V:.3e}/site) at {xn}={best[0]:+.6f} "
            f"{yn}={best[1]:.6f}")
        for ang in np.linspace(0, np.pi, 7)[:-1]:
            dv = np.array([np.cos(ang), np.sin(ang)]) * 1e-3
            gp = [np.diff(cl.sector_eigs(coup(*(best + s_ * dv)), 2)[0])[0] for s_ in (1.0, 2.0)]
            print(f"codim angle={ang:.2f} gap(+1e-3)={gp[0]:.3e} gap(+2e-3)={gp[1]:.3e}", flush=True)
    if a.char:
        for dx in (-0.03, 0.0, 0.03):
            c = coup(best[0] + dx, best[1])
            w, v = cl.sector_eigs(c, 2)
            wg, _ = eigsh(cl.H(c), k=1, which="SA", tol=1e-10,
                          v0=np.random.default_rng(0).standard_normal(cl.D))
            for lv in range(2):
                print(f"char {xn}={c[xn]:+.4f} {yn}={c[yn]:.4f} level {lv} E={w[lv]:.6f} "
                      + describe(cl, cl.full_vec(v[:, lv]))
                      + (f" | global E0={wg[0]:.6f}" if lv == 0 else ""), flush=True)
    if a.out:
        np.savez(a.out, xs=xs, ys=ys, gap=gap, xn=xn, yn=yn)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
