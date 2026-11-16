#!/usr/bin/env python3
"""
P2.1 round-7 tier-1 (T1) toy computations for the refilled route portfolio (Stage 3.5).
=====================================================================================

Kill rules pre-registered in the leaf artifact (section "Round 7", before any run):
KILL if rel. error > 1e-6, or slope of t_c (time per connection) in N_e >= 1.8, or the printed
operation count per connection grows faster than linearly in N_e. PASS if rel. error <= 1e-9 and
the operation count is linear in N_e. A timing slope below the operation-count prediction is
overhead-bound (round-6 Minor 2) and is never a kill.

Families (one augmented ("hidden-fermion") matrix per determinant; V hidden fermions, one per
site m; local features h_m = tanh(W x_patch(m) + b) in R^F, both spins in the patch, OBC-aware
patches from lh_geom.py):

  SH-HFDS  (Route 7.3; --lam 0 gives LH-HFDS, Route 6.1')   Slater frame, per spin s:
      M_s = [[ Phi0_s[occ_s],      K_s[occ_s, :] ],
             [ -C_s(n)[:, :N_s],   I + L_s(n)    ]]          psi = prod_s det M_s
      C_s[m, :] = h_m . Omega_{s,m}  (Omega_{s,m}: F x N_s),
      L_s[m, nb_d(m)] = h_m . lam_{s,m,d}  (d = +x, -x, +y, -y; sparse local hidden-hidden block).
      det M_s = det(I + L_s) det(Phi0_s[occ] + K_s[occ] (I + L_s)^{-1} C_s): the resolvent
      (I + L)^{-1} gives products of local features along lattice paths (non-local, nonlinear),
      while a hop changes only 1 + b rows of M_s (b = number of patches containing a touched site).
  LH-HFPS  (Route 7.1)   singlet-pairing (AGP) frame, N_up = N_dn = N:
      M = [[ F0[occ_up, occ_dn],   K[occ_up, :] ],
           [ -C(n)[:, occ_dn],     I            ]]           psi = det M
      C[m, d] = h_m . Omega_m[:, d]  (Omega_m: F x V). An up hop changes one row, a down hop one
      column, and both change the b hidden rows: rank <= 2 + b.

Fast ratio (both families): carried M^{-1} plus row/column Green's functions
      Gr = [rows of all sites] M^{-1}  (V x n),  Gc = M^{-1} [columns of all sites]  (n x V),
      X = Gr [columns of all sites]  (V x V, AGP only).
Rows: S = I + D M^{-1}[:, R] (D = changed rows, |R| = 1 + b); column (AGP down hop): the
two-stage formula documented in ratio_agp(). Cost per connection O(b F N + n b^2), i.e. linear in
N_e at fixed (F, b); the op count printed is the multiply-add count of the implemented steps.
Brute force: every connected amplitude rebuilt from scratch with sorted rows and columns and
explicit Jordan-Wigner signs (tj_core.connections convention), independent of the fast algebra.

Model: t-t'-J, t = 1, t' = -0.25, J = 0.4, random configuration, random weights scaled so that
|E_loc| = O(N_e). Exactness on L = 4, 6, 8 PBC (doping 1/8) and on the 5x3 OBC 6+6 instance-4
geometry; timing on --sizes (PBC).

Options
  --route {sh,lh,hfps,all}  sh = SH-HFDS (7.3), lh = LH-HFDS (6.1', lam = 0), hfps = LH-HFPS (7.1)
  --sizes LIST    L for timing (default 16,24,32)
  --exact-L LIST  PBC L for exactness (default 4,6,8); the 5x3 OBC geometry is always added
  --patch KIND    site | plus | 3x3 (default plus)
  --F INT         feature channels (default 8)
  --nconf INT     configurations per timing size (default 1)
  --seed INT      RNG seed (default 7)
Threads: set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS <= 4. Output: printed log.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lh_geom import geometry, neighbours4, patch_index, patch_input  # noqa: E402
from tj_core import connections  # noqa: E402  (path added by lh_geom)

TP, JX = -0.25, 0.4


def filling(V):
    nh = int(round(V / 8))
    if (V - nh) % 2:
        nh += 1
    return (V - nh) // 2


def rand_cfg(V, n, rng):
    p = rng.permutation(V)
    ou = np.zeros(V, int); od = np.zeros(V, int)
    ou[p[:n]] = 1; od[p[n:2 * n]] = 1
    return ou, od


def conn_list(T, nn, ou, od):
    diag, hops, exch = connections(T, nn, JX, ou, od)
    out = [(c, [(s, x, y)]) for (s, x, y, c) in hops]
    out += [(-0.5 * JX, [(0, b, a), (1, a, b)]) for (a, b) in exch]
    return diag, out


def apply_moves(ou, od, moves):
    u, d = ou.copy(), od.copy()
    for (s, x, y) in moves:
        o = u if s == 0 else d
        o[x] = 0; o[y] = 1
    return u, d


def eloc_brute(amp, T, nn, ou, od):
    diag, hops, exch = connections(T, nn, JX, ou, od)
    p0 = amp(ou, od)
    E = diag
    for (s, x, y, c) in hops:
        o = (ou if s == 0 else od).copy()
        sg = (-1) ** int(o[min(x, y) + 1:max(x, y)].sum())
        o[x] = 0; o[y] = 1
        E += c * sg * (amp(o, od) if s == 0 else amp(ou, o)) / p0
    for (a, b) in exch:
        u2, d2 = ou.copy(), od.copy()
        su = (-1) ** int(ou[min(a, b) + 1:max(a, b)].sum())
        sd = (-1) ** int(od[min(a, b) + 1:max(a, b)].sum())
        u2[b] = 0; u2[a] = 1; d2[a] = 0; d2[b] = 1
        E += -0.5 * JX * su * sd * amp(u2, d2) / p0
    return E


class Local:
    def __init__(self, P, F, rng):
        self.P, self.F = P, F
        p = P.shape[1]
        self.W = rng.standard_normal((2 * p, F)) / np.sqrt(2 * p)
        self.b = 0.1 * rng.standard_normal(F)
        V = P.shape[0]
        self.inv = [np.nonzero((P == s).any(1))[0] for s in range(V)]

    def feats(self, ou, od, sites=None):
        P = self.P if sites is None else self.P[sites]
        return np.tanh(patch_input(ou, od, P) @ self.W + self.b)

    def changed(self, touched):
        return np.unique(np.concatenate([self.inv[s] for s in touched]))


# ------------------------------------------------------------------ SH-HFDS / LH-HFDS (Slater frame)
class SHHFDS:
    def __init__(self, T, Lx, Ly, bc, n, F, patch, lam, rng):
        V = T.shape[0]
        self.V, self.n, self.F = V, n, F
        e, v = np.linalg.eigh(-T)
        self.loc = Local(patch_index(Lx, Ly, bc, patch), F, rng)
        self.nb = neighbours4(Lx, Ly, bc)
        self.P0 = [v[:, :n] + 0.1 * rng.standard_normal((V, n)) / np.sqrt(V) for _ in range(2)]
        self.K = [0.5 * rng.standard_normal((V, V)) / np.sqrt(V) for _ in range(2)]
        self.Om = [0.5 * rng.standard_normal((V, F, n)) / np.sqrt(F * n) for _ in range(2)]
        self.lam = [lam * rng.standard_normal((V, 4, F)) / np.sqrt(F) for _ in range(2)]
        self.ops = 0

    def build(self, s, ou, od, H):
        o = ou if s == 0 else od
        rows = np.nonzero(o)[0]
        V, N = self.V, len(rows)
        M = np.zeros((N + V, N + V))
        M[:N, :N] = self.P0[s][rows]
        M[:N, N:] = self.K[s][rows]
        M[N:, :N] = -np.einsum("mf,mfn->mn", H, self.Om[s])
        M[N:, N:] = np.eye(V)
        cp = np.einsum("mf,mdf->md", H, self.lam[s])
        for d in range(4):
            ok = self.nb[:, d] >= 0
            M[N + np.nonzero(ok)[0], N + self.nb[ok, d]] += cp[ok, d]
        return M, rows

    def amp(self, ou, od):
        H = self.loc.feats(ou, od)
        return np.prod([np.linalg.det(self.build(s, ou, od, H)[0]) for s in (0, 1)])

    def setup(self, ou, od):
        self.ou, self.od = ou, od
        self.H = self.loc.feats(ou, od)
        self.c = []
        for s in (0, 1):
            M, rows = self.build(s, ou, od, self.H)
            Mi = np.linalg.inv(M)
            allrows = np.hstack([self.P0[s], self.K[s]])            # V x n
            self.c.append(dict(Mi=Mi, Gr=allrows @ Mi, N=len(rows),
                               slot={int(x): i for i, x in enumerate(rows)}))

    def ratio(self, moves):
        ou, od = apply_moves(self.ou, self.od, moves)
        B = self.loc.changed([x for (_, x, _) in moves] + [y for (_, _, y) in moves])
        dH = self.loc.feats(ou, od, B) - self.H[B]                  # b x F
        b, F = len(B), self.F
        mv = {s: (x, y) for (s, x, y) in moves}
        rho = 1.0
        for s in (0, 1):
            c = self.c[s]
            N, V = c["N"], self.V
            n = N + V
            Dh = np.zeros((b, n))
            Dh[:, :N] = -np.einsum("bf,bfn->bn", dH, self.Om[s][B])   # O(b F N)
            dcp = np.einsum("bf,bdf->bd", dH, self.lam[s][B])
            for d in range(4):
                nbB = self.nb[B, d]
                ok = nbB >= 0
                Dh[np.nonzero(ok)[0], N + nbB[ok]] += dcp[ok, d]
            Rh = N + B
            if s in mv:
                x, y = mv[s]
                i = c["slot"][x]
                R = np.concatenate([[i], Rh])
                S = np.eye(1 + b)
                S[0] += c["Gr"][y, R]
                S[0, 0] -= 1.0                                         # Gr[x] = e_i
                S[1:] += Dh @ c["Mi"][:, R]                            # O(n b (1+b))
            else:
                R = Rh
                S = np.eye(b) + Dh @ c["Mi"][:, R]
            self.ops += b * F * N + n * b * len(R) + len(R) ** 3
            rho *= np.linalg.det(S)
        return rho


# ------------------------------------------------------------------ LH-HFPS (singlet-pairing frame)
class LHHFPS:
    def __init__(self, T, Lx, Ly, bc, n, F, patch, rng):
        V = T.shape[0]
        self.V, self.n, self.F = V, n, F
        e, v = np.linalg.eigh(-T)
        # pairing function: Fermi-sea pair amplitude + noise (a singlet geminal)
        self.F0 = v[:, :n] @ v[:, :n].T + 0.1 * rng.standard_normal((V, V)) / np.sqrt(V)
        self.loc = Local(patch_index(Lx, Ly, bc, patch), F, rng)
        self.K = 0.5 * rng.standard_normal((V, V)) / np.sqrt(V)
        self.Om = 0.5 * rng.standard_normal((V, F, V)) / np.sqrt(F * V)
        self.ops = 0

    def build(self, ou, od, H):
        ru, rd = np.nonzero(ou)[0], np.nonzero(od)[0]
        V, N = self.V, len(ru)
        assert len(rd) == N
        C = np.einsum("mf,mfv->mv", H, self.Om)                      # V x V
        M = np.zeros((N + V, N + V))
        M[:N, :N] = self.F0[np.ix_(ru, rd)]
        M[:N, N:] = self.K[ru]
        M[N:, :N] = -C[:, rd]
        M[N:, N:] = np.eye(V)
        return M, ru, rd, C

    def amp(self, ou, od):
        return np.linalg.det(self.build(ou, od, self.loc.feats(ou, od))[0])

    def setup(self, ou, od):
        self.ou, self.od = ou, od
        self.H = self.loc.feats(ou, od)
        M, ru, rd, C = self.build(ou, od, self.H)
        self.N = len(ru)
        self.Mi = np.linalg.inv(M)
        self.rd = rd
        allrows = np.hstack([self.F0[:, rd], self.K])                 # V x n (row of an up at site y)
        allcols = np.vstack([self.F0[ru, :], -C])                    # n x V (column of a down at site y)
        self.Gr = allrows @ self.Mi
        self.Gc = self.Mi @ allcols
        self.X = self.Gr @ allcols
        self.su = {int(x): i for i, x in enumerate(ru)}
        self.sd = {int(x): j for j, x in enumerate(rd)}

    def ratio(self, moves):
        """Two-stage exact ratio.
        Stage 1 (rows R = {i (up row)} + hidden rows B), new rows evaluated on the OLD column labels:
          S = I + D M^{-1}[:, R],  rho1 = det S.
        Stage 2 (column j of the down particle x_d -> y_d), with delta_R(s) the change of the rows R
        at column-site s and w = delta_R(y_d) - delta_R(x_d):
          M^{-1} dc       = Gc[:, y_d] - e_j + M^{-1}[:, R] w
          a               = Gc[j, y_d] - 1 + M^{-1}[j, R] w
          D M^{-1} dc     = D Gc[:, y_d] - delta_R(x_d) + (S - I) w
          rho2            = 1 + a - M^{-1}[j, R] S^{-1} (D M^{-1} dc)."""
        ou, od = apply_moves(self.ou, self.od, moves)
        B = self.loc.changed([x for (_, x, _) in moves] + [y for (_, _, y) in moves])
        dH = self.loc.feats(ou, od, B) - self.H[B]
        b, F, N, V = len(B), self.F, self.N, self.V
        n = N + V
        mv = {s: (x, y) for (s, x, y) in moves}
        rd = self.rd
        Dh = np.zeros((b, n))
        Dh[:, :N] = -np.einsum("bf,bfv->bv", dH, self.Om[B][:, :, rd])   # O(b F N)
        Rh = N + B
        up = 0 in mv
        if up:
            xu, yu = mv[0]
            i = self.su[xu]
            R = np.concatenate([[i], Rh])
        else:
            R = Rh
        k = len(R)
        S = np.eye(k)
        if up:
            S[0] += self.Gr[yu, R]
            S[0, 0] -= 1.0
            S[1:] += Dh @ self.Mi[:, R]
        else:
            S += Dh @ self.Mi[:, R]
        rho = np.linalg.det(S)
        self.ops += b * F * N + n * b * k + k ** 3
        if 1 in mv:
            xd, yd = mv[1]
            j = self.sd[xd]

            def delta(sv):
                dh = -np.einsum("bf,bf->b", dH, self.Om[B][:, :, sv])
                return np.concatenate([[self.F0[yu, sv] - self.F0[xu, sv]], dh]) if up else dh
            dx_, dy_ = delta(xd), delta(yd)
            w = dy_ - dx_
            MjR = self.Mi[j, R]
            a = self.Gc[j, yd] - 1.0 + MjR @ w
            DGy = Dh @ self.Gc[:, yd]                                # O(b n)
            if up:
                DGy = np.concatenate([[self.X[yu, yd] - self.F0[xu, yd]], DGy])   # M[i] M^-1 cols = cols[i]
            bv = DGy - dx_ + (S - np.eye(k)) @ w
            rho *= 1.0 + a - MjR @ np.linalg.solve(S, bv)
            self.ops += b * n + 2 * b * F + k * k
        return rho


def make(route, T, Lx, Ly, bc, n, args, rng):
    if route == "hfps":
        return LHHFPS(T, Lx, Ly, bc, n, args.F, args.patch, rng)
    return SHHFDS(T, Lx, Ly, bc, n, args.F, args.patch, 0.0 if route == "lh" else 0.3, rng)


def eloc_fast(obj, T, nn, ou, od):
    diag, cl = conn_list(T, nn, ou, od)
    obj.ops = 0
    t0 = time.perf_counter()
    E = diag + sum(c * obj.ratio(m) for (c, m) in cl)
    return E, time.perf_counter() - t0, len(cl), obj.ops


def run_route(route, args):
    name = {"sh": "SH-HFDS (Route 7.3)", "lh": "LH-HFDS (Route 6.1', lam=0)", "hfps": "LH-HFPS (Route 7.1)"}[route]
    rng = np.random.default_rng(args.seed)
    print(f"== {name}: exactness (fast E_loc vs brute, sorted rows/cols + JW signs), patch={args.patch} F={args.F}")
    emax = 0.0
    geoms = [(L, L, "pbc", filling(L * L)) for L in args.exact_list] + [(5, 3, "obc", 6)]
    for (Lx, Ly, bc, n) in geoms:
        T, nn = geometry(Lx, Ly, bc, TP)
        obj = make(route, T, Lx, Ly, bc, n, args, rng)
        for _ in range(2):
            ou, od = rand_cfg(Lx * Ly, n, rng)
            obj.setup(ou, od)
            Ef, _, nc, _ = eloc_fast(obj, T, nn, ou, od)
            Eb = eloc_brute(obj.amp, T, nn, ou, od)
            e = abs(Ef - Eb) / (1 + abs(Eb))
            emax = max(emax, e)
            print(f"   {Lx}x{Ly} {bc} Ne={2 * n:4d} n_conn={nc:4d} E_fast={Ef:+.12f} E_brute={Eb:+.12f} rel_err={e:.1e}")
    print(f"== {name}: timing and operation count per connection")
    Ns, tcs, ocs = [], [], []
    for L in args.size_list:
        T, nn = geometry(L, L, "pbc", TP)
        n = filling(L * L)
        obj = make(route, T, L, L, "pbc", n, args, rng)
        tl, ol = [], []
        for _ in range(args.nconf):
            ou, od = rand_cfg(L * L, n, rng)
            t0 = time.perf_counter(); obj.setup(ou, od); ts = time.perf_counter() - t0
            E, te, nc, ops = eloc_fast(obj, T, nn, ou, od)
            tl.append(te / nc); ol.append(ops / nc)
        diag, cl = conn_list(T, nn, ou, od)
        t0 = time.perf_counter()
        for (c, m) in cl[:10]:
            u2, d2 = apply_moves(ou, od, m); obj.amp(u2, d2)
        tb = (time.perf_counter() - t0) / 10
        Ns.append(2 * n); tcs.append(float(np.median(tl))); ocs.append(float(np.mean(ol)))
        print(f"   L={L:2d} Ne={2 * n:4d} n_conn={nc:5d} |E_loc|/Ne={abs(E) / (2 * n):.2f} setup={ts:.2e}s "
              f"t_c={tcs[-1]:.3e}s ops/conn={ocs[-1]:.3e} t_brute/conn={tb:.3e}s speedup={tb / tcs[-1]:.0f}")
    sl = float(np.polyfit(np.log(Ns), np.log(tcs), 1)[0])
    so = float(np.polyfit(np.log(Ns), np.log(ocs), 1)[0])
    kill = emax > 1e-6 or sl >= 1.8 or so > 1.15
    verdict = "KILL" if kill else ("PASS" if (emax <= 1e-9 and so <= 1.15) else "AMBIGUOUS")
    note = " (timing overhead-bound, cost rests on the op count)" if (not kill and sl < so - 0.2) else ""
    print(f"== {name}: max rel_err={emax:.1e} slope(t_c)={sl:.2f} slope(ops/conn)={so:.2f} -> T1 {verdict}{note}")
    return verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", choices=["sh", "lh", "hfps", "all"], default="all")
    ap.add_argument("--sizes", default="16,24,32")
    ap.add_argument("--exact-L", default="4,6,8")
    ap.add_argument("--patch", choices=["site", "plus", "3x3"], default="plus")
    ap.add_argument("--F", type=int, default=8)
    ap.add_argument("--nconf", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    args.size_list = [int(s) for s in args.sizes.split(",")]
    args.exact_list = [int(s) for s in args.exact_L.split(",")]
    t0 = time.time()
    for r in (["hfps", "sh", "lh"] if args.route == "all" else [args.route]):
        run_route(r, args)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
