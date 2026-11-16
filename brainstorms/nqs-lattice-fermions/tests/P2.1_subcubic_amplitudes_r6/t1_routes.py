#!/usr/bin/env python3
"""
P2.1 round-6 tier-1 (T1) toy computations for the route portfolio (Stage 3.5 of the skill).
=========================================================================================

Three checks, each with a kill rule pre-registered in the leaf artifact (Tier log, round 6)
before this script was run:

  --mode gate4     Retroactive Gate 4 for the in-flight route SK-LR (round-5 tj_core.LRCarried):
                   time per connection t_c at three sizes with the sign-trunk width H_s fixed
                   (64) and with H_s grown proportionally to N_e (H_s = N_e / 4).
                   Prediction (T0): slope of t_c in N_e is ~1 at fixed H_s (C_E = O(N_e^2)) and
                   ~2 at H_s ~ N_e (C_E = O(N_e^3)).
  --mode plr       Route 6.2, PLR (product of local rank-1 determinant-ratio factors):
                   psi = prod_s det A0_s * prod_{m,s} (1 + h_m(n) . Z_s[:, m]),
                   Z_s = Omega_s A0_s^{-1} Phih_s[occ_s]  (F x M, M = V factors, one per site),
                   h_m = tanh(W x_patch(m) + b)  (F local features, 3x3 patch, both spins).
                   Carried-table local energy versus brute force (sorted rows, explicit
                   Jordan-Wigner signs, every amplitude rebuilt) and t_c at three sizes.
  --mode gkbf      Route 6.1, GK-BF (global-kernel linear-readout backflow):
                   row of particle j (spin s, site x_j):
                   a_j = Phi0_s[x_j] + (K_s[x_j, :] H(n)) Omega_s,  H(n) = (h_m)_m in R^{V x F}.
                   A hop changes H only on the patch set B (|B| <= 18), so each spin's matrix
                   changes by rank <= 1 + |B|; the ratio is a (1+|B|)-dimensional determinant
                   built from carried Ainv, T = Ainv K[occ, :], R = Omega Ainv, Q = Omega T,
                   KH = K H. Same checks as plr.
  --mode lhhfds    Route 6.1' (mutation of 6.1 after its T0 Gate-4 kill), LH-HFDS: one hidden
                   fermion per site with a patch-local hidden row C_m = h_m Omega_m and hidden
                   orbitals K: A_s = Phi0_s[occ] + K_s[occ, :] C_s(n). Per-connection (unbatched)
                   path only: exactness is the gated quantity, timing is reported only.
  --mode tree      Route 6.4, Tree-HFDS: LH-HFDS plus one hidden fermion per block of a 2^l x 2^l
                   block hierarchy with nonlinear block features; same checks as lhhfds.
  --mode all       gate4, plr, gkbf, lhhfds, tree in sequence.

Model: L x L PBC t-t'-J (tj_core / p11_core lattice), t = 1, t' = -0.25, J = 0.4, hole doping
1/8 (N_up = N_dn), random configuration, random weights (an algorithmic test, not training).
Exactness is checked at the smallest size (and at L = 4 for plr/gkbf), timing at --sizes.

Options
  --mode {gate4,plr,gkbf,lhhfds,tree,all}   what to run (default all)
  --sizes LIST    comma-separated L for timing (default 12,16,24; N_e = 126, 224, 504)
  --exact-L LIST  sizes with the brute-force exactness check (default 4,8)
  --F INT         local feature channels for plr/gkbf (default 8)
  --nconf INT     configurations per size (default 1)
  --seed INT      RNG seed (default 5)
Threads: set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS <= 4 (node policy for T1 runs).
Output: printed log only (copied to t1_output.txt next to this script by the caller).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r5"))
from p11_core import build_lattice            # noqa: E402
from tj_core import connections, LRCarried    # noqa: E402

TP, JX = -0.25, 0.4


def lattice(L):
    T, coords, bonds = build_lattice(L, L, "pbc", "pbc", 1.0, TP)
    nn = sorted({(min(i, j), max(i, j)) for (i, j, _) in bonds})
    V = L * L
    idx = lambda x, y: (x % L) + L * (y % L)
    patch = np.array([[idx(int(c[0]) + dx, int(c[1]) + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
                      for c in coords])                     # (V, 9) 3x3 patch
    nh = int(round(V / 8))
    if (V - nh) % 2:
        nh += 1
    return T, nn, patch, (V - nh) // 2


def rand_cfg(V, n, rng):
    p = rng.permutation(V)
    ou = np.zeros(V, int); od = np.zeros(V, int)
    ou[p[:n]] = 1; od[p[n:2 * n]] = 1
    return ou, od


def slope(N, t):
    return float(np.polyfit(np.log(N), np.log(t), 1)[0])


# ---------------------------------------------------------------- shared local feature net
class Local:
    def __init__(self, V, F, patch, rng):
        self.patch, self.V, self.F = patch, V, F
        self.W = rng.standard_normal((18, F)) / np.sqrt(18)
        self.b = 0.1 * rng.standard_normal(F)
        # which patches contain site s
        self.inv = [np.nonzero((patch == s).any(1))[0] for s in range(V)]

    def feats(self, ou, od, sites=None):
        P = self.patch if sites is None else self.patch[sites]
        x = np.concatenate([2 * ou[P] - 1, 2 * od[P] - 1], -1).astype(float)
        return np.tanh(x @ self.W + self.b)

    def changed(self, touched):
        return np.unique(np.concatenate([self.inv[s] for s in touched]))


def apply_moves(ou, od, moves):
    u, d = ou.copy(), od.copy()
    for (s, x, y) in moves:
        o = u if s == 0 else d
        o[x] = 0; o[y] = 1
    return u, d


def conn_list(T, nn, ou, od):
    """[(coef_inplace, [(spin, x, y), ...])] plus diag."""
    diag, hops, exch = connections(T, nn, JX, ou, od)
    out = [(c, [(s, x, y)]) for (s, x, y, c) in hops]
    out += [(-0.5 * JX, [(0, b, a), (1, a, b)]) for (a, b) in exch]
    return diag, out


def eloc_brute(amp, T, nn, ou, od):
    """Sorted rows with explicit Jordan-Wigner signs (as tj_core.eloc_brute)."""
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


# ---------------------------------------------------------------- Route 6.2: PLR
class PLR:
    def __init__(self, T, patch, n, F, rng):
        V = T.shape[0]
        self.V, self.n, self.F = V, n, F
        e, v = np.linalg.eigh(-T)
        self.P0 = [v[:, :n] + 0.1 * rng.standard_normal((V, n)) / np.sqrt(V) for _ in range(2)]
        self.Ph = [rng.standard_normal((V, V)) / np.sqrt(V) for _ in range(2)]   # M = V hidden rows
        self.Om = [0.05 * rng.standard_normal((F, n)) / np.sqrt(F * n) for _ in range(2)]
        self.loc = Local(V, F, patch, rng)

    def amp(self, ou, od):          # sorted rows, from scratch: O(N^2 M)
        h = self.loc.feats(ou, od)
        val = 1.0
        for s, o in ((0, ou), (1, od)):
            rows = np.nonzero(o)[0]
            A0 = self.P0[s][rows]
            Z = self.Om[s] @ np.linalg.solve(A0, self.Ph[s][rows])
            val *= np.linalg.det(A0) * np.prod(1 + np.einsum("mf,fm->m", h, Z))
        return val

    def setup(self, ou, od):
        self.ou, self.od = ou, od
        self.h = self.loc.feats(ou, od)
        self.c = []
        for s, o in ((0, ou), (1, od)):
            rows = np.nonzero(o)[0]
            Ai = np.linalg.inv(self.P0[s][rows])
            G0 = self.P0[s] @ Ai                       # V x N
            Hk = G0 @ self.Ph[s][rows]                 # V x M
            R = self.Om[s] @ Ai                        # F x N
            Z = R @ self.Ph[s][rows]                   # F x M
            hz = np.einsum("mf,fm->m", self.h, Z)
            f = 1 + hz
            slot = {int(x): i for i, x in enumerate(rows)}
            self.c.append(dict(G0=G0, Hk=Hk, R=R, Z=Z, f=f, hz=hz, hR=self.h @ R, slot=slot))

    def ratio(self, moves):
        ou, od = apply_moves(self.ou, self.od, moves)
        B = self.loc.changed([x for (_, x, _) in moves] + [y for (_, _, y) in moves])
        hB = self.loc.feats(ou, od, B)
        h2 = self.h.copy(); h2[B] = hB                 # O(|B| F) change (copy is bookkeeping)
        rho = 1.0
        mv = {s: (x, y) for (s, x, y) in moves}
        for s in (0, 1):
            c = self.c[s]
            if s in mv:
                x, y = mv[s]
                i = c["slot"][x]
                r0 = c["G0"][y, i]
                w = (self.Ph[s][y] - c["Hk"][y]) / r0  # M
                # f'_m = 1 + h'_m . (Z[:, m] + R[:, i] w_m)     O(M F)
                fn = 1 + np.einsum("mf,fm->m", h2, c["Z"]) + (h2 @ c["R"][:, i]) * w
                rho *= r0 * np.prod(fn / c["f"])
            else:
                fB = 1 + np.einsum("mf,fm->m", hB, c["Z"][:, B])
                rho *= np.prod(fB / c["f"][B])
        return rho


# ---------------------------------------------------------------- Route 6.1: GK-BF
class GKBF:
    def __init__(self, T, patch, n, F, rng):
        V = T.shape[0]
        self.V, self.n, self.F = V, n, F
        e, v = np.linalg.eigh(-T)
        self.P0 = [v[:, :n] + 0.1 * rng.standard_normal((V, n)) / np.sqrt(V) for _ in range(2)]
        self.K = [rng.standard_normal((V, V)) / np.sqrt(V) for _ in range(2)]   # global kernel
        self.Om = [0.3 * rng.standard_normal((F, n)) / np.sqrt(F) for _ in range(2)]
        self.loc = Local(V, F, patch, rng)

    def amp(self, ou, od):
        H = self.loc.feats(ou, od)
        val = 1.0
        for s, o in ((0, ou), (1, od)):
            rows = np.nonzero(o)[0]
            A = self.P0[s][rows] + (self.K[s][rows] @ H) @ self.Om[s]
            val *= np.linalg.det(A)
        return val

    def setup(self, ou, od):
        self.ou, self.od = ou, od
        self.H = self.loc.feats(ou, od)
        self.c = []
        for s, o in ((0, ou), (1, od)):
            rows = np.nonzero(o)[0]
            KH = self.K[s] @ self.H                                  # V x F
            A = self.P0[s][rows] + KH[rows] @ self.Om[s]
            Ai = np.linalg.inv(A)
            Tt = Ai @ self.K[s][rows]                                # N x V
            self.c.append(dict(Ai=Ai, T=Tt, R=self.Om[s] @ Ai, Q=self.Om[s] @ Tt, KH=KH,
                               slot={int(x): i for i, x in enumerate(rows)}))

    def ratio(self, moves):
        ou, od = apply_moves(self.ou, self.od, moves)
        B = self.loc.changed([x for (_, x, _) in moves] + [y for (_, _, y) in moves])
        dH = self.loc.feats(ou, od, B) - self.H[B]                   # b x F
        rho = 1.0
        mv = {s: (x, y) for (s, x, y) in moves}
        b = len(B)
        for s in (0, 1):
            c = self.c[s]
            QB = c["Q"][:, B]                                        # F x b
            if s in mv:
                x, y = mv[s]
                i = c["slot"][x]
                kap = self.K[s][y] - self.K[s][x]
                kH = c["KH"][y] - c["KH"][x] + kap[B] @ dH            # F  (row-change features)
                d0 = self.P0[s][y] - self.P0[s][x]                   # N
                Ri = c["R"][:, i]
                M = np.eye(1 + b)
                M[0, 0] += d0 @ c["Ai"][:, i] + kH @ Ri
                M[0, 1:] += d0 @ c["T"][:, B] + kH @ QB              # O(N b)
                M[1:, 0] += dH @ Ri
                M[1:, 1:] += dH @ QB
                rho *= np.linalg.det(M)
            else:
                rho *= np.linalg.det(np.eye(b) + dH @ QB)
        return rho


# ---------------------------------------------------------------- Route 6.1': LH-HFDS
class LHHFDS:
    """Local-hidden-row HFDS: one hidden fermion per site m, hidden row C_m(n) = h_m(n) Omega_m
    (patch-local), hidden orbitals K (V x V): A_s = Phi0_s[occ] + K_s[occ, :] C_s(n).
    Carried: Ai, T = Ai K[occ, :], R[m] = Omega_m Ai (V x F x N), Q[m] = Omega_m T (V x F x V)."""

    def __init__(self, T, patch, n, F, rng):
        V = T.shape[0]
        self.V, self.n, self.F = V, n, F
        e, v = np.linalg.eigh(-T)
        self.P0 = [v[:, :n] + 0.1 * rng.standard_normal((V, n)) / np.sqrt(V) for _ in range(2)]
        self.loc = Local(V, F, patch, rng)
        self.build_tree(rng)
        Rn = self.nrow
        self.K = [rng.standard_normal((V, Rn)) / np.sqrt(Rn) for _ in range(2)]
        self.Om = [0.3 * rng.standard_normal((Rn, F, n)) / np.sqrt(F * Rn) for _ in range(2)]

    def build_tree(self, rng):          # LH-HFDS: hidden rows = sites only
        self.nrow = self.V

    def hid_all(self, ou, od):
        return self.loc.feats(ou, od)

    def hid_delta(self, ou, od, touched):
        B = self.loc.changed(touched)
        return B, self.loc.feats(ou, od, B) - self.H[B]

    def amp(self, ou, od):
        H = self.hid_all(ou, od)
        val = 1.0
        for s, o in ((0, ou), (1, od)):
            rows = np.nonzero(o)[0]
            C = np.einsum("mf,mfn->mn", H, self.Om[s])
            val *= np.linalg.det(self.P0[s][rows] + self.K[s][rows] @ C)
        return val

    def setup(self, ou, od):
        self.ou, self.od = ou, od
        self.H = self.hid_all(ou, od)
        self.c = []
        for s, o in ((0, ou), (1, od)):
            rows = np.nonzero(o)[0]
            C = np.einsum("mf,mfn->mn", self.H, self.Om[s])
            Ai = np.linalg.inv(self.P0[s][rows] + self.K[s][rows] @ C)
            Tt = Ai @ self.K[s][rows]
            self.c.append(dict(Ai=Ai, T=Tt, R=np.einsum("mfn,np->mfp", self.Om[s], Ai),
                               Q=np.einsum("mfn,nv->mfv", self.Om[s], Tt),
                               slot={int(x): i for i, x in enumerate(rows)}))

    def ratio(self, moves):
        ou, od = apply_moves(self.ou, self.od, moves)
        B, dH = self.hid_delta(ou, od, [x for (_, x, _) in moves] + [y for (_, _, y) in moves])
        H2 = self.H.copy(); H2[B] += dH
        rho = 1.0
        mv = {s: (x, y) for (s, x, y) in moves}
        b = len(B)
        for s in (0, 1):
            c = self.c[s]
            QBB = c["Q"][B][:, :, B]                                   # b x F x b
            if s in mv:
                x, y = mv[s]
                i = c["slot"][x]
                kh = (self.K[s][y] - self.K[s][x])[:, None] * H2       # V x F   O(V F)
                d0 = self.P0[s][y] - self.P0[s][x]
                M = np.eye(1 + b)
                M[0, 0] += d0 @ c["Ai"][:, i] + np.einsum("mf,mf->", kh, c["R"][:, :, i])
                M[0, 1:] += d0 @ c["T"][:, B] + np.einsum("mf,mfb->b", kh, c["Q"][:, :, B])
                M[1:, 0] += np.einsum("bf,bf->b", dH, c["R"][B][:, :, i])
                M[1:, 1:] += np.einsum("bf,bfk->bk", dH, QBB)
                rho *= np.linalg.det(M)
            else:
                rho *= np.linalg.det(np.eye(b) + np.einsum("bf,bfk->bk", dH, QBB))
        return rho



class TreeHFDS(LHHFDS):
    """Route 6.4: LH-HFDS plus one hidden fermion per block of a 2^l x 2^l block hierarchy
    (l = 1.. while 2^l divides L); block feature g = tanh(W_l mean(child features) + b_l), so a
    hop changes only the ancestors of the changed sites (O(log V) extra hidden rows)."""

    def build_tree(self, rng):
        V, F = self.V, self.F
        L = int(round(np.sqrt(V)))
        self.levels = []               # list of (parent index array over previous level rows, W, b)
        prev_xy = np.array([(m % L, m // L) for m in range(V)])
        prev_size, off = 1, V
        while L % (2 * prev_size) == 0:
            size = 2 * prev_size
            nb = L // size
            par = (prev_xy[:, 0] * prev_size // size) + nb * (prev_xy[:, 1] * prev_size // size)
            W = rng.standard_normal((F, F)) / np.sqrt(F)
            b = 0.1 * rng.standard_normal(F)
            self.levels.append(dict(par=par, nb=nb * nb, off=off, W=W, b=b))
            off += nb * nb
            prev_xy = np.array([(k % nb, k // nb) for k in range(nb * nb)])
            prev_size = size
        self.nrow = off

    def _up(self, child, lev):
        nb, par = lev["nb"], lev["par"]
        cnt = np.bincount(par, minlength=nb)[:, None]
        mean = np.zeros((nb, self.F)); np.add.at(mean, par, child)
        return np.tanh((mean / cnt) @ lev["W"] + lev["b"])

    def hid_all(self, ou, od):
        h = self.loc.feats(ou, od)
        out, cur = [h], h
        for lev in self.levels:
            cur = self._up(cur, lev); out.append(cur)
        return np.concatenate(out, 0)

    def hid_delta(self, ou, od, touched):
        B = self.loc.changed(touched)
        newH = {int(m): v for m, v in zip(B, self.loc.feats(ou, od, B))}
        ch, base = B, 0
        rows, vals = list(B), [newH[int(m)] for m in B]
        prev_n = self.V
        for lev in self.levels:
            par = lev["par"]
            ps = np.unique(par[ch])
            nxt = {}
            for p in ps:
                kids = np.nonzero(par == p)[0]
                feats = np.array([newH.get(base + int(k), None) if (base + int(k)) in newH
                                  else self.H[base + int(k)] for k in kids])
                g = np.tanh(feats.mean(0) @ lev["W"] + lev["b"])
                nxt[lev["off"] + int(p)] = g
            newH.update(nxt)
            rows += list(nxt.keys()); vals += list(nxt.values())
            ch, base, prev_n = ps, lev["off"], lev["nb"]
        rows = np.array(rows)
        return rows, np.array(vals) - self.H[rows]

# ---------------------------------------------------------------- batched (vectorised) ratios
def batch_groups(obj, cl, chunk=256):
    """Group connections by moving spins; yield (coef, xs, ys, Bp, dH) per chunk.
    xs/ys: dict spin -> (nc,) arrays; Bp: (nc, bmax) padded patch sets; dH: (nc, bmax, F) feature
    change on Bp (zero on padding, which repeats a valid site so it is a no-op)."""
    groups = {}
    for (c, m) in cl:
        groups.setdefault(tuple(s for (s, _, _) in m), []).append((c, m))
    for sig, items in groups.items():
        for st in range(0, len(items), chunk):
            blk = items[st:st + chunk]
            nc = len(blk)
            coef = np.array([c for (c, _) in blk])
            xs = {s: np.array([m[k][1] for (_, m) in blk]) for k, s in enumerate(sig)}
            ys = {s: np.array([m[k][2] for (_, m) in blk]) for k, s in enumerate(sig)}
            Bs = [obj.loc.changed([x for (_, x, _) in m] + [y for (_, _, y) in m]) for (_, m) in blk]
            bmax = max(len(B) for B in Bs)
            Bp = np.array([np.concatenate([B, np.full(bmax - len(B), B[0])]) for B in Bs])
            mask = np.array([[1.0] * len(B) + [0.0] * (bmax - len(B)) for B in Bs])
            OU = np.repeat(obj.ou[None], nc, 0); OD = np.repeat(obj.od[None], nc, 0)
            r = np.arange(nc)
            for s in sig:
                O = OU if s == 0 else OD
                O[r, xs[s]] = 0; O[r, ys[s]] = 1
            P = obj.loc.patch[Bp]                                      # (nc, bmax, 9)
            xin = np.concatenate([2 * OU[r[:, None, None], P] - 1, 2 * OD[r[:, None, None], P] - 1], -1)
            hnew = np.tanh(xin.astype(float) @ obj.loc.W + obj.loc.b)
            Hold = obj.H if hasattr(obj, "H") else obj.h
            dH = (hnew - Hold[Bp]) * mask[:, :, None]
            yield coef, sig, xs, ys, Bp, dH


def plr_batch(o, sig, xs, ys, Bp, dH):
    nc = Bp.shape[0]
    r = np.arange(nc)
    logr = np.zeros(nc); sgn = np.ones(nc)
    for s in (0, 1):
        c = o.c[s]
        dHZ = np.einsum("cbf,fcb->cb", dH, c["Z"][:, Bp])
        if s in sig:
            x, y = xs[s], ys[s]
            i = np.array([c["slot"][int(v)] for v in x])
            r0 = c["G0"][y, i]
            w = (o.Ph[s][y] - c["Hk"][y]) / r0[:, None]                # (nc, M)
            sfull = np.repeat(c["hz"][None], nc, 0)
            np.add.at(sfull, (r[:, None], Bp), dHZ)
            g = c["hR"][:, i].T.copy()                                 # (nc, M)
            np.add.at(g, (r[:, None], Bp), np.einsum("cbf,fc->cb", dH, c["R"][:, i]))
            fn = 1 + sfull + g * w
            q = fn / c["f"][None]
            logr += np.log(np.abs(q)).sum(1) + np.log(np.abs(r0)); sgn *= np.sign(q).prod(1) * np.sign(r0)
        else:
            q = (1 + c["hz"][Bp] + dHZ) / c["f"][Bp]
            logr += np.log(np.abs(q)).sum(1); sgn *= np.sign(q).prod(1)
    return sgn * np.exp(logr)


def gkbf_batch(o, sig, xs, ys, Bp, dH):
    nc, bm = Bp.shape
    r = np.arange(nc)
    rho = np.ones(nc)
    for s in (0, 1):
        c = o.c[s]
        QB = np.transpose(c["Q"][:, Bp], (1, 0, 2))                   # (nc, F, bm)
        if s in sig:
            x, y = xs[s], ys[s]
            i = np.array([c["slot"][int(v)] for v in x])
            kapB = o.K[s][y[:, None], Bp] - o.K[s][x[:, None], Bp]
            kH = c["KH"][y] - c["KH"][x] + np.einsum("cb,cbf->cf", kapB, dH)
            d0 = o.P0[s][y] - o.P0[s][x]                               # (nc, N)
            Ri = c["R"][:, i].T                                         # (nc, F)
            M = np.repeat(np.eye(1 + bm)[None], nc, 0)
            M[:, 0, 0] += np.einsum("cn,nc->c", d0, c["Ai"][:, i]) + np.einsum("cf,cf->c", kH, Ri)
            M[:, 0, 1:] += np.einsum("cn,ncb->cb", d0, c["T"][:, Bp]) + np.einsum("cf,cfb->cb", kH, QB)
            M[:, 1:, 0] += np.einsum("cbf,cf->cb", dH, Ri)
            M[:, 1:, 1:] += np.einsum("cbf,cfk->cbk", dH, QB)
            rho *= np.linalg.det(M)
        else:
            rho *= np.linalg.det(np.eye(bm)[None] + np.einsum("cbf,cfk->cbk", dH, QB))
    return rho


def eloc_fast(obj, T, nn, ou, od, batched=True):
    diag, cl = conn_list(T, nn, ou, od)
    t0 = time.perf_counter()
    if not batched or isinstance(obj, LHHFDS):   # includes TreeHFDS
        E = diag + sum(c * obj.ratio(m) for (c, m) in cl)
    else:
        f = plr_batch if isinstance(obj, PLR) else gkbf_batch
        E = diag
        for (coef, sig, xs, ys, Bp, dH) in batch_groups(obj, cl):
            E += float(np.sum(coef * f(obj, sig, xs, ys, Bp, dH)))
    return E, time.perf_counter() - t0, len(cl)


def run_family(name, cls, args):
    rng = np.random.default_rng(args.seed)
    print(f"== {name}: exactness (fast carried-table E_loc vs brute, sorted rows + JW signs)")
    emax = 0.0
    for L in args.exact_list:
        T, nn, patch, n = lattice(L)
        obj = cls(T, patch, n, args.F, rng)
        ou, od = rand_cfg(L * L, n, rng)
        obj.setup(ou, od)
        Ef, _, nc = eloc_fast(obj, T, nn, ou, od)
        Ef1, _, _ = eloc_fast(obj, T, nn, ou, od, batched=False)
        Eb = eloc_brute(obj.amp, T, nn, ou, od)
        e = max(abs(Ef - Eb), abs(Ef1 - Eb)) / (1 + abs(Eb))
        emax = max(emax, e)
        print(f"   L={L:2d} Ne={2 * n:4d} n_conn={nc:5d} E_fast={Ef:+.12f} E_brute={Eb:+.12f} rel_err={e:.1e}")
    print(f"== {name}: timing per connection")
    Ns, tcs = [], []
    for L in args.size_list:
        T, nn, patch, n = lattice(L)
        obj = cls(T, patch, n, args.F, rng)
        tl = []
        for _ in range(args.nconf):
            ou, od = rand_cfg(L * L, n, rng)
            t0 = time.perf_counter(); obj.setup(ou, od); ts = time.perf_counter() - t0
            _, te, nc = eloc_fast(obj, T, nn, ou, od)
            tl.append(te / nc)
        # brute per connection on 20 connections (sampled) for the speedup
        diag, cl = conn_list(T, nn, ou, od)
        t0 = time.perf_counter()
        for (c, m) in cl[:20]:
            u2, d2 = apply_moves(ou, od, m); obj.amp(u2, d2)
        tb = (time.perf_counter() - t0) / 20
        Ns.append(2 * n); tcs.append(float(np.median(tl))); spd = tb / tcs[-1]
        print(f"   L={L:2d} Ne={2 * n:4d} n_conn={nc:5d} setup={ts:.3e}s t_c={tcs[-1]:.3e}s "
              f"t_brute/conn={tb:.3e}s speedup={tb / tcs[-1]:.0f}")
    sl = slope(Ns, tcs)
    verdict = ("KILL" if (emax > 1e-6 or sl >= 1.8 or spd < 2) else
               "PASS" if (emax <= 1e-9 and 0.5 <= sl <= 1.3 and spd >= 10) else "AMBIGUOUS")
    print(f"== {name}: max rel_err={emax:.1e} slope(t_c vs N_e)={sl:.2f} speedup(largest L)={spd:.0f} "
          f"-> T1 {verdict}")
    return verdict


def run_gate4(args):
    print("== gate4 (in-flight SK-LR, K=4 r=1, Ha=128): t_c at fixed Hs=64 and at Hs=N_e/4")
    from micro_test import random_ansatz
    res = {}
    for tag in ("Hs=64", "Hs=Ne/4"):
        Ns, tcs = [], []
        for L in args.size_list:
            T, nn, patch, n = lattice(L)
            Hs = 64 if tag == "Hs=64" else max(8, (2 * n) // 4)
            a = random_ansatz("lr", L, n, T, None, args.seed + L, Hs=Hs)
            rng = np.random.default_rng(args.seed + L)
            ou, od = rand_cfg(L * L, n, rng)
            c = LRCarried(a, T, nn, JX, ou, od)
            t0 = time.perf_counter(); c.eloc(); te = time.perf_counter() - t0
            _, hops, exch = connections(T, nn, JX, ou, od)
            nc = len(hops) + len(exch)
            Ns.append(2 * n); tcs.append(te / nc)
            print(f"   {tag:8s} L={L:2d} Ne={2 * n:4d} Hs={Hs:4d} n_conn={nc:5d} t_c={te / nc:.3e}s")
        res[tag] = slope(Ns, tcs)
        print(f"   {tag}: slope(t_c vs N_e) = {res[tag]:.2f}  => C_E exponent ~ {1 + res[tag]:.2f}")
    ok = res["Hs=64"] <= 1.3 and res["Hs=Ne/4"] >= 1.6
    print(f"== gate4: prediction (fixed-width slope <= 1.3, growing-width slope >= 1.6) "
          f"{'CONFIRMED' if ok else 'NOT CONFIRMED'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["gate4", "plr", "gkbf", "lhhfds", "tree", "all"], default="all")
    ap.add_argument("--sizes", default="12,16,24")
    ap.add_argument("--exact-L", default="4,8")
    ap.add_argument("--F", type=int, default=8)
    ap.add_argument("--nconf", type=int, default=1)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args(argv)
    args.size_list = [int(s) for s in args.sizes.split(",")]
    args.exact_list = [int(s) for s in args.exact_L.split(",")]
    t0 = time.time()
    if args.mode in ("gate4", "all"):
        run_gate4(args)
    if args.mode in ("plr", "all"):
        run_family("PLR (Route 6.2)", PLR, args)
    if args.mode in ("gkbf", "all"):
        run_family("GK-BF (Route 6.1)", GKBF, args)
    if args.mode in ("lhhfds", "all"):
        run_family("LH-HFDS (Route 6.1')", LHHFDS, args)
    if args.mode in ("tree", "all"):
        run_family("Tree-HFDS (Route 6.4)", TreeHFDS, args)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
