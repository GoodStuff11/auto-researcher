#!/usr/bin/env python3
"""
P2.1 Round 11c: compiled delayed-update falsifier of the Saturation Diagnostic (cost only).
==========================================================================================

Rule pre-registered in the leaf artifact ("Round 11c ... Pre-registration"), before any run.

Arms (parameters taken from the existing Python objects at the same seeds, imported read-only):
  full     Route 7.1 (local-feature HFPS, plus patch, F = 4), hidden rows on every site
  half     sublattice variant of round 11b (checkerboard, v_h = 1/2)
  quarter  sublattice variant of round 11b (x and y even, v_h = 1/4)
  dense    dense HFPS F = 4 (cost_8c.DenseHFPS): ratio = F N^2 gather + one LAPACK LU
  dense2   same wave function and ratio as dense, with the gathered blocks F0[ru, rd] and KO_f[ru, rd]
           carried in slot order (patched in O(N F) per accepted move); added after the first
           scored runs as a strengthened (cheaper) baseline; the verdict uses it when present
Route engine (numba): augmented matrix M (n = N + V_h, rows = up slots + hidden rows, columns =
down slots + hidden rows); only the inverse is carried, as Mi = Mi0 - P^T Z (rank r <= rmax + k).
A move replaces rows R (up row i and the b hidden rows whose patch contains a touched site) and, for
a down move, column j:  M' = M + E_R D + ct e_j^T  (ct = column change outside R). Ratio:
  det [[I + D Mi[:, R], D Mi ct], [Mi[j, R], 1 + (Mi ct)_j]]   (matrix-determinant lemma).
Accept: append U = [Mi[:, R], Mi ct] and Z_new = S^-1 [D Mi; Mi[j, :]] (Woodbury); when the
pending rank would exceed rmax, flush Mi0 -= P^T Z with one GEMM (n x r x n). rmax = 0 flushes
at every accepted move.

Modes
  --mode check      T0 gates (pre-registered): route arms on 5x3 OBC, 6x6 and 8x8 PBC at --rmax:
                    (i) connection ratios == brute ratios (SubHFPS.amp_explicit, sorted rows, JW signs):
                        amendment A1 gates max_c |rho_fast - rho_ref| / max_c |rho_ref| <= 1e-12 per
                        configuration (the per-ratio relative error is printed and reported, not gated;
                        see diag_ratio.py); compiled E_loc == t1_r7.eloc_brute to rel <= 1e-12;
                    (ii) drift <= 1e-8 after --accepts accepted moves without refresh (Mi and C
                        normalised max error vs rebuild, E_loc vs rebuilt engine).
                    dense, dense2: E_loc == eloc_brute(DenseHFPS.amp_sorted) <= 1e-12; carried log|det|,
                    pre-activation (and dense2's carried blocks) drift <= 1e-8 after --accepts accepted moves. Exit 2 on BUG.
  --mode scan       t_sw over --scan-sweeps sweeps at L = --scan-L for rmax in 0,16,32,64 (route arms)
  --mode run        --arm {full,half,quarter,dense,dense2} [--rmax]: c = t_sw * tau_int at --sizes with CPU
                    --budgets per size; sizes listed in --tsw-only get t_sw only (>= --min-sweeps).
                    Writes results/<arm>.json.
  --mode aggregate  table, R_D(L) = c_dense / c_arm with 1 sigma, and the pre-registered verdict, against
                    each dense arm present; the final verdict uses dense2 when present.
Options: --sizes (8,12,16,24) --budgets (10,20,90,40) --tsw-only (24) --min-sweeps (3)
         --rmax (32) --seed (11) --accepts (10000) --check (1000) --scan-L (16) --scan-sweeps (10)
Threads: OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=1, one process per
arm, at most 4 at a time, `ulimit -t` per process. Use NUMBA_CACHE_DIR outside the repository.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("P2.1_subcubic_amplitudes_r11b", "P2.1_subcubic_amplitudes_r10", "P2.1_subcubic_amplitudes_r8b",
          "P2.1_subcubic_amplitudes_r7"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import t1_r7 as T1  # noqa: E402  (read-only)
import cost_per_sample as CP  # noqa: E402  (read-only)
import cost_8c as C8  # noqa: E402  (read-only)
import sub_hfps as SH  # noqa: E402  (read-only)
from lh_geom import geometry  # noqa: E402

OUT = os.path.join(HERE, "results")
JX = T1.JX
TOL_EXACT, TOL_DRIFT = 1e-12, 1e-8


# ================================================================== lattice / Hamiltonian
def lattice(Lx, Ly, bc):
    T, nn = geometry(Lx, Ly, bc, T1.TP)
    V = T.shape[0]
    nbl = [np.nonzero(T[x])[0] for x in range(V)]
    nmax = max(len(v) for v in nbl)
    nbr = -np.ones((V, nmax), np.int64)
    nnb = np.zeros(V, np.int64)
    for x, v in enumerate(nbl):
        nbr[x, :len(v)] = v
        nnb[x] = len(v)
    isnn = np.zeros((V, V), np.bool_)
    for (i, j) in nn:
        isnn[i, j] = isnn[j, i] = True
    return T, nn, np.ascontiguousarray(T, dtype=np.float64), nbr, nnb, isnn, np.array(nn, np.int64).reshape(-1, 2)


@njit(cache=True)
def seed_nb(s):
    np.random.seed(s)


@njit(cache=True)
def propose(ou, od, nbr, nnb, isnn):
    """8b proposal. Returns (xu, yu, xd, yd); -1 = no move of that spin; xu = -2 = null attempt."""
    V = ou.shape[0]
    x = np.random.randint(V)
    y = nbr[x, np.random.randint(nnb[x])]
    ex = ou[x] + od[x]
    ey = ou[y] + od[y]
    if ex > 0 and ey == 0:
        if ou[x] == 1:
            return x, y, -1, -1
        return -1, -1, x, y
    if ex > 0 and ey > 0 and isnn[x, y] and ou[x] != ou[y]:
        if ou[x] == 1:
            return x, y, y, x
        return y, x, x, y
    return -2, -2, -2, -2


@njit(cache=True)
def connections(ou, od, T, nbr, nnb, nnl, J):
    """t-t'-J connections (tj_core.connections, in-place coefficients). Returns diag, moves (C, 4), coef."""
    V = ou.shape[0]
    diag = 0.0
    for e in range(nnl.shape[0]):
        i, j = nnl[e, 0], nnl[e, 1]
        if (ou[i] == 1 and od[j] == 1) or (od[i] == 1 and ou[j] == 1):
            diag += -0.5 * J
    cmax = 2 * V * nbr.shape[1] + 2 * nnl.shape[0]
    mv = np.empty((cmax, 4), np.int64)
    cf = np.empty(cmax)
    c = 0
    for s in range(2):
        for x in range(V):
            if (ou[x] if s == 0 else od[x]) == 1:
                for t in range(nnb[x]):
                    y = nbr[x, t]
                    if ou[y] + od[y] == 0:
                        if s == 0:
                            mv[c, 0] = x; mv[c, 1] = y; mv[c, 2] = -1; mv[c, 3] = -1
                        else:
                            mv[c, 0] = -1; mv[c, 1] = -1; mv[c, 2] = x; mv[c, 3] = y
                        cf[c] = -T[y, x]
                        c += 1
    for e in range(nnl.shape[0]):
        for o in range(2):
            a = nnl[e, o]; b = nnl[e, 1 - o]
            if ou[b] == 1 and od[a] == 1:          # up b -> a, down a -> b
                mv[c, 0] = b; mv[c, 1] = a; mv[c, 2] = a; mv[c, 3] = b
                cf[c] = -0.5 * J
                c += 1
    return diag, mv[:c], cf[:c]


# ================================================================== route engine (delayed SMW)
@njit(cache=True)
def feat_row(m, ou, od, P, W, bvec, out):
    p = P.shape[1]
    F = W.shape[1]
    for f in range(F):
        acc = bvec[f]
        for a in range(p):
            s = P[m, a]
            if s >= 0:
                acc += (2.0 * ou[s] - 1.0) * W[a, f] + (2.0 * od[s] - 1.0) * W[p + a, f]
        out[f] = np.tanh(acc)


@njit(cache=True)
def r_ratio(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, H, C, Mi0, PT, Z, rr,
            F0, K, Om, P, W, bvec, iptr, iidx, mark, sc_B, sc_dH, sc_R, sc_D, sc_ct, sc_MiR, sc_y, sc_S, sc_k):
    """Exact amplitude ratio psi(n') / psi(n) (in-place slots). Fills the scratch for r_accept."""
    N = ru.shape[0]
    Vh = H.shape[0]
    n = N + Vh
    F = W.shape[1]
    r = rr[0]
    up = xu >= 0
    dn = xd >= 0
    # changed hidden rows B
    b = 0
    for t in range(4):
        s = xu if t == 0 else (yu if t == 1 else (xd if t == 2 else yd))
        if s < 0:
            continue
        for q in range(iptr[s], iptr[s + 1]):
            m = iidx[q]
            if mark[m] == 0:
                mark[m] = 1
                sc_B[b] = m
                b += 1
    for t in range(b):
        mark[sc_B[t]] = 0
    # temporarily apply the move to evaluate new features
    if up:
        ou[xu] = 0; ou[yu] = 1
    if dn:
        od[xd] = 0; od[yd] = 1
    hn = np.empty(F)
    for t in range(b):
        m = sc_B[t]
        feat_row(m, ou, od, P, W, bvec, hn)
        for f in range(F):
            sc_dH[t, f] = hn[f] - H[m, f]
    if up:
        ou[xu] = 1; ou[yu] = 0
    if dn:
        od[xd] = 1; od[yd] = 0
    i = posu[xu] if up else -1
    j = posd[xd] if dn else -1
    # rows R and their change D (k x n)
    k = 0
    if up:
        sc_R[0] = i
        k = 1
    for t in range(b):
        sc_R[k + t] = N + sc_B[t]
    k += b
    D = sc_D[:k]
    if up:
        for q in range(N):
            cn = rd[q] if q != j else yd
            D[0, q] = F0[yu, cn] - F0[xu, rd[q]]
        for q in range(Vh):
            D[0, N + q] = K[yu, q] - K[xu, q]
    o = 1 if up else 0
    for t in range(b):
        m = sc_B[t]
        row = D[o + t]
        for q in range(N):
            cn = rd[q]
            acc = 0.0
            for f in range(F):
                acc += sc_dH[t, f] * Om[m, f, cn]
            row[q] = -acc
        if dn:
            acc = C[m, yd] - C[m, xd]
            for f in range(F):
                acc += sc_dH[t, f] * Om[m, f, yd]
            row[j] = -acc
        for q in range(N, n):
            row[q] = 0.0
    # Mi[:, R] = Mi0[:, R] - P^T Z[:, R]
    MiR = sc_MiR[:n * k].reshape((n, k))
    for a in range(n):
        for c in range(k):
            MiR[a, c] = Mi0[a, sc_R[c]]
    if r > 0:
        ZR = np.empty((r, k))
        for a in range(r):
            for c in range(k):
                ZR[a, c] = Z[a, sc_R[c]]
        MiR -= np.dot(PT[:r].T, ZR)
    kk = k + 1 if dn else k
    S = sc_S[:kk * kk].reshape((kk, kk))
    S[:k, :k] = np.dot(D, MiR)
    for c in range(k):
        S[c, c] += 1.0
    if dn:
        ct = sc_ct
        for p in range(N):
            ct[p] = F0[ru[p], yd] - F0[ru[p], xd]
        for m in range(Vh):
            ct[N + m] = -(C[m, yd] - C[m, xd])
        for c in range(k):
            ct[sc_R[c]] = 0.0
        y = np.dot(Mi0, ct)
        if r > 0:
            y -= np.dot(PT[:r].T, np.dot(Z[:r], ct))
        sc_y[:] = y
        S[:k, k] = np.dot(D, y)
        for c in range(k):
            S[k, c] = MiR[j, c]
        S[k, k] = 1.0 + y[j]
    sc_k[0] = k
    sc_k[1] = b
    return np.linalg.det(S)


@njit(cache=True)
def r_flush(Mi0, PT, Z, rr):
    r = rr[0]
    if r > 0:
        Mi0 -= np.dot(PT[:r].T, Z[:r])
        rr[0] = 0


@njit(cache=True)
def r_accept(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, H, C, Mi0, PT, Z, rr, rmax,
             Om, P, W, bvec, sc_B, sc_dH, sc_R, sc_D, sc_MiR, sc_y, sc_S, sc_k):
    """Accept the move last passed to r_ratio (scratch reused)."""
    N = ru.shape[0]
    Vh = H.shape[0]
    F = W.shape[1]
    k = sc_k[0]
    b = sc_k[1]
    up = xu >= 0
    dn = xd >= 0
    kk = k + 1 if dn else k
    if rr[0] + kk > rmax:
        r_flush(Mi0, PT, Z, rr)
    r = rr[0]
    j = posd[xd] if dn else -1
    D = sc_D[:k]
    VtMi = np.empty((kk, Mi0.shape[0]))
    VtMi[:k] = np.dot(D, Mi0)
    if r > 0:
        VtMi[:k] -= np.dot(np.dot(D, PT[:r].T), Z[:r])
    if dn:
        VtMi[k] = Mi0[j]
        if r > 0:
            VtMi[k] -= np.dot(PT[:r, j].copy(), Z[:r])
    S = sc_S[:kk * kk].reshape((kk, kk)).copy()
    Zn = np.linalg.solve(S, VtMi)
    MiR = sc_MiR[:Mi0.shape[0] * k].reshape((Mi0.shape[0], k))
    for c in range(k):
        PT[r + c] = MiR[:, c]
    if dn:
        PT[r + k] = sc_y
    Z[r:r + kk] = Zn
    rr[0] = r + kk
    # configuration-dependent state
    for t in range(b):
        m = sc_B[t]
        for f in range(F):
            d = sc_dH[t, f]
            H[m, f] += d
            for v in range(C.shape[1]):
                C[m, v] += d * Om[m, f, v]
    if up:
        i = posu[xu]
        ou[xu] = 0; ou[yu] = 1
        ru[i] = yu; posu[xu] = -1; posu[yu] = i
    if dn:
        od[xd] = 0; od[yd] = 1
        rd[j] = yd; posd[xd] = -1; posd[yd] = j


class Route:
    """Holds the compiled-engine state for one route arm; parameters from sub_hfps.SubHFPS."""

    def __init__(self, sub, T, Lx, Ly, bc, n, rmax, seed, F=4):
        rng = np.random.default_rng(seed)
        self.py = SH.SubHFPS(T, Lx, Ly, bc, n, F, "plus", sub, rng)
        o = self.py
        self.rng = rng
        V, Vh = T.shape[0], len(o.hs)
        self.F0 = np.ascontiguousarray(o.F0)
        self.K = np.ascontiguousarray(o.K)
        self.Om = np.ascontiguousarray(o.Om)
        self.P = np.ascontiguousarray(o.loc.P, dtype=np.int64)
        self.W = np.ascontiguousarray(o.loc.W)
        self.bvec = np.ascontiguousarray(o.loc.b)
        lens = np.array([len(o.loc.inv[s]) for s in range(V)])
        self.iptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        self.iidx = np.concatenate([o.loc.inv[s] for s in range(V)]).astype(np.int64)
        self.mark = np.zeros(Vh, np.int64)
        self.N, self.Vh, self.V, self.rmax = n, Vh, V, rmax
        nn_ = n + Vh
        kmax = 1 + int(lens.max()) * 4 + 2
        self.sc = dict(B=np.zeros(4 * int(lens.max()), np.int64), dH=np.zeros((4 * int(lens.max()), F)),
                       R=np.zeros(kmax, np.int64), D=np.zeros((kmax, nn_)), ct=np.zeros(nn_),
                       MiR=np.zeros(nn_ * kmax), y=np.zeros(nn_), S=np.zeros((kmax + 1) ** 2),
                       k=np.zeros(2, np.int64))
        self.PT = np.zeros((rmax + kmax + 1, nn_))
        self.Z = np.zeros((rmax + kmax + 1, nn_))
        self.rr = np.zeros(1, np.int64)

    def build(self, ou, od, ru, rd):
        """Rebuild M (slot order), C, H from scratch (numpy)."""
        o = self.py
        H = o.loc.feats(ou, od)
        C = np.einsum("mf,mfv->mv", H, self.Om)
        N, Vh = len(ru), self.Vh
        M = np.zeros((N + Vh, N + Vh))
        M[:N, :N] = self.F0[np.ix_(ru, rd)]
        M[:N, N:] = self.K[ru]
        M[N:, :N] = -C[:, rd]
        M[N:, N:] = np.eye(Vh)
        return M, np.ascontiguousarray(C), np.ascontiguousarray(H)

    def setup(self, ou, od, ru=None, rd=None):
        self.ou, self.od = ou.astype(np.int64).copy(), od.astype(np.int64).copy()
        self.ru = (np.nonzero(ou)[0] if ru is None else np.asarray(ru)).astype(np.int64).copy()
        self.rd = (np.nonzero(od)[0] if rd is None else np.asarray(rd)).astype(np.int64).copy()
        self.posu = -np.ones(self.V, np.int64); self.posu[self.ru] = np.arange(self.N)
        self.posd = -np.ones(self.V, np.int64); self.posd[self.rd] = np.arange(self.N)
        M, self.C, self.H = self.build(self.ou, self.od, self.ru, self.rd)
        self.Mi0 = np.ascontiguousarray(np.linalg.inv(M))
        self.rr[0] = 0

    def args_ratio(self):
        s = self.sc
        return (self.ou, self.od, self.ru, self.rd, self.posu, self.posd, self.H, self.C, self.Mi0, self.PT, self.Z,
                self.rr, self.F0, self.K, self.Om, self.P, self.W, self.bvec, self.iptr, self.iidx, self.mark,
                s["B"], s["dH"], s["R"], s["D"], s["ct"], s["MiR"], s["y"], s["S"], s["k"])

    def args_accept(self):
        s = self.sc
        return (self.ou, self.od, self.ru, self.rd, self.posu, self.posd, self.H, self.C, self.Mi0, self.PT, self.Z,
                self.rr, self.rmax, self.Om, self.P, self.W, self.bvec, s["B"], s["dH"], s["R"], s["D"], s["MiR"],
                s["y"], s["S"], s["k"])

    def Mi(self):
        r = self.rr[0]
        return self.Mi0 - self.PT[:r].T @ self.Z[:r]


@njit(cache=True)
def r_eloc(T, nbr, nnb, nnl, J, ou, od, ru, rd, posu, posd, H, C, Mi0, PT, Z, rr, F0, K, Om, P, W, bvec,
           iptr, iidx, mark, sc_B, sc_dH, sc_R, sc_D, sc_ct, sc_MiR, sc_y, sc_S, sc_k):
    diag, mv, cf = connections(ou, od, T, nbr, nnb, nnl, J)
    E = diag
    for c in range(mv.shape[0]):
        E += cf[c] * r_ratio(mv[c, 0], mv[c, 1], mv[c, 2], mv[c, 3], ou, od, ru, rd, posu, posd, H, C, Mi0, PT,
                             Z, rr, F0, K, Om, P, W, bvec, iptr, iidx, mark, sc_B, sc_dH, sc_R, sc_D, sc_ct,
                             sc_MiR, sc_y, sc_S, sc_k)
    return E


@njit(cache=True)
def r_sweep(natt, do_eloc, T, nbr, nnb, nnl, isnn, J, rmax, ou, od, ru, rd, posu, posd, H, C, Mi0, PT, Z, rr,
            F0, K, Om, P, W, bvec, iptr, iidx, mark, sc_B, sc_dH, sc_R, sc_D, sc_ct, sc_MiR, sc_y, sc_S, sc_k):
    nprop = 0
    nacc = 0
    for _ in range(natt):
        xu, yu, xd, yd = propose(ou, od, nbr, nnb, isnn)
        if xu == -2:
            continue
        nprop += 1
        rho = r_ratio(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, H, C, Mi0, PT, Z, rr, F0, K, Om, P, W, bvec,
                      iptr, iidx, mark, sc_B, sc_dH, sc_R, sc_D, sc_ct, sc_MiR, sc_y, sc_S, sc_k)
        if np.random.random() < min(1.0, rho * rho):
            r_accept(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, H, C, Mi0, PT, Z, rr, rmax, Om, P, W, bvec,
                     sc_B, sc_dH, sc_R, sc_D, sc_MiR, sc_y, sc_S, sc_k)
            nacc += 1
    E = 0.0
    if do_eloc:
        E = r_eloc(T, nbr, nnb, nnl, J, ou, od, ru, rd, posu, posd, H, C, Mi0, PT, Z, rr, F0, K, Om, P, W, bvec,
                   iptr, iidx, mark, sc_B, sc_dH, sc_R, sc_D, sc_ct, sc_MiR, sc_y, sc_S, sc_k)
    return E, nprop, nacc


# ================================================================== dense HFPS (compiled)
@njit(cache=True)
def d_ratio(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, a, st, F0, KOt, W1, b1, sc_a, sc_st, A):
    V = ou.shape[0]
    N = ru.shape[0]
    F = W1.shape[1]
    an = sc_a
    for f in range(F):
        an[f] = a[f]
    if xu >= 0:
        for f in range(F):
            an[f] += -2.0 * W1[xu, f] + 2.0 * W1[yu, f]
    if xd >= 0:
        for f in range(F):
            an[f] += -2.0 * W1[V + xd, f] + 2.0 * W1[V + yd, f]
    g = np.tanh(an)
    i = posu[xu] if xu >= 0 else -1
    j = posd[xd] if xd >= 0 else -1
    for p in range(N):
        rp = yu if p == i else ru[p]
        for q in range(N):
            cq = yd if q == j else rd[q]
            acc = F0[rp, cq]
            for f in range(F):
                acc += g[f] * KOt[rp, cq, f]
            A[p, q] = acc
    sg, la = np.linalg.slogdet(A)
    sc_st[0] = sg
    sc_st[1] = la
    return sg * st[0] * np.exp(la - st[1])


@njit(cache=True)
def d_accept(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, a, st, sc_a, sc_st):
    a[:] = sc_a
    st[0] = sc_st[0]
    st[1] = sc_st[1]
    if xu >= 0:
        i = posu[xu]
        ou[xu] = 0; ou[yu] = 1
        ru[i] = yu; posu[xu] = -1; posu[yu] = i
    if xd >= 0:
        j = posd[xd]
        od[xd] = 0; od[yd] = 1
        rd[j] = yd; posd[xd] = -1; posd[yd] = j


@njit(cache=True)
def d_eloc(T, nbr, nnb, nnl, J, ou, od, ru, rd, posu, posd, a, st, F0, KOt, W1, b1, sc_a, sc_st, A):
    diag, mv, cf = connections(ou, od, T, nbr, nnb, nnl, J)
    E = diag
    for c in range(mv.shape[0]):
        E += cf[c] * d_ratio(mv[c, 0], mv[c, 1], mv[c, 2], mv[c, 3], ou, od, ru, rd, posu, posd, a, st, F0, KOt,
                             W1, b1, sc_a, sc_st, A)
    return E


@njit(cache=True)
def d_sweep(natt, do_eloc, T, nbr, nnb, nnl, isnn, J, ou, od, ru, rd, posu, posd, a, st, F0, KOt, W1, b1,
            sc_a, sc_st, A):
    nprop = 0
    nacc = 0
    for _ in range(natt):
        xu, yu, xd, yd = propose(ou, od, nbr, nnb, isnn)
        if xu == -2:
            continue
        nprop += 1
        rho = d_ratio(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, a, st, F0, KOt, W1, b1, sc_a, sc_st, A)
        if np.random.random() < min(1.0, rho * rho):
            d_accept(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, a, st, sc_a, sc_st)
            nacc += 1
    E = 0.0
    if do_eloc:
        E = d_eloc(T, nbr, nnb, nnl, J, ou, od, ru, rd, posu, posd, a, st, F0, KOt, W1, b1, sc_a, sc_st, A)
    return E, nprop, nacc


class Dense:
    def __init__(self, T, n, seed, F=4):
        rng = np.random.default_rng(seed)
        self.py = C8.DenseHFPS(T, n, F, rng)
        self.rng = rng
        o = self.py
        self.V, self.N = T.shape[0], n
        self.F0 = np.ascontiguousarray(o.F0)
        self.KOt = np.ascontiguousarray(np.transpose(o.KO, (1, 2, 0)))
        self.W1 = np.ascontiguousarray(o.W1)
        self.b1 = np.ascontiguousarray(o.b1)
        self.sc_a = np.zeros(F); self.sc_st = np.zeros(2); self.A = np.zeros((n, n))

    def setup(self, ou, od, ru=None, rd=None):
        self.ou, self.od = ou.astype(np.int64).copy(), od.astype(np.int64).copy()
        self.ru = (np.nonzero(ou)[0] if ru is None else np.asarray(ru)).astype(np.int64).copy()
        self.rd = (np.nonzero(od)[0] if rd is None else np.asarray(rd)).astype(np.int64).copy()
        self.posu = -np.ones(self.V, np.int64); self.posu[self.ru] = np.arange(self.N)
        self.posd = -np.ones(self.V, np.int64); self.posd[self.rd] = np.arange(self.N)
        x = np.concatenate([self.ou, self.od]).astype(float) * 2 - 1
        self.a = x @ self.W1 + self.b1
        g = np.tanh(self.a)
        A = self.F0[np.ix_(self.ru, self.rd)] + np.einsum("f,ijf->ij", g, self.KOt[np.ix_(self.ru, self.rd)])
        sg, la = np.linalg.slogdet(A)
        self.st = np.array([sg, la])

    def args(self):
        return (self.ou, self.od, self.ru, self.rd, self.posu, self.posd, self.a, self.st, self.F0, self.KOt,
                self.W1, self.b1, self.sc_a, self.sc_st, self.A)


# ================================================================== dense HFPS, carried gather (dense2)
# Post-hoc strengthening of the baseline (added after the first scored runs, see the leaf): the gathered
# blocks G[p, q, 0] = F0[ru_p, rd_q], G[p, q, 1 + f] = KO_f[ru_p, rd_q] are carried in slot order and
# patched in O(N F) on accept, so a ratio streams (F + 1) N^2 contiguous entries before the LU instead
# of a scattered gather. Same wave function and ratio as dense.
@njit(cache=True)
def d2_ratio(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, a, st, F0, KOt, W1, b1, sc_a, sc_st, A, G):
    V = ou.shape[0]
    N = ru.shape[0]
    F = W1.shape[1]
    an = sc_a
    for f in range(F):
        an[f] = a[f]
    if xu >= 0:
        for f in range(F):
            an[f] += -2.0 * W1[xu, f] + 2.0 * W1[yu, f]
    if xd >= 0:
        for f in range(F):
            an[f] += -2.0 * W1[V + xd, f] + 2.0 * W1[V + yd, f]
    g = np.tanh(an)
    for p in range(N):
        for q in range(N):
            acc = G[p, q, 0]
            for f in range(F):
                acc += g[f] * G[p, q, 1 + f]
            A[p, q] = acc
    if xu >= 0:
        i = posu[xu]
        for q in range(N):
            cq = rd[q] if (xd < 0 or q != posd[xd]) else yd
            acc = F0[yu, cq]
            for f in range(F):
                acc += g[f] * KOt[yu, cq, f]
            A[i, q] = acc
    if xd >= 0:
        j = posd[xd]
        for p in range(N):
            rp = yu if (xu >= 0 and p == posu[xu]) else ru[p]
            acc = F0[rp, yd]
            for f in range(F):
                acc += g[f] * KOt[rp, yd, f]
            A[p, j] = acc
    sg, la = np.linalg.slogdet(A)
    sc_st[0] = sg
    sc_st[1] = la
    return sg * st[0] * np.exp(la - st[1])


@njit(cache=True)
def d2_accept(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, a, st, F0, KOt, sc_a, sc_st, G):
    N = ru.shape[0]
    F = KOt.shape[2]
    i = posu[xu] if xu >= 0 else -1
    j = posd[xd] if xd >= 0 else -1
    d_accept(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, a, st, sc_a, sc_st)
    if i >= 0:
        for q in range(N):
            G[i, q, 0] = F0[yu, rd[q]]
            for f in range(F):
                G[i, q, 1 + f] = KOt[yu, rd[q], f]
    if j >= 0:
        for p in range(N):
            G[p, j, 0] = F0[ru[p], yd]
            for f in range(F):
                G[p, j, 1 + f] = KOt[ru[p], yd, f]


@njit(cache=True)
def d2_eloc(T, nbr, nnb, nnl, J, ou, od, ru, rd, posu, posd, a, st, F0, KOt, W1, b1, sc_a, sc_st, A, G):
    diag, mv, cf = connections(ou, od, T, nbr, nnb, nnl, J)
    E = diag
    for c in range(mv.shape[0]):
        E += cf[c] * d2_ratio(mv[c, 0], mv[c, 1], mv[c, 2], mv[c, 3], ou, od, ru, rd, posu, posd, a, st, F0, KOt,
                              W1, b1, sc_a, sc_st, A, G)
    return E


@njit(cache=True)
def d2_sweep(natt, do_eloc, T, nbr, nnb, nnl, isnn, J, ou, od, ru, rd, posu, posd, a, st, F0, KOt, W1, b1,
             sc_a, sc_st, A, G):
    nprop = 0
    nacc = 0
    for _ in range(natt):
        xu, yu, xd, yd = propose(ou, od, nbr, nnb, isnn)
        if xu == -2:
            continue
        nprop += 1
        rho = d2_ratio(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, a, st, F0, KOt, W1, b1, sc_a, sc_st, A, G)
        if np.random.random() < min(1.0, rho * rho):
            d2_accept(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, a, st, F0, KOt, sc_a, sc_st, G)
            nacc += 1
    E = 0.0
    if do_eloc:
        E = d2_eloc(T, nbr, nnb, nnl, J, ou, od, ru, rd, posu, posd, a, st, F0, KOt, W1, b1, sc_a, sc_st, A, G)
    return E, nprop, nacc


class Dense2(Dense):
    def setup(self, ou, od, ru=None, rd=None):
        Dense.setup(self, ou, od, ru, rd)
        ix = np.ix_(self.ru, self.rd)
        self.G = np.ascontiguousarray(np.concatenate([self.F0[ix][:, :, None], self.KOt[ix]], axis=2))

    def args(self):
        return Dense.args(self) + (self.G,)


# ================================================================== drivers
DENSE_ARMS = ("dense", "dense2")


def make(arm, T, Lx, Ly, bc, n, rmax, seed):
    if arm == "dense":
        return Dense(T, n, seed)
    if arm == "dense2":
        return Dense2(T, n, seed)
    return Route(arm, T, Lx, Ly, bc, n, rmax, seed)


def do_sweep(obj, lat, natt, do_eloc=True):
    T, nn, Tc, nbr, nnb, isnn, nnl = lat
    if isinstance(obj, Dense2):
        return d2_sweep(natt, do_eloc, Tc, nbr, nnb, nnl, isnn, JX, *obj.args())
    if isinstance(obj, Dense):
        return d_sweep(natt, do_eloc, Tc, nbr, nnb, nnl, isnn, JX, *obj.args())
    return r_sweep(natt, do_eloc, Tc, nbr, nnb, nnl, isnn, JX, obj.rmax, *obj.args_ratio())


def eloc_c(obj, lat):
    T, nn, Tc, nbr, nnb, isnn, nnl = lat
    if isinstance(obj, Dense2):
        return d2_eloc(Tc, nbr, nnb, nnl, JX, *obj.args())
    if isinstance(obj, Dense):
        return d_eloc(Tc, nbr, nnb, nnl, JX, *obj.args())
    return r_eloc(Tc, nbr, nnb, nnl, JX, *obj.args_ratio())


def ratio_c(obj, mv):
    if isinstance(obj, Dense2):
        return d2_ratio(*mv, *obj.args())
    if isinstance(obj, Dense):
        return d_ratio(*mv, *obj.args())
    return r_ratio(*mv, *obj.args_ratio())


def parse_geom(g):
    if g == "5x3obc":
        return 5, 3, "obc", 6
    L = int(g)
    return L, L, "pbc", T1.filling(L * L)


def nerr(a, b):
    return float(np.max(np.abs(a - b)) / max(1e-300, np.max(np.abs(b))))


def brute_ratios(obj, lat):
    """Per-connection reference ratios in the in-place convention: JW sign * amp_explicit(n') / amp_explicit(n)."""
    T, nn = lat[0], lat[1]
    amp = obj.py.amp_explicit if isinstance(obj, Route) else obj.py.amp_sorted
    ou, od = obj.ou.copy(), obj.od.copy()
    p0 = amp(ou, od)
    diag, mv, cf = connections(obj.ou, obj.od, lat[2], lat[3], lat[4], lat[6], JX)
    refs = []
    for (xu, yu, xd, yd) in mv:
        u2, d2 = ou.copy(), od.copy()
        sg = 1
        if xu >= 0:
            sg *= (-1) ** int(ou[min(xu, yu) + 1:max(xu, yu)].sum()); u2[xu] = 0; u2[yu] = 1
        if xd >= 0:
            sg *= (-1) ** int(od[min(xd, yd) + 1:max(xd, yd)].sum()); d2[xd] = 0; d2[yd] = 1
        refs.append(sg * amp(u2, d2) / p0)
    return mv, np.array(refs)


def check(args):
    ok = okrel = True
    print(f"== route arms, rmax={args.rmax}: (i) per-connection ratio vs brute force, E_loc vs eloc_brute", flush=True)
    for g in ("5x3obc", "6", "8"):
        Lx, Ly, bc, n = parse_geom(g)
        lat = lattice(Lx, Ly, bc)
        for arm in ("full", "half", "quarter", "dense", "dense2"):
            obj = make(arm, lat[0], Lx, Ly, bc, n, args.rmax, 21)
            obj.setup(*T1.rand_cfg(Lx * Ly, n, obj.rng))
            seed_nb(5)
            er = en = eE = 0.0
            nconn = 0
            for rep in range(3):
                do_sweep(obj, lat, 2 * Lx * Ly, False)       # random accepted moves (slots permuted)
                mv, ref = brute_ratios(obj, lat)
                fast = np.array([ratio_c(obj, tuple(int(v) for v in m)) for m in mv])
                er = max(er, float(np.max(np.abs(fast - ref) / np.abs(ref))))
                en = max(en, float(np.max(np.abs(fast - ref)) / np.max(np.abs(ref))))
                nconn += len(mv)
                Ef = eloc_c(obj, lat)
                amp = obj.py.amp_explicit if arm not in DENSE_ARMS else obj.py.amp_sorted
                Eb = T1.eloc_brute(amp, lat[0], lat[1], obj.ou, obj.od)
                eE = max(eE, abs(Ef - Eb) / abs(Eb))
            ok &= en <= TOL_EXACT and eE <= TOL_EXACT          # amendment A1 (leaf): normalised ratio error
            okrel &= er <= TOL_EXACT
            pend = obj.rr[0] if arm not in DENSE_ARMS else 0
            print(f"   {g:6s} {arm:7s}: {nconn} connections, ratio err normalised (A1) {en:.1e}, per-ratio rel "
                  f"{er:.1e}, E_loc fast={Ef:+.12f} brute={Eb:+.12f} rel err {eE:.1e} (pending rank {pend})",
                  flush=True)
    print(f"   original per-ratio relative gate (superseded by A1): {'PASS' if okrel else 'FAIL'}")
    print(f"== (ii) drift over {args.accepts} accepted moves, no refresh (checkpoint every {args.check})", flush=True)
    for g in ("5x3obc", "6", "8"):
        Lx, Ly, bc, n = parse_geom(g)
        lat = lattice(Lx, Ly, bc)
        for arm in ("full", "half", "quarter", "dense", "dense2"):
            obj = make(arm, lat[0], Lx, Ly, bc, n, args.rmax, 23)
            obj.setup(*T1.rand_cfg(Lx * Ly, n, obj.rng))
            seed_nb(7)
            acc = prop = 0
            dmax = 0.0
            t0 = time.perf_counter()
            while acc < args.accepts:
                _, npr, nac = do_sweep(obj, lat, 1, False)
                prop += npr; acc += nac
                if nac and acc % args.check == 0:
                    f = make(arm, lat[0], Lx, Ly, bc, n, args.rmax, 23)
                    f.setup(obj.ou, obj.od, obj.ru, obj.rd)
                    if arm in DENSE_ARMS:
                        et = max(abs(obj.st[1] - f.st[1]) / max(1.0, abs(f.st[1])), nerr(obj.a, f.a),
                                 float(obj.st[0] != f.st[0]))
                        if arm == "dense2":
                            et = max(et, nerr(obj.G, f.G))
                    else:
                        et = max(nerr(obj.Mi(), f.Mi0), nerr(obj.C, f.C))
                    Ec, Ef = eloc_c(obj, lat), eloc_c(f, lat)
                    dmax = max(dmax, et, abs(Ec - Ef) / (1 + abs(Ef)))
            ok &= dmax <= TOL_DRIFT
            print(f"   {g:6s} {arm:7s}: accepts={acc} acceptance={acc / prop:.2f} max drift={dmax:.1e} "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)
    print("CHECK", "PASS" if ok else "BUG")
    return 0 if ok else 2


def scan(args):
    L = args.scan_L
    lat = lattice(L, L, "pbc")
    n = T1.filling(L * L)
    for arm in ("full", "half", "quarter"):
        line = []
        for rmax in (0, 16, 32, 64):
            obj = make(arm, lat[0], L, L, "pbc", n, rmax, args.seed + L)
            obj.setup(*T1.rand_cfg(L * L, n, obj.rng))
            seed_nb(args.seed + L)
            do_sweep(obj, lat, 4 * L * L)                                # warm-up / equilibration
            t0 = time.perf_counter()
            for _ in range(args.scan_sweeps):
                do_sweep(obj, lat, 4 * L * L)
            line.append(f"rmax={rmax}: t_sw={(time.perf_counter() - t0) / args.scan_sweeps:.4f}s")
        print(f"[scan L={L}] {arm:7s} " + "  ".join(line), flush=True)


def warm():
    lat = lattice(4, 4, "pbc")
    for arm in ("full", "dense"):
        o = make(arm, lat[0], 4, 4, "pbc", T1.filling(16), 8, 1)
        o.setup(*T1.rand_cfg(16, T1.filling(16), o.rng))
        do_sweep(o, lat, 64)


def run(args):
    warm()
    out = {"arm": args.arm, "rmax": args.rmax, "sizes": {}}
    tsw_only = {int(s) for s in args.tsw_only.split(",") if s}
    for L, bud in zip(args.size_list, args.budget_list):
        lat = lattice(L, L, "pbc")
        n = T1.filling(L * L)
        obj = make(args.arm, lat[0], L, L, "pbc", n, args.rmax, args.seed + L)
        obj.setup(*T1.rand_cfg(L * L, n, obj.rng))
        seed_nb(args.seed + L)
        c0 = time.process_time()
        Es, tsw, npr_t, nac_t = [], [], 0, 0
        while time.process_time() - c0 < bud or (L in tsw_only and len(Es) < args.min_sweeps):
            w0 = time.perf_counter()
            E, npr, nac = do_sweep(obj, lat, 4 * L * L)
            tsw.append(time.perf_counter() - w0)
            Es.append(E); npr_t += npr; nac_t += nac
        ns = len(Es)
        if L in tsw_only:
            burn = 1 if ns > 1 else 0
            tau, terr = float("nan"), float("nan")
        else:
            burn = max(10, ns // 10)
            Ek = np.array(Es[burn:])
            tau, terr, _ = CP.tau_int(Ek) if len(Ek) > 20 else (float("nan"), float("nan"), 0)
        t = float(np.mean(tsw[burn:])) if ns > burn else float(np.mean(tsw))
        Vh = getattr(obj, "Vh", None)
        out["sizes"][str(L)] = dict(L=L, Ne=2 * n, V_h=Vh, sweeps=ns, t_sw=t, tau=tau, tau_err=terr,
                                    cost=t * tau, acceptance=nac_t / max(npr_t, 1),
                                    E_per_Ne=float(np.mean(Es[burn:]) / (2 * n)) if ns > burn else float("nan"),
                                    cpu_s=time.process_time() - c0)
        s = out["sizes"][str(L)]
        print(f"[{args.arm}] L={L:2d} Ne={2 * n:3d} V_h={Vh} sweeps={ns:5d} t_sw={t:.3e}s tau={tau:.2f}+-{terr:.2f} "
              f"cost={t * tau:.3e}s acc={s['acceptance']:.2f} <E>/Ne={s['E_per_Ne']:+.3f} cpu={s['cpu_s']:.0f}s",
              flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, args.arm + ".json"), "w"), indent=1)


def ratio(a, b):
    R = a["cost"] / b["cost"]
    return R, R * np.hypot(a["tau_err"] / a["tau"], b["tau_err"] / b["tau"]), a["t_sw"] / b["t_sw"]


def aggregate(args):
    R = {f[:-5]: json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith(".json")}
    for k, r in R.items():
        print(f"{k:8s} rmax={r['rmax']:3d} " + "  ".join(
            f"L={L}: t_sw={s['t_sw']:.3e} tau={s['tau']:.2f}+-{s['tau_err']:.2f} sweeps={s['sweeps']}"
            for L, s in r["sizes"].items()))
    refs = [d for d in DENSE_ARMS if d in R]
    if not refs:
        print("INCOMPLETE: no dense arm"); return
    final = None
    for dref in refs:
        print(f"-- scored against {dref}")
        fals, unres, inc = [], [], []
        for v in ("full", "half", "quarter"):
            if v not in R:
                print(f"{v}: INCOMPLETE"); inc.append(v); continue
            txt = []
            for L in R[v]["sizes"]:
                Rv, e, rt = ratio(R[dref]["sizes"][L], R[v]["sizes"][L])
                txt.append(f"L={L}: {Rv:.2f}+-{e:.2f} (t_sw ratio {rt:.2f})")
            print(f"R_D[{v}] = c_{dref} / c_{v}: " + "; ".join(txt))
            R16, e16, _ = ratio(R[dref]["sizes"]["16"], R[v]["sizes"]["16"])
            if not (np.isfinite(R16) and np.isfinite(e16)):
                print(f"   {v}: INCOMPLETE (tau at L=16)"); inc.append(v); continue
            hit = R16 - e16 >= 3
            if "24" in R[v]["sizes"]:
                R24, e24, rt24 = ratio(R[dref]["sizes"]["24"], R[v]["sizes"]["24"])
                if np.isfinite(R24) and np.isfinite(e24):
                    hit |= R24 - e24 >= 4.5
                elif rt24 >= 4.5:
                    unres.append(v)
            if hit:
                fals.append(v)
            print(f"   {v}: R_D(16) - sigma = {R16 - e16:.2f} -> {'FALSIFIED' if hit else 'below 3'}"
                  + ("; L=24 t_sw ratio >= 4.5 (tau not measured)" if v in unres else ""))
        if fals:
            final = f"Saturation Diagnostic FALSIFIED by {', '.join(fals)} (against {dref})"
        elif inc:
            final = f"INCOMPLETE (against {dref})"
        elif unres:
            final = f"L=24 unresolved for {', '.join(unres)} (against {dref})"
        else:
            final = f"Saturation Diagnostic stands (compiled delayed updates; against {dref})"
        print(f"   verdict against {dref}: {final}")
    print("VERDICT (cheapest dense baseline, last listed):", final)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["check", "scan", "run", "aggregate"], default="check")
    ap.add_argument("--arm", choices=["full", "half", "quarter", "dense", "dense2"], default="full")
    ap.add_argument("--sizes", default="8,12,16,24")
    ap.add_argument("--budgets", default="10,20,90,40")
    ap.add_argument("--tsw-only", default="24")
    ap.add_argument("--min-sweeps", type=int, default=3)
    ap.add_argument("--rmax", type=int, default=32)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--accepts", type=int, default=10000)
    ap.add_argument("--check", type=int, default=1000)
    ap.add_argument("--scan-L", type=int, default=16)
    ap.add_argument("--scan-sweeps", type=int, default=10)
    args = ap.parse_args(argv)
    args.size_list = [int(s) for s in args.sizes.split(",")]
    args.budget_list = [float(s) for s in args.budgets.split(",")]
    if args.mode == "check":
        sys.exit(check(args))
    if args.mode == "scan":
        scan(args); return
    if args.mode == "aggregate":
        aggregate(args); return
    run(args)


if __name__ == "__main__":
    main()
