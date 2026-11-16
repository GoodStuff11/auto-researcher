#!/usr/bin/env python3
"""
P2.1 Round 13: Schur-reduced (physical-inverse) engine for the local-feature HFPS family (cost T0/T1).
=====================================================================================================

Pre-registered in the leaf ("Round 13: refill ... Pre-registered T1 rules"), before this file was run.

Wave function (pairing frame, plus patch, F = 4, parameters from sub_hfps.SubHFPS(..., 'full') at the seed):
    psi = det A,  A = Phi[up, dn],  Phi = F0 + K C(n),  C[m, :] = g_m(n) h_m(n) . Omega_m
    g_m(n) = 1 if m is static, = 1 if m is gated and empty (n_m = 0), else 0.
Arms of this engine (--arm):
    full_S   static = all sites            (Route 7.1 itself; identical psi to delayed.Route('full'))
    hg_S     gated  = all sites            (13.2, hole-gated hidden rows)
    cbh_S    static = checkerboard, gated = the other sublattice   (13.3)
    half_S   static = checkerboard         (engine control only; not scored)
Engine: carries Ainv = A^-1 (N x N), Q = Ainv K[up, :] (N x V), C (V x V), Crd = C[:, dn] and Kru = K[up, :]
(slot order). No augmented (N + V_h) inverse and no V x V Green's table. A move (up slot i: xu -> yu,
down slot j: xd -> yd, changed hidden terms S with feature change Delta) is
    A' = A + U W^T,  U = [dc, e_i, K[up', S]],  W = [e_j, dr, (Delta . Omega_S)[:, dn']^T]
    dc = Phi[up, yd] - Phi[up, xd],  dr = Phi[yu, dn'] - Phi[xu, dn']   (built on demand, O(N V_active))
    ratio = det(I + W^T Ainv U);   Ainv U = [Ainv dc, Ainv[:, i], Q[:, S] + Ainv[:, i] (K[yu, S] - K[xu, S])].
Accept (Woodbury): Ainv -= X S^-1 (W^T Ainv); Q <- Q - X S^-1 (W^T Q) + Ainv'[:, i] (K[yu] - K[xu]).

Modes
  --mode check   T0 gates (pre-registered) on 5x3 OBC, 6x6 and 8x8 PBC for every arm:
                 (i) per-configuration normalised ratio error vs an explicit gated det(F0 + K C) build with
                     JW signs <= 1e-12, and E_loc vs t1_r7.eloc_brute <= 1e-12 relative;
                 (ii) full_S ratios == delayed.Route('full') ratios at equal parameters, normalised <= 1e-12;
                 (iii) drift <= 1e-8 after --accepts accepted moves without refresh (Ainv, Q, C, E_loc).
                 Prints CHECK PASS or BUG (exit 2).
  --mode ops     move statistics of each arm at --ops-sizes; multiply-adds per sweep for the arm on this
                 engine, for the 11c delayed engine on 'full' (rmax 32, mean pending rank 16) and for dense2;
                 predicted R_D(16) = 1.78 * ops_11c(full) / ops_arm (11c measured R_D(16) of full = 1.78).
  --mode run     --arm {full_S,hg_S,cbh_S,half_S,dense2,full_11c} --tag WAVE: c = t_sw * tau_int at --sizes with
                 CPU --budgets (the 11c delayed.run metric, unchanged). Writes results/<arm>_<tag>.json.
  --mode aggregate  R_D(L) = c_dense2 / c_arm (same wave), 1 sigma from the tau errors, and the T1-cost verdicts.
Options: --sizes 12,16  --budgets 20,90  --seed 11  --accepts 2000  --check 500  --ops-sizes 12,16  --tag A
Threads: OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1, `ulimit -t 600` per process, at most 4 processes.
Use NUMBA_CACHE_DIR outside the repository.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("P2.1_subcubic_amplitudes_r11c", "P2.1_subcubic_amplitudes_r11b", "P2.1_subcubic_amplitudes_r10",
          "P2.1_subcubic_amplitudes_r8b", "P2.1_subcubic_amplitudes_r7"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import delayed as DL  # noqa: E402  (read-only: lattice, propose, connections, Route, Dense2)
import t1_r7 as T1  # noqa: E402  (read-only)
import cost_per_sample as CP  # noqa: E402  (read-only)
import sub_hfps as SH  # noqa: E402  (read-only)

OUT = os.path.join(HERE, "results")
JX = T1.JX
TOL_EXACT, TOL_DRIFT = 1e-12, 1e-8
ARMS_S = ("full_S", "hg_S", "cbh_S", "half_S")
R11C_FULL = 1.78     # measured R_D(16) of full on the 11c engine (leaf, Round 11c)


def masks(arm, Lx, Ly):
    V = Lx * Ly
    xs, ys = np.arange(V) % Lx, np.arange(V) // Lx
    cb = ((xs + ys) % 2 == 0).astype(np.int64)
    z = np.zeros(V, np.int64)
    o = np.ones(V, np.int64)
    return {"full_S": (o, z), "hg_S": (z, o), "cbh_S": (cb, 1 - cb), "half_S": (cb, z)}[arm]


# ================================================================== engine
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
def s_ratio(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, stat, gat, act, alist, na, Hf, C, Crd, Kru, Ainv, Q,
            F0, K, Om, P, W, bvec, iptr, iidx, mark, sc_S, sc_dH, sc_gn, sc_X, sc_Wt, sc_M, sc_k, st):
    N = ru.shape[0]
    V = ou.shape[0]
    F = W.shape[1]
    up = xu >= 0
    dn = xd >= 0
    i = posu[xu] if up else -1
    j = posd[xd] if dn else -1
    # candidate hidden terms
    nc = 0
    for t in range(4):
        s = xu if t == 0 else (yu if t == 1 else (xd if t == 2 else yd))
        if s < 0:
            continue
        for q in range(iptr[s], iptr[s + 1]):
            m = iidx[q]
            if mark[m] == 0 and (stat[m] + gat[m]) > 0:
                mark[m] = 1
                sc_S[nc] = m
                nc += 1
    for t in range(nc):
        mark[sc_S[t]] = 0
    if up:
        ou[xu] = 0; ou[yu] = 1
    if dn:
        od[xd] = 0; od[yd] = 1
    hn = np.empty(F)
    b = 0
    for t in range(nc):
        m = sc_S[t]
        gn = 1 if (stat[m] == 1 or ou[m] + od[m] == 0) else 0
        if gn == 0 and act[m] == 0:
            continue
        if gn == 1:
            feat_row(m, ou, od, P, W, bvec, hn)
        else:
            for f in range(F):
                hn[f] = 0.0
        sc_S[b] = m
        sc_gn[b] = gn
        for f in range(F):
            sc_dH[b, f] = hn[f] - Hf[m, f]
        b += 1
    if up:
        ou[xu] = 1; ou[yu] = 0
    if dn:
        od[xd] = 1; od[yd] = 0
    k = b + (1 if up else 0) + (1 if dn else 0)
    X = np.empty((N, k))
    Wt = sc_Wt[:k]
    c = 0
    if dn:
        dc = np.empty(N)
        for p in range(N):
            dc[p] = F0[ru[p], yd] - F0[ru[p], xd]
        if na == V:
            dcol = C[:, yd] - C[:, xd]
            dc += np.dot(Kru, dcol)
        else:
            for p in range(N):
                acc = 0.0
                for t in range(na):
                    m = alist[t]
                    acc += Kru[p, m] * (C[m, yd] - C[m, xd])
                dc[p] += acc
        X[:, c] = np.dot(Ainv, dc)
        for q in range(N):
            Wt[c, q] = 0.0
        Wt[c, j] = 1.0
        c += 1
    if up:
        for p in range(N):
            X[p, c] = Ainv[p, i]
        for q in range(N):
            cq = yd if q == j else rd[q]
            Wt[c, q] = F0[yu, cq] - F0[xu, cq]
        if na == V:
            dk = K[yu] - K[xu]
            Wt[c] += np.dot(dk, Crd)
            if dn:
                acc = 0.0
                for m in range(V):
                    acc += dk[m] * (C[m, yd] - Crd[m, j])
                Wt[c, j] += acc
        else:
            for q in range(N):
                cq = yd if q == j else rd[q]
                acc = 0.0
                for t in range(na):
                    m = alist[t]
                    acc += (K[yu, m] - K[xu, m]) * C[m, cq]
                Wt[c, q] += acc
        c += 1
    for t in range(b):
        m = sc_S[t]
        for p in range(N):
            X[p, c + t] = Q[p, m]
        if up:
            d = K[yu, m] - K[xu, m]
            for p in range(N):
                X[p, c + t] += Ainv[p, i] * d
        for q in range(N):
            cq = yd if q == j else rd[q]
            acc = 0.0
            for f in range(F):
                acc += sc_dH[t, f] * Om[m, f, cq]
            Wt[c + t, q] = acc
    sc_X[:, :k] = X
    M = sc_M[:k * k].reshape((k, k))
    M[:, :] = np.dot(Wt, X)
    for a in range(k):
        M[a, a] += 1.0
    sc_k[0] = k
    sc_k[1] = b
    st[0] += 1.0
    st[1] += k
    st[2] += b
    if dn:
        st[3] += 1.0
    if up:
        st[4] += 1.0
    st[5] += na
    return np.linalg.det(M)


@njit(cache=True)
def s_accept(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, stat, gat, act, alist, nav, Hf, C, Crd, Kru, Ainv, Q,
             K, Om, sc_S, sc_dH, sc_gn, sc_X, sc_Wt, sc_M, sc_k, st):
    N = ru.shape[0]
    V = ou.shape[0]
    F = Om.shape[1]
    k = sc_k[0]
    b = sc_k[1]
    up = xu >= 0
    dn = xd >= 0
    i = posu[xu] if up else -1
    j = posd[xd] if dn else -1
    X = sc_X[:, :k].copy()
    Wt = sc_Wt[:k].copy()
    M = sc_M[:k * k].reshape((k, k)).copy()
    WA = np.dot(Wt, Ainv)
    WQ = np.dot(Wt, Q)
    Ainv -= np.dot(X, np.linalg.solve(M, WA))
    Q -= np.dot(X, np.linalg.solve(M, WQ))
    if up:
        for p in range(N):
            a = Ainv[p, i]
            for m in range(V):
                Q[p, m] += a * (K[yu, m] - K[xu, m])
        for m in range(V):
            Kru[i, m] = K[yu, m]
        ou[xu] = 0; ou[yu] = 1
        ru[i] = yu; posu[xu] = -1; posu[yu] = i
    if dn:
        od[xd] = 0; od[yd] = 1
        rd[j] = yd; posd[xd] = -1; posd[yd] = j
    for t in range(b):
        m = sc_S[t]
        for f in range(F):
            d = sc_dH[t, f]
            Hf[m, f] += d
            for v in range(V):
                C[m, v] += d * Om[m, f, v]
        act[m] = sc_gn[t]
        for q in range(N):
            Crd[m, q] = C[m, rd[q]]
    if dn:
        for m in range(V):
            Crd[m, j] = C[m, yd]
    if b > 0:
        n = 0
        for m in range(V):
            if act[m] == 1:
                alist[n] = m
                n += 1
        nav[0] = n
    st[6] += 1.0
    st[7] += k


@njit(cache=True)
def s_eloc(T, nbr, nnb, nnl, J, ou, od, ru, rd, posu, posd, stat, gat, act, alist, nav, Hf, C, Crd, Kru, Ainv, Q,
           F0, K, Om, P, W, bvec, iptr, iidx, mark, sc_S, sc_dH, sc_gn, sc_X, sc_Wt, sc_M, sc_k, st):
    diag, mv, cf = DL.connections(ou, od, T, nbr, nnb, nnl, J)
    E = diag
    for c in range(mv.shape[0]):
        E += cf[c] * s_ratio(mv[c, 0], mv[c, 1], mv[c, 2], mv[c, 3], ou, od, ru, rd, posu, posd, stat, gat, act,
                             alist, nav[0], Hf, C, Crd, Kru, Ainv, Q, F0, K, Om, P, W, bvec, iptr, iidx, mark,
                             sc_S, sc_dH, sc_gn, sc_X, sc_Wt, sc_M, sc_k, st)
    return E


@njit(cache=True)
def s_sweep(natt, do_eloc, T, nbr, nnb, nnl, isnn, J, ou, od, ru, rd, posu, posd, stat, gat, act, alist, nav, Hf,
            C, Crd, Kru, Ainv, Q, F0, K, Om, P, W, bvec, iptr, iidx, mark, sc_S, sc_dH, sc_gn, sc_X, sc_Wt, sc_M,
            sc_k, st):
    nprop = 0
    nacc = 0
    for _ in range(natt):
        xu, yu, xd, yd = DL.propose(ou, od, nbr, nnb, isnn)
        if xu == -2:
            continue
        nprop += 1
        rho = s_ratio(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, stat, gat, act, alist, nav[0], Hf, C, Crd, Kru,
                      Ainv, Q, F0, K, Om, P, W, bvec, iptr, iidx, mark, sc_S, sc_dH, sc_gn, sc_X, sc_Wt, sc_M,
                      sc_k, st)
        if np.random.random() < min(1.0, rho * rho):
            s_accept(xu, yu, xd, yd, ou, od, ru, rd, posu, posd, stat, gat, act, alist, nav, Hf, C, Crd, Kru, Ainv,
                     Q, K, Om, sc_S, sc_dH, sc_gn, sc_X, sc_Wt, sc_M, sc_k, st)
            nacc += 1
    E = 0.0
    if do_eloc:
        E = s_eloc(T, nbr, nnb, nnl, J, ou, od, ru, rd, posu, posd, stat, gat, act, alist, nav, Hf, C, Crd, Kru,
                   Ainv, Q, F0, K, Om, P, W, bvec, iptr, iidx, mark, sc_S, sc_dH, sc_gn, sc_X, sc_Wt, sc_M, sc_k,
                   st)
    return E, nprop, nacc


class Schur:
    """Engine state for one arm; parameters from sub_hfps.SubHFPS(..., 'full') at the given seed."""

    def __init__(self, arm, T, Lx, Ly, bc, n, seed, F=4):
        rng = np.random.default_rng(seed)
        self.py = SH.SubHFPS(T, Lx, Ly, bc, n, F, "plus", "full", rng)
        o = self.py
        self.rng, self.arm = rng, arm
        V = T.shape[0]
        self.V, self.N, self.F = V, n, F
        self.stat, self.gat = masks(arm, Lx, Ly)
        self.F0 = np.ascontiguousarray(o.F0)
        self.K = np.ascontiguousarray(o.K)
        self.Om = np.ascontiguousarray(o.Om)
        self.P = np.ascontiguousarray(o.loc.P, dtype=np.int64)
        self.W = np.ascontiguousarray(o.loc.W)
        self.bvec = np.ascontiguousarray(o.loc.b)
        lens = np.array([len(o.loc.inv[s]) for s in range(V)])
        self.iptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        self.iidx = np.concatenate([o.loc.inv[s] for s in range(V)]).astype(np.int64)
        self.mark = np.zeros(V, np.int64)
        bmax = 4 * int(lens.max())
        kmax = bmax + 2
        self.sc = dict(S=np.zeros(bmax, np.int64), dH=np.zeros((bmax, F)), gn=np.zeros(bmax, np.int64),
                       X=np.zeros((n, kmax)), Wt=np.zeros((kmax, n)), M=np.zeros(kmax * kmax),
                       k=np.zeros(2, np.int64))
        self.st = np.zeros(8)

    def gate(self, ou, od):
        return ((self.stat == 1) | ((self.gat == 1) & (ou + od == 0))).astype(np.int64)

    def amp_explicit(self, ou, od):
        """Reference: det(F0 + sum_{m active} K[:, m] h_m . Omega_m)[sorted up, sorted dn], explicit site loop."""
        o = self.py
        P, W, b = o.loc.P, o.loc.W, o.loc.b
        p = P.shape[1]
        g = self.gate(ou, od)
        Phi = self.F0.copy()
        for m in range(self.V):
            if not g[m]:
                continue
            x = np.zeros(2 * p)
            for a in range(p):
                s = P[m, a]
                if s >= 0:
                    x[a] = 2.0 * ou[s] - 1.0
                    x[p + a] = 2.0 * od[s] - 1.0
            h = np.tanh(x @ W + b)
            Phi += np.outer(self.K[:, m], h @ self.Om[m])
        return np.linalg.det(Phi[np.ix_(np.nonzero(ou)[0], np.nonzero(od)[0])])

    def setup(self, ou, od, ru=None, rd=None):
        self.ou, self.od = ou.astype(np.int64).copy(), od.astype(np.int64).copy()
        self.ru = (np.nonzero(ou)[0] if ru is None else np.asarray(ru)).astype(np.int64).copy()
        self.rd = (np.nonzero(od)[0] if rd is None else np.asarray(rd)).astype(np.int64).copy()
        self.posu = -np.ones(self.V, np.int64); self.posu[self.ru] = np.arange(self.N)
        self.posd = -np.ones(self.V, np.int64); self.posd[self.rd] = np.arange(self.N)
        self.act = self.gate(self.ou, self.od)
        H = self.py.loc.feats(self.ou, self.od)
        self.Hf = np.ascontiguousarray(H * self.act[:, None])
        self.C = np.ascontiguousarray(np.einsum("mf,mfv->mv", self.Hf, self.Om))
        self.Crd = np.ascontiguousarray(self.C[:, self.rd])
        self.Kru = np.ascontiguousarray(self.K[self.ru])
        A = self.F0[np.ix_(self.ru, self.rd)] + self.Kru @ self.Crd
        self.Ainv = np.ascontiguousarray(np.linalg.inv(A))
        self.Q = np.ascontiguousarray(self.Ainv @ self.Kru)
        self.alist = np.zeros(self.V, np.int64)
        a = np.nonzero(self.act)[0]
        self.alist[:len(a)] = a
        self.nav = np.array([len(a)], np.int64)

    def args(self):
        s = self.sc
        return (self.ou, self.od, self.ru, self.rd, self.posu, self.posd, self.stat, self.gat, self.act, self.alist,
                self.nav, self.Hf, self.C, self.Crd, self.Kru, self.Ainv, self.Q, self.F0, self.K, self.Om, self.P,
                self.W, self.bvec, self.iptr, self.iidx, self.mark, s["S"], s["dH"], s["gn"], s["X"], s["Wt"],
                s["M"], s["k"], self.st)


def make(arm, T, Lx, Ly, bc, n, seed):
    if arm == "dense2":
        return DL.Dense2(T, n, seed)
    if arm == "full_11c":
        return DL.Route("full", T, Lx, Ly, bc, n, 32, seed)
    return Schur(arm, T, Lx, Ly, bc, n, seed)


def do_sweep(obj, lat, natt, do_eloc=True):
    if isinstance(obj, Schur):
        T, nn, Tc, nbr, nnb, isnn, nnl = lat
        return s_sweep(natt, do_eloc, Tc, nbr, nnb, nnl, isnn, JX, *obj.args())
    return DL.do_sweep(obj, lat, natt, do_eloc)


def ratio_s(obj, mv):
    a = obj.args()
    return s_ratio(*mv, *a[:10], a[10][0], *a[11:])


def eloc_s(obj, lat):
    T, nn, Tc, nbr, nnb, isnn, nnl = lat
    return s_eloc(Tc, nbr, nnb, nnl, JX, *obj.args())


# ================================================================== modes
def brute_ratios(obj, lat):
    ou, od = obj.ou.copy(), obj.od.copy()
    p0 = obj.amp_explicit(ou, od)
    diag, mv, cf = DL.connections(obj.ou, obj.od, lat[2], lat[3], lat[4], lat[6], JX)
    refs = []
    for (xu, yu, xd, yd) in mv:
        u2, d2 = ou.copy(), od.copy()
        sg = 1
        if xu >= 0:
            sg *= (-1) ** int(ou[min(xu, yu) + 1:max(xu, yu)].sum()); u2[xu] = 0; u2[yu] = 1
        if xd >= 0:
            sg *= (-1) ** int(od[min(xd, yd) + 1:max(xd, yd)].sum()); d2[xd] = 0; d2[yd] = 1
        refs.append(sg * obj.amp_explicit(u2, d2) / p0)
    return mv, np.array(refs)


def nerr(a, b):
    return float(np.max(np.abs(a - b)) / max(1e-300, np.max(np.abs(b))))


def check(args):
    ok = True
    print("== (i) ratios vs explicit gated det build, E_loc vs eloc_brute; (ii) full_S vs delayed.Route('full')",
          flush=True)
    for g in ("5x3obc", "6", "8"):
        Lx, Ly, bc, n = DL.parse_geom(g)
        lat = DL.lattice(Lx, Ly, bc)
        for arm in ARMS_S:
            obj = Schur(arm, lat[0], Lx, Ly, bc, n, 21)
            obj.setup(*T1.rand_cfg(Lx * Ly, n, obj.rng))
            DL.seed_nb(5)
            en = eE = e11 = 0.0
            nconn = 0
            for rep in range(3):
                do_sweep(obj, lat, 2 * Lx * Ly, False)
                mv, ref = brute_ratios(obj, lat)
                fast = np.array([ratio_s(obj, tuple(int(v) for v in m)) for m in mv])
                en = max(en, nerr(fast, ref))
                nconn += len(mv)
                Ef = eloc_s(obj, lat)
                Eb = T1.eloc_brute(obj.amp_explicit, lat[0], lat[1], obj.ou, obj.od)
                eE = max(eE, abs(Ef - Eb) / abs(Eb))
                if arm == "full_S":
                    r = DL.Route("full", lat[0], Lx, Ly, bc, n, 32, 21)
                    r.setup(obj.ou, obj.od, obj.ru, obj.rd)
                    r11 = np.array([DL.ratio_c(r, tuple(int(v) for v in m)) for m in mv])
                    e11 = max(e11, nerr(fast, r11))
            ok &= en <= TOL_EXACT and eE <= TOL_EXACT and e11 <= TOL_EXACT
            print(f"   {g:6s} {arm:7s}: {nconn} conns, active rows {obj.nav[0]}/{obj.V}, ratio err (normalised) "
                  f"{en:.1e}, E_loc rel err {eE:.1e}" + (f", vs 11c full engine {e11:.1e}" if arm == "full_S" else ""),
                  flush=True)
    print(f"== (iii) drift over {args.accepts} accepted moves, no refresh", flush=True)
    for g in ("5x3obc", "6", "8"):
        Lx, Ly, bc, n = DL.parse_geom(g)
        lat = DL.lattice(Lx, Ly, bc)
        for arm in ARMS_S:
            obj = Schur(arm, lat[0], Lx, Ly, bc, n, 23)
            obj.setup(*T1.rand_cfg(Lx * Ly, n, obj.rng))
            DL.seed_nb(7)
            acc = prop = 0
            dmax = 0.0
            while acc < args.accepts:
                _, npr, nac = do_sweep(obj, lat, 1, False)
                prop += npr; acc += nac
                if nac and acc % args.check == 0:
                    f = Schur(arm, lat[0], Lx, Ly, bc, n, 23)
                    f.setup(obj.ou, obj.od, obj.ru, obj.rd)
                    et = max(nerr(obj.Ainv, f.Ainv), nerr(obj.Q, f.Q), nerr(obj.C, f.C) if np.any(f.C) else 0.0,
                             nerr(obj.Crd, f.Crd) if np.any(f.Crd) else 0.0, float(np.any(obj.act != f.act)))
                    Ec, Ef = eloc_s(obj, lat), eloc_s(f, lat)
                    dmax = max(dmax, et, abs(Ec - Ef) / (1 + abs(Ef)))
            ok &= dmax <= TOL_DRIFT
            print(f"   {g:6s} {arm:7s}: accepts={acc} acceptance={acc / prop:.2f} max drift={dmax:.1e}", flush=True)
    print("CHECK", "PASS" if ok else "BUG")
    return 0 if ok else 2


def ops(args):
    """T0 op count (multiply-adds per sweep) from sampled move statistics."""
    for L in [int(s) for s in args.ops_sizes.split(",")]:
        lat = DL.lattice(L, L, "pbc")
        n = T1.filling(L * L)
        V, N, F = L * L, n, 4
        res = {}
        for arm in ARMS_S:
            obj = Schur(arm, lat[0], L, L, "pbc", n, args.seed + L)
            obj.setup(*T1.rand_cfg(V, n, obj.rng))
            DL.seed_nb(args.seed + L)
            for _ in range(5):
                do_sweep(obj, lat, 4 * V)
            obj.st[:] = 0
            ns = 3
            for _ in range(ns):
                do_sweep(obj, lat, 4 * V)
            s = obj.st / ns
            nr, kbar, bbar, ndn, nup, nabar, nac, kac = s[0], s[1] / s[0], s[2] / s[0], s[3], s[4], s[5] / s[0], s[6], s[7] / max(s[6], 1)
            # this engine
            per_ratio = nabar * N * (ndn + nup) / nr + N * N * ndn / nr + N * bbar * (1 + F) + N * kbar * kbar
            per_acc = 2 * kac * N * (N + V) + bbar * F * V + N * V
            ops_s = nr * per_ratio + nac * per_acc
            # 11c delayed engine on the same move stream with n = N + V_h (V_h = active rows for gated arms
            # would need a slot engine; the 11c reference is 'full' only), mean pending rank 16
            nn_ = N + V
            r_ = 16
            k11 = bbar + 1 + 1   # rows R = up row + b hidden rows (+1 column on down moves)
            pr11 = (nup / nr) * (N + V) + bbar * F * N + nn_ * r_ * k11 + k11 * k11 * nn_ + (ndn / nr) * (nn_ * nn_ + 2 * nn_ * r_)
            pa11 = 2 * nn_ * nn_ * k11 + bbar * F * V
            ops_11 = nr * pr11 + nac * pa11
            res[arm] = (ops_s, ops_11, nr, kbar, bbar, nabar, nac, kac, ndn / nr)
        dense = res["full_S"][2] * ((F + 1) * N * N + (2.0 / 3.0) * N ** 3)
        o11 = res["full_S"][1]
        print(f"L={L} N={N} V={V}: dense2 ops/sweep={dense:.3e}; 11c full engine ops/sweep={o11:.3e} "
              f"(R_ops dense2/11c-full = {dense / o11:.2f})")
        for arm, (o, _, nr, kb, bb, na, nac, kac, fdn) in res.items():
            pred = R11C_FULL * o11 / o
            print(f"   {arm:7s}: ratios/sweep={nr:.0f} (down frac {fdn:.2f}) <k>={kb:.2f} <b>={bb:.2f} "
                  f"<active rows>={na:.1f} accepts/sweep={nac:.0f} <k_acc>={kac:.2f} ops/sweep={o:.3e} "
                  f"R_ops(dense2/arm)={dense / o:.2f} ops_11c(full)/ops_arm={o11 / o:.2f}"
                  + (f" predicted R_D(16)={pred:.2f}" if L == 16 else ""))


def warm():
    lat = DL.lattice(4, 4, "pbc")
    for arm in ("full_S", "hg_S"):
        o = Schur(arm, lat[0], 4, 4, "pbc", T1.filling(16), 1)
        o.setup(*T1.rand_cfg(16, T1.filling(16), o.rng))
        do_sweep(o, lat, 64)
    DL.warm()


def run(args):
    warm()
    out = {"arm": args.arm, "tag": args.tag, "sizes": {}}
    for L, bud in zip(args.size_list, args.budget_list):
        lat = DL.lattice(L, L, "pbc")
        n = T1.filling(L * L)
        obj = make(args.arm, lat[0], L, L, "pbc", n, args.seed + L)
        obj.setup(*T1.rand_cfg(L * L, n, obj.rng))
        DL.seed_nb(args.seed + L)
        c0 = time.process_time()
        Es, tsw, npr_t, nac_t = [], [], 0, 0
        while time.process_time() - c0 < bud:
            w0 = time.perf_counter()
            E, npr, nac = do_sweep(obj, lat, 4 * L * L)
            tsw.append(time.perf_counter() - w0)
            Es.append(E); npr_t += npr; nac_t += nac
        ns = len(Es)
        burn = max(10, ns // 10)
        Ek = np.array(Es[burn:])
        tau, terr, _ = CP.tau_int(Ek) if len(Ek) > 20 else (float("nan"), float("nan"), 0)
        t = float(np.mean(tsw[burn:])) if ns > burn else float(np.mean(tsw))
        s = dict(L=L, Ne=2 * n, sweeps=ns, t_sw=t, tau=tau, tau_err=terr, cost=t * tau,
                 acceptance=nac_t / max(npr_t, 1), E_per_Ne=float(np.mean(Ek) / (2 * n)) if len(Ek) else float("nan"),
                 cpu_s=time.process_time() - c0,
                 active=int(obj.nav[0]) if isinstance(obj, Schur) else None)
        out["sizes"][str(L)] = s
        print(f"[{args.arm}/{args.tag}] L={L:2d} sweeps={ns:5d} t_sw={t:.3e}s tau={tau:.2f}+-{terr:.2f} "
              f"cost={t * tau:.3e}s acc={s['acceptance']:.2f} <E>/Ne={s['E_per_Ne']:+.3f} cpu={s['cpu_s']:.0f}s",
              flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"{args.arm}_{args.tag}.json"), "w"), indent=1)


def rat(a, b):
    R = a["cost"] / b["cost"]
    return R, R * np.hypot(a["tau_err"] / a["tau"], b["tau_err"] / b["tau"]), a["t_sw"] / b["t_sw"]


def aggregate(args):
    R = {f[:-5]: json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith(".json")}
    for k, r in R.items():
        print(f"{k:14s} " + "  ".join(f"L={L}: t_sw={s['t_sw']:.3e} tau={s['tau']:.2f}+-{s['tau_err']:.2f} "
                                      f"sweeps={s['sweeps']}" for L, s in r["sizes"].items()))
    verdict = {}
    for k, r in R.items():
        arm, tag = k.rsplit("_", 1)
        if arm == "dense2" or f"dense2_{tag}" not in R:
            continue
        d = R[f"dense2_{tag}"]
        txt = []
        for L in r["sizes"]:
            Rv, e, rt = rat(d["sizes"][L], r["sizes"][L])
            txt.append(f"L={L}: R_D={Rv:.2f}+-{e:.2f} (t_sw ratio {rt:.2f})")
        print(f"{k:14s} vs dense2_{tag}: " + "; ".join(txt))
        if arm == "full_11c":
            continue
        R16, e16, _ = rat(d["sizes"]["16"], r["sizes"]["16"])
        if not (np.isfinite(R16) and np.isfinite(e16)):
            verdict[arm] = "INCOMPLETE (tau at L=16)"; continue
        v = "PASS-cost" if R16 - e16 >= 3 else "KILL-cost"
        extra = ""
        if arm == "full_S" and f"full_11c_{tag}" in R:
            Rf, ef, rtf = rat(R[f"full_11c_{tag}"]["sizes"]["16"], r["sizes"]["16"])
            extra = f"; c_11c/c_S = {Rf:.2f}+-{ef:.2f} (t_sw ratio {rtf:.2f})"
            if not (Rf - ef > 1):
                v = "KILL (Gate 2: not cheaper than the 11c engine at 1 sigma)"
        verdict[arm] = f"{v}: R_D(16) - sigma = {R16 - e16:.2f}{extra}"
    for arm, v in verdict.items():
        print(f"VERDICT {arm}: {v}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["check", "ops", "run", "aggregate"], default="check")
    ap.add_argument("--arm", choices=list(ARMS_S) + ["dense2", "full_11c"], default="full_S")
    ap.add_argument("--tag", default="A")
    ap.add_argument("--sizes", default="12,16")
    ap.add_argument("--budgets", default="20,90")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--accepts", type=int, default=2000)
    ap.add_argument("--check", type=int, default=500)
    ap.add_argument("--ops-sizes", default="12,16")
    args = ap.parse_args(argv)
    args.size_list = [int(s) for s in args.sizes.split(",")]
    args.budget_list = [float(s) for s in args.budgets.split(",")]
    if args.mode == "check":
        sys.exit(check(args))
    if args.mode == "ops":
        ops(args); return
    if args.mode == "aggregate":
        aggregate(args); return
    run(args)


if __name__ == "__main__":
    main()
