#!/usr/bin/env python3
"""
P4.1 round-2 decisive test: do exact-SU(2) fermionic ansaetze beat a spin-resolved NNBF of
matched parameter count, and is the difference in the SIGN?
==========================================================================================

All quantities are EXACT sums over the full fixed-(N_up, N_dn) Hilbert space of a small
2D Hubbard cluster (no Monte Carlo noise). Every arm gets the same optimiser (L-BFGS on the
exact energy with an analytic gradient), the same iteration cap, the same wall-clock cap,
the same seeds, the same charge Jastrow / Gutzwiller / doublon-holon correlator and the
same K=0 translation projection.

Physics
-------
H = -sum_{ij,s} t_ij c+_{is} c_{js} + U sum_i n_{i up} n_{i dn}  on an Lx x Ly torus,
t = 1 (nearest), t' = --tp (diagonal). Fermion order: all up modes (site 0..L-1), then all
down modes; site index i = x*Ly + y. A wave function is a matrix Psi[iu, id].
Default instance = P1.1 instance 1: 4x3 PBC, N_up = N_dn = 4, U = 8, t' = -0.25
(dim 245025; ED: S=0, K=0 ground state E0 = -12.0429038; the near-degenerate level
E1 = -11.9567 at Delta_1/V = 0.0072 t has S = 1).

Arms (--ansatz)
---------------
Shared charge correlator (all arms): J(c) = sum_{i<j} v_ij n_i n_j + g sum_i D_i
  + sum_{m=0..4} (a_m N^d_m(c) + b_m N^h_m(c)),  N^d_m = number of doublons with exactly m
  holons among their 4 nearest neighbours (mVMC-style doublon-holon factor), N^h_m likewise.
  c = charge configuration (n_1..n_L), D_i = doublon indicator. J depends on c only.
Shared lattice projection (all arms, --kproj 1): P_K = (1/L) sum_T T (K = 0 translations).

  foil : mVMC-like Minimal Contrast Foil  P_S P_K e^{J(c)} det f[up, dn],  f general L x L
         (singlet + triplet / AF pairing), P_S exact total-spin projector. Native size
         (no hidden layer; not parameter-matched, it is the field-standard reference).
  pcg  : ROUTE 2.2 "projected charge-conditioned Gaussian":
         P_S P_K e^{J(c)} det f_c[up, dn],  f_c = B1(c) f0 B2(c)^T,  B_a(c) = 1 + eta_a(c),
         eta_1, eta_2 two heads of one charge-only MLP (doublon / holon inputs), restricted
         to |i-j|_inf <= 1. f0 general, so f_c carries hole-bound spin texture
         (B1 != B2: up and down electrons are back-flowed differently). Exact S because
         every c-conditioned Gaussian rotates in closed form; cost N_grid determinants.
  scf  : ROUTE 2.5 "spin-charge factorised SU(2) MPS":
         psi(x) = s_ord(x) e^{J(c)} det Psi0[occ2(c)] * chi_c(sigma),
         occ2(c) = doubled spinless charge orbitals (site i, slot a) occupied if n_i >= 1,
         (site i, slot b) if n_i = 2 (N_e x N_e determinant, carries the charge JW sign);
         chi_c = l^T prod_{k singly occ, ascending} M_k^{sigma_k}(c) r, an SU(2)-symmetric
         MPS on the squeezed spin chain: virtual space (+)_k (k/2)^{m_k} (see --mps-mult),
         M = (reduced tensor W_k(c)) x (Clebsch-Gordan), W_k(c) = W0_k + U_k h(c),
         h(c) = tanh(A feat(c) + a) charge-only. l in the j=0 sector, r in j=0 sector ->
         exact singlet. s_ord converts site-major to up-then-down operator order.
         Cost O(L D^3) + one N_e x N_e determinant per charge configuration.
  nnbf : spin-resolved NNBF control (Luo & Clark): P_K e^{J(c)} det Phi_up(x)[up]
         det Phi_dn(x)[dn], Phi_s(x) = Phi0_s + reshape(W2 tanh(W1 x + b1) + b2),
         x = 2 n_{i s} - 1 (spin-resolved input). Hidden width chosen to MATCH the parameter
         count of pcg (see --target-params). NOT spin-projected (that is the point); the
         run also reports the post-hoc exact projection P_S psi_nnbf (diagnostic only).
  r1   : round-1 survivor (ablation): P_K e^{J(c)} det F_c[up, dn], F_c = B(c) F0 B(c)^T,
         F0 symmetric (charge-only singlet backflow, exact S, no projection). Matched.

Reported per run: E, eps = (E - E0)/|E0|, <S^2> - S(S+1), C_s(R) and P_d(R) errors, and the
P1.1 sign metric:  E_sign = min_{a >= 0} a^T K a / a^T a,  K = S H S, S = diag(sgn psi),
bracketed exactly as in tests/P1.1_sign_vs_amplitude (E_sign_up = min over the ten Haaf
self-consistent fixed-node energy and 4 L-BFGS-B starts; E_stoq a rigorous lower bound),
dE_sign = E_sign_up - E0, dE_amp = E - E_sign_up, f_E = dE_sign / (E - E0).

Usage
-----
  python micro_test.py --mode MODE [options]

--mode {smoke, time, run, report}            (required)
    smoke  : <= ~30 s self-test on a 2x3 torus (N_e = 6, U = 6, t' = -0.3; code check
             only, half filling, global ground state S=0, K=0): (a) analytic
             gradient vs central finite differences for every arm, (b) <S^2> exact for
             pcg / scf / r1 / foil at random parameters, (c) translation projector is
             idempotent and commutes with H, (d) E_sign bracket on the exact ground
             state returns E0, (e) a 150-iteration optimisation of every arm.
    time   : build the production instance (lattice options below), time ONE energy +
             gradient evaluation of every arm and print the parameter counts. Used for
             the cost estimate in submit_test.sbatch (takes ~1-3 min; not a smoke test).
    run    : optimise ONE arm (--ansatz) with one --seed on the instance, then compute
             observables and the sign budget; writes <outdir>/<tag>_<ansatz>_seed<seed>.json
             (skips if it exists unless --overwrite).
    report : read all JSONs for --tag, take the best seed per arm, print the table and
             the pre-committed verdict for each route (pcg, scf); writes <tag>_report.json.

Lattice / model options:
  --Lx INT        (default 4)    torus length in x, >= 2
  --Ly INT        (default 3)    torus length in y, >= 2
  --Ne INT        (default 8)    electron number (even), 2 <= Ne <= 2L-2; S^z = 0, S = 0
  --U FLOAT       (default 8.0)  on-site repulsion (t = 1)
  --tp FLOAT      (default -0.25) diagonal hopping t'
  --kproj {0,1}   (default 1)    apply the K=0 translation projection to every arm

Ansatz / optimisation options:
  --ansatz {foil, pcg, scf, nnbf, r1}  (run mode; default pcg)
  --hidden INT         (default 8)  hidden width of the pcg MLP; it defines the target
                                    parameter count unless --target-params is given
  --target-params INT  (default 0 = use pcg's count at --hidden) every matched arm
                                    (scf, nnbf, r1) picks the hidden width whose
                                    parameter count is closest to this number
  --mps-mult STR       (default "2,2,1,1") multiplicities m_0, m_1/2, m_1, m_3/2, ... of
                                    the scf virtual space (2 to 6 integers >= 1; virtual spin
                                    j_k = k/2, D = sum_k m_k (k+1)); default D = 13
  --seed INT           (default 0)  0 = paramagnetic / free-fermion init, >= 1 = AF (UHF)
                                    init with seed-dependent noise
  --maxiter INT        (default 3000) L-BFGS iteration cap (same for all arms)
  --max-hours FLOAT    (default 16.0) wall-clock cap of the optimisation (same for all)
  --chunk INT          (default 16384) configurations per evaluation chunk

Sign-budget options (as in P1.1):
  --lbfgs-iters INT    (default 4000) L-BFGS-B iterations per start
  --fn-iters INT       (default 10)   self-consistent fixed-node iterations
  --fn-floor FLOAT     (default 1e-3) guide floor
  --no-sign            (flag) skip the sign budget (debugging only)

I/O:
  --outdir PATH  (default <script dir>/results)
  --tag STR      (default auto "L{Lx}x{Ly}_N{Ne}_U{U}_tp{tp}_K{kproj}")
  --overwrite    (flag)

Decision rule (report mode; pre-committed, applied separately to R in {pcg, scf}; best
seed per arm; eps = relative energy error; dE_sign as above):
  g1 (exactness)   : |<S^2>_R| < 1e-10.
  g2 (correlations): dCs(R) <= dCs(nnbf) and dPd(R) <= dPd(nnbf) + 1e-4
                     (dCs = max_R |C_s - C_s^ED|, dPd = mean over non-overlapping d-wave
                     displacements |P_d - P_d^ED|).
  PASS      : g1, eps_R <= 0.7 eps_nnbf, dE_sign(R) <= 0.7 dE_sign(nnbf),
              eps_R <= 0.5 eps_foil, and g2.
  FAIL      : g1 violated, or eps_R >= eps_nnbf, or eps_R >= eps_foil.
  AMBIGUOUS : anything else. Pre-committed action: rerun the same array on P1.1 instance 0
              (--Ne 10, dim 627264); promote R only if it is not FAIL there and
              dE_sign(R) < dE_sign(nnbf) on both instances; otherwise KILL R.
  Diagnostic (no verdict): rho_proj = dE_sign(P_S psi_nnbf) / dE_sign(nnbf); rho_proj < 1
  means exact spin projection alone repairs part of the NNBF sign error.
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

ARMS = ["foil", "pcg", "scf", "nnbf", "r1"]
ROUTES = ["pcg", "scf"]
PASS_E, PASS_SIGN, PASS_FOIL = 0.7, 0.7, 0.5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ lattice / basis
class Lattice:
    def __init__(self, Lx, Ly, tp):
        self.Lx, self.Ly, self.L = Lx, Ly, Lx * Ly
        L = self.L
        self.T = np.zeros((L, L))
        for i in range(L):
            for (dx, dy, amp) in [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                                  (1, 1, tp), (1, -1, tp), (-1, 1, tp), (-1, -1, tp)]:
                j = self.shift(i, dx, dy)
                if j != i:
                    self.T[i, j] += amp
        self.T = 0.5 * (self.T + self.T.T)
        self.mask = np.zeros((L, L), bool)
        for i in range(L):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    self.mask[i, self.shift(i, dx, dy)] = True
        self.nn = [sorted({self.shift(i, d[0], d[1]) for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]} - {i})
                   for i in range(L)]

    def xy(self, i):
        return divmod(i, self.Ly)

    def shift(self, i, dx, dy):
        x, y = self.xy(i)
        return ((x + dx) % self.Lx) * self.Ly + (y + dy) % self.Ly

    def displacements(self):
        return [(dx, dy) for dx in range(self.Lx) for dy in range(self.Ly)]


class Species:
    def __init__(self, L, N):
        self.L, self.N = L, N
        self.states = np.array(sorted(sum(1 << k for k in c) for c in itertools.combinations(range(L), N)),
                               dtype=np.int64)
        self.dim = len(self.states)
        self.lookup = -np.ones(1 << L, dtype=np.int64)
        self.lookup[self.states] = np.arange(self.dim)
        self.bits = ((self.states[:, None] >> np.arange(L)[None, :]) & 1).astype(np.int8)
        self.occ = np.array([np.nonzero(b)[0] for b in self.bits], dtype=np.int64).reshape(self.dim, N)
        self._E = {}

    def E(self, i, j):
        key = (i, j)
        if key in self._E:
            return self._E[key]
        s = self.states
        if i == j:
            m = sp.diags(self.bits[:, i].astype(float), format="csr")
        else:
            ok = ((s >> j) & 1).astype(bool) & ~((s >> i) & 1).astype(bool)
            src = np.nonzero(ok)[0]
            s0 = s[src]
            lo, hi = min(i, j), max(i, j)
            between = np.zeros(len(src), dtype=np.int64)
            for k in range(lo + 1, hi):
                between += (s0 >> k) & 1
            sign = 1.0 - 2.0 * (between % 2)
            dst = self.lookup[s0 - (1 << j) + (1 << i)]
            m = sp.csr_matrix((sign, (dst, src)), shape=(self.dim, self.dim))
        self._E[key] = m
        return m

    def perm(self, p):
        """index map and fermion sign of the site permutation i -> p[i]."""
        new = p[self.occ]
        sgn = np.ones(self.dim)
        for a in range(new.shape[1]):
            for b in range(a + 1, new.shape[1]):
                sgn *= np.where(new[:, a] > new[:, b], -1.0, 1.0)
        return self.lookup[(1 << new).sum(1)], sgn


class Sector:
    def __init__(self, lat, Nu, Nd, U):
        self.lat, self.Nu, self.Nd, self.U = lat, Nu, Nd, U
        L = lat.L
        self.up, self.dn = Species(L, Nu), Species(L, Nd)
        self.shape = (self.up.dim, self.dn.dim)
        self.dim = self.up.dim * self.dn.dim
        self.Ku = sp.csr_matrix(sum(-lat.T[i, j] * self.up.E(i, j) for i in range(L) for j in range(L)
                                    if lat.T[i, j] != 0))
        self.Kd = sp.csr_matrix(sum(-lat.T[i, j] * self.dn.E(i, j) for i in range(L) for j in range(L)
                                    if lat.T[i, j] != 0))
        bu = self.up.bits.astype(float)
        bd = self.dn.bits.astype(float)
        self.bu, self.bd = bu, bd
        self.docc = bu @ bd.T
        nu = np.repeat(self.up.bits, self.dn.dim, axis=0)
        nd = np.tile(self.dn.bits, (self.up.dim, 1))
        n = (nu + nd).astype(np.int64)
        code = n @ (3 ** np.arange(L, dtype=np.int64))
        uniq, self.cidx = np.unique(code, return_inverse=True)
        first = np.zeros(len(uniq), dtype=np.int64)
        first[self.cidx[::-1]] = np.arange(self.dim)[::-1]
        self.n_c = n[first].astype(float)
        self.nC = len(uniq)
        self.iu = np.repeat(np.arange(self.up.dim), self.dn.dim)
        self.id = np.tile(np.arange(self.dn.dim), self.up.dim)
        self.Sz = 0.5 * (Nu - Nd)
        self.Smax = 0.5 * min(Nu + Nd, 2 * L - Nu - Nd)
        self._pp = None
        self._Hs = None
        # translation tables
        self.trans = []
        for dx in range(lat.Lx):
            for dy in range(lat.Ly):
                p = np.array([lat.shift(i, dx, dy) for i in range(L)])
                iu, su = self.up.perm(p)
                idd, sd = self.dn.perm(p)
                self.trans.append((iu, su, idd, sd))

    def H(self, P):
        return self.Ku @ P + (self.Kd @ P.T).T + self.U * self.docc * P

    def Hsparse(self):
        if self._Hs is None:
            self._Hs = (sp.kron(self.Ku, sp.identity(self.dn.dim)) + sp.kron(sp.identity(self.up.dim), self.Kd)
                        + sp.diags(self.U * self.docc.ravel())).tocsr()
        return self._Hs

    def spsm_pairs(self):
        if self._pp is None:
            L = self.lat.L
            self._pp = [(self.up.E(i, j), self.dn.E(j, i)) for i in range(L) for j in range(L) if i != j]
        return self._pp

    def S2(self, P):
        L = self.lat.L
        out = (self.Sz ** 2 - self.Sz) * P
        diag = sum(self.bu[:, i][:, None] * (1.0 - self.bd[:, i][None, :]) for i in range(L))
        out = out + diag * P
        for Eu, Ed in self.spsm_pairs():
            out -= Eu @ (Ed @ P.T).T
        return out

    def projS(self, S):
        ks = [k for k in np.arange(abs(self.Sz), self.Smax + 0.5, 1.0) if abs(k - S) > 1e-9]

        def apply(P):
            for k in ks:
                P = (self.S2(P) - k * (k + 1) * P) / (S * (S + 1) - k * (k + 1))
            return P
        return apply

    def projK(self, P):
        out = np.zeros_like(P)
        for iu, su, idd, sd in self.trans:
            out[np.ix_(iu, idd)] += (su[:, None] * sd[None, :]) * P
        return out / len(self.trans)


def make_sector(Lx, Ly, Ne, U, tp):
    return Sector(Lattice(Lx, Ly, tp), Ne // 2, Ne // 2, U)


# ============================================================================ ED + observables
def exact_states(sec, kproj, mu=5.0, k=2):
    """lowest k eigenpairs of H + mu S^2 (restricted to K=0 if kproj); S=0 targets."""
    shp = sec.shape

    def mv(x):
        P = x.reshape(shp)
        if kproj:
            P = sec.projK(P)
        Q = sec.H(P) + mu * sec.S2(P)
        if kproj:
            Q = sec.projK(Q)
        return Q.ravel()
    op = LinearOperator((sec.dim, sec.dim), matvec=mv, dtype=float)
    v0 = np.random.default_rng(1).standard_normal(sec.dim)
    if kproj:
        v0 = sec.projK(v0.reshape(shp)).ravel()
    w, v = eigsh(op, k=k, which="SA", v0=v0, tol=1e-11)
    o = np.argsort(w)
    out = []
    for kk in o:
        P = v[:, kk].reshape(shp)
        nrm = np.sum(P * P)
        out.append((P, float(np.sum(P * sec.H(P)) / nrm), float(np.sum(P * sec.S2(P)) / nrm)))
    return out


def observables(sec, P):
    lat = sec.lat
    L = lat.L
    nrm = np.sum(P * P)
    E = float(np.sum(P * sec.H(P)) / nrm)
    s2 = float(np.sum(P * sec.S2(P)) / nrm)
    w = P * P / nrm
    szu, szd = 0.5 * sec.bu, 0.5 * sec.bd
    Cs = {}
    for (dx, dy) in lat.displacements():
        if (dx, dy) == (0, 0):
            continue
        acc = 0.0
        for i in range(L):
            j = lat.shift(i, dx, dy)
            zi = szu[:, i][:, None] - szd[:, i][None, :]
            zj = szu[:, j][:, None] - szd[:, j][None, :]
            acc += np.sum(w * zi * zj)
        Cs[f"{dx},{dy}"] = acc / L
    deltas = [((1, 0), 1.0), ((-1, 0), 1.0), ((0, 1), -1.0), ((0, -1), -1.0)]

    def plus(i):
        return {i} | {lat.shift(i, *d) for d, _ in deltas}
    far = [R for R in lat.displacements() if not (plus(0) & plus(lat.shift(0, *R)))]
    Pd = {}
    for R in far:
        acc = 0.0
        for i in range(L):
            k = lat.shift(i, *R)
            for (d1, g1) in deltas:
                j = lat.shift(i, *d1)
                for (d2, g2) in deltas:
                    l = lat.shift(k, *d2)
                    for (a, b) in [(i, j), (j, i)]:
                        for (c, d) in [(k, l), (l, k)]:
                            Z = sec.up.E(a, c) @ (sec.dn.E(b, d) @ P.T).T
                            acc += 0.5 * g1 * g2 * np.sum(P * Z)
        Pd[f"{R[0]},{R[1]}"] = acc / L / nrm
    return dict(E=E, S2=s2, Cs=Cs, Pd=Pd)


# ============================================================================ sign budget (from P1.1)
def sign_constrained_min(K, starts, maxiter):
    res_all, best = [], (np.inf, None)

    def fg(a):
        Ka = K @ a
        nn = a @ a
        R = a @ Ka / nn
        return R, 2.0 * (Ka - R * a) / nn
    for name, a0 in starts:
        a0 = np.abs(a0) / np.linalg.norm(a0)
        r = minimize(fg, a0, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * len(a0),
                     options=dict(maxiter=maxiter, ftol=1e-15, gtol=1e-12, maxcor=30))
        res_all.append((name, float(r.fun), int(r.nit), bool(r.success)))
        if r.fun < best[0]:
            best = (float(r.fun), r.x / np.linalg.norm(r.x))
    return best[0], best[1], res_all


def _lowest(A, v0=None):
    if A.shape[0] <= 400:
        w, U = np.linalg.eigh(A.toarray())
        return w[:1], U[:, :1]
    try:
        return eigsh(A, k=1, which="SA", tol=1e-9, v0=v0, maxiter=50000)
    except ArpackNoConvergence as e:
        if len(e.eigenvalues):
            return e.eigenvalues, e.eigenvectors
        return eigsh(A, k=2, which="SA", tol=1e-8, maxiter=100000)


def fixed_node(K, iters, floor):
    Kc = K.tocoo()
    off = Kc.row != Kc.col
    r, c, v = Kc.row[off], Kc.col[off], Kc.data[off]
    diag = K.diagonal()
    bad = v > 0
    Hk = sp.csr_matrix((v[~bad], (r[~bad], c[~bad])), shape=K.shape)
    rb, cb, vb = r[bad], c[bad], v[bad]
    D = K.shape[0]
    g, v0, best, hist = np.ones(D), None, (np.inf, None), []
    for _ in range(iters + 1):
        gg = np.maximum(g, floor * g.max())
        d = diag + np.bincount(rb, weights=vb * gg[cb] / gg[rb], minlength=D)
        w, U = _lowest((Hk + sp.diags(d)).tocsr(), v0)
        g = np.abs(U[:, 0]); g /= np.linalg.norm(g)
        v0 = g
        hist.append(float(w[0]))
        if w[0] < best[0]:
            best = (float(w[0]), g.copy())
    phi = best[1]
    return best[0], float(phi @ (K @ phi)), phi, hist


def stoq_bound(K):
    Kc = K.tocoo()
    keep = (Kc.row == Kc.col) | (Kc.data < 0)
    Ks = sp.csr_matrix((Kc.data[keep], (Kc.row[keep], Kc.col[keep])), shape=K.shape)
    if K.shape[0] <= 400:
        return float(np.linalg.eigvalsh(Ks.toarray())[0])
    return float(eigsh(Ks, k=1, which="SA", tol=1e-10)[0][0])


def sign_budget(Hs, psi, p0, E0, args, seed=0):
    psi = psi / np.linalg.norm(psi)
    if p0 @ psi < 0:
        p0 = -p0
    Et = float(psi @ (Hs @ psi))
    s_t = np.where(psi >= 0, 1.0, -1.0)
    S = sp.diags(s_t)
    K = (S @ Hs @ S).tocsr()
    dE = Et - E0
    E_FN, E_FN_var, phi_fn, hist = fixed_node(K, args.fn_iters, args.fn_floor)
    uni = np.ones(len(psi)) + 0.01 * np.random.default_rng(seed).random(len(psi))
    E_lb, a_lb, starts = sign_constrained_min(
        K, [("abs_theta", psi), ("abs_exact", p0), ("uniform", uni), ("abs_fixed_node", phi_fn)],
        args.lbfgs_iters)
    E_up, _, src = min([(E_lb, a_lb, "lbfgsb"), (E_FN_var, phi_fn, "fixed_node_var"),
                        (E_FN, phi_fn, "fixed_node")], key=lambda z: z[0])
    E_stoq = stoq_bound(K)
    w_sign = float(np.sum(p0[s_t * np.sign(p0) < 0] ** 2))
    return dict(E=Et, dE_tot=dE, E_sign_up=E_up, E_sign_src=src, E_FN=E_FN, E_stoq=E_stoq,
                dE_sign=E_up - E0, dE_amp=Et - E_up, f_E_up=(E_up - E0) / dE if dE > 0 else float("nan"),
                f_E_lo=max(0.0, E_stoq - E0) / dE if dE > 0 else float("nan"), w_sign=w_sign,
                sign_starts=starts, fn_history=hist)


# ============================================================================ ansatz helpers
def cofactor(A):
    """batched cofactor det(A) A^{-T}, robust for singular A (SVD)."""
    U, s, Vt = np.linalg.svd(A)
    n = s.shape[1]
    prod_others = np.empty_like(s)
    for k in range(n):
        prod_others[:, k] = np.prod(np.delete(s, k, axis=1), axis=1)
    sgn = np.linalg.det(U) * np.linalg.det(Vt)
    adj = np.einsum("bji,bj,bkj->bik", Vt, prod_others, U)
    return sgn[:, None, None] * adj.transpose(0, 2, 1)


def charge_features(sec):
    """J features per charge config: n_i n_j (i<j), sum D, doublon-holon counts (10)."""
    lat, n = sec.lat, sec.n_c
    L = lat.L
    iu, ju = np.triu_indices(L, 1)
    d = (n == 2).astype(float)
    h = (n == 0).astype(float)
    nn = np.zeros((L, L))
    for i in range(L):
        nn[i, lat.nn[i]] = 1.0
    hn = h @ nn.T  # number of holon neighbours of each site
    dn = d @ nn.T
    dh = [np.sum(d * (hn == m), 1) for m in range(5)] + [np.sum(h * (dn == m), 1) for m in range(5)]
    F = np.concatenate([n[:, iu] * n[:, ju], d.sum(1, keepdims=True), np.array(dh).T], 1)
    return F, np.concatenate([d, h], 1)


def uhf(lat, Nu, Nd, U, seed, iters=3000, mix=0.3):
    rng = np.random.default_rng(seed)
    L = lat.L
    T = lat.T
    stag = np.array([(-1) ** sum(lat.xy(i)) for i in range(L)], float)
    nu = (Nu / L) * (1 + 0.3 * stag) + 0.02 * rng.standard_normal(L)
    nd = (Nd / L) * (1 - 0.3 * stag) + 0.02 * rng.standard_normal(L)
    for _ in range(iters):
        _, vu = np.linalg.eigh(-T + np.diag(U * nd))
        _, vd = np.linalg.eigh(-T + np.diag(U * nu))
        Pu, Pd = vu[:, :Nu], vd[:, :Nd]
        a, b = np.sum(Pu ** 2, 1), np.sum(Pd ** 2, 1)
        if max(np.abs(a - nu).max(), np.abs(b - nd).max()) < 1e-10:
            break
        nu, nd = (1 - mix) * nu + mix * a, (1 - mix) * nd + mix * b
    return Pu.copy(), Pd.copy()


class Base:
    """parameter packing shared by all arms."""

    def setup(self, shapes):
        self.shapes = shapes
        self.sizes = [int(np.prod(s)) for _, s in shapes]
        self.nparam = sum(self.sizes)

    def unpack(self, th):
        out, k = {}, 0
        for (name, shp), sz in zip(self.shapes, self.sizes):
            out[name] = th[k:k + sz].reshape(shp)
            k += sz
        return out

    def pack(self, d):
        return np.concatenate([np.asarray(d[name], float).ravel() for name, _ in self.shapes])

    def jastrow(self, p):
        return self.Fj @ p["jv"]


# ---------------------------------------------------------------------------- determinant family
class DetArm(Base):
    """foil (B=1, f general), pcg (B1, B2 independent, f general), r1 (B1=B2, F0 symmetric).
    phi(x) = e^{J(c)} det f_c[up(x), dn(x)], f_c = B1(c) f0 B2(c)^T."""

    def __init__(self, sec, kind, hidden, seed):
        self.sec, self.kind = sec, kind
        L = sec.lat.L
        self.L = L
        self.Fj, self.feat = charge_features(sec)
        self.bf = kind in ("pcg", "r1")
        self.two = kind == "pcg"
        self.symF = kind == "r1"
        self.H = hidden if self.bf else 0
        self.mpos = np.nonzero(sec.lat.mask)
        self.nmask = len(self.mpos[0])
        shapes = [("f0", (L, L)), ("jv", (self.Fj.shape[1],))]
        if self.bf:
            shapes += [("W", (self.H, 2 * L)), ("b", (self.H,)), ("V1", (self.H, self.nmask))]
            if self.two:
                shapes += [("V2", (self.H, self.nmask))]
        self.setup(shapes)
        self.theta0 = self.init(seed)

    @staticmethod
    def count(L, nf, nmask, kind, H):
        n = L * L + nf
        if kind == "pcg":
            n += H * (2 * L + 1) + 2 * H * nmask
        elif kind == "r1":
            n += H * (2 * L + 1) + H * nmask
        return n

    def init(self, seed):
        rng = np.random.default_rng(seed)
        sec, L = self.sec, self.L
        e, U = np.linalg.eigh(-sec.lat.T)
        F = U[:, :sec.Nd] @ U[:, :sec.Nd].T
        if seed >= 1 and not self.symF:
            Pu, Pd = uhf(sec.lat, sec.Nu, sec.Nd, sec.U, seed)
            F = Pu @ Pd.T
        noise = 0.02 if seed == 0 else 0.05
        R = rng.standard_normal((L, L))
        p = {"f0": F + noise * (0.5 * (R + R.T) if self.symF else R), "jv": np.zeros(self.Fj.shape[1])}
        p["jv"][L * (L - 1) // 2] = -1.0  # Gutzwiller start
        if self.bf:
            p["W"] = 0.3 * rng.standard_normal((self.H, 2 * L))
            p["b"] = 0.1 * rng.standard_normal(self.H)
            p["V1"] = 0.01 * rng.standard_normal((self.H, self.nmask))
            if self.two:
                p["V2"] = 0.01 * rng.standard_normal((self.H, self.nmask))
        return self.pack(p)

    def _B(self, hh, V):
        nC, L = hh.shape[0], self.L
        B = np.zeros((nC, L, L))
        B[:, self.mpos[0], self.mpos[1]] = hh @ V
        return B + np.eye(L)[None]

    def forward(self, th, chunk):
        p = self.unpack(th)
        sec = self.sec
        f0 = 0.5 * (p["f0"] + p["f0"].T) if self.symF else p["f0"]
        J = self.jastrow(p)
        cc = dict(f0=f0, expJ=np.exp(J - J.max()))
        if self.bf:
            hh = np.tanh(self.feat @ p["W"].T + p["b"])
            B1 = self._B(hh, p["V1"])
            B2 = self._B(hh, p["V2"]) if self.two else B1
            cc.update(h=hh, B1=B1, B2=B2, Fc=B1 @ f0 @ B2.transpose(0, 2, 1))
        phi = np.empty(sec.dim)
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            phi[sl] = np.linalg.det(self._A(cc, sl)) * cc["expJ"][sec.cidx[sl]]
        cc["phi"] = phi
        cc["p"] = p
        return phi.reshape(sec.shape), cc

    def _A(self, cc, sl):
        sec = self.sec
        ou = sec.up.occ[sec.iu[sl]]
        od = sec.dn.occ[sec.id[sl]]
        if self.bf:
            c = sec.cidx[sl]
            return cc["Fc"][c[:, None, None], ou[:, :, None], od[:, None, :]]
        return cc["f0"][ou[:, :, None], od[:, None, :]]

    def vjp(self, cc, vbar, chunk):
        sec, L, p = self.sec, self.L, cc["p"]
        vb = vbar.ravel()
        phi = cc["phi"]
        g = {}
        gJ = np.bincount(sec.cidx, weights=vb * phi, minlength=sec.nC)
        g["jv"] = self.Fj.T @ gJ
        Fbar = np.zeros(sec.nC * L * L if self.bf else L * L)
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            cof = cofactor(self._A(cc, sl))
            w = vb[sl] * cc["expJ"][sec.cidx[sl]]
            ou = sec.up.occ[sec.iu[sl]]
            od = sec.dn.occ[sec.id[sl]]
            base = (sec.cidx[sl] * L * L)[:, None, None] if self.bf else 0
            idx = base + ou[:, :, None] * L + od[:, None, :]
            Fbar += np.bincount(idx.ravel(), weights=(w[:, None, None] * cof).ravel(), minlength=Fbar.size)
        f0 = cc["f0"]
        if self.bf:
            Fb = Fbar.reshape(sec.nC, L, L)
            B1, B2 = cc["B1"], cc["B2"]
            f0bar = np.einsum("cki,ckl,clj->ij", B1, Fb, B2)
            B1bar = Fb @ B2 @ f0.T
            B2bar = Fb.transpose(0, 2, 1) @ B1 @ f0
            hh = cc["h"]
            e1 = B1bar[:, self.mpos[0], self.mpos[1]]
            if self.two:
                e2 = B2bar[:, self.mpos[0], self.mpos[1]]
                g["V1"], g["V2"] = hh.T @ e1, hh.T @ e2
                hb = e1 @ p["V1"].T + e2 @ p["V2"].T
            else:
                e1 = e1 + B2bar[:, self.mpos[0], self.mpos[1]]
                g["V1"] = hh.T @ e1
                hb = e1 @ p["V1"].T
            pre = hb * (1 - hh ** 2)
            g["W"] = pre.T @ self.feat
            g["b"] = pre.sum(0)
        else:
            f0bar = Fbar.reshape(L, L)
        g["f0"] = 0.5 * (f0bar + f0bar.T) if self.symF else f0bar
        return self.pack(g)


# ---------------------------------------------------------------------------- spin-resolved NNBF
class NNBF(Base):
    def __init__(self, sec, hidden, seed):
        self.sec = sec
        L, Nu, Nd = sec.lat.L, sec.Nu, sec.Nd
        self.L, self.Nu, self.Nd, self.H = L, Nu, Nd, hidden
        self.Fj, _ = charge_features(sec)
        nout = L * (Nu + Nd)
        self.setup([("P0", (nout,)), ("jv", (self.Fj.shape[1],)), ("W1", (2 * L, hidden)),
                    ("b1", (hidden,)), ("W2", (hidden, nout)), ("b2", (nout,))])
        rng = np.random.default_rng(seed)
        Pu, Pd = uhf(sec.lat, Nu, Nd, sec.U, seed)
        jv = np.zeros(self.Fj.shape[1]); jv[L * (L - 1) // 2] = -1.0
        self.theta0 = self.pack(dict(P0=np.concatenate([Pu.ravel(), Pd.ravel()]), jv=jv,
                                     W1=rng.standard_normal((2 * L, hidden)) / np.sqrt(2 * L),
                                     b1=np.zeros(hidden), W2=1e-3 * rng.standard_normal((hidden, nout)),
                                     b2=np.zeros(nout)))

    @staticmethod
    def count(L, nf, Nu, Nd, H):
        nout = L * (Nu + Nd)
        return nout + nf + 2 * L * H + H + H * nout + nout

    def _orb(self, p, sl):
        sec, L, Nu, Nd = self.sec, self.L, self.Nu, self.Nd
        x = np.concatenate([sec.up.bits[sec.iu[sl]], sec.dn.bits[sec.id[sl]]], 1).astype(float) * 2 - 1
        h = np.tanh(x @ p["W1"] + p["b1"])
        out = h @ p["W2"] + p["b2"] + p["P0"]
        B = x.shape[0]
        Ou = out[:, :L * Nu].reshape(B, L, Nu)
        Od = out[:, L * Nu:].reshape(B, L, Nd)
        ru = sec.up.occ[sec.iu[sl]]
        rd = sec.dn.occ[sec.id[sl]]
        Au = np.take_along_axis(Ou, ru[:, :, None], axis=1)
        Ad = np.take_along_axis(Od, rd[:, :, None], axis=1)
        return x, h, Au, Ad, ru, rd

    def forward(self, th, chunk):
        p = self.unpack(th)
        sec = self.sec
        J = self.jastrow(p)
        expJ = np.exp(J - J.max())
        phi = np.empty(sec.dim)
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            _, _, Au, Ad, _, _ = self._orb(p, sl)
            phi[sl] = np.linalg.det(Au) * np.linalg.det(Ad) * expJ[sec.cidx[sl]]
        return phi.reshape(sec.shape), dict(p=p, phi=phi, expJ=expJ)

    def vjp(self, cc, vbar, chunk):
        sec, p, L, Nu, Nd = self.sec, cc["p"], self.L, self.Nu, self.Nd
        vb = vbar.ravel()
        g = {k: np.zeros(s) for k, s in self.shapes}
        gJ = np.bincount(sec.cidx, weights=vb * cc["phi"], minlength=sec.nC)
        g["jv"] = self.Fj.T @ gJ
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            x, h, Au, Ad, ru, rd = self._orb(p, sl)
            B = x.shape[0]
            du, dd = np.linalg.det(Au), np.linalg.det(Ad)
            w = vb[sl] * cc["expJ"][sec.cidx[sl]]
            Gu = cofactor(Au) * (w * dd)[:, None, None]
            Gd = cofactor(Ad) * (w * du)[:, None, None]
            dOu = np.zeros((B, L, Nu)); dOd = np.zeros((B, L, Nd))
            bi = np.arange(B)[:, None]
            np.add.at(dOu, (bi, ru), Gu)
            np.add.at(dOd, (bi, rd), Gd)
            dout = np.concatenate([dOu.reshape(B, -1), dOd.reshape(B, -1)], 1)
            g["P0"] += dout.sum(0)
            g["b2"] += dout.sum(0)
            g["W2"] += h.T @ dout
            dpre = (dout @ p["W2"].T) * (1 - h ** 2)
            g["W1"] += x.T @ dpre
            g["b1"] += dpre.sum(0)
        return self.pack(g)


# ---------------------------------------------------------------------------- spin-charge factorised SU(2) MPS
def cg_half(j, m, s, J, M):
    """<j m; 1/2 s | J M> for s = +-1/2."""
    if abs(m + s - M) > 1e-9 or abs(m) > j + 1e-9 or abs(M) > J + 1e-9:
        return 0.0
    if abs(J - (j + 0.5)) < 1e-9:
        return np.sqrt((j + M + 0.5) / (2 * j + 1)) if s > 0 else np.sqrt((j - M + 0.5) / (2 * j + 1))
    if abs(J - (j - 0.5)) < 1e-9 and j > 0:
        return -np.sqrt((j - M + 0.5) / (2 * j + 1)) if s > 0 else np.sqrt((j + M + 0.5) / (2 * j + 1))
    return 0.0


class SCF(Base):
    """psi(x) = s_ord(x) e^{J(c)} det Psi0[occ2(c)] chi_c(sigma) (see module docstring)."""

    def __init__(self, sec, hidden, seed, mult=(2, 2, 1, 1)):
        self.sec = sec
        lat = sec.lat
        L, Ne = lat.L, sec.Nu + sec.Nd
        self.L, self.Ne, self.H = L, Ne, hidden
        self.Fj, self.feat = charge_features(sec)
        # virtual space: list of (j, multiplicity index a, m)
        js = [0.5 * k for k in range(len(mult))]
        self.vs = [(j, a, -j + q) for j, mu in zip(js, mult) for a in range(mu) for q in range(int(2 * j + 1))]
        self.D = len(self.vs)
        blocks = [(js[k], js[k + 1]) for k in range(len(js) - 1)] + [(js[k + 1], js[k]) for k in range(len(js) - 1)]
        mdict = dict(zip(js, mult))
        self.wlist = [(jl, jr, a, b) for (jl, jr) in blocks for a in range(mdict[jl]) for b in range(mdict[jr])]
        self.nw = len(self.wlist)
        G = np.zeros((self.nw, 2, self.D, self.D))
        for w, (jl, jr, a, b) in enumerate(self.wlist):
            for si, s in enumerate((0.5, -0.5)):
                for p_, (j1, a1, m1) in enumerate(self.vs):
                    if j1 != jl or a1 != a:
                        continue
                    for q_, (j2, a2, m2) in enumerate(self.vs):
                        if j2 != jr or a2 != b:
                            continue
                        G[w, si, p_, q_] = cg_half(j1, m1, s, j2, m2)
        self.G = G
        self.j0 = [k for k, (j, a, m) in enumerate(self.vs) if j == 0.0]
        self.lvec = np.zeros(self.D); self.lvec[self.j0[0]] = 1.0
        # charge configurations: doubled spinless orbitals (2L) occupied
        n = sec.n_c.astype(int)
        occ2 = np.array([sorted([2 * i for i in range(L) if n[c, i] >= 1] + [2 * i + 1 for i in range(L) if n[c, i] == 2])
                         for c in range(sec.nC)], dtype=np.int64)
        self.occ2 = occ2
        # per basis state: spin sigma (+1 up, -1 dn, 0 otherwise) per site, order sign
        bu = sec.up.bits.astype(np.int64)
        bd = sec.dn.bits.astype(np.int64)
        self.bu, self.bd = bu, bd
        # cumulative count of up electrons at sites > j: cu[u, j]
        self.cu_gt = np.cumsum(bu[:, ::-1], axis=1)[:, ::-1] - bu  # sum_{i>j} n_i_up
        self.setup([("jv", (self.Fj.shape[1],)), ("Psi0", (2 * L, Ne)), ("A", (hidden, 2 * L)), ("a", (hidden,)),
                    ("W0", (L, self.nw)), ("Uw", (L, self.nw, hidden)), ("r", (mult[0],))])
        rng = np.random.default_rng(seed)
        # charge orbitals: lowest Ne orbitals of the doubled hopping (slot a and b copies)
        T2 = np.kron(sec.lat.T, np.ones((2, 2)))
        _, U2 = np.linalg.eigh(-T2 + 1e-3 * np.diag(np.tile([0, 1], L)))
        jv = np.zeros(self.Fj.shape[1]); jv[L * (L - 1) // 2] = -1.0
        W0 = np.zeros((L, self.nw))
        for w, (jl, jr, a, b) in enumerate(self.wlist):
            if (jl, jr) in ((0.0, 0.5), (0.5, 0.0)):
                W0[:, w] = 1.0 if (a == 0 and b == 0) else 0.1
        W0 += 0.05 * rng.standard_normal(W0.shape)
        self.theta0 = self.pack(dict(jv=jv, Psi0=U2[:, :Ne] + 0.02 * rng.standard_normal((2 * L, Ne)),
                                     A=0.3 * rng.standard_normal((hidden, 2 * L)), a=0.1 * rng.standard_normal(hidden),
                                     W0=W0, Uw=0.01 * rng.standard_normal((L, self.nw, hidden)),
                                     r=np.eye(mult[0])[0] + 0.05 * rng.standard_normal(mult[0])))

    @staticmethod
    def count(L, nf, Ne, nw, m0, H):
        return nf + 2 * L * Ne + H * (2 * L + 1) + L * nw * (H + 1) + m0

    def _order_sign(self, sl):
        sec = self.sec
        iu, idd = sec.iu[sl], sec.id[sl]
        par = np.sum(self.bd[idd] * self.cu_gt[iu], 1)
        return 1.0 - 2.0 * (par % 2)

    def _charge(self, p):
        A = p["Psi0"][self.occ2]  # (nC, Ne, Ne)
        return A, np.linalg.det(A)

    def _chain(self, p, hh, sl):
        """returns sigma index (B,L) in {0 up,1 dn,-1 none}, M (B,L,D,D), left/right partials, chi."""
        sec = self.sec
        iu, idd, c = sec.iu[sl], sec.id[sl], sec.cidx[sl]
        u, d = self.bu[iu], self.bd[idd]
        single = (u + d) == 1
        sig = np.where(single, np.where(u == 1, 0, 1), -1)
        B, L, D, nw = len(c), self.L, self.D, self.nw
        w = p["W0"][None] + (hh[c] @ p["Uw"].reshape(L * nw, self.H).T).reshape(B, L, nw)
        Mall = (w.reshape(B * L, nw) @ self.G.reshape(nw, 2 * D * D)).reshape(B, L, 2, D, D)
        M = np.take_along_axis(Mall, np.maximum(sig, 0)[:, :, None, None, None], axis=2)[:, :, 0]
        M[~single] = np.eye(D)
        rv = np.zeros(D); rv[self.j0] = p["r"]
        left = np.empty((B, L + 1, D)); left[:, 0] = self.lvec
        for k in range(L):
            left[:, k + 1] = np.matmul(left[:, k, None, :], M[:, k])[:, 0]
        right = np.empty((B, L + 1, D)); right[:, L] = rv
        for k in range(L - 1, -1, -1):
            right[:, k] = np.matmul(M[:, k], right[:, k + 1, :, None])[:, :, 0]
        chi = left[:, L] @ rv
        return sig, single, left, right, chi

    def forward(self, th, chunk):
        p = self.unpack(th)
        sec = self.sec
        J = self.jastrow(p)
        expJ = np.exp(J - J.max())
        A, detA = self._charge(p)
        hh = np.tanh(self.feat @ p["A"].T + p["a"])
        amp_c = expJ * detA
        phi = np.empty(sec.dim)
        chis = np.empty(sec.dim)
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            chi = self._chain(p, hh, sl)[4]
            chis[sl] = chi
            phi[sl] = self._order_sign(sl) * amp_c[sec.cidx[sl]] * chi
        return phi.reshape(sec.shape), dict(p=p, phi=phi, chi=chis, expJ=expJ, A=A, detA=detA, hh=hh)

    def vjp(self, cc, vbar, chunk):
        sec, p, L = self.sec, cc["p"], self.L
        vb = vbar.ravel()
        g = {k: np.zeros(s) for k, s in self.shapes}
        gJ = np.bincount(sec.cidx, weights=vb * cc["phi"], minlength=sec.nC)
        g["jv"] = self.Fj.T @ gJ
        cof = cofactor(cc["A"])
        # d phi / d Psi0 = s_ord chi expJ cof(A_c); accumulated per charge configuration
        wdet = np.zeros(sec.nC)
        amp_c = cc["expJ"] * cc["detA"]
        hh = cc["hh"]
        hbar = np.zeros_like(hh)
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            c = sec.cidx[sl]
            so = self._order_sign(sl)
            sig, single, left, right, chi = self._chain(p, hh, sl)
            wdet += np.bincount(c, weights=vb[sl] * so * chi, minlength=sec.nC)
            wchi = vb[sl] * so * amp_c[c]  # dE/dchi
            # dchi/dM_k = left_k (x) right_{k+1}
            Mbar = wchi[:, None, None, None] * left[:, :L, :, None] * right[:, 1:, None, :]  # (B,L,D,D)
            Mbar[~single] = 0.0
            B, D, nw = len(c), self.D, self.nw
            M2 = np.zeros((B, L, 2, D, D))
            M2[:, :, 0] = Mbar * (sig == 0)[:, :, None, None]
            M2[:, :, 1] = Mbar * (sig == 1)[:, :, None, None]
            wbar = (M2.reshape(B * L, 2 * D * D) @ self.G.reshape(nw, 2 * D * D).T).reshape(B, L * nw)
            g["W0"] += wbar.sum(0).reshape(L, nw)
            g["Uw"] += (wbar.T @ hh[c]).reshape(L, nw, self.H)
            hb = wbar @ p["Uw"].reshape(L * nw, self.H)
            for hcol in range(self.H):
                hbar[:, hcol] += np.bincount(c, weights=hb[:, hcol], minlength=sec.nC)
            # r gradient
            rb = np.einsum("bd,b->d", left[:, L], wchi)
            g["r"] += rb[self.j0]
        # Psi0
        gA = (wdet * cc["expJ"])[:, None, None] * cof  # (nC, Ne, Ne)
        Ps = np.zeros((2 * L, self.Ne))
        np.add.at(Ps, self.occ2, gA)
        g["Psi0"] = Ps
        pre = hbar * (1 - hh ** 2)
        g["A"] = pre.T @ self.feat
        g["a"] = pre.sum(0)
        return self.pack(g)


# ============================================================================ state wrapper
class State:
    """psi = P_K P_S phi_theta; exact energy and gradient."""

    def __init__(self, sec, arm, sproj, kproj, chunk):
        self.sec, self.arm, self.chunk = sec, arm, chunk
        self.PS = sec.projS(0) if sproj else None
        self.kproj = kproj

    def psi(self, phi):
        if self.PS is not None:
            phi = self.PS(phi)
        if self.kproj:
            phi = self.sec.projK(phi)
        return phi

    def energy_grad(self, th):
        sec = self.sec
        phi, cc = self.arm.forward(th, self.chunk)
        psi = self.psi(phi)
        nrm = np.sum(psi * psi)
        Hp = sec.H(psi)
        E = float(np.sum(psi * Hp) / nrm)
        v = 2.0 * (Hp - E * psi) / nrm
        vbar = self.psi(v)  # P_S, P_K symmetric and commuting
        return E, self.arm.vjp(cc, vbar, self.chunk), psi


def build_arm(sec, kind, hidden, seed, target, mult):
    L = sec.lat.L
    nf = charge_features(sec)[0].shape[1]
    nmask = int(sec.lat.mask.sum())
    if kind == "foil":
        return DetArm(sec, "foil", 0, seed), True
    if kind == "pcg" and target == 0:
        return DetArm(sec, "pcg", hidden, seed), True
    if target == 0:
        target = DetArm.count(L, nf, nmask, "pcg", hidden)
    nw = 2 * sum(mult[k] * mult[k + 1] for k in range(len(mult) - 1))
    cnt = {"pcg": lambda H: DetArm.count(L, nf, nmask, "pcg", H),
           "r1": lambda H: DetArm.count(L, nf, nmask, "r1", H),
           "nnbf": lambda H: NNBF.count(L, nf, sec.Nu, sec.Nd, H),
           "scf": lambda H: SCF.count(L, nf, sec.Nu + sec.Nd, nw, mult[0], H)}[kind]
    Hb = min(range(1, 200), key=lambda H: abs(cnt(H) - target))
    if kind == "pcg":
        return DetArm(sec, "pcg", Hb, seed), True
    if kind == "r1":
        return DetArm(sec, "r1", Hb, seed), False
    if kind == "nnbf":
        return NNBF(sec, Hb, seed), False
    return SCF(sec, Hb, seed, mult), False


class Stop(Exception):
    pass


def optimise(st, th0, maxiter, max_hours, logf=log):
    t0 = time.time()
    best = {"E": np.inf, "th": th0.copy()}
    it = [0]

    def f(th):
        E, g, _ = st.energy_grad(th)
        if not np.isfinite(E):
            return 1e3, np.zeros_like(th)
        if E < best["E"]:
            best["E"], best["th"] = E, th.copy()
        return E, g

    def cb(th):
        it[0] += 1
        if it[0] % 50 == 0:
            logf(f"  iter {it[0]:5d}  E_best = {best['E']:.10f}  ({(time.time() - t0) / 60:.1f} min)")
        if time.time() - t0 > 3600 * max_hours:
            raise StopIteration
    try:
        minimize(f, th0, jac=True, method="L-BFGS-B", callback=cb,
                 options=dict(maxiter=maxiter, maxcor=30, ftol=1e-15, gtol=1e-10))
    except StopIteration:
        logf("  wall-clock cap reached")
    return best["th"], best["E"], it[0], time.time() - t0


# ============================================================================ modes
def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="P4.1 round-2 exact-SU(2) test (see module docstring)")
    p.add_argument("--mode", required=True, choices=["smoke", "time", "run", "report"])
    p.add_argument("--Lx", type=int, default=4)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--Ne", type=int, default=8)
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--tp", type=float, default=-0.25)
    p.add_argument("--kproj", type=int, default=1, choices=[0, 1])
    p.add_argument("--ansatz", default="pcg", choices=ARMS)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--target-params", type=int, default=0, dest="target_params")
    p.add_argument("--mps-mult", default="2,2,1,1", dest="mps_mult")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=3000)
    p.add_argument("--max-hours", type=float, default=16.0, dest="max_hours")
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--lbfgs-iters", type=int, default=4000, dest="lbfgs_iters")
    p.add_argument("--fn-iters", type=int, default=10, dest="fn_iters")
    p.add_argument("--fn-floor", type=float, default=1e-3, dest="fn_floor")
    p.add_argument("--no-sign", action="store_true", dest="no_sign")
    p.add_argument("--outdir", default=os.path.join(here, "results"))
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    a.mult = tuple(int(v) for v in a.mps_mult.split(","))
    if not (2 <= len(a.mult) <= 6) or min(a.mult) < 1:
        p.error("--mps-mult needs 2 to 6 positive integers")
    L = a.Lx * a.Ly
    if a.Lx < 2 or a.Ly < 2 or a.Ne % 2 or not (2 <= a.Ne <= 2 * L - 2):
        p.error("invalid lattice / Ne")
    if a.tag is None:
        a.tag = f"L{a.Lx}x{a.Ly}_N{a.Ne}_U{a.U:g}_tp{a.tp:g}_K{a.kproj}"
    return a


def fd_check(st, th, n=10, h=1e-5, seed=11):
    E, g, _ = st.energy_grad(th)
    rng = np.random.default_rng(seed)
    err = 0.0
    for k in rng.choice(len(th), size=min(n, len(th)), replace=False):
        e = np.zeros(len(th)); e[k] = h
        fd = (st.energy_grad(th + e)[0] - st.energy_grad(th - e)[0]) / (2 * h)
        err = max(err, abs(fd - g[k]) / max(1e-6, abs(fd) + abs(g[k])))
    return err


def smoke(a):
    t0 = time.time()
    ok = True
    sec = make_sector(2, 3, 6, 6.0, -0.3)
    Hs = sec.Hsparse()
    # (c) translation projector
    X = np.random.default_rng(0).standard_normal(sec.shape)
    PX = sec.projK(X)
    idem = np.abs(sec.projK(PX) - PX).max()
    comm = np.abs(sec.projK(sec.H(X)) - sec.H(PX)).max()
    cs = idem < 1e-12 and comm < 1e-12
    ok &= cs
    gs = exact_states(sec, 1, k=2)
    P0, E0, s20 = gs[0]
    print(f"2x3 torus Ne=6 U=6 t'=-0.3 dim={sec.dim}: E0(S=0,K=0)={E0:.8f} <S^2>={s20:.1e}; "
          f"P_K idempotent {idem:.1e}, [P_K,H] {comm:.1e} [{'PASS' if cs else 'FAIL'}]")
    for kind in ARMS:
        arm, sproj = build_arm(sec, kind, 3, 3, 0, (1, 1, 1, 1))
        st = State(sec, arm, sproj, 1, 4096)
        th = arm.theta0 + 0.05 * np.random.default_rng(7).standard_normal(arm.nparam)
        E, g, psi = st.energy_grad(th)
        s2 = np.sum(psi * sec.S2(psi)) / np.sum(psi * psi)
        err = fd_check(st, th)
        exact = kind != "nnbf"
        s2ok = abs(s2) < 1e-10 if exact else True
        gok = err < 1e-5
        ok &= s2ok and gok
        print(f"  {kind:5s} P={arm.nparam:4d} E={E:+.6f} <S^2>={s2:+.2e} "
              f"[{'PASS' if s2ok else 'FAIL'}{'' if exact else ' (not required)'}]  grad rel.err={err:.1e} "
              f"[{'PASS' if gok else 'FAIL'}]")
    # (d) sign budget on the exact ground state returns E0
    sb = sign_budget(Hs, P0.ravel(), P0.ravel(), E0, argparse.Namespace(fn_iters=3, fn_floor=1e-3, lbfgs_iters=500))
    sok = abs(sb["E_sign_up"] - E0) < 1e-8
    ok &= sok
    print(f"  sign budget on psi_0: E_sign_up - E0 = {sb['E_sign_up'] - E0:.1e} [{'PASS' if sok else 'FAIL'}]")
    for kind in ARMS:
        arm, sproj = build_arm(sec, kind, 3, 0, 0, (1, 1, 1, 1))
        st = State(sec, arm, sproj, 1, 4096)
        th, E, nit, dt = optimise(st, arm.theta0, 150, 0.01, logf=lambda *x: None)
        psi = st.psi(arm.forward(th, 4096)[0])
        sb = sign_budget(Hs, psi.ravel(), P0.ravel(), E0,
                         argparse.Namespace(fn_iters=3, fn_floor=1e-3, lbfgs_iters=500))
        print(f"  opt {kind:5s}: E={E:.8f} eps={(E - E0) / abs(E0):.2e} dE_sign={sb['dE_sign']:.2e} "
              f"({nit} it, {dt:.1f}s)")
    print(f"(the 2x3 cluster has fewer states than most arms have parameters: optimisation lines only show "
          f"that the code runs)\nSMOKE {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.1f}s)")


def timing(a):
    t0 = time.time()
    sec = make_sector(a.Lx, a.Ly, a.Ne, a.U, a.tp)
    log(f"instance {a.tag}: dim={sec.dim} nC={sec.nC} (built in {time.time() - t0:.1f}s)")
    for kind in ARMS:
        arm, sproj = build_arm(sec, kind, a.hidden, 1, a.target_params, a.mult)
        st = State(sec, arm, sproj, a.kproj, a.chunk)
        t1 = time.time()
        E, g, psi = st.energy_grad(arm.theta0)
        log(f"  {kind:5s} P={arm.nparam:5d} hidden={getattr(arm, 'H', 0):3d} sproj={int(sproj)}  "
            f"E(init)={E:+.5f}  energy+grad {time.time() - t1:.1f}s")


def run(a):
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"{a.tag}_{a.ansatz}_seed{a.seed}.json")
    if os.path.exists(out) and not a.overwrite:
        print(f"exists: {out}  (use --overwrite)")
        return
    t0 = time.time()
    sec = make_sector(a.Lx, a.Ly, a.Ne, a.U, a.tp)
    log(f"instance {a.tag}: dim={sec.dim} charge configs={sec.nC}")
    gs = exact_states(sec, a.kproj, k=2)
    (P0, E0, s20), (P1, E1, s21) = gs
    L = sec.lat.L
    log(f"ED (S=0{', K=0' if a.kproj else ''}): E0={E0:.10f} <S^2>={s20:.1e}; next same-sector level "
        f"E1={E1:.8f} <S^2>={s21:.2f} gap/V={(E1 - E0) / L:.4f} ({time.time() - t0:.0f}s)")
    arm, sproj = build_arm(sec, a.ansatz, a.hidden, a.seed, a.target_params, a.mult)
    st = State(sec, arm, sproj, a.kproj, a.chunk)
    log(f"arm {a.ansatz}: {arm.nparam} parameters, hidden {getattr(arm, 'H', 0)}, spin projection {int(sproj)}")
    th, E, nit, dt = optimise(st, arm.theta0, a.maxiter, a.max_hours)
    psi = st.psi(arm.forward(th, a.chunk)[0])
    obs, obs0 = observables(sec, psi), observables(sec, P0)
    res = dict(tag=a.tag, ansatz=a.ansatz, seed=a.seed, nparam=arm.nparam, hidden=getattr(arm, "H", 0),
               sproj=int(sproj), iters=nit, seconds=dt, E0=E0, E1_same_sector=E1, S2_E1=s21,
               gap_per_site=(E1 - E0) / L, E=obs["E"], eps=(obs["E"] - E0) / abs(E0), S2=obs["S2"],
               Cs=obs["Cs"], Pd=obs["Pd"], Cs_exact=obs0["Cs"], Pd_exact=obs0["Pd"], args=vars(a))
    Hs = sec.Hsparse()
    if not a.no_sign:
        t1 = time.time()
        res["sign"] = sign_budget(Hs, psi.ravel(), P0.ravel(), E0, a, a.seed)
        log(f"sign budget: dE_tot={res['sign']['dE_tot']:.5f} dE_sign={res['sign']['dE_sign']:.5f} "
            f"f_E_up={res['sign']['f_E_up']:.3f} ({time.time() - t1:.0f}s)")
    if a.ansatz == "nnbf":  # post-hoc exact spin projection (diagnostic, not re-optimised)
        pp = sec.projS(0)(psi)
        o2 = observables(sec, pp)
        res["proj"] = dict(E=o2["E"], eps=(o2["E"] - E0) / abs(E0), S2=o2["S2"], Cs=o2["Cs"], Pd=o2["Pd"],
                           weight_S0=float(np.sum(pp * pp) / np.sum(psi * psi)))
        if not a.no_sign:
            res["proj"]["sign"] = sign_budget(Hs, pp.ravel(), P0.ravel(), E0, a, a.seed)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    log(f"done: E={obs['E']:.10f} eps={res['eps']:.3e} <S^2>={obs['S2']:.3e} -> {out}")


def report(a):
    best = {}
    for fn in glob.glob(os.path.join(a.outdir, f"{a.tag}_*_seed*.json")):
        r = json.load(open(fn))
        if r["ansatz"] not in best or r["E"] < best[r["ansatz"]]["E"]:
            best[r["ansatz"]] = r
    need = {"foil", "nnbf"}
    if not need <= set(best):
        print(f"missing results (have {sorted(best)}); need foil and nnbf")
        sys.exit(1)

    def dcs(r):
        return max(abs(r["Cs"][k] - r["Cs_exact"][k]) for k in r["Cs"])

    def dpd(r):
        v = [abs(r["Pd"][k] - r["Pd_exact"][k]) for k in r["Pd"]]
        return float(np.mean(v)) if v else 0.0

    def dsg(r):
        return r["sign"]["dE_sign"] if "sign" in r else float("nan")
    rows = []
    for k in ARMS:
        if k not in best:
            continue
        r = best[k]
        rows.append(dict(ansatz=k, seed=r["seed"], nparam=r["nparam"], E=r["E"], eps=r["eps"], S2=r["S2"],
                         dE_sign=dsg(r), f_E=r.get("sign", {}).get("f_E_up", float("nan")), dCs=dcs(r), dPd=dpd(r)))
        print(f"{k:5s} seed={r['seed']} P={r['nparam']:5d} E={r['E']:.8f} eps={r['eps']:.3e} <S2>={r['S2']:+.1e} "
              f"dE_sign={dsg(r):.4e} f_E={rows[-1]['f_E']:.3f} dCs={dcs(r):.2e} dPd={dpd(r):.2e}")
    nb, fo = best["nnbf"], best["foil"]
    rho = float("nan")
    if "proj" in nb and "sign" in nb["proj"] and "sign" in nb:
        rho = nb["proj"]["sign"]["dE_sign"] / nb["sign"]["dE_sign"]
        print(f"diagnostic: P_S psi_nnbf eps={nb['proj']['eps']:.3e} weight(S=0)={nb['proj']['weight_S0']:.4f} "
              f"rho_proj = dE_sign(P_S nnbf)/dE_sign(nnbf) = {rho:.3f}")
    verdicts = {}
    for R in ROUTES:
        if R not in best:
            continue
        r = best[R]
        g1 = abs(r["S2"]) < 1e-10
        g2 = dcs(r) <= dcs(nb) and dpd(r) <= dpd(nb) + 1e-4
        mE, mS, mF = r["eps"] / nb["eps"], dsg(r) / dsg(nb), r["eps"] / fo["eps"]
        if (not g1) or mE >= 1.0 or mF >= 1.0:
            v = "FAIL"
        elif mE <= PASS_E and mS <= PASS_SIGN and mF <= PASS_FOIL and g2:
            v = "PASS"
        else:
            v = "AMBIGUOUS"
        verdicts[R] = dict(M_E=mE, M_sign=mS, M_foil=mF, g1=g1, g2=g2, verdict=v)
        print(f"{R}: M_E=eps/eps_nnbf={mE:.3f}  M_sign=dE_sign/dE_sign_nnbf={mS:.3f}  M_foil={mF:.3f}  "
              f"g1={'ok' if g1 else 'VIOLATED'} g2={'ok' if g2 else 'VIOLATED'} -> {v}")
    with open(os.path.join(a.outdir, f"{a.tag}_report.json"), "w") as fh:
        json.dump(dict(rows=rows, rho_proj=rho, verdicts=verdicts), fh, indent=1)


def main(argv=None):
    a = parse_args(argv)
    {"smoke": smoke, "time": timing, "run": run, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
