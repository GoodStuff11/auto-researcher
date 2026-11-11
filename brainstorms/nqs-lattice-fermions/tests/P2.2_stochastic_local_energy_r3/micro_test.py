#!/usr/bin/env python3
"""
P2.2 round-3 decisive test: exactly-cheap local energies by window-keyed
backflow, versus exact enumeration, screening and semistochastic local
energies of an unrestricted neural-network-backflow-Jastrow (NNBF-J) state,
on stretched hydrogen rings with polarization (p) functions, at N = 6 and
N = 8 electrons, all numbers measured on the same instance at the same N.

Routes under test
-----------------
R3.1 WIBJ   Window-input backflow x all-orbital Jastrow (an ansatz whose EXACT
            local energy is cheap).  psi(n) = J(n) det Phi(n_W)[occ n], where
            the backflow network sees only the occupations n_W of the window
            W (the lowest K_W spatial MOs, both spins; K_W = N by default),
            Phi is M x N over ALL orbitals, and
            J(n) = exp(sum_p u_p n_p + sum_{p<q} v_pq n_p n_q) over all
            spin-orbitals.  All connected n' that share the same window
            change share Phi(n'_W): the local energy is exact with one
            backflow evaluation per distinct window key (at most
            ~N^2 W_v^2 / 4 keys, independent of M) plus O(1) arithmetic per
            connected term (2x2 Green's-function minor x Jastrow ratio).
            Estimator: KEYX (exact).
R3.2 SHRJ   Unrestricted NNBF-J trained with an external-input shrinkage
            penalty mu * ||W1[:, ext]||^2 (lambda-continuation from WIBJ,
            mu = inf, to NNBFJ, mu = 0), with the window-keyed evaluation
            used as an additive control variate (PFCV): the surrogate
            evaluates the network with the external input frozen at n, it
            is summed exactly over all terms, the terms with |y| > eps are
            evaluated exactly, and the residual x - y on the rest gets K
            draws from p ~ 0.9 |y| + 0.1 |h|.  Unbiased.  The penalty makes
            the residual (the external-input response) small by
            construction.
Controls (same instance, same N, unrestricted NNBFJ state, same seed)
--------------------------------------------------------------------
EXACT       full enumeration with exact incremental NNBF-J amplitudes.
SCREEN      keep |H_nn'| > eps (biased; admissible only if
            |bias| + 2 SE <= --bias-tol).
SS          semistochastic: |H| > eps exact + K heat-bath draws p ~ |H|.
PFCV        (ablation) the R3.2 estimator on the unpenalised NNBFJ state.
All estimators are run per sample with the real per-term code (row
generation by Slater-Condon rules, incremental exact amplitudes for every
term actually evaluated, window-keyed backflow), on samples n drawn iid
from |psi|^2.  The iid draw uses the full state vector (an exact sampler
standing in for MCMC; it is not part of any estimator's cost).  Training
uses the exact energy on the full space (also not part of any estimator's
cost).

Metric
------
For each estimator X on state s: per-sample estimates Ehat_X(n) on B
samples; C_X = s^2_X * t_X, with s^2_X the empirical per-sample variance
and t_X the measured mean wall time per sample (row generation included,
single thread).  Tunable estimators (SCREEN eps; SS and PFCV eps, K) are
selected on the first half of the samples and evaluated on the second
half.  Control cost C_ctrl = min(C_EXACT, C_SS, C_SCREEN[admissible]) on
the NNBFJ state of the same seed.
  M_R3.1 = C_KEYX(WIBJ) / C_ctrl(NNBFJ)
  M_R3.2 = C_PFCV(SHRJ) / C_ctrl(NNBFJ)
plus dE_X = E_var - E_FCI for every arm.  Seeds: --seeds (default 3);
per-instance values are medians over seeds of the per-seed (paired) ratios.

Verdict (pre-committed; computed by --aggregate)
------------------------------------------------
Per route, per N in {6, 8}, with K_min, K_max the smallest/largest active
spaces run:
  dE_ok      median dE_route(K_max) <= max(1.6 mHa, 1.25 * median dE_NNBFJ(K_max))
  PASS       dE_ok and M(K_max) <= 0.5 and M(K_max) < M(K_min)
  FAIL       not dE_ok, or M(K_max) >= 1
  AMBIGUOUS  otherwise.  Pre-committed action: rerun N = 6 with
             --n-active 15; PASS iff M(15) <= min(M(14), 0.5), else KILL.
Global per route: PASS iff PASS at N = 6 and N = 8; FAIL if FAIL at either.
Hard Core: the instances carry non-stoquastic sign structure and strong
multireference character (reported: HF weight, frustrated fraction);
same-sector competitors are not examined: SIGN HALF ONLY.

Command-line options
--------------------
--n-atoms INT          H atoms in the ring = N electrons (even, >= 2). Default 6.
--n-active INT [...]   Active-space sizes K (lowest RHF MOs kept; K <= #AOs).
                       Default 8 10 12 14.
--bond-length FLOAT    H-H distance in bohr (> 0). Default 2.8.
--window INT           K_W, window size in spatial MOs (>= N/2). Default: N.
--arms LIST            Subset of NNBFJ WIBJ SHRJ. Default all three.
--seeds INT [...]      Network-initialisation seeds. Default 0 1 2.
--hidden INT           Backflow hidden width. Default 48.
--maxiter INT          L-BFGS iterations on the exact energy. Default 1500.
--mu FLOAT             SHRJ external-input shrinkage. Default 1e-2.
--n-samples INT        Samples B per state (even, >= 20). Default 2000.
--eps-screen FLOAT...  SCREEN thresholds. Default 1e-5 1e-4 3e-4 1e-3 3e-3 1e-2.
--eps-grid FLOAT...    SS / PFCV thresholds. Default 0 1e-3 1e-2 3e-2.
--k-grid INT...        SS / PFCV draws per sample. Default 1 4 16 64.
--bias-tol FLOAT       Admissible SCREEN bias in Ha. Default 1.6e-4.
--out PATH             Result JSON. Default results_<tag>.json here.
--smoke                Integral, Hamiltonian, ansatz and estimator checks plus
                       a tiny H4 run (< 30 s).
--aggregate DIR        Read every results_*.json in DIR and print the verdict.
"""
import argparse
import glob
import json
import os
import time
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import gammainc, gamma

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["NNBFJ", "WIBJ", "SHRJ"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P2.2 round-3 local-energy test")
    ap.add_argument("--n-atoms", type=int, default=6)
    ap.add_argument("--n-active", type=int, nargs="+", default=[8, 10, 12, 14])
    ap.add_argument("--bond-length", type=float, default=2.8)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--maxiter", type=int, default=1500)
    ap.add_argument("--mu", type=float, default=1e-2)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--eps-screen", type=float, nargs="+", default=[1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    ap.add_argument("--eps-grid", type=float, nargs="+", default=[0.0, 1e-3, 1e-2, 3e-2])
    ap.add_argument("--k-grid", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--bias-tol", type=float, default=1.6e-4)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", type=str, default=None)
    a = ap.parse_args(argv)
    if a.aggregate:
        return a
    if a.n_atoms < 2 or a.n_atoms % 2:
        ap.error("--n-atoms must be even and >= 2")
    if a.window is None:
        a.window = a.n_atoms
    if a.window < a.n_atoms // 2:
        ap.error("--window must be >= N/2")
    if a.bond_length <= 0 or a.hidden < 1 or a.maxiter < 1 or a.mu < 0:
        ap.error("--bond-length > 0, --hidden >= 1, --maxiter >= 1, --mu >= 0")
    if a.n_samples < 20 or a.n_samples % 2:
        ap.error("--n-samples must be even and >= 20")
    if min(a.k_grid) < 1 or min(a.eps_grid) < 0:
        ap.error("--k-grid >= 1, --eps-grid >= 0")
    if a.smoke:
        a.n_atoms, a.n_active, a.window, a.hidden, a.maxiter = 4, [6, 8], 4, 8, 200
        a.seeds, a.n_samples, a.arms = [0, 1, 2], 200, ARMS
        a.k_grid, a.eps_grid = [1, 4, 16], [0.0, 1e-2]
    if a.out is None:
        tag = "smoke" if a.smoke else f"N{a.n_atoms}_K{'-'.join(map(str, a.n_active))}_{'-'.join(a.arms)}"
        a.out = os.path.join(HERE, f"results_{tag}.json")
    return a


# ----------------------------------------------------------------------------
# Gaussian integrals: McMurchie-Davidson for Cartesian s and p functions
# ----------------------------------------------------------------------------

# 6-31G** hydrogen: contracted s (3 prim), uncontracted s, uncontracted p.
H631GSS = [(0, [18.7311370, 2.8253937, 0.6401217], [0.03349460, 0.23472695, 0.81375733]),
           (0, [0.1612778], [1.0]),
           (1, [1.1], [1.0])]
STO3G_H = [(0, [3.42525091, 0.62391373, 0.16885540], [0.15432897, 0.53532814, 0.44463454])]
CART = {0: [(0, 0, 0)], 1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)]}


def boys(n, T):
    T = np.asarray(T, float)
    out = np.empty_like(T)
    small = T < 1e-7
    out[small] = 1.0 / (2 * n + 1) - T[small] / (2 * n + 3)
    Ts = T[~small]
    out[~small] = 0.5 * Ts ** (-(n + 0.5)) * gamma(n + 0.5) * gammainc(n + 0.5, Ts)
    return out


def hermite_E(i, j, t, Qx, a, b):
    """E^{ij}_t (Helgaker), vectorised over arrays Qx = A_x - B_x, a, b."""
    p = a + b; q = a * b / p
    if t < 0 or t > i + j:
        return np.zeros_like(Qx)
    if i == j == t == 0:
        return np.exp(-q * Qx * Qx)
    if j == 0:
        return (1 / (2 * p)) * hermite_E(i - 1, j, t - 1, Qx, a, b) - (q * Qx / a) * hermite_E(i - 1, j, t, Qx, a, b) \
            + (t + 1) * hermite_E(i - 1, j, t + 1, Qx, a, b)
    return (1 / (2 * p)) * hermite_E(i, j - 1, t - 1, Qx, a, b) + (q * Qx / b) * hermite_E(i, j - 1, t, Qx, a, b) \
        + (t + 1) * hermite_E(i, j - 1, t + 1, Qx, a, b)


def hermite_R(L, p, PC):
    """R_{tuv}(p, PC) for t+u+v <= L; p: (n,), PC: (n,3). Returns dict."""
    RPC2 = (PC ** 2).sum(-1)
    T = p * RPC2
    Fn = [boys(n, T) for n in range(L + 1)]
    memo = {}

    def R(t, u, v, n):
        key = (t, u, v, n)
        if key in memo:
            return memo[key]
        if t < 0 or u < 0 or v < 0:
            val = 0.0
        elif t == u == v == 0:
            val = (-2 * p) ** n * Fn[n]
        elif t > 0:
            val = (t - 1) * R(t - 2, u, v, n + 1) + PC[:, 0] * R(t - 1, u, v, n + 1)
        elif u > 0:
            val = (u - 1) * R(t, u - 2, v, n + 1) + PC[:, 1] * R(t, u - 1, v, n + 1)
        else:
            val = (v - 1) * R(t, u, v - 2, n + 1) + PC[:, 2] * R(t, u, v - 1, n + 1)
        memo[key] = val
        return val
    out = {}
    for t in range(L + 1):
        for u in range(L + 1 - t):
            for v in range(L + 1 - t - u):
                out[(t, u, v)] = np.broadcast_to(R(t, u, v, 0), p.shape).astype(float)
    return out


def build_basis(X, basis):
    """Primitive list and contraction matrix T (prims x AOs)."""
    prims, cols, aolab = [], [], []
    k = 0
    for ia, A in enumerate(X):
        for l, exps, coefs in basis:
            for lmn in CART[l]:
                for e, c in zip(exps, coefs):
                    nrm = (2 * e / np.pi) ** 0.75 * (4 * e) ** (l / 2)
                    prims.append((e, A, lmn)); cols.append((k, c * nrm))
                k += 1
                aolab.append((ia, l))
    T = np.zeros((len(prims), k))
    for ip, (kk, c) in enumerate(cols):
        T[ip, kk] = c
    return prims, T, aolab


def _pairs(prims):
    n = len(prims)
    a = np.array([p[0] for p in prims]); A = np.array([p[1] for p in prims]); L = np.array([p[2] for p in prims])
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); I = I.ravel(); J = J.ravel()
    ai, aj = a[I], a[J]; p = ai + aj
    P = (ai[:, None] * A[I] + aj[:, None] * A[J]) / p[:, None]
    Q = A[I] - A[J]
    # Hermite expansion coefficients E_{tuv} for t+u+v <= 2 (l <= 1)
    htuv = [(t, u, v) for t in range(3) for u in range(3) for v in range(3) if t + u + v <= 2]
    Eh = np.zeros((len(I), len(htuv)))
    li, lj = L[I], L[J]
    for combo in set(map(tuple, np.concatenate([li, lj], 1))):
        m = np.all(np.concatenate([li, lj], 1) == combo, 1)
        idx = np.nonzero(m)[0]
        (ix, iy, iz, jx, jy, jz) = combo
        for h, (t, u, v) in enumerate(htuv):
            Eh[idx, h] = hermite_E(ix, jx, t, Q[idx, 0], ai[idx], aj[idx]) * \
                hermite_E(iy, jy, u, Q[idx, 1], ai[idx], aj[idx]) * hermite_E(iz, jz, v, Q[idx, 2], ai[idx], aj[idx])
    return dict(I=I, J=J, a=ai, b=aj, p=p, P=P, Q=Q, Eh=Eh, htuv=htuv, li=li, lj=lj, n=n)


def one_electron(prims, X, Z):
    pr = _pairs(prims)
    n = pr["n"]; p = pr["p"]; Q = pr["Q"]; ai, aj = pr["a"], pr["b"]
    li, lj = pr["li"], pr["lj"]

    Sd = np.zeros((3, len(p))); Td = np.zeros((3, len(p)))
    for d in range(3):
        for i in (0, 1):
            for j in (0, 1):
                m = (li[:, d] == i) & (lj[:, d] == j)
                if not m.any():
                    continue
                ii = np.nonzero(m)[0]
                pa, qa, A_, B_ = p[ii], Q[ii, d], ai[ii], aj[ii]
                def s(i_, j_):
                    if j_ < 0:
                        return 0.0
                    return hermite_E(i_, j_, 0, qa, A_, B_) * np.sqrt(np.pi / pa)
                Sd[d, ii] = s(i, j)
                Td[d, ii] = -0.5 * (j * (j - 1) * s(i, j - 2) - 2 * B_ * (2 * j + 1) * s(i, j) + 4 * B_ ** 2 * s(i, j + 2))
    S = Sd[0] * Sd[1] * Sd[2]
    Tk = Td[0] * Sd[1] * Sd[2] + Sd[0] * Td[1] * Sd[2] + Sd[0] * Sd[1] * Td[2]
    V = np.zeros(len(p))
    for C, z in zip(X, Z):
        R = hermite_R(2, p, pr["P"] - C)
        acc = np.zeros(len(p))
        for h, tuv in enumerate(pr["htuv"]):
            acc += pr["Eh"][:, h] * R[tuv]
        V += -z * 2 * np.pi / p * acc
    return S.reshape(n, n), Tk.reshape(n, n), V.reshape(n, n), pr


def eri_prims(pr, chunk=64):
    n = pr["n"]; npair = n * n
    p, P, Eh, htuv = pr["p"], pr["P"], pr["Eh"], pr["htuv"]
    sign = np.array([(-1) ** (t + u + v) for (t, u, v) in htuv])
    Ek = Eh * sign[None]
    nh = len(htuv)
    comb = {}
    G = np.zeros((npair, npair))
    for s0 in range(0, npair, chunk):
        s1 = min(npair, s0 + chunk)
        pb = p[s0:s1]
        alpha = pb[:, None] * p[None, :] / (pb[:, None] + p[None, :])
        PQ = P[s0:s1, None, :] - P[None, :, :]
        R = hermite_R(4, alpha.ravel(), PQ.reshape(-1, 3))
        acc = np.zeros((s1 - s0, npair))
        for h1, (t, u, v) in enumerate(htuv):
            inner = np.zeros((s1 - s0) * npair)
            for h2, (t2, u2, v2) in enumerate(htuv):
                inner += np.tile(Ek[:, h2], s1 - s0) * R[(t + t2, u + u2, v + v2)]
            acc += Eh[s0:s1, h1][:, None] * inner.reshape(s1 - s0, npair)
        pref = 2 * np.pi ** 2.5 / (pb[:, None] * p[None, :] * np.sqrt(pb[:, None] + p[None, :]))
        G[s0:s1] = pref * acc
    return G.reshape(n, n, n, n)


def integrals(X, basis, Z=None):
    if Z is None:
        Z = np.ones(len(X))
    prims, T, aolab = build_basis(X, basis)
    Sp, Tp, Vp, pr = one_electron(prims, X, Z)
    S = T.T @ Sp @ T
    nrm = 1 / np.sqrt(np.diag(S)); T = T * nrm[None]
    S = T.T @ Sp @ T
    h = T.T @ (Tp + Vp) @ T
    Gp = eri_prims(pr)
    G = np.einsum("ijkl,ia->ajkl", Gp, T, optimize=True)
    G = np.einsum("ajkl,jb->abkl", G, T, optimize=True)
    G = np.einsum("abkl,kc->abcl", G, T, optimize=True)
    G = np.einsum("abcl,ld->abcd", G, T, optimize=True)
    enuc = sum(Z[i] * Z[j] / np.linalg.norm(X[i] - X[j]) for i in range(len(X)) for j in range(i))
    return S, h, G, enuc, aolab


# ----------------------------------------------------------------------------
# Copied from tests/P2.2_stochastic_local_energy_r2/micro_test.py (geometry, RHF,
# Slater-Condon Hamiltonian, Hilbert space, ED); not imported across folders
# ----------------------------------------------------------------------------
def geometry(n_atoms, R, kind):
    if kind == "chain":
        return np.array([[i * R, 0.0, 0.0] for i in range(n_atoms)])
    rad = R / (2 * np.sin(np.pi / n_atoms))
    ang = 2 * np.pi * np.arange(n_atoms) / n_atoms
    return np.stack([rad * np.cos(ang), rad * np.sin(ang), 0 * ang], 1)

def rhf(S, h, G, nocc, maxit=1000, damp=0.5, lindep=1e-7):
    s, U = np.linalg.eigh(S)
    keep = s > lindep
    Xh = U[:, keep] / np.sqrt(s[keep])
    e, Cp = np.linalg.eigh(Xh.T @ h @ Xh)
    Cm = Xh @ Cp
    D = 2 * Cm[:, :nocc] @ Cm[:, :nocc].T
    Eold = 0.0
    for _ in range(maxit):
        F = h + np.einsum("pqrs,rs->pq", G, D) - 0.5 * np.einsum("prqs,rs->pq", G, D)
        e, Cp = np.linalg.eigh(Xh.T @ F @ Xh)
        Cm = Xh @ Cp
        Dn = 2 * Cm[:, :nocc] @ Cm[:, :nocc].T
        Eel = 0.5 * np.sum(D * (h + F))
        D = damp * D + (1 - damp) * Dn
        if abs(Eel - Eold) < 1e-11:
            break
        Eold = Eel
    return Cm, Eel, e


# ----------------------------------------------------------------------------
# Second-quantized Hamiltonian via Slater-Condon rows
# Spin-orbital order: alpha_0..alpha_{K-1}, beta_0..beta_{K-1}; ascending-order
# Jordan-Wigner phases.
# ----------------------------------------------------------------------------
class SQHam:
    def __init__(self, h, g, enuc, K, N):
        self.K, self.M, self.N, self.enuc = K, 2 * K, N, enuc
        M = self.M
        self.hso = np.zeros((M, M)); self.hso[:K, :K] = h; self.hso[K:, K:] = h
        gp = g.transpose(0, 2, 1, 3)  # <pq|rs> = (pr|qs)
        Gs = np.zeros((M, M, M, M))
        for s1 in (0, 1):
            for s2 in (0, 1):
                Gs[s1 * K:(s1 + 1) * K, s2 * K:(s2 + 1) * K, s1 * K:(s1 + 1) * K, s2 * K:(s2 + 1) * K] = gp
        self.gas = Gs - Gs.transpose(0, 1, 3, 2)
        del Gs
        self.Gd = np.einsum("pqpq->pq", self.gas).copy()  # <pq||pq>
        self.spin = (np.arange(M) >= K).astype(np.int64)
        self.iu_occ = np.triu_indices(N, 1)

    def diag(self, occ):
        return float(np.trace(self.hso[np.ix_(occ, occ)]) + 0.5 * self.Gd[np.ix_(occ, occ)].sum() + self.enuc)

    def row(self, code, occ):
        """All connected n' != n: dict with new codes, H_{n'n}, H_{n'n'}, excitation indices."""
        M, N = self.M, self.N
        o = np.zeros(M, np.int64); o[occ] = 1
        c = np.cumsum(o) - o
        virt = np.nonzero(o == 0)[0]
        u = self.Gd[:, occ].sum(1)
        f = np.diag(self.hso) + u
        Hnn = self.diag(occ)
        Fn = self.hso + self.gas[:, occ, :, occ].sum(0)
        # singles
        I, A = np.meshgrid(occ, virt, indexing="ij"); I = I.ravel(); A = A.ravel()
        m = self.spin[I] == self.spin[A]; I, A = I[m], A[m]
        hs = Fn[A, I] * (1 - 2 * ((c[I] + c[A] - (I < A)) & 1))
        dEs = f[A] - f[I] - self.Gd[A, I]
        # doubles
        oi, oj = occ[self.iu_occ[0]], occ[self.iu_occ[1]]
        vu = np.triu_indices(len(virt), 1)
        va, vb = virt[vu[0]], virt[vu[1]]
        nI = len(oi); nA = len(va)
        Ii = np.repeat(oi, nA); Jj = np.repeat(oj, nA)
        Aa = np.tile(va, nI); Bb = np.tile(vb, nI)
        m = (self.spin[Ii] + self.spin[Jj]) == (self.spin[Aa] + self.spin[Bb])
        Ii, Jj, Aa, Bb = Ii[m], Jj[m], Aa[m], Bb[m]
        val = self.gas[Aa, Bb, Ii, Jj]
        ph = c[Ii] + c[Jj] - 1 + c[Bb] - (Ii < Bb) - (Jj < Bb) + c[Aa] - (Ii < Aa) - (Jj < Aa)
        hd = val * (1 - 2 * (ph & 1))
        Gd = self.Gd
        dEd = (f[Aa] + f[Bb] - f[Ii] - f[Jj] + Gd[Ii, Jj] + Gd[Aa, Bb]
               - Gd[Aa, Ii] - Gd[Aa, Jj] - Gd[Bb, Ii] - Gd[Bb, Jj])
        one = np.int64(1)
        cs = code - (one << I) + (one << A)
        cd = code - (one << Ii) - (one << Jj) + (one << Aa) + (one << Bb)
        out = {
            "code": np.concatenate([cs, cd]),
            "h": np.concatenate([hs, hd]),
            "Hjj": Hnn + np.concatenate([dEs, dEd]),
            "rem": np.concatenate([np.stack([I, -np.ones_like(I)], 1), np.stack([Ii, Jj], 1)]),
            "add": np.concatenate([np.stack([A, -np.ones_like(A)], 1), np.stack([Aa, Bb], 1)]),
            "Hnn": Hnn,
        }
        keep = np.abs(out["h"]) > 1e-13
        for k in ("code", "h", "Hjj", "rem", "add"):
            out[k] = out[k][keep]
        return out


def build_space(K, na, nb):
    def strings(n):
        return [sum(1 << i for i in c) for c in combinations(range(K), n)]
    codes = np.array(sorted(a | (b << K) for a in strings(na) for b in strings(nb)), dtype=np.int64)
    M = 2 * K
    occ = ((codes[:, None] >> np.arange(M)) & 1).astype(np.int8)
    occ_idx = np.array([np.nonzero(o)[0] for o in occ])
    return codes, occ, occ_idx


def build_hamiltonian(ham, codes, occ_idx):
    D = len(codes)
    indptr = [0]; idx = []; dat = []
    for n in range(D):
        r = ham.row(codes[n], occ_idx[n])
        j = np.searchsorted(codes, r["code"])
        assert np.all(codes[j] == r["code"]), "connected config outside sector"
        idx.append(np.concatenate([[n], j]).astype(np.int32))
        dat.append(np.concatenate([[r["Hnn"]], r["h"]]))
        indptr.append(indptr[-1] + len(j) + 1)
    H = sp.csr_matrix((np.concatenate(dat), np.concatenate(idx), np.array(indptr, dtype=np.int64)), shape=(D, D))
    H.sort_indices()
    return H


def ground_states(H, k=2):
    D = H.shape[0]
    if D <= 2500:
        w, v = np.linalg.eigh(H.toarray())
        return w[:k], v[:, :k]
    from scipy.sparse.linalg import eigsh
    w, v = eigsh(H, k=k, which="SA", tol=1e-10)
    o = np.argsort(w)
    return w[o], v[:, o]




# ----------------------------------------------------------------------------
# NNBF-Jastrow ansatz with an input mask (window) and per-sample kernels
# ----------------------------------------------------------------------------
class Ansatz:
    """psi(n) = exp(u.n + sum_{p<q} v_pq n_p n_q) * det Phi(x)[occ n],
    Phi(x) = Phi0 + reshape(W2 tanh(W1 (mask * x) + b1)), x = 2n - 1.
    mask = window (WIBJ) or all ones (NNBFJ, SHRJ).  pen_mask/mu: external-
    input shrinkage mu * ||W1[:, pen_mask]||^2 added to the objective."""

    def __init__(self, M, N, hidden, Phi0, scale, rng, in_mask, pen_mask=None, mu=0.0):
        self.M, self.N, self.Hh = M, N, hidden
        self.mask = in_mask.astype(float)
        self.pen = np.zeros(M, bool) if pen_mask is None else pen_mask
        self.mu = mu
        self.iu = np.triu_indices(M, 1)
        self.shapes = [("Phi0", (M, N)), ("W2", (M * N, hidden)), ("W1", (hidden, M)), ("b1", (hidden,)),
                       ("u", (M,)), ("v", (len(self.iu[0]),))]
        init = {"Phi0": Phi0, "W2": scale * rng.standard_normal((M * N, hidden)),
                "W1": scale * rng.standard_normal((hidden, M)) * self.mask[None],
                "b1": np.zeros(hidden), "u": np.zeros(M), "v": np.zeros(len(self.iu[0]))}
        self.theta0 = np.concatenate([init[k].ravel() for k, _ in self.shapes])

    def unpack(self, th):
        out, i = {}, 0
        for k, s in self.shapes:
            n = int(np.prod(s)); out[k] = th[i:i + n].reshape(s); i += n
        V = np.zeros((self.M, self.M)); V[self.iu] = out["v"]; out["V"] = V + V.T
        out["W1m"] = out["W1"] * self.mask[None]
        out["W2r"] = out["W2"].reshape(self.M, self.N, self.Hh)
        return out

    def amplitudes(self, th, occ, occ_idx):
        P = self.unpack(th)
        x = (2.0 * occ - 1.0)
        hid = np.tanh(x @ P["W1m"].T + P["b1"])
        Phi = P["Phi0"][None] + (hid @ P["W2"].T).reshape(-1, self.M, self.N)
        A = np.take_along_axis(Phi, occ_idx[:, :, None], axis=1)
        o = occ.astype(float)
        logJ = o @ P["u"] + 0.5 * np.einsum("bp,bp->b", o @ P["V"], o)
        return np.linalg.det(A) * np.exp(logJ), (Phi, A, hid, x, P, o)

    def energy_grad(self, th, H, occ, occ_idx):
        psi, (Phi, A, hid, x, P, o) = self.amplitudes(th, occ, occ_idx)
        nrm = psi @ psi
        Hpsi = H @ psi
        E = psi @ Hpsi / nrm
        c = 2 * (Hpsi - E * psi) / nrm
        cp = c * psi
        Ainv = np.linalg.inv(A)
        gA = cp[:, None, None] * np.transpose(Ainv, (0, 2, 1))
        gPhi = np.zeros_like(Phi)
        np.put_along_axis(gPhi, occ_idx[:, :, None], gA, axis=1)
        gflat = gPhi.reshape(len(psi), -1)
        g = {"Phi0": gPhi.sum(0), "W2": gflat.T @ hid}
        dz = (gflat @ P["W2"]) * (1 - hid ** 2)
        g["W1"] = (dz.T @ x) * self.mask[None]
        g["b1"] = dz.sum(0)
        g["u"] = o.T @ cp
        g["v"] = (o.T @ (o * cp[:, None]))[self.iu]
        pen = 0.0
        if self.mu > 0 and self.pen.any():
            W1p = P["W1"][:, self.pen]
            pen = self.mu * np.sum(W1p ** 2)
            g["W1"][:, self.pen] += 2 * self.mu * W1p
        return E + pen, np.concatenate([g[k].ravel() for k, _ in self.shapes])

    # ---- per-sample kernels -------------------------------------------------
    def setup(self, P, occ_n):
        x = 2.0 * np.isin(np.arange(self.M), occ_n) - 1.0
        z = P["W1m"] @ x + P["b1"]
        Phi = P["Phi0"] + (P["W2"] @ np.tanh(z)).reshape(self.M, self.N)
        A = Phi[occ_n]
        o = (x + 1) / 2
        return {"x": x, "z": z, "detA": np.linalg.det(A), "f": P["V"] @ o, "occ": occ_n}

    def jastrow_delta(self, P, st, rem, add):
        u, V, f = P["u"], P["V"], st["f"]
        i, a = rem[:, 0], add[:, 0]
        d = u[a] - u[i] + f[a] - f[i] - V[a, i]
        d2 = rem[:, 1] >= 0
        j, b = rem[d2, 1], add[d2, 1]
        d[d2] += u[b] - u[j] + f[b] - f[j] + V[a[d2], b] + V[i[d2], j] - V[b, i[d2]] - V[a[d2], j] - V[b, j]
        return d

    def ratio_exact(self, P, st, rem, add, occ_j):
        """Exact psi(n')/psi(n) for a batch of connected n' (incremental hidden
        update; only the N occupied rows of Phi(n') are formed: O(N^2 h + N^3))."""
        if len(rem) == 0:
            return np.zeros(0)
        W1T = P["W1m"].T
        dz = 2 * (W1T[add[:, 0]] - W1T[rem[:, 0]])
        d2 = add[:, 1] >= 0
        dz[d2] += 2 * (W1T[add[d2, 1]] - W1T[rem[d2, 1]])
        hid = np.tanh(st["z"][None] + dz)
        A = P["Phi0"][occ_j] + np.einsum("bknh,bh->bkn", P["W2r"][occ_j], hid)
        return np.linalg.det(A) / st["detA"] * np.exp(self.jastrow_delta(P, st, rem, add))

    def ratio_key(self, P, st, codes_j, rem, add, wcode, win_idx):
        """Window-keyed ratio: the backflow is evaluated once per distinct window
        occupation n'_W with the external input frozen at n; every term then
        costs a 2x2 Green's-function minor.  Exact when mask = window (WIBJ);
        the frozen-external-input surrogate otherwise (PFCV)."""
        keys, inv = np.unique(codes_j & wcode, return_inverse=True)
        bits = ((keys[:, None] >> win_idx[None]) & 1).astype(float)
        X = np.repeat(st["x"][None], len(keys), 0)
        X[:, win_idx] = 2 * bits - 1
        Z = X @ P["W1m"].T + P["b1"]
        Phi = P["Phi0"][None] + (np.tanh(Z) @ P["W2"].T).reshape(-1, self.M, self.N)
        occ_n = st["occ"]
        A = Phi[:, occ_n, :]
        Ainv = np.linalg.inv(A)
        dk = np.linalg.det(A)
        N = self.N
        pos = np.full(self.M, -1); pos[occ_n] = np.arange(N)
        ki = pos[rem[:, 0]]; a = add[:, 0]
        d2 = rem[:, 1] >= 0
        # rows a (and b) of G_k = Phi_k A_k^{-1}
        ga = np.einsum("tn,tnc->tc", Phi[inv, a, :], Ainv[inv])
        r = ga[np.arange(len(a)), ki].copy()
        if d2.any():
            t2 = np.nonzero(d2)[0]
            kj = pos[rem[t2, 1]]; b = add[t2, 1]
            gb = np.einsum("tn,tnc->tc", Phi[inv[t2], b, :], Ainv[inv[t2]])
            r[t2] = ga[t2, ki[t2]] * gb[np.arange(len(t2)), kj] - ga[t2, kj] * gb[np.arange(len(t2)), ki[t2]]
        L = np.repeat(occ_n[None], len(rem), 0)
        L[np.arange(len(rem)), ki] = a
        if d2.any():
            L[t2, kj] = b
        par = np.zeros(len(rem), np.int64)
        for p_ in range(N):
            for q_ in range(p_ + 1, N):
                par += L[:, p_] > L[:, q_]
        return r * (1 - 2 * (par & 1)) * dk[inv] / st["detA"] * np.exp(self.jastrow_delta(P, st, rem, add)), len(keys)


def decode(codes, M, N):
    bits = (codes[:, None] >> np.arange(M)[None]) & 1
    return np.nonzero(bits)[1].reshape(-1, N)


# ----------------------------------------------------------------------------
# Estimators (each runs the real per-term code it is charged for)
# ----------------------------------------------------------------------------
def est_exact(net, P, st, row, occ_j):
    return row["Hnn"] + row["h"] @ net.ratio_exact(P, st, row["rem"], row["add"], occ_j)


def est_screen(net, P, st, row, occ_j, eps):
    m = np.abs(row["h"]) > eps
    return row["Hnn"] + row["h"][m] @ net.ratio_exact(P, st, row["rem"][m], row["add"][m], occ_j[m])


def _draw(rng, w, K):
    p = w / w.sum()
    return rng.choice(len(p), size=K, p=p), p


def est_ss(net, P, st, row, occ_j, eps, K, rng):
    h = row["h"]; m = np.abs(h) > eps
    val = row["Hnn"] + h[m] @ net.ratio_exact(P, st, row["rem"][m], row["add"][m], occ_j[m])
    T = np.nonzero(~m)[0]
    if len(T):
        k, p = _draw(rng, np.abs(h[T]), K)
        t = T[k]
        val += np.mean(h[t] * net.ratio_exact(P, st, row["rem"][t], row["add"][t], occ_j[t]) / p[k])
    return val


def est_pfcv(net, P, st, row, occ_j, y, eps, K, rng, alpha=0.9):
    """y = h * window-keyed surrogate ratio (computed once per sample, charged
    separately as t_key)."""
    h = row["h"]; m = np.abs(y) > eps
    val = row["Hnn"] + h[m] @ net.ratio_exact(P, st, row["rem"][m], row["add"][m], occ_j[m])
    T = np.nonzero(~m)[0]
    if len(T):
        yT = y[T]; val += yT.sum()
        w = alpha * np.abs(yT) / max(np.abs(yT).sum(), 1e-300) + (1 - alpha) * np.abs(h[T]) / np.abs(h[T]).sum()
        k, p = _draw(rng, w, K)
        t = T[k]
        x = h[t] * net.ratio_exact(P, st, row["rem"][t], row["add"][t], occ_j[t])
        val += np.mean((x - yT[k]) / p[k])
    return val


def analyse_state(net, th, ham, samples_codes, args, rng, wcode, win_idx, arm):
    """Run every estimator on every sample; return per-sample values and times."""
    P = net.unpack(th)
    M, N = net.M, net.N
    B = len(samples_codes)
    res = {"EXACT": [], "KEYX": []}
    tim = {"row": [], "EXACT": [], "KEY": []}
    cfg = []
    if arm == "NNBFJ":
        cfg += [("SCREEN", e, 0) for e in args.eps_screen]
        cfg += [("SS", e, K) for e in args.eps_grid for K in args.k_grid]
    if arm in ("NNBFJ", "SHRJ"):
        cfg += [("PFCV", e, K) for e in args.eps_grid for K in args.k_grid]
    for c in cfg:
        res[c] = []; tim[c] = []
    nkeys, ncon = [], []
    for code in samples_codes:
        t0 = time.perf_counter()
        occ_n = decode(np.array([code]), M, N)[0]
        row = ham.row(code, occ_n)
        occ_j = decode(row["code"], M, N)
        t1 = time.perf_counter()
        st = net.setup(P, occ_n)
        t1b = time.perf_counter()
        ex = est_exact(net, P, st, row, occ_j)
        t2 = time.perf_counter()
        rk, nk = net.ratio_key(P, st, row["code"], row["rem"], row["add"], wcode, win_idx)
        y = row["h"] * rk
        kx = row["Hnn"] + y.sum()
        t3 = time.perf_counter()
        tset = t1b - t1
        tim["row"].append(t1 - t0 + tset); tim["EXACT"].append(t2 - t1b); tim["KEY"].append(t3 - t2)
        res["EXACT"].append(ex); res["KEYX"].append(kx)
        nkeys.append(nk); ncon.append(len(row["h"]))
        for c in cfg:
            s0 = time.perf_counter()
            if c[0] == "SCREEN":
                v = est_screen(net, P, st, row, occ_j, c[1])
            elif c[0] == "SS":
                v = est_ss(net, P, st, row, occ_j, c[1], c[2], rng)
            else:
                v = est_pfcv(net, P, st, row, occ_j, y, c[1], c[2], rng)
            tim[c].append(time.perf_counter() - s0)
            res[c].append(v)
    res = {k: np.array(v) for k, v in res.items()}
    tim = {k: np.array(v) for k, v in tim.items()}
    return res, tim, np.array(nkeys), np.array(ncon), cfg


def costs(res, tim, cfg, args):
    """C = s^2 * t on the second half; tunables selected on the first half."""
    B = len(res["EXACT"]); h1, h2 = slice(0, B // 2), slice(B // 2, B)
    trow = tim["row"]
    out = {}

    def C(vals, t, sl):
        return float(np.var(vals[sl], ddof=1) * np.mean(t[sl]))
    out["EXACT"] = {"C": C(res["EXACT"], trow + tim["EXACT"], h2), "var": float(np.var(res["EXACT"][h2], ddof=1)),
                    "t": float(np.mean((trow + tim["EXACT"])[h2]))}
    out["EXACT_norow"] = {"C": C(res["EXACT"], tim["EXACT"], h2)}
    out["KEYX"] = {"C": C(res["KEYX"], trow + tim["KEY"], h2), "var": float(np.var(res["KEYX"][h2], ddof=1)),
                   "t": float(np.mean((trow + tim["KEY"])[h2])),
                   "max_dev_from_exact": float(np.max(np.abs(res["KEYX"] - res["EXACT"])))}
    out["KEYX_norow"] = {"C": C(res["KEYX"], tim["KEY"], h2)}
    for fam in ("SCREEN", "SS", "PFCV"):
        best = None
        for c in [c for c in cfg if c[0] == fam]:
            extra = tim["KEY"] if fam == "PFCV" else 0.0
            vals = res[c]
            if fam == "SCREEN":
                d = vals - res["EXACT"]
                bias = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(B))
                if abs(bias) + 2 * se > args.bias_tol:
                    continue
            c1 = C(vals, trow + tim[c] + extra, h1)
            if best is None or c1 < best[0]:
                best = (c1, c)
        if best is None:
            continue
        c = best[1]; extra = tim["KEY"] if fam == "PFCV" else 0.0
        vals = res[c]
        d = vals - res["EXACT"]
        dev = vals[h2] - vals[h2].mean()
        top = np.sort(dev ** 2)[::-1]
        out[fam] = {"C": C(vals, trow + tim[c] + extra, h2), "C_norow": C(vals, tim[c] + extra, h2),
                    "eps": c[1], "K": c[2], "var": float(np.var(vals[h2], ddof=1)),
                    "t": float(np.mean((trow + tim[c] + extra)[h2])),
                    "bias": float(d.mean()), "bias_se": float(d.std(ddof=1) / np.sqrt(B)),
                    "top1pct_var_share": float(top[:max(1, len(top) // 100)].sum() / max(top.sum(), 1e-300))}
    return out


# ----------------------------------------------------------------------------
# Instance and driver
# ----------------------------------------------------------------------------
def build_instance(n_atoms, R, K_act):
    X = geometry(n_atoms, R, "ring")
    S, hao, Gao, enuc, lab = integrals(X, H631GSS)
    nocc = n_atoms // 2
    Cm, Eel, eps = rhf(S, hao, Gao, nocc)
    if K_act > Cm.shape[1]:
        raise ValueError(f"K_act={K_act} > number of MOs {Cm.shape[1]}")
    C = Cm[:, :K_act]
    h = C.T @ hao @ C
    g = np.einsum("pqrs,pa,qb,rc,sd->abcd", Gao, C, C, C, C, optimize=True)
    isp = np.array([l == 1 for (_, l) in lab])
    mull = (C * (S @ C))  # Mulliken populations per AO and MO
    p_weight = float(mull[isp, nocc:].sum(0).mean()) if K_act > nocc else 0.0
    return h, g, enuc, Eel + enuc, p_weight, float(eps[K_act - 1] if K_act <= len(eps) else np.nan)


def train(net, H, occ, oi, maxiter):
    r = minimize(lambda t: net.energy_grad(t, H, occ, oi), net.theta0, jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "maxcor": 30, "gtol": 1e-12, "ftol": 1e-16})
    return r.x


def run_instance(K, args):
    t0 = time.time()
    N = args.n_atoms; nocc = N // 2
    h, g, enuc, Erhf, pw, _ = build_instance(N, args.bond_length, K)
    ham = SQHam(h, g, enuc, K, N)
    codes, occ, oi = build_space(K, nocc, nocc)
    H = build_hamiltonian(ham, codes, oi)
    w2, v2 = ground_states(H, 2)
    E0, psi0 = w2[0], v2[:, 0]
    hf = sum(1 << i for i in range(nocc)); hf |= hf << K
    hfw = float(psi0[np.searchsorted(codes, hf)] ** 2)
    Hc = H.tocoo(); m = Hc.row != Hc.col
    prod = Hc.data[m] * psi0[Hc.row[m]] * psi0[Hc.col[m]]
    frustr = float(np.abs(prod[prod > 0]).sum() / np.abs(prod).sum())
    M = 2 * K
    KW = min(args.window, K)
    win = np.zeros(M, bool); win[:KW] = True; win[K:K + KW] = True
    win_idx = np.nonzero(win)[0]; wcode = int(sum(1 << int(p) for p in win_idx))
    print(f"\n=== H{N} ring R={args.bond_length} 6-31G** active K={K} M={M} dim={len(codes)} "
          f"E_RHF={Erhf:.6f} E_FCI={E0:.6f} Ecorr={1e3*(E0-Erhf):.1f} mHa gap={w2[1]-E0:.4f} HF wt={hfw:.3f} "
          f"frustr={frustr:.3f} p-weight(virt)={pw:.3f} window K_W={KW} nnz/row={H.nnz/len(codes):.0f} "
          f"(build {time.time()-t0:.1f}s)", flush=True)
    out = {"K": K, "M": M, "N": N, "dim": len(codes), "E_FCI": float(E0), "E_RHF": float(Erhf),
           "HF_weight": hfw, "frustrated_fraction": frustr, "p_weight_virtual": pw, "K_W": KW,
           "gap": float(w2[1] - E0), "runs": []}
    Phi0 = np.zeros((M, N)); Phi0[:K, :nocc] = np.eye(K)[:, :nocc]; Phi0[K:, nocc:] = Phi0[:K, :nocc]
    for seed in args.seeds:
        for arm in args.arms:
            rng = np.random.default_rng(1000 * seed + 7)
            mask = win if arm == "WIBJ" else np.ones(M, bool)
            net = Ansatz(M, N, args.hidden, Phi0, 0.05, rng, mask, pen_mask=~win if arm == "SHRJ" else None,
                         mu=args.mu if arm == "SHRJ" else 0.0)
            t1 = time.time()
            th = train(net, H, occ, oi, args.maxiter)
            psi, _ = net.amplitudes(th, occ, oi)
            wv = psi ** 2 / (psi @ psi)
            Ev = float(psi @ (H @ psi) / (psi @ psi))
            EL = (H @ psi) / psi
            varx = float(wv @ (EL - Ev) ** 2)
            srng = np.random.default_rng(10 ** 6 + 1000 * seed + 13)
            samp = codes[srng.choice(len(codes), size=args.n_samples, p=wv)]
            topt = time.time() - t1
            res, tim, nk, nc, cfg = analyse_state(net, th, ham, samp, args, srng, wcode, win_idx, arm)
            elchk = float(np.max(np.abs(res["EXACT"] - EL[np.searchsorted(codes, samp)])))
            cst = costs(res, tim, cfg, args)
            run = {"seed": seed, "arm": arm, "dE_mHa": 1e3 * (Ev - E0), "var_exact": varx, "EL_check": elchk,
                   "mean_keys": float(nk.mean()), "mean_conn": float(nc.mean()), "costs": cst,
                   "W1_ext_norm": float(np.linalg.norm(net.unpack(th)["W1"][:, ~win])),
                   "t_train_s": topt, "t_analyse_s": time.time() - t1 - topt}
            out["runs"].append(run)
            ce = cst["EXACT"]["C"]
            line = " ".join(f"{k}={v['C']/ce:.3g}" for k, v in cst.items() if "C" in v and not k.endswith("norow"))
            print(f"  seed={seed} {arm:5s} dE={run['dE_mHa']:8.3f} mHa Var(EL)={varx:.2e} <|C|>={nc.mean():.0f} "
                  f"<keys>={nk.mean():.1f} ELchk={elchk:.1e} KEYX-EXACT max={cst['KEYX']['max_dev_from_exact']:.1e} "
                  f"|W1_ext|={run['W1_ext_norm']:.3g} (train {topt:.0f}s)\n      C/C_EXACT(own state): {line}",
                  flush=True)
            for fam in ("SCREEN", "SS", "PFCV"):
                if fam in cst:
                    d = cst[fam]
                    print(f"      {fam:6s} eps={d['eps']:g} K={d['K']} bias={1e3*d['bias']:.4f}+-{1e3*d['bias_se']:.4f} mHa "
                          f"top1%share={d['top1pct_var_share']:.2f}")
    return out


def ratios_for_instance(inst):
    """Per seed (paired with the NNBFJ control of the same seed): M_R3.1, M_R3.2,
    the PFCV ablation, and the no-row-generation sensitivity."""
    by = {}
    for r in inst["runs"]:
        by.setdefault(r["seed"], {})[r["arm"]] = r
    rows = []
    for seed, d in sorted(by.items()):
        if "NNBFJ" not in d:
            continue
        c = d["NNBFJ"]["costs"]
        ctrl = min(c[k]["C"] for k in ("EXACT", "SS", "SCREEN") if k in c)
        ctrl_nr = min([c["EXACT_norow"]["C"]] + [c[k]["C_norow"] for k in ("SS", "SCREEN") if k in c])
        rr = {"seed": seed, "dE_NNBFJ": d["NNBFJ"]["dE_mHa"],
              "ctrl": min((k for k in ("EXACT", "SS", "SCREEN") if k in c), key=lambda k: c[k]["C"]),
              "M_PFCV_ablation": c["PFCV"]["C"] / ctrl if "PFCV" in c else np.nan}
        if "WIBJ" in d:
            w = d["WIBJ"]["costs"]
            rr.update({"dE_WIBJ": d["WIBJ"]["dE_mHa"], "M_R31": w["KEYX"]["C"] / ctrl,
                       "M_R31_norow": w["KEYX_norow"]["C"] / ctrl_nr})
        if "SHRJ" in d and "PFCV" in d["SHRJ"]["costs"]:
            s = d["SHRJ"]["costs"]
            rr.update({"dE_SHRJ": d["SHRJ"]["dE_mHa"], "M_R32": s["PFCV"]["C"] / ctrl,
                       "M_R32_norow": s["PFCV"]["C_norow"] / ctrl_nr})
        rows.append(rr)
    return rows


def verdict_for(insts_by_K, route):
    key, dkey = {"R3.1": ("M_R31", "dE_WIBJ"), "R3.2": ("M_R32", "dE_SHRJ")}[route]
    Ks = sorted(insts_by_K)
    med = {}
    for K in Ks:
        rows = [r for r in ratios_for_instance(insts_by_K[K]) if key in r]
        if not rows:
            return {"verdict": "INCOMPLETE"}
        med[K] = {"M": float(np.median([r[key] for r in rows])),
                  "M_norow": float(np.median([r[key + "_norow"] for r in rows])),
                  "M_spread": [float(min(r[key] for r in rows)), float(max(r[key] for r in rows))],
                  "dE": float(np.median([r[dkey] for r in rows])),
                  "dE_ctrl": float(np.median([r["dE_NNBFJ"] for r in rows])), "n_seeds": len(rows)}
    kmin, kmax = Ks[0], Ks[-1]
    m = med[kmax]
    dE_ok = m["dE"] <= max(1.6, 1.25 * m["dE_ctrl"])
    if not dE_ok or m["M"] >= 1:
        v = "FAIL"
    elif m["M"] <= 0.5 and m["M"] < med[kmin]["M"] and len(Ks) > 1:
        v = "PASS"
    else:
        v = "AMBIGUOUS"
    return {"verdict": v, "dE_ok": bool(dE_ok), "by_K": med}


def aggregate(path):
    files = sorted(glob.glob(os.path.join(path, "results_*.json")))
    files = [f for f in files if "smoke" not in os.path.basename(f)]
    data = {}
    for f in files:
        with open(f) as fh:
            for inst in json.load(fh)["instances"]:
                d = data.setdefault(inst["N"], {}).setdefault(inst["K"], {**{k: v for k, v in inst.items() if k != "runs"},
                                                                          "runs": []})
                d["runs"].extend(inst["runs"])
    out = {}
    for route in ("R3.1", "R3.2"):
        out[route] = {}
        for N in sorted(data):
            out[route][N] = verdict_for(data[N], route)
        vs = [out[route][N]["verdict"] for N in out[route]]
        out[route]["global"] = ("FAIL" if "FAIL" in vs else
                                "PASS" if vs and all(v == "PASS" for v in vs) and set(data) >= {6, 8} else
                                "AMBIGUOUS" if vs else "INCOMPLETE")
    print(json.dumps(out, indent=1))
    return out


def smoke_checks(args):
    t0 = time.time()
    rng = np.random.default_rng(3)
    # (1) integrals: s-part, symmetry, PSD, rotation invariance, H2 reference
    X = np.array([[0, 0, 0], [1.4, 0, 0.]])
    S, h, G, e, _ = integrals(X, H631GSS)
    C, E, _ = rhf(S, h, G, 1)
    X4 = geometry(4, 2.8, "ring")
    S, h, G, e4, _ = integrals(X4, H631GSS)
    K = S.shape[0]
    sym = max(abs(G - G.transpose(1, 0, 2, 3)).max(), abs(G - G.transpose(2, 3, 0, 1)).max())
    psd = np.linalg.eigvalsh(G.reshape(K * K, K * K)).min()
    C4, E4, _ = rhf(S, h, G, 2)
    Qm, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    S2, h2, G2, e42, _ = integrals(X4 @ Qm.T + 0.3, H631GSS)
    C42, E42, _ = rhf(S2, h2, G2, 2)
    print(f"H2 6-31G** R=1.4: E_RHF={E+e:.6f} (lit. -1.13128); H4 ring: ERI symmetry err={sym:.1e}, "
          f"min eig(ERI)={psd:.1e}, rotation invariance |dE|={abs(E42+e42-E4-e4):.1e}")
    ok = abs(E + e + 1.13128) < 2e-5 and sym < 1e-12 and psd > -1e-10 and abs(E42 + e42 - E4 - e4) < 1e-9
    # (2) ansatz kernels: exact incremental ratio, window-keyed ratio (exact for WIB), Jastrow delta, gradient
    hh, gg, en, Er, pw, _ = build_instance(4, 2.8, 6)
    ham = SQHam(hh, gg, en, 6, 4)
    codes, occ, oi = build_space(6, 2, 2)
    Hs = build_hamiltonian(ham, codes, oi)
    M = 12; win = np.zeros(M, bool); win[:4] = True; win[6:10] = True
    win_idx = np.nonzero(win)[0]; wcode = int(sum(1 << int(p) for p in win_idx))
    Phi0 = np.zeros((M, 4)); Phi0[:6, :2] = np.eye(6)[:, :2]; Phi0[6:, 2:] = Phi0[:6, :2]
    errs = {}
    for name, mask in (("full", np.ones(M, bool)), ("window", win)):
        net = Ansatz(M, 4, 5, Phi0, 0.3, rng, mask, pen_mask=~win, mu=0.1)
        th = net.theta0 + 0.2 * rng.standard_normal(net.theta0.size)
        P = net.unpack(th)
        psi, _ = net.amplitudes(th, occ, oi)
        n = 17
        row = ham.row(codes[n], oi[n]); j = np.searchsorted(codes, row["code"])
        st = net.setup(P, oi[n])
        rx = net.ratio_exact(P, st, row["rem"], row["add"], decode(row["code"], M, 4))
        rk, _ = net.ratio_key(P, st, row["code"], row["rem"], row["add"], wcode, win_idx)
        errs[name] = (float(np.max(np.abs(rx - psi[j] / psi[n]))), float(np.max(np.abs(rk - psi[j] / psi[n]))))
        idx = rng.choice(th.size, 12, replace=False); gn = []
        _, ga = net.energy_grad(th, Hs, occ, oi)
        for i in idx:
            dd = np.zeros_like(th); dd[i] = 1e-6
            gn.append((net.energy_grad(th + dd, Hs, occ, oi)[0] - net.energy_grad(th - dd, Hs, occ, oi)[0]) / 2e-6)
        errs[name + "_grad"] = float(np.max(np.abs(ga[idx] - gn)) / max(np.max(np.abs(gn)), 1e-12))
    print(f"  exact incremental ratio err: full={errs['full'][0]:.1e} window={errs['window'][0]:.1e}; "
          f"window-keyed ratio err: WIB (must be ~0)={errs['window'][1]:.1e}, full-input surrogate (nonzero)="
          f"{errs['full'][1]:.1e}; gradient rel err: full={errs['full_grad']:.1e} window={errs['window_grad']:.1e}")
    ok = ok and errs["full"][0] < 1e-10 and errs["window"][0] < 1e-10 and errs["window"][1] < 1e-10 \
        and errs["full"][1] > 1e-6 and errs["full_grad"] < 1e-5 and errs["window_grad"] < 1e-5
    # (3) unbiasedness of the stochastic estimators on one row (average of 20000 draws)
    net = Ansatz(M, 4, 5, Phi0, 0.3, rng, np.ones(M, bool))
    th = net.theta0 + 0.2 * rng.standard_normal(net.theta0.size); P = net.unpack(th)
    n = 5; row = ham.row(codes[n], oi[n]); occ_j = decode(row["code"], M, 4); st = net.setup(P, oi[n])
    ex = est_exact(net, P, st, row, occ_j)
    rk, _ = net.ratio_key(P, st, row["code"], row["rem"], row["add"], wcode, win_idx); y = row["h"] * rk
    vs = np.array([est_ss(net, P, st, row, occ_j, 1e-2, 4, rng) for _ in range(20000)])
    vp = np.array([est_pfcv(net, P, st, row, occ_j, y, 1e-2, 4, rng) for _ in range(20000)])
    zs = abs(vs.mean() - ex) / (vs.std() / np.sqrt(len(vs)))
    zp = abs(vp.mean() - ex) / (vp.std() / np.sqrt(len(vp)))
    print(f"  unbiasedness (20000 draws, one row): SS z={zs:.2f}  PFCV z={zp:.2f}   ({time.time()-t0:.1f}s)")
    ok = ok and zs < 4 and zp < 4
    return bool(ok)


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        aggregate(args.aggregate)
        return
    out = {"args": {k: v for k, v in vars(args).items()}, "instances": []}
    if args.smoke:
        out["smoke_ok"] = smoke_checks(args)
        print(f"SMOKE CHECKS {'PASS' if out['smoke_ok'] else 'FAIL'}")
    for K in args.n_active:
        out["instances"].append(run_instance(K, args))
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    if args.smoke:
        insts = {inst["K"]: inst for inst in out["instances"]}
        for route in ("R3.1", "R3.2"):
            print(f"[smoke-scale verdict logic, H4 only, not decisive] {route}: "
                  f"{json.dumps(verdict_for(insts, route))}")


if __name__ == "__main__":
    main()
