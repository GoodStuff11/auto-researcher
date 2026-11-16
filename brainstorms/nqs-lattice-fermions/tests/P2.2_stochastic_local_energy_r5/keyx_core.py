#!/usr/bin/env python3
# Verbatim copy of tests/P2.2_stochastic_local_energy_r4/micro_test.py (round-4 code; imported by the round-5 micro_test.py).
"""
P2.2 round-4 decisive test: exact keyed local energies for key-structured
backflow ansaetze, in two separate parts.

  Part S (scaling, ED-free):  operation count and wall-clock per sample of the
         exact keyed local energy (KEYX) against EXACT enumeration, screening
         (SCREEN, generation charged only for kept terms, as with HCI-style
         pre-sorted integrals), semistochastic (SS) and the truncated local
         energy of Liu & Clark arXiv:2502.18843 (TRUNC), on a distorted
         (non-symmetric) H8 chain in s+p Gaussian bases of growing size
         (K = 40 .. 136 spatial MOs by default; 168 optional), with samples from
         a CISD-like proposal (HF + MP1-weighted singles/doubles).
  Part A (accuracy, ED):  genuinely multireference distorted H_n chains
         (HF weight < 0.6, checked in code), 6-31G**, every arm trained by
         L-BFGS on the exact energy; accuracy loss of each restricted ansatz
         against the unrestricted NNBF-J of the same seed; exact biases of the
         biased controls; sampled variance increments of SS.

Arms (ansatz families; every one is evaluated by the SAME generic keyed code)
-----------------------------------------------------------------------------
NNBFJ  control. psi = J(n) det Phi(n)[occ n]; backflow input = all M
       occupations; Jastrow = one-body + all pairs + neural Jastrow (width hJ)
       on all occupations.
WIBJ   R3.1' (G = 0 end of R4.1): backflow input = window occupations only
       (window = lowest K_W spatial MOs, both spins); neural Jastrow on the
       window occupations (per key, not per term); pair Jastrow on all pairs.
CGK    R4.1 coarse-grained-key backflow: backflow input = window occupations
       plus 2G coarse external counts (electrons per spin in each of G
       external energy shells); neural Jastrow on the same features.
XSK    R4.2 cross-spin-keyed backflow: two blocks, det_alpha * det_beta;
       alpha orbitals are driven by the beta window occupations only and beta
       orbitals by the alpha window occupations only; neural Jastrow on the
       window; pair Jastrow on all pairs.
FJ     R4.0 diagnostic (critic's direction): WIBJ with a Jastrow that is
       factorizable across the window boundary (no external-external pair
       terms), the condition for the per-key Cholesky contraction.

Estimators
----------
EXACT  all connected n', one exact incremental amplitude each.
KEYX   all connected n', grouped by backflow key (the value of the backflow
       input features of n'); per key: occupied block A_k (n^2 h + n^3); per
       (key, added orbital): one orbital row and one Green's-function row
       (n h + n^2); per term: <= 2x2 minor, parity, O(1) pair Jastrow;
       neural Jastrow once per Jastrow key.  Exact for every arm (for NNBFJ
       every term is its own key, so there is no saving).
SCREEN keep |H_nn'| > eps (biased).  TRUNC sum only over n' in U = the
       unique configurations of the current batch of B_U samples (biased).
SS     |H| > eps exact + K_d heat-bath draws p ~ |H| (unbiased).
CONTR  (R4.0, operation count + identity check only) per-key Cholesky
       contraction of the (i,j in window -> a,b external) class under FJ.

Pre-committed verdict (python3 micro_test.py --aggregate DIR)
--------------------------------------------------------------
tau = 0.2 mHa (absolute).  Per route R in {R4.1 = CGK, R4.2 = XSK} (and
R3.1' = WIBJ, reported):
 Accuracy (Part A, per instance, at the largest K of that instance):
   premise: HF weight < 0.6 (enforced at build) and median NNBFJ dE < 1.6 mHa;
   delta_R = median over seeds of (dE_R - dE_NNBFJ), paired by seed.
   ACC_FAIL if delta_R > tau on any premise-valid instance;
   ACC_OK if delta_R <= tau on every instance and every premise holds;
   ACC_AMBIG otherwise (a premise fails).
 Control admissibility (Part A, NNBFJ state, largest K of every instance):
   SCREEN(eps) admissible iff |exact bias| <= tau on every instance;
   TRUNC(B_U) admissible iff |mean batch bias| + 2 SE <= tau on every
   instance; SS always admissible, charged its measured variance-increment
   ratio (selection half A, evaluation half B, tail rule on both halves).
 Cost (Part S, per K): C_R = ops_KEYX(R) * v_R, v_R = max over instances of
   the median (Var_R / Var_NNBFJ) from Part A; C_ctrl = min(EXACT,
   SCREEN(largest admissible eps), SS(best, x (1+inc)), TRUNC(smallest
   admissible B_U)), all from the NNBFJ arm, all generous to the controls
   (SCREEN/SS generation charged only for kept terms; admissibility at ED
   size assumed to transfer).  M_op(K) = C_R / C_ctrl; M_wall(K) the same with
   measured single-thread seconds, with a bootstrap 95% CI.
 PASS  ACC_OK and M_op(K_max) <= 0.5 and M_op(K_max) < M_op(K_min) and the
       wall-clock does not veto (veto: CI lower bound of M_wall(K_max) > 1).
 FAIL  ACC_FAIL, or M_op(K_max) >= 1, or wall-clock veto.
 AMBIGUOUS otherwise.  Pre-committed actions: ACC_AMBIG -> rerun Part A with
       --hidden 96 --maxiter 6000; cost-ambiguous -> rerun Part S with
       --basis-series 6,5 (K = 168): PASS iff M_op(168) <= 0.5, else KILL.
Hard Core: sign half only (same-sector competitors not examined).

Command-line options
--------------------
--part {S,A}            S = scaling (ED-free), A = accuracy (ED). Required unless
                        --smoke or --aggregate.
--n-atoms INT           H atoms = electrons N (even, >= 4). Default 8 (S), 6 (A).
--bond-length FLOAT     Mean H-H spacing in bohr (> 0). Default 2.0 (S), 3.2 (A).
--distort FLOAT         Uniform random displacement amplitude (bohr, >= 0) per
                        Cartesian coordinate, applied to a straight chain.
                        Default 0.25.
--geom-seed INT         Seed of the displacement. Default 1.
--basis-series LIST     Part S only: space-separated "ns,np" pairs; per H atom ns
                        s shells (6-31G** two s shells, then extra uncontracted
                        s exponents 0.05, 0.6, 2.5, 8.0, 0.02) and np p shells
                        (1.1, then 0.35, 3.5, 0.12, 9.0). Functions per atom
                        ns + 3 np. Default "2,1 3,2 4,3 5,4" (K = 40 72 104 136
                        for H8 before linear-dependence removal).
--n-active INT [...]    Part A only: active-space sizes K (valence MOs, then
                        p-type virtuals, then the rest). Default 8 10 12.
--window INT            K_W, window size in spatial MOs (>= N/2). Default N.
--groups INT            CGK external shells G (>= 1). Default 2.
--arms LIST             Subset of NNBFJ WIBJ CGK XSK FJ. Default all five.
--seeds INT [...]       Network seeds. Default 0 1 2.
--hidden INT            Backflow hidden width h. Default 64.
--hj INT                Neural-Jastrow width. Default 32.
--maxiter INT           Part A L-BFGS iterations on the exact energy. Default 3000.
--n-samples INT         Samples per state (even, >= 20). Default 2000 (A), 200 (S).
--n-wall INT            Part S: samples per arm that also get the full wall-clock
                        EXACT/SCREEN/SS timing and the KEYX-vs-EXACT check.
                        Default 12.
--eps-screen FLOAT...   SCREEN / SS thresholds. Default 1e-6 1e-5 1e-4 3e-4 1e-3 3e-3.
--k-grid INT...         SS draws per sample. Default 1 4 16 64.
--bu-grid INT...        TRUNC batch sizes B_U. Default 1000 4000 16000.
--chol-tol FLOAT        Pivoted-Cholesky tolerance (Part S). Default 1e-6.
--tau FLOAT             Absolute tolerance in Ha. Default 2e-4.
--allow-weak            Part A: do not abort when the HF weight is >= 0.6.
--out PATH              Result JSON. Default results_<part>_<tag>.json here.
--smoke                 Checks (integrals; spatial vs spin-orbital Hamiltonian;
                        keyed = exact ratios for every arm; gradients; Cholesky
                        contraction identity; TRUNC bias formula) plus tiny
                        Part A and Part S runs and a smoke-scale aggregate.
--aggregate DIR         Read results_A_*.json and results_S_*.json in DIR and
                        print the verdict.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # single-thread timings (set before numpy import)
import argparse
import glob
import json
import time
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import gammainc, gamma

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["NNBFJ", "WIBJ", "CGK", "XSK", "FJ"]
ROUTE_ARM = {"R4.1": "CGK", "R4.2": "XSK", "R3.1'": "WIBJ"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P2.2 round-4 keyed local-energy test")
    ap.add_argument("--part", choices=["S", "A"], default=None)
    ap.add_argument("--n-atoms", type=int, default=None)
    ap.add_argument("--bond-length", type=float, default=None)
    ap.add_argument("--distort", type=float, default=0.25)
    ap.add_argument("--geom-seed", type=int, default=1)
    ap.add_argument("--basis-series", nargs="+", default=["2,1", "3,2", "4,3", "5,4"])
    ap.add_argument("--n-active", type=int, nargs="+", default=[8, 10, 12])
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--hj", type=int, default=32)
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--n-wall", type=int, default=12)
    ap.add_argument("--eps-screen", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--k-grid", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--bu-grid", type=int, nargs="+", default=[1000, 4000, 16000])
    ap.add_argument("--chol-tol", type=float, default=1e-6)
    ap.add_argument("--tau", type=float, default=2e-4)
    ap.add_argument("--allow-weak", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", type=str, default=None)
    a = ap.parse_args(argv)
    if a.aggregate:
        return a
    if a.smoke:
        return a
    if a.part is None:
        ap.error("--part S or --part A is required")
    if a.n_atoms is None:
        a.n_atoms = 8 if a.part == "S" else 6
    if a.bond_length is None:
        a.bond_length = 2.0 if a.part == "S" else 3.2
    if a.n_samples is None:
        a.n_samples = 200 if a.part == "S" else 2000
    if a.n_atoms < 4 or a.n_atoms % 2:
        ap.error("--n-atoms must be even and >= 4")
    if a.window is None:
        a.window = a.n_atoms
    if a.window < a.n_atoms // 2 or a.groups < 1:
        ap.error("--window >= N/2 and --groups >= 1")
    if a.bond_length <= 0 or a.distort < 0 or a.hidden < 1 or a.hj < 1 or a.maxiter < 1:
        ap.error("bad --bond-length/--distort/--hidden/--hj/--maxiter")
    if a.n_samples < 20 or a.n_samples % 2:
        ap.error("--n-samples must be even and >= 20")
    if min(a.k_grid) < 1 or min(a.eps_screen) <= 0 or min(a.bu_grid) < 1:
        ap.error("--k-grid >= 1, --eps-screen > 0, --bu-grid >= 1")
    try:
        a.basis_series = [tuple(int(v) for v in s.split(",")) for s in a.basis_series]
    except ValueError:
        ap.error("--basis-series entries must look like ns,np")
    for ns, npp in a.basis_series:
        if not (2 <= ns <= 7 and 1 <= npp <= 5):
            ap.error("--basis-series: 2 <= ns <= 7, 1 <= np <= 5")
    if a.out is None:
        tag = (f"N{a.n_atoms}_" + ("-".join(f"{s}{p}" for s, p in a.basis_series) if a.part == "S"
                                   else "K" + "-".join(map(str, a.n_active))) + "_" + "-".join(a.arms))
        a.out = os.path.join(HERE, f"results_{a.part}_{tag}.json")
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
# Round-4 additions: larger s+p bases, distorted chains, spatial-ERI rows
# ----------------------------------------------------------------------------
EXTRA_S = [0.05, 0.6, 2.5, 8.0, 0.02]
EXTRA_P = [0.35, 3.5, 0.12, 9.0]


def make_basis(ns, npp):
    """Per-H basis: the two 6-31G** s shells (+ ns-2 extra uncontracted s) and
    the 6-31G** p shell (+ npp-1 extra uncontracted p). (2, 1) = 6-31G**."""
    b = [H631GSS[0], H631GSS[1]] + [(0, [e], [1.0]) for e in EXTRA_S[:ns - 2]]
    b += [(1, [1.1], [1.0])] + [(1, [e], [1.0]) for e in EXTRA_P[:npp - 1]]
    return b


def geometry_distorted(n_atoms, R, delta, seed):
    X = np.array([[i * R, 0.0, 0.0] for i in range(n_atoms)])
    rng = np.random.default_rng(seed)
    return X + delta * rng.uniform(-1, 1, X.shape)


def mo_transform(G, C):
    """(pq|rs) AO -> MO with four tensordots (memory ~ 2 K^4)."""
    for _ in range(4):
        G = np.tensordot(G, C, axes=([0], [0]))
    return np.ascontiguousarray(G)


def build_molecule(n_atoms, R, delta, gseed, basis, K_act=None, order="energy"):
    """RHF on a distorted H chain.  order='energy': all MOs by orbital energy
    (Part S).  order='valence-p': the N valence MOs, then virtual MOs with
    Mulliken p weight > 0.5, then the rest (Part A, as round 3).  Returns
    h, g (spatial MO integrals, chemists' notation), enuc, E_RHF, eps."""
    X = geometry_distorted(n_atoms, R, delta, gseed)
    S, hao, Gao, enuc, lab = integrals(X, basis)
    nocc = n_atoms // 2
    Cm, Eel, eps = rhf(S, hao, Gao, nocc)
    nmo = Cm.shape[1]
    if order == "valence-p":
        isp = np.array([l == 1 for (_, l) in lab])
        pw = (Cm * (S @ Cm))[isp].sum(0)
        rest = np.arange(n_atoms, nmo)
        idx = np.concatenate([np.arange(n_atoms), rest[pw[rest] > 0.5], rest[pw[rest] <= 0.5]])
    else:
        idx = np.arange(nmo)
    if K_act is not None:
        if K_act > nmo:
            raise ValueError(f"K_act={K_act} > number of MOs {nmo}")
        idx = idx[:K_act]
    C = Cm[:, idx]
    h = C.T @ hao @ C
    del S
    g = mo_transform(Gao, C)
    del Gao
    return h, g, enuc, Eel + enuc, eps[idx]


class SpatialHam:
    """Slater-Condon rows from SPATIAL integrals (memory K^4, not (2K)^4).
    Same conventions and output as the round-3 spin-orbital SQHam.row (checked
    elementwise in --smoke): spin-orbital p = spatial + K*spin, ascending-order
    Jordan-Wigner phases."""

    def __init__(self, h, g, enuc, K, N):
        self.K, self.M, self.N, self.enuc = K, 2 * K, N, enuc
        self.h, self.g = h, g
        self.J = np.einsum("ppqq->pq", g).copy()   # (pp|qq)
        self.X = np.einsum("pqqp->pq", g).copy()   # (pq|qp)
        self.spin = (np.arange(self.M) >= K).astype(np.int64)
        self.sp = np.arange(self.M) % K
        self.iu_occ = np.triu_indices(N, 1)
        # configuration codes: exact bit masks for M <= 62 (Part A, ED indexing);
        # 64-bit Zobrist hashes (random keys, wrap-around sums) for M > 62 (Part S)
        if self.M <= 62:
            self.z = np.int64(1) << np.arange(self.M, dtype=np.int64)
        else:
            self.z = np.random.default_rng(2024).integers(0, 2 ** 63, self.M, dtype=np.int64).astype(np.uint64)

    def code_of(self, occ):
        return self.z[occ].sum(dtype=self.z.dtype)

    def _G(self, a, i, b, j):
        """<ab||ij> for spin-orbital index arrays."""
        s, q = self.spin, self.sp
        d1 = (s[a] == s[i]) & (s[b] == s[j])
        d2 = (s[a] == s[j]) & (s[b] == s[i])
        return d1 * self.g[q[a], q[i], q[b], q[j]] - d2 * self.g[q[a], q[j], q[b], q[i]]

    def diag(self, occ):
        q, s = self.sp[occ], self.spin[occ]
        same = (s[:, None] == s[None, :])
        return float(self.h[q, q].sum() + 0.5 * (self.J[np.ix_(q, q)] - same * self.X[np.ix_(q, q)]).sum() + self.enuc)

    def row(self, code, occ):
        M, N, K = self.M, self.N, self.K
        o = np.zeros(M, np.int64); o[occ] = 1
        c = np.cumsum(o) - o
        virt = np.nonzero(o == 0)[0]
        Hnn = self.diag(occ)
        q, s = self.sp, self.spin
        # singles
        I, A = np.meshgrid(occ, virt, indexing="ij"); I = I.ravel(); A = A.ravel()
        m = s[I] == s[A]; I, A = I[m], A[m]
        qo, so = q[occ], s[occ]
        F = self.h[q[A], q[I]] + self.g[q[A][:, None], q[I][:, None], qo[None], qo[None]].sum(1) \
            - ((s[A][:, None] == so[None]) * self.g[q[A][:, None], qo[None], qo[None], q[I][:, None]]).sum(1)
        hs = F * (1 - 2 * ((c[I] + c[A] - (I < A)) & 1))
        # doubles
        oi, oj = occ[self.iu_occ[0]], occ[self.iu_occ[1]]
        vu = np.triu_indices(len(virt), 1)
        va, vb = virt[vu[0]], virt[vu[1]]
        nI = len(oi); nA = len(va)
        Ii = np.repeat(oi, nA); Jj = np.repeat(oj, nA)
        Aa = np.tile(va, nI); Bb = np.tile(vb, nI)
        m = (s[Ii] + s[Jj]) == (s[Aa] + s[Bb])
        Ii, Jj, Aa, Bb = Ii[m], Jj[m], Aa[m], Bb[m]
        val = self._G(Aa, Ii, Bb, Jj)
        ph = c[Ii] + c[Jj] - 1 + c[Bb] - (Ii < Bb) - (Jj < Bb) + c[Aa] - (Ii < Aa) - (Jj < Aa)
        hd = val * (1 - 2 * (ph & 1))
        z = self.z
        code = z.dtype.type(code)
        with np.errstate(over="ignore"):
            cs = code - z[I] + z[A]
            cd = code - z[Ii] - z[Jj] + z[Aa] + z[Bb]
        out = {"code": np.concatenate([cs, cd]), "h": np.concatenate([hs, hd]),
               "rem": np.concatenate([np.stack([I, -np.ones_like(I)], 1), np.stack([Ii, Jj], 1)]),
               "add": np.concatenate([np.stack([A, -np.ones_like(A)], 1), np.stack([Aa, Bb], 1)]),
               "Hnn": Hnn}
        keep = np.abs(out["h"]) > 1e-13
        for k in ("code", "h", "rem", "add"):
            out[k] = out[k][keep]
        return out


def pivoted_cholesky(g, tol, maxrank=None):
    """(pq|rs) = sum_gamma L[gamma,pq] L[gamma,rs] to max residual diagonal tol."""
    K = g.shape[0]
    V = g.reshape(K * K, K * K)
    d = np.diag(V).copy()
    maxrank = maxrank or 20 * K   # Cholesky rank of molecular ERIs is ~ 5-10 K
    L = np.zeros((min(maxrank, K * K), K * K))
    n = 0
    while n < L.shape[0]:
        p = int(np.argmax(d))
        if d[p] < tol:
            break
        col = V[:, p] - L[:n].T @ L[:n, p]
        L[n] = col / np.sqrt(d[p])
        d -= L[n] ** 2
        n += 1
    return L[:n].reshape(n, K, K)


def decode(codes, M, N):
    bits = (np.asarray(codes)[:, None] >> np.arange(M)[None]) & 1
    return np.nonzero(bits)[1].reshape(-1, N)


def occ_from_row(occ_n, rem, add):
    """Sorted occupations of every connected n' (no bit codes; any M)."""
    T = len(rem); N = len(occ_n)
    pos = {p: k for k, p in enumerate(occ_n)}
    L = np.repeat(occ_n[None], T, 0)
    ar = np.arange(T)
    L[ar, np.vectorize(pos.get)(rem[:, 0])] = add[:, 0]
    d2 = rem[:, 1] >= 0
    if d2.any():
        L[ar[d2], np.vectorize(pos.get)(rem[d2, 1])] = add[d2, 1]
    return np.sort(L, 1)


_HASHW = {}


def _rowkey(Mx):
    """Integer rows -> int64 keys (random-weight hash; exact with probability ~1).
    Replaces np.unique(axis=0), which is slow."""
    d = Mx.shape[1]
    if d not in _HASHW:
        _HASHW[d] = np.random.default_rng(77 + d).integers(1, 2 ** 40, d, dtype=np.int64)
    with np.errstate(over="ignore"):
        return Mx.astype(np.int64) @ _HASHW[d]


# ----------------------------------------------------------------------------
# Generic key-structured backflow-Jastrow ansatz
# ----------------------------------------------------------------------------
class Ansatz:
    """psi(n) = J(n) * prod_s det Phi_s(f_s(n))[occ_s(n)]
    Block s covers spin-orbitals [lo, lo+nr) and determinant columns
    [c0, c0+ncol) of the sorted occupation list.  f_s(n) = F_s x, x = 2n-1,
    F_s an integer feature matrix; Phi_s = Phi0_s + reshape(W2_s tanh(W1_s f_s + b1_s)).
    ln J = u.n + sum_{p<q, Vmask} v_pq n_p n_q + a.tanh(Wj (F_J x) + bj).
    The backflow key of n' is the value of f_s(n'); the Jastrow key is F_J x'."""

    def __init__(self, K, N, arm, hidden, hj, rng, scale=0.05, KW=None, G=2):
        M = 2 * K; nh = N // 2
        self.K, self.M, self.N, self.arm, self.Hh, self.hj = K, M, N, arm, hidden, hj
        KW = N if KW is None else min(KW, K)
        win = np.zeros(M, bool); win[:KW] = True; win[K:K + KW] = True
        self.win = win
        I = np.eye(M, dtype=np.int64)
        alpha = np.arange(M) < K
        if arm == "NNBFJ":
            blocks = [(0, M, 0, N, I)]; FJ = I
        elif arm in ("WIBJ", "FJ"):
            blocks = [(0, M, 0, N, I[win])]; FJ = I[win]
        elif arm == "CGK":
            ext = np.arange(KW, K)
            groups = np.array_split(ext, G) if len(ext) else []
            S = []
            for sp in (0, 1):
                for gr in groups:
                    r = np.zeros(M, np.int64); r[gr + sp * K] = 1; S.append(r)
            F = np.vstack([I[win]] + ([np.array(S)] if S else []))
            blocks = [(0, M, 0, N, F)]; FJ = F
        elif arm == "XSK":
            blocks = [(0, K, 0, nh, I[win & ~alpha]), (K, K, nh, nh, I[win & alpha])]
            FJ = I[win]
        else:
            raise ValueError(arm)
        self.blocks = blocks
        self.FJ = FJ
        iu = np.triu_indices(M, 1)
        vm = np.ones(len(iu[0]), bool)
        if arm == "FJ":
            vm = ~((~win)[iu[0]] & (~win)[iu[1]])   # no external-external pair terms
        self.iu = (iu[0][vm], iu[1][vm])
        self.shapes = []
        init = {}
        for s, (lo, nr, c0, ncol, F) in enumerate(blocks):
            P0 = np.zeros((nr, ncol))
            if len(blocks) == 1:
                P0[:K, :nh] = np.eye(K)[:, :nh]; P0[K:, nh:] = P0[:K, :nh]
            else:
                P0[:, :] = np.eye(K)[:, :nh]
            d = F.shape[0]
            self.shapes += [(f"Phi0_{s}", (nr, ncol)), (f"W2_{s}", (nr * ncol, hidden)), (f"W1_{s}", (hidden, d)),
                            (f"b1_{s}", (hidden,))]
            init.update({f"Phi0_{s}": P0, f"W2_{s}": scale * rng.standard_normal((nr * ncol, hidden)),
                         f"W1_{s}": scale * rng.standard_normal((hidden, d)), f"b1_{s}": np.zeros(hidden)})
        dJ = FJ.shape[0]
        self.shapes += [("u", (M,)), ("v", (len(self.iu[0]),)), ("Wj", (hj, dJ)), ("bj", (hj,)), ("aj", (hj,))]
        init.update({"u": np.zeros(M), "v": np.zeros(len(self.iu[0])), "Wj": scale * rng.standard_normal((hj, dJ)),
                     "bj": np.zeros(hj), "aj": scale * rng.standard_normal(hj)})
        self.theta0 = np.concatenate([init[k].ravel() for k, _ in self.shapes])

    def unpack(self, th):
        out, i = {}, 0
        for k, s in self.shapes:
            n = int(np.prod(s)); out[k] = th[i:i + n].reshape(s); i += n
        V = np.zeros((self.M, self.M)); V[self.iu] = out["v"]; out["V"] = V + V.T
        for s, (lo, nr, c0, ncol, F) in enumerate(self.blocks):
            out[f"W1F_{s}"] = out[f"W1_{s}"] @ F            # h x M
            out[f"W2r_{s}"] = out[f"W2_{s}"].reshape(nr, ncol, self.Hh)
        out["WjF"] = out["Wj"] @ self.FJ                    # hj x M
        return out

    # ---- full-vector amplitudes and exact-energy gradient (training only) ----
    def amplitudes(self, th, occ, occ_idx):
        P = self.unpack(th)
        x = 2.0 * occ - 1.0
        psi = np.ones(len(occ)); cache = []
        for s, (lo, nr, c0, ncol, F) in enumerate(self.blocks):
            f = x @ F.T
            hid = np.tanh(f @ P[f"W1_{s}"].T + P[f"b1_{s}"])
            Phi = P[f"Phi0_{s}"][None] + (hid @ P[f"W2_{s}"].T).reshape(-1, nr, ncol)
            rows = occ_idx[:, c0:c0 + ncol] - lo
            A = np.take_along_axis(Phi, rows[:, :, None], axis=1)
            psi = psi * np.linalg.det(A)
            cache.append((f, hid, Phi, A, rows))
        o = occ.astype(float)
        fJ = x @ self.FJ.T
        tj = np.tanh(fJ @ P["Wj"].T + P["bj"])
        logJ = o @ P["u"] + 0.5 * np.einsum("bp,bp->b", o @ P["V"], o) + tj @ P["aj"]
        return psi * np.exp(logJ), (cache, P, o, tj, fJ)

    def energy_grad(self, th, H, occ, occ_idx):
        psi, (cache, P, o, tj, fJ) = self.amplitudes(th, occ, occ_idx)
        nrm = psi @ psi
        Hpsi = H @ psi
        E = psi @ Hpsi / nrm
        c = 2 * (Hpsi - E * psi) / nrm
        cp = c * psi
        g = {}
        for s, (lo, nr, c0, ncol, F) in enumerate(self.blocks):
            f, hid, Phi, A, rows = cache[s]
            gA = cp[:, None, None] * np.transpose(np.linalg.inv(A), (0, 2, 1))
            gPhi = np.zeros_like(Phi)
            np.put_along_axis(gPhi, rows[:, :, None], gA, axis=1)
            gflat = gPhi.reshape(len(psi), -1)
            g[f"Phi0_{s}"] = gPhi.sum(0)
            g[f"W2_{s}"] = gflat.T @ hid
            dz = (gflat @ P[f"W2_{s}"]) * (1 - hid ** 2)
            g[f"W1_{s}"] = dz.T @ f
            g[f"b1_{s}"] = dz.sum(0)
        g["u"] = o.T @ cp
        g["v"] = (o.T @ (o * cp[:, None]))[self.iu]
        g["aj"] = tj.T @ cp
        dzj = cp[:, None] * (1 - tj ** 2) * P["aj"][None]
        g["Wj"] = dzj.T @ fJ
        g["bj"] = dzj.sum(0)
        return E, np.concatenate([g[k].ravel() for k, _ in self.shapes])

    # ---- per-sample kernels ---------------------------------------------------
    def setup(self, P, occ_n):
        x = 2.0 * np.isin(np.arange(self.M), occ_n) - 1.0
        st = {"x": x, "occ": occ_n, "blk": []}
        detA = 1.0
        for s, (lo, nr, c0, ncol, F) in enumerate(self.blocks):
            f = F @ x
            z = P[f"W1_{s}"] @ f + P[f"b1_{s}"]
            rows = occ_n[c0:c0 + ncol] - lo
            A = P[f"Phi0_{s}"][rows] + P[f"W2r_{s}"][rows] @ np.tanh(z)
            d = np.linalg.det(A); detA *= d
            st["blk"].append({"f": f.astype(np.int64), "z": z, "rows": rows, "det": d})
        st["detA"] = detA
        o = (x + 1) / 2
        st["fp"] = P["V"] @ o
        st["fJ"] = (self.FJ @ x).astype(np.int64)
        st["zj"] = P["Wj"] @ st["fJ"] + P["bj"]
        return st

    def jastrow_pair(self, P, st, rem, add):
        u, V, f = P["u"], P["V"], st["fp"]
        i, a = rem[:, 0], add[:, 0]
        d = u[a] - u[i] + f[a] - f[i] - V[a, i]
        d2 = rem[:, 1] >= 0
        j, b = rem[d2, 1], add[d2, 1]
        d[d2] += u[b] - u[j] + f[b] - f[j] + V[a[d2], b] + V[i[d2], j] - V[b, i[d2]] - V[a[d2], j] - V[b, j]
        return d

    def _dcols(self, Fm, rem, add):
        """2 * (F[:,a] - F[:,i] (+ F[:,b] - F[:,j])) per term (T x d)."""
        FT = Fm.T
        d = 2 * (FT[add[:, 0]] - FT[rem[:, 0]])
        d2 = rem[:, 1] >= 0
        d[d2] += 2 * (FT[add[d2, 1]] - FT[rem[d2, 1]])
        return d

    def ratio_exact(self, P, st, rem, add, occ_j, chunk=2048):
        """Exact psi(n')/psi(n), one incremental amplitude per term:
        O(sum_s (n_s^2 h + n_s^3) + hj) per term (chunked)."""
        T = len(rem)
        out = np.empty(T)
        for c0_ in range(0, T, chunk):
            sl = slice(c0_, min(T, c0_ + chunk))
            rm, ad, oj = rem[sl], add[sl], occ_j[sl]
            r = np.ones(len(rm))
            for s, (lo, nr, c0, ncol, F) in enumerate(self.blocks):
                dz = self._dcols(P[f"W1F_{s}"], rm, ad)
                hid = np.tanh(st["blk"][s]["z"][None] + dz)
                rows = oj[:, c0:c0 + ncol] - lo
                A = P[f"Phi0_{s}"][rows] + np.einsum("bknh,bh->bkn", P[f"W2r_{s}"][rows], hid)
                r *= np.linalg.det(A) / st["blk"][s]["det"]
            dzj = self._dcols(P["WjF"], rm, ad)
            dj = (np.tanh(st["zj"][None] + dzj) - np.tanh(st["zj"])[None]) @ P["aj"]
            out[sl] = r * np.exp(self.jastrow_pair(P, st, rm, ad) + dj)
        return out

    def ratio_key(self, P, st, rem, add):
        """Exact psi(n')/psi(n) for all terms, grouped by backflow key.
        Returns ratios and a counter dict for the operation-count model."""
        T = len(rem)
        r = np.ones(T)
        cnt = {"keys": [], "pairs": [], "jkeys": 0, "T": T}
        occ_n = st["occ"]
        ar = np.arange(T)
        for s, (lo, nr, c0, ncol, F) in enumerate(self.blocks):
            bs = st["blk"][s]
            dF = self._dcols(F, rem, add)
            fk = bs["f"][None] + dF
            _, first, inv = np.unique(_rowkey(fk), return_index=True, return_inverse=True)
            keys = fk[first]
            inv = inv.ravel()
            Th = np.tanh(keys @ P[f"W1_{s}"].T + P[f"b1_{s}"])           # nk x h
            rows_n = bs["rows"]
            Ak = P[f"Phi0_{s}"][rows_n][None] + np.einsum("rnh,kh->krn", P[f"W2r_{s}"][rows_n], Th)
            dk = np.linalg.det(Ak)
            Akinv = np.linalg.inv(Ak)
            # replacements inside this block: slot (position in rows_n) <- added local orbital
            inb_r = (rem >= lo) & (rem < lo + nr)
            inb_a = (add >= lo) & (add < lo + nr)
            nrep = inb_r.sum(1)
            assert np.all(nrep == inb_a.sum(1))
            pos = np.full(nr, -1); pos[rows_n] = np.arange(ncol)
            # first and second replacement of each term (ordered as in rem/add)
            o1 = np.where(inb_r[:, 0], 0, 1)
            sl1 = np.where(nrep >= 1, pos[np.clip(rem[ar, o1] - lo, 0, nr - 1)], -1)
            ad1 = np.where(nrep >= 1, add[ar, np.where(inb_a[:, 0], 0, 1)] - lo, -1)
            sl2 = np.where(nrep == 2, pos[np.clip(rem[:, 1] - lo, 0, nr - 1)], -1)
            ad2 = np.where(nrep == 2, add[:, 1] - lo, -1)
            # distinct (key, added orbital) rows and Green's-function rows
            need = np.concatenate([np.stack([inv[nrep >= 1], ad1[nrep >= 1]], 1),
                                   np.stack([inv[nrep == 2], ad2[nrep == 2]], 1)])
            minor = np.ones(T)
            npairs = 0
            if len(need):
                pc = need[:, 0] * nr + need[:, 1]
                up, pinv = np.unique(pc, return_inverse=True)
                npairs = len(up)
                kk, aa = up // nr, up % nr
                rowsv = P[f"Phi0_{s}"][aa] + np.einsum("pnh,ph->pn", P[f"W2r_{s}"][aa], Th[kk])
                Gr = np.einsum("pn,pnc->pc", rowsv, Akinv[kk])               # G rows
                m1 = nrep >= 1; m2 = nrep == 2
                n1 = int(m1.sum())
                g1 = Gr[pinv[:n1]]
                t1 = np.nonzero(m1)[0]
                minor[t1] = g1[np.arange(n1), sl1[t1]]
                if m2.any():
                    t2 = np.nonzero(m2)[0]
                    w1 = np.searchsorted(t1, t2)
                    ga = g1[w1]; gb = Gr[pinv[n1:]]
                    q = np.arange(len(t2))
                    minor[t2] = ga[q, sl1[t2]] * gb[q, sl2[t2]] - ga[q, sl2[t2]] * gb[q, sl1[t2]]
                # parity of the slot-ordered row list relative to sorted order
                L = np.repeat(rows_n[None], T, 0)
                L[t1, sl1[t1]] = ad1[t1]
                if m2.any():
                    L[t2, sl2[t2]] = ad2[t2]
                par = np.zeros(T, np.int64)
                for p_ in range(ncol):
                    for q_ in range(p_ + 1, ncol):
                        par += L[:, p_] > L[:, q_]
                minor *= 1 - 2 * (par & 1)
            r *= dk[inv] / bs["det"] * minor
            cnt["keys"].append(len(keys)); cnt["pairs"].append(npairs)
        # Jastrow: pair part per term, neural part per Jastrow key
        dJ = self._dcols(self.FJ, rem, add)
        fj = st["fJ"][None] + dJ
        _, jfirst, jinv = np.unique(_rowkey(fj), return_index=True, return_inverse=True)
        jk = fj[jfirst]
        cnt["jkeys"] = len(jk)
        nj = np.tanh(jk @ P["Wj"].T + P["bj"]) @ P["aj"] - np.tanh(st["zj"]) @ P["aj"]
        r *= np.exp(self.jastrow_pair(P, st, rem, add) + nj[jinv.ravel()])
        return r, cnt


# ----------------------------------------------------------------------------
# Operation-count model (flops-like units; the same constants for all arms)
# ----------------------------------------------------------------------------
C_GEN = 30      # Slater-Condon generation of one connected term (gather + phase)
C_HASH = 5      # TRUNC: hash lookup of one connected code
C_PAIRJ = 15    # one-body + pair Jastrow ratio of one term


def amp_ops(net):
    return sum(ncol * ncol * net.Hh + ncol ** 3 + 4 * net.Hh for (_, _, _, ncol, _) in net.blocks) \
        + 6 * net.hj + C_PAIRJ


def keyx_ops(net, cnt):
    o = 0.0
    for (lo, nr, c0, ncol, F), nk, npair in zip(net.blocks, cnt["keys"], cnt["pairs"]):
        o += nk * (4 * net.Hh + ncol * ncol * net.Hh + 2 * ncol ** 3) + npair * (ncol * net.Hh + ncol * ncol)
        o += cnt["T"] * (10 + 0.5 * ncol * ncol)
    o += cnt["jkeys"] * (net.FJ.shape[0] * net.hj + net.hj) + cnt["T"] * C_PAIRJ
    return o


def setup_ops(net):
    return sum(F.shape[0] * net.Hh + ncol * ncol * net.Hh + ncol ** 3 for (_, _, _, ncol, F) in net.blocks) \
        + net.FJ.shape[0] * net.hj + net.M


# ----------------------------------------------------------------------------
# R4.0: per-key Cholesky contraction of the (i,j in window -> a,b external)
# class under the factorizable (FJ) Jastrow
# ----------------------------------------------------------------------------
def contract_ij(net, P, st, ham, i, j, Lc):
    """Sum over all n' = n - i - j + a + b with a, b external (unoccupied in n)
    of H_nn' psi(n')/psi(n), by generalized Wick + Cholesky:
      S = sum_g (sum_a L_ai G~_a,pi)(sum_b L_bj G~_b,pj)
          - d(s_i,s_j) (sum_a L_aj G~_a,pi)(sum_b L_bi G~_b,pj),
    G~ = diag(exp(lambda)) Phi_k A_k^{-1} on external rows.  Returns
    (value, ops)."""
    assert net.arm == "FJ" and len(net.blocks) == 1
    K, M, N = net.K, net.M, net.N
    lo, nr, c0, ncol, F = net.blocks[0]
    occ = st["occ"]
    fk = st["blk"][0]["f"] - 2 * F[:, i] - 2 * F[:, j]
    t = np.tanh(P["W1_0"] @ fk + P["b1_0"])
    Phik = P["Phi0_0"] + P["W2r_0"] @ t
    Ak = Phik[occ]
    G = Phik @ np.linalg.inv(Ak)
    ext = np.nonzero(~net.win)[0]
    u, V, f = P["u"], P["V"], st["fp"]
    lam = u[ext] + f[ext] - V[ext, i] - V[ext, j]
    jk = np.tanh(P["Wj"] @ (st["fJ"] - 2 * net.FJ[:, i] - 2 * net.FJ[:, j]) + P["bj"]) @ P["aj"] \
        - np.tanh(st["zj"]) @ P["aj"]
    cJ = -u[i] - u[j] - f[i] - f[j] + V[i, j] + jk
    pos = {p: k for k, p in enumerate(occ)}
    Gt = G[ext] * np.exp(lam)[:, None]
    sp, sq = ham.spin, ham.sp
    si, sj = sp[i], sp[j]
    ea, qa = sp[ext], sq[ext]
    Li = Lc[:, qa, sq[i]]; Lj = Lc[:, qa, sq[j]]                     # Ng x |ext|
    gi, gj = Gt[:, pos[i]], Gt[:, pos[j]]
    mi, mj = (ea == si), (ea == sj)
    # Coulomb-like: a with spin of i, b with spin of j; exchange-like: a with
    # spin of j, b with spin of i (nonzero also for opposite spins, because the
    # backflow determinant is a generalized, spin-mixing one: G_{a beta, i alpha} != 0)
    S = np.sum((Li[:, mi] @ gi[mi]) * (Lj[:, mj] @ gj[mj]))
    S -= np.sum((Lj[:, mj] @ gi[mj]) * (Li[:, mi] @ gj[mi]))
    val = np.linalg.det(Ak) / st["detA"] * np.exp(cJ) * S
    Ng = Lc.shape[0]
    ops = (4 * net.Hh + M * ncol * net.Hh + 2 * ncol ** 3 + M * ncol * ncol) + 4 * Ng * len(ext) / 2
    return float(val), float(ops)


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def hf_code(K, N):
    hf = sum(1 << i for i in range(N // 2))
    return hf | (hf << K)


def make_net(K, N, arm, args, seed):
    rng = np.random.default_rng(1000 * seed + 7)
    return Ansatz(K, N, arm, args.hidden, args.hj, rng, 0.05, args.window, args.groups)


def tail_ok(inc):
    """Variance increment estimable: top 1% of samples carry <= 50% of it."""
    d = (inc - inc.mean()) ** 2
    if d.sum() <= 0:
        return True
    return np.sort(d)[::-1][:max(1, len(d) // 100)].sum() / d.sum() <= 0.5


def boot_ratio_ci(a, b, n=2000, seed=0):
    """95% bootstrap CI of mean(a)/mean(b), paired resampling."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    idx = rng.integers(0, len(a), (n, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


# ----------------------------------------------------------------------------
# Part A: accuracy on multireference ED instances
# ----------------------------------------------------------------------------
def run_part_A_instance(K, args):
    t0 = time.time()
    N = args.n_atoms; nocc = N // 2
    h, g, enuc, Erhf, eps = build_molecule(N, args.bond_length, args.distort, args.geom_seed, H631GSS, K, "valence-p")
    ham = SpatialHam(h, g, enuc, K, N)
    codes, occ, oi = build_space(K, nocc, nocc)
    H = build_hamiltonian(ham, codes, oi)
    w2, v2 = ground_states(H, 2)
    E0, psi0 = w2[0], v2[:, 0]
    hfw = float(psi0[np.searchsorted(codes, hf_code(K, N))] ** 2)
    Hc = H.tocoo(); m = Hc.row != Hc.col
    prod = Hc.data[m] * psi0[Hc.row[m]] * psi0[Hc.col[m]]
    frustr = float(np.abs(prod[prod > 0]).sum() / np.abs(prod).sum())
    print(f"\n=== [A] H{N} distorted chain R={args.bond_length} d={args.distort} K={K} dim={len(codes)} "
          f"E_RHF={Erhf:.6f} E_FCI={E0:.6f} HF wt={hfw:.3f} frustr={frustr:.3f} nnz/row={H.nnz/len(codes):.0f} "
          f"({time.time()-t0:.1f}s)", flush=True)
    if hfw >= 0.6 and not args.allow_weak:
        raise SystemExit(f"instance not multireference: HF weight {hfw:.3f} >= 0.6")
    out = {"K": K, "N": N, "dim": len(codes), "E_FCI": float(E0), "HF_weight": hfw, "frustrated_fraction": frustr,
           "gap": float(w2[1] - E0), "runs": []}
    Hoff = H - sp.diags(H.diagonal())
    for seed in args.seeds:
        for arm in args.arms:
            net = make_net(K, N, arm, args, seed)
            t1 = time.time()
            r = minimize(lambda t: net.energy_grad(t, H, occ, oi), net.theta0, jac=True, method="L-BFGS-B",
                         options={"maxiter": args.maxiter, "maxcor": 30, "gtol": 1e-12, "ftol": 1e-16})
            th = r.x
            psi, _ = net.amplitudes(th, occ, oi)
            nrm = psi @ psi
            wv = psi ** 2 / nrm
            Hpsi = H @ psi
            Ev = float(psi @ Hpsi / nrm)
            EL = Hpsi / psi
            varx = float(wv @ (EL - Ev) ** 2)
            run = {"seed": seed, "arm": arm, "dE_mHa": 1e3 * (Ev - E0), "var_exact": varx,
                   "t_train_s": time.time() - t1}
            srng = np.random.default_rng(10 ** 6 + 1000 * seed + 13)
            sidx = srng.choice(len(codes), size=args.n_samples, p=wv)
            P = net.unpack(th)
            # exactness of the keyed local energy on the samples; per-sample ops
            dev, opsK, opsE = [], [], []
            for k in sidx[: min(len(sidx), 200)]:
                row = ham.row(codes[k], oi[k])
                st = net.setup(P, oi[k])
                rk, cnt = net.ratio_key(P, st, row["rem"], row["add"])
                dev.append(abs(row["Hnn"] + row["h"] @ rk - EL[k]))
                opsK.append(keyx_ops(net, cnt) + C_GEN * len(row["h"]) + setup_ops(net))
                opsE.append(len(row["h"]) * (C_GEN + amp_ops(net)) + setup_ops(net))
            run.update({"KEYX_max_dev": float(max(dev)), "ops_KEYX_over_EXACT_ED": float(np.mean(opsK) / np.mean(opsE))})
            if arm == "NNBFJ":
                run["controls"] = control_block_A(args, H, Hoff, psi, EL, codes, oi, ham, net, P, sidx, srng)
            out["runs"].append(run)
            print(f"  seed={seed} {arm:5s} dE={run['dE_mHa']:8.3f} mHa Var(EL)={varx:.2e} KEYX-EL max={run['KEYX_max_dev']:.1e} "
                  f"ops(KEYX/EXACT, ED size)={run['ops_KEYX_over_EXACT_ED']:.3f} (train {run['t_train_s']:.0f}s)", flush=True)
            if arm == "NNBFJ":
                cb = run["controls"]
                print("      SCREEN exact bias (mHa): " + " ".join(f"{e}:{1e3*b:.4f}" for e, b in cb["screen_bias"].items()))
                print("      TRUNC batch bias (mHa): " + " ".join(f"{bu}:{1e3*v[0]:.4f}+-{1e3*v[1]:.4f}" for bu, v in cb["trunc_bias"].items()))
                print(f"      SS selected (half A) eps={cb['ss_sel']['eps']:g} K={cb['ss_sel']['K']} inc ratio (half B)={cb['ss_sel']['inc_ratio_B']:.3f}")
    return out


def control_block_A(args, H, Hoff, psi, EL, codes, oi, ham, net, P, sidx, srng):
    """Exact biases of SCREEN and TRUNC; sampled variance increments of SS
    (grid selection on half A, evaluation on half B, tail rule on both)."""
    nrm = psi @ psi
    res = {"screen_bias": {}, "trunc_bias": {}, "ss_grid": {}}
    Hc = Hoff.tocsr()
    for e in args.eps_screen:
        He = Hc.copy(); He.data[np.abs(He.data) <= e] = 0.0
        res["screen_bias"][str(e)] = float(psi @ ((He - Hc) @ psi) / nrm)
    wv = psi ** 2 / nrm
    for bu in args.bu_grid:
        bs = []
        for rep in range(5):
            batch = srng.choice(len(codes), size=bu, p=wv)
            inU = np.zeros(len(codes), bool); inU[batch] = True
            Hout = Hc[batch][:, ~inU]
            e = -(Hout @ psi[~inU]) / psi[batch]
            bs.append(float(e.mean()))
        res["trunc_bias"][str(bu)] = [float(np.mean(bs)), float(np.std(bs, ddof=1) / np.sqrt(len(bs)))]
    # SS sampled on the iid samples, per-sample rows
    B = len(sidx); hA, hB = slice(0, B // 2), slice(B // 2, B)
    vals = {(e, Kd): np.empty(B) for e in args.eps_screen for Kd in args.k_grid}
    ops = {(e, Kd): np.empty(B) for e in args.eps_screen for Kd in args.k_grid}
    amp = amp_ops(net)
    for t, k in enumerate(sidx):
        row = ham.row(codes[k], oi[k])
        hrow = row["h"]
        j = np.searchsorted(codes, row["code"])
        x = hrow * psi[j] / psi[k]          # exact terms (the amplitudes the estimator would evaluate)
        for e in args.eps_screen:
            mk = np.abs(hrow) > e
            base = row["Hnn"] + x[mk].sum()
            Tl = np.nonzero(~mk)[0]
            for Kd in args.k_grid:
                v = base
                if len(Tl):
                    p = np.abs(hrow[Tl]); p = p / p.sum()
                    dr = srng.choice(len(Tl), size=Kd, p=p)
                    v += np.mean(x[Tl[dr]] / p[dr])
                vals[(e, Kd)][t] = v
                ops[(e, Kd)][t] = mk.sum() * (C_GEN + amp) + (Kd if len(Tl) else 0) * amp
    ELs = EL[sidx]
    varpi = float(wv @ (EL - (wv @ EL)) ** 2)
    best = None
    for key, v in vals.items():
        inc = v - ELs
        incA = np.var(v[hA], ddof=1) - np.var(ELs[hA], ddof=1)
        incB = np.var(v[hB], ddof=1) - np.var(ELs[hB], ddof=1)
        okA, okB = tail_ok(inc[hA]), tail_ok(inc[hB])
        res["ss_grid"][f"{key[0]:g},{key[1]}"] = {"inc_ratio_A": float(max(incA, 0) / varpi),
                                                  "inc_ratio_B": float(max(incB, 0) / varpi),
                                                  "tail_ok_A": bool(okA), "tail_ok_B": bool(okB),
                                                  "mean_ops": float(ops[key].mean())}
        if okA:
            cA = (1 + max(incA, 0) / varpi) * ops[key][hA].mean()
            if best is None or cA < best[0]:
                best = (cA, key)
    if best is None:
        res["ss_sel"] = {"eps": -1, "K": -1, "inc_ratio_B": float("inf"), "tail_ok_B": False}
    else:
        key = best[1]; d = res["ss_grid"][f"{key[0]:g},{key[1]}"]
        res["ss_sel"] = {"eps": key[0], "K": key[1], "inc_ratio_B": d["inc_ratio_B"], "tail_ok_B": d["tail_ok_B"]}
    res["var_pi"] = varpi
    return res


# ----------------------------------------------------------------------------
# Part S: ED-free scaling
# ----------------------------------------------------------------------------
def cisd_sampler(ham, eps_orb, K, N, B, rng):
    """CISD-like proposal: HF with probability 0.4, a single with 0.1 (uniform),
    a double with 0.5 drawn with probability ~ |t|^2, t = <ab||ij>/(e_i+e_j-e_a-e_b).
    Returns a list of sorted occupation arrays."""
    occ = np.concatenate([np.arange(N // 2), K + np.arange(N // 2)])
    row = ham.row(ham.code_of(occ), occ)
    e = np.concatenate([eps_orb, eps_orb])
    isd = row["rem"][:, 1] >= 0
    i, j, a, b = row["rem"][:, 0], row["rem"][:, 1], row["add"][:, 0], row["add"][:, 1]
    den = np.where(isd, e[i] + e[np.maximum(j, 0)] - e[a] - e[np.maximum(b, 0)], 1.0)
    t2 = np.where(isd, (row["h"] / den) ** 2, 0.0)
    sing = np.nonzero(~isd)[0]
    occs = occ_from_row(occ, row["rem"], row["add"])
    out = []
    for _ in range(B):
        u = rng.random()
        if u < 0.4:
            out.append(occ.copy())
        elif u < 0.5:
            out.append(occs[rng.choice(sing)])
        else:
            out.append(occs[rng.choice(len(t2), p=t2 / t2.sum())])
    return out


def run_part_S_basis(ns, npp, args):
    t0 = time.time()
    N = args.n_atoms
    h, g, enuc, Erhf, eps = build_molecule(N, args.bond_length, args.distort, args.geom_seed, make_basis(ns, npp), None, "energy")
    K = h.shape[0]; M = 2 * K
    ham = SpatialHam(h, g, enuc, K, N)
    tb = time.time() - t0
    Lc = pivoted_cholesky(g, args.chol_tol)
    Ng = Lc.shape[0]
    Ng4 = pivoted_cholesky(g, 1e-4).shape[0]
    print(f"\n=== [S] H{N} distorted chain R={args.bond_length} d={args.distort} basis(ns,np)=({ns},{npp}) K={K} M={M} "
          f"E_RHF={Erhf:.6f} N_chol(1e-6)={Ng} ({Ng/K:.2f} K) N_chol(1e-4)={Ng4} (build {tb:.0f}s)", flush=True)
    rng = np.random.default_rng(12345)
    samples = cisd_sampler(ham, eps, K, N, args.n_samples, rng)
    U = np.unique([ham.code_of(o) for o in samples])
    out = {"K": K, "M": M, "N": N, "ns": ns, "np": npp, "N_chol": Ng, "N_chol_1e-4": Ng4, "arms": {}}
    rows = []
    for occ_n in samples:
        rows.append((occ_n, ham.row(ham.code_of(occ_n), occ_n)))
    nconn = np.array([len(r[1]["h"]) for r in rows])
    kept = {str(e): np.array([(np.abs(r[1]["h"]) > e).sum() for r in rows]) for e in args.eps_screen}
    out["mean_conn"] = float(nconn.mean())
    out["kept_frac"] = {e: float(v.mean() / nconn.mean()) for e, v in kept.items()}
    print(f"  <|C|>={nconn.mean():.0f}  kept fraction by eps: " + " ".join(f"{e}:{v:.3f}" for e, v in out["kept_frac"].items()), flush=True)
    for arm in args.arms:
        net = make_net(K, N, arm, args, args.seeds[0])
        P = net.unpack(net.theta0 + 0.02 * np.random.default_rng(5).standard_normal(net.theta0.size))
        amp = amp_ops(net); so = setup_ops(net)
        opsK, wK, opsE, wE, dev = [], [], [], [], []
        wS = {e: [] for e in args.eps_screen}; wT = []
        con = {"ops_contr": 0.0, "ops_enum": 0.0, "max_rel_dev": 0.0, "n": 0}
        floor_share = []
        for t, (occ_n, row) in enumerate(rows):
            st = net.setup(P, occ_n)
            s0 = time.perf_counter()
            rk, cnt = net.ratio_key(P, st, row["rem"], row["add"])
            ek = row["Hnn"] + row["h"] @ rk
            wK.append(time.perf_counter() - s0)
            ok = keyx_ops(net, cnt) + C_GEN * len(row["h"]) + so
            opsK.append(ok)
            floor_share.append((cnt["T"] * (10 + 0.5 * sum(b[3] ** 2 for b in net.blocks) + C_PAIRJ + C_GEN)) / ok)
            opsE.append(len(row["h"]) * (C_GEN + amp) + so)
            if t < args.n_wall:
                occ_j = occ_from_row(occ_n, row["rem"], row["add"])
                s0 = time.perf_counter()
                rx = net.ratio_exact(P, st, row["rem"], row["add"], occ_j)
                ex = row["Hnn"] + row["h"] @ rx
                wE.append(time.perf_counter() - s0)
                dev.append(abs(ex - ek) / max(1.0, abs(ex)))
                for e in args.eps_screen:
                    mk = np.abs(row["h"]) > e
                    s0 = time.perf_counter()
                    _ = row["Hnn"] + row["h"][mk] @ net.ratio_exact(P, st, row["rem"][mk], row["add"][mk], occ_j[mk])
                    wS[e].append(time.perf_counter() - s0)
                s0 = time.perf_counter()
                inU = np.isin(row["code"], U)
                _ = row["Hnn"] + row["h"][inU].sum()
                wT.append(time.perf_counter() - s0)
                if arm == "FJ":
                    occW = [p for p in occ_n if net.win[p]]
                    for (i, j) in list(combinations(occW, 2))[:3]:
                        cls = (row["rem"][:, 0] == i) & (row["rem"][:, 1] == j) & (~net.win[row["add"][:, 0]]) & \
                              (row["add"][:, 1] >= 0) & (~net.win[np.maximum(row["add"][:, 1], 0)])
                        if not cls.any():
                            continue
                        se = float(row["h"][cls] @ rk[cls])
                        sc, oc = contract_ij(net, P, st, ham, i, j, Lc)
                        nk = 4 * net.Hh + M * N * net.Hh + 2 * N ** 3 + M * N * N
                        con["ops_contr"] += oc
                        con["ops_enum"] += nk + cls.sum() * (10 + 0.5 * N * N + C_PAIRJ + C_GEN)
                        con["max_rel_dev"] = max(con["max_rel_dev"], abs(sc - se) / max(abs(se), 1e-12))
                        con["n"] += 1
        a = {"ops_KEYX": float(np.mean(opsK)), "ops_EXACT": float(np.mean(opsE)),
             "ops_amp": amp, "keyx_floor_share": float(np.mean(floor_share)),
             "wall_KEYX": wK, "wall_EXACT": wE, "wall_SCREEN": {str(e): v for e, v in wS.items()}, "wall_TRUNC": wT,
             "KEYX_max_rel_dev": float(max(dev)) if dev else None,
             "ops_SCREEN": {str(e): float(np.mean(kept[str(e)]) * (C_GEN + amp) + so) for e in args.eps_screen},
             "ops_TRUNC": float(nconn.mean() * (C_GEN + C_HASH) + amp + so)}
        if arm == "FJ" and con["n"]:
            a["contraction"] = {"ratio_ops_contr_over_enum": con["ops_contr"] / con["ops_enum"], "max_rel_dev": con["max_rel_dev"],
                                "n_keys_checked": con["n"]}
        out["arms"][arm] = a
        msg = (f"  {arm:5s} ops KEYX/EXACT={a['ops_KEYX']/a['ops_EXACT']:.4f} floor share={a['keyx_floor_share']:.2f} "
               f"wall KEYX/EXACT={np.mean(wK[:len(wE)])/max(np.mean(wE),1e-12):.4f} KEYX-EXACT rel dev={a['KEYX_max_rel_dev']:.1e}")
        if "contraction" in a:
            msg += f" | R4.0 contraction/enumeration ops={a['contraction']['ratio_ops_contr_over_enum']:.2f} (identity dev {a['contraction']['max_rel_dev']:.1e})"
        print(msg, flush=True)
    return out


# ----------------------------------------------------------------------------
# Aggregate: the pre-committed verdict
# ----------------------------------------------------------------------------
def load_results(path):
    A, S = [], []
    for f in sorted(glob.glob(os.path.join(path, "results_*.json"))):
        if "smoke" in os.path.basename(f) and not os.path.basename(path.rstrip("/")).startswith("smoke"):
            pass
        with open(f) as fh:
            d = json.load(fh)
        (A if d.get("part") == "A" else S).append(d)
    return A, S


def verdict(A, S, tau=2e-4):
    # ---- Part A: merge runs per (instance, K)
    inst = {}
    for d in A:
        key = d["instance"]
        for it in d["instances"]:
            e = inst.setdefault(key, {}).setdefault(it["K"], {**{k: v for k, v in it.items() if k != "runs"}, "runs": []})
            e["runs"].extend(it["runs"])
    out = {"tau_mHa": 1e3 * tau, "instances": {}, "routes": {}}
    if not inst or not S:
        out["verdict"] = "INCOMPLETE"
        return out
    ctrl = {"screen_ok": None, "trunc_ok": None, "ss_inc": {}}
    varratio = {}
    acc = {}
    for name, byK in inst.items():
        Kmax = max(byK)
        it = byK[Kmax]
        runs = it["runs"]
        by = {}
        for r in runs:
            by.setdefault(r["seed"], {})[r["arm"]] = r
        dEn = [d["NNBFJ"]["dE_mHa"] for d in by.values() if "NNBFJ" in d]
        prem = bool(it["HF_weight"] < 0.6 and dEn and np.median(dEn) < 1.6)
        info = {"K_max": Kmax, "HF_weight": it["HF_weight"], "median_dE_NNBFJ": float(np.median(dEn)) if dEn else None,
                "premise": prem, "delta_mHa": {}, "var_ratio": {}}
        for arm in ("WIBJ", "CGK", "XSK", "FJ"):
            dl = [d[arm]["dE_mHa"] - d["NNBFJ"]["dE_mHa"] for d in by.values() if arm in d and "NNBFJ" in d]
            vr = [d[arm]["var_exact"] / d["NNBFJ"]["var_exact"] for d in by.values() if arm in d and "NNBFJ" in d]
            if dl:
                info["delta_mHa"][arm] = [float(np.median(dl)), float(min(dl)), float(max(dl))]
                info["var_ratio"][arm] = float(np.median(vr))
                varratio[arm] = max(varratio.get(arm, 0), float(np.median(vr)))
                acc.setdefault(arm, []).append((prem, float(np.median(dl))))
        # controls on NNBFJ
        sb, tb, ssi = {}, {}, {}
        for d in by.values():
            c = d.get("NNBFJ", {}).get("controls")
            if not c:
                continue
            for e, b in c["screen_bias"].items():
                sb.setdefault(e, []).append(abs(b))
            for bu, (m, se) in c["trunc_bias"].items():
                tb.setdefault(bu, []).append(abs(m) + 2 * se)
            for k, g in c["ss_grid"].items():
                ssi.setdefault(k, []).append(g["inc_ratio_B"] if (g["tail_ok_A"] and g["tail_ok_B"]) else float("inf"))
        sok = {e for e, v in sb.items() if max(v) <= tau}
        tok = {bu for bu, v in tb.items() if max(v) <= tau}
        ctrl["screen_ok"] = sok if ctrl["screen_ok"] is None else ctrl["screen_ok"] & sok
        ctrl["trunc_ok"] = tok if ctrl["trunc_ok"] is None else ctrl["trunc_ok"] & tok
        for k, v in ssi.items():
            ctrl["ss_inc"][k] = max(ctrl["ss_inc"].get(k, 0), float(np.median(v)))
        info["screen_admissible"] = sorted(sok, key=float); info["trunc_admissible"] = sorted(tok, key=int)
        out["instances"][name] = info
    eps_star = max(ctrl["screen_ok"], key=float) if ctrl["screen_ok"] else None
    bu_star = min(ctrl["trunc_ok"], key=int) if ctrl["trunc_ok"] else None
    out["controls_transferred"] = {"SCREEN_eps": eps_star, "TRUNC_BU": bu_star,
                                   "SS_inc_ratio": ctrl["ss_inc"]}
    # ---- Part S: per K
    Srows = sorted([it for d in S for it in d["instances"]], key=lambda it: it["K"])
    for route, arm in ROUTE_ARM.items():
        a = acc.get(arm, [])
        if not a:
            out["routes"][route] = {"verdict": "INCOMPLETE"}
            continue
        if any(p and dl > 1e3 * tau for p, dl in a):
            accv = "ACC_FAIL"
        elif all(p for p, _ in a) and all(dl <= 1e3 * tau for _, dl in a):
            accv = "ACC_OK"
        else:
            accv = "ACC_AMBIG"
        byK = {}
        for it in Srows:
            if arm not in it["arms"] or "NNBFJ" not in it["arms"]:
                continue
            R = it["arms"][arm]; Cn = it["arms"]["NNBFJ"]
            cands = {"EXACT": Cn["ops_EXACT"]}
            wc = {"EXACT": np.array(Cn["wall_EXACT"])}
            if eps_star is not None and eps_star in Cn["ops_SCREEN"]:
                cands["SCREEN"] = Cn["ops_SCREEN"][eps_star]; wc["SCREEN"] = np.array(Cn["wall_SCREEN"][eps_star])
            for k, inc in ctrl["ss_inc"].items():
                e, Kd = k.split(",")
                if np.isfinite(inc) and e in Cn["ops_SCREEN"]:
                    c = (Cn["ops_SCREEN"][e] + int(Kd) * Cn["ops_amp"]) * (1 + inc)
                    if c < cands.get("SS", np.inf):
                        cands["SS"] = c
                        wc["SS"] = (np.array(Cn["wall_SCREEN"][e]) + int(Kd) * np.mean(Cn["wall_EXACT"]) /
                                    max(1.0, it["mean_conn"])) * (1 + inc)
            if bu_star is not None:
                cands["TRUNC"] = Cn["ops_TRUNC"]; wc["TRUNC"] = np.array(Cn["wall_TRUNC"])
            best = min(cands, key=cands.get)
            v = varratio.get(arm, 1.0)
            Mop = R["ops_KEYX"] * v / cands[best]
            nW = len(wc[best])
            wr = np.array(R["wall_KEYX"][:nW]) * v
            Mw = float(wr.mean() / wc[best].mean())
            ci = boot_ratio_ci(wr, wc[best])
            byK[it["K"]] = {"M_op": float(Mop), "ctrl": best, "M_wall": Mw, "M_wall_CI": ci,
                            "ops_KEYX_over_EXACT": R["ops_KEYX"] / R["ops_EXACT"]}
        if len(byK) < 2:
            out["routes"][route] = {"verdict": "INCOMPLETE", "accuracy": accv, "byK": byK}
            continue
        Ks = sorted(byK); kmin, kmax = Ks[0], Ks[-1]
        m = byK[kmax]
        veto = m["M_wall_CI"][0] > 1
        if accv == "ACC_FAIL" or m["M_op"] >= 1 or veto:
            v = "FAIL"
        elif accv == "ACC_OK" and m["M_op"] <= 0.5 and m["M_op"] < byK[kmin]["M_op"]:
            v = "PASS"
        else:
            v = "AMBIGUOUS"
        out["routes"][route] = {"verdict": v, "accuracy": accv, "wall_veto": bool(veto), "var_ratio": varratio.get(arm),
                                "byK": byK}
    return out


def aggregate(path, tau=2e-4):
    A, S = load_results(path)
    A = [d for d in A if not d.get("smoke")]; S = [d for d in S if not d.get("smoke")]
    out = verdict(A, S, tau)
    print(json.dumps(out, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))
    return out


# ----------------------------------------------------------------------------
# Smoke checks
# ----------------------------------------------------------------------------
def smoke_checks():
    t0 = time.time()
    rng = np.random.default_rng(3)
    ok = True
    # (1) integrals (as round 3) and larger basis sanity
    X = np.array([[0, 0, 0], [1.4, 0, 0.]])
    S, h, G, e, _ = integrals(X, H631GSS)
    C, E, _ = rhf(S, h, G, 1)
    S2, h2, G2, e2, _ = integrals(X, make_basis(4, 3))
    C2, E2, _ = rhf(S2, h2, G2, 1)
    print(f"H2 R=1.4: E_RHF 6-31G**={E+e:.6f} (lit. -1.13128); (ns,np)=(4,3) E_RHF={E2+e2:.6f} (must be lower; HF limit -1.13363)")
    ok &= abs(E + e + 1.13128) < 2e-5 and (E2 + e2) < (E + e) and (E2 + e2) > -1.13363 - 1e-5
    # (2) spatial-ERI rows == spin-orbital rows (round-3 code), elementwise
    N, K = 4, 6
    hh, gg, en, Er, eps = build_molecule(N, 3.0, 0.25, 1, H631GSS, K, "valence-p")
    hamS = SpatialHam(hh, gg, en, K, N); hamQ = SQHam(hh, gg, en, K, N)
    codes, occ, oi = build_space(K, 2, 2)
    err = 0.0
    for n in range(0, len(codes), 37):
        a, b = hamS.row(codes[n], oi[n]), hamQ.row(codes[n], oi[n])
        ia, ib = np.argsort(a["code"]), np.argsort(b["code"])
        assert np.array_equal(a["code"][ia], b["code"][ib])
        err = max(err, np.abs(a["h"][ia] - b["h"][ib]).max(), abs(a["Hnn"] - b["Hnn"]))
    print(f"  spatial vs spin-orbital Slater-Condon rows: max |diff| = {err:.1e}")
    ok &= err < 1e-12
    H = build_hamiltonian(hamS, codes, oi)
    # (3) per arm: keyed ratios == exact ratios == full-vector ratios; gradient
    class A_: pass
    args = A_(); args.hidden, args.hj, args.window, args.groups = 5, 4, 4, 2
    M = 2 * K
    for arm in ARMS:
        net = make_net(K, N, arm, args, 0)
        th = net.theta0 + 0.3 * rng.standard_normal(net.theta0.size)
        P = net.unpack(th)
        psi, _ = net.amplitudes(th, occ, oi)
        e1 = e2_ = 0.0
        for n in (5, 17, 101):
            row = hamS.row(codes[n], oi[n]); j = np.searchsorted(codes, row["code"])
            st = net.setup(P, oi[n])
            rx = net.ratio_exact(P, st, row["rem"], row["add"], decode(row["code"], M, N))
            rk, cnt = net.ratio_key(P, st, row["rem"], row["add"])
            e1 = max(e1, np.abs(rx - psi[j] / psi[n]).max()); e2_ = max(e2_, np.abs(rk - psi[j] / psi[n]).max())
        idx = rng.choice(th.size, 10, replace=False); gn = []
        _, ga = net.energy_grad(th, H, occ, oi)
        for i in idx:
            dd = np.zeros_like(th); dd[i] = 1e-6
            gn.append((net.energy_grad(th + dd, H, occ, oi)[0] - net.energy_grad(th - dd, H, occ, oi)[0]) / 2e-6)
        gerr = float(np.max(np.abs(ga[idx] - gn)) / max(np.max(np.abs(gn)), 1e-12))
        print(f"  {arm:5s}: exact-ratio err={e1:.1e} keyed-ratio err={e2_:.1e} grad rel err={gerr:.1e} "
              f"blocks={len(net.blocks)} features={[b[4].shape[0] for b in net.blocks]} jastrow features={net.FJ.shape[0]}")
        ok &= e1 < 1e-9 and e2_ < 1e-9 and gerr < 1e-5
    # (4) R4.0 contraction identity on the FJ arm (n with two external electrons
    #     too, so spectator dressing is exercised)
    net = make_net(K, N, "FJ", args, 1)
    P = net.unpack(net.theta0 + 0.3 * rng.standard_normal(net.theta0.size))
    Lc = pivoted_cholesky(gg, 1e-12)
    rec = np.abs(np.einsum("gpq,grs->pqrs", Lc, Lc) - gg).max()
    dev = 0.0; ncheck = 0
    for n in range(0, len(codes), 23):
        row = hamS.row(codes[n], oi[n]); st = net.setup(P, oi[n])
        rk, _ = net.ratio_key(P, st, row["rem"], row["add"])
        occW = [p for p in oi[n] if net.win[p]]
        for (i, j) in combinations(occW, 2):
            cls = (row["rem"][:, 0] == i) & (row["rem"][:, 1] == j) & (~net.win[row["add"][:, 0]]) & \
                  (row["add"][:, 1] >= 0) & (~net.win[np.maximum(row["add"][:, 1], 0)])
            if not cls.any():
                continue
            se = row["h"][cls] @ rk[cls]
            sc, _ = contract_ij(net, P, st, hamS, i, j, Lc)
            dev = max(dev, abs(sc - se) / max(abs(se), 1e-10)); ncheck += 1
    print(f"  R4.0 Cholesky contraction == enumerated class sum: max rel dev = {dev:.1e} over {ncheck} keys "
          f"(Cholesky reconstruction err {rec:.1e})")
    ok &= dev < 1e-8 and ncheck > 0
    # (5) TRUNC bias formula (sparse, used in Part A) == direct per-sample row sums
    net = make_net(K, N, "NNBFJ", args, 2)
    psi, _ = net.amplitudes(net.theta0 + 0.3 * rng.standard_normal(net.theta0.size), occ, oi)
    batch = rng.choice(len(codes), size=40, p=psi ** 2 / (psi @ psi))
    inU = np.zeros(len(codes), bool); inU[batch] = True
    Hoff = (H - sp.diags(H.diagonal())).tocsr()
    e_sp = -(Hoff[batch][:, ~inU] @ psi[~inU]) / psi[batch]
    e_dir = []
    for k in batch:
        row = hamS.row(codes[k], oi[k]); j = np.searchsorted(codes, row["code"])
        full = row["h"] @ psi[j] / psi[k]
        tr = row["h"][inU[j]] @ psi[j[inU[j]]] / psi[k]
        e_dir.append(tr - full)
    terr = float(np.abs(e_sp - np.array(e_dir)).max())
    print(f"  TRUNC per-sample truncation error, sparse formula vs direct rows: max |diff| = {terr:.1e}")
    ok &= terr < 1e-10
    print(f"  ({time.time()-t0:.1f}s)")
    return bool(ok)


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        aggregate(args.aggregate)
        return
    if args.smoke:
        ok = smoke_checks()
        print(f"SMOKE CHECKS {'PASS' if ok else 'FAIL'}", flush=True)
        sdir = os.path.join(HERE, "smoke_results")
        os.makedirs(sdir, exist_ok=True)
        # tiny Part A: H4 distorted chain, R = 4.4 (HF weight 0.49), K = 5, 6
        a = parse_args(["--part", "A", "--n-atoms", "4", "--bond-length", "4.4", "--n-active", "5", "6",
                        "--window", "4", "--hidden", "8", "--hj", "4", "--maxiter", "150", "--seeds", "0", "1",
                        "--n-samples", "200", "--bu-grid", "50", "200", "--k-grid", "1", "8",
                        "--eps-screen", "1e-4", "1e-3", "1e-2", "--out", os.path.join(sdir, "results_A_smoke.json")])
        resA = {"part": "A", "smoke": True, "instance": "H4smoke", "args": {k: v for k, v in vars(a).items()},
                "instances": [run_part_A_instance(K, a) for K in a.n_active]}
        with open(a.out, "w") as fh:
            json.dump(resA, fh, indent=1, default=float)
        # tiny Part S: H4 distorted chain, bases (2,1) and (3,2)
        s = parse_args(["--part", "S", "--n-atoms", "4", "--basis-series", "2,1", "3,2", "--window", "4",
                        "--hidden", "8", "--hj", "4", "--n-samples", "30", "--n-wall", "4",
                        "--eps-screen", "1e-4", "1e-3", "1e-2", "--k-grid", "1", "8",
                        "--out", os.path.join(sdir, "results_S_smoke.json")])
        resS = {"part": "S", "smoke": True, "args": {k: v for k, v in vars(s).items()},
                "instances": [run_part_S_basis(ns, npp, s) for ns, npp in s.basis_series]}
        with open(s.out, "w") as fh:
            json.dump(resS, fh, indent=1, default=float)
        print("[smoke-scale verdict logic, H4 only, not decisive]")
        v = verdict([resA], [resS], 2e-4)
        print(json.dumps({r: {"verdict": d.get("verdict"), "accuracy": d.get("accuracy"),
                              "M_op_by_K": {K: round(x["M_op"], 3) for K, x in d.get("byK", {}).items()}}
                          for r, d in v["routes"].items()}))
        print("controls transferred:", json.dumps({k: v["controls_transferred"][k] for k in ("SCREEN_eps", "TRUNC_BU")}))
        return
    res = {"part": args.part, "args": {k: v for k, v in vars(args).items()}, "instances": []}
    if args.part == "A":
        res["instance"] = f"H{args.n_atoms}_R{args.bond_length}_d{args.distort}_g{args.geom_seed}"
        for K in args.n_active:
            res["instances"].append(run_part_A_instance(K, args))
            with open(args.out, "w") as fh:
                json.dump(res, fh, indent=1, default=float)
    else:
        for ns, npp in args.basis_series:
            res["instances"].append(run_part_S_basis(ns, npp, args))
            with open(args.out, "w") as fh:
                json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
