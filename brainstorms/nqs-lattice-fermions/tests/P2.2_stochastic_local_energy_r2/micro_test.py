#!/usr/bin/env python3
"""
P2.2 round-2 decisive test: sign-stratified (positivised) and perturbative-tail
importance-sampled local energies versus exact enumeration, screening,
surrogate screening, surrogate completion, heat-bath and semistochastic local
energies, for a trained neural-network-backflow (NNBF) state of a stretched
hydrogen cluster, as the orbital basis grows at fixed electron number.

Why this test exists
--------------------
Round 1 (frozen-backflow additive control variate) died on H6/H8 STO-3G,
where |C(n)| (number of connected configurations) is ~100, so the O(M^4)
cost of exact enumeration never appears, and the surrogate cost kappa was
assumed (1e-3) rather than measured.  This test:
  * grows the basis at fixed N (H4: N = 4 electrons) with an s-Gaussian basis
    built here (STO-3G 1s + extra uncontracted s shells per atom), so
    K = 4*ns spatial orbitals, M = 2K spin-orbitals, |C| ~ N^2 (M-N)^2 / 4
    reaching ~2500 at ns = 6;
  * MEASURES kappa (cost of one cheap surrogate term, and of one Hamiltonian
    element generation, in units of one NNBF forward pass) on the instance
    itself, with batched numpy implementations of both;
  * compares two round-2 routes against the best of a strong control set.

Routes under test (all unbiased)
--------------------------------
R2.0 SCVSS   Round-1 frozen-backflow additive control variate, revived under
             the corrected test (larger M, measured kappa): exact |y_FB| > eps,
             sum_tail y_FB exactly, K draws of the residual x - y_FB.  Judged
             against the controls WITHOUT itself; for R2.1-R2.5 it is a control.
R2.1 POS_FB  Positivisation (Owen & Zhou 2000) with the frozen-backflow
             surrogate y = H_{nn'} * rt(n'), rt = det Phi(n)[occ n'] / psi(n):
             terms with |y| > eps evaluated exactly; the tail is split into
             the strata sgn(y) = +1 and sgn(y) = -1, each sampled with its own
             proposal p_s ~ alpha |y| + (1-alpha) |H| (stratified, integer
             allocation K_s ~ stratum mass, at least 1 per non-empty stratum).
             POS_HYB: strata from sgn(y_FB), magnitudes/proposal from y_EN.
R2.2 PT2IS   P/Q-partitioned local energy with a perturbative (Epstein-Nesbet)
             tail proposal: P = {n' : |y_EN| > eps or H_{n'n'} < E + Delta_P}
             exact; Q sampled with p ~ alpha |y_EN| + (1-alpha) |H|, where
             y_EN = H_{nn'}^2 / min(E - H_{n'n'}, -delta).  Hamiltonian-only
             surrogate (architecture-agnostic), O(1) per term.

R2.3 LRX     Mutation of round-1's failed component (the frozen surrogate):
             rt is replaced by a ratio in which the two NEW occupied rows of
             Phi(n') are formed exactly (exact hidden update; O(N h) each) and
             enter via a 2x2 minor of Phi_new A^{-1}, while the response of the
             N-2 unchanged rows is linearised, exp(sum_k Tk . dhid), with
             Tk[k,h] = sum_l (A^{-1})_{lk} dPhi[occ_k, l]/dhid_h.  O(N h) per
             term versus O(N^2 h + N^3) for an exact amplitude.  Used as an
             additive control variate (SCV_LRX) and as a positivised proposal
             (POS_LRX); R2.3 = best of the two.

R2.4 MLMC    Multilevel Monte Carlo over surrogate fidelity (Giles): exact
             set D = {|y_FB| > eps}; on the tail, sum y_FB exactly (cheap
             level), the correction y_LRX - y_FB with K1 draws (LRX level) and
             the correction x - y_LRX with K2 draws (exact amplitudes), both
             with p ~ alpha |y_FB| + (1-alpha) |H|.

R2.5 ADA     Cost-aware cross-sample allocation (active probing): the
             round-1 additive frozen-backflow control variate, but each sample
             n gets K_n = max(1, round(K Z_n / E_pi[Z])) residual draws, with
             Z_n = sum_tail |y_FB| known before any amplitude is evaluated
             (SCV_ADA).  MLMC_ADA combines R2.4 and R2.5 (reported as
             "R2.4+R2.5").

Controls (the route must beat the BEST of these)
------------------------------------------------
EXACT      full enumeration (zero variance at the eigenstate).
SCREEN_H   keep |H_{nn'}| > eps (biased; admissible only if |bias| <= tol;
           charged generation cost only for kept terms -- generous).
SCREEN_FB  keep |y_FB| > eps (biased, admissible if |bias| <= tol).
SCREEN_EN  keep P-space (|y_EN| > eps or low H_{n'n'}) (biased).
COMP_FB    SCREEN_FB plus deterministic completion sum_tail y_FB (biased).
COMP_EN    SCREEN_EN plus deterministic completion sum_tail y_EN (biased;
           a per-sample EN-PT2 tail correction).
HB         heat-bath K draws p ~ |H| (unbiased; zero enumeration cost).
SS         semistochastic: exact |H| > eps + heat-bath tail (unbiased).
SCVSS      round-1 additive frozen-backflow control variate (unbiased).
SCREEN_LRX, COMP_LRX  surrogate screening / completion with the R2.3
           surrogate (biased; admissible if |bias| <= tol).
Diagnostics (not in the verdict): ISFB (single proposal ~|y_FB|),
ORACLE_SIGN (strata from the TRUE sign, |y_EN| magnitudes), ORACLE_IS
(single proposal ~|x|, the single-proposal floor).

Cost model (NNBF forward-pass units, per sample n)
--------------------------------------------------
1 (own amplitude) + #amplitudes of connected n' evaluated + kH * #Hamiltonian
elements generated + kFB * #frozen-backflow terms + kEN * #EN terms
+ kLRX * #LRX terms + kSetup (+ kSetupLRX) per-sample setups where used.
All kappas are MEASURED (see --kappa-scale): decisive values at chemistry
scale, instance values (N = 4) as a sensitivity.
C_X = (Var_pi(E_L^X) + E_pi[tau_X^2(n)] / K-allocation) * cost_X; every method
is minimised over K (draws), eps (thresholds), alpha (defensive weight) and
Delta_P.  Var_pi(E_L) is exact (full space); E_pi[tau^2], costs and screening
bias/variance shifts use common rows for every method: all rows if
dim <= --exact-rows-max, else top-weight rows plus defensive draws.
All tau^2 are analytic per row (no inner Monte Carlo noise).

Verdict (per route X in {R2.0, ..., R2.5, R2.4+R2.5}, per geometry, final snapshot)
------------------------------------------------------------------------
M_X(ns) = C_X / min(controls).  With ns_max the largest basis:
  PASS       M_X(ns_max) <= 0.5  and  M_X(ns_max) < M_X(ns_min)
             and dE(ns_max) <= --de-max-mHa
  FAIL       M_X(ns_max) >= 1.0
  AMBIGUOUS  otherwise.  Pre-committed action: rerun with --ns 6 7 and
             --hidden 64; PASS iff M_X(7) <= min(M_X(6), 0.5), else KILL.

Command-line options
--------------------
--n-atoms INT            Number of H atoms (even, >= 2). Default 4.
--ns INT [INT ...]       Basis sizes: s functions per atom (1 = STO-3G; each
                         extra adds one uncontracted s shell; 1..9).
                         K = n_atoms*ns (minus any linearly dependent
                         combinations). Default: 2 3 4 5 6.
--bond-length FLOAT      Nearest-neighbour H-H distance in bohr (> 0).
                         Default 2.8 (stretched, strongly multireference).
--geometry {chain,ring}  Linear chain or regular polygon. Default ring.
--hidden INT             NNBF hidden width (>= 1). Default 64.
--maxiter INT            L-BFGS iterations on the exact energy (>= 1).
                         Default 2000.
--snapshots INT [...]    Extra iterations to analyse (final always). Default 200.
--init-scale FLOAT       Std of random init of W1, W2 (>= 0). Default 0.05.
--seed INT               RNG seed. Default 1234.
--n-rows INT             When dim > --exact-rows-max: at most this many
                         top-weight rows taken exactly, plus this many draws
                         from the remaining rows with q ~ sqrt(pi) and
                         importance weights (>= 10). Default 3000.
--exact-rows-max INT     Use every configuration (exact pi-average) if
                         dim <= this. Default 15000.
--top-tail FLOAT         Mass left outside the exact top-weight rows, in
                         (0,1). Default 1e-3.
--kappa-scale N M H T    DECISIVE cost model: the kappas (cost of one
                         surrogate / Hamiltonian term and of one per-sample
                         setup, in units of one exact NNBF amplitude) are
                         measured with the same kernels on a chemistry-scale
                         NNBF with N electrons, M spin-orbitals, hidden width H,
                         timing T random connected terms (integers >= 2).
                         Default 20 80 256 20000.  The kappas of the H4 test
                         instance itself (N=4, its M and --hidden) are also
                         measured and every metric is re-reported under them
                         as a sensitivity.
--bias-tol FLOAT         Max |bias| (Ha) for biased arms. Default 1.6e-4.
--de-max-mHa FLOAT       Max NNBF energy error (mHa) for a PASS. Default 10.
--en-delta FLOAT         Denominator clip delta (Ha, > 0) for y_EN. Default 0.05.
--alphas F [F ...]       Defensive weights in [0,1]. Default 0.5 0.9.
--delta-p F [F ...]      P-space energy windows Delta_P (Ha) for EN arms.
                         Default 0.0 0.5 1.0.
--k-grid INT [...]       Draw counts. Default 1 2 4 8 16 32 64 128 256 512.
--n-eps INT              Log-spaced thresholds in [1e-8, 1] (>= 2). Default 33.
--kappa-override F F F F F F  Use given kH kFB kEN kLRX kSetup kSetupLRX
                         instead of the measured decisive kappas (sensitivity
                         only; never for a reported verdict).
--out PATH               JSON output. Default results.json next to script.
--aggregate DIR          Do not compute: read every results_*_*.json in DIR
                         (the Slurm array tasks) and print the global verdict:
                         a route PASSES only if it PASSES on every task, FAILS
                         if it FAILS on any task, else AMBIGUOUS.
--smoke                  Fast login-node run (< 30 s): checks the Slater-Condon
                         builder against an independent second-quantized
                         builder, the 2x2 frozen-ratio formula against direct
                         determinants, the EN diagonal update against direct
                         diagonals, the LRX error order, the NNBF gradient, then runs ns = 1 2 with
                         maxiter 150, hidden 8, and times the decisive
                         kappas at N=8, M=32, hidden 64 (4000 terms).

Example
-------
python3 micro_test.py --smoke
python3 micro_test.py --ns 2 3 4 5 6 --geometry ring --bond-length 2.8
"""
import argparse
import json
import os
import time
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P2.2 round-2 local-energy test")
    ap.add_argument("--n-atoms", type=int, default=4)
    ap.add_argument("--ns", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--bond-length", type=float, default=2.8)
    ap.add_argument("--geometry", choices=["chain", "ring"], default="ring")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--snapshots", type=int, nargs="*", default=[200])
    ap.add_argument("--init-scale", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-rows", type=int, default=3000)
    ap.add_argument("--exact-rows-max", type=int, default=15000)
    ap.add_argument("--top-tail", type=float, default=1e-3)
    ap.add_argument("--kappa-scale", type=int, nargs=4, default=[20, 80, 256, 20000])
    ap.add_argument("--bias-tol", type=float, default=1.6e-4)
    ap.add_argument("--de-max-mHa", type=float, default=10.0)
    ap.add_argument("--en-delta", type=float, default=0.05)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.9])
    ap.add_argument("--delta-p", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--k-grid", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--n-eps", type=int, default=33)
    ap.add_argument("--kappa-override", type=float, nargs=6, default=None)
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "results.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", type=str, default=None)
    a = ap.parse_args(argv)
    if a.smoke:
        a.ns, a.maxiter, a.snapshots, a.hidden, a.n_rows = [1, 2], 150, [], 8, 400
        a.n_eps, a.k_grid = 17, [1, 4, 16, 64, 256]
        a.kappa_scale = [8, 32, 64, 4000]
        if a.out == os.path.join(HERE, "results.json"):
            a.out = os.path.join(HERE, "results_smoke.json")
    if a.n_atoms < 2 or a.n_atoms % 2:
        ap.error("--n-atoms must be even and >= 2")
    if any(s < 1 or s > 1 + len(EXTRA_EXPS) for s in a.ns):
        ap.error(f"--ns entries must be in 1..{1+len(EXTRA_EXPS)}")
    if a.bond_length <= 0 or a.en_delta <= 0:
        ap.error("--bond-length and --en-delta must be > 0")
    if a.hidden < 1 or a.maxiter < 1 or a.n_eps < 2 or a.n_rows < 10:
        ap.error("--hidden, --maxiter >= 1, --n-eps >= 2, --n-rows >= 10")
    if min(a.kappa_scale) < 2 or a.kappa_scale[0] >= a.kappa_scale[1]:
        ap.error("--kappa-scale N M H T: integers >= 2 with N < M")
    if any(not (0 <= x <= 1) for x in a.alphas) or min(a.k_grid) < 1:
        ap.error("--alphas in [0,1], --k-grid >= 1")
    return a


# ----------------------------------------------------------------------------
# s-Gaussian basis and integrals (numpy only)
# ----------------------------------------------------------------------------
STO3G_A = np.array([3.42525091, 0.62391373, 0.16885540])
STO3G_D = np.array([0.15432897, 0.53532814, 0.44463454])
# Extra uncontracted s exponents, added in this (nested) order.
EXTRA_EXPS = [0.08, 1.4, 0.3, 4.5, 0.035, 12.0, 0.75, 0.15]


def boys0(t):
    t = np.asarray(t, dtype=float)
    out = np.ones_like(t)
    m = t > 1e-12
    st = np.sqrt(t[m])
    out[m] = 0.5 * np.sqrt(np.pi) * erf(st) / st
    out[~m] = 1.0 - t[~m] / 3.0
    return out


def geometry(n_atoms, R, kind):
    if kind == "chain":
        return np.array([[i * R, 0.0, 0.0] for i in range(n_atoms)])
    rad = R / (2 * np.sin(np.pi / n_atoms))
    ang = 2 * np.pi * np.arange(n_atoms) / n_atoms
    return np.stack([rad * np.cos(ang), rad * np.sin(ang), 0 * ang], 1)


def ao_integrals(X, ns):
    """Contracted s basis: STO-3G 1s + (ns-1) extra uncontracted s per atom."""
    prim_a, prim_c, prim_C, owner = [], [], [], []
    k = 0
    for A in X:
        shells = [(STO3G_A, STO3G_D)] + [(np.array([e]), np.array([1.0])) for e in EXTRA_EXPS[:ns - 1]]
        for ex, d in shells:
            for e, dd in zip(ex, d):
                prim_a.append(e); prim_c.append(dd * (2 * e / np.pi) ** 0.75)
                prim_C.append(A); owner.append(k)
            k += 1
    a = np.array(prim_a); C = np.array(prim_C); own = np.array(owner)
    P = len(a); K = k
    T = np.zeros((P, K)); T[np.arange(P), own] = prim_c
    A_, B_ = a[:, None], a[None, :]
    p = A_ + B_
    d2 = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    E = np.exp(-A_ * B_ / p * d2)
    S_p = (np.pi / p) ** 1.5 * E
    T_p = A_ * B_ / p * (3 - 2 * A_ * B_ / p * d2) * S_p
    Pc = (A_[..., None] * C[:, None, :] + B_[..., None] * C[None, :, :]) / p[..., None]
    V_p = np.zeros_like(S_p)
    for Cn in X:
        V_p += -2 * np.pi / p * E * boys0(p * ((Pc - Cn) ** 2).sum(-1))
    pq = p[:, :, None, None] * p[None, None, :, :]
    ps = p[:, :, None, None] + p[None, None, :, :]
    dPQ = ((Pc[:, :, None, None, :] - Pc[None, None, :, :, :]) ** 2).sum(-1)
    G_p = 2 * np.pi ** 2.5 / (pq * np.sqrt(ps)) * E[:, :, None, None] * E[None, None] * boys0(pq / ps * dPQ)
    S = T.T @ S_p @ T
    nrm = 1 / np.sqrt(np.diag(S))
    T = T * nrm[None, :]
    S = T.T @ S_p @ T
    h = T.T @ (T_p + V_p) @ T
    G = np.einsum("ijkl,ia,jb,kc,ld->abcd", G_p, T, T, T, T, optimize=True)
    enuc = sum(1.0 / np.linalg.norm(X[i] - X[j]) for i in range(len(X)) for j in range(i))
    return S, h, G, enuc


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


# Independent reference builder (round-1 E_pq construction), for --smoke only
def epq_coo(codes, occ, K, p, q):
    rows, cols, vals = [], [], []
    for s in (0, 1):
        P, Q = p + s * K, q + s * K
        if P == Q:
            i = np.nonzero(occ[:, Q])[0]
            rows.append(i); cols.append(i); vals.append(np.ones(len(i)))
            continue
        i = np.nonzero((occ[:, Q] == 1) & (occ[:, P] == 0))[0]
        if len(i) == 0:
            continue
        o = occ[i].astype(np.int64)
        s1 = o[:, :Q].sum(1); o[:, Q] = 0; s2 = o[:, :P].sum(1)
        newc = codes[i] - (1 << Q) + (1 << P)
        j = np.searchsorted(codes, newc)
        rows.append(j); cols.append(i); vals.append((-1.0) ** (s1 + s2))
    if not rows:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def build_hamiltonian_ref(h, g, enuc, codes, occ, K):
    """Round-1 builder: H = sum h'_pq E_pq + 1/2 sum g_pqrs E_pq E_rs."""
    D = len(codes)
    E = {(p, q): epq_coo(codes, occ, K, p, q) for p in range(K) for q in range(K)}
    allr = np.concatenate([E[k][0] for k in E]); allc = np.concatenate([E[k][1] for k in E])
    allv = np.concatenate([E[k][2] for k in E])
    lab_p = np.concatenate([np.full(len(E[k][0]), k[0]) for k in E])
    lab_q = np.concatenate([np.full(len(E[k][0]), k[1]) for k in E])
    hp = h - 0.5 * np.einsum("prrq->pq", g)
    H = sp.coo_matrix((hp[lab_p, lab_q] * allv, (allr, allc)), shape=(D, D)).tocsr()
    for p in range(K):
        for q in range(K):
            r, c, v = E[p, q]
            if len(r) == 0:
                continue
            Epq = sp.csr_matrix((v, (r, c)), shape=(D, D))
            Cpq = sp.coo_matrix((g[p, q, lab_p, lab_q] * allv, (allr, allc)), shape=(D, D)).tocsr()
            H = H + 0.5 * (Epq @ Cpq)
    return (H + enuc * sp.identity(D, format="csr")).tocsr()


# ----------------------------------------------------------------------------
# NNBF ansatz
# ----------------------------------------------------------------------------
class NNBF:
    def __init__(self, Morb, N, hidden, Phi0, scale, rng):
        self.M, self.N, self.Hh = Morb, N, hidden
        self.shapes = [("Phi0", (Morb, N)), ("W2", (Morb * N, hidden)),
                       ("W1", (hidden, Morb)), ("b1", (hidden,))]
        init = {"Phi0": Phi0, "W2": scale * rng.standard_normal((Morb * N, hidden)),
                "W1": scale * rng.standard_normal((hidden, Morb)), "b1": np.zeros(hidden)}
        self.theta0 = np.concatenate([init[k].ravel() for k, _ in self.shapes])

    def unpack(self, th):
        out, i = {}, 0
        for k, s in self.shapes:
            n = int(np.prod(s)); out[k] = th[i:i + n].reshape(s); i += n
        return out

    def amplitudes(self, th, occ, occ_idx):
        P = self.unpack(th)
        x = 2.0 * occ - 1.0
        hid = np.tanh(x @ P["W1"].T + P["b1"])
        Phi = P["Phi0"][None] + (hid @ P["W2"].T).reshape(-1, self.M, self.N)
        A = np.take_along_axis(Phi, occ_idx[:, :, None], axis=1)
        return np.linalg.det(A), (Phi, A, hid, x, P)

    def energy_grad(self, th, H, occ, occ_idx):
        psi, (Phi, A, hid, x, P) = self.amplitudes(th, occ, occ_idx)
        nrm = psi @ psi
        Hpsi = H @ psi
        E = psi @ Hpsi / nrm
        c = 2 * (Hpsi - E * psi) / nrm
        Ainv = np.linalg.inv(A)
        gA = (c * psi)[:, None, None] * np.transpose(Ainv, (0, 2, 1))
        gPhi = np.zeros_like(Phi)
        np.put_along_axis(gPhi, occ_idx[:, :, None], gA, axis=1)
        gflat = gPhi.reshape(len(psi), -1)
        g = {"Phi0": gPhi.sum(0), "W2": gflat.T @ hid}
        dz = (gflat @ P["W2"]) * (1 - hid ** 2)
        g["W1"] = dz.T @ x
        g["b1"] = dz.sum(0)
        return E, np.concatenate([g[k].ravel() for k, _ in self.shapes])

    # --- fast per-sample kernels used for timing (kappa) -------------------
    def setup_one(self, P, occ_n):
        """Hidden pre-activation, Phi(n), A^{-1}, G = Phi(n) A^{-1} for one sample."""
        x = 2.0 * np.isin(np.arange(self.M), occ_n) - 1.0
        z = P["W1"] @ x + P["b1"]
        Phi = P["Phi0"] + (P["W2"] @ np.tanh(z)).reshape(self.M, self.N)
        Ainv = np.linalg.inv(Phi[occ_n])
        return z, Phi, Phi @ Ainv

    def forward_connected(self, P, W2r, z, occ_j, rem, add):
        """Exact amplitudes psi(n') for a batch of connected n' (incremental
        hidden update; only occupied rows of Phi(n') are formed)."""
        W1T = P["W1"].T
        dz = 2 * (W1T[add[:, 0]] - W1T[rem[:, 0]])
        d2 = add[:, 1] >= 0
        dz[d2] += 2 * (W1T[add[d2, 1]] - W1T[rem[d2, 1]])
        hid = np.tanh(z[None] + dz)
        A = P["Phi0"][occ_j] + np.einsum("bknh,bh->bkn", W2r[occ_j], hid)
        return np.linalg.det(A)


def frozen_ratio(G, occ_n, occ_j, rem, add):
    """rt(n') = det(Phi(n)[occ n']) / det(Phi(n)[occ n]) via 2x2 Green's-function
    minors (O(1) per term) times the row-sorting parity (O(N^2), N small)."""
    pos = np.full(G.shape[0], -1); pos[occ_n] = np.arange(len(occ_n))
    ki = pos[rem[:, 0]]; a = add[:, 0]
    r = G[a, ki].copy()
    d2 = rem[:, 1] >= 0
    kj = pos[rem[d2, 1]]; b = add[d2, 1]
    r[d2] = G[a[d2], ki[d2]] * G[b, kj] - G[a[d2], kj] * G[b, ki[d2]]
    # parity of sorting the in-place-replaced row list
    L = np.repeat(occ_n[None], len(rem), 0)
    L[np.arange(len(rem)), ki] = a
    L[np.nonzero(d2)[0], kj] = b
    inv = np.zeros(len(rem), np.int64)
    N = len(occ_n)
    for p_ in range(N):
        for q_ in range(p_ + 1, N):
            inv += L[:, p_] > L[:, q_]
    return r * (1 - 2 * (inv & 1))


def lrx_setup(W2r, Ainv, occ_n):
    """Z[m, c, h] = sum_l W2r[occ_n[m], l, h] (A^{-1})_{l c}: response of row m
    of Phi(n) A^{-1} to hid_h.  One-off O(N^3 h) per sample."""
    return np.einsum("mlh,lc->mch", W2r[occ_n], Ainv)


def lrx_ratio(P, W2r, z, Ainv, Z, occ_n, rem, add):
    """R2.3 surrogate ratio psi(n')/psi(n) for single/double excitations.
    Let B = Phi(n)[occ n'] with the NEW rows a, b replaced by the exact rows
    of Phi(n') (exact hidden update, O(N h) each); det B / det A is a 2x2 minor
    of G_j = B A^{-1}.  The response Delta of the N-2 unchanged rows is
    linearised: log det(B + Delta) - log det B ~ tr(G_j^{-1} Delta A^{-1}),
    with G_j^{-1} the exact rank-2 inverse.  O(N h) per term versus
    O(N^2 h + N^3) for an exact amplitude.  Linear (adjugate) form, so the
    error is O(|Delta|^2) also near nodes of B."""
    N = len(occ_n)
    M = W2r.shape[0]
    pos = np.full(M, -1); pos[occ_n] = np.arange(N)
    I, J, A, B = rem[:, 0], rem[:, 1], add[:, 0], add[:, 1]
    d2 = J >= 0
    nb = len(I); bi = np.arange(nb)
    W1T = P["W1"].T
    dz = 2 * (W1T[A] - W1T[I])
    dz[d2] += 2 * (W1T[B[d2]] - W1T[J[d2]])
    hj = np.tanh(z[None] + dz)
    dh = hj - np.tanh(z)[None]
    ki = pos[I]
    Jx = np.where(d2, J, I); Bx = np.where(d2, B, A)
    kj = pos[Jx]
    ga = (P["Phi0"][A] + np.einsum("bnh,bh->bn", W2r[A], hj)) @ Ainv
    gb = (P["Phi0"][Bx] + np.einsum("bnh,bh->bn", W2r[Bx], hj)) @ Ainv
    unch = np.ones((nb, N), bool); unch[bi, ki] = False; unch[bi[d2], kj[d2]] = False
    # diagonal part: sum_{k unchanged} Z[k,k,:] . dh
    Zd = np.einsum("kkh->kh", Z)
    s = np.einsum("bk,bk->b", dh @ Zd.T, unch)
    # off-diagonal part from the two changed rows of G_j^{-1}
    Vi = np.einsum("bmh,bh->bm", Z[:, ki, :].transpose(1, 0, 2), dh)
    Vj = np.einsum("bmh,bh->bm", Z[:, kj, :].transpose(1, 0, 2), dh)
    Xa = np.where(unch, ga, 0.0); Xb = np.where(unch, gb, 0.0)
    # det(B + Delta)/det A ~ r (1 + s_diag) + tr(adj(G_j) restricted) : no
    # division by det B, so the expansion stays O(Delta^2) near nodes of B.
    r = ga[bi, ki].copy()
    s1 = ~d2
    q = np.zeros(nb)
    q[s1] = -np.einsum("bm,bm->b", Xa[s1], Vi[s1])
    w11, w12 = ga[bi[d2], ki[d2]], ga[bi[d2], kj[d2]]
    w21, w22 = gb[bi[d2], ki[d2]], gb[bi[d2], kj[d2]]
    r[d2] = w11 * w22 - w12 * w21
    a_i = -(w22[:, None] * Xa[d2] - w12[:, None] * Xb[d2])
    a_j = -(-w21[:, None] * Xa[d2] + w11[:, None] * Xb[d2])
    q[d2] = np.einsum("bm,bm->b", a_i, Vi[d2]) + np.einsum("bm,bm->b", a_j, Vj[d2])
    val = r * (1 + s) + q
    L = occ_n[None].repeat(nb, 0)
    L[bi, ki] = A
    L[bi[d2], kj[d2]] = B[d2]
    inv = np.zeros(nb, np.int64)
    for p_ in range(N):
        for q_ in range(p_ + 1, N):
            inv += L[:, p_] > L[:, q_]
    return val * (1 - 2 * (inv & 1))


# ----------------------------------------------------------------------------
# Row data and kappa measurement
# ----------------------------------------------------------------------------
def collect_rows(ham, net, th, codes, occ_idx, psi, rows_idx, E_var, en_delta):
    P = net.unpack(th)
    W2r = P["W2"].reshape(net.M, net.N, net.Hh)
    flat = {k: [] for k in ("rid", "h", "x", "yFB", "yLRX", "yEN", "dHjj")}
    Hnn, ncon = [], []
    for r_i, n in enumerate(rows_idx):
        r = ham.row(codes[n], occ_idx[n])
        j = np.searchsorted(codes, r["code"])
        z, Phi, G = net.setup_one(P, occ_idx[n])
        rt = frozen_ratio(G, occ_idx[n], occ_idx[j], r["rem"], r["add"])
        Ainv = np.linalg.inv(Phi[occ_idx[n]])
        rx = lrx_ratio(P, W2r, z, Ainv, lrx_setup(W2r, Ainv, occ_idx[n]), occ_idx[n], r["rem"], r["add"])
        h = r["h"]
        flat["rid"].append(np.full(len(h), r_i))
        flat["h"].append(h)
        flat["x"].append(h * psi[j] / psi[n])
        flat["yFB"].append(h * rt)
        flat["yLRX"].append(h * rx)
        den = np.minimum(E_var - r["Hjj"], -en_delta)
        flat["yEN"].append(h ** 2 / den)
        flat["dHjj"].append(r["Hjj"] - E_var)
        Hnn.append(r["Hnn"]); ncon.append(len(h))
    out = {k: np.concatenate(v) for k, v in flat.items()}
    out["Hnn"] = np.array(Hnn); out["ncon"] = np.array(ncon, float)
    return out


def _best_time(fn, reps=5):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)


def kernel_times(P, M, N, hso, gas, Gd, occ_n, rid, rem, add, occ_j, en_delta, chunk=8192, reps=5):
    """Wall time per call of every kernel, batched over the given connected
    terms (rows occ_n, term->row map rid, excitations rem/add, targets occ_j):
      T_f    one exact NNBF amplitude of a connected configuration
             (incremental hidden update, occupied rows only, det),
      T_h    one Hamiltonian element (Slater-Condon gather + phase, per-row
             Fock build amortised over the given terms),
      T_fb   one frozen-backflow ratio (2x2 Green's-function minor + parity),
      T_en   one Epstein-Nesbet term (diagonal update + division),
      T_lx   one LRX ratio (R2.3),
      T_s    one per-row setup (MLP, Phi(n), inverse, G),
      T_slx  one per-row LRX setup (Z tensor)."""
    nr = len(occ_n)
    nterm = len(rid)
    I, J, A, B = rem[:, 0], rem[:, 1], add[:, 0], add[:, 1]
    d2 = J >= 0; s1 = ~d2
    Gkk = np.einsum("pkqk->kpq", gas)
    hdiag = np.diag(hso)

    def t_H():
        O = np.zeros((nr, M)); O[np.arange(nr)[:, None], occ_n] = 1
        Cc = (np.cumsum(O, 1) - O).astype(np.int64)
        Fn = hso[None] + Gkk[occ_n].sum(1)
        v = np.empty(nterm)
        r1 = rid[s1]
        v[s1] = Fn[r1, A[s1], I[s1]] * (1 - 2 * ((Cc[r1, I[s1]] + Cc[r1, A[s1]] - (I[s1] < A[s1])) & 1))
        r2 = rid[d2]; i2, j2, a2, b2 = I[d2], J[d2], A[d2], B[d2]
        ph = (Cc[r2, i2] + Cc[r2, j2] - 1 + Cc[r2, b2] - (i2 < b2) - (j2 < b2)
              + Cc[r2, a2] - (i2 < a2) - (j2 < a2))
        v[d2] = gas[a2, b2, i2, j2] * (1 - 2 * (ph & 1))
        return v

    def setup(Pn):
        O = np.zeros((nr, M)); O[np.arange(nr)[:, None], occ_n] = 1
        z = (2 * O - 1) @ Pn["W1"].T + Pn["b1"]
        Phi = Pn["Phi0"][None] + (np.tanh(z) @ Pn["W2"].T).reshape(nr, M, N)
        An = np.take_along_axis(Phi, occ_n[:, :, None], axis=1)
        Ainv = np.linalg.inv(An)
        G = Phi @ Ainv
        return z, G, Ainv

    def setup_lrx(Pn, Ainv):
        W2r = Pn["W2"].reshape(M, N, -1)
        return np.einsum("rmlh,rlc->rmch", W2r[occ_n], Ainv)

    def t_LRX(Pn, z, Ainv, Zt):
        W2r = Pn["W2"].reshape(M, N, -1)
        out = np.empty(nterm)
        for r_ in range(nr):
            idx = np.nonzero(rid == r_)[0]
            for s0 in range(0, len(idx), chunk):
                m = idx[s0:s0 + chunk]
                out[m] = lrx_ratio(Pn, W2r, z[r_], Ainv[r_], Zt[r_], occ_n[r_], rem[m], add[m])
        return out

    def t_fwd(Pn, z):
        W1T = Pn["W1"].T; W2r = Pn["W2"].reshape(M, N, -1)
        out = np.empty(nterm)
        for s0 in range(0, nterm, chunk):
            sl = slice(s0, s0 + chunk)
            dz = 2 * (W1T[A[sl]] - W1T[I[sl]])
            dd = d2[sl]
            dz[dd] += 2 * (W1T[B[sl][dd]] - W1T[J[sl][dd]])
            hid = np.tanh(z[rid[sl]] + dz)
            oj = occ_j[sl]
            Am = Pn["Phi0"][oj] + np.einsum("bknh,bh->bkn", W2r[oj], hid)
            out[sl] = np.linalg.det(Am)
        return out

    def t_FB(G):
        pos = np.full((nr, M), -1); pos[np.arange(nr)[:, None], occ_n] = np.arange(N)[None]
        ki = pos[rid, I]
        r = G[rid, A, ki]
        r2 = rid[d2]; kj = pos[r2, J[d2]]
        r[d2] = G[r2, A[d2], ki[d2]] * G[r2, B[d2], kj] - G[r2, A[d2], kj] * G[r2, B[d2], ki[d2]]
        L = occ_n[rid].copy()
        L[np.arange(nterm), ki] = A
        L[np.nonzero(d2)[0], kj] = B[d2]
        inv = np.zeros(nterm, np.int64)
        for p_ in range(N):
            for q_ in range(p_ + 1, N):
                inv += L[:, p_] > L[:, q_]
        return r * (1 - 2 * (inv & 1))

    def t_EN(hv):
        O = np.zeros((nr, M)); O[np.arange(nr)[:, None], occ_n] = 1
        f = hdiag[None] + O @ Gd
        dE = np.empty(nterm)
        r1 = rid[s1]
        dE[s1] = f[r1, A[s1]] - f[r1, I[s1]] - Gd[A[s1], I[s1]]
        r2 = rid[d2]; i2, j2, a2, b2 = I[d2], J[d2], A[d2], B[d2]
        dE[d2] = (f[r2, a2] + f[r2, b2] - f[r2, i2] - f[r2, j2] + Gd[i2, j2] + Gd[a2, b2]
                  - Gd[a2, i2] - Gd[a2, j2] - Gd[b2, i2] - Gd[b2, j2])
        return hv ** 2 / np.minimum(-dE, -en_delta)

    hv = t_H()
    z, G, Ainv = setup(P)
    Zt = setup_lrx(P, Ainv)
    amp = t_fwd(P, z)
    T = {"T_f": _best_time(lambda: t_fwd(P, z), reps) / nterm,
         "T_h": _best_time(t_H, reps) / nterm,
         "T_fb": _best_time(lambda: t_FB(G), reps) / nterm,
         "T_en": _best_time(lambda: t_EN(hv), reps) / nterm,
         "T_lx": _best_time(lambda: t_LRX(P, z, Ainv, Zt), max(2, reps // 2)) / nterm,
         "T_s": _best_time(lambda: setup(P), reps) / nr,
         "T_slx": _best_time(lambda: setup_lrx(P, Ainv), reps) / nr}
    return T, hv, amp


def kappas_from_times(T):
    f = T["T_f"]
    return {"kH": T["T_h"] / f, "kFB": T["T_fb"] / f, "kEN": T["T_en"] / f, "kLRX": T["T_lx"] / f,
            "kSetup": T["T_s"] / f, "kSetupLRX": T["T_slx"] / f, "t_forward_us": 1e6 * f}


def measure_kappa(ham, net, P, codes, occ_idx, rows_idx, en_delta, nmax=40):
    """Kappas measured on the test instance itself (its N, M and trained width)."""
    M, N = net.M, net.N
    rows_idx = np.asarray(rows_idx[:nmax])
    rows = [ham.row(codes[n], occ_idx[n]) for n in rows_idx]
    rid = np.concatenate([np.full(len(r["h"]), i) for i, r in enumerate(rows)])
    rem = np.concatenate([r["rem"] for r in rows]); add = np.concatenate([r["add"] for r in rows])
    jj = np.concatenate([np.searchsorted(codes, r["code"]) for r in rows])
    T, hv, amp = kernel_times(P, M, N, ham.hso, ham.gas, ham.Gd, occ_idx[rows_idx], rid, rem, add,
                              occ_idx[jj], en_delta)
    out = kappas_from_times(T)
    out.update({"N": N, "M": M, "hidden": int(net.Hh), "n_terms_timed": int(len(rid)),
                "kernel_H_err": float(np.max(np.abs(hv - np.concatenate([r["h"] for r in rows]))))})
    out["_fwd_amp"] = amp; out["_jj"] = jj
    return out


def measure_kappa_scale(N, M, hidden, nterm, en_delta, rng, nr=4):
    """Kappas for a chemistry-scale NNBF (N electrons, M spin-orbitals, hidden
    width), measured with the SAME kernels on random parameters, random
    integrals and random single/double excitations of nr random samples."""
    Phi0 = np.linalg.qr(rng.standard_normal((M, N)))[0]
    P = {"Phi0": Phi0, "W1": 0.05 * rng.standard_normal((hidden, M)), "b1": np.zeros(hidden),
         "W2": 0.05 * rng.standard_normal((M * N, hidden))}
    hso = rng.standard_normal((M, M)); hso = hso + hso.T
    gas = rng.standard_normal((M, M, M, M))
    Gd = np.einsum("pqpq->pq", gas).copy()
    occ_n = np.array([np.sort(rng.choice(M, N, replace=False)) for _ in range(nr)])
    per = nterm // nr
    rid, rem, add, occ_j = [], [], [], []
    for r_ in range(nr):
        occ = occ_n[r_]; vir = np.setdiff1d(np.arange(M), occ)
        ij = np.sort(np.array([rng.choice(occ, 2, replace=False) for _ in range(per)]), 1)
        ab = np.sort(np.array([rng.choice(vir, 2, replace=False) for _ in range(per)]), 1)
        single = rng.random(per) < 0.05
        ij[single, 1] = -1; ab[single, 1] = -1
        for t in range(per):
            o = set(occ.tolist()); o.discard(ij[t, 0]); o.add(ab[t, 0])
            if ij[t, 1] >= 0:
                o.discard(ij[t, 1]); o.add(ab[t, 1])
            occ_j.append(sorted(o))
        rid.append(np.full(per, r_)); rem.append(ij); add.append(ab)
    T, _, _ = kernel_times(P, M, N, hso, gas, Gd, occ_n, np.concatenate(rid), np.concatenate(rem),
                           np.concatenate(add), np.array(occ_j), en_delta, reps=3)
    out = kappas_from_times(T)
    out.update({"N": N, "M": M, "hidden": hidden, "n_terms_timed": int(per * nr)})
    return out


# ----------------------------------------------------------------------------
# Estimator analysis (all tau^2 analytic per row)
# ----------------------------------------------------------------------------
class Analyzer:
    def __init__(self, d, wrow, E_L, kap, args, E_exact, V_exact):
        """wrow: row weights (pi-weights, or stratified importance weights);
        normalised here. Var_pi(E_L) and E are the EXACT full-space values;
        row-dependent quantities (tau^2, costs, screening bias/variance shifts)
        are weighted row averages with common rows for every method."""
        self.d, self.w, self.EL, self.kap, self.a = d, wrow / wrow.sum(), E_L, kap, args
        self.R = len(wrow)
        self.rid = d["rid"]
        self.ah = np.abs(d["h"])
        self.E = float(E_exact)
        self.V = float(V_exact)
        self.kgrid = np.array(args.k_grid, float)
        self.eps = np.logspace(-8, 0, args.n_eps)
        self.ncon = d["ncon"]

    def rs(self, v, mask=None):
        if mask is not None:
            v = np.where(mask, v, 0.0)
        return np.bincount(self.rid, weights=v, minlength=self.R)

    def tau2(self, x, mask, q, alpha):
        """Per-row single-draw variance of sum_{mask} x with p ~ alpha q/Zq + (1-alpha)|h|/Zh."""
        Zq = self.rs(q, mask); Zh = self.rs(self.ah, mask)
        Zq_e = Zq[self.rid]; Zh_e = Zh[self.rid]
        with np.errstate(divide="ignore", invalid="ignore"):
            pq = np.where(Zq_e > 0, q / Zq_e, self.ah / np.where(Zh_e > 0, Zh_e, 1))
            p = alpha * pq + (1 - alpha) * self.ah / np.where(Zh_e > 0, Zh_e, 1)
            t = np.where(mask & (p > 0), x ** 2 / np.where(p > 0, p, 1), 0.0)
        bad = mask & (p <= 0) & (x != 0)
        tau = self.rs(t) - self.rs(x, mask) ** 2
        tau[np.bincount(self.rid, weights=bad.astype(float), minlength=self.R) > 0] = np.inf
        return np.maximum(tau, 0.0), Zq, Zh

    def best_K(self, strata, base):
        """strata: list of (tau_row, alloc_weight_row, nonempty_row). Integer
        allocation K_s = max(1, round(K w_s)) on non-empty strata."""
        best = None
        for K in self.kgrid:
            var_row = np.zeros(self.R); cost_row = np.zeros(self.R)
            for tau, wgt, ne in strata:
                Ks = np.where(ne, np.maximum(1, np.round(K * wgt)), 0)
                var_row += np.where(Ks > 0, tau / np.maximum(Ks, 1), 0)
                cost_row += Ks
            C = (self.V + self.w @ var_row) * (self.w @ base + self.w @ cost_row)
            if best is None or C < best["C"]:
                best = {"C": float(C), "K": int(K), "tau2_over_var": float(self.w @ var_row * K / max(self.V, 1e-300))}
        return best

    def unbiased(self, name, score, extraD, q, sign, base_fn, resid_cv=None, alphas=None, adaptive=False):
        """Semistochastic: D = {score > eps} | extraD exact; tail sampled.
        sign: None (single stratum) or array of +-1 for stratification.
        resid_cv: if given (array y), additive control variate: sample x - y on
        the tail and add sum_tail y exactly (round-1 SCV).
        adaptive: per-sample draw allocation K_n = max(1, round(K Z_n / E_pi Z))
        with the cheap, known tail mass Z_n = sum_tail q (R2.5)."""
        x = self.d["x"]
        best = None
        alphas = self.a.alphas if alphas is None else alphas
        for eps in self.eps:
            D = score > eps
            if extraD is not None:
                D = D | extraD
            tail = ~D
            nD = self.rs(np.ones_like(x), D)
            base = base_fn(nD)
            xt = x - resid_cv if resid_cv is not None else x
            for al in alphas:
                if sign is None:
                    tau, Zq, Zh = self.tau2(xt, tail, q, al)
                    wgt = np.ones(self.R)
                    if adaptive:
                        zm = float(self.w @ Zq)
                        wgt = Zq / zm if zm > 0 else wgt
                    strata = [(tau, wgt, self.rs(np.ones_like(x), tail) > 0)]
                else:
                    strata = []
                    Zq_tot = self.rs(q, tail); Zh_tot = self.rs(self.ah, tail)
                    for s in (1, -1):
                        m = tail & (sign == s)
                        tau, Zq, Zh = self.tau2(xt, m, q, al)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            wq = np.where(Zq_tot > 0, Zq / Zq_tot, 0)
                            wh = np.where(Zh_tot > 0, Zh / Zh_tot, 0)
                        strata.append((tau, al * wq + (1 - al) * wh, self.rs(np.ones_like(x), m) > 0))
                b = self.best_K(strata, base)
                if best is None or b["C"] < best["C"]:
                    best = dict(b, eps=float(eps), alpha=al, nD=float(self.w @ nD))
        best["name"] = name
        return best

    def multilevel(self, name, y0, y1, base_fn, lvl1_cost, lvl1_setup, alphas=None, adaptive=False):
        """R2.4 three-level estimator (multilevel Monte Carlo over surrogate
        fidelity): D = {|y0| > eps} exact; on the tail T,
          sum_T x = sum_T y0 (exact, cheap)
                  + sum_T (y1 - y0)  [K1 draws, cost lvl1_cost each]
                  + sum_T (x - y1)   [K2 draws, one amplitude each],
        both corrections with the defensive proposal p ~ a|y0| + (1-a)|H|
        (y1 is never needed outside the drawn terms)."""
        x = self.d["x"]
        best = None
        alphas = self.a.alphas if alphas is None else alphas
        a0 = np.abs(y0)
        for eps in self.eps:
            D = a0 > eps
            tail = ~D
            nD = self.rs(np.ones_like(x), D)
            base = self.w @ base_fn(nD)
            ne = self.rs(np.ones_like(x), tail) > 0
            for al in alphas:
                t1, Zq, _ = self.tau2(y1 - y0, tail, a0, al)
                t2, _, _ = self.tau2(x - y1, tail, a0, al)
                wgt = np.ones(self.R)
                if adaptive:
                    zm = float(self.w @ Zq)
                    wgt = Zq / zm if zm > 0 else wgt
                for K1 in self.kgrid:
                    K1n = np.maximum(1, np.round(K1 * wgt))
                    v1 = self.w @ np.where(ne, t1 / K1n, 0)
                    c1 = self.w @ np.where(ne, K1n * lvl1_cost + lvl1_setup, 0)
                    for K2 in self.kgrid:
                        K2n = np.maximum(1, np.round(K2 * wgt))
                        v2 = self.w @ np.where(ne, t2 / K2n, 0)
                        c2 = self.w @ np.where(ne, K2n, 0)
                        C = (self.V + v1 + v2) * (base + c1 + c2)
                        if best is None or C < best["C"]:
                            best = {"C": float(C), "K": int(K2), "K1": int(K1), "eps": float(eps), "alpha": al,
                                    "nD": float(self.w @ nD)}
        best["name"] = name
        return best

    def biased(self, name, score, extraD, base_fn, completion=None):
        x = self.d["x"]
        best = None
        for eps in self.eps:
            D = score > eps
            if extraD is not None:
                D = D | extraD
            ELs = self.d["Hnn"] + self.rs(x, D)
            if completion is not None:
                ELs = ELs + self.rs(completion, ~D)
            bias = float(self.w @ (ELs - self.EL))
            if abs(bias) > self.a.bias_tol:
                continue
            var_s = self.V + float(self.w @ ((ELs - self.E - bias) ** 2 - (self.EL - self.E) ** 2))
            nD = self.rs(np.ones_like(x), D)
            C = max(var_s, 0.0) * float(self.w @ base_fn(nD))
            if best is None or C < best["C"]:
                best = {"C": C, "eps": float(eps), "bias": bias, "nD": float(self.w @ nD)}
        if best is None:
            best = {"C": float("inf"), "note": "no admissible threshold"}
        best["name"] = name
        return best

    def run(self):
        d, k = self.d, self.kap
        nc = self.ncon
        aFB, aEN = np.abs(d["yFB"]), np.abs(d["yEN"])
        sFB = np.where(d["yFB"] >= 0, 1, -1)
        sX = np.where(d["x"] >= 0, 1, -1)
        out = {"Var_sample_rows": float(self.w @ (self.EL - self.E) ** 2), "mean_ncon": float(self.w @ nc)}
        cFB = lambda nD: 1 + k["kSetup"] + (k["kH"] + k["kFB"]) * nc + nD
        cEN = lambda nD: 1 + (k["kH"] + k["kEN"]) * nc + nD
        cHYB = lambda nD: 1 + k["kSetup"] + (k["kH"] + k["kFB"] + k["kEN"]) * nc + nD
        out["EXACT"] = {"C": self.V * float(self.w @ (1 + (1 + k["kH"]) * nc)), "name": "EXACT"}
        out["SCREEN_H"] = self.biased("SCREEN_H", self.ah, None, lambda nD: 1 + (1 + k["kH"]) * nD)
        out["SCREEN_FB"] = self.biased("SCREEN_FB", aFB, None, cFB)
        out["COMP_FB"] = self.biased("COMP_FB", aFB, None, cFB, completion=d["yFB"])
        best_sen, best_cen, best_pt2, best_hyb, best_or = None, None, None, None, None
        for dP in self.a.delta_p:
            P = d["dHjj"] < dP
            r1 = self.biased("SCREEN_EN", aEN, P, cEN)
            r2 = self.biased("COMP_EN", aEN, P, cEN, completion=d["yEN"])
            r3 = self.unbiased("PT2IS", aEN, P, aEN, None, cEN)
            r4 = self.unbiased("POS_HYB", aEN, P, aEN, sFB, cHYB)
            r5 = self.unbiased("ORACLE_SIGN", aEN, P, aEN, sX, cEN)
            for r in (r1, r2, r3, r4, r5):
                r["delta_P"] = dP
            best_sen = r1 if best_sen is None or r1["C"] < best_sen["C"] else best_sen
            best_cen = r2 if best_cen is None or r2["C"] < best_cen["C"] else best_cen
            best_pt2 = r3 if best_pt2 is None or r3["C"] < best_pt2["C"] else best_pt2
            best_hyb = r4 if best_hyb is None or r4["C"] < best_hyb["C"] else best_hyb
            best_or = r5 if best_or is None or r5["C"] < best_or["C"] else best_or
        out["SCREEN_EN"], out["COMP_EN"], out["PT2IS"], out["POS_HYB"], out["ORACLE_SIGN"] = \
            best_sen, best_cen, best_pt2, best_hyb, best_or
        allm = np.ones(len(d["x"]), bool)
        tauHB, _, _ = self.tau2(d["x"], allm, self.ah, 0.0)
        out["HB"] = dict(self.best_K([(tauHB, np.ones(self.R), nc > 0)], np.ones(self.R)), name="HB")
        out["SS"] = self.unbiased("SS", self.ah, None, self.ah, None, lambda nD: 1 + (1 + k["kH"]) * nD, alphas=[0.0])
        out["SCVSS"] = self.unbiased("SCVSS", aFB, None, aFB, None, cFB, resid_cv=d["yFB"])
        out["POS_FB"] = self.unbiased("POS_FB", aFB, None, aFB, sFB, cFB)
        # R2.3: exact-new-rows + linear-response backflow surrogate (LRX)
        for tag, kk in (("LRX", "kLRX"),):
            y = d["y" + tag]; ay = np.abs(y); sy = np.where(y >= 0, 1, -1)
            cst = (lambda kk_: (lambda nD: 1 + k["kSetup"] + k["kSetupLRX"]
                                + (k["kH"] + k[kk_]) * nc + nD))(kk)
            out["SCREEN_" + tag] = self.biased("SCREEN_" + tag, ay, None, cst)
            out["COMP_" + tag] = self.biased("COMP_" + tag, ay, None, cst, completion=y)
            out["SCV_" + tag] = self.unbiased("SCV_" + tag, ay, None, ay, None, cst, resid_cv=y)
            out["POS_" + tag] = self.unbiased("POS_" + tag, ay, None, ay, sy, cst)
            out["sign_mismatch_" + tag] = float(self.w @ self.rs(np.abs(d["x"]), sy != sX) / (self.w @ self.rs(np.abs(d["x"]))))
            out["resid_frac_" + tag] = float(self.w @ self.rs(np.abs(d["x"] - y)) / (self.w @ self.rs(np.abs(d["x"]))))
        out["ISFB"] = self.unbiased("ISFB", aFB, None, aFB, None, cFB)
        out["MLMC"] = self.multilevel("MLMC", d["yFB"], d["yLRX"], cFB, k["kLRX"], k["kSetupLRX"])
        out["SCV_ADA"] = self.unbiased("SCV_ADA", aFB, None, aFB, None, cFB, resid_cv=d["yFB"], adaptive=True)
        out["MLMC_ADA"] = self.multilevel("MLMC_ADA", d["yFB"], d["yLRX"], cFB, k["kLRX"], k["kSetupLRX"],
                                          adaptive=True)
        tauO, _, _ = self.tau2(d["x"], allm, np.abs(d["x"]), 1.0)
        out["ORACLE_IS"] = dict(self.best_K([(tauO, np.ones(self.R), nc > 0)], np.ones(self.R)), name="ORACLE_IS")
        # diagnostics
        ax = np.abs(d["x"])
        out["sign_mismatch_FB"] = float(self.w @ self.rs(ax, sFB != sX) / (self.w @ self.rs(ax)))
        out["sign_mismatch_EN"] = float(self.w @ self.rs(ax, sX > 0) / (self.w @ self.rs(ax)))
        out["resid_frac_FB"] = float(self.w @ self.rs(np.abs(d["x"] - d["yFB"])) / (self.w @ self.rs(ax)))
        out["resid_frac_EN"] = float(self.w @ self.rs(np.abs(d["x"] - d["yEN"])) / (self.w @ self.rs(ax)))
        ctrl = ["EXACT", "SCREEN_H", "SCREEN_FB", "SCREEN_EN", "COMP_FB", "COMP_EN", "HB", "SS", "SCVSS",
                "SCREEN_LRX", "COMP_LRX"]
        out["C_ctrl"] = min(out[c]["C"] for c in ctrl)
        # R2.0 (revived round-1 SCV) is judged against the controls WITHOUT itself
        ctrl0 = [c for c in ctrl if c != "SCVSS"]
        out["C_ctrl0"] = min(out[c]["C"] for c in ctrl0)
        out["ctrl0_best"] = min(ctrl0, key=lambda c: out[c]["C"])
        out["M_R20"] = out["SCVSS"]["C"] / out["C_ctrl0"]
        out["ctrl_best"] = min(ctrl, key=lambda c: out[c]["C"])
        out["C_R21"] = min(out["POS_FB"]["C"], out["POS_HYB"]["C"])
        out["C_R22"] = out["PT2IS"]["C"]
        out["M_R21"] = out["C_R21"] / out["C_ctrl"]
        out["M_R22"] = out["C_R22"] / out["C_ctrl"]
        out["C_R23"] = min(out[m]["C"] for m in ("SCV_LRX", "POS_LRX"))
        out["M_R23"] = out["C_R23"] / out["C_ctrl"]
        out["C_R24"] = out["MLMC"]["C"]
        out["M_R24"] = out["C_R24"] / out["C_ctrl"]
        out["C_R25"] = out["SCV_ADA"]["C"]
        out["M_R25"] = out["C_R25"] / out["C_ctrl"]
        out["M_R24R25"] = out["MLMC_ADA"]["C"] / out["C_ctrl"]
        return out


def choose_rows(wfull, args, rng):
    """Rows for the pi-averages. dim <= --exact-rows-max: every configuration
    with weight pi(n) (exact). Otherwise: the top-weight configurations holding
    1 - --top-tail of the mass (at most --n-rows of them) exactly, plus
    --n-rows draws from the rest with q ~ sqrt(pi) (defensive for the heavy
    E_L tail), importance weights pi/(q S)."""
    D = len(wfull)
    if D <= args.exact_rows_max:
        return np.arange(D), wfull.copy()
    order = np.argsort(-wfull)
    cm = np.cumsum(wfull[order])
    nt = min(int(np.searchsorted(cm, 1 - args.top_tail)) + 1, args.n_rows)
    top, rest = order[:nt], order[nt:]
    q = np.sqrt(wfull[rest]); q /= q.sum()
    dr = rng.choice(len(rest), size=args.n_rows, p=q)
    u, cnt = np.unique(dr, return_counts=True)
    idx = np.concatenate([top, rest[u]])
    w = np.concatenate([wfull[top], wfull[rest[u]] * cnt / (q[u] * args.n_rows)])
    return idx, w


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def build_instance(n_atoms, ns, R, kind):
    X = geometry(n_atoms, R, kind)
    S, hao, Gao, enuc = ao_integrals(X, ns)
    nocc = n_atoms // 2
    Cm, Eel, eps = rhf(S, hao, Gao, nocc)
    h = Cm.T @ hao @ Cm
    g = np.einsum("pqrs,pa,qb,rc,sd->abcd", Gao, Cm, Cm, Cm, Cm, optimize=True)
    return h, g, enuc, Eel + enuc, Cm.shape[1]


def smoke_checks(rng):
    t0 = time.time()
    # (1) H2 STO-3G reference energies
    h, g, en, Erhf, K = build_instance(2, 1, 1.4, "chain")
    ham = SQHam(h, g, en, K, 2)
    codes, occ, oi = build_space(K, 1, 1)
    E0 = ground_states(build_hamiltonian(ham, codes, oi), 1)[0][0]
    print(f"H2 STO-3G R=1.4: E_RHF={Erhf:.6f} (ref -1.116714)  E_FCI={E0:.6f} (ref -1.137284)")
    # (2) Slater-Condon builder vs independent E_pq builder, elementwise
    h, g, en, Erhf, K = build_instance(4, 2, 2.0, "ring")
    ham = SQHam(h, g, en, K, 4)
    codes, occ, oi = build_space(K, 2, 2)
    Hs = build_hamiltonian(ham, codes, oi)
    Hr = build_hamiltonian_ref(h, g, en, codes, occ, K)
    err = abs(Hs - Hr).max()
    print(f"H4 ring ns=2 (K={K}, dim={len(codes)}): max |H_SC - H_Epq| = {err:.2e}")
    # (3) EN diagonal update and frozen 2x2 ratio vs direct determinants
    net = NNBF(2 * K, 4, 6, np.eye(2 * K)[:, [0, 1, K, K + 1]], 0.3, rng)
    th = net.theta0 + 0.1 * rng.standard_normal(net.theta0.size)
    P = net.unpack(th)
    n = 37
    r = ham.row(codes[n], oi[n])
    j = np.searchsorted(codes, r["code"])
    dHjj = np.max(np.abs(r["Hjj"] - Hs.diagonal()[j]))
    _, Phi, G = net.setup_one(P, oi[n])
    rt = frozen_ratio(G, oi[n], oi[j], r["rem"], r["add"])
    rt_ref = np.linalg.det(Phi[oi[j]]) / np.linalg.det(Phi[oi[n]])
    W2r = P["W2"].reshape(net.M, net.N, net.Hh)
    z, _, _ = net.setup_one(P, oi[n])
    amp = net.forward_connected(P, W2r, z, oi[j], r["rem"], r["add"])
    amp_ref = net.amplitudes(th, occ[j], oi[j])[0]
    print(f"  EN diagonal update err={dHjj:.2e}  frozen-ratio err={np.max(np.abs(rt-rt_ref)):.2e}  "
          f"fast-forward err={np.max(np.abs(amp-amp_ref)):.2e}")
    # LRX order check: with backflow scaled by s, the frozen ratio errs at O(s)
    # and LRX at O(s^2) (new rows exact, old rows linearised).
    errs = []
    for sc in (1e-2, 1e-3):
        Ps = dict(P); Ps["W2"] = P["W2"] * sc
        th_s = np.concatenate([Ps[k_].ravel() for k_, _ in net.shapes])
        W2s = Ps["W2"].reshape(net.M, net.N, net.Hh)
        zs, Phis, Gs = net.setup_one(Ps, oi[n])
        Ai = np.linalg.inv(Phis[oi[n]])
        ex = net.amplitudes(th_s, occ[j], oi[j])[0] / net.amplitudes(th_s, occ[[n]], oi[[n]])[0][0]
        fb = frozen_ratio(Gs, oi[n], oi[j], r["rem"], r["add"])
        lx = lrx_ratio(Ps, W2s, zs, Ai, lrx_setup(W2s, Ai, oi[n]), oi[n], r["rem"], r["add"])
        errs.append((np.max(np.abs(fb - ex)), np.max(np.abs(lx - ex))))
    print(f"  backflow x1e-2: |FB-exact|={errs[0][0]:.2e} |LRX-exact|={errs[0][1]:.2e};  "
          f"x1e-3: |FB-exact|={errs[1][0]:.2e} |LRX-exact|={errs[1][1]:.2e}")
    lrx_order_ok = errs[1][1] < 0.03 * errs[0][1] and errs[1][0] > 0.03 * errs[0][0]
    # (4) gradient
    _, ga = net.energy_grad(th, Hs, occ, oi)
    idx = rng.choice(th.size, 15, replace=False); gn = []
    for i in idx:
        dd = np.zeros_like(th); dd[i] = 1e-6
        gn.append((net.energy_grad(th + dd, Hs, occ, oi)[0] - net.energy_grad(th - dd, Hs, occ, oi)[0]) / 2e-6)
    gerr = np.max(np.abs(ga[idx] - gn)) / max(np.max(np.abs(gn)), 1e-12)
    print(f"  gradient check: max rel err = {gerr:.2e}   ({time.time()-t0:.1f}s)")
    ok = err < 1e-10 and dHjj < 1e-10 and np.max(np.abs(rt - rt_ref)) < 1e-10 and gerr < 1e-5 \
        and np.max(np.abs(amp - amp_ref)) < 1e-10 and lrx_order_ok
    return {"H_builder_err": float(err), "EN_diag_err": float(dHjj),
            "frozen_ratio_err": float(np.max(np.abs(rt - rt_ref))), "grad_rel_err": float(gerr), "all_ok": bool(ok)}


def run_instance(ns, args, rng):
    t0 = time.time()
    n_atoms = args.n_atoms
    h, g, enuc, Erhf, K = build_instance(n_atoms, ns, args.bond_length, args.geometry)
    nocc = n_atoms // 2
    ham = SQHam(h, g, enuc, K, n_atoms)
    codes, occ, occ_idx = build_space(K, nocc, nocc)
    H = build_hamiltonian(ham, codes, occ_idx)
    tb = time.time() - t0
    w2, v2 = ground_states(H, 2)
    E0, psi0 = w2[0], v2[:, 0]
    hf_code = sum(1 << i for i in range(nocc)) | (sum(1 << i for i in range(nocc)) << K)
    hf_w = float(psi0[np.searchsorted(codes, hf_code)] ** 2)
    Hc = H.tocoo(); m = Hc.row != Hc.col
    prod = Hc.data[m] * psi0[Hc.row[m]] * psi0[Hc.col[m]]
    frustr = float(np.abs(prod[prod > 0]).sum() / np.abs(prod).sum())
    print(f"\n=== H{n_atoms} {args.geometry} R={args.bond_length} ns={ns}: K={K} M={2*K} dim={len(codes)} "
          f"E_RHF={Erhf:.6f} E_FCI={E0:.6f} Ecorr={1e3*(E0-Erhf):.1f} mHa gap(S_z=0)={w2[1]-E0:.4f} "
          f"HF wt={hf_w:.3f} frustr={frustr:.3f} nnz/row={H.nnz/len(codes):.0f} (build {tb:.1f}s)", flush=True)
    Morb, N = 2 * K, n_atoms
    Phi0 = np.zeros((Morb, N))
    Phi0[:K, :nocc] = np.eye(K)[:, :nocc]
    Phi0[K:, nocc:] = Phi0[:K, :nocc]
    net = NNBF(Morb, N, args.hidden, Phi0, args.init_scale, rng)
    res = {"ns": ns, "K": K, "M": 2 * K, "dim": len(codes), "E_FCI": float(E0), "E_RHF": float(Erhf),
           "gap_Sz0": float(w2[1] - E0), "HF_weight": hf_w, "frustrated_fraction": frustr,
           "nnz_per_row": H.nnz / len(codes), "snapshots": []}
    snaps = sorted(set([s for s in args.snapshots if s < args.maxiter] + [args.maxiter]))
    th = net.theta0.copy(); done = 0
    for s in snaps:
        t1 = time.time()
        r = minimize(lambda t: net.energy_grad(t, H, occ, occ_idx), th, jac=True, method="L-BFGS-B",
                     options={"maxiter": s - done, "maxcor": 30, "gtol": 1e-10, "ftol": 1e-15})
        th = r.x; done = s
        psi, _ = net.amplitudes(th, occ, occ_idx)
        wfull = psi ** 2 / (psi @ psi)
        Ev = float(psi @ (H @ psi) / (psi @ psi))
        EL_full = (H @ psi) / psi
        var_exact = float(wfull @ (EL_full - Ev) ** 2)
        topt = time.time() - t1
        rows_idx, wrow = choose_rows(wfull, args, rng)
        d = collect_rows(ham, net, th, codes, occ_idx, psi, rows_idx, Ev, args.en_delta)
        EL = d["Hnn"] + np.bincount(d["rid"], weights=d["x"], minlength=len(rows_idx))
        el_err = float(np.max(np.abs(EL - EL_full[rows_idx])))
        order = np.argsort(-wfull[rows_idx])
        P_ = net.unpack(th)
        kap_inst = measure_kappa(ham, net, P_, codes, occ_idx, rows_idx[order], args.en_delta)
        fwd_err = float(np.max(np.abs(kap_inst["_fwd_amp"] - psi[kap_inst["_jj"]])) / np.max(np.abs(psi)))
        kap_inst = {k_: v for k_, v in kap_inst.items() if not k_.startswith("_")}
        kap_used = dict(args._kappa_scale)
        if args.kappa_override is not None:
            kap_used.update(dict(zip(["kH", "kFB", "kEN", "kLRX", "kSetup", "kSetupLRX"], args.kappa_override)))
        an = Analyzer(d, wrow, EL, kap_used, args, Ev, var_exact).run()
        a2 = Analyzer(d, wrow, EL, kap_inst, args, Ev, var_exact).run()
        sens = {"instance_kappa": {"M_R20": a2["M_R20"], "M_R21": a2["M_R21"], "M_R22": a2["M_R22"], "M_R23": a2["M_R23"],
                                   "M_R24": a2["M_R24"], "M_R25": a2["M_R25"], "M_R24R25": a2["M_R24R25"],
                                   "ctrl_best": a2["ctrl_best"], "C_ctrl_over_exact": a2["C_ctrl"] / a2["EXACT"]["C"],
                                   "C_route_over_exact": {kk: a2[kk]["C"] / a2["EXACT"]["C"] for kk in
                                                          ("POS_FB", "PT2IS", "SCV_LRX", "POS_LRX", "MLMC", "SCV_ADA",
                                                           "MLMC_ADA", "SCVSS",
                                                           "SCREEN_H", "SS")}}}
        an.update({"iter": s, "dE_mHa": 1e3 * (Ev - E0), "var_exact": var_exact, "EL_check": el_err,
                   "fwd_kernel_rel_err": fwd_err, "n_rows": int(len(rows_idx)),
                   "kappa": kap_used, "kappa_instance": kap_inst, "kappa_sensitivity": sens})
        res["snapshots"].append(an)
        print(f"  it={s:5d} dE={an['dE_mHa']:8.3f} mHa VarEx={var_exact:.2e} VarRows={an['Var_sample_rows']:.2e} "
              f"<|C|>={an['mean_ncon']:.0f} rows={len(rows_idx)} ELchk={el_err:.1e} fwdchk={fwd_err:.1e} "
              f"(opt {topt:.1f}s, total {time.time()-t1:.1f}s)", flush=True)
        for lab_, kp in (("decisive kappa (chemistry scale)", an["kappa"]), ("instance kappa", kap_inst)):
            print(f"     {lab_}: kH={kp['kH']:.3g} kFB={kp['kFB']:.3g} kEN={kp['kEN']:.3g} "
                  f"kLRX={kp['kLRX']:.3g} kSetup={kp['kSetup']:.3g} "
                  f"kSetupLRX={kp['kSetupLRX']:.3g} (t_fwd={kp['t_forward_us']:.2f} us/amp)")
        print(f"     sign mismatch: FB={an['sign_mismatch_FB']:.3f} LRX={an['sign_mismatch_LRX']:.3f} "
              f"EN={an['sign_mismatch_EN']:.3f}  resid frac: FB={an['resid_frac_FB']:.3f} "
              f"LRX={an['resid_frac_LRX']:.3f} EN={an['resid_frac_EN']:.3f}")
        ce = an["EXACT"]["C"]
        for key in ["EXACT", "SCREEN_H", "SCREEN_FB", "SCREEN_EN", "COMP_FB", "COMP_EN", "SCREEN_LRX",
                    "COMP_LRX", "HB", "SS", "SCVSS", "POS_FB", "POS_HYB", "PT2IS", "SCV_LRX", "POS_LRX",
                    "MLMC", "SCV_ADA", "MLMC_ADA", "ISFB", "ORACLE_SIGN", "ORACLE_IS"]:
            dd = an[key]
            extra = " ".join(f"{q}={dd[q]:.3g}" for q in ("K", "K1", "eps", "alpha", "delta_P", "nD", "bias") if q in dd)
            print(f"     {key:11s} C/C_exact={dd['C']/ce:10.4g}  {extra}")
        print(f"     -> R2.0 (revived SCV) vs best non-SCV control ({an['ctrl0_best']}): M_R2.0 = {an['M_R20']:.4g}")
        print(f"     -> control best = {an['ctrl_best']};  M_R2.1 = {an['M_R21']:.4g};  M_R2.2 = {an['M_R22']:.4g};"
              f"  M_R2.3 = {an['M_R23']:.4g};  M_R2.4 = {an['M_R24']:.4g};  M_R2.5 = {an['M_R25']:.4g};"
              f"  M_(R2.4+R2.5) = {an['M_R24R25']:.4g}",
              flush=True)
        sv = sens["instance_kappa"]
        print(f"     [sensitivity, instance kappa] M_R2.0={sv['M_R20']:.4g} ctrl={sv['ctrl_best']} M_R2.1={sv['M_R21']:.4g} "
              f"M_R2.2={sv['M_R22']:.4g} M_R2.3={sv['M_R23']:.4g} M_R2.4={sv['M_R24']:.4g} "
              f"M_R2.5={sv['M_R25']:.4g} M_(R2.4+R2.5)={sv['M_R24R25']:.4g}", flush=True)
    return res


def verdict(results, args):
    st = sorted(results, key=lambda r: r["ns"])
    out = {"ns": [r["ns"] for r in st], "M_orb": [r["M"] for r in st],
           "dE_final_mHa": [r["snapshots"][-1]["dE_mHa"] for r in st]}
    for key, lab in (("M_R20", "R2.0"), ("M_R21", "R2.1"), ("M_R22", "R2.2"), ("M_R23", "R2.3"), ("M_R24", "R2.4"),
                     ("M_R25", "R2.5"), ("M_R24R25", "R2.4+R2.5")):
        Ms = [r["snapshots"][-1][key] for r in st]
        Mx = Ms[-1]
        if Mx >= 1.0:
            v = "FAIL"
        elif Mx <= 0.5 and (len(Ms) == 1 or Mx < Ms[0]) and out["dE_final_mHa"][-1] <= args.de_max_mHa:
            v = "PASS"
        else:
            v = "AMBIGUOUS"
        out[lab] = {"M_by_ns": Ms, "verdict": v}
    return out


def aggregate(path):
    import glob
    files = sorted(glob.glob(os.path.join(path, "results_*_*.json")))
    files = [f for f in files if "smoke" not in f]
    per = {}
    for f in files:
        with open(f) as fh:
            dat = json.load(fh)
        if "verdict" not in dat:
            print(f"{os.path.basename(f)}: incomplete (no verdict)")
            continue
        v = dat["verdict"]
        per[os.path.basename(f)] = v
        print(f"{os.path.basename(f)}: geometry={dat['args']['geometry']} M_orb={v['M_orb']} "
              f"dE={['%.2f' % x for x in v['dE_final_mHa']]} "
              f"R2.0 M={['%.3g' % x for x in v['R2.0']['M_by_ns']]} {v['R2.0']['verdict']}  "
              f"R2.1 M={['%.3g' % x for x in v['R2.1']['M_by_ns']]} {v['R2.1']['verdict']}  "
              f"R2.2 M={['%.3g' % x for x in v['R2.2']['M_by_ns']]} {v['R2.2']['verdict']}  "
              f"R2.3 M={['%.3g' % x for x in v['R2.3']['M_by_ns']]} {v['R2.3']['verdict']}  "
              f"R2.4 M={['%.3g' % x for x in v['R2.4']['M_by_ns']]} {v['R2.4']['verdict']}  "
              f"R2.5 M={['%.3g' % x for x in v['R2.5']['M_by_ns']]} {v['R2.5']['verdict']}  "
              f"R2.4+R2.5 M={['%.3g' % x for x in v['R2.4+R2.5']['M_by_ns']]} {v['R2.4+R2.5']['verdict']}")
    glob_v = {}
    for lab in ("R2.0", "R2.1", "R2.2", "R2.3", "R2.4", "R2.5", "R2.4+R2.5"):
        vs = [v[lab]["verdict"] for v in per.values()]
        if not vs:
            glob_v[lab] = "NO DATA"
        elif "FAIL" in vs:
            glob_v[lab] = "FAIL"
        elif all(x == "PASS" for x in vs) and len(vs) >= 2:
            glob_v[lab] = "PASS"
        else:
            glob_v[lab] = "AMBIGUOUS"
    print("GLOBAL VERDICT:", json.dumps(glob_v))
    return glob_v


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        aggregate(args.aggregate)
        return
    rng = np.random.default_rng(args.seed)
    print("args:", vars(args), flush=True)
    payload = {"args": {k_: v for k_, v in vars(args).items() if not k_.startswith("_")}}
    if args.smoke:
        payload["smoke_checks"] = smoke_checks(rng)
    t0 = time.time()
    Nk, Mk, hk, nk = args.kappa_scale
    args._kappa_scale = measure_kappa_scale(Nk, Mk, hk, nk, args.en_delta, rng)
    ks = args._kappa_scale
    print(f"chemistry-scale kappa (N={Nk}, M={Mk}, hidden={hk}, {nk} terms): kH={ks['kH']:.3g} "
          f"kFB={ks['kFB']:.3g} kEN={ks['kEN']:.3g} kLRX={ks['kLRX']:.3g} kSetup={ks['kSetup']:.3g} "
          f"kSetupLRX={ks['kSetupLRX']:.3g} t_fwd={ks['t_forward_us']:.2f} us ({time.time()-t0:.1f}s)", flush=True)
    payload["kappa_scale"] = ks
    results = []
    for ns in args.ns:
        results.append(run_instance(ns, args, rng))
        payload["results"] = results
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1, default=float)
    payload["verdict"] = verdict(results, args)
    print("\nVERDICT:", json.dumps(payload["verdict"]), flush=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1, default=float)


if __name__ == "__main__":
    main()
