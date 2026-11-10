#!/usr/bin/env python3
"""
P2.2 decisive test: frozen-backflow control-variate (SCV) local energy versus
exact enumeration, magnitude screening, heat-bath and semistochastic local
energies, for a neural-network-backflow (NNBF) state on a stretched hydrogen
chain in a delocalized (RHF) orbital basis.

What the script does
--------------------
For every (n_atoms, R) instance:
  1. Builds STO-3G integrals for a linear hydrogen chain (or ring) with bond
     length R (bohr) from analytic s-Gaussian formulas (numpy/scipy only),
     runs closed-shell RHF, and transforms to RHF molecular orbitals, so the
     two-body tensor V_{pqrs} is dense (all-to-all, O(M^4) connected terms).
  2. Builds the exact second-quantized Hamiltonian as a sparse matrix in the
     S_z = 0, N_e = n_atoms sector (Jordan-Wigner sign convention, spin-orbital
     order alpha_0..alpha_{K-1}, beta_0..beta_{K-1}) and its FCI ground state.
     It reports Hard-Core diagnostics: the HF-determinant weight in psi_0 and
     the frustrated fraction  F = sum_{n!=n'} |H psi0 psi0| [H psi0(n) psi0(n') > 0]
     / sum_{n!=n'} |H psi0 psi0|  (F = 0 would mean a sign-free basis).
  3. Optimizes a real NNBF state psi(n) = det(Phi(n)[occ(n), :]),
     Phi(n) = Phi0 + reshape(W2 tanh(W1 (2n-1) + b1)), by L-BFGS on the exact
     energy (analytic gradient), saving snapshots at chosen iterations.
  4. For every snapshot, computes EXACTLY (no Monte Carlo noise: all variances
     are evaluated analytically over |psi|^2 and over the proposal
     distributions) the cost to reach a fixed statistical error for:
        EXACT      exact enumeration of all connected n'
        SCREEN     keep |H_{nn'}| > eps only (biased; admissible only if
                   |bias| <= --bias-tol)
        HB         heat-bath: K i.i.d. draws with p ~ |H_{nn'}| (unbiased)
        SS         semistochastic: exact for |H|>eps + heat-bath tail (unbiased)
        SCV        route: exact sum of H * frozen-backflow ratio (cheap
                   arithmetic) + K draws of the residual H*(r - r~) with a
                   defensive mixture proposal (unbiased)
        SCVSS      SCV with the terms |H r~| > eps evaluated exactly
        FROZEN     surrogate only, no correction (biased; diagnostic)
     Cost per sample is counted in NQS-forward-pass units: 1 (own amplitude)
     + number of NQS amplitudes of connected configurations evaluated
     + kappa * (number of surrogate terms) + 1 (Green's-function setup) for
     SCV/SCVSS/FROZEN.  Cost-to-target C = Var_total * cost_per_sample, with
     Var_total = Var_{|psi|^2}(E_loc) + E_{|psi|^2}[tau^2(n)/K].
     Every method is optimized over K (and eps).
  5. Decisive metric:  M = C_SCVbest / min(C_HB, C_SS, C_SCREEN)  at the final
     snapshot of the largest system at the largest bond length.

Command-line options (all optional)
-----------------------------------
--sizes INT [INT ...]      Number of H atoms per instance. Each must be even
                           and >= 2 (closed-shell S_z=0). Valid practical range
                           2..10; 10 needs ~4-8 GB RAM and ~1-3 h. Default: 6 8
--bond-lengths F [F ...]   H-H distances in bohr (> 0). The LARGEST value is
                           treated as the "stretched" Hard-Core instance used
                           for the verdict. Default: 1.8 3.0
--geometry {chain,ring}    chain: linear equally spaced chain (default).
                           ring: regular polygon with nearest-neighbour
                           distance R.
--hidden INT               Hidden width of the backflow MLP (>= 1). Default 32.
--maxiter INT              Total L-BFGS iterations for the NNBF (>= 1).
                           Default 3000.
--snapshots INT [INT ...]  Iteration numbers at which to analyse the state
                           (values > maxiter are ignored; the final state is
                           always analysed). Default: 50 300 1000
--init-scale FLOAT         Std of random init for W1, W2 (>= 0). Default 0.05.
--seed INT                 RNG seed. Default 1234.
--kappa FLOAT              Cost of one surrogate (Green's-function) term in
                           NQS-forward-pass units (>= 0). Default 1e-3
                           (conservative; ~1e-4 for a 1e5-parameter network).
--bias-tol FLOAT           Max |bias| (Hartree) for SCREEN to be admissible.
                           Default 1.6e-4 (10% of chemical accuracy).
--mix-alpha FLOAT          Weight in [0,1] of the |H r~| component in the SCV
                           proposal p = a*|H r~|/Z1 + (1-a)*|H|/Z2. Default 0.5.
--k-grid INT [INT ...]     Candidate numbers of stochastic draws per sample.
                           Default: 1 2 4 8 16 32 64 128 256
--n-eps INT                Number of log-spaced eps thresholds in [1e-7, 1]
                           for SCREEN/SS/SCVSS (>= 2). Default 30.
--out PATH                 JSON output path. Default: results.json next to
                           this script.
--smoke                    Tiny fast run: sizes=[4], bond-lengths=[1.4, 3.0],
                           maxiter=150, snapshots=[30], hidden=8. Overrides the
                           corresponding options. Also checks H2 FCI energy
                           (-1.1373 Ha at R=1.4) and the analytic gradient.
--check-grad               Compare analytic vs finite-difference NNBF gradient
                           on the first instance and print the max rel. error.

Verdict thresholds (printed at the end, evaluated on the largest size at the
largest bond length, final snapshot):
  PASS       M <= 0.3 and M(largest size) <= M(smallest size) (non-degrading)
             and C_SCVbest / C_EXACT < 1
  FAIL       M >= 1.0  or  C_SCVbest / C_EXACT >= 1
  AMBIGUOUS  anything else -> pre-committed action: rerun with --sizes 8 10
             at the stretched bond length only; PASS iff M(10) <= min(M(8),0.5).

Example
-------
python3 micro_test.py --sizes 6 8 --bond-lengths 1.8 3.0 --maxiter 3000
python3 micro_test.py --smoke
"""
import argparse
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import erf

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="P2.2 SCV local-energy decisive test")
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--bond-lengths", type=float, nargs="+", default=[1.8, 3.0])
    ap.add_argument("--geometry", choices=["chain", "ring"], default="chain")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--snapshots", type=int, nargs="+", default=[50, 300, 1000])
    ap.add_argument("--init-scale", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--kappa", type=float, default=1e-3)
    ap.add_argument("--bias-tol", type=float, default=1.6e-4)
    ap.add_argument("--mix-alpha", type=float, default=0.5)
    ap.add_argument("--k-grid", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--n-eps", type=int, default=30)
    ap.add_argument("--out", type=str, default=os.path.join(here, "results.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check-grad", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.sizes, a.bond_lengths, a.maxiter, a.snapshots, a.hidden = [4], [1.4, 3.0], 150, [30], 8
        a.check_grad = True
        if a.out == os.path.join(here, "results.json"):
            a.out = os.path.join(here, "results_smoke.json")
    for s in a.sizes:
        if s < 2 or s % 2:
            ap.error("--sizes entries must be even and >= 2")
    if any(r <= 0 for r in a.bond_lengths):
        ap.error("--bond-lengths must be > 0")
    if a.hidden < 1 or a.maxiter < 1 or a.n_eps < 2:
        ap.error("--hidden, --maxiter >= 1 and --n-eps >= 2 required")
    if not (0.0 <= a.mix_alpha <= 1.0):
        ap.error("--mix-alpha must be in [0,1]")
    if a.kappa < 0 or a.init_scale < 0 or min(a.k_grid) < 1:
        ap.error("--kappa, --init-scale >= 0 and --k-grid entries >= 1 required")
    return a


# ----------------------------------------------------------------------------
# Integrals (STO-3G hydrogen, analytic s-Gaussians)
# ----------------------------------------------------------------------------
STO3G_A = np.array([3.42525091, 0.62391373, 0.16885540])
STO3G_D = np.array([0.15432897, 0.53532814, 0.44463454])


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


def ao_integrals(X):
    K = len(X)
    a = np.tile(STO3G_A, K)
    c = np.tile(STO3G_D, K) * (2 * a / np.pi) ** 0.75
    C = np.repeat(X, 3, axis=0)
    own = np.repeat(np.arange(K), 3)
    T = np.zeros((3 * K, K))
    T[np.arange(3 * K), own] = c  # primitive -> contracted
    A, B = a[:, None], a[None, :]
    p = A + B
    d2 = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    E = np.exp(-A * B / p * d2)
    S_p = (np.pi / p) ** 1.5 * E
    T_p = A * B / p * (3 - 2 * A * B / p * d2) * S_p
    P = (A[..., None] * C[:, None, :] + B[..., None] * C[None, :, :]) / p[..., None]
    V_p = np.zeros_like(S_p)
    for Cn in X:
        V_p += -2 * np.pi / p * E * boys0(p * ((P - Cn) ** 2).sum(-1))
    # ERI (ij|kl)
    pq = p[:, :, None, None] * p[None, None, :, :]
    ps = p[:, :, None, None] + p[None, None, :, :]
    dPQ = ((P[:, :, None, None, :] - P[None, None, :, :, :]) ** 2).sum(-1)
    G_p = 2 * np.pi ** 2.5 / (pq * np.sqrt(ps)) * E[:, :, None, None] * E[None, None] \
        * boys0(pq / ps * dPQ)
    S = T.T @ S_p @ T
    h = T.T @ (T_p + V_p) @ T
    G = np.einsum("ijkl,ia,jb,kc,ld->abcd", G_p, T, T, T, T, optimize=True)
    enuc = sum(1.0 / np.linalg.norm(X[i] - X[j]) for i in range(K) for j in range(i))
    return S, h, G, enuc


def rhf(S, h, G, nocc, maxit=500, damp=0.5):
    s, U = np.linalg.eigh(S)
    Xh = U @ np.diag(s ** -0.5) @ U.T
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
    return Cm, Eel


# ----------------------------------------------------------------------------
# Fock space, Hamiltonian
# ----------------------------------------------------------------------------


def build_space(K, na, nb):
    def strings(n):
        return [sum(1 << i for i in c) for c in combinations(range(K), n)]
    codes = np.array(sorted(a | (b << K) for a in strings(na) for b in strings(nb)), dtype=np.int64)
    M = 2 * K
    occ = ((codes[:, None] >> np.arange(M)) & 1).astype(np.int8)
    occ_idx = np.array([np.nonzero(o)[0] for o in occ])
    return codes, occ, occ_idx


def epq_coo(codes, occ, K, p, q):
    """Spin-summed E_pq = sum_s c^dag_{p s} c_{q s}; returns (row=new, col=old, val)."""
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
        s1 = o[:, :Q].sum(1)
        o[:, Q] = 0
        s2 = o[:, :P].sum(1)
        newc = codes[i] - (1 << Q) + (1 << P)
        j = np.searchsorted(codes, newc)
        rows.append(j); cols.append(i); vals.append((-1.0) ** (s1 + s2))
    if not rows:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def build_hamiltonian(h, g, enuc, codes, occ, K):
    D = len(codes)
    E = {}
    for p in range(K):
        for q in range(K):
            E[p, q] = epq_coo(codes, occ, K, p, q)
    allr = np.concatenate([E[k][0] for k in E])
    allc = np.concatenate([E[k][1] for k in E])
    allv = np.concatenate([E[k][2] for k in E])
    lab_p = np.concatenate([np.full(len(E[k][0]), k[0]) for k in E])
    lab_q = np.concatenate([np.full(len(E[k][0]), k[1]) for k in E])
    hp = h - 0.5 * np.einsum("prrq->pq", g)
    H = sp.csr_matrix((D, D))
    one = sp.coo_matrix((hp[lab_p, lab_q] * allv, (allr, allc)), shape=(D, D)).tocsr()
    H = H + one
    for p in range(K):
        for q in range(K):
            r, c, v = E[p, q]
            if len(r) == 0:
                continue
            Epq = sp.csr_matrix((v, (r, c)), shape=(D, D))
            Cpq = sp.coo_matrix((g[p, q, lab_p, lab_q] * allv, (allr, allc)), shape=(D, D)).tocsr()
            H = H + 0.5 * (Epq @ Cpq)
    H = H + enuc * sp.identity(D, format="csr")
    H = ((H + H.T) * 0.5).tocsr()
    H.eliminate_zeros()
    return H


def ground_state(H):
    D = H.shape[0]
    if D <= 2500:
        w, v = np.linalg.eigh(H.toarray())
        return w[0], v[:, 0]
    from scipy.sparse.linalg import eigsh
    w, v = eigsh(H, k=1, which="SA", tol=1e-12)
    return w[0], v[:, 0]


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

    def phi_all(self, th, occ):
        P = self.unpack(th)
        x = 2.0 * occ - 1.0
        hid = np.tanh(x @ P["W1"].T + P["b1"])
        Phi = P["Phi0"][None] + (hid @ P["W2"].T).reshape(-1, self.M, self.N)
        return Phi, hid, x, P

    def amplitudes(self, th, occ, occ_idx):
        Phi, hid, x, P = self.phi_all(th, occ)
        A = np.take_along_axis(Phi, occ_idx[:, :, None], axis=1)
        return np.linalg.det(A), (Phi, A, hid, x, P)

    def energy_grad(self, th, H, occ, occ_idx):
        psi, (Phi, A, hid, x, P) = self.amplitudes(th, occ, occ_idx)
        nrm = psi @ psi
        Hpsi = H @ psi
        E = psi @ Hpsi / nrm
        c = 2 * (Hpsi - E * psi) / nrm  # dE/dpsi(n)
        Ainv = np.linalg.inv(A)
        gA = (c * psi)[:, None, None] * np.transpose(Ainv, (0, 2, 1))
        gPhi = np.zeros_like(Phi)
        np.put_along_axis(gPhi, occ_idx[:, :, None], gA, axis=1)  # rows unique per config
        gflat = gPhi.reshape(len(psi), -1)
        g = {"Phi0": gPhi.sum(0), "W2": gflat.T @ hid}
        dz = (gflat @ P["W2"]) * (1 - hid ** 2)
        g["W1"] = dz.T @ x
        g["b1"] = dz.sum(0)
        return E, np.concatenate([g[k].ravel() for k, _ in self.shapes])


# ----------------------------------------------------------------------------
# Local-energy estimator analysis
# ----------------------------------------------------------------------------


def estimator_data(net, th, H, occ, occ_idx):
    psi, (Phi, A, hid, x, P) = net.amplitudes(th, occ, occ_idx)
    D = len(psi)
    w = psi ** 2 / (psi @ psi)
    Hd = H.diagonal()
    rows = []
    for n in range(D):
        lo, hi = H.indptr[n], H.indptr[n + 1]
        j = H.indices[lo:hi]; h = H.data[lo:hi]
        m = (j != n) & (np.abs(h) > 1e-12)  # drop exact-zero (symmetry-forbidden) elements
        j, h = j[m], h[m]
        r = psi[j] / psi[n]
        Asur = Phi[n][occ_idx[j]]  # frozen backflow orbitals of n, rows of n'
        rt = np.linalg.det(Asur) / psi[n]
        rows.append((h, r, rt))
    EL = np.array([Hd[n] + rows[n][0] @ rows[n][1] for n in range(D)])
    return psi, w, Hd, rows, EL


def is_tau2(x, p):
    """Single-draw variance of sum(x) estimated by x_j/p_j, j~p (p normalized on support)."""
    if len(x) == 0:
        return 0.0
    m = p > 0
    if np.any((~m) & (x != 0)):
        return np.inf
    return float(np.sum(x[m] ** 2 / p[m]) - np.sum(x) ** 2)


def analyse(w, Hd, rows, EL, kappa, bias_tol, alpha, kgrid, n_eps):
    E = w @ EL
    var_ex = float(w @ (EL - E) ** 2)
    nconn = np.array([len(r[0]) for r in rows], float)
    eps_grid = np.logspace(-7, 0, n_eps)
    kgrid = np.array(kgrid, float)
    out = {"E_var": float(E), "var_exact": var_ex, "mean_nconn": float(w @ nconn)}
    c_exact = var_ex * (1 + w @ nconn)
    out["C_EXACT"] = c_exact

    def best_over_K(tau2_mean, base_cost):
        v = var_ex + tau2_mean / kgrid
        c = v * (base_cost + kgrid)
        i = int(np.argmin(c))
        return float(c[i]), int(kgrid[i]), float(v[i])

    # HB
    tau = np.array([is_tau2(h * r, np.abs(h) / np.abs(h).sum()) if len(h) else 0.0 for h, r, rt in rows])
    out["HB"] = dict(zip(["C", "K", "var"], best_over_K(w @ tau, 1.0)))
    # SCREEN / SS
    best_scr, best_ss = None, None
    for eps in eps_grid:
        keep = [np.abs(h) > eps for h, r, rt in rows]
        nk = np.array([k.sum() for k in keep], float)
        ELs = np.array([Hd[n] + rows[n][0][keep[n]] @ rows[n][1][keep[n]] for n in range(len(rows))])
        bias = float(w @ ELs - E)
        if abs(bias) <= bias_tol:
            Es = w @ ELs
            c = float(w @ (ELs - Es) ** 2) * (1 + w @ nk)
            if best_scr is None or c < best_scr["C"]:
                best_scr = {"C": c, "eps": float(eps), "bias": bias, "nkeep": float(w @ nk)}
        tau = []
        for n, (h, r, rt) in enumerate(rows):
            t = ~keep[n]
            ht = h[t]
            tau.append(is_tau2(ht * r[t], np.abs(ht) / np.abs(ht).sum()) if ht.size and np.abs(ht).sum() > 0 else 0.0)
        c, K, v = best_over_K(w @ np.array(tau), 1.0 + w @ nk)
        if best_ss is None or c < best_ss["C"]:
            best_ss = {"C": c, "K": K, "var": v, "eps": float(eps)}
    out["SCREEN"] = best_scr if best_scr else {"C": float("inf")}
    out["SS"] = best_ss

    # SCV
    def scv_tau(h, r, rt, mask):
        x = h[mask] * (r[mask] - rt[mask])
        if x.size == 0:
            return 0.0
        a1 = np.abs(h[mask] * rt[mask]); a2 = np.abs(h[mask])
        p = np.zeros_like(a2)
        if a1.sum() > 0:
            p += alpha * a1 / a1.sum()
        else:
            p += alpha * a2 / a2.sum()
        p += (1 - alpha) * a2 / a2.sum()
        return is_tau2(x, p)

    tau = np.array([scv_tau(h, r, rt, np.ones(len(h), bool)) if len(h) else 0.0 for h, r, rt in rows])
    base = 2.0 + kappa * (w @ nconn)
    out["SCV"] = dict(zip(["C", "K", "var"], best_over_K(w @ tau, base)))
    out["SCV"]["tau2_mean"] = float(w @ tau)
    best = None
    for eps in eps_grid:
        tau, nk = [], []
        for h, r, rt in rows:
            keep = np.abs(h * rt) > eps
            nk.append(keep.sum())
            tau.append(scv_tau(h, r, rt, ~keep) if len(h) else 0.0)
        c, K, v = best_over_K(w @ np.array(tau), base + w @ np.array(nk, float))
        if best is None or c < best["C"]:
            best = {"C": c, "K": K, "var": v, "eps": float(eps)}
    out["SCVSS"] = best
    # FROZEN only (biased diagnostic)
    ELf = np.array([Hd[n] + rows[n][0] @ rows[n][2] for n in range(len(rows))])
    out["FROZEN_bias"] = float(w @ ELf - E)
    # residual fraction
    num = sum(w[n] * np.abs(h * (r - rt)).sum() for n, (h, r, rt) in enumerate(rows))
    den = sum(w[n] * np.abs(h * r).sum() for n, (h, r, rt) in enumerate(rows))
    out["residual_fraction"] = float(num / den)
    # oracle IS lower bound for any purely stochastic sampler (needs psi(n'))
    tau = np.array([is_tau2(h * r, np.abs(h * r) / np.abs(h * r).sum()) if len(h) and np.abs(h * r).sum() > 0 else 0.0 for h, r, rt in rows])
    out["ORACLE_IS"] = dict(zip(["C", "K", "var"], best_over_K(w @ tau, 1.0)))
    out["C_SCVbest"] = min(out["SCV"]["C"], out["SCVSS"]["C"])
    out["C_baseline_best"] = min(out["HB"]["C"], out["SS"]["C"], out["SCREEN"]["C"])
    out["M"] = out["C_SCVbest"] / out["C_baseline_best"]
    out["SCV_over_EXACT"] = out["C_SCVbest"] / c_exact
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def run_instance(n_atoms, R, args, rng):
    t0 = time.time()
    X = geometry(n_atoms, R, args.geometry)
    S, hao, Gao, enuc = ao_integrals(X)
    nocc = n_atoms // 2
    Cm, Eel = rhf(S, hao, Gao, nocc)
    h = Cm.T @ hao @ Cm
    g = np.einsum("pqrs,pa,qb,rc,sd->abcd", Gao, Cm, Cm, Cm, Cm, optimize=True)
    K = n_atoms
    codes, occ, occ_idx = build_space(K, nocc, nocc)
    H = build_hamiltonian(h, g, enuc, codes, occ, K)
    E0, psi0 = ground_state(H)
    hf_code = sum(1 << i for i in range(nocc)) | (sum(1 << i for i in range(nocc)) << K)
    hf_w = float(psi0[np.searchsorted(codes, hf_code)] ** 2)
    Hc = H.tocoo(); m = Hc.row != Hc.col
    prod = Hc.data[m] * psi0[Hc.row[m]] * psi0[Hc.col[m]]
    frustr = float(np.abs(prod[prod > 0]).sum() / np.abs(prod).sum())
    print(f"\n=== H{n_atoms} {args.geometry} R={R}: dim={len(codes)}  E_RHF={Eel+enuc:.6f}  "
          f"E_FCI={E0:.6f}  HF weight={hf_w:.3f}  frustrated frac={frustr:.3f}  "
          f"nnz/row={H.nnz/len(codes):.0f}  ({time.time()-t0:.1f}s)", flush=True)
    Morb, N = 2 * K, n_atoms
    Phi0 = np.zeros((Morb, N))
    Phi0[:K, :nocc] = np.eye(K)[:, :nocc]  # orbitals are RHF MOs: HF = lowest nocc MOs
    Phi0[K:, nocc:] = Phi0[:K, :nocc]
    net = NNBF(Morb, N, args.hidden, Phi0, args.init_scale, rng)
    res = {"n_atoms": n_atoms, "R": R, "dim": len(codes), "E_FCI": float(E0),
           "E_RHF": float(Eel + enuc), "HF_weight": hf_w, "frustrated_fraction": frustr,
           "nnz_per_row": H.nnz / len(codes), "snapshots": []}

    if args.check_grad:
        th = net.theta0 + 0.05 * rng.standard_normal(net.theta0.size)
        _, ga = net.energy_grad(th, H, occ, occ_idx)
        idx = rng.choice(th.size, 20, replace=False)
        gn = []
        for i in idx:
            d = np.zeros_like(th); d[i] = 1e-6
            gn.append((net.energy_grad(th + d, H, occ, occ_idx)[0] - net.energy_grad(th - d, H, occ, occ_idx)[0]) / 2e-6)
        err = np.max(np.abs(ga[idx] - gn)) / max(np.max(np.abs(gn)), 1e-12)
        print(f"  gradient check: max rel err = {err:.2e}")
        res["grad_check_rel_err"] = float(err)
        args.check_grad = False

    snaps = sorted(set([s for s in args.snapshots if s < args.maxiter] + [args.maxiter]))
    th = net.theta0.copy()
    done = 0
    for s in snaps:
        r = minimize(lambda t: net.energy_grad(t, H, occ, occ_idx), th, jac=True,
                     method="L-BFGS-B", options={"maxiter": s - done, "maxcor": 30, "gtol": 1e-10, "ftol": 1e-15})
        th = r.x
        done = s
        Ev = net.energy_grad(th, H, occ, occ_idx)[0]
        t1 = time.time()
        psi, w, Hd, rows, EL = estimator_data(net, th, H, occ, occ_idx)
        a = analyse(w, Hd, rows, EL, args.kappa, args.bias_tol, args.mix_alpha, args.k_grid, args.n_eps)
        a["iter"] = s
        a["dE_mHa"] = 1e3 * (Ev - E0)
        res["snapshots"].append(a)
        print(f"  it={s:5d} dE={a['dE_mHa']:9.3f} mHa  VarEx={a['var_exact']:.2e}  "
              f"<nconn>={a['mean_nconn']:.0f}  resid.frac={a['residual_fraction']:.3f}  "
              f"FROZEN bias={1e3*a['FROZEN_bias']:.3f} mHa  ({time.time()-t1:.1f}s)", flush=True)
        ce = a["C_EXACT"]
        for k in ["HB", "SS", "SCREEN", "SCV", "SCVSS", "ORACLE_IS"]:
            d = a[k]
            extra = " ".join(f"{q}={d[q]:.3g}" for q in ("K", "eps", "bias") if q in d)
            print(f"     {k:9s} C/C_exact={d['C']/ce:10.4g}  {extra}")
        print(f"     -> M = C_SCVbest/C_baseline = {a['M']:.4g};  C_SCVbest/C_EXACT = {a['SCV_over_EXACT']:.4g}", flush=True)
    return res


def verdict(results):
    Rmax = max(r["R"] for r in results)
    st = [r for r in results if r["R"] == Rmax]
    st.sort(key=lambda r: r["n_atoms"])
    Ms = [r["snapshots"][-1]["M"] for r in st]
    ex = st[-1]["snapshots"][-1]["SCV_over_EXACT"]
    M = Ms[-1]
    if M >= 1.0 or ex >= 1.0:
        v = "FAIL"
    elif M <= 0.3 and Ms[-1] <= Ms[0] and ex < 1.0:
        v = "PASS"
    else:
        v = "AMBIGUOUS"
    return {"R_stretched": Rmax, "sizes": [r["n_atoms"] for r in st], "M_by_size": Ms,
            "M": M, "SCV_over_EXACT": ex, "verdict": v}


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    print("args:", vars(args), flush=True)
    if args.smoke:
        S, h, G, en = ao_integrals(geometry(2, 1.4, "chain"))
        Cm, Eel = rhf(S, h, G, 1)
        hm = Cm.T @ h @ Cm
        gm = np.einsum("pqrs,pa,qb,rc,sd->abcd", G, Cm, Cm, Cm, Cm)
        codes, occ, oi = build_space(2, 1, 1)
        E0 = ground_state(build_hamiltonian(hm, gm, en, codes, occ, 2))[0]
        print(f"H2 STO-3G R=1.4: E_RHF={Eel+en:.6f} (ref -1.116714)  E_FCI={E0:.6f} (ref -1.137284)")
    results = []
    for n in args.sizes:
        for R in args.bond_lengths:
            results.append(run_instance(n, R, args, rng))
            with open(args.out, "w") as f:
                json.dump({"args": vars(args), "results": results}, f, indent=1, default=float)
    v = verdict(results)
    print("\nVERDICT:", json.dumps(v), flush=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results, "verdict": v}, f, indent=1, default=float)


if __name__ == "__main__":
    main()
