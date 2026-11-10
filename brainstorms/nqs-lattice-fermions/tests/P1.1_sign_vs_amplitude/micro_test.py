#!/usr/bin/env python3
"""
P1.1 decisive test: sign-versus-amplitude error budget of fermionic NQS
=======================================================================

What it does
------------
For one doped 2D Hubbard cluster (t, t', U) it

1. builds the Hamiltonian in the full fixed-(N_up, N_dn) occupation basis
   (Jordan-Wigner order: all up modes 0..V-1, then all down modes 0..V-1;
   a basis state is |u, d> with u, d bit patterns) and solves it by Lanczos
   (scipy.sparse.linalg.eigsh). The lowest `--n-ed-states` eigenpairs are kept;
   if the ground state is degenerate (levels within --degeneracy-tol of E_0), the
   reference psi_0 used for each variational state is the normalized projection of
   that state onto the degenerate ground-state manifold.
2. prepares variational states, all evaluated EXACTLY on the full Hilbert space
   (no Monte Carlo noise in energies or correlations):
     * "uhf"        unrestricted Hartree-Fock Slater determinant (mean-field state),
     * "sj"         Slater-Jastrow foil: Slater determinant (orbitals trainable,
                    initialised at UHF) times exp(sum_{i<j} v_{d(i,j)} n_i n_j
                    + g sum_i n_{i,up} n_{i,dn}), optimised by stochastic reconfiguration
                    (SR) with the exact energy gradient,
     * "nnbf_H"     neural-network backflow (Luo & Clark, PRL 122, 226401) for each
                    hidden width H in --widths: orbitals Phi_sigma(n) = Phi0_sigma +
                    reshape(W2 tanh(W1 x + b1) + b2), amplitude det(Phi_up(n)[occ_up])
                    * det(Phi_dn(n)[occ_dn]) (optionally times the Jastrow factor, see
                    --nnbf-jastrow); optimised by SR with exact gradient.
3. for each variational state psi_theta computes the sign/amplitude error budget:
     s_theta(n) = sgn psi_theta(n),
     E_sign  = min_{a >= 0} a^T K a / a^T a with K = S H S, S = diag(s_theta)
              (best energy reachable with the sign structure of psi_theta and
              optimal non-negative amplitudes; computed by L-BFGS-B with bounds from
              several starts, so it is an UPPER estimate of the true constrained min),
     E_stoq  = lowest eigenvalue of K with its positive off-diagonal elements deleted
              (a rigorous LOWER bound on E_sign, since a^T K_+ a >= 0 for a >= 0),
     dE_tot  = E_theta - E_0,  dE_sign = E_sign - E_0,  dE_amp = E_theta - E_sign,
     f_E     = dE_sign / dE_tot  (sign fraction of the energy error),
     f_E_lo  = max(0, E_stoq - E_0) / dE_tot (rigorous lower bound on the sign fraction),
     pairing: singlet bond-pair correlations P(b0, b) = <Delta_b0^dag Delta_b>,
              Delta_b = (c_{j dn} c_{i up} - c_{j up} c_{i dn})/sqrt(2) for nearest-
              neighbour bond b=(i,j), reference bond b0 = x-bond at site 0, evaluated
              for psi_0, psi_theta and the sign-constrained optimum a* s_theta;
              eC_tot = max_{b: dist >= r_min} |P_theta - P_0|,
              eC_sign = max_{b: dist >= r_min} |P_{a* s_theta} - P_0|,
              f_C = eC_sign / eC_tot,
     sign-mismatch weight w = sum_{n: s_theta s_0 = -1} |psi_0(n)|^2 (after global sign
     alignment), and the "hybrid swap" energies E[s_theta |psi_0|], E[s_0 |psi_theta|].
   An optimizer calibration is also run once: the sign-constrained minimisation with
   the EXACT signs s_0 started from uniform amplitudes must return E_0 (reported as
   calib_err = E_sign(s_0) - E_0; a value above 1e-3 * dE_tot of the best NNBF flags
   L-BFGS-B local-minimum trouble).

Decision metric (pre-committed, see P1.1_sign_vs_amplitude.md):
   M = f_E of the widest NNBF state. PASS (sign binding) if M >= 0.6 on every instance,
   FAIL (amplitude binding) if M <= 0.3 on every instance, AMBIGUOUS otherwise.

Usage
-----
    python3 micro_test.py [options]

Options (all optional)
----------------------
  --instance K       Integer 0, 1, 2 or -1 (default -1). Selects a pre-committed
                     instance and OVERRIDES --Lx/--Ly/--bcx/--bcy/--nup/--ndn/--U/--tp:
                       0 : 4x3, PBC x PBC, N_up=N_dn=5 (doping 1/6),  U=8, t'=-0.25
                       1 : 4x3, PBC x PBC, N_up=N_dn=4 (doping 1/3),  U=8, t'=-0.25
                       2 : 4x3, PBC x PBC, N_up=N_dn=5 (doping 1/6),  U=4, t'=-0.25
                       3 : 2x2 smoke test, OBC x OBC, N_up=N_dn=1,      U=4, t'=0
                     -1 : use the explicit lattice options below.
  --Lx INT           Sites along x (default 4). Any integer >= 2.
  --Ly INT           Sites along y (default 3). Any integer >= 1 (1 gives a chain;
                     not a Hard-Core instance, only for checks).
  --bcx {pbc,obc}    Boundary condition along x (default pbc). "pbc" periodic, "obc" open.
  --bcy {pbc,obc}    Boundary condition along y (default pbc).
  --nup INT          Number of up electrons (default 5).
  --ndn INT          Number of down electrons (default 5).
  --t FLOAT          Nearest-neighbour hopping t (default 1.0). Hopping term is
                     -t_ij c^dag_i c_j + h.c.
  --tp FLOAT         Next-nearest (diagonal) hopping t' (default -0.25), enters as
                     -t'_ij; t'=0 disables diagonal bonds.
  --U FLOAT          On-site repulsion (default 8.0).
  --widths LIST      Comma-separated NNBF hidden widths (default "8,32,128"); the LAST
                     entry defines M. Empty string "" skips NNBF.
  --nnbf-jastrow {0,1}  1 multiplies NNBF by the same Jastrow factor as the foil
                     (default 0, plain NNBF as in Luo & Clark).
  --sr-iters INT     SR iterations per variational state (default 1500).
  --sr-lr FLOAT      SR step size eta (default 0.05).
  --sr-shift FLOAT   SR diagonal shift lambda (default 1e-3).
  --sr-samples INT   Configurations drawn exactly from |psi|^2 to build the SR metric
                     (default 2000). The force/gradient itself is always exact.
  --chunk INT        Configurations per forward/backward chunk (default 65536);
                     lower it to save memory.
  --lbfgs-iters INT  Max L-BFGS-B iterations per start in the sign-constrained
                     minimisation (default 4000).
  --n-ed-states INT  Number of lowest eigenpairs computed by Lanczos (default 3).
  --degeneracy-tol FLOAT  Energy window for a degenerate ground manifold (default 1e-8).
  --r-min FLOAT      Minimal bond-centre distance used in the pairing error (default 1.5).
  --seed INT         RNG seed (default 1234).
  --out PATH         JSON output path (default: results/<tag>.json next to this script,
                     where tag encodes the instance). Existing files are NOT overwritten
                     unless --overwrite 1.
  --overwrite {0,1}  1 allows overwriting an existing output JSON (default 0).
  --checks-only {0,1}  1 runs only the correctness checks (U=0 Slater energy equals
                     U=0 ED; two-site Hubbard analytic energy; NNBF gradient versus finite
                     differences) and exits (default 0). The checks also run at the start
                     of every normal run, which aborts with exit code 2 if they fail.
  --aggregate DIR    If given, do no computation: read every results/*.json in DIR that
                     contains "M" and print the per-instance M, f_E_lo, f_C of the widest
                     NNBF and the GLOBAL verdict (PASS if every M >= 0.6, FAIL if every
                     M <= 0.3, AMBIGUOUS otherwise). Default "" (off).

Outputs
-------
  JSON with keys: "params", "ed" (E0 list, gap, degenerate flag), "calib_err",
  "states" -> {name: {E, dE_tot, E_sign, E_stoq, dE_sign, dE_amp, f_E, f_E_lo,
  E_hyb_signtheta_ampexact, E_hyb_signexact_amptheta, w_sign, eC_tot, eC_sign, f_C,
  n_params, sign_starts}}, "M", "verdict_instance".
  Progress is printed to stdout (redirected to the Slurm log by submit_test.sbatch).
"""
import argparse
import itertools
import json
import os
import sys
import time
from math import comb

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

INSTANCES = {
    0: dict(Lx=4, Ly=3, bcx="pbc", bcy="pbc", nup=5, ndn=5, U=8.0, tp=-0.25),
    1: dict(Lx=4, Ly=3, bcx="pbc", bcy="pbc", nup=4, ndn=4, U=8.0, tp=-0.25),
    2: dict(Lx=4, Ly=3, bcx="pbc", bcy="pbc", nup=5, ndn=5, U=4.0, tp=-0.25),
    3: dict(Lx=2, Ly=2, bcx="obc", bcy="obc", nup=1, ndn=1, U=4.0, tp=0.0),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- lattice
def build_lattice(Lx, Ly, bcx, bcy, t, tp):
    """Return hopping matrix T (V x V, H_kin = -sum T_ij c^dag_i c_j), site coords,
    and list of NN bonds (i, j, direction)."""
    V = Lx * Ly
    idx = lambda x, y: x + Lx * y
    T = np.zeros((V, V))
    bonds = []
    coords = np.array([(x, y) for y in range(Ly) for x in range(Lx)], float)

    def add(x, y, dx, dy, amp, record=None):
        x2, y2 = x + dx, y + dy
        if not (0 <= x2 < Lx):
            if bcx == "obc":
                return
            x2 %= Lx
        if not (0 <= y2 < Ly):
            if bcy == "obc":
                return
            y2 %= Ly
        i, j = idx(x, y), idx(x2, y2)
        if i == j:
            return
        T[i, j] += amp
        T[j, i] += amp
        if record is not None:
            bonds.append((i, j, record))

    for y in range(Ly):
        for x in range(Lx):
            if Lx > 1 and not (Lx == 2 and bcx == "pbc" and x == 1):
                add(x, y, 1, 0, t, "x")
            elif Lx == 2 and bcx == "pbc" and x == 1:
                add(x, y, 1, 0, t)  # double wrap bond of a length-2 ring
            if Ly > 1 and not (Ly == 2 and bcy == "pbc" and y == 1):
                add(x, y, 0, 1, t, "y")
            elif Ly == 2 and bcy == "pbc" and y == 1:
                add(x, y, 0, 1, t)
            if tp != 0.0 and Lx > 1 and Ly > 1:
                add(x, y, 1, 1, tp)
                add(x, y, 1, -1, tp)
    return T, coords, bonds


def min_image(d, L, bc):
    if bc == "pbc":
        d = d - L * np.round(d / L)
    return d


# ----------------------------------------------------------------------------- basis
def species_basis(V, N):
    states = []
    for occ in itertools.combinations(range(V), N):
        b = 0
        for i in occ:
            b |= 1 << i
        states.append(b)
    states = np.array(sorted(states), dtype=np.int64)
    lookup = -np.ones(1 << V, dtype=np.int64)
    lookup[states] = np.arange(len(states))
    occ = ((states[:, None] >> np.arange(V)[None, :]) & 1).astype(np.int8)
    rows = np.array([np.nonzero(o)[0] for o in occ], dtype=np.int64).reshape(len(states), N)
    return states, lookup, occ, rows


def popcount_below(b, i):
    return bin(b & ((1 << i) - 1)).count("1")


def species_hop(V, N, T):
    """Sparse matrix of -sum_ij T_ij a^dag_i a_j within one species (JW sign)."""
    states, lookup, _, _ = species_basis(V, N)
    r, c, v = [], [], []
    for k, b in enumerate(states):
        for j in range(V):
            if not (b >> j) & 1:
                continue
            b1 = b ^ (1 << j)
            s1 = popcount_below(b, j)
            for i in range(V):
                if T[i, j] == 0.0 or (b1 >> i) & 1:
                    continue
                b2 = b1 | (1 << i)
                s2 = popcount_below(b1, i)
                r.append(lookup[b2]); c.append(k); v.append(-T[i, j] * (-1) ** (s1 + s2))
    D = len(states)
    return sp.csr_matrix((v, (r, c)), shape=(D, D))


def species_annihilate(V, N, i):
    """Matrix of a_i from N-particle to (N-1)-particle species space."""
    st, _, _, _ = species_basis(V, N)
    st2, lk2, _, _ = species_basis(V, N - 1)
    r, c, v = [], [], []
    for k, b in enumerate(st):
        if (b >> i) & 1:
            r.append(lk2[b ^ (1 << i)]); c.append(k); v.append((-1) ** popcount_below(b, i))
    return sp.csr_matrix((v, (r, c)), shape=(len(st2), len(st)))


class Hubbard:
    def __init__(self, Lx, Ly, bcx, bcy, nup, ndn, t, tp, U):
        self.Lx, self.Ly, self.bcx, self.bcy = Lx, Ly, bcx, bcy
        self.V = V = Lx * Ly
        self.nup, self.ndn, self.U = nup, ndn, U
        self.T, self.coords, self.bonds = build_lattice(Lx, Ly, bcx, bcy, t, tp)
        self.su, _, self.occ_u, self.rows_u = species_basis(V, nup)
        self.sd, _, self.occ_d, self.rows_d = species_basis(V, ndn)
        self.Du, self.Dd = len(self.su), len(self.sd)
        self.D = self.Du * self.Dd
        Hu = species_hop(V, nup, self.T)
        Hd = species_hop(V, ndn, self.T)
        self.Hu, self.Hd = Hu, Hd
        dbl = (self.occ_u.astype(float) @ self.occ_d.T.astype(float)).ravel()  # index iu*Dd+id
        self.dbl = dbl
        self.H = (sp.kron(Hu, sp.identity(self.Dd)) + sp.kron(sp.identity(self.Du), Hd)
                  + sp.diags(U * dbl)).tocsr()

    def iu_id(self):
        iu = np.repeat(np.arange(self.Du), self.Dd)
        idd = np.tile(np.arange(self.Dd), self.Du)
        return iu, idd

    # pairing ---------------------------------------------------------------
    def pair_setup(self, r_min):
        if self.nup < 1 or self.ndn < 1:
            self.pair_ops = []
            return
        V = self.V
        au = [species_annihilate(V, self.nup, i) for i in range(V)]
        ad = [species_annihilate(V, self.ndn, i) for i in range(V)]
        self.pair_ops = []
        self.pair_bonds = []
        b0 = next(b for b in self.bonds if b[2] == "x" and b[0] == 0) if any(
            b[2] == "x" for b in self.bonds) else self.bonds[0]
        self.b0 = b0
        c0 = self.bond_center(b0)
        for b in self.bonds:
            i, j, _ = b
            self.pair_ops.append((au[i], ad[j], au[j], ad[i]))
            self.pair_bonds.append(b)
        self.pair_dist = np.array([self.dist(c0, self.bond_center(b)) for b in self.pair_bonds])
        self.pair_mask = self.pair_dist >= r_min
        self.ib0 = self.pair_bonds.index(b0)

    def bond_center(self, b):
        i, j, d = b
        ci = self.coords[i]
        dx = min_image(self.coords[j] - ci, np.array([self.Lx, self.Ly]), "pbc")
        if self.bcx == "obc":
            dx[0] = self.coords[j][0] - ci[0]
        if self.bcy == "obc":
            dx[1] = self.coords[j][1] - ci[1]
        return ci + 0.5 * dx

    def dist(self, a, b):
        d = b - a
        d0 = min_image(d[0], self.Lx, self.bcx)
        d1 = min_image(d[1], self.Ly, self.bcy)
        return float(np.hypot(d0, d1))

    def pairing(self, psi):
        """Return vector P(b0, b) for all bonds b."""
        if not self.pair_ops:
            return np.zeros(0)
        Psi = (psi / np.linalg.norm(psi)).reshape(self.Du, self.Dd)
        vecs = []
        for (aui, adj, auj, adi) in self.pair_ops:
            # Delta_b ~ (a_i^up (x) a_j^dn + a_j^up (x) a_i^dn)/sqrt2 (global sign irrelevant)
            M = (aui @ (adj @ Psi.T).T + auj @ (adi @ Psi.T).T) / np.sqrt(2.0)
            vecs.append(np.asarray(M).ravel())
        v0 = vecs[self.ib0]
        return np.array([v0 @ v for v in vecs])


# ----------------------------------------------------------------------------- ED
def solve_ed(model, k, tol_deg, seed):
    rng = np.random.default_rng(seed)
    k = min(k, model.D - 1) if model.D > 2 else 1
    if model.D <= 400:
        w, v = np.linalg.eigh(model.H.toarray())
        w, v = w[:max(k, 1)], v[:, :max(k, 1)]
    else:
        w, v = eigsh(model.H, k=k, which="SA", v0=rng.standard_normal(model.D), tol=1e-12)
        o = np.argsort(w); w, v = w[o], v[:, o]
    deg = int(np.sum(w - w[0] < tol_deg))
    return w, v, deg


# ----------------------------------------------------------------------------- UHF
def uhf_orbitals(model, seed, iters=2000, mix=0.3):
    rng = np.random.default_rng(seed)
    V, T, U = model.V, model.T, model.U
    stag = np.array([(-1) ** int(x + y) for x, y in model.coords])
    nu = (model.nup / V) * (1 + 0.3 * stag) + 0.01 * rng.standard_normal(V)
    nd = (model.ndn / V) * (1 - 0.3 * stag) + 0.01 * rng.standard_normal(V)
    for _ in range(iters):
        eu, vu = np.linalg.eigh(-T + np.diag(U * nd))
        ed, vd = np.linalg.eigh(-T + np.diag(U * nu))
        Pu = vu[:, :model.nup]; Pd = vd[:, :model.ndn]
        nu_new = np.sum(Pu ** 2, 1); nd_new = np.sum(Pd ** 2, 1)
        if max(np.abs(nu_new - nu).max(), np.abs(nd_new - nd).max()) < 1e-10:
            nu, nd = nu_new, nd_new
            break
        nu = (1 - mix) * nu + mix * nu_new
        nd = (1 - mix) * nd + mix * nd_new
    return Pu.copy(), Pd.copy()


# ----------------------------------------------------------------------------- ansatz
class Ansatz:
    """Backflow-Slater(-Jastrow) ansatz, evaluated on the full basis.

    kind = "slater" (Phi0 only, frozen; mean-field), "sj" (Phi0 + Jastrow),
    "nnbf" (Phi0 + MLP backflow, optional Jastrow)."""

    def __init__(self, model, Pu, Pd, kind, width=0, jastrow=False, seed=0):
        self.m = model
        self.kind, self.H, self.jas = kind, width, jastrow
        rng = np.random.default_rng(seed)
        V, nu, nd = model.V, model.nup, model.ndn
        self.V, self.nu, self.nd = V, nu, nd
        iu, idd = model.iu_id()
        self.iu, self.idd = iu, idd
        # Jastrow features
        dists = sorted({round(model.dist(model.coords[i], model.coords[j]), 6)
                        for i in range(V) for j in range(i + 1, V)})
        self.dclass = dists
        p = {"Pu": Pu.copy(), "Pd": Pd.copy()}
        if jastrow:
            p["vj"] = np.zeros(len(dists) + 1)
        if kind == "nnbf":
            nin, nout = 2 * V, V * (nu + nd)
            p["W1"] = rng.standard_normal((nin, width)) / np.sqrt(nin)
            p["b1"] = np.zeros(width)
            p["W2"] = 1e-3 * rng.standard_normal((width, nout))
            p["b2"] = np.zeros(nout)
        self.p = p
        self.keys = list(p.keys())
        self._jfeat_cache = None

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def flat(self):
        return np.concatenate([self.p[k].ravel() for k in self.keys])

    def set_flat(self, x):
        o = 0
        for k in self.keys:
            n = self.p[k].size
            self.p[k] = x[o:o + n].reshape(self.p[k].shape); o += n

    def _xin(self, sl):
        m = self.m
        return np.concatenate([m.occ_u[self.iu[sl]], m.occ_d[self.idd[sl]]], 1).astype(float) * 2 - 1

    def _jfeat(self, sl):
        m = self.m
        nu = m.occ_u[self.iu[sl]].astype(float)
        nd = m.occ_d[self.idd[sl]].astype(float)
        n = nu + nd
        F = np.zeros((n.shape[0], len(self.dclass) + 1))
        for i in range(self.V):
            for j in range(i + 1, self.V):
                c = self.dclass.index(round(m.dist(m.coords[i], m.coords[j]), 6))
                F[:, c] += n[:, i] * n[:, j]
        F[:, -1] = np.sum(nu * nd, 1)
        return F

    def _orbitals(self, sl):
        """Return per-config orbitals (B, V, nu), (B, V, nd) and hidden activations."""
        B = sl.stop - sl.start
        Pu, Pd = self.p["Pu"], self.p["Pd"]
        if self.kind != "nnbf":
            return (np.broadcast_to(Pu, (B,) + Pu.shape), np.broadcast_to(Pd, (B,) + Pd.shape),
                    None, None)
        x = self._xin(sl)
        h = np.tanh(x @ self.p["W1"] + self.p["b1"])
        out = h @ self.p["W2"] + self.p["b2"]
        V, nu, nd = self.V, self.nu, self.nd
        Ou = Pu[None] + out[:, :V * nu].reshape(B, V, nu)
        Od = Pd[None] + out[:, V * nu:].reshape(B, V, nd)
        return Ou, Od, x, h

    def _mats(self, Ou, Od, sl):
        ru = self.m.rows_u[self.iu[sl]]
        rd = self.m.rows_d[self.idd[sl]]
        Au = np.take_along_axis(Ou, ru[:, :, None], axis=1)
        Ad = np.take_along_axis(Od, rd[:, :, None], axis=1)
        return Au, Ad, ru, rd

    def amplitudes(self, chunk):
        out = np.empty(self.m.D)
        for s in range(0, self.m.D, chunk):
            sl = slice(s, min(s + chunk, self.m.D))
            Ou, Od, _, _ = self._orbitals(sl)
            Au, Ad, _, _ = self._mats(Ou, Od, sl)
            a = np.linalg.det(Au) * np.linalg.det(Ad)
            if self.jas:
                a = a * np.exp(self._jfeat(sl) @ self.p["vj"])
            out[sl] = a
        return out

    @staticmethod
    def _inv_T(A):
        """A^{-T} with tiny regularisation for (near-)singular matrices."""
        try:
            return np.swapaxes(np.linalg.inv(A), 1, 2)
        except np.linalg.LinAlgError:
            eps = 1e-12 * np.eye(A.shape[-1])
            return np.swapaxes(np.linalg.inv(A + eps), 1, 2)

    def _local_grads(self, sl, w):
        """Return gradient of sum_n w(n) log psi(n) (flattened, summed) if w is 1-D
        weights; implemented through dlogpsi/dA = A^{-T}."""
        B = sl.stop - sl.start
        Ou, Od, x, h = self._orbitals(sl)
        Au, Ad, ru, rd = self._mats(Ou, Od, sl)
        Gu = self._inv_T(Au) * w[:, None, None]
        Gd = self._inv_T(Ad) * w[:, None, None]
        dOu = np.zeros((B, self.V, self.nu)); dOd = np.zeros((B, self.V, self.nd))
        bi = np.arange(B)[:, None]
        np.add.at(dOu, (bi, ru), Gu)
        np.add.at(dOd, (bi, rd), Gd)
        g = {"Pu": dOu.sum(0), "Pd": dOd.sum(0)}
        if self.jas:
            g["vj"] = self._jfeat(sl).T @ w
        if self.kind == "nnbf":
            dout = np.concatenate([dOu.reshape(B, -1), dOd.reshape(B, -1)], 1)
            g["W2"] = h.T @ dout
            g["b2"] = dout.sum(0)
            dpre = (dout @ self.p["W2"].T) * (1 - h ** 2)
            g["W1"] = x.T @ dpre
            g["b1"] = dpre.sum(0)
        return g

    def grad_weighted(self, w_full, chunk):
        """sum_n w(n) dlog psi(n)/dtheta, exact over the full basis."""
        tot = None
        for s in range(0, self.m.D, chunk):
            sl = slice(s, min(s + chunk, self.m.D))
            g = self._local_grads(sl, w_full[sl])
            tot = g if tot is None else {k: tot[k] + g[k] for k in tot}
        return np.concatenate([tot[k].ravel() for k in self.keys])

    def per_sample_O_fast(self, idx):
        """Vectorised per-sample log-derivatives (gather configs into a fake slice)."""
        # build a temporary view by remapping iu/idd
        iu0, id0 = self.iu, self.idd
        self.iu, self.idd = iu0[idx], id0[idx]
        try:
            B = len(idx)
            sl = slice(0, B)
            Ou, Od, x, h = self._orbitals(sl)
            Au, Ad, ru, rd = self._mats(Ou, Od, sl)
            Gu = self._inv_T(Au); Gd = self._inv_T(Ad)
            dOu = np.zeros((B, self.V, self.nu)); dOd = np.zeros((B, self.V, self.nd))
            bi = np.arange(B)[:, None]
            np.add.at(dOu, (bi, ru), Gu)
            np.add.at(dOd, (bi, rd), Gd)
            parts = {"Pu": dOu.reshape(B, -1), "Pd": dOd.reshape(B, -1)}
            if self.jas:
                parts["vj"] = self._jfeat(sl)
            if self.kind == "nnbf":
                dout = np.concatenate([dOu.reshape(B, -1), dOd.reshape(B, -1)], 1)
                parts["W2"] = (h[:, :, None] * dout[:, None, :]).reshape(B, -1)
                parts["b2"] = dout
                dpre = (dout @ self.p["W2"].T) * (1 - h ** 2)
                parts["W1"] = (x[:, :, None] * dpre[:, None, :]).reshape(B, -1)
                parts["b1"] = dpre
            return np.concatenate([parts[k] for k in self.keys], 1)
        finally:
            self.iu, self.idd = iu0, id0


def energy(H, psi):
    Hp = H @ psi
    nn = psi @ psi
    return float(psi @ Hp / nn), Hp, nn


def optimize_sr(ans, H, args, rng, label):
    """Stochastic reconfiguration with EXACT force and a sampled metric (Woodbury)."""
    best_E, best_x = np.inf, ans.flat().copy()
    for it in range(args.sr_iters):
        psi = ans.amplitudes(args.chunk)
        E, Hp, nn = energy(H, psi)
        if not np.isfinite(E):
            log(f"{label}: non-finite energy at it {it}; restoring best")
            ans.set_flat(best_x); break
        if E < best_E:
            best_E, best_x = E, ans.flat().copy()
        # exact force f_k = <O_k (E_loc - E)> = sum psi (Hpsi - E psi) O / ||psi||^2
        w = psi * (Hp - E * psi) / nn  # weight multiplying dlog psi
        f = ans.grad_weighted(w, args.chunk)
        p = psi ** 2 / nn
        idx = rng.choice(len(psi), size=args.sr_samples, p=p)
        O = ans.per_sample_O_fast(idx)
        Oc = (O - O.mean(0)) / np.sqrt(len(idx))
        lam = args.sr_shift
        Kmat = Oc @ Oc.T + lam * np.eye(len(idx))
        dx = (f - Oc.T @ np.linalg.solve(Kmat, Oc @ f)) / lam
        nrm = np.linalg.norm(dx)
        step = args.sr_lr * dx
        if nrm * args.sr_lr > 1.0:  # trust region on parameter step
            step *= 1.0 / (nrm * args.sr_lr)
        ans.set_flat(ans.flat() - step)
        if it % 50 == 0 or it == args.sr_iters - 1:
            log(f"{label}: it {it:5d}  E = {E:.8f}  |f| = {np.linalg.norm(f):.2e}")
    psi = ans.amplitudes(args.chunk)
    E, _, _ = energy(H, psi)
    if E > best_E:
        ans.set_flat(best_x)
    return ans


# ----------------------------------------------------------------------------- sign budget
def sign_constrained_min(K, starts, maxiter):
    """min_{a>=0} a^T K a / a^T a by L-BFGS-B from several starts. Returns (E, a, list)."""
    res_all = []
    best = (np.inf, None)

    def fg(a):
        Ka = K @ a
        nn = a @ a
        R = a @ Ka / nn
        return R, 2.0 * (Ka - R * a) / nn

    for name, a0 in starts:
        a0 = np.abs(a0) / np.linalg.norm(a0)
        r = minimize(fg, a0, jac=True, method="L-BFGS-B",
                     bounds=[(0.0, None)] * len(a0),
                     options=dict(maxiter=maxiter, ftol=1e-15, gtol=1e-12, maxcor=30))
        res_all.append((name, float(r.fun), int(r.nit)))
        if r.fun < best[0]:
            best = (float(r.fun), r.x / np.linalg.norm(r.x))
    return best[0], best[1], res_all


def stoq_bound(K):
    Kc = K.tocoo()
    keep = (Kc.row == Kc.col) | (Kc.data < 0)
    Ks = sp.csr_matrix((Kc.data[keep], (Kc.row[keep], Kc.col[keep])), shape=K.shape)
    if K.shape[0] <= 400:
        return float(np.linalg.eigvalsh(Ks.toarray())[0])
    return float(eigsh(Ks, k=1, which="SA", tol=1e-10)[0][0])


def ground_ref(psi_t, W, Vv, deg):
    """psi_0 aligned with psi_t inside the degenerate ground manifold."""
    G = Vv[:, :deg]
    c = G.T @ psi_t
    if np.linalg.norm(c) < 1e-14:
        c = np.zeros(deg); c[0] = 1.0
    p0 = G @ c
    return p0 / np.linalg.norm(p0)


def budget(model, psi_t, W, Vv, deg, args, rng):
    H = model.H
    psi_t = psi_t / np.linalg.norm(psi_t)
    p0 = ground_ref(psi_t, W, Vv, deg)
    if p0 @ psi_t < 0:
        p0 = -p0
    E0 = float(W[0])
    Et, _, _ = energy(H, psi_t)
    s_t = np.where(psi_t >= 0, 1.0, -1.0)
    s_0 = np.where(p0 >= 0, 1.0, -1.0)
    S = sp.diags(s_t)
    K = (S @ H @ S).tocsr()
    uni = np.ones(model.D) + 0.01 * rng.random(model.D)
    E_sign, a_star, starts = sign_constrained_min(
        K, [("abs_theta", psi_t), ("abs_exact", p0), ("uniform", uni)], args.lbfgs_iters)
    E_stoq = stoq_bound(K)
    dE = Et - E0
    E_h1, _, _ = energy(H, s_t * np.abs(p0))
    E_h2, _, _ = energy(H, s_0 * np.abs(psi_t))
    w_sign = float(np.sum(p0[s_t * s_0 < 0] ** 2))
    P0 = model.pairing(p0); Pt = model.pairing(psi_t); Ps = model.pairing(s_t * a_star)
    if len(P0) and model.pair_mask.any():
        msk = model.pair_mask
        eC_tot = float(np.max(np.abs(Pt - P0)[msk])); eC_sign = float(np.max(np.abs(Ps - P0)[msk]))
    else:
        eC_tot = eC_sign = float("nan")
    return dict(E=Et, E0=E0, dE_tot=dE, E_sign=E_sign, E_stoq=E_stoq,
                dE_sign=E_sign - E0, dE_amp=Et - E_sign,
                f_E=(E_sign - E0) / dE if dE > 0 else float("nan"),
                f_E_lo=max(0.0, E_stoq - E0) / dE if dE > 0 else float("nan"),
                E_hyb_signtheta_ampexact=E_h1, E_hyb_signexact_amptheta=E_h2,
                w_sign=w_sign, eC_tot=eC_tot, eC_sign=eC_sign,
                f_C=eC_sign / eC_tot if eC_tot > 0 else float("nan"),
                P0=P0.tolist(), Ptheta=Pt.tolist(), Psign=Ps.tolist(),
                sign_starts=starts)


# ----------------------------------------------------------------------------- checks
def run_checks(seed):
    ok = True
    # (a) two-site Hubbard, one up one down, open: E0 = (U - sqrt(U^2+16 t^2))/2
    for U in (0.0, 4.0, 8.0):
        m = Hubbard(2, 1, "obc", "obc", 1, 1, 1.0, 0.0, U)
        E = np.linalg.eigvalsh(m.H.toarray())[0]
        ex = 0.5 * (U - np.sqrt(U * U + 16.0))
        log(f"check two-site U={U}: ED {E:.12f}  analytic {ex:.12f}")
        ok &= abs(E - ex) < 1e-10
    # (b) U=0 Slater determinant energy from lowest orbitals == U=0 ED (tests JW signs)
    for (Lx, Ly, bcx, bcy, nu, nd, tp) in [(2, 3, "pbc", "obc", 2, 1, -0.25),
                                           (3, 2, "obc", "pbc", 2, 2, 0.3)]:
        m = Hubbard(Lx, Ly, bcx, bcy, nu, nd, 1.0, tp, 0.0)
        e, v = np.linalg.eigh(-m.T)
        Esp = e[:nu].sum() + e[:nd].sum()
        a = Ansatz(m, v[:, :nu], v[:, :nd], "slater")
        psi = a.amplitudes(4096)
        Es, _, _ = energy(m.H, psi)
        Eed = solve_ed(m, 2, 1e-8, seed)[0][0]
        log(f"check U=0 {Lx}x{Ly} {bcx}/{bcy} N=({nu},{nd}) t'={tp}: sum eps {Esp:.12f} "
            f"Slater {Es:.12f} ED {Eed:.12f}")
        ok &= abs(Es - Esp) < 1e-10 and abs(Eed - Esp) < 1e-10
    # (c) gradient check of NNBF+Jastrow against finite differences
    m = Hubbard(2, 3, "pbc", "obc", 2, 2, 1.0, -0.25, 4.0)
    Pu, Pd = uhf_orbitals(m, seed)
    a = Ansatz(m, Pu, Pd, "nnbf", width=4, jastrow=True, seed=seed)
    a.p["W2"] *= 100
    x0 = a.flat().copy()
    psi = a.amplitudes(4096); E, Hp, nn = energy(m.H, psi)
    g = 2 * a.grad_weighted(psi * (Hp - E * psi) / nn, 4096)
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(len(x0)); d /= np.linalg.norm(d)
    h = 1e-6
    a.set_flat(x0 + h * d); Ep = energy(m.H, a.amplitudes(4096))[0]
    a.set_flat(x0 - h * d); Em = energy(m.H, a.amplitudes(4096))[0]
    a.set_flat(x0)
    fd = (Ep - Em) / (2 * h)
    log(f"check gradient: analytic {g @ d:.10e}  finite-diff {fd:.10e}")
    ok &= abs(g @ d - fd) < 1e-6 * max(1.0, abs(fd))
    log(f"ALL CHECKS {'PASSED' if ok else 'FAILED'}")
    return ok


# ----------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.1 sign-vs-amplitude error budget (see module docstring)")
    ap.add_argument("--instance", type=int, default=-1, choices=[-1, 0, 1, 2, 3])
    ap.add_argument("--Lx", type=int, default=4)
    ap.add_argument("--Ly", type=int, default=3)
    ap.add_argument("--bcx", choices=["pbc", "obc"], default="pbc")
    ap.add_argument("--bcy", choices=["pbc", "obc"], default="pbc")
    ap.add_argument("--nup", type=int, default=5)
    ap.add_argument("--ndn", type=int, default=5)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--tp", type=float, default=-0.25)
    ap.add_argument("--U", type=float, default=8.0)
    ap.add_argument("--widths", type=str, default="8,32,128")
    ap.add_argument("--nnbf-jastrow", type=int, choices=[0, 1], default=0)
    ap.add_argument("--sr-iters", type=int, default=1500)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--lbfgs-iters", type=int, default=4000)
    ap.add_argument("--n-ed-states", type=int, default=3)
    ap.add_argument("--degeneracy-tol", type=float, default=1e-8)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    ap.add_argument("--checks-only", type=int, choices=[0, 1], default=0)
    ap.add_argument("--aggregate", type=str, default="")
    a = ap.parse_args(argv)
    if a.instance >= 0:
        for k, v in INSTANCES[a.instance].items():
            setattr(a, k, v)
    a.width_list = [int(w) for w in a.widths.split(",") if w.strip()]
    if not a.out:
        tag = (f"L{a.Lx}x{a.Ly}_{a.bcx}{a.bcy}_n{a.nup}-{a.ndn}_U{a.U:g}_tp{a.tp:g}"
               f"_w{'-'.join(map(str, a.width_list)) or 'none'}_j{a.nnbf_jastrow}_s{a.seed}")
        a.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", tag + ".json")
    return a


def aggregate(d):
    import glob
    Ms = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        if "M" not in r:
            continue
        w = r["params"]["width_list"][-1]
        st = r["states"][f"nnbf_{w}"]
        Ms.append(r["M"])
        print(f"{os.path.basename(f)}: M={r['M']:.4f} f_E_lo={st['f_E_lo']:.4f} f_C={st['f_C']:.4f} "
              f"dE_tot={st['dE_tot']:.5f} calib_err={r['calib_err']:.2e}")
    if not Ms:
        print("no finished results"); return 1
    v = "PASS" if min(Ms) >= 0.6 else ("FAIL" if max(Ms) <= 0.3 else "AMBIGUOUS")
    print(f"GLOBAL VERDICT over {len(Ms)} instance(s): {v}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        return aggregate(args.aggregate)
    log("args: " + json.dumps({k: v for k, v in vars(args).items()}))
    if not run_checks(args.seed):
        log("correctness checks failed; aborting")
        return 2
    if args.checks_only:
        return 0
    if os.path.exists(args.out) and not args.overwrite:
        log(f"output {args.out} exists; use --overwrite 1 to recompute. Exiting.")
        return 0
    rng = np.random.default_rng(args.seed)
    m = Hubbard(args.Lx, args.Ly, args.bcx, args.bcy, args.nup, args.ndn, args.t, args.tp, args.U)
    m.pair_setup(args.r_min)
    log(f"model V={m.V} N=({m.nup},{m.ndn}) D={m.D} nnz(H)={m.H.nnz}")
    W, Vv, deg = solve_ed(m, args.n_ed_states, args.degeneracy_tol, args.seed)
    log(f"ED levels {W.tolist()}  ground degeneracy {deg}")
    res = dict(params=vars(args), ed=dict(levels=W.tolist(), degeneracy=deg), states={})
    # optimizer calibration with exact signs
    p0 = Vv[:, 0]
    S0 = sp.diags(np.where(p0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0 @ m.H @ S0).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng.random(m.D))],
                                    args.lbfgs_iters)
    res["calib_err"] = Ec - W[0]
    log(f"calibration: E_sign(s_0) from uniform start - E0 = {res['calib_err']:.3e}")

    Pu, Pd = uhf_orbitals(m, args.seed)
    states = []
    a = Ansatz(m, Pu, Pd, "slater")
    states.append(("uhf", a))
    a = Ansatz(m, Pu, Pd, "sj", jastrow=True, seed=args.seed)
    optimize_sr(a, m.H, args, rng, "sj")
    states.append(("sj", a))
    for w in args.width_list:
        a = Ansatz(m, Pu, Pd, "nnbf", width=w, jastrow=bool(args.nnbf_jastrow), seed=args.seed + w)
        optimize_sr(a, m.H, args, rng, f"nnbf_{w}")
        states.append((f"nnbf_{w}", a))
    for name, a in states:
        psi = a.amplitudes(args.chunk)
        b = budget(m, psi, W, Vv, deg, args, rng)
        b["n_params"] = a.n_params()
        res["states"][name] = b
        log(f"{name:>10s}: dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} "
            f"dE_amp={b['dE_amp']:.5f} f_E={b['f_E']:.3f} f_E_lo={b['f_E_lo']:.3f} "
            f"w_sign={b['w_sign']:.3e} eC_tot={b['eC_tot']:.3e} eC_sign={b['eC_sign']:.3e} "
            f"f_C={b['f_C']:.3f}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    if args.width_list:
        M = res["states"][f"nnbf_{args.width_list[-1]}"]["f_E"]
        res["M"] = M
        res["verdict_instance"] = "PASS" if M >= 0.6 else ("FAIL" if M <= 0.3 else "AMBIGUOUS")
        log(f"M = f_E(nnbf_{args.width_list[-1]}) = {M:.4f} -> instance verdict {res['verdict_instance']}")
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
