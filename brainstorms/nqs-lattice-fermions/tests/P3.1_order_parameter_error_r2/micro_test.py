#!/usr/bin/env python3
"""
P3.1 round-2 decisive test: can a cheap, VMC-accessible quantity predict the error of the
long-range d-wave pair correlation of REAL trained fermionic NQS (NNBF) and of the
Slater-Jastrow foil, better than the one-Lanczos-step baseline?
=====================================================================================

Population (no synthetic contaminations)
----------------------------------------
For each of the three P1.1 instances (4x3 PBC, t'=-0.25; 0: N_up=N_dn=5, U=8;
1: N_up=N_dn=4, U=8, has a same-sector near-degenerate competitor, Delta_1/V=0.007t;
2: N_up=N_dn=5, U=4) the script retrains, with the IDENTICAL P1.1 code, seed and
hyper-parameters (exact-force SR, 1500 iterations), the Slater-Jastrow foil "sj" and
NNBF of hidden widths 8, 32, 128 (Luo & Clark backflow). The P1.1 random-number
consumption is replicated, so the final states reproduce the P1.1 states (their
dE_tot is checked against the P1.1 JSON when present). Each network contributes
SNAPSHOTS: the current SR state at iterations round(f * sr_iters) for f in
--snap-fracs (default 0.2, 0.5) plus the final (best-energy) state, i.e. 4 networks
x 3 snapshots = 12 trained states per instance, 36 in total (9 foil, 27 NNBF).

Observable (off-diagonal only, as restricted by the critic)
-----------------------------------------------------------
A = (1/N_pairs) sum_{(b0,b) ordered, d(b0,b) >= r_min} eta_b0 eta_b Delta_b0^dag Delta_b,
Delta_b = (c_{j dn} c_{i up} + c_{i dn} c_{j up})/sqrt2 for the nearest-neighbour bond
b=(i,j) (singlet; same convention as P1.1), eta=+1 for x-bonds and -1 for y-bonds. This
is the translation- and orientation-averaged long-range d-wave pair correlation; for a
symmetric psi_0 it equals P1.1's Pd. a(psi) = <psi|A|psi>/<psi|psi>, a_0 = a(psi_0).
TRUE error delta = a_0 - a(psi) (exact).

Arms (all predict delta; "phi" is the improved state an arm produces)
--------------------------------------------------------------------
  L1  baseline : lowest Ritz vector of H in span{psi, H psi}            (one Lanczos step)
  L2  baseline : lowest Ritz vector of H in span{psi, H psi, H^2 psi}   (two steps; costs
                 >= the routes per sample: matched-or-larger budget control)
  AR  route 2.1: lowest Ritz vector in span{psi, H psi, A psi}          (A-Ritz, real states)
  SR  route 2.2: lowest Ritz vector in span{psi, H psi, P_F psi}        (sign-defect Ritz)
                 P_F psi = psi restricted to the locally sign-inconsistent set
                 F_tau = {n : kappa(n) > tau},  kappa(n) = (H_off psi)(n)/psi(n),
                 H_off = H - diag(H). For the exact ground state kappa(n) = E_0 - H_nn < 0
                 for every n (E_0 < 0 <= H_nn), so F is empty: sign-inconsistency is an
                 exact zero-bias sign witness. tau is chosen from --taus (units of t) by
                 minimal Ritz energy.
  AS  secondary: span{psi, H psi, A psi, P_F psi} (reported, not in any verdict)
  XW  control  : cross-width zero-variance extrapolation (current NQS practice): least-
                 squares line a_k = a* + c sigma2_k through the three NNBF widths at the
                 same snapshot, dhat = a* - a(widest). Defined only for the widest NNBF.
For L1, L2, AR, SR, AS the pre-committed first-order extrapolation is
    rhat = 1 - sigma_phi / sigma_psi    (fraction of the error vector removed),
    dhat = (a_phi - a_psi) / rhat,
which follows from psi = psi_0 + e, phi ~ psi_0 + (1 - r) e (a is FIRST order in e,
sigma is first order in e, so the linear-in-variance rule is not used).

Trust rule (redesigned; abstention is penalised)
------------------------------------------------
An arm issues a certificate ("trusted") iff rhat >= --r-min-trust (default 0.05, i.e. the
step removes at least ~10% of the variance, not 50%) AND, in Monte Carlo mode, the
bootstrap relative standard error of dhat is at most --max-rel-se (default 0.5).
XW is always "trusted" (that is how it is used in practice).
A scored state (|delta| > 0.05 |a_0|) is
    tracked  iff trusted and 1/2 <= dhat/delta <= 2,
    false    iff trusted and dhat/delta < 1/2 (under-statement, includes wrong sign).
Metrics over ALL scored states (abstentions count as NOT tracked):
    M_X = #tracked / #scored,  G_X = #false / #scored,  Cov_X = #trusted NNBF / #NNBF.
Spearman rank correlation of |dhat| with |delta| (no abstention) is reported.

Pre-committed verdict (per route X in {AR, SR}; B = max(M_L1, M_L2); primary mode =
Monte Carlo with --mc-samples samples drawn exactly from |psi|^2, the default 200000):
    PASS      M_X >= 0.5  and  M_X - B >= 0.2  and  G_X <= 0.10  and  Cov_X >= 0.5
    FAIL      M_X <  0.3  or   M_X - B <  0.05 or   G_X >  0.20
    AMBIGUOUS otherwise -> pre-committed action: rerun (same training seed) with
              --mc-samples 2000000 --mc-seed 4321; a route that is not PASS then is KILLED.
Pre-registered prediction (P1.1 transfer, not a route verdict): XW under-states the
error (dhat/delta < 1/2) on at least 2/3 of the widest-NNBF states, because width
shrinks the variance (amplitude error) but not the sign-dominated pair error.
Exact-matrix-element results (--mc-samples 0 equivalent) are always also reported.

Monte Carlo emulation
---------------------
Every matrix element is estimated as a VMC code would: n ~ |psi|^2 (exact sampling,
no autocorrelation), <v_i|O|v_j> ~ mean[(v_i(n)/psi(n)) (O v_j)(n)/psi(n)] with the
local ratios read from exactly computed full vectors (so heavy tails of second-order
estimators such as <H A psi|H A psi> are present). Bootstrap over samples gives the
relative standard error. The P_F / tau selection is redone inside MC mode.

Usage
-----
    python3 micro_test.py [options]
    python3 micro_test.py --aggregate results          # verdict over finished JSONs

Options
-------
  --instance K        0, 1, 2 (P1.1 instances above) or -1 (default -1: explicit lattice
                      options below are used).
  --Lx INT, --Ly INT  Lattice size (defaults 4, 3); integers >= 2 (Ly >= 1 for checks).
  --bcx, --bcy        "pbc" or "obc" (default pbc, pbc).
  --nup, --ndn INT    Particle numbers (default 5, 5); both >= 1.
  --t, --tp, --U      Hopping t (1.0), diagonal hopping t' (-0.25), on-site U (8.0); floats.
  --widths LIST       Comma-separated NNBF hidden widths (default "8,32,128"); >= 1 entry;
                      XW needs >= 2 widths and uses all of them, the last is "widest".
  --sr-iters INT      SR iterations per network (default 1500, as P1.1); >= 2.
  --sr-lr FLOAT       SR step (default 0.05).  --sr-shift FLOAT  SR diagonal shift (1e-3).
  --sr-samples INT    Samples for the SR metric (default 2000).
  --chunk INT         Configurations per forward chunk (default 65536).
  --snap-fracs LIST   Comma-separated fractions in (0,1) of --sr-iters at which snapshots
                      are taken (default "0.2,0.5"); "" = final states only.
  --taus LIST         Comma-separated non-negative thresholds tau (units of t) for the
                      sign-defect set F_tau (default "0,0.5,1,2").
  --r-min FLOAT       Minimal bond-centre distance in A (default 1.5, as P1.1).
  --mc-samples INT    Samples per state for the primary MC mode (default 200000);
                      0 = exact mode only (then the verdict uses exact mode).
  --n-boot INT        Bootstrap resamples (default 100); >= 2.
  --mc-seed INT       Seed of the MC sampling / bootstrap stream (default 99).
  --r-min-trust FLOAT Trust threshold on rhat (default 0.05); in [0, 1).
  --max-rel-se FLOAT  Trust threshold on bootstrap relative SE of dhat (default 0.5); > 0.
  --score-floor FLOAT States with |delta| <= floor * |a_0| are not scored (default 0.05).
  --n-ed-states INT   Lowest ED eigenpairs (default 4; raised to >= 2): used for the
                      competitor-admixture diagnostic |<k|psi>|^2 and a(psi_k).
  --degeneracy-tol FLOAT  Ground-manifold energy window (default 1e-8).
  --seed INT          Training seed (default 1234, as P1.1).
  --p11-json PATH     P1.1 result JSON to cross-check final dE_tot (default: auto-located
                      in ../P1.1_sign_vs_amplitude/results by instance tag; "" disables).
  --out PATH          Output JSON (default results/<tag>.json next to this script).
                      Existing output is not overwritten unless --overwrite 1.
  --overwrite {0,1}   (default 0).
  --checks-only {0,1} Run only the correctness checks (P1.1 checks + A hermiticity + A
                      equals P1.1 Pd on a symmetric state) and exit (default 0).
  --aggregate DIR     No computation: pool every finished *.json in DIR, print per-arm
                      metrics for MC and exact mode, the XW prediction, and the verdicts.

Outputs
-------
JSON: "params", "ed" (levels, degeneracy, a_k for the lowest levels, Pd_P11 check),
"a0", "states" -> list of rows {name, net, iter, E, dE, sigma2, a, delta, fid_k (overlap
weights with the lowest ED states), w_sign (sign-mismatch weight vs psi_0), and per mode
("exact", "mc") per arm: E_phi, sigma2_phi, a_phi, rhat, dhat, rel_se, trusted, tau},
"xw" -> per snapshot {exact: dhat, mc: dhat}, "metrics" and "verdicts" for this file.
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
from scipy.sparse.linalg import eigsh

INSTANCES = {
    0: dict(Lx=4, Ly=3, bcx="pbc", bcy="pbc", nup=5, ndn=5, U=8.0, tp=-0.25),
    1: dict(Lx=4, Ly=3, bcx="pbc", bcy="pbc", nup=4, ndn=4, U=8.0, tp=-0.25),
    2: dict(Lx=4, Ly=3, bcx="pbc", bcy="pbc", nup=5, ndn=5, U=4.0, tp=-0.25),
}

# pre-committed verdict constants
PASS_M, PASS_GAIN, PASS_G, PASS_COV = 0.5, 0.2, 0.10, 0.5
FAIL_M, FAIL_GAIN, FAIL_G = 0.3, 0.05, 0.20
XW_PRED_FRAC = 2.0 / 3.0
ARMS = ["L1", "L2", "AR", "SR", "AS"]
ROUTES = ["AR", "SR"]


# ---- copied verbatim from tests/P1.1_sign_vs_amplitude/micro_test.py (lattice, ED, ansatz, checks)
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

    def dwave(self, psi):
        """d-wave long-range pair correlation: mean over bonds b with d(b, b0) >= r_min of
        eta_b P(b0, b), eta = +1 for x-bonds, -1 for y-bonds (b0 is an x-bond).
        Returns nan if there are no pairing operators or no such bonds."""
        if not self.pair_ops or not self.pair_mask.any():
            return float("nan")
        P = self.pairing(psi)
        eta = np.array([1.0 if b[2] == "x" else -1.0 for b in self.pair_bonds])
        if self.pair_bonds[self.ib0][2] != "x":
            eta = -eta
        return float(np.mean((eta * P)[self.pair_mask]))


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



# ============================================================================ new for P3.1 r2
class PairOp:
    """Translation- and orientation-averaged long-range d-wave pair correlation
    A = (1/N_pairs) sum_{(b0,b): d >= r_min} eta_b0 eta_b Delta_b0^dag Delta_b,
    applied matrix-free on the (D_up x D_dn) amplitude matrix. Needs model.pair_setup()."""

    def __init__(self, model, r_min):
        self.m = model
        bonds = model.pair_bonds
        nb = len(bonds)
        eta = np.array([1.0 if b[2] == "x" else -1.0 for b in bonds])
        cen = [model.bond_center(b) for b in bonds]
        W = np.zeros((nb, nb))
        for i in range(nb):
            for j in range(nb):
                if model.dist(cen[i], cen[j]) >= r_min:
                    W[i, j] = eta[i] * eta[j]
        self.n_pairs = int(np.count_nonzero(W))
        if self.n_pairs == 0:
            raise ValueError("no bond pairs with d >= r_min; lower --r-min")
        self.W = W / self.n_pairs
        self.ops = model.pair_ops

    def deltas(self, psi):
        Psi = psi.reshape(self.m.Du, self.m.Dd)
        return np.stack([(aui @ (adj @ Psi.T).T + auj @ (adi @ Psi.T).T) / np.sqrt(2.0)
                         for (aui, adj, auj, adi) in self.ops])

    def apply(self, psi):
        X = self.deltas(psi)
        Y = np.tensordot(self.W, X, axes=(1, 0))
        out = np.zeros((self.m.Du, self.m.Dd))
        for (aui, adj, auj, adi), Yb in zip(self.ops, Y):
            out += (aui.T @ (adj.T @ Yb.T).T + auj.T @ (adi.T @ Yb.T).T) / np.sqrt(2.0)
        return out.ravel()

    def brute(self, psi):
        """Reference value sum_{b0,b} W P(b0,b) via explicit bond vectors (checks)."""
        X = self.deltas(psi / np.linalg.norm(psi)).reshape(len(self.ops), -1)
        return float(np.sum(self.W * (X @ X.T)))


def optimize_sr_snap(ans, H, args, rng, label, snap_iters):
    """P1.1 optimize_sr (identical update and RNG use) plus snapshots of the CURRENT
    parameters at the iterations in snap_iters. Returns {iter: flat}, final = best."""
    snaps = {}
    best_E, best_x = np.inf, ans.flat().copy()
    for it in range(args.sr_iters):
        psi = ans.amplitudes(args.chunk)
        E, Hp, nn = energy(H, psi)
        if not np.isfinite(E):
            log(f"{label}: non-finite energy at it {it}; restoring best")
            ans.set_flat(best_x); break
        if it in snap_iters:
            snaps[it] = ans.flat().copy()
        if E < best_E:
            best_E, best_x = E, ans.flat().copy()
        w = psi * (Hp - E * psi) / nn
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
        if nrm * args.sr_lr > 1.0:
            step *= 1.0 / (nrm * args.sr_lr)
        ans.set_flat(ans.flat() - step)
        if it % 100 == 0 or it == args.sr_iters - 1:
            log(f"{label}: it {it:5d}  E = {E:.8f}")
    psi = ans.amplitudes(args.chunk)
    E, _, _ = energy(H, psi)
    if E > best_E:
        ans.set_flat(best_x)
    snaps["final"] = ans.flat().copy()
    return snaps


def sym(M):
    return 0.5 * (M + M.T)


def ritz(S, Hm, HH, AA, cond=1e-10):
    """Lowest Ritz vector of (Hm, S); returns (E, sigma2, a) of that vector."""
    lam, U = np.linalg.eigh(S)
    keep = lam > cond * lam.max()
    X = U[:, keep] / np.sqrt(lam[keep])
    e, y = np.linalg.eigh(sym(X.T @ Hm @ X))
    c = X @ y[:, 0]
    nn = c @ S @ c
    E = c @ Hm @ c / nn
    return float(E), float(c @ HH @ c / nn - E * E), float(c @ AA @ c / nn)


def arm_from_gram(G, idx, cond):
    S, Hm, HH, AA = (g[np.ix_(idx, idx)] for g in G)
    return ritz(S, Hm, HH, AA, cond)


def psi_stats(G):
    S, Hm, HH, AA = G
    E = Hm[0, 0] / S[0, 0]
    return float(E), float(HH[0, 0] / S[0, 0] - E * E), float(AA[0, 0] / S[0, 0])


def first_order(sig2_psi, sig2_phi, a_psi, a_phi):
    rhat = 1.0 - np.sqrt(max(sig2_phi, 0.0) / sig2_psi) if sig2_psi > 0 else float("nan")
    dhat = (a_phi - a_psi) / rhat if (np.isfinite(rhat) and rhat > 1e-12) else float("nan")
    return float(rhat), float(dhat)


def arm_indices(n_tau):
    """Master basis order: 0 psi, 1 H psi, 2 H^2 psi, 3 A psi, 4.. P_tau psi."""
    base = {"L1": [0, 1], "L2": [0, 1, 2], "AR": [0, 1, 3]}
    sr = {k: [0, 1, 4 + k] for k in range(n_tau)}
    asx = {k: [0, 1, 3, 4 + k] for k in range(n_tau)}
    return base, sr, asx


def evaluate_arms(G, n_tau, cond, tau_choice=None):
    """Point estimates of every arm from the master Gram tuple G. If tau_choice is None
    the SR tau is selected by minimal Ritz energy. Returns (dict arm->(E,s2,a), tau_k)."""
    base, sr, asx = arm_indices(n_tau)
    out = {k: arm_from_gram(G, v, cond) for k, v in base.items()}
    if tau_choice is None:
        res = [arm_from_gram(G, sr[k], cond) for k in range(n_tau)]
        tau_choice = int(np.argmin([r[0] for r in res]))
        out["SR"] = res[tau_choice]
    else:
        out["SR"] = arm_from_gram(G, sr[tau_choice], cond)
    out["AS"] = arm_from_gram(G, asx[tau_choice], cond)
    return out, tau_choice


def master_vectors(model, Aop, psi, taus):
    psi = psi / np.linalg.norm(psi)
    H = model.H
    Hp = H @ psi
    diag = H.diagonal()
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(psi != 0, (Hp - diag * psi) / psi, -np.inf)
    vecs = [psi, Hp, H @ Hp, Aop.apply(psi)]
    flip_w = []
    for tau in taus:
        F = kappa > tau
        vecs.append(np.where(F, psi, 0.0))
        flip_w.append(float(np.sum(psi[F] ** 2)))
    V = np.stack(vecs, 1)
    HV = np.stack([Hp, vecs[2], H @ vecs[2]] + [H @ v for v in vecs[3:]], 1)
    AV = np.stack([vecs[3]] + [Aop.apply(v) for v in vecs[1:]], 1)
    return V, HV, AV, flip_w


def analyze_state(model, Aop, psi, a0, args, mc_rng):
    V, HV, AV, flip_w = master_vectors(model, Aop, psi, args.tau_list)
    n_tau = len(args.tau_list)
    res = {"flip_weight": flip_w}
    # ---- exact mode
    G = (V.T @ V, sym(V.T @ HV), HV.T @ HV, sym(V.T @ AV))
    E, s2, a = psi_stats(G)
    arms, tk = evaluate_arms(G, n_tau, 1e-10)
    ex = {}
    for k, (Ep, s2p, ap) in arms.items():
        r, d = first_order(s2, s2p, a, ap)
        ex[k] = dict(E_phi=Ep, sigma2_phi=s2p, a_phi=ap, rhat=r, dhat=d, rel_se=0.0,
                     trusted=bool(np.isfinite(r) and r >= args.r_min_trust and np.isfinite(d)))
    ex["SR"]["tau"] = ex["AS"]["tau"] = args.tau_list[tk]
    res.update(E=E, sigma2=s2, a=a, exact=ex)
    if args.mc_samples <= 0:
        return res
    # ---- Monte Carlo mode
    p = V[:, 0] ** 2
    idx = mc_rng.choice(len(p), size=args.mc_samples, p=p / p.sum())
    w0 = V[idx, 0]
    R, Hl, Al = V[idx] / w0[:, None], HV[idx] / w0[:, None], AV[idx] / w0[:, None]
    N = args.mc_samples

    def gram(w):
        Rw = R * w[:, None]
        return (Rw.T @ R, sym(Rw.T @ Hl), (Hl * w[:, None]).T @ Hl, sym(Rw.T @ Al))

    Gm = gram(np.full(N, 1.0 / N))
    Em, s2m, am = psi_stats(Gm)
    arms_m, tkm = evaluate_arms(Gm, n_tau, 1e-8)
    point = {k: first_order(s2m, v[1], am, v[2]) for k, v in arms_m.items()}
    boot = {k: [] for k in arms_m}
    for _ in range(args.n_boot):
        w = mc_rng.multinomial(N, np.full(N, 1.0 / N)) / N
        Gb = gram(w)
        _, s2b, ab = psi_stats(Gb)
        arms_b, _ = evaluate_arms(Gb, n_tau, 1e-8, tau_choice=tkm)
        for k, v in arms_b.items():
            boot[k].append(first_order(s2b, v[1], ab, v[2])[1])
    mc = {}
    for k, (Ep, s2p, ap) in arms_m.items():
        r, d = point[k]
        b = np.array(boot[k], float)
        b = b[np.isfinite(b)]
        rse = float(np.std(b) / abs(d)) if (len(b) > 1 and np.isfinite(d) and d != 0) else float("inf")
        mc[k] = dict(E_phi=Ep, sigma2_phi=s2p, a_phi=ap, rhat=r, dhat=d, rel_se=rse,
                     trusted=bool(np.isfinite(r) and r >= args.r_min_trust and np.isfinite(d)
                                  and rse <= args.max_rel_se))
    mc["SR"]["tau"] = mc["AS"]["tau"] = args.tau_list[tkm]
    res.update(E_mc=Em, sigma2_mc=s2m, a_mc=am, mc=mc)
    return res


def xw_estimates(rows, widths, modes):
    """Cross-width zero-variance extrapolation per snapshot label."""
    out = {}
    labels = sorted({r["iter"] for r in rows if r["net"].startswith("nnbf")}, key=str)
    for lab in labels:
        sel = {r["net"]: r for r in rows if r["iter"] == lab and r["net"].startswith("nnbf")}
        nets = [f"nnbf_{w}" for w in widths]
        if not all(n in sel for n in nets) or len(nets) < 2:
            continue
        o = {"state": f"{nets[-1]}@{lab}"}
        for mode in modes:
            sk, ak = ("sigma2", "a") if mode == "exact" else ("sigma2_mc", "a_mc")
            x = np.array([sel[n][sk] for n in nets]); y = np.array([sel[n][ak] for n in nets])
            c1, c0 = np.polyfit(x, y, 1)
            o[mode] = float(c0 - y[-1])
        out[lab] = o
    return out


def is_nnbf(row):
    return row["net"].startswith("nnbf")


def compute_metrics(rows, xw_rows, mode):
    from scipy.stats import spearmanr
    scored = [r for r in rows if r["scored"] and mode in r]
    out = {"n_scored": len(scored), "n_nnbf": sum(is_nnbf(r) for r in scored)}
    for arm in ARMS:
        tr = fa = tru = tru_nn = 0
        dh, dt = [], []
        for r in scored:
            e = r[mode][arm]
            ratio = e["dhat"] / r["delta"] if np.isfinite(e["dhat"]) else float("nan")
            if e["trusted"]:
                tru += 1
                tru_nn += is_nnbf(r)
                if 0.5 <= ratio <= 2.0:
                    tr += 1
                elif not (ratio > 2.0):
                    fa += 1
            dh.append(abs(e["dhat"]) if np.isfinite(e["dhat"]) else 0.0)
            dt.append(abs(r["delta"]))
        n = max(len(scored), 1)
        rho = float(spearmanr(dh, dt)[0]) if len(scored) > 2 else float("nan")
        out[arm] = dict(M=tr / n, G=fa / n, trusted=tru / n,
                        Cov=tru_nn / max(out["n_nnbf"], 1), spearman=rho)
    under = n_xw = 0
    for x in xw_rows:
        if mode not in x or not x.get("scored", False):
            continue
        n_xw += 1
        ratio = x[mode] / x["delta"]
        under += not (ratio >= 0.5)
    out["XW"] = dict(n=n_xw, under=under, frac_under=under / max(n_xw, 1))
    return out


def verdicts(met):
    B = max(met["L1"]["M"], met["L2"]["M"])
    v = {}
    for X in ROUTES:
        m = met[X]
        gain = m["M"] - B
        if m["M"] >= PASS_M and gain >= PASS_GAIN and m["G"] <= PASS_G and m["Cov"] >= PASS_COV:
            v[X] = "PASS"
        elif m["M"] < FAIL_M or gain < FAIL_GAIN or m["G"] > FAIL_G:
            v[X] = "FAIL"
        else:
            v[X] = "AMBIGUOUS"
    x = met["XW"]
    v["XW_prediction"] = ("CONFIRMED" if x["n"] and x["frac_under"] >= XW_PRED_FRAC
                          else ("REFUTED" if x["n"] else "n/a"))
    return v


def print_metrics(met, mode, title):
    print(f"--- {title} [{mode}]  scored={met['n_scored']} (NNBF {met['n_nnbf']})")
    print("arm   M      G      trusted  Cov(NNBF)  spearman")
    for arm in ARMS:
        m = met[arm]
        print(f"{arm:4s} {m['M']:.3f}  {m['G']:.3f}  {m['trusted']:.3f}    {m['Cov']:.3f}      {m['spearman']:+.3f}")
    x = met["XW"]
    print(f"XW (cross-width extrapolation): under-states on {x['under']}/{x['n']} widest-NNBF states")


def report(rows, xw_rows, primary, title):
    out = {}
    for mode in ("exact", "mc"):
        if not any(mode in r for r in rows):
            continue
        met = compute_metrics(rows, xw_rows, mode)
        print_metrics(met, mode, title)
        out[mode] = dict(metrics=met, verdicts=verdicts(met))
    if primary in out:
        v = out[primary]["verdicts"]
        print(f"VERDICT ({primary} mode): AR (route 2.1 A-Ritz) = {v['AR']};  "
              f"SR (route 2.2 sign-defect Ritz) = {v['SR']};  XW prediction = {v['XW_prediction']}")
    return out


def table(rows, mode):
    print(f"state                 dE       sig2      a         delta      " +
          "  ".join(f"{k}:dhat/d(t)" for k in ARMS))
    for r in rows:
        if mode not in r:
            continue
        cells = []
        for k in ARMS:
            e = r[mode][k]
            q = e["dhat"] / r["delta"] if np.isfinite(e["dhat"]) else float("nan")
            cells.append(f"{q:+8.2f}{'*' if e['trusted'] else ' '}")
        print(f"{r['name']:20s} {r['dE']:.4f}  {r['sigma2']:.4f}  {r['a']:+.4e} {r['delta']:+.3e}  "
              + "  ".join(cells))


def run_extra_checks(seed):
    """A hermiticity and A versus explicit bond-pair sum on a small doped cluster."""
    m = Hubbard(3, 2, "pbc", "obc", 2, 2, 1.0, -0.25, 8.0)
    m.pair_setup(1.0)
    A = PairOp(m, 1.0)
    rng = np.random.default_rng(seed)
    u, v = rng.standard_normal(m.D), rng.standard_normal(m.D)
    herm = abs(u @ A.apply(v) - v @ A.apply(u))
    v1 = v / np.linalg.norm(v)
    diff = abs(v1 @ A.apply(v1) - A.brute(v1))
    log(f"check PairOp: hermiticity {herm:.2e}  vs explicit bond sum {diff:.2e}")
    W, Vv, _ = solve_ed(m, 2, 1e-8, seed)
    psi = Vv[:, 0]
    nz = np.abs(psi) > 1e-6 * np.abs(psi).max()  # symmetry zeros excluded (kappa undefined)
    kap = (m.H @ psi - m.H.diagonal() * psi)[nz] / psi[nz]
    log(f"check sign witness on {nz.sum()}/{m.D} non-zero configs: max |kappa(psi_0) - (E0 - H_nn)| = "
        f"{np.max(np.abs(kap - (W[0] - m.H.diagonal()[nz]))):.2e}, max kappa = {kap.max():.3f} (< 0 expected)")
    return herm < 1e-10 and diff < 1e-10 and kap.max() < 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P3.1 round-2 decisive test (see module docstring)")
    ap.add_argument("--instance", type=int, default=-1, choices=[-1, 0, 1, 2])
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
    ap.add_argument("--sr-iters", type=int, default=1500)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--snap-fracs", type=str, default="0.2,0.5")
    ap.add_argument("--taus", type=str, default="0,0.5,1,2")
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--mc-samples", type=int, default=200000)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--mc-seed", type=int, default=99)
    ap.add_argument("--r-min-trust", type=float, default=0.05)
    ap.add_argument("--max-rel-se", type=float, default=0.5)
    ap.add_argument("--score-floor", type=float, default=0.05)
    ap.add_argument("--n-ed-states", type=int, default=4)
    ap.add_argument("--degeneracy-tol", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--p11-json", type=str, default=None)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    ap.add_argument("--checks-only", type=int, choices=[0, 1], default=0)
    ap.add_argument("--aggregate", type=str, default="")
    a = ap.parse_args(argv)
    if a.instance >= 0:
        for k, v in INSTANCES[a.instance].items():
            setattr(a, k, v)
    a.n_ed_states = max(a.n_ed_states, 2)
    a.width_list = [int(w) for w in a.widths.split(",") if w.strip()]
    a.tau_list = [float(x) for x in a.taus.split(",") if x.strip()]
    a.snap_list = [float(x) for x in a.snap_fracs.split(",") if x.strip()]
    if not a.width_list or not a.tau_list or a.sr_iters < 2 or a.n_boot < 2:
        ap.error("need >= 1 width, >= 1 tau, --sr-iters >= 2, --n-boot >= 2")
    if any(not (0 < f < 1) for f in a.snap_list):
        ap.error("--snap-fracs entries must lie in (0, 1)")
    here = os.path.dirname(os.path.abspath(__file__))
    a.tag = (f"L{a.Lx}x{a.Ly}_{a.bcx}{a.bcy}_n{a.nup}-{a.ndn}_U{a.U:g}_tp{a.tp:g}"
             f"_w{'-'.join(map(str, a.width_list))}_j0_s{a.seed}")
    if not a.out:
        a.out = os.path.join(here, "results", a.tag + ".json")
    if a.p11_json is None:
        cand = os.path.join(here, "..", "P1.1_sign_vs_amplitude", "results", a.tag + ".json")
        a.p11_json = cand if os.path.exists(cand) else ""
    return a


def aggregate(d):
    rows, xw = [], []
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    for f in files:
        r = json.load(open(f))
        if "metrics" not in r:
            continue
        rows += r["states"]
        xw += r["xw"]
        print(f"{os.path.basename(f)}: {len(r['states'])} states, a0={r['a0']:+.4e}, "
              f"gap/V={r['ed']['gap_per_site']:.4f}")
    if not rows:
        print("no finished results"); return 1
    primary = "mc" if any("mc" in r for r in rows) else "exact"
    report(rows, xw, primary, f"POOLED over {len(files)} file(s)")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        return aggregate(args.aggregate)
    t_start = time.time()
    log("args: " + json.dumps({k: v for k, v in vars(args).items()}))
    if not (run_checks(args.seed) and run_extra_checks(args.seed)):
        log("correctness checks failed; aborting")
        return 2
    if args.checks_only:
        return 0
    if os.path.exists(args.out) and not args.overwrite:
        log(f"output {args.out} exists; use --overwrite 1. Exiting.")
        return 0
    rng = np.random.default_rng(args.seed)
    m = Hubbard(args.Lx, args.Ly, args.bcx, args.bcy, args.nup, args.ndn, args.t, args.tp, args.U)
    m.pair_setup(args.r_min)
    Aop = PairOp(m, args.r_min)
    log(f"model V={m.V} N=({m.nup},{m.ndn}) D={m.D} nnz(H)={m.H.nnz} A pairs={Aop.n_pairs}")
    t0 = time.time(); Aop.apply(np.ones(m.D)); tA = time.time() - t0
    t0 = time.time(); m.H @ np.ones(m.D); tH = time.time() - t0
    log(f"timing: one A application {tA:.3f}s, one H application {tH:.3f}s")
    W, Vv, deg = solve_ed(m, args.n_ed_states, args.degeneracy_tol, args.seed)
    a_k = [float(Vv[:, k] @ Aop.apply(Vv[:, k])) for k in range(len(W))]
    gap = float(W[1] - W[0])
    ed = dict(levels=W.tolist(), degeneracy=deg, a_k=a_k, gap=gap, gap_per_site=gap / (m.V * args.t),
              Pd_P11_E0=m.dwave(Vv[:, 0]))
    log(f"ED levels {np.round(W, 6).tolist()} deg={deg}  a_k={np.round(a_k, 5).tolist()}  "
        f"Delta1/V={ed['gap_per_site']:.4f}t  (P1.1 Pd(E0)={ed['Pd_P11_E0']:.4e})")
    rng.random(m.D)  # replicate the P1.1 calibration draw so the SR trajectories match P1.1
    Pu, Pd = uhf_orbitals(m, args.seed)
    snap_iters = sorted({int(round(f * args.sr_iters)) for f in args.snap_list})
    nets = [("sj", Ansatz(m, Pu, Pd, "sj", jastrow=True, seed=args.seed))]
    for w in args.width_list:
        nets.append((f"nnbf_{w}", Ansatz(m, Pu, Pd, "nnbf", width=w, jastrow=False, seed=args.seed + w)))
    states = []
    for name, a in nets:
        t0 = time.time()
        snaps = optimize_sr_snap(a, m.H, args, rng, name, set(snap_iters))
        log(f"{name}: trained in {time.time() - t0:.1f}s")
        for lab, x in snaps.items():
            a.set_flat(x)
            states.append((name, str(lab), a.amplitudes(args.chunk)))
        a.set_flat(snaps["final"])
    p11 = json.load(open(args.p11_json)) if args.p11_json else None
    mc_rng = np.random.default_rng(args.mc_seed)
    rows = []
    for name, lab, psi in states:
        t0 = time.time()
        psi = psi / np.linalg.norm(psi)
        G = Vv[:, :deg]
        c = G.T @ psi
        p0 = G @ c if np.linalg.norm(c) > 1e-14 else Vv[:, 0]
        p0 = p0 / np.linalg.norm(p0)
        a0 = float(p0 @ Aop.apply(p0))
        if p0 @ psi < 0:
            p0 = -p0
        res = analyze_state(m, Aop, psi, a0, args, mc_rng)
        row = dict(name=f"{name}@{lab}", net=name, iter=lab, a0=a0, dE=res["E"] - float(W[0]),
                   delta=a0 - res["a"],
                   fid_k=[float((Vv[:, k] @ psi) ** 2) for k in range(len(W))],
                   w_sign=float(np.sum(p0[np.sign(psi) != np.sign(p0)] ** 2)), **res)
        row["scored"] = bool(abs(row["delta"]) > args.score_floor * abs(a0))
        if p11 and lab == "final" and name in p11.get("states", {}):
            row["p11_dE"] = p11["states"][name]["dE_tot"]
        rows.append(row)
        log(f"{row['name']:16s} dE={row['dE']:.4f} a={res['a']:+.4e} a0={a0:+.4e} delta={row['delta']:+.3e} "
            f"fid={np.round(row['fid_k'], 3).tolist()} w_sign={row['w_sign']:.3f} "
            + " ".join(f"{k}:{res['exact'][k]['dhat']:+.2e}/{res['exact'][k]['rhat']:.3f}" for k in ARMS)
            + (f"  [P1.1 dE {row['p11_dE']:.4f}]" if "p11_dE" in row else "")
            + f" ({time.time() - t0:.1f}s)")
    modes = ["exact"] + (["mc"] if args.mc_samples > 0 else [])
    xw = xw_estimates(rows, args.width_list, modes)
    xw_rows = []
    for lab, o in xw.items():
        r = next(r for r in rows if r["name"] == o["state"])
        o.update(delta=r["delta"], scored=r["scored"])
        xw_rows.append(o)
    a0_main = rows[0]["a0"]
    out = dict(params=vars(args), ed=ed, a0=a0_main, states=rows, xw=xw_rows)
    for mode in modes:
        print(f"=== per-state dhat/delta [{mode}] (* = trusted)")
        table(rows, mode)
    rep = report(rows, xw_rows, modes[-1], "this instance")
    out["metrics"] = {k: v["metrics"] for k, v in rep.items()}
    out["verdicts"] = {k: v["verdicts"] for k, v in rep.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    log(f"wrote {args.out}  (total {time.time() - t_start:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
