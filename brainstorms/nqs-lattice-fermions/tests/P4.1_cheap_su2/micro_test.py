#!/usr/bin/env python3
"""
P4.1 decisive micro-test: exact spin-SU(2) fermionic ansatz at single-determinant cost
versus the spin-projected Jastrow-pair-product foil (mVMC style).

Everything is computed by EXACT sums over the full fixed-(N_up, N_dn) Hilbert space of a
small 2D Hubbard cluster (no Monte Carlo noise), so energy differences between ansaetze are
differences in expressivity + optimisation, never sampling error.

Physics
-------
Hubbard model  H = -sum_{ij,s} t_ij c+_{is} c_{js} + U sum_i n_{i up} n_{i dn}
on an Lx x Ly torus (periodic in both directions), nearest-neighbour hopping t (=1) and
next-nearest (diagonal) hopping tp.  Fermion ordering: all up modes (site 0..L-1) then all
down modes, so a basis state is |u, d> with u, d bit strings and the wave function is stored
as a matrix Psi[u, d].

Ansaetze (all share the charge Jastrow  J(c) = sum_{i<j} v_ij n_i n_j + g sum_i D_i,
which depends only on the charge configuration c = (n_1..n_L) and therefore commutes
with every spin rotation):
  foil  : P_S [ e^J det f[up, dn] ]       f general L x L (singlet+triplet / AF pairing),
          P_S the exact total-spin projector  (= mVMC  P_J L^S |phi_PP>).
          Real VMC cost: N_grid Pfaffians per amplitude.
  agp   : e^J det F0[up, dn]              F0 = F0^T (singlet geminal), no backflow.
          Exact S=0 by construction, 1 determinant.
  r1    : e^J det F_c[up, dn],  F_c = B_c F0 B_c^T,  B_c = 1 + eta(c)
          eta(c) = MLP of the CHARGE configuration only (doublon/holon features),
          restricted to |i-j|^2 <= 2.  Charge-conditioned singlet geminal: exact S,
          1 determinant per amplitude (the route under test, "R1").
  r1x   : (1 + sum_b alpha_b(c) X_b) |r1>,  X_b = 2 S_i.S_j + n_i n_j / 2 the SU(2)-scalar
          spin exchange on bond b=(i,j) (NN and NNN bonds), alpha_b(c) charge-only MLP head.
          Still exact S; extra cost = one rank-2 determinant ratio per bond ("R1x").
  r2    : P_S [ e^J det (B_c f B_c^T)[up, dn] ]  f general: charge backflow + exact
          projection (what the quadrature would buy on top of R1; upper-bound reference).
For S>0 (option --spin) the sector is N_up - N_dn = 2S and every determinant gets 2S extra
columns of unpaired orbitals Phi_c = B_c Phi0 (highest-weight construction; exact S).

Optimisation: deterministic L-BFGS (scipy) on the exact energy with an analytic gradient
(hand-written vector-Jacobian products).  Several random seeds; best energy kept.

Observables (reported for exact ground state and for each ansatz):
  E, eps = (E - E0)/|E0|, <S^2>, C_s(R) = (1/L) sum_i <S^z_i S^z_{i+R}>,
  P_d(R) = (1/L) sum_i <Delta_d^+(i) Delta_d(i+R)>  (singlet d-wave bond pairs).

Usage
-----
  python micro_test.py --mode MODE [options]

--mode {smoke, run, report}           (required)
    smoke  : tiny self-test on a 2x3 torus with N_e=4: checks (a) analytic gradient against
             central finite differences for every ansatz, (b) <S^2> = S(S+1) to machine
             precision for agp/r1/r1x/foil/r2 at random parameters, both for --spin 0 and
             the highest-weight S=1 construction, (c) a short optimisation of every ansatz.
             Runs in a few seconds; prints PASS/FAIL per check.
    run    : optimise ONE ansatz (--ansatz) on the instance given by the lattice options,
             compute observables, write JSON to --outdir/<tag>_<ansatz>_seed<seed>.json.
             Skips the work if that JSON already exists (unless --overwrite).
    report : read all JSONs for --tag in --outdir, compute the decisive metric
             M = eps(route)/eps(foil) (best seed per ansatz) and the guards, print the
             verdict PASS / FAIL / AMBIGUOUS, and write <tag>_report.json.

Lattice / model options (used by run and report; smoke ignores them):
  --Lx INT        (default 3)   torus length in x, >= 2
  --Ly INT        (default 4)   torus length in y, >= 2
  --Ne INT        (default 10)  electron number, 2 <= Ne <= 2L - 2, Ne - 2*spin even
  --U FLOAT       (default 8.0) on-site repulsion in units of t
  --tp FLOAT      (default -0.3) next-nearest-neighbour hopping t' (t = 1)
  --spin INT      (default 0)   target total spin S; the sector is S^z = S
                                (N_up = Ne/2 + S, N_dn = Ne/2 - S)

Ansatz / optimisation options:
  --ansatz {foil, agp, r1, r1x, r2}   (run mode; default r1)  see "Ansaetze" above
  --hidden INT     (default 16)  hidden units of the charge MLP (r1, r1x, r2)
  --seed INT       (default 0)   RNG seed for initial parameters (seed 0 = deterministic
                                 free-fermion pairing init + small noise; seed >= 1 adds an
                                 antiferromagnetic (staggered) component to general f for
                                 foil/r2, larger noise for the others)
  --maxiter INT    (default 4000) L-BFGS iteration cap
  --max-hours FLOAT (default 10.0) wall-clock cap for the optimisation (stops gracefully)
  --chunk INT      (default 200000) configurations per chunk in determinant evaluation

I/O options:
  --outdir PATH    (default: <script dir>/results)
  --tag STR        (default: auto "L{Lx}x{Ly}_N{Ne}_U{U}_tp{tp}_S{spin}")
  --route {r1, r1x} (report mode; default r1x) which ansatz the decisive metric M uses
  --overwrite      (flag) recompute even if the output JSON exists

Decisive metric (report mode)
-----------------------------
  eps_A = (E_A - E0)/|E0| with E_A the best (lowest) energy of ansatz A over seeds.
  M     = eps_route / eps_foil.
  Guards: dC_s(route) <= dC_s(foil) and dP_d(route) <= dP_d(foil) + 1e-4, where
  dC_s = max_{R != 0} |C_s(R) - C_s^exact(R)|, dP_d = |P_d(R_far) - P_d^exact(R_far)| averaged
  over the displacements R_far whose d-wave "plus" stencils do not overlap; and
  |<S^2>_route - S(S+1)| < 1e-10.
  PASS      : M <= 0.5 and all guards hold.
  FAIL      : M >= 1.0, or the <S^2> guard fails.
  AMBIGUOUS : otherwise (pre-committed action: rerun on the second instance
              --Ne 8 with the same lattice; promote only if M < 1 with guards there too).
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
from scipy.sparse.linalg import LinearOperator, eigsh


# ----------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="P4.1 exact-SU(2) ansatz micro-test (see module docstring)")
    p.add_argument("--mode", required=True, choices=["smoke", "run", "report"])
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=4)
    p.add_argument("--Ne", type=int, default=10)
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--tp", type=float, default=-0.3)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--ansatz", default="r1", choices=["foil", "agp", "r1", "r1x", "r2"])
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=4000)
    p.add_argument("--max-hours", type=float, default=10.0, dest="max_hours")
    p.add_argument("--chunk", type=int, default=200000)
    p.add_argument("--outdir", default=os.path.join(here, "results"))
    p.add_argument("--tag", default=None)
    p.add_argument("--route", default="r1x", choices=["r1", "r1x"])
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    L = a.Lx * a.Ly
    if a.mode != "smoke":
        if a.Lx < 2 or a.Ly < 2:
            p.error("Lx, Ly must be >= 2")
        if not (2 <= a.Ne <= 2 * L - 2) or (a.Ne - 2 * a.spin) % 2 or a.spin < 0 or a.Ne // 2 - a.spin < 1:
            p.error("invalid Ne/spin combination")
    if a.tag is None:
        a.tag = f"L{a.Lx}x{a.Ly}_N{a.Ne}_U{a.U:g}_tp{a.tp:g}_S{a.spin}"
    return a


# ----------------------------------------------------------------------------- lattice
class Lattice:
    def __init__(self, Lx, Ly, tp):
        self.Lx, self.Ly, self.L = Lx, Ly, Lx * Ly
        self.tp = tp
        L = self.L
        self.T = np.zeros((L, L))  # hopping amplitudes t_ij (H = -sum t_ij c+_i c_j)
        for i in range(L):
            for (dx, dy, amp) in [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                                  (1, 1, tp), (1, -1, tp), (-1, 1, tp), (-1, -1, tp)]:
                j = self.shift(i, dx, dy)
                if j != i:
                    self.T[i, j] += amp
        self.T = 0.5 * (self.T + self.T.T)
        # backflow mask |i-j|^2 <= 2 (minimum image), bond list for exchange dressing
        self.mask = np.zeros((L, L), bool)
        bonds = set()
        for i in range(L):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    j = self.shift(i, dx, dy)
                    self.mask[i, j] = True
                    if j != i:
                        bonds.add((min(i, j), max(i, j)))
        self.bonds = sorted(bonds)

    def xy(self, i):
        return divmod(i, self.Ly)

    def shift(self, i, dx, dy):
        x, y = self.xy(i)
        return ((x + dx) % self.Lx) * self.Ly + (y + dy) % self.Ly

    def displacements(self):
        return [(dx, dy) for dx in range(self.Lx) for dy in range(self.Ly)]


# ----------------------------------------------------------------------------- species basis
class Species:
    """Basis of N fermions of one spin species on L sites with c+_i c_j matrices (JW signs)."""

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
        """sparse matrix of c+_i c_j on this species (i == j: number operator)."""
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


class Sector:
    """Fixed (N_up, N_dn) sector; wave functions are matrices Psi[iu, id]."""

    def __init__(self, lat, Nu, Nd, U):
        self.lat, self.Nu, self.Nd, self.U = lat, Nu, Nd, U
        L = lat.L
        self.up, self.dn = Species(L, Nu), Species(L, Nd)
        self.shape = (self.up.dim, self.dn.dim)
        self.dim = self.up.dim * self.dn.dim
        self.Ku = sum(-lat.T[i, j] * self.up.E(i, j) for i in range(L) for j in range(L) if lat.T[i, j] != 0)
        self.Kd = sum(-lat.T[i, j] * self.dn.E(i, j) for i in range(L) for j in range(L) if lat.T[i, j] != 0)
        self.Ku = sp.csr_matrix(self.Ku)
        self.KdT = sp.csr_matrix(self.Kd).T.tocsr()
        bu = self.up.bits.astype(float)
        bd = self.dn.bits.astype(float)
        self.docc = bu @ bd.T  # number of doubly occupied sites per (u, d)
        self.nup_i = lambda i: bu[:, i][:, None] * np.ones((1, self.dn.dim))
        self.bu, self.bd = bu, bd
        # charge configuration per basis state
        nu = np.repeat(self.up.bits, self.dn.dim, axis=0)
        nd = np.tile(self.dn.bits, (self.up.dim, 1))
        n = (nu + nd).astype(np.int64)
        code = n @ (3 ** np.arange(L, dtype=np.int64))
        uniq, self.cidx = np.unique(code, return_inverse=True)
        first = np.zeros(len(uniq), dtype=np.int64)
        first[self.cidx[::-1]] = np.arange(self.dim)[::-1]
        self.n_c = n[first].astype(float)  # (nC, L)
        self.nC = len(uniq)
        self.iu = np.repeat(np.arange(self.up.dim), self.dn.dim)
        self.id = np.tile(np.arange(self.dn.dim), self.up.dim)
        Sz = 0.5 * (Nu - Nd)
        self.Sz = Sz
        self.Smax = 0.5 * min(Nu + Nd, 2 * L - Nu - Nd)
        self._pp = None

    # --- operators acting on matrices Psi (nU, nD)
    def H(self, P):
        return self.Ku @ P + (self.KdT.T @ P.T).T + self.U * self.docc * P

    def spsm_pairs(self):
        if self._pp is None:
            L = self.lat.L
            self._pp = [(self.up.E(i, j), self.dn.E(j, i).T.tocsr()) for i in range(L) for j in range(L) if i != j]
        return self._pp

    def S2(self, P):
        L = self.lat.L
        out = (self.Sz ** 2 - self.Sz) * P
        diag = sum(self.bu[:, i][:, None] * (1.0 - self.bd[:, i][None, :]) for i in range(L))
        out = out + diag * P
        for Eu, EdT in self.spsm_pairs():
            out -= Eu @ (EdT.T @ P.T).T
        return out

    def exchange(self, P, i, j):
        """X_ij = 2 S_i.S_j + n_i n_j / 2 (SU(2)-scalar spin exchange), i != j."""
        szi = 0.5 * (self.bu[:, i][:, None] - self.bd[:, i][None, :])
        szj = 0.5 * (self.bu[:, j][:, None] - self.bd[:, j][None, :])
        ni = self.bu[:, i][:, None] + self.bd[:, i][None, :]
        nj = self.bu[:, j][:, None] + self.bd[:, j][None, :]
        out = (2 * szi * szj + 0.5 * ni * nj) * P
        # 2 * 1/2 (S+_i S-_j + S-_i S+_j);  S+_i S-_j = -E^u_ij Psi (E^d_ji)^T
        out -= self.up.E(i, j) @ (self.dn.E(j, i) @ P.T).T
        out -= self.up.E(j, i) @ (self.dn.E(i, j) @ P.T).T
        return out

    def projector(self, S):
        """returns function applying the exact projector onto total spin S (within this S^z)."""
        ks = [k for k in np.arange(abs(self.Sz), self.Smax + 0.5, 1.0) if abs(k - S) > 1e-9]

        def apply(P):
            for k in ks:
                P = (self.S2(P) - k * (k + 1) * P) / (S * (S + 1) - k * (k + 1))
            return P
        return apply


# ----------------------------------------------------------------------------- exact diagonalisation
def exact_ground_state(sec, S, mu=5.0):
    """lowest eigenstate of H + mu (S^2 - S(S+1))^2-free shift: uses H + mu*S^2 restricted
    to the S^z = S sector, whose lowest state has total spin S whenever an S state lies
    within mu*2(S+1) of the global minimum (checked a posteriori)."""
    shp = sec.shape

    def mv(x):
        P = x.reshape(shp)
        return (sec.H(P) + mu * sec.S2(P)).ravel()
    op = LinearOperator((sec.dim, sec.dim), matvec=mv, dtype=float)
    rng = np.random.default_rng(1)
    w, v = eigsh(op, k=1, which="SA", v0=rng.standard_normal(sec.dim), tol=1e-12)
    P = v[:, 0].reshape(shp)
    E = float(np.sum(P * sec.H(P)) / np.sum(P * P))
    s2 = float(np.sum(P * sec.S2(P)) / np.sum(P * P))
    return P, E, s2


# ----------------------------------------------------------------------------- observables
def observables(sec, P):
    lat = sec.lat
    L = lat.L
    nrm = np.sum(P * P)
    E = float(np.sum(P * sec.H(P)) / nrm)
    s2 = float(np.sum(P * sec.S2(P)) / nrm)
    w = P * P / nrm
    szu = 0.5 * sec.bu  # (nU, L)
    szd = 0.5 * sec.bd  # (nD, L)
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
    # d-wave pair correlation at non-overlapping displacements
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
                            # c+_{a up} c+_{b dn} c_{d dn} c_{c up} = E^u_ac (x) E^d_bd
                            Z = sec.up.E(a, c) @ (sec.dn.E(b, d) @ P.T).T
                            acc += 0.5 * g1 * g2 * np.sum(P * Z)
        Pd[f"{R[0]},{R[1]}"] = acc / L / nrm
    return dict(E=E, S2=s2, Cs=Cs, Pd=Pd)


# ----------------------------------------------------------------------------- ansatz
class Ansatz:
    def __init__(self, sec, kind, S, hidden, seed):
        self.sec, self.kind, self.S = sec, kind, S
        lat = sec.lat
        L = lat.L
        self.L = L
        self.nunp = sec.Nu - sec.Nd  # unpaired columns (= 2S)
        self.backflow = kind in ("r1", "r1x", "r2")
        self.symF = kind in ("agp", "r1", "r1x")
        self.proj = kind in ("foil", "r2")
        self.dress = kind == "r1x"
        self.H = hidden if self.backflow else 0
        self.mpos = np.nonzero(lat.mask)
        self.nmask = len(self.mpos[0])
        self.bonds = lat.bonds
        self.iu_, self.ju_ = np.triu_indices(L, 1)
        shapes = [("F0", (L, L)), ("Jv", (len(self.iu_),)), ("Jg", (1,))]
        if self.nunp:
            shapes.append(("Phi0", (L, self.nunp)))
        if self.backflow:
            shapes += [("W", (self.H, 2 * L)), ("b", (self.H,)), ("V", (self.H, self.nmask))]
        if self.dress:
            shapes += [("a", (len(self.bonds),)), ("Ua", (self.H, len(self.bonds)))]
        self.shapes = shapes
        self.sizes = [int(np.prod(s)) for _, s in shapes]
        self.nparam = sum(self.sizes)
        n = sec.n_c
        d = (n == 2).astype(float)
        h = (n == 0).astype(float)
        self.feat = np.concatenate([d, h], axis=1)
        self.dbl = d.sum(1)
        self.projS = sec.projector(S) if self.proj else None
        self.theta0 = self.init(seed)

    # ---- parameters
    def unpack(self, th):
        out, k = {}, 0
        for (name, shp), sz in zip(self.shapes, self.sizes):
            out[name] = th[k:k + sz].reshape(shp)
            k += sz
        return out

    def pack(self, d):
        return np.concatenate([d[name].ravel() for name, _ in self.shapes])

    def init(self, seed):
        rng = np.random.default_rng(seed)
        lat, L = self.sec.lat, self.L
        h0 = -lat.T
        e, U = np.linalg.eigh(h0)
        Nd = self.sec.Nd
        F = U[:, :Nd] @ U[:, :Nd].T
        p = {}
        noise = 0.02 if seed == 0 else 0.1
        if self.symF:
            R = rng.standard_normal((L, L))
            p["F0"] = F + noise * 0.5 * (R + R.T)
        else:
            Fg = F.copy()
            if seed >= 1:  # antiferromagnetic (staggered) unrestricted pairing
                stag = np.array([(-1) ** sum(lat.xy(i)) for i in range(L)], float)
                eu, Uu = np.linalg.eigh(h0 + 1.0 * np.diag(stag))
                ed, Ud = np.linalg.eigh(h0 - 1.0 * np.diag(stag))
                Fg = Uu[:, :Nd] @ Ud[:, :Nd].T
            p["F0"] = Fg + noise * rng.standard_normal((L, L))
        p["Jv"] = np.zeros(len(self.iu_))
        p["Jg"] = np.array([-1.0])
        if self.nunp:
            p["Phi0"] = U[:, Nd:Nd + self.nunp] + noise * rng.standard_normal((L, self.nunp))
        if self.backflow:
            p["W"] = 0.3 * rng.standard_normal((self.H, 2 * L))
            p["b"] = 0.1 * rng.standard_normal(self.H)
            p["V"] = 0.01 * rng.standard_normal((self.H, self.nmask))
        if self.dress:
            p["a"] = np.zeros(len(self.bonds))
            p["Ua"] = 0.01 * rng.standard_normal((self.H, len(self.bonds)))
        return self.pack(p)

    # ---- charge-level quantities
    def charge_level(self, p):
        L, nC = self.L, self.sec.nC
        F0 = 0.5 * (p["F0"] + p["F0"].T) if self.symF else p["F0"]
        n = self.sec.n_c
        Vm = np.zeros((L, L))
        Vm[self.iu_, self.ju_] = p["Jv"]
        J = np.einsum("ci,ij,cj->c", n, Vm, n) + p["Jg"][0] * self.dbl
        cache = dict(F0=F0, J=J)
        if self.backflow:
            hpre = self.feat @ p["W"].T + p["b"]
            hh = np.tanh(hpre)
            eta_m = hh @ p["V"]  # (nC, nmask)
            B = np.zeros((nC, L, L))
            B[:, self.mpos[0], self.mpos[1]] = eta_m
            B += np.eye(L)[None]
            Fc = B @ F0 @ B.transpose(0, 2, 1)
            cache.update(h=hh, B=B, Fc=Fc)
            if self.nunp:
                cache["Phic"] = B @ p["Phi0"]
            if self.dress:
                cache["alpha"] = p["a"][None, :] + hh @ p["Ua"]
        else:
            cache["Fc"] = None
        return cache

    # ---- forward: returns psi matrix and cache
    def forward(self, th, chunk=200000):
        p = self.unpack(th)
        sec = self.sec
        cc = self.charge_level(p)
        expJ = np.exp(cc["J"] - cc["J"].max())
        cc["expJ"] = expJ
        phi = np.empty(sec.dim)
        A_all = []
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            A = self.build_A(cc, p, sl)
            phi[sl] = np.linalg.det(A) * expJ[sec.cidx[sl]]
            A_all.append(A)
        cc["A"] = A_all
        phiM = phi.reshape(sec.shape)
        cc["phi"] = phiM
        if self.proj:
            psi = self.projS(phiM)
        elif self.dress:
            Y = [sec.exchange(phiM, i, j) for (i, j) in self.bonds]
            al = cc["alpha"][sec.cidx].reshape(sec.shape + (len(self.bonds),))
            psi = phiM + sum(al[:, :, b] * Y[b] for b in range(len(self.bonds)))
            cc["Y"], cc["al"] = Y, al
        else:
            psi = phiM
        return psi, cc, p

    def build_A(self, cc, p, sl):
        sec = self.sec
        c = sec.cidx[sl]
        ou = sec.up.occ[sec.iu[sl]]
        od = sec.dn.occ[sec.id[sl]]
        if cc["Fc"] is None:
            A = cc["F0"][ou[:, :, None], od[:, None, :]]
        else:
            A = cc["Fc"][c[:, None, None], ou[:, :, None], od[:, None, :]]
        if self.nunp:
            Phi = cc["Phic"][c[:, None], ou] if self.backflow else p["Phi0"][ou]
            A = np.concatenate([A, Phi], axis=2)
        return A

    # ---- energy and gradient
    def energy_grad(self, th, chunk=200000):
        sec = self.sec
        psi, cc, p = self.forward(th, chunk)
        nrm = np.sum(psi * psi)
        Hp = sec.H(psi)
        E = np.sum(psi * Hp) / nrm
        v = 2.0 * (Hp - E * psi) / nrm  # dE/dpsi
        g = {name: np.zeros(shp) for name, shp in self.shapes}
        hbar = None
        if self.proj:
            vphi = self.projS(v)
        elif self.dress:
            nb = len(self.bonds)
            abar = np.zeros((sec.nC, nb))
            vphi = v.copy()
            for b, (i, j) in enumerate(self.bonds):
                abar[:, b] = np.bincount(sec.cidx, weights=(v * cc["Y"][b]).ravel(), minlength=sec.nC)
                vphi += sec.exchange(cc["al"][:, :, b] * v, i, j)
            g["a"] = abar.sum(0)
            g["Ua"] = cc["h"].T @ abar
            hbar = abar @ p["Ua"].T
        else:
            vphi = v
        phi = cc["phi"].ravel()
        vphi = vphi.ravel()
        gJ = np.bincount(sec.cidx, weights=vphi * phi, minlength=sec.nC)
        L = self.L
        Nd = sec.Nd
        Fbar = np.zeros(sec.nC * L * L) if self.backflow else np.zeros(L * L)
        Phibar = np.zeros((sec.nC if self.backflow else 1) * L * max(self.nunp, 1))
        for k, s in enumerate(range(0, sec.dim, chunk)):
            sl = slice(s, min(s + chunk, sec.dim))
            A = cc["A"][k]
            cof = cofactor(A)
            w = vphi[sl] * cc["expJ"][sec.cidx[sl]]
            gA = w[:, None, None] * cof
            c = sec.cidx[sl]
            ou = sec.up.occ[sec.iu[sl]]
            od = sec.dn.occ[sec.id[sl]]
            base = (c * L * L)[:, None, None] if self.backflow else 0
            idx = base + ou[:, :, None] * L + od[:, None, :]
            Fbar += np.bincount(idx.ravel(), weights=gA[:, :, :Nd].ravel(), minlength=Fbar.size)
            if self.nunp:
                m = self.nunp
                base2 = (c * L * m)[:, None, None] if self.backflow else 0
                idx2 = base2 + ou[:, :, None] * m + np.arange(m)[None, None, :]
                Phibar += np.bincount(idx2.ravel(), weights=gA[:, :, Nd:].ravel(), minlength=Phibar.size)
        # Jastrow (gradient of log-amplitude times phi already folded in gJ)
        n = sec.n_c
        Mn = n.T @ (gJ[:, None] * n)
        g["Jv"] = Mn[self.iu_, self.ju_]
        g["Jg"] = np.array([np.dot(gJ, self.dbl)])
        F0 = cc["F0"]
        if self.backflow:
            Fb = Fbar.reshape(sec.nC, L, L)
            B = cc["B"]
            F0bar = np.einsum("cki,ckl,clj->ij", B, Fb, B)
            Bbar = Fb @ B @ F0.T + Fb.transpose(0, 2, 1) @ B @ F0
            if self.nunp:
                Pb = Phibar.reshape(sec.nC, L, self.nunp)
                Bbar += Pb @ p["Phi0"].T
                g["Phi0"] = np.einsum("cki,ckm->im", B, Pb)
            etabar = Bbar[:, self.mpos[0], self.mpos[1]]
            hh = cc["h"]
            g["V"] = hh.T @ etabar
            hb = etabar @ p["V"].T
            if hbar is not None:
                hb = hb + hbar
            pre = hb * (1 - hh ** 2)
            g["W"] = pre.T @ self.feat
            g["b"] = pre.sum(0)
        else:
            F0bar = Fbar.reshape(L, L)
            if self.nunp:
                g["Phi0"] = Phibar.reshape(L, self.nunp)
        g["F0"] = 0.5 * (F0bar + F0bar.T) if self.symF else F0bar
        return float(E), self.pack(g), psi


def cofactor(A):
    """batched cofactor matrix cof(A) = det(A) A^{-T}, robust for singular A (via SVD)."""
    U, s, Vt = np.linalg.svd(A)
    n = s.shape[1]
    prod_others = np.empty_like(s)
    for k in range(n):
        prod_others[:, k] = np.prod(np.delete(s, k, axis=1), axis=1)
    sgn = np.linalg.det(U) * np.linalg.det(Vt)
    adj = np.einsum("bji,bj,bkj->bik", Vt, prod_others, U)  # V diag U^T = adj(A)
    return sgn[:, None, None] * adj.transpose(0, 2, 1)


# ----------------------------------------------------------------------------- optimisation
class Stop(Exception):
    pass


def optimise(ans, maxiter, max_hours, chunk, log=print):
    t0 = time.time()
    best = {"E": np.inf, "th": ans.theta0.copy()}
    it = [0]

    def f(th):
        E, g, _ = ans.energy_grad(th, chunk)
        if E < best["E"]:
            best["E"], best["th"] = E, th.copy()
        return E, g

    def cb(th):
        it[0] += 1
        if it[0] % 50 == 0:
            log(f"  iter {it[0]:5d}  E_best = {best['E']:.10f}  ({(time.time()-t0)/60:.1f} min)")
            sys.stdout.flush()
        if time.time() - t0 > 3600 * max_hours:
            raise StopIteration

    try:
        minimize(f, ans.theta0, jac=True, method="L-BFGS-B", callback=cb,
                 options=dict(maxiter=maxiter, maxcor=30, ftol=1e-15, gtol=1e-10))
    except StopIteration:
        log("  wall-clock cap reached")
    return best["th"], best["E"], it[0], time.time() - t0


# ----------------------------------------------------------------------------- modes
def make_sector(Lx, Ly, Ne, U, tp, S):
    lat = Lattice(Lx, Ly, tp)
    Nu, Nd = Ne // 2 + S, Ne // 2 - S
    return Sector(lat, Nu, Nd, U)


def smoke():
    ok_all = True
    for S in (0, 1):
        sec = make_sector(2, 3, 4, 6.0, -0.3, S)
        _, E0, s20 = exact_ground_state(sec, S)
        print(f"[S={S}] 2x3 torus Ne=4 dim={sec.dim} E0={E0:.8f} <S^2>_ED={s20:.2e}")
        for kind in ["agp", "r1", "r1x", "foil", "r2"]:
            ans = Ansatz(sec, kind, S, hidden=4, seed=3)
            th = ans.theta0 + 0.05 * np.random.default_rng(7).standard_normal(ans.nparam)
            E, g, psi = ans.energy_grad(th)
            s2 = np.sum(psi * sec.S2(psi)) / np.sum(psi * psi)
            # finite-difference check on 12 random coordinates
            rng = np.random.default_rng(11)
            idx = rng.choice(ans.nparam, size=min(12, ans.nparam), replace=False)
            err = 0.0
            for k in idx:
                e = np.zeros(ans.nparam); e[k] = 1e-5
                fd = (ans.energy_grad(th + e)[0] - ans.energy_grad(th - e)[0]) / 2e-5
                err = max(err, abs(fd - g[k]) / max(1e-6, abs(fd) + abs(g[k])))
            s2ok = abs(s2 - S * (S + 1)) < 1e-10
            gok = err < 1e-5
            ok_all &= s2ok and gok
            print(f"  {kind:5s} P={ans.nparam:4d} E={E:+.6f}  <S^2>-S(S+1)={s2-S*(S+1):+.2e} [{'PASS' if s2ok else 'FAIL'}]"
                  f"  grad rel.err={err:.1e} [{'PASS' if gok else 'FAIL'}]")
    # short optimisation, Sz=0
    sec = make_sector(2, 3, 4, 6.0, -0.3, 0)
    P0, E0, _ = exact_ground_state(sec, 0)
    for kind in ["foil", "agp", "r1", "r1x", "r2"]:
        ans = Ansatz(sec, kind, 0, hidden=4, seed=0)
        th, E, nit, dt = optimise(ans, 300, 0.05, 200000, log=lambda *a: None)
        print(f"  opt {kind:5s}: E={E:.8f} eps={(E-E0)/abs(E0):.2e} ({nit} it, {dt:.1f}s)")
    print("SMOKE", "PASS" if ok_all else "FAIL")


def run(a):
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"{a.tag}_{a.ansatz}_seed{a.seed}.json")
    if os.path.exists(out) and not a.overwrite:
        print(f"exists: {out}  (use --overwrite)")
        return
    t0 = time.time()
    sec = make_sector(a.Lx, a.Ly, a.Ne, a.U, a.tp, a.spin)
    print(f"instance {a.tag}: dim={sec.dim} charge configs={sec.nC}")
    P0, E0, s20 = exact_ground_state(sec, a.spin)
    print(f"ED: E0={E0:.10f} <S^2>={s20:.3e} ({time.time()-t0:.0f}s)")
    ans = Ansatz(sec, a.ansatz, a.spin, a.hidden, a.seed)
    print(f"ansatz {a.ansatz}: {ans.nparam} parameters")
    th, E, nit, dt = optimise(ans, a.maxiter, a.max_hours, a.chunk)
    psi = ans.forward(th, a.chunk)[0]
    obs = observables(sec, psi)
    obs0 = observables(sec, P0)
    res = dict(tag=a.tag, ansatz=a.ansatz, seed=a.seed, nparam=ans.nparam, iters=nit, seconds=dt,
               E0=E0, S2_exact=s20, E=obs["E"], eps=(obs["E"] - E0) / abs(E0), S2=obs["S2"],
               target_S2=a.spin * (a.spin + 1), Cs=obs["Cs"], Pd=obs["Pd"], Cs_exact=obs0["Cs"],
               Pd_exact=obs0["Pd"], args=vars(a))
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"done: E={obs['E']:.10f} eps={res['eps']:.3e} <S^2>={obs['S2']:.3e} -> {out}")


def report(a):
    files = glob.glob(os.path.join(a.outdir, f"{a.tag}_*_seed*.json"))
    best = {}
    for fn in files:
        r = json.load(open(fn))
        k = r["ansatz"]
        if k not in best or r["E"] < best[k]["E"]:
            best[k] = r
    if "foil" not in best or a.route not in best:
        print(f"missing results (have {sorted(best)}); need foil and {a.route}")
        sys.exit(1)

    def dcs(r):
        return max(abs(r["Cs"][k] - r["Cs_exact"][k]) for k in r["Cs"])

    def dpd(r):
        vals = [abs(r["Pd"][k] - r["Pd_exact"][k]) for k in r["Pd"]]
        return float(np.mean(vals)) if vals else 0.0  # no non-overlapping R: guard vacuous
    rows = []
    for k, r in sorted(best.items()):
        rows.append(dict(ansatz=k, seed=r["seed"], nparam=r["nparam"], E=r["E"], eps=r["eps"],
                         S2_dev=r["S2"] - r["target_S2"], dCs=dcs(r), dPd=dpd(r)))
        print(f"{k:5s} seed={r['seed']} P={r['nparam']:5d} E={r['E']:.8f} eps={r['eps']:.3e} "
              f"<S2>-tgt={r['S2']-r['target_S2']:+.1e} dCs={dcs(r):.2e} dPd={dpd(r):.2e}")
    f, rt = best["foil"], best[a.route]
    M = rt["eps"] / f["eps"]
    s2ok = abs(rt["S2"] - rt["target_S2"]) < 1e-10
    guards = (dcs(rt) <= dcs(f)) and (dpd(rt) <= dpd(f) + 1e-4)
    if (M >= 1.0) or not s2ok:
        verdict = "FAIL"
    elif M <= 0.5 and guards:
        verdict = "PASS"
    else:
        verdict = "AMBIGUOUS"
    print(f"M = eps({a.route})/eps(foil) = {M:.3f}; S2 guard {'ok' if s2ok else 'VIOLATED'}; "
          f"correlation guards {'ok' if guards else 'VIOLATED'} -> {verdict}")
    with open(os.path.join(a.outdir, f"{a.tag}_report.json"), "w") as fh:
        json.dump(dict(rows=rows, M=M, route=a.route, verdict=verdict), fh, indent=1)


def main(argv=None):
    a = parse_args(argv)
    if a.mode == "smoke":
        smoke()
    elif a.mode == "run":
        run(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
