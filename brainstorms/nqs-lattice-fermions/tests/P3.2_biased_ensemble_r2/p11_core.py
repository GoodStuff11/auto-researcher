"""Core routines copied verbatim from tests/P1.1_sign_vs_amplitude/micro_test.py: lattice,
species basis, Hubbard model (JW signs, pairing), ED, UHF, backflow-Slater(-Jastrow) Ansatz and
exact energy. Copied rather than imported, per the round-2 brief. No CLI."""
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


