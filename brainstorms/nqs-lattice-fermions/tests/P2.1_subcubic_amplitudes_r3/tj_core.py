"""
tj_core.py -- physics core of the P2.1 round-3 test (no command line; imported by micro_test.py)
=================================================================================================

Contents
--------
* TJ           t-t'-J model on an Lx x Ly lattice in the fixed-(N_up, N_dn) sector with no
               double occupancy. Jordan-Wigner order: all up modes 0..V-1, then all down modes
               (the convention of p11_core.py / P1.1). Basis state = (iu, id): indices of the up
               and down bit patterns in the full single-species bases, restricted to disjoint
               patterns. H = -sum_{ij,s} T_ij P c^dag_is c_js P + J sum_<ij> (S_i.S_j - n_i n_j/4),
               T from p11_core.build_lattice (t on NN bonds, t' on diagonals); J on NN bonds only.
               Also: S^+ (sparse, to the (N_up+1, N_dn-1) t-J sector), total-spin Loewdin
               projector, and the singlet bond-pair operators used by p11_core.budget().
* SpaceGroup   the unitary representation U_g of the lattice space group on the t-J basis
               (translations x {1, Mx, My, MxMy} [x diagonal mirror D when Lx == Ly]; |G| = 4V
               or 8V) with the fermionic permutation sign, and the 1-D projector
               P_chi = |G|^-1 sum_g chi(g) U_g with chi(g) = <psi_0|U_g|psi_0> read from ED.
* LRAnsatz     psi(n) = exp(J(n)) sum_k w_k det B_k(n) with three modes:
                 "full": NNBF (Luo & Clark), B_{k,s} = (Phi0_{k,s} + out_{k,s}(n))[occ_s];
                 "lr"  : shared-reference low-rank (SK-LR / HFDS when K=1),
                         B_{k,s} = Phi0_s[occ_s] + Phih_{k,s}[occ_s] C_{k,s}(n), C r x N_s;
                 "agp" : geminal-frame shared-reference low-rank (SK-AGP; AGP x J when r=0),
                         B_k = F0[occ_up, occ_dn] + Phih_k[occ_up] C_k(n)[:, occ_dn], C r x V.
               out / C from a one-hidden-layer tanh trunk (width Hs) on x = 2n-1; J(n) from a
               separate tanh net (width Ha). Exact gradients (checked against finite differences).
* Fast local energies with carried tables (the per-amplitude algorithms whose cost is claimed):
                 LRCarried  ("lr" mode): setup, all-hop/exchange local energy, single-move
                            ratio and accepted-move table update (used for Metropolis sweeps);
                 eloc_fast_agp ("agp" mode): setup and all-hop/exchange local energy;
                 eloc_brute: rebuild every connected amplitude from scratch (sorted rows,
                            explicit Jordan-Wigner signs), independent of the fast algebra.
"""
import itertools
import time
from math import comb

import numpy as np
import scipy.sparse as sp

from p11_core import build_lattice, min_image, species_basis

# ============================================================================ bit helpers
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def popcount(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.int64))
    return _POP8[a.view(np.uint8).reshape(-1, 8)].sum(1).reshape(a.shape)


def between_mask(a, b):
    """Bit mask of sites strictly between a and b (arrays or ints)."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    one = np.int64(1)
    return ((one << hi) - one) ^ ((one << (lo + 1)) - one)


def below_mask(i):
    return (np.int64(1) << np.asarray(i, dtype=np.int64)) - np.int64(1)


# ============================================================================ model
class TJ:
    def __init__(self, Lx, Ly, bcx, bcy, nup, ndn, t, tp, J, build_H=True):
        self.Lx, self.Ly, self.bcx, self.bcy = Lx, Ly, bcx, bcy
        self.V = V = Lx * Ly
        self.nup, self.ndn, self.t, self.tp, self.J = nup, ndn, t, tp, J
        self.U = 8.0  # only used by p11_core.uhf_orbitals (Hubbard UHF start)
        self.T, self.coords, self.bonds = build_lattice(Lx, Ly, bcx, bcy, t, tp)
        self.nn = sorted({(min(i, j), max(i, j)) for (i, j, _) in self.bonds})
        self.su, self.lk_u, self.occ_u, self.rows_u = species_basis(V, nup)
        self.sd, self.lk_d, self.occ_d, self.rows_d = species_basis(V, ndn)
        self.Du, self.Dd = len(self.su), len(self.sd)
        iu_l, id_l = [], []
        free_combos = np.array(list(itertools.combinations(range(V - nup), ndn)), dtype=np.int64)
        free_combos = free_combos.reshape(-1, ndn)
        for iu, u in enumerate(self.su):
            free = np.array([s for s in range(V) if not (u >> s) & 1], dtype=np.int64)
            dm = (np.int64(1) << free[free_combos]).sum(1) if ndn > 0 else np.zeros(1, np.int64)
            iu_l.append(np.full(len(dm), iu)); id_l.append(self.lk_d[dm])
        iu_a, id_a = np.concatenate(iu_l), np.concatenate(id_l)
        keys = iu_a * self.Dd + id_a
        o = np.argsort(keys)
        self.keys, self._iu, self._id = keys[o], iu_a[o], id_a[o]
        self.D = len(self.keys)
        assert self.D == comb(V, nup) * comb(V - nup, ndn)
        self.bu, self.bd = self.su[self._iu], self.sd[self._id]   # bit patterns per basis state
        if build_H:
            self.H = self._build_H()

    def iu_id(self):
        return self._iu, self._id

    def index(self, bu, bd):
        """t-J basis index of up/down bit patterns (arrays); -1 if absent."""
        k = self.lk_u[bu] * self.Dd + self.lk_d[bd]
        pos = np.searchsorted(self.keys, k)
        pos = np.minimum(pos, self.D - 1)
        return np.where(self.keys[pos] == k, pos, -1)

    def _build_H(self):
        V, T, J = self.V, self.T, self.J
        bu, bd = self.bu, self.bd
        one = np.int64(1)
        diag = np.zeros(self.D)
        r_, c_, v_ = [], [], []
        idx = np.arange(self.D)
        for (i, j) in self.nn:
            anti = (((bu >> i) & 1) & ((bd >> j) & 1)) | (((bd >> i) & 1) & ((bu >> j) & 1))
            diag += -0.5 * J * anti
        empty = ~(bu | bd)
        for i in range(V):
            for j in range(V):
                if i == j or T[i, j] == 0.0:
                    continue
                bm = between_mask(i, j)
                for s in (0, 1):
                    b = bu if s == 0 else bd
                    src = np.nonzero(((b >> j) & 1) & ((empty >> i) & 1))[0]
                    if len(src) == 0:
                        continue
                    nb = b[src] ^ (one << j) ^ (one << i)
                    sg = 1.0 - 2.0 * (popcount(b[src] & bm) & 1)
                    tgt = self.index(nb, bd[src]) if s == 0 else self.index(bu[src], nb)
                    assert (tgt >= 0).all()
                    r_.append(tgt); c_.append(idx[src]); v_.append(-T[i, j] * sg)
        for (i, j) in self.nn:
            for (a, b) in ((i, j), (j, i)):   # up at b, down at a  ->  up at a, down at b
                src = np.nonzero(((bu >> b) & 1) & ((bd >> a) & 1))[0]
                if len(src) == 0:
                    continue
                bm = between_mask(a, b)
                nu_ = bu[src] ^ (one << b) ^ (one << a)
                nd_ = bd[src] ^ (one << a) ^ (one << b)
                sg = (1.0 - 2.0 * (popcount(bu[src] & bm) & 1)) * (1.0 - 2.0 * (popcount(bd[src] & bm) & 1))
                tgt = self.index(nu_, nd_)
                assert (tgt >= 0).all()
                r_.append(tgt); c_.append(src); v_.append(-0.5 * J * sg)
        r_.append(idx); c_.append(idx); v_.append(diag)
        H = sp.csr_matrix((np.concatenate(v_), (np.concatenate(r_), np.concatenate(c_))),
                          shape=(self.D, self.D))
        return H

    # ---------------------------------------------------------------- spin
    def spin_setup(self):
        V, nu, nd = self.V, self.nup, self.ndn
        self.Sz = 0.5 * (nu - nd)
        Ntot = nu + nd
        self.Smax = 0.5 * min(Ntot, 2 * V - Ntot)
        self.S0 = abs(self.Sz)
        if nd == 0:
            self.Sp = sp.csr_matrix((1, self.D)); return
        tgt = TJ(self.Lx, self.Ly, self.bcx, self.bcy, nu + 1, nd - 1, self.t, self.tp, self.J,
                 build_H=False)
        one = np.int64(1)
        r_, c_, v_ = [], [], []
        for i in range(V):
            src = np.nonzero((self.bd >> i) & 1)[0]
            if len(src) == 0:
                continue
            nu_ = self.bu[src] | (one << i); nd_ = self.bd[src] ^ (one << i)
            sg = 1.0 - 2.0 * ((popcount(self.bu[src] & below_mask(i)) + popcount(self.bd[src] & below_mask(i))) & 1)
            r_.append(tgt.index(nu_, nd_)); c_.append(src); v_.append(sg)
        self.Sp = sp.csr_matrix((np.concatenate(v_), (np.concatenate(r_), np.concatenate(c_))),
                                shape=(tgt.D, self.D))

    def S2(self, psi):
        return self.Sp.T @ (self.Sp @ psi) + (self.Sz ** 2 + self.Sz) * psi

    def S2_expect(self, psi):
        return float(psi @ self.S2(psi) / (psi @ psi))

    def spin_project(self, psi, S):
        out = psi.copy()
        tgt = S * (S + 1)
        s = self.S0
        while s <= self.Smax + 1e-9:
            if abs(s - S) > 1e-9:
                ss = s * (s + 1)
                out = (self.S2(out) - ss * out) / (tgt - ss)
            s += 1
        return out

    # ---------------------------------------------------------------- pairing (P1.1 definition)
    def pair_setup(self, r_min):
        self.pair_ops = []
        if self.nup < 1 or self.ndn < 1:
            return
        tgt = TJ(self.Lx, self.Ly, self.bcx, self.bcy, self.nup - 1, self.ndn - 1, self.t, self.tp,
                 self.J, build_H=False)
        one = np.int64(1)

        def cc(iu_site, jd_site):   # c_{i up} c_{j dn}; global (-1)^N_up dropped (common)
            src = np.nonzero(((self.bu >> iu_site) & 1) & ((self.bd >> jd_site) & 1))[0]
            nu_ = self.bu[src] ^ (one << iu_site); nd_ = self.bd[src] ^ (one << jd_site)
            sg = 1.0 - 2.0 * ((popcount(self.bu[src] & below_mask(iu_site))
                               + popcount(self.bd[src] & below_mask(jd_site))) & 1)
            return sp.csr_matrix((sg, (tgt.index(nu_, nd_), src)), shape=(tgt.D, self.D))

        self.pair_bonds = list(self.bonds)
        for (i, j, _) in self.pair_bonds:
            self.pair_ops.append(((cc(i, j) + cc(j, i)) / np.sqrt(2.0)).tocsr())
        b0 = next(b for b in self.bonds if b[2] == "x" and b[0] == 0)
        self.ib0 = self.pair_bonds.index(b0)
        c0 = self.bond_center(b0)
        self.pair_dist = np.array([self.dist(c0, self.bond_center(b)) for b in self.pair_bonds])
        self.pair_mask = self.pair_dist >= r_min

    def bond_center(self, b):
        i, j, _ = b
        ci = self.coords[i]
        dx = min_image(self.coords[j] - ci, np.array([self.Lx, self.Ly]), "pbc")
        if self.bcx == "obc":
            dx[0] = self.coords[j][0] - ci[0]
        if self.bcy == "obc":
            dx[1] = self.coords[j][1] - ci[1]
        return ci + 0.5 * dx

    def dist(self, a, b):
        d = b - a
        return float(np.hypot(min_image(d[0], self.Lx, self.bcx), min_image(d[1], self.Ly, self.bcy)))

    def pairing(self, psi):
        if not self.pair_ops:
            return np.zeros(0)
        p = psi / np.linalg.norm(psi)
        vecs = [M @ p for M in self.pair_ops]
        v0 = vecs[self.ib0]
        return np.array([v0 @ v for v in vecs])


# ============================================================================ symmetry
def group_elements(Lx, Ly, diag):
    """(label, site permutation p) for g = T^(a,b) M^(c,d) D^e: site (x,y) -> D^e -> mirrors ->
    translation. D (x <-> y) only if diag and Lx == Ly."""
    V = Lx * Ly
    out = []
    for e in ((0, 1) if (diag and Lx == Ly) else (0,)):
        for c in (0, 1):
            for d in (0, 1):
                for a in range(Lx):
                    for b in range(Ly):
                        p = np.empty(V, dtype=np.int64)
                        for y in range(Ly):
                            for x in range(Lx):
                                x2, y2 = (y, x) if e else (x, y)
                                x2 = (-x2) % Lx if c else x2
                                y2 = (-y2) % Ly if d else y2
                                p[x + Lx * y] = (x2 + a) % Lx + Lx * ((y2 + b) % Ly)
                        out.append(((a, b, c, d, e), p))
    return out


def species_map(rows, lookup, p):
    """Image index and permutation parity sign for every species state under site perm p."""
    img = p[rows]                                         # (Ds, N)
    N = rows.shape[1]
    inv = np.zeros(len(rows), dtype=np.int64)
    for a in range(N):
        for b in range(a + 1, N):
            inv += img[:, a] > img[:, b]
    bits = (np.int64(1) << img).sum(1) if N > 0 else np.zeros(len(rows), np.int64)
    return lookup[bits], 1 - 2 * (inv & 1)


class SpaceGroup:
    """U_g psi (n) = sinv[g][n] psi(inv[g][n]); P_chi = |G|^-1 sum_g chi_g U_g."""

    def __init__(self, m, diag=True):
        self.m = m
        self.elems, inv_l, sinv_l = [], [], []
        idx = np.arange(m.D)
        for e, p in group_elements(m.Lx, m.Ly, diag):
            mu, su = species_map(m.rows_u, m.lk_u, p)
            md, sd = species_map(m.rows_d, m.lk_d, p)
            img = m.index(m.su[mu[m._iu]], m.sd[md[m._id]])
            assert (img >= 0).all()
            sgn = (su[m._iu] * sd[m._id]).astype(np.int8)
            inv = np.empty(m.D, dtype=np.int32); inv[img] = idx
            sinv = np.empty(m.D, dtype=np.int8); sinv[img] = sgn
            self.elems.append(e); inv_l.append(inv); sinv_l.append(sinv)
        self.inv, self.sinv = np.array(inv_l), np.array(sinv_l)
        self.G = len(self.elems)
        self.chi = np.ones(self.G)

    def apply(self, psi, k):
        return self.sinv[k] * psi[self.inv[k]]

    def set_from(self, psi0):
        psi0 = psi0 / np.linalg.norm(psi0)
        chi = np.array([psi0 @ self.apply(psi0, k) for k in range(self.G)])
        if np.max(np.abs(np.abs(chi) - 1)) > 1e-6:
            raise RuntimeError(f"psi_0 not in a 1-D irrep of this group: |chi| range "
                               f"[{np.abs(chi).min():.3f}, {np.abs(chi).max():.3f}]")
        self.chi = np.round(chi)
        gen = {}
        for k, e in enumerate(self.elems):
            if e in ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)):
                gen[{(1, 0, 0, 0, 0): "Tx", (0, 1, 0, 0, 0): "Ty", (0, 0, 1, 0, 0): "Mx",
                     (0, 0, 0, 1, 0): "My", (0, 0, 0, 0, 1): "D"}[e]] = int(self.chi[k])
        return gen

    def project(self, psi):
        out = np.zeros_like(psi)
        for k in range(self.G):
            out += self.chi[k] * self.apply(psi, k)
        return out / self.G


# ============================================================================ ansatz
class LRAnsatz:
    """psi = exp(J) * sum_k w_k det B_k; see module docstring. Requires nup == ndn for 'agp'."""

    def __init__(self, model, ref, mode, K=1, r=0, Hs=0, Ha=0, shared_ref=True, seed=0,
                 w2_scale=1e-3):
        rng = np.random.default_rng(seed)
        self.m = model
        V, nu, nd = model.V, model.nup, model.ndn
        self.V, self.nu, self.nd = V, nu, nd
        self.mode, self.K = mode, K
        self.r = r if mode in ("lr", "agp") else 0
        self.Kref = 1 if (shared_ref or mode != "full") else K
        p = {}
        if mode == "agp":
            assert nu == nd
            p["F0"] = ref.copy()
            if self.r > 0:
                p["Ph"] = rng.standard_normal((K, V, self.r)) / np.sqrt(V)
            self.per = self.r * V
        else:
            Pu, Pd = ref
            p["P0u"] = np.repeat(Pu[None], self.Kref, 0).copy()
            p["P0d"] = np.repeat(Pd[None], self.Kref, 0).copy()
            if self.Kref > 1:
                p["P0u"][1:] += 0.01 * rng.standard_normal(p["P0u"][1:].shape)
                p["P0d"][1:] += 0.01 * rng.standard_normal(p["P0d"][1:].shape)
            if self.r > 0:
                p["Phu"] = rng.standard_normal((K, V, self.r)) / np.sqrt(V)
                p["Phd"] = rng.standard_normal((K, V, self.r)) / np.sqrt(V)
            self.per = self.r * (nu + nd) if mode == "lr" else V * (nu + nd)
        self.outdim = K * self.per
        self.trunk = Hs > 0 and self.outdim > 0
        self.Hs, self.Ha = (Hs if self.trunk else 0), Ha
        if self.trunk:
            p["W1"] = rng.standard_normal((2 * V, Hs)) / np.sqrt(2 * V)
            p["b1"] = np.zeros(Hs)
            p["W2"] = w2_scale * rng.standard_normal((Hs, self.outdim))
            p["b2"] = np.zeros(self.outdim)
        if Ha > 0:
            p["W1a"] = rng.standard_normal((2 * V, Ha)) / np.sqrt(2 * V)
            p["b1a"] = np.zeros(Ha)
            p["w2a"] = np.zeros(Ha)
        if K > 1:
            p["wk"] = np.ones(K)
        self.p = p
        self.keys = list(p.keys())

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def flat(self):
        return np.concatenate([self.p[k].ravel() for k in self.keys])

    def set_flat(self, x):
        o = 0
        for k in self.keys:
            n = self.p[k].size
            self.p[k] = x[o:o + n].reshape(self.p[k].shape)
            o += n

    def _wk(self):
        return self.p["wk"] if self.K > 1 else np.ones(1)

    def forward(self, ou, od, ru, rd):
        p, K, V, nu, nd, r = self.p, self.K, self.V, self.nu, self.nd, self.r
        B = ou.shape[0]
        x = np.concatenate([ou, od], 1).astype(float) * 2 - 1
        c = dict(x=x, ru=ru, rd=rd, B=B)
        if self.trunk:
            h = np.tanh(x @ p["W1"] + p["b1"])
            out = (h @ p["W2"] + p["b2"]).reshape(B, K, self.per)
            c["h"] = h
        if self.mode == "agp":
            F0r = p["F0"][ru[:, :, None], rd[:, None, :]]            # (B, N, N)
            if r > 0:
                C = out.reshape(B, K, r, V)
                Cr = np.take_along_axis(C, np.broadcast_to(rd[:, None, None, :], (B, K, r, nu)), 3)
                Ph = np.transpose(p["Ph"][:, ru], (1, 0, 2, 3))      # (B, K, N, r)
                M = F0r[:, None] + Ph @ Cr
                c["Cr"], c["Ph"] = Cr, Ph
            else:
                M = F0r[:, None]
            c["M"] = M
            dk = np.linalg.det(M)
        else:
            Bm = {}
            for s, rows, N in (("u", ru, nu), ("d", rd, nd)):
                P0 = p["P0" + s]
                P0r = np.transpose(P0[:, rows], (1, 0, 2, 3))
                if self.mode == "lr":
                    if r > 0:
                        o0 = 0 if s == "u" else r * nu
                        C = out[:, :, o0:o0 + r * N].reshape(B, K, r, N)
                        Ph = np.transpose(p["Ph" + s][:, rows], (1, 0, 2, 3))
                        M = P0r + Ph @ C
                        c["C" + s], c["Ph" + s] = C, Ph
                    else:
                        M = np.broadcast_to(P0r, (B, K, N, N)).copy()
                else:
                    o0 = 0 if s == "u" else V * nu
                    Of = out[:, :, o0:o0 + V * N].reshape(B, K, V, N) + P0[None]
                    M = np.take_along_axis(Of, rows[:, None, :, None], axis=2)
                Bm[s] = M
            c["Bm"] = Bm
            du = np.linalg.det(Bm["u"]) if nu > 0 else np.ones((B, K))
            dd = np.linalg.det(Bm["d"]) if nd > 0 else np.ones((B, K))
            dk = du * dd
        c["dk"] = dk
        S = dk @ self._wk()
        if self.Ha > 0:
            ha = np.tanh(x @ p["W1a"] + p["b1a"])
            J = ha @ p["w2a"]
            c["ha"] = ha
        else:
            J = np.zeros(B)
        c["J"], c["S"] = J, S
        c["psi"] = np.exp(J) * S
        return c

    @staticmethod
    def _invT(M):
        try:
            return np.swapaxes(np.linalg.inv(M), -1, -2)
        except np.linalg.LinAlgError:
            return np.swapaxes(np.linalg.pinv(M), -1, -2)

    def grad(self, c, coef, per_sample):
        """sum_n coef(n) dpsi(n)/dtheta (per_sample=False) or rows coef(n) dpsi(n)/dtheta."""
        p, K, V, B, r = self.p, self.K, self.V, c["B"], self.r
        eJ = np.exp(c["J"])
        wk = self._wk()
        tk = (coef * eJ)[:, None] * wk[None] * c["dk"]
        bi = np.arange(B)[:, None]
        red = (lambda a: a) if per_sample else (lambda a: a.sum(0))
        g = {}
        dout = np.zeros((B, K, self.per)) if self.trunk else None
        if self.mode == "agp":
            ru, rd = c["ru"], c["rd"]
            N = self.nu
            M = tk[:, :, None, None] * self._invT(c["M"])           # (B, K, N, N)
            Ms = M.sum(1)
            if per_sample:
                Z = np.zeros((B, V, V))
                Z[bi[:, :, None], ru[:, :, None], rd[:, None, :]] = Ms
                g["F0"] = Z
            else:
                fi = (ru[:, :, None] * V + rd[:, None, :]).ravel()
                g["F0"] = np.bincount(fi, weights=Ms.ravel(), minlength=V * V).reshape(V, V)
            if r > 0:
                Cr, Ph = c["Cr"], c["Ph"]
                gPh = M @ np.swapaxes(Cr, -1, -2)                     # (B, K, N, r)
                Zh = np.zeros((B, K, V, r))
                for kk in range(K):
                    Zh[bi, kk, ru] = gPh[:, kk]
                g["Ph"] = red(Zh)
                dCr = np.swapaxes(Ph, -1, -2) @ M                     # (B, K, r, N)
                dC = np.zeros((B, K, r, V))
                for kk in range(K):
                    for q in range(r):
                        dC[bi, kk, q, rd] = dCr[:, kk, q]
                dout[:] = dC.reshape(B, K, -1)
        else:
            for s, rows, N in (("u", c["ru"], self.nu), ("d", c["rd"], self.nd)):
                if N == 0:
                    continue
                M = tk[:, :, None, None] * self._invT(c["Bm"][s])
                Msum = M.sum(1, keepdims=True) if self.Kref == 1 else M
                Z = np.zeros((B, self.Kref, V, N))
                for kk in range(self.Kref):
                    Z[bi, kk, rows] = Msum[:, kk]
                g["P0" + s] = red(Z)
                if self.mode == "lr" and r > 0:
                    C, Ph = c["C" + s], c["Ph" + s]
                    Zh = np.zeros((B, K, V, r))
                    gPh = M @ np.swapaxes(C, -1, -2)
                    for kk in range(K):
                        Zh[bi, kk, rows] = gPh[:, kk]
                    g["Ph" + s] = red(Zh)
                    dC = np.swapaxes(Ph, -1, -2) @ M
                    o0 = 0 if s == "u" else r * self.nu
                    dout[:, :, o0:o0 + r * N] = dC.reshape(B, K, -1)
                elif self.mode == "full":
                    Zo = np.zeros((B, K, V, N))
                    for kk in range(K):
                        Zo[bi, kk, rows] = M[:, kk]
                    o0 = 0 if s == "u" else V * self.nu
                    dout[:, :, o0:o0 + V * N] = Zo.reshape(B, K, -1)
        x = c["x"]
        if self.trunk:
            do = dout.reshape(B, -1)
            h = c["h"]
            dpre = (do @ p["W2"].T) * (1 - h ** 2)
            if per_sample:
                g["W2"] = (h[:, :, None] * do[:, None, :]).reshape(B, -1)
                g["W1"] = (x[:, :, None] * dpre[:, None, :]).reshape(B, -1)
            else:
                g["W2"], g["W1"] = h.T @ do, x.T @ dpre
            g["b2"], g["b1"] = red(do), red(dpre)
        if self.Ha > 0:
            cp = coef * c["psi"]
            ha = c["ha"]
            dprea = cp[:, None] * p["w2a"][None] * (1 - ha ** 2)
            g["W1a"] = (x[:, :, None] * dprea[:, None, :]).reshape(B, -1) if per_sample else x.T @ dprea
            g["b1a"] = red(dprea)
            g["w2a"] = red(cp[:, None] * ha)
        if K > 1:
            g["wk"] = red((coef * eJ)[:, None] * c["dk"])
        if per_sample:
            return np.concatenate([g[k].reshape(B, -1) for k in self.keys], 1)
        return np.concatenate([g[k].ravel() for k in self.keys])

    def _occ(self, idx):
        m = self.m
        iu, idd = m.iu_id()
        iu, idd = iu[idx], idd[idx]
        return m.occ_u[iu], m.occ_d[idd], m.rows_u[iu], m.rows_d[idd]

    def amplitudes(self, chunk):
        D = self.m.D
        out = np.empty(D)
        for s in range(0, D, chunk):
            idx = np.arange(s, min(s + chunk, D))
            out[idx] = self.forward(*self._occ(idx))["psi"]
        return out

    def force(self, rcoef, chunk):
        D, tot = self.m.D, None
        for s in range(0, D, chunk):
            idx = np.arange(s, min(s + chunk, D))
            gg = self.grad(self.forward(*self._occ(idx)), rcoef[idx], False)
            tot = gg if tot is None else tot + gg
        return tot

    def per_sample_O(self, idx):
        c = self.forward(*self._occ(idx))
        psi = c["psi"]
        safe = np.where(np.abs(psi) > 1e-300, psi, 1e-300)
        return self.grad(c, 1.0 / safe, True)

    def per_sample_O_proj(self, sg, idx, psiP, o_chunk=8):
        """O_P(n) = d log psi_P(n)/d theta with psi_P = P_chi psi_theta, evaluated per amplitude
        on the |G| image configurations of each sample."""
        G = sg.G
        out = np.empty((len(idx), self.n_params()))
        for s in range(0, len(idx), o_chunk):
            ii = idx[s:s + o_chunk]
            B = len(ii)
            img = sg.inv[:, ii].T.astype(np.int64)                      # (B, G)
            coef = sg.chi[None, :] * sg.sinv[:, ii].T / (G * psiP[ii][:, None])
            c = self.forward(*self._occ(img.ravel()))
            gr = self.grad(c, coef.ravel(), True)
            out[s:s + B] = gr.reshape(B, G, -1).sum(1)
        return out

    def psiP_per_amplitude(self, sg, idx):
        """psi_P(n) for basis indices idx from per-amplitude forward passes on |G| images."""
        img = sg.inv[:, idx].T.astype(np.int64)
        ps = self.forward(*self._occ(img.ravel()))["psi"].reshape(len(idx), sg.G)
        return (ps * sg.chi[None, :] * sg.sinv[:, idx].T).sum(1) / sg.G


# ============================================================================ connected configs
def connections(T, nn, J, ou, od):
    """All connected configurations of a t-J configuration (0/1 arrays ou, od).
    Returns (diag, hops, exch): hops = list of (spin, x, y, coef_inplace) with coef_inplace the
    coefficient multiplying the in-place-row ratio (-T_yx); exch = list of (a, b) meaning the up
    particle at b moves to a and the down particle at a moves to b (coef -J/2)."""
    V = len(ou)
    diag = 0.0
    for (i, j) in nn:
        if (ou[i] and od[j]) or (od[i] and ou[j]):
            diag += -0.5 * J
    hops = []
    occ = ou + od
    for s, o in ((0, ou), (1, od)):
        for x in np.nonzero(o)[0]:
            for y in np.nonzero(T[x])[0]:
                if not occ[y]:
                    hops.append((s, x, y, -T[y, x]))
    exch = []
    for (i, j) in nn:
        for (a, b) in ((i, j), (j, i)):
            if ou[b] and od[a]:
                exch.append((a, b))
    return diag, hops, exch


def eloc_brute(ans, T, nn, J, ou, od, chunk=64):
    """Local energy by rebuilding every connected amplitude (sorted rows, explicit JW sign)."""
    def amp(OU, OD):
        ru = np.array([np.nonzero(o)[0] for o in OU])
        rd = np.array([np.nonzero(o)[0] for o in OD])
        return ans.forward(OU, OD, ru, rd)["psi"]
    psi0 = amp(ou[None], od[None])[0]
    diag, hops, exch = connections(T, nn, J, ou, od)
    cfg, cf = [], []
    for (s, x, y, cfi) in hops:
        o = (ou if s == 0 else od).copy()
        sgn = (-1) ** int(o[min(x, y) + 1:max(x, y)].sum())
        o[x] = 0; o[y] = 1
        cfg.append((o, od) if s == 0 else (ou, o)); cf.append(cfi * sgn)
    for (a, b) in exch:
        u2, d2 = ou.copy(), od.copy()
        su = (-1) ** int(ou[min(a, b) + 1:max(a, b)].sum())
        sd = (-1) ** int(od[min(a, b) + 1:max(a, b)].sum())
        u2[b] = 0; u2[a] = 1; d2[a] = 0; d2[b] = 1
        cfg.append((u2, d2)); cf.append(-0.5 * J * su * sd)
    E = diag
    for s in range(0, len(cfg), chunk):
        blk = cfg[s:s + chunk]
        ps = amp(np.array([b[0] for b in blk]), np.array([b[1] for b in blk]))
        E += float(np.sum(np.array(cf[s:s + chunk]) * ps / psi0))
    return E


class LRCarried:
    """Carried tables of an 'lr'-mode LRAnsatz at one configuration (both spins), with the exact
    all-connection local energy, single-move ratios and accepted-move updates."""

    def __init__(self, ans, T, nn, J, ou, od):
        assert ans.mode == "lr" and ans.Kref == 1
        self.a, self.T, self.nn, self.J = ans, T, nn, J
        self.nbr = [np.nonzero(T[x])[0] for x in range(T.shape[0])]
        self.setup(ou.copy(), od.copy())

    def setup(self, ou, od):
        a, p = self.a, self.a.p
        self.occ = {"u": ou, "d": od}
        self.x = np.concatenate([ou, od]).astype(float) * 2 - 1
        self.sd = {}
        for s in ("u", "d"):
            rows = np.nonzero(self.occ[s])[0]
            A0 = p["P0" + s][0][rows]
            Ainv = np.linalg.inv(A0)
            G0 = p["P0" + s][0] @ Ainv
            e = dict(rows=rows, Ainv=Ainv, G0=G0)
            if a.r > 0:
                Phr = p["Ph" + s][:, rows]
                e["Y"] = Ainv[None] @ Phr
                e["Hk"] = G0[None] @ Phr
            self.sd[s] = e
        if a.trunk:
            self.pre = self.x @ p["W1"] + p["b1"]
        if a.Ha > 0:
            self.prea = self.x @ p["W1a"] + p["b1a"]
        self.S = self._termsum(self._out(self.pre[None]) if a.trunk else None,
                               {s: self.sd[s]["Y"][None] for s in self.sd} if a.r > 0 else None)[0]
        self.Jv = self._J(self.prea[None])[0] if a.Ha > 0 else 0.0

    # helpers
    def _out(self, pre):
        p = self.a.p
        return np.tanh(pre) @ p["W2"] + p["b2"]

    def _J(self, prea):
        return np.tanh(prea) @ self.a.p["w2a"]

    def _C(self, out, s):
        a = self.a
        B = out.shape[0]
        o0 = 0 if s == "u" else a.r * a.nu
        N = a.nu if s == "u" else a.nd
        return out.reshape(B, a.K, -1)[:, :, o0:o0 + a.r * N].reshape(B, a.K, a.r, N)

    def _termsum(self, out, Ys):
        a = self.a
        if a.r == 0:
            return np.full(1 if out is None else out.shape[0], float(a._wk().sum()))
        d = np.ones((out.shape[0], a.K))
        for s in ("u", "d"):
            d = d * np.linalg.det(np.eye(a.r) + self._C(out, s) @ Ys[s])
        return d @ a._wk()

    def _Ynew(self, s, ii, ys, rho0):
        e = self.sd[s]
        p = self.a.p
        vec = (p["Ph" + s][:, ys] - e["Hk"][:, ys]) / rho0[None, :, None]           # (K, n, r)
        return e["Y"][None] + e["Ainv"][:, ii].T[:, None, :, None] * np.transpose(vec, (1, 0, 2))[:, :, None, :]

    def _dpre(self, W, moves):
        """Pre-activation shift for a list of moves [(spin_index, x, y), ...] (arrays)."""
        V = self.a.V
        d = 0
        for (si, xs, ys) in moves:
            d = d + 2 * (W[si * V + ys] - W[si * V + xs])
        return d

    def ratios(self, spins_moves):
        """In-place ratios psi(n')/psi(n) for a batch of connected configurations, each given by
        a list of single-species moves. spins_moves: list (length = moves per config, 1 or 2) of
        tuples (s, ii, xs, ys) with arrays of equal length nb (one entry per connected config)."""
        a, p = self.a, self.a.p
        nb = len(spins_moves[0][1])
        mv = [(0 if s == "u" else 1, xs, ys) for (s, ii, xs, ys) in spins_moves]
        rho = np.ones(nb)
        Ys = {s: np.broadcast_to(self.sd[s]["Y"][None], (nb,) + self.sd[s]["Y"].shape)
              for s in self.sd} if a.r > 0 else None
        for (s, ii, xs, ys) in spins_moves:
            r0 = self.sd[s]["G0"][ys, ii]
            rho = rho * r0
            if a.r > 0:
                Ys[s] = self._Ynew(s, ii, ys, r0)
        out = self._out(self.pre[None] + self._dpre(p["W1"], mv)) if a.trunk else None
        tot = self._termsum(out, Ys) if a.r > 0 else np.full(nb, self.S)
        fac = np.exp(self._J(self.prea[None] + self._dpre(p["W1a"], mv)) - self.Jv) if a.Ha > 0 else 1.0
        return fac * rho * tot / self.S

    def eloc(self):
        ou, od = self.occ["u"], self.occ["d"]
        diag, hops, exch = connections(self.T, self.nn, self.J, ou, od)
        E = diag
        slot = {s: {int(x): i for i, x in enumerate(self.sd[s]["rows"])} for s in self.sd}
        for s, si in (("u", 0), ("d", 1)):
            hs = [h for h in hops if h[0] == si]
            if not hs:
                continue
            xs = np.array([h[1] for h in hs]); ys = np.array([h[2] for h in hs])
            ii = np.array([slot[s][int(x)] for x in xs]); cf = np.array([h[3] for h in hs])
            E += float(np.sum(cf * self.ratios([(s, ii, xs, ys)])))
        if exch:
            A = np.array([e[0] for e in exch]); Bb = np.array([e[1] for e in exch])
            iu = np.array([slot["u"][int(b)] for b in Bb]); jd = np.array([slot["d"][int(x)] for x in A])
            E += float(np.sum(-0.5 * self.J * self.ratios([("u", iu, Bb, A), ("d", jd, A, Bb)])))
        return E

    def accept(self, moves):
        """Apply an accepted move (list of (s, i, x, y)) with Sherman-Morrison table updates."""
        a, p = self.a, self.a.p
        for (s, i, x, y) in moves:
            e = self.sd[s]
            rho0 = e["G0"][y, i]
            if a.r > 0:
                e["Y"] = self._Ynew(s, np.array([i]), np.array([y]), np.array([rho0]))[0]
            g = e["G0"][y].copy(); g[i] -= 1.0
            e["Ainv"] = e["Ainv"] - np.outer(e["Ainv"][:, i], g) / rho0
            e["G0"] = e["G0"] - np.outer(e["G0"][:, i], g) / rho0
            e["rows"] = e["rows"].copy(); e["rows"][i] = y
            if a.r > 0:
                e["Hk"] = e["G0"][None] @ p["Ph" + s][:, e["rows"]]
            si = 0 if s == "u" else 1
            self.occ[s] = self.occ[s].copy(); self.occ[s][x] = 0; self.occ[s][y] = 1
            if a.trunk:
                self.pre = self.pre + 2 * (p["W1"][si * a.V + y] - p["W1"][si * a.V + x])
            if a.Ha > 0:
                self.prea = self.prea + 2 * (p["W1a"][si * a.V + y] - p["W1a"][si * a.V + x])
        self.S = self._termsum(self._out(self.pre[None]) if a.trunk else None,
                               {s: self.sd[s]["Y"][None] for s in self.sd} if a.r > 0 else None)[0]
        self.Jv = self._J(self.prea[None])[0] if a.Ha > 0 else 0.0

    def sweep(self, rng, nprop):
        """nprop Metropolis proposals (random hop of a random particle, or a random NN
        exchange), each costing one single-configuration ratio; accepted moves update tables."""
        acc = 0
        V = self.a.V
        for _ in range(nprop):
            if rng.random() < 0.5 and self.nn:
                i, j = self.nn[rng.integers(len(self.nn))]
                ou, od = self.occ["u"], self.occ["d"]
                if ou[j] and od[i]:
                    a_, b_ = i, j
                elif ou[i] and od[j]:
                    a_, b_ = j, i
                else:
                    continue
                iu = int(np.nonzero(self.sd["u"]["rows"] == b_)[0][0])
                jd = int(np.nonzero(self.sd["d"]["rows"] == a_)[0][0])
                mv = [("u", iu, b_, a_), ("d", jd, a_, b_)]
            else:
                s = "u" if rng.random() < 0.5 else "d"
                rows = self.sd[s]["rows"]
                i = int(rng.integers(len(rows))); x = int(rows[i])
                nb = self.nbr[x]
                y = int(nb[rng.integers(len(nb))])
                if self.occ["u"][y] or self.occ["d"][y]:
                    continue
                mv = [(s, i, x, y)]
            rt = self.ratios([(s_, np.array([i_]), np.array([x_]), np.array([y_])) for (s_, i_, x_, y_) in mv])[0]
            if rng.random() < min(1.0, rt * rt):
                self.accept(mv)
                acc += 1
        return acc

    def drift(self):
        """Max deviation of carried tables from a fresh setup at the current configuration."""
        old = {s: {k: v.copy() for k, v in self.sd[s].items()} for s in self.sd}
        S_old = self.S
        self.setup(self.occ["u"].copy(), self.occ["d"].copy())
        d = abs(S_old - self.S) / (abs(self.S) + 1e-300)
        for s in self.sd:
            for k in ("Ainv", "G0", "Y", "Hk"):
                if k in self.sd[s]:
                    ref = self.sd[s][k]
                    d = max(d, float(np.max(np.abs(old[s][k] - ref)) / (1 + np.max(np.abs(ref)))))
        return d


def eloc_fast_agp(ans, T, nn, J, ou, od):
    """Exact local energy of an 'agp'-mode LRAnsatz with carried tables (setup timed separately).
    Reference B0 = F0[occ_up, occ_dn]; tables Binv, W = Binv F0[ru,:], Wp = F0[:,rd] Binv,
    Y_k = Binv Phih_k[ru], Hk = Wp Phih_k[ru]. Per connection O(N r + N) plus the network update."""
    a, p, r, K, V = ans, ans.p, ans.r, ans.K, ans.V
    t0 = time.perf_counter()
    ru, rd = np.nonzero(ou)[0], np.nonzero(od)[0]
    F0 = p["F0"]
    Binv = np.linalg.inv(F0[np.ix_(ru, rd)])
    W = Binv @ F0[ru, :]                      # (N, V)
    Wp = F0[:, rd] @ Binv                     # (V, N)
    if r > 0:
        Ph = p["Ph"]
        Y = Binv[None] @ Ph[:, ru]            # (K, N, r)
        Hk = Wp[None] @ Ph[:, ru]             # (K, V, r)
    x = np.concatenate([ou, od]).astype(float) * 2 - 1
    pre = x @ p["W1"] + p["b1"] if a.trunk else None
    prea = x @ p["W1a"] + p["b1a"] if a.Ha > 0 else None
    t_setup = time.perf_counter() - t0
    t0 = time.perf_counter()
    wk = a._wk()

    def termsum(pre_b, Yb, cols):
        nb = Yb.shape[0]
        if r == 0:
            return np.full(nb, float(wk.sum()))
        out = (np.tanh(pre_b) @ p["W2"] + p["b2"]).reshape(nb, K, r, V)
        Cr = np.take_along_axis(out, np.broadcast_to(cols[:, None, None, :], (nb, K, r, len(rd))), 3)
        return np.linalg.det(np.eye(r) + Cr @ Yb) @ wk

    def Jv(prea_b):
        return np.tanh(prea_b) @ p["w2a"] if a.Ha > 0 else 0.0

    S = termsum(pre[None], Y[None], rd[None])[0] if r > 0 else float(wk.sum())
    J0 = Jv(prea[None])[0] if a.Ha > 0 else 0.0
    diag, hops, exch = connections(T, nn, J, ou, od)
    E = diag
    su = {int(s_): i for i, s_ in enumerate(ru)}
    sdn = {int(s_): i for i, s_ in enumerate(rd)}
    for si in (0, 1):
        hs = [h for h in hops if h[0] == si]
        if not hs:
            continue
        xs = np.array([h[1] for h in hs]); ys = np.array([h[2] for h in hs]); cf = np.array([h[3] for h in hs])
        nb = len(xs)
        d = (2 * (p["W1"][si * V + ys] - p["W1"][si * V + xs])) if a.trunk else 0
        da = (2 * (p["W1a"][si * V + ys] - p["W1a"][si * V + xs])) if a.Ha > 0 else 0
        cols = np.broadcast_to(rd[None], (nb, len(rd))).copy()
        if si == 0:
            ii = np.array([su[int(v)] for v in xs])
            rho0 = Wp[ys, ii]
            if r > 0:
                vec = (Ph[:, ys] - Hk[:, ys]) / rho0[None, :, None]                 # (K, nb, r)
                Yb = Y[None] + Binv[:, ii].T[:, None, :, None] * np.transpose(vec, (1, 0, 2))[:, :, None, :]
        else:
            jj = np.array([sdn[int(v)] for v in xs])
            rho0 = W[jj, ys]
            cols[np.arange(nb), jj] = ys
            if r > 0:
                u = W[:, ys].T.copy(); u[np.arange(nb), jj] -= 1.0                   # (nb, N)
                Yj = Y[:, jj, :]                                                      # (K, nb, r)
                Yb = Y[None] - u[:, None, :, None] * np.transpose(Yj, (1, 0, 2))[:, :, None, :] / rho0[:, None, None, None]
        tot = termsum(pre[None] + d, Yb, cols) if r > 0 else np.full(nb, S)
        fac = np.exp(Jv(prea[None] + da) - J0) if a.Ha > 0 else 1.0
        E += float(np.sum(cf * fac * rho0 * tot / S))
    if exch:
        A = np.array([e[0] for e in exch]); Bb = np.array([e[1] for e in exch])
        nb = len(A)
        ii = np.array([su[int(v)] for v in Bb]); jj = np.array([sdn[int(v)] for v in A])
        rho_u = Wp[A, ii]
        q = np.sum(Wp[A] * F0[ru][:, Bb].T, 1)                                        # Wp[a] . F0[ru, b]
        fab, fbb = F0[A, Bb], F0[Bb, Bb]
        kap = (fab - fbb) - (q - fbb + (rho_u - 1) * (fab - fbb)) / rho_u
        bji = Binv[jj, ii]
        rho_d = W[jj, Bb] + bji * kap
        cols = np.broadcast_to(rd[None], (nb, len(rd))).copy()
        cols[np.arange(nb), jj] = Bb
        d = (2 * (p["W1"][A] - p["W1"][Bb]) + 2 * (p["W1"][V + Bb] - p["W1"][V + A])) if a.trunk else 0
        da = (2 * (p["W1a"][A] - p["W1a"][Bb]) + 2 * (p["W1a"][V + Bb] - p["W1a"][V + A])) if a.Ha > 0 else 0
        if r > 0:
            vec = (Ph[:, A] - Hk[:, A]) / rho_u[None, :, None]
            Y1 = Y[None] + Binv[:, ii].T[:, None, :, None] * np.transpose(vec, (1, 0, 2))[:, :, None, :]   # (nb,K,N,r)
            w1b = W[:, Bb].T + Binv[:, ii].T * kap[:, None]                                # (nb, N)
            w1b[np.arange(nb), jj] -= 1.0
            Y1j = Y1[np.arange(nb), :, jj, :]                                              # (nb, K, r)
            Y2 = Y1 - w1b[:, None, :, None] * Y1j[:, :, None, :] / rho_d[:, None, None, None]
            tot = termsum(pre[None] + d, Y2, cols)
        else:
            tot = np.full(nb, S)
        fac = np.exp(Jv(prea[None] + da) - J0) if a.Ha > 0 else 1.0
        E += float(np.sum(-0.5 * J * fac * rho_u * rho_d * tot / S))
    return E, t_setup, time.perf_counter() - t0


# ============================================================================ references
def dwave_bcs_F(m, delta0=0.5, s_mix=0.15, noise=0.02, seed=0):
    """Real pairing function F0[i, j] = V^-1 sum_k phi_k cos(k.(r_i - r_j)), phi_k = Delta_k /
    (xi_k + sqrt(xi_k^2 + Delta_k^2)), Delta_k = delta0 (cos kx - cos ky) + s_mix delta0 (a small
    extended-s admixture and noise break the d-wave irrep so that projection onto any 1-D irrep is
    non-zero), mu by bisection so that sum_k v_k^2 = N_up. Lattice PBC (k grid Lx x Ly)."""
    rng = np.random.default_rng(seed)
    Lx, Ly, V = m.Lx, m.Ly, m.V
    kx = 2 * np.pi * np.arange(Lx) / Lx
    ky = 2 * np.pi * np.arange(Ly) / Ly
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    KX, KY = KX.ravel(), KY.ravel()
    eps = -2 * m.t * (np.cos(KX) + np.cos(KY)) - 4 * m.tp * np.cos(KX) * np.cos(KY)
    dk = delta0 * (np.cos(KX) - np.cos(KY)) + s_mix * delta0 * (np.cos(KX) + np.cos(KY))

    def nk(mu):
        xi = eps - mu
        Ek = np.sqrt(xi ** 2 + dk ** 2 + 1e-12)
        return 0.5 * (1 - xi / Ek)
    lo, hi = eps.min() - 10, eps.max() + 10
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        if nk(mu).sum() < m.nup:
            lo = mu
        else:
            hi = mu
    xi = eps - mu
    Ek = np.sqrt(xi ** 2 + dk ** 2 + 1e-12)
    phi = dk / (xi + Ek + 1e-12)
    phi = np.clip(phi, -50, 50)
    R = m.coords
    dx = R[:, None, 0] - R[None, :, 0]
    dy = R[:, None, 1] - R[None, :, 1]
    F = np.zeros((V, V))
    for a in range(len(KX)):
        F += phi[a] * np.cos(KX[a] * dx + KY[a] * dy)
    F /= V
    F = F / np.abs(F).max()
    return F + noise * rng.standard_normal((V, V))
