#!/usr/bin/env python3
"""
P4.1 r23 (R22 `spin-tree`): coupling-tree basis |c, T> for the Hubbard model, hop matrix elements by recoupling.
================================================================================================================

Library + validation script. Run with the lit_env interpreter from this directory:
    ~/.conda/envs/lit_env/bin/python spin_tree.py --mode MODE

Modes (--mode, required)
  validate   run every INVALID check of RULE.md that concerns the basis and write results/validate.json:
               (b) brute force: build every tree state as a Fock vector (explicit Clebsch-Gordan expansion) on small
                   clusters and compare V^T H_Fock V with the recoupling Hamiltonian elementwise (tolerance 1e-12),
                   orthonormality V^T V = 1, and V^T U_g V against the signed permutation of the point group;
                   clusters: 4-site chain (N_e = 2, 3, 4; S = 0 and S = 1 for N_e = 4), 2x2 with t' (N_e = 2, 4;
                   mirrors checked), 2x3 with t' (N_e = 4, 6).
               (c) connected states: for every basis state of the 12-site sector and every allowed hop, the number of
                   triangle-allowed T' produced by the recoupling code equals r21 tree_count.hop_count on the same
                   tree; accidental zeros (allowed T' with a zero coefficient) are counted separately.
               (a) ED: the lowest (+,+) eigenvalue of the 12-site tree-basis H equals E0 to 1e-10 (also the lowest
                   S = 0 eigenvalue of every irrep, for information), and U_g commutes with H to 1e-12.
             Also caches the 12-site sector (basis, H, point-group maps, features) in results/sector12.npz.
  sector     only build and cache results/sector12.npz (no checks).
Options
  --threads INT   BLAS threads (default 1)

System (12 sites): open 4x3 strip, site i = x*3 + y, H = -sum_{i != j} T_ij sum_s c^dag_{j s} c_{i s} + U sum n_up n_dn,
T_ij = 1 (nearest neighbours), t' = -0.25 (diagonals), U = 8, N_e = 10, S = 0 (the vmc.py convention;
E0 = -7.827899034271753 in the (+,+) irrep).

Basis convention
  Tree: nested tuples of site indices (binary). Fermion order = depth-first leaf order; on a doubly occupied site
  up before down. A leaf state is |0>, c^dag_{i s}|0> (spin 1/2, m = s) or c^dag_{i up} c^dag_{i dn}|0> (spin 0).
  A node state is sum <j_a m_a j_b m_b | j m> (A-string)(B-string)|0> with the A string (first child) to the left.
  Spins are stored doubled (2j).
Recoupling (no Fock expansion): reduced matrix elements R of c^dag (convention <j' m'|T_q|j m> = <j m 1/2 q|j' m'> R)
  are carried from each leaf of the hop up to the children of the lowest common ancestor (LCA); each node multiplies
  by a node coefficient X (a 6j-type recoupling coefficient, computed from Clebsch-Gordan sums and cached) and by the
  fermion parity (-1)^{N_a} when the operator acts in the second child; at the LCA the two rank-1/2 tensors are
  coupled to rank 0 (one cached coefficient); every node above the LCA is unchanged and contributes 1.
"""
import argparse
import functools
import itertools
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
E0 = -7.827899034271753
SQ2 = np.sqrt(2.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ Clebsch-Gordan (doubled arguments)
@functools.lru_cache(maxsize=None)
def cg(j1, m1, j2, m2, j, m):
    """<j1/2 m1/2, j2/2 m2/2 | j/2 m/2> (all arguments doubled), float, via sympy (exact) and cached."""
    if m1 + m2 != m or abs(m1) > j1 or abs(m2) > j2 or abs(m) > j:
        return 0.0
    if (j1 + m1) % 2 or (j2 + m2) % 2 or (j + m) % 2:
        return 0.0
    if not (abs(j1 - j2) <= j <= j1 + j2) or (j1 + j2 + j) % 2:
        return 0.0
    from sympy import Rational
    from sympy.physics.wigner import clebsch_gordan
    return float(clebsch_gordan(Rational(j1, 2), Rational(j2, 2), Rational(j, 2),
                                Rational(m1, 2), Rational(m2, 2), Rational(m, 2)))


def mrange(j):
    return range(-j, j + 1, 2)


@functools.lru_cache(maxsize=None)
def X_node(side, ja, jb, j, xa, x):
    """node coefficient for a rank-1/2 tensor T acting in child `side` (0 = first, 1 = second) of a node:
    side 0: <(xa jb) x||T(a)||(ja jb) j> = X * <xa||T||ja>;  side 1: <(ja xa) x||T(b)||(ja jb) j> = X * <xa||T||jb>
    (fermion parity excluded). Computed as a ratio of Clebsch-Gordan sums (Wigner-Eckart)."""
    for m in mrange(j):
        for q in (-1, 1):
            den = cg(j, m, 1, q, x, m + q)
            if abs(den) < 1e-8:
                continue
            num = 0.0
            if side == 0:
                for ma in mrange(ja):
                    mb = m - ma
                    num += cg(ja, ma, jb, mb, j, m) * cg(xa, ma + q, jb, mb, x, m + q) * cg(ja, ma, 1, q, xa, ma + q)
            else:
                for mb in mrange(jb):
                    ma = m - mb
                    num += cg(ja, ma, jb, mb, j, m) * cg(ja, ma, xa, mb + q, x, m + q) * cg(jb, mb, 1, q, xa, mb + q)
            return num / den
    return 0.0


@functools.lru_cache(maxsize=None)
def C_lca(case, ja, jb, J, xa, xb):
    """rank-0 coupling at the LCA, <(xa xb) J M| sum_s O1_s(a) O2_s(b) |(ja jb) J M> per unit reduced elements:
    case 1: O1 = c^dag (a: ja -> xa, R = <xa||c^dag||ja>), O2 = c (b: jb -> xb, via R = <jb||c^dag||xb>);
    case 2: O1 = c (a: ja -> xa, via R = <ja||c^dag||xa>), O2 = c^dag (b: jb -> xb, R = <xb||c^dag||jb>).
    Fermion parity excluded. Independent of M; M = J is used."""
    M = J
    tot = 0.0
    for s in (-1, 1):
        for ma in mrange(ja):
            mb = M - ma
            if abs(mb) > jb:
                continue
            c0 = cg(ja, ma, jb, mb, J, M)
            if c0 == 0.0:
                continue
            if case == 1:
                xma, xmb = ma + s, mb - s
                t = cg(ja, ma, 1, s, xa, xma) * cg(xb, xmb, 1, s, jb, mb)
            else:
                xma, xmb = ma - s, mb + s
                t = cg(xa, xma, 1, s, ja, ma) * cg(jb, mb, 1, s, xb, xmb)
            if t == 0.0:
                continue
            tot += c0 * cg(xa, xma, xb, xmb, J, M) * t
    return tot


def tri(a, b, c):
    return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


# ============================================================================ tree
class Tree:
    """binary coupling tree from nested tuples of site indices; nodes numbered in pre-order, root 0."""

    def __init__(self, nested, L):
        self.L = L
        self.parent, self.children, self.sites = [], [], []
        self.leaf_of = np.full(L, -1)

        def add(t, par):
            v = len(self.parent)
            self.parent.append(par)
            self.children.append(None)
            self.sites.append(None)
            if isinstance(t, (int, np.integer)):
                self.leaf_of[int(t)] = v
                self.sites[v] = [int(t)]
                return v
            assert len(t) == 2, t
            a = add(t[0], v)
            b = add(t[1], v)
            self.children[v] = (a, b)
            self.sites[v] = self.sites[a] + self.sites[b]
            return v

        add(nested, -1)
        assert sorted(self.sites[0]) == list(range(L))
        self.n = len(self.parent)
        self.order = self.sites[0]  # fermion (DFS leaf) order
        self.pos = np.empty(L, dtype=int)
        self.pos[self.order] = np.arange(L)
        self.internal = [v for v in range(self.n) if self.children[v] is not None]
        self.anc = {}
        for s in range(L):
            v, lst = self.leaf_of[s], []
            while v != -1:
                lst.append(v)
                v = self.parent[v]
            self.anc[s] = lst

    def counts(self, c):
        """electron count per node."""
        N = np.zeros(self.n, dtype=int)
        for v in reversed(range(self.n)):
            if self.children[v] is None:
                N[v] = c[self.sites[v][0]]
            else:
                a, b = self.children[v]
                N[v] = N[a] + N[b]
        return N

    def trees(self, c, S2):
        """all valid T (tuple of doubled spins per node) for occupation c and doubled root spin S2."""
        opts = [None] * self.n
        for v in reversed(range(self.n)):
            if self.children[v] is None:
                opts[v] = {int(c[self.sites[v][0]] == 1)}
            else:
                a, b = self.children[v]
                opts[v] = {s for x in opts[a] for y in opts[b] for s in range(abs(x - y), x + y + 1, 2)}
        if S2 not in opts[0]:
            return []
        out = []

        def rec(T, stack):
            if not stack:
                out.append(tuple(T))
                return
            v = stack[0]
            a, b = self.children[v]
            for x in sorted(opts[a]):
                for y in sorted(opts[b]):
                    if tri(x, y, T[v]):
                        T2 = list(T)
                        T2[a], T2[b] = x, y
                        rest = stack[1:] + [u for u in (a, b) if self.children[u] is not None]
                        rec(T2, rest)

        T0 = [0] * self.n
        T0[0] = S2
        rec(T0, [0])
        return out


# ============================================================================ sector and Hamiltonian by recoupling
class Sector:
    def __init__(self, tree, Tmat, U, Ne, S2):
        self.tree, self.T, self.U, self.Ne, self.S2 = tree, Tmat, U, Ne, S2
        L = tree.L
        self.L = L
        cs = [c for c in itertools.product(range(3), repeat=L) if sum(c) == Ne]
        self.states, self.index = [], {}
        for c in cs:
            for T in tree.trees(c, S2):
                self.index[(c, T)] = len(self.states)
                self.states.append((c, T))
        self.dim = len(self.states)
        bi, bj = np.nonzero(Tmat)
        self.hops = [(int(i), int(j), float(Tmat[i, j])) for i, j in zip(bi, bj) if i != j]

    def hop(self, c, T, N, i, j):
        """c^dag_j c_i (spin summed) on |c, T>: list of (T', value, allowed) for every triangle-allowed T'."""
        tr = self.tree
        if c[i] == 0 or c[j] == 2:
            return None, []
        cn = list(c)
        cn[i] -= 1
        cn[j] += 1
        cn = tuple(cn)
        ai = tr.anc[i]
        aj = tr.anc[j]
        sai = set(ai)
        lca = next(x for x in aj if x in sai)
        ka, kb = tr.children[lca]
        ti = ka if ka in sai else kb
        tj = kb if ti == ka else ka

        def path(site, top, newleaf, oldleaf, Rleaf, creation):
            """list of (dict node->new spin, R) along leaf..top. creation: R = <new||c^dag||old>, else <old||c^dag||new>."""
            res = [({tr.leaf_of[site]: newleaf}, Rleaf)]
            v = tr.leaf_of[site]
            while v != top:
                p = tr.parent[v]
                a, b = tr.children[p]
                side = 0 if a == v else 1
                w = b if side == 0 else a
                sw = T[w]
                nxt = []
                for nd, R in res:
                    xv = nd[v]
                    for sp_ in (T[p] - 1, T[p] + 1):
                        if sp_ < 0 or not tri(xv, sw, sp_):
                            continue
                        if creation:   # ket old, bra new
                            kv, bv, kp, bp = T[v], xv, T[p], sp_
                        else:          # ket new, bra old
                            kv, bv, kp, bp = xv, T[v], sp_, T[p]
                        if side == 0:
                            Xc = X_node(0, kv, sw, kp, bv, bp)
                            sgn = 1
                        else:
                            Xc = X_node(1, sw, kv, kp, bv, bp)
                            sgn = -1 if N[w] % 2 else 1
                        d2 = dict(nd)
                        d2[p] = sp_
                        nxt.append((d2, R * Xc * sgn))
                res = nxt
                v = p
            return res

        # annihilation side (site i): <old||c^dag||new>
        Ri = 1.0 if c[i] == 1 else -SQ2
        Pi = path(i, ti, int(cn[i] == 1), int(c[i] == 1), Ri, False)
        Rj = 1.0 if c[j] == 0 else -SQ2
        Pj = path(j, tj, int(cn[j] == 1), int(c[j] == 1), Rj, True)
        J = T[lca]
        out = []
        for di, R1 in Pi:
            for dj, R2 in Pj:
                xi, xj = di[ti], dj[tj]
                if not tri(xi, xj, J):
                    continue
                if ti == ka:   # i in first child: case 2, parity (-1)^{N_a - 1}
                    val = C_lca(2, T[ka], T[kb], J, xi, xj) * R1 * R2 * (-1 if (N[ka] - 1) % 2 else 1)
                else:          # j in first child: case 1, parity (-1)^{N_a}
                    val = C_lca(1, T[ka], T[kb], J, xj, xi) * R1 * R2 * (-1 if N[ka] % 2 else 1)
                Tn = list(T)
                for d in (di, dj):
                    for k, s in d.items():
                        Tn[k] = s
                out.append((tuple(Tn), val))
        return cn, out

    def hamiltonian(self, stats=False):
        rows, cols, vals = [], [], []
        nallowed, nzero, per_state = 0, 0, []
        for k, (c, T) in enumerate(self.states):
            N = self.tree.counts(c)
            nd = sum(1 for x in c if x == 2)
            if nd:
                rows.append(k)
                cols.append(k)
                vals.append(self.U * nd)
            cnt = []
            for i, j, t in self.hops:
                cn, lst = self.hop(c, T, N, i, j)
                if cn is None:
                    continue
                cnt.append(len(lst))
                for Tn, v in lst:
                    nallowed += 1
                    if abs(v) < 1e-13:
                        nzero += 1
                        continue
                    rows.append(self.index[(cn, Tn)])
                    cols.append(k)
                    vals.append(-t * v)
            per_state.append(cnt)
        H = sp.csr_matrix((vals, (rows, cols)), shape=(self.dim, self.dim))
        H.sum_duplicates()
        if stats:
            return H, dict(allowed=nallowed, accidental_zeros=nzero), per_state
        return H

    # -------------------------------------------------------------- point group (site permutation = tree automorphism)
    def pg_map(self, perm):
        """U_g |c,T> = sgn |c', T'>, U_g c^dag_{i s} U_g^-1 = c^dag_{perm[i] s}. Returns (idx, sgn) arrays."""
        tr = self.tree
        # node map: v -> node whose site set is perm(sites[v])
        key = {frozenset(tr.sites[v]): v for v in range(tr.n)}
        nmap = np.array([key[frozenset(int(perm[s]) for s in tr.sites[v])] for v in range(tr.n)])
        swapped = []
        for v in tr.internal:
            a, b = tr.children[v]
            a2, b2 = tr.children[nmap[v]]
            assert {nmap[a], nmap[b]} == {a2, b2}, "not a tree automorphism"
            if nmap[a] == b2:
                swapped.append(v)
        img_pos = tr.pos[np.asarray(perm)[tr.order]]  # position of the image of the k-th leaf (in fermion order)
        idx = np.empty(self.dim, dtype=np.int64)
        sgn = np.empty(self.dim)
        for k, (c, T) in enumerate(self.states):
            cn = [0] * self.L
            for s in range(self.L):
                cn[perm[s]] = c[s]
            Tn = [0] * tr.n
            for v in range(tr.n):
                Tn[nmap[v]] = T[v]
            # fermion sign: reorder the image string (leaves in old DFS order, at new positions) to new DFS order
            occ = [c[s] for s in tr.order]
            f = 0
            for p in range(self.L):
                for q in range(p + 1, self.L):
                    if img_pos[p] > img_pos[q]:
                        f += occ[p] * occ[q]
            ph = 0
            for v in swapped:
                a, b = tr.children[v]
                ph += (T[a] + T[b] - T[v]) // 2
            idx[k] = self.index[(tuple(cn), tuple(Tn))]
            sgn[k] = (-1.0) ** (f + ph)
        return idx, sgn


# ============================================================================ brute force (Fock) for validation
def fock_vec(tree, c, T, m_root):
    """dict bitmask -> amplitude; modes 2*pos + (0 up, 1 dn), canonical ascending creation order."""
    def rec(v, m):
        if tree.children[v] is None:
            s = tree.sites[v][0]
            p = tree.pos[s]
            if c[s] == 0:
                return {0: 1.0} if m == 0 else {}
            if c[s] == 2:
                return {(1 << (2 * p)) | (1 << (2 * p + 1)): 1.0} if m == 0 else {}
            if m == 1:
                return {1 << (2 * p): 1.0}
            if m == -1:
                return {1 << (2 * p + 1): 1.0}
            return {}
        a, b = tree.children[v]
        out = {}
        for ma in mrange(T[a]):
            mb = m - ma
            if abs(mb) > T[b]:
                continue
            w = cg(T[a], ma, T[b], mb, T[v], m)
            if w == 0.0:
                continue
            A, B = rec(a, ma), rec(b, mb)
            for ka, va in A.items():
                for kb, vb in B.items():
                    out[ka | kb] = out.get(ka | kb, 0.0) + w * va * vb
        return out
    return rec(0, m_root)


def fock_ops(tree, Tmat, U, keys):
    """H in the Fock basis listed by `keys` (bitmasks), modes 2*pos(site) + s."""
    L = tree.L
    ix = {k: n for n, k in enumerate(keys)}
    H = np.zeros((len(keys), len(keys)))

    def ann(k, m):
        if not (k >> m) & 1:
            return None, 0
        return k ^ (1 << m), (-1) ** bin(k & ((1 << m) - 1)).count("1")

    def cre(k, m):
        if (k >> m) & 1:
            return None, 0
        return k | (1 << m), (-1) ** bin(k & ((1 << m) - 1)).count("1")

    for k, n in ix.items():
        for s in range(L):
            p = tree.pos[s]
            if (k >> (2 * p)) & 1 and (k >> (2 * p + 1)) & 1:
                H[n, n] += U
        for i in range(L):
            for j in range(L):
                if i == j or Tmat[i, j] == 0:
                    continue
                for sg in (0, 1):
                    k1, s1 = ann(k, 2 * tree.pos[i] + sg)
                    if k1 is None:
                        continue
                    k2, s2 = cre(k1, 2 * tree.pos[j] + sg)
                    if k2 is None:
                        continue
                    H[ix[k2], n] += -Tmat[i, j] * s1 * s2
    return H


def fock_perm(tree, perm, keys):
    """U_g in the Fock basis: c^dag_{i s} -> c^dag_{perm[i] s}."""
    L = tree.L
    ix = {k: n for n, k in enumerate(keys)}
    Ug = np.zeros((len(keys), len(keys)))
    for k, n in ix.items():
        modes = [m for m in range(2 * L) if (k >> m) & 1]  # ascending = canonical creation order
        img = []
        for m in modes:
            p, s = divmod(m, 2)
            site = tree.order[p]
            img.append(2 * tree.pos[perm[site]] + s)
        # sign of sorting img
        sgn = 1
        arr = list(img)
        for a in range(len(arr)):
            for b in range(a + 1, len(arr)):
                if arr[a] > arr[b]:
                    sgn = -sgn
        k2 = sum(1 << m for m in img)
        Ug[ix[k2], n] = sgn
    return Ug


# ============================================================================ geometry
def strip_T(Lx, Ly, tp):
    """identical to vmc.strip_T: site x*Ly + y, NN 1, diagonal tp, open."""
    L = Lx * Ly
    T = np.zeros((L, L))
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            for dx, dy, a in [(1, 0, 1.0), (0, 1, 1.0), (1, 1, tp), (1, -1, tp)]:
                X, Y = x + dx, y + dy
                if 0 <= X < Lx and 0 <= Y < Ly:
                    j = X * Ly + Y
                    T[i, j] += a
                    T[j, i] += a
    return T


def mirror_perms(Lx, Ly):
    """[(E), Mx, My, MxMy] as site permutations, same order and characters as vmc.Model (chi = (+,+))."""
    out = []
    for c, d in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        out.append(np.array([(Lx - 1 - x if c else x) * Ly + (Ly - 1 - y if d else y)
                             for x in range(Lx) for y in range(Ly)]))
    return out


def tree12():
    s = lambda x, y: x * 3 + y  # noqa: E731
    left = ((s(0, 1), s(1, 1)), ((s(0, 0), s(1, 0)), (s(0, 2), s(1, 2))))
    right = ((s(3, 1), s(2, 1)), ((s(3, 0), s(2, 0)), (s(3, 2), s(2, 2))))
    return Tree((left, right), 12)


# ============================================================================ validation
def brute_cluster(name, tree, Tmat, U, Ne, S2, perms=None):
    sec = Sector(tree, Tmat, U, Ne, S2)
    Hr = sec.hamiltonian().toarray()
    vecs = [fock_vec(tree, c, T, S2) for c, T in sec.states]  # m = S (highest weight)
    keys = sorted(set(k for v in vecs for k in v))
    V = np.zeros((len(keys), sec.dim))
    kx = {k: n for n, k in enumerate(keys)}
    for col, v in enumerate(vecs):
        for k, a in v.items():
            V[kx[k], col] = a
    # close the Fock key set under H (so V^T H V is exact): all states with the same N_e and S^z
    L = tree.L
    Sz2 = S2
    allk = []
    for k in range(1 << (2 * L)):
        if bin(k).count("1") != Ne:
            continue
        up = sum((k >> (2 * p)) & 1 for p in range(L))
        if 2 * up - Ne == Sz2:
            allk.append(k)
    ax = {k: n for n, k in enumerate(allk)}
    VV = np.zeros((len(allk), sec.dim))
    for k, n in kx.items():
        VV[ax[k]] = V[n]
    Hf = fock_ops(tree, Tmat, U, allk)
    Hb = VV.T @ Hf @ VV
    out = dict(cluster=name, Ne=Ne, S=S2 / 2, dim=sec.dim, fock_dim=len(allk),
               orth=float(np.abs(VV.T @ VV - np.eye(sec.dim)).max()),
               H_maxdiff=float(np.abs(Hb - Hr).max()), H_norm=float(np.abs(Hr).max()))
    if perms is not None:
        pd = []
        for g, pm in enumerate(perms):
            idx, sgn = sec.pg_map(pm)
            Pm = np.zeros((sec.dim, sec.dim))
            Pm[idx, np.arange(sec.dim)] = sgn
            Ub = VV.T @ fock_perm(tree, pm, allk) @ VV
            pd.append(float(np.abs(Ub - Pm).max()))
        out["pg_maxdiff"] = max(pd)
    return out


def build12(threads=1):
    L, tree = 12, tree12()
    Tm = strip_T(4, 3, -0.25)
    t0 = time.time()
    sec = Sector(tree, Tm, 8.0, 10, 0)
    log(f"12-site sector: dim {sec.dim} ({time.time() - t0:.0f} s)")
    H, st, per_state = sec.hamiltonian(stats=True)
    log(f"H built: nnz {H.nnz}, allowed T' {st['allowed']}, accidental zeros {st['accidental_zeros']} "
        f"({time.time() - t0:.0f} s)")
    maps = [sec.pg_map(pm) for pm in mirror_perms(4, 3)]
    return sec, H, st, per_state, maps


def save12(sec, H, maps, fn):
    tr = sec.tree
    C = np.array([c for c, _ in sec.states], dtype=np.int8)
    Tt = np.array([T for _, T in sec.states], dtype=np.int8)
    np.savez_compressed(fn, C=C, T=Tt, order=np.array(tr.order), internal=np.array(tr.internal),
                        node_size=np.array([len(s) for s in tr.sites]),
                        Hdata=H.data, Hind=H.indices, Hptr=H.indptr, dim=sec.dim,
                        pg_idx=np.stack([m[0] for m in maps]), pg_sgn=np.stack([m[1] for m in maps]))


def validate(a):
    c0 = time.process_time()
    res = dict(brute=[])
    # (b) brute force on small clusters
    chain = Tree(((0, 1), (2, 3)), 4)
    Tc = np.zeros((4, 4))
    for i in range(3):
        Tc[i, i + 1] = Tc[i + 1, i] = 1.0
    Tc[0, 2] = Tc[2, 0] = Tc[1, 3] = Tc[3, 1] = -0.3  # next-nearest hops cross the root split and stay inside pairs
    for Ne, S2 in [(2, 0), (3, 1), (4, 0), (4, 2), (5, 1), (6, 0)]:
        res["brute"].append(brute_cluster("chain4+nnn", chain, Tc, 3.0, Ne, S2))
    sq = Tree(((0, 2), (1, 3)), 4)  # 2x2, site x*2 + y; pairs = rows y = 0, 1
    T22 = strip_T(2, 2, -0.25)
    for Ne, S2 in [(2, 0), (3, 1), (4, 0), (4, 2)]:
        res["brute"].append(brute_cluster("2x2", sq, T22, 8.0, Ne, S2, perms=mirror_perms(2, 2)))
    s = lambda x, y: x * 3 + y  # noqa: E731
    t23 = Tree(((s(0, 1), s(1, 1)), ((s(0, 0), s(1, 0)), (s(0, 2), s(1, 2)))), 6)  # one half of the 12-site tree
    T23 = strip_T(2, 3, -0.25)
    for Ne, S2 in [(4, 0), (5, 1), (6, 0)]:
        res["brute"].append(brute_cluster("2x3(half-tree)", t23, T23, 8.0, Ne, S2, perms=None))
    # 4x2 strip with the 12-site tree's mirror structure, to test both mirrors on an Lx = 4 tree with fermion signs
    s2 = lambda x, y: x * 2 + y  # noqa: E731
    t42 = Tree((((s2(0, 0), s2(1, 0)), (s2(0, 1), s2(1, 1))), ((s2(3, 0), s2(2, 0)), (s2(3, 1), s2(2, 1)))), 8)
    T42 = strip_T(4, 2, -0.25)
    for Ne, S2 in [(4, 0), (6, 0)]:
        res["brute"].append(brute_cluster("4x2(mirror tree)", t42, T42, 8.0, Ne, S2, perms=mirror_perms(4, 2)))
    for r in res["brute"]:
        log(f"brute {r['cluster']:18s} Ne {r['Ne']} S {r['S']}: dim {r['dim']:4d} (Fock {r['fock_dim']:5d}); "
            f"orth {r['orth']:.1e}; H maxdiff {r['H_maxdiff']:.1e} (|H|max {r['H_norm']:.2f})"
            + (f"; U_g maxdiff {r['pg_maxdiff']:.1e}" if "pg_maxdiff" in r else ""))
    res["brute_ok"] = all(r["orth"] < 1e-12 and r["H_maxdiff"] < 1e-12 and r.get("pg_maxdiff", 0) < 1e-12
                          for r in res["brute"])
    log(f"(b) brute force OK: {res['brute_ok']} [{(time.process_time() - c0) / 60:.2f} CPU-min]")

    # 12-site sector
    sec, H, st, per_state, maps = build12()
    res["dim12"] = sec.dim
    res["allowed_Tprime"], res["accidental_zeros"] = st["allowed"], st["accidental_zeros"]
    sym = float(abs(H - H.T).max())
    res["H_asym"] = sym
    # (c) connected counts vs r21 tree_count.hop_count on the same tree
    sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r21"))
    import tree_count as TC
    tr = sec.tree
    par = np.array(tr.parent)
    ch = tr.children
    anc = {tr.leaf_of[s]: tr.anc[s] for s in range(12)}
    mism, tot = 0, 0
    for k, (c, T) in enumerate(sec.states):
        occ = np.array(c)
        Ta = np.array(T)
        mine = iter(per_state[k])
        for i, j, _ in sec.hops:
            r = TC.hop_count(i, j, occ, tr.leaf_of, par, ch, Ta, anc)
            if r is None:
                continue
            tot += 1
            if r != next(mine):
                mism += 1
    res["count_checked_hops"], res["count_mismatch"] = tot, mism
    log(f"(c) connected T' counts vs r21 tree_count.hop_count: {tot} (state, hop) pairs, {mism} mismatches; "
        f"allowed T' {st['allowed']}, accidental zeros {st['accidental_zeros']}")
    # (a) ED; point group
    comm = 0.0
    for idx, sgn in maps:
        Pm = sp.csr_matrix((sgn, (idx, np.arange(sec.dim))), shape=(sec.dim, sec.dim))
        comm = max(comm, float(abs(Pm @ H - H @ Pm).max()))
    res["pg_comm"] = comm
    from scipy.sparse.linalg import eigsh
    irreps = {}
    for chi in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        chars = [1, chi[0], chi[1], chi[0] * chi[1]]
        # orthonormal basis of the image of P_G: column r = P_G e_r for each orbit representative r (smallest index
        # in its orbit), P_G e_r = (1/4) sum_g chi_g sgn[g, r] e_{idx[g, r]}; zero columns (sign-frustrated orbits)
        # are dropped
        idx_all = np.stack([m[0] for m in maps])
        sgn_all = np.stack([m[1] for m in maps])
        reps = np.nonzero(idx_all.min(0) == np.arange(sec.dim))[0]
        rr = np.concatenate([idx_all[g, reps] for g in range(4)])
        cc = np.concatenate([np.arange(len(reps))] * 4)
        vv = np.concatenate([0.25 * chars[g] * sgn_all[g, reps] for g in range(4)])
        Bm = sp.csc_matrix((vv, (rr, cc)), shape=(sec.dim, len(reps)))
        Bm.sum_duplicates()
        Bm.eliminate_zeros()
        nv = np.sqrt(np.asarray(Bm.multiply(Bm).sum(0)).ravel())
        keepc = nv > 1e-12
        B = (Bm[:, keepc] @ sp.diags(1 / nv[keepc])).tocsr()
        n = int(keepc.sum())
        Hs = (B.T @ H @ B).tocsr()
        w = eigsh(Hs, k=1, which="SA", tol=1e-14)[0][0]
        irreps[f"{chi}"] = dict(dim=n, E=float(w))
        log(f"irrep {chi}: dim {n}, lowest S = 0 energy {w:.12f}")
    res["irreps"] = irreps
    res["E_pp"] = irreps["(1, 1)"]["E"]
    res["E_pp_minus_E0"] = abs(res["E_pp"] - E0)
    res["ok"] = (res["brute_ok"] and mism == 0 and comm < 1e-12 and res["E_pp_minus_E0"] < 1e-10 and sym < 1e-12)
    log(f"(a) tree-basis ED (+,+): {res['E_pp']:.13f}  E0 {E0:.13f}  |diff| {res['E_pp_minus_E0']:.1e}; "
        f"[H, U_g] max {comm:.1e}; H asym {sym:.1e}")
    log(f"ALL BASIS CHECKS PASS: {res['ok']}")
    res["cpu_min"] = (time.process_time() - c0) / 60
    json.dump(res, open(os.path.join(HERE, "results/validate.json"), "w"), indent=1)
    save12(sec, H, maps, os.path.join(HERE, "results/sector12.npz"))
    log(f"wrote results/validate.json and results/sector12.npz [{res['cpu_min']:.2f} CPU-min]")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["validate", "sector"])
    ap.add_argument("--threads", default="1")
    return ap.parse_args()


if __name__ == "__main__":
    A = parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = A.threads
    if A.mode == "validate":
        validate(A)
    else:
        sec, H, st, _, maps = build12()
        save12(sec, H, maps, os.path.join(HERE, "results/sector12.npz"))
