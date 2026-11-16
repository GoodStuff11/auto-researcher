#!/usr/bin/env python3
"""
P1.2 decisive test: can a LOCAL, LOW-DEGREE sign layer on top of the Minimal
Contrast Foil's determinant repair the foil's nodal (sign) errors in the doped
t-t'-J model, without backflow and without a sum over determinants?

Route under test ("residual Walsh sign layer", artifact Route 4):
    psi(n) = s_foil(n) * c(n) * A(n),   c(n) = sgn( b + sum_S w_S chi_S(n) ),
where s_foil is the sign of the best foil reference (Gutzwiller-projected d-wave
BCS or Gutzwiller-projected antiferromagnetic Slater determinant), chi_S(n) =
prod_{v in S} (1 - 2 x_v) are Walsh characters of the binary occupation
variables x_v in {n_{i,up}, n_{i,dn}}, and S runs over subsets of size 1..k of
the 8 binary variables of one 2x2 plaquette (all plaquettes, weights not
shared).  A(n) > 0 is an amplitude that is NOT learned here: the test measures
sign quality through the fixed-node energy, which optimises amplitudes for a
given sign.

What it computes per instance
-----------------------------
  1. Exact ground state psi_0 by sparse Lanczos in the t-t'-J Hilbert space
     (no double occupancy, N_up = N_dn), site-ordered Fock sign convention
     |n> = prod_{i ascending, occupied} c^dag_{i,sigma_i}|0>.  If the ground
     state is degenerate, psi_0 is the normalised projection of the foil's
     fixed-node state onto the degenerate manifold (this favours the foil).
  2. Foil sign: Nelder-Mead over 3 parameters of each family
       slater_af (t'_v, m, a): projected Slater of mean-field H with hopping t
                 (x-bonds scaled by 1+a), t'_v and staggered field m;
       bcs_d (Delta, mu, t'_v): projected d-wave BCS at fixed N, PBC only;
     objective = fixed-node energy with --fn-opt-iters guiding iterations; the
     family with the lower final fixed-node energy is the foil.
  3. Fixed-node energy E_FN[s] for a sign vector s: lowest eigenvalue of the
     ten Haaf effective Hamiltonian (PRB 51, 13039 (1995)) built from
     H' = S H S, with guiding amplitudes iterated self-consistently
     (--fn-iters iterations, starting from g = 1, floor 1e-3 max g).  Every
     iterate is a rigorous upper bound on the lowest energy of any state with
     sign s, hence on E_0, and is non-increasing in the iteration count.
  4. Residual sign r(n) = sgn psi_0(n) * s_foil(n).  The layer c(n) is fitted by
     L2-regularised logistic regression on --ntrain configurations drawn from
     |psi_0|^2 (oracle labels: this tests representability and sample
     efficiency, not VMC trainability).  Two feature sets of identical size:
       local : plaquette-local Walsh characters of degree 1..--degree;
       null  : the same number of characters with the same degrees, but each
               built from binary variables drawn uniformly over the whole
               lattice (non-local control of equal capacity).
  5. Diagnostics: weighted sign disagreement W(s) = sum_n |psi_0|^2 [s != sgn psi_0]
     (global sign chosen to minimise it), and for the fixed-node state
     psi = s * a_FN the max error over r of the staggered spin correlation
     C_s(r) and the hole-hole correlation g_hh(r) versus exact.

Scalar metric per instance
--------------------------
  M_local = (E_FN[s_foil * c_local] - E_0) / (E_FN[s_foil] - E_0)
  M_null  = (E_FN[s_foil * c_null ] - E_0) / (E_FN[s_foil] - E_0)
Verdict over all instances:
  PASS      : on EVERY instance M_local <= 0.5, M_null - M_local >= 0.1, and the
              corrected state's C_s and g_hh errors are <= the foil's.
  FAIL      : on ANY instance M_local >= 0.9, or M_local >= M_null.
  AMBIGUOUS : otherwise.  Pre-committed action: rerun once with --degree 4;
              if that run is not PASS, the route is recorded as FAIL.
The script prints "VERDICT: PASS|FAIL|AMBIGUOUS".

Command-line options
--------------------
  --mode {decisive,smoke,holon_check,pairwise_check}     default: decisive
      decisive       : run every instance in --instances and print the verdict.
      smoke          : login-node-safe checks (a few seconds):
                       (a) 2-site, 1 electron: E_0 = -t exactly;
                       (b) 2x2 open (4-site ring) Heisenberg: E_0 = -3J and the
                           Marshall sign gives E_FN = E_0 exactly;
                       (c) fully polarised 2x3 and 3x4 open, t'=-0.3, 1 and 2
                           holes: E_0 equals the free-fermion sum, and the
                           holon-determinant sign and the Slater sign both have
                           weighted disagreement W < 1e-10 with psi_0;
                       (d) fixed-node bound sanity: with the exact sign E_FN
                           - E_0 decreases with iterations and stays >= 0;
                       (e) a reduced decisive run on 3x4 open, 2 holes, t'=-0.3.
      holon_check    : evidence for killing Route 3 (slave-fermion anchor):
                       on 3x4 open, 2 holes, t'=0, for J in {100,10,2,0.4},
                       prints min over ALL hole-only functions f of
                       W(M * f), M = Marshall sign.  Small values mean a holon
                       antisymmetric layer times Marshall can carry the sign.
      pairwise_check : evidence for killing Route 2 (pairwise antisymmetric
                       product): tests whether the free-fermion ground-state
                       sign on 3x4 open, N=4, t'=-0.3 is a quadratic GF(2)
                       polynomial of the occupations.
  --instances STR    default "4x4p:2:-0.3,4x4p:2:0.3,3x4o:2:0.0"
                     comma-separated LxXLy{p|o}:N_holes:tprime ("p" periodic,
                     "o" open).  N_up = N_dn = (V - N_holes)/2, so V - N_holes
                     must be even.  bcs_d is used only for periodic lattices.
  --J FLOAT          default 0.4   nearest-neighbour exchange J/t.
  --degree INT       default 3     max Walsh degree k of the sign layer (1..4).
  --ntrain INT       default 4000  training configurations sampled from |psi_0|^2.
  --l2 FLOAT         default 1e-3  base L2 penalty; the fit tries {l2/10, l2, 10*l2,
                                   100*l2} and the constant layer c=+1, and keeps
                                   whichever has the lowest validation sign error
                                   (validation sample: ntrain/2 draws from |psi_0|^2).
  --maxfev INT       default 80    Nelder-Mead evaluations per family per start.
  --starts INT       default 2     Nelder-Mead starts per family (first is a
                                   physical guess, the rest seeded perturbations).
  --fn-opt-iters INT default 2     guiding iterations of E_FN inside the foil
                                   optimisation.
  --fn-iters INT     default 10    guiding iterations of every reported E_FN.
  --seed INT         default 0     RNG seed (starts, training samples, null set).
  --out PATH         default ""    if set, write a JSON summary to PATH.

Units t = 1.  4x4 with 2 holes has 411,840 states (Sz = 0).
"""
import argparse
import itertools
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize

_POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)


def popcount(x):
    x = np.asarray(x, dtype=np.int64)
    return _POP16[x & 0xFFFF] + _POP16[(x >> 16) & 0xFFFF] + _POP16[(x >> 32) & 0xFFFF]


# ----------------------------------------------------------------------------- lattice
class Lattice:
    def __init__(self, Lx, Ly, pbc):
        self.Lx, self.Ly, self.pbc, self.V = Lx, Ly, pbc, Lx * Ly
        self.xy = np.array([(i % Lx, i // Lx) for i in range(self.V)])
        self.sub = (self.xy[:, 0] + self.xy[:, 1]) % 2  # sublattice A = 0

    def idx(self, x, y):
        if self.pbc:
            return (x % self.Lx) + (y % self.Ly) * self.Lx
        if 0 <= x < self.Lx and 0 <= y < self.Ly:
            return x + y * self.Lx
        return None

    def bonds(self, shifts):
        out = set()
        for i in range(self.V):
            x, y = self.xy[i]
            for dx, dy in shifts:
                j = self.idx(x + dx, y + dy)
                if j is not None and j != i:
                    out.add((min(i, j), max(i, j), 1 if dy == 0 else 0))
        return sorted(out)

    def nn(self):
        return self.bonds([(1, 0), (0, 1)])

    def nnn(self):
        return [(i, j, 0) for i, j, _ in self.bonds([(1, 1), (1, -1)])]

    def n3(self):
        return [(i, j, 0) for i, j, _ in self.bonds([(2, 0), (0, 2)])]

    def disp(self, i, j):
        d = self.xy[j] - self.xy[i]
        if self.pbc:
            d = np.array([d[0] % self.Lx, d[1] % self.Ly])
        return int(d[0]), int(d[1])


def _lowest(A, v0=None):
    try:
        return spla.eigsh(A, k=1, which="SA", tol=1e-9, v0=v0, maxiter=50000)
    except spla.ArpackNoConvergence as e:
        if len(e.eigenvalues):
            return e.eigenvalues, e.eigenvectors
        return spla.eigsh(A, k=2, which="SA", tol=1e-8, maxiter=100000)


# ----------------------------------------------------------------------------- basis / H
class TJ:
    def __init__(self, lat, Nup, Ndn, tp, J):
        self.lat, self.Nup, self.Ndn, self.tp, self.J = lat, Nup, Ndn, tp, J
        V = lat.V
        ups, dns = [], []
        for u in itertools.combinations(range(V), Nup):
            um = sum(1 << i for i in u)
            rest = [i for i in range(V) if not (um >> i) & 1]
            for d in itertools.combinations(rest, Ndn):
                ups.append(um)
                dns.append(sum(1 << i for i in d))
        up, dn = np.array(ups, np.int64), np.array(dns, np.int64)
        key = (up << V) | dn
        o = np.argsort(key)
        self.up, self.dn, self.key = up[o], dn[o], key[o]
        self.D = len(self.key)
        self.occ = self.up | self.dn
        bits = ((self.up[:, None] >> np.arange(V)) & 1, (self.dn[:, None] >> np.arange(V)) & 1)
        self.nup, self.ndn = bits[0].astype(np.int8), bits[1].astype(np.int8)
        self.hole = (1 - self.nup - self.ndn).astype(np.int8)
        self.up_pos = self._pos(self.nup, Nup)
        self.dn_pos = self._pos(self.ndn, Ndn)
        self.h_pos = self._pos(self.hole, V - Nup - Ndn)
        self.fn_iters, self.fn_opt_iters = 10, 2
        self._build()

    @staticmethod
    def _pos(bits, n):
        if n == 0:
            return np.zeros((bits.shape[0], 0), np.int64)
        return np.nonzero(bits)[1].reshape(bits.shape[0], n)

    def _lookup(self, up, dn):
        k = (up << self.lat.V) | dn
        p = np.searchsorted(self.key, k)
        assert np.all(self.key[p] == k)
        return p

    def _build(self):
        lat, V = self.lat, self.lat.V
        rows, cols, vals = [], [], []
        diag = np.zeros(self.D)
        hops = [(i, j, 1.0) for i, j, _ in lat.nn()] + [(i, j, self.tp) for i, j, _ in lat.nnn()]
        for i, j, t in hops:
            if t == 0.0:
                continue
            between = sum(1 << k for k in range(i + 1, j))
            for a, b in ((i, j), (j, i)):  # electron hops b -> a
                for spin in (0, 1):
                    src = self.up if spin == 0 else self.dn
                    m = (((src >> b) & 1) == 1) & (((self.occ >> a) & 1) == 0)
                    idx = np.nonzero(m)[0]
                    s_new = (src[idx] & ~(1 << b)) | (1 << a)
                    if spin == 0:
                        nidx = self._lookup(s_new, self.dn[idx])
                    else:
                        nidx = self._lookup(self.up[idx], s_new)
                    sgn = 1 - 2 * (popcount(self.occ[idx] & between) % 2)
                    rows.append(nidx); cols.append(idx); vals.append(-t * sgn)
        for i, j, _ in lat.nn():
            ui, uj = (self.up >> i) & 1, (self.up >> j) & 1
            di, dj = (self.dn >> i) & 1, (self.dn >> j) & 1
            anti = (ui & dj) | (di & uj)
            diag -= 0.5 * self.J * anti
            idx = np.nonzero(anti)[0]
            fl = (1 << i) | (1 << j)
            nidx = self._lookup(self.up[idx] ^ fl, self.dn[idx] ^ fl)
            rows.append(nidx); cols.append(idx); vals.append(np.full(len(idx), 0.5 * self.J))
        r = np.concatenate(rows); c = np.concatenate(cols); v = np.concatenate(vals)
        self.offr, self.offc, self.offv = r, c, v
        self.diag = diag
        self.H = (sp.csr_matrix((v, (r, c)), shape=(self.D, self.D)) + sp.diags(diag)).tocsr()

    # --- exact
    def ground(self, k=4):
        k = min(k, self.D - 1)
        w, U = spla.eigsh(self.H, k=k, which="SA", tol=1e-11)
        o = np.argsort(w)
        w, U = w[o], U[:, o]
        deg = int(np.sum(w < w[0] + 1e-7))
        return w[0], U[:, :deg], deg

    # --- fixed node
    def fixed_node(self, s, iters=None, floor=1e-3):
        """Self-consistent ten Haaf fixed-node energy for the sign vector s.

        iteration 0 uses guiding amplitudes g = 1 (sign-only bound); each further
        iteration uses g = Perron vector of the previous step.  Every iterate is a
        rigorous upper bound on min_{a>=0} a^T (S H S) a >= E_0 and the sequence is
        non-increasing; with the exact sign it converges to E_0.
        """
        iters = self.fn_iters if iters is None else iters
        hv = s[self.offr] * s[self.offc] * self.offv
        bad = hv > 0
        keep = ~bad
        Hk = sp.csr_matrix((hv[keep], (self.offr[keep], self.offc[keep])), shape=(self.D, self.D))
        rb, cb, vb = self.offr[bad], self.offc[bad], hv[bad]
        g = np.ones(self.D)
        v0 = None
        for _ in range(iters + 1):
            gg = np.maximum(g, floor * g.max())
            d = self.diag + np.bincount(rb, weights=vb * gg[cb] / gg[rb], minlength=self.D)
            w, U = _lowest((Hk + sp.diags(d)).tocsr(), v0)
            g = np.abs(U[:, 0])
            v0 = g
        return w[0], s * g

    # --- diagonal correlations
    def correlations(self, prob):
        lat = self.lat
        sz = 0.5 * (self.nup - self.ndn).astype(float)
        h = self.hole.astype(float)
        Cs, G, cnt = {}, {}, {}
        for i in range(lat.V):
            for j in range(lat.V):
                if i == j:
                    continue
                d = lat.disp(i, j)
                sign = 1 - 2 * ((lat.sub[i] + lat.sub[j]) % 2)
                Cs[d] = Cs.get(d, 0.0) + sign * np.dot(prob, sz[:, i] * sz[:, j])
                G[d] = G.get(d, 0.0) + np.dot(prob, h[:, i] * h[:, j])
                cnt[d] = cnt.get(d, 0) + 1
        keys = sorted(cnt)
        return np.array([Cs[k] / cnt[k] for k in keys]), np.array([G[k] / cnt[k] for k in keys])


# ----------------------------------------------------------------------------- sign families
def batched_sign_det(Mats):
    if Mats.shape[-1] == 0:
        return np.ones(Mats.shape[0])
    s, _ = np.linalg.slogdet(Mats)
    s = np.where(s == 0, 1.0, s)
    return s


def block_to_site_sign(m):
    """pi(n): <n_site| (ups block, then downs block) state> parity."""
    V = m.lat.V
    cnt = np.zeros(m.D, np.int64)
    for i in range(V):
        below = (1 << i) - 1
        cnt += ((m.up >> i) & 1) * popcount(m.dn & below)
    return 1.0 - 2.0 * (cnt % 2)


def hopping_matrix(lat, t1, tp, t3, a, sign1=-1.0):
    V = lat.V
    T = np.zeros((V, V))
    for i, j, isx in lat.nn():
        T[i, j] += sign1 * t1 * (1 + a * isx)
    for i, j, _ in lat.nnn():
        T[i, j] += -tp
    for i, j, _ in lat.n3():
        T[i, j] += -t3
    return T + T.T


def lowest_orbitals(T, n):
    w, U = np.linalg.eigh(T)
    return U[:, :n]


def sign_slater_af(m, p, pi):
    tpv, mag, a = p
    T = hopping_matrix(m.lat, 1.0, tpv, 0.0, a)
    stag = 1 - 2 * m.lat.sub
    Pu = lowest_orbitals(T + np.diag(mag * stag), m.Nup)
    Pd = lowest_orbitals(T - np.diag(mag * stag), m.Ndn)
    s = batched_sign_det(Pu[m.up_pos]) * batched_sign_det(Pd[m.dn_pos])
    return s * pi


def sign_bcs_d(m, p, pi):
    Delta, mu, tpv = p
    lat = m.lat
    kx = 2 * np.pi * np.arange(lat.Lx) / lat.Lx
    ky = 2 * np.pi * np.arange(lat.Ly) / lat.Ly
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    xi = -2 * (np.cos(KX) + np.cos(KY)) - 4 * tpv * np.cos(KX) * np.cos(KY) - mu
    Dk = Delta * (np.cos(KX) - np.cos(KY)) + 1e-3
    phi = Dk / (xi + np.sqrt(xi ** 2 + Dk ** 2))
    V = lat.V
    F = np.zeros((V, V))
    for i in range(V):
        for j in range(V):
            dx, dy = lat.xy[i] - lat.xy[j]
            F[i, j] = np.real(np.sum(phi * np.exp(1j * (KX * dx + KY * dy)))) / V
    Ms = F[m.up_pos[:, :, None], m.dn_pos[:, None, :]]
    return batched_sign_det(Ms) * pi


def sign_holon_anchor(m, p, sh):
    tph, t3h, a = p
    lat = m.lat
    Nh = m.h_pos.shape[1]
    # holon hopping matrix: in the kappa gauge holons hop with +t (sh=+1) on nn bonds
    T = hopping_matrix(lat, 1.0, -tph, -t3h, a, sign1=sh)
    X = lowest_orbitals(T, Nh)
    kappa = 1.0 - 2.0 * (m.h_pos.sum(axis=1) % 2) if Nh else np.ones(m.D)
    marshall = 1.0 - 2.0 * ((m.ndn[:, m.lat.sub == 0].sum(axis=1)) % 2)
    return kappa * marshall * batched_sign_det(X[m.h_pos])


def gf2_solvable(A, b):
    A = A.copy() % 2
    b = b.copy() % 2
    nr, nc = A.shape
    r = 0
    for c in range(nc):
        piv = np.nonzero(A[r:, c])[0]
        if len(piv) == 0:
            continue
        p = r + piv[0]
        A[[r, p]] = A[[p, r]]; b[[r, p]] = b[[p, r]]
        rows = np.nonzero(A[:, c])[0]
        rows = rows[rows != r]
        A[rows] ^= A[r]; b[rows] ^= b[r]
        r += 1
        if r == nr:
            break
    return not np.any(b[r:] & 1) and not np.any((A.sum(axis=1) == 0) & (b == 1))


def pairwise_check():
    lat = Lattice(3, 4, False)
    N = 4
    T = np.zeros((lat.V, lat.V))
    for i, j, _ in lat.nn():
        T[i, j] = T[j, i] = -1.0
    for i, j, _ in lat.nnn():
        T[i, j] = T[j, i] = 0.3
    w, U = np.linalg.eigh(T)
    print(f"orbital gap e_N - e_(N-1) = {w[N]-w[N-1]:.4f}")
    confs = list(itertools.combinations(range(lat.V), N))
    amps = np.array([np.linalg.det(U[list(c), :N]) for c in confs])
    keep = np.abs(amps) > 1e-10
    pairs = list(itertools.combinations(range(lat.V), 2))
    rows = []
    for c in confs:
        n = np.zeros(lat.V, np.int64); n[list(c)] = 1
        rows.append(np.concatenate([[1], n, [n[i] * n[j] for i, j in pairs]]))
    A = np.array(rows, np.int64)[keep]
    b = (amps[keep] < 0).astype(np.int64)
    sol = gf2_solvable(A.astype(np.uint8), b.astype(np.uint8))
    print(f"configs={keep.sum()} unknowns={A.shape[1]}  sign is quadratic GF(2) polynomial: {sol}")
    return sol




# ----------------------------------------------------------------------------- Walsh sign layer
def plaquette_supports(lat, degree):
    """All subsets (size 1..degree) of the 8 binary variables of each 2x2 plaquette.
    Binary variable index: 2*i (n_i,up) and 2*i+1 (n_i,dn)."""
    sups = set()
    for i in range(lat.V):
        x, y = lat.xy[i]
        sites = [lat.idx(x + dx, y + dy) for dx in (0, 1) for dy in (0, 1)]
        if any(s is None for s in sites) or len(set(sites)) < 4:
            continue
        var = [2 * s + b for s in sites for b in (0, 1)]
        for d in range(1, degree + 1):
            for S in itertools.combinations(sorted(var), d):
                sups.add(S)
    return sorted(sups)


def null_supports(local, nvar, rng):
    out = set()
    for S in local:
        while True:
            T = tuple(sorted(rng.choice(nvar, size=len(S), replace=False)))
            if T not in out:
                out.add(T)
                break
    return sorted(out)


def walsh_features(m, sups, rows):
    z = np.empty((len(rows), 2 * m.lat.V), np.int8)
    z[:, 0::2] = 1 - 2 * m.nup[rows]
    z[:, 1::2] = 1 - 2 * m.ndn[rows]
    F = np.empty((len(rows), len(sups)), np.float32)
    for k, S in enumerate(sups):
        F[:, k] = np.prod(z[:, list(S)], axis=1)
    return F


def fit_layer(m, sups, y_all, p0, ntrain, l2, rng):
    """Logistic fit of c(n) = sgn(b + w.chi(n)) to the residual sign y on configurations
    sampled from |psi_0|^2.  The L2 penalty is chosen from {l2/10, l2, 10 l2, 100 l2}
    by the sign error on an independent validation sample of size ntrain/2; the
    constant layer c = +1 (i.e. the bare foil) is kept if it validates better."""
    rows = rng.choice(m.D, size=ntrain, replace=True, p=p0)
    vrows = rng.choice(m.D, size=max(ntrain // 2, 1), replace=True, p=p0)
    X = walsh_features(m, sups, rows).astype(np.float64)
    Xv = walsh_features(m, sups, vrows).astype(np.float64)
    y, yv = y_all[rows], y_all[vrows]
    P = X.shape[1]

    def loss(wb, lam):
        w, b = wb[:P], wb[P]
        z = y * (X @ w + b)
        l = np.logaddexp(0, -z).mean() + lam * np.dot(w, w)
        sgm = -np.exp(-np.logaddexp(0, z))
        g = X.T @ (sgm * y) / len(y) + 2 * lam * w
        return l, np.append(g, np.mean(sgm * y))

    best = (np.mean(yv < 0), None)  # constant layer c = +1
    for lam in (l2 / 10, l2, 10 * l2, 100 * l2):
        r = minimize(loss, np.zeros(P + 1), args=(lam,), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=2000))
        err = np.mean(np.sign(Xv @ r.x[:P] + r.x[P]) != yv)
        if err < best[0]:
            best = (err, r.x)
    c = np.ones(m.D)
    if best[1] is not None:
        w, b = best[1][:P], best[1][P]
        for a in range(0, m.D, 50000):
            rr = np.arange(a, min(a + 50000, m.D))
            c[rr] = np.sign(walsh_features(m, sups, rr) @ w + b)
        c[c == 0] = 1.0
    return c, float(best[0]), best[1] is not None


# ----------------------------------------------------------------------------- driver
def foil_sign(m, fam, pi, maxfev, starts, rng):
    if fam == "slater_af":
        f, x0 = (lambda p: sign_slater_af(m, p, pi)), np.array([m.tp, 0.3, 0.02])
    else:
        f, x0 = (lambda p: sign_bcs_d(m, p, pi)), np.array([0.3, -0.5, m.tp])
    best = (np.inf, None)
    for st in range(starts):
        x = x0 if st == 0 else x0 + 0.3 * rng.standard_normal(3)
        r = minimize(lambda p: m.fixed_node(f(p), iters=m.fn_opt_iters)[0], x, method="Nelder-Mead",
                     options=dict(maxfev=maxfev, xatol=1e-3, fatol=1e-7))
        if r.fun < best[0]:
            best = (r.fun, r.x)
    return f(best[1]), [float(v) for v in best[1]]


def weighted_W(p0, s0, s):
    w = float(np.sum(p0 * (s != s0)))
    return min(w, 1 - w)


def run_instance(spec, args, rng):
    latspec, nh, tp = spec.split(":")
    Lx, Ly = map(int, latspec[:-1].split("x"))
    lat = Lattice(Lx, Ly, latspec[-1] == "p")
    nh, tp = int(nh), float(tp)
    ne = lat.V - nh
    assert ne % 2 == 0, "need an even electron number"
    t0 = time.time()
    m = TJ(lat, ne // 2, ne // 2, tp, args.J)
    m.fn_iters, m.fn_opt_iters = args.fn_iters, args.fn_opt_iters
    E0, U0, deg = m.ground()
    print(f"[{spec}] D={m.D} E0={E0:.8f} deg={deg} ({time.time()-t0:.1f}s)", flush=True)
    pi = block_to_site_sign(m)
    fams = ["slater_af"] + (["bcs_d"] if lat.pbc else [])
    cand = {}
    for fam in fams:
        t1 = time.time()
        s, par = foil_sign(m, fam, pi, args.maxfev, args.starts, rng)
        e, psi = m.fixed_node(s)
        cand[fam] = (e, s, psi, par)
        print(f"  foil {fam:9s} E_FN={e:.8f} dE={e-E0:.4e} params={np.round(par,4).tolist()} ({time.time()-t1:.0f}s)", flush=True)
    foil = min(cand, key=lambda k: cand[k][0])
    eF, sF, psiF, parF = cand[foil]
    if deg == 1:
        psi0 = U0[:, 0]
    else:
        psi0 = U0 @ (U0.T @ psiF)
        psi0 /= np.linalg.norm(psi0)
    p0 = psi0 ** 2
    p0 = p0 / p0.sum()
    s0 = np.where(psi0 >= 0, 1.0, -1.0)
    if np.sum(p0 * s0 * sF) < 0:  # fix the global sign so that the residual is mostly +1
        s0, psi0 = -s0, -psi0
    Cs0, G0 = m.correlations(p0)
    eEx, _ = m.fixed_node(s0)
    y = s0 * sF
    local = plaquette_supports(lat, args.degree)
    null = null_supports(local, 2 * lat.V, rng)
    rows = {}

    def record(name, s, e, psi):
        Cs, G = m.correlations(psi ** 2 / np.sum(psi ** 2))
        rows[name] = dict(dE=float(e - E0), W=weighted_W(p0, s0, s),
                          errCs=float(np.max(np.abs(Cs - Cs0))), errG=float(np.max(np.abs(G - G0))))

    record("exact_sign", s0, eEx, s0 * np.abs(psi0))
    record("foil", sF, eF, psiF)
    for name, sups in (("local", local), ("null", null)):
        t1 = time.time()
        c, verr, used = fit_layer(m, sups, y, p0, args.ntrain, args.l2, rng)
        e, psi = m.fixed_node(sF * c)
        record(name, sF * c, e, psi)
        print(f"  layer {name:5s} P={len(sups)} val_err={verr:.4f} (const-layer weighted err {np.sum(p0*(y<0)):.4f}) "
              f"nontrivial={used} E_FN={e:.8f} ({time.time()-t1:.0f}s)", flush=True)
    for k, v in rows.items():
        print(f"  {k:10s} dE={v['dE']:.4e} W={v['W']:.4f} errCs={v['errCs']:.3e} errG={v['errG']:.3e}")
    dF = rows["foil"]["dE"]
    Ml, Mn = rows["local"]["dE"] / dF, rows["null"]["dE"] / dF
    corr_ok = rows["local"]["errCs"] <= rows["foil"]["errCs"] and rows["local"]["errG"] <= rows["foil"]["errG"]
    print(f"  foil={foil} params={np.round(parF,4).tolist()}  M_local={Ml:.4f}  M_null={Mn:.4f}  corr_ok={corr_ok}", flush=True)
    return dict(spec=spec, D=m.D, E0=float(E0), deg=deg, foil=foil, foil_params=parF, rows=rows,
                M_local=Ml, M_null=Mn, corr_ok=bool(corr_ok), P=len(local))


def verdict(results):
    if any(r["M_local"] >= 0.9 or r["M_local"] >= r["M_null"] for r in results):
        return "FAIL"
    if all(r["M_local"] <= 0.5 and r["M_null"] - r["M_local"] >= 0.1 and r["corr_ok"] for r in results):
        return "PASS"
    return "AMBIGUOUS"


# ----------------------------------------------------------------------------- checks
def holon_check():
    lat = Lattice(3, 4, False)
    for J in (100.0, 10.0, 2.0, 0.4):
        m = TJ(lat, 5, 5, 0.0, J)
        E0, U0, _ = m.ground()
        p, s0 = U0[:, 0] ** 2, np.sign(U0[:, 0])
        M = 1.0 - 2.0 * (m.ndn[:, lat.sub == 0].sum(axis=1) % 2)
        r = M * s0
        hk = m.h_pos[:, 0] * lat.V + m.h_pos[:, 1]
        W = 0.0
        for h in np.unique(hk):
            sel = hk == h
            a = np.sum(p[sel] * (r[sel] > 0))
            W += min(a, np.sum(p[sel]) - a)
        print(f"J/t={J:6.1f}  min_f W(Marshall * f(holes)) = {W:.4f}")


def smoke():
    ok = True
    lat = Lattice(2, 1, False)
    m = TJ(lat, 1, 0, 0.0, 0.4)
    E0 = np.linalg.eigvalsh(m.H.toarray())[0]
    print(f"(a) 2-site 1 electron: E0={E0:.10f} (expect -1)"); ok &= abs(E0 + 1) < 1e-10
    m = TJ(Lattice(2, 2, False), 2, 2, 0.0, 1.0)
    E0 = np.linalg.eigvalsh(m.H.toarray())[0]
    Ef, _ = m.fixed_node(sign_holon_anchor(m, (0.0, 0.0, 0.0), 1.0), iters=0)
    print(f"(b) 4-site ring Heisenberg: E0={E0:.10f} (expect -3)  E_FN[Marshall]={Ef:.10f}")
    ok &= abs(E0 + 3) < 1e-9 and abs(Ef - E0) < 1e-8
    for (Lx, Ly) in ((2, 3), (3, 4)):
        lat = Lattice(Lx, Ly, False)
        T = hopping_matrix(lat, 1.0, -0.3, 0.0, 0.0)
        for nh in (1, 2):
            m = TJ(lat, lat.V - nh, 0, -0.3, 0.4)
            w, U = np.linalg.eigh(m.H.toarray())
            Efree = np.sum(np.linalg.eigvalsh(T)[:lat.V - nh])
            p0, s0 = U[:, 0] ** 2, np.sign(U[:, 0])
            Wh = weighted_W(p0, s0, sign_holon_anchor(m, (-0.3, 0.0, 0.0), 1.0))
            Ws = weighted_W(p0, s0, sign_slater_af(m, (-0.3, 0.0, 0.0), block_to_site_sign(m)))
            print(f"(c) polarised {Lx}x{Ly} N_h={nh}: E0-E_free={w[0]-Efree:.2e}  W[holon det]={Wh:.2e}  W[Slater]={Ws:.2e}")
            ok &= abs(w[0] - Efree) < 1e-9 and Wh < 1e-10 and Ws < 1e-10
    m = TJ(Lattice(3, 4, False), 5, 5, -0.3, 0.4)
    E0, U0, _ = m.ground()
    s0 = np.where(U0[:, 0] >= 0, 1.0, -1.0)
    es = [m.fixed_node(s0, iters=K)[0] - E0 for K in (0, 5, 10)]
    print(f"(d) doped 3x4 exact sign: E_FN-E0 at K=0,5,10: {np.round(es,5).tolist()}")
    ok &= es[0] >= es[1] >= es[2] >= -1e-9
    a = parse_args(["--ntrain", "1500", "--maxfev", "40", "--starts", "1"])
    r = run_instance("3x4o:2:-0.3", a, np.random.default_rng(0))
    ok &= r["rows"]["foil"]["dE"] >= -1e-9 and r["rows"]["local"]["dE"] >= -1e-9
    print("SMOKE:", "OK" if ok else "FAILED")
    return ok


# ----------------------------------------------------------------------------- CLI
def parse_args(argv):
    p = argparse.ArgumentParser(description="P1.2: local Walsh sign layer on the foil determinant, fixed-node test")
    p.add_argument("--mode", choices=["decisive", "smoke", "holon_check", "pairwise_check"], default="decisive")
    p.add_argument("--instances", default="4x4p:2:-0.3,4x4p:2:0.3,3x4o:2:0.0")
    p.add_argument("--J", type=float, default=0.4)
    p.add_argument("--degree", type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--ntrain", type=int, default=4000)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--maxfev", type=int, default=80)
    p.add_argument("--starts", type=int, default=2)
    p.add_argument("--fn-opt-iters", type=int, default=2)
    p.add_argument("--fn-iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.mode == "smoke":
        return 0 if smoke() else 1
    if args.mode == "holon_check":
        holon_check()
        return 0
    if args.mode == "pairwise_check":
        pairwise_check()
        return 0
    rng = np.random.default_rng(args.seed)
    results = [run_instance(s.strip(), args, rng) for s in args.instances.split(",")]
    v = verdict(results)
    print("M_local per instance:", {r["spec"]: round(r["M_local"], 4) for r in results})
    print("M_null  per instance:", {r["spec"]: round(r["M_null"], 4) for r in results})
    print(f"VERDICT: {v}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(dict(args=vars(args), results=results, verdict=v), fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
