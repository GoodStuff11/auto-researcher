#!/usr/bin/env python3
"""
P3.1 decisive micro-test: order-operator-augmented Ritz step (A-Ritz) as an
estimator of the error of a long-range correlation C_theta(r) of an imperfect
variational state, versus the standard one-Lanczos-step + variance
extrapolation (L1-vex) baseline, on exactly diagonalizable doped Hubbard
clusters with t' != 0.

Physics being tested
--------------------
Given a normalized trial state psi (a stand-in for a VMC/NQS state) and a
Hermitian, lattice-averaged correlation operator A (pairing, spin or charge at
the largest separation r), the A-Ritz step solves the Rayleigh-Ritz problem of
H in the small subspace
    own3 : span{ psi, H psi, A psi }
    own4 : span{ psi, H psi, A psi, H A psi }
    l1   : span{ psi, H psi }                       (baseline: one Lanczos step)
and takes the lowest Ritz vector phi.  From the two points
(sigma2_psi, a_psi) and (sigma2_phi, a_phi), where sigma2 = <H^2> - <H>^2 and
a = <A>, a linear zero-variance extrapolation gives
    a_vex = a_psi + (a_phi - a_psi) * sigma2_psi / (sigma2_psi - sigma2_phi).
The PREDICTED error of the trial state is  dhat = a_vex - a_psi;  the TRUE
error is  d = a_0 - a_psi  with a_0 = <psi_0|A|psi_0> from ED.

A state is "trusted" (the method issues a certificate) iff the Ritz step
removed at least half of the variance: (sigma2_psi - sigma2_phi) / sigma2_psi
>= 0.5.  A state counts as "tracked" iff it is trusted, |d| > floor, and
1/2 <= dhat/d <= 2.  A state is a "false certificate" iff it is trusted,
|d| > floor, and dhat/d < 1/2 (this includes the wrong sign).

Metric (pre-committed, primary = --subspace own3, --estimator vex, exact
matrix elements):
    M   = fraction of near-ground-state trial states (families nd with eta>0
          and amp) with |d| > floor that are tracked;
    M_L = same quantity for the l1 baseline;
    G   = fraction of ALL trial states (nd, amp, mf) with |d| > floor that are
          false certificates for the A-Ritz method.
PASS: M >= 0.70 and M - M_L >= 0.20 and G <= 0.05.
FAIL: M < 0.50 or M - M_L < 0.05 or G > 0.15.
Otherwise AMBIGUOUS (pre-committed action: rerun with --subspace own4 and
--mc_samples 100000; if still not PASS the route is killed).

Trial-state families (all real, fixed N_up = N_dn):
    nd  : psi_0 + s * psi_k  with psi_k one of the lowest same-(N_up,N_dn)
          excited states that couple to A (|<k|A|0>| non-negligible), plus
          multiplicative amplitude noise psi_0(n)*eta*xi_n (xi_n ~ N(0,1)).
          s in {0.1,0.3,0.6}, eta in {0,0.1,0.3}.  eta=0 states are pure
          two-level mixtures (trivially solved by one Lanczos step) and are
          reported but excluded from M.
    amp : psi_0(n) * exp(sum_d g_d J_d(n) + g_D D(n) + eta*xi_n) with random
          symmetric Jastrow couplings g (density-density at displacement d,
          double occupancy D) plus sign flips on a fraction f of the
          configurations with the smallest |psi_0|.  Mimics NQS-type
          amplitude and sign errors.
    mf  : Minimal Contrast Foil.  Gutzwiller-Jastrow projected d-wave BCS
          pair-product state  det[phi(up_i, dn_j)] * exp(g D + v1 J_nn + v2 J_nnn),
          phi from a real-space BdG mean field (t, t', mu, Delta_d).  One
          member is energy-optimized by Nelder-Mead; further members scan
          Delta_d (wrong pairing amplitude) at fixed optimized Jastrow.

Observables (A, all Hermitian, averaged over lattice translations i with
both i and i+r inside the lattice):
    pair   : (1/n_i) sum_i (Delta_i^dag Delta_{i+r} + h.c.)/2,
             Delta_i = Delta_{i,i+x} - Delta_{i,i+y} (d-wave),
             Delta_{ij} = (c_{i up} c_{j dn} - c_{i dn} c_{j up})/sqrt(2)
    spin   : (1/n_i) sum_i S_i . S_{i+r}
    charge : (1/n_i) sum_i n_i n_{i+r}
r is the largest displacement (floor(Lx/2), floor(Ly/2)) unless --rx/--ry
given.

CLI options (all have defaults; run `python micro_test.py --help`)
------------------------------------------------------------------
--clusters STR      Semicolon-separated list "Lx,Ly,Ne[,bc]".  bc is one of
                    pbc  : periodic in x and y (duplicate bonds on width-2
                           directions are counted once),
                    cyl  : periodic in x, open in y,
                    obc  : open in both.
                    Default "4,2,4,cyl" (smoke size). Ne must be even (N_up = N_dn = Ne/2).
                    Note: 4x2 cyl Ne=6 and 6x2 cyl Ne=10 have a ground state in a
                    symmetry sector orthogonal to the foil (foil fidelity 0); avoid
                    them for the mf family. Decisive run: see submit_test.sbatch.
                    Hilbert dimension = C(Lx*Ly, Ne/2)^2; keep it below ~1e6.
--U FLOAT           On-site repulsion U (default 8.0). t = 1 fixed.
--tp STR            Comma-separated list of t' values (NNN hopping, sign
                    convention H = -t sum_nn - t' sum_nnn); every value is a
                    separate instance.  Default "-0.25".
--rx INT, --ry INT  Correlation displacement; default -1 means floor(L/2).
--observables STR   Comma-separated subset of {pair,spin,charge}.
                    Default "pair,spin,charge".
--families STR      Comma-separated subset of {nd,amp,mf}. Default "nd,amp,mf".
--subspace STR      A-Ritz subspace: own3 (default, cost ~ #H * #A per sample)
                    or own4 (adds H A psi, cost ~ #H^2 * #A per sample).  The
                    l1 baseline is always computed.
--estimator STR     vex (default; linear zero-variance extrapolation through
                    psi and phi) or ritz (dhat = a_phi - a_psi, no
                    extrapolation).  Both are always printed; this flag only
                    selects which one enters M, M_L, G and the verdict.
--mc_samples INT    0 (default) = exact matrix elements.  >0 = every matrix
                    element is replaced by its VMC local estimator over this
                    many i.i.d. configurations drawn from |psi|^2 (emulates
                    a VMC run; no autocorrelation).
--n_coupled INT     Number of lowest A-coupled excited states used in nd
                    (default 3).
--n_eig INT         Number of lowest eigenpairs computed by eigsh (default 12).
--n_seeds INT       Random repetitions per family grid point (default 2).
--mf_maxiter INT    Nelder-Mead iterations for the foil (default 300).
--err_floor FLOAT   Relative floor: states with |d| <= err_floor*max(|a_0|,1e-3)
                    are excluded from M, M_L, G (default 0.05).
--seed INT          Master RNG seed (default 1234).
--out STR           Path of a JSON results file (default: results_<timestamp>.json
                    next to this script).  "none" disables writing.
--quick             Smoke mode: one seed, s in {0.3}, eta in {0.1}, few amp
                    and mf members, mf_maxiter 40.  For verifying the script.

Exit code is 0 regardless of verdict; the verdict is printed on the last line
as  VERDICT: PASS | FAIL | AMBIGUOUS.
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
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.optimize import minimize


# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="P3.1 A-Ritz order-parameter error micro-test "
                                            "(see module docstring for full documentation)")
    p.add_argument("--clusters", default="4,2,4,cyl")
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--tp", default="-0.25")
    p.add_argument("--rx", type=int, default=-1)
    p.add_argument("--ry", type=int, default=-1)
    p.add_argument("--observables", default="pair,spin,charge")
    p.add_argument("--families", default="nd,amp,mf")
    p.add_argument("--subspace", default="own3", choices=["own3", "own4"])
    p.add_argument("--estimator", default="vex", choices=["vex", "ritz"])
    p.add_argument("--mc_samples", type=int, default=0)
    p.add_argument("--n_coupled", type=int, default=3)
    p.add_argument("--n_eig", type=int, default=12)
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--mf_maxiter", type=int, default=300)
    p.add_argument("--err_floor", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    clusters = []
    for c in a.clusters.split(";"):
        f = c.strip().split(",")
        Lx, Ly, Ne = int(f[0]), int(f[1]), int(f[2])
        bc = f[3].strip() if len(f) > 3 else "pbc"
        if bc not in ("pbc", "cyl", "obc"):
            p.error(f"bad bc {bc}")
        if Ne % 2:
            p.error("Ne must be even")
        clusters.append((Lx, Ly, Ne, bc))
    a.cluster_list = clusters
    a.tp_list = [float(x) for x in a.tp.split(",")]
    a.obs_list = [x.strip() for x in a.observables.split(",")]
    for o in a.obs_list:
        if o not in ("pair", "spin", "charge"):
            p.error(f"bad observable {o}")
    a.fam_list = [x.strip() for x in a.families.split(",")]
    for f_ in a.fam_list:
        if f_ not in ("nd", "amp", "mf"):
            p.error(f"bad family {f_}")
    if a.quick:
        a.n_seeds = 1
        a.mf_maxiter = min(a.mf_maxiter, 40)
    return a


# ----------------------------------------------------------------------------
# Lattice and single-species Fock space
# ----------------------------------------------------------------------------
class Lattice:
    def __init__(self, Lx, Ly, bc):
        self.Lx, self.Ly, self.bc = Lx, Ly, bc
        self.V = Lx * Ly
        self.px = bc in ("pbc", "cyl")
        self.py = bc == "pbc"

    def site(self, x, y):
        if self.px:
            x %= self.Lx
        if self.py:
            y %= self.Ly
        if 0 <= x < self.Lx and 0 <= y < self.Ly:
            return x + self.Lx * y
        return None

    def coords(self, s):
        return s % self.Lx, s // self.Lx

    def bonds(self, disps):
        out = set()
        for s in range(self.V):
            x, y = self.coords(s)
            for dx, dy in disps:
                j = self.site(x + dx, y + dy)
                if j is not None and j != s:
                    out.add((min(s, j), max(s, j)))
        return sorted(out)


class Species:
    """Basis of bit-masks with n particles on V sites, sorted ascending."""

    def __init__(self, V, n):
        self.V, self.n = V, n
        masks = [sum(1 << i for i in c) for c in itertools.combinations(range(V), n)]
        self.masks = np.array(sorted(masks), dtype=np.int64)
        self.dim = len(self.masks)
        self.occ = ((self.masks[:, None] >> np.arange(V)[None, :]) & 1).astype(np.int8)

    def index(self, m):
        return np.searchsorted(self.masks, m)

    def hop(self, a, b):
        """Sparse matrix of c^dag_a c_b within this species (string sign)."""
        m = self.masks
        if a == b:
            return sp.diags(self.occ[:, a].astype(float)).tocsr()
        sel = ((m >> b) & 1).astype(bool) & ~((m >> a) & 1).astype(bool)
        src = np.nonzero(sel)[0]
        ms = m[src]
        new = (ms ^ (1 << b)) | (1 << a)
        lo, hi = min(a, b), max(a, b)
        between = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
        cnt = np.array([bin(int(v)).count("1") for v in (ms & between)])
        sgn = np.where(cnt % 2 == 0, 1.0, -1.0)
        dst = self.index(new)
        return sp.csr_matrix((sgn, (dst, src)), shape=(self.dim, self.dim))


# ----------------------------------------------------------------------------
# Operators on psi stored as matrices Psi[iu, id]
# ----------------------------------------------------------------------------
class Op:
    """Sum_k (Up_k (x) Dn_k) + diag.  Up_k / Dn_k sparse; diag array (du,dd)."""

    def __init__(self, du, dd):
        self.du, self.dd = du, dd
        self.terms = {}  # key -> [Up, Dn]  (Dn accumulates)
        self.diag = np.zeros((du, dd))

    def add(self, key, Up, Dn, c=1.0):
        if key in self.terms:
            self.terms[key][1] = self.terms[key][1] + c * Dn
        else:
            self.terms[key] = [Up, c * Dn]

    def apply(self, v):
        P = v.reshape(self.du, self.dd)
        out = self.diag * P
        for Up, Dn in self.terms.values():
            out = out + (Dn @ (Up @ P).T).T
        return out.reshape(-1)


class Hubbard:
    def __init__(self, lat, Ne, U, tp):
        self.lat, self.U, self.tp = lat, U, tp
        V = lat.V
        self.up = Species(V, Ne // 2)
        self.dn = Species(V, Ne // 2)
        du, dd = self.up.dim, self.dn.dim
        self.du, self.dd, self.dim = du, dd, du * dd
        self.nn = lat.bonds([(1, 0), (0, 1)])
        self.nnn = lat.bonds([(1, 1), (1, -1)])
        Tu = sp.csr_matrix((du, du))
        Td = sp.csr_matrix((dd, dd))
        for (i, j), t in [(b, 1.0) for b in self.nn] + [(b, tp) for b in self.nnn]:
            if t == 0:
                continue
            Tu = Tu - t * (self.up.hop(i, j) + self.up.hop(j, i))
            Td = Td - t * (self.dn.hop(i, j) + self.dn.hop(j, i))
        self.Tu, self.Td = Tu.tocsr(), Td.tocsr()
        nu = self.up.occ.astype(float)
        nd = self.dn.occ.astype(float)
        self.nu, self.nd = nu, nd
        self.D = (nu @ nd.T)  # double occupancy, shape (du, dd)
        self.Ddiag = U * self.D

    def apply(self, v):
        P = v.reshape(self.du, self.dd)
        out = self.Tu @ P + (self.Td @ P.T).T + self.Ddiag * P
        return out.reshape(-1)

    def linop(self):
        return LinearOperator((self.dim, self.dim), matvec=self.apply, dtype=float)

    # density on site i as (du,dd) arrays
    def n_site(self, i):
        return self.nu[:, i][:, None] + self.nd[:, i][None, :]

    def sz_site(self, i):
        return 0.5 * (self.nu[:, i][:, None] - self.nd[:, i][None, :])


def corr_pairs(lat, rx, ry):
    out = []
    for s in range(lat.V):
        x, y = lat.coords(s)
        j = lat.site(x + rx, y + ry)
        if j is not None:
            out.append((s, j))
    return out


def build_observable(ham, name, rx, ry):
    lat = ham.lat
    A = Op(ham.du, ham.dd)
    pairs = corr_pairs(lat, rx, ry)
    if not pairs:
        raise ValueError("no site pairs at displacement r")
    if name == "charge":
        for i, j in pairs:
            A.diag += ham.n_site(i) * ham.n_site(j)
        A.diag /= len(pairs)
        return A
    if name == "spin":
        w = 1.0 / len(pairs)
        for i, j in pairs:
            A.diag += w * ham.sz_site(i) * ham.sz_site(j)
            if i == j:
                raise ValueError("r = 0 not supported")
            # (S+_i S-_j + S-_i S+_j)/2 = -(E^u_ij E^d_ji + E^u_ji E^d_ij)/2
            A.add(("u", i, j), ham.up.hop(i, j), ham.dn.hop(j, i), -0.5 * w)
            A.add(("u", j, i), ham.up.hop(j, i), ham.dn.hop(i, j), -0.5 * w)
        return A
    if name == "pair":
        # d-wave site pair: Delta_i = sum_{delta in {x,y}} f_delta Delta_{i,i+delta}
        def dwave_bonds(i):
            x, y = lat.coords(i)
            out = []
            for (dx, dy), f in (((1, 0), 1.0), ((0, 1), -1.0)):
                j = lat.site(x + dx, y + dy)
                if j is not None and j != i:
                    out.append((i, j, f))
            return out
        # Delta^dag_{kl} Delta_{ij} = 1/2 sum_{(a,b) in {(k,l),(l,k)}} sum_{(c,d) in {(i,j),(j,i)}}
        #     c^dag_{a up} c^dag_{b dn} c_{d dn} c_{c up} = 1/2 sum E^u_{ac} (x) E^d_{bd}
        terms = []
        for i0, j0 in pairs:
            for (k, l, f1) in dwave_bonds(i0):
                for (i, j, f2) in dwave_bonds(j0):
                    if len({k, l, i, j}) < 4:
                        continue
                    terms.append((k, l, i, j, f1 * f2))
        w = 1.0 / len(pairs)
        for (k, l, i, j, f) in terms:
            for (a, b) in ((k, l), (l, k)):
                for (c, d) in ((i, j), (j, i)):
                    # Hermitian part: (X + X^dag)/2
                    coef = 0.5 * f * w * 0.5
                    A.add(("u", a, c), ham.up.hop(a, c), ham.dn.hop(b, d), coef)
                    A.add(("u", c, a), ham.up.hop(c, a), ham.dn.hop(d, b), coef)
        return A
    raise ValueError(name)


def total_S2(ham):
    """Operator S_tot^2 for a sanity check of the spin-operator signs."""
    lat = ham.lat
    A = Op(ham.du, ham.dd)
    V = lat.V
    for i in range(V):
        for j in range(V):
            A.diag += ham.sz_site(i) * ham.sz_site(j)
            if i != j:
                A.add(("u", i, j), ham.up.hop(i, j), ham.dn.hop(j, i), -0.5)
                A.add(("u", j, i), ham.up.hop(j, i), ham.dn.hop(i, j), -0.5)
            else:
                # (S+_i S-_i + S-_i S+_i)/2 = (n_iu + n_id - 2 n_iu n_id)/2
                A.diag += 0.5 * (ham.nu[:, i][:, None] + ham.nd[:, i][None, :]
                                 - 2 * ham.nu[:, i][:, None] * ham.nd[:, i][None, :])
    return A


# ----------------------------------------------------------------------------
# Ritz step and estimators
# ----------------------------------------------------------------------------
def ritz_estimate(ham, A, psi, basis_kind, mc_samples, rng):
    """Return dict with a_psi, s2_psi, a_phi, s2_phi, E_psi, E_phi."""
    Hpsi = ham.apply(psi)
    vecs = [psi, Hpsi]
    if basis_kind in ("own3", "own4"):
        Apsi = A.apply(psi)
        vecs.append(Apsi)
        if basis_kind == "own4":
            vecs.append(ham.apply(Apsi))
    Hv = [Hpsi] + [ham.apply(v) for v in vecs[1:]]
    Av = [A.apply(v) for v in vecs]
    Vm = np.array(vecs)
    HVm = np.array(Hv)
    AVm = np.array(Av)
    if mc_samples > 0:
        p = psi ** 2
        p = p / p.sum()
        idx = rng.choice(len(psi), size=mc_samples, p=p)
        w = 1.0 / psi[idx]
        X = Vm[:, idx] * w
        HX = HVm[:, idx] * w
        AX = AVm[:, idx] * w
        n = mc_samples
        S = X @ X.T / n
        Hm = X @ HX.T / n
        Am = X @ AX.T / n
        H2 = HX @ HX.T / n
    else:
        S = Vm @ Vm.T
        Hm = Vm @ HVm.T
        Am = Vm @ AVm.T
        H2 = HVm @ HVm.T
    Hm = 0.5 * (Hm + Hm.T)
    Am = 0.5 * (Am + Am.T)
    nrm0 = S[0, 0]
    E_psi = Hm[0, 0] / nrm0
    a_psi = Am[0, 0] / nrm0
    s2_psi = H2[0, 0] / nrm0 - E_psi ** 2
    # canonical orthogonalization
    sev, sU = np.linalg.eigh(S)
    keep = sev > 1e-10 * sev.max()
    X = sU[:, keep] / np.sqrt(sev[keep])
    hev, hU = np.linalg.eigh(X.T @ Hm @ X)
    c = X @ hU[:, 0]
    nc = c @ S @ c
    E_phi = c @ Hm @ c / nc
    a_phi = c @ Am @ c / nc
    s2_phi = c @ H2 @ c / nc - E_phi ** 2
    return dict(E_psi=E_psi, a_psi=a_psi, s2_psi=s2_psi,
                E_phi=E_phi, a_phi=a_phi, s2_phi=s2_phi)


def estimates(r):
    red = (r["s2_psi"] - r["s2_phi"]) / r["s2_psi"] if r["s2_psi"] > 0 else 0.0
    trusted = red >= 0.5
    if r["s2_psi"] - r["s2_phi"] > 1e-14:
        a_vex = r["a_psi"] + (r["a_phi"] - r["a_psi"]) * r["s2_psi"] / (r["s2_psi"] - r["s2_phi"])
    else:
        a_vex = r["a_phi"]
    return dict(var_reduction=red, trusted=bool(trusted),
                dhat_vex=a_vex - r["a_psi"], dhat_ritz=r["a_phi"] - r["a_psi"])


# ----------------------------------------------------------------------------
# Trial-state families
# ----------------------------------------------------------------------------
def normalize(v):
    return v / np.linalg.norm(v)


def jastrow_features(ham):
    """Symmetric (translation-summed) Jastrow features as flat arrays."""
    lat = ham.lat
    feats = [ham.D.reshape(-1)]
    for disp in ([(1, 0), (0, 1)], [(1, 1), (1, -1)], [(2, 0), (0, 2)]):
        b = lat.bonds(disp)
        if not b:
            continue
        f = np.zeros((ham.du, ham.dd))
        for i, j in b:
            f += ham.n_site(i) * ham.n_site(j)
        feats.append(f.reshape(-1))
    return feats


def family_nd(psi0, excited, rng, quick, n_seeds):
    out = []
    s_vals = [0.3] if quick else [0.1, 0.3, 0.6]
    eta_vals = [0.1] if quick else [0.0, 0.1, 0.3]
    for k, (Ek, vk) in enumerate(excited):
        for s in s_vals:
            for eta in eta_vals:
                for sd in range(n_seeds if eta > 0 else 1):
                    sign = rng.choice([-1.0, 1.0])
                    junk = psi0 * eta * rng.standard_normal(psi0.shape)
                    v = normalize(psi0 + sign * s * vk + junk)
                    out.append((dict(family="nd", k=k, s=s, eta=eta, seed=sd), v))
    return out


def family_amp(ham, psi0, rng, quick, n_seeds):
    out = []
    feats = jastrow_features(ham)
    g_vals = [0.2] if quick else [0.05, 0.2, 0.5]
    flip_vals = [0.05] if quick else [0.0, 0.05, 0.2]
    eta = 0.1
    order = np.argsort(np.abs(psi0))
    for g in g_vals:
        for fl in flip_vals:
            for sd in range(n_seeds):
                logw = sum(g * rng.standard_normal() * f / max(np.abs(f).max(), 1) for f in feats)
                logw = logw + eta * rng.standard_normal(psi0.shape)
                v = psi0 * np.exp(logw - logw.max())
                nflip = int(fl * len(v))
                if nflip:
                    pool = order[: max(nflip * 4, 1)]
                    fidx = rng.choice(pool, size=nflip, replace=False)
                    v[fidx] *= -1
                out.append((dict(family="amp", g=g, flip=fl, seed=sd), normalize(v)))
    return out


def bdg_pair_function(ham, mu, delta):
    lat = ham.lat
    V = lat.V
    K = np.zeros((V, V))
    for i, j in ham.nn:
        K[i, j] = K[j, i] = -1.0
    for i, j in ham.nnn:
        K[i, j] = K[j, i] = -ham.tp
    K -= mu * np.eye(V)
    Dm = np.zeros((V, V))
    for s in range(V):
        x, y = lat.coords(s)
        for (dx, dy), f in (((1, 0), 1.0), ((0, 1), -1.0)):
            j = lat.site(x + dx, y + dy)
            if j is not None and j != s:
                Dm[s, j] += delta * f
                Dm[j, s] += delta * f
    Dm += 1e-3 * np.eye(V)  # tiny s-wave regulator to keep phi finite
    M = np.block([[K, Dm], [Dm, -K]])
    ev, W = np.linalg.eigh(M)
    pos = ev > 0
    Um, Vm = W[:V, pos], W[V:, pos]
    phi = -np.linalg.solve(Um.T, Vm.T)
    return phi


def mf_state(ham, params, feats):
    mu, delta, g, v1, v2 = params
    phi = bdg_pair_function(ham, mu, delta)
    up_occ = ham.up.occ.astype(bool)
    dn_occ = ham.dn.occ.astype(bool)
    n = ham.up.n
    upi = np.array([np.nonzero(r)[0] for r in up_occ])  # (du, n)
    dni = np.array([np.nonzero(r)[0] for r in dn_occ])  # (dd, n)
    amp = np.empty((ham.du, ham.dd))
    for iu in range(ham.du):
        sub = phi[upi[iu]][:, dni]  # (n, dd, n)
        amp[iu] = np.linalg.det(np.transpose(sub, (1, 0, 2)))
    amp = amp.reshape(-1)
    jw = g * feats[0] + v1 * (feats[1] if len(feats) > 1 else 0) + v2 * (feats[2] if len(feats) > 2 else 0)
    jw = jw - jw.max()
    v = amp * np.exp(jw)
    nv = np.linalg.norm(v)
    if not np.isfinite(nv) or nv == 0:
        return None
    return v / nv


def family_mf(ham, rng, quick, maxiter):
    feats = jastrow_features(ham)

    def energy(x):
        v = mf_state(ham, x, feats)
        if v is None:
            return 1e6
        return float(v @ ham.apply(v))

    x0 = np.array([0.0, 0.3, -1.0, 0.0, 0.0])
    res = minimize(energy, x0, method="Nelder-Mead",
                   options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-6))
    xopt = res.x
    out = [(dict(family="mf", label="opt", params=xopt.tolist(), E=res.fun), mf_state(ham, xopt, feats))]
    for d in ([0.0, 0.8] if quick else [0.0, 0.05, 0.6, 1.2]):
        x = xopt.copy()
        x[1] = d
        v = mf_state(ham, x, feats)
        if v is not None:
            out.append((dict(family="mf", label=f"delta={d}", params=x.tolist()), v))
    return out


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------
def run_instance(args, Lx, Ly, Ne, bc, tp, rng, log):
    lat = Lattice(Lx, Ly, bc)
    t0 = time.time()
    ham = Hubbard(lat, Ne, args.U, tp)
    log(f"\n=== cluster {Lx}x{Ly} bc={bc} Ne={Ne} U={args.U} t'={tp}  dim={ham.dim} ===")
    k = min(args.n_eig, ham.dim - 2)
    ev, evec = eigsh(ham.linop(), k=k, which="SA", tol=1e-11)
    o = np.argsort(ev)
    ev, evec = ev[o], evec[:, o]
    E0, psi0 = ev[0], normalize(evec[:, 0])
    gap1 = ev[1] - ev[0]
    S2 = total_S2(ham)
    s2val = psi0 @ S2.apply(psi0)
    log(f"E0={E0:.8f}  E0/V={E0/lat.V:.6f}  gap1={gap1:.3e}  <S^2>={s2val:.6f}  (ED {time.time()-t0:.1f}s)")
    if gap1 < 1e-7:
        log("WARNING: ground state degenerate within the (N_up,N_dn) sector; instance skipped")
        return []
    rx = args.rx if args.rx >= 0 else Lx // 2
    ry = args.ry if args.ry >= 0 else Ly // 2
    obs = {name: build_observable(ham, name, rx, ry) for name in args.obs_list}
    # sanity: hermiticity of A on random vectors
    for name, A in obs.items():
        x, y = rng.standard_normal(ham.dim), rng.standard_normal(ham.dim)
        herm = abs(x @ A.apply(y) - y @ A.apply(x)) / (np.linalg.norm(x) * np.linalg.norm(y))
        log(f"obs {name}: r=({rx},{ry})  a0={psi0 @ A.apply(psi0):+.6e}  hermiticity_err={herm:.1e}")
    # excited states coupled to any A
    coupled = []
    for kk in range(1, len(ev)):
        vk = normalize(evec[:, kk])
        c = max(abs(vk @ A.apply(psi0)) for A in obs.values())
        if c > 1e-6:
            coupled.append((ev[kk], vk, c))
    log("A-coupled excited states (E-E0, max|<k|A|0>|): " +
        ", ".join(f"({e-E0:.4f}, {c:.2e})" for e, _, c in coupled[: args.n_coupled + 3]))
    excited = [(e, v) for e, v, _ in coupled[: args.n_coupled]]
    states = []
    if "nd" in args.fam_list:
        states += family_nd(psi0, excited, rng, args.quick, args.n_seeds)
    if "amp" in args.fam_list:
        states += family_amp(ham, psi0, rng, args.quick, args.n_seeds)
    if "mf" in args.fam_list:
        t1 = time.time()
        mfs = family_mf(ham, rng, args.quick, args.mf_maxiter)
        log(f"foil optimized: E_mf={mfs[0][0]['E']:.6f} (rel err {(mfs[0][0]['E']-E0)/abs(E0):.3e}) "
            f"params={np.round(mfs[0][0]['params'],3).tolist()}  ({time.time()-t1:.1f}s)")
        states += mfs
    rows = []
    for meta, psi in states:
        for name, A in obs.items():
            a0 = psi0 @ A.apply(psi0)
            d = a0 - psi @ A.apply(psi)
            rA = ritz_estimate(ham, A, psi, args.subspace, args.mc_samples, rng)
            rL = ritz_estimate(ham, A, psi, "l1", args.mc_samples, rng)
            eA, eL = estimates(rA), estimates(rL)
            rows.append(dict(cluster=f"{Lx}x{Ly}{bc}N{Ne}", tp=tp, obs=name, a0=float(a0), d=float(d),
                             E0=float(E0), Epsi=float(rA["E_psi"]), s2psi=float(rA["s2_psi"]),
                             fidelity=float((psi0 @ psi) ** 2),
                             A=dict({k_: float(v) for k_, v in rA.items()}, **eA),
                             L=dict({k_: float(v) for k_, v in rL.items()}, **eL),
                             **{k_: v for k_, v in meta.items() if k_ != "params"}))
    log(f"instance done: {len(states)} states x {len(obs)} observables ({time.time()-t0:.1f}s)")
    return rows


def classify(row, method, est, floor_rel):
    a0, d = row["a0"], row["d"]
    floor = floor_rel * max(abs(a0), 1e-3)
    if abs(d) <= floor:
        return "below_floor"
    m = row[method]
    dhat = m["dhat_" + est]
    ratio = dhat / d
    if not m["trusted"]:
        return "untrusted"
    if 0.5 <= ratio <= 2.0:
        return "tracked"
    if ratio < 0.5:
        return "false_cert"
    return "over"


def summarize(rows, args, log):
    est = args.estimator
    near = [r for r in rows if (r["family"] == "nd" and r.get("eta", 0) > 0) or r["family"] == "amp"]

    def frac(rs, method, label):
        cl = [classify(r, method, est, args.err_floor) for r in rs]
        valid = [c for c in cl if c != "below_floor"]
        if not valid:
            return float("nan"), 0
        return sum(c == label for c in valid) / len(valid), len(valid)

    M, nM = frac(near, "A", "tracked")
    ML, _ = frac(near, "L", "tracked")
    G, nG = frac(rows, "A", "false_cert")
    GL, _ = frac(rows, "L", "false_cert")
    log("\n--- per-family breakdown (estimator=%s) ---" % est)
    log(f"{'family':8s} {'obs':7s} {'n':>4s} | A: tracked untrusted false over | L1: tracked untrusted false over")
    for fam in ("nd0", "nd", "amp", "mf"):
        for name in args.obs_list:
            if fam == "nd0":
                rs = [r for r in rows if r["family"] == "nd" and r.get("eta", 1) == 0 and r["obs"] == name]
            elif fam == "nd":
                rs = [r for r in rows if r["family"] == "nd" and r.get("eta", 0) > 0 and r["obs"] == name]
            else:
                rs = [r for r in rows if r["family"] == fam and r["obs"] == name]
            if not rs:
                continue
            ca = [classify(r, "A", est, args.err_floor) for r in rs]
            cb = [classify(r, "L", est, args.err_floor) for r in rs]
            n = sum(c != "below_floor" for c in ca)
            f = lambda cl, lab: sum(c == lab for c in cl)
            log(f"{fam:8s} {name:7s} {n:4d} | A: {f(ca,'tracked'):4d} {f(ca,'untrusted'):4d} {f(ca,'false_cert'):4d} {f(ca,'over'):4d}"
                f"    | L1: {f(cb,'tracked'):4d} {f(cb,'untrusted'):4d} {f(cb,'false_cert'):4d} {f(cb,'over'):4d}")
    log("\n--- sample rows (first 12 above floor) ---")
    log(f"{'fam':4s} {'obs':6s} {'fid':>7s} {'d_true':>10s} {'dhatA':>10s} {'dhatL':>10s} {'redA':>6s} {'redL':>6s}")
    shown = 0
    for r in rows:
        if classify(r, "A", est, args.err_floor) == "below_floor":
            continue
        log(f"{r['family']:4s} {r['obs']:6s} {r['fidelity']:7.4f} {r['d']:+10.3e} {r['A']['dhat_'+est]:+10.3e} "
            f"{r['L']['dhat_'+est]:+10.3e} {r['A']['var_reduction']:6.3f} {r['L']['var_reduction']:6.3f}")
        shown += 1
        if shown >= 12:
            break
    log(f"\nM (A-Ritz {args.subspace}, tracked fraction, near-GS families) = {M:.3f}  (n={nM})")
    log(f"M_L (L1 baseline)                                       = {ML:.3f}")
    log(f"G (A-Ritz false-certificate fraction, all families)      = {G:.3f}  (n={nG});  G_L1 = {GL:.3f}")
    if any(np.isnan(x) for x in (M, ML, G)):
        verdict = "AMBIGUOUS"
    elif M >= 0.70 and M - ML >= 0.20 and G <= 0.05:
        verdict = "PASS"
    elif M < 0.50 or M - ML < 0.05 or G > 0.15:
        verdict = "FAIL"
    else:
        verdict = "AMBIGUOUS"
    return dict(M=M, M_L=ML, G=G, G_L=GL, nM=nM, nG=nG, verdict=verdict)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    log = lambda s: print(s, flush=True)
    log(f"P3.1 micro_test  args={vars(args)}")
    rows = []
    t0 = time.time()
    for (Lx, Ly, Ne, bc) in args.cluster_list:
        dim = comb(Lx * Ly, Ne // 2) ** 2
        if dim > 2_000_000:
            log(f"skip {Lx}x{Ly} Ne={Ne}: dim {dim} too large")
            continue
        for tp in args.tp_list:
            rows += run_instance(args, Lx, Ly, Ne, bc, tp, rng, log)
    summ = summarize(rows, args, log)
    log(f"total time {time.time()-t0:.1f}s")
    if args.out.lower() != "none":
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       f"results_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(out, "w") as fh:
            json.dump(dict(args={k: v for k, v in vars(args).items()}, summary=summ, rows=rows), fh, indent=1)
        log(f"results written to {out}")
    log(f"VERDICT: {summ['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
