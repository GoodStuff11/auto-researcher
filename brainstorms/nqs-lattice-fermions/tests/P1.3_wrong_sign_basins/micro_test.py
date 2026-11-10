#!/usr/bin/env python3
"""
P1.3 decisive test: do fermionic NQS get stuck in wrong-sign basins, and does
orbital-swap basin hopping (OSBH) get them out at matched optimizer cost?

What the script does
--------------------
1. Builds the 2D Hubbard model with nearest (t) and next-nearest (t') hopping
   on an Lx x Ly cluster at fixed (N_up, N_dn), as a sparse matrix over the
   full fixed-particle-number Hilbert space, and finds the exact ground state
   psi_0 and energy E_0 by Lanczos (scipy.sparse.linalg.eigsh).
2. Defines a real-valued fermionic NQS
       log psi(n) = J(n) + sum_h a_h log cosh(w_h . x + b_h)
                    + log det Phi_up(x)[occ_up] + log det Phi_dn(x)[occ_dn]
   with x = (n_up, n_dn) in {0,1}^{2 Ns}, a distance-class two-body Jastrow J,
   Ha hidden "amplitude" units, and a neural backflow
       Phi_s(x) = Phi0_s + sum_h tanh(Wb_h . x + cb_h) B_{s,h}
   with Hb hidden units. The sign of psi comes only from the two determinants.
   With --ansatz foil, Ha = Hb = 0: this is the Minimal Contrast Foil
   (Jastrow-Slater) of the brainstorm.
3. Optimizes it by stochastic reconfiguration (SR) with EXACT sums over the
   whole Hilbert space (no Monte Carlo noise), from several initializations:
     - "mf"   : free-fermion (t, t') orbitals plus a tiny random perturbation
                (noise seeds differ); this is the standard mean-field start.
     - "rand" : Gaussian random orbitals.
   For each init it runs two optimizers with the SAME total number of SR steps:
     - "sr"   : plain SR for --steps steps.
     - "osbh" : plain SR for --steps-pre steps, then repeated cycles of
                orbital-swap basin hopping: for each spin channel (up, dn, and
                both together) and each pair (a, b) of the --window highest
                "occupied" and --window lowest "virtual" one-body directions of
                the current reference orbitals Phi0 (defined by diagonalising
                the hopping matrix inside the occupied span and its
                complement), replace occupied direction a by virtual direction
                b, keep all other (Jastrow, amplitude, backflow) parameters,
                relax --relax-steps SR steps, and accept the lowest-energy
                candidate if it beats the current state relaxed by the same
                number of steps. All proposal relaxations count against the
                step budget. Leftover budget is spent on plain SR.
4. For every final state it reports the variational energy, the overlap with
   psi_0, the weighted sign agreement, and the LATTICE FIXED-NODE energy
   E_FN[s] of its sign structure s(n) = sgn psi_theta(n) (ten Haaf et al.,
   PRB 51, 13039 (1995)), computed with the guiding function g = s |psi_0|
   (exact amplitudes, trial signs) so that it measures the SIGN alone:
   sign-violating off-diagonal elements H_nm g_n g_m > 0 are removed and
   H_nm g_m / g_n is added to H_nn; E_FN is the ground state of the result.
   E_FN >= E_0, with equality iff s = sgn psi_0 on the support of psi_0.
   eps_s = (E_FN - E_0)/|E_0| is the sign-only error. Also reported:
   E_sign = <g|H|g>, the energy of exact amplitudes with trial signs.
5. Computes the decisive scalar (see the P1.3 artifact):
       eps_SR   = median over mf seeds of eps_s for plain SR,
       eps_OS   = median over mf seeds of eps_s for OSBH,
       eps_best = min over ALL runs (all inits, both optimizers) of eps_s,
       G        = eps_SR - eps_best         (basin gap; premise check)
       M        = (eps_SR - eps_OS) / G     (fraction of the basin gap closed)
   and prints PASS / FAIL / AMBIGUOUS according to the pre-committed
   thresholds (also listed below).

Thresholds (pre-committed; also in the artifact)
    basin exists   : G > 0.25 * eps_SR
    no basin       : G <= 0.10 * eps_SR       -> verdict FAIL (premise false)
    PASS           : basin exists AND M >= 0.5 AND median variational energy
                     of OSBH <= median variational energy of plain SR
    FAIL           : basin exists AND M < 0.2, OR no basin
    AMBIGUOUS      : everything else (0.2 <= M < 0.5, or
                     0.10*eps_SR < G <= 0.25*eps_SR, or M >= 0.5 with higher
                     variational energy). Action: rerun with --tp +0.3 and
                     doubled --n-mf/--n-rand; if still not PASS, kill the route.

Command-line options (all optional)
-----------------------------------
--lx INT            lattice length in x (default 4). Must be >= 1.
--ly INT            lattice length in y (default 3). Must be >= 1.
--bc STR            boundary conditions. Valid values:
                      "pbc"  periodic in both directions (default),
                      "obc"  open in both directions,
                      "pbcx" periodic in x, open in y,
                      "pbcy" open in x, periodic in y.
                    Bonds are de-duplicated, so a periodic direction of length
                    2 does not double the hopping, and length 1 adds no bond.
--nup INT           number of spin-up electrons (default 5).
--ndn INT           number of spin-down electrons (default 5).
--t FLOAT           nearest-neighbour hopping t (default 1.0), H has -t c^dag c.
--tp FLOAT          next-nearest (diagonal) hopping t' (default -0.3), H has
                    -t' c^dag c on diagonal bonds. Must be nonzero for the
                    Hard Core unless the filling is away from half filling
                    anyway (it is by default).
--U FLOAT           on-site repulsion U (default 8.0).
--ansatz STR        variational family. Valid values:
                      "nqs"  Jastrow + Ha amplitude units + Hb backflow units
                             (default),
                      "foil" Jastrow-Slater only (forces Ha = Hb = 0).
--ha INT            number of amplitude hidden units (default 4; ignored for foil).
--hb INT            number of backflow hidden units (default 1; ignored for foil).
--methods STR       comma-separated optimizers to run. Valid entries: "sr",
                    "osbh" (default "sr,osbh").
--n-mf INT          number of mean-field-init seeds (default 3).
--n-rand INT        number of random-init seeds (default 3).
--steps INT         total SR-step budget per run (default 800).
--steps-pre INT     plain SR steps before the first OSBH cycle (default 300).
                    Must be < --steps.
--relax-steps INT   SR steps used to relax each OSBH proposal (default 15).
--window INT        number of occupied and virtual one-body directions per
                    spin considered for swaps (default 2). Proposals per cycle
                    = 3 * window^2 (up, dn, and both spins).
--eta FLOAT         SR imaginary-time step (default 0.03).
--diag-shift FLOAT  relative diagonal shift of the SR matrix (default 1e-3);
                    an absolute shift of 1e-4 is always added.
--max-step FLOAT    clip on the parameter-update norm per step (default 0.5).
--seed INT          base random seed (default 1234).
--out PATH          JSON output file (default results_<instance>_<ansatz>.json
                    next to this script).
--check             run self-checks only (U=0 Slater energy vs ED, two-site
                    Hubbard energy vs closed form, finite-difference check of
                    the log-derivative matrix) and exit.
--quiet             print only the summary.

Example
-------
    python3 micro_test.py --check --lx 3 --ly 2 --nup 2 --ndn 2
    python3 micro_test.py --lx 3 --ly 2 --nup 2 --ndn 2 --steps 200 \
        --steps-pre 80 --n-mf 1 --n-rand 1
    python3 micro_test.py            # main instance 4x3 PBC, 5+5, t'=-0.3, U=8
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------
def parse_args(argv):
    p = argparse.ArgumentParser(description="P1.3 wrong-sign-basin test")
    p.add_argument("--lx", type=int, default=4)
    p.add_argument("--ly", type=int, default=3)
    p.add_argument("--bc", type=str, default="pbc",
                   choices=["pbc", "obc", "pbcx", "pbcy"])
    p.add_argument("--nup", type=int, default=5)
    p.add_argument("--ndn", type=int, default=5)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--tp", type=float, default=-0.3)
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--ansatz", type=str, default="nqs", choices=["nqs", "foil"])
    p.add_argument("--ha", type=int, default=4)
    p.add_argument("--hb", type=int, default=1)
    p.add_argument("--methods", type=str, default="sr,osbh")
    p.add_argument("--n-mf", type=int, default=3)
    p.add_argument("--n-rand", type=int, default=3)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--steps-pre", type=int, default=300)
    p.add_argument("--relax-steps", type=int, default=15)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--eta", type=float, default=0.03)
    p.add_argument("--diag-shift", type=float, default=1e-3)
    p.add_argument("--max-step", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--check", action="store_true")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ("sr", "osbh"):
            p.error(f"invalid method {m!r}; valid: sr, osbh")
    a.methods = methods
    if a.steps_pre >= a.steps:
        p.error("--steps-pre must be < --steps")
    ns = a.lx * a.ly
    if not (0 < a.nup <= ns and 0 < a.ndn <= ns):
        p.error("need 0 < nup, ndn <= lx*ly")
    if a.ansatz == "foil":
        a.ha, a.hb = 0, 0
    return a


# ----------------------------------------------------------------------------
# Lattice and Hamiltonian
# ----------------------------------------------------------------------------
def lattice_bonds(lx, ly, bc):
    px = bc in ("pbc", "pbcx")
    py = bc in ("pbc", "pbcy")

    def site(x, y):
        return x + lx * y

    nn, nnn = set(), set()
    for x in range(lx):
        for y in range(ly):
            i = site(x, y)
            for dx, dy, store in ((1, 0, nn), (0, 1, nn), (1, 1, nnn), (1, -1, nnn)):
                xx, yy = x + dx, y + dy
                if not (0 <= xx < lx):
                    if not px:
                        continue
                    xx %= lx
                if not (0 <= yy < ly):
                    if not py:
                        continue
                    yy %= ly
                j = site(xx, yy)
                if i != j:
                    store.add((min(i, j), max(i, j)))
    nnn -= nn  # tiny lattices: a pair cannot be both
    return sorted(nn), sorted(nnn)


def hopping_matrix(ns, nn, nnn, t, tp):
    T = np.zeros((ns, ns))
    for i, j in nn:
        T[i, j] = T[j, i] = -t
    for i, j in nnn:
        T[i, j] = T[j, i] = -tp
    return T


def sector_configs(ns, n):
    confs = []
    for occ in itertools.combinations(range(ns), n):
        b = 0
        for i in occ:
            b |= 1 << i
        confs.append(b)
    confs = np.array(sorted(confs), dtype=np.int64)
    occ = np.array([[i for i in range(ns) if (b >> i) & 1] for b in confs],
                   dtype=np.int64)
    return confs, occ


def one_species_hop(confs, T):
    """Sparse matrix of sum_ij T_ij c^dag_i c_j in a fixed-N sector with
    creation strings ordered by increasing site index."""
    ns = T.shape[0]
    index = {int(b): k for k, b in enumerate(confs)}
    rows, cols, vals = [], [], []
    pairs = [(i, j) for i in range(ns) for j in range(ns) if i != j and T[i, j] != 0]
    for k, b in enumerate(confs):
        b = int(b)
        for i, j in pairs:
            if (b >> j) & 1 and not (b >> i) & 1:
                lo, hi = min(i, j), max(i, j)
                between = bin(b & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))).count("1")
                nb = b ^ (1 << i) ^ (1 << j)
                rows.append(index[nb])
                cols.append(k)
                vals.append(T[i, j] * (-1) ** between)
    n = len(confs)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


class Model:
    def __init__(self, lx, ly, bc, nup, ndn, t, tp, U):
        self.lx, self.ly, self.ns = lx, ly, lx * ly
        self.nup, self.ndn = nup, ndn
        nn, nnn = lattice_bonds(lx, ly, bc)
        self.T = hopping_matrix(self.ns, nn, nnn, t, tp)
        self.cu, self.occ_u = sector_configs(self.ns, nup)
        self.cd, self.occ_d = sector_configs(self.ns, ndn)
        nu, nd = len(self.cu), len(self.cd)
        self.dim = nu * nd
        Tu = one_species_hop(self.cu, self.T)
        Td = one_species_hop(self.cd, self.T)
        bits_u = ((self.cu[:, None] >> np.arange(self.ns)) & 1).astype(float)
        bits_d = ((self.cd[:, None] >> np.arange(self.ns)) & 1).astype(float)
        dbl = bits_u @ bits_d.T  # (nu, nd) number of doubly occupied sites
        self.diag = U * dbl.reshape(-1)
        Hoff = sp.kron(Tu, sp.identity(nd)) + sp.kron(sp.identity(nu), Td)
        Hoff = Hoff.tocoo()
        self.off_r, self.off_c, self.off_v = Hoff.row, Hoff.col, Hoff.data
        self.H = (Hoff + sp.diags(self.diag)).tocsr()
        # full configuration arrays: index = iu * nd + id
        self.iu = np.repeat(np.arange(nu), nd)
        self.id = np.tile(np.arange(nd), nu)
        self.X = np.concatenate([bits_u[self.iu], bits_d[self.id]], axis=1)
        self.bits_u, self.bits_d = bits_u, bits_d
        self.coords = np.array([(i % lx, i // lx) for i in range(self.ns)])
        self.bc = bc

    def ground_state(self):
        if self.dim <= 400:
            w, v = np.linalg.eigh(self.H.toarray())
            return w[0], v[:, 0]
        w, v = spla.eigsh(self.H, k=1, which="SA", tol=1e-10)
        return w[0], v[:, 0]

    def fixed_node_energy(self, s, amp):
        """Lattice fixed-node energy (ten Haaf et al., PRB 51, 13039 (1995)) for
        the guiding function g(n) = s(n) * amp(n), restricted to the support
        amp(n) > 1e-10 * max(amp). Off-diagonal H_nm with H_nm g_n g_m > 0
        (sign-violating) are removed and H_nm g_m / g_n is added to H_nn.
        With amp = |psi_0| the result depends only on the sign s, is >= E_0,
        and equals E_0 iff s agrees with sgn psi_0 on the support."""
        s = np.where(s >= 0, 1.0, -1.0)
        supp = amp > 1e-10 * amp.max()
        newidx = -np.ones(self.dim, dtype=np.int64)
        newidx[supp] = np.arange(supp.sum())
        r, c, v = self.off_r, self.off_c, self.off_v
        keep_s = supp[r] & supp[c]
        r, c, v = r[keep_s], c[keep_s], v[keep_s]
        g = s * amp
        viol = v * g[r] * g[c] > 0
        nsup = int(supp.sum())
        diag = self.diag[supp] + np.bincount(newidx[r[viol]],
                                             weights=v[viol] * g[c[viol]] / g[r[viol]],
                                             minlength=nsup)
        keep = ~viol
        Hfn = sp.csr_matrix((v[keep], (newidx[r[keep]], newidx[c[keep]])),
                            shape=(nsup, nsup)) + sp.diags(diag)
        if nsup <= 400:
            return np.linalg.eigvalsh(Hfn.toarray())[0]
        w = spla.eigsh(Hfn, k=1, which="SA", tol=1e-10, return_eigenvectors=False)
        return w[0]


# ----------------------------------------------------------------------------
# Ansatz
# ----------------------------------------------------------------------------
def jastrow_features(model):
    """Distance-class two-body Jastrow features F (dim x Nc): for each class c
    of (same-spin / opposite-spin, |dx|, |dy|) pairs, F_c(n) = sum over pairs
    (a<=b) in class of x_a x_b; opposite-spin onsite pair = double occupancy."""
    ns, lx, ly = model.ns, model.lx, model.ly
    px = model.bc in ("pbc", "pbcx")
    py = model.bc in ("pbc", "pbcy")
    classes = {}
    for a in range(2 * ns):
        for b in range(a, 2 * ns):
            ia, ib = a % ns, b % ns
            same = (a < ns) == (b < ns)
            if same and ia == ib:
                continue  # n^2 = n, absorbed in normalisation
            dx = abs(model.coords[ia, 0] - model.coords[ib, 0])
            dy = abs(model.coords[ia, 1] - model.coords[ib, 1])
            if px:
                dx = min(dx, lx - dx)
            if py:
                dy = min(dy, ly - dy)
            key = (same, min(dx, dy), max(dx, dy))
            classes.setdefault(key, []).append((a, b))
    keys = sorted(classes)
    X = model.X
    F = np.zeros((model.dim, len(keys)))
    for c, k in enumerate(keys):
        for a, b in classes[k]:
            F[:, c] += X[:, a] * X[:, b]
    return F


class NQS:
    def __init__(self, model, ha, hb):
        self.m = model
        self.ha, self.hb = ha, hb
        self.F = jastrow_features(model)
        ns, nu, nd = model.ns, model.nup, model.ndn
        self.shapes = [("v", (self.F.shape[1],)),
                       ("a", (ha,)), ("w", (ha, 2 * ns)), ("b", (ha,)),
                       ("P0u", (ns, nu)), ("P0d", (ns, nd)),
                       ("Wb", (hb, 2 * ns)), ("cb", (hb,)),
                       ("Bu", (hb, ns, nu)), ("Bd", (hb, ns, nd))]
        self.sizes = [int(np.prod(s)) for _, s in self.shapes]
        self.npar = sum(self.sizes)

    def unpack(self, th):
        out, k = {}, 0
        for (name, shp), sz in zip(self.shapes, self.sizes):
            out[name] = th[k:k + sz].reshape(shp)
            k += sz
        return out

    def pack(self, d):
        return np.concatenate([np.asarray(d[n], float).reshape(-1) for n, _ in self.shapes])

    def init(self, kind, rng):
        m = self.m
        ns = m.ns
        d = {}
        d["v"] = np.zeros(self.F.shape[1])
        d["a"] = 0.05 * rng.standard_normal(self.ha)
        d["w"] = 0.05 * rng.standard_normal((self.ha, 2 * ns))
        d["b"] = 0.05 * rng.standard_normal(self.ha)
        if kind == "mf":
            e, U = np.linalg.eigh(m.T)
            d["P0u"] = U[:, :m.nup] + 1e-3 * rng.standard_normal((ns, m.nup))
            d["P0d"] = U[:, :m.ndn] + 1e-3 * rng.standard_normal((ns, m.ndn))
        elif kind == "rand":
            d["P0u"] = rng.standard_normal((ns, m.nup)) / np.sqrt(ns)
            d["P0d"] = rng.standard_normal((ns, m.ndn)) / np.sqrt(ns)
        else:
            raise ValueError(kind)
        d["Wb"] = 0.1 * rng.standard_normal((self.hb, 2 * ns))
        d["cb"] = 0.1 * rng.standard_normal(self.hb)
        d["Bu"] = 1e-3 * rng.standard_normal((self.hb, ns, m.nup))
        d["Bd"] = 1e-3 * rng.standard_normal((self.hb, ns, m.ndn))
        return self.pack(d)

    def _spin_part(self, P0, B, tb, occ_idx, occ_table, nconf_species, want_grad):
        """Return sign, logabs det and M (dim x ns x n) for one spin species."""
        m = self.m
        n = P0.shape[1]
        if self.hb == 0:
            occ = occ_table  # (n_species, n)
            A = P0[occ]  # (n_species, n, n)
            sgn, lad = np.linalg.slogdet(A)
            if not want_grad:
                return sgn[occ_idx], lad[occ_idx], None
            G = np.linalg.inv(A)  # (n_s, n, n): G[alpha, k]
            Ms = np.zeros((len(occ_table), m.ns, n))
            Ms[np.arange(len(occ_table))[:, None], occ, :] = np.transpose(G, (0, 2, 1))
            return sgn[occ_idx], lad[occ_idx], Ms[occ_idx]
        Phi = P0[None] + np.einsum("dh,hin->din", tb, B)  # (dim, ns, n)
        occ = occ_table[occ_idx]  # (dim, n)
        A = np.take_along_axis(Phi, occ[:, :, None], axis=1)  # (dim, n, n)
        sgn, lad = np.linalg.slogdet(A)
        if not want_grad:
            return sgn, lad, None
        G = np.linalg.inv(A)
        M = np.zeros((m.dim, m.ns, n))
        M[np.arange(m.dim)[:, None], occ, :] = np.transpose(G, (0, 2, 1))
        return sgn, lad, M

    def evaluate(self, th, want_grad=True):
        """Return (psi normalised, O log-derivative matrix (dim x npar) or None)."""
        m, d = self.m, self.unpack(th)
        X = m.X
        logpsi = self.F @ d["v"]
        if self.ha:
            theta = X @ d["w"].T + d["b"]
            logpsi = logpsi + np.logaddexp(theta, -theta) @ d["a"]
        tb = np.tanh(X @ d["Wb"].T + d["cb"]) if self.hb else None
        su, lu, Mu = self._spin_part(d["P0u"], d["Bu"], tb, m.iu, m.occ_u, len(m.cu), want_grad)
        sd, ld, Md = self._spin_part(d["P0d"], d["Bd"], tb, m.id, m.occ_d, len(m.cd), want_grad)
        logpsi = logpsi + lu + ld
        sgn = su * sd
        psi = sgn * np.exp(logpsi - logpsi[np.isfinite(logpsi)].max())
        psi[~np.isfinite(logpsi)] = 0.0
        psi /= np.linalg.norm(psi)
        if not want_grad:
            return psi, None
        cols = [self.F]
        if self.ha:
            tt = np.tanh(theta)
            cols += [np.logaddexp(theta, -theta),
                     ((tt * d["a"])[:, :, None] * X[:, None, :]).reshape(m.dim, -1),
                     tt * d["a"]]
        cols += [Mu.reshape(m.dim, -1), Md.reshape(m.dim, -1)]
        if self.hb:
            g = (1 - tb ** 2) * (np.einsum("din,hin->dh", Mu, d["Bu"])
                                 + np.einsum("din,hin->dh", Md, d["Bd"]))
            cols += [(g[:, :, None] * X[:, None, :]).reshape(m.dim, -1), g,
                     (tb[:, :, None, None] * Mu[:, None]).reshape(m.dim, -1),
                     (tb[:, :, None, None] * Md[:, None]).reshape(m.dim, -1)]
        O = np.concatenate(cols, axis=1)
        return psi, O


# ----------------------------------------------------------------------------
# Optimisation
# ----------------------------------------------------------------------------
def energy(model, psi):
    return float(psi @ (model.H @ psi))


def sr_step(model, ans, th, args):
    psi, O = ans.evaluate(th, True)
    Hpsi = model.H @ psi
    E = float(psi @ Hpsi)
    p = psi ** 2
    Ob = O - p @ O
    S = Ob.T @ (p[:, None] * Ob)
    F = Ob.T @ (psi * Hpsi - E * p)
    S += args.diag_shift * np.diag(np.diag(S)) + 1e-4 * np.eye(len(th))
    try:
        dth = np.linalg.solve(S, F)
    except np.linalg.LinAlgError:
        dth = np.linalg.lstsq(S, F, rcond=None)[0]
    step = args.eta * dth
    nrm = np.linalg.norm(step)
    if nrm > args.max_step:
        step *= args.max_step / nrm
    return th - step, E


def run_sr(model, ans, th, nsteps, args):
    E = None
    for _ in range(nsteps):
        th, E = sr_step(model, ans, th, args)
    psi, _ = ans.evaluate(th, False)
    return th, energy(model, psi)


def swap_proposals(model, ans, th, window):
    """One-body occupied/virtual swaps of the reference orbitals Phi0."""
    d = ans.unpack(th)
    out = []
    per_spin = {}
    for key in ("P0u", "P0d"):
        P0 = d[key]
        n = P0.shape[1]
        Q, R = np.linalg.qr(P0)
        Pfull = np.linalg.qr(np.concatenate([Q, np.eye(model.ns)], axis=1))[0]
        Pv = Pfull[:, n:]  # orthonormal complement (ns x ns-n)
        eo, Vo = np.linalg.eigh(Q.T @ model.T @ Q)
        ev, Vv = np.linalg.eigh(Pv.T @ model.T @ Pv)
        Qr = Q @ Vo  # occupied directions, ascending one-body energy
        coeff = Vo.T @ R  # P0 = Qr @ coeff
        Pr = Pv @ Vv
        props = []
        for a in range(max(0, n - window), n):
            for b in range(min(window, Pr.shape[1])):
                Qn = Qr.copy()
                Qn[:, a] = Pr[:, b]
                props.append((a, b, Qn @ coeff))
        per_spin[key] = props
    for key in ("P0u", "P0d"):
        for a, b, newP in per_spin[key]:
            dd = {k: v.copy() for k, v in d.items()}
            dd[key] = newP
            out.append((f"{key}:{a}->{b}", ans.pack(dd)))
    for (a, b, nu_), (a2, b2, nd_) in zip(per_spin["P0u"], per_spin["P0d"]):
        dd = {k: v.copy() for k, v in d.items()}
        dd["P0u"], dd["P0d"] = nu_, nd_
        out.append((f"both:{a}->{b}", ans.pack(dd)))
    return out


def run_osbh(model, ans, th, args, log):
    budget = args.steps
    th, E = run_sr(model, ans, th, args.steps_pre, args)
    used = args.steps_pre
    nacc = 0
    while True:
        props = swap_proposals(model, ans, th, args.window)
        cost = (len(props) + 1) * args.relax_steps
        if used + cost > budget:
            break
        th_c, E_c = run_sr(model, ans, th, args.relax_steps, args)
        best = (E_c, th_c, "stay")
        for label, thp in props:
            thr, Er = run_sr(model, ans, thp, args.relax_steps, args)
            if np.isfinite(Er) and Er < best[0]:
                best = (Er, thr, label)
        used += cost
        th, E = best[1], best[0]
        if best[2] != "stay":
            nacc += 1
        log(f"    osbh cycle: used={used} E={E:.6f} move={best[2]}")
    th, E = run_sr(model, ans, th, budget - used, args)
    return th, E, nacc


def diagnostics(model, psi, psi0, E0):
    E = energy(model, psi)
    ov = abs(psi @ psi0)
    sgn = np.sign(psi)
    agree = (psi0 ** 2) @ (sgn * np.sign(psi0))
    efn = model.fixed_node_energy(sgn, np.abs(psi0))
    g = sgn * np.abs(psi0)
    e_sign = float(g @ (model.H @ g))
    return dict(E=E, rel_err=(E - E0) / abs(E0), overlap=ov,
                sign_agree=abs(float(agree)), E_FN=float(efn), E_sign=e_sign,
                eps_s=float((efn - E0) / abs(E0)))


# ----------------------------------------------------------------------------
# Self checks
# ----------------------------------------------------------------------------
def self_checks(args):
    ok = True
    # (1) two-site Hubbard, 1+1 electrons: E0 = (U - sqrt(U^2 + 16 t^2))/2
    m2 = Model(2, 1, "obc", 1, 1, 1.0, 0.0, 4.0)
    e2, _ = m2.ground_state()
    ex = (4.0 - np.sqrt(16.0 + 16.0)) / 2
    print(f"[check] two-site Hubbard U=4: ED={e2:.10f} exact={ex:.10f}")
    ok &= abs(e2 - ex) < 1e-9
    # (2) U=0 Slater determinant energy equals ED and free-fermion sum
    m0 = Model(args.lx, args.ly, args.bc, args.nup, args.ndn, args.t, args.tp, 0.0)
    e0, _ = m0.ground_state()
    eps = np.linalg.eigvalsh(m0.T)
    efree = eps[:args.nup].sum() + eps[:args.ndn].sum()
    ans0 = NQS(m0, 0, 0)
    th = ans0.init("mf", np.random.default_rng(0))
    d = ans0.unpack(th)
    _, U = np.linalg.eigh(m0.T)
    d["P0u"], d["P0d"] = U[:, :args.nup], U[:, :args.ndn]
    psi, _ = ans0.evaluate(ans0.pack(d), False)
    es = energy(m0, psi)
    print(f"[check] U=0 {args.lx}x{args.ly} {args.bc} {args.nup}+{args.ndn}: "
          f"ED={e0:.10f} free={efree:.10f} Slater={es:.10f}")
    shell_deg = (args.nup < m0.ns and abs(eps[args.nup] - eps[args.nup - 1]) < 1e-9) or \
                (args.ndn < m0.ns and abs(eps[args.ndn] - eps[args.ndn - 1]) < 1e-9)
    ok &= abs(e0 - efree) < 1e-8
    if not shell_deg:
        ok &= abs(es - efree) < 1e-8
    elif abs(es - efree) > 1e-8:
        print("[check] note: open shell at U=0, Slater energy test skipped")
    # (3) fixed-node energy with the exact sign equals E0 (interacting)
    m = Model(args.lx, args.ly, args.bc, args.nup, args.ndn, args.t, args.tp, args.U)
    E0, psi0 = m.ground_state()
    efn = m.fixed_node_energy(np.sign(psi0), np.abs(psi0))
    efn_mf = m.fixed_node_energy(np.sign(psi), np.abs(psi0))
    print(f"[check] U={args.U}: E0={E0:.10f}  E_FN[exact sign]={efn:.10f}  "
          f"E_FN[U=0 Slater sign]={efn_mf:.10f}")
    ok &= abs(efn - E0) < 1e-7 and efn_mf >= E0 - 1e-9
    # (4) finite-difference check of O = d log psi / d theta (nqs, hb>0)
    ans = NQS(m, 2, 1)
    rng = np.random.default_rng(1)
    th = ans.init("mf", rng)
    th = th + 0.1 * rng.standard_normal(th.shape)
    psi_c, O = ans.evaluate(th, True)
    wmask = psi_c ** 2 > 1e-4 * (psi_c ** 2).max()  # away from nodes

    def logpsi(t_):
        d_ = ans.unpack(t_)
        X = m.X
        lp = ans.F @ d_["v"]
        theta = X @ d_["w"].T + d_["b"]
        lp = lp + np.logaddexp(theta, -theta) @ d_["a"]
        tb = np.tanh(X @ d_["Wb"].T + d_["cb"])
        _, lu, _ = ans._spin_part(d_["P0u"], d_["Bu"], tb, m.iu, m.occ_u, 0, False)
        _, ld, _ = ans._spin_part(d_["P0d"], d_["Bd"], tb, m.id, m.occ_d, 0, False)
        return lp + lu + ld

    idx = rng.choice(len(th), size=min(25, len(th)), replace=False)
    err = 0.0
    for k in idx:
        e = np.zeros_like(th)
        e[k] = 1e-7
        fd = (logpsi(th + e) - logpsi(th - e)) / 2e-7
        err = max(err, np.max(np.abs(fd - O[:, k])[wmask]))
    print(f"[check] max |O - finite difference| over 25 params = {err:.2e}")
    ok &= err < 1e-4
    print("[check] ALL OK" if ok else "[check] SOME CHECK FAILED")
    return ok


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def verdict(res, methods):
    def med(method, init):
        v = [r for r in res if r["method"] == method and r["init"] == init]
        return (float(np.median([r["eps_s"] for r in v])),
                float(np.median([r["E"] for r in v]))) if v else (None, None)
    eps_sr, e_sr = med("sr", "mf")
    eps_os, e_os = med("osbh", "mf")
    eps_best = float(min(r["eps_s"] for r in res))
    out = dict(eps_SR=eps_sr, eps_OS=eps_os, eps_best=eps_best,
               E_SR_median=e_sr, E_OS_median=e_os)
    if eps_sr is None or eps_os is None:
        out["verdict"] = "INCOMPLETE (need both sr and osbh)"
        return out
    G = eps_sr - eps_best
    out["G"] = G
    out["M"] = (eps_sr - eps_os) / G if G > 0 else None
    if G <= 0.10 * eps_sr:
        out["verdict"] = "FAIL (no basin: mean-field-init plain SR already reaches best sign)"
    elif G > 0.25 * eps_sr and out["M"] >= 0.5 and e_os <= e_sr:
        out["verdict"] = "PASS"
    elif G > 0.25 * eps_sr and out["M"] < 0.2:
        out["verdict"] = "FAIL (basin exists, OSBH does not escape it)"
    else:
        out["verdict"] = "AMBIGUOUS (rerun with --tp +0.3 and doubled seeds; kill if not PASS)"
    return out


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    log = (lambda s: None) if args.quiet else (lambda s: print(s, flush=True))
    if args.check:
        sys.exit(0 if self_checks(args) else 1)
    t0 = time.time()
    model = Model(args.lx, args.ly, args.bc, args.nup, args.ndn, args.t, args.tp, args.U)
    E0, psi0 = model.ground_state()
    ans = NQS(model, args.ha, args.hb)
    log(f"instance {args.lx}x{args.ly} {args.bc} N=({args.nup},{args.ndn}) t'={args.tp} "
        f"U={args.U}: dim={model.dim} E0={E0:.8f} npar={ans.npar} ansatz={args.ansatz}")
    res = []
    inits = [("mf", k) for k in range(args.n_mf)] + [("rand", k) for k in range(args.n_rand)]
    for kind, k in inits:
        for method in args.methods:
            rng = np.random.default_rng(args.seed + 1000 * k + (0 if kind == "mf" else 500))
            th = ans.init(kind, rng)
            ts = time.time()
            nacc = 0
            if method == "sr":
                th, _ = run_sr(model, ans, th, args.steps, args)
            else:
                th, _, nacc = run_osbh(model, ans, th, args, log)
            psi, _ = ans.evaluate(th, False)
            dg = diagnostics(model, psi, psi0, E0)
            dg.update(init=kind, seed=k, method=method, accepted_swaps=nacc,
                      seconds=time.time() - ts)
            res.append(dg)
            log(f"  {kind}{k} {method:5s}: E={dg['E']:.6f} relE={dg['rel_err']:.3e} "
                f"ov={dg['overlap']:.4f} signagree={dg['sign_agree']:.4f} "
                f"eps_s={dg['eps_s']:.3e} swaps={nacc} ({dg['seconds']:.0f}s)")
    v = verdict(res, args.methods)
    print("SUMMARY " + json.dumps(v))
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"results_{args.lx}x{args.ly}_{args.bc}_{args.nup}_{args.ndn}_tp{args.tp}_U{args.U}_{args.ansatz}.json")
    with open(out, "w") as f:
        json.dump(dict(args=vars(args), E0=E0, dim=model.dim, npar=ans.npar,
                       runs=res, summary=v, wall=time.time() - t0), f, indent=1)
    log(f"wrote {out} (wall {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
