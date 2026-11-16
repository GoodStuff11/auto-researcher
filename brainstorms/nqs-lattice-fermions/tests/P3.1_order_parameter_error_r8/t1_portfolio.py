#!/usr/bin/env python3
"""
P3.1 round 8 (Stage 3.5 portfolio): tier-1 toy computations, exact linear algebra only
=====================================================================================

What it computes (all exact, no sampling, no training):
  Instances: three doped OBC t-t'-J clusters at the same electron density n = 2/3 (doping 1/3),
  t = 1, t' = -0.25, J = 0.4 (the parameters of the shared 4x4 OBC 6+6 instance):
     3x2 (2+2, D = 90), 3x3 (3+3, D = 1680), 4x3 (4+4, D = 34650); 6x2 (4+4, D = 34650) is a reported
     second 12-site geometry (added after the census, before any K_half was computed).
  psi_0 = ED ground state, sector = its space-group irrep chi and total spin S_0.
  Observables (Hermitian, symmetric):
     pair / spin / charge : r5_core definitions (long-range real-space aggregates), with the cut-offs
                            scaled to the cluster: R_PAIR = 0.82 x max bond-centre distance,
                            R_SITE = 0.82 x max site distance (0.82 reproduces the 4x4 cut-offs 3.0/3.5).
     pairSF               : d-wave pair structure factor (1/V) sum_{b,b'} eta_b eta_b' Delta_b^dag Delta_b'.
     spinSF / chargeSF    : (2/V) sum_{i<j} (-1)^{dx+dy} S_i.S_j  and the same with (n_i-nbar)(n_j-nbar).
  Trial states (error models, fixed before any K_half was seen):
     psi = N[psi_0 + lam e + mu j],  e = -(H - E_0)^{-1} Q_0 V psi_0 inside the sector (first-order
     ground state of H + V: the "variational state = ground state of a slightly wrong Hamiltonian"
     model), lam such that |<psi_0|psi_0 + lam e>|/|.| = 0.94 (the seed-11 overlap), and j high-omega
     junk (orthogonal to psi_0 and to the N_LOW lowest levels) with mu such that the junk carries
     2x the energy error of lam e (real state: F_E(1t) = 0.31).
     V in: tp (t' -> t'+0.1), J (J -> J+0.2), dens_s{0,1}, spin_s{0,1}, pair_s{0,1}: random
     short-range symmetric density-density, S.S and d-wave pair-hopping couplings
     g_c = N(0,1) exp(-(d_c - 1)) per distance class c (seeds 0, 1).  8 trial states per size.
  Arms (Ritz of H, exact matrices; delta_hat = a(lowest Ritz vector) - a(psi); ratio = delta_hat/delta):
     L1R      span{psi, (H-E)psi}                                      (saturated control)
     ORC(K)   span{psi, (H-E)psi, |1..K>}  exact in-sector levels      (ESR0 ceiling, energy order)
     RWR*     span{psi, (H-E)psi, y*_pair, y*_spin, y*_charge}         y*_A = (H-E_0)^{-1} Q_0 (A-a_0) psi_0
     RWR      span{psi, (H-E)psi, y_pair, y_spin, y_charge}            y_A from psi only:
              y_A = (Q (H - s) Q)^{-1} Q (A - a_psi) psi, Q = 1 - |psi><psi|, s = E_psi - 2 (E_psi - E_L1R)
     RWR-R / RWR-H : RWR with injected training error per response,
              y~ = unit(y) + eta_low g + eta_high j_y, g a random unit vector in span{|1..N_LOW>},
              j_y unit high-omega junk; grade R: (0.3, 0.5); grade H: (0.5, 1.0).
     RWR1     span{psi, (H-E)psi, y_A} for the scored observable only (grade 0).
  K_half(obs, state) = smallest K such that ORC(K') ratio is in [1/2, 2] for every K' in [K, N_LOW];
  "inf" if ORC(N_LOW) is outside. N_LOW = min(60, sector dimension - 1).
  Scored item: |delta| >= 0.10 |a_0|. Tracked: ratio in [1/2, 2]. (Exact arms: no abstention.)

Pre-registered rules (written in the leaf, Round 8 tier log, before --mode main was run):
  G4  (ESR0 Gate 4, per real-space observable): med = median K_half over scored states (inf counts
      as larger than any number). FAIL if med(4x3) >= 2 med(3x2) or med(4x3) = inf; PASS if
      med(4x3) <= med(3x2) + 1; UNCLEAR otherwise.
  RWR (route 8.1): KILL if RWR* tracks < 0.9 of scored items (lemma check) or RWR-R tracks < 0.5
      overall or tracked(RWR-R, 4x3) < tracked(RWR-R, 3x2) - 0.25. PASS if RWR* >= 0.9 and RWR-R
      >= 0.75 overall and tracked(RWR-R, 4x3) >= tracked(RWR-R, 3x2) - 0.1 and
      tracked(RWR-R, 4x3) - tracked(ORC(3), 4x3) >= 0.2. UNCLEAR otherwise.
  SF  (route 8.3, structure-factor targeting): PASS if for every SF observable med K_half <= 3 at every
      size and med(4x3) <= med(3x2) + 1; KILL if the G4 FAIL condition holds for 2 of 3 SF observables,
      or med K_half(SF) >= med K_half(real-space partner) at 4x3 for 2 of 3 pairs; UNCLEAR otherwise.

Usage
  python3 t1_portfolio.py --mode census            # sizes, sector dims, low levels (no K_half)
  python3 t1_portfolio.py --mode main --out results/t1_main.json
Options
  --mode {census, main}   census: build each instance, print D, sector dimension (or number of levels
                          found), the lowest omegas and #levels with omega <= 1t; main: everything above.
  --sizes LIST            comma list from {3x2, 3x3, 6x2, 4x3} (default all four; the rules compare
                          the first and the last size of the list).
  --n-low INT             cap on N_LOW (default 60).
  --out PATH              JSON output for --mode main (default results/t1_main.json; "none" = no file).
Runs on <= 4 threads (OMP/OPENBLAS/MKL_NUM_THREADS set to 4 below). Pure numpy/scipy.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as sla  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(HERE, "..", "P3.1_order_parameter_error_r5")
sys.path.insert(0, R5)
import r5_core as rc  # noqa: E402
from p11_core import solve_ed  # noqa: E402
from tj_core import TJ, popcount  # noqa: E402,F401

SIZES = {"3x2": dict(Lx=3, Ly=2, nup=2, ndn=2), "3x3": dict(Lx=3, Ly=3, nup=3, ndn=3),
         "6x2": dict(Lx=6, Ly=2, nup=4, ndn=4), "4x3": dict(Lx=4, Ly=3, nup=4, ndn=4)}
TP, JJ = -0.25, 0.4
REAL = ("pair", "spin", "charge")
SF = ("pairSF", "spinSF", "chargeSF")
ALL = REAL + SF
GRADES = {"0": (0.0, 0.0), "R": (0.3, 0.5), "H": (0.5, 1.0)}
OVERLAP, JUNK_RATIO, SCORE_REL = 0.94, 2.0, 0.10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unit(v):
    return v / np.linalg.norm(v)


# ============================================================================ structure factors
def sf_operators(m, ob):
    """(M_sf, A_spinSF, d_chargeSF): d-wave pair SF bond matrix (1/V) eta eta'; (2/V) sum_{i<j} (-1)^{dx+dy}
    S_i.S_j (sparse); the same for (n_i - nbar)(n_j - nbar) (diagonal vector)."""
    V = m.V
    eta = np.array([1.0 if b[2] == "x" else -1.0 for b in m.pair_bonds])
    M_sf = np.outer(eta, eta) / V
    nu = np.array([(m.bu >> i) & 1 for i in range(V)], dtype=float)
    nd = np.array([(m.bd >> i) & 1 for i in range(V)], dtype=float)
    szs = 0.5 * (nu - nd)
    plus, minus = [], []
    dg_s = np.zeros(m.D); dg_c = np.zeros(m.D)
    for i in range(V):
        for j in range(i + 1, V):
            dx, dy = (m.coords[i] - m.coords[j]).astype(int)
            s = 1.0 if (abs(dx) + abs(dy)) % 2 == 0 else -1.0
            (plus if s > 0 else minus).append((i, j))
            dg_s += s * szs[i] * szs[j]
            dg_c += s * (ob.n_site_occ[i] - ob.nbar) * (ob.n_site_occ[j] - ob.nbar)
    ex = ob._exchange(plus) - (ob._exchange(minus) if minus else 0)
    return M_sf, ((sp.diags(dg_s) + ex) * (2.0 / V)).tocsr(), dg_c * (2.0 / V)


def apply_obs(ob, sf, name, psi):
    if name in REAL:
        return ob.apply(name, psi)
    if name == "pairSF":
        vs = np.array([Dop @ psi for Dop in ob.D_ops])
        u = sf[0] @ vs
        out = np.zeros_like(psi)
        for b, Dop in enumerate(ob.D_ops):
            out += Dop.T @ u[b]
        return out
    if name == "spinSF":
        return sf[1] @ psi
    if name == "chargeSF":
        return sf[2] * psi
    raise KeyError(name)


# ============================================================================ instance
class Inst:
    def __init__(self, name, n_low_cap):
        t0 = time.time()
        c = SIZES[name]
        self.name = name
        m = TJ(c["Lx"], c["Ly"], "obc", "obc", c["nup"], c["ndn"], 1.0, TP, JJ)
        m.spin_setup()
        self.m = m
        W, Vv, deg = solve_ed(m, 4, 1e-8, rc.UHF_SEED)
        assert deg == 1, f"{name}: degenerate ground level {W}"
        self.E0 = float(W[0])
        self.psi0 = unit(Vv[:, 0])
        self.sg, self.gen = rc.build_symmetry(m, self.psi0)
        self.S0 = float(round(0.5 * (np.sqrt(1 + 4 * m.S2_expect(self.psi0)) - 1)))
        # observables with size-scaled cut-offs
        dsite = max(np.hypot(*(m.coords[i] - m.coords[j])) for i in range(m.V) for j in range(m.V))
        m.pair_setup(0.0)
        cen = [m.bond_center(b) for b in m.pair_bonds]
        dbond = max(np.hypot(*(a - b)) for a in cen for b in cen)
        rc.R_PAIR, rc.R_SITE = 0.82 * dbond, 0.82 * dsite
        self.ob = rc.Observables(m)
        self._build_sf(cen)
        self.levels(n_low_cap)
        log(f"{name}: D={m.D} sector levels found {self.n_found} (N_LOW={self.n_low}) E0={self.E0:.6f} "
            f"chi={self.gen} S0={self.S0:g} R_PAIR={rc.R_PAIR:.2f} ({self.ob.n_pair} pairs) "
            f"R_SITE={rc.R_SITE:.2f} ({self.ob.n_site} pairs) ({time.time() - t0:.1f}s)")

    def _build_sf(self, cen):
        self.sf = sf_operators(self.m, self.ob)

    def apply(self, name, psi):
        return apply_obs(self.ob, self.sf, name, psi)

    def two_mode_ops(self):
        """Callables v -> O_m^dag O_m v for O in (rho, S, Delta_d) and the 8 lowest OBC standing waves."""
        if hasattr(self, "_tm"):
            return self._tm
        m, ob = self.m, self.ob
        modes = [(a, b) for a in range(3) for b in range(3) if (a, b) != (0, 0)]

        def f(mode, xy):
            return (math.cos(math.pi * mode[0] * (xy[0] + 0.5) / m.Lx)
                    * math.cos(math.pi * mode[1] * (xy[1] + 0.5) / m.Ly))
        nu = np.array([(m.bu >> i) & 1 for i in range(m.V)], dtype=float)
        nd = np.array([(m.bd >> i) & 1 for i in range(m.V)], dtype=float)
        szs = 0.5 * (nu - nd)
        eta = np.array([1.0 if b[2] == "x" else -1.0 for b in m.pair_bonds])
        cen = [m.bond_center(b) for b in m.pair_bonds]
        ex1 = {(i, j): ob._exchange([(i, j)]) for i in range(m.V) for j in range(i + 1, m.V)}
        ops = []
        for mode in modes:
            fs = np.array([f(mode, m.coords[i]) for i in range(m.V)])
            rho = fs @ (ob.n_site_occ - ob.nbar)
            ops.append(lambda v, d=rho ** 2: d * v)
            dg = (fs @ szs) ** 2
            Ssum = sp.diags(dg) + sum(2 * fs[i] * fs[j] * E for (i, j), E in ex1.items()) \
                + sp.diags(sum(fs[i] ** 2 * (0.75 * (ob.n_site_occ[i] % 2) - szs[i] ** 2) for i in range(m.V)))
            ops.append(lambda v, S=Ssum.tocsr(): S @ v)
            fb = eta * np.array([f(mode, c) for c in cen])
            Mb = np.outer(fb, fb)
            ops.append(lambda v, Mb=Mb: ob.apply_pieces_all("pair", v, [Mb])[0])
        self._tm = ops
        return ops

    def proj(self, v):
        return self.m.spin_project(self.sg.project(v), self.S0)

    def levels(self, cap):
        m = self.m
        if m.D <= 2000:
            Pm = np.column_stack([self.proj(np.eye(m.D)[:, i]) for i in range(m.D)])
            Pm = 0.5 * (Pm + Pm.T)
            lam, Ub = np.linalg.eigh(Pm)
            B = Ub[:, lam > 0.5]
            Hs = B.T @ (m.H @ B)
            e, X = np.linalg.eigh(0.5 * (Hs + Hs.T))
            U = B @ X
            self.dim_sector = B.shape[1]
        else:
            k = cap + 2

            def PH(v):
                x = self.proj(v)
                return self.proj(m.H @ x) + 50.0 * (v - x)   # out-of-sector pushed up to +50
            L = sla.LinearOperator((m.D, m.D), matvec=PH, dtype=float)
            v0 = self.proj(np.random.default_rng(3).standard_normal(m.D))
            e, U = sla.eigsh(L, k=k, which="SA", v0=v0, tol=1e-10)
            o = np.argsort(e); e, U = e[o], U[:, o]
            self.dim_sector = None
        assert abs(e[0] - self.E0) < 1e-7, (e[0], self.E0)
        for n in range(min(len(e), cap + 1)):
            u = unit(U[:, n]); U[:, n] = u
            assert np.linalg.norm(m.H @ u - e[n] * u) < 1e-6 and np.linalg.norm(self.proj(u) - u) < 1e-8
        self.n_found = len(e)
        self.n_low = min(cap, len(e) - 1)
        if U[:, 0] @ self.psi0 < 0:
            U[:, 0] *= -1
        assert abs(U[:, 0] @ self.psi0) > 1 - 1e-8
        self.om = e[1:self.n_low + 1] - e[0]
        self.U = U[:, 1:self.n_low + 1]

    # ------------------------------------------------------------- linear solves inside the sector
    def solve(self, rhs, shift, orth=None):
        """x = (P (H - shift) P)^{-1} rhs on the sector (optionally inside the complement of orth)."""
        m = self.m

        def q(v):
            v = self.proj(v)
            if orth is not None:
                v = v - orth * (orth @ v)
            return v

        def op(v):
            x = q(v)
            return q(m.H @ x - shift * x) + (v - x)
        L = sla.LinearOperator((m.D, m.D), matvec=op, dtype=float)
        x, info = sla.minres(L, q(rhs), rtol=1e-11, maxiter=20000)
        res = np.linalg.norm(op(x) - q(rhs)) / np.linalg.norm(q(rhs))
        assert res < 1e-6, ("minres", info, res)
        return q(x)

    def junk(self, rng, base):
        x = self.proj(base * rng.standard_normal(self.m.D))
        j = self.m.H @ x - (base @ (self.m.H @ base)) * x
        j = self.proj(j)
        j -= self.U @ (self.U.T @ j)
        j -= self.psi0 * (self.psi0 @ j)
        return unit(j)


# ============================================================================ error models
def perturbations(I, rng_seed=0):
    m, ob = I.m, I.ob
    out = {}
    for lab, kw in (("tp", dict(tp=TP + 0.1, J=JJ)), ("J", dict(tp=TP, J=JJ + 0.2))):
        m2 = TJ(m.Lx, m.Ly, m.bcx, m.bcy, m.nup, m.ndn, 1.0, kw["tp"], kw["J"])
        dH = (m2.H - m.H).tocsr()
        out[lab] = lambda v, dH=dH: dH @ v
    dists = {k: np.array(v) for k, v in ob.piece_dists.items()}
    for s in (0, 1):
        rng = np.random.default_rng(100 + s)
        for kind, key in (("dens", "charge"), ("spin", "spin"), ("pair", "pair")):
            d = dists[key]
            g = rng.standard_normal(len(d)) * np.exp(-(np.maximum(d, 1.0) - 1.0))
            g[d < 0.5] = 0.0 if key != "pair" else g[d < 0.5]
            pcs = ob.pieces[key]
            if key == "pair":
                Mg = sum(gi * P for gi, P in zip(g, pcs))
                out[f"{kind}_s{s}"] = lambda v, Mg=Mg: ob.apply_pieces_all("pair", v, [Mg])[0]
            elif key == "spin":
                Sg = sum(gi * P for gi, P in zip(g, pcs)).tocsr()
                out[f"{kind}_s{s}"] = lambda v, Sg=Sg: Sg @ v
            else:
                dg = sum(gi * P for gi, P in zip(g, pcs))
                out[f"{kind}_s{s}"] = lambda v, dg=dg: dg * v
    return out


def trial_state(I, Vop, rng):
    psi0, E0 = I.psi0, I.E0
    w = I.proj(Vop(psi0))
    w -= psi0 * (psi0 @ w)
    e = -I.solve(w, E0, orth=psi0)
    e = unit(e)
    lam = math.sqrt(1.0 / OVERLAP ** 2 - 1.0)
    Ee = e @ (I.m.H @ e) - E0
    j = I.junk(rng, psi0)
    Ej = j @ (I.m.H @ j) - E0
    mu = math.sqrt(JUNK_RATIO * lam ** 2 * Ee / Ej)
    return unit(psi0 + lam * e + mu * j)


# ============================================================================ Ritz
def ritz(I, vecs, names):
    """vecs[0] = psi. Returns {obs: a(Ritz) - a(psi)} and the Ritz energy."""
    B = np.array([unit(v) for v in vecs])
    HB = np.array([I.m.H @ b for b in B])
    S = B @ B.T; H = B @ HB.T
    S = 0.5 * (S + S.T); H = 0.5 * (H + H.T)
    lam, V = np.linalg.eigh(S)
    keep = lam > 1e-11 * lam.max()
    X = V[:, keep] / np.sqrt(lam[keep])
    ev, Y = np.linalg.eigh(X.T @ H @ X)
    c = X @ Y[:, 0]
    phi = unit(c @ B)
    out = {}
    for k in names:
        out[k] = float(phi @ I.apply(k, phi) - B[0] @ I.apply(k, B[0]))
    return out, float(ev[0])


class Oracle:
    """Precomputed matrices for span{psi, (H-E)psi, |1..N_LOW>}; ORC(K) for every K cheaply."""

    def __init__(self, I, psi, names):
        r = I.m.H @ psi - (psi @ (I.m.H @ psi)) * psi
        B = np.vstack([psi, unit(r), I.U.T])
        self.B = B
        self.S = B @ B.T
        self.H = B @ np.array([I.m.H @ b for b in B]).T
        self.A = {k: B @ np.array([I.apply(k, b) for b in B]).T for k in names}
        self.a_psi = {k: float(psi @ I.apply(k, psi)) for k in names}

    def shift(self, K, k, perm=None):
        idx = np.arange(K + 2) if perm is None else np.concatenate([[0, 1], 2 + np.asarray(perm[:K], int)])
        n = len(idx)
        S = self.S[np.ix_(idx, idx)]; H = self.H[np.ix_(idx, idx)]
        lam, V = np.linalg.eigh(0.5 * (S + S.T))
        keep = lam > 1e-11 * lam.max()
        X = V[:, keep] / np.sqrt(lam[keep])
        ev, Y = np.linalg.eigh(X.T @ (0.5 * (H + H.T)) @ X)
        c = X @ Y[:, 0]
        A = self.A[k][np.ix_(idx, idx)]
        return float(c @ A @ c / (c @ S @ c) - self.a_psi[k])


def k_half(ratios):
    ok = [(0.5 <= r <= 2.0) for r in ratios]   # ratios[K-1] for K = 1..N_LOW
    if not ok[-1]:
        return math.inf
    K = len(ok)
    while K - 1 >= 1 and ok[K - 2]:
        K -= 1
    return K


# ============================================================================ main
def run_instance(I, rng):
    a0 = {k: float(I.psi0 @ I.apply(k, I.psi0)) for k in ALL}
    # exact responses of psi_0
    ystar = {}
    for k in REAL:
        w = I.apply(k, I.psi0) - a0[k] * I.psi0
        ystar[k] = I.solve(w, I.E0, orth=I.psi0)
    rows = []
    for lab, Vop in perturbations(I).items():
        psi = trial_state(I, Vop, rng)
        if psi @ I.psi0 < 0:
            psi = -psi
        Epsi = float(psi @ (I.m.H @ psi))
        r = I.m.H @ psi - Epsi * psi
        _, E_L1 = ritz(I, [psi, r], [])
        s = Epsi - 2.0 * (Epsi - E_L1)
        orc = Oracle(I, psi, ALL)
        delta = {k: a0[k] - orc.a_psi[k] for k in ALL}
        c = I.U.T @ psi; alpha = float(I.psi0 @ psi)
        d1 = {k: float(-2 * alpha * np.sum(c * (I.U.T @ (I.apply(k, I.psi0) - a0[k] * I.psi0))))
              for k in ALL}
        row = dict(size=I.name, state=lab, overlap=alpha, dE=Epsi - I.E0, dE_L1=E_L1 - I.E0,
                   shift_below_E0=bool(s < I.E0), a0=a0, delta=delta, delta1_window=d1)
        ratios = {k: [orc.shift(K, k) / delta[k] for K in range(1, I.n_low + 1)] for k in ALL}
        row["Khalf"] = {k: k_half(ratios[k]) for k in ALL}
        row["ORC3"] = {k: ratios[k][min(3, I.n_low) - 1] for k in ALL}
        row["KhalfA"] = {}
        for k in REAL:   # A-ordered window (route 8.4): key |<n|(A - a_psi) psi>|^2 / omega_n from psi
            wpsi = I.U.T @ (I.apply(k, psi) - orc.a_psi[k] * psi)
            perm = np.argsort(-(wpsi ** 2) / I.om)
            rA = [orc.shift(K, k, perm) / delta[k] for K in range(1, I.n_low + 1)]
            row["KhalfA"][k] = k_half(rA)
        row["ORC6"] = {k: ratios[k][min(6, I.n_low) - 1] for k in ALL}
        row["ORCmax"] = {k: ratios[k][-1] for k in ALL}
        l1, _ = ritz(I, [psi, r], REAL)
        row["L1R"] = {k: l1[k] / delta[k] for k in REAL}
        # responses from psi only
        y = {}
        for k in REAL:
            w = I.apply(k, psi) - orc.a_psi[k] * psi
            y[k] = unit(I.solve(w, s, orth=psi))
        rs, _ = ritz(I, [psi, r] + [ystar[k] for k in REAL], REAL)
        row["RWRstar"] = {k: rs[k] / delta[k] for k in REAL}
        for g, (el, eh) in GRADES.items():
            ys = []
            for k in REAL:
                gl = unit(I.U @ rng.standard_normal(I.n_low))
                jy = I.junk(rng, unit(psi))
                ys.append(unit(y[k] + el * gl + eh * jy))
            rr, _ = ritz(I, [psi, r] + ys, REAL)
            row[f"RWR_{g}"] = {k: rr[k] / delta[k] for k in REAL}
        tm = []
        for op in I.two_mode_ops():
            v = op(psi); v = I.proj(v - (psi @ v) * psi)
            if np.linalg.norm(v) > 1e-10:
                tm.append(v)
        rt, _ = ritz(I, [psi, r] + tm, REAL)
        row["TMW"] = {k: rt[k] / delta[k] for k in REAL}
        row["RWR1"] = {}
        for k in REAL:
            r1, _ = ritz(I, [psi, r, y[k]], [k])
            row["RWR1"][k] = r1[k] / delta[k]
        # route 8.8 (RWR on the concentrated structure factors), separate random stream
        rng2 = np.random.default_rng(11 + len(rows))
        ysf = {}
        for k in ("pairSF", "chargeSF"):
            w = I.apply(k, psi) - orc.a_psi[k] * psi
            yk = unit(I.solve(w, s, orth=psi))
            ysf[k] = unit(yk + 0.3 * unit(I.U @ rng2.standard_normal(I.n_low)) + 0.5 * I.junk(rng2, unit(psi)))
        rsf, _ = ritz(I, [psi, r] + [ysf[k] for k in ("pairSF", "chargeSF")], ("pairSF", "chargeSF"))
        row["RWRSF_R"] = {k: rsf[k] / delta[k] for k in ("pairSF", "chargeSF")}
        row["scored"] = {k: bool(abs(delta[k]) >= SCORE_REL * abs(a0[k])) for k in ALL}
        rows.append(row)
        log(f"{I.name} {lab:7s} ov={alpha:.3f} dE={Epsi - I.E0:.4f} dE_L1={E_L1 - I.E0:.4f} s<E0={s < I.E0} | "
            + " ".join(f"{k}:K={row['Khalf'][k]} d/a0={delta[k] / a0[k]:+.2f}{'' if row['scored'][k] else '(ns)'}"
                       for k in ALL))
        log("      ratios  " + " | ".join(
            f"{k}: L1R {row['L1R'][k]:+.2f} ORC3 {row['ORC3'][k]:+.2f} RWR* {row['RWRstar'][k]:+.2f} "
            f"RWR0 {row['RWR_0'][k]:+.2f} R {row['RWR_R'][k]:+.2f} H {row['RWR_H'][k]:+.2f} RWR1 {row['RWR1'][k]:+.2f}"
            for k in REAL))
    return rows


def med(xs):
    if not xs:
        return float("nan")
    return float(np.median(np.array(xs, dtype=float)))


def frac(rows, arm, sizes=None):
    it = [(r, k) for r in rows for k in REAL if r["scored"][k] and (sizes is None or r["size"] in sizes)]
    if not it:
        return float("nan"), 0
    t = sum(1 for r, k in it if 0.5 <= r[arm][k] <= 2.0)
    return t / len(it), len(it)


def verdicts(rows, sizes):
    lo, hi = sizes[0], sizes[-1]
    out = {}
    print("--- K_half medians over scored states (inf = not reached by N_LOW)")
    mk = {}
    for k in ALL:
        mk[k] = {s: med([r["Khalf"][k] for r in rows if r["size"] == s and r["scored"][k]]) for s in sizes}
        print(f"  {k:9s} " + "  ".join(f"{s}: {mk[k][s]}" for s in sizes))

    def g4(k):
        a, b = mk[k][lo], mk[k][hi]
        if not np.isfinite(b) or (np.isfinite(a) and b >= 2 * a):
            return "FAIL"
        if np.isfinite(a) and b <= a + 1:
            return "PASS"
        return "UNCLEAR"
    for k in REAL:
        out[f"G4_{k}"] = g4(k)
    mA = {k: {s: med([r["KhalfA"][k] for r in rows if r["size"] == s and r["scored"][k]]) for s in sizes}
          for k in REAL}
    for k in REAL:
        print(f"  {k + '-A':9s} " + "  ".join(f"{s}: {mA[k][s]}" for s in sizes))
    npass = sum(1 for k in REAL if all(mA[k][s] <= 3 for s in sizes) and mA[k][hi] <= mA[k][lo] + 1)
    nkill = sum(1 for k in REAL if (not np.isfinite(mA[k][hi])) or mA[k][hi] >= 2 * mA[k][lo]
                or mA[k][hi] >= mk[k][hi])
    out["AOW"] = "PASS" if npass >= 2 else ("KILL" if nkill >= 2 else "UNCLEAR")
    print(f"AOW (route 8.4): pass-observables {npass}/3, kill-observables {nkill}/3 -> {out['AOW']}")
    print("G4 (ESR0 Gate 4): " + "  ".join(f"{k} {out[f'G4_{k}']}" for k in REAL))
    # RWR
    fs, ns = frac(rows, "RWRstar")
    fR, nR = frac(rows, "RWR_R")
    fRlo, _ = frac(rows, "RWR_R", [lo]); fRhi, _ = frac(rows, "RWR_R", [hi])
    fO3hi, _ = frac(rows, "ORC3", [hi])
    for arm in ("L1R", "ORC3", "ORC6", "ORCmax", "RWRstar", "RWR_0", "RWR_R", "RWR_H", "RWR1", "TMW"):
        print(f"  tracked {arm:8s} all {frac(rows, arm)[0]:.2f} (n={frac(rows, arm)[1]})  "
              + "  ".join(f"{s} {frac(rows, arm, [s])[0]:.2f}" for s in sizes))
    if fs < 0.9 or fR < 0.5 or fRhi < fRlo - 0.25:
        v = "KILL"
    elif fs >= 0.9 and fR >= 0.75 and fRhi >= fRlo - 0.1 and fRhi - fO3hi >= 0.2:
        v = "PASS"
    else:
        v = "UNCLEAR"
    out["RWR"] = v
    print(f"RWR (route 8.1): RWR* {fs:.2f} (n={ns}); RWR-R {fR:.2f} (n={nR}); RWR-R {lo} {fRlo:.2f} {hi} {fRhi:.2f}; "
          f"ORC3 {hi} {fO3hi:.2f} -> {v}")
    fT, _ = frac(rows, "TMW"); fTlo, _ = frac(rows, "TMW", [lo]); fThi, _ = frac(rows, "TMW", [hi])
    fL1hi, _ = frac(rows, "L1R", [hi])
    if fT < 0.5 or fThi < fTlo - 0.25 or fThi < fL1hi + 0.2:
        v = "KILL"
    elif fT >= 0.75 and fThi >= fTlo - 0.1 and fThi >= fO3hi + 0.2:
        v = "PASS"
    else:
        v = "UNCLEAR"
    out["TMW"] = v
    print(f"TMW (route 8.5): all {fT:.2f}; {lo} {fTlo:.2f} {hi} {fThi:.2f}; ORC3 {hi} {fO3hi:.2f} L1R {hi} {fL1hi:.2f} -> {v}")
    it = lambda ss: [(r, k) for r in rows for k in ("pairSF", "chargeSF") if r["scored"][k] and r["size"] in ss]
    def fsf(ss):
        x = it(ss)
        return (sum(1 for r, k in x if 0.5 <= r["RWRSF_R"][k] <= 2.0) / len(x) if x else float("nan")), len(x)
    fa, na = fsf(sizes); fl, nl_ = fsf([lo]); fh, nh = fsf([hi])
    v = "KILL" if (fa < 0.5 or fh < fl - 0.25) else ("PASS" if fa >= 0.75 and fh >= fl - 0.1 else "UNCLEAR")
    out["RWRSF"] = v
    print(f"RWR-SF (route 8.8): all {fa:.2f} (n={na}); {lo} {fl:.2f} (n={nl_}) {hi} {fh:.2f} (n={nh}); "
          + " ".join(f"{s_} {fsf([s_])[0]:.2f}" for s_ in sizes) + f" -> {v}")
    # SF
    g4sf = [g4(k) for k in SF]
    worse = sum(1 for kr, ks in zip(REAL, SF) if mk[ks][hi] >= mk[kr][hi])
    small = all((mk[k][s] <= 3) for k in SF for s in sizes) and all(mk[k][hi] <= mk[k][lo] + 1 for k in SF)
    if g4sf.count("FAIL") >= 2 or worse >= 2:
        v = "KILL"
    elif small:
        v = "PASS"
    else:
        v = "UNCLEAR"
    out["SF"] = v
    print(f"SF (route 8.3): G4 per SF obs {g4sf}; SF not smaller than partner at {hi}: {worse}/3 -> {v}")
    return out, mk


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("census", "main"), required=True)
    p.add_argument("--sizes", default="3x2,3x3,6x2,4x3")
    p.add_argument("--n-low", type=int, default=60)
    p.add_argument("--out", default=os.path.join(HERE, "results", "t1_main.json"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    sizes = args.sizes.split(",")
    rows, cen = [], {}
    for s in sizes:
        I = Inst(s, args.n_low)
        n1 = int(np.sum(I.om <= 1.0))
        cen[s] = dict(D=I.m.D, dim_sector=I.dim_sector, n_low=I.n_low, omega=I.om[:10].tolist(), n_le_1t=n1)
        log(f"census {s}: sector dim {I.dim_sector} lowest omegas {np.round(I.om[:10], 3).tolist()} "
            f"#levels(omega<=1t) {n1}{'+' if n1 == I.n_low else ''}")
        if args.mode == "main":
            rows += run_instance(I, np.random.default_rng(7))
    if args.mode == "main":
        out, mk = verdicts(rows, sizes)
        if args.out != "none":
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(dict(census=cen, rows=rows, verdicts=out,
                               Khalf_median={k: v for k, v in mk.items()}), f, indent=1, default=str)
    log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
