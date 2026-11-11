#!/usr/bin/env python3
"""
P4.1 round-3 decisive test: structurally exact-SU(2) fermionic ansaetze at two determinants per
amplitude versus a SPIN- AND SPACE-GROUP-PROJECTED spin-resolved NNBF trained with projection.
=================================================================================================

Shared physics core (lattice, Hubbard sector, ED helpers, observables, P1.1 sign budget,
NNBF / AGP / SCF arms, L-BFGS driver) is r2_micro_test.py in this folder: a verbatim copy of
tests/P4.1_cheap_su2_r2/micro_test.py. Nothing is imported across folders.

Instance (default) : 4x3 CYLINDER (periodic along x, open along y; --bc-y obc), Hubbard,
                     N_up = N_dn = 4, U = 8, t' = -0.25, dim 245025 (the lower-symmetry geometry
                     P1.2 round 3 uses). Chosen because on the 4x3 TORUS the projected NNBF control
                     is within ~1e-5 of E_0 (P1.1-P interim), so accuracy cannot discriminate there.
                     SIGN HALF OF THE HARD CORE ONLY (no same-sector competitor is claimed).
                     --bc-y pbc reproduces the round-2 torus.
Symmetry           : every arm is trained as psi = P_G P_S phi or P_G phi, where
                     P_G = (1/|G|) sum_g chi(g) U_g over the space group (x-translations x
                     {1, Mx, My, MxMy}; |G| = 16 on the 4x3 cylinder, 48 on the 4x3 torus),
                     chi = psi_0's real 1-D irrep (psi_0 from a symmetry-free Lanczos of
                     H + 5 S^2, irrep read and asserted), and P_S = exact Loewdin projector onto S = 0.

Arms (--ansatz)
---------------
  sqrvb  ROUTE 3.4 "spin-charge factorised RVB with charge-conditioned spinon pairing":
         psi(x) = s_ord(x) e^{J(c)} det Psi_c[occ2(c)] * (-1)^{inv(sigma)} det A_c[U_s, D_s]
         c = charge configuration, occ2(c) = doubled spinless charge orbitals (slot a of site i
         if n_i >= 1, slot b if n_i = 2), Psi_c = Psi0 + reshape(Vc h(c)) (2L x N_e),
         A_c = sym(A0 + reshape(Va h(c))) (L x L symmetric spinon pairing), h(c) =
         tanh(W feat(c) + b) a charge-only MLP (doublon and holon indicators), U_s / D_s =
         up / down singly occupied sites, inv(sigma) = #{i<j: i down-single, j up-single},
         s_ord = site-major -> up-then-down reordering sign. EXACT S = 0 FOR EVERY THETA
         (charge factor x singlet AGP of the singles), no projection quadrature.
         PER-AMPLITUDE COST: one N_e x N_e determinant + one (N_s/2) x (N_s/2) determinant.
  sqrvb0 ABLATION of sqrvb: Va = 0 (spinon pairing A0 independent of the charge
         configuration; the uncoupled slave-particle product). Parameter-matched by width.
  scf    ROUTE 3.5 = round-2 SCF (charge determinant x charge-conditioned SU(2)-symmetric MPS
         on the lattice snake, identity on holes / doublons), unchanged except budget; the
         round-2 "stall" was a 150-iteration artefact (diag_scf_stall.py:
         eps 7e-3 at 150 it -> 1.8e-15 at 370 it). Cost O(L D^2) + one N_e determinant.
  nnbf_SG PROJECTED CONTROL (brief rule 1): spin-resolved Luo-Clark NNBF trained as
         P_G P_S phi_nnbf. P_S is applied to the full Hilbert vector (oracle); its
         per-amplitude cost in VMC is an orbit sum exponential in N_s. Matched parameters.
  nnbf_G spin-resolved NNBF trained as P_G phi_nnbf (S^z only). Reported with post-hoc P_S.
  che    EXACT-S PROJECTED CONTROL: the same NNBF architecture with CHARGE-ONLY input (doublon /
         holon indicators replace the +-1 spin occupations; identical parameter count),
         trained as P_G P_S phi. This is the family of Che, arXiv:2608.22925 (spin-scalar
         backflow features + Gauss-Legendre spin projection with LU reuse). Training uses the
         full-vector P_S, which is the SAME state as Che's per-amplitude quadrature; the
         per-amplitude Che algorithm is implemented in che_single() and checked against it.
  foil   mVMC-like Minimal Contrast Foil: P_G P_S e^{J(c)} det f[up, dn], f general (native
         size, not matched).
Every arm shares J(c) (charge Jastrow + Gutzwiller + doublon-holon factor).

Usage
-----
  python micro_test.py --mode MODE [options]

--mode {smoke, time, cost, run, report}       (required)
    smoke  : <= 2 min self-test on a 2x3 torus (N_e = 6, U = 6, t' = -0.3; code check only):
             (a) space-group projector idempotent, symmetric, commutes with H, P_S commutes
             with P_G; (b) <S^2> = 0 at random parameters for sqrvb, sqrvb0, scf, che, foil,
             nnbf_SG; (c) analytic gradient vs central finite differences for every arm;
             (d) per-amplitude single-configuration functions sqrvb_single and che_single
             agree with the batched / full-vector states; (e) E_sign bracket on psi_0 = E0;
             (f) a 300-iteration optimisation of every arm (code runs; not evidence).
    time   : build the production instance, time ONE energy + gradient of every arm, print
             parameter counts (used for the sbatch budget).
    cost   : per-amplitude wall-time benchmark (single configurations, random parameters,
             no training) on Lx x Lx tori, Lx in --cost-sizes, N_e = 2*round(7L/16):
             sqrvb_single (2 dets), scf_single (MPS chain + 1 det), che_single (Che's LU-reused
             Gauss-Legendre spin projection, N_q = ceil((N_e/2+1)/2) nodes), quad_single (naive
             quadrature, N_q full N_e x N_e determinants). Writes <outdir>/cost.json (used by
             report for gate g_cost).
    run    : optimise ONE arm (--ansatz) with one --seed; then observables, P1.1 sign budget,
             post-hoc projections, sampled local-energy diagnostic; writes
             <outdir>/<tag>_<ansatz>_seed<seed>.json (skips if present unless --overwrite).
    report : aggregate all JSONs for --tag; MEDIANS over seeds (>= 3 required per arm,
             otherwise the route is AMBIGUOUS); prints the table and the verdict.

Lattice / model options:
  --Lx INT (4)  --Ly INT (3)  lattice, both >= 2; x is always periodic
  --bc-y {obc, pbc} (obc)  boundary along y: obc = cylinder (default), pbc = torus
  --Ne INT (8)      even, S^z = 0, target S = 0
  --U FLOAT (8.0)   on-site repulsion (t = 1)
  --tp FLOAT (-0.25) diagonal hopping t'
Ansatz / optimisation options:
  --ansatz {sqrvb, sqrvb0, scf, nnbf_SG, nnbf_G, che, foil}   (run; default sqrvb)
  --nnbf-hidden INT (16)  width of the NNBF control; its parameter count is the target every
                          matched arm (sqrvb, sqrvb0, scf, che) is matched to
  --mps-mult STR ("2,2,1,1")  scf virtual multiplicities m_0, m_1/2, ... (D = sum m_k (k+1))
  --seed INT (0)          seeds 0, 1, 2 in production; seed >= 1 uses a UHF start where the arm
                          has one (nnbf, che, foil)
  --maxiter INT (2000)    L-BFGS iteration cap, SAME for every arm (budget matched by iterations)
  --max-hours FLOAT (20)  wall-clock safety cap, same for every arm; a run that hits it is
                          flagged "capped" and report treats the arm as AMBIGUOUS for budget
  --chunk INT (16384)     configurations per chunk
Sign budget (as P1.1): --lbfgs-iters INT (4000)  --fn-iters INT (10)  --fn-floor FLOAT (1e-3)
  --no-sign  skip it (debug only)
Sampled-estimator diagnostic: --n-samples INT (20000) draws x ~ |psi|^2 (exact distribution);
  reports std(E_loc) and the tail ratio q_99.9(|E_loc - E|)/std. Not a gate (no route needs
  cross-overlaps or amplitude ratios beyond standard VMC local energies).
Cost benchmark: --cost-sizes STR ("6,8,12,16,20")  --cost-reps INT (60); slopes fitted over the
  three largest sizes
I/O: --outdir PATH (<script dir>/results)  --tag STR (auto)  --overwrite

Pre-committed decision rule (report; route R in {sqrvb, scf}; m(.) = median over seeds 0,1,2;
nSG = nnbf_SG):
  g1 exactness   : max_seeds |<S^2>_R| < 1e-10.
  g_cost         : t_R / t_che < 1 at the largest benchmarked size, and the log-log slope of
                   t_R in N_e over the three largest sizes is below that of t_che (cost.json).
  g_mech (sqrvb) : m(eps_sqrvb) <= 0.8 m(eps_sqrvb0) (the charge-conditioned spinon pairing is
                   what works); not applied to scf.
  g2 correlations: m(dCs_R) <= m(dCs_nSG) and m(dPd_R) <= m(dPd_nSG) + 1e-4.
  PASS      : g1, g_cost, g_mech, g2, m(eps_R) <= 0.8 m(eps_nSG),
              m(dE_sign_R) <= 0.8 m(dE_sign_nSG), m(eps_R) < m(eps_che), m(eps_R) <= 0.5 m(eps_foil).
  FAIL      : not g1, or not g_cost, or m(eps_R) >= m(eps_nSG), or m(eps_R) >= m(eps_che).
  UNINFORMATIVE (premise gate) : not FAIL and m(eps_nSG) < 1e-4 (the projected control leaves no
              measurable error, so accuracy cannot discriminate). Pre-committed action: rerun on a
              6x2 ladder (--Lx 6 --Ly 2 --bc-y obc --Ne 10) and report the cost claim alone.
  AMBIGUOUS : otherwise (or < 3 seeds, or a capped run). Pre-committed action: rerun the array on
              P1.1 instance 0 (--Ne 10, dim 627264); promote R only if it is not FAIL there and
              m(dE_sign_R) < m(dE_sign_nSG) on both instances; otherwise KILL R.
  Reported, not gating: che vs nnbf_SG (does charge-only conditioning lose against
  spin-resolved conditioning once both are projected?), nnbf_G before / after post-hoc P_S,
  seed spread (min, max) of every metric.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.special import roots_legendre

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_micro_test as R  # noqa: E402  (local verbatim copy of the round-2 core)

ARMS = ["sqrvb", "sqrvb0", "scf", "nnbf_SG", "nnbf_G", "che", "foil"]
ROUTES = ["sqrvb", "scf"]
SPROJ = {"sqrvb": False, "sqrvb0": False, "scf": False, "nnbf_SG": True, "nnbf_G": False,
         "che": True, "foil": True}
EXACT = {"sqrvb", "sqrvb0", "scf", "nnbf_SG", "che", "foil"}
PREMISE_EPS = 1e-4  # premise gate: m(eps_nnbf_SG) must be >= this for accuracy to discriminate
log = R.log


# ============================================================================ geometry
class CylLattice(R.Lattice):
    """Lx x Ly lattice, periodic along x; along y periodic (bc_y = "pbc", the round-2 torus) or
    open (bc_y = "obc": cylinder). Same attributes as R.Lattice (T, mask, nn, xy, shift);
    shift() always wraps (it is only used for displacement-averaged observables and for the
    x-translations of the space group)."""

    def __init__(self, Lx, Ly, tp, bc_y="obc"):
        self.Lx, self.Ly, self.L, self.bc_y = Lx, Ly, Lx * Ly, bc_y
        L = self.L
        self.T = np.zeros((L, L))
        for i in range(L):
            x, y = self.xy(i)
            for (dx, dy, amp) in [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                                  (1, 1, tp), (1, -1, tp), (-1, 1, tp), (-1, -1, tp)]:
                if bc_y == "obc" and not (0 <= y + dy < Ly):
                    continue
                j = self.shift(i, dx, dy)
                if j != i:
                    self.T[i, j] += amp
        self.T = 0.5 * (self.T + self.T.T)
        self.mask = np.zeros((L, L), bool)
        for i in range(L):
            x, y = self.xy(i)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if bc_y == "obc" and not (0 <= y + dy < Ly):
                        continue
                    self.mask[i, self.shift(i, dx, dy)] = True
        self.nn = [sorted({self.shift(i, dx, dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                           if not (bc_y == "obc" and not (0 <= self.xy(i)[1] + dy < Ly))} - {i})
                   for i in range(L)]


def make_sector(Lx, Ly, Ne, U, tp, bc_y):
    return R.Sector(CylLattice(Lx, Ly, tp, bc_y), Ne // 2, Ne // 2, U)


# ============================================================================ space group
def space_group(sec):
    """(name exponents, (iu, su, id, sd)) for all g = T^(a,b) M^(c,d); site i = x*Ly + y."""
    lat = sec.lat
    out = []
    for c in (0, 1):
        for d in (0, 1):
            for a in range(lat.Lx):
                for b in (range(lat.Ly) if getattr(lat, "bc_y", "pbc") == "pbc" else [0]):
                    p = np.empty(lat.L, dtype=np.int64)
                    for i in range(lat.L):
                        x, y = lat.xy(i)
                        x2 = (-x) % lat.Lx if c else x
                        if getattr(lat, "bc_y", "pbc") == "pbc":
                            y2 = (-y) % lat.Ly if d else y
                        else:
                            y2 = lat.Ly - 1 - y if d else y
                        p[i] = ((x2 + a) % lat.Lx) * lat.Ly + (y2 + b) % lat.Ly
                    iu, su = sec.up.perm(p)
                    idd, sd = sec.dn.perm(p)
                    out.append(((a, b, c, d), (iu, su, idd, sd)))
    return out


class SG:
    GEN = {(1, 0, 0, 0): "Tx", (0, 1, 0, 0): "Ty", (0, 0, 1, 0): "Mx", (0, 0, 0, 1): "My"}

    def __init__(self, sec):
        self.sec = sec
        self.el = space_group(sec)
        self.G = len(self.el)
        self.chi = np.ones(self.G)

    def apply(self, P, k):
        iu, su, idd, sd = self.el[k][1]
        out = np.zeros_like(P)
        out[np.ix_(iu, idd)] = (su[:, None] * sd[None, :]) * P
        return out

    def set_irrep(self, P0):
        nn = np.sum(P0 * P0)
        lam = {}
        for k, (e, _) in enumerate(self.el):
            if e in self.GEN:
                v = float(np.sum(P0 * self.apply(P0, k)) / nn)
                if abs(abs(v) - 1) > 1e-6:
                    raise RuntimeError(f"psi_0 not in a real 1-D irrep: {self.GEN[e]} -> {v}")
                lam[self.GEN[e]] = int(round(v))
        self.lam = lam
        lam = {k: lam.get(k, 1) for k in ("Tx", "Ty", "Mx", "My")}
        self.chi = np.array([lam["Tx"] ** a * lam["Ty"] ** b * lam["Mx"] ** c * lam["My"] ** d
                             for (a, b, c, d), _ in self.el], float)
        return self.lam

    def project(self, P):
        out = np.zeros_like(P)
        for k in range(self.G):
            out += self.chi[k] * self.apply(P, k)
        return out / self.G


def ground_state(sec, sg, mu=5.0):
    """psi_0: lowest state of H + mu S^2 in the whole fixed-(N_up, N_dn) space (no symmetry
    assumed; S > 0 states are pushed up by mu S(S+1)); its space-group irrep is then read from
    the generators (asserted real 1-D) and set in sg. Also returns the next level."""
    shp = sec.shape
    op = R.LinearOperator((sec.dim, sec.dim), dtype=float,
                          matvec=lambda x: (sec.H(x.reshape(shp)) + mu * sec.S2(x.reshape(shp))).ravel())
    w, v = R.eigsh(op, k=2, which="SA", v0=np.random.default_rng(1).standard_normal(sec.dim), tol=1e-11)
    o = np.argsort(w)
    out = []
    for kk in o:
        P = v[:, kk].reshape(shp)
        nrm = np.sum(P * P)
        out.append((P / np.sqrt(nrm), float(np.sum(P * sec.H(P)) / nrm), float(np.sum(P * sec.S2(P)) / nrm)))
    P0, E0, s20 = out[0]
    if abs(s20) > 1e-8 or out[1][1] + mu * out[1][2] - E0 < 1e-6:
        raise RuntimeError(f"psi_0 not a non-degenerate singlet: <S^2>={s20}, levels {w}")
    lam = sg.set_irrep(P0)
    return P0, E0, s20, lam, out[1]


# ============================================================================ new arm: SQ-RVB
class SQRVB(R.Base):
    """psi(x) = s_ord e^{J(c)} det Psi_c[occ2(c)] (-1)^inv det A_c[U_s, D_s] (module docstring)."""

    def __init__(self, sec, hidden, seed, couple=True):
        self.sec, self.couple = sec, couple
        lat = sec.lat
        L, Ne = lat.L, sec.Nu + sec.Nd
        self.L, self.Ne, self.H = L, Ne, hidden
        self.Fj, self.feat = R.charge_features(sec)
        n = sec.n_c.astype(int)
        self.occ2 = np.array([sorted([2 * i for i in range(L) if n[c, i] >= 1]
                                     + [2 * i + 1 for i in range(L) if n[c, i] == 2])
                              for c in range(sec.nC)], dtype=np.int64)
        bu = sec.up.bits.astype(np.int64)
        bd = sec.dn.bits.astype(np.int64)
        cu_gt = np.cumsum(bu[:, ::-1], axis=1)[:, ::-1] - bu
        # per basis state: order sign, spin-singles bookkeeping, grouped by m = N_s / 2
        self.sord = np.empty(sec.dim)
        self.inv = np.empty(sec.dim)
        self.m = np.empty(sec.dim, dtype=np.int64)
        Us, Ds = {}, {}
        for s in range(0, sec.dim, 65536):
            sl = slice(s, min(s + 65536, sec.dim))
            u, d = bu[sec.iu[sl]], bd[sec.id[sl]]
            par = np.sum(d * cu_gt[sec.iu[sl]], 1)
            self.sord[sl] = 1.0 - 2.0 * (par % 2)
            su = u * (1 - d)
            sdn = d * (1 - u)
            below_dn = np.cumsum(sdn, 1) - sdn  # down-singles at sites < j
            self.inv[sl] = 1.0 - 2.0 * (np.sum(su * below_dn, 1) % 2)
            self.m[sl] = su.sum(1)
        self.groups = {}
        for m in np.unique(self.m):
            idx = np.nonzero(self.m == m)[0]
            u, d = bu[sec.iu[idx]], bd[sec.id[idx]]
            su = u * (1 - d)
            sdn = d * (1 - u)
            U = np.nonzero(su)[1].reshape(len(idx), m) if m else np.zeros((len(idx), 0), np.int64)
            D = np.nonzero(sdn)[1].reshape(len(idx), m) if m else np.zeros((len(idx), 0), np.int64)
            self.groups[int(m)] = (idx, U, D)
        shapes = [("jv", (self.Fj.shape[1],)), ("Psi0", (2 * L, Ne)), ("A0", (L, L)),
                  ("W", (hidden, 2 * L)), ("b", (hidden,)), ("Vc", (hidden, 2 * L * Ne))]
        if couple:
            shapes.append(("Va", (hidden, L * L)))
        self.setup(shapes)
        rng = np.random.default_rng(seed)
        T2 = np.kron(lat.T, np.ones((2, 2)))
        _, U2 = np.linalg.eigh(-T2 + 1e-3 * np.diag(np.tile([0, 1], L)))
        _, Uh = np.linalg.eigh(-lat.T)
        A0 = Uh[:, :sec.Nd] @ Uh[:, :sec.Nd].T  # free-fermion RVB start
        jv = np.zeros(self.Fj.shape[1]); jv[L * (L - 1) // 2] = -1.0
        p = dict(jv=jv, Psi0=U2[:, :Ne] + 0.02 * rng.standard_normal((2 * L, Ne)),
                 A0=A0 + 0.02 * rng.standard_normal((L, L)), W=0.3 * rng.standard_normal((hidden, 2 * L)),
                 b=0.1 * rng.standard_normal(hidden), Vc=0.01 * rng.standard_normal((hidden, 2 * L * Ne)))
        if couple:
            p["Va"] = 0.01 * rng.standard_normal((hidden, L * L))
        self.theta0 = self.pack(p)

    @staticmethod
    def count(L, nf, Ne, H, couple=True):
        return nf + 2 * L * Ne + L * L + H * (2 * L + 1) + H * 2 * L * Ne + (H * L * L if couple else 0)

    def _fields(self, p):
        L, Ne, H = self.L, self.Ne, self.H
        hh = np.tanh(self.feat @ p["W"].T + p["b"])
        Psi = p["Psi0"][None] + (hh @ p["Vc"]).reshape(-1, 2 * L, Ne)
        M = np.broadcast_to(p["A0"], (hh.shape[0], L, L)).copy()
        if self.couple:
            M = M + (hh @ p["Va"]).reshape(-1, L, L)
        A = 0.5 * (M + M.transpose(0, 2, 1))
        return hh, Psi, A

    def forward(self, th, chunk):
        p = self.unpack(th)
        sec = self.sec
        J = self.jastrow(p)
        expJ = np.exp(J - J.max())
        hh, Psi, A = self._fields(p)
        Ac = np.take_along_axis(Psi, self.occ2[:, :, None], axis=1)  # (nC, Ne, Ne)
        detC = np.linalg.det(Ac)
        chi = np.empty(sec.dim)
        for m, (idx, U, D) in self.groups.items():
            c = sec.cidx[idx]
            if m == 0:
                chi[idx] = 1.0
                continue
            for s in range(0, len(idx), chunk):
                sl = slice(s, s + chunk)
                sub = A[c[sl][:, None, None], U[sl][:, :, None], D[sl][:, None, :]]
                chi[idx[sl]] = np.linalg.det(sub)
        chi *= self.inv
        phi = self.sord * expJ[sec.cidx] * detC[sec.cidx] * chi
        return phi.reshape(sec.shape), dict(p=p, phi=phi, chi=chi, expJ=expJ, Ac=Ac, detC=detC,
                                            hh=hh, A=A)

    def vjp(self, cc, vbar, chunk):
        sec, p, L, Ne = self.sec, cc["p"], self.L, self.Ne
        vb = vbar.ravel()
        g = {}
        gJ = np.bincount(sec.cidx, weights=vb * cc["phi"], minlength=sec.nC)
        g["jv"] = self.Fj.T @ gJ
        # charge determinant
        wC = np.bincount(sec.cidx, weights=vb * self.sord * cc["chi"], minlength=sec.nC) * cc["expJ"]
        Acbar = wC[:, None, None] * R.cofactor(cc["Ac"])
        Psibar = np.zeros((sec.nC, 2 * L, Ne))
        np.add.at(Psibar, (np.arange(sec.nC)[:, None], self.occ2), Acbar)
        # spinon determinant
        Abar = np.zeros((sec.nC, L, L))
        wS = vb * self.sord * cc["expJ"][sec.cidx] * cc["detC"][sec.cidx] * self.inv
        for m, (idx, U, D) in self.groups.items():
            if m == 0:
                continue
            c = sec.cidx[idx]
            for s in range(0, len(idx), chunk):
                sl = slice(s, s + chunk)
                sub = cc["A"][c[sl][:, None, None], U[sl][:, :, None], D[sl][:, None, :]]
                cof = R.cofactor(sub) * wS[idx[sl]][:, None, None]
                flat = (c[sl][:, None, None] * L * L + U[sl][:, :, None] * L + D[sl][:, None, :]).ravel()
                Abar += np.bincount(flat, weights=cof.ravel(), minlength=sec.nC * L * L).reshape(sec.nC, L, L)
        Mbar = 0.5 * (Abar + Abar.transpose(0, 2, 1))
        hh = cc["hh"]
        g["Psi0"] = Psibar.sum(0)
        g["Vc"] = hh.T @ Psibar.reshape(sec.nC, -1)
        hbar = Psibar.reshape(sec.nC, -1) @ p["Vc"].T
        g["A0"] = Mbar.sum(0)
        if self.couple:
            g["Va"] = hh.T @ Mbar.reshape(sec.nC, -1)
            hbar = hbar + Mbar.reshape(sec.nC, -1) @ p["Va"].T
        pre = hbar * (1 - hh ** 2)
        g["W"] = pre.T @ self.feat
        g["b"] = pre.sum(0)
        return self.pack(g)


# ============================================================================ charge-input NNBF (Che family)
class ChargeNNBF(R.NNBF):
    """NNBF with the +-1 spin occupations replaced by charge-only features (doublon, holon
    indicators, mapped to +-1); identical parameter count. phi = e^J det Phi_up(c)[up] det Phi_dn(c)[dn]."""

    def __init__(self, sec, hidden, seed):
        super().__init__(sec, hidden, seed)
        _, f = R.charge_features(sec)
        self.cfeat = 2.0 * f - 1.0

    def _orb(self, p, sl):
        sec, L, Nu, Nd = self.sec, self.L, self.Nu, self.Nd
        x = self.cfeat[sec.cidx[sl]]
        h = np.tanh(x @ p["W1"] + p["b1"])
        out = h @ p["W2"] + p["b2"] + p["P0"]
        B = x.shape[0]
        Ou = out[:, :L * Nu].reshape(B, L, Nu)
        Od = out[:, L * Nu:].reshape(B, L, Nd)
        ru = sec.up.occ[sec.iu[sl]]
        rd = sec.dn.occ[sec.id[sl]]
        Au = np.take_along_axis(Ou, ru[:, :, None], axis=1)
        Ad = np.take_along_axis(Od, rd[:, :, None], axis=1)
        return x, h, Au, Ad, ru, rd


# ============================================================================ per-amplitude algorithms
def gl_nodes(n_half):
    """Gauss-Legendre nodes in u = cos(beta) exact for the S = 0 projection of an S^z = 0 state
    whose spin is at most n_half: N_q = ceil((n_half + 1) / 2)."""
    nq = int(np.ceil((n_half + 1) / 2))
    return roots_legendre(nq)


def che_single(Phiu, Phid, U, D, nodes):
    """<x|P_{S=0}|Phi> for a charge-conditioned UHF determinant (orbitals Phiu (L,Nu), Phid (L,Nd)
    already evaluated at c(x)), x = (up sites U, down sites D), by Che's LU-reused quadrature:
    <x|R(beta)|Phi> = c^{N_e} det A det(Dm + t^2 C A^{-1} B), c = cos(beta/2), t = tan(beta/2),
    A = Phiu[U], B = Phid[U], C = Phiu[D], Dm = Phid[D]; P_{S=0} = (1/2) int_{-1}^{1} du."""
    u, w = nodes
    A, B, C, Dm = Phiu[U], Phid[U], Phiu[D], Phid[D]
    lu = np.linalg.slogdet(A)
    X = C @ np.linalg.solve(A, B)
    Ne = len(U) + len(D)
    sg, lg = np.empty(len(u)), np.empty(len(u))
    for k, uk in enumerate(u):
        s_, l_ = np.linalg.slogdet(Dm + (1 - uk) / (1 + uk) * X)
        sg[k], lg[k] = s_, l_ + 0.5 * Ne * np.log(0.5 * (1 + uk))
    ref = lg.max()  # signed log-sum-exp (overflow-safe at large N_e)
    return 0.5 * lu[0] * np.exp(lu[1] + ref) * np.sum(w * sg * np.exp(lg - ref))


def quad_single(Phiu, Phid, U, D, nodes):
    """Naive quadrature: N_q full N_e x N_e determinants of the rotated orbitals."""
    u, w = nodes
    sg, lg = np.empty(len(u)), np.empty(len(u))
    for k, uk in enumerate(u):
        c, s = np.sqrt(0.5 * (1 + uk)), np.sqrt(0.5 * (1 - uk))
        M = np.block([[c * Phiu[U], -s * Phid[U]], [s * Phiu[D], c * Phid[D]]])
        sg[k], lg[k] = np.linalg.slogdet(M)
    ref = lg.max()
    return 0.5 * np.exp(ref) * np.sum(w * sg * np.exp(lg - ref))


def sqrvb_single(p, nU, nD, L, Ne):
    """Single-configuration sqrvb amplitude without J and s_ord (both O(L^2)): one N_e x N_e and
    one m x m determinant after the O(H L N_e + H L^2) charge-conditioned fields."""
    n = nU + nD
    feat = np.concatenate([(n == 2), (n == 0)]).astype(float)
    h = np.tanh(p["W"] @ feat + p["b"])
    Psi = p["Psi0"] + (h @ p["Vc"]).reshape(2 * L, Ne)
    occ2 = np.sort(np.concatenate([2 * np.nonzero(n >= 1)[0], 2 * np.nonzero(n == 2)[0] + 1]))
    dc = np.linalg.slogdet(Psi[occ2])  # (sign, log); slogdet as in che_single
    dc = dc[0] * np.exp(dc[1])
    M = p["A0"] + (h @ p["Va"]).reshape(L, L)
    U = np.nonzero(nU * (1 - nD))[0]
    D = np.nonzero(nD * (1 - nU))[0]
    if len(U) == 0:
        return dc
    sub = 0.5 * (M[np.ix_(U, D)] + M[np.ix_(D, U)].T)
    below = np.cumsum(nD * (1 - nU)) - nD * (1 - nU)
    sgn = 1.0 - 2.0 * (int(np.sum((nU * (1 - nD)) * below)) % 2)
    ds = np.linalg.slogdet(sub)
    return dc * sgn * ds[0] * np.exp(ds[1])


def che_fields(p, nU, nD, L, Nu, Nd):
    n = nU + nD
    x = 2.0 * np.concatenate([(n == 2), (n == 0)]).astype(float) - 1.0
    h = np.tanh(x @ p["W1"] + p["b1"])
    out = h @ p["W2"] + p["b2"] + p["P0"]
    return out[:L * Nu].reshape(L, Nu), out[L * Nu:].reshape(L, Nd)


def scf_single(nU, nD, Psi0, G, W0, Uw, Wh, bh, lvec, rv):
    """Single-configuration SCF chain (no J, no s_ord): h(c), tensors, chain of vector-matrix
    products over the lattice snake, one charge determinant."""
    n = nU + nD
    L = len(n)
    feat = np.concatenate([(n == 2), (n == 0)]).astype(float)
    h = np.tanh(Wh @ feat + bh)
    occ2 = np.sort(np.concatenate([2 * np.nonzero(n >= 1)[0], 2 * np.nonzero(n == 2)[0] + 1]))
    dc = np.linalg.slogdet(Psi0[occ2])
    dc = dc[0] * np.exp(dc[1])
    ks = np.nonzero(n == 1)[0]
    w = W0[ks] + Uw[ks] @ h                                   # (Ls, nw)
    sig = np.where(nU[ks] == 1, 0, 1)
    M = np.einsum("kw,wkab->kab", w, G[:, sig])               # (Ls, D, D) site tensors
    v = lvec
    for Mk in M:
        v = v @ Mk
    return dc * (v @ rv)


def sord_single(nU, nD):
    """site-major -> up-then-down operator-order sign: (-1)^{sum_j nD_j #{i > j: nU_i = 1}}."""
    above = np.cumsum(nU[::-1])[::-1] - nU
    return 1.0 - 2.0 * (int(np.sum(nD * above)) % 2)


def nnbf_orbit_single(p, nU, nD, L, Nu, Nd):
    """<x|P_{S=0}|phi_nnbf> for a SPIN-RESOLVED NNBF (no J; J is charge-only and factors out) by
    the exact orbit sum: in the site-major product basis
    <sigma|P_0|sigma'> = (-1)^k k! (m-k)! / (m+1)!, m = N_s / 2, k = half the Hamming distance
    between sigma and sigma' on the N_s singly occupied sites (from Loewdin's
    P_0 = sum_j (-1)^j (S^-)^j (S^+)^j / (j! (j+1)!) summed over intermediate states). The sum runs over all
    binom(N_s, N_s/2) spin arrangements of the singles with the charge fixed, each one NNBF
    forward pass + 2 determinants: exponential in N_s. Batched over the orbit."""
    from itertools import combinations
    from math import factorial
    sing = np.nonzero((nU + nD) == 1)[0]
    m = int(np.sum(nU * (1 - nD)))
    upset = set(np.nonzero(nU * (1 - nD))[0].tolist())
    dbl = (nU * nD).astype(int)
    orb = []
    for ups in combinations(sing.tolist(), m):
        u = dbl.copy(); d = dbl.copy()
        u[list(ups)] = 1
        d[[i for i in sing.tolist() if i not in ups]] = 1
        k = m - len(upset & set(ups))
        orb.append((u, d, (-1.0) ** k * factorial(k) * factorial(m - k) / factorial(m + 1)))
    U = np.array([o[0] for o in orb]); D = np.array([o[1] for o in orb]); w = np.array([o[2] for o in orb])
    x = np.concatenate([U, D], 1).astype(float) * 2 - 1
    h = np.tanh(x @ p["W1"] + p["b1"])
    out = h @ p["W2"] + p["b2"] + p["P0"]
    B = len(w)
    Ou = out[:, :L * Nu].reshape(B, L, Nu)
    Od = out[:, L * Nu:].reshape(B, L, Nd)
    ru = np.nonzero(U)[1].reshape(B, Nu)
    rd = np.nonzero(D)[1].reshape(B, Nd)
    au = np.take_along_axis(Ou, ru[:, :, None], axis=1)
    ad = np.take_along_axis(Od, rd[:, :, None], axis=1)
    so = np.array([sord_single(U[b], D[b]) for b in range(B)])
    val = np.linalg.det(au) * np.linalg.det(ad) * so
    return sord_single(nU, nD) * float(np.sum(w * val)), B


# ============================================================================ state
class State:
    """psi = P_G [P_S] phi_theta; exact energy and gradient (P_G, P_S symmetric, commuting)."""

    def __init__(self, sec, arm, sproj, sg, chunk):
        self.sec, self.arm, self.chunk, self.sg = sec, arm, chunk, sg
        self.PS = sec.projS(0) if sproj else None

    def psi(self, phi):
        if self.PS is not None:
            phi = self.PS(phi)
        return self.sg.project(phi) if self.sg is not None else phi

    def energy_grad(self, th):
        sec = self.sec
        phi, cc = self.arm.forward(th, self.chunk)
        psi = self.psi(phi)
        nrm = np.sum(psi * psi)
        Hp = sec.H(psi)
        E = float(np.sum(psi * Hp) / nrm)
        v = 2.0 * (Hp - E * psi) / nrm
        return E, self.arm.vjp(cc, self.psi(v), self.chunk), psi


def build_arm(sec, kind, nnbf_hidden, seed, mult):
    L = sec.lat.L
    nf = R.charge_features(sec)[0].shape[1]
    Ne = sec.Nu + sec.Nd
    target = R.NNBF.count(L, nf, sec.Nu, sec.Nd, nnbf_hidden)
    nw = 2 * sum(mult[k] * mult[k + 1] for k in range(len(mult) - 1))

    def best(cnt):
        return min(range(1, 400), key=lambda H: abs(cnt(H) - target))
    if kind in ("nnbf_SG", "nnbf_G"):
        return R.NNBF(sec, nnbf_hidden, seed)
    if kind == "che":
        return ChargeNNBF(sec, nnbf_hidden, seed)
    if kind == "foil":
        return R.DetArm(sec, "foil", 0, seed)
    if kind in ("sqrvb", "sqrvb0"):
        cpl = kind == "sqrvb"
        return SQRVB(sec, best(lambda H: SQRVB.count(L, nf, Ne, H, cpl)), seed, cpl)
    if kind == "scf":
        return R.SCF(sec, best(lambda H: R.SCF.count(L, nf, Ne, nw, mult[0], H)), seed, mult)
    raise ValueError(kind)


# ============================================================================ modes
def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="P4.1 round-3 exact-SU(2) test (see module docstring)")
    p.add_argument("--mode", required=True, choices=["smoke", "time", "cost", "run", "report"])
    p.add_argument("--Lx", type=int, default=4)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--Ne", type=int, default=8)
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--tp", type=float, default=-0.25)
    p.add_argument("--bc-y", default="obc", choices=["obc", "pbc"], dest="bc_y")
    p.add_argument("--ansatz", default="sqrvb", choices=ARMS)
    p.add_argument("--nnbf-hidden", type=int, default=16, dest="nnbf_hidden")
    p.add_argument("--mps-mult", default="2,2,1,1", dest="mps_mult")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--max-hours", type=float, default=20.0, dest="max_hours")
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--lbfgs-iters", type=int, default=4000, dest="lbfgs_iters")
    p.add_argument("--fn-iters", type=int, default=10, dest="fn_iters")
    p.add_argument("--fn-floor", type=float, default=1e-3, dest="fn_floor")
    p.add_argument("--no-sign", action="store_true", dest="no_sign")
    p.add_argument("--n-samples", type=int, default=20000, dest="n_samples")
    p.add_argument("--cost-sizes", default="6,8,12,16,20", dest="cost_sizes")
    p.add_argument("--cost-reps", type=int, default=60, dest="cost_reps")
    p.add_argument("--outdir", default=os.path.join(here, "results"))
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    a.mult = tuple(int(v) for v in a.mps_mult.split(","))
    if not (2 <= len(a.mult) <= 6) or min(a.mult) < 1:
        p.error("--mps-mult needs 2 to 6 positive integers")
    L = a.Lx * a.Ly
    if a.Lx < 2 or a.Ly < 2 or a.Ne % 2 or not (2 <= a.Ne <= 2 * L - 2):
        p.error("invalid lattice / Ne")
    if a.tag is None:
        a.tag = f"L{a.Lx}x{a.Ly}{a.bc_y}_N{a.Ne}_U{a.U:g}_tp{a.tp:g}_G"
    return a


def fd_check(st, th, n=8, h=1e-5, seed=11):
    E, g, _ = st.energy_grad(th)
    rng = np.random.default_rng(seed)
    err = 0.0
    for k in rng.choice(len(th), size=min(n, len(th)), replace=False):
        e = np.zeros(len(th)); e[k] = h
        fd = (st.energy_grad(th + e)[0] - st.energy_grad(th - e)[0]) / (2 * h)
        err = max(err, abs(fd - g[k]) / max(1e-6, abs(fd) + abs(g[k])))
    return err


def smoke(a):
    t0 = time.time()
    ok = True
    sec = make_sector(2, 3, 6, 6.0, -0.3, "pbc")
    sg = SG(sec)
    Hs = sec.Hsparse()
    P0, E0, s20, lam, _ = ground_state(sec, sg)
    rng = np.random.default_rng(0)
    X, Y = rng.standard_normal(sec.shape), rng.standard_normal(sec.shape)
    PX = sg.project(X)
    idem = np.abs(sg.project(PX) - PX).max()
    sym = abs(np.sum(Y * PX) - np.sum(sg.project(Y) * X))
    comm = np.abs(sg.project(sec.H(X)) - sec.H(PX)).max()
    PS = sec.projS(0)
    scomm = np.abs(PS(PX) - sg.project(PS(X))).max()
    ov = np.sum(sg.project(P0) * P0)
    c1 = max(idem, sym, comm, scomm) < 1e-12 and abs(ov - 1) < 1e-10
    ok &= c1
    sc = make_sector(4, 2, 4, 6.0, -0.3, "obc")
    gc = SG(sc)
    Pc, Ec, sc2, lc, _ = ground_state(sc, gc)
    Xc = rng.standard_normal(sc.shape)
    PXc = gc.project(Xc)
    ec = max(np.abs(gc.project(PXc) - PXc).max(), np.abs(gc.project(sc.H(Xc)) - sc.H(PXc)).max(),
             np.abs(sc.projS(0)(PXc) - gc.project(sc.projS(0)(Xc))).max(),
             max(np.abs(sc.H(gc.apply(Xc, k)) - gc.apply(sc.H(Xc), k)).max() for k in range(gc.G)))
    oc = np.sum(gc.project(Pc) * Pc)
    cc_ = ec < 1e-12 and abs(oc - 1) < 1e-10
    ok &= cc_
    print(f"4x2 cylinder (y open) Ne=4 dim={sc.dim} |G|={gc.G} irrep={lc}: max(P_G idem, [P_G,H], [P_G,P_S], "
          f"[U_g,H]) = {ec:.1e}, <psi0|P_G|psi0>={oc:.12f} [{'PASS' if cc_ else 'FAIL'}]")
    print(f"2x3 Ne=6 U=6 t'=-0.3 dim={sec.dim} |G|={sg.G} irrep={lam}: E0={E0:.8f} <S^2>={s20:.1e}; "
          f"P_G idem {idem:.1e} sym {sym:.1e} [P_G,H] {comm:.1e} [P_G,P_S] {scomm:.1e} "
          f"<psi0|P_G|psi0>={ov:.12f} [{'PASS' if c1 else 'FAIL'}]")
    for kind in ARMS:
        arm = build_arm(sec, kind, 3, 2, (1, 1, 1, 1))
        st = State(sec, arm, SPROJ[kind], sg, 4096)
        th = arm.theta0 + 0.05 * np.random.default_rng(7).standard_normal(arm.nparam)
        E, g, psi = st.energy_grad(th)
        phi = arm.forward(th, 4096)[0]
        s2r = np.sum(phi * sec.S2(phi)) / np.sum(phi * phi)
        s2 = np.sum(psi * sec.S2(psi)) / np.sum(psi * psi)
        err = fd_check(st, th)
        s2ok = abs(s2) < 1e-10 if kind in EXACT else True
        gok = err < 1e-5
        ok &= s2ok and gok
        print(f"  {kind:7s} P={arm.nparam:4d} E={E:+.6f} <S^2>(phi)={s2r:+.2e} <S^2>(psi)={s2:+.2e} "
              f"[{'PASS' if s2ok else 'FAIL'}{'' if kind in EXACT else ' (not required)'}]  grad rel.err={err:.1e} "
              f"[{'PASS' if gok else 'FAIL'}]")
    # (d) per-amplitude algorithms vs batched / full-vector states
    arm = build_arm(sec, "sqrvb", 3, 2, (1, 1, 1, 1))
    th = arm.theta0 + 0.1 * np.random.default_rng(3).standard_normal(arm.nparam)
    phi, cc = arm.forward(th, 4096)
    phi = phi.ravel()
    pp = arm.unpack(th)
    errs = []
    for k in np.random.default_rng(4).choice(sec.dim, 40, replace=False):
        nU, nD = sec.up.bits[sec.iu[k]].astype(int), sec.dn.bits[sec.id[k]].astype(int)
        ref = phi[k] / (arm.sord[k] * cc["expJ"][sec.cidx[k]])
        errs.append(abs(sqrvb_single(pp, nU, nD, sec.lat.L, arm.Ne) - ref) / max(1e-300, np.abs(phi).max()))
    e1 = max(errs)
    arm = build_arm(sec, "che", 3, 2, (1, 1, 1, 1))
    th = arm.theta0 + 0.1 * np.random.default_rng(5).standard_normal(arm.nparam)
    pp = arm.unpack(th)
    pp["jv"] = 0.0 * pp["jv"]
    th = arm.pack(pp)
    phi = arm.forward(th, 4096)[0]
    proj = sec.projS(0)(phi).ravel()
    nodes = gl_nodes(sec.Nu)
    e2 = e3 = 0.0
    for k in np.random.default_rng(6).choice(sec.dim, 40, replace=False):
        nU, nD = sec.up.bits[sec.iu[k]].astype(int), sec.dn.bits[sec.id[k]].astype(int)
        Pu, Pd = che_fields(pp, nU, nD, sec.lat.L, sec.Nu, sec.Nd)
        U, D = sec.up.occ[sec.iu[k]], sec.dn.occ[sec.id[k]]
        sc = np.exp(-arm.jastrow(pp).max())  # forward() normalises e^J by its max
        e2 = max(e2, abs(sc * che_single(Pu, Pd, U, D, nodes) - proj[k]) / np.abs(proj).max())
        e3 = max(e3, abs(sc * quad_single(Pu, Pd, U, D, nodes) - proj[k]) / np.abs(proj).max())
    arm = build_arm(sec, "scf", 3, 2, (1, 1, 1, 1))
    th = arm.theta0 + 0.1 * np.random.default_rng(8).standard_normal(arm.nparam)
    phi, cc = arm.forward(th, 4096)
    phi = phi.ravel()
    pp = arm.unpack(th)
    rv = np.zeros(arm.D); rv[arm.j0] = pp["r"]
    e4 = 0.0
    for k in np.random.default_rng(9).choice(sec.dim, 40, replace=False):
        nU, nD = sec.up.bits[sec.iu[k]].astype(int), sec.dn.bits[sec.id[k]].astype(int)
        ref = phi[k] / (arm._order_sign(slice(k, k + 1))[0] * cc["expJ"][sec.cidx[k]])
        val = scf_single(nU, nD, pp["Psi0"], arm.G, pp["W0"], pp["Uw"], pp["A"], pp["a"], arm.lvec, rv)
        e4 = max(e4, abs(val - ref) / np.abs(phi).max())
    arm = build_arm(sec, "nnbf_SG", 3, 2, (1, 1, 1, 1))
    th = arm.theta0 + 0.1 * np.random.default_rng(10).standard_normal(arm.nparam)
    pp = arm.unpack(th)
    pp["jv"] = 0.0 * pp["jv"]
    th = arm.pack(pp)
    proj = sec.projS(0)(arm.forward(th, 4096)[0]).ravel()
    e5 = 0.0
    for k in np.random.default_rng(11).choice(sec.dim, 40, replace=False):
        nU, nD = sec.up.bits[sec.iu[k]].astype(int), sec.dn.bits[sec.id[k]].astype(int)
        e5 = max(e5, abs(nnbf_orbit_single(pp, nU, nD, sec.lat.L, sec.Nu, sec.Nd)[0] - proj[k]) / np.abs(proj).max())
    c4 = max(e1, e2, e3, e4, e5) < 1e-10
    ok &= c4
    print(f"  per-amplitude: sqrvb_single vs batched {e1:.1e}; che_single (N_q={len(nodes[0])}) vs full-vector "
          f"P_S {e2:.1e}; quad_single vs full-vector P_S {e3:.1e}; scf_single vs batched {e4:.1e} "
          f"nnbf_orbit_single vs full-vector P_S {e5:.1e} [{'PASS' if c4 else 'FAIL'}]")
    sb = R.sign_budget(Hs, P0.ravel(), P0.ravel(), E0, argparse.Namespace(fn_iters=3, fn_floor=1e-3, lbfgs_iters=500))
    sok = abs(sb["E_sign_up"] - E0) < 1e-8
    ok &= sok
    print(f"  sign budget on psi_0: E_sign_up - E0 = {sb['E_sign_up'] - E0:.1e} [{'PASS' if sok else 'FAIL'}]")
    for kind in ARMS:
        arm = build_arm(sec, kind, 3, 0, (1, 1, 1, 1))
        st = State(sec, arm, SPROJ[kind], sg, 4096)
        th, E, nit, dt = R.optimise(st, arm.theta0, 300, 0.02, logf=lambda *x: None)
        print(f"  opt {kind:7s}: P={arm.nparam:4d} eps={(E - E0) / abs(E0):.2e} ({nit} it, {dt:.1f}s)")
    print(f"(2x3: more parameters than symmetric states for most arms; optimisation lines only show the code runs)\n"
          f"SMOKE {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.1f}s)")


def timing(a):
    t0 = time.time()
    sec = make_sector(a.Lx, a.Ly, a.Ne, a.U, a.tp, a.bc_y)
    sg = SG(sec)
    log(f"instance {a.tag}: dim={sec.dim} nC={sec.nC} |G|={sg.G} (built in {time.time() - t0:.1f}s)")
    for kind in ARMS:
        arm = build_arm(sec, kind, a.nnbf_hidden, 1, a.mult)
        st = State(sec, arm, SPROJ[kind], sg, a.chunk)
        t1 = time.time()
        E, g, psi = st.energy_grad(arm.theta0)
        log(f"  {kind:7s} P={arm.nparam:5d} hidden={getattr(arm, 'H', 0):3d} P_S={int(SPROJ[kind])}  "
            f"E(init)={E:+.5f}  energy+grad {time.time() - t1:.1f}s")


def cost(a):
    """Per-amplitude wall time of the single-configuration algorithms (module docstring)."""
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(0)
    res = []
    for Lx in [int(v) for v in a.cost_sizes.split(",")]:
        L = Lx * Lx
        Ne = 2 * int(round(7 * L / 16))
        Nu = Nd = Ne // 2
        H = 8
        pq = dict(Psi0=rng.standard_normal((2 * L, Ne)) / np.sqrt(L), A0=rng.standard_normal((L, L)) / np.sqrt(L),
                  W=rng.standard_normal((H, 2 * L)) * 0.3, b=np.zeros(H),
                  Vc=0.01 * rng.standard_normal((H, 2 * L * Ne)), Va=0.01 * rng.standard_normal((H, L * L)))
        pc = dict(P0=rng.standard_normal(L * Ne) / np.sqrt(L), W1=rng.standard_normal((2 * L, H)) * 0.3,
                  b1=np.zeros(H), W2=0.01 * rng.standard_normal((H, L * Ne)), b2=np.zeros(L * Ne))
        D, nw = 13, 14  # --mps-mult 2,2,1,1
        G = rng.standard_normal((nw, 2, D, D)) / np.sqrt(D)
        W0 = rng.standard_normal((L, nw)); Uw = 0.01 * rng.standard_normal((L, nw, H))
        lvec, rv = np.eye(D)[0], np.eye(D)[0]
        cfgs = []
        for _ in range(a.cost_reps):
            nd_ = int(round(0.05 * Nu))
            sites = rng.permutation(L)
            dbl = sites[:nd_]
            su = sites[nd_:Nu]
            sdn = sites[Nu:Nu + Nd - nd_]
            nU = np.zeros(L, int); nD = np.zeros(L, int)
            nU[np.concatenate([dbl, su])] = 1
            nD[np.concatenate([dbl, sdn])] = 1
            cfgs.append((nU, nD))
        nodes = gl_nodes(Nu)
        row = dict(Lx=Lx, L=L, Ne=Ne, Nq=len(nodes[0]))
        for name in ("sqrvb", "scf", "che", "quad"):
            t = time.perf_counter()
            for nU, nD in cfgs:
                if name == "sqrvb":
                    sqrvb_single(pq, nU, nD, L, Ne)
                elif name == "scf":
                    scf_single(nU, nD, pq["Psi0"], G, W0, Uw, pq["W"], pq["b"], lvec, rv)
                else:
                    Pu, Pd = che_fields(pc, nU, nD, L, Nu, Nd)
                    U, Dd = np.nonzero(nU)[0], np.nonzero(nD)[0]
                    (che_single if name == "che" else quad_single)(Pu, Pd, U, Dd, nodes)
            row[f"t_{name}_us"] = 1e6 * (time.perf_counter() - t) / len(cfgs)
        res.append(row)
        print(f"Lx={Lx:2d} L={L:3d} Ne={Ne:3d} N_q={row['Nq']:2d}: per amplitude  sqrvb {row['t_sqrvb_us']:9.1f} us  "
              f"scf {row['t_scf_us']:9.1f} us  che(LU-reused GL) {row['t_che_us']:9.1f} us  "
              f"naive quadrature {row['t_quad_us']:9.1f} us", flush=True)
    orbit = []
    for (Lx, Ly) in [(4, 3), (4, 4), (5, 4), (6, 4)]:
        L = Lx * Ly
        Ne = 2 * int(round(7 * L / 16))
        Nu = Nd = Ne // 2
        H = 8
        pn = dict(P0=rng.standard_normal(L * Ne) / np.sqrt(L), W1=rng.standard_normal((2 * L, H)) * 0.3,
                  b1=np.zeros(H), W2=0.01 * rng.standard_normal((H, L * Ne)), b2=np.zeros(L * Ne))
        ts, ns = [], []
        for _ in range(3):
            sites = rng.permutation(L)
            nU = np.zeros(L, int); nD = np.zeros(L, int)
            nU[sites[:Nu]] = 1; nD[sites[Nu:Nu + Nd]] = 1  # no doublons: the largest orbit
            t = time.perf_counter()
            _, B = nnbf_orbit_single(pn, nU, nD, L, Nu, Nd)
            ts.append(time.perf_counter() - t); ns.append(B)
        orbit.append(dict(L=L, Ne=Ne, orbit=int(np.median(ns)), t_orbit_us=1e6 * float(np.median(ts))))
        print(f"spin-resolved NNBF exact P_S by orbit sum: L={L:3d} Ne={Ne:3d} orbit={orbit[-1]['orbit']:7d} "
              f"per amplitude {orbit[-1]['t_orbit_us']:12.1f} us", flush=True)
    big = res[-3:]
    ne = np.log([r["Ne"] for r in big])
    slopes = {k: float(np.polyfit(ne, np.log([r[f"t_{k}_us"] for r in big]), 1)[0])
              for k in ("sqrvb", "scf", "che", "quad")}
    print("log-log slopes in N_e over the three largest sizes: " + "  ".join(f"{k} {v:.2f}" for k, v in slopes.items()))
    print("(numpy per-call overhead dominates at small N_e, hence the fit over the largest sizes only)")
    with open(os.path.join(a.outdir, "cost.json"), "w") as fh:
        json.dump(dict(rows=res, slopes=slopes, orbit=orbit), fh, indent=1)


def sampled_eloc(sec, psi, n, seed):
    """x ~ |psi|^2 drawn exactly; E_loc = (H psi)(x) / psi(x)."""
    p = (psi * psi).ravel()
    p = p / p.sum()
    idx = np.random.default_rng(seed).choice(p.size, size=n, p=p)
    el = (sec.H(psi).ravel()[idx]) / psi.ravel()[idx]
    E = float(np.sum(psi * sec.H(psi)) / np.sum(psi * psi))
    sd = float(np.std(el))
    return dict(E_mc=float(np.mean(el)), std=sd, tail=float(np.quantile(np.abs(el - E), 0.999) / max(sd, 1e-300)))


def run(a):
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"{a.tag}_{a.ansatz}_seed{a.seed}.json")
    if os.path.exists(out) and not a.overwrite:
        print(f"exists: {out}  (use --overwrite)")
        return
    t0 = time.time()
    sec = make_sector(a.Lx, a.Ly, a.Ne, a.U, a.tp, a.bc_y)
    sg = SG(sec)
    P0, E0, s20, lam, (P1, E1, s21) = ground_state(sec, sg)
    L = sec.lat.L
    log(f"instance {a.tag}: dim={sec.dim}; psi_0 E0={E0:.10f} <S^2>={s20:.1e} irrep={lam}; "
        f"next K=0 S=0 level {E1:.6f} (<S^2>={s21:.2f}) ({time.time() - t0:.0f}s)")
    arm = build_arm(sec, a.ansatz, a.nnbf_hidden, a.seed, a.mult)
    st = State(sec, arm, SPROJ[a.ansatz], sg, a.chunk)
    log(f"arm {a.ansatz}: {arm.nparam} parameters, hidden {getattr(arm, 'H', 0)}, P_S={int(SPROJ[a.ansatz])}, P_G on")
    th, E, nit, dt = R.optimise(st, arm.theta0, a.maxiter, a.max_hours)
    capped = dt > 3600 * a.max_hours * 0.999
    phi = arm.forward(th, a.chunk)[0]
    psi = st.psi(phi)
    obs, obs0 = R.observables(sec, psi), R.observables(sec, P0)
    res = dict(tag=a.tag, ansatz=a.ansatz, seed=a.seed, nparam=arm.nparam, hidden=getattr(arm, "H", 0),
               sproj=int(SPROJ[a.ansatz]), iters=nit, seconds=dt, capped=bool(capped), E0=E0, irrep=lam,
               E=obs["E"], eps=(obs["E"] - E0) / abs(E0), S2=obs["S2"], Cs=obs["Cs"], Pd=obs["Pd"],
               Cs_exact=obs0["Cs"], Pd_exact=obs0["Pd"], args={k: v for k, v in vars(a).items() if k != "mult"})
    Hs = sec.Hsparse()
    res["sampled"] = sampled_eloc(sec, psi, a.n_samples, a.seed)
    if not a.no_sign:
        t1 = time.time()
        res["sign"] = R.sign_budget(Hs, psi.ravel(), P0.ravel(), E0, a, a.seed)
        log(f"sign budget: dE_tot={res['sign']['dE_tot']:.5f} dE_sign={res['sign']['dE_sign']:.5f} "
            f"f_E_up={res['sign']['f_E_up']:.3f} ({time.time() - t1:.0f}s)")
    # post-hoc full projection P_G P_S of the trained phi (identity for exact-S, P_G-trained arms)
    pp = sg.project(sec.projS(0)(phi))
    o2 = R.observables(sec, pp)
    res["posthoc"] = dict(E=o2["E"], eps=(o2["E"] - E0) / abs(E0), S2=o2["S2"], Cs=o2["Cs"], Pd=o2["Pd"],
                          weight=float(np.sum(pp * pp) / np.sum(psi * psi)))
    if not a.no_sign and not SPROJ[a.ansatz] and a.ansatz not in ("sqrvb", "sqrvb0", "scf"):
        res["posthoc"]["sign"] = R.sign_budget(Hs, pp.ravel(), P0.ravel(), E0, a, a.seed)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    log(f"done: E={obs['E']:.10f} eps={res['eps']:.3e} <S^2>={obs['S2']:.3e} iters={nit} -> {out}")


def report(a):
    runs = {}
    for fn in glob.glob(os.path.join(a.outdir, f"{a.tag}_*_seed*.json")):
        r = json.load(open(fn))
        runs.setdefault(r["ansatz"], []).append(r)

    def dcs(r):
        return max(abs(r["Cs"][k] - r["Cs_exact"][k]) for k in r["Cs"])

    def dpd(r):
        v = [abs(r["Pd"][k] - r["Pd_exact"][k]) for k in r["Pd"]]
        return float(np.mean(v)) if v else 0.0

    def dsg(r):
        return r["sign"]["dE_sign"] if "sign" in r else float("nan")
    met = {"eps": lambda r: r["eps"], "dE_sign": dsg, "dCs": dcs, "dPd": dpd, "S2": lambda r: abs(r["S2"])}
    stats = {}
    for k in ARMS:
        if k not in runs:
            continue
        rs = sorted(runs[k], key=lambda r: r["seed"])
        s = {m: [f(r) for r in rs] for m, f in met.items()}
        stats[k] = dict(n=len(rs), seeds=[r["seed"] for r in rs], capped=any(r.get("capped") for r in rs),
                        iters=[r["iters"] for r in rs], nparam=rs[0]["nparam"],
                        med={m: float(np.median(v)) for m, v in s.items()},
                        lo={m: float(np.min(v)) for m, v in s.items()},
                        hi={m: float(np.max(v)) for m, v in s.items()},
                        maxS2=float(np.max(s["S2"])))
        st = stats[k]
        print(f"{k:7s} n={st['n']} P={st['nparam']:5d} iters={st['iters']} "
              + "  ".join(f"{m}={st['med'][m]:.3e} [{st['lo'][m]:.2e},{st['hi'][m]:.2e}]"
                          for m in ("eps", "dE_sign", "dCs", "dPd")) + f"  max|S2|={st['maxS2']:.1e}")
    if "nnbf_G" in runs:
        ph = [r["posthoc"]["eps"] for r in runs["nnbf_G"]]
        print(f"nnbf_G post-hoc P_S P_G: median eps {np.median(ph):.3e} (trained-projected nnbf_SG: "
              f"{stats.get('nnbf_SG', {}).get('med', {}).get('eps', float('nan')):.3e})")
    if "che" in stats and "nnbf_SG" in stats:
        print(f"diagnostic: m(eps_che)/m(eps_nnbf_SG) = {stats['che']['med']['eps'] / stats['nnbf_SG']['med']['eps']:.3f} "
              f"(charge-only vs spin-resolved conditioning, both exactly projected)")
    cj = os.path.join(a.outdir, "cost.json")
    cost_ok = {}
    if os.path.exists(cj):
        C = json.load(open(cj))
        last = C["rows"][-1]
        for Rn in ROUTES:
            cost_ok[Rn] = (last[f"t_{Rn}_us"] < last["t_che_us"]) and (C["slopes"][Rn] < C["slopes"]["che"])
    verdicts = {}
    need = {"nnbf_SG", "che", "foil"}
    for Rn in ROUTES:
        if Rn not in stats or not need <= set(stats):
            verdicts[Rn] = "AMBIGUOUS (missing arms)"
            continue
        r, n, ch, fo = stats[Rn], stats["nnbf_SG"], stats["che"], stats["foil"]
        few = min(x["n"] for x in (r, n, ch, fo)) < 3 or (Rn == "sqrvb" and stats.get("sqrvb0", {"n": 0})["n"] < 3)
        capped = any(x["capped"] for x in (r, n, ch, fo))
        g1 = r["maxS2"] < 1e-10
        gc = cost_ok.get(Rn, False)
        gm = True if Rn != "sqrvb" else ("sqrvb0" in stats and r["med"]["eps"] <= 0.8 * stats["sqrvb0"]["med"]["eps"])
        g2 = r["med"]["dCs"] <= n["med"]["dCs"] and r["med"]["dPd"] <= n["med"]["dPd"] + 1e-4
        mE, mS = r["med"]["eps"] / n["med"]["eps"], r["med"]["dE_sign"] / n["med"]["dE_sign"]
        mC, mF = r["med"]["eps"] / ch["med"]["eps"], r["med"]["eps"] / fo["med"]["eps"]
        prem = n["med"]["eps"] >= PREMISE_EPS
        if (not g1) or (not gc) or mE >= 1.0 or mC >= 1.0:
            v = "FAIL"
        elif not prem:
            v = "UNINFORMATIVE"
        elif few or capped:
            v = "AMBIGUOUS"
        elif g2 and gm and mE <= 0.8 and mS <= 0.8 and mC < 1.0 and mF <= 0.5:
            v = "PASS"
        else:
            v = "AMBIGUOUS"
        verdicts[Rn] = dict(M_E=mE, M_sign=mS, M_che=mC, M_foil=mF, premise=prem, g1=g1, g_cost=gc, g_mech=gm, g2=g2,
                            few_seeds=few, capped=capped, verdict=v)
        print(f"{Rn}: M_E={mE:.3f} M_sign={mS:.3f} M_che={mC:.3f} M_foil={mF:.3f} g1={g1} g_cost={gc} "
              f"g_mech={gm} g2={g2} -> {v}")
    with open(os.path.join(a.outdir, f"{a.tag}_report.json"), "w") as fh:
        json.dump(dict(stats=stats, verdicts=verdicts), fh, indent=1)


def main(argv=None):
    a = parse_args(argv)
    {"smoke": smoke, "time": timing, "cost": cost, "run": run, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
