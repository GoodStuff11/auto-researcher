#!/usr/bin/env python3
"""
P4.1 round 4: honest cost of exact spin-SU(2), completeness of "one AGP per charge configuration",
the spin-feature dichotomy (hardness + bucketing), and the decisive accuracy test.
=================================================================================================

Local copies (nothing imported across folders): r3_micro_test.py (verbatim copy of
tests/P4.1_cheap_su2_r3/micro_test.py) and r2_micro_test.py (its physics core).

Notation: x = (nU, nD) occupations; c(x) = charge configuration; N_s = number of singly occupied
sites; m = N_s / 2 (S^z = 0); n = N_e / 2; site-major ("sm") and up-then-down ("ud") operator
orders differ by s_ord(x) = (-1)^{sum_j nD_j #{i > j : nU_i = 1}}.

Ansatz families (all exact S = 0)
---------------------------------
  tocchio  single determinant, symmetric charge-conditioned pairing (Tocchio / round-1 R1 family):
           psi(x) = e^{J(c)} det F_c[U(x), D(x)],  F_c = sym(F0 + sum_h h_h(c) V_h),
           h(c) = tanh(W feat(c) + b) (doublon / holon indicators). 1 det of size n.
  kagpK    K-term charge-conditioned AGP sum (Route 4.2): psi = e^{J} sum_{k<K} det F^{(k)}_c[U, D],
           every F^{(k)}_c symmetric, sharing h(c). K dets of size n. (K = 1 is tocchio.)
  che      Che arXiv:2608.22925 family: charge-input NNBF UHF determinant + exact P_{S=0}.
           Per amplitude by the CLOSED FORM (che_closed):
             <x|P_0|Phi> = det A det Dm int_0^1 prod_i (x + (1 - x) mu_i) dx,
             mu = eig(Dm^{-1} C A^{-1} B), A = Phi_up[U], B = Phi_dn[U], C = Phi_up[D], Dm = Phi_dn[D];
           the integrand is a degree-n polynomial, integrated exactly by ceil((n+1)/2) Gauss-Legendre
           nodes at O(n) each once mu is known: O(n^3) total (one eig), equivalently
           sum_k e_k(mu) k! (n-k)! / (n+1)! (checked against the elementary-symmetric form).
  che_gl   round-3 che_single (LU-reused quadrature, N_q determinants of size n): O(n^4).
  sfb      Route 4.1 constructive half, "spin-feature-bucketed backflow": the charge-input NNBF
           additionally reads ONE spin feature, s_A(x) = (2 w - N_sA) / max(N_sA, 1), w = number of
           up-singles on the checkerboard sublattice A, N_sA = number of singles on A; trained with
           exact P_{S=0}. Per amplitude (sfb_single): the orbit is split by w with
           Pi_w = (1/P) sum_j e^{-i th_j w} e^{i th_j (W' + N_sA)/2}, W' = sum_{i in A}(n_iu - n_id),
           th_j = 2 pi j / P, P = N_sA + 1; e^{i th W'/2} is one-body (row phases), so every term is
           one che_closed: (N_sA + 1)^2 closed-form projections, O(N^5). (With R_b bins of w:
           R_b (N_sA + 1), O(R_b N^4).)
  sqrvb    round-3 SQ-RVB (2 dets: N_e x N_e charge, m x m spinon); reported for continuity.
  nnbf     spin-resolved NNBF, UNPROJECTED (2 dets of size n): the no-symmetry reference cost.
  All per-amplitude field evaluations gather only the needed entries (O(H n^2) or O(H n N_e)) with
  the same hidden width H, so the cost comparison is at equal parameterisation.

Usage
-----
  python r4_test.py --mode MODE [options]

--mode {smoke, cost, dims, lemma, agp, run, report}   (required)
  smoke  : <= 1 min code check on the 2x3 torus (N_e = 6) and 4x2 open cluster (N_e = 4):
           <S^2> = 0 for tocchio, kagp2 (no projector) and che, sfb (with P_S) at random parameters;
           analytic gradient vs finite differences for tocchio, kagp2, sfb; kagp_single == batched;
           che_closed == che_gl == elementary-symmetric form == full-vector P_S; sfb_single ==
           full-vector P_S of the sfb state.
  cost   : per-amplitude wall time on Lx x Lx tori, Lx in --cost-sizes, N_e = 2 round(7L/16), random
           parameters, hidden width --cost-H; sfb only for Lx <= --sfb-max-Lx. Writes
           <outdir>/cost_r4.json; slopes over the three largest sizes.
  dims   : generic local dimension (Jacobian rank at random parameters) of the per-charge-configuration
           tabular families tocchio (K = 1, 2, 3) and projected UHF (che tabular) for N_s = 4 ... 12
           without doublons, against the singlet dimension Catalan(N_s / 2).
  lemma  : (a) #P-hardness reduction check: for random 3-CNF formulas on n_v variables (dual-rail
           encoding on L = 2 n_v sites, half filling) the spin-resolved determinant amplitude
           phi(x') = s_ord(x') SAT(bits(x')) / K_{k(x')} (orbitals that select occupied sites, one entry
           scaled) has (P_{S=0} phi)(x_ref) = s_ord(x_ref) #SAT, checked with the full-vector Loewdin
           projector; (b) timing of the bucketed sfb amplitude vs the exponential orbit sum.
  agp    : tabular completeness on one cluster (--Lx --Ly --Ne --U --tp --bc-x --bc-y): ED psi_0 (cached
           in <outdir>/psi0_<tag>.npz), then for every charge configuration c the best fidelity of the
           per-c tabular families tocchio K = 1, K = 2 and projected UHF (che tabular) to psi_0's spin
           factor chi_c; reports per-N_s medians / minima, the weighted infidelity
           1 - sum_c p_c F_c and the energy error of the assembled tabular state. --agp-restarts (3),
           --agp-maxiter (3000), --agp-families (tocchio1,tocchio2,puhf), --agp-min-Ns (0): groups with
           N_s below it are not fitted and count as F_c = 1 (only valid where --mode dims shows local
           completeness of every family, i.e. N_s <= 6). Writes
           <outdir>/agp_<tag>.json. Slurm (submit_agp.sbatch) for the 4x3 open clusters.
  run    : train ONE arm (--ansatz in tocchio, kagp2, che, sfb, nnbf_SG) with P_G (and P_S where the
           arm needs it) on the production instance, L-BFGS with exact gradient, --maxiter (2000),
           --max-hours (20), --seed; writes <outdir>/<tag>_<ansatz>_seed<seed>.json.
  report : medians over seeds and the pre-committed verdicts (see the leaf artifact, Round 4).

Options
  --Lx (4) --Ly (3) --Ne (10) --U (8) --tp (-0.25) --bc-x {obc,pbc} (obc) --bc-y {obc,pbc} (obc)
  --ansatz {tocchio, kagp2, che, sfb, nnbf_SG} (run)   --nnbf-hidden (16) width of the matched control
  --seed (0) --maxiter (2000) --max-hours (20) --chunk (16384)
  --cost-sizes ("8,12,16,20,24") --cost-reps (20) --cost-H (8) --sfb-max-Lx (6)
  --outdir (<script dir>/results) --tag (auto) --overwrite
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_legendre

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_micro_test as M3  # noqa: E402  (local verbatim copy of the round-3 script)

R = M3.R
log = R.log
RUN_ARMS = ["tocchio", "kagp2", "che", "sfb", "nnbf_SG"]
SPROJ = {"tocchio": False, "kagp2": False, "che": True, "sfb": True, "nnbf_SG": True}


# ============================================================================ per-amplitude algorithms
def feat_c(nU, nD):
    n = nU + nD
    return np.concatenate([(n == 2), (n == 0)]).astype(float)


def che_closed(Phiu, Phid, U, D, ek=False):
    """<x|P_{S=0}|Phi> in closed form (module docstring). Complex orbitals allowed."""
    n = len(U)
    A, B, C, Dm = Phiu[U], Phid[U], Phiu[D], Phid[D]
    sA, lA = np.linalg.slogdet(A)
    sD, lD = np.linalg.slogdet(Dm)
    mu = np.linalg.eigvals(np.linalg.solve(Dm, C @ np.linalg.solve(A, B)))
    if ek:  # elementary symmetric polynomials, sum_k e_k k!(n-k)!/(n+1)!
        e = np.zeros(n + 1, complex); e[0] = 1.0
        for v in mu:
            e[1:] = e[1:] + v * e[:-1]
        wk = np.array([math.factorial(k) * math.factorial(n - k) / math.factorial(n + 1) for k in range(n + 1)])
        return sA * sD * np.exp(lA + lD) * np.sum(e * wk)
    xg, wg = roots_legendre(int(np.ceil((n + 1) / 2)))
    xg, wg = 0.5 * (xg + 1), 0.5 * wg
    lg = np.sum(np.log(xg[:, None] + (1 - xg[:, None]) * mu[None, :].astype(complex)), axis=1)
    ref = lg.real.max()
    return sA * sD * np.exp(lA + lD + ref) * np.sum(wg * np.exp(lg - ref))


def che_rows(p, nU, nD, L, Nu, Nd, extra=None):
    """charge-input NNBF orbitals gathered on the occupied rows only: O(H |occ| (Nu + Nd))."""
    n = nU + nD
    x = 2.0 * feat_c(nU, nD) - 1.0
    if extra is not None:
        x = np.concatenate([x, [extra]])
    h = np.tanh(x @ p["W1"] + p["b1"])
    occ = np.nonzero(n)[0]
    H = h.shape[0]
    W2u = p["W2"][:, :L * Nu].reshape(H, L, Nu)[:, occ, :]
    W2d = p["W2"][:, L * Nu:].reshape(H, L, Nd)[:, occ, :]
    c0 = p["b2"] + p["P0"]
    Pu = np.tensordot(h, W2u, 1) + c0[:L * Nu].reshape(L, Nu)[occ]
    Pd = np.tensordot(h, W2d, 1) + c0[L * Nu:].reshape(L, Nd)[occ]
    pos = -np.ones(L, dtype=np.int64); pos[occ] = np.arange(len(occ))
    return Pu, Pd, pos[np.nonzero(nU)[0]], pos[np.nonzero(nD)[0]], occ


def che_gl(Phiu, Phid, U, D):
    return M3.che_single(Phiu, Phid, U, D, M3.gl_nodes(len(U)))


def kagp_single(p, nU, nD):
    """sum_k det F^{(k)}_c[U, D] (no J), gathered fields O(K H n^2)."""
    h = np.tanh(p["W"] @ feat_c(nU, nD) + p["b"])
    U, D = np.nonzero(nU)[0], np.nonzero(nD)[0]
    tot = 0.0
    for k in range(p["F0"].shape[0]):
        V = p["V"][k]
        Gud = p["F0"][k][np.ix_(U, D)] + np.tensordot(h, V[:, U[:, None], D[None, :]], 1)
        Gdu = p["F0"][k][np.ix_(D, U)] + np.tensordot(h, V[:, D[:, None], U[None, :]], 1)
        tot += np.linalg.det(0.5 * (Gud + Gdu.T))
    return tot


def sqrvb_single_g(p, nU, nD, L, Ne):
    """round-3 SQ-RVB amplitude with gathered fields."""
    n = nU + nD
    h = np.tanh(p["W"] @ feat_c(nU, nD) + p["b"])
    H = h.shape[0]
    occ2 = np.sort(np.concatenate([2 * np.nonzero(n >= 1)[0], 2 * np.nonzero(n == 2)[0] + 1]))
    Vc = p["Vc"].reshape(H, 2 * L, Ne)
    Psi = p["Psi0"][occ2] + np.tensordot(h, Vc[:, occ2, :], 1)
    dc = np.linalg.det(Psi)
    U = np.nonzero(nU * (1 - nD))[0]; D = np.nonzero(nD * (1 - nU))[0]
    if len(U) == 0:
        return dc
    Va = p["Va"].reshape(H, L, L)
    sub = 0.5 * (p["A0"][np.ix_(U, D)] + np.tensordot(h, Va[:, U[:, None], D[None, :]], 1)
                 + (p["A0"][np.ix_(D, U)] + np.tensordot(h, Va[:, D[:, None], U[None, :]], 1)).T)
    return dc * np.linalg.det(sub)


def nnbf_single(p, nU, nD, L, Nu, Nd):
    """unprojected spin-resolved NNBF, gathered rows: 2 dets of size n."""
    x = np.concatenate([nU, nD]).astype(float) * 2 - 1
    h = np.tanh(x @ p["W1"] + p["b1"])
    H = h.shape[0]
    U, D = np.nonzero(nU)[0], np.nonzero(nD)[0]
    c0 = p["b2"] + p["P0"]
    Pu = np.tensordot(h, p["W2"][:, :L * Nu].reshape(H, L, Nu)[:, U, :], 1) + c0[:L * Nu].reshape(L, Nu)[U]
    Pd = np.tensordot(h, p["W2"][:, L * Nu:].reshape(H, L, Nd)[:, D, :], 1) + c0[L * Nu:].reshape(L, Nd)[D]
    return np.linalg.det(Pu) * np.linalg.det(Pd)


def sublattice_A(lat):
    return np.array([(sum(lat.xy(i)) % 2 == 0) for i in range(lat.L)])


def sfb_single(p, nU, nD, L, Nu, Nd, Amask):
    """exact P_{S=0} amplitude of the sfb state (no J) by spin-feature bucketing + che_closed."""
    n = nU + nD
    sing = n == 1
    NsA = int(np.sum(sing & Amask))
    P = NsA + 1
    th = 2 * np.pi * np.arange(P) / P
    tot = 0.0 + 0.0j
    for w in range(NsA + 1):
        s = (2 * w - NsA) / max(NsA, 1)
        Pu, Pd, U, D, occ = che_rows(p, nU, nD, L, Nu, Nd, extra=s)
        a = Amask[occ].astype(float)
        for j in range(P):
            ph = np.exp(0.5j * th[j] * a)
            val = che_closed(Pu * ph[:, None], Pd * np.conj(ph)[:, None], U, D)
            tot += np.exp(-1j * th[j] * w + 0.5j * th[j] * NsA) * val / P
    return tot.real


# ============================================================================ trainable arms
class KAGP(R.Base):
    """phi(x) = e^{J(c)} sum_{k<K} det F^{(k)}_c[U(x), D(x)], F^{(k)}_c = sym(F0_k + reshape(V_k h(c)))."""

    def __init__(self, sec, hidden, seed, K):
        self.sec, self.K, self.H = sec, K, hidden
        L = sec.lat.L
        self.L = L
        self.Fj, self.feat = R.charge_features(sec)
        self.ru = sec.up.occ[sec.iu]
        self.rd = sec.dn.occ[sec.id]
        self.setup([("jv", (self.Fj.shape[1],)), ("F0", (K, L, L)), ("W", (hidden, 2 * L)),
                    ("b", (hidden,)), ("V", (K, hidden, L * L))])
        rng = np.random.default_rng(seed)
        _, Uh = np.linalg.eigh(-sec.lat.T)
        Ff = Uh[:, :sec.Nd] @ Uh[:, :sec.Nd].T
        noise = 0.02 if seed == 0 else 0.05
        F0 = np.empty((K, L, L))
        for k in range(K):
            Rn = rng.standard_normal((L, L))
            F0[k] = (1.0 if k == 0 else 0.3) * Ff + (noise if k == 0 else 0.1) * 0.5 * (Rn + Rn.T)
        jv = np.zeros(self.Fj.shape[1]); jv[L * (L - 1) // 2] = -1.0
        self.theta0 = self.pack(dict(jv=jv, F0=F0, W=0.3 * rng.standard_normal((hidden, 2 * L)),
                                     b=0.1 * rng.standard_normal(hidden),
                                     V=0.01 * rng.standard_normal((K, hidden, L * L))))

    @staticmethod
    def count(L, nf, H, K):
        return nf + K * L * L + H * (2 * L + 1) + K * H * L * L

    def _F(self, p, hh, k):
        L = self.L
        M = p["F0"][k][None] + (hh @ p["V"][k]).reshape(-1, L, L)
        return 0.5 * (M + M.transpose(0, 2, 1))

    def forward(self, th, chunk):
        p = self.unpack(th)
        sec = self.sec
        J = self.jastrow(p)
        expJ = np.exp(J - J.max())
        hh = np.tanh(self.feat @ p["W"].T + p["b"])
        Fs = [self._F(p, hh, k) for k in range(self.K)]
        dets = np.zeros((self.K, sec.dim))
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            c = sec.cidx[sl]
            for k in range(self.K):
                dets[k, sl] = np.linalg.det(Fs[k][c[:, None, None], self.ru[sl][:, :, None], self.rd[sl][:, None, :]])
        phi = expJ[sec.cidx] * dets.sum(0)
        return phi.reshape(sec.shape), dict(p=p, phi=phi, expJ=expJ, hh=hh, Fs=Fs)

    def vjp(self, cc, vbar, chunk):
        sec, p, L = self.sec, cc["p"], self.L
        vb = vbar.ravel()
        g = {}
        gJ = np.bincount(sec.cidx, weights=vb * cc["phi"], minlength=sec.nC)
        g["jv"] = self.Fj.T @ gJ
        w = vb * cc["expJ"][sec.cidx]
        hh = cc["hh"]
        g["F0"] = np.zeros((self.K, L, L)); g["V"] = np.zeros((self.K, self.H, L * L))
        hbar = np.zeros_like(hh)
        for k in range(self.K):
            Fbar = np.zeros(sec.nC * L * L)
            for s in range(0, sec.dim, chunk):
                sl = slice(s, min(s + chunk, sec.dim))
                c = sec.cidx[sl]
                sub = cc["Fs"][k][c[:, None, None], self.ru[sl][:, :, None], self.rd[sl][:, None, :]]
                cof = R.cofactor(sub) * w[sl][:, None, None]
                flat = (c[:, None, None] * L * L + self.ru[sl][:, :, None] * L + self.rd[sl][:, None, :]).ravel()
                Fbar += np.bincount(flat, weights=cof.ravel(), minlength=sec.nC * L * L)
            Fbar = Fbar.reshape(sec.nC, L, L)
            Mbar = 0.5 * (Fbar + Fbar.transpose(0, 2, 1)).reshape(sec.nC, -1)
            g["F0"][k] = Mbar.sum(0).reshape(L, L)
            g["V"][k] = hh.T @ Mbar
            hbar += Mbar @ p["V"][k].T
        pre = hbar * (1 - hh ** 2)
        g["W"] = pre.T @ self.feat
        g["b"] = pre.sum(0)
        return self.pack(g)


class SFB(M3.ChargeNNBF):
    """charge-input NNBF + one spin feature s_A(x) (sublattice-A polarisation of the singles)."""

    def __init__(self, sec, hidden, seed):
        super().__init__(sec, hidden, seed)
        A = sublattice_A(sec.lat)
        bu = sec.up.bits[sec.iu].astype(np.int64)
        bd = sec.dn.bits[sec.id].astype(np.int64)
        sing = (bu + bd) == 1
        NsA = (sing & A[None, :]).sum(1)
        w = ((bu * (1 - bd)) * A[None, :]).sum(1)
        self.sA = (2 * w - NsA) / np.maximum(NsA, 1)
        p = self.unpack(self.theta0)
        rng = np.random.default_rng(1000 + seed)
        p["W1"] = np.vstack([p["W1"], 0.3 * rng.standard_normal((1, hidden))])
        shapes = [(k, (2 * self.L + 1, hidden) if k == "W1" else s) for k, s in self.shapes]
        self.setup(shapes)
        self.theta0 = self.pack(p)

    def _orb(self, p, sl):
        sec, L, Nu, Nd = self.sec, self.L, self.Nu, self.Nd
        x = np.concatenate([self.cfeat[sec.cidx[sl]], self.sA[sl][:, None]], 1)
        h = np.tanh(x @ p["W1"] + p["b1"])
        out = h @ p["W2"] + p["b2"] + p["P0"]
        B = x.shape[0]
        Ou = out[:, :L * Nu].reshape(B, L, Nu)
        Od = out[:, L * Nu:].reshape(B, L, Nd)
        ru = sec.up.occ[sec.iu[sl]]
        rd = sec.dn.occ[sec.id[sl]]
        return x, h, np.take_along_axis(Ou, ru[:, :, None], 1), np.take_along_axis(Od, rd[:, :, None], 1), ru, rd


def build_arm(sec, kind, width, seed):
    L = sec.lat.L
    nf = R.charge_features(sec)[0].shape[1]
    target = R.NNBF.count(L, nf, sec.Nu, sec.Nd, width)
    if kind in ("tocchio", "kagp2"):
        K = 1 if kind == "tocchio" else 2
        H = min(range(1, 400), key=lambda h: abs(KAGP.count(L, nf, h, K) - target))
        return KAGP(sec, H, seed, K)
    if kind == "che":
        return M3.ChargeNNBF(sec, width, seed)
    if kind == "sfb":
        return SFB(sec, width, seed)
    if kind == "nnbf_SG":
        return R.NNBF(sec, width, seed)
    raise ValueError(kind)


# ============================================================================ smoke
def full_PS_amp(sec, phi, PS):
    return PS(phi)


def smoke(a):
    t0 = time.time()
    ok = True
    sec = M3.make_sector(2, 3, 6, 6.0, -0.3, "pbc", "pbc")
    log(f"2x3 torus Ne=6 dim={sec.dim}")
    PS = sec.projS(0)
    for kind in ["tocchio", "kagp2", "che", "sfb"]:
        arm = build_arm(sec, kind, 4, 0)
        rng = np.random.default_rng(3)
        th = arm.theta0 + 0.1 * rng.standard_normal(arm.nparam)
        st = M3.State(sec, arm, SPROJ[kind], None, 4096)
        phi, _ = arm.forward(th, 4096)
        psi = st.psi(phi)
        s2 = float(np.sum(psi * sec.S2(psi)) / np.sum(psi * psi))
        rel = M3.fd_check(st, th)
        good = abs(s2) < 1e-10 and rel < 1e-5
        ok &= good
        log(f"  {kind:8s} P={arm.nparam:4d} <S^2>={s2:+.2e} grad rel.err={rel:.1e} [{'PASS' if good else 'FAIL'}]")
    # kagp_single vs batched (J switched off)
    arm = build_arm(sec, "kagp2", 4, 0)
    p = arm.unpack(arm.theta0 + 0.1 * np.random.default_rng(4).standard_normal(arm.nparam))
    p["jv"][:] = 0
    phi, _ = arm.forward(arm.pack(p), 4096)
    phi = phi.ravel()
    pk = dict(F0=p["F0"], V=p["V"].reshape(arm.K, arm.H, sec.lat.L, sec.lat.L), W=p["W"], b=p["b"])
    err = max(abs(kagp_single(pk, sec.up.bits[sec.iu[i]].astype(int), sec.dn.bits[sec.id[i]].astype(int)) - phi[i])
              for i in range(0, sec.dim, 7))
    ok &= err < 1e-12
    log(f"  kagp_single vs batched: {err:.1e}")
    # che_closed / che_gl / e_k / full vector; sfb_single vs full vector (4x2 open, doublons present)
    for (Lx, Ly, Ne, bx, by) in [(2, 3, 6, "pbc", "pbc"), (4, 2, 6, "obc", "obc")]:
        s2 = M3.make_sector(Lx, Ly, Ne, 6.0, -0.3, by, bx)
        PS2 = s2.projS(0)
        L, Nu, Nd = s2.lat.L, s2.Nu, s2.Nd
        for kind in ["che", "sfb"]:
            arm = build_arm(s2, kind, 4, 1)
            p = arm.unpack(arm.theta0 + 0.2 * np.random.default_rng(5).standard_normal(arm.nparam))
            p["jv"][:] = 0
            phi, _ = arm.forward(arm.pack(p), 4096)
            ref = PS2(phi).ravel()
            e1 = e2 = e3 = 0.0
            for i in range(0, s2.dim, max(1, s2.dim // 40)):
                nU = s2.up.bits[s2.iu[i]].astype(int); nD = s2.dn.bits[s2.id[i]].astype(int)
                if kind == "che":
                    Pu, Pd, U, D, _ = che_rows(p, nU, nD, L, Nu, Nd)
                    v = che_closed(Pu, Pd, U, D).real
                    e2 = max(e2, abs(v - che_gl(Pu, Pd, U, D)))
                    e3 = max(e3, abs(v - che_closed(Pu, Pd, U, D, ek=True).real))
                else:
                    v = sfb_single(p, nU, nD, L, Nu, Nd, sublattice_A(s2.lat))
                e1 = max(e1, abs(v - ref[i]))
            scale = np.abs(ref).max()
            good = max(e1, e2, e3) < 1e-10 * max(1, scale)
            ok &= good
            log(f"  {Lx}x{Ly} {bx}/{by} Ne={Ne} {kind}: |single - full-vector P_S| {e1:.1e}"
                + (f", |closed - GL| {e2:.1e}, |closed - e_k| {e3:.1e}" if kind == "che" else "")
                + f" (max|amp| {scale:.2e}) [{'PASS' if good else 'FAIL'}]")
    log(f"SMOKE {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.1f}s)")
    return ok


# ============================================================================ cost
def timeit(fn, cfgs):
    t = time.perf_counter()
    for c in cfgs:
        fn(*c)
    return 1e6 * (time.perf_counter() - t) / len(cfgs)


def cost(a):
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for Lx in [int(v) for v in a.cost_sizes.split(",")]:
        L = Lx * Lx
        Ne = 2 * int(round(7 * L / 16)); Nu = Nd = Ne // 2
        H = a.cost_H
        F0 = rng.standard_normal((4, L, L)) / np.sqrt(L)
        pk = {K: dict(F0=F0[:K], V=0.01 * rng.standard_normal((K, H, L, L)) / np.sqrt(L),
                      W=0.3 * rng.standard_normal((H, 2 * L)), b=np.zeros(H)) for K in (1, 2, 4)}
        pc = dict(P0=rng.standard_normal(L * Ne) / np.sqrt(L), W1=rng.standard_normal((2 * L, H)) * 0.3,
                  b1=np.zeros(H), W2=0.01 * rng.standard_normal((H, L * Ne)), b2=np.zeros(L * Ne))
        pf = dict(pc); pf["W1"] = np.vstack([pc["W1"], 0.3 * rng.standard_normal((1, H))])
        pq = dict(Psi0=rng.standard_normal((2 * L, Ne)) / np.sqrt(L), A0=rng.standard_normal((L, L)) / np.sqrt(L),
                  W=rng.standard_normal((H, 2 * L)) * 0.3, b=np.zeros(H),
                  Vc=0.01 * rng.standard_normal((H, 2 * L * Ne)), Va=0.01 * rng.standard_normal((H, L * L)))
        cfgs = []
        for _ in range(a.cost_reps):
            nd_ = int(round(0.05 * Nu))
            st = rng.permutation(L)
            nU = np.zeros(L, int); nD = np.zeros(L, int)
            nU[np.concatenate([st[:nd_], st[nd_:Nu]])] = 1
            nD[np.concatenate([st[:nd_], st[Nu:Nu + Nd - nd_]])] = 1
            cfgs.append((nU, nD))
        A = np.array([(((i // Lx) + (i % Lx)) % 2 == 0) for i in range(L)])
        row = dict(Lx=Lx, L=L, Ne=Ne)
        row["nnbf_unproj"] = timeit(lambda nU, nD: nnbf_single(pc, nU, nD, L, Nu, Nd), cfgs)
        for K in (1, 2, 4):
            row[f"kagp{K}"] = timeit(lambda nU, nD: kagp_single(pk[K], nU, nD), cfgs)
        row["sqrvb"] = timeit(lambda nU, nD: sqrvb_single_g(pq, nU, nD, L, Ne), cfgs)
        row["che_closed"] = timeit(lambda nU, nD: che_closed(*che_rows(pc, nU, nD, L, Nu, Nd)[:4]), cfgs)
        row["che_gl"] = timeit(lambda nU, nD: che_gl(*che_rows(pc, nU, nD, L, Nu, Nd)[:4]), cfgs)
        if Lx <= a.sfb_max_Lx:
            row["sfb"] = timeit(lambda nU, nD: sfb_single(pf, nU, nD, L, Nu, Nd, A), cfgs[:3])
        rows.append(row)
        print("Lx=%2d L=%3d Ne=%3d " % (Lx, L, Ne) + "  ".join(f"{k} {v:10.1f}us" for k, v in row.items()
                                                                if k not in ("Lx", "L", "Ne")), flush=True)
    big = [r for r in rows][-3:]
    ne = np.log([r["Ne"] for r in big])
    slopes = {k: float(np.polyfit(ne, np.log([r[k] for r in big]), 1)[0])
              for k in big[0] if k not in ("Lx", "L", "Ne") and all(k in r for r in big)}
    print("log-log slopes in N_e (three largest sizes): " + "  ".join(f"{k} {v:.2f}" for k, v in slopes.items()))
    last = rows[-1]
    print("ratios at largest size (t / t_tocchio): " + "  ".join(
        f"{k} {last[k] / last['kagp1']:.1f}" for k in last if k not in ("Lx", "L", "Ne")))
    with open(os.path.join(a.outdir, "cost_r4.json"), "w") as fh:
        json.dump(dict(rows=rows, slopes=slopes), fh, indent=1)


# ============================================================================ tabular per-c families
def kernel(Ns):
    """site-major S = 0 projector on N_s spins (S^z = 0), patterns = up-subsets of size m."""
    m = Ns // 2
    pats = list(combinations(range(Ns), m))
    S = [set(q) for q in pats]
    K = np.array([[(-1) ** (m - len(a & b)) * math.factorial(m - len(a & b)) * math.factorial(len(a & b))
                   / math.factorial(m + 1) for b in S] for a in S])
    return pats, K


def det_and_cof(M):
    return np.linalg.det(M), R.cofactor(M.reshape(-1, *M.shape[-2:])).reshape(M.shape)


class Tab:
    """per-c tabular families on a group of charge configurations sharing N_s and N_occ.
    U, D: (nc, ns, n) row indices into the occupied-site list; sord, chi: (nc, ns) (chi in ud basis)."""

    def __init__(self, U, D, sord, chi, K):
        self.U, self.D, self.sord, self.chi, self.Kmat = U, D, sord, chi, K
        self.nc, self.ns, self.n = U.shape
        self.nocc = int(max(U.max(), D.max())) + 1 if U.size else 0
        self.cc = np.arange(self.nc)[:, None, None, None]

    def fid(self, w, chi):
        a = np.sum(chi * w, 1)
        nw = np.sum(w * w, 1); nc = np.sum(chi * chi, 1)
        F = a * a / (nw * nc)
        dF = 2 * a[:, None] * chi / (nw * nc)[:, None] - 2 * (a * a)[:, None] * w / (nw * nw * nc)[:, None]
        return F, dF

    def toc(self, th, K):
        G = th.reshape(self.nc, K, self.nocc, self.nocc)
        F = 0.5 * (G + G.transpose(0, 1, 3, 2))
        v = np.zeros((self.nc, self.ns)); cofs = []
        for k in range(K):
            sub = F[:, k][self.cc, self.U[..., :, None], self.D[..., None, :]]
            d, cof = det_and_cof(sub)
            v += d; cofs.append(cof)
        return v, cofs

    def toc_obj(self, th, K):
        v, cofs = self.toc(th, K)
        F, dF = self.fid(v, self.chi)
        g = np.zeros((self.nc, K, self.nocc * self.nocc))
        for k in range(K):
            flat = (self.cc * self.nocc * self.nocc + self.U[..., :, None] * self.nocc + self.D[..., None, :])
            gb = np.bincount(flat.ravel(), weights=(cofs[k] * dF[:, :, None, None]).ravel(),
                             minlength=self.nc * self.nocc * self.nocc).reshape(self.nc, self.nocc, self.nocc)
            g[:, k] = (0.5 * (gb + gb.transpose(0, 2, 1))).reshape(self.nc, -1)
        return F, -g.ravel()

    def puhf(self, th):
        P = th.reshape(self.nc, 2, self.nocc, self.n)
        Au = P[:, 0][self.cc[..., 0], self.U]  # (nc, ns, n, n)
        Ad = P[:, 1][self.cc[..., 0], self.D]
        du, cu = det_and_cof(Au)
        dd, cd = det_and_cof(Ad)
        v = du * dd
        w = (self.sord * v) @ self.Kmat
        return v, w, du, dd, cu, cd

    def puhf_obj(self, th):
        v, w, du, dd, cu, cd = self.puhf(th)
        F, dF = self.fid(w, self.sord * self.chi)
        dv = (dF @ self.Kmat) * self.sord
        g = np.zeros((self.nc, 2, self.nocc, self.n))
        np.add.at(g[:, 0], (self.cc[..., 0, 0][:, :, None].repeat(self.ns, 1), self.U),
                  cu * (dv * dd)[:, :, None, None])
        np.add.at(g[:, 1], (self.cc[..., 0, 0][:, :, None].repeat(self.ns, 1), self.D),
                  cd * (dv * du)[:, :, None, None])
        return F, -g.ravel()

    def fit(self, family, restarts, maxiter, seed=0):
        rng = np.random.default_rng(seed)
        best = np.full(self.nc, -1.0); bestw = np.zeros((self.nc, self.ns))
        for r in range(restarts):
            if family.startswith("tocchio"):
                K = int(family[-1])
                th0 = rng.standard_normal(self.nc * K * self.nocc * self.nocc)
                fun = lambda th: (lambda F, g: (float(np.sum(1 - F)), g))(*self.toc_obj(th, K))  # noqa: E731
            else:
                th0 = rng.standard_normal(self.nc * 2 * self.nocc * self.n)
                fun = lambda th: (lambda F, g: (float(np.sum(1 - F)), g))(*self.puhf_obj(th))  # noqa: E731
            res = minimize(fun, th0, jac=True, method="L-BFGS-B",
                           options=dict(maxiter=maxiter, maxcor=30, ftol=1e-16, gtol=1e-12))
            if family.startswith("tocchio"):
                w = self.toc(res.x, K)[0]; F, _ = self.fid(w, self.chi)
            else:
                w = self.puhf(res.x)[1]; F, _ = self.fid(w, self.sord * self.chi)
                w = self.sord * w  # back to ud
            imp = F > best
            best[imp] = F[imp]; bestw[imp] = w[imp]
        return best, bestw


def charge_groups(sec):
    """group basis states by charge configuration: per N_s, per N_occ (= N_s + n_d) arrays."""
    L = sec.lat.L
    bu = sec.up.bits[sec.iu].astype(np.int64)
    bd = sec.dn.bits[sec.id].astype(np.int64)
    n = bu + bd
    cu_gt = np.cumsum(bu[:, ::-1], 1)[:, ::-1] - bu
    sord = 1.0 - 2.0 * (np.sum(bd * cu_gt, 1) % 2)
    Ns = (n == 1).sum(1)
    out = {}
    for ns_ in np.unique(Ns):
        idx = np.nonzero(Ns == ns_)[0]
        m = int(ns_) // 2
        pats, Kmat = kernel(int(ns_))
        lut = {sum(1 << q for q in pat): j for j, pat in enumerate(pats)}
        sing = np.nonzero(n[idx] == 1)[1].reshape(len(idx), ns_) if ns_ else np.zeros((len(idx), 0), np.int64)
        su = np.take_along_axis(bu[idx], sing, 1) * (1 - np.take_along_axis(bd[idx], sing, 1))
        code = su @ (1 << np.arange(ns_, dtype=np.int64))
        pj = np.array([lut[int(cv)] for cv in code])
        cs, cl = np.unique(sec.cidx[idx], return_inverse=True)
        nocc = int((n[idx[0]] > 0).sum())
        occ = np.nonzero(n[idx] > 0)[1].reshape(len(idx), nocc)
        bu_o = np.take_along_axis(bu[idx], occ, 1); bd_o = np.take_along_axis(bd[idx], occ, 1)
        Ur = np.nonzero(bu_o)[1].reshape(len(idx), sec.Nu)
        Dr = np.nonzero(bd_o)[1].reshape(len(idx), sec.Nd)
        nc, nsg = len(cs), len(pats)
        assert nc * nsg == len(idx), "orbit bookkeeping"
        S = np.zeros((nc, nsg), np.int64); S[cl, pj] = idx
        U = np.zeros((nc, nsg, sec.Nu), np.int64); U[cl, pj] = Ur
        D = np.zeros((nc, nsg, sec.Nd), np.int64); D[cl, pj] = Dr
        out[int(ns_)] = dict(state=S, U=U, D=D, sord=sord[S], K=Kmat, m=m)
    return out


def catalan(m):
    return math.comb(2 * m, m) // (m + 1)


def dims(a):
    """Jacobian rank of tabular maps at random parameters (no doublons, N_occ = N_s)."""
    rng = np.random.default_rng(0)
    print(" N_s  singlet_dim  rank(tocchio1)  rank(tocchio2)  rank(tocchio3)  rank(puhf)")
    out = []
    for Ns in (4, 6, 8, 10, 12):
        m = Ns // 2
        pats, Kmat = kernel(Ns)
        U = np.array([list(p) for p in pats])[None]
        D = np.array([[i for i in range(Ns) if i not in p] for p in pats])[None]
        nU = np.zeros((len(pats), Ns), int)
        for j, p in enumerate(pats):
            nU[j, list(p)] = 1
        nD = 1 - nU
        above = np.cumsum(nU[:, ::-1], 1)[:, ::-1] - nU
        sord = (1.0 - 2.0 * (np.sum(nD * above, 1) % 2))[None]
        tb = Tab(U, D, sord, np.ones((1, len(pats))), Kmat)
        row = dict(Ns=Ns, singlet=catalan(m))
        for fam in ("tocchio1", "tocchio2", "tocchio3", "puhf"):
            if fam.startswith("tocchio"):
                K = int(fam[-1])
                f = lambda th: tb.toc(th, K)[0][0]  # noqa: E731
                th = rng.standard_normal(K * Ns * Ns)
            else:
                f = lambda th: tb.puhf(th)[1][0]  # noqa: E731
                th = rng.standard_normal(2 * Ns * m)
            h = 1e-6
            J = np.array([(f(th + h * e) - f(th - h * e)) / (2 * h) for e in np.eye(len(th))]).T
            s = np.linalg.svd(J, compute_uv=False)
            row[fam] = int(np.sum(s > 1e-7 * s[0]))
            if fam.startswith("tocchio"):  # singlet check of the tabular Tocchio family
                v = f(th)
                row["toc_singlet_err"] = float(np.abs(Kmat @ (sord[0] * v) - sord[0] * v).max() / np.abs(v).max())
        out.append(row)
        print(f" {Ns:3d}  {row['singlet']:11d}  {row['tocchio1']:14d}  {row['tocchio2']:14d}  "
              f"{row['tocchio3']:14d}  {row['puhf']:10d}   (tocchio singlet check {row['toc_singlet_err']:.1e})")
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "dims_r4.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def tag_of(a):
    return a.tag or f"L{a.Lx}x{a.Ly}x{a.bc_x}{a.bc_y}_N{a.Ne}_U{a.U:g}_tp{a.tp:g}"


def get_psi0(a, sec, sg):
    os.makedirs(a.outdir, exist_ok=True)
    fn = os.path.join(a.outdir, f"psi0_{tag_of(a)}.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        sg.set_irrep(z["P0"])
        return z["P0"], float(z["E0"])
    t = time.time()
    P0, E0, s20, lam, nxt = M3.ground_state(sec, sg)
    log(f"ED: E0={E0:.10f} <S^2>={s20:.1e} irrep={lam} next level {nxt[1]:.6f} ({time.time() - t:.0f}s)")
    np.savez(fn, P0=P0, E0=E0)
    return P0, E0


def agp(a):
    t0 = time.time()
    sec = M3.make_sector(a.Lx, a.Ly, a.Ne, a.U, a.tp, a.bc_y, a.bc_x)
    sg = M3.SG(sec)
    P0, E0 = get_psi0(a, sec, sg)
    psi0 = P0.ravel()
    groups = charge_groups(sec)
    fams = a.agp_families.split(",")
    fit = {f: np.zeros(sec.dim) for f in fams}
    res = dict(tag=tag_of(a), E0=E0, dim=sec.dim, groups=[])
    for Ns, gdat in sorted(groups.items()):
        S = gdat["state"]
        chi = psi0[S]
        pc = np.sum(chi * chi, 1)
        keep = pc > 1e-14
        row = dict(Ns=Ns, nc=int(S.shape[0]), nc_weighted=int(keep.sum()), weight=float(pc.sum()),
                   singlet_dim=catalan(gdat["m"]), orbit=int(S.shape[1]))
        for f in fams:
            if S.shape[1] == 1 or keep.sum() == 0 or Ns < a.agp_min_Ns:
                # orbit of size 1, no weight, or N_s below --agp-min-Ns (locally complete by --mode dims):
                # chi itself is used (F_c = 1); row["skipped"] records it
                Fc = np.ones(S.shape[0]); w = chi.copy()
                row["skipped"] = bool(Ns < a.agp_min_Ns and S.shape[1] > 1)
            else:
                Fc = np.ones(S.shape[0]); w = chi.copy()
                # split by number of occupied sites (doublon count fixed by N_s at fixed N_e, so one block)
                tb = Tab(gdat["U"][keep], gdat["D"][keep], gdat["sord"][keep], chi[keep], gdat["K"])
                t1 = time.time()
                Fk, wk = tb.fit(f, a.agp_restarts, a.agp_maxiter)
                Fc[keep] = Fk; w[keep] = wk
                log(f"  N_s={Ns:2d} {f:9s}: {int(keep.sum())} configs, median F={np.median(Fk):.12f} "
                    f"min F={Fk.min():.6f} ({time.time() - t1:.0f}s)")
            alpha = np.sum(chi * w, 1) / np.maximum(np.sum(w * w, 1), 1e-300)
            fit[f][S.ravel()] = (alpha[:, None] * w).ravel()
            row[f] = dict(median_F=float(np.median(Fc[keep])) if keep.any() else 1.0,
                          min_F=float(Fc[keep].min()) if keep.any() else 1.0,
                          weighted_infid=float(np.sum(pc * (1 - Fc))))
        res["groups"].append(row)
    for f in fams:
        v = fit[f].reshape(sec.shape)
        nrm = np.sum(v * v)
        E = float(np.sum(v * sec.H(v)) / nrm)
        s2 = float(np.sum(v * sec.S2(v)) / nrm)
        res[f] = dict(infid=float(1 - np.sum(v * P0) ** 2 / nrm), E=E, eps=(E - E0) / abs(E0), S2=s2)
        log(f"{f:9s}: total infidelity {res[f]['infid']:.3e}  eps={res[f]['eps']:.3e}  <S^2>={s2:.1e}")
    print(f"{'N_s':>4} {'#c':>6} {'weight':>9} {'singlet':>7} " + " ".join(f"{f + ' (wInfid, minF)':>30}" for f in fams))
    for g in res["groups"]:
        print(f"{g['Ns']:4d} {g['nc']:6d} {g['weight']:9.2e} {g['singlet_dim']:7d} "
              + " ".join(f"{g[f]['weighted_infid']:16.2e} {g[f]['min_F']:12.6f}" for f in fams))
    res["seconds"] = time.time() - t0
    with open(os.path.join(a.outdir, f"agp_{tag_of(a)}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


# ============================================================================ lemma checks
def lemma(a):
    rng = np.random.default_rng(7)
    ok = True
    for nv in (3, 4):
        L = 2 * nv
        sec = R.Sector(M3.CylLattice(nv, 2, 0.0, "obc", "obc"), nv, nv, 0.0)
        PS = sec.projS(0)
        bu = sec.up.bits[sec.iu].astype(int); bd = sec.dn.bits[sec.id].astype(int)
        cu_gt = np.cumsum(bu[:, ::-1], 1)[:, ::-1] - bu
        sord = 1.0 - 2.0 * (np.sum(bd * cu_gt, 1) % 2)
        nodbl = (bu + bd).max(1) == 1
        valid = nodbl & np.all(bu[:, 0::2] + bu[:, 1::2] == 1, axis=1)
        bits = bu[:, 1::2]  # variable a = 1 iff site 2a+1 is up
        kk = bits.sum(1)
        Kk = np.array([(-1) ** k * math.factorial(k) * math.factorial(nv - k) / math.factorial(nv + 1)
                       for k in range(nv + 1)])
        xref = np.nonzero(valid & (kk == 0))[0][0]
        for trial in range(15):
            ncl = rng.integers(1, 3 * nv)
            cl = [(rng.choice(nv, 3, replace=False), rng.integers(0, 2, 3)) for _ in range(ncl)]
            sat = np.ones(sec.dim, bool)
            for vars_, negs in cl:
                lit = np.zeros(sec.dim, bool)
                for v, ng in zip(vars_, negs):
                    lit |= (bits[:, v] == 1) ^ bool(ng)
                sat &= lit
            # determinant realisation: orbitals select the occupied sites, one entry scaled by g(x')
            g = np.where(valid & sat, 1.0 / Kk[kk], 0.0) * sord
            phi = np.empty(sec.dim)
            for i in range(sec.dim):
                U = np.nonzero(bu[i])[0]; D = np.nonzero(bd[i])[0]
                Pu = np.zeros((L, nv)); Pd = np.zeros((L, nv))
                Pu[U, np.arange(nv)] = 1.0; Pd[D, np.arange(nv)] = 1.0
                Pu[U[0], 0] = g[i]
                phi[i] = np.linalg.det(Pu[U]) * np.linalg.det(Pd[D])
            val = sord[xref] * PS(phi.reshape(sec.shape)).ravel()[xref]
            assigns = np.array([[(t >> v) & 1 for v in range(nv)] for t in range(2 ** nv)])
            cnt = 0
            for asg in assigns:
                cnt += all(any((asg[v] == 1) ^ bool(ng) for v, ng in zip(vs, ngs)) for vs, ngs in cl)
            good = abs(val - cnt) < 1e-9
            ok &= good
            if trial < 4 or not good:
                log(f"  n_v={nv} trial {trial}: {ncl} clauses, (P_S phi)(x_ref) = {val:.12f}, #SAT = {cnt} "
                    f"[{'PASS' if good else 'FAIL'}]")
    log(f"#SAT reduction {'PASS' if ok else 'FAIL'} (all trials)")
    # timing: bucketed sfb amplitude vs exponential orbit sum (spin-resolved NNBF)
    for (Lx, Ly) in [(4, 3), (4, 4), (5, 4), (6, 4)]:
        L = Lx * Ly
        Ne = 2 * int(round(7 * L / 16)); Nu = Nd = Ne // 2
        H = 8
        pc = dict(P0=rng.standard_normal(L * Ne) / np.sqrt(L), W1=rng.standard_normal((2 * L + 1, H)) * 0.3,
                  b1=np.zeros(H), W2=0.01 * rng.standard_normal((H, L * Ne)), b2=np.zeros(L * Ne))
        pn = dict(pc); pn["W1"] = pc["W1"][:2 * L]
        A = np.array([(((i // Ly) + (i % Ly)) % 2 == 0) for i in range(L)])
        st = rng.permutation(L)
        nU = np.zeros(L, int); nD = np.zeros(L, int)
        nU[st[:Nu]] = 1; nD[st[Nu:Nu + Nd]] = 1
        t = time.perf_counter(); sfb_single(pc, nU, nD, L, Nu, Nd, A); t1 = time.perf_counter() - t
        t = time.perf_counter(); _, B = M3.nnbf_orbit_single(pn, nU, nD, L, Nu, Nd); t2 = time.perf_counter() - t
        log(f"  L={L:3d} Ne={Ne:3d}: sfb (bucketed closed form) {1e3 * t1:9.1f} ms   spin-resolved orbit sum "
            f"({B} terms) {1e3 * t2:10.1f} ms")
    return ok


# ============================================================================ accuracy test
def run(a):
    os.makedirs(a.outdir, exist_ok=True)
    fn = os.path.join(a.outdir, f"{tag_of(a)}_w{a.nnbf_hidden}_{a.ansatz}_seed{a.seed}.json")
    if os.path.exists(fn) and not a.overwrite:
        log(f"exists: {fn}")
        return
    sec = M3.make_sector(a.Lx, a.Ly, a.Ne, a.U, a.tp, a.bc_y, a.bc_x)
    sg = M3.SG(sec)
    P0, E0 = get_psi0(a, sec, sg)
    arm = build_arm(sec, a.ansatz, a.nnbf_hidden, a.seed)
    st = M3.State(sec, arm, SPROJ[a.ansatz], sg, a.chunk)
    log(f"{a.ansatz}: P={arm.nparam} (target NNBF width {a.nnbf_hidden})")
    th, E, it, sec_ = R.optimise(st, arm.theta0, a.maxiter, a.max_hours)
    phi, _ = arm.forward(th, a.chunk)
    psi = st.psi(phi)
    nrm = np.sum(psi * psi)
    out = dict(ansatz=a.ansatz, seed=a.seed, P=arm.nparam, E=E, E0=E0, eps=(E - E0) / abs(E0), iters=it,
               seconds=sec_, capped=bool(sec_ > 3600 * a.max_hours * 0.999),
               S2=float(np.sum(psi * sec.S2(psi)) / nrm), overlap2=float(np.sum(psi * P0) ** 2 / nrm))
    log(json.dumps(out))
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=1)


def report(a):
    pat = os.path.join(a.outdir, f"{tag_of(a)}_w{a.nnbf_hidden}_*_seed*.json")
    rs = {}
    for f in glob.glob(pat):
        r = json.load(open(f))
        rs.setdefault(r["ansatz"], []).append(r)
    med = {k: float(np.median([r["eps"] for r in v])) for k, v in rs.items()}
    for k in RUN_ARMS:
        if k in rs:
            e = [r["eps"] for r in rs[k]]
            print(f"{k:8s} n={len(e)} P={rs[k][0]['P']:5d} m(eps)={med[k]:.3e} [{min(e):.2e}, {max(e):.2e}] "
                  f"max|S2|={max(abs(r['S2']) for r in rs[k]):.1e} capped={any(r['capped'] for r in rs[k])}")
    full = all(len(rs.get(k, [])) >= 3 for k in RUN_ARMS)
    cj = json.load(open(os.path.join(a.outdir, "cost_r4.json")))["rows"][-1]
    exact = all(max(abs(r["S2"]) for r in rs[k]) < 1e-10 for k in rs)
    # Route 4.2 (kagp2)
    if not full or not exact:
        v2 = "AMBIGUOUS (incomplete or exactness violated)"
        v1 = v2
    else:
        gc = cj["kagp2"] < cj["che_closed"]
        if not gc:
            v2 = "FAIL (cost)"
        elif med["kagp2"] >= 0.9 * med["tocchio"] or med["kagp2"] > med["che"]:
            v2 = "FAIL (accuracy)"
        elif med["kagp2"] <= 0.7 * med["tocchio"] and med["kagp2"] <= med["che"]:
            v2 = "PASS"
        else:
            v2 = "AMBIGUOUS"
        if med["sfb"] >= 0.95 * med["che"]:
            v1 = "FAIL (the spin feature buys nothing)"
        elif med["sfb"] <= 0.7 * med["che"]:
            v1 = "PASS"
        else:
            v1 = "AMBIGUOUS"
    print(f"premise: m(eps_nnbf_SG) = {med.get('nnbf_SG', float('nan')):.2e} (NO-TARGET if < 1e-4)")
    af = os.path.join(a.outdir, f"agp_{tag_of(a)}.json")
    if os.path.exists(af):
        ag = json.load(open(af))
        e1, e2 = ag["tocchio1"]["eps"], ag["tocchio2"]["eps"]
        prem = e1 >= 1e-4 and e2 <= 0.5 * e1
        print(f"Route 4.2 premise (tabular): eps(tocchio1)={e1:.2e}, eps(tocchio2)={e2:.2e}, eps(puhf)="
              f"{ag['puhf']['eps']:.2e} -> {'MET' if prem else 'NOT MET: Route 4.2 is NO-TARGET on this cluster'}")
        if not prem:
            v2_override = "NO-TARGET (tabular premise not met)"
    else:
        print("Route 4.2 premise: agp json missing (run submit_agp.sbatch first)")
    print(f"che vs nnbf_SG (charge-only vs spin-resolved, both projected): ratio "
          f"{med.get('che', np.nan) / med.get('nnbf_SG', np.nan):.2f}")
    print(f"Route 4.2 kagp2: {locals().get('v2_override', v2)}")
    print(f"Route 4.1 sfb  : {v1}")


# ============================================================================ main
def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="P4.1 round-4 tests (see module docstring)")
    p.add_argument("--mode", required=True, choices=["smoke", "cost", "dims", "lemma", "agp", "run", "report"])
    p.add_argument("--Lx", type=int, default=4)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--Ne", type=int, default=10)
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--tp", type=float, default=-0.25)
    p.add_argument("--bc-x", default="obc", choices=["obc", "pbc"], dest="bc_x")
    p.add_argument("--bc-y", default="obc", choices=["obc", "pbc"], dest="bc_y")
    p.add_argument("--ansatz", default="kagp2", choices=RUN_ARMS)
    p.add_argument("--nnbf-hidden", type=int, default=16, dest="nnbf_hidden")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--max-hours", type=float, default=20.0, dest="max_hours")
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--cost-sizes", default="8,12,16,20,24", dest="cost_sizes")
    p.add_argument("--cost-reps", type=int, default=20, dest="cost_reps")
    p.add_argument("--cost-H", type=int, default=8, dest="cost_H")
    p.add_argument("--sfb-max-Lx", type=int, default=6, dest="sfb_max_Lx")
    p.add_argument("--agp-restarts", type=int, default=3, dest="agp_restarts")
    p.add_argument("--agp-maxiter", type=int, default=3000, dest="agp_maxiter")
    p.add_argument("--agp-min-Ns", type=int, default=0, dest="agp_min_Ns")
    p.add_argument("--agp-families", default="tocchio1,tocchio2,puhf", dest="agp_families")
    p.add_argument("--outdir", default=os.path.join(here, "results"))
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    dict(smoke=smoke, cost=cost, dims=dims, lemma=lemma, agp=agp, run=run, report=report)[a.mode](a)


if __name__ == "__main__":
    main()
