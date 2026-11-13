#!/usr/bin/env python3
"""
P4.1 round 5 (fix): rank-ladder test of charge-conditioned projected UHF mixtures.
=================================================================================

The original round-5 script (che_rbm M=2 vs che_k4) is archived in logs/r5_original/. This version applies
the round-5 critic's fix-then-submit path.

Local copies only: fix_test.py, r4_test.py, r3_micro_test.py, r2_micro_test.py are verbatim copies of
tests/P4.1_cheap_su2_fix/. Data READ (not code): the cached ED state psi0_<tag>.npz from
../P4.1_cheap_su2_r4/results; the reference run JSONs of `che` and `nnbf_SG` from ../P4.1_cheap_su2_fix/results
and their E_best trajectories from ../P4.1_cheap_su2_fix/logs/P4.1fix_451886_*.out (job 451886: same instance,
same common reference start, same optimiser, 2000 L-BFGS iterations, none capped).

What the arms are (all exact S = 0 for every theta, all trained with P_G and exact P_S)
-------------------------------------------------------------------------------------
Every arm is a projected mixture of charge-conditioned UHF determinants (ResHF / multi-component PHF with a
charge backflow). The rank K is the number of Gaussian components per charge configuration c, and the exact
P_S of the mixture costs K closed-form Che evaluations per amplitude.
  che_rbmM  (M = 1, 2, 3; rank 2^M)  phi = e^{J(c)} det Phi_up(c)[U] det Phi_dn(c)[D] prod_{m<M} 2cosh(b_m + w_m(c).sigma).
            Expanding 2cosh a = sum_{s=+-1} e^{s a}: each hidden configuration s is ONE charge-conditioned UHF
            determinant with diagonal up/down row scalings e^{+-f_s(c)}, f_s = sum_m s_m w_m(c). It takes no
            spin input that che lacks; the components VARY with c (row scalings w_m(c) = h(c)[:Hr] Wr_m + w0_m).
  che_kK    (K = 2, 4, 8; rank K)    phi = e^{J} sum_k a_k det(P0_k + B(c))[U] det(...)[D], one shared backflow
            B(c); the components differ only by c-independent reference offsets P0_k and coefficients a_k.
  che_kh4   (rank 4, containing foil) phi = e^{J} sum_k a_k det(P0_k + h(c) W2_k)[U] det(...)[D]: shared trunk
            h(c), each component with its OWN backflow head W2_k. Contains che_k4 (W2_k equal) at equal trunk.
  che       (drift control only) the round-4-fix che arm (H = 16, 2637 parameters), seed 0, run to 4000 iterations.
Parameter matching: all new arms lie within +-1% of P_C = 2589 (see ARM_SPEC; the hidden width H and, for
che_rbmM, the field-head width Hr <= H are the knobs). che and nnbf_SG have 2637 (+1.9%, conservative).
che_kh4 has no separate b2 (P0_k is its head bias); the other arms inherit che's duplicate pair P0 + b2.
Common initialisation: the che start (Fermi sea x Gutzwiller, W2 ~ 1e-3, W1 ~ N(0, 1/fan_in)); every component
starts near it with small independent noise of relative size EPS = 0.05 (P0_k = P0 + EPS rms(P0) xi_k,
a_k = 1 for k-arms; b_m = 0, per-site fields of order EPS for rbm arms).

Code paths: accuracy runs (`run`) evaluate the full 627k vector and apply the exact P_S of the sector
(sec.projS); the cost runs (`cost`) time the per-amplitude polynomial algorithm (K closed-form Che
evaluations). `smoke` (b) checks that the two agree to 1e-10 on small clusters.

Usage
-----
  python r5_test.py --mode MODE [options]
--mode {smoke, cost, timing, run, report, check}
  smoke  : <= 1 min login-node code check on 4x2 open N_e = 6 and 2x3 torus (every arm, small widths).
  cost   : per-amplitude wall time (single core) on L = Lx^2 tori; writes results/cost_r5.json.
  timing : one energy + gradient of che and of every new arm on the production instance, in ONE call;
           writes results/timing_r5.json (used to size --max-hours and --time).
  run    : train one arm; writes results/<tag>_w16_<arm>_seed<seed>.json (E_best trajectory every 50
           iterations included) and the final parameters <...>_theta.npy (+ shapes in the JSON).
           --ansatz che runs the drift control (label che_drift<maxiter>).
  report : premises and pre-committed verdicts (leaf artifact, Round 5 fix).
  check  : report on synthetic result sets, one per verdict path.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fix_test as F  # noqa: E402  (local verbatim copy)

T, M3, R = F.T, F.M3, F.R
log = R.log

# ---------------------------------------------------------------------------- arms (pre-committed)
ARM_SPEC = {
    "che_rbm1": dict(cls="rbm", M=1, H=15, Hr=7),
    "che_rbm2": dict(cls="rbm", M=2, H=14, Hr=9),
    "che_rbm3": dict(cls="rbm", M=3, H=13, Hr=10),
    "che_k2": dict(cls="k", K=2, H=15),
    "che_k4": dict(cls="k", K=4, H=13),
    "che_k8": dict(cls="k", K=8, H=10),
    "che_kh4": dict(cls="kh", K=4, H=4),
}
NEW_ARMS = list(ARM_SPEC)
REF_ARMS = ["che", "nnbf_SG"]
FAMILIES = {"rbm (c-varying row scalings)": ["che_rbm1", "che_rbm2", "che_rbm3"],
            "k (fixed offsets, shared backflow)": ["che_k2", "che_k4", "che_k8"],
            "kh4 (containing foil)": ["che_kh4"]}
P_C = 2589           # parameter centre; every new arm within +-1%
EPS = 0.05           # common component-noise scale


def rank(kind):
    s = ARM_SPEC[kind]
    return 2 ** s["M"] if s["cls"] == "rbm" else s["K"]


# pre-committed thresholds (artifact, Round 5 fix)
N_FINAL = 2000       # verdict iteration
N_CHECK = 1000       # every verdict must also hold here
N_DRIFT = 4000       # che drift-control run length
PASS_GAIN = 0.5      # fraction of the che -> nnbf_SG gap closed, on medians AND on best seeds, at the best rank
FAIL_GAIN = 0.2
STAB_TOL = 0.2       # "increasing or stable with rank": g_m never drops by more than 0.2 between successive ranks
CVARY_CREDIT = 0.8   # rbm credited with c-varying components only if m(rbm_r) <= 0.8 m(k_r) and ranges disjoint
COST_FACTOR = 1.5    # t(arm) <= 1.5 * rank * t(che_closed) at the largest size
COST_SLOPE = 3.0     # log-log slope in N_e over the three largest sizes


# ============================================================================ trainable arms
def fastcof(A):
    """batched cofactor det(A) A^{-T}: LU inverse where the Hadamard ratio exceeds 1e-8, SVD (R.cofactor) elsewhere."""
    out = np.empty_like(A)
    d = np.linalg.det(A)
    had = np.abs(d) / np.prod(np.linalg.norm(A, axis=2), axis=1)  # Hadamard ratio, 1 = orthogonal rows
    ok = np.isfinite(had) & (had > 1e-8)
    if ok.any():
        Ao = A[ok]
        out[ok] = d[ok][:, None, None] * np.swapaxes(np.linalg.inv(Ao), 1, 2)
    if (~ok).any():
        out[~ok] = R.cofactor(A[~ok])
    return out


class CheRBM(M3.ChargeNNBF):
    """che x prod_m 2cosh(b_m + w_m(c).sigma) = 2^M row-scaled charge-conditioned UHF determinants."""

    def __init__(self, sec, hidden, seed, M, Hr):
        super().__init__(sec, hidden, seed)
        self.M, self.Hr = M, Hr
        L = self.L
        bu = sec.up.bits[sec.iu].astype(np.float32)
        bd = sec.dn.bits[sec.id].astype(np.float32)
        self.sig = bu - bd
        p = self.unpack(self.theta0)
        p["Wr"] = np.zeros((Hr, M * L)); p["w0"] = np.zeros(M * L); p["br"] = np.zeros(M)
        self.setup(self.shapes + [("Wr", (Hr, M * L)), ("w0", (M * L,)), ("br", (M,))])
        self.theta0 = self.pack(p)

    @staticmethod
    def count(L, nf, Nu, Nd, H, M, Hr):
        return R.NNBF.count(L, nf, Nu, Nd, H) + Hr * M * L + M * L + M

    def _args(self, p, h, sl):
        B = h.shape[0]
        Wm = (h[:, :self.Hr] @ p["Wr"] + p["w0"]).reshape(B, self.M, self.L)
        return np.einsum("bml,bl->bm", Wm, self.sig[sl]) + p["br"]

    def forward(self, th, chunk):
        p = self.unpack(th)
        sec = self.sec
        J = self.jastrow(p)
        expJ = np.exp(J - J.max())
        phi = np.empty(sec.dim)
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            _, h, Au, Ad, _, _ = self._orb(p, sl)
            cor = np.prod(2.0 * np.cosh(self._args(p, h, sl)), 1)
            phi[sl] = np.linalg.det(Au) * np.linalg.det(Ad) * expJ[sec.cidx[sl]] * cor
        return phi.reshape(sec.shape), dict(p=p, phi=phi, expJ=expJ)

    def vjp(self, cc, vbar, chunk):
        sec, p, L, Nu, Nd, M, Hr = self.sec, cc["p"], self.L, self.Nu, self.Nd, self.M, self.Hr
        vb = vbar.ravel()
        g = {k: np.zeros(s) for k, s in self.shapes}
        gJ = np.bincount(sec.cidx, weights=vb * cc["phi"], minlength=sec.nC)
        g["jv"] = self.Fj.T @ gJ
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            x, h, Au, Ad, ru, rd = self._orb(p, sl)
            B = x.shape[0]
            du, dd = np.linalg.det(Au), np.linalg.det(Ad)
            a = self._args(p, h, sl)
            cor = np.prod(2.0 * np.cosh(a), 1)
            eJ = cc["expJ"][sec.cidx[sl]]
            w = vb[sl] * eJ * cor
            Gu = fastcof(Au) * (w * dd)[:, None, None]
            Gd = fastcof(Ad) * (w * du)[:, None, None]
            dOu = np.zeros((B, L, Nu)); dOd = np.zeros((B, L, Nd))
            bi = np.arange(B)[:, None]
            np.add.at(dOu, (bi, ru), Gu)
            np.add.at(dOd, (bi, rd), Gd)
            dout = np.concatenate([dOu.reshape(B, -1), dOd.reshape(B, -1)], 1)
            g["P0"] += dout.sum(0)
            g["b2"] += dout.sum(0)
            g["W2"] += h.T @ dout
            q = vb[sl] * du * dd * eJ * cor
            da = q[:, None] * np.tanh(a)
            g["br"] += da.sum(0)
            dW = (da[:, :, None] * self.sig[sl][:, None, :]).reshape(B, M * L)
            g["w0"] += dW.sum(0)
            g["Wr"] += h[:, :Hr].T @ dW
            dh = dout @ p["W2"].T
            dh[:, :Hr] += dW @ p["Wr"].T
            dpre = dh * (1 - h ** 2)
            g["W1"] += x.T @ dpre
            g["b1"] += dpre.sum(0)
        return self.pack(g)


class CheK(M3.ChargeNNBF):
    """sum_k a_k che-determinant with reference P0_k and SHARED charge backflow; exact P_S by the State."""

    def __init__(self, sec, hidden, seed, K):
        super().__init__(sec, hidden, seed)
        self.K = K
        p = self.unpack(self.theta0)
        nout = p["P0"].shape[0]
        p["P0"] = np.tile(p["P0"], (K, 1))
        p["ak"] = np.ones(K - 1)
        self.setup([("P0", (K, nout))] + [(k, s) for k, s in self.shapes if k != "P0"] + [("ak", (K - 1,))])
        self.theta0 = self.pack(p)

    @staticmethod
    def count(L, nf, Nu, Nd, H, K):
        return R.NNBF.count(L, nf, Nu, Nd, H) + (K - 1) * L * (Nu + Nd) + (K - 1)

    def _outs(self, p, h):
        shared = h @ p["W2"] + p["b2"]
        return [shared + p["P0"][k] for k in range(self.K)]

    def _orbk(self, p, sl):
        sec, L, Nu, Nd = self.sec, self.L, self.Nu, self.Nd
        x = self.cfeat[sec.cidx[sl]]
        h = np.tanh(x @ p["W1"] + p["b1"])
        B = x.shape[0]
        ru = sec.up.occ[sec.iu[sl]]
        rd = sec.dn.occ[sec.id[sl]]
        As = []
        for out in self._outs(p, h):
            Ou = out[:, :L * Nu].reshape(B, L, Nu)
            Od = out[:, L * Nu:].reshape(B, L, Nd)
            As.append((np.take_along_axis(Ou, ru[:, :, None], 1), np.take_along_axis(Od, rd[:, :, None], 1)))
        return x, h, As, ru, rd

    def coef(self, p):
        return np.concatenate([[1.0], p["ak"]])

    def terms(self, th, chunk):
        p = self.unpack(th)
        sec = self.sec
        J = self.jastrow(p)
        expJ = np.exp(J - J.max())
        out = np.zeros((self.K, sec.dim))
        a = self.coef(p)
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            _, _, As, _, _ = self._orbk(p, sl)
            for k, (Au, Ad) in enumerate(As):
                out[k, sl] = a[k] * np.linalg.det(Au) * np.linalg.det(Ad) * expJ[sec.cidx[sl]]
        return out

    def forward(self, th, chunk):
        p = self.unpack(th)
        sec = self.sec
        J = self.jastrow(p)
        expJ = np.exp(J - J.max())
        a = self.coef(p)
        phi = np.zeros(sec.dim)
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            _, _, As, _, _ = self._orbk(p, sl)
            for k, (Au, Ad) in enumerate(As):
                phi[sl] += a[k] * np.linalg.det(Au) * np.linalg.det(Ad)
            phi[sl] *= expJ[sec.cidx[sl]]
        return phi.reshape(sec.shape), dict(p=p, phi=phi, expJ=expJ)

    def vjp(self, cc, vbar, chunk):
        sec, p, L, Nu, Nd = self.sec, cc["p"], self.L, self.Nu, self.Nd
        vb = vbar.ravel()
        a = self.coef(p)
        g = {k: np.zeros(s) for k, s in self.shapes}
        gJ = np.bincount(sec.cidx, weights=vb * cc["phi"], minlength=sec.nC)
        g["jv"] = self.Fj.T @ gJ
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            x, h, As, ru, rd = self._orbk(p, sl)
            B = x.shape[0]
            eJ = cc["expJ"][sec.cidx[sl]]
            bi = np.arange(B)[:, None]
            dh = np.zeros_like(h)
            dtot = np.zeros((B, L * (Nu + Nd)))
            for k, (Au, Ad) in enumerate(As):
                du, dd = np.linalg.det(Au), np.linalg.det(Ad)
                if k > 0:
                    g["ak"][k - 1] += np.sum(vb[sl] * eJ * du * dd)
                w = vb[sl] * eJ * a[k]
                Gu = fastcof(Au) * (w * dd)[:, None, None]
                Gd = fastcof(Ad) * (w * du)[:, None, None]
                dOu = np.zeros((B, L, Nu)); dOd = np.zeros((B, L, Nd))
                np.add.at(dOu, (bi, ru), Gu)
                np.add.at(dOd, (bi, rd), Gd)
                dout = np.concatenate([dOu.reshape(B, -1), dOd.reshape(B, -1)], 1)
                g["P0"][k] += dout.sum(0)
                if "b2" in g:  # shared head (CheK)
                    dtot += dout
                else:          # own head per component (CheKH)
                    g["W2"][k] += h.T @ dout
                    dh += dout @ p["W2"][k].T
            if "b2" in g:
                g["b2"] += dtot.sum(0)
                g["W2"] += h.T @ dtot
                dh += dtot @ p["W2"].T
            dpre = dh * (1 - h ** 2)
            g["W1"] += x.T @ dpre
            g["b1"] += dpre.sum(0)
        return self.pack(g)


class CheKH(CheK):
    """containing foil: K components, shared trunk h(c), OWN backflow head per component (P0_k is its bias)."""

    def __init__(self, sec, hidden, seed, K):
        M3.ChargeNNBF.__init__(self, sec, hidden, seed)
        self.K = K
        p = self.unpack(self.theta0)
        nout = p["P0"].shape[0]
        q = dict(P0=np.tile(p["P0"], (K, 1)), jv=p["jv"], W1=p["W1"], b1=p["b1"],
                 W2=np.tile(p["W2"][None], (K, 1, 1)), ak=np.ones(K - 1))
        self.setup([("P0", (K, nout)), ("jv", p["jv"].shape), ("W1", p["W1"].shape), ("b1", p["b1"].shape),
                    ("W2", (K, hidden, nout)), ("ak", (K - 1,))])
        self.theta0 = self.pack(q)

    @staticmethod
    def count(L, nf, Nu, Nd, H, K):
        nout = L * (Nu + Nd)
        return nf + K * nout + 2 * L * H + H + K * H * nout + (K - 1)

    def _outs(self, p, h):
        return [h @ p["W2"][k] + p["P0"][k] for k in range(self.K)]


def arm_count(kind, L, nf, Nu, Nd, H=None, Hr=None):
    s = ARM_SPEC[kind]
    H = s["H"] if H is None else H
    if s["cls"] == "rbm":
        return CheRBM.count(L, nf, Nu, Nd, H, s["M"], s["Hr"] if Hr is None else Hr)
    return (CheK if s["cls"] == "k" else CheKH).count(L, nf, Nu, Nd, H, s["K"])


def build_r5(sec, kind, seed, chunk=16384, H=None, Hr=None, width=16):
    """new arm at its pre-committed size (or H, Hr overrides for small smoke clusters), common start."""
    if kind in F.RUN_ARMS:
        return F.build_arm_fix(sec, kind, width, seed, chunk=chunk)
    s = ARM_SPEC[kind]
    H = s["H"] if H is None else H
    L = sec.lat.L
    if s["cls"] == "rbm":
        Hr = min(H, s["Hr"] if Hr is None else Hr)
        arm = CheRBM(sec, H, seed, s["M"], Hr)
    else:
        arm = (CheK if s["cls"] == "k" else CheKH)(sec, H, seed, s["K"])
    Phi, _ = F.fermi_sea(sec)
    rng = np.random.default_rng(10_000 + seed)      # trunk / shared head: as che (fix_test.build_arm_fix)
    rc = np.random.default_rng(20_000 + seed)       # independent component noise
    p = {k: v.copy() for k, v in arm.unpack(arm.theta0).items()}
    p["jv"] = F.gutz(arm, L)
    P0 = np.concatenate([Phi.ravel(), Phi.ravel()])
    p["W2"] = 1e-3 * rng.standard_normal(p["W2"].shape)
    if "b2" in p:
        p["b2"] = np.zeros_like(p["b2"])
    p["W1"] = rng.standard_normal(p["W1"].shape) / np.sqrt(p["W1"].shape[0])
    p["b1"] = np.zeros_like(p["b1"])
    rms = np.sqrt(np.mean(P0 ** 2))
    if s["cls"] == "rbm":
        p["P0"] = P0
        p["br"] = np.zeros(arm.M)
        p["w0"] = 0.5 * EPS * rc.standard_normal(p["w0"].shape)
        p["Wr"] = 0.5 * EPS * rc.standard_normal(p["Wr"].shape) / np.sqrt(arm.Hr)
    else:
        p["P0"] = P0[None] + EPS * rms * rc.standard_normal((arm.K, P0.size))
        p["ak"] = np.ones(arm.K - 1)
    arm.theta0 = arm.pack(p)
    return arm


# ============================================================================ per-amplitude algorithms
def rbm_single(p, nU, nD, L, Nu, Nd, M, Hr):
    """exact <x|P_{S=0}|phi_rbm> (no J): 2^M closed-form Che projections with row-scaled orbitals."""
    Pu, Pd, U, D, occ = T.che_rows(p, nU, nD, L, Nu, Nd)
    x = 2.0 * T.feat_c(nU, nD) - 1.0
    h = np.tanh(x @ p["W1"] + p["b1"])
    Wm = (h[:Hr] @ p["Wr"] + p["w0"]).reshape(M, L)[:, occ]
    tot = 0.0
    for code in range(2 ** M):
        sv = np.array([1.0 if (code >> m) & 1 else -1.0 for m in range(M)])
        f = sv @ Wm
        tot += np.exp(sv @ p["br"]) * T.che_closed(Pu * np.exp(f)[:, None], Pd * np.exp(-f)[:, None], U, D).real
    return tot


def chek_single(p, nU, nD, L, Nu, Nd, K, own_head=False):
    """exact <x|P_{S=0}|phi_k> (no J): K closed-form Che projections."""
    a = np.concatenate([[1.0], p["ak"]])
    tot = 0.0
    for k in range(K):
        q = dict(p); q["P0"] = p["P0"][k]
        if own_head:
            q["W2"] = p["W2"][k]; q["b2"] = np.zeros_like(p["P0"][k])
        tot += a[k] * T.che_closed(*T.che_rows(q, nU, nD, L, Nu, Nd)[:4]).real
    return tot


def single(kind, p, nU, nD, L, Nu, Nd, Hr=None):
    s = ARM_SPEC[kind]
    if s["cls"] == "rbm":
        return rbm_single(p, nU, nD, L, Nu, Nd, s["M"], Hr if Hr is not None else s["Hr"])
    return chek_single(p, nU, nD, L, Nu, Nd, s["K"], own_head=(s["cls"] == "kh"))


# ============================================================================ smoke
SMOKE_SIZE = {"che_rbm1": (4, 2), "che_rbm2": (3, 2), "che_rbm3": (3, 2), "che_k2": (4, None),
              "che_k4": (3, None), "che_k8": (2, None), "che_kh4": (2, None)}


def fd_blocks(st, arm, th, h=1e-5, per=3, seed=11):
    """central-difference check on `per` random indices of EVERY parameter block."""
    _, g, _ = st.energy_grad(th)
    rng = np.random.default_rng(seed)
    off, err = 0, 0.0
    for (_, _), sz in zip(arm.shapes, arm.sizes):
        for k in off + rng.choice(sz, size=min(per, sz), replace=False):
            e = np.zeros(len(th)); e[k] = h
            fd = (st.energy_grad(th + e)[0] - st.energy_grad(th - e)[0]) / (2 * h)
            err = max(err, abs(fd - g[k]) / max(1e-6, abs(fd) + abs(g[k])))
        off += sz
    return err


def smoke(a):
    t0 = time.time()
    ok = True
    sec = M3.make_sector(4, 2, 6, 6.0, -0.3, "obc", "obc")
    sg = M3.SG(sec)
    P0, E0, *_ = M3.ground_state(sec, sg)
    log(f"(a) 4x2 open Ne=6 dim={sec.dim} E0={E0:.6f}")
    Eche = None
    for kind in ["che"] + NEW_ARMS:
        H, Hr = SMOKE_SIZE.get(kind, (4, None))
        arm = build_r5(sec, kind, 0, chunk=4096, H=H, Hr=Hr, width=4)
        st = M3.State(sec, arm, True, sg, 4096)
        phi, _ = arm.forward(arm.theta0, 4096)
        Ei, _ = F.energy_of(sec, st.psi(phi))
        th = arm.theta0 + 0.1 * np.random.default_rng(3).standard_normal(arm.nparam)
        phi, _ = arm.forward(th, 4096)
        _, s2 = F.energy_of(sec, st.psi(phi))
        rel = fd_blocks(st, arm, th)
        if kind == "che":
            Eche = Ei
        good = abs(s2) < 1e-10 and rel < 1e-5 and abs(Ei - Eche) < 0.05 * abs(Eche)
        ok &= good
        log(f"  {kind:8s} P={arm.nparam:4d} H={arm.H} E_init={Ei:.6f} (che {Eche:.6f}) <S^2>={s2:+.1e} "
            f"grad rel.err (all blocks)={rel:.1e} [{'PASS' if good else 'FAIL'}]")
    for (Lx, Ly, Ne, bx, by) in [(2, 3, 6, "pbc", "pbc"), (4, 2, 6, "obc", "obc")]:
        s2c = M3.make_sector(Lx, Ly, Ne, 6.0, -0.3, by, bx)
        PS = s2c.projS(0)
        L, Nu, Nd = s2c.lat.L, s2c.Nu, s2c.Nd
        for kind in NEW_ARMS:
            H, Hr = SMOKE_SIZE[kind]
            arm = build_r5(s2c, kind, 1, chunk=4096, H=H, Hr=Hr)
            p = arm.unpack(arm.theta0 + 0.2 * np.random.default_rng(5).standard_normal(arm.nparam))
            p["jv"][:] = 0
            phi, _ = arm.forward(arm.pack(p), 4096)
            ref = PS(phi).ravel()
            e1 = 0.0
            for i in range(0, s2c.dim, max(1, s2c.dim // 40)):
                nU = s2c.up.bits[s2c.iu[i]].astype(int); nD = s2c.dn.bits[s2c.id[i]].astype(int)
                v = single(kind, p, nU, nD, L, Nu, Nd, Hr=getattr(arm, "Hr", None))
                e1 = max(e1, abs(v - ref[i]))
            scale = np.abs(ref).max()
            good = e1 < 1e-10 * max(1, scale)
            ok &= good
            log(f"(b) {Lx}x{Ly} {bx}/{by} Ne={Ne} {kind:8s}: |polynomial single - full-vector P_S| {e1:.1e} "
                f"(max|amp| {scale:.2e}) [{'PASS' if good else 'FAIL'}]")
    # (c) production parameter counts (no sector build needed: L = 12, nf = 77, 5 + 5)
    L, nf, Nu, Nd = 12, 77, 5, 5
    cnt = {k: arm_count(k, L, nf, Nu, Nd) for k in NEW_ARMS}
    dev = {k: v / P_C - 1 for k, v in cnt.items()}
    good = all(abs(d) <= 0.01 for d in dev.values())
    ok &= good
    log("(c) production counts (4x3, 5+5): " + ", ".join(
        f"{k} {cnt[k]} ({100 * dev[k]:+.2f}%)" for k in NEW_ARMS) + f" vs P_C={P_C} [{'PASS' if good else 'FAIL'}]")
    log(f"SMOKE {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.1f}s)")
    return ok


# ============================================================================ cost
def cost(a):
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for Lx in [int(v) for v in a.cost_sizes.split(",")]:
        L = Lx * Lx
        Ne = 2 * int(round(7 * L / 16)); Nu = Nd = Ne // 2
        H = a.cost_H
        pc = dict(P0=rng.standard_normal(L * Ne) / np.sqrt(L), W1=rng.standard_normal((2 * L, H)) * 0.3,
                  b1=np.zeros(H), W2=0.01 * rng.standard_normal((H, L * Ne)), b2=np.zeros(L * Ne))
        cfgs = []
        for _ in range(a.cost_reps):
            nd_ = int(round(0.05 * Nu))
            st = rng.permutation(L)
            nU = np.zeros(L, int); nD = np.zeros(L, int)
            nU[np.concatenate([st[:nd_], st[nd_:Nu]])] = 1
            nD[np.concatenate([st[:nd_], st[Nu:Nu + Nd - nd_]])] = 1
            cfgs.append((nU, nD))
        row = dict(Lx=Lx, L=L, Ne=Ne)
        row["che_closed"] = T.timeit(lambda nU, nD: T.che_closed(*T.che_rows(pc, nU, nD, L, Nu, Nd)[:4]), cfgs)
        for kind in NEW_ARMS:
            s = ARM_SPEC[kind]
            if s["cls"] == "rbm":
                M = s["M"]
                pk = dict(pc, Wr=0.01 * rng.standard_normal((H, M * L)), w0=0.05 * rng.standard_normal(M * L),
                          br=np.ones(M))
                row[kind] = T.timeit(lambda nU, nD: rbm_single(pk, nU, nD, L, Nu, Nd, M, H), cfgs)
            else:
                K = s["K"]
                pk = dict(pc, P0=rng.standard_normal((K, L * Ne)) / np.sqrt(L), ak=np.full(K - 1, 0.1))
                if s["cls"] == "kh":
                    pk["W2"] = 0.01 * rng.standard_normal((K, H, L * Ne))
                row[kind] = T.timeit(lambda nU, nD: chek_single(pk, nU, nD, L, Nu, Nd, K, s["cls"] == "kh"), cfgs)
        rows.append(row)
        print("Lx=%2d L=%3d Ne=%3d " % (Lx, L, Ne) + "  ".join(f"{k} {v / 1e3:8.2f} ms" for k, v in row.items()
                                                                if k not in ("Lx", "L", "Ne")), flush=True)
    big = rows[-3:]
    ne = np.log([r["Ne"] for r in big])
    keys = ["che_closed"] + NEW_ARMS
    slopes = {k: float(np.polyfit(ne, np.log([r[k] for r in big]), 1)[0]) for k in keys}
    last = rows[-1]
    ratios = {k: last[k] / last["che_closed"] for k in NEW_ARMS}
    print("slopes in N_e (three largest): " + "  ".join(f"{k} {v:.2f}" for k, v in slopes.items()))
    print("ratio to che_closed at largest size: " + "  ".join(f"{k} {v:.2f} (rank {rank(k)})"
                                                              for k, v in ratios.items()))
    with open(os.path.join(a.outdir, "cost_r5.json"), "w") as fh:
        json.dump(dict(rows=rows, slopes=slopes, ratios=ratios), fh, indent=1)


def timing(a):
    """che and every new arm in ONE call, same node and thread count."""
    sec = M3.make_sector(a.Lx, a.Ly, a.Ne, a.U, a.tp, a.bc_y, a.bc_x)
    sg = M3.SG(sec)
    F.load_psi0(a, sec, sg)
    out = {}
    for kind in ["che"] + NEW_ARMS:
        arm = build_r5(sec, kind, 0, a.chunk)
        st = M3.State(sec, arm, True, sg, a.chunk)
        phi, _ = arm.forward(arm.theta0, a.chunk)
        E_init, _ = F.energy_of(sec, st.psi(phi))
        t = time.time()
        E, g, _ = st.energy_grad(arm.theta0)
        dt = time.time() - t
        out[kind] = dict(P=arm.nparam, H=arm.H, Hr=getattr(arm, "Hr", None), rank=1 if kind == "che" else rank(kind),
                         E_init=E_init, sec=dt)
        log(f"{kind:8s}: P={arm.nparam} ({100 * (arm.nparam / P_C - 1):+.2f}% of {P_C}) H={arm.H} "
            f"Hr={getattr(arm, 'Hr', '-')} rank={out[kind]['rank']} E_init={E_init:.6f} "
            f"one energy+gradient {dt:.1f}s ({dt / out['che']['sec']:.2f}x che)")
    out["threads"] = os.environ.get("OMP_NUM_THREADS")
    with open(os.path.join(a.outdir, "timing_r5.json"), "w") as fh:
        json.dump(out, fh, indent=1)


# ============================================================================ run
def run_label(a):
    return f"che_drift{a.maxiter}" if a.ansatz == "che" else a.ansatz


def run(a):
    os.makedirs(a.outdir, exist_ok=True)
    label = run_label(a)
    fn = os.path.join(a.outdir, f"{F.tag_of(a)}_w{a.nnbf_hidden}_{label}_seed{a.seed}.json")
    if os.path.exists(fn) and not a.overwrite:
        log(f"exists: {fn}")
        return
    sec = M3.make_sector(a.Lx, a.Ly, a.Ne, a.U, a.tp, a.bc_y, a.bc_x)
    sg = M3.SG(sec)
    P0, E0 = F.load_psi0(a, sec, sg)
    arm = build_r5(sec, a.ansatz, a.seed, a.chunk, width=a.nnbf_hidden)
    st = M3.State(sec, arm, True, sg, a.chunk)
    phi, _ = arm.forward(arm.theta0, a.chunk)
    E_init, _ = F.energy_of(sec, st.psi(phi))
    log(f"{label}: P={arm.nparam} H={arm.H} Hr={getattr(arm, 'Hr', '-')} E_init={E_init:.8f} E0={E0:.8f}")
    traj = []

    def logf(msg):
        log(msg)
        mt = re.search(r"iter\s+(\d+)\s+E_best = (\S+)", msg)
        if mt:
            traj.append([int(mt.group(1)), float(mt.group(2))])
    th, E, it, sec_ = R.optimise(st, arm.theta0, a.maxiter, a.max_hours, logf=logf)
    phi, _ = arm.forward(th, a.chunk)
    psi = st.psi(phi)
    nrm = np.sum(psi * psi)
    out = dict(ansatz=label, seed=a.seed, P=arm.nparam, H=arm.H, Hr=getattr(arm, "Hr", None),
               rank=1 if a.ansatz == "che" else rank(a.ansatz), E=E, E0=E0, E_init=E_init,
               eps=(E - E0) / abs(E0), iters=it, seconds=sec_, capped=bool(it < a.maxiter and
                                                                          sec_ > 3600 * a.max_hours * 0.999),
               S2=float(np.sum(psi * sec.S2(psi)) / nrm),
               overlap2=float(np.sum(psi * P0) ** 2 / (nrm * np.sum(P0 * P0))), traj=traj,
               shapes=[[k, list(s)] for k, s in arm.shapes])
    p = arm.unpack(th)
    if a.ansatz.startswith("che_rbm"):
        out.update(br=p["br"].tolist(), w0_norm=float(np.linalg.norm(p["w0"])),
                   Wr_norm=float(np.linalg.norm(p["Wr"])))
    elif a.ansatz != "che":
        tv = arm.terms(th, a.chunk)
        out.update(ak=p["ak"].tolist(),
                   term_norm_ratio=[float(np.linalg.norm(tv[k]) / np.linalg.norm(tv[0])) for k in range(1, arm.K)])
    np.save(fn.replace(".json", "_theta.npy"), th)
    log(json.dumps({k: v for k, v in out.items() if k not in ("traj", "shapes")}))
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=1)


# ============================================================================ report
def ref_trajs(logdir):
    """E_best trajectories of the job-451886 runs: {ansatz: {seed: [[it, E], ...]}}."""
    out = {}
    for f in glob.glob(os.path.join(logdir, "P4.1fix_451886_*.out")):
        txt = open(f).read().splitlines()
        res = [json.loads(l[l.index("{"):]) for l in txt if '"ansatz"' in l]
        if not res:
            continue
        traj = [[int(m.group(1)), float(m.group(2))] for l in txt
                for m in [re.search(r"iter\s+(\d+)\s+E_best = (\S+)", l)] if m]
        out.setdefault(res[-1]["ansatz"], {})[res[-1]["seed"]] = traj
    return out


def load_runs(a):
    tag, width = F.tag_of(a), a.nnbf_hidden
    rs = {}
    tr = ref_trajs(a.ref_logs)
    for f in glob.glob(os.path.join(a.ref_dir, f"{tag}_w{width}_*_seed*.json")):
        r = json.load(open(f))
        if r["ansatz"] in REF_ARMS:
            r["traj"] = tr.get(r["ansatz"], {}).get(r["seed"], [])
            rs.setdefault(r["ansatz"], {})[r["seed"]] = r
    for f in glob.glob(os.path.join(a.outdir, f"{tag}_w{width}_*_seed*.json")):
        r = json.load(open(f))
        if r["ansatz"] in NEW_ARMS or r["ansatz"].startswith("che_drift"):
            rs.setdefault(r["ansatz"], {})[r["seed"]] = r
    return {k: list(v.values()) for k, v in rs.items()}


def eps_at(r, n):
    """eps at iteration n from the logged E_best trajectory; final E if the run finished before n uncapped."""
    t = dict((int(i), e) for i, e in r.get("traj", []))
    if n in t:
        return (t[n] - r["E0"]) / abs(r["E0"])
    if not r.get("capped") and r["iters"] < n:
        return r["eps"]
    return None


def last_iter(r):
    if not r.get("capped"):
        return 10 ** 9
    return max([int(i) for i, _ in r.get("traj", [])] or [0])


def verdicts(rs, cost_r5):
    """dict of verdict lines; a pure function of the run dicts and the cost file (pre-committed rules)."""
    out = {}
    for k in REF_ARMS + NEW_ARMS:
        v = rs.get(k, [])
        if len(v) < 3 or any(abs(r["S2"]) >= 1e-10 for r in v):
            return {"ALL": f"INCOMPLETE ({k}: {len(v)} seeds or S^2 not exact)"}
    drift_runs = rs.get(f"che_drift{N_DRIFT}", [])
    if not drift_runs:
        return {"ALL": f"INCOMPLETE (che_drift{N_DRIFT} missing)"}
    arms = REF_ARMS + NEW_ARMS
    n_eq = min(N_FINAL, min(last_iter(r) for k in arms for r in rs[k]))
    n_eq -= n_eq % 50
    if n_eq < N_CHECK:
        return {"ALL": f"INCOMPLETE (a capped arm stopped at iteration {n_eq} < {N_CHECK})"}
    capped = sorted({k for k in arms if any(r.get("capped") for r in rs[k])})
    st = {}
    for n in sorted({n_eq, N_CHECK}):
        e = {k: [eps_at(r, n) for r in rs[k]] for k in arms}
        if any(x is None for v in e.values() for x in v):
            return {"ALL": f"INCOMPLETE (trajectory missing iteration {n})"}
        e = {k: np.array(v) for k, v in e.items()}
        m = {k: float(np.median(v)) for k, v in e.items()}
        lo = {k: float(v.min()) for k, v in e.items()}
        hi = {k: float(v.max()) for k, v in e.items()}
        if not (m["che"] >= 1.5 * m["nnbf_SG"] and lo["che"] > hi["nnbf_SG"]):
            return {"ALL": f"NO-TARGET (che -> nnbf_SG gap not resolved at iteration {n})"}
        gm = {k: (m["che"] - m[k]) / (m["che"] - m["nnbf_SG"]) for k in NEW_ARMS}
        gb = {k: (lo["che"] - lo[k]) / (lo["che"] - lo["nnbf_SG"]) for k in NEW_ARMS}
        st[n] = dict(m=m, lo=lo, hi=hi, gm=gm, gb=gb)
    che = {r["seed"]: r for r in rs["che"]}
    d_drift = float(np.median([eps_at(r, N_CHECK) - eps_at(r, N_FINAL) for r in che.values()]))
    dr = drift_runs[0]
    e_drift = eps_at(dr, N_DRIFT)
    if e_drift is None:
        e_drift = dr["eps"]
    out["drift"] = (f"che 1000->2000 median gain {d_drift:.2e}; che_drift seed {dr['seed']} eps(2000)="
                    f"{eps_at(dr, N_FINAL)} eps({N_DRIFT})={e_drift:.3e} (capped={dr.get('capped')})")

    def fails(k, n):
        """failed parts of the per-arm PASS condition at iteration n (empty list = qualifies)."""
        s = st[n]
        f = []
        if s["gm"][k] < PASS_GAIN or s["gb"][k] < PASS_GAIN:
            f.append("gap fraction < 0.5")
        if s["hi"][k] >= s["lo"]["che"]:
            f.append("seed ranges overlap che")
        if s["m"]["che"] - s["m"][k] <= d_drift:
            f.append("gain <= che 1000->2000 drift")
        return f

    def costok(k):
        return (cost_r5 is not None and k in cost_r5["ratios"] and cost_r5["ratios"][k] <= COST_FACTOR * rank(k)
                and cost_r5["slopes"][k] <= COST_SLOPE)

    s = st[n_eq]
    for k in NEW_ARMS:
        out[k] = (f"rank {rank(k)}: m={s['m'][k]:.3e} [{s['lo'][k]:.3e}, {s['hi'][k]:.3e}] g_m={s['gm'][k]:+.2f} "
                  f"g_b={s['gb'][k]:+.2f} | it {N_CHECK}: g_m={st[N_CHECK]['gm'][k]:+.2f} "
                  f"g_b={st[N_CHECK]['gb'][k]:+.2f} | cost ok={costok(k)}")
    fam_v = {}
    for fam, ks in FAMILIES.items():
        g = [s["gm"][k] for k in ks]
        best = ks[int(np.argmax(g))]
        stable = all(g[i + 1] >= g[i] - STAB_TOL for i in range(len(g) - 1))
        why = []
        if not all(costok(k) for k in ks):
            v = "FAIL (cost gate)"
        elif max(g) < FAIL_GAIN and max(s["gb"][k] for k in ks) < FAIL_GAIN:
            v = "FAIL"
        else:
            for n in sorted({n_eq, N_CHECK}, reverse=True):
                if fails(best, n):
                    why.append(f"{best} at it {n}: " + ", ".join(fails(best, n)))
            if not stable:
                why.append("gain drops with rank")
            if e_drift <= s["m"][best]:
                why.append(f"che_drift at {N_DRIFT} ({e_drift:.3e}) reaches {best}")
            v = "PASS" if not why else "AMBIGUOUS (" + "; ".join(why) + ")"
        fam_v[fam] = (v, best)
        out[f"FAMILY {fam}"] = (f"{v}; best rank {best}; g_m by rank " + ", ".join(f"{x:+.2f}" for x in g)
                                + (f"; compared at equal iteration {n_eq} (capped: {', '.join(capped)})"
                                   if n_eq < N_FINAL else ""))
    # attribution (pre-committed); invalid if any compared arm is capped
    rb_v, rb_best = fam_v["rbm (c-varying row scalings)"]
    if capped:
        out["attribution"] = f"INVALID (capped arm(s): {', '.join(capped)})"
    elif rb_v == "PASS":
        kk = f"che_k{rank(rb_best)}"
        cv = s["m"][rb_best] <= CVARY_CREDIT * s["m"][kk] and s["hi"][rb_best] < s["lo"][kk]
        foil = s["m"]["che_kh4"] <= s["m"]["che_rbm2"]
        out["attribution"] = (("C-VARYING components beat fixed ones at equal rank" if cv else
                               "RANK (rbm not separated from fixed components at equal rank)")
                              + ("; containing foil kh4 matches or beats rbm2 at rank 4" if foil else
                                 "; containing foil kh4 worse than rbm2 at rank 4"))
    elif fam_v["k (fixed offsets, shared backflow)"][0] == "PASS" or fam_v["kh4 (containing foil)"][0] == "PASS":
        out["attribution"] = "RANK (projected-mixture rank; narrow multi-component PHF result)"
    else:
        out["attribution"] = "n/a"
    return out


def report(a):
    rs = load_runs(a)
    cf = os.path.join(a.outdir, "cost_r5.json")
    cost_r5 = json.load(open(cf)) if os.path.exists(cf) else None
    for k in REF_ARMS + NEW_ARMS + [f"che_drift{N_DRIFT}"]:
        for r in sorted(rs.get(k, []), key=lambda r: r["seed"]):
            e1 = eps_at(r, N_CHECK)
            print(f"{k:12s} seed={r['seed']} P={r['P']} eps={r['eps']:.3e} eps@{N_CHECK}="
                  f"{'n/a' if e1 is None else f'{e1:.3e}'} S2={r['S2']:+.1e} it={r['iters']} capped={r.get('capped')}")
    for k, v in verdicts(rs, cost_r5).items():
        print(f"VERDICT {k}: {v}")


def check(_a):
    E0 = -7.827899034271753

    def mk(k, e2, e1=None, s2=0.0, cap_at=None):
        e1 = [x + 0.6e-4 for x in e2] if e1 is None else e1
        out = []
        for i, (x2, x1) in enumerate(zip(e2, e1)):
            traj = [[n, E0 + abs(E0) * (x1 + (x2 - x1) * (n - 1000) / 1000 if n >= 1000 else x1 + 1e-4)]
                    for n in range(50, 2001, 50)]
            r = dict(ansatz=k, seed=i, P=2589, E0=E0, eps=x2, S2=s2, iters=2000, capped=False, traj=traj)
            if cap_at is not None and i == 0:
                r["traj"] = [t for t in traj if t[0] <= cap_at]
                r.update(capped=True, iters=cap_at, eps=(r["traj"][-1][1] - E0) / abs(E0))
            out.append(r)
        return out

    def drift(e4):
        r = dict(ansatz=f"che_drift{N_DRIFT}", seed=0, P=2637, E0=E0, eps=e4, S2=0.0, iters=N_DRIFT, capped=False,
                 traj=[[1000, E0 + abs(E0) * 4.07e-4], [2000, E0 + abs(E0) * 3.43e-4], [4000, E0 + abs(E0) * e4]])
        return {f"che_drift{N_DRIFT}": [r]}

    ref = dict(che=mk("che", [3.43e-4, 4.26e-4, 3.26e-4], [4.07e-4, 5.06e-4, 4.21e-4]),
               nnbf_SG=mk("nnbf_SG", [2.31e-4, 2.07e-4, 1.89e-4], [2.79e-4, 2.55e-4, 2.30e-4]))
    ref.update(drift(3.0e-4))
    bad = [3.4e-4, 3.5e-4, 3.3e-4]
    lad = {"che_rbm1": [2.9e-4, 3.0e-4, 2.8e-4], "che_rbm2": [2.5e-4, 2.55e-4, 2.45e-4],
           "che_rbm3": [2.4e-4, 2.45e-4, 2.35e-4]}

    def arms(**kw):
        d = {}
        for k in NEW_ARMS:
            v = kw.get(k, bad)
            d[k] = v if isinstance(v[0], dict) else mk(k, v)
        return d
    good = dict(ratios={k: 1.1 * rank(k) for k in NEW_ARMS}, slopes={k: 2.3 for k in NEW_ARMS})
    badcost = dict(ratios={k: (3.0 if k == "che_rbm1" else 1.1) * rank(k) for k in NEW_ARMS},
                   slopes={k: 2.3 for k in NEW_ARMS})
    sc = {
        "rbm PASS c-varying; k, kh FAIL": (dict(ref, **arms(**lad)), good),
        "rbm and k PASS equal -> RANK": (dict(ref, **arms(**lad, che_k2=lad["che_rbm1"], che_k4=lad["che_rbm2"],
                                                          che_k8=lad["che_rbm3"])), good),
        "all FAIL": (dict(ref, **arms()), good),
        "gain drops with rank": (dict(ref, **arms(che_rbm1=[2.9e-4, 3.0e-4, 2.8e-4], che_rbm2=[2.4e-4, 2.45e-4, 2.35e-4],
                                                  che_rbm3=[3.3e-4, 3.35e-4, 3.25e-4])), good),
        "PASS at 2000 not at 1000": (dict(ref, **arms(**{k: mk(k, v, [x + 1.6e-4 for x in v]) for k, v in lad.items()})),
                                     good),
        "gain below che 1000->2000 drift": (dict(ref, **arms(**lad), che=mk("che", [3.43e-4, 4.26e-4, 3.26e-4],
                                                                         [5.0e-4, 5.9e-4, 4.9e-4])), good),
        "che_drift(4000) reaches best": (dict(ref, **arms(**lad), **drift(2.3e-4)), good),
        "capped arm -> equal iteration": (dict(ref, **arms(**lad, che_k8=mk("che_k8", bad, cap_at=1500))), good),
        "cost gate": (dict(ref, **arms(**lad)), badcost),
        "capped below 1000": (dict(ref, **arms(**lad, che_k8=mk("che_k8", bad, cap_at=800))), good),
        "missing seed": (dict(ref, **arms(**lad, che_k4=mk("che_k4", bad[:2]))), good),
        "S^2 not exact": (dict(ref, **arms(**dict(lad, che_rbm2=mk("che_rbm2", lad["che_rbm2"], s2=1e-6)))), good),
        "missing drift run": ({k: v for k, v in dict(ref, **arms(**lad)).items() if not k.startswith("che_drift")},
                              good),
        "no gap": (dict(ref, **arms(**lad), nnbf_SG=mk("nnbf_SG", [3.3e-4, 3.4e-4, 3.0e-4])), good),
    }
    for name, (rs, c) in sc.items():
        v = verdicts(rs, c)
        short = [f"{k.replace('FAMILY ', '').split(' (')[0]}: {x.split(';')[0]}" for k, x in v.items()
                 if k.startswith("FAMILY") or k in ("ALL", "attribution")]
        print(f"[{name}] " + " | ".join(short))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="P4.1 round-5 fix tests (see module docstring)")
    p.add_argument("--mode", required=True, choices=["smoke", "cost", "timing", "run", "report", "check"])
    p.add_argument("--Lx", type=int, default=4)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--Ne", type=int, default=10)
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--tp", type=float, default=-0.25)
    p.add_argument("--bc-x", default="obc", choices=["obc", "pbc"], dest="bc_x")
    p.add_argument("--bc-y", default="obc", choices=["obc", "pbc"], dest="bc_y")
    p.add_argument("--ansatz", default="che_rbm2", choices=NEW_ARMS + ["che"])
    p.add_argument("--nnbf-hidden", type=int, default=16, dest="nnbf_hidden")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=N_FINAL)
    p.add_argument("--max-hours", type=float, default=22.0, dest="max_hours")
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--cost-sizes", default="8,12,16,20,24", dest="cost_sizes")
    p.add_argument("--cost-reps", type=int, default=10, dest="cost_reps")
    p.add_argument("--cost-H", type=int, default=8, dest="cost_H")
    p.add_argument("--outdir", default=os.path.join(HERE, "results"))
    p.add_argument("--ref-dir", default=os.path.join(HERE, "..", "P4.1_cheap_su2_fix", "results"), dest="ref_dir")
    p.add_argument("--ref-logs", default=os.path.join(HERE, "..", "P4.1_cheap_su2_fix", "logs"), dest="ref_logs")
    p.add_argument("--psi0-dir", default=os.path.join(HERE, "..", "P4.1_cheap_su2_r4", "results"), dest="psi0_dir")
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    dict(smoke=smoke, cost=cost, timing=timing, run=run, report=report, check=check)[a.mode](a)


if __name__ == "__main__":
    main()
