#!/usr/bin/env python3
"""
P4.1 round 5: polynomial exact-S arms against the che -> nnbf_SG gap.
=====================================================================

Local copies only: fix_test.py, r4_test.py, r3_micro_test.py, r2_micro_test.py are verbatim copies of
tests/P4.1_cheap_su2_fix/. Data READ (not code): the cached ED state psi0_<tag>.npz from
../P4.1_cheap_su2_r4/results, and the reference run JSONs of `che` and `nnbf_SG` from
../P4.1_cheap_su2_fix/results (job 451886, same instance, same common reference start, same optimiser,
2000 L-BFGS iterations).

New arms (both exact S = 0 for every theta, both 4 closed-form Che projections per amplitude)
--------------------------------------------------------------------------------------------
  che_rbm   Route 5.5, hidden-unit-enumerated spin field. Pre-projection state
                phi(x) = e^{J(c)} det Phi_up(c)[U] det Phi_dn(c)[D] prod_{m<M} 2 cosh(b_m + w_m(c) . sigma(x)),
            sigma_j = n_ju - n_jd, w_m(c) = h(c) Wr_m + w0_m (h(c) = the che trunk's hidden layer), trained
            with exact P_S P_G. Since 2 cosh(a) = sum_{s = +-1} e^{s a} and e^{s w . sigma} is a ONE-BODY
            diagonal operator (row scaling e^{+s w_j} of the up orbitals, e^{-s w_j} of the down orbitals),
                <x|P_S|phi> = e^{J(c)} sum_{s in {+-1}^M} e^{s . b} che_closed(Phi_up e^{f_s}, Phi_dn e^{-f_s}, U, D),
            f_s = sum_m s_m w_m(c): 2^M closed forms, O(2^M n^3) per amplitude. The spin dependence is
            read through M learned, charge-conditioned, real linear spin fields (not integer features).
  che_k4    Route 5.4 / mechanism foil at equal cost: K = 4 charge-backflowed UHF components sharing the
            backflow, phi = e^{J} sum_k a_k det(P0_k + B(c))[U] det(...)[D], a_0 = 1, exact P_S P_G.
            K closed forms per amplitude. Gaussian rank 4 per c, NO spin input.
  Both are parameter-matched to NNBF width 16 by the hidden width H (as in round 4).

Usage
-----
  python r5_test.py --mode MODE [options]
--mode {smoke, cost, timing, run, report, check}
  smoke  : <= 1 min login-node code check. (a) 4x2 open N_e = 6: che, che_rbm, che_k4 at random theta:
           <S^2> of P_S P_G phi, analytic gradient vs central differences on indices drawn from EVERY
           parameter block, E_init against che. (b) per-amplitude polynomial algorithms (rbm_single,
           chek_single: sums of che_closed with row-scaled orbitals) against the full-vector P_S of the
           batched state, 2x3 torus and 4x2 open (doublons present), J switched off.
  cost   : per-amplitude wall time (single core) on L = Lx^2 tori, Lx in --cost-sizes: che_closed,
           che_rbm (M = 2), che_k4; writes results/cost_r5.json.
  timing : one energy + gradient evaluation of each new arm on the production instance (8 threads max).
  run    : train one arm (--ansatz che_rbm | che_k4) exactly as fix_test.py run (common reference,
           P_G and exact P_S, L-BFGS, 2000 iterations, --max-hours). Writes
           results/<tag>_w<width>_<ansatz>_seed<seed>.json.
  report : premises and pre-committed verdicts (see the leaf artifact, Round 5).
  check  : report on synthetic result sets, one per verdict path.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fix_test as F  # noqa: E402  (local verbatim copy)

T, M3, R = F.T, F.M3, F.R
log = R.log
NEW_ARMS = ["che_rbm", "che_k4"]
REF_ARMS = ["che", "nnbf_SG"]
M_HID = 2
K_COMP = 4

# pre-committed thresholds (artifact, Round 5)
PASS_GAIN = 0.5      # fraction of the che -> nnbf_SG gap closed, on medians AND on best seeds
FAIL_GAIN = 0.2
SPIN_CREDIT = 0.8    # che_rbm credited with spin-resolved gain only if m(rbm) <= 0.8 m(k4), ranges disjoint
COST_FACTOR = 1.5    # t(new arm) <= 1.5 * 4 * t(che_closed) at the largest size
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
    """che x prod_m 2cosh(b_m + w_m(c).sigma); exact P_S is applied by the State."""

    def __init__(self, sec, hidden, seed, M=M_HID):
        super().__init__(sec, hidden, seed)
        self.M = M
        L = self.L
        bu = sec.up.bits[sec.iu].astype(np.float32)
        bd = sec.dn.bits[sec.id].astype(np.float32)
        self.sig = bu - bd
        p = self.unpack(self.theta0)
        rng = np.random.default_rng(2000 + seed)
        p["Wr"] = 0.01 * rng.standard_normal((hidden, M * L))
        p["w0"] = 0.05 * rng.standard_normal(M * L)
        p["br"] = np.ones(M)
        self.setup(self.shapes + [("Wr", (hidden, M * L)), ("w0", (M * L,)), ("br", (M,))])
        self.theta0 = self.pack(p)

    @staticmethod
    def count(L, nf, Nu, Nd, H, M=M_HID):
        return R.NNBF.count(L, nf, Nu, Nd, H) + H * M * L + M * L + M

    def _args(self, p, h, sl):
        B = h.shape[0]
        Wm = (h @ p["Wr"] + p["w0"]).reshape(B, self.M, self.L)
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
        sec, p, L, Nu, Nd, M = self.sec, cc["p"], self.L, self.Nu, self.Nd, self.M
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
            g["Wr"] += h.T @ dW
            dh = dout @ p["W2"].T + dW @ p["Wr"].T
            dpre = dh * (1 - h ** 2)
            g["W1"] += x.T @ dpre
            g["b1"] += dpre.sum(0)
        return self.pack(g)


class CheK(M3.ChargeNNBF):
    """sum_k a_k che-determinant with reference P0_k and SHARED charge backflow; exact P_S by the State."""

    def __init__(self, sec, hidden, seed, K=K_COMP):
        super().__init__(sec, hidden, seed)
        self.K = K
        p = self.unpack(self.theta0)
        nout = p["P0"].shape[0]
        p["P0"] = np.tile(p["P0"], (K, 1))
        p["ak"] = np.full(K - 1, 0.05)
        self.setup([("P0", (K, nout))] + [(k, s) for k, s in self.shapes if k != "P0"] + [("ak", (K - 1,))])
        self.theta0 = self.pack(p)

    @staticmethod
    def count(L, nf, Nu, Nd, H, K=K_COMP):
        return R.NNBF.count(L, nf, Nu, Nd, H) + (K - 1) * L * (Nu + Nd) + (K - 1)

    def _orbk(self, p, sl):
        sec, L, Nu, Nd = self.sec, self.L, self.Nu, self.Nd
        x = self.cfeat[sec.cidx[sl]]
        h = np.tanh(x @ p["W1"] + p["b1"])
        shared = h @ p["W2"] + p["b2"]
        B = x.shape[0]
        ru = sec.up.occ[sec.iu[sl]]
        rd = sec.dn.occ[sec.id[sl]]
        As = []
        for k in range(self.K):
            out = shared + p["P0"][k]
            Ou = out[:, :L * Nu].reshape(B, L, Nu)
            Od = out[:, L * Nu:].reshape(B, L, Nd)
            As.append((np.take_along_axis(Ou, ru[:, :, None], 1), np.take_along_axis(Od, rd[:, :, None], 1)))
        return x, h, As, ru, rd

    def coef(self, p):
        return np.concatenate([[1.0], p["ak"]])

    def terms(self, th, chunk):
        p = self.unpack(th)
        sec = self.sec
        expJ = np.exp(self.jastrow(p) - self.jastrow(p).max())
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
                dtot += dout
            g["b2"] += dtot.sum(0)
            g["W2"] += h.T @ dtot
            dpre = (dtot @ p["W2"].T) * (1 - h ** 2)
            g["W1"] += x.T @ dpre
            g["b1"] += dpre.sum(0)
        return self.pack(g)


def build_r5(sec, kind, width, seed, chunk=16384):
    """parameter-matched arm with the round-4-fix common reference start."""
    if kind in F.RUN_ARMS:
        return F.build_arm_fix(sec, kind, width, seed, chunk=chunk)
    L = sec.lat.L
    nf = R.charge_features(sec)[0].shape[1]
    target = R.NNBF.count(L, nf, sec.Nu, sec.Nd, width)
    cls = CheRBM if kind == "che_rbm" else CheK
    H = min(range(1, 400), key=lambda h: abs(cls.count(L, nf, sec.Nu, sec.Nd, h) - target))
    arm = cls(sec, H, seed)
    Phi, _ = F.fermi_sea(sec)
    rng = np.random.default_rng(10_000 + seed)
    p = {k: v.copy() for k, v in arm.unpack(arm.theta0).items()}
    p["jv"] = F.gutz(arm, L)
    P0 = np.concatenate([Phi.ravel(), Phi.ravel()])
    p["W2"] = 1e-3 * rng.standard_normal(p["W2"].shape)
    p["b2"] = np.zeros_like(p["b2"])
    p["W1"] = rng.standard_normal(p["W1"].shape) / np.sqrt(p["W1"].shape[0])
    p["b1"] = np.zeros_like(p["b1"])
    if kind == "che_rbm":
        p["P0"] = P0
    else:
        p["P0"][0] = P0
        for k in range(1, arm.K):
            Qu, _ = np.linalg.qr(rng.standard_normal((L, sec.Nu)))
            Qd, _ = np.linalg.qr(rng.standard_normal((L, sec.Nd)))
            p["P0"][k] = np.concatenate([Qu.ravel(), Qd.ravel()])
        p["ak"] = np.ones(arm.K - 1)
        arm.theta0 = arm.pack(p)
        tv = arm.terms(arm.theta0, chunk)
        n0 = np.linalg.norm(tv[0])
        p["ak"] = np.array([0.05 * n0 / np.linalg.norm(tv[k]) for k in range(1, arm.K)])
    arm.theta0 = arm.pack(p)
    return arm


# ============================================================================ per-amplitude algorithms
def rbm_single(p, nU, nD, L, Nu, Nd, M=M_HID):
    """exact <x|P_{S=0}|phi_rbm> (no J): 2^M closed-form Che projections with row-scaled orbitals."""
    Pu, Pd, U, D, occ = T.che_rows(p, nU, nD, L, Nu, Nd)
    x = 2.0 * T.feat_c(nU, nD) - 1.0
    h = np.tanh(x @ p["W1"] + p["b1"])
    Wm = (h @ p["Wr"] + p["w0"]).reshape(M, L)[:, occ]
    tot = 0.0
    for code in range(2 ** M):
        sv = np.array([1.0 if (code >> m) & 1 else -1.0 for m in range(M)])
        f = sv @ Wm
        tot += np.exp(sv @ p["br"]) * T.che_closed(Pu * np.exp(f)[:, None], Pd * np.exp(-f)[:, None], U, D).real
    return tot


def chek_single(p, nU, nD, L, Nu, Nd, K=K_COMP):
    """exact <x|P_{S=0}|phi_k4> (no J): K closed-form Che projections."""
    a = np.concatenate([[1.0], p["ak"]])
    tot = 0.0
    for k in range(K):
        q = dict(p); q["P0"] = p["P0"][k]
        tot += a[k] * T.che_closed(*T.che_rows(q, nU, nD, L, Nu, Nd)[:4]).real
    return tot


# ============================================================================ smoke
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
    for kind in ["che", "che_rbm", "che_k4"]:
        arm = build_r5(sec, kind, 4, 0, chunk=4096)
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
        for kind in ["che_rbm", "che_k4"]:
            arm = build_r5(s2c, kind, 4, 1, chunk=4096)
            p = arm.unpack(arm.theta0 + 0.2 * np.random.default_rng(5).standard_normal(arm.nparam))
            p["jv"][:] = 0
            phi, _ = arm.forward(arm.pack(p), 4096)
            ref = PS(phi).ravel()
            e1 = 0.0
            for i in range(0, s2c.dim, max(1, s2c.dim // 40)):
                nU = s2c.up.bits[s2c.iu[i]].astype(int); nD = s2c.dn.bits[s2c.id[i]].astype(int)
                v = (rbm_single if kind == "che_rbm" else chek_single)(p, nU, nD, L, Nu, Nd)
                e1 = max(e1, abs(v - ref[i]))
            scale = np.abs(ref).max()
            good = e1 < 1e-10 * max(1, scale)
            ok &= good
            log(f"(b) {Lx}x{Ly} {bx}/{by} Ne={Ne} {kind}: |polynomial single - full-vector P_S| {e1:.1e} "
                f"(max|amp| {scale:.2e}) [{'PASS' if good else 'FAIL'}]")
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
        pr = dict(pc, Wr=0.01 * rng.standard_normal((H, M_HID * L)), w0=0.05 * rng.standard_normal(M_HID * L),
                  br=np.ones(M_HID))
        pk = dict(pc, P0=rng.standard_normal((K_COMP, L * Ne)) / np.sqrt(L), ak=np.full(K_COMP - 1, 0.1))
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
        row["che_rbm"] = T.timeit(lambda nU, nD: rbm_single(pr, nU, nD, L, Nu, Nd), cfgs)
        row["che_k4"] = T.timeit(lambda nU, nD: chek_single(pk, nU, nD, L, Nu, Nd), cfgs)
        rows.append(row)
        print("Lx=%2d L=%3d Ne=%3d " % (Lx, L, Ne) + "  ".join(f"{k} {v / 1e3:9.2f} ms" for k, v in row.items()
                                                                if k not in ("Lx", "L", "Ne")), flush=True)
    big = rows[-3:]
    ne = np.log([r["Ne"] for r in big])
    slopes = {k: float(np.polyfit(ne, np.log([r[k] for r in big]), 1)[0]) for k in ("che_closed", "che_rbm", "che_k4")}
    last = rows[-1]
    ratios = {k: last[k] / last["che_closed"] for k in ("che_rbm", "che_k4")}
    print("slopes in N_e (three largest): " + "  ".join(f"{k} {v:.2f}" for k, v in slopes.items()))
    print("ratio to che_closed at largest size: " + "  ".join(f"{k} {v:.2f}" for k, v in ratios.items()))
    with open(os.path.join(a.outdir, "cost_r5.json"), "w") as fh:
        json.dump(dict(rows=rows, slopes=slopes, ratios=ratios), fh, indent=1)


def timing(a):
    sec = M3.make_sector(a.Lx, a.Ly, a.Ne, a.U, a.tp, a.bc_y, a.bc_x)
    sg = M3.SG(sec)
    F.load_psi0(a, sec, sg)
    for kind in NEW_ARMS:
        arm = build_r5(sec, kind, a.nnbf_hidden, 0, a.chunk)
        st = M3.State(sec, arm, True, sg, a.chunk)
        phi, _ = arm.forward(arm.theta0, a.chunk)
        E_init, _ = F.energy_of(sec, st.psi(phi))
        t = time.time()
        E, g, _ = st.energy_grad(arm.theta0)
        log(f"{kind}: P={arm.nparam} H={arm.H} E_init={E_init:.6f} one energy+gradient {time.time() - t:.1f}s "
            f"|g|={np.linalg.norm(g):.2e}")


# ============================================================================ run
def run(a):
    os.makedirs(a.outdir, exist_ok=True)
    fn = os.path.join(a.outdir, f"{F.tag_of(a)}_w{a.nnbf_hidden}_{a.ansatz}_seed{a.seed}.json")
    if os.path.exists(fn) and not a.overwrite:
        log(f"exists: {fn}")
        return
    sec = M3.make_sector(a.Lx, a.Ly, a.Ne, a.U, a.tp, a.bc_y, a.bc_x)
    sg = M3.SG(sec)
    P0, E0 = F.load_psi0(a, sec, sg)
    arm = build_r5(sec, a.ansatz, a.nnbf_hidden, a.seed, a.chunk)
    st = M3.State(sec, arm, True, sg, a.chunk)
    phi, _ = arm.forward(arm.theta0, a.chunk)
    E_init, _ = F.energy_of(sec, st.psi(phi))
    log(f"{a.ansatz}: P={arm.nparam} H={arm.H} E_init={E_init:.8f} E0={E0:.8f}")
    th, E, it, sec_ = R.optimise(st, arm.theta0, a.maxiter, a.max_hours)
    phi, _ = arm.forward(th, a.chunk)
    psi = st.psi(phi)
    nrm = np.sum(psi * psi)
    out = dict(ansatz=a.ansatz, seed=a.seed, P=arm.nparam, H=arm.H, E=E, E0=E0, E_init=E_init,
               eps=(E - E0) / abs(E0), iters=it, seconds=sec_, capped=bool(sec_ > 3600 * a.max_hours * 0.999),
               S2=float(np.sum(psi * sec.S2(psi)) / nrm),
               overlap2=float(np.sum(psi * P0) ** 2 / (nrm * np.sum(P0 * P0))))
    p = arm.unpack(th)
    if a.ansatz == "che_rbm":
        out.update(br=p["br"].tolist(), w0_norm=float(np.linalg.norm(p["w0"])),
                   Wr_norm=float(np.linalg.norm(p["Wr"])))
    else:
        tv = arm.terms(th, a.chunk)
        out.update(term_norm_ratio=[float(np.linalg.norm(tv[k]) / np.linalg.norm(tv[0])) for k in range(1, arm.K)])
    np.save(fn.replace(".json", "_theta.npy"), th)
    log(json.dumps(out))
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=1)


# ============================================================================ report
def load_runs(dirs, tag, width):
    rs = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, f"{tag}_w{width}_*_seed*.json")):
            r = json.load(open(f))
            if r["ansatz"] in NEW_ARMS + REF_ARMS:
                rs.setdefault(r["ansatz"], {})[r["seed"]] = r
    return {k: list(v.values()) for k, v in rs.items()}


def verdicts(rs, cost_r5):
    """returns dict of verdict lines; pure function of the run dicts and the cost file."""
    out = {}
    for k in REF_ARMS + NEW_ARMS:
        v = rs.get(k, [])
        if len(v) < 3 or any(abs(r["S2"]) >= 1e-10 for r in v):
            return {"ALL": f"INCOMPLETE ({k}: {len(v)} seeds or S^2 not exact)"}
    eps = {k: np.array([r["eps"] for r in v]) for k, v in rs.items()}
    m = {k: float(np.median(v)) for k, v in eps.items()}
    lo = {k: float(v.min()) for k, v in eps.items()}
    hi = {k: float(v.max()) for k, v in eps.items()}
    if not (m["che"] >= 1.5 * m["nnbf_SG"] and lo["che"] > hi["nnbf_SG"]):
        return {"ALL": "NO-TARGET (che -> nnbf_SG gap not resolved)"}
    gap_m = m["che"] - m["nnbf_SG"]
    gap_b = lo["che"] - lo["nnbf_SG"]
    costok = {}
    for k in NEW_ARMS:
        if cost_r5 is None:
            costok[k] = False
        else:
            costok[k] = (cost_r5["ratios"][k] <= COST_FACTOR * 4) and (cost_r5["slopes"][k] <= COST_SLOPE)
    for k in NEW_ARMS:
        gm = (m["che"] - m[k]) / gap_m
        gb = (lo["che"] - lo[k]) / gap_b
        capped = any(r.get("capped") for r in rs[k])
        if not costok[k]:
            v = "FAIL (cost gate)"
        elif gm >= PASS_GAIN and gb >= PASS_GAIN and hi[k] < lo["che"]:
            v = "PASS"
        elif gm < FAIL_GAIN and gb < FAIL_GAIN:
            v = "FAIL"
        else:
            v = "AMBIGUOUS"
        if capped and v in ("PASS", "FAIL"):
            v = "AMBIGUOUS (capped)"
        out[k] = (f"{v}: m={m[k]:.3e} [{lo[k]:.3e}, {hi[k]:.3e}]  g_m={gm:+.2f} g_b={gb:+.2f}  "
                  f"(che m={m['che']:.3e}, nnbf_SG m={m['nnbf_SG']:.3e})")
    # mechanism attribution for che_rbm
    if out["che_rbm"].startswith("PASS"):
        spin = m["che_rbm"] <= SPIN_CREDIT * m["che_k4"] and hi["che_rbm"] < lo["che_k4"]
        out["attribution"] = ("SPIN-RESOLVED (che_rbm beats the equal-cost Gaussian-rank-4 foil)" if spin else
                              "GAUSSIAN-RANK (che_rbm not separated from che_k4: credit multi-component PHF)")
    else:
        out["attribution"] = "n/a"
    return out


def report(a):
    rs = load_runs([a.ref_dir, a.outdir], F.tag_of(a), a.nnbf_hidden)
    cf = os.path.join(a.outdir, "cost_r5.json")
    cost_r5 = json.load(open(cf)) if os.path.exists(cf) else None
    for k in REF_ARMS + NEW_ARMS:
        for r in sorted(rs.get(k, []), key=lambda r: r["seed"]):
            print(f"{k:8s} seed={r['seed']} P={r['P']} eps={r['eps']:.3e} S2={r['S2']:+.1e} it={r['iters']} "
                  f"capped={r.get('capped')}")
    for k, v in verdicts(rs, cost_r5).items():
        print(f"VERDICT {k}: {v}")


def check(_a):
    def mk(k, e, s2=0.0):
        return [dict(ansatz=k, seed=i, P=2637, eps=x, S2=s2, iters=2000, capped=False) for i, x in enumerate(e)]
    base = dict(che=mk("che", [3.43e-4, 4.26e-4, 3.26e-4]), nnbf_SG=mk("nnbf_SG", [2.31e-4, 2.07e-4, 1.89e-4]))
    good = dict(ratios=dict(che_rbm=4.2, che_k4=4.1), slopes=dict(che_rbm=2.3, che_k4=2.3))
    sc = {
        "rbm PASS spin, k4 FAIL": (dict(base, che_rbm=mk("che_rbm", [2.4e-4, 2.5e-4, 2.3e-4]),
                                        che_k4=mk("che_k4", [3.4e-4, 3.5e-4, 3.3e-4])), good),
        "both PASS, gain = rank": (dict(base, che_rbm=mk("che_rbm", [2.4e-4, 2.5e-4, 2.3e-4]),
                                        che_k4=mk("che_k4", [2.45e-4, 2.4e-4, 2.35e-4])), good),
        "both FAIL": (dict(base, che_rbm=mk("che_rbm", [3.5e-4, 3.6e-4, 3.3e-4]),
                           che_k4=mk("che_k4", [3.3e-4, 3.6e-4, 3.4e-4])), good),
        "overlap AMBIGUOUS": (dict(base, che_rbm=mk("che_rbm", [2.9e-4, 3.4e-4, 2.6e-4]),
                                   che_k4=mk("che_k4", [3.0e-4, 3.0e-4, 2.9e-4])), good),
        "cost gate": (dict(base, che_rbm=mk("che_rbm", [2.4e-4, 2.5e-4, 2.3e-4]),
                           che_k4=mk("che_k4", [2.4e-4, 2.5e-4, 2.3e-4])),
                      dict(ratios=dict(che_rbm=9.0, che_k4=4.0), slopes=dict(che_rbm=2.3, che_k4=3.4))),
        "missing seed": (dict(base, che_rbm=mk("che_rbm", [2.4e-4, 2.5e-4]),
                              che_k4=mk("che_k4", [2.4e-4, 2.5e-4, 2.3e-4])), good),
        "S^2 not exact": (dict(base, che_rbm=mk("che_rbm", [2.4e-4, 2.5e-4, 2.3e-4], s2=1e-6),
                               che_k4=mk("che_k4", [2.4e-4, 2.5e-4, 2.3e-4])), good),
        "no gap": (dict(base, nnbf_SG=mk("nnbf_SG", [3.3e-4, 3.4e-4, 3.0e-4]),
                        che_rbm=mk("che_rbm", [2.4e-4, 2.5e-4, 2.3e-4]),
                        che_k4=mk("che_k4", [2.4e-4, 2.5e-4, 2.3e-4])), good),
    }
    for name, (rs, c) in sc.items():
        v = verdicts(rs, c)
        print(f"[{name}] " + " | ".join(f"{k}: {x.split(':')[0] if k != 'attribution' else x.split(' (')[0]}"
                                        for k, x in v.items()))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="P4.1 round-5 tests (see module docstring)")
    p.add_argument("--mode", required=True, choices=["smoke", "cost", "timing", "run", "report", "check"])
    p.add_argument("--Lx", type=int, default=4)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--Ne", type=int, default=10)
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--tp", type=float, default=-0.25)
    p.add_argument("--bc-x", default="obc", choices=["obc", "pbc"], dest="bc_x")
    p.add_argument("--bc-y", default="obc", choices=["obc", "pbc"], dest="bc_y")
    p.add_argument("--ansatz", default="che_rbm", choices=NEW_ARMS)
    p.add_argument("--nnbf-hidden", type=int, default=16, dest="nnbf_hidden")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--max-hours", type=float, default=22.0, dest="max_hours")
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--cost-sizes", default="8,12,16,20,24", dest="cost_sizes")
    p.add_argument("--cost-reps", type=int, default=10, dest="cost_reps")
    p.add_argument("--cost-H", type=int, default=8, dest="cost_H")
    p.add_argument("--outdir", default=os.path.join(HERE, "results"))
    p.add_argument("--ref-dir", default=os.path.join(HERE, "..", "P4.1_cheap_su2_fix", "results"), dest="ref_dir")
    p.add_argument("--psi0-dir", default=os.path.join(HERE, "..", "P4.1_cheap_su2_r4", "results"), dest="psi0_dir")
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    dict(smoke=smoke, cost=cost, timing=timing, run=run, report=report, check=check)[a.mode](a)


if __name__ == "__main__":
    main()
