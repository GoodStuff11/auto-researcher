#!/usr/bin/env python3
r"""
P4.2 round-5 per-leaf critic: is the capacity plateau of the size-independent MFNG-O family an
optimisation artefact or an architecture limit, and which component is the bottleneck?
=================================================================================================

Facts this script tests (P4.2 doc, "Enlarged-Capacity Re-pilot"): on the 4x4 OBC t-t'-J 6+6 target
MFNG-O plateaus at relative error 5.97% (426 parameters, job 452196) and 5.71% (818 parameters,
job 604157); the projected NNBF of P2.1 reaches 0.56% with 1734 parameters.

Two modes.

--mode lm      STATIONARITY CHECK on a saved trajectory (4x4 target, theta of a checkpoint record).
               At theta, with b = U^T psi (projected amplitudes) and B = U^T (psi * O) (projected
               log-derivative vectors), the linear method (LM) builds the space
               span{b, B_1 - <B_1> b, ..., B_P - <B_P> b} and returns the LOWEST projected energy
               E_LM of ANY state in it (generalised eigenproblem, overlap eigenvalues below
               --lm-cut * max dropped; several cuts printed). E_LM is the best energy reachable by
               one first-order move in ANY direction with ANY step and ANY preconditioner (it is a
               real variational state in the irrep, so E_LM >= E0). Printed: the gradient norm,
               E, E_LM, and the LM gain fraction g = (E - E_LM) / (E - E0).
               Pre-registered reading: g < 0.10 at the plateau -> the plateau is a tangent-space
               stationary point (not a learning-rate / SR-shift / step-count artefact).
--mode train   COMPONENT ABLATION on a small ED cluster (default 4x3 OBC 5+5, dim 16632, irrep
               4152, ~0.8 s per step). MFNG-O with optional extra components, each initialised to
               ZERO so every variant starts from exactly the base MFNG-O state (same rng):
                 --free-f        site-resolved symmetric pair matrix F_ij added to the geminal
                                 (A_ij += F[r_up_i, r_dn_j]); N(N+1)/2 parameters. NOT size
                                 independent: an oracle for the translation-invariant kernel /
                                 open-boundary restriction.
                 --spin-feat     the local site features v_s seen by the backflow and the
                                 one-body Jastrow get three spin channels (S^z_s, sum_nn S^z/4,
                                 sum_nnn S^z/4; missing neighbour = 0). Size independent; breaks
                                 exact SU(2) (S^z is kept).
                 --jas2 c|cs     two-body Jastrow over C4v displacement classes,
                                 sum_{i<j} v_c(d_ij) h_i h_j  [+ v_s(d_ij) S^z_i S^z_j for cs].
                                 Size independent. "c" keeps exact SU(2); "cs" breaks it.
                 --bnd-f R       boundary-aware pair kernel (SIZE INDEPENDENT replacement of
                                 --free-f): A_ij += G[b_i, b_j, d_ij] for max(|dx|,|dy|) <= R, with
                                 b = number of missing nearest neighbours (0 bulk, 1 edge,
                                 2 corner; unordered pair) and d the C4v displacement class.
                                 R = 0 (default) switches it off.
               Trained by the same projected exact-sum SR as obc_transfer.py
               (r5_common.optimise_ckpt; --lr, --diag-shift as there). The final record gives E,
               relerr, the fixed-sign excess A (P1.1 budget) and its share of E - E0.

Command-line options
--------------------
  --mode STR        lm | train (required).
  --json PATH       (lm) obc_transfer.py result JSON holding the record named by --record.
  --record STR      (lm) record name, default "scr@600" (its "theta" list is used).
  --lm-cut STR      (lm) comma list of relative overlap-eigenvalue cut-offs (default 1e-6,1e-8,1e-10).
  --tgt STR         Target "LxXLy:nup" OBC (lm: taken from the JSON; train default "4x3:5").
  --tp, --J, --t    Couplings (-0.25, 0.4, 1.0) (lm: taken from the JSON).
  --channels, --field-hidden, --bf-hidden, --jas-hidden   MFNG sizes (4, 16, 16, 16) (lm: JSON).
  --free-f, --spin-feat   (train) flags, see above.   --jas2 STR  none | c | cs (default none).
  --bnd-f INT       (train) range R of the boundary-aware pair kernel (default 0 = off).
  --steps INT       (train) SR steps (default 600).   --seed INT  init seed offset (default 0;
                    the init rng is 2000 + seed exactly as obc_transfer.py's scr arm).
  --lr, --diag-shift  SR step and relative diagonal shift (0.05, 0.01).
  --kq INT (8)  --delta0 (0.3)  --eta (0.1)  --chunk INT (50000)  --r-min (2.0)
  --fn-iters (6) --fn-floor (1e-3) --lbfgs-iters (300)   fixed-sign budget settings.
  --out PATH        JSON output (required).
  --selftest        log-derivative check of every extra component by central finite differences
                    (4x2 OBC 3+3, h = 1e-6, configurations with |d log psi| >= 100, i.e. a
                    near-singular geminal, skipped; pass if max relative error < 1e-4) and check
                    that the zero-initialised extras reproduce the base MFNG-O amplitudes; exit.

Examples
--------
  python3 critic_capacity.py --selftest
  python3 critic_capacity.py --mode train --spin-feat --out results_critic/train_spin.json
  python3 critic_capacity.py --mode lm --json results_capacity_c8h32/scr_s0.json --out results_critic/lm_c8h32.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r3_core as rc  # noqa: E402
import obc_transfer as ot  # noqa: E402
from r5_common import optimise_ckpt  # noqa: E402


# ============================================================================ extended ansatz
def bnd_map(lat, rf):
    """Boundary-aware, SIZE-INDEPENDENT pair-kernel index map: for sites i != j with
    max(|dx|,|dy|) <= rf, key = (sorted boundary classes of i and j, C4v displacement class), where
    the boundary class of a site is its number of missing nearest neighbours (0 bulk, 1 edge,
    2 corner). Returns the (N*N, nkeys) 0/1 matrix M with F_b[i, j] = (M theta_b)[i*N + j]."""
    N = lat.N
    nb = (lat.nn == N).sum(1)
    keys, ent = {}, []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx, dy = abs(lat.xs[i] - lat.xs[j]), abs(lat.ys[i] - lat.ys[j])
            if max(dx, dy) > rf:
                continue
            k = (min(nb[i], nb[j]), max(nb[i], nb[j]), min(dx, dy), max(dx, dy))
            keys.setdefault(k, len(keys))
            ent.append((i * N + j, keys[k]))
    M = np.zeros((N * N, len(keys)))
    for r, c in ent:
        M[r, c] = 1.0
    return M


def disp_classes(lat):
    """C4v displacement class index for every pair i<j: key (min(|dx|,|dy|), max(|dx|,|dy|))."""
    keys = {}
    N = lat.N
    M = []
    for i in range(N):
        for j in range(i + 1, N):
            dx, dy = abs(lat.xs[i] - lat.xs[j]), abs(lat.ys[i] - lat.ys[j])
            k = (min(dx, dy), max(dx, dy))
            keys.setdefault(k, len(keys))
            M.append((i, j, keys[k]))
    Mc = np.zeros((len(keys), N, N))
    for i, j, c in M:
        Mc[c, i, j] = 1.0
    return Mc


class MFNGX(ot.MFNGO):
    def __init__(self, lat, nup, C, Hf, Hb, Hj, free_f=False, spin_feat=False, jas2="none", bnd_f=0):
        self.base = ot.MFNGO(lat, nup, C, Hf, Hb, Hj)
        self.lat, self.n, self.C = lat, nup, C
        self.Hf, self.Hb, self.Hj = Hf, (Hb if C > 0 else 0), Hj
        self.free_f, self.spin_feat, self.jas2, self.bnd_f = free_f, spin_feat, jas2, bnd_f
        self.Mb = bnd_map(lat, bnd_f) if bnd_f > 0 else None
        self.Fin = 6 if spin_feat else 3
        ck, cy = np.cos(lat.kx), np.cos(lat.ky)
        self.z = np.stack([(ck + cy) / 2, ck * cy], 1)
        self.gam = ck - cy
        nout = 2 * (C + 1)
        N = lat.N
        self.Mc = disp_classes(lat)
        ncl = self.Mc.shape[0]
        sizes = [("fW1", Hf * 2), ("fb1", Hf), ("fW2", nout * Hf), ("fb2", nout),
                 ("bW", self.Hb * self.Fin), ("bb", self.Hb), ("ba", C * self.Hb),
                 ("jW", Hj * self.Fin), ("jb", Hj), ("ja", Hj),
                 ("F", N * (N + 1) // 2 if free_f else 0),
                 ("vc", ncl if jas2 in ("c", "cs") else 0), ("vs", ncl if jas2 == "cs" else 0),
                 ("Fb", self.Mb.shape[1] if bnd_f > 0 else 0)]
        self.sl, o = {}, 0
        for nm, s_ in sizes:
            self.sl[nm] = slice(o, o + s_); o += s_
        self.npar = o
        self.nf = self.sl["fb2"].stop
        self.nd = lat.Cd.shape[0]
        self.iu = np.triu_indices(N)

    def init(self, rng, density, t, tp, delta0, eta):
        tb = self.base.init(rng, density, t, tp, delta0, eta)
        th = np.zeros(self.npar)
        for nm in ("fW1", "fb1", "fW2", "fb2", "bb", "ba", "jb", "ja"):
            th[self.sl[nm]] = tb[self.base.sl[nm]]
        for nm, H in (("bW", self.Hb), ("jW", self.Hj)):
            W = np.zeros((H, self.Fin)); W[:, :3] = tb[self.base.sl[nm]].reshape(H, 3)
            th[self.sl[nm]] = W.ravel()
        return th

    def Fmat(self, th):
        N = self.lat.N
        F = np.zeros((N, N))
        F[self.iu] = th[self.sl["F"]]
        return F + F.T - np.diag(np.diag(F))

    def feats(self, states):
        lat, N = self.lat, self.lat.N
        nu, nd, hole, v = ot.site_features(lat, states)
        sz = (nu - nd).astype(float)
        if self.spin_feat:
            sp_ = np.concatenate([sz, np.zeros((len(states), 1))], 1)
            v = np.concatenate([v, np.stack([sz, sp_[:, lat.nn].sum(2) / 4, sp_[:, lat.nnn].sum(2) / 4], 2)], 2)
        return nu, nd, hole, sz, v

    def evaluate(self, th, states, need_grad, cache=None):
        lat, N, n, C, Fin = self.lat, self.lat.N, self.n, self.C, self.Fin
        B = len(states)
        if cache is None:
            cache = self.precompute(th, need_grad)
        kap, CJ = cache
        nu, nd, hole, sz, v = self.feats(states)
        jW = self.P(th, "jW", (self.Hj, Fin)); jb = th[self.sl["jb"]]; ja = th[self.sl["ja"]]
        zj = v @ jW.T + jb
        lc = np.logaddexp(zj, -zj) - np.log(2.0)
        logJ = (lc @ ja).sum(1)
        if self.jas2 != "none":
            hh = hole[:, :, None] * hole[:, None, :]
            Ohc = np.einsum("bij,cij->bc", hh, self.Mc)
            logJ = logJ + Ohc @ th[self.sl["vc"]]
            if self.jas2 == "cs":
                ss = sz[:, :, None] * sz[:, None, :]
                Ohs = np.einsum("bij,cij->bc", ss, self.Mc)
                logJ = logJ + Ohs @ th[self.sl["vs"]]
        ups = np.nonzero(nu)[1].reshape(B, n)
        dns = np.nonzero(nd)[1].reshape(B, n)
        D = lat.disp[ups[:, :, None], dns[:, None, :]]
        bi = np.arange(B)[:, None]
        A = kap[0][D].copy()
        if self.free_f:
            A += self.Fmat(th)[ups[:, :, None], dns[:, None, :]]
        if self.bnd_f > 0:
            A += (self.Mb @ th[self.sl["Fb"]]).reshape(N, N)[ups[:, :, None], dns[:, None, :]]
        if C > 0:
            bW = self.P(th, "bW", (self.Hb, Fin)); bb = th[self.sl["bb"]]; ba = self.P(th, "ba", (C, self.Hb))
            tb = np.tanh(v @ bW.T + bb)
            phi = tb @ ba.T
            pu = phi[bi, ups]
            pd = phi[bi, dns]
            for m in range(1, C + 1):
                A += pu[:, :, None, m - 1] * kap[m][D] * pd[:, None, :, m - 1]
        sgn, lad = np.linalg.slogdet(A)
        good = (sgn != 0) & np.isfinite(lad)
        logabs = np.where(good, logJ + lad, -np.inf)
        if not need_grad:
            return logabs, sgn, None
        O = np.zeros((B, self.npar))
        Wt = np.zeros_like(A)
        if good.any():
            Wt[good] = np.transpose(np.linalg.inv(A[good]), (0, 2, 1))
        nd_ = self.nd
        flat = (np.arange(B)[:, None, None] * nd_ + D).ravel()
        for m in range(C + 1):
            Q = Wt if m == 0 else Wt * pu[:, :, None, m - 1] * pd[:, None, :, m - 1]
            Hm = np.bincount(flat, weights=Q.ravel(), minlength=B * nd_).reshape(B, nd_)
            O[:, :self.nf] += Hm @ CJ[m]
        if self.free_f or self.bnd_f > 0:
            fl = (np.arange(B)[:, None, None] * N * N + ups[:, :, None] * N + dns[:, None, :]).ravel()
            G = np.bincount(fl, weights=Wt.ravel(), minlength=B * N * N).reshape(B, N, N)
        if self.bnd_f > 0:
            O[:, self.sl["Fb"]] = G.reshape(B, N * N) @ self.Mb
        if self.free_f:
            Gs = G + np.transpose(G, (0, 2, 1))
            idx = np.arange(N)
            Gs[:, idx, idx] = G[:, idx, idx]
            O[:, self.sl["F"]] = Gs[:, self.iu[0], self.iu[1]]
        if C > 0:
            delta = np.zeros_like(tb)
            for m in range(1, C + 1):
                G = Wt * kap[m][D]
                Rr = (G * pd[:, None, :, m - 1]).sum(2)
                Cc = (G * pu[:, :, None, m - 1]).sum(1)
                dphi = np.zeros((B, N))
                dphi[bi, ups] = Rr
                dphi[bi, dns] = Cc
                O[:, self.sl["ba"].start + (m - 1) * self.Hb:self.sl["ba"].start + m * self.Hb] = \
                    np.einsum("bs,bsh->bh", dphi, tb)
                delta += dphi[:, :, None] * ba[m - 1][None, None, :]
            delta *= 1 - tb ** 2
            O[:, self.sl["bW"]] = np.einsum("bsh,bsf->bhf", delta, v).reshape(B, -1)
            O[:, self.sl["bb"]] = delta.sum(1)
        tj = np.tanh(zj) * ja
        O[:, self.sl["ja"]] = lc.sum(1)
        O[:, self.sl["jb"]] = tj.sum(1)
        O[:, self.sl["jW"]] = np.einsum("bsh,bsf->bhf", tj, v).reshape(B, -1)
        if self.jas2 != "none":
            O[:, self.sl["vc"]] = Ohc
            if self.jas2 == "cs":
                O[:, self.sl["vs"]] = Ohs
        O[~good] = 0.0
        return logabs, sgn, O


def make_net(lat, nup, a):
    return MFNGX(lat, nup, a.channels, a.field_hidden, a.bf_hidden, a.jas_hidden,
                 free_f=a.free_f, spin_feat=a.spin_feat, jas2=a.jas2, bnd_f=a.bnd_f)


# ============================================================================ modes
def selftest(a):
    lat = ot.OBC(4, 2, a.t, a.tp, a.J, a.kq)
    basis = rc.make_basis(lat.N, 3, 3)
    ok = True
    for ff, sf, j2, bf in ((False, False, "none", 0), (True, False, "none", 0), (False, True, "none", 0),
                           (False, False, "cs", 0), (True, True, "cs", 0), (False, False, "none", 3),
                           (False, True, "cs", 3)):
        net = MFNGX(lat, 3, 2, 6, 5, 4, free_f=ff, spin_feat=sf, jas2=j2, bnd_f=bf)
        rng = np.random.default_rng(3)
        th = net.init(rng, 0.75, a.t, a.tp, a.delta0, a.eta)
        th = th + 0.05 * rng.standard_normal(net.npar)
        st = basis[:300]
        la, sg, O = net.evaluate(th, st, True)
        u = rng.standard_normal(net.npar)
        h = 1e-6
        lp, _, _ = net.evaluate(th + h * u, st, False)
        lm, _, _ = net.evaluate(th - h * u, st, False)
        an = O @ u
        fin = np.isfinite(la) & (np.abs(an) < 1e2)   # skip near-singular geminals (|d log psi| huge)
        fd = (lp - lm)[fin] / (2 * h)
        err = np.max(np.abs(fd - an[fin])) / max(1e-12, np.max(np.abs(fd)))
        same = True
        if not (ff or sf or j2 != "none" or bf):
            b = ot.MFNGO(lat, 3, 2, 6, 5, 4)
            tb = b.init(np.random.default_rng(3), 0.75, a.t, a.tp, a.delta0, a.eta)
            t0 = net.init(np.random.default_rng(3), 0.75, a.t, a.tp, a.delta0, a.eta)
            l1, _, _ = b.evaluate(tb, st, False); l2, _, _ = net.evaluate(t0, st, False)
            same = np.allclose(l1, l2)
        else:
            tb = net.base.init(np.random.default_rng(3), 0.75, a.t, a.tp, a.delta0, a.eta)
            t0 = net.init(np.random.default_rng(3), 0.75, a.t, a.tp, a.delta0, a.eta)
            l1, _, _ = net.base.evaluate(tb, st, False); l2, _, _ = net.evaluate(t0, st, False)
            same = np.allclose(l1, l2)
        good = err < 1e-4 and same
        ok &= good
        rc.log(f"selftest free_f={ff} spin_feat={sf} jas2={j2} bnd_f={bf}: npar={net.npar} rel grad err {err:.2e}, "
               f"zero-init equals base MFNG-O: {same} -> {'OK' if good else 'FAIL'}")
    rc.log("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return ok


def setup_target(a):
    R = {}
    a.comp_tol = 1e-3
    T = ot.target_setup(a, R)
    return T, R["target"]


def train(a):
    T, tgt = setup_target(a)
    lat, nup = T["lat"], T["nup"]
    net = make_net(lat, nup, a)
    th = net.init(np.random.default_rng(2000 + a.seed), 2 * nup / lat.N, a.t, a.tp, a.delta0, a.eta)
    cl = rc.Cluster(net, T["basis"], T["U"], T["HG"])
    rc.log(f"train {a.tgt} free_f={a.free_f} spin_feat={a.spin_feat} jas2={a.jas2} bnd_f={a.bnd_f} npar={net.npar} "
           f"lr={a.lr} shift={a.diag_shift}")
    th, h = optimise_ckpt(cl, th, a.steps, a.lr, a.diag_shift, a.chunk, "train", T["E0"], T["V"])
    psi = rc.state_vector(net, th, T["basis"], a.chunk)
    b = T["U"].T @ psi
    w = float(b @ b)
    E = float(b @ (T["HG"] @ b)) / w
    sb = rc.sign_budget(T["HG"], b, T["E0"], a)
    rel = (E - T["E0"]) / abs(T["E0"])
    share = sb["A"] / max(1e-300, E - T["E0"])
    res = dict(params=vars(a), target=tgt, npar=net.npar, hist=h, E=E, relerr=rel, A=sb["A"],
               sign_share=share, irrep_weight=w, eps_at={k: h[k] for k in (50, 150, 300, 600) if k < len(h)})
    rc.log(f"RESULT {a.tgt} free_f={a.free_f} spin_feat={a.spin_feat} jas2={a.jas2} bnd_f={a.bnd_f} npar={net.npar}: "
           f"E={E:.8f} relerr={rel:.4%} A={sb['A']:.4f} sign share={share:.3f} irrep weight={w:.3f} "
           f"eps(150)={h[min(150, len(h) - 1)]:.4e} eps(end)={h[-1]:.4e}")
    return res


def lm(a):
    d = json.load(open(a.json))
    p = d["params"]
    for k in ("tgt", "tp", "J", "t", "channels", "field_hidden", "bf_hidden", "jas_hidden", "kq", "r_min"):
        setattr(a, k, p[k])
    a.free_f, a.spin_feat, a.jas2, a.bnd_f = False, False, "none", 0
    th = np.array(d["arms"][a.record]["theta"])
    T, tgt = setup_target(a)
    net = ot.new_net(T["lat"], T["nup"], a)
    assert net.npar == len(th)
    t0 = time.time()
    psi = rc.state_vector(net, th, T["basis"], a.chunk)
    b = T["U"].T @ psi
    n2 = float(b @ b)
    cache = net.precompute(th, True)
    Bm = np.zeros((T["U"].shape[1], net.npar))
    for s in range(0, len(T["basis"]), a.chunk):
        sl = slice(s, s + a.chunk)
        _, _, O = net.evaluate(th, T["basis"][sl], True, cache)
        Bm += T["U"][sl].T @ (psi[sl, None] * O)
    rc.log(f"LM: B built ({Bm.shape}) in {time.time() - t0:.0f}s")
    p0 = b / np.sqrt(n2)
    Hp0 = T["HG"] @ p0
    E = float(p0 @ Hp0)
    Tm = (Bm - np.outer(b, b @ Bm) / n2) / np.sqrt(n2)
    del Bm
    HT = np.asarray(T["HG"] @ Tm)
    g = 2 * (Tm.T @ Hp0)
    H0k = Tm.T @ Hp0
    Hkl = Tm.T @ HT
    Skl = Tm.T @ Tm
    Hkl = (Hkl + Hkl.T) / 2
    E0, V = T["E0"], T["V"]
    out = dict(record=a.record, json=a.json, npar=net.npar, E=E, E0=E0, relerr=(E - E0) / abs(E0),
               grad_norm=float(np.linalg.norm(g)), lm={})
    rc.log(f"LM {a.json} {a.record}: npar={net.npar} E={E:.8f} relerr={(E - E0) / abs(E0):.4%} |g|={np.linalg.norm(g):.3e}")
    se, sv = np.linalg.eigh(Skl)
    for cut in [float(x) for x in a.lm_cut.split(",")]:
        keep = se > cut * se.max()
        X = sv[:, keep] / np.sqrt(se[keep])
        Hr = np.zeros((keep.sum() + 1, keep.sum() + 1))
        Hr[0, 0] = E
        Hr[0, 1:] = Hr[1:, 0] = X.T @ H0k
        Hr[1:, 1:] = X.T @ Hkl @ X
        w = np.linalg.eigvalsh(Hr)
        Elm = float(w[0])
        gain = (E - Elm) / (E - E0)
        out["lm"][str(cut)] = dict(kept=int(keep.sum()), E_LM=Elm, relerr_LM=(Elm - E0) / abs(E0), gain=gain)
        rc.log(f"  cut {cut:g}: kept {keep.sum()}/{net.npar} E_LM={Elm:.8f} relerr_LM={(Elm - E0) / abs(E0):.4%} "
               f"LM gain fraction (E-E_LM)/(E-E0)={gain:.4f}")
    return out


def parse_args(argv):
    p = argparse.ArgumentParser(description="P4.2 r5 critic capacity diagnostics (see docstring)")
    p.add_argument("--mode", choices=("lm", "train"), default=None)
    p.add_argument("--json", default="")
    p.add_argument("--record", default="scr@600")
    p.add_argument("--lm-cut", default="1e-6,1e-8,1e-10")
    p.add_argument("--tgt", default="4x3:5")
    p.add_argument("--tp", type=float, default=-0.25)
    p.add_argument("--J", type=float, default=0.4)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--field-hidden", type=int, default=16)
    p.add_argument("--bf-hidden", type=int, default=16)
    p.add_argument("--jas-hidden", type=int, default=16)
    p.add_argument("--free-f", action="store_true")
    p.add_argument("--spin-feat", action="store_true")
    p.add_argument("--jas2", default="none", choices=("none", "c", "cs"))
    p.add_argument("--bnd-f", type=int, default=0)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--diag-shift", type=float, default=0.01)
    p.add_argument("--kq", type=int, default=8)
    p.add_argument("--delta0", type=float, default=0.3)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--chunk", type=int, default=50000)
    p.add_argument("--r-min", type=float, default=2.0)
    p.add_argument("--fn-iters", type=int, default=6)
    p.add_argument("--fn-floor", type=float, default=1e-3)
    p.add_argument("--lbfgs-iters", type=int, default=300)
    p.add_argument("--out", default="")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if not a.selftest and (a.mode is None or not a.out):
        p.error("--mode and --out are required unless --selftest")
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.selftest:
        sys.exit(0 if selftest(a) else 2)
    res = train(a) if a.mode == "train" else lm(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    rc.log(f"done ({a.out})")


if __name__ == "__main__":
    main()
