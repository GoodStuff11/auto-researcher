#!/usr/bin/env python3
r"""
P4.2 round 5, part 2: decisive test of the mutated size-transfer routes R5.1 and R5.2 on the shared
4x4 OBC t-t'-J target (6+6 electrons, t' = -0.25, J = 0.4), scored by STEPS-TO-EPS of projected SR
against a projected from-scratch control, with the P2.1 projected NNBF as the premise control.
=================================================================================================

Failed component (Route 3.1, round 3-5): WHAT is transferred. A 122-parameter MFNG fitted exactly on
8-site sources is an arbitrary point of a ~110-dimensional solution manifold; on the 16-site target
it is 2-6x worse than the untrained mean-field (MF) state (pilot 410960), and resetting the
linearised source-null directions to the MF init destroys the source fit (close31 smoke), because
those directions are null only to first order. Operation A replaces the transferred object:

  R5.1 PA  "prior-anchored transfer" (L2-SP). Sources are fitted with a penalty
           anchor * |theta - theta_init|^2 that pulls every direction the sources do not need back
           to the source run's own MF initialisation (a nonlinear, exact version of the null
           filter: the fit, not a linearisation, decides what moves). The whole MFNG is copied
           (size independent) and fine-tuned at the target by projected SR.
  R5.2 ELT "environment-library local transfer". Only the LOCAL parts (backflow phi_m and charge
           Jastrow: functions of the site's hole environment) are transferred; the pair kernel
           (fields f_m(k)) is rebuilt from the target MF. The library adds 12-site OBC clusters with
           4 holes, so the 3- and 4-hole environments of the target are seen at the source. Premise
           (coverage): at least 95% of the target ground state's site-environment weight lies on
           environment tuples that some library ground state populates (weight >= 1e-4).
           Also anchored (same anchor as PA).
Ablation  CP  "copy": same sources as PA, no anchor (Route 3.1 transplanted to OBC).
Control   SCR MFNG from the target MF init (the matched-architecture scratch control).
Premise control: projected NNBF width 6 (1734 parameters) of P2.1, results read from
  ../P2.1_subcubic_amplitudes_r3/results/inst0_nnbf_s{11,22,33}.json (job 410988 and 451706).

Model and lattice
-----------------
t-t'-J on an Lx x Ly OPEN cluster: t on NN bonds, t' on the two diagonals, J (S.S - n n/4) on NN
bonds, no double occupancy, H = -sum t_ij c~^+ c~ + ... (sign convention of r3_core.build_H; the
4x4 6+6 ground energy is checked against P2.1's ED, -11.388010886295234, to 1e-8).
Size-independent MFNG on OBC (class MFNGO): the pair kernels are
kappa_m(r) = (1/Q^2) sum_{k in Q x Q grid} cos(k.r) f_m(k) on a FIXED quadrature grid (--kq, default
8), so kappa_m is the identical function of the displacement r on every cluster and the copy is
exact; site features v_s = (h_s, sum_nn h~/4, sum_nnn h~/4) with h~ = 1 for a missing neighbour
("outside the cluster counts as empty"). Exact singlet. Network sizes --channels/--field-hidden/
--bf-hidden/--jas-hidden (defaults 4/16/16/16: 426 parameters; 30220 / 426 = 71 projected-singlet
states per parameter).

Symmetry: the target is trained and scored in the ED irrep of its point group {1, Mx, My, MxMy} x
{1, D} (|G| = 8, class OBCSector; isometry U, H_G = U^T H U), by the projected exact-sum SR of
r3_core (r5_common.optimise_ckpt: one continuous trajectory). Sources are trained unprojected.

Arms and Slurm roles (--task-id 0..11: arm = ARMS[task // 3], seed = task % 3)
------------------------------------------------------------------------------
  scr  (tasks 0-2)  target MF init (rng 2000 + seed), --steps target steps
  cp   (tasks 3-5)  sources --src-a, anchor 0, best of --src-restarts by source loss; copy; --steps
  pa   (tasks 6-8)  sources --src-a, anchor --anchor; copy; --steps                   [R5.1]
  env  (tasks 9-11) sources --src-env, anchor --anchor; backflow + Jastrow copied, fields from the
                    target MF init (rng 2000 + seed); --steps                            [R5.2]
Every arm records its eps trajectory eps(n) = (E_n - E0)/V (projected), full records at 0 and at
every checkpoint, and the fixed-sign excess A (P1.1 budget in the irrep) at 0 and at the last two
checkpoints. Roles cp, pa and env also record the source fit, and env the coverage.

Decision (pre-committed; --aggregate). m(.) = median over 3 seeds.
---------------------------------------------------------------
  eps_best = lowest eps of any arm, step or seed; eps_opt is bracketed in [0, eps_best] and every
  ratio criterion is required at both ends ("robust").
  PREMISE (brief): median dE_sign of the projected NNBF control (P2.1, 3 seeds) >= 0.005 t, else
           every route UNINFORMATIVE; fewer than 3 control seeds -> PENDING-PREMISE.
  VOID     unless eps_best * V / |E0| <= --void-tol (0.01): the transferable family must itself
           come within 1% of E0 for transfer to matter for finite-size work.
  COUNTED B in --ckpts-rule (default 10,20,50): m eps_scr(B) - eps_best >= 1e-3 t/site and
           m [eps_scr(B-5) - eps_scr(B)] >= 0.02 m [eps_scr(B) - eps_best] (plateau check).
  speed-up S_X(B) = B / m n_X(l_B), l_B = m eps_scr(B), n_X = first step with eps_X <= l_B.
  Route X in {pa (R5.1), env (R5.2)} PASS if all of
    S_X(B) >= 2 at >= 2 counted B;  robust m eps_X(20) - x <= 0.5 (m eps_scr(20) - x);
    ablation: m eps_X(20) < m eps_cp(20) (pa) or < m eps_pa(20) (env: the library must add to PA);
    sign: m A_X(last) <= m A_scr(last);   env only: coverage premise (>= 0.95).
  FAIL if S_X(B) < 1.25 at every counted B, or robust m eps_X(20) - x >= 0.9 (m eps_scr(20) - x),
    or (env) coverage < 0.95.  AMBIGUOUS otherwise (not promoted).
  Hard Core: in-irrep gap per site reported; competitor flag below COMP_TOL = 1e-3 t/site,
  otherwise "sign half only".

Command-line options
--------------------
  --task-id INT     Slurm index 0..11 (sets --arm, --seed). Default -1 (use --arm/--seed).
  --arm STR         scr | cp | pa | env.          --seed INT  0, 1, 2 (default 0).
  --tgt STR         Target "LxXLy:nup" OBC (default "4x4:6").
  --tp, --J, --t    Couplings (-0.25, 0.4, 1.0).
  --src-a STR       Sources of cp/pa, comma list of "LxXLy:nup" OBC clusters (default "4x2:3,2x4:3").
  --src-env STR     Library of env (default "4x2:3,2x4:3,4x3:4,3x4:4").
  --anchor FLOAT    L2-SP strength on |theta - theta_init|^2, per parameter (default 1e-2).
  --src-restarts INT (4)  --steps-src INT (800): source restarts (best by source loss) and steps.
  --steps INT       Target steps per arm (default 150).
  --ckpts STR       Record checkpoints (default "10,20,50,100,150").
  --ckpts-rule STR  B values of the rule (default "10,20,50").
  --channels, --field-hidden, --bf-hidden, --jas-hidden   MFNG sizes (4, 16, 16, 16).
  --kq INT          Quadrature grid of the size-independent kernels (default 8).
  --delta0, --eta   MF-init d-wave amplitude and regulariser (0.3, 0.1).
  --lr, --diag-shift  SR step and relative diagonal shift (0.05, 0.01).
  --fn-iters, --fn-floor, --lbfgs-iters  fixed-sign budget settings (6, 1e-3, 3000).
  --r-min FLOAT     Pd minimal bond distance (2.0; Pd is a diagnostic only, minimum-image distance).
  --chunk INT       Configurations per chunk (50000).
  --wall-hours FLOAT  Stop target steps after this many hours (default 22).
  --nnbf-dir PATH   P2.1 results folder (premise control).
  --out PATH        Output JSON (default results/{arm}_s{seed}.json).
  --aggregate DIR   Decision from every JSON in DIR (no computation).
  --void-tol, --comp-tol, --plateau-frac, --premise-sign  rule constants (0.01, 1e-3, 0.02, 0.005).
  --selftest        Checks: OBC lattice vs P2.1 conventions on 3x2 (E0 vs exact diagonalisation of
                    the dense H, and 3x2 / 4x3 OBC energies vs P2.1 tj_core), [H, P_g] = 0 and U^T U = I on a small OBC cluster, MFNGO
                    log-derivatives vs finite differences, MFNGO kernel identical on two cluster
                    sizes, anchored-wrapper gradient identity. Exit 2 on failure.
  --timing          One gradient and value pass on --chunk target configurations, extrapolated.

Examples
--------
  python3 obc_transfer.py --selftest
  python3 obc_transfer.py --timing
  python3 obc_transfer.py --task-id 7                      # one Slurm task
  python3 obc_transfer.py --aggregate results
  # login smoke (trivial target, 2+2 electrons; checks the code, not the verdict):
  for t in 0..11: python3 obc_transfer.py --task-id $t --tgt 4x4:2 --src-a 4x2:1,2x4:1 \
      --src-env 4x2:1,2x4:1,4x3:2,3x4:2 --steps-src 60 --src-restarts 2 --steps 12 \
      --ckpts 2,4,6,12 --ckpts-rule 6,12 --lbfgs-iters 300 --out results_smoke/task$t.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r3_core as rc  # noqa: E402
from r5_common import med, optimise_ckpt, steps_to  # noqa: E402

ARMS = ("scr", "cp", "pa", "env")
P21_E0 = -11.388010886295234        # P2.1 instance 0 ED ground energy (tests/P2.1_subcubic_amplitudes_r3)
P21_REF = {"3x2:2": -4.062964411903416, "4x3:3": -9.16758855}   # tj_core.TJ OBC ground energies
PREM_EPS = 1e-3
COVER_MIN, COVER_W = 0.95, 1e-4


# ============================================================================ OBC lattice
class OBC:
    """Open Lx x Ly cluster with the interface MFNG / build_H / Pairing expect. Missing
    neighbours in nn / nnn are the ghost index N."""

    def __init__(self, Lx, Ly, t, tp, J, kq=8):
        self.Lx, self.Ly, self.N = Lx, Ly, Lx * Ly
        self.phx = self.phy = 0.0
        N = self.N
        ins = lambda x, y: 0 <= x < Lx and 0 <= y < Ly
        s = lambda x, y: x + Lx * y if ins(x, y) else N
        self.site = s
        self.hop, self.jb, self.xb = [], [], []
        for y in range(Ly):
            for x in range(Lx):
                for (dx, dy) in ((1, 0), (0, 1)):
                    if ins(x + dx, y + dy):
                        self.hop.append((s(x, y), s(x + dx, y + dy), t))
                        self.jb.append((s(x, y), s(x + dx, y + dy), J))
                        self.xb.append((s(x, y), s(x + dx, y + dy), "x" if dx else "y", x, y))
                if tp != 0.0:
                    for dy in (1, -1):
                        if ins(x + 1, y + dy):
                            self.hop.append((s(x, y), s(x + 1, y + dy), tp))
        self.xs = np.array([i % Lx for i in range(N)])
        self.ys = np.array([i // Lx for i in range(N)])
        self.nn = np.array([[s(x + 1, y), s(x - 1, y), s(x, y + 1), s(x, y - 1)]
                            for y in range(Ly) for x in range(Lx)])
        self.nnn = np.array([[s(x + 1, y + 1), s(x - 1, y + 1), s(x + 1, y - 1), s(x - 1, y - 1)]
                             for y in range(Ly) for x in range(Lx)])
        ddx = self.xs[:, None] - self.xs[None, :]
        ddy = self.ys[:, None] - self.ys[None, :]
        self.disp = (ddx + Lx - 1) + (2 * Lx - 1) * (ddy + Ly - 1)
        m, n = np.meshgrid(np.arange(kq), np.arange(kq), indexing="xy")
        self.kx = (2 * np.pi * m / kq).ravel()
        self.ky = (2 * np.pi * n / kq).ravel()
        gx, gy = np.meshgrid(np.arange(-(Lx - 1), Lx), np.arange(-(Ly - 1), Ly), indexing="xy")
        self.gx, self.gy = gx.ravel(), gy.ravel()
        self.Cd = np.cos(np.outer(self.gx, self.kx) + np.outer(self.gy, self.ky)) / (kq * kq)

    def name(self):
        return f"{self.Lx}x{self.Ly}o"

    def square(self):
        return self.Lx == self.Ly


class OBCSector:
    """Point group {1, Mx, My, MxMy} x {1, D} (D only if square) of an open cluster; real 1-D
    irreps lam = {Mx, My[, D]}; isometry U onto Pi_lam (same construction as r3_core.Sector)."""

    def __init__(self, lat, basis):
        self.lat, self.D = lat, len(basis)
        Lx, Ly = lat.Lx, lat.Ly
        self.gens = ["Mx", "My"] + (["D"] if lat.square() else [])
        self.pg = []
        for e in ((0, 1) if lat.square() else (0,)):
            for c in (0, 1):
                for d in (0, 1):
                    perm = np.empty(lat.N, dtype=np.int64)
                    for i in range(lat.N):
                        x, y = lat.xs[i], lat.ys[i]
                        if e:
                            x, y = y, x
                        if c:
                            x = Lx - 1 - x
                        if d:
                            y = Ly - 1 - y
                        perm[i] = x + Lx * y
                    self.pg.append(((c, d, e), rc.site_perm_map(lat, basis, perm)))
        self.genmap = {"Mx": [m for k, m in self.pg if k == (1, 0, 0)][0],
                       "My": [m for k, m in self.pg if k == (0, 1, 0)][0]}
        if lat.square():
            self.genmap["D"] = [m for k, m in self.pg if k == (0, 0, 1)][0]

    def irrep_of(self, psi):
        psi = psi / np.linalg.norm(psi)
        ev = {g: float(psi @ rc.apply_map(self.genmap[g], psi)) for g in self.gens}
        ok = all(abs(abs(v) - 1) < 1e-6 for v in ev.values())
        return ok, {k: int(np.sign(v)) for k, v in ev.items()}, ev

    def isometry(self, lam):
        D = self.D
        rep = np.full(D, D, dtype=np.int64)
        c = np.zeros(D, dtype=np.int64)
        for (cc, dd, ee), P in self.pg:
            ch = lam["Mx"] ** cc * lam["My"] ** dd * (lam.get("D", 1) ** ee)
            idx, s = P
            lt = idx < rep
            eq = idx == rep
            c = np.where(eq, c + ch * s, c)
            c[lt] = ch * s[lt]
            rep[lt] = idx[lt]
        allowed = c != 0
        reps, inv = np.unique(rep[allowed], return_inverse=True)
        norm = np.bincount(inv, weights=c[allowed].astype(float) ** 2)
        rows = np.nonzero(allowed)[0]
        return sp.csr_matrix((c[allowed] / np.sqrt(norm[inv]), (rows, inv)), shape=(D, len(reps)))


# ============================================================================ ansatz
def site_features(lat, states):
    """(hole (B, N), v (B, N, 3)) with v_s = (h_s, sum_nn h~ / 4, sum_nnn h~ / 4), h~ = 1 outside."""
    N = lat.N
    nu = ((states[:, None] >> np.arange(N, dtype=rc.U64)) & rc.U64(1)).astype(np.int64)
    nd = ((states[:, None] >> np.arange(N, 2 * N, dtype=rc.U64)) & rc.U64(1)).astype(np.int64)
    hole = (1 - nu - nd).astype(float)
    hp = np.concatenate([hole, np.ones((len(states), 1))], 1)
    v = np.stack([hole, hp[:, lat.nn].sum(2) / 4, hp[:, lat.nnn].sum(2) / 4], 2)
    return nu, nd, hole, v


class MFNGO(rc.MFNG):
    """MFNG on an open cluster: identical to r3_core.MFNG except the site features treat a missing
    neighbour as a hole (ghost index N). evaluate() is r3_core.MFNG.evaluate with that change."""

    def evaluate(self, th, states, need_grad, cache=None):
        lat, N, n, C = self.lat, self.lat.N, self.n, self.C
        B = len(states)
        if cache is None:
            cache = self.precompute(th, need_grad)
        kap, CJ = cache
        nu, nd, hole, v = site_features(lat, states)
        jW = self.P(th, "jW", (self.Hj, 3)); jb = th[self.sl["jb"]]; ja = th[self.sl["ja"]]
        zj = v @ jW.T + jb
        lc = np.logaddexp(zj, -zj) - np.log(2.0)
        logJ = (lc @ ja).sum(1)
        ups = np.nonzero(nu)[1].reshape(B, n)
        dns = np.nonzero(nd)[1].reshape(B, n)
        D = lat.disp[ups[:, :, None], dns[:, None, :]]
        bi = np.arange(B)[:, None]
        A = kap[0][D].copy()
        if C > 0:
            bW = self.P(th, "bW", (self.Hb, 3)); bb = th[self.sl["bb"]]; ba = self.P(th, "ba", (C, self.Hb))
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
        O[~good] = 0.0
        return logabs, sgn, O


class Anchored:
    """theta -> net(theta_init + theta): lets r3_core.optimise's lam |theta|^2 act as the L2-SP
    anchor lam |theta_net - theta_init|^2."""

    def __init__(self, net, th_init):
        self.net, self.th0 = net, th_init
        self.lat, self.npar, self.jas_mask = net.lat, net.npar, net.jas_mask

    def precompute(self, th, need_grad):
        return self.net.precompute(th + self.th0, need_grad)

    def evaluate(self, th, states, need_grad, cache=None):
        return self.net.evaluate(th + self.th0, states, need_grad, cache)


def new_net(lat, nup, a, C=None):
    return MFNGO(lat, nup, a.channels if C is None else C, a.field_hidden, a.bf_hidden, a.jas_hidden)


def local_mask(net):
    """Parameters of the local parts (backflow phi_m and Jastrow)."""
    m = np.zeros(net.npar, bool)
    for nm in ("bW", "bb", "ba", "jW", "jb", "ja"):
        m[net.sl[nm]] = True
    return m


# ============================================================================ clusters
def parse_spec(s):
    l, n = s.split(":")
    Lx, Ly = rc.parse_lat(l)
    return Lx, Ly, int(n)


def build_cluster(spec, a, target=False):
    Lx, Ly, nup = parse_spec(spec)
    lat = OBC(Lx, Ly, a.t, a.tp, a.J, a.kq)
    basis = rc.make_basis(lat.N, nup, nup)
    H = rc.build_H(lat, basis)
    w, v = rc.ed_lowest(H, k=3)
    return dict(lat=lat, basis=basis, H=H, E0=float(w[0]), psi0=v[:, 0], nup=nup, w=w)


def env_weights(lat, basis, psi, chunk=200000):
    """Weight of each site-environment tuple (h, a, b) (integers: own hole, holes+missing among
    nn, among nnn) in |psi|^2, averaged over sites."""
    out = {}
    p = psi ** 2 / float(psi @ psi)
    for s0 in range(0, len(basis), chunk):
        sl = slice(s0, s0 + chunk)
        _, _, _, v = site_features(lat, basis[sl])
        key = (v[:, :, 0] * 100 + np.round(v[:, :, 1] * 4) * 10 + np.round(v[:, :, 2] * 4)).astype(np.int64)
        w = np.repeat(p[sl][:, None], lat.N, 1) / lat.N
        u, inv = np.unique(key.ravel(), return_inverse=True)
        acc = np.bincount(inv, weights=w.ravel())
        for k, x in zip(u.tolist(), acc.tolist()):
            out[k] = out.get(k, 0.0) + x
    return out


def coverage(tgt_w, srcs_w):
    seen = set(k for sw in srcs_w for k, x in sw.items() if x >= COVER_W)
    return float(sum(x for k, x in tgt_w.items() if k in seen)), sorted(k for k in tgt_w if k not in seen)


def train_sources(a, specs, anchor, seed, tag):
    src = [build_cluster(s, a) for s in specs]
    dens = 2 * src[0]["nup"] / src[0]["lat"].N
    best, losses = None, []
    for r in range(a.src_restarts):
        net0 = new_net(src[0]["lat"], src[0]["nup"], a)
        th_init = net0.init(np.random.default_rng(1000 + 100 * seed + r), dens, a.t, a.tp, a.delta0, a.eta)
        cls = [rc.Cluster(Anchored(new_net(s["lat"], s["nup"], a), th_init), s["basis"],
                          sp.identity(len(s["basis"]), format="csr"), s["H"]) for s in src]
        d, Es, _ = rc.optimise(cls, np.zeros(net0.npar), a.steps_src, a.lr, a.diag_shift, a.chunk,
                               f"{tag} restart {r}", lam=anchor, Eref=[s["E0"] for s in src])
        L = sum(e / c.V for e, c in zip(Es, cls)) + anchor * float(d @ d)
        losses.append(L)
        if best is None or L < best[0]:
            best = (L, th_init + d, th_init, Es)
    _, th, th_init, Es = best
    rel = [float((e - s["E0"]) / abs(s["E0"])) for e, s in zip(Es, src)]
    rc.log(f"source {tag}: relerr={['%.2e' % x for x in rel]} |theta-theta_init|={np.linalg.norm(th - th_init):.3f}")
    return th, th_init, src, dict(specs=specs, anchor=anchor, E=[float(e) for e in Es], relerr=rel,
                                  restart_losses=losses, dist_init=float(np.linalg.norm(th - th_init)))


# ============================================================================ run
def target_setup(a, R):
    T = build_cluster(a.tgt, a, target=True)
    lat, basis, H = T["lat"], T["basis"], T["H"]
    if a.tgt == "4x4:6" and abs(a.tp + 0.25) < 1e-12 and abs(a.J - 0.4) < 1e-12:
        assert abs(T["E0"] - P21_E0) < 1e-8, f"E0 {T['E0']} differs from P2.1 {P21_E0}"
    deg = T["w"][1] - T["w"][0] < 1e-8 * abs(T["w"][0])
    sec = OBCSector(lat, basis)
    ok, lam, ev = sec.irrep_of(T["psi0"])
    s2 = rc.S2_expect(lat, basis, T["psi0"])
    assert ok and not deg and abs(s2) < 1e-6, f"target GS not a non-degenerate 1-D singlet: {ev} S2={s2}"
    U = sec.isometry(lam)
    HG = (U.T @ H @ U).tocsr()
    w, v = rc.ed_lowest(HG, k=2)
    E0 = float(w[0]); V = lat.N
    psi0 = U @ v[:, 0]
    pair = rc.Pairing(lat, basis, T["nup"], T["nup"], a.r_min)
    npar = new_net(lat, T["nup"], a).npar
    gap = float(w[1] - w[0]) / V
    R["target"] = dict(spec=a.tgt, tp=a.tp, J=a.J, irrep=lam, dim=len(basis), n_irrep=int(U.shape[1]), npar=npar,
                       E0=E0, E0_full=T["E0"], gap_per_site=gap, competitor=bool(gap < a.comp_tol),
                       Pd_E0=pair.Pd(psi0))
    rc.log(f"target {lat.name()} {a.tgt} irrep={lam} dim={len(basis)} n_irrep={U.shape[1]} E0={E0:.10f} "
           f"gap/V={gap:.4e} ({'COMPETITOR' if gap < a.comp_tol else 'sign half only'}) npar={npar}")
    return dict(lat=lat, basis=basis, H=H, U=U, HG=HG, E0=E0, V=V, psi0=psi0, pair=pair, nup=T["nup"])


def run(a):
    t0 = time.time()
    out = a.out or os.path.join(HERE, "results", f"{a.arm}_s{a.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    R = dict(params=vars(a).copy(), arm=a.arm, seed=a.seed, arms={})

    def dump():
        with open(out, "w") as fh:
            json.dump(R, fh, indent=1, default=float)

    T = target_setup(a, R)
    dump()
    a.tp_sel, a.J_sel = a.tp, a.J
    record = rc.make_recorder(a, R, T, dump)
    lat, nup = T["lat"], T["nup"]
    net = new_net(lat, nup, a)
    th_tgt_init = net.init(np.random.default_rng(2000 + a.seed), 2 * nup / lat.N, a.t, a.tp, a.delta0, a.eta)
    if a.arm == "scr":
        th = th_tgt_init
    elif a.arm in ("cp", "pa"):
        th, _, _, info = train_sources(a, a.src_a.split(","), 0.0 if a.arm == "cp" else a.anchor, a.seed, a.arm)
        R["source"] = info
    else:
        ths, _, src, info = train_sources(a, a.src_env.split(","), a.anchor, a.seed, "env")
        lm = local_mask(net)
        th = th_tgt_init.copy()
        th[lm] = ths[lm]
        tw = env_weights(lat, T["basis"], T["psi0"])
        sws = [env_weights(s["lat"], s["basis"], s["psi0"]) for s in src]
        cov, miss = coverage(tw, sws)
        cov_a, _ = coverage(tw, [sw for s, sw in zip(a.src_env.split(","), sws) if s in a.src_a.split(",")])
        info.update(coverage=cov, coverage_src_a=cov_a, uncovered=miss)
        R["source"] = info
        rc.log(f"env coverage of target environments: library {cov:.4f}, src-a only {cov_a:.4f}; uncovered {miss}")
    dump()
    ck = sorted({int(x) for x in a.ckpts.split(",")})
    sign_at = {0} | set(ck[-2:])
    cl = rc.Cluster(net, T["basis"], T["U"], T["HG"])
    _, h = optimise_ckpt(cl, th, a.steps, a.lr, a.diag_shift, a.chunk, a.arm, T["E0"], T["V"], ckpts=[0] + ck,
                         on_ckpt=lambda n, t_: record(f"{a.arm}@{n}", net, t_, sign=(n in sign_at)),
                         deadline=t0 + a.wall_hours * 3600)
    R["hist"] = h
    R["done"] = True
    R["wall_s"] = time.time() - t0
    dump()
    rc.log(f"done ({out})")


# ============================================================================ aggregate
def nnbf_premise(d):
    fs = sorted(glob.glob(os.path.join(d, "inst0_nnbf_s*.json")))
    vals = []
    for f in fs:
        r = json.load(open(f))
        b = (r.get("budgets") or {}).get("P")
        if b:
            vals.append(dict(seed=r["seed"], dE_sign=b["dE_sign"], dE_tot=b["dE_tot"]))
    return vals


def aggregate(a):
    rs = {}
    for f in sorted(glob.glob(os.path.join(a.aggregate, "*.json"))):
        r = json.load(open(f))
        if r.get("done"):
            rs.setdefault(r["arm"], []).append(r)
    miss = [k for k in ARMS if len(rs.get(k, [])) < 3]
    if miss:
        print(f"INCOMPLETE: arms with < 3 seeds: {miss}")
        return
    T = rs["scr"][0]["target"]
    V = rc.parse_lat(T["spec"].split(":")[0])[0] * rc.parse_lat(T["spec"].split(":")[0])[1]
    H = {k: [r["hist"] for r in sorted(rs[k], key=lambda r: r["seed"])] for k in ARMS}
    at = lambda hs, n: [h[n] if n < len(h) else float("nan") for h in hs]
    eps_best = min(min(h) for k in H for h in H[k])
    relerr = eps_best * V / abs(T["E0"])
    prem = nnbf_premise(a.nnbf_dir)
    print(f"target {T['spec']} OBC t'={T['tp']} J={T['J']} irrep={T['irrep']} E0={T['E0']:.10f} n_irrep={T['n_irrep']} "
          f"npar={T['npar']} gap/V={T['gap_per_site']:.4e} -> {'COMPETITOR' if T['competitor'] else 'sign half only'}")
    print(f"  premise control (P2.1 projected NNBF): {prem}")
    print(f"  eps_best={eps_best:.4e} relerr={relerr:.3e} (VOID unless <= {a.void_tol})")
    rb = sorted({int(x) for x in a.ckpts_rule.split(",")})
    rows, counted = {}, []
    for B in rb:
        ms = med(at(H["scr"], B))
        mov = med([h[B - 5] - h[B] if B < len(h) and B >= 5 else float("nan") for h in H["scr"]])
        c = (ms - eps_best >= PREM_EPS) and (mov >= a.plateau_frac * (ms - eps_best))
        row = dict(m_scr=ms, counted=bool(c))
        for k in ("cp", "pa", "env"):
            n = [steps_to(h, ms) for h in H[k]]
            nm = float("inf") if sum(x is None for x in n) >= 2 else med([x if x is not None else float("inf") for x in n])
            row[f"S_{k}"] = (B / nm if nm > 0 else float("inf")) if np.isfinite(nm) else 0.0
            row[f"m_{k}"] = med(at(H[k], B))
        rows[B] = row
        if c:
            counted.append(B)
        print(f"  B={B:4d}: " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    m20 = {k: med(at(H[k], 20)) for k in ARMS}
    Alast = {k: med([r["arms"].get(f"{k}@{max(int(x) for x in a.ckpts.split(','))}", {}).get("A", float("nan"))
                     for r in rs[k]]) for k in ARMS}
    cov = med([r["source"].get("coverage") for r in rs["env"]])
    print(f"  m eps(20)={ {k: round(v, 5) for k, v in m20.items()} }  m A(last)={ {k: round(v, 5) for k, v in Alast.items()} }  "
          f"env coverage={cov:.4f}")
    if len(prem) < 3:
        print("GLOBAL: PENDING-PREMISE (fewer than 3 projected-NNBF control seeds)")
        return
    if med([p["dE_sign"] for p in prem]) < a.premise_sign:
        print(f"GLOBAL: UNINFORMATIVE (projected NNBF median dE_sign < {a.premise_sign} t)")
        return
    if relerr > a.void_tol:
        print("GLOBAL: VOID (the size-independent family does not reach 1% of E0)")
        return
    for X, abl, name in (("pa", "cp", "R5.1 PA"), ("env", "pa", "R5.2 ELT")):
        sp_ok = sum(rows[B][f"S_{X}"] >= 2 for B in counted) >= 2
        b_ok = all(m20[X] - x <= 0.5 * (m20["scr"] - x) for x in (0.0, eps_best))
        b_fail = all(m20[X] - x >= 0.9 * (m20["scr"] - x) for x in (0.0, eps_best))
        slow = len(counted) >= 1 and all(rows[B][f"S_{X}"] < 1.25 for B in counted)
        cov_ok = X != "env" or cov >= COVER_MIN
        P = dict(speed=sp_ok, b20=b_ok, ablation=m20[X] < m20[abl], sign=Alast[X] <= Alast["scr"], coverage=cov_ok)
        if slow or b_fail or not cov_ok:
            v = "FAIL (" + ", ".join(n for n, x in (("slow", slow), ("b-fail", b_fail), ("coverage", not cov_ok)) if x) + ")"
        elif all(P.values()):
            v = "PASS"
        else:
            v = "AMBIGUOUS (" + ", ".join(k for k, x in P.items() if not x) + " not met)"
        print(f"GLOBAL {name}: {v}  {P}")


# ============================================================================ self-tests
def selftest(a):
    ok = True
    b = argparse.Namespace(**vars(a))
    b.t, b.tp, b.J = 1.0, -0.25, 0.4
    # (a) OBC 3x2, 2+2: sparse H vs dense ED, Hermitian
    lat = OBC(3, 2, 1.0, -0.25, 0.4)
    basis = rc.make_basis(lat.N, 2, 2)
    H = rc.build_H(lat, basis)
    herm = abs(H - H.T).max()
    w = np.linalg.eigvalsh(H.toarray())
    rc.log(f"selftest(a) OBC 3x2 2+2 dim={len(basis)} |H-H^T|={herm:.1e} E0={w[0]:.10f} bonds hop={len(lat.hop)} "
           f"(expect 7 NN + 4 diag) {'OK' if herm < 1e-12 and len(lat.hop) == 11 else 'FAIL'}")
    ok &= herm < 1e-12 and len(lat.hop) == 11
    # (a') conventions vs P2.1 tj_core (reference energies computed with TJ(..., 'obc', 'obc', ...))
    l43 = OBC(4, 3, 1.0, -0.25, 0.4)
    e43 = rc.ed_lowest(rc.build_H(l43, rc.make_basis(12, 3, 3)), k=1)[0][0]
    good = abs(w[0] - P21_REF["3x2:2"]) < 1e-9 and abs(e43 - P21_REF["4x3:3"]) < 1e-7
    rc.log(f"selftest(a') vs P2.1 tj_core: 3x2:2 {w[0]:.10f} ({P21_REF['3x2:2']}), 4x3:3 {e43:.8f} "
           f"({P21_REF['4x3:3']}) {'OK' if good else 'FAIL'}")
    ok &= good
    # (b) symmetry on 4x4 OBC, 2+2
    lat = OBC(4, 4, 1.0, -0.25, 0.4)
    basis = rc.make_basis(lat.N, 2, 2)
    H = rc.build_H(lat, basis)
    sec = OBCSector(lat, basis)
    v = np.random.default_rng(0).standard_normal(len(basis))
    comm = max(np.abs(H @ rc.apply_map(m, v) - rc.apply_map(m, H @ v)).max() for _, m in sec.pg)
    w0, V0 = rc.ed_lowest(H, k=3)
    okk, lam, ev = sec.irrep_of(V0[:, 0])
    U = sec.isometry(lam)
    orth = abs((U.T @ U - sp.identity(U.shape[1])).toarray()).max()
    wG = np.linalg.eigvalsh((U.T @ H @ U).toarray())
    good = comm < 1e-12 and orth < 1e-12 and okk and abs(wG[0] - w0[0]) < 1e-9
    rc.log(f"selftest(b) 4x4 OBC 2+2: max|[H,P_g]v|={comm:.1e} U^T U=I ({orth:.1e}) GS irrep {lam} "
           f"E0 irrep {wG[0]:.10f} vs full {w0[0]:.10f} {'OK' if good else 'FAIL'}")
    ok &= good
    # (c) log-derivatives vs finite differences; (d) kernel identical on two sizes
    net = new_net(lat, 2, a)
    th = net.init(np.random.default_rng(1), 4 / 16, 1.0, -0.25, 0.3, 0.1)
    th = th + 0.05 * np.random.default_rng(2).standard_normal(len(th))
    st = basis[:40]
    la, sg, O = net.evaluate(th, st, True)
    fd = np.zeros_like(O)
    for p in range(len(th)):
        e = np.zeros(len(th)); e[p] = 1e-6
        fd[:, p] = (net.evaluate(th + e, st, False)[0] - net.evaluate(th - e, st, False)[0]) / 2e-6
    rel = np.abs(fd - O).max() / max(1e-12, np.abs(O).max())
    rc.log(f"selftest(c) MFNGO log-derivative max rel err over {len(th)} params = {rel:.2e} {'OK' if rel < 1e-5 else 'FAIL'}")
    ok &= rel < 1e-5
    small = OBC(4, 2, 1.0, -0.25, 0.4)
    ks = new_net(small, 1, a).precompute(th, False)[0]
    kb = net.precompute(th, False)[0]
    diffs = []
    for gi, (gx, gy) in enumerate(zip(small.gx, small.gy)):
        j = np.nonzero((lat.gx == gx) & (lat.gy == gy))[0][0]
        diffs.append(np.abs(ks[:, gi] - kb[:, j]).max())
    rc.log(f"selftest(d) kernel on 4x2o vs 4x4o at shared displacements: max diff {max(diffs):.1e} "
           f"{'OK' if max(diffs) < 1e-14 else 'FAIL'}")
    ok &= max(diffs) < 1e-14
    # (e) anchored wrapper: same amplitudes at theta = theta_init + d
    an = Anchored(net, th)
    d = 0.01 * np.random.default_rng(3).standard_normal(len(th))
    dd = np.abs(an.evaluate(d, st, False)[0] - net.evaluate(th + d, st, False)[0]).max()
    rc.log(f"selftest(e) anchored wrapper amplitude identity {dd:.1e} {'OK' if dd < 1e-14 else 'FAIL'}")
    ok &= dd < 1e-14
    rc.log("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return ok


def parse_args(argv):
    p = argparse.ArgumentParser(description="P4.2 r5 decisive test on the 4x4 OBC instance (see docstring)")
    p.add_argument("--task-id", type=int, default=-1)
    p.add_argument("--arm", default="scr", choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tgt", default="4x4:6")
    p.add_argument("--tp", type=float, default=-0.25)
    p.add_argument("--J", type=float, default=0.4)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--src-a", default="4x2:3,2x4:3")
    p.add_argument("--src-env", default="4x2:3,2x4:3,4x3:4,3x4:4")
    p.add_argument("--anchor", type=float, default=1e-2)
    p.add_argument("--src-restarts", type=int, default=4)
    p.add_argument("--steps-src", type=int, default=800)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--ckpts", default="10,20,50,100,150")
    p.add_argument("--ckpts-rule", default="10,20,50")
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--field-hidden", type=int, default=16)
    p.add_argument("--bf-hidden", type=int, default=16)
    p.add_argument("--jas-hidden", type=int, default=16)
    p.add_argument("--kq", type=int, default=8)
    p.add_argument("--delta0", type=float, default=0.3)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--diag-shift", type=float, default=0.01)
    p.add_argument("--fn-iters", type=int, default=6)
    p.add_argument("--fn-floor", type=float, default=1e-3)
    p.add_argument("--lbfgs-iters", type=int, default=3000)
    p.add_argument("--r-min", type=float, default=2.0)
    p.add_argument("--chunk", type=int, default=50000)
    p.add_argument("--wall-hours", type=float, default=22.0)
    p.add_argument("--nnbf-dir", default=os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3", "results"))
    p.add_argument("--out", default=None)
    p.add_argument("--aggregate", default="")
    p.add_argument("--void-tol", type=float, default=0.01)
    p.add_argument("--comp-tol", type=float, default=1e-3)
    p.add_argument("--plateau-frac", type=float, default=0.02)
    p.add_argument("--premise-sign", type=float, default=0.005)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--timing", action="store_true")
    a = p.parse_args(argv)
    if a.task_id >= 0:
        if a.task_id > 11:
            p.error("--task-id must be 0..11")
        a.arm, a.seed = ARMS[a.task_id // 3], a.task_id % 3
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.aggregate:
        aggregate(a)
        return
    if a.selftest:
        sys.exit(0 if selftest(a) else 2)
    if a.timing:
        Lx, Ly, nup = parse_spec(a.tgt)
        lat = OBC(Lx, Ly, a.t, a.tp, a.J, a.kq)
        basis = rc.make_basis(lat.N, nup, nup)
        net = new_net(lat, nup, a)
        th = net.init(np.random.default_rng(0), 2 * nup / lat.N, a.t, a.tp, a.delta0, a.eta)
        t1 = time.time(); net.evaluate(th, basis[:a.chunk], True); t2 = time.time()
        net.evaluate(th, basis[:a.chunk], False); t3 = time.time()
        nch = int(np.ceil(len(basis) / a.chunk))
        rc.log(f"target {a.tgt} OBC dim={len(basis)} npar={net.npar}: chunk {a.chunk} grad {t2 - t1:.2f}s value "
               f"{t3 - t2:.2f}s -> full grad+value ~{nch * (t3 - t1):.0f}s, value ~{nch * (t3 - t2):.0f}s per pass")
        return
    if not selftest(a):
        sys.exit(2)
    run(a)


if __name__ == "__main__":
    main()
