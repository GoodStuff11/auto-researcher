#!/usr/bin/env python3
r"""
P4.2 round 5, part 2 (critic-fixed): decisive test of the size-transfer route R5.1 (PA) on the shared
4x4 OBC t-t'-J target (6+6 electrons, t' = -0.25, J = 0.4), scored by OFFSET-FREE WALL-TIME
speed-up of projected SR against the stronger scratch control scr+, with the source cost charged.
=================================================================================================

Failed component (Route 3.1, rounds 3-5): WHAT is transferred. A 122-parameter MFNG fitted exactly
on 8-site sources is an arbitrary point of a ~110-dimensional solution manifold; on the 16-site
target it is 2-6x worse than the untrained mean-field (MF) state (pilot 410960). Operation A:

  R5.1 PA  "prior-anchored transfer" (L2-SP). Sources are fitted with a penalty
           anchor * |theta - theta_init|^2 toward the source run's own MF initialisation; the
           whole size-independent MFNG-O is copied and fine-tuned at the target by projected SR.
  R5.2 ELT was KILLED at Gate 0 by the per-leaf critic (routine VMC warm-starting from a library
           of small-cluster fits; no amortised one-library / many-target claim is made), so it has
           no arm here.
Ablation  cp    unanchored copy from the same sources (Route 3.1 moved to OBC).
Controls  scr   MFNG-O from the target MF init (capacity reference; also the CAPACITY PILOT).
          scr+  MFNG-O from the target MF init, steps 1..--scrp-k (default 20) on the KERNEL (field
                network) and JASTROW parameters only, then all parameters. The k cheap steps are
                charged (they are steps of its trajectory and of its clock). PA must beat scr+.

Critic fixes applied (P4.2 round-5 per-leaf critic)
---------------------------------------------------
 1 Capacity pilot first (--capacity DIR reads it): one scr task, 600 steps. Its plateau eps_inf
   replaces the old [0, eps_best] bracket everywhere (x = eps_inf below). Pre-committed: if
   eps_inf * V / |E0| > 0.03, do NOT run the array; mutate (enlarge --channels / --bf-hidden).
 2 VOID gates PASS only. FAIL, AMBIGUOUS and INVALID always print. VOID (moot) means
   eps_inf > m eps_nnbf(n*), the P2.1 projected NNBF's eps at the pilot's step count n*
   (parsed from the P2.1 logs; median over the seeds that reached n*). The absolute 1%
   (eps_best V / |E0| <= 0.01) is printed as a FLAG only.
 3 scr+ arm; S is offset-free: for route X and control C = scr+, the common start level is
   l0 = min(m eps_C(0), m eps_X(0)); each seed's clock starts at the first step reaching l0, so a
   route that starts above scr+ is counted from when it first matches scr+'s step-0 energy, and a
   route that only starts lower on scr+'s own curve (a static better start) gets S = 1.
 4 S is in WALL TIME with the source cost charged (source ED + all restarts + the whole anchor
   ladder). Target setup (shared by every arm) and checkpoint records are not charged. The
   step-count version (source cost converted to target-step equivalents) is reported alongside.
 5 No 451706 / NNBF-premise gate. A SIGN claim is made only if MFNG-O's own scr sign share at step
   --sign-share-step (150), m A_scr / (V m eps_scr), is >= SIGN_SHARE_MIN = 0.10. Steps-to-eps are
   reported separately for the sign part A and the amplitude part E - E0 - A (checkpoint
   resolution, offset-free as in 3).
 6 PA with anchor_ok = False in any seed -> INVALID. For the selected anchor rung the fraction of
   |theta - theta_init| inside the source-identified subspace (joint source SR metric, eigenvalues
   >= --id-tol * max) is recorded and printed.

Model and lattice
-----------------
t-t'-J on an Lx x Ly OPEN cluster: t on NN bonds, t' on the two diagonals, J (S.S - n n/4) on NN
bonds, no double occupancy (sign convention of r3_core.build_H; the 4x4 6+6 ground energy is
checked against P2.1's ED, -11.388010886295234, to 1e-8). Size-independent MFNG on OBC (MFNGO):
kappa_m(r) = (1/Q^2) sum_{k in Q x Q grid} cos(k.r) f_m(k) on a FIXED quadrature grid (--kq 8), so
kappa_m is the same function of r on every cluster; site features v_s = (h_s, sum_nn h~/4,
sum_nnn h~/4), h~ = 1 for a missing neighbour. Exact singlet. Sizes --channels/--field-hidden/
--bf-hidden/--jas-hidden (defaults 4/16/16/16: 426 parameters).
Symmetry: the target is trained and scored in the ED irrep of {1, Mx, My, MxMy} x {1, D} (|G| = 8;
isometry U, H_G = U^T H U) by projected exact-sum SR (r5_common.optimise_ckpt, one continuous
trajectory). Sources are trained unprojected.

Arms and Slurm roles (--task-id 0..11: arm = ARMS[task // 3], seed = task % 3)
------------------------------------------------------------------------------
  scr  (0-2)   target MF init (rng 2000 + seed)
  scr+ (3-5)   same init; first --scrp-k steps on kernel + Jastrow only (arm name "scrp")
  cp   (6-8)   sources --src-a, anchor 0, best of --src-restarts by source loss; copy
  pa   (9-11)  sources --src-a, anchor by the --anchors ladder rule; copy            [R5.1]
Every arm records eps(n) = (E_n - E0)/V, the SR clock (wall s, records excluded), full records
and the fixed-sign excess A at 0 and every checkpoint, and (cp, pa) the charged source wall time
and the identified-subspace fraction. JSON is rewritten every 10 steps.

Decision (pre-committed; --aggregate DIR --capacity-dir DIR). m(.) = median over 3 seeds.
--------------------------------------------------------------------------------------------
  x = eps_inf (capacity pilot).  C = scr+.  b = --b-step (50).
  COUNTED B in --ckpts-rule (20,50,100,150): m eps_C(B) - x >= 1e-3 t/site and
           m [eps_C(B-5) - eps_C(B)] >= 0.02 (m eps_C(B) - x) (plateau check).
  S_X(B)   offset-free wall-time speed-up (fix 3, 4): level l_B = m eps_C(B);
           S = m[T_C(l_B) - T_C(l0)] / m[T_X(l_B) - T_X(l0) + t_src,X]; S = 0 if X misses l_B
           in 2 or more seeds.
  PA INVALID  if anchor_ok = False in any pa seed.
  PA FAIL     S_pa(B) < 1.25 at every counted B, or m eps_pa(b) - x >= 0.9 (m eps_C(b) - x).
  PA PASS     S_pa(B) >= 2 at >= 2 counted B, m eps_pa(b) - x <= 0.5 (m eps_C(b) - x), and
              m eps_pa(b) < m eps_cp(b) (ablation); printed as VOID (moot) instead if
              eps_inf > m eps_nnbf(n*) or that NNBF value is unavailable.
  AMBIGUOUS   otherwise (also when no B is counted).
  SIGN claim  only if the scr sign share at 150 >= 0.10: PA sign speed-up S^A >= 2 at >= 2 counted B.
  Hard Core: in-irrep gap per site; competitor flag below COMP_TOL = 1e-3 t/site, else "sign half
  only".

Capacity read-out (--capacity DIR, pre-committed). n* = last step of the pilot, w = --plateau-window
(100). eps_inf = geometric extrapolation eps(n*) - d2 r / (1 - r), with d1 = eps(n*-2w) - eps(n*-w),
d2 = eps(n*-w) - eps(n*), r = d2 / d1, used only if 0 < r < 1 (else eps_inf = eps(n*)); clipped at
0. Rule: eps_inf V / |E0| > --capacity-tol (0.03) -> MUTATE (do not run the array); else RUN ARRAY.

Command-line options
--------------------
  --task-id INT     Slurm index 0..11 (sets --arm, --seed). Default -1 (use --arm/--seed).
  --arm STR         scr | scrp | cp | pa.        --seed INT  0, 1, 2 (default 0).
  --tgt STR         Target "LxXLy:nup" OBC (default "4x4:6").
  --tp, --J, --t    Couplings (-0.25, 0.4, 1.0).
  --src-a STR       Sources of cp/pa, comma list of "LxXLy:nup" OBC clusters (default "4x2:3,2x4:3").
  --anchors STR     L2-SP ladder (default "1e-2,1e-3,1e-4"). Rule: the LARGEST anchor whose source
                    fit has max relative error <= --src-tol (1e-2) on every source; if none, the
                    smallest, flagged anchor_ok = False (INVALID in the aggregate).
  --src-restarts INT (4)  --steps-src INT (800): source restarts and steps.
  --id-tol FLOAT    Identified-subspace eigenvalue threshold (default 1e-6).
  --scrp-k INT      Charged kernel + Jastrow steps of scr+ (default 20).
  --steps INT       Target steps per arm (default 150; the capacity pilot uses 600).
  --ckpts STR       Record checkpoints (default "10,20,50,100,150").
  --sign-at STR     Checkpoints with the sign budget A: "all" (default) or a comma list.
  --ckpts-rule STR  B values of the rule (default "20,50,100,150").
  --b-step INT      Matched-budget step of the ratio and ablation criteria (default 50).
  --sign-share-step INT  Step of the scr sign share (default 150).
  --channels, --field-hidden, --bf-hidden, --jas-hidden   MFNG sizes (4, 16, 16, 16).
  --kq INT          Quadrature grid (8).   --delta0, --eta  MF-init d-wave amplitude, regulariser.
  --lr, --diag-shift  SR step and relative diagonal shift (0.05, 0.01).
  --fn-iters, --fn-floor, --lbfgs-iters  fixed-sign budget settings (6, 1e-3, 3000).
  --r-min FLOAT     Pd minimal bond distance (2.0; diagnostic).
  --chunk INT       Configurations per chunk (50000).
  --wall-hours FLOAT  Stop target steps after this many hours (default 22).
  --nnbf-dir PATH   P2.1 test folder; its *.out logs give the NNBF eps trajectory (moot condition).
  --out PATH        Output JSON (default results/{arm}_s{seed}.json).
  --aggregate DIR   Decision from every JSON in DIR (no computation); needs --capacity-dir.
  --capacity-dir DIR  Capacity-pilot folder (default results_capacity).
  --capacity DIR    Capacity read-out and pre-committed MUTATE / RUN ARRAY decision.
  --plateau-window INT, --capacity-tol FLOAT  capacity rule constants (100, 0.03).
  --comp-tol, --plateau-frac  rule constants (1e-3, 0.02).
  --selftest        OBC lattice vs P2.1 conventions, [H, P_g] = 0 and U^T U = I, MFNGO
                    log-derivatives vs finite differences, kernel identical on two sizes,
                    anchored-wrapper identity. Exit 2 on failure.
  --logiccheck      Decision logic on synthetic trajectories (static better start -> not PASS;
                    VOID gates PASS only; FAIL prints under VOID; anchor_ok False -> INVALID;
                    source cost charged; sign gate). Exit 2 on failure.
  --timing          One gradient and value pass on --chunk target configurations, extrapolated.

Examples
--------
  python3 obc_transfer.py --selftest ; python3 obc_transfer.py --logiccheck
  python3 obc_transfer.py --arm scr --seed 0 --steps 600 --ckpts 10,20,50,100,150,300,450,600 \
      --wall-hours 28 --out results_capacity/scr_s0.json          # capacity pilot (Slurm)
  python3 obc_transfer.py --capacity results_capacity
  python3 obc_transfer.py --task-id 7                              # one array task
  python3 obc_transfer.py --aggregate results --capacity-dir results_capacity
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r3_core as rc  # noqa: E402
from r5_common import identified_fraction, med, optimise_ckpt, steps_to  # noqa: E402

ARMS = ("scr", "scrp", "cp", "pa")
LABEL = {"scr": "scr", "scrp": "scr+", "cp": "cp", "pa": "pa"}
P21_E0 = -11.388010886295234        # P2.1 instance 0 ED ground energy (tests/P2.1_subcubic_amplitudes_r3)
P21_REF = {"3x2:2": -4.062964411903416, "4x3:3": -9.16758855}   # tj_core.TJ OBC ground energies
PREM_EPS = 1e-3
SIGN_SHARE_MIN = 0.10


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


def kernel_jastrow_mask(net):
    """scr+ cheap phase: kernel (field network fW1, fb1, fW2, fb2) and Jastrow parameters; the
    backflow (bW, bb, ba) is frozen."""
    m = np.ones(net.npar, bool)
    for nm in ("bW", "bb", "ba"):
        m[net.sl[nm]] = False
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


def train_sources(a, specs, anchor, seed, tag):
    """Best of --src-restarts source fits (by source loss). Wall time (ED + all restarts) and the
    identified-subspace fraction of the selected fit are returned in info."""
    t0 = time.time()
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
            best = (L, th_init + d, th_init, Es, cls, d)
    _, th, th_init, Es, cls, d = best
    wall = time.time() - t0
    idf = identified_fraction(cls, d, a.chunk, a.id_tol)
    rel = [float((e - s["E0"]) / abs(s["E0"])) for e, s in zip(Es, src)]
    rc.log(f"source {tag}: relerr={['%.2e' % x for x in rel]} |theta-theta_init|={np.linalg.norm(th - th_init):.3f} "
           f"identified {idf['n_identified']}/{idf['npar']} frac_identified={idf['frac_identified']:.3f} wall={wall:.1f}s")
    return th, th_init, src, dict(specs=specs, anchor=anchor, E=[float(e) for e in Es], relerr=rel,
                                  restart_losses=losses, dist_init=float(np.linalg.norm(th - th_init)),
                                  identified=idf, wall_s=wall)


def anchored_sources(a, specs, tag):
    """Pre-committed anchor rule: the LARGEST anchor of --anchors (descending) whose source fit has
    max relative error <= --src-tol on every source cluster; if none qualifies, the smallest is
    used and info['anchor_ok'] = False. Every rung tried is charged (wall_s_charged)."""
    ladder = sorted((float(x) for x in a.anchors.split(",")), reverse=True)
    tried, charged = [], 0.0
    for lam in ladder:
        res = train_sources(a, specs, lam, a.seed, f"{tag} anchor {lam:g}")
        charged += res[3]["wall_s"]
        tried.append(dict(anchor=lam, relerr=res[3]["relerr"], identified=res[3]["identified"],
                          wall_s=res[3]["wall_s"]))
        if max(res[3]["relerr"]) <= a.src_tol:
            break
    res[3].update(ladder=tried, anchor_ok=bool(max(res[3]["relerr"]) <= a.src_tol), wall_s_charged=charged)
    return res


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
    mask = mask_steps = None
    if a.arm in ("scr", "scrp"):
        th = th_tgt_init
        R["source"] = dict(wall_s_charged=0.0, anchor_ok=None)
        if a.arm == "scrp":
            mask, mask_steps = kernel_jastrow_mask(net), a.scrp_k
            R["source"]["scrp_k"] = a.scrp_k
            R["source"]["n_cheap_params"] = int(mask.sum())
    elif a.arm == "cp":
        th, _, _, info = train_sources(a, a.src_a.split(","), 0.0, a.seed, "cp")
        info.update(anchor_ok=None, wall_s_charged=info["wall_s"])
        R["source"] = info
    else:
        th, _, _, info = anchored_sources(a, a.src_a.split(","), "pa")
        R["source"] = info
    rc.log(f"arm {LABEL[a.arm]}: charged source wall time {R['source']['wall_s_charged']:.1f}s")
    dump()
    ck = sorted({int(x) for x in a.ckpts.split(",")})
    sign_at = set([0] + ck) if a.sign_at == "all" else {0} | {int(x) for x in a.sign_at.split(",")}
    clock = []
    R["clock"] = clock

    def on_step(n, h):
        R["hist"] = h
        if n % 10 == 0:
            dump()

    cl = rc.Cluster(net, T["basis"], T["U"], T["HG"])
    _, h = optimise_ckpt(cl, th, a.steps, a.lr, a.diag_shift, a.chunk, LABEL[a.arm], T["E0"], T["V"],
                         ckpts=[0] + ck,
                         on_ckpt=lambda n, t_: record(f"{a.arm}@{n}", net, t_, sign=(n in sign_at)),
                         deadline=t0 + a.wall_hours * 3600, mask=mask, mask_steps=mask_steps,
                         clock=clock, on_step=on_step)
    R["hist"] = h
    R["done"] = True
    R["wall_s"] = time.time() - t0
    dump()
    rc.log(f"done ({out})")


# ============================================================================ decision helpers
def medi(x):
    """Median keeping +inf (missing levels), dropping nan / None."""
    x = [v for v in x if v is not None and not np.isnan(v)]
    return float(np.median(x)) if x else float("nan")


def tgt_V(T):
    Lx, Ly = rc.parse_lat(T["spec"].split(":")[0])
    return Lx * Ly


def first_reach(series, level):
    for n, v in series:
        if v <= level:
            return n
    return None


def offset_free(X, C, B):
    """Offset-free speed-up of X over control C at budget B (fix 3, 4). X, C: per-seed dicts with
    series [(step, value)] sorted by step, clock (list, s) and src (charged source s). Common start
    level l0 = min(m value_C(0), m value_X(0)); level l_B = m value_C(B). Each seed's span runs from
    its first step at or below l0 to its first step at or below l_B; X's span is charged src (wall)
    or src / (its mean target-step time) (steps). S = m span_C / m span_X; 0 if X misses in >= 2 seeds."""
    v = lambda r, n: dict(r["series"]).get(n, float("nan"))
    lB = med([v(c, B) for c in C])
    l0 = min(med([v(c, 0) for c in C]), med([v(x, 0) for x in X]))
    res = dict(lB=lB, l0=l0, S_wall=float("nan"), S_step=float("nan"))
    if not lB < l0:
        return res

    def spans(rr):
        out = []
        for r in rr:
            n0, n1 = first_reach(r["series"], l0), first_reach(r["series"], lB)
            if n0 is None or n1 is None:
                out.append((float("inf"), float("inf")))
                continue
            step_s = r["clock"][-1] / max(1, len(r["clock"]) - 1)
            out.append((r["clock"][n1] - r["clock"][n0] + r["src"], n1 - n0 + r["src"] / max(1e-300, step_s)))
        return out

    sc, sx = spans(C), spans(X)
    for i, k in enumerate(("S_wall", "S_step")):
        if sum(not np.isfinite(s[i]) for s in sx) >= 2:
            res[k] = 0.0
            continue
        num, den = medi([s[i] for s in sc]), medi([s[i] for s in sx])
        res[k] = num / den if den > 0 else (float("inf") if num > 0 else float("nan"))
    return res


def per_seed(rs, k, kind, V):
    """kind 'E': every step of eps; 'A' / 'amp': checkpoint records of the sign part A / V and the
    amplitude part (E - E0 - A) / V (per site, so levels compare with eps)."""
    out = []
    for r in sorted(rs[k], key=lambda r: r["seed"]):
        if kind == "E":
            ser = list(enumerate(r["hist"]))
        else:
            ser = []
            for nm, rec in r["arms"].items():
                if "A" in rec and np.isfinite(rec.get("eps", float("nan"))):
                    n = int(nm.split("@")[1])
                    ser.append((n, rec["A"] / V if kind == "A" else rec["eps"] - rec["A"] / V))
            ser.sort()
        out.append(dict(series=ser, clock=r["clock"], src=float((r.get("source") or {}).get("wall_s_charged") or 0.0)))
    return out


def nnbf_eps_at(d, n, E0, V):
    """P2.1 projected NNBF (instance 0) eps at step n per seed, from the *.out logs ("nnbf: it N
    E = ..." every 50 steps; the last logged step <= n). None if that seed has not reached n."""
    traj = {}
    for f in sorted(glob.glob(os.path.join(d, "*.out"))):
        seed = None
        for line in open(f, errors="replace"):
            m = re.search(r"instance (\d+) \{.*\} D=\d+ arm=(\w+) seed=(\d+)", line)
            if m:
                seed = int(m.group(3)) if (m.group(1) == "0" and m.group(2) == "nnbf") else None
                if seed is not None:
                    traj[seed] = []
                continue
            m = re.search(r"nnbf: it\s+(\d+)\s+E = (-?[0-9.]+)", line)
            if m and seed is not None:
                traj[seed].append((int(m.group(1)), float(m.group(2))))
    out = {}
    for s, pts in traj.items():
        ok = [E for it, E in pts if it <= n]
        out[s] = (ok[-1] - E0) / V if (pts and max(it for it, _ in pts) >= n and ok) else None
    return out


def capacity_estimate(h, w):
    n = len(h) - 1
    e = h[n]
    info = dict(n_star=n, eps_last=e, window=w)
    if n >= 2 * w:
        d1, d2 = h[n - 2 * w] - h[n - w], h[n - w] - h[n]
        r = d2 / d1 if d1 > 0 else float("nan")
        info.update(d1=d1, d2=d2, r=r, drop_last_window_frac=d2 / max(1e-300, e))
        if 0 < r < 1:
            info.update(eps_inf=max(0.0, e - d2 * r / (1 - r)), how="geometric extrapolation")
        else:
            info.update(eps_inf=e, how="last value (r not in (0,1))")
    else:
        info.update(eps_inf=e, how="last value (fewer than 2w steps)")
    return info


def load_capacity(d, a):
    rr = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        if r.get("arm") == "scr" and r.get("hist"):
            rr.append(r)
    if not rr:
        return None
    T = rr[0]["target"]
    V = tgt_V(T)
    ests = [capacity_estimate(r["hist"], a.plateau_window) for r in rr]
    einf = med([e["eps_inf"] for e in ests])
    shares = []
    for r in rr:
        rec = r["arms"].get(f"scr@{a.sign_share_step}", {})
        if "A" in rec and rec.get("eps", 0) > 0:
            shares.append(rec["A"] / (V * rec["eps"]))
    return dict(T=T, V=V, eps_inf=einf, relerr_inf=einf * V / abs(T["E0"]), n_star=min(e["n_star"] for e in ests),
                ests=ests, done=[bool(r.get("done")) for r in rr], sign_share=med(shares), seeds=[r["seed"] for r in rr])


def capacity(a):
    cap = load_capacity(a.capacity, a)
    if cap is None:
        print(f"CAPACITY: no scr JSON with a trajectory in {a.capacity}")
        return None
    T, V = cap["T"], cap["V"]
    print(f"target {T['spec']} OBC t'={T['tp']} J={T['J']} E0={T['E0']:.10f} n_irrep={T['n_irrep']} npar={T['npar']}")
    for s, e, dn in zip(cap["seeds"], cap["ests"], cap["done"]):
        print(f"  scr seed {s}: n*={e['n_star']} done={dn} eps(n*)={e['eps_last']:.5e} eps_inf={e['eps_inf']:.5e} ({e['how']}"
              + (f", r={e['r']:.3f}, drop over last {e['window']} steps = {e['drop_last_window_frac']:.3f} of eps(n*)" if "r" in e else "") + ")")
    nn = nnbf_eps_at(a.nnbf_dir, cap["n_star"], T["E0"], V)
    m_nn = med([v for v in nn.values() if v is not None])
    print(f"  eps_inf={cap['eps_inf']:.5e}  relerr_inf={cap['relerr_inf']:.4f}  (capacity rule threshold {a.capacity_tol})")
    print(f"  scr sign share at step {a.sign_share_step}: {cap['sign_share']:.4f} (sign claim allowed only if >= {SIGN_SHARE_MIN})")
    print(f"  P2.1 projected NNBF eps at n*={cap['n_star']}: {nn} median {m_nn:.5e} -> "
          + ("moot (VOID would gate PASS)" if not np.isfinite(m_nn) or cap["eps_inf"] > m_nn else "not moot"))
    print(f"  FLAG abs 1%: relerr_inf {'<=' if cap['relerr_inf'] <= 0.01 else '>'} 0.01 (flag only)")
    if cap["relerr_inf"] > a.capacity_tol:
        print(f"CAPACITY DECISION: MUTATE (relerr_inf {cap['relerr_inf']:.4f} > {a.capacity_tol}); do NOT run the array; "
              "enlarge the size-independent family (--channels, --bf-hidden) and re-pilot")
    else:
        print(f"CAPACITY DECISION: RUN ARRAY (relerr_inf {cap['relerr_inf']:.4f} <= {a.capacity_tol})")
    return cap


def decide(rs, cap, m_nn, a, say=print):
    T = rs["scr"][0]["target"]
    V, E0 = tgt_V(T), T["E0"]
    x = cap["eps_inf"]
    H = {k: [r["hist"] for r in sorted(rs[k], key=lambda r: r["seed"])] for k in ARMS}
    at = lambda hs, n: [h[n] if n < len(h) else float("nan") for h in hs]
    eps_best = min(min(h) for k in H for h in H[k])
    flag1 = eps_best * V / abs(E0) <= 0.01
    moot = (m_nn is None) or (not np.isfinite(m_nn)) or x > m_nn
    say(f"target {T['spec']} OBC t'={T['tp']} J={T['J']} irrep={T['irrep']} E0={E0:.10f} n_irrep={T['n_irrep']} "
        f"npar={T['npar']} gap/V={T['gap_per_site']:.4e} -> {'COMPETITOR' if T['competitor'] else 'sign half only'}")
    say(f"  capacity: x = eps_inf = {x:.4e} (relerr {cap['relerr_inf']:.4f}, n* = {cap['n_star']}); "
        f"NNBF eps at n*: {m_nn:.4e} -> {'MOOT: VOID gates PASS' if moot else 'not moot'}")
    say(f"  FLAG abs 1%: eps_best={eps_best:.4e} relerr={eps_best * V / abs(E0):.3e} "
        f"({'within' if flag1 else 'NOT within'} 1% of E0; flag only)")
    E = {k: per_seed(rs, k, "E", V) for k in ARMS}
    src = {k: med([e["src"] for e in E[k]]) for k in ARMS}
    say(f"  charged source wall s (median): {src}")
    rb = sorted({int(v) for v in a.ckpts_rule.split(",")})
    rows, counted = {}, []
    for B in rb:
        mc = med(at(H["scrp"], B))
        mov = med([h[B - 5] - h[B] if B < len(h) and B >= 5 else float("nan") for h in H["scrp"]])
        c = bool((mc - x >= PREM_EPS) and (mov >= a.plateau_frac * (mc - x)))
        row = dict(m_scrp=mc, counted=c)
        for k in ("scr", "cp", "pa"):
            s = offset_free(E[k], E["scrp"], B)
            row[f"S_{k}"], row[f"Sstep_{k}"] = s["S_wall"], s["S_step"]
        rows[B] = row
        if c:
            counted.append(B)
        say(f"  B={B:4d}: " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    mb = {k: med(at(H[k], a.b_step)) for k in ARMS}
    say(f"  m eps({a.b_step})={ {LABEL[k]: round(v, 5) for k, v in mb.items()} }")
    srcs = [r["source"] for r in sorted(rs["pa"], key=lambda r: r["seed"])]
    aok = [s.get("anchor_ok") for s in srcs]
    say(f"  pa anchors {[s.get('anchor') for s in srcs]} anchor_ok {aok} frac of |theta - theta_init| in the "
        f"source-identified subspace {[round(s.get('identified', {}).get('frac_identified', float('nan')), 4) for s in srcs]}")
    fin = [B for B in counted if np.isfinite(rows[B]["S_pa"])]
    speed = sum(rows[B]["S_pa"] >= 2 for B in counted) >= 2
    b_ok = mb["pa"] - x <= 0.5 * (mb["scrp"] - x)
    b_fail = mb["pa"] - x >= 0.9 * (mb["scrp"] - x)
    slow = len(fin) >= 1 and all(rows[B]["S_pa"] < 1.25 for B in fin)
    P = dict(counted=len(counted) >= 1, speed=speed, b=b_ok, ablation=mb["pa"] < mb["cp"])
    if not all(v is True for v in aok):
        v = f"INVALID (anchor_ok={aok})"
    elif slow or b_fail:
        v = "FAIL (" + ", ".join(n for n, f in (("slow", slow), ("b-fail", b_fail)) if f) + ")"
    elif all(P.values()):
        v = ("VOID (PASS conditions met, but moot: eps_inf above the NNBF at the same step count)" if moot else "PASS")
    else:
        v = "AMBIGUOUS (" + ", ".join(k for k, f in P.items() if not f) + " not met)"
    say(f"GLOBAL R5.1 PA: {v}  {P}")
    # sign part (fix 5)
    ss = a.sign_share_step
    sh = med([r["arms"][f"scr@{ss}"]["A"] / (V * r["arms"][f"scr@{ss}"]["eps"]) for r in rs["scr"]
              if "A" in r["arms"].get(f"scr@{ss}", {}) and r["arms"][f"scr@{ss}"]["eps"] > 0])
    parts = {}
    for kind in ("A", "amp"):
        Xs, Cs = per_seed(rs, "pa", kind, V), per_seed(rs, "scrp", kind, V)
        parts[kind] = {B: offset_free(Xs, Cs, B)["S_step"] for B in counted}
    say(f"  steps-to-eps by part (pa vs scr+, offset-free, checkpoint resolution, source charged in steps): "
        f"sign A {parts['A']}  amplitude E-E0-A {parts['amp']}")
    if np.isfinite(sh) and sh >= SIGN_SHARE_MIN:
        sv = "PASS" if sum(s >= 2 for s in parts["A"].values()) >= 2 else "NOT MET"
        sign = f"SIGN CLAIM (scr sign share at {ss} = {sh:.3f} >= {SIGN_SHARE_MIN}): {sv}"
    else:
        sign = f"SIGN CLAIM NOT MADE (scr sign share at {ss} = {sh:.3f} < {SIGN_SHARE_MIN} or missing)"
    say(sign)
    return dict(pa=v, sign=sign, rows=rows, counted=counted, moot=moot, flag1=flag1)


def aggregate(a):
    rs = {}
    for f in sorted(glob.glob(os.path.join(a.aggregate, "*.json"))):
        r = json.load(open(f))
        if r.get("done"):
            rs.setdefault(r["arm"], []).append(r)
    miss = [LABEL[k] for k in ARMS if len(rs.get(k, [])) < 3]
    if miss:
        print(f"INCOMPLETE: arms with < 3 seeds: {miss}")
        return
    cap = load_capacity(a.capacity_dir, a)
    if cap is None:
        print(f"NO CAPACITY PILOT in {a.capacity_dir}: x = eps_inf is undefined, no decision")
        return
    T = rs["scr"][0]["target"]
    nn = nnbf_eps_at(a.nnbf_dir, cap["n_star"], T["E0"], tgt_V(T))
    m_nn = med([v for v in nn.values() if v is not None])
    print(f"  P2.1 NNBF eps at n*={cap['n_star']} per seed: {nn}")
    decide(rs, cap, m_nn, a)


# ============================================================================ logic check
def logiccheck(a):
    """Decision logic on synthetic trajectories (no physics)."""
    V, E0, x, n = 16, -10.0, 0.002, 150
    ck = (0, 10, 20, 50, 100, 150)
    ctrl = lambda m: x + 0.3 * np.exp(-m / 60.0)
    fast = lambda m: x + 0.3 * np.exp(-m / 20.0)
    T = dict(spec="4x4:6", tp=-0.25, J=0.4, irrep={}, E0=E0, n_irrep=1, npar=426, gap_per_site=0.05, competitor=False)

    def syn(arm, f, src=0.0, aok=True, share=0.3):
        out = []
        for s in range(3):
            h = [float(f(m)) for m in range(n + 1)]
            r = dict(arm=arm, seed=s, done=True, hist=h, clock=[100.0 * m for m in range(n + 1)], target=T,
                     arms={f"{arm}@{c}": dict(eps=h[c], A=share * V * h[c]) for c in ck},
                     source=dict(anchor_ok=(aok if s == 0 else True) if arm == "pa" else None, wall_s_charged=src,
                                 anchor=1e-3, identified=dict(frac_identified=0.5)))
            out.append(r)
        return out

    b = argparse.Namespace(**vars(a))
    b.ckpts_rule, b.b_step, b.sign_share_step, b.plateau_frac = "20,50,100,150", 50, 150, 0.02
    cap = dict(eps_inf=x, relerr_inf=x * V / abs(E0), n_star=600)
    quiet = lambda *_: None
    ok = True

    def case(name, pa, m_nn, want, share=0.3, cp=None):
        nonlocal ok
        rs = dict(scr=syn("scr", ctrl, share=share), scrp=syn("scrp", ctrl, share=share),
                  cp=cp or syn("cp", ctrl), pa=pa)
        d = decide(rs, cap, m_nn, b, say=quiet)
        good = d["pa"].startswith(want[0]) and (len(want) < 2 or d["sign"].startswith(want[1]))
        ok &= good
        rc.log(f"logiccheck {name}: PA -> {d['pa'].split('  ')[0]} | {d['sign'][:26]} (expect {want}) {'OK' if good else 'FAIL'}")

    old_S = 100 / steps_to([ctrl(m + 60) for m in range(n + 1)], ctrl(100))
    rc.log(f"logiccheck (context) old rule S = B / n_X(l_B) for a 60-step static head start at B=100: {old_S:.2f} (>= 2 would PASS)")
    case("static better start, no source cost", syn("pa", lambda m: ctrl(m + 60)), 1.0, ("FAIL",))
    case("3x faster descent, not moot", syn("pa", fast), 1.0, ("PASS",))
    case("3x faster descent, moot", syn("pa", fast), 1e-4, ("VOID",))
    case("static better start, moot (FAIL still prints)", syn("pa", lambda m: ctrl(m + 60)), 1e-4, ("FAIL",))
    case("3x faster but source costs 100 target steps", syn("pa", fast, src=1e4), 1.0, ("FAIL",))
    case("3x faster, anchor_ok False in one seed", syn("pa", fast, aok=False), 1.0, ("INVALID",))
    case("sign share 0.3: claim made", syn("pa", fast), 1.0, ("PASS", "SIGN CLAIM (scr"), share=0.3)
    case("sign share 0.05: claim not made", syn("pa", fast), 1.0, ("PASS", "SIGN CLAIM NOT MADE"), share=0.05)
    case("pa no better than cp at b (ablation)", syn("pa", fast), 1.0, ("AMBIGUOUS",), cp=syn("cp", fast))
    rc.log("LOGICCHECK " + ("PASSED" if ok else "FAILED"))
    return ok


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
    p.add_argument("--anchors", default="1e-2,1e-3,1e-4")
    p.add_argument("--src-tol", type=float, default=1e-2)
    p.add_argument("--src-restarts", type=int, default=4)
    p.add_argument("--steps-src", type=int, default=800)
    p.add_argument("--id-tol", type=float, default=1e-6)
    p.add_argument("--scrp-k", type=int, default=20)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--ckpts", default="10,20,50,100,150")
    p.add_argument("--sign-at", default="all")
    p.add_argument("--ckpts-rule", default="20,50,100,150")
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
    p.add_argument("--nnbf-dir", default=os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3"))
    p.add_argument("--out", default=None)
    p.add_argument("--aggregate", default="")
    p.add_argument("--capacity-dir", default=os.path.join(HERE, "results_capacity"))
    p.add_argument("--capacity", default="")
    p.add_argument("--plateau-window", type=int, default=100)
    p.add_argument("--capacity-tol", type=float, default=0.03)
    p.add_argument("--comp-tol", type=float, default=1e-3)
    p.add_argument("--plateau-frac", type=float, default=0.02)
    p.add_argument("--b-step", type=int, default=50)
    p.add_argument("--sign-share-step", type=int, default=150)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--logiccheck", action="store_true")
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
    if a.capacity:
        capacity(a)
        return
    if a.logiccheck:
        sys.exit(0 if logiccheck(a) else 2)
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
