#!/usr/bin/env python3
"""
P2.1 round-2 decisive test: does a LOW-RANK-HOP-RESPONSE determinant family lose
SIGN quality relative to full neural-network backflow (NNBF) at matched parameter count?
=======================================================================================

Physics question
----------------
P1.1 (Slurm job 329057) showed that on doped 4x3 PBC Hubbard clusters the sign
structure carries 61-77 % of the NNBF energy error and that network width fixes
amplitudes, not signs. The round-1 P2.1 node result is that an exact sub-cubic local
energy needs a family whose hop response A(n') - A(n) has low rank. This script asks
whether such a restriction costs SIGN quality (E_sign) and total energy against full
NNBF at the same parameter count, measured with exact sums over the full Hilbert space.

Families (all evaluated exactly on the full fixed-(N_up, N_dn) basis; Jordan-Wigner
order and Hamiltonian from p11_core.py, a verbatim copy of the P1.1 test):
  psi(n) = exp(J(n)) * sum_{k=1..K} w_k prod_{sigma} det B_{k,sigma}(n)
  * "full" mode (NNBF; Luo & Clark PRL 122, 226401):
        B_{k,s}(n) = (Phi0_{k,s} + reshape(out_{k,s}(n)))[occ_s]      (V x N_s output)
  * "lr" mode (low-rank hop response; hidden-fermion form of Liu & Clark 2311.09450):
        B_{k,s}(n) = Phi0_s[occ_s] + Phih_{k,s}[occ_s] C_{k,s}(n)      (C is r x N_s)
    so a hop changes one row plus a rank-r term: rank(A(n')-A(n)) <= r+1 per term.
  * out / C come from a one-hidden-layer tanh trunk of width Hs on x = 2n-1 (2V inputs);
    J(n) = w2a . tanh(W1a x + b1a) is a separate positive amplitude network of width Ha.
Arms (names used by --arm; widths are solved so that the parameter count matches
NNBF of width 128 on the same instance, within one hidden unit):
  nnbf128 : full, K=1, Hs=128, Ha=0                    (control: P1.1's widest NNBF)
  mdnnbf  : full, K=4 separate references, Ha=0, Hs free (control: unrestricted multi-det)
  lchfds  : lr, K=1, r, Ha=0, Hs free                  (Liu-Clark / HFDS arm, no amplitude net)
  schfds  : lr, K=1, r, Hs=--hs-sign, Ha free          (ROUTE 2.1: split-capacity HFDS)
  sklr    : lr, K=4 shared reference, r, Hs=--hs-sign, Ha free (ROUTE 2.2: shared-reference
            low-rank multi-determinant, table-method form)
  sjnet   : lr with r=0, K=1, Ha free                  (Gate-2 control: Slater x neural Jastrow)
All arms are trained by stochastic reconfiguration with the EXACT force (same settings as
P1.1: 1500 iterations, step 0.05, shift 1e-3, 2000-sample metric) from the same UHF
reference (UHF seed fixed at 1234), then budgeted with P1.1's budget(): E_sign (upper
estimate: min over 4 L-BFGS-B starts and the ten Haaf fixed-node state), dE_tot, dE_sign,
dE_amp and the d-wave pair-correlation error eC_tot.

Decision rule (pre-committed; applied by --aggregate)
-----------------------------------------------------
For every (instance, arm) the seed with the lowest variational energy is used.
Ref_sign = min(dE_sign(nnbf128), dE_sign(mdnnbf)); Ref_tot and Ref_C likewise with dE_tot
and eC_tot. For a route R in {schfds (2.1), sklr (2.2)} and each instance:
  rho_s = dE_sign(R)/Ref_sign, rho_t = dE_tot(R)/Ref_tot, rho_C = eC_tot(R)/Ref_C,
  gamma = dE_sign(R)/dE_sign(sjnet)  (does the low-rank channel do sign work?).
  PASS     on every instance: rho_s <= 1.15 and rho_t <= 1.30 and rho_C <= 1.30 and
           gamma <= 0.75; for route 2.1 additionally dE_tot(schfds) < dE_tot(lchfds)
           (the amplitude channel must repair the Liu-Clark deficit).
  FAIL     on any instance: rho_s >= 1.5 or gamma >= 0.9 or rho_C >= 2.0;
           for route 2.1 also if dE_tot(schfds) >= dE_tot(lchfds) on every instance.
  AMBIGUOUS otherwise. Pre-committed action: rerun the AMBIGUOUS route and both controls
           with two more seeds (--seeds 2 more); if still AMBIGUOUS, declare FAIL.
The cost side check (--mode cost) must also pass for a route to survive:
  eps <= 1e-8 (fast versus brute-force local energy) and fitted slope of the fast local
  energy time over the three largest sizes <= 2.4, with brute slope >= 3.3 (regime guard;
  if the guard fails the side check is AMBIGUOUS and is rerun with larger --sizes).

Usage
-----
  python3 micro_test.py --mode train --task ID [options]   one Slurm array task
  python3 micro_test.py --mode cost [--sizes 8,12,16,20,24] fast-vs-brute local energy
  python3 micro_test.py --mode aggregate                    verdicts from results/
  python3 micro_test.py --mode smoke                        login-node checks (< 30 s)

Options
-------
  --mode {train,cost,aggregate,smoke}  What to run (default smoke).
  --task INT         Array index 0..35 for --mode train; decodes to instance = task//12
                     (0,1,2 = P1.1 instances), arm = ARMS[(task%12)//2], seed index =
                     task%2 (seeds 1234, 4321). Overridden by explicit --instance/--arm/--seed.
  --instance INT     P1.1 instance 0, 1 or 2 (4x3 PBC, t'=-0.25; 0: 5+5 U=8; 1: 4+4 U=8;
                     2: 5+5 U=4), or 9 = 3x2 smoke cluster (not a Hard-Core instance).
  --arm NAME         One of nnbf128, mdnnbf, lchfds, schfds, sklr, sjnet.
  --seed INT         Network-initialisation and SR-sampling seed (default from --task).
  --r INT            Rank r of the low-rank channel per spin and term (default 2).
  --K INT            Number of terms for mdnnbf and sklr (default 4).
  --hs-sign INT      Sign-trunk width Hs of schfds and sklr (default 64).
  --match-width INT  NNBF width whose parameter count all arms match (default 128).
  --sr-iters INT     SR iterations (default 1500).  --sr-lr FLOAT (default 0.05).
  --sr-shift FLOAT   SR diagonal shift (default 1e-3). --sr-samples INT (default 2000).
  --chunk INT        Configurations per forward chunk (default 65536).
  --lbfgs-iters INT  L-BFGS-B iterations per start in E_sign (default 4000).
  --fn-iters INT     ten Haaf self-consistent iterations (default 10).
  --fn-floor FLOAT   Fixed-node guide floor (default 1e-3).
  --r-min FLOAT      Minimal bond distance in the pairing error (default 1.5).
  --sizes LIST       --mode cost: comma-separated L for L x L PBC lattices (default
                     8,12,16,20,24); brute force is run only for L <= --brute-max.
  --brute-max INT    Largest L with brute-force comparison (default 16).
  --nconf INT        --mode cost: configurations per size (default 2).
  --tp FLOAT         --mode cost: t' (default -0.25). --U FLOAT: U (default 8).
  --doping FLOAT     --mode cost: hole doping (default 0.125).
  --results DIR      Results directory (default results/ next to this script).
  --out PATH         Output JSON (default results/<tag>.json); not overwritten
                     unless --overwrite 1.
  --overwrite {0,1}  Allow overwriting (default 0).

Outputs
-------
  train: results/inst{I}_{arm}_s{seed}.json with params, arm spec, n_params, P_target,
         E, E0, ED levels, Hard-Core gap, the full P1.1 budget dict, timings.
  cost:  results/cost_side_check.json with per-size timings, eps, slopes, verdict.
  aggregate: printed table, per-route verdicts, results/aggregate.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p11_core import (INSTANCES, Hubbard, Ansatz, budget, build_lattice, energy,  # noqa: E402
                      solve_ed, uhf_orbitals)

ARMS = ["nnbf128", "mdnnbf", "lchfds", "schfds", "sklr", "sjnet"]
SEEDS = [1234, 4321]
UHF_SEED = 1234
SMOKE_INST = dict(Lx=3, Ly=2, bcx="pbc", bcy="obc", nup=2, ndn=2, U=4.0, tp=-0.25)
# pre-committed thresholds
RHO_S_PASS, RHO_T_PASS, RHO_C_PASS, GAMMA_PASS = 1.15, 1.30, 1.30, 0.75
RHO_S_FAIL, GAMMA_FAIL, RHO_C_FAIL = 1.5, 0.9, 2.0
EPS_SIDE, SLOPE_SIDE, BRUTE_GUARD = 1e-8, 2.4, 3.3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ ansatz
class LRAnsatz:
    """psi = exp(J) * sum_k w_k prod_s det B_{k,s}; see module docstring."""

    def __init__(self, model, Pu, Pd, mode, K=1, r=0, Hs=0, Ha=0, shared_ref=True, seed=0,
                 w2_scale=1e-3):
        rng = np.random.default_rng(seed)
        self.m = model
        V, nu, nd = model.V, model.nup, model.ndn
        self.V, self.nu, self.nd = V, nu, nd
        self.mode, self.K, self.r = mode, K, (r if mode == "lr" else 0)
        self.Kref = 1 if (shared_ref or mode == "lr") else K
        p = {}
        p["P0u"] = np.repeat(Pu[None], self.Kref, 0).copy()
        p["P0d"] = np.repeat(Pd[None], self.Kref, 0).copy()
        if self.Kref > 1:
            p["P0u"][1:] += 0.01 * rng.standard_normal(p["P0u"][1:].shape)
            p["P0d"][1:] += 0.01 * rng.standard_normal(p["P0d"][1:].shape)
        if self.r > 0:
            p["Phu"] = rng.standard_normal((K, V, self.r)) / np.sqrt(V)
            p["Phd"] = rng.standard_normal((K, V, self.r)) / np.sqrt(V)
        self.per = self.r * (nu + nd) if mode == "lr" else V * (nu + nd)
        self.outdim = K * self.per
        self.trunk = Hs > 0 and self.outdim > 0
        self.Hs, self.Ha = (Hs if self.trunk else 0), Ha
        if self.trunk:
            p["W1"] = rng.standard_normal((2 * V, Hs)) / np.sqrt(2 * V)
            p["b1"] = np.zeros(Hs)
            p["W2"] = w2_scale * rng.standard_normal((Hs, self.outdim))
            p["b2"] = np.zeros(self.outdim)
        if Ha > 0:
            p["W1a"] = rng.standard_normal((2 * V, Ha)) / np.sqrt(2 * V)
            p["b1a"] = np.zeros(Ha)
            p["w2a"] = np.zeros(Ha)
        if K > 1:
            p["wk"] = np.ones(K)
        self.p = p
        self.keys = list(p.keys())

    # -- parameter plumbing (same interface as p11_core.Ansatz)
    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def flat(self):
        return np.concatenate([self.p[k].ravel() for k in self.keys])

    def set_flat(self, x):
        o = 0
        for k in self.keys:
            n = self.p[k].size
            self.p[k] = x[o:o + n].reshape(self.p[k].shape)
            o += n

    def _wk(self):
        return self.p["wk"] if self.K > 1 else np.ones(1)

    # -- forward on explicit occupations
    def forward(self, ou, od, ru, rd):
        """ou, od: (B, V) 0/1 occupations; ru, rd: (B, N_s) sorted occupied sites."""
        p, K, V, nu, nd = self.p, self.K, self.V, self.nu, self.nd
        B = ou.shape[0]
        x = np.concatenate([ou, od], 1).astype(float) * 2 - 1
        c = dict(x=x, ru=ru, rd=rd, B=B)
        if self.trunk:
            h = np.tanh(x @ p["W1"] + p["b1"])
            out = (h @ p["W2"] + p["b2"]).reshape(B, K, self.per)
            c["h"] = h
        Bm = {}
        for s, rows, N, off in (("u", ru, nu, 0), ("d", rd, nd, None)):
            P0 = p["P0" + s]                                  # (Kref, V, N)
            P0r = np.transpose(P0[:, rows], (1, 0, 2, 3))     # (B, Kref, N, N)
            if self.mode == "lr":
                if self.r > 0:
                    o0 = 0 if s == "u" else self.r * nu
                    C = out[:, :, o0:o0 + self.r * N].reshape(B, K, self.r, N)
                    Ph = np.transpose(p["Ph" + s][:, rows], (1, 0, 2, 3))  # (B, K, N, r)
                    M = P0r + Ph @ C
                    c["C" + s], c["Ph" + s] = C, Ph
                else:
                    M = np.broadcast_to(P0r, (B, K, N, N)).copy()
            else:
                o0 = 0 if s == "u" else V * nu
                Of = out[:, :, o0:o0 + V * N].reshape(B, K, V, N) + P0[None]
                M = np.take_along_axis(Of, rows[:, None, :, None], axis=2)
            Bm[s] = M
        c["Bm"] = Bm
        du = np.linalg.det(Bm["u"]) if nu > 0 else np.ones((B, K))
        dd = np.linalg.det(Bm["d"]) if nd > 0 else np.ones((B, K))
        c["du"], c["dd"], c["dk"] = du, dd, du * dd
        S = c["dk"] @ self._wk()
        if self.Ha > 0:
            ha = np.tanh(x @ p["W1a"] + p["b1a"])
            J = ha @ p["w2a"]
            c["ha"] = ha
        else:
            J = np.zeros(B)
        c["J"], c["S"] = J, S
        c["psi"] = np.exp(J) * S
        return c

    @staticmethod
    def _invT(M):
        try:
            return np.swapaxes(np.linalg.inv(M), -1, -2)
        except np.linalg.LinAlgError:
            return np.swapaxes(np.linalg.inv(M + 1e-12 * np.eye(M.shape[-1])), -1, -2)

    def grad(self, c, coef, per_sample):
        """Gradient of sum_n coef(n) psi(n) (per_sample=False) or the per-sample
        coef(n) dpsi(n)/dtheta (per_sample=True, shape (B, P))."""
        p, K, V, B = self.p, self.K, self.V, c["B"]
        eJ = np.exp(c["J"])
        wk = self._wk()
        tk = (coef * eJ)[:, None] * wk[None] * c["dk"]           # (B, K)
        bi = np.arange(B)[:, None]
        red = (lambda a: a) if per_sample else (lambda a: a.sum(0))
        g = {}
        dout = np.zeros((B, K, self.per)) if self.trunk else None
        for s, rows, N in (("u", c["ru"], self.nu), ("d", c["rd"], self.nd)):
            if N == 0:
                continue
            M = tk[:, :, None, None] * self._invT(c["Bm"][s])     # (B, K, N, N)
            Msum = M.sum(1, keepdims=True) if self.Kref == 1 else M
            Z = np.zeros((B, self.Kref, V, N))
            for kk in range(self.Kref):
                Z[bi, kk, rows] = Msum[:, kk]
            g["P0" + s] = red(Z)
            if self.mode == "lr" and self.r > 0:
                C, Ph = c["C" + s], c["Ph" + s]
                Zh = np.zeros((B, K, V, self.r))
                gPh = M @ np.swapaxes(C, -1, -2)                     # (B, K, N, r)
                for kk in range(K):
                    Zh[bi, kk, rows] = gPh[:, kk]
                g["Ph" + s] = red(Zh)
                dC = np.swapaxes(Ph, -1, -2) @ M                     # (B, K, r, N)
                o0 = 0 if s == "u" else self.r * self.nu
                dout[:, :, o0:o0 + self.r * N] = dC.reshape(B, K, -1)
            elif self.mode == "full":
                Zo = np.zeros((B, K, V, N))
                for kk in range(K):
                    Zo[bi, kk, rows] = M[:, kk]
                o0 = 0 if s == "u" else V * self.nu
                dout[:, :, o0:o0 + V * N] = Zo.reshape(B, K, -1)
        x = c["x"]
        if self.trunk:
            do = dout.reshape(B, -1)
            h = c["h"]
            dpre = (do @ p["W2"].T) * (1 - h ** 2)
            if per_sample:
                g["W2"] = (h[:, :, None] * do[:, None, :]).reshape(B, -1)
                g["W1"] = (x[:, :, None] * dpre[:, None, :]).reshape(B, -1)
            else:
                g["W2"], g["W1"] = h.T @ do, x.T @ dpre
            g["b2"], g["b1"] = red(do), red(dpre)
        if self.Ha > 0:
            cp = coef * c["psi"]
            ha = c["ha"]
            dprea = cp[:, None] * p["w2a"][None] * (1 - ha ** 2)
            if per_sample:
                g["W1a"] = (x[:, :, None] * dprea[:, None, :]).reshape(B, -1)
            else:
                g["W1a"] = x.T @ dprea
            g["b1a"] = red(dprea)
            g["w2a"] = red(cp[:, None] * ha)
        if K > 1:
            g["wk"] = red((coef * eJ)[:, None] * c["dk"])
        if per_sample:
            return np.concatenate([g[k].reshape(B, -1) for k in self.keys], 1)
        return np.concatenate([g[k].ravel() for k in self.keys])

    # -- full-basis interface used by SR
    def _occ(self, idx):
        m = self.m
        iu, idd = m.iu_id()
        iu, idd = iu[idx], idd[idx]
        return m.occ_u[iu], m.occ_d[idd], m.rows_u[iu], m.rows_d[idd]

    def amplitudes(self, chunk):
        D = self.m.D
        out = np.empty(D)
        for s in range(0, D, chunk):
            idx = np.arange(s, min(s + chunk, D))
            out[idx] = self.forward(*self._occ(idx))["psi"]
        return out

    def force(self, rcoef, chunk):
        """sum_n rcoef(n) dpsi(n)/dtheta over the full basis."""
        D, tot = self.m.D, None
        for s in range(0, D, chunk):
            idx = np.arange(s, min(s + chunk, D))
            gg = self.grad(self.forward(*self._occ(idx)), rcoef[idx], False)
            tot = gg if tot is None else tot + gg
        return tot

    def per_sample_O(self, idx):
        c = self.forward(*self._occ(idx))
        psi = c["psi"]
        safe = np.where(np.abs(psi) > 1e-300, psi, 1e-300)
        return self.grad(c, 1.0 / safe, True)


ARM_DEF = {
    "nnbf128": dict(mode="full", K=1, free=None, shared_ref=True),
    "mdnnbf": dict(mode="full", K="K", free="Hs", shared_ref=False),
    "lchfds": dict(mode="lr", K=1, free="Hs", Ha=0),
    "schfds": dict(mode="lr", K=1, free="Ha"),
    "sklr": dict(mode="lr", K="K", free="Ha"),
    "sjnet": dict(mode="lr", K=1, free="Ha", r0=True, Hs=0),
}


def arm_kwargs(arm, model, args, Pu, Pd):
    """Return constructor kwargs for `arm` with its free width solved to match the
    parameter count of NNBF(width=args.match_width) on `model`."""
    target = LRAnsatz(model, Pu, Pd, "full", K=1, Hs=args.match_width).n_params()
    d = ARM_DEF[arm]
    kw = dict(mode=d["mode"], K=(args.K if d["K"] == "K" else d["K"]),
              r=(0 if d.get("r0") else args.r), shared_ref=d.get("shared_ref", True),
              Hs=d.get("Hs", args.hs_sign), Ha=d.get("Ha", 0))
    if arm == "nnbf128":
        kw.update(Hs=args.match_width, Ha=0)
        return kw, target
    f = d["free"]
    if f == "Hs":
        kw["Ha"] = 0
    def cnt(w):
        k2 = dict(kw); k2[f] = w
        return LRAnsatz(model, Pu, Pd, **k2).n_params()
    c1, c2 = cnt(1), cnt(2)
    slope, base = c2 - c1, c1 - (c2 - c1)
    kw[f] = max(1, int(round((target - base) / slope)))
    return kw, target


# ============================================================================ SR
def optimize_sr(ans, H, args, rng, label):
    """SR with the EXACT force and a sampled metric (same algorithm as P1.1)."""
    best_E, best_x = np.inf, ans.flat().copy()
    for it in range(args.sr_iters):
        psi = ans.amplitudes(args.chunk)
        E, Hp, nn = energy(H, psi)
        if not np.isfinite(E):
            log(f"{label}: non-finite energy at it {it}; restoring best")
            ans.set_flat(best_x)
            break
        if E < best_E:
            best_E, best_x = E, ans.flat().copy()
        f = ans.force((Hp - E * psi) / nn, args.chunk)
        idx = rng.choice(len(psi), size=args.sr_samples, p=psi ** 2 / nn)
        O = ans.per_sample_O(idx)
        Oc = (O - O.mean(0)) / np.sqrt(len(idx))
        lam = args.sr_shift
        Km = Oc @ Oc.T + lam * np.eye(len(idx))
        dx = (f - Oc.T @ np.linalg.solve(Km, Oc @ f)) / lam
        step = args.sr_lr * dx
        nrm = np.linalg.norm(dx)
        if nrm * args.sr_lr > 1.0:
            step *= 1.0 / (nrm * args.sr_lr)
        ans.set_flat(ans.flat() - step)
        if it % 50 == 0 or it == args.sr_iters - 1:
            log(f"{label}: it {it:5d}  E = {E:.8f}  |f| = {np.linalg.norm(f):.2e}")
    E, _, _ = energy(H, ans.amplitudes(args.chunk))
    if E > best_E:
        ans.set_flat(best_x)
    return ans


# ============================================================================ fast local energy
def neighbours(T):
    return [np.nonzero(T[x])[0] for x in range(T.shape[0])]


def eloc_fast(ans, T, nbr, U, ou, od):
    """Exact local energy of one configuration using the low-rank hop response.
    Setup (A^{-1}, G0 = Phi0 A^{-1}, Y, Hk: carried across Markov steps in VMC) is timed
    separately; the per-hop work is O(N (Hs K r + K r^2) + Ha). Requires mode 'lr'."""
    p, K, r, V = ans.p, ans.K, ans.r, ans.V
    assert ans.mode == "lr" and ans.Kref == 1
    t0 = time.perf_counter()
    occ = {"u": ou, "d": od}
    x = np.concatenate([ou, od]).astype(float) * 2 - 1
    sd = {}
    for s in ("u", "d"):
        rows = np.nonzero(occ[s])[0]
        A0 = p["P0" + s][0][rows]
        Ainv = np.linalg.inv(A0)
        G0 = p["P0" + s][0] @ Ainv
        e = dict(rows=rows, Ainv=Ainv, G0=G0, N=len(rows))
        if r > 0:
            Phr = p["Ph" + s][:, rows]                       # (K, N, r)
            e["Y"] = Ainv[None] @ Phr                         # (K, N, r)
            e["Hk"] = G0[None] @ Phr                          # (K, V, r)
        sd[s] = e
    t_setup = time.perf_counter() - t0
    t0 = time.perf_counter()
    wk = ans._wk()
    nu, nd = ans.nu, ans.nd

    def Cs(out, s):
        B = out.shape[0]
        o0 = 0 if s == "u" else r * nu
        N = nu if s == "u" else nd
        return out.reshape(B, K, -1)[:, :, o0:o0 + r * N].reshape(B, K, r, N)

    def dets(C, Y):  # C (B,K,r,N), Y (B or 1,K,N,r)
        if r == 0:
            return np.ones(C.shape[:2])
        return np.linalg.det(np.eye(r) + C @ Y)

    if ans.trunk:
        pre = x @ p["W1"] + p["b1"]
        out = (np.tanh(pre) @ p["W2"] + p["b2"])[None]
    if ans.Ha > 0:
        prea = x @ p["W1a"] + p["b1a"]
        J = np.tanh(prea) @ p["w2a"]
    else:
        J = 0.0
    Dold = {s: (dets(Cs(out, s), sd[s]["Y"][None]) if r > 0 else np.ones((1, K))) for s in sd}
    S = float((Dold["u"] * Dold["d"])[0] @ wk)
    E = U * float(ou @ od)
    for si, s in enumerate(("u", "d")):
        e = sd[s]
        ii, xs, ys = [], [], []
        for i, xx in enumerate(e["rows"]):
            for y in nbr[xx]:
                if not occ[s][y]:
                    ii.append(i); xs.append(xx); ys.append(y)
        if not ii:
            continue
        ii, xs, ys = np.array(ii), np.array(xs), np.array(ys)
        nh = len(ii)
        rho0 = e["G0"][ys, ii]
        if ans.trunk:
            pre2 = pre[None] - 2 * p["W1"][si * V + xs] + 2 * p["W1"][si * V + ys]
            out2 = np.tanh(pre2) @ p["W2"] + p["b2"]
        fac = np.ones(nh)
        if ans.Ha > 0:
            pa = prea[None] - 2 * p["W1a"][si * V + xs] + 2 * p["W1a"][si * V + ys]
            fac = np.exp(np.tanh(pa) @ p["w2a"] - J)
        if r > 0:
            o = "d" if s == "u" else "u"
            vec = (p["Ph" + s][:, ys] - e["Hk"][:, ys]) / rho0[None, :, None]   # (K, nh, r)
            Y2 = e["Y"][None] + e["Ainv"][:, ii].T[:, None, :, None] * np.transpose(vec, (1, 0, 2))[:, :, None, :]
            Dn_s = dets(Cs(out2, s), Y2)
            Dn_o = dets(Cs(out2, o), sd[o]["Y"][None])
            tot = (Dn_s * Dn_o) @ wk
        else:
            tot = np.full(nh, S)
        ratio = fac * rho0 * tot / S
        E += float(np.sum(-T[ys, xs] * ratio))
    return E, t_setup, time.perf_counter() - t0


def eloc_brute(ans, T, nbr, U, ou, od, chunk=128):
    """Local energy by rebuilding every connected amplitude from scratch (sorted rows,
    explicit Jordan-Wigner sign). Independent of eloc_fast's algebra."""
    V = ans.V
    def amp(OU, OD):
        ru = np.array([np.nonzero(o)[0] for o in OU])
        rd = np.array([np.nonzero(o)[0] for o in OD])
        return ans.forward(OU, OD, ru, rd)["psi"]
    psi0 = amp(ou[None], od[None])[0]
    E = U * float(ou @ od)
    configs, coefs = [], []
    for s in ("u", "d"):
        o = ou if s == "u" else od
        for xx in np.nonzero(o)[0]:
            for y in nbr[xx]:
                if o[y]:
                    continue
                n2 = o.copy(); n2[xx] = 0; n2[y] = 1
                lo, hi = min(xx, y), max(xx, y)
                sgn = (-1) ** int(o[lo + 1:hi].sum())
                configs.append((n2, od) if s == "u" else (ou, n2))
                coefs.append(-T[y, xx] * sgn)
    for a in range(0, len(configs), chunk):
        blk = configs[a:a + chunk]
        ps = amp(np.array([b[0] for b in blk]), np.array([b[1] for b in blk]))
        E += float(np.sum(np.array(coefs[a:a + chunk]) * ps / psi0))
    return E


def random_lr(V, nup, ndn, T, arm_kw, seed, scale=0.3):
    class Dims:
        pass
    m = Dims(); m.V, m.nup, m.ndn = V, nup, ndn
    rng = np.random.default_rng(seed)
    e, v = np.linalg.eigh(-T)
    Pu = v[:, :nup] + 0.1 * rng.standard_normal((V, nup)) / np.sqrt(V)
    Pd = v[:, :ndn] + 0.1 * rng.standard_normal((V, ndn)) / np.sqrt(V)
    a = LRAnsatz(m, Pu, Pd, seed=seed, w2_scale=scale, **arm_kw)
    if a.Ha > 0:
        a.p["w2a"] = 0.1 * rng.standard_normal(a.Ha)
    if a.K > 1:
        a.p["wk"] = rng.standard_normal(a.K)
    return a


def cost_side_check(args):
    arm_kw = dict(mode="lr", K=args.K, r=args.r, Hs=args.hs_sign, Ha=128)
    rows = []
    for L in args.size_list:
        V = L * L
        Ne = 2 * int(round(V * (1 - args.doping) / 2))
        n = Ne // 2
        T, _, _ = build_lattice(L, L, "pbc", "pbc", 1.0, args.tp)
        nbr = neighbours(T)
        ans = random_lr(V, n, n, T, arm_kw, args.seed + L)
        rng = np.random.default_rng(args.seed + 7 * L)
        tf, ts, tb, eps = [], [], [], 0.0
        for _ in range(args.nconf):
            ou = np.zeros(V, int); od = np.zeros(V, int)
            ou[rng.choice(V, n, replace=False)] = 1
            od[rng.choice(V, n, replace=False)] = 1
            Ef, t_set, t_f = eloc_fast(ans, T, nbr, args.U, ou, od)
            tf.append(t_f); ts.append(t_set)
            if L <= args.brute_max:
                t0 = time.perf_counter()
                Eb = eloc_brute(ans, T, nbr, args.U, ou, od)
                tb.append(time.perf_counter() - t0)
                eps = max(eps, abs(Ef - Eb) / (1 + abs(Eb)))
        row = dict(L=L, V=V, Ne=Ne, T_E_fast=float(np.median(tf)), T_setup=float(np.median(ts)),
                   T_E_brute=float(np.median(tb)) if tb else None, eps=eps if tb else None)
        rows.append(row)
        log(f"cost L={L:3d} Ne={Ne:4d} T_E_fast={row['T_E_fast']:.3e} T_setup={row['T_setup']:.3e} "
            f"T_E_brute={row['T_E_brute'] if tb else float('nan'):.3e} eps={row['eps'] if tb else float('nan'):.2e}")
    def slope(key, rr):
        rr = [q for q in rr if q[key]]
        if len(rr) < 2:
            return float("nan")
        return float(np.polyfit(np.log([q["Ne"] for q in rr]), np.log([q[key] for q in rr]), 1)[0])
    s_fast = slope("T_E_fast", rows[-3:])
    s_brute = slope("T_E_brute", [q for q in rows if q["T_E_brute"]][-3:])
    eps = max([q["eps"] for q in rows if q["eps"] is not None] or [float("nan")])
    if not (eps <= EPS_SIDE):
        v = "FAIL"
    elif s_fast <= SLOPE_SIDE and s_brute >= BRUTE_GUARD:
        v = "PASS"
    elif s_fast > 2.8:
        v = "FAIL"
    else:
        v = "AMBIGUOUS"
    res = dict(rows=rows, slope_fast=s_fast, slope_brute=s_brute, eps=eps, verdict=v, arm=arm_kw)
    log(f"side check: slope_fast={s_fast:.2f} slope_brute={s_brute:.2f} eps={eps:.2e} -> {v}")
    return res


# ============================================================================ checks
def run_checks(seed):
    ok = True
    m = Hubbard(**{**SMOKE_INST, "t": 1.0})
    Pu, Pd = uhf_orbitals(m, seed)
    iu, idd = m.iu_id()
    # (a) reduction: full mode K=1 equals P1.1 NNBF with identical parameters
    a1 = Ansatz(m, Pu, Pd, "nnbf", width=6, seed=seed); a1.p["W2"] *= 300
    a2 = LRAnsatz(m, Pu, Pd, "full", K=1, Hs=6, seed=seed)
    a2.p["P0u"][0], a2.p["P0d"][0] = a1.p["Pu"], a1.p["Pd"]
    for k in ("W1", "b1", "W2", "b2"):
        a2.p[k] = a1.p[k].copy()
    d = np.max(np.abs(a1.amplitudes(4096) - a2.amplitudes(4096)))
    log(f"check full-mode == P1.1 NNBF: max|dpsi| = {d:.2e}"); ok &= d < 1e-12
    # (b) gradient checks for each arm type
    for name, kw in [("full K=2 sep", dict(mode="full", K=2, Hs=4, Ha=3, shared_ref=False)),
                     ("lr K=1 r=2 + J", dict(mode="lr", K=1, r=2, Hs=4, Ha=3)),
                     ("lr K=3 r=1 + J", dict(mode="lr", K=3, r=1, Hs=4, Ha=3)),
                     ("slater x J", dict(mode="lr", K=1, r=0, Hs=0, Ha=3))]:
        a = LRAnsatz(m, Pu, Pd, seed=seed, w2_scale=0.3, **kw)
        rng = np.random.default_rng(seed)
        if a.Ha:
            a.p["w2a"] = 0.3 * rng.standard_normal(a.Ha)
        if a.K > 1:
            a.p["wk"] = 1 + 0.5 * rng.standard_normal(a.K)
        x0 = a.flat().copy()
        psi = a.amplitudes(4096); E, Hp, nn = energy(m.H, psi)
        g = 2 * a.force((Hp - E * psi) / nn, 4096)
        dv = rng.standard_normal(len(x0)); dv /= np.linalg.norm(dv)
        h = 1e-6
        a.set_flat(x0 + h * dv); Ep = energy(m.H, a.amplitudes(4096))[0]
        a.set_flat(x0 - h * dv); Em = energy(m.H, a.amplitudes(4096))[0]
        a.set_flat(x0)
        fd = (Ep - Em) / (2 * h)
        # per-sample O consistency: O(n) psi(n) summed with r(n) equals force
        idx = np.arange(m.D)
        O = a.per_sample_O(idx)
        g2 = 2 * (((Hp - E * psi) / nn) * psi) @ O
        e1, e2 = abs(g @ dv - fd), np.max(np.abs(g2 - g))
        log(f"check grad [{name}]: analytic {g @ dv:.10e} fd {fd:.10e} |O-path diff| {e2:.1e}")
        ok &= e1 < 1e-6 * max(1, abs(fd)) and e2 < 1e-8 * max(1, np.abs(g).max())
        # (c) fast local energy == (H psi)(n)/psi(n) pointwise, all configs (lr only)
        if a.mode == "lr":
            nbr = neighbours(m.T)
            Hpsi = m.H @ psi
            err = 0.0
            for j in range(m.D):
                if abs(psi[j]) < 1e-8 * np.abs(psi).max():
                    continue
                El, _, _ = eloc_fast(a, m.T, nbr, m.U, m.occ_u[iu[j]].astype(int),
                                     m.occ_d[idd[j]].astype(int))
                err = max(err, abs(El - Hpsi[j] / psi[j]) / (1 + abs(Hpsi[j] / psi[j])))
            log(f"check fast E_loc == (H psi)/psi on all {m.D} configs [{name}]: max rel err {err:.1e}")
            ok &= err < 1e-9
    log(f"ALL CHECKS {'PASSED' if ok else 'FAILED'}")
    return ok


# ============================================================================ train task
def train_task(args):
    inst = dict(SMOKE_INST) if args.instance == 9 else dict(INSTANCES[args.instance])
    m = Hubbard(inst["Lx"], inst["Ly"], inst["bcx"], inst["bcy"], inst["nup"], inst["ndn"],
                1.0, inst["tp"], inst["U"])
    m.pair_setup(args.r_min)
    log(f"instance {args.instance} {inst} D={m.D} arm={args.arm} seed={args.seed}")
    t0 = time.time()
    W, Vv, deg = solve_ed(m, 3, 1e-8, UHF_SEED)
    gap = float(W[1] - W[0])
    log(f"ED levels {W.tolist()} deg {deg} Delta1/V={gap / m.V:.4f} ({time.time() - t0:.0f} s)")
    Pu, Pd = uhf_orbitals(m, UHF_SEED)
    kw, target = arm_kwargs(args.arm, m, args, Pu, Pd)
    ans = LRAnsatz(m, Pu, Pd, seed=args.seed, **kw)
    log(f"arm {args.arm}: {kw}  n_params={ans.n_params()}  target={target}")
    rng = np.random.default_rng(args.seed + 1000 * ARMS.index(args.arm))
    t1 = time.time()
    optimize_sr(ans, m.H, args, rng, args.arm)
    t_sr = time.time() - t1
    psi = ans.amplitudes(args.chunk)
    t2 = time.time()
    b = budget(m, psi, W, Vv, deg, args, rng)
    log(f"{args.arm}: dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} dE_amp={b['dE_amp']:.5f} "
        f"f_E_up={b['f_E_up']:.3f} eC_tot={b['eC_tot']:.3e} spread={b['spread']:.1e} "
        f"(SR {t_sr:.0f} s, budget {time.time() - t2:.0f} s)")
    res = dict(params={k: v for k, v in vars(args).items()}, instance=args.instance, arm=args.arm,
               seed=args.seed, arm_kwargs=kw, n_params=ans.n_params(), P_target=target,
               ed_levels=W.tolist(), gap_per_site=gap / m.V, competitor_flag=bool(gap / m.V < 0.02),
               budget=b, t_sr=t_sr)
    return res


# ============================================================================ aggregate
def aggregate(d):
    import glob
    best = {}
    for f in sorted(glob.glob(os.path.join(d, "inst*_*.json"))):
        r = json.load(open(f))
        key = (r["instance"], r["arm"])
        if key not in best or r["budget"]["E"] < best[key]["budget"]["E"]:
            best[key] = r
    insts = sorted({k[0] for k in best})
    verdicts = {}
    out = dict(rows=[], routes={})
    for inst in insts:
        line = [f"instance {inst}:"]
        for a in ARMS:
            if (inst, a) in best:
                bb = best[(inst, a)]["budget"]
                line.append(f"{a}: dE_tot={bb['dE_tot']:.4f} dE_sign={bb['dE_sign']:.4f} eC={bb['eC_tot']:.2e}")
        print("  ".join(line[:1]) + "\n    " + "\n    ".join(line[1:]))
    for route, arm in (("2.1", "schfds"), ("2.2", "sklr")):
        per = []
        for inst in insts:
            need = [arm, "sjnet", "nnbf128", "mdnnbf"] + (["lchfds"] if route == "2.1" else [])
            if not all((inst, a) in best for a in need):
                per.append(("MISSING", inst, {}))
                continue
            B = {a: best[(inst, a)]["budget"] for a in need}
            rs = min(B["nnbf128"]["dE_sign"], B["mdnnbf"]["dE_sign"])
            rt = min(B["nnbf128"]["dE_tot"], B["mdnnbf"]["dE_tot"])
            rc = min(B["nnbf128"]["eC_tot"], B["mdnnbf"]["eC_tot"])
            q = dict(rho_s=B[arm]["dE_sign"] / rs, rho_t=B[arm]["dE_tot"] / rt,
                     rho_C=B[arm]["eC_tot"] / rc, gamma=B[arm]["dE_sign"] / B["sjnet"]["dE_sign"])
            if route == "2.1":
                q["beats_lc"] = bool(B[arm]["dE_tot"] < B["lchfds"]["dE_tot"])
                q["lc_rho_s"] = B["lchfds"]["dE_sign"] / rs
                q["lc_rho_t"] = B["lchfds"]["dE_tot"] / rt
            ok = (q["rho_s"] <= RHO_S_PASS and q["rho_t"] <= RHO_T_PASS and q["rho_C"] <= RHO_C_PASS
                  and q["gamma"] <= GAMMA_PASS and q.get("beats_lc", True))
            bad = q["rho_s"] >= RHO_S_FAIL or q["gamma"] >= GAMMA_FAIL or q["rho_C"] >= RHO_C_FAIL
            per.append(("PASS" if ok else ("FAIL" if bad else "AMB"), inst, q))
            print(f"route {route} ({arm}) instance {inst}: " +
                  " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in q.items()))
        tags = [p[0] for p in per]
        if any(t == "MISSING" for t in tags):
            v = "INCOMPLETE"
        elif any(t == "FAIL" for t in tags):
            v = "FAIL"
        elif route == "2.1" and not any(p[2].get("beats_lc") for p in per):
            v = "FAIL"
        elif all(t == "PASS" for t in tags):
            v = "PASS"
        else:
            v = "AMBIGUOUS"
        verdicts[route] = v
        out["routes"][route] = dict(verdict=v, per_instance=[(t, i, q) for t, i, q in per])
        print(f"ROUTE {route} ({arm}) VERDICT: {v}")
    cs = os.path.join(d, "cost_side_check.json")
    if os.path.exists(cs):
        c = json.load(open(cs))
        print(f"cost side check: slope_fast={c['slope_fast']:.2f} slope_brute={c['slope_brute']:.2f} "
              f"eps={c['eps']:.1e} -> {c['verdict']}")
        out["cost"] = c["verdict"]
    with open(os.path.join(d, "aggregate.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    return 0


# ============================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P2.1 round-2 test (see module docstring)")
    ap.add_argument("--mode", choices=["train", "cost", "aggregate", "smoke"], default="smoke")
    ap.add_argument("--task", type=int, default=-1)
    ap.add_argument("--instance", type=int, default=-1, choices=[-1, 0, 1, 2, 9])
    ap.add_argument("--arm", choices=[""] + ARMS, default="")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--hs-sign", type=int, default=64)
    ap.add_argument("--match-width", type=int, default=128)
    ap.add_argument("--sr-iters", type=int, default=1500)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--lbfgs-iters", type=int, default=4000)
    ap.add_argument("--fn-iters", type=int, default=10)
    ap.add_argument("--fn-floor", type=float, default=1e-3)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--sizes", type=str, default="8,12,16,20,24")
    ap.add_argument("--brute-max", type=int, default=16)
    ap.add_argument("--nconf", type=int, default=2)
    ap.add_argument("--tp", type=float, default=-0.25)
    ap.add_argument("--U", type=float, default=8.0)
    ap.add_argument("--doping", type=float, default=0.125)
    ap.add_argument("--results", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    a = ap.parse_args(argv)
    if a.task >= 0:
        if a.instance < 0:
            a.instance = a.task // 12
        if not a.arm:
            a.arm = ARMS[(a.task % 12) // 2]
        if a.seed < 0:
            a.seed = SEEDS[a.task % 2]
    if a.seed < 0:
        a.seed = SEEDS[0]
    a.size_list = [int(s) for s in a.sizes.split(",") if s.strip()]
    return a


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.results, exist_ok=True)
    if args.mode == "aggregate":
        return aggregate(args.results)
    if args.mode == "cost":
        res = cost_side_check(args)
        out = args.out or os.path.join(args.results, "cost_side_check.json")
        json.dump(res, open(out, "w"), indent=1, default=float)
        return 0
    if args.mode == "smoke":
        t0 = time.time()
        if not run_checks(UHF_SEED):
            return 2
        # tiny end-to-end training of every arm on the 3x2 smoke cluster (not Hard Core)
        sm = parse_args(["--instance", "9", "--sr-iters", "15", "--sr-samples", "100",
                         "--lbfgs-iters", "200", "--fn-iters", "1", "--match-width", "8"])
        for arm in ARMS:
            sm.arm = arm
            train_task(sm)
        sc = parse_args(["--sizes", "4,6,8", "--brute-max", "8", "--nconf", "1"])
        cost_side_check(sc)
        log(f"smoke total {time.time() - t0:.1f} s")
        return 0
    if args.instance < 0 or not args.arm:
        print("--mode train needs --task or --instance and --arm"); return 2
    out = args.out or os.path.join(args.results, f"inst{args.instance}_{args.arm}_s{args.seed}.json")
    if os.path.exists(out) and not args.overwrite:
        log(f"{out} exists; use --overwrite 1"); return 0
    if not run_checks(UHF_SEED):
        return 2
    res = train_task(args)
    json.dump(res, open(out, "w"), indent=1, default=float)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
