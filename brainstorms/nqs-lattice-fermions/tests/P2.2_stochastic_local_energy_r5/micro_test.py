#!/usr/bin/env python3
"""
P2.2 round-5 decisive test.  Two routes against the refined (post-Saturation-
Diagnostic) formulation of the leaf, in three parts.

Saturation result this test is built on (see the leaf artifact, Round 5):
an unbiased estimator of E_L(n) = H_nn + sum_j x_j that queries term j with
inclusion probability pi_j has anticipated variance >= sum_j s_j^2 (1/pi_j - 1)
(Godambe-Joshi), s_j = the uncertainty of x_j given everything the estimator
knows for free.  Exactness forces pi_j = 1 for every j with s_j > 0, i.e.
Omega(|C|) work per sample.  A gain is possible only by (F_A) making each x_j
cheap arithmetic by construction (architecture), or (F_B) making s_j small with
free information, e.g. the same network at an earlier parameter (amortization).

Parts
-----
L  (route R5.2, F_B) Anchored local energy (ALE): SVRG on the inner sum of
   E_L.  SR epochs of T steps on one fixed, reweighted sample set.  At the
   epoch anchor theta~: exact enumeration x~_j = h_j psi~(j)/psi~(n), stored.
   Inner step at theta:  E^ALE0(n) = H_nn + sum_j x~_j
                                 + (1/Kd) sum_k [x_jk(theta) - x~_jk] / p_jk,
                         E^ALE1(n) = E^ALE0 with the first-order anchor term
                                 g~(n).dtheta added and subtracted per draw
                                 (g~(n) = grad_theta E_L(n) at the anchor, one
                                 backward pass through the anchor enumeration).
   p_j = 0.9 |x~_j| / sum|x~| + 0.1 |h_j| / sum|h|.  Unbiased given the samples
   for any theta; the increment vanishes as dtheta -> 0 (no sign floor).
   Samples are reweighted by (psi(n)/psi~(n))^2; an epoch ends after T steps or
   when ESS < --ess-min * B.  Every arm runs SR (minSR form) from ONE common
   L-BFGS warm start of the general NNBF-J, until its cumulative local-energy
   cost reaches the budget C_b = --n-ref steps of EXACT.  The exact energy of
   every iterate is logged (evaluation only).
   Arms (one --arm per task; 3 SR seeds per task):
     EXACT        all connected terms, fresh samples every step.
     EXACTNF      noise-floor arm: EXACT with SR seeds 10 11 12.
     SCREEN:e     |h| > e only (biased; judged by the exact energy it reaches).
     SS:e:Kd      |h| > e exact + Kd heat-bath draws p ~ |h| (unbiased).
     ALE0:T:Kd    route R5.2, zeroth order.
     ALE1:T:Kd    route R5.2, first order.
     suffix +B    (any arm except EXACT/EXACTNF) equal-step-cost mode: the arm
                  spends its saving on samples per step instead of on steps,
                  B = --batch x (EXACT cost / arm cost per sample per step,
                  measured on 300 warm-state samples), capped at --b-cap x
                  --batch; leftover budget buys steps.  Both modes of every
                  control and route enter the verdict (best of each set).
   Cost models (both logged per step, cumulative):
     ops : amplitudes x amp_ops + generated terms x C_GEN (keyx_core units);
           SCREEN/SS charged only for kept terms (generous: HCI-style sorted
           integrals); ALE anchor = EXACT (+ 2 amp per term for the ALE1
           backward); ALE inner = 1 amp (psi(n)) + Kd amp (+ 2 Kd amp for the
           ALE1 JVPs) + Kd C_HASH (alias draws).
     cal : the same counts priced with single-thread seconds measured in this
           task (t_amp from keyx_core.ratio_exact on full rows, t_gen from
           SpatialHam.row); the ALE1 JVP is priced at 2 t_amp.
   Values: amplitudes are looked up in the full ED vector of the same network
   (identical numbers to per-amplitude evaluation); the per-amplitude kernels
   are timed separately on --n-wall samples (EXACT row vs ALE0 inner
   estimator, real ratio_exact calls) and reported with a bootstrap CI.
W  (route R5.1, F_A) Window-input backflow x all-orbital Jastrow (WIBJ):
   accuracy at large external/window ratio.  Wraps keyx_core Part A (L-BFGS
   on the exact energy) with arms NNBFJ (seeds 0 1 2), NNBFJ noise-floor
   (seeds 3 4 5) and WIBJ (seeds 0 1 2), extended SCREEN/SS grids.
S  (route R5.1) ED-free cost scaling: keyx_core Part S, arms NNBFJ and WIBJ,
   extended SCREEN/SS grids.

Pre-committed verdicts (python3 micro_test.py --aggregate DIR), tau = 0.2 mHa
----------------------------------------------------------------------------
R5.2 (Part L), per K, under BOTH cost models (ops and cal):
  dE_X(c) = mean exact dE over the iterates whose cumulative cost lies in
  (0.9 c, c]; medians over the 3 SR seeds.
  ctrl* = the control config (EXACT, SCREEN:*, SS:*) with the lowest median
  dE(C_b); D_ctrl its median.  nu = max(|median dE_EXACT(C_b) - median
  dE_EXACTNF(C_b)|, half the range over the six EXACT/EXACTNF seeds) (noise
  floor).  Both modes (steps, +B) count.  r* = the ALE config (ALE0, ALE1 pooled)
  with the lowest median dE(C_b/2).
  premise: median dE_EXACT(C_b) < 1.6 mHa and nu < tau.
  PASS(K): premise, median dE_r*(C_b/2) <= D_ctrl, and best-of-seeds
           dE_r*(C_b/2) <= best-of-seeds dE_ctrl*(C_b).
  FAIL(K): median dE_r*(C_b) > D_ctrl + nu (worse at EQUAL cost).
  AMBIGUOUS(K) otherwise.  A K passes only if it passes under both cost
  models and fails if it fails under either.
  Global: PASS iff PASS at every K; FAIL iff FAIL at the largest K;
  otherwise AMBIGUOUS -> pre-committed rerun at K = 24 with the same grid:
  PASS iff PASS(24), else KILL.
R5.1 (Parts W + S):
  Accuracy, per instance at its largest K: delta_med = median over seeds 0-2
  of (dE_WIBJ - dE_NNBFJ), paired; delta_best = min_seed dE_WIBJ - min over
  all six NNBFJ seeds; nu_W = |median NNBFJ(0-2) - median NNBFJ(3-5)|;
  premise: HF weight < 0.6, median NNBFJ dE < 1.6 mHa, nu_W < tau.
  ACC_OK if delta_med <= tau and delta_best <= tau on every instance (all
  premises hold); ACC_FAIL if delta_med > tau on a premise-valid instance;
  ACC_AMBIG otherwise.
  Cost (Part S, keyx_core accounting, variance-ratio corrected, controls on
  NNBFJ with SCREEN eps admissible iff its exact bias <= tau on every W
  instance and SS charged its measured increment): M_op(K), M_wall(K) with
  bootstrap 95% CI.
  PASS: ACC_OK, M_op(K_max) <= 0.5, M_op(K_max) < M_op(K_min), and the upper
        CI bound of M_wall(K_max) <= 0.8.
  FAIL: ACC_FAIL, or M_op(K_max) >= 1, or the lower CI bound of M_wall(K_max)
        > 1.
  AMBIGUOUS otherwise -> pre-committed: Part S with --basis-series 6,5
        (K = 168); PASS iff M_op <= 0.5 and M_wall CI upper <= 0.8, else KILL.
Hard Core: sign half only (same-sector competitors not examined).

Command-line options
--------------------
--part {L,W,S}          Required unless --smoke or --aggregate.
--n-atoms INT           H atoms = electrons (even, >= 4). Default 4 (L, W), 8 (S).
--bond-length FLOAT     Mean H-H spacing in bohr. Default 4.4 (L, W), 2.0 (S).
--distort FLOAT         Random displacement amplitude (bohr). Default 0.25.
--geom-seed INT         Displacement seed. Default 1.
--n-active INT [...]    L, W: active spatial MOs K. Default 16 (L), 20 24 (W).
--basis-series LIST     S: "ns,np" pairs (keyx_core). Default "2,1 3,2 4,3 5,4".
--arm STR               L only, one arm spec: EXACT, EXACTNF, SCREEN:e,
                        SS:e:Kd, ALE0:T:Kd, ALE1:T:Kd, each but the first two
                        optionally suffixed +B.  Default EXACT.
--w-arms LIST           W only: subset of NNBFJ NNBFJNF WIBJ (NNBFJNF = NNBFJ
                        with seeds 3 4 5). Default all three.
--seeds INT [...]       L: SR seeds (EXACTNF adds 10). W: network seeds for
                        NNBFJ/WIBJ. Default 0 1 2.
--hidden INT            Backflow hidden width. Default 64 (W, S), 32 (L).
--hj INT                Neural-Jastrow width. Default 32 (W, S), 16 (L).
--maxiter INT           W: L-BFGS iterations. Default 6000.
--warm-iters INT        L: L-BFGS iterations of the common warm start. Default 400.
--batch INT             L: samples per SR step B. Default 500.
--lr FLOAT              L: SR step size. Default 0.5.
--shift FLOAT           L: minSR diagonal shift. Default 1e-4.
--max-dtheta FLOAT      L: step-norm clip. Default 1.0.
--n-ref INT             L: budget C_b = n-ref EXACT steps. Default 200.
--max-steps-factor INT  L: cap on steps = factor x n-ref. Default 12.
--ess-min FLOAT         L: ALE re-anchors when ESS < ess-min * B. Default 0.5.
--b-cap INT             L: +B mode cap on B / --batch. Default 4.
--n-wall INT            L: samples with real-kernel wall-clock timing. Default 40.
                        S: as keyx_core. Default 12.
--n-samples INT         W/S: as keyx_core. Default 2000 (W), 200 (S).
--eps-screen FLOAT...   W/S grids. Default 1e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2.
--k-grid INT...         W/S SS draws. Default 1 4 16 64 256.
--tau FLOAT             Ha. Default 2e-4.
--out PATH              Result JSON (default: results_<part>_<tag>.json here).
--smoke                 Correctness checks (log-derivatives, VJP, ALE
                        unbiasedness and first-order identity, R5.3 certified
                        bound), a tiny Part L on H4 K=8 with every arm type,
                        tiny Parts W and S, and the smoke-scale aggregates.
--aggregate DIR         Read results_{L,W,S}_*.json in DIR, print verdicts.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import keyx_core as kc  # noqa: E402

TAU = 2e-4


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P2.2 round-5 test (ALE and WIBJ)")
    ap.add_argument("--part", choices=["L", "W", "S"], default=None)
    ap.add_argument("--n-atoms", type=int, default=None)
    ap.add_argument("--bond-length", type=float, default=None)
    ap.add_argument("--distort", type=float, default=0.25)
    ap.add_argument("--geom-seed", type=int, default=1)
    ap.add_argument("--n-active", type=int, nargs="+", default=None)
    ap.add_argument("--basis-series", nargs="+", default=["2,1", "3,2", "4,3", "5,4"])
    ap.add_argument("--arm", type=str, default="EXACT")
    ap.add_argument("--w-arms", nargs="+", default=["NNBFJ", "NNBFJNF", "WIBJ"], choices=["NNBFJ", "NNBFJNF", "WIBJ"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--hj", type=int, default=None)
    ap.add_argument("--maxiter", type=int, default=6000)
    ap.add_argument("--warm-iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--shift", type=float, default=1e-4)
    ap.add_argument("--max-dtheta", type=float, default=1.0)
    ap.add_argument("--n-ref", type=int, default=200)
    ap.add_argument("--max-steps-factor", type=int, default=12)
    ap.add_argument("--ess-min", type=float, default=0.5)
    ap.add_argument("--b-cap", type=int, default=4)
    ap.add_argument("--n-wall", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--eps-screen", type=float, nargs="+", default=[1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--k-grid", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", type=str, default=None)
    a = ap.parse_args(argv)
    if a.aggregate or a.smoke:
        return a
    if a.part is None:
        ap.error("--part L, W or S is required")
    d = {"L": (4, 4.4, [16], 32, 16, 40, None), "W": (4, 4.4, [20, 24], 64, 32, None, 2000),
         "S": (8, 2.0, None, 64, 32, 12, 200)}[a.part]
    a.n_atoms = a.n_atoms or d[0]
    a.bond_length = a.bond_length or d[1]
    a.n_active = a.n_active or d[2]
    a.hidden = a.hidden or d[3]
    a.hj = a.hj or d[4]
    a.n_wall = a.n_wall if a.n_wall is not None else d[5]
    a.n_samples = a.n_samples or d[6]
    if a.n_atoms < 4 or a.n_atoms % 2:
        ap.error("--n-atoms must be even and >= 4")
    if a.part == "L":
        parse_arm(a.arm)  # validates
        if a.batch < 10 or a.lr <= 0 or a.shift <= 0 or a.n_ref < 5 or not 0 < a.ess_min < 1:
            ap.error("bad --batch/--lr/--shift/--n-ref/--ess-min")
    if a.out is None:
        tag = f"N{a.n_atoms}_R{a.bond_length}_" + ("K" + "-".join(map(str, a.n_active)) if a.n_active else "S")
        tag += "_" + (a.arm.replace(":", "-") if a.part == "L" else "-".join(a.w_arms) if a.part == "W" else "NNBFJ-WIBJ")
        a.out = os.path.join(HERE, f"results_{a.part}_{tag}.json")
    return a


def parse_arm(s):
    eq = s.endswith("+B")
    f = (s[:-2] if eq else s).split(":")
    kind = f[0]
    if kind in ("EXACT", "EXACTNF") and len(f) == 1 and not eq:
        return {"kind": kind, "eqB": False}
    if kind == "SCREEN" and len(f) == 2:
        return {"kind": kind, "eps": float(f[1]), "eqB": eq}
    if kind == "SS" and len(f) == 3:
        return {"kind": kind, "eps": float(f[1]), "Kd": int(f[2]), "eqB": eq}
    if kind in ("ALE0", "ALE1") and len(f) == 3:
        return {"kind": kind, "T": int(f[1]), "Kd": int(f[2]), "eqB": eq}
    raise SystemExit(f"bad --arm {s!r}: use EXACT, EXACTNF, SCREEN:e, SS:e:Kd, ALE0:T:Kd, ALE1:T:Kd, "
                     f"optionally with the suffix +B (not for EXACT/EXACTNF)")


def per_sample_ops(inst, arm, sidx, amp):
    """Mean per-sample, per-step local-energy cost (ops) of an arm on samples sidx
    (ALE: amortized over an epoch of T steps)."""
    gen = kc.C_GEN
    C = np.diff(inst.Hoff.indptr)[sidx].astype(float)
    k = arm["kind"]
    if k in ("EXACT", "EXACTNF"):
        return float(np.mean(C * (gen + amp)))
    if k in ("SCREEN", "SS"):
        v = []
        for n in sidx:
            _, h = inst.row(n)
            m = int((np.abs(h) > arm["eps"]).sum())
            v.append(m * (gen + amp) + (arm["Kd"] * amp if k == "SS" and m < len(h) else 0))
        return float(np.mean(v))
    fo = 3 if k == "ALE1" else 1
    T, Kd = arm["T"], arm["Kd"]
    return float(np.mean((C * (gen + fo * amp) + (T - 1) * (amp + Kd * (fo * amp + kc.C_HASH))) / T))


# ----------------------------------------------------------------------------
# Per-sample log-derivatives and VJPs of the keyx_core Ansatz
# ----------------------------------------------------------------------------
def logderiv(net, th, occ_s, oi_s):
    """psi and O[b,k] = d ln psi(n_b) / d theta_k (B x P)."""
    psi, (cache, P, o, tj, fJ) = net.amplitudes(th, occ_s, oi_s)
    B = len(psi)
    parts = {}
    for s, (lo, nr, c0, ncol, F) in enumerate(net.blocks):
        f, hid, Phi, A, rows = cache[s]
        gA = np.transpose(np.linalg.inv(A), (0, 2, 1))
        gPhi = np.zeros_like(Phi)
        np.put_along_axis(gPhi, rows[:, :, None], gA, axis=1)
        gflat = gPhi.reshape(B, -1)
        parts[f"Phi0_{s}"] = gflat
        parts[f"W2_{s}"] = (gflat[:, :, None] * hid[:, None, :]).reshape(B, -1)
        dz = (gflat @ P[f"W2_{s}"]) * (1 - hid ** 2)
        parts[f"W1_{s}"] = (dz[:, :, None] * f[:, None, :]).reshape(B, -1)
        parts[f"b1_{s}"] = dz
    parts["u"] = o
    parts["v"] = o[:, net.iu[0]] * o[:, net.iu[1]]
    parts["aj"] = tj
    dzj = (1 - tj ** 2) * P["aj"][None]
    parts["Wj"] = (dzj[:, :, None] * fJ[:, None, :]).reshape(B, -1)
    parts["bj"] = dzj
    return psi, np.concatenate([parts[k] for k, _ in net.shapes], 1)


def vjp(net, th, occ_c, oi_c, cp):
    """sum_b cp_b d ln psi(n_b)/d theta (P-vector), batched backward pass."""
    psi, (cache, P, o, tj, fJ) = net.amplitudes(th, occ_c, oi_c)
    g = {}
    for s, (lo, nr, c0, ncol, F) in enumerate(net.blocks):
        f, hid, Phi, A, rows = cache[s]
        gA = cp[:, None, None] * np.transpose(np.linalg.inv(A), (0, 2, 1))
        gPhi = np.zeros_like(Phi)
        np.put_along_axis(gPhi, rows[:, :, None], gA, axis=1)
        gflat = gPhi.reshape(len(psi), -1)
        g[f"Phi0_{s}"] = gPhi.sum(0)
        g[f"W2_{s}"] = gflat.T @ hid
        dz = (gflat @ P[f"W2_{s}"]) * (1 - hid ** 2)
        g[f"W1_{s}"] = dz.T @ f
        g[f"b1_{s}"] = dz.sum(0)
    g["u"] = o.T @ cp
    g["v"] = (o.T @ (o * cp[:, None]))[net.iu]
    g["aj"] = tj.T @ cp
    dzj = cp[:, None] * (1 - tj ** 2) * P["aj"][None]
    g["Wj"] = dzj.T @ fJ
    g["bj"] = dzj.sum(0)
    return np.concatenate([g[k].ravel() for k, _ in net.shapes])


# ----------------------------------------------------------------------------
# Instance
# ----------------------------------------------------------------------------
class Inst:
    def __init__(self, N, R, distort, gseed, K, allow_weak=False):
        t0 = time.time()
        h, g, enuc, Erhf, eps = kc.build_molecule(N, R, distort, gseed, kc.H631GSS, K, "valence-p")
        self.ham = kc.SpatialHam(h, g, enuc, K, N)
        self.codes, self.occ, self.oi = kc.build_space(K, N // 2, N // 2)
        self.H = kc.build_hamiltonian(self.ham, self.codes, self.oi)
        w, v = kc.ground_states(self.H, 2)
        self.E0 = float(w[0])
        self.hfw = float(v[np.searchsorted(self.codes, kc.hf_code(K, N)), 0] ** 2)
        self.Hd = self.H.diagonal().copy()
        self.Hoff = (self.H - sp.diags(self.Hd)).tocsr()
        self.Hoff.eliminate_zeros()
        self.K, self.N, self.D = K, N, len(self.codes)
        self.info = {"N": N, "R": R, "K": K, "dim": self.D, "E_FCI": self.E0, "E_RHF": float(Erhf),
                     "HF_weight": self.hfw, "nnz_row": float(self.Hoff.nnz / self.D)}
        print(f"=== H{N} distorted chain R={R} K={K} dim={self.D} E_FCI={self.E0:.6f} HF wt={self.hfw:.3f} "
              f"<|C|>={self.Hoff.nnz/self.D:.0f} ({time.time()-t0:.1f}s)", flush=True)
        if self.hfw >= 0.6 and not allow_weak:
            raise SystemExit(f"instance not multireference: HF weight {self.hfw:.3f} >= 0.6")

    def row(self, n):
        a, b = self.Hoff.indptr[n], self.Hoff.indptr[n + 1]
        return self.Hoff.indices[a:b], self.Hoff.data[a:b]


class A_:
    pass


def make_net(K, N, arm, hidden, hj, seed):
    a = A_(); a.hidden, a.hj, a.window, a.groups = hidden, hj, N, 2
    return kc.make_net(K, N, arm, a, seed)


# ----------------------------------------------------------------------------
# Part L: SR with local-energy estimators
# ----------------------------------------------------------------------------
def calibrate(inst, net, th, n_wall, rng, Kd=4):
    """Single-thread wall-clock model of the real per-sample kernels, including
    per-call (interpreter) overhead, fitted on n_wall samples drawn from pi:
      amplitudes : t = a + b m   (keyx_core.ratio_exact on m terms of one row)
      generation : t = ag + bg |C| (SpatialHam.row)
      setup      : ts (net.setup, needed before any incremental amplitude)
    plus a direct real-kernel comparison: EXACT row (generation + setup + all
    amplitudes) against the ALE0 inner estimator (setup + Kd drawn amplitudes
    from a stored anchor), with a paired bootstrap CI."""
    P = net.unpack(th)
    psi, _ = net.amplitudes(th, inst.occ, inst.oi)
    wv = psi ** 2 / (psi @ psi)
    idx = rng.choice(inst.D, size=n_wall, p=wv)
    am, at, gm, gt, ts, wE, wA = [], [], [], [], [], [], []
    for n in idx:
        s0 = time.perf_counter()
        row = inst.ham.row(inst.codes[n], inst.oi[n])
        tg = time.perf_counter() - s0
        gm.append(len(row["h"])); gt.append(tg)
        s0 = time.perf_counter()
        st = net.setup(P, inst.oi[n])
        t_s = time.perf_counter() - s0
        ts.append(t_s)
        occ_j = kc.decode(row["code"], inst.ham.M, inst.N)
        C = len(row["h"])
        for m_ in sorted({1, 4, 16, 64, C}):
            m_ = min(m_, C)
            s0 = time.perf_counter()
            rx = net.ratio_exact(P, st, row["rem"][:m_], row["add"][:m_], occ_j[:m_])
            _ = row["h"][:m_] @ rx
            am.append(m_); at.append(time.perf_counter() - s0)
        s0 = time.perf_counter()
        st = net.setup(P, inst.oi[n])
        rx = net.ratio_exact(P, st, row["rem"], row["add"], occ_j)
        _ = row["Hnn"] + row["h"] @ rx
        wE.append(tg + time.perf_counter() - s0)
        xt = row["h"] * rx
        p = 0.9 * np.abs(xt) / np.abs(xt).sum() + 0.1 * np.abs(row["h"]) / np.abs(row["h"]).sum()
        s0 = time.perf_counter()
        st2 = net.setup(P, inst.oi[n])
        d = rng.choice(len(p), size=Kd, p=p)
        r2 = net.ratio_exact(P, st2, row["rem"][d], row["add"][d], occ_j[d])
        _ = xt.sum() + np.mean((row["h"][d] * r2 - xt[d]) / p[d])
        wA.append(time.perf_counter() - s0)
    b, a = np.polyfit(am, at, 1)
    bg, ag = np.polyfit(gm, gt, 1) if len(set(gm)) > 1 else (np.mean(gt) / np.mean(gm), 0.0)
    return {"a": float(max(a, 0)), "b": float(b), "ag": float(max(ag, 0)), "bg": float(bg), "ts": float(np.mean(ts)),
            "Kd_wall": Kd,
            "wall_EXACT_row": wE, "wall_ALE0_inner": wA,
            "wall_ratio_ALE0_over_EXACT": float(np.mean(wA) / np.mean(wE)),
            "wall_ratio_CI": kc.boot_ratio_ci(wA, wE)}


def cal_amp(c, m):
    """Seconds for one call of m incremental amplitudes (0 if m == 0)."""
    return 0.0 if m <= 0 else c["a"] + c["b"] * m


def cal_gen(c, m):
    return c["ag"] + c["bg"] * m


def sr_update(O, w, e, lr, shift, maxd):
    Ob = w @ O
    sw = np.sqrt(w)
    X = sw[:, None] * (O - Ob)
    eb = sw * (e - w @ e)
    T = X @ X.T
    T[np.diag_indices_from(T)] += shift
    d = -lr * (X.T @ np.linalg.solve(T, eb))
    nd = np.linalg.norm(d)
    if nd > maxd:
        d *= maxd / nd
    return d


class Anchor:
    """Stored anchor of one ALE epoch: samples, theta~, per-sample rows, x~,
    proposals, (ALE1) g~(n) = grad E_L(n) at theta~."""

    def __init__(self, inst, net, th, sidx, psi, first_order):
        self.th = th.copy(); self.sidx = sidx; self.psi_n = psi[sidx].copy()
        self.rows, self.xt, self.p, self.S = [], [], [], np.empty(len(sidx))
        for b, n in enumerate(sidx):
            j, h = inst.row(n)
            xt = h * psi[j] / psi[n]
            ax = np.abs(xt); ah = np.abs(h)
            p = 0.9 * ax / ax.sum() + 0.1 * ah / ah.sum()
            self.rows.append((j, h)); self.xt.append(xt); self.p.append(p / p.sum())
            self.S[b] = inst.Hd[n] + xt.sum()
        self.G = None
        if first_order:
            _, On = logderiv(net, th, inst.occ[sidx], inst.oi[sidx])
            G = np.empty((len(sidx), len(th)))
            for b, n in enumerate(sidx):
                j, _ = self.rows[b]
                G[b] = vjp(net, th, inst.occ[j], inst.oi[j], self.xt[b]) - self.xt[b].sum() * On[b]
            self.G = G


def ale_estimate(inst, net, anc, th, psi, Kd, rng, first_order, fd_eps=1e-4):
    """Inner-step ALE estimates on the anchor samples at theta (psi = full
    vector at theta).  First order: the per-draw JVP of x~_j along dtheta by a
    central difference on the drawn configurations only."""
    B = len(anc.sidx)
    e = anc.S.copy()
    dth = th - anc.th
    draws = [rng.choice(len(anc.p[b]), size=Kd, p=anc.p[b]) for b in range(B)]
    if first_order:
        e += anc.G @ dth
        nd = np.linalg.norm(dth)
        if nd > 0:
            u = dth / nd
            cfg = np.unique(np.concatenate([anc.sidx] + [anc.rows[b][0][draws[b]] for b in range(B)]))
            pp, _ = net.amplitudes(anc.th + fd_eps * u, inst.occ[cfg], inst.oi[cfg])
            pm, _ = net.amplitudes(anc.th - fd_eps * u, inst.occ[cfg], inst.oi[cfg])
            loc = {c: i for i, c in enumerate(cfg)}
    for b, n in enumerate(anc.sidx):
        j, h = anc.rows[b]
        d = draws[b]
        x = h[d] * psi[j[d]] / psi[n]
        delta = x - anc.xt[b][d]
        if first_order and np.linalg.norm(dth) > 0:
            jj = np.array([loc[c] for c in j[d]]); nn = loc[n]
            xp = h[d] * pp[jj] / pp[nn]; xm = h[d] * pm[jj] / pm[nn]
            delta -= (xp - xm) / (2 * fd_eps) * np.linalg.norm(dth)
        e[b] += np.mean(delta / anc.p[b][d])
    return e


def run_sr(inst, net, th0, arm, seed, args, costs, budget_ops, budget_cal, B=None):
    """SR (minSR form) with the arm's local-energy estimator.  Mode 'steps'
    (B = --batch): cheaper estimators buy more steps.  Mode '+B' (B given):
    cheaper estimators buy more samples per step (B = --batch x cost ratio,
    capped at --b-cap x --batch); any leftover budget buys steps."""
    rng = np.random.default_rng(10 ** 5 + 97 * seed + 1)
    th = th0.copy()
    amp = costs["amp_ops"]; gen = kc.C_GEN
    c = costs["cal"]
    max_steps = args.max_steps_factor * args.n_ref
    if B is None:
        B = args.batch
    kind = arm["kind"]
    traj = {"step": [], "ops": [], "cal": [], "dE_mHa": [], "ess": [], "anchor": []}
    cum_ops = cum_cal = 0.0
    anc = None; age = 0
    for step in range(max_steps):
        psi, _ = net.amplitudes(th, inst.occ, inst.oi)
        nrm = psi @ psi
        E = float(psi @ (inst.H @ psi) / nrm)
        traj["step"].append(step); traj["ops"].append(cum_ops); traj["cal"].append(cum_cal)
        traj["dE_mHa"].append(1e3 * (E - inst.E0))
        if not np.isfinite(E) or (cum_ops >= budget_ops and cum_cal >= budget_cal):
            traj["ess"].append(float("nan")); traj["anchor"].append(False)
            break
        wv = psi ** 2 / nrm
        is_anchor = False
        if kind in ("ALE0", "ALE1"):
            fo = kind == "ALE1"
            ess = B
            if anc is not None:
                r = (psi[anc.sidx] / anc.psi_n) ** 2
                w = r / r.sum(); ess = 1.0 / (w @ w)
            if anc is None or age >= arm["T"] or ess < args.ess_min * B:
                sidx = rng.choice(inst.D, size=B, p=wv)
                anc = Anchor(inst, net, th, sidx, psi, fo)
                age = 0; is_anchor = True
                w = np.full(B, 1.0 / B); ess = B
                e = anc.S.copy()
                nC = sum(len(rw[0]) for rw in anc.rows)
                o_ = nC * (gen + amp) + (2 * nC * amp if fo else 0)
                c_ = sum(cal_gen(c, len(rw[0])) + c["ts"] + cal_amp(c, len(rw[0])) * (3 if fo else 1) for rw in anc.rows)
            else:
                e = ale_estimate(inst, net, anc, th, psi, arm["Kd"], rng, fo)
                Kd = arm["Kd"]
                o_ = B * (amp + Kd * amp + Kd * kc.C_HASH + (2 * Kd * amp if fo else 0))
                c_ = B * (c["ts"] + cal_amp(c, Kd) * (3 if fo else 1))
            age += 1
            sidx = anc.sidx
        else:
            sidx = rng.choice(inst.D, size=B, p=wv)
            w = np.full(B, 1.0 / B); ess = B
            e = np.empty(B); o_ = c_ = 0.0
            for b, n in enumerate(sidx):
                j, h = inst.row(n)
                x = h * psi[j] / psi[n]
                if kind in ("EXACT", "EXACTNF"):
                    e[b] = inst.Hd[n] + x.sum(); m = len(x)
                    o_ += m * (gen + amp); c_ += cal_gen(c, m) + c["ts"] + cal_amp(c, m)
                else:
                    mk = np.abs(h) > arm["eps"]; m = int(mk.sum())
                    e[b] = inst.Hd[n] + x[mk].sum()
                    o_ += m * (gen + amp); c_ += cal_gen(c, m) + c["ts"] + cal_amp(c, m)
                    if kind == "SS" and m < len(x):
                        tl = np.nonzero(~mk)[0]
                        p = np.abs(h[tl]); p /= p.sum()
                        d = rng.choice(len(tl), size=arm["Kd"], p=p)
                        e[b] += np.mean(x[tl[d]] / p[d])
                        o_ += arm["Kd"] * amp; c_ += cal_amp(c, arm["Kd"])
        cum_ops += o_; cum_cal += c_
        traj["ess"].append(float(ess)); traj["anchor"].append(bool(is_anchor))
        _, O = logderiv(net, th, inst.occ[sidx], inst.oi[sidx])
        th = th + sr_update(O, w, e, args.lr, args.shift, args.max_dtheta)
    return traj


def run_part_L(args):
    out = {"part": "L", "arm": args.arm, "args": {k: v for k, v in vars(args).items()}, "instances": []}
    arm = parse_arm(args.arm)
    seeds = [s + 10 for s in args.seeds] if arm["kind"] == "EXACTNF" else args.seeds
    for K in args.n_active:
        inst = Inst(args.n_atoms, args.bond_length, args.distort, args.geom_seed, K)
        net = make_net(K, args.n_atoms, "NNBFJ", args.hidden, args.hj, 0)
        t0 = time.time()
        r = minimize(lambda t: net.energy_grad(t, inst.H, inst.occ, inst.oi), net.theta0, jac=True,
                     method="L-BFGS-B", options={"maxiter": args.warm_iters, "maxcor": 30, "gtol": 1e-12, "ftol": 1e-16})
        thw = r.x
        Ew = 1e3 * (r.fun - inst.E0)
        cal = calibrate(inst, net, thw, args.n_wall, np.random.default_rng(4242), arm.get("Kd", 4))
        amp = kc.amp_ops(net)
        mC = inst.Hoff.nnz / inst.D
        psi, _ = net.amplitudes(thw, inst.occ, inst.oi)
        wv = psi ** 2 / (psi @ psi)
        mCpi = float(wv @ np.diff(inst.Hoff.indptr))
        costs = {"amp_ops": amp, "cal": cal}
        budget_ops = args.n_ref * args.batch * mCpi * (kc.C_GEN + amp)
        budget_cal = args.n_ref * args.batch * (cal_gen(cal, mCpi) + cal["ts"] + cal_amp(cal, mCpi))
        print(f"  warm start: {args.warm_iters} L-BFGS its, dE={Ew:.3f} mHa ({time.time()-t0:.0f}s); P={net.theta0.size} "
              f"<|C|>_pi={mCpi:.0f}; cal: a={cal['a']*1e6:.0f}us b={cal['b']*1e6:.2f}us/amp ts={cal['ts']*1e6:.0f}us "
              f"bg={cal['bg']*1e6:.2f}us/term; real-kernel wall ALE0(Kd={cal['Kd_wall']})/EXACT per sample={cal['wall_ratio_ALE0_over_EXACT']:.4f} CI={cal['wall_ratio_CI']}", flush=True)
        rec = {**inst.info, "warm_dE_mHa": Ew, "P": int(net.theta0.size), "mean_conn_pi": mCpi, "calibration": cal,
               "costs": costs, "budget_ops": budget_ops, "budget_cal": budget_cal, "runs": [],
               "cert_bound": cert_bound(inst, net, thw, args.eps_screen, np.random.default_rng(7), 100)}
        Barm = None
        if arm["eqB"]:
            s0 = np.random.default_rng(99).choice(inst.D, size=300, p=wv)
            ratio = per_sample_ops(inst, {"kind": "EXACT"}, s0, amp) / per_sample_ops(inst, arm, s0, amp)
            Barm = int(min(args.b_cap * args.batch, max(args.batch, round(args.batch * ratio))))
            rec["eqB"] = {"cost_ratio_EXACT_over_arm": ratio, "B": Barm}
            print(f"  +B mode: per-sample cost ratio EXACT/arm = {ratio:.2f} -> B = {Barm}", flush=True)
        for seed in seeds:
            t1 = time.time()
            tr = run_sr(inst, net, thw, arm, seed, args, costs, budget_ops, budget_cal, Barm)
            rec["runs"].append({"seed": seed, **tr})
            fo = final_dE(tr, "ops", budget_ops); fc = final_dE(tr, "cal", budget_cal)
            print(f"  {args.arm} seed={seed}: steps={len(tr['step'])} anchors={sum(tr['anchor'])} "
                  f"dE(C_b)[ops]={fo:.3f} [cal]={fc:.3f} dE(C_b/2)[ops]={final_dE(tr,'ops',budget_ops/2):.3f} "
                  f"last dE={tr['dE_mHa'][-1]:.3f} mHa ({time.time()-t1:.0f}s)", flush=True)
        out["instances"].append(rec)
        with open(args.out, "w") as fh:
            json.dump(out, fh, default=float)
    return out


def final_dE(tr, axis, c):
    x = np.array(tr[axis]); y = np.array(tr["dE_mHa"])
    m = (x > 0.9 * c) & (x <= c)
    if not m.any():
        m = x <= c
        if not m.any():
            return float("nan")
        return float(y[m][-1])
    return float(y[m].mean())


# ----------------------------------------------------------------------------
# R5.3 pre-check: certified screening bias via a configuration-independent
# Hadamard bound (|det| <= prod of row-norm bounds; |tanh| <= 1)
# ----------------------------------------------------------------------------
def cert_bound(inst, net, th, eps_grid, rng, nsamp):
    P = net.unpack(th)
    lo, nr, c0, ncol, F = net.blocks[0]
    R = np.linalg.norm(P["Phi0_0"], axis=1) + np.linalg.norm(P["W2r_0"], axis=1).sum(1)
    psi, (cache, _, o, tj, _) = net.amplitudes(th, inst.occ, inst.oi)
    logJ = o @ P["u"] + 0.5 * np.einsum("bp,bp->b", o @ P["V"], o) + tj @ P["aj"]
    logR = np.log(R)[inst.oi].sum(1)
    wv = psi ** 2 / (psi @ psi)
    idx = rng.choice(inst.D, size=nsamp, p=wv)
    res = {}
    for e in eps_grid:
        bnd, act = [], []
        for n in idx:
            j, h = inst.row(n)
            m = np.abs(h) <= e
            bnd.append(np.sum(np.abs(h[m]) * np.exp(logJ[j[m]] + logR[j[m]])) / abs(psi[n]))
            act.append(abs(np.sum(h[m] * psi[j[m]])) / abs(psi[n]))
        res[str(e)] = {"certified_mHa": 1e3 * float(np.mean(bnd)), "actual_abs_mHa": 1e3 * float(np.mean(act))}
    return res


# ----------------------------------------------------------------------------
# Parts W and S: wrappers around keyx_core
# ----------------------------------------------------------------------------
def kc_args(args, part):
    lst = ["--part", "A" if part == "W" else "S", "--n-atoms", str(args.n_atoms), "--bond-length", str(args.bond_length),
           "--distort", str(args.distort), "--geom-seed", str(args.geom_seed), "--hidden", str(args.hidden),
           "--hj", str(args.hj), "--n-samples", str(args.n_samples), "--eps-screen"] + [str(e) for e in args.eps_screen] + \
          ["--k-grid"] + [str(k) for k in args.k_grid] + ["--out", args.out]
    if part == "W":
        lst += ["--maxiter", str(args.maxiter), "--n-active"] + [str(k) for k in args.n_active]
    else:
        lst += ["--n-wall", str(args.n_wall), "--basis-series"] + list(args.basis_series) + ["--arms", "NNBFJ", "WIBJ"]
    if getattr(args, "allow_weak", False):
        lst += ["--allow-weak"]
    return kc.parse_args(lst)


def run_part_W(args):
    res = {"part": "A", "r5": True, "instance": f"H{args.n_atoms}_R{args.bond_length}_d{args.distort}_g{args.geom_seed}",
           "args": {k: v for k, v in vars(args).items()}, "instances": []}
    for K in args.n_active:
        merged = None
        for wa in args.w_arms:
            a = kc_args(args, "W")
            a.arms = ["WIBJ"] if wa == "WIBJ" else ["NNBFJ"]
            a.seeds = [s + 3 for s in args.seeds] if wa == "NNBFJ" + "NF" else list(args.seeds)
            it = kc.run_part_A_instance(K, a)
            for r in it["runs"]:
                r["w_arm"] = wa
            if merged is None:
                merged = it
            else:
                merged["runs"].extend(it["runs"])
        res["instances"].append(merged)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    return res


def run_part_S(args):
    a = kc_args(args, "S")
    res = {"part": "S", "r5": True, "args": {k: v for k, v in vars(args).items()}, "instances": []}
    for ns, npp in a.basis_series:
        res["instances"].append(kc.run_part_S_basis(ns, npp, a))
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    return res


# ----------------------------------------------------------------------------
# Verdicts
# ----------------------------------------------------------------------------
def verdict_L(Ls, tau=TAU):
    tauM = 1e3 * tau
    byK = {}
    for d in Ls:
        for it in d["instances"]:
            e = byK.setdefault(it["K"], {"arms": {}, "budget_ops": it["budget_ops"], "budget_cal": it["budget_cal"]})
            e["arms"][d["arm"]] = it["runs"]
    out = {"tau_mHa": tauM, "byK": {}}
    if not byK:
        out["verdict"] = "INCOMPLETE"
        return out
    for K, e in sorted(byK.items()):
        arms = e["arms"]
        if "EXACT" not in arms or "EXACTNF" not in arms or not any(a.startswith("ALE") for a in arms):
            out["byK"][K] = {"verdict": "INCOMPLETE", "arms": sorted(arms)}
            continue
        rk = {}
        for axis, Cb in (("ops", e["budget_ops"]), ("cal", e["budget_cal"])):
            full = {a: [final_dE(r, axis, Cb) for r in runs] for a, runs in arms.items()}
            half = {a: [final_dE(r, axis, Cb / 2) for r in runs] for a, runs in arms.items()}
            med = lambda v: float(np.median(v)) if np.all(np.isfinite(v)) else float("inf")  # noqa: E731
            ctrls = [a for a in arms if a == "EXACT" or a.startswith("SCREEN") or a.startswith("SS")]
            routes = [a for a in arms if a.startswith("ALE")]
            cstar = min(ctrls, key=lambda a: med(full[a]))
            rstar = min(routes, key=lambda a: med(half[a]))
            D = med(full[cstar])
            pool = np.array(full["EXACT"] + full["EXACTNF"])
            nu = max(abs(med(full["EXACT"]) - med(full["EXACTNF"])), 0.5 * float(np.ptp(pool)))
            prem = med(full["EXACT"]) < 1.6 and nu < tauM
            Rh, Rf = med(half[rstar]), med(full[rstar])
            best_r = float(np.nanmin(half[rstar])); best_c = float(np.nanmin(full[cstar]))
            if Rf > D + nu:
                v = "FAIL"
            elif prem and Rh <= D and best_r <= best_c:
                v = "PASS"
            else:
                v = "AMBIGUOUS"
            rk[axis] = {"verdict": v, "ctrl_star": cstar, "D_ctrl_mHa": D, "nu_mHa": nu, "premise": bool(prem),
                        "route_star": rstar, "route_dE_half_mHa": Rh, "route_dE_full_mHa": Rf,
                        "best_seed_route_half": best_r, "best_seed_ctrl_full": best_c,
                        "median_dE_full": {a: med(v_) for a, v_ in full.items()},
                        "median_dE_half": {a: med(v_) for a, v_ in half.items()},
                        "seed_spread_full": {a: float(np.nanmax(v_) - np.nanmin(v_)) for a, v_ in full.items()}}
        vs = [rk[a]["verdict"] for a in rk]
        kv = "FAIL" if "FAIL" in vs else "PASS" if all(v == "PASS" for v in vs) else "AMBIGUOUS"
        out["byK"][K] = {"verdict": kv, **rk}
    done = {K: d["verdict"] for K, d in out["byK"].items() if d["verdict"] != "INCOMPLETE"}
    if not done:
        out["verdict"] = "INCOMPLETE"
    elif done[max(done)] == "FAIL":
        out["verdict"] = "FAIL"
    elif all(v == "PASS" for v in done.values()) and len(done) == len(out["byK"]):
        out["verdict"] = "PASS"
    else:
        out["verdict"] = "AMBIGUOUS (pre-committed: rerun Part L at K=24; PASS iff PASS(24), else KILL)"
    return out


def verdict_R51(Ws, Ss, tau=TAU):
    tauM = 1e3 * tau
    kc.ROUTE_ARM.clear(); kc.ROUTE_ARM["R5.1"] = "WIBJ"
    base = kc.verdict(Ws, Ss, tau) if Ws and Ss else {"routes": {}, "instances": {}}
    out = {"tau_mHa": tauM, "instances": {}, "cost_byK": base.get("routes", {}).get("R5.1", {}).get("byK", {}),
           "controls_transferred": base.get("controls_transferred")}
    inst = {}
    for d in Ws:
        for it in d["instances"]:
            e = inst.setdefault(d["instance"], {}).setdefault(it["K"], {"HF_weight": it["HF_weight"], "runs": []})
            e["runs"].extend(it["runs"])
    acc = []
    for name, byK in inst.items():
        Km = max(byK); runs = byK[Km]["runs"]
        n0 = {r["seed"]: r["dE_mHa"] for r in runs if r["arm"] == "NNBFJ" and r.get("w_arm", "NNBFJ") == "NNBFJ"}
        nf = {r["seed"]: r["dE_mHa"] for r in runs if r.get("w_arm") == "NNBFJNF"}
        wb = {r["seed"]: r["dE_mHa"] for r in runs if r["arm"] == "WIBJ"}
        if not (n0 and nf and wb):
            out["instances"][name] = {"K_max": Km, "status": "INCOMPLETE"}
            acc.append(None)
            continue
        dl = [wb[s] - n0[s] for s in wb if s in n0]
        dmed = float(np.median(dl)); dbest = float(min(wb.values()) - min(list(n0.values()) + list(nf.values())))
        nu = float(abs(np.median(list(n0.values())) - np.median(list(nf.values()))))
        prem = byK[Km]["HF_weight"] < 0.6 and np.median(list(n0.values())) < 1.6 and nu < tauM
        out["instances"][name] = {"K_max": Km, "delta_med_mHa": dmed, "delta_best_mHa": dbest, "nu_W_mHa": nu,
                                  "premise": bool(prem), "dE_NNBFJ": n0, "dE_NNBFJNF": nf, "dE_WIBJ": wb}
        acc.append((prem, dmed, dbest))
    if not acc or any(a is None for a in acc):
        accv = "ACC_INCOMPLETE"
    elif any(p and dm > tauM for p, dm, _ in acc):
        accv = "ACC_FAIL"
    elif all(p and dm <= tauM and db <= tauM for p, dm, db in acc):
        accv = "ACC_OK"
    else:
        accv = "ACC_AMBIG"
    out["accuracy"] = accv
    cb = out["cost_byK"]
    if accv == "ACC_FAIL":
        v = "FAIL"
    elif len(cb) < 2 or accv == "ACC_INCOMPLETE":
        v = "INCOMPLETE"
    else:
        Ks = sorted(cb, key=float); m, m0 = cb[Ks[-1]], cb[Ks[0]]
        if m["M_op"] >= 1 or m["M_wall_CI"][0] > 1:
            v = "FAIL"
        elif accv == "ACC_OK" and m["M_op"] <= 0.5 and m["M_op"] < m0["M_op"] and m["M_wall_CI"][1] <= 0.8:
            v = "PASS"
        else:
            v = "AMBIGUOUS (pre-committed: Part S --basis-series 6,5; PASS iff M_op<=0.5 and M_wall CI upper<=0.8, else KILL)"
    out["verdict"] = v
    return out


def aggregate(path, tau=TAU, smoke=False):
    L, W, S = [], [], []
    for f in sorted(glob.glob(os.path.join(path, "results_*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        if bool(d.get("smoke")) != smoke:
            continue
        b = os.path.basename(f)
        (L if b.startswith("results_L") else W if b.startswith("results_W") else S).append(d)
    out = {"R5.2 (ALE, Part L)": verdict_L(L, tau), "R5.1 (WIBJ, Parts W+S)": verdict_R51(W, S, tau)}
    print(json.dumps(out, indent=1, default=str))
    return out


# ----------------------------------------------------------------------------
# Smoke
# ----------------------------------------------------------------------------
def smoke_checks():
    t0 = time.time()
    ok = True
    rng = np.random.default_rng(5)
    inst = Inst(4, 4.4, 0.25, 1, 6)
    net = make_net(6, 4, "NNBFJ", 6, 4, 0)
    th = net.theta0 + 0.2 * rng.standard_normal(net.theta0.size)
    # (1) log-derivatives vs finite differences
    sidx = rng.choice(inst.D, 8, replace=False)
    psi, O = logderiv(net, th, inst.occ[sidx], inst.oi[sidx])
    d = rng.standard_normal(th.size) * 1e-6
    pp, _ = net.amplitudes(th + d, inst.occ[sidx], inst.oi[sidx])
    pm, _ = net.amplitudes(th - d, inst.occ[sidx], inst.oi[sidx])
    fd = (np.log(np.abs(pp)) - np.log(np.abs(pm))) / 2
    e1 = float(np.max(np.abs(O @ d - fd)) / np.max(np.abs(fd)))
    # (2) VJP == sum of per-sample log-derivatives weighted
    cp = rng.standard_normal(8)
    e2 = float(np.max(np.abs(vjp(net, th, inst.occ[sidx], inst.oi[sidx], cp) - cp @ O)))
    print(f"  logderiv vs FD rel err={e1:.1e}; VJP vs per-sample O err={e2:.1e}")
    ok &= e1 < 1e-5 and e2 < 1e-10
    # (3) ALE0 / ALE1 unbiasedness on fixed samples at a shifted theta, and
    #     the first-order identity sum_j JVP_j == g~(n).dtheta
    psiA, _ = net.amplitudes(th, inst.occ, inst.oi)
    s3 = rng.choice(inst.D, 4, p=psiA ** 2 / (psiA @ psiA))
    anc0 = Anchor(inst, net, th, s3, psiA, False)
    anc1 = Anchor(inst, net, th, s3, psiA, True)
    dth = 0.02 * rng.standard_normal(th.size)
    th2 = th + dth
    psiB, _ = net.amplitudes(th2, inst.occ, inst.oi)
    ELB = (inst.H @ psiB)[s3] / psiB[s3]
    z = {}
    for name, anc, fo in (("ALE0", anc0, False), ("ALE1", anc1, True)):
        vals = np.array([ale_estimate(inst, net, anc, th2, psiB, 4, rng, fo) for _ in range(3000)])
        se = vals.std(0, ddof=1) / np.sqrt(len(vals))
        z[name] = float(np.max(np.abs(vals.mean(0) - ELB) / se))
        z[name + "_sd"] = float(vals.std(0).mean())
    # first-order identity
    n = s3[0]; j, h = inst.row(n)
    u = dth / np.linalg.norm(dth); fe = 1e-5
    allc = np.concatenate([[n], j])
    pp, _ = net.amplitudes(th + fe * u, inst.occ[allc], inst.oi[allc])
    pm, _ = net.amplitudes(th - fe * u, inst.occ[allc], inst.oi[allc])
    jv = np.sum(h * (pp[1:] / pp[0] - pm[1:] / pm[0]) / (2 * fe)) * np.linalg.norm(dth)
    e3 = abs(jv - anc1.G[0] @ dth) / max(abs(jv), 1e-12)
    print(f"  ALE unbiasedness (3000 draws, 4 rows, |dtheta|={np.linalg.norm(dth):.2f}): max z ALE0={z['ALE0']:.2f} "
          f"ALE1={z['ALE1']:.2f}; per-draw sd ALE0={z['ALE0_sd']:.2e} ALE1={z['ALE1_sd']:.2e}; "
          f"sum_j JVP_j vs g~.dtheta rel err={e3:.1e}")
    ok &= z["ALE0"] < 4.5 and z["ALE1"] < 4.5 and e3 < 1e-5 and z["ALE1_sd"] < z["ALE0_sd"]
    print(f"  ({time.time()-t0:.1f}s)")
    return bool(ok)


def verdict_selftest():
    """Synthetic trajectories: verdict_L must return PASS when the route reaches
    the control's full-budget energy at half budget, and FAIL when it is worse
    at equal cost."""
    def run(level_half, level_full):
        c = np.linspace(0, 1.0, 101)
        y = np.where(c <= 0.5, level_half, level_full)
        return {"ops": list(c), "cal": list(c), "dE_mHa": list(y)}
    out = {}
    for name, (rh, rf) in {"pass": (0.9, 0.8), "fail": (1.3, 1.3)}.items():
        arms = {"EXACT": [run(1.2, 1.0 + 0.01 * s) for s in range(3)],
                "EXACTNF": [run(1.2, 1.0 + 0.01 * s) for s in range(3)],
                "SCREEN:1e-2": [run(1.3, 1.1) for _ in range(3)],
                "ALE0:4:4": [run(rh, rf) for _ in range(3)]}
        Ls = [{"arm": a, "instances": [{"K": 16, "budget_ops": 1.0, "budget_cal": 1.0, "runs": r}]}
              for a, r in arms.items()]
        out[name] = verdict_L(Ls)["verdict"]
    return out


def staleness_precheck(inst, net, thw, B=200, Kd=4, rng=None, steps=16, lr=0.05):
    """Variance increment of ALE0/ALE1 (over Var_pi(E_L)) as a function of the
    anchor age, along an EXACT-SR trajectory from the warm start."""
    rng = rng or np.random.default_rng(3)
    psi, _ = net.amplitudes(thw, inst.occ, inst.oi)
    wv = psi ** 2 / (psi @ psi)
    s = rng.choice(inst.D, B, p=wv)
    a0 = Anchor(inst, net, thw, s, psi, False); a1 = Anchor(inst, net, thw, s, psi, True)
    th = thw.copy(); out = []
    for t in range(1, steps + 1):
        sidx = rng.choice(inst.D, B, p=wv)
        e = (inst.H @ psi)[sidx] / psi[sidx]
        _, O = logderiv(net, th, inst.occ[sidx], inst.oi[sidx])
        th = th + sr_update(O, np.full(B, 1 / B), e, lr, 1e-3, 0.05)
        psi, _ = net.amplitudes(th, inst.occ, inst.oi)
        wv = psi ** 2 / (psi @ psi)
        EL = (inst.H @ psi) / psi
        varpi = float(wv @ (EL - wv @ EL) ** 2)
        r = []
        for anc, fo in ((a0, False), (a1, True)):
            reps = np.array([ale_estimate(inst, net, anc, th, psi, Kd, rng, fo) for _ in range(20)])
            r.append(float(reps.var(0, ddof=1).mean() / varpi))
        out.append((t, float(np.linalg.norm(th - thw)), r[0], r[1]))
    return out


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        aggregate(args.aggregate)
        return
    if args.smoke:
        ok = smoke_checks()
        vs = verdict_selftest()
        print(f"  verdict_L self-test on synthetic trajectories: {vs} (must be PASS / FAIL)")
        ok &= vs == {"pass": "PASS", "fail": "FAIL"}
        print(f"SMOKE CHECKS {'PASS' if ok else 'FAIL'}", flush=True)
        sd = os.path.join(HERE, "smoke_results"); os.makedirs(sd, exist_ok=True)
        # tiny Part L: H4 K=8, every arm type
        arms = ["EXACT", "EXACTNF", "SCREEN:1e-2", "SCREEN:1e-2+B", "SS:1e-2:4", "SS:1e-2:4+B",
                "ALE0:8:4", "ALE0:8:4+B", "ALE1:8:4"]
        for a in arms:
            la = parse_args(["--part", "L", "--n-active", "8", "--arm", a, "--hidden", "8", "--hj", "4",
                             "--warm-iters", "150", "--batch", "100", "--n-ref", "30", "--max-steps-factor", "4",
                             "--n-wall", "6", "--seeds", "0", "1", "2",
                             "--out", os.path.join(sd, f"results_L_smoke_{a.replace(':', '-')}.json")])
            r = run_part_L(la); r["smoke"] = True
            with open(la.out, "w") as fh:
                json.dump(r, fh, default=float)
        inst = Inst(4, 4.4, 0.25, 1, 8)
        net = make_net(8, 4, "NNBFJ", 8, 4, 0)
        thw = minimize(lambda t: net.energy_grad(t, inst.H, inst.occ, inst.oi), net.theta0, jac=True,
                       method="L-BFGS-B", options={"maxiter": 150}).x
        print("  staleness pre-check (age, |dtheta|, inc ALE0/Var_pi, inc ALE1/Var_pi), Kd=4:")
        for t, nd, r0, r1 in staleness_precheck(inst, net, thw):
            if t in (1, 2, 4, 8, 16):
                print(f"    age={t:2d} |dth|={nd:.3f} ALE0={r0:.3e} ALE1={r1:.3e}")
        print("  R5.3 certified screening bound at the warm state (mean per sample, mHa):")
        for e, v in cert_bound(inst, net, thw, [1e-3, 1e-2], np.random.default_rng(1), 50).items():
            print(f"    eps={e}: certified={v['certified_mHa']:.3e} actual |dropped|={v['actual_abs_mHa']:.3e}")
        # tiny Parts W and S
        wa = parse_args(["--part", "W", "--n-active", "6", "7", "--hidden", "8", "--hj", "4", "--maxiter", "150",
                         "--n-samples", "200", "--eps-screen", "1e-4", "1e-3", "1e-2", "--k-grid", "1", "8",
                         "--out", os.path.join(sd, "results_W_smoke.json")])
        rw = run_part_W(wa); rw["smoke"] = True
        with open(wa.out, "w") as fh:
            json.dump(rw, fh, default=float)
        sa = parse_args(["--part", "S", "--n-atoms", "4", "--basis-series", "2,1", "3,2", "--hidden", "8", "--hj", "4",
                         "--n-samples", "30", "--n-wall", "4", "--eps-screen", "1e-4", "1e-3", "1e-2", "--k-grid", "1", "8",
                         "--out", os.path.join(sd, "results_S_smoke.json")])
        rs = run_part_S(sa); rs["smoke"] = True
        with open(sa.out, "w") as fh:
            json.dump(rs, fh, default=float)
        print("[smoke-scale verdict logic, H4 only, not decisive]")
        aggregate(sd, smoke=True)
        return
    {"L": run_part_L, "W": run_part_W, "S": run_part_S}[args.part](args)


if __name__ == "__main__":
    main()
