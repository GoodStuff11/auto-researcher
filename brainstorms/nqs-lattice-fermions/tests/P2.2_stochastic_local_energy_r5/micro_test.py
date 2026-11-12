#!/usr/bin/env python3
"""
P2.2 round-5 decisive test (critic-fix revision, 2026-11-12).  Route R5.1
(WIBJ, regime F_A) in two parts; route R5.2 (ALE, Part L) is DEFERRED.

Saturation statement this test is built on (see the leaf artifact, Round 5)
--------------------------------------------------------------------------
Black-box query model: each connected term x_j = H_nn' psi(n')/psi(n) is
revealed only by an oracle call psi(n').  (a) An EXACT local energy needs
Omega(|C|) oracle calls whenever no x_j can be inferred from the others.
(b) An estimator that is unbiased over its own randomness and queries term j
with inclusion probability pi_j has variance >= sum_j s_j^2 (1/pi_j - 1) when
x_j is known to the estimator only up to s_j (the inclusion-probability /
Horvitz-Thompson bound).  Architectures that expose algebraic structure can
beat |C|: either each term becomes cheap arithmetic ("cheaper per term",
F_A1, which is what KEYX does for WIBJ), or whole classes of terms are summed
by contraction without enumerating them ("sub-|C| by contraction", F_A2: the
round-4 R4.0 Wick/Cholesky contraction for Slater-Jastrow with a factorizable
Jastrow; THC/Cholesky factorized sums over external indices).  This test
measures F_A1 only.

Parts
-----
W  (route R5.1 accuracy, ED, L-BFGS on the exact energy).  Instances:
   H4 distorted chain R=4.4, K = 12 16 20 24 (the K-trend), H6 R=3.6,
   K = 12 14.  Arms (one per task):
     NNBFJ    general NNBF x neural/pair Jastrow, width --hidden (seeds 0 1 2)
     NNBFJNF  the same, seeds 3 4 5 (noise floor)
     WIBJ     window-input backflow x all-orbital Jastrow (seeds 0 1 2)
     NNBFJh4  minimal-contrast foil: NNBFJ with hidden 4, hJ 4 (per-term
              cost of the order of WIBJ's keyed per-term cost)
     NNBFJh8  the same with hidden 8, hJ 8
     CASJ     minimal-contrast foil: window-only multi-determinant x pair
              Jastrow, no neural network (CAS(N, K_W) with trainable orbital
              rows, all C(K_W, N/2)^2 window determinants)
   NNBFJ also gets the exact SCREEN/TRUNC biases and the sampled SS
   increments (controls), and every arm reports per-sample operation counts
   on |psi|^2 samples of its own TRAINED state.
S  (route R5.1 cost, ED-free).  Distorted H8 chain R=2.0, s+p bases with
   K = 40 72 104 136.  --s-arms main = NNBFJ + WIBJ (timed in the same task),
   --s-arms foils = NNBFJh4 + NNBFJh8 + CASJ (operation counts).  Samples come
   from a CISD-like proposal (HF + MP1-weighted singles/doubles); no trained
   state exists at these sizes (stated limitation).  Wall model (the Part L
   calibration): generation is timed per row and fitted t = ag + bg m;
   ratio_exact is timed on prefixes m = 1 .. |C| and fitted t = a + b m
   (per-call overhead a).  EXACT/KEYX/TRUNC are charged generation of all
   |C| terms, SCREEN/SS only of kept terms (generous: HCI-sorted integrals),
   and an SS draw call is charged a + b Kd.  WIBJ+SCREEN (keyed on kept terms)
   is reported against NNBFJ+SCREEN.
L  (route R5.2, anchored local energy) DEFERRED: not in the array.  Its
   decisive regime (cold start, step-limited optimization) needs a separate
   micro-pilot.  The code is kept for that pilot.

Pre-committed verdict for R5.1 (python3 micro_test.py --aggregate DIR), tau = 0.2 mHa
------------------------------------------------------------------------------------
Accuracy, per W instance at its largest K:
  delta_med  = median over seeds 0-2 of (dE_WIBJ - dE_NNBFJ), paired by seed;
  delta_best = min over the 3 WIBJ seeds - min over the 6 NNBFJ seeds
               (0-5; asymmetric, against WIBJ);
  nu_W       = max(|median NNBFJ(0-2) - median NNBFJ(3-5)|, half the range
               over the six NNBFJ seeds)   (the Part L noise-floor definition);
  premise: HF weight < 0.6, median NNBFJ dE < 1.6 mHa, nu_W < tau.
  ACC_OK if delta_med <= tau and delta_best <= tau on every instance (all
  premises hold); ACC_FAIL if delta_med > tau on a premise-valid instance;
  ACC_AMBIG otherwise.
Trend (every instance with >= 3 K values, i.e. H4 K = 12..24): TREND_FAIL if
  delta_med(K) rises strictly monotonically AND its least-squares linear fit
  in K, extrapolated to K = 40 (the smallest Part S K), exceeds tau.
Foils: FOIL_FAIL if some foil F in {NNBFJh4, NNBFJh8, CASJ} is within the
  noise floor of WIBJ on every instance (median dE_F <= median dE_WIBJ +
  nu_W at the largest K) AND is no more expensive in Part S at K_max:
  C_F = min(ops EXACT_F, ops SCREEN_F(eps*)) x v_F <= C_WIBJ = min(ops KEYX,
  ops KEYX on SCREEN(eps*) kept terms) x v_WIBJ.
Cost (Part S, variance-ratio corrected; v_X = max over instances of the
  median Var_X / Var_NNBFJ; controls on NNBFJ: EXACT, SCREEN(eps*) with eps*
  the largest eps whose exact bias <= tau on every W instance, SS charged its
  measured increment, TRUNC(smallest admissible B_U)):
  M_op(K) = ops_KEYX(WIBJ) v_WIBJ / min control ops;
  M_wall(K) = mean route wall / min over controls of mean control wall, both
  with the wall model above, paired bootstrap 95% CI.
PASS: ACC_OK, no TREND_FAIL, no FOIL_FAIL, M_op(K_max) <= 0.5,
      M_op(K_max) < M_op(K_min), and the upper CI bound of M_wall(K_max) <= 0.8.
FAIL: ACC_FAIL, TREND_FAIL, FOIL_FAIL, M_op(K_max) >= 1, or the lower CI bound
      of M_wall(K_max) > 1.
AMBIGUOUS otherwise -> pre-committed: Part S with --basis-series 6,5
      (K = 168); PASS iff M_op <= 0.5 and M_wall CI upper <= 0.8, else KILL.
Hard Core: sign half only (same-sector competitors not examined).

Command-line options
--------------------
--part {W,S,L}          Required unless --smoke or --aggregate (L: deferred).
--n-atoms INT           H atoms = electrons (even, >= 4). Default 4 (L, W), 8 (S).
--bond-length FLOAT     Mean H-H spacing in bohr. Default 4.4 (L, W), 2.0 (S).
--distort FLOAT         Random displacement amplitude (bohr). Default 0.25.
--geom-seed INT         Displacement seed. Default 1.
--n-active INT [...]    L, W: active spatial MOs K. Default 16 (L), 12 16 20 24 (W).
--basis-series LIST     S: "ns,np" pairs (keyx_core). Default "2,1 3,2 4,3 5,4".
--w-arms LIST           W: subset of NNBFJ NNBFJNF WIBJ NNBFJh4 NNBFJh8 CASJ.
                        Default all six.
--s-arms {main,foils,all} S: arm group. Default all.
--arm STR               L only (deferred), as before.
--seeds INT [...]       W: network seeds (NNBFJNF adds 3). Default 0 1 2.
--hidden INT            Backflow hidden width. Default 64 (W, S), 32 (L).
--hj INT                Neural-Jastrow width. Default 32 (W, S), 16 (L).
--maxiter INT           W: L-BFGS iterations. Default 6000.
--n-wall INT            S: samples with wall-clock timing. Default 60 (>= 50 for
                        a decisive run); L: 40.
--n-samples INT         W/S samples per state. Default 2000 (W), 200 (S).
--eps-screen FLOAT...   W/S grids. Default 1e-5 1e-4 3e-4 1e-3 1.5e-3 2e-3
                        2.5e-3 3e-3 1e-2 3e-2.
--k-grid INT...         W/S SS draws. Default 1 4 16 64 256.
--bu-grid INT...        W TRUNC batch sizes. Default 1000 4000 16000.
--tau FLOAT             Ha. Default 2e-4.
--out PATH              Result JSON (default: results_<part>_<tag>.json here).
--smoke                 Correctness checks (log-derivatives, VJP, ALE identities,
                        CASJ gradient and per-sample kernel), tiny Parts W and S
                        with every arm, the smoke-scale aggregate, and the
                        verdict self-test (PASS / FAIL / TREND_FAIL / FOIL_FAIL).
--aggregate DIR         Read results_{W,S,L}_*.json in DIR, print verdicts.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
from itertools import combinations
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
    ap = argparse.ArgumentParser(description="P2.2 round-5 test (WIBJ; ALE deferred)")
    ap.add_argument("--part", choices=["W", "S", "L"], default=None)
    ap.add_argument("--n-atoms", type=int, default=None)
    ap.add_argument("--bond-length", type=float, default=None)
    ap.add_argument("--distort", type=float, default=0.25)
    ap.add_argument("--geom-seed", type=int, default=1)
    ap.add_argument("--n-active", type=int, nargs="+", default=None)
    ap.add_argument("--basis-series", nargs="+", default=["2,1", "3,2", "4,3", "5,4"])
    ap.add_argument("--arm", type=str, default="EXACT")
    ap.add_argument("--w-arms", nargs="+", default=["NNBFJ", "NNBFJNF", "WIBJ", "NNBFJh4", "NNBFJh8", "CASJ"],
                    choices=["NNBFJ", "NNBFJNF", "WIBJ", "NNBFJh4", "NNBFJh8", "CASJ"])
    ap.add_argument("--s-arms", choices=["main", "foils", "all"], default="all")
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
    ap.add_argument("--eps-screen", type=float, nargs="+",
                    default=[1e-5, 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--k-grid", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    ap.add_argument("--bu-grid", type=int, nargs="+", default=[1000, 4000, 16000])
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", type=str, default=None)
    a = ap.parse_args(argv)
    if a.aggregate or a.smoke:
        return a
    if a.part is None:
        ap.error("--part W, S (or the deferred L) is required")
    d = {"L": (4, 4.4, [16], 32, 16, 40, None), "W": (4, 4.4, [12, 16, 20, 24], 64, 32, None, 2000),
         "S": (8, 2.0, None, 64, 32, 60, 200)}[a.part]
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
    if min(a.eps_screen) <= 0 or min(a.k_grid) < 1 or min(a.bu_grid) < 1:
        ap.error("--eps-screen > 0, --k-grid >= 1, --bu-grid >= 1")
    if a.out is None:
        tag = f"N{a.n_atoms}_R{a.bond_length}_" + ("K" + "-".join(map(str, a.n_active)) if a.n_active else "S")
        tag += "_" + (a.arm.replace(":", "-") if a.part == "L" else "-".join(a.w_arms) if a.part == "W" else a.s_arms)
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
# CASJ foil: window-only multi-determinant x pair Jastrow (no neural network)
# ----------------------------------------------------------------------------
def cofactor(A):
    """d det(A) / dA for a stack (..., n, n); closed form for n <= 3."""
    n = A.shape[-1]
    if n == 1:
        return np.ones_like(A)
    if n == 2:
        C = np.empty_like(A)
        C[..., 0, 0] = A[..., 1, 1]; C[..., 0, 1] = -A[..., 1, 0]
        C[..., 1, 0] = -A[..., 0, 1]; C[..., 1, 1] = A[..., 0, 0]
        return C
    if n == 3:
        r0, r1, r2 = A[..., 0, :], A[..., 1, :], A[..., 2, :]
        return np.stack([np.cross(r1, r2), np.cross(r2, r0), np.cross(r0, r1)], -2)
    return np.linalg.det(A)[..., None, None] * np.swapaxes(np.linalg.inv(A), -1, -2)


class CASJ:
    """psi(n) = J(n) * sum_{I,I'} C[I,I'] det Ua[occ_a(n), S_I] det Ub[occ_b(n), S_I'].
    S_I runs over the C(K_W, N/2) subsets of the K_W window spatial MOs (the CAS
    determinants); Ua, Ub (K x K_W) are trainable orbital rows, so external
    occupations enter through orbital mixing (CAS with orbital optimization);
    ln J = u.n + sum_{p<q} v_pq n_p n_q over all 2K spin orbitals (the WIBJ pair
    Jastrow, no neural Jastrow).  K_W = N, as for WIBJ."""

    arm = "CASJ"

    def __init__(self, K, N, rng, KW=None, scale=0.01):
        self.K, self.M, self.N = K, 2 * K, N
        nh = N // 2
        KW = min(N if KW is None else KW, K)
        self.nh, self.KW = nh, KW
        self.subs = np.array(list(combinations(range(KW), nh)))
        self.ns = ns = len(self.subs)
        self.iu = np.triu_indices(self.M, 1)
        self.shapes = [("Ua", (K, KW)), ("Ub", (K, KW)), ("C", (ns, ns)), ("u", (self.M,)),
                       ("v", (len(self.iu[0]),))]
        U0 = np.eye(K)[:, :KW]
        C0 = scale * rng.standard_normal((ns, ns)); C0[0, 0] = 1.0     # subset 0 = HF
        init = {"Ua": U0 + scale * rng.standard_normal((K, KW)), "Ub": U0 + scale * rng.standard_normal((K, KW)),
                "C": C0, "u": np.zeros(self.M), "v": np.zeros(len(self.iu[0]))}
        self.theta0 = np.concatenate([init[k].ravel() for k, _ in self.shapes])
        self.E = np.zeros((ns, nh, KW))
        self.E[np.arange(ns)[:, None], np.arange(nh)[None], self.subs] = 1.0

    def unpack(self, th):
        out, i = {}, 0
        for k, s in self.shapes:
            n = int(np.prod(s)); out[k] = th[i:i + n].reshape(s); i += n
        V = np.zeros((self.M, self.M)); V[self.iu] = out["v"]; out["V"] = V + V.T
        return out

    def _minors(self, U, rows):
        """rows (B, nh) spatial MO indices -> A (B, ns, nh, nh), dets (B, ns)."""
        A = U[rows][:, :, self.subs].transpose(0, 2, 1, 3)
        return A, np.linalg.det(A)

    def amplitudes(self, th, occ, occ_idx):
        P = self.unpack(th)
        nh, K = self.nh, self.K
        Aa, ma = self._minors(P["Ua"], occ_idx[:, :nh])
        Ab, mb = self._minors(P["Ub"], occ_idx[:, nh:] - K)
        Phi = np.einsum("bi,ij,bj->b", ma, P["C"], mb)
        o = occ.astype(float)
        J = np.exp(o @ P["u"] + 0.5 * np.einsum("bp,bp->b", o @ P["V"], o))
        return Phi * J, (P, Aa, ma, Ab, mb, J, o)

    def energy_grad(self, th, H, occ, occ_idx):
        psi, (P, Aa, ma, Ab, mb, J, o) = self.amplitudes(th, occ, occ_idx)
        nrm = psi @ psi
        Hpsi = H @ psi
        E = psi @ Hpsi / nrm
        c = 2 * (Hpsi - E * psi) / nrm
        cp = c * psi
        w = c * J
        nh, K, B = self.nh, self.K, len(psi)
        g = {"C": (ma * w[:, None]).T @ mb}
        for key, A, k, rows in (("Ua", Aa, w[:, None] * (mb @ P["C"].T), occ_idx[:, :nh]),
                                ("Ub", Ab, w[:, None] * (ma @ P["C"]), occ_idx[:, nh:] - K)):
            gX = np.einsum("bi,birc,icw->brw", k, cofactor(A), self.E, optimize=True)
            S = sp.csr_matrix((np.ones(B * nh), (rows.ravel(), np.arange(B * nh))), shape=(K, B * nh))
            g[key] = S @ gX.reshape(B * nh, self.KW)
        g["u"] = o.T @ cp
        g["v"] = (o.T @ (o * cp[:, None]))[self.iu]
        return E, np.concatenate([g[k].ravel() for k, _ in self.shapes])

    # ---- per-sample kernel (Part S) and its generous incremental op count ----
    def setup(self, P, occ_n):
        nh, K = self.nh, self.K
        _, ma = self._minors(P["Ua"], occ_n[None, :nh])
        _, mb = self._minors(P["Ub"], occ_n[None, nh:] - K)
        o = np.zeros(self.M); o[occ_n] = 1
        return {"occ": occ_n, "Phi": float(ma[0] @ P["C"] @ mb[0]), "fp": P["V"] @ o}

    def ratio(self, P, st, rem, add, occ_j, chunk=1024):
        """Exact psi(n')/psi(n) by direct minor evaluation (vectorized)."""
        out = np.empty(len(rem))
        for c0 in range(0, len(rem), chunk):
            sl = slice(c0, c0 + chunk)
            _, ma = self._minors(P["Ua"], occ_j[sl, :self.nh])
            _, mb = self._minors(P["Ub"], occ_j[sl, self.nh:] - self.K)
            Phi = np.einsum("bi,ij,bj->b", ma, P["C"], mb)
            out[sl] = Phi / st["Phi"] * np.exp(kc.Ansatz.jastrow_pair(self, P, st, rem[sl], add[sl]))
        return out

    def ops_terms(self, rem):
        """Incremental cost per term (generous to the foil): a replaced row in one
        spin costs ns (nh r + r^3) for the ns window minors of that spin (r = rows
        replaced, Sherman-Morrison / 2x2 minor updates from stored inverses); the
        contraction costs ns (one spin changed) or ns^2 (both); + pair Jastrow +
        generation."""
        K, ns, nh = self.K, self.ns, self.nh
        ra = ((rem >= 0) & (rem < K)).sum(1); rb = (rem >= K).sum(1)
        c = np.zeros(len(rem))
        for r in (ra, rb):
            c += np.where(r > 0, ns * (nh * r + r ** 3), 0)
        c += np.where((ra > 0) & (rb > 0), ns * ns, ns) + kc.C_PAIRJ + kc.C_GEN
        return c

    def setup_ops(self):
        """Minors and inverses of both spins, the two contracted vectors, and the
        Green's-function rows of every orbital for every minor."""
        return 2 * self.ns * self.nh ** 3 + 2 * self.ns ** 2 + 2 * self.K * self.ns * self.nh ** 2 + self.M


# ----------------------------------------------------------------------------
# Part W: accuracy, all arms, one task per (instance, arm)
# ----------------------------------------------------------------------------
W_ARMS = {"NNBFJ": ("NNBFJ", None, None, 0), "NNBFJNF": ("NNBFJ", None, None, 3), "WIBJ": ("WIBJ", None, None, 0),
          "NNBFJh4": ("NNBFJ", 4, 4, 0), "NNBFJh8": ("NNBFJ", 8, 8, 0), "CASJ": ("CASJ", None, None, 0)}
FOILS = ("NNBFJh4", "NNBFJh8", "CASJ")


def build_net(K, N, w_arm, hidden, hj, seed):
    kind, hh, hjj, off = W_ARMS[w_arm]
    s = seed + off
    if kind == "CASJ":
        return CASJ(K, N, np.random.default_rng(1000 * s + 7)), s
    return make_net(K, N, kind, hh or hidden, hjj or hj, s), s


def run_part_W(args):
    tag = f"H{args.n_atoms}_R{args.bond_length}_d{args.distort}_g{args.geom_seed}"
    res = {"part": "W", "instance": tag, "w_arms": list(args.w_arms),
           "args": {k: v for k, v in vars(args).items()}, "instances": []}
    ka = A_()
    ka.eps_screen, ka.k_grid, ka.bu_grid = args.eps_screen, args.k_grid, args.bu_grid
    N = args.n_atoms
    for K in args.n_active:
        t0 = time.time()
        h, g, enuc, Erhf, eps = kc.build_molecule(N, args.bond_length, args.distort, args.geom_seed, kc.H631GSS, K,
                                                  "valence-p")
        ham = kc.SpatialHam(h, g, enuc, K, N)
        codes, occ, oi = kc.build_space(K, N // 2, N // 2)
        H = kc.build_hamiltonian(ham, codes, oi)
        w2, v2 = kc.ground_states(H, 2)
        E0, psi0 = w2[0], v2[:, 0]
        hfw = float(psi0[np.searchsorted(codes, kc.hf_code(K, N))] ** 2)
        print(f"\n=== [W] {tag} K={K} dim={len(codes)} E_FCI={E0:.6f} HF wt={hfw:.3f} nnz/row={H.nnz/len(codes):.0f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if hfw >= 0.6:
            raise SystemExit(f"instance not multireference: HF weight {hfw:.3f} >= 0.6")
        it = {"K": K, "N": N, "dim": len(codes), "E_FCI": float(E0), "HF_weight": hfw, "gap": float(w2[1] - E0),
              "runs": []}
        Hoff = H - sp.diags(H.diagonal())
        for wa in args.w_arms:
            for seed in args.seeds:
                net, s = build_net(K, N, wa, args.hidden, args.hj, seed)
                t1 = time.time()
                r = minimize(lambda t: net.energy_grad(t, H, occ, oi), net.theta0, jac=True, method="L-BFGS-B",
                             options={"maxiter": args.maxiter, "maxcor": 30, "gtol": 1e-12, "ftol": 1e-16})
                th = r.x
                psi, _ = net.amplitudes(th, occ, oi)
                nrm = psi @ psi
                wv = psi ** 2 / nrm
                Hpsi = H @ psi
                Ev = float(psi @ Hpsi / nrm)
                EL = Hpsi / psi
                run = {"seed": s, "arm": wa, "dE_mHa": 1e3 * (Ev - E0), "var_exact": float(wv @ (EL - Ev) ** 2),
                       "P": int(net.theta0.size), "t_train_s": time.time() - t1}
                srng = np.random.default_rng(10 ** 6 + 1000 * s + 13)
                sidx = srng.choice(len(codes), size=args.n_samples, p=wv)
                P = net.unpack(th)
                # per-sample operation counts on |psi|^2 samples of THIS trained state
                nS = min(len(sidx), 200)
                opsE, opsK, opsKs, opsS, dev = [], [], {str(e): [] for e in args.eps_screen}, \
                    {str(e): [] for e in args.eps_screen}, []
                for k in sidx[:nS]:
                    row = ham.row(codes[k], oi[k])
                    hab = np.abs(row["h"])
                    if wa == "CASJ":
                        so = net.setup_ops(); ct = net.ops_terms(row["rem"])
                        opsE.append(ct.sum() + so)
                        for e in args.eps_screen:
                            opsS[str(e)].append(ct[hab > e].sum() + so)
                        continue
                    amp = kc.amp_ops(net); so = kc.setup_ops(net)
                    opsE.append(len(row["h"]) * (kc.C_GEN + amp) + so)
                    for e in args.eps_screen:
                        opsS[str(e)].append(int((hab > e).sum()) * (kc.C_GEN + amp) + so)
                    if wa == "WIBJ":
                        st = net.setup(P, oi[k])
                        rk, cnt = net.ratio_key(P, st, row["rem"], row["add"])
                        dev.append(abs(row["Hnn"] + row["h"] @ rk - EL[k]))
                        opsK.append(kc.keyx_ops(net, cnt) + kc.C_GEN * len(row["h"]) + so)
                        for e in args.eps_screen:
                            mk = hab > e
                            if mk.any():
                                _, ce = net.ratio_key(P, st, row["rem"][mk], row["add"][mk])
                                opsKs[str(e)].append(kc.keyx_ops(net, ce) + kc.C_GEN * int(mk.sum()) + so)
                            else:
                                opsKs[str(e)].append(so)
                run["ops_ED"] = {"EXACT": float(np.mean(opsE)), "SCREEN": {e: float(np.mean(v)) for e, v in opsS.items()}}
                if wa == "WIBJ":
                    run["ops_ED"].update({"KEYX": float(np.mean(opsK)),
                                          "KEYX_SCREEN": {e: float(np.mean(v)) for e, v in opsKs.items()}})
                    run["KEYX_max_dev"] = float(max(dev))
                if wa == "NNBFJ":
                    run["controls"] = kc.control_block_A(ka, H, Hoff, psi, EL, codes, oi, ham, net, P, sidx, srng)
                it["runs"].append(run)
                msg = (f"  {wa:8s} seed={s} dE={run['dE_mHa']:8.3f} mHa Var(EL)={run['var_exact']:.2e} P={run['P']} "
                       f"(train {run['t_train_s']:.0f}s)")
                if wa == "WIBJ":
                    msg += (f" KEYX-EL max={run['KEYX_max_dev']:.1e} ops KEYX/EXACT(trained |psi|^2)="
                            f"{run['ops_ED']['KEYX']/run['ops_ED']['EXACT']:.3f}")
                print(msg, flush=True)
                if wa == "NNBFJ":
                    cb = run["controls"]
                    print("      SCREEN exact bias (mHa): " + " ".join(f"{e}:{1e3*b:.4f}" for e, b in cb["screen_bias"].items()))
                    print(f"      SS selected eps={cb['ss_sel']['eps']:g} Kd={cb['ss_sel']['K']} inc ratio (half B)="
                          f"{cb['ss_sel']['inc_ratio_B']:.3f}", flush=True)
        res["instances"].append(it)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    return res


# ----------------------------------------------------------------------------
# Part S: ED-free cost scaling with the calibrated wall model
# ----------------------------------------------------------------------------
S_GROUPS = {"main": ["NNBFJ", "WIBJ"], "foils": list(FOILS), "all": ["NNBFJ", "WIBJ"] + list(FOILS)}


def run_part_S_basis(ns, npp, args, arms):
    t0 = time.time()
    N = args.n_atoms
    h, g, enuc, Erhf, eps = kc.build_molecule(N, args.bond_length, args.distort, args.geom_seed,
                                              kc.make_basis(ns, npp), None, "energy")
    K = h.shape[0]
    ham = kc.SpatialHam(h, g, enuc, K, N)
    print(f"\n=== [S] H{N} distorted chain R={args.bond_length} basis(ns,np)=({ns},{npp}) K={K} E_RHF={Erhf:.6f} "
          f"(build {time.time()-t0:.0f}s)", flush=True)
    samples = kc.cisd_sampler(ham, eps, K, N, args.n_samples, np.random.default_rng(12345))
    U = np.unique([ham.code_of(o) for o in samples])
    rows, tgen = [], []
    for t, occ_n in enumerate(samples):
        s0 = time.perf_counter()
        rows.append((occ_n, ham.row(ham.code_of(occ_n), occ_n)))
        tgen.append(time.perf_counter() - s0)
    nconn = np.array([len(r[1]["h"]) for r in rows])
    kept = {str(e): np.array([(np.abs(r[1]["h"]) > e).sum() for r in rows]) for e in args.eps_screen}
    bg, ag = np.polyfit(nconn, tgen, 1) if len(set(nconn)) > 1 else (np.mean(tgen) / np.mean(nconn), 0.0)
    nW = min(args.n_wall, len(rows))
    out = {"K": K, "M": 2 * K, "N": N, "ns": ns, "np": npp, "mean_conn": float(nconn.mean()), "n_wall": nW,
           "conn": nconn.tolist(), "kept": {e: v.tolist() for e, v in kept.items()},
           "gen": {"t": tgen, "ag": float(max(ag, 0)), "bg": float(bg)}, "arms": {}}
    print(f"  <|C|>={nconn.mean():.0f}  gen fit: {max(ag,0)*1e6:.0f}us + {bg*1e6:.3f}us/term; kept fraction: "
          + " ".join(f"{e}:{v.mean()/nconn.mean():.3f}" for e, v in kept.items()), flush=True)
    for arm in arms:
        net, _ = build_net(K, N, arm, args.hidden, args.hj, 0)
        P = net.unpack(net.theta0 + 0.02 * np.random.default_rng(5).standard_normal(net.theta0.size))
        a = {"wall_EXACT": [], "wall_SCREEN": {str(e): [] for e in args.eps_screen}}
        if arm == "CASJ":
            so = net.setup_ops()
            ops_all, ops_S = [], {str(e): [] for e in args.eps_screen}
            for t, (occ_n, row) in enumerate(rows):
                ct = net.ops_terms(row["rem"]); hab = np.abs(row["h"])
                ops_all.append(ct.sum() + so)
                for e in args.eps_screen:
                    ops_S[str(e)].append(ct[hab > e].sum() + so)
                if t < min(nW, 5):   # the direct (non-incremental) kernel is slow; timing reported only
                    st = net.setup(P, occ_n)
                    occ_j = kc.occ_from_row(occ_n, row["rem"], row["add"])
                    s0 = time.perf_counter()
                    _ = row["Hnn"] + row["h"] @ net.ratio(P, st, row["rem"], row["add"], occ_j)
                    a["wall_EXACT"].append(time.perf_counter() - s0)
            a.update({"ops_EXACT": float(np.mean(ops_all)),
                      "ops_SCREEN": {e: float(np.mean(v)) for e, v in ops_S.items()}, "ns_dets": net.ns ** 2})
            out["arms"][arm] = a
            print(f"  {arm:8s} ops(all terms)/term={a['ops_EXACT']/nconn.mean():.0f} dets={net.ns**2}", flush=True)
            continue
        amp = kc.amp_ops(net); so = kc.setup_ops(net)
        a.update({"ops_amp": amp, "ops_setup": so,
                  "ops_EXACT": float(np.mean(nconn * (kc.C_GEN + amp) + so)),
                  "ops_SCREEN": {e: float(np.mean(v * (kc.C_GEN + amp) + so)) for e, v in kept.items()},
                  "ops_TRUNC": float(nconn.mean() * (kc.C_GEN + kc.C_HASH) + amp + so)})
        if arm == "WIBJ":
            opsK, wK, dev = [], [], []
            opsKs = {str(e): [] for e in args.eps_screen}; wKs = {str(e): [] for e in args.eps_screen}
        fitm, fitt = [], []
        wT = []
        for t, (occ_n, row) in enumerate(rows):
            st = net.setup(P, occ_n)
            hab = np.abs(row["h"])
            if arm == "WIBJ":
                s0 = time.perf_counter()
                rk, cnt = net.ratio_key(P, st, row["rem"], row["add"])
                ek = row["Hnn"] + row["h"] @ rk
                wK.append(time.perf_counter() - s0)
                opsK.append(kc.keyx_ops(net, cnt) + kc.C_GEN * len(row["h"]) + so)
                for e in args.eps_screen:
                    mk = hab > e
                    s0 = time.perf_counter()
                    if mk.any():
                        rke, ce = net.ratio_key(P, st, row["rem"][mk], row["add"][mk])
                        _ = row["Hnn"] + row["h"][mk] @ rke
                        opsKs[str(e)].append(kc.keyx_ops(net, ce) + kc.C_GEN * int(mk.sum()) + so)
                    else:
                        opsKs[str(e)].append(so)
                    wKs[str(e)].append(time.perf_counter() - s0)
            if t >= nW or (arm in FOILS and t >= min(nW, 10)):
                continue
            occ_j = kc.occ_from_row(occ_n, row["rem"], row["add"])
            s0 = time.perf_counter()
            rx = net.ratio_exact(P, st, row["rem"], row["add"], occ_j)
            ex = row["Hnn"] + row["h"] @ rx
            a["wall_EXACT"].append(time.perf_counter() - s0)
            if arm == "WIBJ":
                dev.append(abs(ex - ek) / max(1.0, abs(ex)))
            if arm == "NNBFJ":
                for e in args.eps_screen:
                    mk = hab > e
                    s0 = time.perf_counter()
                    _ = row["Hnn"] + row["h"][mk] @ net.ratio_exact(P, st, row["rem"][mk], row["add"][mk], occ_j[mk])
                    a["wall_SCREEN"][str(e)].append(time.perf_counter() - s0)
                for m_ in sorted({1, 4, 16, 64, 256, 1024, len(row["h"])}):
                    m_ = min(m_, len(row["h"]))
                    s0 = time.perf_counter()
                    _ = row["h"][:m_] @ net.ratio_exact(P, st, row["rem"][:m_], row["add"][:m_], occ_j[:m_])
                    fitm.append(m_); fitt.append(time.perf_counter() - s0)
                s0 = time.perf_counter()
                _ = row["Hnn"] + row["h"][np.isin(row["code"], U)].sum()
                wT.append(time.perf_counter() - s0)
        if arm == "NNBFJ":
            b, a0 = np.polyfit(fitm, fitt, 1)
            a.update({"cal": {"a": float(max(a0, 0)), "b": float(b)}, "wall_TRUNC": wT})
        if arm == "WIBJ":
            a.update({"ops_KEYX": float(np.mean(opsK)), "wall_KEYX": wK,
                      "ops_KEYX_SCREEN": {e: float(np.mean(v)) for e, v in opsKs.items()}, "wall_KEYX_SCREEN": wKs,
                      "KEYX_max_rel_dev": float(max(dev)) if dev else None})
        out["arms"][arm] = a
        msg = f"  {arm:8s} ops/term(EXACT)={a['ops_EXACT']/nconn.mean():.0f}"
        if arm == "WIBJ":
            msg += (f" ops KEYX/EXACT={a['ops_KEYX']/a['ops_EXACT']:.4f} wall KEYX/EXACT="
                    f"{np.mean(wK[:len(a['wall_EXACT'])])/max(np.mean(a['wall_EXACT']),1e-12):.4f} "
                    f"KEYX-EXACT rel dev={a['KEYX_max_rel_dev']:.1e}")
        if arm == "NNBFJ":
            msg += f" ratio_exact fit: a={a['cal']['a']*1e6:.0f}us b={a['cal']['b']*1e6:.2f}us/amp"
        print(msg, flush=True)
    return out


def run_part_S(args):
    a = kc_basis(args)
    arms = S_GROUPS[args.s_arms]
    res = {"part": "S", "r5": True, "s_arms": args.s_arms, "args": {k: v for k, v in vars(args).items()},
           "instances": []}
    for ns, npp in a:
        res["instances"].append(run_part_S_basis(ns, npp, args, arms))
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    return res


def kc_basis(args):
    return [tuple(int(v) for v in s.split(",")) for s in args.basis_series]


# ----------------------------------------------------------------------------
# Verdicts
# ----------------------------------------------------------------------------
def verdict_L(Ls, tau=TAU):
    """Part L verdict (kept for the deferred R5.2 micro-pilot)."""
    tauM = 1e3 * tau
    byK = {}
    for d in Ls:
        for it in d["instances"]:
            e = byK.setdefault(it["K"], {"arms": {}, "budget_ops": it["budget_ops"], "budget_cal": it["budget_cal"]})
            e["arms"][d["arm"]] = it["runs"]
    out = {"tau_mHa": tauM, "byK": {}}
    if not byK:
        out["verdict"] = "UNTESTED (deferred: needs a cold-start, step-limited micro-pilot)"
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
                        "route_star": rstar, "route_dE_half_mHa": Rh, "route_dE_full_mHa": Rf}
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
        out["verdict"] = "AMBIGUOUS"
    return out


def noise_floor(a, b):
    """Part L definition: max(|median a - median b|, half the range of a + b)."""
    pool = np.array(list(a) + list(b))
    return float(max(abs(np.median(list(a)) - np.median(list(b))), 0.5 * np.ptp(pool)))


def trend_rule(Ks, dmed, tauM, K_ext=40):
    """TREND_FAIL iff delta_med rises strictly monotonically in K and its
    least-squares line extrapolated to K_ext exceeds tau."""
    o = np.argsort(Ks); K_ = np.array(Ks, float)[o]; d_ = np.array(dmed)[o]
    mono = bool(np.all(np.diff(d_) > 0))
    slope, icpt = np.polyfit(K_, d_, 1)
    ext = float(slope * K_ext + icpt)
    return {"K": K_.tolist(), "delta_med": d_.tolist(), "monotone_rise": mono, "slope_mHa_per_K": float(slope),
            "delta_ext_K40": ext, "trend_fail": bool(mono and ext > tauM)}


def verdict_R51(Ws, Ss, tau=TAU):
    tauM = 1e3 * tau
    out = {"tau_mHa": tauM, "instances": {}, "trend": {}, "foils": {}}
    # ---- merge W runs per (instance, K)
    inst = {}
    for d in Ws:
        for it in d["instances"]:
            e = inst.setdefault(d["instance"], {}).setdefault(it["K"], {"HF_weight": it["HF_weight"], "runs": []})
            e["runs"].extend(it["runs"])

    def arm_runs(runs, arm):
        return {r["seed"]: r for r in runs if r["arm"] == arm}

    acc, var_ratio, ctrl = [], {}, {"screen": None, "trunc": None, "ss": {}}
    for name, byK in inst.items():
        per_K = {}
        for K, e in sorted(byK.items()):
            n0, nf, wb = arm_runs(e["runs"], "NNBFJ"), arm_runs(e["runs"], "NNBFJNF"), arm_runs(e["runs"], "WIBJ")
            if not (n0 and nf and wb):
                per_K[K] = None
                continue
            dn0 = {s: r["dE_mHa"] for s, r in n0.items()}
            dl = [wb[s]["dE_mHa"] - dn0[s] for s in wb if s in dn0]
            per_K[K] = {"delta_med_mHa": float(np.median(dl)),
                        "delta_best_mHa": float(min(r["dE_mHa"] for r in wb.values())
                                                - min([r["dE_mHa"] for r in n0.values()] + [r["dE_mHa"] for r in nf.values()])),
                        "nu_W_mHa": noise_floor(dn0.values(), [r["dE_mHa"] for r in nf.values()]),
                        "median_dE_mHa": {a: float(np.median([r["dE_mHa"] for r in arm_runs(e["runs"], a).values()]))
                                          for a in W_ARMS if arm_runs(e["runs"], a)},
                        "dE_mHa": {a: {s: r["dE_mHa"] for s, r in arm_runs(e["runs"], a).items()}
                                   for a in W_ARMS if arm_runs(e["runs"], a)}}
        Km = max(byK)
        top = per_K.get(Km)
        out["instances"][name] = {"K_max": Km, "byK": per_K}
        if top is None:
            acc.append(None)
            continue
        e = byK[Km]
        n0 = arm_runs(e["runs"], "NNBFJ")
        prem = bool(e["HF_weight"] < 0.6 and top["median_dE_mHa"]["NNBFJ"] < 1.6 and top["nu_W_mHa"] < tauM)
        out["instances"][name]["premise"] = prem
        acc.append((prem, top["delta_med_mHa"], top["delta_best_mHa"]))
        # variance ratios (paired by seed) and NNBFJ controls at the largest K
        for a in ("WIBJ",) + FOILS:
            ar = arm_runs(e["runs"], a)
            vr = [ar[s]["var_exact"] / n0[s]["var_exact"] for s in ar if s in n0]
            if vr:
                var_ratio[a] = max(var_ratio.get(a, 0.0), float(np.median(vr)))
        sb, tb, ssi = {}, {}, {}
        for r in n0.values():
            c = r["controls"]
            for k_, b in c["screen_bias"].items():
                sb.setdefault(k_, []).append(abs(b))
            for k_, (m, se) in c["trunc_bias"].items():
                tb.setdefault(k_, []).append(abs(m) + 2 * se)
            for k_, gg in c["ss_grid"].items():
                ssi.setdefault(k_, []).append(gg["inc_ratio_B"] if (gg["tail_ok_A"] and gg["tail_ok_B"]) else float("inf"))
        sok = {k_ for k_, v in sb.items() if max(v) <= tau}
        tok = {k_ for k_, v in tb.items() if max(v) <= tau}
        ctrl["screen"] = sok if ctrl["screen"] is None else ctrl["screen"] & sok
        ctrl["trunc"] = tok if ctrl["trunc"] is None else ctrl["trunc"] & tok
        for k_, v in ssi.items():
            ctrl["ss"][k_] = max(ctrl["ss"].get(k_, 0.0), float(np.median(v)))
        # cross-check on trained |psi|^2 samples (reported only)
        wb = arm_runs(e["runs"], "WIBJ")
        out["instances"][name]["ops_KEYX_over_EXACT_trained"] = float(np.median(
            [r["ops_ED"]["KEYX"] / r["ops_ED"]["EXACT"] for r in wb.values()]))
        # trend
        ks = [K for K, v in per_K.items() if v is not None]
        if len(ks) >= 3:
            out["trend"][name] = {**trend_rule(ks, [per_K[K]["delta_med_mHa"] for K in ks], tauM),
                                  "WIBJ_median_dE_mHa": [per_K[K]["median_dE_mHa"]["WIBJ"] for K in sorted(ks)]}
    eps_star = max(ctrl["screen"], key=float) if ctrl["screen"] else None
    eps_star = None if eps_star is None else str(float(eps_star))   # keys are str(float) in W and S
    bu_star = min(ctrl["trunc"], key=int) if ctrl["trunc"] else None
    out["controls"] = {"SCREEN_eps_star": eps_star, "TRUNC_BU_star": bu_star, "SS_inc_ratio": ctrl["ss"],
                       "var_ratio": var_ratio}
    if not acc or any(a is None for a in acc):
        accv = "ACC_INCOMPLETE"
    elif any(p and dm > tauM for p, dm, _ in acc):
        accv = "ACC_FAIL"
    elif all(p and dm <= tauM and db <= tauM for p, dm, db in acc):
        accv = "ACC_OK"
    else:
        accv = "ACC_AMBIG"
    out["accuracy"] = accv
    trend_fail = any(t["trend_fail"] for t in out["trend"].values())
    out["trend_fail"] = trend_fail
    # ---- Part S: merge arms per K
    SK = {}
    for d in Ss:
        for it in d["instances"]:
            e = SK.setdefault(it["K"], {k: v for k, v in it.items() if k != "arms"} | {"arms": {}})
            e["arms"].update(it["arms"])
            if "NNBFJ" in it["arms"]:
                e["gen"] = it["gen"]; e["kept"] = it["kept"]; e["conn"] = it["conn"]
    vW = var_ratio.get("WIBJ", 1.0)
    cost = {}
    for K, it in sorted(SK.items()):
        A = it["arms"]
        if "NNBFJ" not in A or "WIBJ" not in A:
            continue
        Cn, R = A["NNBFJ"], A["WIBJ"]
        nW = len(Cn["wall_EXACT"])
        tg = np.array(it["gen"]["t"][:nW])
        gm = lambda m: it["gen"]["ag"] + it["gen"]["bg"] * np.asarray(m)  # noqa: E731
        cal = Cn["cal"]
        ops = {"EXACT": Cn["ops_EXACT"]}
        wall = {"EXACT": np.array(Cn["wall_EXACT"]) + tg}
        if eps_star is not None:
            ops["SCREEN"] = Cn["ops_SCREEN"][eps_star]
            wall["SCREEN"] = np.array(Cn["wall_SCREEN"][eps_star]) + gm(it["kept"][eps_star][:nW])
        best_ss = None
        for k_, inc in ctrl["ss"].items():
            e_, Kd = k_.split(",")
            ek = next((x for x in Cn["ops_SCREEN"] if float(x) == float(e_)), None)
            if ek is None or not np.isfinite(inc):
                continue
            c_ = (Cn["ops_SCREEN"][ek] + int(Kd) * Cn["ops_amp"]) * (1 + inc)
            if best_ss is None or c_ < best_ss[0]:
                w_ = (np.array(Cn["wall_SCREEN"][ek]) + gm(it["kept"][ek][:nW]) + cal["a"] + cal["b"] * int(Kd)) * (1 + inc)
                best_ss = (c_, w_, k_)
        if best_ss:
            ops["SS"] = best_ss[0]; wall["SS"] = best_ss[1]
        if bu_star is not None:
            ops["TRUNC"] = Cn["ops_TRUNC"]; wall["TRUNC"] = np.array(Cn["wall_TRUNC"]) + tg
        cop = min(ops, key=ops.get)
        cwl = min(wall, key=lambda k_: wall[k_].mean())
        rw = (np.array(R["wall_KEYX"][:nW]) + tg) * vW
        c = {"M_op": float(R["ops_KEYX"] * vW / ops[cop]), "ctrl_op": cop,
             "M_wall": float(rw.mean() / wall[cwl].mean()), "ctrl_wall": cwl, "M_wall_CI": kc.boot_ratio_ci(rw, wall[cwl]),
             "ops_KEYX_over_EXACT": R["ops_KEYX"] / R["ops_EXACT"], "SS_config": best_ss[2] if best_ss else None}
        if eps_star is not None:
            rws = (np.array(R["wall_KEYX_SCREEN"][eps_star][:nW]) + gm(it["kept"][eps_star][:nW])) * vW
            c["WIBJ+SCREEN_over_NNBFJ+SCREEN"] = {"M_op": float(R["ops_KEYX_SCREEN"][eps_star] * vW / Cn["ops_SCREEN"][eps_star]),
                                                  "M_wall": float(rws.mean() / wall["SCREEN"].mean())}
        CW = min(R["ops_KEYX"], R["ops_KEYX_SCREEN"][eps_star] if eps_star else np.inf) * vW
        c["C_WIBJ_best_ops"] = float(CW)
        for f in FOILS:
            if f in A:
                Cf = min(A[f]["ops_EXACT"], A[f]["ops_SCREEN"][eps_star] if eps_star else np.inf) * var_ratio.get(f, 1.0)
                c[f"C_{f}_over_C_WIBJ"] = float(Cf / CW)
        cost[K] = c
    out["cost_byK"] = cost
    # ---- foils
    foil_fail = False
    Kmax_S = max(cost) if cost else None
    for f in FOILS:
        dom = []
        for name, d in out["instances"].items():
            top = d["byK"].get(d["K_max"])
            if top is None or f not in top["median_dE_mHa"]:
                dom = None
                break
            dom.append(top["median_dE_mHa"][f] <= top["median_dE_mHa"]["WIBJ"] + top["nu_W_mHa"])
        cr = cost[Kmax_S].get(f"C_{f}_over_C_WIBJ") if Kmax_S is not None else None
        if dom is None or cr is None:
            out["foils"][f] = {"status": "INCOMPLETE"}
            continue
        ff = bool(all(dom) and cr <= 1.0)
        out["foils"][f] = {"accuracy_within_nu_of_WIBJ_everywhere": bool(all(dom)), "cost_over_WIBJ_Kmax": cr,
                           "foil_fail": ff}
        foil_fail |= ff
    foil_incomplete = any(v.get("status") == "INCOMPLETE" for v in out["foils"].values())
    out["foil_fail"] = foil_fail
    if accv == "ACC_FAIL" or trend_fail or foil_fail:
        v = "FAIL"
    elif len(cost) < 2 or accv == "ACC_INCOMPLETE" or foil_incomplete:
        v = "INCOMPLETE"
    else:
        Ks = sorted(cost); m, m0 = cost[Ks[-1]], cost[Ks[0]]
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
    out = {"R5.1 (WIBJ, Parts W+S)": verdict_R51(W, S, tau), "R5.2 (ALE, Part L)": verdict_L(L, tau)}
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
    # (1) log-derivatives vs finite differences; (2) VJP (kept for the deferred R5.2 pilot)
    sidx = rng.choice(inst.D, 8, replace=False)
    psi, O = logderiv(net, th, inst.occ[sidx], inst.oi[sidx])
    d = rng.standard_normal(th.size) * 1e-6
    pp, _ = net.amplitudes(th + d, inst.occ[sidx], inst.oi[sidx])
    pm, _ = net.amplitudes(th - d, inst.occ[sidx], inst.oi[sidx])
    fd = (np.log(np.abs(pp)) - np.log(np.abs(pm))) / 2
    e1 = float(np.max(np.abs(O @ d - fd)) / np.max(np.abs(fd)))
    cp = rng.standard_normal(8)
    e2 = float(np.max(np.abs(vjp(net, th, inst.occ[sidx], inst.oi[sidx], cp) - cp @ O)))
    print(f"  logderiv vs FD rel err={e1:.1e}; VJP vs per-sample O err={e2:.1e}")
    ok &= e1 < 1e-5 and e2 < 1e-10
    # (3) CASJ: cofactor, gradient vs FD, per-sample kernel vs full-vector ratios, ops model
    A = rng.standard_normal((5, 3, 3)); dA = np.zeros_like(A); dA[:, 1, 2] = 1e-6
    ec = float(np.max(np.abs((np.linalg.det(A + dA) - np.linalg.det(A - dA)) / 2e-6 - cofactor(A)[:, 1, 2])))
    for N_, K_ in ((4, 6), (6, 7)):
        ins = Inst(N_, 4.4 if N_ == 4 else 3.6, 0.25, 1, K_, allow_weak=True)
        cj = CASJ(K_, N_, np.random.default_rng(3))
        thc = cj.theta0 + 0.3 * rng.standard_normal(cj.theta0.size)
        _, ga = cj.energy_grad(thc, ins.H, ins.occ, ins.oi)
        idx = rng.choice(thc.size, 12, replace=False); gn = []
        for i in idx:
            dd = np.zeros_like(thc); dd[i] = 1e-6
            gn.append((cj.energy_grad(thc + dd, ins.H, ins.occ, ins.oi)[0] - cj.energy_grad(thc - dd, ins.H, ins.occ, ins.oi)[0]) / 2e-6)
        gerr = float(np.max(np.abs(ga[idx] - gn)) / max(np.max(np.abs(gn)), 1e-12))
        P = cj.unpack(thc)
        psiC, _ = cj.amplitudes(thc, ins.occ, ins.oi)
        rerr = 0.0
        for n in (3, 40, 101):
            row = ins.ham.row(ins.codes[n], ins.oi[n]); j = np.searchsorted(ins.codes, row["code"])
            st = cj.setup(P, ins.oi[n])
            occ_j = kc.decode(row["code"], ins.ham.M, N_)
            rerr = max(rerr, float(np.max(np.abs(cj.ratio(P, st, row["rem"], row["add"], occ_j) - psiC[j] / psiC[n]))))
        print(f"  CASJ N={N_} K={K_}: dets={cj.ns**2} P={cj.theta0.size} grad rel err={gerr:.1e} "
              f"per-sample ratio err={rerr:.1e}; cofactor err={ec:.1e}")
        ok &= gerr < 1e-5 and rerr < 1e-9 and ec < 1e-6
    print(f"  ({time.time()-t0:.1f}s)")
    return bool(ok)


def _syn_W(case):
    """Synthetic Part W results for the verdict self-test."""
    Ws = []
    for name, Ks in (("H4", [12, 16, 20, 24]), ("H6", [12, 14])):
        its = []
        for K in Ks:
            runs = []
            base = 0.5
            wib = {"pass": 0.55, "fail": 0.95, "trend": 0.4 + 0.012 * (K - 12) if name == "H4" else 0.5,
                   "foil": 0.55}[case]
            for arm in W_ARMS:
                for s in (0, 1, 2):
                    seed = s + (3 if arm == "NNBFJNF" else 0)
                    dE = {"NNBFJ": base, "NNBFJNF": base + 0.01, "WIBJ": wib,
                          "NNBFJh4": 0.55 if case == "foil" else 3.0, "NNBFJh8": 2.0, "CASJ": 5.0}[arm]
                    dE += 0.005 * s
                    r = {"seed": seed, "arm": arm, "dE_mHa": dE, "var_exact": 1e-3}
                    if arm == "NNBFJ":
                        r["controls"] = {"screen_bias": {"0.001": 1e-5, "0.01": 1e-3}, "trunc_bias": {"1000": [1e-3, 1e-4]},
                                         "ss_grid": {"0.001,4": {"inc_ratio_B": 0.2, "tail_ok_A": True, "tail_ok_B": True}}}
                    if arm == "WIBJ":
                        r["ops_ED"] = {"KEYX": 1.0, "EXACT": 5.0}
                    runs.append(r)
            its.append({"K": K, "HF_weight": 0.5, "runs": runs})
        Ws.append({"part": "W", "instance": name, "instances": its})
    return Ws


def _syn_S(case, Mop):
    Ss = []
    for i, K in enumerate((40, 72)):
        nW = 20
        rng = np.random.default_rng(K)
        tE = 1.0 + 0.1 * rng.random(nW)
        m = Mop[i]
        arms = {"NNBFJ": {"ops_EXACT": 100.0, "ops_SCREEN": {"0.001": 80.0, "0.01": 40.0}, "ops_amp": 1.0, "ops_TRUNC": 500.0,
                          "wall_EXACT": tE.tolist(), "wall_SCREEN": {"0.001": (0.8 * tE).tolist(), "0.01": (0.4 * tE).tolist()},
                          "wall_TRUNC": (5 * tE).tolist(), "cal": {"a": 1e-4, "b": 1e-6}},
                "WIBJ": {"ops_EXACT": 100.0, "ops_KEYX": 80.0 * m, "ops_KEYX_SCREEN": {"0.001": 80.0 * m, "0.01": 40 * m},
                         "wall_KEYX": (0.8 * m * tE).tolist(), "wall_KEYX_SCREEN": {"0.001": (0.8 * m * tE).tolist(),
                                                                                   "0.01": (0.4 * m * tE).tolist()}},
                "NNBFJh4": {"ops_EXACT": 10.0 if case == "foil" else 300.0, "ops_SCREEN": {"0.001": 10.0 if case == "foil" else 300.0}},
                "NNBFJh8": {"ops_EXACT": 400.0, "ops_SCREEN": {"0.001": 400.0}},
                "CASJ": {"ops_EXACT": 900.0, "ops_SCREEN": {"0.001": 900.0}}}
        Ss.append({"part": "S", "instances": [{"K": K, "arms": arms, "conn": [100] * nW, "kept": {"0.001": [80] * nW, "0.01": [40] * nW},
                                               "gen": {"t": [0.0] * nW, "ag": 0.0, "bg": 0.0}}]})
    return Ss


def verdict_selftest():
    out = {}
    for case, Mop in (("pass", (0.5, 0.3)), ("fail", (0.5, 0.3)), ("trend", (0.5, 0.3)), ("foil", (0.5, 0.3))):
        out[case] = verdict_R51(_syn_W(case), _syn_S(case, Mop))["verdict"]
    return out


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        aggregate(args.aggregate)
        return
    if args.smoke:
        ok = smoke_checks()
        vs = verdict_selftest()
        want = {"pass": "PASS", "fail": "FAIL", "trend": "FAIL", "foil": "FAIL"}
        print(f"  verdict_R51 self-test on synthetic results: {vs} (must be {want})")
        ok &= vs == want
        print(f"SMOKE CHECKS {'PASS' if ok else 'FAIL'}", flush=True)
        sd = os.path.join(HERE, "smoke_results"); os.makedirs(sd, exist_ok=True)
        for f in glob.glob(os.path.join(sd, "results_*.json")):
            os.remove(f)
        wa = parse_args(["--part", "W", "--n-active", "6", "7", "8", "--hidden", "8", "--hj", "4", "--maxiter", "150",
                         "--n-samples", "200", "--eps-screen", "1e-4", "1e-3", "2e-3", "1e-2", "--k-grid", "1", "8",
                         "--bu-grid", "100", "--out", os.path.join(sd, "results_W_smoke.json")])
        rw = run_part_W(wa); rw["smoke"] = True
        with open(wa.out, "w") as fh:
            json.dump(rw, fh, default=float)
        for grp in ("main", "foils"):
            sa = parse_args(["--part", "S", "--n-atoms", "4", "--basis-series", "2,1", "3,2", "--hidden", "8", "--hj", "4",
                             "--n-samples", "30", "--n-wall", "6", "--eps-screen", "1e-4", "1e-3", "2e-3", "1e-2",
                             "--k-grid", "1", "8", "--s-arms", grp, "--out", os.path.join(sd, f"results_S_smoke_{grp}.json")])
            rs = run_part_S(sa); rs["smoke"] = True
            with open(sa.out, "w") as fh:
                json.dump(rs, fh, default=float)
        print("[smoke-scale verdict logic, H4 only, not decisive]")
        aggregate(sd, smoke=True)
        return
    {"L": run_part_L, "W": run_part_W, "S": run_part_S}[args.part](args)


if __name__ == "__main__":
    main()
