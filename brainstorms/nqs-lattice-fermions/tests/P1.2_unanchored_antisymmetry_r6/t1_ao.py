#!/usr/bin/env python3
"""
P1.2 round 6 critic, tier T1: shared premise of the amplitude-off-loading (AO) family (routes AO-F and
AO-L). Does a SIGN-INERT factor driven by exact Hamiltonian-neighbour ratios lower the sign error of a
co-trained NNBF more than a plain neural Jastrow or extra backflow width at the same parameter count?
=====================================================================================================

Why unprojected toys: every PROJECTED toy up to ~1e4 states is NO-TARGET (round-3 pilot 410216, round-5
smoke), but the unprojected NNBF on the 2-leg ladders keeps a sign error (UHF anchor dEs 0.04-0.11 in
t1_vote.py). Unprojected, a positive factor cannot change sgn psi at all (AO lemma), so ANY sign gain
here comes from mechanism 2 of AO ("the factor frees backflow capacity for the nodes"), which is the
only AO mechanism that is not a finite-size projection effect (critic objection 4). The exact ratios
(aoK) carry more information than the frozen ratios of AO-F and are exactly AO-L's feature, so a kill
here kills both routes.

Toys (t = 1, t' = -0.25, J = 0.4, open x open 2-leg ladders, 2 holes; same code as t1_vote.py):
  size 0 : 4x2 3+3 D = 560 (V = 8);  size 1 : 5x2 4+4 D = 3150 (V = 10);  size 2 : 6x2 5+5 D = 16632 (V = 12)
Arms (UHF start, unprojected, exact-sum SR of the round-5 harness, p21_harness.optimize proj="none"):
  nnbfW  NNBF width Hc (control by width; Hc = base width + 2)         -> parameter target T
  nnbfJ  NNBF width Hb=2 x exp(MLP Jastrow), LRAnsatz Ha chosen so n_params <= T (largest such Ha)
  aoK    NNBF width Hb=2 x exp(sum_e alpha_e(n) g(R_e(n))), R_e = c_e Phi(m_e)/Phi(n) EXACT ratios on the
         hole-hop frame K, g(R) = R/sqrt(1+R^2); alpha net width chosen so n_params <= T (largest)
Metric: dEs(X) = min_{a>=0} <a|S H S|a>/<a|a> - E_0 with S = diag(sgn psi_X) (t1_vote.dE_sign), median
over seeds 11 and 22 (the shift 1e-3 run). r_J = dEs(aoK)/dEs(nnbfJ), r_W = dEs(aoK)/dEs(nnbfW),
r* = max(r_J, r_W). Also reported: dE_tot, and r_JW = dEs(nnbfJ)/dEs(nnbfW) (does ANY positive factor
beat width).

PRE-REGISTERED RULES (written 2026-11-16 before the first run):
  premise per size: counted iff med dEs(nnbfW) >= 0.005 (else that size is NO-TARGET).
  KILL (AO-F and AO-L): r* >= 0.9 at >= 2 counted sizes including the largest counted size.
  PASS: r* <= 0.7 at every counted size AND r*(largest counted) <= r*(smallest counted) + 0.1.
  UNDECIDED otherwise; NO-TARGET if fewer than 2 sizes are counted.
  Validity: the AOK analytic force must match finite differences (rel < 1e-5) on size 0, and the
  jastrow-free aoK at alpha = 0 must reproduce the base amplitudes exactly; else INVALID.

Usage:  OMP_NUM_THREADS=4 python3 t1_ao.py [--sizes 0,1,2] [--iters 300] [--seeds 11,22]
Options:
  --sizes LIST   subset of {0,1,2} (default all; the rule needs all three).
  --iters INT    SR iterations per arm (default 300).
  --seeds LIST   comma-separated seeds (default 11,22).
  --out PATH     JSON output (default results/t1_ao.json).
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(os.path.dirname(HERE), "P1.2_unanchored_antisymmetry_r5")
sys.path.insert(0, R5)
sys.path.insert(0, HERE)
from p11_core import solve_ed, uhf_orbitals, energy  # noqa: E402
from tj_core import TJ, LRAnsatz  # noqa: E402
import p21_harness as H21  # noqa: E402
import micro_test as R5T  # noqa: E402
from t1_vote import dE_sign, SIZES  # noqa: E402

HB = 2


class AOK(R5T.FRD):
    """psi = Phi * prod_e exp(alpha_e g(R_e)): FRD-K with every factor made positive (sign-inert)."""

    def _fac_idx(self, idx, phi):
        f = super()._fac_idx(idx, phi)
        f["F"] = np.exp(f["al"] * f["g"])
        return f

    def _coefs(self, f, w):
        P = np.prod(f["F"], 1)
        psi = f["v"] * P
        al = f["al"]
        cn = w * P * (1.0 - np.sum(al * f["q"], 1))
        cm = np.where(f["act"], (w * psi)[:, None] * al * f["C"] * f["dg_du"], 0.0)
        dal = (w * psi)[:, None] * f["g"]
        return cn, cm, dal


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.2 r6 critic T1: AO-family premise (see docstring)")
    ap.add_argument("--sizes", default="0,1,2")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seeds", default="11,22")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "t1_ao.json"))
    a = ap.parse_args(argv)
    a.size_list = [int(s) for s in a.sizes.split(",")]
    a.seed_list = [int(s) for s in a.seeds.split(",")]
    return a


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make(arm, m, ref, seed, T):
    if arm == "nnbfW":
        return LRAnsatz(m, ref, "full", K=1, Hs=HB + 2, seed=seed)
    if arm == "nnbfJ":
        best = None
        for ha in range(1, 400):
            if LRAnsatz(m, ref, "full", K=1, Hs=HB, Ha=ha, seed=seed).n_params() <= T:
                best = ha
            else:
                break
        return LRAnsatz(m, ref, "full", K=1, Hs=HB, Ha=best, seed=seed)
    best = None
    for w in range(1, 400):
        if AOK(m, LRAnsatz(m, ref, "full", K=1, Hs=HB, seed=seed), "K", w, seed).n_params() <= T:
            best = w
        else:
            break
    a = AOK(m, LRAnsatz(m, ref, "full", K=1, Hs=HB, seed=seed), "K", best, seed)
    a.tables(); a.set_flat(a.flat())
    return a


def checks(m, ref):
    a = make("aoK", m, ref, 3, make("nnbfW", m, ref, 3, 0).n_params())
    x = a.flat().copy()
    x[a.base.n_params():] += 0.05 * np.random.default_rng(1).standard_normal(len(x) - a.base.n_params())
    a.set_flat(x)
    psi = a.amplitudes(65536)
    r = np.random.default_rng(2).standard_normal(m.D)
    f = a.force(r, 65536)
    worst = 0.0
    for k in np.random.default_rng(4).choice(len(x), 6, replace=False):
        e = np.zeros(len(x)); e[k] = 1e-6
        a.set_flat(x + e); p1 = a.amplitudes(65536)
        a.set_flat(x - e); p2 = a.amplitudes(65536)
        fd = r @ (p1 - p2) / 2e-6
        worst = max(worst, abs(fd - f[k]) / max(abs(fd), 1e-8))
    a.set_flat(x)
    q = a.q.copy()
    a.q["W2a"] = np.zeros_like(a.q["W2a"]); a.q["b2a"] = np.zeros_like(a.q["b2a"])
    a._phi = None
    d0 = float(np.max(np.abs(a.amplitudes(65536) - a.base.amplitudes(65536))))
    a.q = q
    return worst, d0, bool(np.all(np.isfinite(psi)))


def run_size(k, args):
    Lx, Ly, nu, nd = SIZES[k]
    m = TJ(Lx, Ly, "obc", "obc", nu, nd, 1.0, -0.25, 0.4)
    W, Vv, deg = solve_ed(m, 2, 1e-8, H21.UHF_SEED)
    psi0, E0 = Vv[:, 0], float(W[0])
    ref = uhf_orbitals(m, H21.UHF_SEED)
    T = make("nnbfW", m, ref, 0, 0).n_params()
    res = dict(size=k, D=m.D, V=m.V, E0=E0, T=T, arms={})
    if k == min(args.size_list):
        fdw, d0, fin = checks(m, ref)
        res["check_force_rel"], res["check_alpha0"], res["check_finite"] = fdw, d0, fin
        log(f"size {k} checks: aoK force vs fd rel {fdw:.1e}; alpha=0 vs base {d0:.1e}; finite {fin}")
    sra = SimpleNamespace(sr_iters=args.iters, sr_lr=0.05, sr_shift=1e-3, sr_samples=600, chunk=65536,
                          o_chunk=4)
    for arm in ("nnbfW", "nnbfJ", "aoK"):
        rows = []
        for seed in args.seed_list:
            t0 = time.time()
            a = make(arm, m, ref, seed, T)
            H21.optimize(a, m, None, "none", 0, sra, np.random.default_rng(seed + 5), f"s{k} {arm} {seed}")
            psi = a.amplitudes(65536)
            Et = energy(m.H, psi)[0]
            s = np.where(psi >= 0, 1.0, -1.0)
            if (psi0 @ psi) < 0:
                s = -s
            des = dE_sign(m.H, s, psi0, E0, 3000)
            rows.append(dict(seed=seed, n_params=a.n_params(), dEt=Et - E0, dEs=des, sec=time.time() - t0))
            log(f"size {k} {arm} seed {seed}: n_params {a.n_params()} (T {T}) dE_tot {Et - E0:.5f} "
                f"dE_sign {des:.5f} ({time.time() - t0:.0f} s)")
        res["arms"][arm] = rows
    return res


def verdict(R):
    med = lambda r, arm, key: float(np.median([x[key] for x in r["arms"][arm]]))
    rows, counted = [], []
    for r in R:
        dW, dJ, dA = med(r, "nnbfW", "dEs"), med(r, "nnbfJ", "dEs"), med(r, "aoK", "dEs")
        rs = max(dA / dJ, dA / dW) if min(dJ, dW) > 0 else float("inf")
        row = dict(size=r["size"], V=r["V"], dEs_W=dW, dEs_J=dJ, dEs_aoK=dA, r_J=dA / dJ if dJ > 0 else None,
                   r_W=dA / dW if dW > 0 else None, r_star=rs, r_JW=dJ / dW if dW > 0 else None,
                   counted=dW >= 0.005)
        rows.append(row)
        if row["counted"]:
            counted.append(row)
    valid = all(r.get("check_force_rel", 0) < 1e-5 and r.get("check_alpha0", 0) < 1e-12 for r in R)
    if not valid:
        v = "INVALID"
    elif len(counted) < 2:
        v = "NO-TARGET"
    else:
        big = counted[-1]
        nkill = sum(c["r_star"] >= 0.9 for c in counted)
        if nkill >= 2 and big["r_star"] >= 0.9:
            v = "KILL"
        elif all(c["r_star"] <= 0.7 for c in counted) and big["r_star"] <= counted[0]["r_star"] + 0.1:
            v = "PASS"
        else:
            v = "UNDECIDED"
    return dict(rows=rows, valid=valid, verdict=v)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    R = [run_size(k, args) for k in args.size_list]
    V = verdict(R)
    for row in V["rows"]:
        print("  " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    print(f"VERDICT AO-family premise (AO-F, AO-L): {V['verdict']} (valid {V['valid']})")
    print(f"{time.time() - t0:.1f} s wall")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(results=R, verdict=V), open(args.out, "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
