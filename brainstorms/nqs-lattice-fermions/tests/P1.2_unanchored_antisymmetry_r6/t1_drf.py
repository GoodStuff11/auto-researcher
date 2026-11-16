#!/usr/bin/env python3
"""
P1.2 round 6 refill, tier T1 for route 7.2 DRF (determinant-ratio-feature backflow): do the reference
determinant's own hop ratios tell a SMALL model where the reference's signs are wrong, beyond what the
occupations alone tell it? (Information premise; three-size Gate 4 trend.)
=====================================================================================================

DRF feeds the first-pass determinant's single-hop ratios R_e(n) = c_e det(O_n[rows(m_e)]) / det(O_n[rows(n)])
(the Green's-function entries; for the UHF start these equal the exact ratios) into a second backflow pass,
so the orbitals can deform where the first pass's nodes are wrong. If the ratios add no information to
the occupations for a model of the size a backflow MLP layer has, DRF has nothing to add.

Toys: the three ladders of t1_vote.py (V = 8, 10, 12). Reference: UHF (t1_vote.uhf_amplitudes). Label
y(n) = [sgn Phi(n) != sgn psi_0(n)] (globally aligned), weight p(n) = psi_0(n)^2. Configurations with
|Phi| < 1e-10 max are dropped (no ratio). Random half split (seed 7) into fit and test.
Models (weighted L2-logistic, scipy L-BFGS, penalty 1e-3 per unit weight):
  A: x = 2n - 1 (2V occupations)
  B: x plus g(R_e) = R_e/sqrt(1+R_e^2) for every edge e of the frame (0 if inactive), same penalty.
     Frames (--frame, sibling routes, the same rules for each): K = hole hops (DRF-K, route 7.2),
     X = spin exchange (DRF-X, route 7.3), H = K u X (DRF-H, route 7.4). For X the ratio is the
     exchange element over J/2 times the neighbour ratio, as in FRD-X (scale does not change the AUC).
  C: (capacity control) x plus the same number of random +-1 features as B adds (seed 9).
Score: weighted AUC on the test half. dAUC = AUC_B - max(AUC_A, AUC_C).

PRE-REGISTERED RULES (written 2026-11-16 before the first run):
  KILL route 7.2 if dAUC < 0.03 at >= 2 of 3 sizes, including V = 12.
  PASS if dAUC >= 0.05 at all three sizes and dAUC(12) >= 0.5 dAUC(8).
  UNDECIDED otherwise.
  Validity: AUC of a model scoring with the true label must be 1 (check of the AUC code); the ratio
  features are built exactly as the K-frame terms of t1_vote.vote.
Usage: OMP_NUM_THREADS=4 python3 t1_drf.py [--frame {K,X,H}] [--out PATH]
  --frame  edge frame of the ratio features (default K); --out default results/t1_drf_<frame>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from t1_vote import SIZES, uhf_amplitudes, frame_tables  # noqa: E402  (sets the r5 path)
from p11_core import solve_ed, uhf_orbitals  # noqa: E402
from tj_core import TJ  # noqa: E402
import p21_harness as H21  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.2 r6 T1 for DRF (see docstring)")
    ap.add_argument("--frame", choices=["K", "X", "H"], default="K")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = os.path.join(HERE, "results", f"t1_drf_{a.frame}.json")
    return a


def wauc(score, y, w):
    o = np.argsort(score, kind="mergesort")
    s, y, w = score[o], y[o], w[o]
    wp, wn = w * y, w * (1 - y)
    # ties: average rank via grouping
    auc, cum_n, i = 0.0, 0.0, 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        p_blk, n_blk = wp[i:j].sum(), wn[i:j].sum()
        auc += p_blk * (cum_n + 0.5 * n_blk)
        cum_n += n_blk
        i = j
    return auc / (wp.sum() * wn.sum())


def fit(X, y, w, lam=1e-3):
    X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
    w = w / w.sum()

    def fg(b):
        z = X1 @ b
        pz = 1 / (1 + np.exp(-z))
        L = -np.sum(w * (y * np.log(pz + 1e-300) + (1 - y) * np.log(1 - pz + 1e-300))) + lam * b[:-1] @ b[:-1]
        g = X1.T @ (w * (pz - y))
        g[:-1] += 2 * lam * b[:-1]
        return L, g
    return minimize(fg, np.zeros(X1.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=2000)).x


def run(k, frame):
    Lx, Ly, nu, nd = SIZES[k]
    m = TJ(Lx, Ly, "obc", "obc", nu, nd, 1.0, -0.25, 0.4)
    W, Vv, _ = solve_ed(m, 2, 1e-8, H21.UHF_SEED)
    psi0 = Vv[:, 0]
    phi = uhf_amplitudes(m, *uhf_orbitals(m, H21.UHF_SEED))
    if phi @ psi0 < 0:
        phi = -phi
    Mi, hc = frame_tables(m, frame)
    good = np.abs(phi) > 1e-10 * np.abs(phi).max()
    act = Mi >= 0
    R = np.where(act & good[:, None], hc * phi[np.maximum(Mi, 0)], 0.0) / np.where(good, phi, 1.0)[:, None]
    gR = R / np.sqrt(1 + R ** 2)
    iu, idd = m.iu_id()
    x = np.concatenate([m.occ_u[iu], m.occ_d[idd]], 1).astype(float) * 2 - 1
    y = (np.sign(phi) != np.sign(psi0)).astype(float)
    p = psi0 ** 2 / (psi0 @ psi0)
    sel = good & (p > 0)
    x, gR, y, p = x[sel], gR[sel], y[sel], p[sel]
    rnd = np.random.default_rng(9).choice([-1.0, 1.0], size=gR.shape)
    tr = np.random.default_rng(7).random(len(y)) < 0.5
    te = ~tr
    out = dict(size=k, V=m.V, D=m.D, w_wrong=float((p * y).sum() / p.sum()), auc_true=wauc(y[te], y[te], p[te]))
    for lab, X in (("A", x), ("B", np.concatenate([x, gR], 1)), ("C", np.concatenate([x, rnd], 1))):
        b = fit(X[tr], y[tr], p[tr])
        sc = np.concatenate([X[te], np.ones((te.sum(), 1))], 1) @ b
        out[f"auc_{lab}"] = wauc(sc, y[te], p[te])
    out["dAUC"] = out["auc_B"] - max(out["auc_A"], out["auc_C"])
    return out


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    rows = []
    for k in (0, 1, 2):
        r = run(k, args.frame)
        rows.append(r)
        print(" ".join(f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}" for a, b in r.items()), flush=True)
    valid = all(abs(r["auc_true"] - 1) < 1e-12 for r in rows)
    d = [r["dAUC"] for r in rows]
    if not valid:
        v = "INVALID"
    elif sum(x < 0.03 for x in d) >= 2 and d[2] < 0.03:
        v = "KILL"
    elif all(x >= 0.05 for x in d) and d[2] >= 0.5 * d[0]:
        v = "PASS"
    else:
        v = "UNDECIDED"
    print(f"VERDICT DRF-{args.frame} information premise: {v} (dAUC {[round(x, 4) for x in d]}) {time.time() - t0:.1f} s")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(rows=rows, verdict=v), open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
