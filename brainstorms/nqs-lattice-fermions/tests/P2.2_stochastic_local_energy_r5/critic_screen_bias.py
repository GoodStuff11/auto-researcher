#!/usr/bin/env python3
"""
P2.2 round-5 per-leaf critic check: is SCREEN admissible for WIBJ?

The decisive job 452238 measured the exact SCREEN energy bias only for NNBFJ
(eps* = 0.002 is the largest eps with |bias| <= tau = 0.2 mHa on every NNBFJ
seed at each instance's largest K).  The "symmetric" cost ratio
WIBJ+SCREEN / NNBFJ+SCREEN assumes SCREEN at eps* is also admissible for
WIBJ, which was never checked, and trained parameters were not saved.

This script retrains one (instance, arm, seed) exactly as Part W does
(same build_molecule, build_net, L-BFGS options, maxiter) and reports
  dE_mHa  (must reproduce the 452238 value: determinism check),
  screen_bias_mHa[eps] = psi^T (H_eps - H_off) psi / psi^T psi  (exact, ED),
the same formula as keyx_core.control_block_A.

Usage
-----
  python3 critic_screen_bias.py --n-atoms N --bond-length R --K K --arm ARM --seed S --out FILE
    --n-atoms INT      4 (H4, R=4.4) or 6 (H6, R=3.6).
    --bond-length F    4.4 for H4, 3.6 for H6 (as in submit_test.sbatch).
    --K INT            active orbitals (H4: 12, 16, 20; H6: 12, 14).
    --arm STR          one of NNBFJ, WIBJ (any key of micro_test.W_ARMS works).
    --seed INT         seed index as in Part W (0, 1, 2); the arm offset is added
                       by build_net exactly as in Part W.
    --maxiter INT      L-BFGS iterations (default 6000, as in Part W).
    --out PATH         JSON output.
  python3 critic_screen_bias.py --aggregate DIR   prints the table from all
                       critic_sb_*.json files in DIR.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keyx_core as kc  # noqa: E402
import micro_test as mt  # noqa: E402

EPS = [1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 1e-2]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-atoms", type=int, default=4)
    ap.add_argument("--bond-length", type=float, default=4.4)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--arm", default="WIBJ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=6000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--aggregate", default=None)
    return ap.parse_args(argv)


def run(a):
    N, K = a.n_atoms, a.K
    h, g, enuc, Erhf, eps = kc.build_molecule(N, a.bond_length, 0.25, 1, kc.H631GSS, K, "valence-p")
    ham = kc.SpatialHam(h, g, enuc, K, N)
    codes, occ, oi = kc.build_space(K, N // 2, N // 2)
    H = kc.build_hamiltonian(ham, codes, oi)
    w2, v2 = kc.ground_states(H, 2)
    E0 = w2[0]
    net, s = mt.build_net(K, N, a.arm, 64, 32, a.seed)
    t0 = time.time()
    r = minimize(lambda t: net.energy_grad(t, H, occ, oi), net.theta0, jac=True, method="L-BFGS-B",
                 options={"maxiter": a.maxiter, "maxcor": 30, "gtol": 1e-12, "ftol": 1e-16})
    psi, _ = net.amplitudes(r.x, occ, oi)
    nrm = psi @ psi
    Ev = float(psi @ (H @ psi) / nrm)
    Hc = (H - sp.diags(H.diagonal())).tocsr()
    sb, kept = {}, {}
    for e in EPS:
        He = Hc.copy(); He.data[np.abs(He.data) <= e] = 0.0
        sb[str(e)] = 1e3 * float(psi @ ((He - Hc) @ psi) / nrm)
        kept[str(e)] = float((np.abs(Hc.data) > e).sum() / Hc.nnz)
    out = {"N": N, "K": K, "arm": a.arm, "seed": s, "dE_mHa": 1e3 * (Ev - E0), "screen_bias_mHa": sb,
           "kept_frac_nnz": kept, "t_s": time.time() - t0}
    print(json.dumps(out), flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


def aggregate(d):
    rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "critic_sb_*.json")))]
    rs.sort(key=lambda r: (r["N"], r["K"], r["arm"], r["seed"]))
    print("H_N K arm seed dE_mHa | SCREEN bias (mHa) at eps = " + " ".join(f"{e:g}" for e in EPS))
    for r in rs:
        print(f"H{r['N']} K={r['K']:2d} {r['arm']:6s} s={r['seed']} dE={r['dE_mHa']:.3f} | "
              + " ".join(f"{r['screen_bias_mHa'][str(e)]:+.4f}" for e in EPS))
    print("\nmax |bias| over seeds (mHa); admissible iff <= 0.2")
    groups = {}
    for r in rs:
        groups.setdefault((r["N"], r["K"], r["arm"]), []).append(r)
    for (N, K, arm), g in sorted(groups.items()):
        mx = {e: max(abs(x["screen_bias_mHa"][str(e)]) for x in g) for e in EPS}
        adm = [e for e in EPS if mx[e] <= 0.2]
        print(f"H{N} K={K:2d} {arm:6s} n={len(g)} max|bias| @2e-3={mx[2e-3]:.4f} @1e-3={mx[1e-3]:.4f} "
              f"largest admissible eps={max(adm) if adm else None}")


if __name__ == "__main__":
    a = parse_args()
    aggregate(a.aggregate) if a.aggregate else run(a)
