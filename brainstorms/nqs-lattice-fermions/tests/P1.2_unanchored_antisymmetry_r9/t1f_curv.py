#!/usr/bin/env python3
"""
P1.2 reopening review (2026-11-16): second-order premise of the PHASE-carrier family (RPC and any mutation of it).
Rules written here BEFORE any run.
====================================================================================================================
At a real state psi (the trained t1c foil), the energy of psi exp(i eps g) is E + eps^2 q(g) + O(eps^4) with
    q(g) = -(1/N) sum_{n,m} psi_n psi_m H_nm (g_n - g_m)^2 / 2 ,  N = sum psi^2
(odd orders vanish for real psi and real symmetric H). A phase carrier can leave the real foil downhill only along
directions with q(g) < 0. Normalise by the psi^2-weighted variance of g, v(g) = sum psi_n^2 (g_n - <g>)^2 / N, and
report lam(S) = min over g in span S of q(g) / v(g) (a small generalised eigenproblem on the centred columns of S).
Spans (columns are functions of the configuration n, all at the same foil):
  ratio  [x_up, x_dn, f_0, f_1, f_2] per site, NOT weight-shared (5V columns): every phase that is linear in the
         per-site ratio features; the weight-shared RPC kernel at small amplitude lies inside the span of its
         tanh features, and this linear span is its generous linear stand-in.
  ratio2 ratio plus all products of two per-site columns at the same site (5V + 10V columns).
  occ    [x_up, x_dn] per site (2V columns): occupation-only control.
  rand   5V random +-1 columns: capacity control.
  free   the direction of the trained unrestricted phase (results/t1f/<size>_free_<seed>.json is not enough, so
         recomputed here with the same L-BFGS run as t1f_phase.py, seed for seed) -- reported only.
RULE (median over foil seeds 11 and 22; counted sizes 1 and 2 as in t1f_phase.py):
  FAMILY-KILL  if lam(ratio2) >= 0 at both counted sizes: no extensive phase direction built from the ratio
               features (up to second order in them) descends from the foil, so every phase carrier fed these
               features is stuck at the real foil to second order, as RPC was.
  FAMILY-OPEN  if lam(ratio2) < 0 and lam(ratio2) < lam(rand) at both counted sizes (a descending direction that
               a random span of equal size does not have): RPC's kill is then an optimiser failure, and the mutation
               RPC-L (a linear phase on ratio2, warm-started along the lam(ratio2) eigenvector) goes to T1.
  UNDECIDED    otherwise.
Budget: RLIMIT_CPU 300 s; one thread by default (at most 4).

Usage
-----
  python3 t1f_curv.py [--threads INT]    (sizes 0, 1, 2; seeds 11, 22; writes results/t1f/curv.json)
"""
import os
import sys

_n = 1
for _i, _a in enumerate(sys.argv):
    if _a == "--threads" and _i + 1 < len(sys.argv):
        _n = max(1, min(4, int(sys.argv[_i + 1])))
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = str(_n)

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sl  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t1f_phase as P  # noqa: E402


def lam_min(H, psi, S):
    N = psi @ psi
    S = S - (psi ** 2 @ S / N)[None, :]
    A = sp.diags(psi) @ H @ sp.diags(psi)
    A = A - sp.diags(A.diagonal())
    deg = np.asarray(A.sum(1)).ravel()
    # q(g) = (1/N) g^T (diag(deg) - A) g  * (-1)  ... from q = -(1/2N) sum A_nm (g_n - g_m)^2
    Lg = (deg[:, None] * S - A @ S)
    Q = -(S.T @ Lg) / N
    Wm = (S * (psi ** 2)[:, None]).T @ S / N
    ev, U = np.linalg.eigh(Wm)
    keep = ev > 1e-10 * ev.max()
    T = U[:, keep] / np.sqrt(ev[keep])
    w = sl.eigh(T.T @ Q @ T, eigvals_only=True)
    return float(w[0])


def q_of(H, psi, g):
    return lam_min(H, psi, g[:, None])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None):
    parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (300, 310))
    rows = {}
    for k in (0, 1, 2):
        per = []
        for seed in (11, 22):
            d = P.setup(k, seed, "G")
            H, psi, Phi = d["H"], d["psiF"], d["Phi"]
            D, V = Phi.shape[0], Phi.shape[1]
            ratio = Phi.reshape(D, V * 5)
            prods = [Phi[:, :, i] * Phi[:, :, j] for i in range(5) for j in range(i + 1, 5)]
            ratio2 = np.concatenate([ratio] + prods, 1)
            occ = Phi[:, :, :2].reshape(D, 2 * V)
            rand = np.random.default_rng(seed).choice([-1.0, 1.0], size=(D, 5 * V))
            from scipy.optimize import minimize
            x0 = 0.1 * np.random.default_rng(seed + 7).standard_normal(D)
            r = minimize(lambda p_: P.e_and_dphi(H, psi, p_), x0, jac=True, method="L-BFGS-B",
                         options=dict(maxiter=500))
            res = dict(seed=seed, ratio=lam_min(H, psi, ratio), ratio2=lam_min(H, psi, ratio2),
                       occ=lam_min(H, psi, occ), rand=lam_min(H, psi, rand), free_dir=q_of(H, psi, r.x))
            per.append(res)
            P.log(f"size {k} seed {seed}: " + " ".join(f"{a} {res[a]:+.4f}" for a in
                                                      ("ratio", "ratio2", "occ", "rand", "free_dir")))
        rows[k] = {a: float(np.median([x[a] for x in per])) for a in ("ratio", "ratio2", "occ", "rand", "free_dir")}
    cnt = (1, 2)
    if all(rows[k]["ratio2"] >= 0 for k in cnt):
        v = "FAMILY-KILL"
    elif all(rows[k]["ratio2"] < 0 and rows[k]["ratio2"] < rows[k]["rand"] for k in cnt):
        v = "FAMILY-OPEN"
    else:
        v = "UNDECIDED"
    for k in rows:
        P.log(f"median size {k}: " + " ".join(f"{a} {rows[k][a]:+.4f}" for a in rows[k]))
    P.log(f"VERDICT phase-family second-order premise: {v}; CPU {time.process_time():.0f} s")
    json.dump(dict(rows=rows, verdict=v), open(os.path.join(P.OUT, "curv.json"), "w"), indent=1)
    P.ledger_add("curv", time.process_time(), "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
