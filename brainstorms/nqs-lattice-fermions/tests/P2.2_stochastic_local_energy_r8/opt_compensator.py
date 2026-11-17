#!/usr/bin/env python3
r"""
P2.2 R7.4 (C3') T1: best possible compensator supported on diagonal + kept pairs (rule in RULE.md).
=================================================================================================

Primal (per eps):  b_opt = min psi^T Y psi,  Y = C - H_drop >= 0,  C on diag + kept pairs.
Dual (Lagrange):   b_opt = max <H_drop, S> - psi^T H_drop psi  over PSD S with S_ij = psi_i psi_j on diag + kept
                   (a PSD *completion* of psi psi^T from the kept pattern; Grone et al. 1984).
Writing S = D_psi T D_psi (psi_i != 0): T is a correlation matrix with T_ij = 1 on kept pairs, which forces
T_ij = 1 across every connected component of the kept graph (restricted to psi_i != 0).  Hence
    b_opt = max_{T corr. over components}  sum_{dropped pairs i<j} e_ij (T_{c(i)c(j)} - 1),  e_ij = 2 h_ij psi_i psi_j
a Goemans-Williamson-type SDP over the components.  Dropping T's off-diagonal freedom to [-1,1] per pair gives
back the diagonal-compensator bound sum (|e| - e).  Dual Slater point T = I exists, so the primal infimum equals
this value (attainment checked separately with a full 400x400 primal SDP).

Arms printed per eps: SCREEN |b_S|, VPS1 (diagonal, as in reopen_t1.py), OPT (this SDP), and OPTk (full primal
with free entries of Y bounded by kappa * max|H_offdiag|, a "sane-norm" compensator), at selected eps.

Options
-------
--psi-thr FLOAT    |psi_i| below which a determinant counts as psi-zero (default 1e-9)
--kappa FLOAT...   bounds for the OPTk arm (default 1 10)
--primal-at EPS... grid eps values (nearest grid point) at which to run the full primal (default 2.24e-3 1e-1)
--delta D...       relative amplitude noise for the off-psi0 check psi' = psi0 (1 + D xi) (default 1e-3 1e-2)
--out PATH         JSON output (default results/h6_opt.json)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"
import argparse
import json
import resource
import sys
import time

resource.setrlimit(resource.RLIMIT_CPU, (600, 605))

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import cvxpy as cp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P2.2_reopening_review"))
sys.path.insert(0, os.path.join(HERE, "..", "P2.2_stochastic_local_energy"))
import reopen_t1 as rt  # noqa: E402  (same builder, pair list, VPS1 code)

TAU = 2e-4
GRID = np.logspace(-7, -1, 41)


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def opt_dual(P, mask, nz, n):
    """Exact b_opt via the component SDP.  mask = dropped pairs."""
    keep = ~mask
    r, c = P["r"][keep], P["c"][keep]
    ok = nz[r] & nz[c]
    A = sp.coo_matrix((np.ones(ok.sum()), (r[ok], c[ok])), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    rd, cd, e = P["r"][mask], P["c"][mask], P["x2"][mask]
    live = nz[rd] & nz[cd] & (lab[rd] != lab[cd])
    if not live.any():
        return 0.0, 0, ncomp, float(e[nz[rd] & nz[cd] & (lab[rd] == lab[cd])].sum()) if len(e) else 0.0
    a, b = lab[rd[live]], lab[cd[live]]
    used, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    m = len(used)
    ia, ib = inv[: len(a)], inv[len(a):]
    W = np.zeros((m, m))
    np.add.at(W, (ia, ib), e[live])
    W = (W + W.T) / 2  # sum_{pairs} e (T_ab - 1) = <W, T> - sum e   (W symmetric, both triangles)
    if m == 1:
        return 0.0, 1, ncomp, 0.0
    T = cp.Variable((m, m), PSD=True)
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(W, T))), [cp.diag(T) == 1])
    prob.solve(solver=cp.CLARABEL)
    val = float(prob.value - e[live].sum())
    return max(val, 0.0), m, ncomp, 0.0


def primal(H, P, mask, psi, kappa=None):
    """Full primal on all n determinants with SCS; returns repaired certified excess and max |free entry|."""
    n = H.shape[0]
    Y = cp.Variable((n, n), symmetric=True)
    fixed = np.zeros((n, n))
    free = np.eye(n, dtype=bool)
    rk, ck = P["r"][~mask], P["c"][~mask]
    free[rk, ck] = free[ck, rk] = True
    rd, cd = P["r"][mask], P["c"][mask]
    fixed[rd, cd] = fixed[cd, rd] = -P["h"][mask]
    Fm = (~free).astype(float)
    cons = [Y >> 0, cp.multiply(Fm, Y) == fixed]
    if kappa is not None:
        cons.append(cp.abs(cp.multiply(free.astype(float), Y)) <= kappa)
    prob = cp.Problem(cp.Minimize(psi @ Y @ psi), cons)
    prob.solve(solver=cp.SCS, eps=1e-8, max_iters=20000)
    Yv = Y.value
    Yv = np.where(free, Yv, fixed)
    Yv = (Yv + Yv.T) / 2
    lam = np.linalg.eigvalsh(Yv).min()
    if lam < 0:
        Yv = Yv - lam * np.eye(n)
    return float(psi @ Yv @ psi), float(np.abs(Yv[free]).max()), float(prob.value), prob.status, Yv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psi-thr", type=float, default=1e-9)
    ap.add_argument("--kappa", type=float, nargs="*", default=[1.0, 10.0])
    ap.add_argument("--primal-at", type=float, nargs="*", default=[2.24e-3, 1e-1])
    ap.add_argument("--delta", type=float, nargs="+", default=[1e-3, 1e-2])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "h6_opt.json"))
    a = ap.parse_args()
    t0 = time.time()
    R, N = 3.0, 6
    Hc, *_ = rt.build(N, R, "CAN")
    wc, vc = rt.lowest(Hc, 1)
    occ_c = rt.build(N, R, "CAN")[2]
    nEX = float((lambda P: (P["pr"] + P["pc"]).sum())(rt.pair_data(Hc, vc[:, 0], occ_c, N)))
    H, codes, occ, K, nocc = rt.build(N, R, "LOW")
    w, v = rt.lowest(H, 2)
    psi = v[:, 0]
    n = H.shape[0]
    P = rt.pair_data(H, psi, occ, K)
    nz = np.abs(psi) > a.psi_thr
    print(f"H6 LOW: n={n} E0={w[0]:.8f} gap={w[1]-w[0]:.4f} pairs={len(P['h'])} psi-zero dets={int((~nz).sum())} "
          f"maxoff|h|={np.abs(P['h']).max():.3e}", flush=True)
    hmax = float(np.abs(P["h"]).max())
    rows = []
    for eps in GRID:
        mask = np.abs(P["h"]) <= eps
        nk = nEX * 0 + (P["pr"] + P["pc"]).sum() - float((P["pr"][mask] + P["pc"][mask]).sum())
        f = nk / nEX
        bS = abs(float(P["x2"][mask].sum()))
        bV, _ = rt.vps_excess(P, mask)
        sum_abs = float((np.abs(P["x2"][mask]) - P["x2"][mask]).sum())
        bO, m, ncomp, _ = opt_dual(P, mask, nz, n)
        rows.append(dict(eps=float(eps), f=float(f), SCREEN=bS, VPS1=float(bV), DIAGLB=sum_abs, OPT=bO,
                         m_comp=int(m), ncomp=int(ncomp)))
        print(f"eps={eps:.2e} f={f:.4f} SCREEN={bS:.2e} VPS1={bV:.2e} sum(|e|-e)={sum_abs:.2e} OPT={bO:.2e} "
              f"comps(kept graph)={ncomp} in-SDP={m}  cpu={cpu():.0f}s", flush=True)

    def best(key):
        ok = [i for i, r in enumerate(rows) if r[key] <= TAU]
        return rows[max(ok)] if ok else None

    res = {k: best(k) for k in ("SCREEN", "VPS1", "OPT")}
    for k, b in res.items():
        print(f"BEST {k}: f={b['f']:.4f} eps*={b['eps']:.2e} err={b[k]:.2e}", flush=True)

    # full primal at selected eps: attainment, entry size, kappa-bounded arms, excess off psi0
    rng = np.random.default_rng(0)
    xis = [rng.standard_normal(n) for _ in range(5)]
    prim = []
    for e_req in a.primal_at:
        i = int(np.argmin(np.abs(np.log(GRID / e_req))))
        eps = rows[i]["eps"]
        mask = np.abs(P["h"]) <= eps
        for kap in [None] + list(a.kappa):
            if cpu() > 500:
                print("CPU guard: skipping remaining primal solves", flush=True)
                break
            kv = None if kap is None else kap * hmax
            bP, Ymax, raw, st, Yv = primal(H, P, mask, psi, kv)
            rob = {}
            for d in a.delta:
                vals = []
                for xi in xis:
                    p2 = psi * (1 + d * xi)
                    p2 /= np.linalg.norm(p2)
                    vals.append(float(p2 @ Yv @ p2))
                rob[str(d)] = float(np.median(vals))
            prim.append(dict(eps=eps, f=rows[i]["f"], kappa=kap, b_certified=bP, max_free_entry=Ymax,
                             raw=raw, status=st, excess_perturbed_median=rob, diag_LB=rows[i]["DIAGLB"]))
            print(f"PRIMAL eps={eps:.2e} f={rows[i]['f']:.4f} kappa={kap} b_cert={bP:.3e} (raw {raw:.3e}, {st}) "
                  f"max|Y_free|={Ymax:.3e} (= {Ymax/hmax:.1f} max|h|) excess(noisy psi0, median of 5): "
                  + ", ".join(f"d={k}: {v:.2e}" for k, v in rob.items()) + f"  cpu={cpu():.0f}s", flush=True)
    fS = res["SCREEN"]["f"]
    ratio = res["OPT"]["f"] / fS
    verdict = "KILL" if ratio >= 3 else "PASS-T1"
    print(f"f_S*={fS:.4f} f_diag={res['VPS1']['f']:.4f} (ratio {res['VPS1']['f']/fS:.2f}) "
          f"f_opt={res['OPT']['f']:.4f} ratio={ratio:.3f} -> {verdict}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(rows=rows, best=res, primal=prim, ratio=ratio, ratio_diag=res["VPS1"]["f"] / fS,
                       verdict=verdict, psi_thr=a.psi_thr, n_psi_zero=int((~nz).sum()), hmax=hmax,
                       cpu_s=cpu(), wall_s=time.time() - t0), fh, indent=1)
    print(f"cpu {cpu():.0f} s wall {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
