#!/usr/bin/env python3
r"""
P2.2 R7.4: complete sparse zero-excess compensator (path gadgets) and its robustness off psi0.
===========================================================================================

Laplacian form.  Any Y with Y psi0 = 0 can be written Y = D^{-1} L D^{-1} (D = diag psi0, on psi0 != 0), with
L = D Y D a symmetric zero-row-sum ("signed Laplacian") matrix: L_ab = -g_ab, Y_ab = -g_ab / (psi_a psi_b).
Dropped pair (i,j): Y_ij = -h_ij  <=>  g_ij = h_ij psi_i psi_j = e_ij / 2  (fixed).  Kept pairs: g free.
  * e_ij > 0: positive conductance, the 2x2 piece g L_ij is PSD on its own.
  * e_ij < 0: negative conductance -w (w = |e|/2).  Take a shortest kept-graph path i = v0..vm = j and put
    conductance m*w on each of its m edges.  Series rule (Cauchy-Schwarz): sum_k m w (x_k - x_{k+1})^2
    >= w (x_i - x_j)^2, so the cycle piece is PSD.  Pieces add, so L >= 0 and Y >= 0 exactly.
Exact robustness.  psi' = psi0 (1 + delta xi), E xi xi^T = I:  E[psi'^T Y psi'] = delta^2 sum_i psi_i^2 Y_ii
  = delta^2 tr L,  tr L = sum_{e>0} e + sum_{e<0} (m_ij^2 - 1)|e_ij|   (no noise draws needed for the mean).
Determinants with psi0 = 0 (symmetry zeros) keep psi' = 0 under relative noise; dropped pairs touching them
get 2x2 blocks [[a, -h], [-h, h^2/a]] with excess a psi'_i^2 -> 0 as a -> 0 (counted, contribute 0 in the limit).

Metric (voi.md): f_rob = f (CAN denominator) at the largest grid eps with median-of-5 excess <= tau at delta.
Prints per grid eps: f, n_dropped, #negative dropped pairs, path-length histogram, unreachable pairs (fall back to
the 2x2 diagonal block, cost 2|e| on psi0), tr L, mean and median-of-5 noisy excess at delta = 1e-3, 1e-2.
Checks (at --check-eps): L 1 = 0, lambda_min(L) (dense), psi0^T Y psi0, explicit noisy excess vs formula.

Usage: path_gadget.py --n 6 [--n 8] [--check-eps 2.24e-3 1e-1] [--out results/path_gadget_H{n}.json]
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
from scipy.sparse.csgraph import shortest_path, connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P2.2_reopening_review"))
sys.path.insert(0, os.path.join(HERE, "..", "P2.2_stochastic_local_energy"))
import reopen_t1 as rt  # noqa: E402

TAU = 2e-4
GRID = np.logspace(-7, -1, 41)
DELTAS = (1e-3, 1e-2)


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def build_L(P, mask, nz, n):
    """Returns sparse L (on all n, zero rows for psi-zero dets), stats."""
    rk, ck = P["r"][~mask], P["c"][~mask]
    okk = nz[rk] & nz[ck]
    A = sp.coo_matrix((np.ones(okk.sum()), (rk[okk], ck[okk])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float).tocsr()
    rd, cd, e = P["r"][mask], P["c"][mask], P["x2"][mask]
    live = nz[rd] & nz[cd]
    n_touch_zero = int((~live).sum())
    rd, cd, e = rd[live], cd[live], e[live]
    rows, cols, vals = [], [], []

    def add_edge(a, b, g):  # adds g * (E_aa + E_bb - E_ab - E_ba)
        rows.extend([a, b, a, b]); cols.extend([a, b, b, a]); vals.extend([g, g, -g, -g])

    pos = e > 0
    for a, b, ee in zip(rd[pos], cd[pos], e[pos]):
        add_edge(a, b, ee / 2)
    neg = np.where(e < 0)[0]
    src = np.unique(rd[neg])
    plen = {}
    unreach, diag_cost = 0, 0.0
    if len(src):
        dist, pred = shortest_path(A, directed=False, unweighted=True, indices=src, return_predecessors=True)
        srow = {s: k for k, s in enumerate(src)}
        for t in neg:
            i, j, w = rd[t], cd[t], -e[t] / 2
            k = srow[i]
            if not np.isfinite(dist[k, j]):
                unreach += 1
                diag_cost += 2 * abs(e[t])
                # 2x2 block with Y psi0 != 0: a psi_i^2 + b psi_j^2 + e = |e| + |e| ; in L form a 2x2 PSD
                # piece with diagonal w + w and off-diagonal +w (not zero-row-sum)
                rows.extend([i, j, i, j]); cols.extend([i, j, j, i]); vals.extend([w, w, w, w])
                continue
            m = int(dist[k, j])
            plen[m] = plen.get(m, 0) + 1
            add_edge(i, j, -w)
            v = j
            while v != i:
                u = pred[k, v]
                add_edge(u, v, m * w)
                v = u
    L = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    ncomp = connected_components(A[nz][:, nz], directed=False)[0]
    return L, dict(n_dropped=int(mask.sum()), n_neg=int(len(neg)), n_pos=int(pos.sum()),
                   n_touch_psi_zero=n_touch_zero, unreachable=unreach, diag_cost_psi0=diag_cost,
                   path_len_hist={str(k): v for k, v in sorted(plen.items())}, kept_comps_nz=int(ncomp))


def run(nat, check_eps, psi_thr, eps_min=0.0, seed=0):
    t0 = time.time()
    R = 3.0
    Hc, _, occ_c, Kc, _ = rt.build(nat, R, "CAN")
    wc, vc = rt.lowest(Hc, 1)
    Pc = rt.pair_data(Hc, vc[:, 0], occ_c, Kc)
    nEX = float((Pc["pr"] + Pc["pc"]).sum())
    del Hc, Pc
    H, codes, occ, K, nocc = rt.build(nat, R, "LOW")
    w, v = rt.lowest(H, 1)
    psi = v[:, 0]
    n = H.shape[0]
    P = rt.pair_data(H, psi, occ, K)
    nz = np.abs(psi) > psi_thr
    print(f"H{nat} LOW n={n} E0={w[0]:.8f} pairs={len(P['h'])} psi-zero={int((~nz).sum())} nEX(CAN)={nEX:.4f} "
          f"cpu={cpu():.0f}s", flush=True)
    rng = np.random.default_rng(seed)
    xis = [rng.standard_normal(n) for _ in range(5)]
    ipsi = np.where(nz, 1 / np.where(nz, psi, 1), 0.0)
    rows = []
    checks = []
    ck_idx = {int(np.argmin(np.abs(np.log(GRID / c)))) for c in check_eps}
    for gi, eps in enumerate(GRID):
        mask = np.abs(P["h"]) <= eps
        f = float((P["pr"] + P["pc"]).sum() - (P["pr"][mask] + P["pc"][mask]).sum()) / nEX
        bS = abs(float(P["x2"][mask].sum()))
        if eps < eps_min * (1 - 1e-9):  # gadget skipped (cost); SCREEN row only
            rows.append(dict(eps=float(eps), f=f, SCREEN=bS, b_psi0=np.inf,
                             robust={str(d): dict(median=np.inf) for d in DELTAS}))
            continue
        L, st = build_L(P, mask, nz, n)
        trL = float(L.diagonal().sum())
        b0 = float(np.ones(n)[nz] @ (L @ np.where(nz, 1.0, 0.0))[nz])  # psi0^T Y psi0 = 1^T L 1 (on nz)
        rob = {}
        for d in DELTAS:
            vals = []
            for xi in xis:
                p2 = psi * (1 + d * xi)
                p2 /= np.linalg.norm(p2)
                u = p2 * ipsi  # D^{-1} psi'
                vals.append(float(u @ (L @ u)))
            rob[str(d)] = dict(median=float(np.median(vals)), mean_formula=d * d * trL / (1 + d * d) + b0)
        row = dict(eps=float(eps), f=f, SCREEN=bS, trL=trL, b_psi0=b0, robust=rob, **st)
        rows.append(row)
        print(f"eps={eps:.2e} f={f:.4f} SCREEN={bS:.2e} drop={st['n_dropped']} neg={st['n_neg']} "
              f"zero-touch={st['n_touch_psi_zero']} unreach={st['unreachable']} comps={st['kept_comps_nz']} "
              f"paths={st['path_len_hist']} trL={trL:.3e} b(psi0)={b0:.1e} "
              + " ".join(f"d={k}: med {r['median']:.2e} formula {r['mean_formula']:.2e}" for k, r in rob.items())
              + f" cpu={cpu():.0f}s", flush=True)
        if gi in ck_idx:
            Ld = L.toarray()[np.ix_(nz, nz)]
            lam = float(np.linalg.eigvalsh(Ld)[0])
            rowsum = float(np.abs(Ld.sum(1)).max())
            Y = Ld * np.outer(ipsi[nz], ipsi[nz])
            # Y must equal -h on every dropped pair (both psi-nonzero)
            rd, cd, hd = P["r"][mask], P["c"][mask], P["h"][mask]
            lv = nz[rd] & nz[cd]
            idx = np.arange(n) - np.cumsum(~nz)
            fixerr = float(np.abs(Y[idx[rd[lv]], idx[cd[lv]]] + hd[lv]).max()) if lv.any() else 0.0
            # Y must vanish off diag + kept + dropped(nz) pattern
            allowed = np.eye(nz.sum(), dtype=bool)
            r_all, c_all = P["r"], P["c"]
            la = nz[r_all] & nz[c_all]
            allowed[idx[r_all[la]], idx[c_all[la]]] = True
            allowed[idx[c_all[la]], idx[r_all[la]]] = True
            offpat = float(np.abs(Y[~allowed]).max()) if (~allowed).any() else 0.0
            ymax = float(np.abs(Y[~np.eye(len(Y), dtype=bool)]).max())
            ck = dict(eps=float(eps), lam_min_L=lam, lam_min_L_rel=lam / max(np.abs(Ld).max(), 1e-300),
                      max_rowsum=rowsum, max_fixed_entry_err=fixerr, max_entry_off_pattern=offpat,
                      max_offdiag_Y=ymax, max_offdiag_Y_over_hmax=ymax / float(np.abs(P["h"]).max()))
            checks.append(ck)
            print("CHECK " + json.dumps(ck), flush=True)
            del Ld, Y, allowed
    def best(fn):
        ok = [r for r in rows if fn(r) <= TAU]
        return max(ok, key=lambda r: r["eps"]) if ok else None
    out = dict(n_atoms=nat, n=n, nEX=nEX, rows=rows, checks=checks,
               f_S=best(lambda r: r["SCREEN"])["f"],
               f_rob={str(d): (best(lambda r, d=d: r["robust"][str(d)]["median"]) or {}).get("f") for d in DELTAS},
               f_psi0=best(lambda r: r["b_psi0"])["f"], grid_end_f=rows[-1]["f"],
               cpu_s=cpu(), wall_s=time.time() - t0)
    print(f"H{nat}: f_S*={out['f_S']:.4f} f_psi0={out['f_psi0']:.4f} f_rob={out['f_rob']} grid-end f={rows[-1]['f']:.4f}"
          f" cpu={cpu():.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[6])
    ap.add_argument("--check-eps", type=float, nargs="*", default=[2.2387211385683377e-3, 1e-1])
    ap.add_argument("--psi-thr", type=float, default=1e-9)
    ap.add_argument("--eps-min", type=float, default=0.0,
                    help="build gadgets only for grid eps >= this (f_rob is the LARGEST admissible eps, so a pass "
                         "at the grid end already fixes f_rob; smaller eps are diagnostics)")
    a = ap.parse_args()
    for nat in a.n:
        out = run(nat, a.check_eps, a.psi_thr, a.eps_min)
        p = os.path.join(HERE, "results", f"path_gadget_H{nat}.json")
        with open(p, "w") as fh:
            json.dump(out, fh, indent=1)
    print(f"total cpu {cpu():.0f} s", flush=True)


if __name__ == "__main__":
    main()
