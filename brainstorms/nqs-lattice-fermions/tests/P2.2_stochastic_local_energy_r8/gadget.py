#!/usr/bin/env python3
"""Explicit zero-excess compensator (no SDP): Y = sum over dropped pairs (i,j) of w w^T, w supported on {i,j,k}
with k a common kept neighbour, w_i w_j = -h_ij, w . psi_ref = 0.  Y >= 0, Y psi_ref = 0, and Y only touches
diagonal + kept pairs, so it certifies <psi|H_kept + Y + diag-part|psi> >= <psi|H|psi> for every psi.
Reports: pairs without a common kept neighbour, max |Y| entry, excess on psi0 and on noisy psi0.
Usage: gadget.py [eps ...]   (H6 LOW, R = 3.0; default eps 2.24e-3 1e-1)"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
import numpy as np
import scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P2.2_reopening_review"))
sys.path.insert(0, os.path.join(HERE, "..", "P2.2_stochastic_local_energy"))
import reopen_t1 as rt  # noqa: E402

eps_list = [float(x) for x in sys.argv[1:]] or [2.2387211385683377e-3, 1e-1]
H, codes, occ, K, nocc = rt.build(6, 3.0, "LOW")
w, v = rt.lowest(H, 1)
psi = v[:, 0]
n = H.shape[0]
P = rt.pair_data(H, psi, occ, K)
hmax = float(np.abs(P["h"]).max())
rng = np.random.default_rng(0)
xis = [rng.standard_normal(n) for _ in range(5)]
out = []
for eps in eps_list:
    mask = np.abs(P["h"]) <= eps
    rk, ck = P["r"][~mask], P["c"][~mask]
    A = sp.coo_matrix((np.ones(len(rk)), (rk, ck)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8).tocsr()
    Y = np.zeros((n, n))
    nofix = 0
    for i, j, h in zip(P["r"][mask], P["c"][mask], P["h"][mask]):
        common = np.intersect1d(A[i].indices, A[j].indices)
        if len(common) == 0:
            nofix += 1
            continue
        k = common[np.argmax(np.abs(psi[common]))]  # largest-|psi| pivot keeps w_k small
        s = np.sqrt(abs(h) * abs(psi[j]) / max(abs(psi[i]), 1e-300))
        wi, wj = s, -h / s
        wk = -(wi * psi[i] + wj * psi[j]) / psi[k]
        idx = np.array([i, j, k]); wv = np.array([wi, wj, wk])
        Y[np.ix_(idx, idx)] += np.outer(wv, wv)
    lam = np.linalg.eigvalsh(Y).min()
    rob = {}
    for d in (1e-3, 1e-2):
        vals = []
        for xi in xis:
            p2 = psi * (1 + d * xi); p2 /= np.linalg.norm(p2)
            vals.append(float(p2 @ Y @ p2))
        rob[str(d)] = float(np.median(vals))
    nk = float((P["pr"] + P["pc"]).sum() - (P["pr"][mask] + P["pc"][mask]).sum())
    rec = dict(eps=eps, n_dropped=int(mask.sum()), no_common_neighbour=nofix, lam_min=float(lam),
               excess_psi0=float(psi @ Y @ psi), max_entry=float(np.abs(Y).max()),
               max_entry_over_hmax=float(np.abs(Y).max() / hmax), excess_noisy_median=rob, kept_own=nk)
    out.append(rec)
    print(json.dumps(rec), flush=True)
json.dump(out, open(os.path.join(HERE, "results", "h6_gadget.json"), "w"), indent=1)
r = resource.getrusage(resource.RUSAGE_SELF); print(f"cpu {r.ru_utime + r.ru_stime:.0f} s")
