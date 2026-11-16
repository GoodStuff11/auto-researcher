#!/usr/bin/env python3
"""
P2.1 round-6 per-leaf critic, T0/T1 check of Route 6.2 (PLR) for poles at reference nodes.

Claim under test (critic, T0): with f_m = 1 + h_m . Omega A0^{-1} phi_m[occ] = det(A0 + phi_m c_m^T)/det A0,
the PLR amplitude per spin is det(A0) * prod_{m=1}^{M} f_m = det(A0)^{1-M} * prod_m det(A0 + phi_m c_m^T),
so |psi| ~ eps^{1-M} when the smallest singular value eps of A0 -> 0 (a pole of order M-1 at every
reference node). LH-HFDS, det(A0 + sum_m phi_m c_m^T), the same corrections placed inside one
determinant, stays finite (it is a polynomial in the matrix entries).

Check 1: one configuration of the T1 model (t1_routes.PLR / LHHFDS weights, L = 6), A0's smallest
singular value set to eps = 1e-1 ... 1e-6; fit d log|psi| / d log eps per spin for PLR and LH-HFDS.
Check 2: 400 random configurations at L = 6 and L = 8 with the T1 PLR weights: Spearman correlation of
log|prod f| with -log|det A0|, and the largest log10|prod f|.
Run: OMP_NUM_THREADS=1 python3 critic_plr_poles.py > critic_plr_poles_output.txt
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from t1_routes import lattice, rand_cfg, PLR, LHHFDS  # noqa: E402


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def check1(L=6, F=8, seed=5):
    T, nn, patch, n = lattice(L)
    rng = np.random.default_rng(seed)
    plr = PLR(T, patch, n, F, rng)
    lh = LHHFDS(T, patch, n, F, np.random.default_rng(seed))
    ou, od = rand_cfg(T.shape[0], n, np.random.default_rng(seed + 1))
    rows = np.nonzero(ou)[0]
    h = plr.loc.feats(ou, od)
    hl = lh.loc.feats(ou, od)
    U, s, Wt = np.linalg.svd(plr.P0[0][rows])
    eps_list = 10.0 ** -np.arange(1, 7)
    lp, ll = [], []
    for eps in eps_list:
        s2 = s.copy(); s2[-1] = eps
        A0 = (U * s2) @ Wt
        Z = plr.Om[0] @ np.linalg.solve(A0, plr.Ph[0][rows])
        f = 1 + np.einsum("mf,fm->m", h, Z)
        lp.append(np.log(abs(np.linalg.det(A0))) + np.sum(np.log(np.abs(f))))
        C = np.einsum("mf,mfn->mn", hl, lh.Om[0])
        ll.append(np.log(abs(np.linalg.det(A0 + lh.K[0][rows] @ C))))
        print(f"  eps={eps:.0e}  log10|psi_PLR,up|={lp[-1] / np.log(10):+9.2f}"
              f"  log10|psi_LH,up|={ll[-1] / np.log(10):+7.2f}")
    sp = np.polyfit(np.log(eps_list), lp, 1)[0]
    sl = np.polyfit(np.log(eps_list), ll, 1)[0]
    M = T.shape[0]
    sp_tail = np.polyfit(np.log(eps_list[3:]), lp[3:], 1)[0]
    print(f"check1 asymptotic slope (eps=1e-4..1e-6): PLR={sp_tail:+.2f}, predicted 1-M={1 - M}")
    print(f"check1 L={L} M={M}: slope dlog|psi|/dlog eps  PLR={sp:+.2f} (pole order predicted 1-M={1 - M})"
          f"  LH-HFDS={sl:+.2f} (finite: predicted 0)")


def check2(L, F=8, seed=5, ncfg=400):
    T, nn, patch, n = lattice(L)
    plr = PLR(T, patch, n, F, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 7)
    ld, lf = [], []
    for _ in range(ncfg):
        ou, od = rand_cfg(T.shape[0], n, rng)
        h = plr.loc.feats(ou, od)
        for s, o in ((0, ou), (1, od)):
            rows = np.nonzero(o)[0]
            A0 = plr.P0[s][rows]
            Z = plr.Om[s] @ np.linalg.solve(A0, plr.Ph[s][rows])
            f = 1 + np.einsum("mf,fm->m", h, Z)
            ld.append(np.log10(abs(np.linalg.det(A0))))
            lf.append(np.sum(np.log10(np.abs(f))))
    ld, lf = np.array(ld), np.array(lf)
    print(f"check2 L={L} (V={T.shape[0]}, {2 * ncfg} spin blocks): spearman(log|prod f|, -log|det A0|)="
          f"{spearman(lf, -ld):+.2f}; log10|prod f| median {np.median(lf):+.2f}, max {lf.max():+.2f};"
          f" log10|det A0| min {ld.min():+.2f}")


if __name__ == "__main__":
    check1()
    for L in (6, 8):
        check2(L)
