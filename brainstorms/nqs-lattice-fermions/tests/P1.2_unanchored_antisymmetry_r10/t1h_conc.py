#!/usr/bin/env python3
"""Context only (no decision weight; written AFTER the t1h_curv2 verdict, prompted by t1h_global showing the
unrestricted lambda_min is a single-configuration spike near a node). For the lambda(S) eigenvector g_S, the
free-phase direction g_f and the rand eigenvector: psi^2-weighted participation number of (g - <g>)^2, and the
sup-normalised curvature q(g) / max|g - <g>|^2 (energy gain per unit squared phase amplitude, which a phase bounded
by pi can actually realise). Sizes 1, 2; seeds 11, 22."""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sl  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t1h_curv2 as T  # noqa: E402
P = T.P


def lam_vec(H, psi, S):
    N = psi @ psi
    S = S - (psi ** 2 @ S / N)[None, :]
    A = sp.diags(psi) @ H @ sp.diags(psi)
    A = A - sp.diags(A.diagonal())
    deg = np.asarray(A.sum(1)).ravel()
    Q = -(S.T @ (deg[:, None] * S - A @ S)) / N
    Wm = (S * (psi ** 2)[:, None]).T @ S / N
    ev, U = np.linalg.eigh(Wm)
    keep = ev > 1e-10 * ev.max()
    Tm = U[:, keep] / np.sqrt(ev[keep])
    w, X = sl.eigh(Tm.T @ Q @ Tm)
    return float(w[0]), S @ (Tm @ X[:, 0])


def q_abs(H, psi, g):
    N = psi @ psi
    g = g - (psi ** 2 @ g) / N
    A = sp.diags(psi) @ H @ sp.diags(psi)
    A = A - sp.diags(A.diagonal())
    deg = np.asarray(A.sum(1)).ravel()
    q = -(g @ (deg * g - A @ g)) / N
    p = psi ** 2 * g ** 2
    p /= p.sum()
    return q / np.max(np.abs(g)) ** 2, 1.0 / np.sum(p ** 2)


resource.setrlimit(resource.RLIMIT_CPU, (300, 310))
for k in (1, 2):
    for seed in (11, 22):
        d = P.setup(k, seed, "G")
        m, H, psi, Phi = d["m"], d["H"], d["psiF"], d["Phi"]
        D, V = Phi.shape[0], Phi.shape[1]
        w = psi ** 2
        ratio = Phi.reshape(D, V * 5)
        prods = [Phi[:, :, i] * Phi[:, :, j] for i in range(5) for j in range(i + 1, 5)]
        hb, E = T.edges(m)
        cross = np.concatenate([(Phi[:, i, :, None] * Phi[:, j, None, :]).reshape(D, 25) for (i, j) in hb], 1)
        G = T.edge_ratios(H, psi, E)
        pairs = [(p, q) for p in range(len(E)) for q in range(p + 1, len(E)) if set(E[p][1]) & set(E[q][1])]
        EP = np.stack([G[:, p] * G[:, q] for (p, q) in pairs], 1)
        S = np.concatenate([T.clean(np.concatenate([ratio] + prods + [cross], 1), w),
                            T.clean(np.concatenate([G, EP], 1), w)], 1)
        rand = np.random.default_rng(seed).choice([-1.0, 1.0], size=(D, T.rank_w(S, psi)))
        lS, gS = lam_vec(H, psi, S)
        lR, gR = lam_vec(H, psi, rand)
        x0 = 0.1 * np.random.default_rng(seed + 7).standard_normal(D)
        gf = minimize(lambda p_: P.e_and_dphi(H, psi, p_), x0, jac=True, method="L-BFGS-B",
                      options=dict(maxiter=500)).x
        out = []
        for name, g in (("S", gS), ("free", gf), ("rand", gR)):
            qa, pn = q_abs(H, psi, g)
            out.append(f"{name}: q/max|g|^2 {qa:+.2e} PN {pn:.0f}")
        P.log(f"size {k} seed {seed}: lam(S) {lS:+.4f} | " + " | ".join(out) + f" | CPU {time.process_time():.0f} s")
