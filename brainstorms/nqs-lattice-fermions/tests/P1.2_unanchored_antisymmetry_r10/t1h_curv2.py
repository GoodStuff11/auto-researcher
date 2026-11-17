#!/usr/bin/env python3
"""
P1.2 r10 (2026-11-17): curvature screen of the cross-site / two-hop phase span. Rule in RULE.md (written first).
Reuses t1f_curv.lam_min and t1f_phase.setup from ../P1.2_unanchored_antisymmetry_r9 unchanged.
Usage: python3 t1h_curv2.py [--sizes 1,2]   (writes results/curv2.json; one thread; RLIMIT_CPU 540 s)
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R9 = os.path.join(os.path.dirname(HERE), "P1.2_unanchored_antisymmetry_r9")
sys.path.insert(0, R9)
import t1f_phase as P  # noqa: E402
from t1f_curv import lam_min  # noqa: E402

OUT = os.path.join(HERE, "results")
ONE = np.int64(1)


def edges(m):
    """List of (label, sites, tgt index array (-1 if not allowed))."""
    bu, bd, V = m.bu, m.bd, m.V
    empty = ~(bu | bd)
    hb = sorted({(min(i, j), max(i, j)) for i in range(V) for j in range(V) if i != j and m.T[i, j] != 0.0})
    out = []
    for (i, j) in hb:
        for s in (0, 1):
            b = bu if s == 0 else bd
            tgt = np.full(m.D, -1, np.int64)
            for (a, c) in ((i, j), (j, i)):        # particle at c hops to empty a
                src = np.nonzero(((b >> c) & 1) & ((empty >> a) & 1))[0]
                nb = b[src] ^ (ONE << c) ^ (ONE << a)
                tgt[src] = m.index(nb, bd[src]) if s == 0 else m.index(bu[src], nb)
            out.append((f"hop{s}_{i}_{j}", (i, j), tgt))
    for (i, j) in m.nn:
        tgt = np.full(m.D, -1, np.int64)
        for (a, c) in ((i, j), (j, i)):            # up at c, down at a -> swap
            src = np.nonzero(((bu >> c) & 1) & ((bd >> a) & 1))[0]
            tgt[src] = m.index(bu[src] ^ (ONE << c) ^ (ONE << a), bd[src] ^ (ONE << a) ^ (ONE << c))
        out.append((f"ex_{i}_{j}", (i, j), tgt))
    assert all((e[2][e[2] != -1] >= 0).all() for e in out)
    return hb, out


def edge_ratios(H, psi, E):
    D = len(psi)
    G = np.zeros((D, len(E)))
    for k, (_, _, tgt) in enumerate(E):
        ok = np.nonzero(tgt >= 0)[0]
        h = np.asarray(H[tgt[ok], ok]).ravel()
        assert np.all(h != 0.0)
        R = h * psi[tgt[ok]] / psi[ok]
        G[ok, k] = R / (1.0 + np.abs(R))
    return G


def clean(S, w):
    """Drop zero columns, normalise (span unchanged)."""
    nrm = np.sqrt((S ** 2 * w[:, None]).sum(0))
    keep = nrm > 1e-12 * max(nrm.max(), 1e-300)
    return S[:, keep] / nrm[keep]


def rank_w(S, psi):
    N = psi @ psi
    Sc = S - (psi ** 2 @ S / N)[None, :]
    ev = np.linalg.eigvalsh((Sc * (psi ** 2)[:, None]).T @ Sc / N)
    return int((ev > 1e-10 * ev.max()).sum())


def capture(S, psi, g):
    """psi^2-weighted fraction of centred g captured by span(S)."""
    w = psi ** 2 / (psi @ psi)
    gc = g - w @ g
    Sc = S - (w @ S)[None, :]
    sw = np.sqrt(w)
    A, b = Sc * sw[:, None], gc * sw
    coef, *_ = np.linalg.lstsq(A, b, rcond=1e-10)
    return float(np.sum((A @ coef) ** 2) / np.sum(b ** 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,2")
    a = ap.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (540, 550))
    keys = ("S", "rand", "site", "edge", "ratio2", "free_dir")
    rows, per_all = {}, {}
    for k in [int(x) for x in a.sizes.split(",")]:
        per = []
        for seed in (11, 22):
            d = P.setup(k, seed, "G")
            m, H, psi, Phi = d["m"], d["H"], d["psiF"], d["Phi"]
            D, V = Phi.shape[0], Phi.shape[1]
            w = psi ** 2
            ratio = Phi.reshape(D, V * 5)
            prods = [Phi[:, :, i] * Phi[:, :, j] for i in range(5) for j in range(i + 1, 5)]
            ratio2 = np.concatenate([ratio] + prods, 1)
            hb, E = edges(m)
            cross = [Phi[:, i, :, None] * Phi[:, j, None, :] for (i, j) in hb]
            cross = np.concatenate([c.reshape(D, 25) for c in cross], 1)
            G = edge_ratios(H, psi, E)
            pairs = [(p, q) for p in range(len(E)) for q in range(p + 1, len(E))
                     if set(E[p][1]) & set(E[q][1])]
            EP = np.stack([G[:, p] * G[:, q] for (p, q) in pairs], 1)
            site = clean(np.concatenate([ratio2, cross], 1), w)
            edge = clean(np.concatenate([G, EP], 1), w)
            S = np.concatenate([site, edge], 1)
            rS = rank_w(S, psi)
            rand = np.random.default_rng(seed).choice([-1.0, 1.0], size=(D, rS))
            x0 = 0.1 * np.random.default_rng(seed + 7).standard_normal(D)
            r = minimize(lambda p_: P.e_and_dphi(H, psi, p_), x0, jac=True, method="L-BFGS-B",
                         options=dict(maxiter=500))
            gf = r.x
            res = dict(seed=seed, D=D, cols_S=S.shape[1], rank_S=rS, n_edges=len(E), n_pairs=len(pairs),
                       S=lam_min(H, psi, S), rand=lam_min(H, psi, rand), site=lam_min(H, psi, site),
                       edge=lam_min(H, psi, edge), ratio2=lam_min(H, psi, ratio2),
                       free_dir=lam_min(H, psi, gf[:, None]),
                       cap_S=capture(S, psi, gf), cap_site=capture(site, psi, gf),
                       cap_edge=capture(edge, psi, gf), cap_ratio2=capture(ratio2, psi, gf),
                       cap_rand=capture(rand, psi, gf), cpu=time.process_time())
            per.append(res)
            P.log(f"size {k} seed {seed}: D {D} rank(S) {rS} edges {len(E)} pairs {len(pairs)} | "
                  + " ".join(f"{x} {res[x]:+.4f}" for x in keys) + " | capture "
                  + " ".join(f"{x} {res['cap_' + x]:.3f}" for x in ("S", "site", "edge", "ratio2", "rand"))
                  + f" | CPU {res['cpu']:.0f} s")
        per_all[k] = per
        rows[k] = {x: float(np.median([p[x] for p in per])) for x in keys + tuple(
            "cap_" + y for y in ("S", "site", "edge", "ratio2", "rand"))}
    cnt = (1, 2)
    if all(rows[k]["S"] >= 0 for k in cnt):
        v = "FAMILY-KILL"
    elif all(rows[k]["S"] < 0 and rows[k]["S"] < rows[k]["rand"] for k in cnt):
        v = "FAMILY-OPEN"
    else:
        v = "AMBIGUOUS"
    for k in rows:
        P.log(f"median size {k}: " + " ".join(f"{x} {rows[k][x]:+.4f}" for x in rows[k]))
    cpu = time.process_time()
    P.log(f"VERDICT cross-site/two-hop phase span: {v}; CPU {cpu:.0f} s")
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(rows=rows, per=per_all, verdict=v, cpu_s=cpu), open(os.path.join(OUT, "curv2.json"), "w"),
              indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
