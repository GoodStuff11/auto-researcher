#!/usr/bin/env python3
"""
P5.1 round 4, route R51.10 `kappa`: exact noise-to-signal ratio of the sampled SR step (ridge-regression sample
complexity), computed from exact sums at one state.  Rule: RULE_kappa.md.
    ~/.conda/envs/lit_env/bin/python kappa.py --Lx LX --Ly LY --Ne NE [--start FILE] [--theta FILE] [--tag TAG]

At theta, with p = |psi|^2 over orbit representatives, O_c = O - <O>, e = E_loc - E, lambda = 1e-3 (R51.1's shift):
  S = <O_c O_c^T>_p,  f = <O_c e>_p,  x = (S + lambda)^-1 f   (the exact step direction before length scaling).
A sampled step from N i.i.d. draws from q (weights w = p/q) has, to first order in 1/N,
  E || x_N - x ||_S^2 = (1/N) tr[ A C_q A S ],   A = (S + lambda)^-1,   C_q = <w (g - <g>)(g - <g>)^T>_p,
with g = O_c r and r = e - O_c^T x the regression residual (the first-order error of the ridge solution, force and
metric fluctuations together).  kappa_q = tr[A C_q A S] / (x^T S x): the draws at which the expected relative error
in the metric norm is 1.  Reported for q = p (Born) and q = guided (0.8 p + 0.2 p_flat), plus d_eff =
tr[S (S + lambda)^-1] and exp(H[p]).
Memory: rows are processed in chunks of --chunk representatives in two passes (pass 1: S, f; pass 2: C_q), so the
14-site sector (2.25 million representatives) fits; rows are computed twice (cost 2x one exact metric build).
Output: results/kappa_<tag>.json.
"""
import argparse
import json
import os
import sys

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ns_run as NS  # noqa: E402
import torch  # noqa: E402

LAM = 1e-3


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--Lx", type=int, required=True)
    pa.add_argument("--Ly", type=int, required=True)
    pa.add_argument("--Ne", type=int, required=True)
    pa.add_argument("--start", default=None)
    pa.add_argument("--theta", default=None, help="evaluate at this theta instead of the start")
    pa.add_argument("--tag", default=None)
    pa.add_argument("--chunk", type=int, default=32768)
    a = pa.parse_args()
    if a.start is None:
        a.start = NS.START12
    c0 = NS.cpu_min()
    ctx = NS.GCtx(a.Lx, a.Ly, a.Ne, a.start)
    th = ctx.th0 if a.theta is None else torch.tensor(np.load(a.theta))
    E, pr, phi0 = ctx.energy(th)
    p = ctx.size * pr * pr
    p = p / p.sum()
    uq = np.nonzero(p > 0)[0]
    pu = p[uq] / p[uq].sum()
    el = (ctx.Hr @ pr)[uq] / pr[uq]
    pf = ctx.size[uq] * np.abs(pr[uq])
    pf = pf / pf.sum()
    wq = {"born": np.ones_like(pu), "guided": pu / (0.8 * pu + 0.2 * pf)}
    P_ = ctx.m.nparam
    chunks = [slice(i, min(i + a.chunk, len(uq))) for i in range(0, len(uq), a.chunk)]
    # pass 1
    Ob, M2, Eb, OE = np.zeros(P_), np.zeros((P_, P_)), float(pu @ el), np.zeros(P_)
    for c in chunks:
        O = NS.T.orows(ctx, th, phi0, pr, uq[c])
        Ob += pu[c] @ O
        M2 += (O * pu[c][:, None]).T @ O
        OE += O.T @ (pu[c] * el[c])
    S = M2 - np.outer(Ob, Ob)
    S = 0.5 * (S + S.T)
    f = OE - Ob * Eb
    del M2
    w_, U = np.linalg.eigh(S)
    A = (U / (w_ + LAM)) @ U.T
    x = A @ f
    xSx = float(x @ S @ x)
    ASA = A @ S @ A
    NS.log(f"pass 1 done ({NS.cpu_min() - c0:.2f} CPU-min)")
    # pass 2
    acc = {k: dict(sg=np.zeros(P_), sw=0.0, sgw=np.zeros(P_), ggw=np.zeros((P_, P_))) for k in wq}
    for c in chunks:
        Oc = NS.T.orows(ctx, th, phi0, pr, uq[c]) - Ob
        r = (el[c] - Eb) - Oc @ x
        G = Oc * r[:, None]
        for k, w in wq.items():
            pw = pu[c] * w[c]
            d = acc[k]
            d["sg"] += pu[c] @ G
            d["sw"] += float(pw.sum())
            d["sgw"] += pw @ G
            d["ggw"] += (G * pw[:, None]).T @ G
    out = dict(Lx=a.Lx, Ly=a.Ly, Ne=a.Ne, P=int(P_), reps=int(len(p)), theta=a.theta or a.start, E=E,
               xSx=xSx, d_eff=float(np.sum(w_ / (w_ + LAM))), expH=float(np.exp(-np.sum(pu * np.log(pu)))))
    for k, d in acc.items():
        gb = d["sg"]
        Cq = d["ggw"] - np.outer(gb, d["sgw"]) - np.outer(d["sgw"], gb) + d["sw"] * np.outer(gb, gb)
        tr = float(np.sum(ASA * Cq))
        out[f"kappa_{k}"] = tr / xSx
        out[f"N_eps0.3_{k}"] = tr / xSx / 0.09
    out["cpu_min"] = NS.cpu_min() - c0
    tag = a.tag or f"S{a.Lx}x{a.Ly}_N{a.Ne}"
    json.dump(out, open(os.path.join(HERE, "results", f"kappa_{tag}.json"), "w"), indent=1)
    NS.log(json.dumps(out))


if __name__ == "__main__":
    main()
