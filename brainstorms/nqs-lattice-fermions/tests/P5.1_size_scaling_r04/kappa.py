#!/usr/bin/env python3
"""
P5.1 round 4, route R51.10 `kappa`: exact noise-to-signal ratio of the sampled SR step (ridge-regression sample
complexity), computed from exact sums at one state.  Rule: RULE_kappa.md.
    ~/.conda/envs/lit_env/bin/python kappa.py --Lx LX --Ly LY --Ne NE [--start FILE | --theta FILE] [--tag TAG]

At theta, with p = |psi|^2 over orbit representatives, O_c = O - <O>, e = E_loc - E, lambda = 1e-3 (R51.1's shift):
  S = <O_c O_c^T>_p,  f = <O_c e>_p,  x = (S + lambda)^-1 f   (the exact step direction before length scaling).
A sampled step from N i.i.d. draws from q (weights w = p/q) has, to first order in 1/N,
  E || x_N - x ||_S^2 = (1/N) tr[ A C_q A S ],   A = (S + lambda)^-1,   C_q = <w (O_c e - f)(O_c e - f)^T>_p
(the force fluctuation dominates; the metric fluctuation's first-order term is added with the same formula on
 the residual r = O_c (e - O_c^T x), i.e. C_q is built from O_c r, the regression residual, which is the exact
 first-order error of the ridge solution).  kappa_q = tr[A C_q A S] / (x^T S x) is the number of samples at which
the step's expected relative error in the metric norm is 1; N_eps = kappa / eps^2.  Reported for q = p (Born) and
q = guided (0.8 p + 0.2 p_flat), plus d_eff = tr[S (S + lambda)^-1] and the effective support exp(H[p]).
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
    O = NS.T.orows(ctx, th, phi0, pr, uq)
    el = (ctx.Hr @ pr)[uq] / pr[uq]
    Oc = O - pu @ O
    e = el - pu @ el
    S = (Oc * pu[:, None]).T @ Oc
    S = 0.5 * (S + S.T)
    f = Oc.T @ (pu * e)
    w_, U = np.linalg.eigh(S)
    A = (U / (w_ + LAM)) @ U.T
    x = A @ f
    xSx = float(x @ S @ x)
    r = e - Oc @ x                      # regression residual of E_loc on O at the ridge solution
    G = Oc * r[:, None]                 # per-sample first-order error contributions (mean zero up to lambda x)
    G = G - pu @ G
    pf = ctx.size[uq] * np.abs(pr[uq])
    pf = pf / pf.sum()
    out = dict(Lx=a.Lx, Ly=a.Ly, Ne=a.Ne, P=int(ctx.m.nparam), reps=int(len(p)), theta=a.theta or a.start, E=E,
               xSx=xSx, d_eff=float(np.sum(w_ / (w_ + LAM))), expH=float(np.exp(-np.sum(pu * np.log(pu)))))
    ASA = A @ S @ A
    for name, q in (("born", pu), ("guided", 0.8 * pu + 0.2 * pf)):
        wq = pu / q
        Cq = (G * (pu * wq)[:, None]).T @ G
        tr = float(np.sum(ASA * Cq))
        out[f"kappa_{name}"] = tr / xSx
        out[f"N_eps0.3_{name}"] = tr / xSx / 0.09
    out["cpu_min"] = NS.cpu_min() - c0
    tag = a.tag or f"S{a.Lx}x{a.Ly}_N{a.Ne}"
    json.dump(out, open(os.path.join(HERE, "results", f"kappa_{tag}.json"), "w"), indent=1)
    NS.log(json.dumps(out))


if __name__ == "__main__":
    main()
