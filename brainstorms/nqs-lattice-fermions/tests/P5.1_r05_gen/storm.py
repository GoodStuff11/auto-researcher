#!/usr/bin/env python3
"""
P5.1 round 5, R51.13 `svrg` multi-step T1 (RULE_storm.md): STORM-type recursive control variate on the SR direction.
    ~/.conda/envs/lit_env/bin/python storm.py --nsamp 2048 --a 0.1 [--steps 120]     (5x2, N_e 8, S10 start)
Same draws, step rule and seeds as R51.8's ns_run.py (guided q = 0.8 p + 0.2 p_flat, shift 1e-3, metric length
tau = 0.005, seeds 9000 + n).  At step n the draws give xhat_n = (S_n + lam)^-1 f_n at theta_n and, re-weighted to
theta_{n-1} (weights p_{n-1}/q_n, rows and local energies at theta_{n-1}), xhat'_{n-1}; the direction is
v_n = xhat_n + (1 - a)(v_{n-1} - xhat'_{n-1})  (STORM, Cutkosky & Orabona 2019), v_1 = xhat_1; the step is
d = -tau v / sqrt(v^T S_n v).  Each draw is evaluated at two parameter sets, so the arm is compared with plain
guided SR at twice its N.  Exact energy every step scores only.  Output: results/storm_N<N>_a<a>/run.json.
"""
import argparse
import json
import os
import sys

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P5.1_size_scaling_r04"))
import ns_run as NS  # noqa: E402
import torch  # noqa: E402

LAM = 1e-3
START10 = os.path.join(HERE, "..", "P5.1_size_scaling_r04", "starts", "S5x2_N8_U8_tp-0.25_pg11_che_seed0_theta.npy")


def solve(ctx, th, phi0, pr, uq, w):
    O = NS.T.orows(ctx, th, phi0, pr, uq)
    el = (ctx.Hr @ pr)[uq] / pr[uq]
    Oc = O - w @ O
    S = (Oc * w[:, None]).T @ Oc
    S = 0.5 * (S + S.T)
    f = (Oc * w[:, None]).T @ (el - w @ el)
    x = np.linalg.solve(S + LAM * np.eye(len(f)), f)
    return S, x, f


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--nsamp", type=int, default=2048)
    pa.add_argument("--a", type=float, default=0.1)
    pa.add_argument("--steps", type=int, default=120)
    pa.add_argument("--tau", type=float, default=0.005)
    pa.add_argument("--seed0", type=int, default=9000)
    pa.add_argument("--mode", default="x", choices=["x", "sf"],
                    help="x: recursion on the solved direction (R51.13); sf: recursion on S and f, then solve (R51.18)")
    a = pa.parse_args()
    c0 = NS.cpu_min()
    ctx = NS.GCtx(5, 2, 8, START10)
    th = ctx.th0.clone()
    E00, pr, phi0 = ctx.energy(th)
    E0 = -6.550697008720983  # results/info_S5x2_N8.json
    prev = None
    v = None
    hist = []
    res = os.path.join(HERE, "results", f"storm_N{a.nsamp}_a{a.a}" + ("_sf" if a.mode == "sf" else ""))
    os.makedirs(res, exist_ok=True)
    for n in range(1, a.steps + 1):
        p = ctx.size * pr * pr
        p = p / p.sum()
        pf = ctx.size * np.abs(pr)
        q = 0.8 * p + 0.2 * pf / pf.sum()
        rg = np.random.default_rng(a.seed0 + n).choice(len(q), size=a.nsamp, p=q)
        uq, inv = np.unique(rg, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(uq)).astype(float)
        w = cnt * p[uq] / q[uq]
        w = w / w.sum()
        S, xh, fh = solve(ctx, th, phi0, pr, uq, w)
        if prev is None:
            v = xh
            Sr, fr = S, fh
        else:
            thp, prp, phip = prev
            pp = ctx.size * prp * prp
            pp = pp / pp.sum()
            wp = cnt * pp[uq] / q[uq]
            wp = wp / wp.sum()
            Sp, xp, fp = solve(ctx, thp, phip, prp, uq, wp)
            if a.mode == "x":
                v = xh + (1 - a.a) * (v - xp)
            else:
                Sr = S + (1 - a.a) * (Sr - Sp)
                fr = fh + (1 - a.a) * (fr - fp)
                Sr = 0.5 * (Sr + Sr.T)
                v = np.linalg.solve(Sr + LAM * np.eye(len(fr)), fr)
                S = Sr
        ml = np.sqrt(max(float(v @ S @ v), 1e-300))
        d = -a.tau * v / ml
        prev = (th, pr, phi0)
        th = th + torch.tensor(d)
        En, pr, phi0 = ctx.energy(th)
        hist.append(dict(E=En, R=(E00 - En) / (E00 - E0), n_unique=int(len(uq))))
        if n % 10 == 0:
            NS.log(f"step {n}: R {hist[-1]['R']:.4f} [{NS.cpu_min() - c0:.2f} CPU-min]")
    R = np.array([h["R"] for h in hist])
    out = dict(mode=a.mode, nsamp=a.nsamp, a=a.a, tau=a.tau, seed0=a.seed0, E_start=E00, E0=E0, steps=hist,
               R_plat=float(R[-20:].mean()), cpu_min=NS.cpu_min() - c0)
    json.dump(out, open(os.path.join(res, "run.json"), "w"), indent=1)
    NS.log(f"done: R_plat {out['R_plat']:.4f}, {out['cpu_min']:.2f} CPU-min")


if __name__ == "__main__":
    main()
