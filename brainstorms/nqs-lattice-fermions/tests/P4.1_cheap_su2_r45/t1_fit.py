#!/usr/bin/env python3
"""
P4.1 R45 T1 (premise "fit traction"): does supervised infidelity fitting move a 12-site exact-S state that 4000
exact-gradient energy L-BFGS iterations have left crawling?  Rule: RULE_T1.md (written before this run).

System: open 4x3, N_e = 10 (5+5), U = 8, t' = -0.25, (+,+) via P_G, exact S = 0 (Che closed form inside `che`).
Start: che drift4000 seed 0 (eps 3.0585e-4, overlap2 0.99968); at its end energy L-BFGS lowered eps by ~0.18%
per 50 iterations (stored trajectory).
Run: L-BFGS-B (maxcor 30) on loss = -log F, F = <g|P phi>^2 / <P phi|P phi>, g = exact ground state, for
--maxfun function evaluations; every evaluation logs (1-F, eps).  Usage (interactive, <= 10 CPU-min):
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -t 600
    ~/.conda/envs/lit_env/bin/python t1_fit.py --threads 1 --maxfun 40
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--threads", default="1")
ap.add_argument("--maxfun", type=int, default=40)
A = ap.parse_args()
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = A.threads
sys.argv = [sys.argv[0], "--threads", A.threads]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r11"))
import realise as R  # noqa: E402  (read only)

V = R.V
torch.set_num_threads(int(A.threads))
CHE_START = os.path.join(TESTS, "P4.1_cheap_su2_r5/results/L4x3xobcobc_N10_U8_tp-0.25_w16_che_drift4000_seed0")
E0 = -7.827899034271753
c0 = time.process_time()


def cpu():
    return (time.process_time() - c0) / 60


ctx = R.Ctx()
ctx.m = V.Model("che", 4, 3, 10, 16, -0.25, 8.0, (1, 1))
g = ctx.gs / np.linalg.norm(ctx.gs)
dim, CH = ctx.sec.dim, 8192


def phi_of(th):
    ls, ss = [], []
    with torch.no_grad():
        for s0 in range(0, dim, CH):
            l, s = ctx.m.logpsi0(th, ctx.nU[s0:s0 + CH], ctx.nD[s0:s0 + CH])
            ls.append(l)
            ss.append(s)
    l, s = torch.cat(ls), torch.cat(ss)
    return (s * torch.exp(l - l.max())).numpy()


def pfull(phi):
    pr = 0.25 * (ctx.sg[:, ctx.reps] * phi[ctx.idx[:, ctx.reps]]).sum(0)
    return ctx.E @ pr, pr


hist = []


def loss_grad(th_np):
    th = torch.tensor(th_np)
    phi = phi_of(th)
    Pphi, pr = pfull(phi)
    a = float(g @ Pphi)
    n2 = float(Pphi @ Pphi)
    F = a * a / n2
    prn = pr / np.sqrt(np.sum(ctx.size * pr * pr))
    E = float(np.sum(ctx.size * prn * (ctx.Hr @ prn)))
    w = torch.tensor(-(2.0 * g / a - 2.0 * Pphi / n2) * phi)
    thg = th.clone().requires_grad_(True)
    for s0 in range(0, dim, CH):
        l, _ = ctx.m.logpsi0(thg, ctx.nU[s0:s0 + CH], ctx.nD[s0:s0 + CH])
        (w[s0:s0 + CH] * l).sum().backward()
    hist.append(dict(nfev=len(hist) + 1, infid=1 - F, eps=(E - E0) / abs(E0), cpu=cpu()))
    print(json.dumps(hist[-1]), flush=True)
    return -np.log(F), thg.grad.numpy().copy()


th0 = np.load(CHE_START + "_theta.npy")
rec = json.load(open(CHE_START + ".json"))
# checks: start energy reproduces its record; g is P_G-invariant (so <g|P phi> = <g|phi>); FD gradient
phi = phi_of(torch.tensor(th0))
Pphi, _ = pfull(phi)
chk = dict(gsym=float(np.abs(pfull(g)[0] - g).max()),
           gphi=float(abs(g @ Pphi - g @ phi) / abs(g @ phi)))
f0, g0 = loss_grad(th0)
rng = np.random.default_rng(0)
fd = []
for j in rng.choice(len(th0), 3, replace=False):
    e = np.zeros(len(th0))
    e[j] = 1e-5
    fp = loss_grad(th0 + e)[0]
    fm = loss_grad(th0 - e)[0]
    fd.append(abs((fp - fm) / 2e-5 - g0[j]) / np.abs(g0).max())
chk["grad_fd_rel"] = float(max(fd))
chk["start_eps_vs_record"] = abs(hist[0]["eps"] - rec["eps"])
chk["start_infid_vs_record"] = abs(hist[0]["infid"] - (1 - rec["overlap2"]))
print("checks", json.dumps(chk), flush=True)
start = dict(hist[0])
hist.clear()
res = minimize(loss_grad, th0, jac=True, method="L-BFGS-B",
               options=dict(maxcor=30, maxfun=A.maxfun, maxiter=A.maxfun, ftol=1e-15, gtol=1e-12))
best = min(hist, key=lambda h: h["infid"])
out = dict(start=start, checks=chk, best=best, final=hist[-1], n=len(hist),
           infid_drop=1 - best["infid"] / start["infid"], eps_drop_at_best=1 - best["eps"] / start["eps"],
           eps_drop_min=1 - min(h["eps"] for h in hist) / start["eps"], cpu_min=cpu(), hist=hist)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "results", "t1_fit.json"), "w"), indent=1)
print("RESULT", json.dumps({k: v for k, v in out.items() if k != "hist"}), flush=True)
