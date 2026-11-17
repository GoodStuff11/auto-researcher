#!/usr/bin/env python3
"""
P5.1 r01 route R1 `nfloor`: does the guided-SR plateau fall with samples per step?  Rule: RULE_B.md.
====================================================================================================

Run from this directory with the lit_env interpreter, thread variables exported before python:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/.conda/envs/lit_env/bin/python nf_run.py [options]

Step (every arm): at the current theta of the 12-site `che` state (open 4x3 Hubbard, 5+5, U=8, t'=-0.25, 2637
parameters), draw N configurations from the guided mixture q = 0.8 |psi|^2 + 0.2 |psi| (R38's sampler, copied from
../P4.1_cheap_su2_r25/guided.py), form the self-normalised weighted S_T and force f, and take
    d = -tau * x / sqrt(x^T S_T x),   x = (S_T + 1e-3 I)^{-1} f
i.e. R21/R38's shifted SR direction at FIXED metric length tau, ALWAYS taken (no held-out step-length selection;
RULE_A's T1a showed that the selection, not the direction, caused most of the 16,384-sample plateau).  The exact
energy is computed after every step for scoring only; it never feeds back.

Options
  --nsamp INT      samples per step N (default 16384).  The arms of RULE_B use 16384 and 65536.
  --tau FLOAT      fixed metric length (default 0.005).  RULE_B uses 0.005 and 0.0025.
  --steps INT      number of steps (default 120); the run resumes from <res>/run.json if present.
  --seed0 INT      base seed; step n draws with seed seed0 + n (default 9000).
  --start {sr, guided60}
                   sr        the SR-trained `che` theta0 (R38's start; default);
                   guided60  the final theta of R38's replica seed0=6000 (R = 0.5355), used only by the
                             pre-flight timing.
  --cpu-cap FLOAT  CPU-minutes; stop before a step that would exceed it (default 600).
  --res DIR        results directory (default results/N<nsamp>_tau<tau>_s<seed0>).
Output: <res>/run.json (per step: exact E, R, dR, sampled E, |dtheta|, CPU-min) and <res>/theta.npy.
Reuses read only: ../P4.1_cheap_su2_r24/trsr.py (orows, SHIFT), ../P4.1_cheap_su2_r19/polish.py (ctx, E0).
"""
import argparse
import json
import os
import sys
import time

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r24"))
import trsr as T  # noqa: E402  (read only; imports polish as T.P)
import torch  # noqa: E402

P = T.P
R25 = os.path.join(TESTS, "P4.1_cheap_su2_r25")


def guided_stats(ctx, th, phi0, pr, N, seed):
    p = ctx.size * pr * pr
    p = p / p.sum()
    pf = ctx.size * np.abs(pr)
    pf = pf / pf.sum()
    q = 0.8 * p + 0.2 * pf
    rg = np.random.default_rng(seed).choice(len(q), size=N, p=q)
    uq, inv = np.unique(rg, return_inverse=True)
    O = T.orows(ctx, th, phi0, pr, uq)[inv]
    w = p[rg] / q[rg]
    w = w / w.sum()
    elg = (ctx.Hr @ pr)[rg] / pr[rg]
    Oc = O - w @ O
    ST = (Oc * w[:, None]).T @ Oc
    f = (Oc * w[:, None]).T @ (elg - w @ elg)
    return 0.5 * (ST + ST.T), f, float(w @ elg), len(uq)


def parse():
    pa = argparse.ArgumentParser()
    pa.add_argument("--nsamp", type=int, default=16384)
    pa.add_argument("--tau", type=float, default=0.005)
    pa.add_argument("--steps", type=int, default=120)
    pa.add_argument("--seed0", type=int, default=9000)
    pa.add_argument("--start", default="sr", choices=["sr", "guided60"])
    pa.add_argument("--cpu-cap", type=float, default=600.0)
    pa.add_argument("--res", default=None)
    a = pa.parse_args()
    if a.res is None:
        a.res = os.path.join(HERE, "results", f"N{a.nsamp}_tau{a.tau}_s{a.seed0}")
    return a


def main():
    a = parse()
    os.makedirs(a.res, exist_ok=True)
    t0 = time.time()
    ctx = P.ctx_for("che")
    E00 = ctx.energy(ctx.th0)[0]
    gap = E00 - P.E0
    th = ctx.th0.clone()
    if a.start == "guided60":
        th = torch.tensor(np.load(os.path.join(R25, "results_guided_s6000", "theta_shift.npy")))
    fn, fth = os.path.join(a.res, "run.json"), os.path.join(a.res, "theta.npy")
    hist = []
    if os.path.exists(fn):
        hist = json.load(open(fn))["steps"]
        th = torch.tensor(np.load(fth))
    E = ctx.energy(th)[0]
    per = None
    while len(hist) < a.steps:
        c0 = P.cpu_min()
        if per is not None and c0 + per > a.cpu_cap:
            P.log(f"stop: next step would exceed the CPU cap ({c0:.1f} + {per:.2f} > {a.cpu_cap}); TRUNCATED")
            break
        n = len(hist) + 1
        _, pr, phi0 = ctx.energy(th)
        ST, f, Es, nu = guided_stats(ctx, th, phi0, pr, a.nsamp, a.seed0 + n)
        w, U = np.linalg.eigh(ST)
        d = T.d_shift(w, U, f, a.tau)[0]
        th = th + torch.tensor(d)
        En = ctx.energy(th)[0]
        rec = dict(E=En, R=(E00 - En) / gap, dR=(E - En) / gap, E_samp=Es, n_unique=nu,
                   dth=float(np.linalg.norm(d)), eps=(En - P.E0) / abs(P.E0))
        E = En
        hist.append(rec)
        per = P.cpu_min() - c0
        rec["cpu_step"] = per
        P.log(f"N {a.nsamp} tau {a.tau} step {n}: R {rec['R']:+.4f} (dR {rec['dR']:+.4f}); unique {nu}; "
              f"|dtheta| {rec['dth']:.3f} [{P.cpu_min():.2f} CPU-min, step {per:.3f}]")
        np.save(fth, th.numpy())
        json.dump(dict(nsamp=a.nsamp, tau=a.tau, seed0=a.seed0, start=a.start, E_theta0=E00, gap=gap, steps=hist,
                       cpu_min=P.cpu_min(), wall=time.time() - t0), open(fn, "w"), indent=1)
    P.log(f"done: {len(hist)} steps, {P.cpu_min():.2f} CPU-min, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
