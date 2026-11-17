#!/usr/bin/env python3
"""
P4.1 r21 item 3 (R19 `ladder`, cheapest falsifier): one exact LM step per cutoff from each r19 N8 endpoint.
=========================================================================================================

Run with the lit_env interpreter:
    ~/.conda/envs/lit_env/bin/python probe.py --arm {pcg, che} [--threads INT (1)] [--cpu-cap FLOAT (CPU-min, 6)]

--arm      pcg : the r19-polished R2 state (../P4.1_cheap_su2_r19/results/theta_pcg.npy)
           che : the r19-polished control (theta_che.npy)
--threads  torch/BLAS threads (1, as r19, so CPU time is comparable)
--cpu-cap  the process stops before the line searches if building S, H already used more than this (no partial
           verdict is then written).

Per arm: check that the endpoint's exact energy reproduces r19's recorded step-8 energy (|diff| < 1e-8, else INVALID);
build the exact S, H once (r19 polish.build / exact_SH, sums over all 627,264 configurations via orbit
representatives); for each relative metric cutoff in {1e-5 (control), 1e-6, 1e-7, 1e-8}: Round 11 Tan.reduce,
LM candidate with |z| <= 0.05 (tau), line search s in {1, 1/2, 1/4} on the exact energy (first lowering s, as r19).
Records kept rank, predicted gain, realised relative drop of eps, and the fraction of the tangent-space energy
gradient outside the kept subspace (Euclidean, and natural-gradient sum_i (u_i.g)^2 / w_i over eigenvalues above
1e-9 * max, the P4.2 check).
Rule (RULE.md, R19, per arm): FROZEN-ARTEFACT if some cutoff below 1e-5 lowers eps by at least 3% in that one step
and beats the control step by at least 2 percentage points; GENUINE if no cutoff lowers eps by more than 1%;
UNCLEAR otherwise. Writes results/probe_<arm>.json.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["pcg", "che"])
ap.add_argument("--threads", default="1")
ap.add_argument("--cpu-cap", type=float, default=6.0)
a = ap.parse_args()
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = a.threads
sys.argv = [sys.argv[0], "--threads", a.threads]

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R19 = os.path.join(os.path.dirname(HERE), "P4.1_cheap_su2_r19")
sys.path.insert(0, R19)
import polish as P  # noqa: E402  (read only)

R = P.R
E0, TAU = P.E0, 0.05
CUTS = [1e-5, 1e-6, 1e-7, 1e-8]
c0, t0 = time.process_time(), time.time()
cpu = lambda: (time.process_time() - c0) / 60  # noqa: E731
ctx = P.ctx_for(a.arm)
seq = json.load(open(os.path.join(R19, f"results/seq_{a.arm}.json")))
th = torch.tensor(np.load(os.path.join(R19, f"results/theta_{a.arm}.npy")))
Erec = seq["steps"][-1]["E_new"]
E, Y, HY, pr = P.build(ctx, th)
valid = abs(E - Erec) < 1e-8
eps = (E - E0) / abs(E0)
gap = seq["gap"]
P.log(f"{a.arm} r19 N8: E {E:.10f} recorded {Erec:.10f} |diff| {abs(E - Erec):.1e} VALID {valid}; eps {eps:.5e} "
      f"[{cpu():.2f} CPU-min]")
S, Hm = P.exact_SH(Y, HY)
del Y, HY
Pn = ctx.m.nparam
tan = R.Tan(S, Hm, E, list(range(Pn)))
if cpu() > a.cpu_cap:
    sys.exit(f"stop: S, H build used {cpu():.1f} > {a.cpu_cap} CPU-min; no verdict")
w, U = tan.w, tan.U
gu = U.T @ tan.g
ng_ok = w > 1e-9 * w.max()
ng_tot = float(np.sum(gu[ng_ok] ** 2 / w[ng_ok]))
rows = []
for c in CUTS:
    r = tan.reduce(c)
    k = w > c * w.max()
    z, lam = tan.candidate(r, "lm", TAU)
    pred = tan.pred(r, z)
    d = tan.dtheta(r, z, Pn)
    s, Es, trials = R.line_search(ctx, th, d, E)
    eps_new = (Es - E0) / abs(E0) if s is not None else eps
    row = dict(cut=c, rank=r["rank"], zn=float(np.linalg.norm(z)), lam=lam, pred_gain_gap=(E - pred) / gap,
               pred_drop=(E - pred) / (E - E0), s=s, E_new=Es, eps_new=eps_new, drop=1 - eps_new / eps,
               trials=[(t, (e - E0) / abs(E0)) for t, e in trials],
               g_out_euclid=float(np.sum(gu[~k] ** 2) / np.sum(gu ** 2)),
               g_out_natural=float(np.sum(gu[ng_ok & ~k] ** 2 / w[ng_ok & ~k]) / ng_tot))
    rows.append(row)
    P.log(f"cut {c:.0e}: rank {row['rank']}; |z| {row['zn']:.3f}; predicted drop of eps {row['pred_drop']:.4f}; "
          f"s {s}; eps {eps:.5e} -> {eps_new:.5e} (drop {100 * row['drop']:.2f}%); grad outside kept: euclid "
          f"{row['g_out_euclid']:.3e}, natural {row['g_out_natural']:.3e} [{cpu():.2f} CPU-min]")
ctrl = rows[0]["drop"]
frozen = any(r["drop"] >= 0.03 and r["drop"] - ctrl >= 0.02 for r in rows[1:])
genuine = all(r["drop"] <= 0.01 for r in rows)
v = "INVALID" if not valid else "FROZEN-ARTEFACT" if frozen else "GENUINE" if genuine else "UNCLEAR"
out = dict(arm=a.arm, valid=bool(valid), E=E, eps=eps, gap=gap, metric_wmax=float(w.max()), rows=rows,
           verdict=v, cpu_min=cpu(), wall_s=time.time() - t0)
P.log(f"RULE ({a.arm}): control drop {100 * ctrl:.2f}%; best ladder drop "
      f"{100 * max(r['drop'] for r in rows[1:]):.2f}% -> {v} [{cpu():.2f} CPU-min, {time.time() - t0:.0f} s wall]")
json.dump(out, open(os.path.join(HERE, f"results/probe_{a.arm}.json"), "w"), indent=1)
