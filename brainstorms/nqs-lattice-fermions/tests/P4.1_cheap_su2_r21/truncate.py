#!/usr/bin/env python3
"""
P4.1 r21 item 1 (R20 `abl-rank`, step 1): zero-step rank truncation of pcg's pairing matrix f0.
================================================================================================

Run with the lit_env interpreter:
    ~/.conda/envs/lit_env/bin/python truncate.py [--threads INT (default 4)]

No other options. Loads the r19-polished pcg parameters (../P4.1_cheap_su2_r19/results/theta_pcg.npy), checks that
their exact energy reproduces r19's recorded step-8 energy (|diff| < 1e-8, else INVALID), replaces f0 by its best
rank-k approximation (truncated SVD) for k = 5 (the rule) and, for context only, k = 4, 6, 8, 10, 12, keeps every
other parameter fixed, and evaluates the exact projected energy (sum over all 627,264 configurations) in pcg's own
code (vmc_t2.DetModel via r19 polish.ctx_for). Rule (RULE.md): KILL R2's pairing mechanism at T1 if
eps_trunc(k=5) <= 1.10 eps_pcg; otherwise go to step 2. Writes results/truncate.json.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--threads", default="4")
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

E0 = P.E0
c0, t0 = time.process_time(), time.time()
ctx = P.ctx_for("pcg")
seq = json.load(open(os.path.join(R19, "results/seq_pcg.json")))
th = torch.tensor(np.load(os.path.join(R19, "results/theta_pcg.npy")))
E = ctx.energy(th)[0]
Erec = seq["steps"][-1]["E_new"]
valid = abs(E - Erec) < 1e-8
eps = (E - E0) / abs(E0)
P.log(f"pcg r19 N8: E {E:.10f} recorded {Erec:.10f} |diff| {abs(E - Erec):.1e} VALID {valid}; eps {eps:.5e}")
m = ctx.m
off = int(np.cumsum([0] + m.sizes)[[n for n, _ in m.shapes].index("f0")])
L = m.L
f0 = th[off:off + L * L].numpy().reshape(L, L)
sv = np.linalg.svd(f0, compute_uv=False)
P.log("f0 singular values (relative to s1): " + " ".join(f"{x:.3e}" for x in sv / sv[0]))
U_, s_, Vt = np.linalg.svd(f0)
out = dict(E_pcg=E, E_recorded=Erec, valid=bool(valid), eps_pcg=eps, sv=sv.tolist(), trunc={})
for k in (5, 4, 6, 8, 10, 12):
    fk = (U_[:, :k] * s_[:k]) @ Vt[:k]
    tk = th.clone()
    tk[off:off + L * L] = torch.tensor(fk.ravel())
    Ek = ctx.energy(tk)[0]
    ek = (Ek - E0) / abs(E0)
    out["trunc"][k] = dict(E=Ek, eps=ek, ratio=ek / eps, frob_rel=float(np.linalg.norm(f0 - fk) / np.linalg.norm(f0)))
    P.log(f"rank {k:2d}: E {Ek:.8f} eps {ek:.5e} eps/eps_pcg {ek / eps:.3f} |f0-fk|/|f0| "
          f"{out['trunc'][k]['frob_rel']:.3e}")
r5 = out["trunc"][5]["ratio"]
out["verdict"] = ("INVALID" if not valid else
                  "KILL (pairing mechanism, T1)" if r5 <= 1.10 else "PASS to step 2")
out["cpu_min"] = (time.process_time() - c0) / 60
out["wall_s"] = time.time() - t0
P.log(f"RULE: eps_trunc(5)/eps_pcg = {r5:.3f} vs 1.10 -> {out['verdict']}  [{out['cpu_min']:.2f} CPU-min, "
      f"{out['wall_s']:.0f} s wall]")
json.dump(out, open(os.path.join(HERE, "results/truncate.json"), "w"), indent=1)
