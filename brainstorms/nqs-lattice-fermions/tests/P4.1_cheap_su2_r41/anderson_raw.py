#!/usr/bin/env python3
"""
P4.1 round 26, R41: type-II Anderson mixing on the raw fixed-cutoff LM direction (`che`; RULE.md).
=================================================================================================

Run (single thread, hard CPU cap):
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; ulimit -t 600
    ~/.conda/envs/lit_env/bin/python anderson_raw.py [--builds N] [--mmax M]

Map G(x) = x + beta d(x): d = LM direction from one exact S, H build at fixed cutoff 1e-7, |z| <= 0.05, beta = 1/4.
After build k, candidates G(x_k) and x_AA(m) = G(x_k) - DeltaG gamma, gamma = lstsq(DeltaF, f_k), m = 1..min(k, mmax);
lowest exact energy accepted. Writes results/anderson_raw_che.json after every build.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--builds", type=int, default=6)
ap.add_argument("--mmax", type=int, default=3)
A = ap.parse_args()
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
sys.argv = [sys.argv[0], "--threads", "1"]

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r19"))
import polish as P  # noqa: E402

R, log, E0 = P.R, P.log, P.E0
TAU, CUT, BETA, MIN_GAIN, GAMMA_LM = 0.05, 1e-7, 0.25, 1e-7, 7.27e-6
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "anderson_raw_che.json")
c0 = time.process_time()
cpu = lambda: (time.process_time() - c0) / 60  # noqa: E731
eps = lambda E: (E - E0) / abs(E0)  # noqa: E731


def raw_dir(ctx, th, E):
    Pn = ctx.m.nparam
    Eb, Y, HY, pr = P.build(ctx, th)
    assert abs(Eb - E) < 1e-9, (Eb, E)
    S, Hm = P.exact_SH(Y, HY)
    del Y, HY, pr
    tan = R.Tan(S, Hm, E, list(range(Pn)))
    del S, Hm
    r = tan.reduce(CUT)
    z, lam = tan.candidate(r, "lm", TAU)
    return np.asarray(tan.dtheta(r, z, Pn), dtype=float), int(r["rank"])


def summarise(out, t0):
    B = out["builds"]
    if not B:
        return
    eps_end = B[-1]["eps_new"]
    span = B[-1]["cpu_total"] - t0
    out.update(eps_end=eps_end, span_cpu=span, gamma=(out["eps_start"] - eps_end) / span)
    out["ratio"] = out["gamma"] / GAMMA_LM
    if len(B) >= 2:
        g1 = (B[0]["eps_new"] - eps_end) / (B[-1]["cpu_total"] - B[0]["cpu_total"])
        out.update(gamma_late=g1, ratio_late=g1 / GAMMA_LM)
    out["n_AA_accepted"] = sum(b["choice"].startswith("AA") for b in B)
    out["verdict"] = ("TOO-COSTLY" if len(B) < 3 else
                      "KILL" if out["ratio"] < 1.5 else "PASS-T1" if out["ratio"] >= 3 else "AMBIGUOUS")


def main():
    ctx = P.ctx_for("che")
    x = torch.tensor(np.load(os.path.join(TESTS, "P4.1_cheap_su2_r22/results/theta_che.npy")))
    E = ctx.energy(x)[0]
    out = dict(eps_start=eps(E), valid=abs(eps(E) - 7.217364129368478e-4) < 1e-9, setup_cpu=cpu(), builds=[],
               cut=CUT, beta=BETA, tau=TAU)
    log(f"start eps {eps(E):.6e} (valid {out['valid']}) [set-up {cpu():.2f} CPU-min]")
    if not out["valid"]:
        out["verdict"] = "INVALID"
        json.dump(out, open(OUT, "w"), indent=1)
        return
    t0 = cpu()
    X, F, D = [], [], []
    for k in range(A.builds):
        s0 = cpu()
        d, rank = raw_dir(ctx, x, E)
        tb = cpu() - s0
        xk = x.numpy().copy()
        f = BETA * d
        X.append(xk); F.append(f); D.append(d)
        cos_prev = float(d @ D[-2] / np.linalg.norm(d) / np.linalg.norm(D[-2])) if k else None
        g = xk + f
        Eg = ctx.energy(torch.tensor(g))[0]
        rec = dict(k=k, eps_x=eps(E), rank=rank, raw_norm=float(np.linalg.norm(d)), cos_prev=cos_prev,
                   eps_plain=eps(Eg), build_cpu=tb, cands=[])
        bestE, bestx, choice = Eg, g, "plain"
        for m in range(1, min(k, A.mmax) + 1):
            dF = np.stack([F[j + 1] - F[j] for j in range(k - m, k)], 1)
            dX = np.stack([X[j + 1] - X[j] for j in range(k - m, k)], 1)
            gam = np.linalg.lstsq(dF, f, rcond=1e-10)[0]
            xa = g - (dX + dF) @ gam
            Ea = ctx.energy(torch.tensor(xa))[0]
            rec["cands"].append(dict(m=m, gamma=gam.tolist(), eps=eps(Ea),
                                     step_norm=float(np.linalg.norm(xa - xk)), plain_norm=float(np.linalg.norm(f))))
            if np.isfinite(Ea) and Ea < bestE:
                bestE, bestx, choice = Ea, xa, f"AA m={m}"
        if (E - bestE) / abs(E0) <= MIN_GAIN:
            xh = xk + 0.5 * f
            Eh = ctx.energy(torch.tensor(xh))[0]
            rec["eps_half"] = eps(Eh)
            if (E - Eh) / abs(E0) > MIN_GAIN:
                bestE, bestx, choice = Eh, xh, "half"
            else:
                choice = "STALLED"
        rec.update(choice=choice, eps_new=eps(bestE) if choice != "STALLED" else eps(E),
                   cpu_total=cpu())
        rec["gain"] = rec["eps_x"] - rec["eps_new"]
        out["builds"].append(rec)
        log(f"build {k}: eps {eps(E):.6e} plain {eps(Eg):.6e} rank {rank} |d| {rec['raw_norm']:.3f} cos_prev {cos_prev}; "
            + " ".join(f"m{c['m']}:{c['eps']:.6e}" for c in rec["cands"])
            + f"; accepted {choice} eps {rec['eps_new']:.6e} [build {tb:.2f}, total {cpu():.2f} CPU-min]")
        if choice == "STALLED":
            out["stop"] = "STALLED"
            summarise(out, t0)
            json.dump(out, open(OUT, "w"), indent=1)
            break
        x, E = torch.tensor(bestx), bestE
        np.save(os.path.join(RES, "theta_anderson_raw.npy"), bestx)
        summarise(out, t0)
        json.dump(out, open(OUT, "w"), indent=1)
        log(f"running: gamma {out['gamma']:.3e} = {out['ratio']:.2f} gamma_LM; verdict-so-far {out['verdict']}")


if __name__ == "__main__":
    main()
