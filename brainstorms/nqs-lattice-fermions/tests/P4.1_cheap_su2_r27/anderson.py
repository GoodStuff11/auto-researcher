#!/usr/bin/env python3
"""
P4.1 round 23, R30: safeguarded Anderson (type-II) mixing over exact ladder-LM steps on `che` (RULE_R30.md).
===========================================================================================================

Run with the lit_env interpreter (single thread):
    ~/.conda/envs/lit_env/bin/python anderson.py [--builds N] [--cpu-cap MIN] [--mmax M]

Options
  --builds INT     number of ladder builds (map evaluations G) to run (default 3, RULE_R30.md)
  --cpu-cap FLOAT  total CPU-minute cap including set-up (default 10, RULE_R30.md HARD LIMIT); before each
                   build the script stops if elapsed CPU + the last build's CPU would exceed it
  --mmax INT       largest Anderson history depth tried (default 4; with 3 builds only m <= 2 is reachable)

Start: ../P4.1_cheap_su2_r22/results/theta_che.npy (job 742206 endpoint). One map G(x) = one ladder step, copied
from ../P4.1_cheap_su2_r22/ladder.py (exact S, H build, tangent eigendecomposition, cutoffs 1e-5..1e-9, LM
|z| <= 0.05, line search s in {1, 1/2, 1/4}, lowest exact energy over rungs; G(x) = x if no rung gains > 1e-7).
After build k: f_k = G(x_k) - x_k; for m = 1..min(k, mmax), x_AA(m) = G(x_k) - DeltaG gamma with
gamma = lstsq(DeltaF, f_k) over the last m differences; x_{k+1} = lowest exact-energy candidate if below
E(G(x_k)), else G(x_k). Writes results/anderson_che.json (trajectory, CPU, verdict) and results/theta_anderson.npy.
"""
import argparse
import json
import os
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", type=int, default=3)
    ap.add_argument("--cpu-cap", type=float, default=10.0)
    ap.add_argument("--mmax", type=int, default=4)
    return ap.parse_args()


A = parse_args()
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
TAU, MIN_GAIN = 0.05, 1e-7
LADDER = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
GAMMA_LM, R_LM = 7.27e-6, 0.975
RES = os.path.join(HERE, "results")
c0 = time.process_time()
cpu = lambda: (time.process_time() - c0) / 60  # noqa: E731
eps = lambda E: (E - E0) / abs(E0)  # noqa: E731


def ladder_step(ctx, th, E):
    """one ladder-LM map evaluation; returns (G(th), E(G(th)), info)."""
    Pn = ctx.m.nparam
    Eb, Y, HY, pr = P.build(ctx, th)
    assert abs(Eb - E) < 1e-9, (Eb, E)
    S, Hm = P.exact_SH(Y, HY)
    del Y, HY, pr
    tan = R.Tan(S, Hm, E, list(range(Pn)))
    del S, Hm
    best, prev = None, None
    for c in LADDER:
        r = tan.reduce(c)
        if r["rank"] == prev:
            continue
        prev = r["rank"]
        z, lam = tan.candidate(r, "lm", TAU)
        d = tan.dtheta(r, z, Pn)
        for s in (1.0, 0.5, 0.25):
            Es = ctx.energy(th + s * torch.tensor(d))[0]
            if np.isfinite(Es) and (E - Es) / abs(E0) > MIN_GAIN:
                if best is None or Es < best[1]:
                    best = (c, Es, s, d)
                break
    if best is None:
        return th.clone(), E, dict(cut=None, s=None)
    c, Es, s, d = best
    return th + s * torch.tensor(d), Es, dict(cut=c, s=s)


def main():
    ctx = P.ctx_for("che")
    x = torch.tensor(np.load(os.path.join(TESTS, "P4.1_cheap_su2_r22/results/theta_che.npy")))
    E = ctx.energy(x)[0]
    eps_start = eps(E)
    valid = abs(eps_start - 7.217364129368478e-4) < 1e-9
    log(f"start eps {eps_start:.6e} (valid {valid}) [set-up {cpu():.2f} CPU-min]")
    out = dict(eps_start=eps_start, valid=valid, setup_cpu=cpu(), builds=[], stop=None)
    if not valid:
        out["verdict"] = "INVALID"
        json.dump(out, open(os.path.join(RES, "anderson_che.json"), "w"), indent=1)
        return
    t_build0, per = cpu(), None
    X, G = [], []  # iterates x_k and their images G(x_k)
    for k in range(A.builds):
        if per is not None and cpu() + per > A.cpu_cap:
            out["stop"] = f"CPU cap before build {k} ({cpu():.2f} + {per:.2f} > {A.cpu_cap})"
            log(out["stop"])
            break
        s0 = cpu()
        g, Eg, info = ladder_step(ctx, x, E)
        per = cpu() - s0
        X.append(x.numpy().copy())
        G.append(g.numpy().copy())
        rec = dict(k=k, eps_x=eps(E), eps_G=eps(Eg), plain_gain=eps(E) - eps(Eg), build_cpu=per, **info, cands=[])
        bestE, bestx, choice = Eg, g, "plain"
        F = [G[j] - X[j] for j in range(len(X))]
        for m in range(1, min(k, A.mmax) + 1):
            dF = np.stack([F[j + 1] - F[j] for j in range(k - m, k)], 1)
            dG = np.stack([G[j + 1] - G[j] for j in range(k - m, k)], 1)
            gam = np.linalg.lstsq(dF, F[k], rcond=1e-10)[0]
            xa = torch.tensor(G[k] - dG @ gam)
            Ea = ctx.energy(xa)[0]
            rec["cands"].append(dict(m=m, gamma=gam.tolist(), eps=eps(Ea),
                                     step_norm=float(np.linalg.norm(xa.numpy() - X[k])),
                                     plain_norm=float(np.linalg.norm(F[k]))))
            if np.isfinite(Ea) and Ea < bestE:
                bestE, bestx, choice = Ea, xa, f"AA m={m}"
        rec.update(choice=choice, eps_new=eps(bestE), gain=eps(E) - eps(bestE), cpu_total=cpu())
        out["builds"].append(rec)
        log(f"build {k}: eps {eps(E):.6e} -> plain {eps(Eg):.6e} (cut {info['cut']}, s {info['s']}); "
            + " ".join(f"m{c['m']}:{c['eps']:.6e}" for c in rec["cands"])
            + f"; accepted {choice} eps {eps(bestE):.6e} [build {per:.2f}, total {cpu():.2f} CPU-min]")
        x, E = bestx, bestE
        np.save(os.path.join(RES, "theta_anderson.npy"), x.numpy())
        json.dump(out, open(os.path.join(RES, "anderson_che.json"), "w"), indent=1)
    # verdict (RULE_R30.md)
    B = out["builds"]
    out["cpu_total"] = cpu()
    if len(B) < 3:
        out["verdict"] = "TOO-COSTLY"
    else:
        span = cpu() - t_build0
        eps_end = B[-1]["eps_new"]
        gam = (eps_start - eps_end) / span
        late_cpu = B[-1]["cpu_total"] - B[0]["cpu_total"]
        gam_late = (B[0]["eps_new"] - eps_end) / late_cpu
        d = [b["gain"] for b in B]
        r = (d[2] / d[0]) ** 0.5 if d[0] > 0 and d[2] > 0 else float("inf")
        tail = d[2] * r / (1 - r) if r < 1 else float("inf")
        shrinks = bool(r < R_LM and tail / eps_end < 0.5)
        ratio = gam / GAMMA_LM
        v = "KILL" if ratio < 1.5 else ("PASS-T1" if ratio >= 3 and shrinks else "AMBIGUOUS")
        out.update(eps_end=eps_end, span_cpu=span, gamma=gam, ratio=ratio, gamma_late=gam_late,
                   ratio_late=gam_late / GAMMA_LM, gains=d, r=r, tail=tail, tail_frac=tail / eps_end,
                   tail_shrinks=shrinks, n_AA_accepted=sum(b["choice"] != "plain" for b in B), verdict=v)
        log(f"eps {eps_start:.6e} -> {eps_end:.6e} in {span:.2f} CPU-min: gamma {gam:.3e} = {ratio:.2f} gamma_LM "
            f"(builds 1-2: {gam_late / GAMMA_LM:.2f}); gains {d}; r {r:.3f} tail/eps {tail / eps_end:.3f} "
            f"shrinks {shrinks}; AA accepted {out['n_AA_accepted']}; VERDICT {v} (R28: 1.12)")
    json.dump(out, open(os.path.join(RES, "anderson_che.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
