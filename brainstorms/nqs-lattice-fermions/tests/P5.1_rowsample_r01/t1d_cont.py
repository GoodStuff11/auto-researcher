#!/usr/bin/env python3
"""
P5.1 r02 T1d comparison arm: R51.3 leverage-row steps continued from T1c's endpoint until the ladder's eps_L.
Rule: RULE_T1d.md.  Gated quick job (submit_t1d_cont.sbatch); not run interactively.

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    ~/.conda/envs/lit_env/bin/python -u t1d_cont.py [--max-steps 20]
Needs results/t1c_theta.npy, results/lev0.npy (T1c's scores, reused) and results/t1d_ladder.json (eps_L, gamma_L).
Step protocol = t1c.py's, unchanged (seeds continue at 11).  Output: results/t1d_cont.json with the verdict.
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
import t1c as C  # noqa: E402

P, RS, torch, EG = C.P, C.RS, C.torch, C.EG
SCORE, AMORT = 1.896, 1.896 / 30
RES = C.RES


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--max-steps", type=int, default=20)
    a = pa.parse_args()
    lad = json.load(open(os.path.join(RES, "t1d_ladder.json")))
    eps_L, gam_L = lad["eps_L"], lad["gamma_L"]
    ctx = P.ctx_for("che")
    th = torch.tensor(np.load(os.path.join(RES, "t1c_theta.npy")))
    lev = np.load(os.path.join(RES, "lev0.npy"))
    E, pr, phi0 = ctx.energy(th)
    out = dict(rule="RULE_T1d.md", eps_start=C.eps(E), eps_L=eps_L, gamma_L=gam_L, steps=[], reached=False)
    P.log(f"start eps {C.eps(E):.5e}; target eps_L {eps_L:.5e}; gamma_L {gam_L:.3e}")
    Pn, m, cpu = ctx.m.nparam, 8 * ctx.m.nparam, 0.0
    for k in range(11, 11 + a.max_steps):
        s0 = P.cpu_min()
        _, G = EG.energy_grad(ctx, th)
        f = G / 2
        born = ctx.size * pr * pr
        q = 0.5 * born / born.sum() + 0.5 * lev / lev.sum()
        r = np.random.default_rng(k).choice(len(q), size=m, p=q)
        u, inv = np.unique(r, return_inverse=True)
        Yu = C.rows(ctx, th, phi0, u)
        Ys = np.concatenate([(pr * ctx.sw)[u][:, None], Yu], 1)[inv] / np.sqrt(m * q[r])[:, None]
        del Yu
        St, _ = RS.centred(Ys.T @ Ys, None, E)
        del Ys
        w, U = np.linalg.eigh(St)
        best = None
        for c in C.CUTS:
            kk = w > c * w.max()
            Pm = U[:, kk] / np.sqrt(w[kk])
            gt = Pm.T @ f
            d = -(Pm @ gt) / np.linalg.norm(gt)
            for tau in C.TAUS:
                ev = ctx.energy(th + torch.tensor(tau * d))
                if np.isfinite(ev[0]) and (E - ev[0]) / abs(P.E0) > C.MIN_GAIN:
                    if best is None or ev[0] < best["ev"][0]:
                        best = dict(cut=c, tau=tau, d=d, ev=ev)
                    break
        per = P.cpu_min() - s0
        cpu += per
        if best is None:
            out["steps"].append(dict(k=k, accepted=False, cpu_min=per))
            P.log(f"step {k}: STOP-NOGAIN")
            break
        th = th + best["tau"] * torch.tensor(best["d"])
        E, pr, phi0 = best["ev"]
        out["steps"].append(dict(k=k, accepted=True, cut=best["cut"], tau=best["tau"], eps_new=C.eps(E), cpu_min=per))
        P.log(f"step {k}: cut {best['cut']:.0e} tau {best['tau']} eps -> {C.eps(E):.5e} [step {per:.2f}]")
        json.dump(out, open(os.path.join(RES, "t1d_cont.json"), "w"), indent=1)
        if C.eps(E) <= eps_L:
            out["reached"] = True
            break
    n = len(out["steps"])
    drop = out["eps_start"] - C.eps(E)
    out["gamma_R"] = drop / (cpu + AMORT * n)
    out["gamma_R_fullscore"] = drop / (cpu + SCORE)
    out["rho"] = out["gamma_R"] / gam_L
    out["rho_fullscore"] = out["gamma_R_fullscore"] / gam_L
    out["verdict"] = "PASS" if out["rho"] >= 3 else "KILLED" if out["rho"] < 1.5 else "MARGINAL"
    P.log(f"T1d: {n} steps, reached eps_L {out['reached']}; rho {out['rho']:.2f} (full score cost "
          f"{out['rho_fullscore']:.2f}) -> {out['verdict']}")
    json.dump(out, open(os.path.join(RES, "t1d_cont.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
