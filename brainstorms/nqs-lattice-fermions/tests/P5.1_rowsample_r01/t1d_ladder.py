#!/usr/bin/env python3
"""
P5.1 r02 T1d baseline arm: the exact cutoff ladder (P4.1 r22 ladder.py protocol) from R51.3's T1c endpoint.
Rule: RULE_T1d.md.

Run from this directory, thread variables exported before python and a hard CPU limit set:
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; ulimit -t 600
    ~/.conda/envs/lit_env/bin/python t1d_ladder.py [--cpu-cap 9.4]
Start: results/t1c_theta.npy.  Output: results/t1d_ladder.json.  Reuses read only ../P4.1_cheap_su2_r22/ladder.py
(its constants and its r19 polish / r11 realise imports); the step body below is run_arm's, unchanged.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
cap = float(sys.argv[sys.argv.index("--cpu-cap") + 1]) if "--cpu-cap" in sys.argv else 9.4
sys.argv = [sys.argv[0], "--mode", "verdict", "--threads", "1"]  # ladder.py parses argv at import
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r22"))
import ladder as L  # noqa: E402

np, torch, P, R = L.np, L.torch, L.P, L.R
eps = lambda E: (E - L.E0) / abs(L.E0)  # noqa: E731


def main():
    ctx = P.ctx_for("che")
    th = torch.tensor(np.load(os.path.join(HERE, "results", "t1c_theta.npy")))
    E = ctx.energy(th)[0]
    out = dict(rule="RULE_T1d.md", eps_start=eps(E), setup_cpu=L.cpu(), steps=[], stop=None)
    L.log(f"start eps {eps(E):.5e}; setup {L.cpu():.2f} CPU-min (not charged)")
    Pn, per = ctx.m.nparam, None
    while True:
        s0 = L.cpu()
        if per is not None and s0 + per > cap:
            out["stop"] = "CPU-CAP"
            break
        Eb, Y, HY, pr = P.build(ctx, th)
        S, Hm = P.exact_SH(Y, HY)
        del Y, HY, pr
        tan = R.Tan(S, Hm, E, list(range(Pn)))
        del S, Hm
        best, prev = None, None
        for c in L.LADDER:
            r = tan.reduce(c)
            if r["rank"] == prev:
                continue
            prev = r["rank"]
            z, lam = tan.candidate(r, "lm", L.TAU)
            d = tan.dtheta(r, z, Pn)
            for s in (1.0, 0.5, 0.25):
                Es = ctx.energy(th + s * torch.tensor(d))[0]
                if np.isfinite(Es) and (E - Es) / abs(L.E0) > L.MIN_GAIN:
                    if best is None or Es < best["E"]:
                        best = dict(E=Es, s=s, d=d, cut=c)
                    break
        per = L.cpu() - s0
        if best is None:
            out["steps"].append(dict(accepted=False, cpu_min=per))
            out["stop"] = "STOP-NOGAIN"
            L.log(f"STOP-NOGAIN [{per:.2f}]")
            break
        th = th + best["s"] * torch.tensor(best["d"])
        out["steps"].append(dict(accepted=True, eps=eps(E), eps_new=eps(best["E"]), cut=best["cut"], s=best["s"],
                                 cpu_min=per))
        E = best["E"]
        L.log(f"step {len(out['steps'])}: cut {best['cut']:.0e} s {best['s']} eps -> {eps(E):.5e} "
              f"[step {per:.2f}, process {L.cpu():.2f} CPU-min]")
        json.dump(out, open(os.path.join(HERE, "results", "t1d_ladder.json"), "w"), indent=1)
    acc = [s for s in out["steps"] if s["accepted"]]
    if acc:
        out["eps_L"] = acc[-1]["eps_new"]
        out["cpu_steps"] = sum(s["cpu_min"] for s in acc)
        out["gamma_L"] = (out["eps_start"] - out["eps_L"]) / out["cpu_steps"]
        L.log(f"ladder: eps {out['eps_start']:.5e} -> {out['eps_L']:.5e} in {len(acc)} steps, "
              f"{out['cpu_steps']:.2f} CPU-min; gamma_L {out['gamma_L']:.3e} ({out['gamma_L'] / 7.27e-6:.2f} gamma_LM)")
    np.save(os.path.join(HERE, "results", "t1d_ladder_theta.npy"), th.numpy())
    json.dump(out, open(os.path.join(HERE, "results", "t1d_ladder.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
