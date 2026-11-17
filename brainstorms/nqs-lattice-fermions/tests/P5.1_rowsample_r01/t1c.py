#!/usr/bin/env python3
"""
P5.1 r02 T1c for R51.3 `levrow`: 10 polishing steps with a leverage-row metric.  Rule: RULE_T1c.md.
=================================================================================================

Run from this directory, thread variables exported before python and a hard CPU limit set:
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; ulimit -t 600
    ~/.conda/envs/lit_env/bin/python t1c.py [--steps 10] [--cpu-cap 9.3]
Options
  --steps INT      total steps (default 10, RULE_T1c.md)
  --cpu-cap FLOAT  process CPU-min of THIS invocation; before each step stop (resumable) if elapsed + the last
                   step's cost would exceed it (default 9.3, under ulimit -t 600)
Resumes from results/t1c_state.json, results/t1c_theta.npy and results/lev0.npy if they exist.
Output: results/t1c_state.json (per-step record, charged CPU, verdict).  Log to stdout.
Reuses read only: rowsample.py (jac, centred, P = r19 polish), ../P4.1_cheap_su2_r25/exgrad.py (energy_grad).
"""
import argparse
import json
import os
import sys

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r25"))
import rowsample as RS  # noqa: E402
import exgrad as EG  # noqa: E402

P, torch = RS.P, RS.torch
CUTS, TAUS, MIN_GAIN = (1e-5, 1e-6, 1e-7), (0.002, 0.001, 0.0005), 1e-7
GLM = 7.27e-6
RES = os.path.join(HERE, "results")
FS, FT, FL = (os.path.join(RES, f) for f in ("t1c_state.json", "t1c_theta.npy", "lev0.npy"))
eps = lambda E: (E - P.E0) / abs(P.E0)  # noqa: E731


def rows(ctx, th, phi0, u):
    """Jacobian rows [psi, d psi/d theta] * sqrt(orbit size) for representative indices u only (as RS.jac)."""
    Pn = ctx.m.nparam
    out = np.empty((len(u), Pn))
    for r0 in range(0, len(u), 2048):
        rr = ctx.reps[u[r0:r0 + 2048]]
        acc = np.zeros((len(rr), Pn))
        for g in range(4):
            ix = ctx.idx[g, rr]
            w = 0.25 * ctx.sg[g, rr] * phi0[ix]
            acc += w[:, None] * P.logder(ctx.m, th, ctx.nU[ix], ctx.nD[ix])
        out[r0:r0 + 2048] = acc * ctx.sw[u[r0:r0 + 2048], None]
    return out


def verdict(st):
    acc = [s for s in st["steps"] if s.get("accepted")]
    e_end = acc[-1]["eps_new"] if acc else st["eps_start"]
    g = (st["eps_start"] - e_end) / st["cpu_charged"]
    g_noscore = (st["eps_start"] - e_end) / (st["cpu_charged"] - st["cpu_scores"])
    v = "PASS" if g >= 3 * GLM else "KILLED" if g < 1.5 * GLM else "MARGINAL"
    done = len(st["steps"]) >= st["cap"] or st["stop"] == "STOP-NOGAIN"
    return dict(gamma=g, ratio=g / GLM, ratio_noscore=g_noscore / GLM, verdict=v if done else f"PRELIMINARY {v}",
                eps_end=e_end)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--steps", type=int, default=10)
    pa.add_argument("--cpu-cap", type=float, default=9.3)
    a = pa.parse_args()
    ctx = P.ctx_for("che")
    c_setup = P.cpu_min()
    P.log(f"setup {c_setup:.2f} CPU-min (not charged)")
    if os.path.exists(FS):
        st = json.load(open(FS))
        th = torch.tensor(np.load(FT))
        lev = np.load(FL)
        P.log(f"resuming at step {len(st['steps'])}, charged {st['cpu_charged']:.2f} CPU-min")
    else:
        th = torch.tensor(np.load(os.path.join(TESTS, "P4.1_cheap_su2_r22", "results", "theta_che.npy")))
        c = P.cpu_min()
        E, pr, Y = RS.jac(ctx, th)
        S = Y.T @ Y
        w, U = np.linalg.eigh(S)
        Z = Y @ (U / np.sqrt(w + RS.LAM_REL * w.max()))
        lev = np.einsum("ij,ij->i", Z, Z)
        del Z, U
        c_sc = P.cpu_min() - c
        np.save(FL, lev)
        P.log(f"leverage scores at start: {c_sc:.3f} CPU-min; sum {lev.sum():.1f}")
        # validity check (not charged): exact-gradient force == full-Jacobian force
        c = P.cpu_min()
        Hc0 = Y.T @ ((ctx.Hr @ pr) * ctx.sw)
        STx, s = RS.centred(S, None, E)
        f_full = Hc0[1:] / S[0, 0] - E * s
        del Y, S, STx
        E2, G = EG.energy_grad(ctx, th)
        chk = float(np.abs(G / 2 - f_full).max() / np.abs(f_full).max())
        c_chk = P.cpu_min() - c
        P.log(f"check: |G/2 - f_full|/|f_full| = {chk:.2e}; E diff {abs(E2 - E):.1e} [{c_chk:.2f} CPU-min, not charged]")
        st = dict(rule="RULE_T1c.md", cap=a.steps, eps_start=eps(E), E_start=E, cpu_scores=c_sc, cpu_charged=c_sc,
                  cpu_check=c_chk, check=chk, valid=chk < 1e-6, steps=[], stop=None)
        if not st["valid"]:
            st["stop"] = "INVALID"
            json.dump(st, open(FS, "w"), indent=1)
            P.log("INVALID")
            return
        json.dump(st, open(FS, "w"), indent=1)
        np.save(FT, th.numpy())
    Pn, m = ctx.m.nparam, 8 * ctx.m.nparam
    per, cache = None, None
    while st["stop"] is None and len(st["steps"]) < st["cap"]:
        s0 = P.cpu_min()
        if per is not None and s0 + per > a.cpu_cap:
            P.log(f"stop: next step would exceed the CPU cap ({s0:.2f} + {per:.2f} > {a.cpu_cap}); resumable")
            break
        k = len(st["steps"]) + 1
        E, pr, phi0 = cache if cache is not None else ctx.energy(th)
        _, G = EG.energy_grad(ctx, th)
        f = G / 2
        born = ctx.size * pr * pr
        born = born / born.sum()
        q = 0.5 * born + 0.5 * lev / lev.sum()
        r = np.random.default_rng(k).choice(len(q), size=m, p=q)
        u, inv = np.unique(r, return_inverse=True)
        Yu = rows(ctx, th, phi0, u)
        Ys = np.concatenate([(pr * ctx.sw)[u][:, None], Yu], 1)[inv] / np.sqrt(m * q[r])[:, None]
        del Yu
        St, _ = RS.centred(Ys.T @ Ys, None, E)
        del Ys
        w, U = np.linalg.eigh(St)
        rungs, best = [], None
        for c in CUTS:
            kk = w > c * w.max()
            Pm = U[:, kk] / np.sqrt(w[kk])
            gt = Pm.T @ f
            d = -(Pm @ gt) / np.linalg.norm(gt)
            row = dict(cut=c, rank=int(kk.sum()), trials=[], tau=None)
            for tau in TAUS:
                ev = ctx.energy(th + torch.tensor(tau * d))
                En = ev[0]
                row["trials"].append((tau, eps(En)))
                if np.isfinite(En) and (E - En) / abs(P.E0) > MIN_GAIN:
                    row.update(tau=tau, E_new=En)
                    if best is None or En < best["E_new"]:
                        best = dict(cut=c, tau=tau, E_new=En, d=d, ev=ev)
                    break
            rungs.append(row)
        per = P.cpu_min() - s0
        st["cpu_charged"] += per
        rec = dict(k=k, eps=eps(E), n_distinct=int(len(u)), rungs=rungs, cpu_min=per)
        if best is None:
            rec["accepted"] = False
            st["stop"] = "STOP-NOGAIN"
            P.log(f"step {k}: no rung lowers eps by > {MIN_GAIN}; STOP-NOGAIN [{per:.2f} CPU-min]")
        else:
            th = th + best["tau"] * torch.tensor(best["d"])
            cache = best["ev"]  # exact (E, pr, phi0) at the new theta, reused by the next step
            rec.update(accepted=True, cut=best["cut"], tau=best["tau"], eps_new=eps(best["E_new"]),
                       gain=eps(E) - eps(best["E_new"]))
            np.save(FT, th.numpy())
            P.log(f"step {k}: cut {best['cut']:.0e} tau {best['tau']} eps {rec['eps']:.5e} -> {rec['eps_new']:.5e} "
                  f"(gain {rec['gain']:.2e}); distinct rows {len(u)}; rungs " +
                  " ".join(f"{x['cut']:.0e}:{x['tau']}" for x in rungs) +
                  f" [step {per:.2f}, charged {st['cpu_charged']:.2f}, process {P.cpu_min():.2f} CPU-min]")
        st["steps"].append(rec)
        st.update(verdict(st))
        json.dump(st, open(FS, "w"), indent=1)
    if len(st["steps"]) >= st["cap"] and st["stop"] is None:
        st["stop"] = "STOP-CAP"
    if st["steps"]:
        st.update(verdict(st))
        acc = [s for s in st["steps"] if s.get("accepted")]
        if len(acc) >= 10:
            h = st["steps"][5:10]
            st["ratio_second_half"] = (h[0]["eps"] - h[-1]["eps_new"]) / sum(x["cpu_min"] for x in h) / GLM
        P.log(f"T1c: eps {st['eps_start']:.5e} -> {st['eps_end']:.5e} in {len(st['steps'])} steps, charged "
              f"{st['cpu_charged']:.2f} CPU-min (scores {st['cpu_scores']:.2f}); gamma/gamma_LM {st['ratio']:.2f} "
              f"(without score cost {st['ratio_noscore']:.2f}) -> {st['verdict']}")
    json.dump(st, open(FS, "w"), indent=1)


if __name__ == "__main__":
    main()
