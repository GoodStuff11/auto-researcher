#!/usr/bin/env python3
"""
P4.1 rung 1 (Round 11 VOI note): polish `pcg` at 12 sites with exact truncated-metric LM steps, then test sampled S.
================================================================================================================

Run with the lit_env interpreter from this directory:
    ~/.conda/envs/lit_env/bin/python polish.py --mode MODE [options]

Adapted from tests/P4.1_cheap_su2_r12/polish.py (written 2026-11-16, never run: its results/ is empty).  Changes:
the rule is the leaf's recorded rung-1 rule (RULE.md), only the `che` and `pcg` arms are polished, the sampled
polish runs per arm with 8192 samples per step, and the verdict is written to results/verdict.json so the batch
script can decide whether the sampled `pcg` arm is needed.  The pre-registered rule is RULE.md (mode 'rule' prints
its path); nothing here overrides it.

System: open 4x3 strip, N_e = 10, U = 8, t' = -0.25, irrep (+,+) via P_G, exact S = 0; E0 = -7.827899034271753;
eps = (E - E0) / |E0|; g = E(theta0) - E0 per arm (theta0 = the SR-trained T2a / pilot parameters).

Imports (read only): tests/P4.1_cheap_su2_r11/realise.py (Ctx: sector, orbit reduction, exact energy; Tan:
truncated metric and LM candidate; line_search), and through it tangent.py, vmc.py, vmc_t2.py.

Modes
  rule      print RULE.md.
  seq       exact polish of --arm for at most --steps total steps (resumes from results/seq_<arm>.json; `che`
            without that file starts from Round 11's step-4 parameters and history, after checking that their
            exact energy reproduces Round 11's recorded step-4 energy).
  sampled   sampled polish of --arm from theta0, --nsamp i.i.d. samples per step, --steps steps;
            writes results/sampled_<arm>_N<nsamp>.json.
  verdict   apply RULE.md to results/*.json, print it, write results/verdict.json.
  need-pcg-sampled   exit 0 if the exact verdict is KEEP-LIVE or CONFOUNDED (sampled pcg needed), 1 otherwise.
Options
  --arm {che, pcg}   --steps INT (8)   --nsamp INT (8192)   --threads INT (1)   --cpu-cap FLOAT (CPU-min, 60)
  --res DIR          results directory (default ./results; the timing probe used ./results_probe)
"""
import argparse
import json
import os
import sys
import time

THR = "1"
for i, s in enumerate(sys.argv):
    if s == "--threads" and i + 1 < len(sys.argv):
        THR = sys.argv[i + 1]
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = THR

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r11"))
import realise as R  # noqa: E402  (read only)

TG, V = R.TG, R.V
T2 = TG.T2
log = V.log
torch.set_num_threads(int(THR))
E0 = TG.E0
THRCUT, TAU = 1e-5, 0.05
RHO_RETRACT, RHO_KEEP, F_KEEP = 0.85, 0.60, 0.5
PAR = dict(TG.PAR)
R11_CHE = os.path.join(TESTS, "P4.1_cheap_su2_r11/results/seq_full_lm_thr1e-05_tau0.05.json")
R11_CHE_TH = os.path.join(TESTS, "P4.1_cheap_su2_r11/results/theta_full_lm_thr1e-05_tau0.05.npy")
R11_SH = os.path.join(TESTS, "P4.1_cheap_su2_r11/results/S_H_theta0.npz")
RES = os.path.join(HERE, "results")


def cpu_min():
    return time.process_time() / 60.0


def model(arm):
    if arm == "che":
        return V.Model("che", 4, 3, 10, 16, -0.25, 8.0, (1, 1))
    return T2.DetModel(arm, 4, 3, 10, {"pcg": 15, "r1": 25}[arm], -0.25, 8.0, (1, 1))


def ctx_for(arm):
    """Round 11 Ctx with the arm's model and parameters swapped in (orbit reduction is arm independent)."""
    ctx = R.Ctx()
    ctx.m = model(arm)
    ctx.arm = arm
    ctx.th0 = torch.tensor(np.load(PAR[arm] + "_theta.npy"))
    ctx.stored = json.load(open(PAR[arm] + ".json"))["E"]
    assert ctx.th0.numel() == ctx.m.nparam, (arm, ctx.th0.numel(), ctx.m.nparam)
    return ctx


def logder(m, th, nU, nD):
    rep = th.detach().unsqueeze(0).repeat(nU.shape[0], 1).requires_grad_(True)
    l, _ = m.logpsi0(rep, nU, nD, rep=True)
    l.sum().backward()
    return rep.grad.numpy()


def build(ctx, th, CH=2048):
    """Y (orbit-weighted tangents [Psi, d Psi/d theta]) and HY over all representatives; returns E, Y, HY, pr."""
    E, pr, phi0 = ctx.energy(th)
    P, nrep = ctx.m.nparam, len(ctx.reps)
    Y = np.empty((nrep, 1 + P))
    Y[:, 0] = pr * ctx.sw
    for r0 in range(0, nrep, CH):
        rr = ctx.reps[r0:r0 + CH]
        acc = np.zeros((len(rr), P))
        for g in range(4):
            ix = ctx.idx[g, rr]
            w = 0.25 * ctx.sg[g, rr] * phi0[ix]
            acc += w[:, None] * logder(ctx.m, th, ctx.nU[ix], ctx.nD[ix])
        Y[r0:r0 + CH, 1:] = acc * ctx.sw[r0:r0 + CH, None]
    HY = np.empty_like(Y)
    for c0 in range(0, 1 + P, 512):
        cb = slice(c0, min(c0 + 512, 1 + P))
        HY[:, cb] = (ctx.Hr @ (Y[:, cb] / ctx.sw[:, None])) * ctx.sw[:, None]
    return E, Y, HY, pr


def exact_SH(Y, HY):
    S, Hm = Y.T @ Y, Y.T @ HY
    return 0.5 * (S + S.T), 0.5 * (Hm + Hm.T)


def sampled_SH(ctx, Y, HY, pr, N, seed):
    """i.i.d. samples r ~ p(r) = |orbit(r)| Psi(r)^2; unbiased S, H estimators normalised by the sampled S_00."""
    p = ctx.size * pr * pr
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    r = rng.choice(len(p), size=N, p=p)
    wt = 1.0 / np.sqrt(N * p[r])
    Ys, HYs = Y[r] * wt[:, None], HY[r] * wt[:, None]
    S, Hm = Ys.T @ Ys, Ys.T @ HYs
    S, Hm = 0.5 * (S + S.T), 0.5 * (Hm + Hm.T)
    s0 = S[0, 0]
    S, Hm = S / s0, Hm / s0
    eloc = (ctx.Hr @ pr)[r] / pr[r]
    return S, Hm, r, eloc


def step_candidate(S, Hm, E, P):
    tan = R.Tan(S, Hm, E, list(range(P)))
    r = tan.reduce(THRCUT)
    z, lam = tan.candidate(r, "lm", TAU)
    return tan.dtheta(r, z, P), float(np.linalg.norm(z)), lam, r["rank"], (E - tan.pred(r, z))


def dump(fn, d):
    json.dump(d, open(fn, "w"), indent=1)


def seq(a):
    t0 = time.time()
    ctx = ctx_for(a.arm)
    fn, fth = os.path.join(RES, f"seq_{a.arm}.json"), os.path.join(RES, f"theta_{a.arm}.npy")
    E00 = ctx.energy(ctx.th0)[0]
    gap = E00 - E0
    valid = abs(E00 - ctx.stored) < 1e-8
    log(f"{a.arm}: P {ctx.m.nparam}; E(theta0) {E00:.10f} stored {ctx.stored:.10f} (diff {abs(E00 - ctx.stored):.1e}); "
        f"eps0 {gap / abs(E0):.4e}; VALID {valid} [{cpu_min():.1f} CPU-min]")
    th, hist, chk = ctx.th0.clone(), [], {}
    if os.path.exists(fn):
        d = json.load(open(fn))
        hist, chk = d["steps"], d.get("check", {})
        th = torch.tensor(np.load(fth))
    elif a.arm == "che":
        hist = json.load(open(R11_CHE))["steps"]
        th = torch.tensor(np.load(R11_CHE_TH))
        Er = ctx.energy(th)[0]
        chk = dict(r11_step4_E=hist[-1]["E_new"], recomputed=Er, diff=abs(Er - hist[-1]["E_new"]))
        valid = valid and chk["diff"] < 1e-10
        log(f"che resumes from Round 11 step 4: E {Er:.10f} vs recorded {hist[-1]['E_new']:.10f} "
            f"(diff {chk['diff']:.1e})")
    stopped = any(not s.get("accepted") for s in hist)
    P, per = ctx.m.nparam, None
    while len(hist) < a.steps and not stopped:
        c0 = cpu_min()
        if per is not None and c0 + per > a.cpu_cap:
            log(f"stop: next step would exceed the CPU cap ({c0:.1f} + {per:.1f} > {a.cpu_cap}); TRUNCATED")
            break
        E, Y, HY, pr = build(ctx, th)
        S, Hm = exact_SH(Y, HY)
        del Y, HY
        d, zn, lam, rank, pred = step_candidate(S, Hm, E, P)
        s, Es, trials = R.line_search(ctx, th, d, E)
        rec = dict(E=E, pred=pred / gap, lam=lam, zn=zn, dth=float(np.linalg.norm(d)), rank=rank, s=s, E_new=Es,
                   trials=[(t, (E00 - e) / gap) for t, e in trials])
        if s is None:
            rec["accepted"] = False
            hist.append(rec)
            stopped = True
            log(f"step {len(hist)}: no lowering trial; stop. trials {rec['trials']}")
            break
        th = th + s * torch.tensor(d)
        rec.update(accepted=True, R=(E00 - Es) / gap, eps=(Es - E0) / abs(E0))
        hist.append(rec)
        per = cpu_min() - c0
        log(f"step {len(hist)}: E {Es:.8f} eps {rec['eps']:.4e} R {rec['R']:.4f} (predicted gain {rec['pred']:.4f}); "
            f"s {s}; |z| {zn:.3f}; |dtheta| {s * rec['dth']:.3f}; rank {rank} [{cpu_min():.1f} CPU-min, step {per:.1f}, "
            f"wall {time.time() - t0:.0f} s]")
        np.save(fth, th.numpy())
        dump(fn, dict(arm=a.arm, E_theta0=E00, gap=gap, valid=valid, check=chk, steps=hist, cpu_min=cpu_min(),
                      wall=time.time() - t0))
    dump(fn, dict(arm=a.arm, E_theta0=E00, gap=gap, valid=valid, check=chk, steps=hist, cpu_min=cpu_min(),
                  wall=time.time() - t0, stopped=stopped))
    log(f"wrote {os.path.basename(fn)} ({time.time() - t0:.0f} s, {cpu_min():.1f} CPU-min)")


def sampled(a):
    t0 = time.time()
    ctx = ctx_for(a.arm)
    fn = os.path.join(RES, f"sampled_{a.arm}_N{a.nsamp}.json")
    th = ctx.th0.clone()
    E00 = ctx.energy(th)[0]
    gap = E00 - E0
    valid = abs(E00 - ctx.stored) < 1e-8
    P, per, hist, chk = ctx.m.nparam, None, [], {}
    while len(hist) < a.steps:
        c0 = cpu_min()
        if per is not None and c0 + per > a.cpu_cap:
            log(f"stop: next step would exceed the CPU cap ({c0:.1f} + {per:.1f} > {a.cpu_cap}); TRUNCATED")
            break
        E, Y, HY, pr = build(ctx, th)
        if not hist and a.arm == "che":
            S, Hm = exact_SH(Y, HY)
            ref = np.load(R11_SH)
            chk.update(S=float(np.abs(S - ref["S"]).max() / np.abs(ref["S"]).max()),
                       H=float(np.abs(Hm - ref["H"]).max() / np.abs(ref["H"]).max()))
            valid = valid and chk["S"] < 1e-8 and chk["H"] < 1e-8
            log(f"INVALID check: generic vs Round 11 S rel {chk['S']:.1e}, H rel {chk['H']:.1e}")
            del S, Hm
        Ss, Hs, r, eloc = sampled_SH(ctx, Y, HY, pr, a.nsamp, len(hist) + 1)
        del Y, HY
        Eold_s = float(eloc.mean())
        se = float(eloc.std() / np.sqrt(len(eloc)))
        if not hist:
            chk.update(E_samp=Eold_s, E_exact=E, se=se, z=abs(Eold_s - E) / se)
            valid = valid and chk["z"] <= 6.0
            log(f"INVALID check: sampled E {Eold_s:.6f} +- {se:.1e} vs exact {E:.6f} ({chk['z']:.1f} se)")
        d, zn, lam, rank, pred = step_candidate(Ss, Hs, float(Hs[0, 0]), P)
        acc_s, trials = None, []
        for s in (1.0, 0.5, 0.25):
            Ex, prn, _ = ctx.energy(th + s * torch.tensor(d))
            w = (prn[r] / pr[r]) ** 2
            el_new = (ctx.Hr @ prn)[r] / prn[r]
            Es_s = float(np.sum(w * el_new) / np.sum(w))
            trials.append(dict(s=s, E_samp=Es_s, E_exact=Ex))
            if Es_s < Eold_s:
                acc_s = (s, Ex)
                break
        rec = dict(E=E, E_samp=Eold_s, se=se, zn=zn, lam=lam, rank=rank, dth=float(np.linalg.norm(d)), trials=trials)
        if acc_s is None:
            rec["accepted"] = False
            hist.append(rec)
            log(f"step {len(hist)}: no sampled-lowering trial; stop. trials {trials}")
            break
        s, Ex = acc_s
        th = th + s * torch.tensor(d)
        rec.update(accepted=True, s=s, E_new=Ex, R=(E00 - Ex) / gap, eps=(Ex - E0) / abs(E0))
        hist.append(rec)
        per = cpu_min() - c0
        log(f"step {len(hist)}: exact E {Ex:.8f} R {rec['R']:.4f}; s {s}; sampled E_old {Eold_s:.6f} -> "
            f"{trials[-1]['E_samp']:.6f}; |z| {zn:.3f}; rank {rank} [{cpu_min():.1f} CPU-min, step {per:.1f}, "
            f"wall {time.time() - t0:.0f} s]")
        dump(fn, dict(arm=a.arm, nsamp=a.nsamp, E_theta0=E00, gap=gap, valid=valid, check=chk, steps=hist,
                      cpu_min=cpu_min(), wall=time.time() - t0))
    dump(fn, dict(arm=a.arm, nsamp=a.nsamp, E_theta0=E00, gap=gap, valid=valid, check=chk, steps=hist,
                  cpu_min=cpu_min(), wall=time.time() - t0, done=len(hist) >= a.steps or
                  any(not s.get("accepted") for s in hist)))
    log(f"wrote {os.path.basename(fn)} ({time.time() - t0:.0f} s, {cpu_min():.1f} CPU-min)")


def at_step(d, n):
    """(eps, R, accepted steps, complete) after at most n steps; complete = n steps taken or an early stop."""
    acc = [s for s in d["steps"][:n] if s.get("accepted")]
    E = acc[-1]["E_new"] if acc else d["E_theta0"]
    done = len(d["steps"]) >= n or any(not s.get("accepted") for s in d["steps"])
    return (E - E0) / abs(E0), (d["E_theta0"] - E) / d["gap"], len(acc), done


def exact_verdict(D, n):
    if not all(k in D for k in ("che", "pcg")):
        return None, "INCOMPLETE"
    ec, _, _, dc = at_step(D["che"], n)
    ep, _, _, dp = at_step(D["pcg"], n)
    if not (dc and dp):
        return ep / ec, "INCOMPLETE"
    rho = ep / ec
    return rho, ("RETRACT" if rho >= RHO_RETRACT else "KEEP-LIVE" if rho <= RHO_KEEP else "CONFOUNDED")


def load_exact():
    D = {}
    for arm in ("che", "pcg"):
        f = os.path.join(RES, f"seq_{arm}.json")
        if os.path.exists(f):
            D[arm] = json.load(open(f))
    return D


def verdict(a):
    D = load_exact()
    out = dict(valid=all(D[k].get("valid", False) for k in D) and len(D) == 2)
    for k in D:
        for n in (4, 8):
            e, Rr, na, done = at_step(D[k], n)
            print(f"exact {k} N{n}: eps {e:.4e} (SR-trained {D[k]['gap'] / abs(E0):.4e}; R {Rr:.4f}) after {na} "
                  f"accepted steps{'' if done else ' [TRUNCATED]'}")
    for n in (4, 8):
        rho, v = exact_verdict(D, n)
        out[f"rho_N{n}"], out[f"verdict_N{n}"] = rho, v
        print(f"N{n}: rho_pol = eps_pcg / eps_che = {rho if rho is None else round(rho, 4)} -> {v}"
              f"{'  (the decision uses N8)' if n == 4 else ''}")
    out["exact_verdict"] = out["verdict_N8"]
    samp = {}
    for arm in ("che", "pcg"):
        f = os.path.join(RES, f"sampled_{arm}_N{a.nsamp}.json")
        if not os.path.exists(f) or arm not in D:
            continue
        d = json.load(open(f))
        e, Rs, na, done = at_step(d, 4)
        Rx = at_step(D[arm], 4)[1]
        fN = Rs / Rx
        samp[arm] = dict(R_sampled=Rs, R_exact=Rx, f=fN, eps=e, steps=na, done=done, valid=d.get("valid"),
                         check=d.get("check"))
        out["valid"] = out["valid"] and bool(d.get("valid"))
        print(f"sampled {arm} N={a.nsamp}: R {Rs:.4f} after {na} accepted steps (exact 4-step R {Rx:.4f}); "
              f"f = {fN:.3f}{'' if done else ' [TRUNCATED]'}; check {d.get('check')}")
    need = ["che"] + (["pcg"] if out["exact_verdict"] in ("KEEP-LIVE", "CONFOUNDED") else [])
    if all(k in samp and samp[k]["done"] for k in need):
        sv = "SAMPLED-KEEPS" if all(samp[k]["f"] >= F_KEEP for k in need) else "SAMPLED-LOSES"
    else:
        sv = "INCOMPLETE"
    out.update(sampled=samp, sampled_arms_required=need, sampled_verdict=sv)
    print(f"VALID {out['valid']}; EXACT VERDICT (N8): {out['exact_verdict']}; SAMPLED VERDICT "
          f"(arms {need}): {sv}")
    dump(os.path.join(RES, "verdict.json"), out)


def need_pcg_sampled(a):
    D = load_exact()
    _, v = exact_verdict(D, 8)
    need = v in ("KEEP-LIVE", "CONFOUNDED")
    # RULE.md amendment 1: if sampled che already completed with f < 0.5, SAMPLED-LOSES is fixed; pcg decides nothing
    f, fche = os.path.join(RES, f"sampled_che_N{a.nsamp}.json"), None
    if need and os.path.exists(f) and "che" in D:
        e, Rs, na, done = at_step(json.load(open(f)), 4)
        fche = Rs / at_step(D["che"], 4)[1]
        need = not (done and fche < F_KEEP)
    print(f"exact verdict {v}; sampled che f {fche}; sampled pcg {'NEEDED' if need else 'NOT NEEDED'}")
    sys.exit(0 if need else 1)


def main():
    global RES
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", required=True, choices=["rule", "seq", "sampled", "verdict", "need-pcg-sampled"])
    pa.add_argument("--arm", default="pcg", choices=["che", "pcg"])
    pa.add_argument("--steps", type=int, default=8)
    pa.add_argument("--nsamp", type=int, default=8192)
    pa.add_argument("--threads", default="1")
    pa.add_argument("--cpu-cap", type=float, default=60.0)
    pa.add_argument("--res", default=os.path.join(HERE, "results"))
    a = pa.parse_args()
    RES = a.res
    os.makedirs(RES, exist_ok=True)
    if a.mode == "rule":
        print(open(os.path.join(HERE, "RULE.md")).read())
    else:
        {"seq": seq, "sampled": sampled, "verdict": verdict, "need-pcg-sampled": need_pcg_sampled}[a.mode](a)


if __name__ == "__main__":
    main()
