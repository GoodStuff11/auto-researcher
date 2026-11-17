#!/usr/bin/env python3
"""
P4.1 Round 12 T1: polish every SR-trained 12-site arm with the Round 11 linear-method protocol, and test it sampled.
=================================================================================================================

Run with the lit_env interpreter from this directory:
    ~/.conda/envs/lit_env/bin/python polish.py --mode MODE [options]

Question
--------
Round 11 (realise.py) showed that four exact truncated-metric linear-method (LM) steps remove 0.539 of the 12-site
`che` gap (eps 2.86e-3 -> 1.32e-3), so every SR-trained ratio in the leaf carries an optimiser term.  This script
(1) polishes `pcg` (R2, T2a H = 15), `r1` (T2a H = 25) and `che_k8` (pilot, H = 10, K = 8) with exactly the protocol
that polished `che`, and extends all four arms by the same number of further steps; (2) reruns the `che` polish with
S and H estimated from i.i.d. samples of |Psi|^2 instead of exact sums, the only form that can run at 24 sites.
System: open 4x3 strip, N_e = 10, U = 8, t' = -0.25, irrep (+,+) via P_G, exact S = 0; E0 = -7.827899034271753;
eps = (E - E0) / |E0|; g = E(theta0) - E0 per arm.

Pre-registered rule (written before any polish output existed; mode 'rule' prints it)
  Protocol (identical for every arm, the Round 11 `che` variant): block full (all parameters); relative cutoff 1e-5
  of the tangent metric (tangents orthogonalised against Psi); mode lm with trust radius tau = 0.05 in the metric
  norm; line search s in {1, 1/2, 1/4} on the exact energy, first lowering s taken, otherwise the sequence stops
  (an early stop is the arm's final value; it is not re-tried).  S and H rebuilt exactly after every accepted step.
  N4 = after at most 4 steps (the `che` budget of Round 11).  N6 = after at most 6 steps, run for ALL four arms or
  for none (che continues from its Round 11 step-4 parameters).
  R2 (pcg against che, both polished):  rho_pol = eps_pcg / eps_che.
    SURVIVES  if rho_pol <= 0.75 and eps_pcg < eps_r1 (polished r1);
    KILLED    if rho_pol >= 0.90 or eps_pcg >= eps_r1;
    AMBIGUOUS otherwise.
  The verdict is computed at N4 and, if run, at N6.  If the two differ, the verdict is AMBIGUOUS (both reported).
  R1 (reported, no verdict): rho_8 = eps_che_k8 / eps_che, both polished, at N4 and N6.
  Sampled polish (che only, from the SR-trained theta0): every step draws N i.i.d. representatives r with
  probability p(r) = |orbit(r)| Psi(r)^2 (exact sampling, no Markov chain; fresh samples per step, seed = step),
  estimates S_kl = mean Y_rk Y_rl / p(r) and H_kl = mean Y_rk (H Y)_rl / p(r) (symmetrised; normalised by S_00;
  this is the standard sampled LM estimator including the local-energy-derivative term), and builds the step with
  the same cutoff, tau and lm mode.  The line search uses the sampled energy only: correlated reweighting of the
  same samples, E_s = sum w E_loc,new / sum w with w = Psi_new(r)^2 / Psi_old(r)^2, against the sampled E_old.
  Exact energies are computed for scoring only.  f_N = R_sampled(4 steps) / R_exact(4 steps) with R_exact = 0.5392.
    SAMPLED-KEEPS  if f_N >= 0.5 at N = 1e5;   SAMPLED-LOSES if f_N < 0.5 at N = 1e5.
    N = 2e4 is reported (no verdict) if the CPU cap allows.
  Budget clause: 40 CPU-minutes for the whole round (sum of process CPU times, 8 threads).  Order of execution:
  sampled che N = 1e5, pcg, r1, che_k8, sampled N = 2e4, then the N6 extension.  A sequence cut by the cap is
  reported as truncated; if the pcg or r1 N4 sequence is cut, the R2 verdict is INCOMPLETE, not a pass or a kill.
  INVALID if: an arm's E(theta0) differs from its stored exact energy by > 1e-8; or the generic (autograd) S and H
  at the `che` theta0 differ from Round 11's hand-assembled S_H_theta0.npz by more than 1e-8 relative (max-norm).

Implementation notes
  * Reuses realise.py (Round 11) read only: Ctx (sector, orbit reduction, exact energy), Tan (truncated metric and
    LM candidate), line_search.  The orbit reduction depends only on geometry and irrep, so one Ctx serves all
    arms; each arm's E(theta0) check validates that.
  * Log-derivatives are generic: per-sample parameter copies (vmc.Model.logderiv's 'rep' trick) through
    vmc.Model.logpsi0 (che, che_k8) or vmc_t2.DetModel.logpsi0 (pcg, r1), computed for the four images of each
    representative and accumulated into Y as realise.build does.
  * The CPU time of the process is logged after every stage (time.process_time), and a sequence stops before a step
    that would push the process past --cpu-cap minutes.

Modes
  rule     print the rule above.
  probe    time one rep-logderiv chunk (2048 configurations) and one forward chunk per arm; no polish.
  seq      polish --arm with the protocol above for at most --steps steps (resumes from results/seq_<arm>.json if it
           exists; for che without that file, starts from Round 11's step-4 parameters and history).
  sampled  sampled polish of che from theta0 with --nsamp samples per step, --steps steps; writes
           results/sampled_N<nsamp>.json.  Step 1 also runs the INVALID check against S_H_theta0.npz.
  verdict  apply the rule to results/*.json and print it.
Options
  --mode {rule, probe, seq, sampled, verdict} (required)
  --arm {che, pcg, r1, che_k8}   seq only (pcg)
  --steps INT (4)                seq / sampled: maximum total number of steps in the stored sequence
  --nsamp INT (100000)           sampled only
  --threads INT (8)              torch/BLAS threads
  --cpu-cap FLOAT (12)           CPU-minutes for this process
"""
import argparse
import json
import os
import sys
import time

THR = "8"
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
RULE = __doc__.split("Pre-registered rule")[1].split("Implementation notes")[0]
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
E0 = TG.E0
THRCUT, TAU = 1e-5, 0.05
R_EXACT4 = 0.5391510169766579
PAR = dict(TG.PAR)
PAR["che_k8"] = os.path.join(TESTS, "P4.1_cheap_su2_scaling/logs/pilot/pg_che_k8/S4x3_N10_U8_tp-0.25_pg11_che_k8_seed0")
R11_CHE = os.path.join(TESTS, "P4.1_cheap_su2_r11/results/seq_full_lm_thr1e-05_tau0.05.json")
R11_CHE_TH = os.path.join(TESTS, "P4.1_cheap_su2_r11/results/theta_full_lm_thr1e-05_tau0.05.npy")


def cpu_min():
    return time.process_time() / 60.0


def model(arm):
    if arm == "che":
        return V.Model("che", 4, 3, 10, 16, -0.25, 8.0, (1, 1))
    if arm == "che_k8":
        return V.Model("che_k8", 4, 3, 10, 10, -0.25, 8.0, (1, 1))
    return T2.DetModel(arm, 4, 3, 10, {"pcg": 15, "r1": 25}[arm], -0.25, 8.0, (1, 1))


def ctx_for(arm, ctx=None):
    """Round 11 Ctx with the arm's model and parameters swapped in (orbit reduction is arm independent)."""
    if ctx is None:
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
    """Y (orbit-weighted tangents [Psi, d Psi/d theta]) and HY; returns E, Y, HY, pr."""
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
    return S, Hm, float(Hm[0, 0]), r, eloc


def step_candidate(S, Hm, E, P):
    tan = R.Tan(S, Hm, E, list(range(P)))
    r = tan.reduce(THRCUT)
    z, lam = tan.candidate(r, "lm", TAU)
    return tan.dtheta(r, z, P), float(np.linalg.norm(z)), lam, r["rank"], (E - tan.pred(r, z))


def probe(a):
    ctx = R.Ctx()
    for arm in ("che", "pcg", "r1", "che_k8"):
        ctx_for(arm, ctx)
        ix = np.arange(2048)
        c0 = cpu_min()
        logder(ctx.m, ctx.th0, ctx.nU[ix], ctx.nD[ix])
        c1 = cpu_min()
        with torch.no_grad():
            ctx.m.logpsi0(ctx.th0, ctx.nU[ix], ctx.nD[ix])
        c2 = cpu_min()
        n = ctx.sec.dim / 2048
        log(f"{arm}: P {ctx.m.nparam}; logderiv {60 * (c1 - c0):.2f} CPU-s per 2048 (full space ~{n * (c1 - c0):.2f} "
            f"CPU-min); forward {60 * (c2 - c1):.2f} CPU-s per 2048 (~{n * (c2 - c1):.2f} CPU-min)")


def seq(a):
    t0 = time.time()
    ctx = ctx_for(a.arm)
    fn = os.path.join(RES, f"seq_{a.arm}.json")
    fth = os.path.join(RES, f"theta_{a.arm}.npy")
    E00, _, _ = ctx.energy(ctx.th0)
    gap = E00 - E0
    log(f"{a.arm}: P {ctx.m.nparam}; E(theta0) {E00:.10f} stored {ctx.stored:.10f} (diff {abs(E00 - ctx.stored):.1e}); "
        f"eps0 {gap / abs(E0):.4e} [{cpu_min():.1f} CPU-min]")
    valid = abs(E00 - ctx.stored) < 1e-8
    th, hist = ctx.th0.clone(), []
    if os.path.exists(fn):
        hist = json.load(open(fn))["steps"]
        th = torch.tensor(np.load(fth))
    elif a.arm == "che":
        hist = json.load(open(R11_CHE))["steps"]
        th = torch.tensor(np.load(R11_CHE_TH))
    stopped = any(not s.get("accepted") for s in hist)
    P, per = ctx.m.nparam, None
    while len(hist) < a.steps and not stopped:
        c0 = cpu_min()
        if per is not None and c0 + per > a.cpu_cap:
            log(f"stop: next step would exceed the CPU cap ({c0:.1f} + {per:.1f} > {a.cpu_cap})")
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
            f"s {s}; |z| {zn:.3f}; |dtheta| {s * rec['dth']:.3f}; rank {rank} [{cpu_min():.1f} CPU-min, step {per:.1f}]")
        np.save(fth, th.numpy())
        json.dump(dict(arm=a.arm, E_theta0=E00, gap=gap, valid=valid, steps=hist, cpu_min=cpu_min(),
                       wall=time.time() - t0), open(fn, "w"), indent=1)
    json.dump(dict(arm=a.arm, E_theta0=E00, gap=gap, valid=valid, steps=hist, cpu_min=cpu_min(),
                   wall=time.time() - t0, stopped=stopped), open(fn, "w"), indent=1)
    log(f"wrote {os.path.basename(fn)} ({time.time() - t0:.0f} s, {cpu_min():.1f} CPU-min)")


def sampled(a):
    t0 = time.time()
    ctx = ctx_for("che")
    fn = os.path.join(RES, f"sampled_N{a.nsamp}.json")
    th = ctx.th0.clone()
    E00 = ctx.energy(th)[0]
    gap = E00 - E0
    P, per, hist, chk = ctx.m.nparam, None, [], None
    while len(hist) < a.steps:
        c0 = cpu_min()
        if per is not None and c0 + per > a.cpu_cap:
            log(f"stop: next step would exceed the CPU cap ({c0:.1f} + {per:.1f} > {a.cpu_cap})")
            break
        E, Y, HY, pr = build(ctx, th)
        if not hist:
            S, Hm = exact_SH(Y, HY)
            ref = np.load(os.path.join(TESTS, "P4.1_cheap_su2_r11/results/S_H_theta0.npz"))
            chk = dict(S=float(np.abs(S - ref["S"]).max() / np.abs(ref["S"]).max()),
                       H=float(np.abs(Hm - ref["H"]).max() / np.abs(ref["H"]).max()), E=abs(E - E00))
            log(f"INVALID check: generic vs Round 11 S rel {chk['S']:.1e}, H rel {chk['H']:.1e}")
            del S, Hm
        Ss, Hs, Esamp, r, eloc = sampled_SH(ctx, Y, HY, pr, a.nsamp, len(hist) + 1)
        del Y, HY
        d, zn, lam, rank, pred = step_candidate(Ss, Hs, Esamp, P)
        Eold_s = float(eloc.mean())
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
        rec = dict(E=E, E_samp=Esamp, Eold_s=Eold_s, zn=zn, lam=lam, rank=rank, dth=float(np.linalg.norm(d)),
                   trials=trials)
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
            f"{trials[-1]['E_samp']:.6f}; |z| {zn:.3f}; rank {rank} [{cpu_min():.1f} CPU-min, step {per:.1f}]")
        json.dump(dict(nsamp=a.nsamp, E_theta0=E00, gap=gap, check=chk, steps=hist, cpu_min=cpu_min(),
                       wall=time.time() - t0), open(fn, "w"), indent=1)
    json.dump(dict(nsamp=a.nsamp, E_theta0=E00, gap=gap, check=chk, steps=hist, cpu_min=cpu_min(),
                   wall=time.time() - t0), open(fn, "w"), indent=1)
    log(f"wrote {os.path.basename(fn)} ({time.time() - t0:.0f} s, {cpu_min():.1f} CPU-min)")


def eps_at(d, n):
    acc = [s for s in d["steps"][:n] if s.get("accepted")]
    E = acc[-1]["E_new"] if acc else d["E_theta0"]
    return (E - E0) / abs(E0), len(acc), len(d["steps"]) >= n or any(not s.get("accepted") for s in d["steps"])


def r2_verdict(e_pcg, e_che, e_r1):
    rho = e_pcg / e_che
    if rho >= 0.90 or e_pcg >= e_r1:
        return rho, "KILLED"
    if rho <= 0.75 and e_pcg < e_r1:
        return rho, "SURVIVES"
    return rho, "AMBIGUOUS"


def verdict(a):
    D = {}
    for arm in ("che", "pcg", "r1", "che_k8"):
        f = os.path.join(RES, f"seq_{arm}.json")
        if not os.path.exists(f) and arm == "che":
            f = R11_CHE
        if os.path.exists(f):
            D[arm] = json.load(open(f))
            D[arm].setdefault("E_theta0", D[arm].get("E_theta0"))
    valid = all(D[k].get("valid", True) for k in D)
    out = {}
    for n in (4, 6):
        row = {k: eps_at(D[k], n) for k in D}
        for k, (e, na, done) in row.items():
            e0 = (D[k]["E_theta0"] - E0) / abs(E0)
            print(f"N{n} {k}: eps {e:.4e} (SR-trained {e0:.4e}; R {(e0 - e) / e0:.3f}) after {na} accepted steps"
                  f"{'' if done else ' [sequence shorter than N: truncated]'}")
        if all(k in row and row[k][2] for k in ("che", "pcg", "r1")):
            rho, v = r2_verdict(row["pcg"][0], row["che"][0], row["r1"][0])
            out[n] = v
            print(f"N{n} R2: rho_pol {rho:.3f}; pcg {'<' if row['pcg'][0] < row['r1'][0] else '>='} r1; verdict {v}")
        else:
            print(f"N{n} R2: INCOMPLETE")
        if "che_k8" in row and row["che_k8"][2]:
            print(f"N{n} R1: rho_8 = eps_k8 / eps_che = {row['che_k8'][0] / row['che'][0]:.3f}")
    vs = list(out.values())
    final = "INCOMPLETE" if not vs else (vs[0] if len(set(vs)) == 1 else "AMBIGUOUS (N4 and N6 differ)")
    print(f"VALID {valid}; R2 VERDICT: {final}")
    for N in (100000, 20000):
        f = os.path.join(RES, f"sampled_N{N}.json")
        if os.path.exists(f):
            d = json.load(open(f))
            acc = [s for s in d["steps"] if s.get("accepted")]
            Rs = acc[-1]["R"] if acc else 0.0
            fN = Rs / R_EXACT4
            tag = ("SAMPLED-KEEPS" if fN >= 0.5 else "SAMPLED-LOSES") if N == 100000 else "reported"
            print(f"sampled N={N}: R {Rs:.4f} after {len(acc)} accepted steps of {len(d['steps'])}; f_N {fN:.3f}; "
                  f"check {d['check']}; {tag}")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", required=True, choices=["rule", "probe", "seq", "sampled", "verdict"])
    pa.add_argument("--arm", default="pcg", choices=["che", "pcg", "r1", "che_k8"])
    pa.add_argument("--steps", type=int, default=4)
    pa.add_argument("--nsamp", type=int, default=100000)
    pa.add_argument("--threads", default="8")
    pa.add_argument("--cpu-cap", type=float, default=12.0)
    a = pa.parse_args()
    if a.mode == "rule":
        print(RULE)
    else:
        {"probe": probe, "seq": seq, "sampled": sampled, "verdict": verdict}[a.mode](a)


if __name__ == "__main__":
    main()
