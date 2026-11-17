#!/usr/bin/env python3
"""
P4.1 r21 / route R21 `trSR`: sampled SR with a truncated metric and many small steps, 12 sites.  Rule: RULE.md.
=============================================================================================================

Run with the lit_env interpreter from this directory:
    ~/.conda/envs/lit_env/bin/python trsr.py --mode MODE [options]

Reuses (read only) tests/P4.1_cheap_su2_r19/polish.py (ctx_for, logder, cpu_min, dump) and, through it,
tests/P4.1_cheap_su2_r11/realise.py (Ctx.energy, orbit reduction).  r20's i.i.d. draw is copied (draw()).

Step (all arms): at the current theta, draw N i.i.d. representatives r ~ p(r) = |orbit(r)| Psi(r)^2 (seed = step
number), compute the orbit-weighted log-derivatives O_i(r) only on the drawn rows, and form
    S_T = <O O> - <O><O>,   f = <O E_loc> - <O><E_loc>      (standard SR force; no HY needed).
  trsr     eigen-truncate S_T at relative cutoff 1e-4 (keep w > 1e-4 w_max), d = -tau * Pm g~/|g~|,
           g~ = Pm^T f, Pm = U_k / sqrt(w_k)  =>  metric length sqrt(d^T S_T d) = tau = 0.05.
  shift    d proportional to -(S_T + 1e-3 I)^{-1} f, scaled to the same metric length sqrt(d^T S_T d) = 0.05
           (the training's absolute Tikhonov shift; RULE.md "Interpretation").
  ladder   (reported only) candidates at relative cutoffs 1e-4..1e-8, each at tau = 0.05; a fresh held-out set
           (N, seed 1000 + step) scores each by the correlated-sampling dE; take argmin if dE < 0, else skip.
RULE_amend1 (supersedes the fixed tau = 0.05 for trsr and shift): each step tries metric lengths {0.005, 0.01,
0.02, 0.05}, scores them by the correlated-sampling dE on a held-out set (N, seed 1000 + step), takes the argmin
if dE < 0, else skips the step.  The exact energy is logged per candidate (oracle) and never used to choose.
Verdict on the best of trsr and shift.

Modes
  exact0   free pre-run diagnostic: the infinite-sample (exact S, H from r11 results/S_H_theta0.npz) step 1 of the
           trsr and shift arms at tau 0.05 (and 0.02), scored by the exact energy.  Decides nothing.
  run      --arm {trsr, shift, ladder} --steps 15 --nsamp 16384; resumes from results/<arm>.json.
  verdict  apply RULE.md to results/*.json; print and write results/verdict.json.
Options  --threads INT (1)   --cpu-cap FLOAT (CPU-min, 60)   --res DIR
"""
import argparse
import json
import os
import sys
import time

THR = "1"
for i, s_ in enumerate(sys.argv):
    if s_ == "--threads" and i + 1 < len(sys.argv):
        THR = sys.argv[i + 1]
for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = THR

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r19"))
import polish as P  # noqa: E402  (read only)
import torch  # noqa: E402

log = P.log
E0 = P.E0
CUT, TAU, SHIFT = 1e-4, 0.05, 1e-3
TAUS = (0.005, 0.01, 0.02, 0.05)          # RULE_amend1: metric-length ladder for trsr and shift
LADDER = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
R_VIABLE, R_DEAD, RISE_MAX = 0.20, 0.08, 0.02
R11_SH = P.R11_SH
RES = os.path.join(HERE, "results")


def draw(ctx, pr, N, seed):
    p = ctx.size * pr * pr
    p = p / p.sum()
    return np.random.default_rng(seed).choice(len(p), size=N, p=p)


def orows(ctx, th, phi0, pr, q, CH=2048):
    """O_i(r) = (d pr_i / d theta)(r) / pr(r) for representative positions q (same expansion as polish.build)."""
    P_ = ctx.m.nparam
    out = np.empty((len(q), P_))
    for c0 in range(0, len(q), CH):
        qq = q[c0:c0 + CH]
        rr = ctx.reps[qq]
        acc = np.zeros((len(rr), P_))
        for g in range(4):
            ix = ctx.idx[g, rr]
            w = 0.25 * ctx.sg[g, rr] * phi0[ix]
            acc += w[:, None] * P.logder(ctx.m, th, ctx.nU[ix], ctx.nD[ix])
        out[c0:c0 + CH] = acc / pr[qq][:, None]
    return out


def sr_stats(ctx, th, phi0, pr, r):
    """sampled S_T and force f from i.i.d. draws r (positions in reps); duplicates are computed once."""
    q, inv = np.unique(r, return_inverse=True)
    O = orows(ctx, th, phi0, pr, q)[inv]
    el = (ctx.Hr @ pr)[r] / pr[r]
    Ob = O.mean(0)
    Oc = O - Ob
    ST = Oc.T @ Oc / len(r)
    f = Oc.T @ (el - el.mean()) / len(r)
    return 0.5 * (ST + ST.T), f, el


def d_trunc(w, U, f, cut, tau=TAU):
    k = w > cut * w.max()
    Pm = U[:, k] / np.sqrt(w[k])
    gt = Pm.T @ f
    return -tau * (Pm @ gt) / np.linalg.norm(gt), int(k.sum()), float(np.linalg.norm(gt))


def d_shift(w, U, f, tau=TAU):
    x = U @ ((U.T @ f) / (w + SHIFT))
    ml = np.sqrt(max(x @ (U @ (w * (U.T @ x))), 0.0))
    return -tau * x / ml, float(ml)


def exact0(a):
    """infinite-sample step 1 at theta0 from Round 11's exact S, H (diagnostic only)."""
    ctx = P.ctx_for("che")
    E00, pr0, _ = ctx.energy(ctx.th0)
    gap = E00 - E0
    z = np.load(R11_SH)
    S, Hm = z["S"], z["H"]
    c = np.arange(1, S.shape[0])
    s = S[0, c]
    ST = S[np.ix_(c, c)] - np.outer(s, s)
    f = 0.5 * (Hm[0, c] + Hm[c, 0]) - E00 * s
    w, U = np.linalg.eigh(0.5 * (ST + ST.T))
    out = dict(E_theta0=E00, gap=gap, wmax=float(w.max()), wmin=float(w.min()),
               n_below_shift=int((w < SHIFT).sum()), rank_cut=int((w > CUT * w.max()).sum()))
    for tau in (0.05, 0.02):
        for arm in ("trsr", "shift"):
            d = d_trunc(w, U, f, CUT, tau)[0] if arm == "trsr" else d_shift(w, U, f, tau)[0]
            Ex = ctx.energy(ctx.th0 + torch.tensor(d))[0]
            out[f"{arm}_tau{tau}"] = dict(R=(E00 - Ex) / gap, dth=float(np.linalg.norm(d)))
            log(f"exact step 1 {arm} tau {tau}: R {(E00 - Ex) / gap:+.4f}; |dtheta| {np.linalg.norm(d):.3f}")
    log(f"S_T w_max {w.max():.3e}, w_min {w.min():.1e}, rank at cutoff {out['rank_cut']}, "
        f"eigenvalues below the 1e-3 shift: {out['n_below_shift']} of {len(w)}")
    P.dump(os.path.join(RES, "exact0.json"), out)


def run(a):
    t0 = time.time()
    ctx = P.ctx_for("che")
    fn, fth = os.path.join(RES, f"{a.arm}.json"), os.path.join(RES, f"theta_{a.arm}.npy")
    th = ctx.th0.clone()
    E00 = ctx.energy(th)[0]
    gap = E00 - E0
    valid = abs(E00 - ctx.stored) < 1e-8
    hist, chk, per = [], dict(E_theta0_diff=abs(E00 - ctx.stored)), None
    if os.path.exists(fn):
        d0 = json.load(open(fn))
        hist, chk, valid = d0["steps"], d0["check"], d0["valid"]
        th = torch.tensor(np.load(fth))
    while len(hist) < a.steps:
        c0 = P.cpu_min()
        if per is not None and c0 + per > a.cpu_cap:
            log(f"stop: next step would exceed the CPU cap ({c0:.1f} + {per:.1f} > {a.cpu_cap}); TRUNCATED")
            break
        n = len(hist) + 1
        E, pr, phi0 = ctx.energy(th)
        r = draw(ctx, pr, a.nsamp, n)
        ST, f, el = sr_stats(ctx, th, phi0, pr, r)
        E_s, se = float(el.mean()), float(el.std() / np.sqrt(len(el)))
        if n == 1:
            chk.update(E_samp=E_s, se=se, z=abs(E_s - E) / se)
            valid = valid and chk["z"] <= 6.0
            log(f"INVALID check: sampled E {E_s:.6f} +- {se:.1e} vs exact {E:.6f} ({chk['z']:.1f} se)")
        w, U = np.linalg.eigh(ST)
        rec = dict(E=E, E_samp=E_s, se=se, wmax=float(w.max()), fnorm=float(np.linalg.norm(f)))
        rv = draw(ctx, pr, a.nsamp, 1000 + n)
        elv = (ctx.Hr @ pr)[rv] / pr[rv]

        def score(dc, **kw):
            Ex, prn, _ = ctx.energy(th + torch.tensor(dc))
            wt = (prn[rv] / pr[rv]) ** 2
            eln = (ctx.Hr @ prn)[rv] / prn[rv]
            En = float(np.sum(wt * eln) / np.sum(wt))
            x = (wt / wt.mean()) * (eln - En) - (elv - elv.mean())
            return dict(kw, dE_val=En - float(elv.mean()), sig_val=float(x.std() / np.sqrt(len(x))), E_exact=Ex,
                        R_oracle=(E00 - Ex) / gap, dth=float(np.linalg.norm(dc)))

        cands, best = [], None
        if a.arm in ("trsr", "shift"):      # RULE_amend1: metric length chosen per step on the held-out set
            key = "tau"
            for tau in TAUS:
                if a.arm == "trsr":
                    dc, rank, gtn = d_trunc(w, U, f, CUT, tau)
                    cd = score(dc, tau=tau, rank=rank)
                else:
                    dc = d_shift(w, U, f, tau)[0]
                    cd = score(dc, tau=tau)
                cands.append(cd)
                if best is None or cd["dE_val"] < best[0]["dE_val"]:
                    best = (cd, dc)
        else:                               # reported-only cutoff ladder at metric length 0.05
            key = "cut"
            for cut in LADDER:
                dc, rank, gtn = d_trunc(w, U, f, cut)
                cd = score(dc, cut=cut, rank=rank, gt=gtn)
                cands.append(cd)
                if best is None or cd["dE_val"] < best[0]["dE_val"]:
                    best = (cd, dc)
        take = best[0]["dE_val"] < 0.0
        d = best[1] if take else np.zeros_like(best[1])
        rec.update(cands=cands, taken=bool(take), chosen=best[0][key] if take else None,
                   oracle=min(cands, key=lambda c: c["E_exact"])[key])
        th = th + torch.tensor(d)
        E_new = best[0]["E_exact"] if take else E
        rec.update(E_new=E_new, dth=float(np.linalg.norm(d)), R=(E00 - E_new) / gap,
                   dR=(E - E_new) / gap, eps=(E_new - E0) / abs(E0))
        hist.append(rec)
        per = P.cpu_min() - c0
        extra = "\n    " + "; ".join(f"{key} {c[key]:.0e}: val dE {c['dE_val']:+.2e}+-"
                                     f"{c['sig_val']:.1e} exact R {c['R_oracle']:+.4f}" for c in rec["cands"])
        log(f"{a.arm} step {n}: {key} {rec['chosen']}; exact R {rec['R']:+.4f} (step dR {rec['dR']:+.4f}); |dtheta| {rec['dth']:.3f}; "
            f"sampled E {E_s:.5f}+-{se:.1e}; w_max {rec['wmax']:.2e} [{P.cpu_min():.1f} CPU-min, step {per:.2f}, "
            f"wall {time.time() - t0:.0f} s]{extra}")
        np.save(fth, th.numpy())
        P.dump(fn, dict(arm=a.arm, nsamp=a.nsamp, E_theta0=E00, gap=gap, valid=valid, check=chk, steps=hist,
                        cpu_min=P.cpu_min(), wall=time.time() - t0))
    P.dump(fn, dict(arm=a.arm, nsamp=a.nsamp, E_theta0=E00, gap=gap, valid=valid, check=chk, steps=hist,
                    cpu_min=P.cpu_min(), wall=time.time() - t0, done=len(hist) >= a.steps))
    log(f"wrote {os.path.basename(fn)} ({time.time() - t0:.0f} s, {P.cpu_min():.1f} CPU-min)")


def summ(d, n=15):
    st = d["steps"][:n]
    return dict(R=st[-1]["R"] if st else 0.0, max_rise=max([-s["dR"] for s in st] + [0.0]),
                done=len(st) >= n, valid=bool(d.get("valid")), R_traj=[round(s["R"], 4) for s in st])


def verdict(a):
    S = {}
    for k in ("trsr", "shift", "ladder"):
        p = os.path.join(RES, f"{k}.json")
        if os.path.exists(p):
            S[k] = summ(json.load(open(p)))
            print(f"{k}: {S[k]}")
        else:
            print(f"{k}: missing")
    if not all(k in S and S[k]["done"] for k in ("trsr", "shift")):
        v = "INCOMPLETE"
    elif not (S["trsr"]["valid"] and S["shift"]["valid"]):
        v = "INVALID"
    else:                                   # RULE_amend1: best of the two main arms
        if any(S[k]["R"] >= R_VIABLE and S[k]["max_rise"] <= RISE_MAX for k in ("trsr", "shift")):
            v = "VIABLE"
        elif all(S[k]["R"] < R_DEAD for k in ("trsr", "shift")):
            v = "NOT-VIABLE"
        else:
            v = "MARGINAL"
    diff = S["trsr"]["R"] - S["shift"]["R"] if "trsr" in S and "shift" in S else None
    print(f"reported only: R_trsr(15) - R_shift(15) = {diff}")
    print(f"VERDICT (best of trsr, shift; RULE_amend1; ladder reported only): {v}")
    P.dump(os.path.join(RES, "verdict.json"), dict(arms=S, verdict=v, trsr_minus_shift=diff))


def main():
    global RES
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", required=True, choices=["exact0", "run", "verdict"])
    pa.add_argument("--arm", default="trsr", choices=["trsr", "shift", "ladder"])
    pa.add_argument("--nsamp", type=int, default=16384)
    pa.add_argument("--steps", type=int, default=15)
    pa.add_argument("--threads", default="1")
    pa.add_argument("--cpu-cap", type=float, default=60.0)
    pa.add_argument("--res", default=os.path.join(HERE, "results"))
    a = pa.parse_args()
    RES = a.res
    os.makedirs(RES, exist_ok=True)
    {"exact0": exact0, "run": run, "verdict": verdict}[a.mode](a)


if __name__ == "__main__":
    main()
