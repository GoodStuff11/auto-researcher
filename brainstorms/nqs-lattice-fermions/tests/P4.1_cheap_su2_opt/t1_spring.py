#!/usr/bin/env python3
"""
P4.1 round 9, route R15 T1: does a cost-neutral better optimiser (SPRING = minSR + Kaczmarz momentum) lower the
energy of the exact-S `che` ansatz at equal wall time, in the sample-starved regime of the 24-site runs?
================================================================================================================

Run with the lit_env interpreter (torch, float64, CPU):
    ~/.conda/envs/lit_env/bin/python t1_spring.py --mode MODE [options]

IMPORTS the frozen ../P4.1_cheap_su2_scaling/vmc.py read-only (Model, Run, lr_at, exact sums). It never edits it
(jobs 686990 and, through vmc_t2.py, 699824 / 700568 / 703587 run it). SPRING is a subclass of vmc.Run in this
file; the baseline arm is vmc.Run itself, unchanged.

Why this toy (written before any run)
-------------------------------------
The 24-site production runs have N_s / P = 1024 / 9711 = 0.105 samples per parameter per step (sample-starved
SR). T1 reproduces that ratio on the largest exact-sum toy where the che ansatz is far from its capacity ceiling:
open 5x2 strip, N_e = 8, U = 8, t' = -0.25, irrep (+,+) via P_G (1,1), che at H = 4 (P = 620 parameters),
N_s = 64 chains (64 / 620 = 0.103). ED E0 = -6.5490569 (round-8 smoke). Exact-gradient L-BFGS of the same ansatz
reaches eps = 7.1e-4 (round-8 T1), so any eps far above that is optimisation-limited, which is the regime claimed
for 24 sites. The toy is a MECHANISM test (does the optimiser lower a noise floor at equal cost); it is not a
Gate 4 size trend (round-6 anti-pattern: exact-sum trends below about 16 sites cannot decide such routes).

Arms (each seeds 0, 1, 2; every arm uses the 686990 schedule: lr 0.05 constant for the first half, cosine decay
to 10 %, Polyak average of the last 25 %, diag shift 1e-3, update-norm cap 0.5, ceil(N_e/2) proposals per step)
  base     vmc.Run sample-space SR (minSR / Rende et al. form), N_s = 64, T = --iters steps.
  spring   SPRING (Goldshlager, Abrahamsen, Lin, arXiv:2401.10190): phi_k = mu phi_{k-1} +
           O^T (O O^T + lam I)^{-1} (eps - mu O phi_{k-1}), theta -= lr phi_k (same cap), mu = --mu (0.9),
           lam = diag shift; N_s = 64, T steps. Extra cost per step: one O @ phi product (negligible).
  base4    the trivial variance-reduction foil: vmc.Run with N_s = 256 and round(0.58 T) steps, i.e. equal WALL
           time. Amendment A1 (before any run output): the draft gave T/4 steps on the assumption that the cost is
           linear in N_s; the timing mode measured 0.0708 s/step at N_s = 64 and 0.1218 s/step at N_s = 256 on 2
           threads (logs/timing_output.txt), so T/4 steps would give base4 only 0.43 of base's wall time. The
           fraction is 0.0708 / 0.1218 = 0.58.
  ctrl     POSITIVE CONTROL (premise "noise binds"): vmc.Run with N_s = 256 and T steps (4 times the compute).

Score: eps = (E - E0)/|E0| of the Polyak-averaged parameters, EXACT (all 44,100 configurations, P_G applied).
Wall time is measured per run; "equal wall time" is checked, not assumed (rule clause W below).

Pre-registered rule (written before any `run` output existed; `--mode report` applies it)
-----------------------------------------------------------------------------------------
  Means over the 3 seeds: e_X = mean eps_X; SE_X = std(eps_X, ddof=1)/sqrt(3); s_XY = sqrt(SE_X^2 + SE_Y^2).
  INVALID      if any final |<S^2>| > 1e-10 or any energy is non-finite.
  W            wall-time check: if mean wall(spring) > 1.15 * mean wall(base), spring is NOT at equal wall time
               and the verdict is INCOMPLETE-WALL (rerun base with ceil(ratio * T) steps, then report again).
  NO-TARGET    if e_ctrl >= e_base - 2 s_ctrl,base: 4x the samples does not help, so the toy is not noise-limited
               and the optimiser question has no target here.
  KILL         if e_spring >= e_base - 2 s_spring,base  (SPRING does not lower the energy beyond 2 sigma at equal
               wall time), OR e_spring >= e_base4 (it does not beat the trivial more-samples foil).
  PASS         if e_spring <= 0.8 e_base and e_spring < e_base4 - 2 s_spring,base4.
  AMBIGUOUS    otherwise.
  Reported, not gating: e_base4 / e_base (does plain variance reduction at equal cost help?), all per-seed eps,
  walls, and the constant-lr-phase plateau of each arm (exact energy of the current parameters at T/2).

Modes
-----
  timing   time 20 steps of base and spring (N_s = 64) and of base (N_s = 256); prints s/step. No files.
  run      train one (--arm, --seed) and write results/<arm>_seed<seed>.json (skip if it exists).
  all      run every arm and seed in sequence (respects --budget-min: aborts before a run that would exceed it,
           using the measured per-step time), then report.
  report   apply the rule to results/*.json; prints the table and the verdict.
  check    logic check of the rule on synthetic inputs (prints CHECK PASS n/n).

Options
-------
  --mode {timing, run, all, report, check}  (required)
  --arm {base, spring, base4, ctrl} (base)   --seed INT (0)
  --iters INT (250)   T, the step budget of base and spring
  --mu FLOAT (0.9)    SPRING momentum
  --threads INT (2)   torch threads (cap 4)
  --budget-min FLOAT (9.5)  CPU-minute cap for --mode all (threads * wall)
  --outdir PATH (<script dir>/results)
"""
import argparse
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCAL = os.path.join(os.path.dirname(HERE), "P4.1_cheap_su2_scaling")
sys.path.insert(0, SCAL)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vmc as V  # noqa: E402  (frozen; read-only)

E0_5x2 = -6.5490569
LX, LY, NE, H = 5, 2, 8, 4
ARMS = {"base": (64, 1.0, False), "spring": (64, 1.0, True), "base4": (256, 0.58, False), "ctrl": (256, 1.0, False)}


class SpringRun(V.Run):
    """vmc.Run with the SPRING update; everything else (sampling, local energy, log-derivatives) inherited."""

    def __init__(self, model, a, seed, mu):
        super().__init__(model, a, seed)
        self.mu, self.phi = mu, None

    def step(self, th, lr=None):
        m, a = self.m, self.a
        lr = a.lr if lr is None else lr
        t0 = time.time()
        self.la, self.sa = m.logpsi_batched(th, self.nU, self.nD, a.chunk)
        acc = m.mcmc(th, self.nU, self.nD, self.la, self.sa, self.nprop, self.gen, a.chunk)
        t1 = time.time()
        El = m.local_energy(th, self.nU, self.nD, self.la, self.sa, a.chunk)
        t2 = time.time()
        O = m.logderiv(th, self.nU, self.nD, a.grad_chunk)
        t3 = time.time()
        N = El.shape[0]
        Ob = (O - O.mean(0)) / math.sqrt(N)
        eb = (El - El.mean()) / math.sqrt(N)
        Tm = Ob @ Ob.T
        Tm.diagonal().add_(a.diag_shift)
        if self.phi is None:
            self.phi = torch.zeros(Ob.shape[1], dtype=Ob.dtype)
        rhs = eb - self.mu * (Ob @ self.phi)
        self.phi = self.mu * self.phi + Ob.T @ torch.linalg.solve(Tm, rhs)
        upd = lr * self.phi
        nrm = float(torch.linalg.norm(upd))
        if nrm > a.max_step:
            upd = upd * (a.max_step / nrm)
        t4 = time.time()
        info = dict(E=float(El.mean()), Estd=float(El.std()), acc=acc, upd=nrm,
                    t_mcmc=t1 - t0, t_eloc=t2 - t1, t_grad=t3 - t2, t_sr=t4 - t3)
        return th - upd, info


def ns(arm, a):
    nc, frac, _ = ARMS[arm]
    return argparse.Namespace(n_chains=nc, iters=max(1, int(round(a.iters * frac))), lr=0.05, lr_decay=0.1,
                              avg_frac=0.25, diag_shift=1e-3, max_step=0.5, sr_norm=0, sweeps_per_step=0.5,
                              therm_sweeps=50, chunk=4096, grad_chunk=256)


def model():
    return V.Model("che", LX, LY, NE, H, -0.25, 8.0, (1, 1))


def train(arm, seed, a, sec=None, m=None):
    torch.set_num_threads(a.threads)
    m = m or model()
    sec = sec or V.exact_sector(m)
    b = ns(arm, a)
    run = SpringRun(m, b, seed, a.mu) if ARMS[arm][2] else V.Run(m, b, seed)
    th = m.init(seed)
    run.start_chains(th)
    avg_start = int(round(b.iters * 0.75))
    avg, n = torch.zeros_like(th), 0
    t0 = time.time()
    Ehalf = None
    for it in range(1, b.iters + 1):
        th, info = run.step(th, V.lr_at(b, it))
        if not np.isfinite(info["E"]):
            return dict(arm=arm, seed=seed, finite=False)
        if it > avg_start:
            avg += th
            n += 1
        if it == b.iters // 2:
            Ehalf = V.exact_energy(m, th, sec)[0]
    wall = time.time() - t0
    th = avg / n
    E, psi = V.exact_energy(m, th, sec)
    S2 = float(np.sum(psi * sec.S2(psi)))
    return dict(arm=arm, seed=seed, finite=bool(np.isfinite(E)), iters=b.iters, n_chains=b.n_chains, mu=a.mu,
                P=m.nparam, E=E, eps=(E - E0_5x2) / abs(E0_5x2), S2=S2, wall=wall, threads=a.threads,
                eps_half_current=(Ehalf - E0_5x2) / abs(E0_5x2) if Ehalf is not None else None)


def rule(R):
    """R: dict arm -> list of result dicts. Returns (verdict, lines)."""
    lines = []
    for arm in ARMS:
        if arm not in R or len(R[arm]) < 3:
            return "INCOMPLETE", [f"missing seeds for {arm}"]
    if any((not r["finite"]) or abs(r["S2"]) > 1e-10 for rs in R.values() for r in rs):
        return "INVALID", ["non-finite energy or |<S^2>| > 1e-10"]
    e = {k: float(np.mean([r["eps"] for r in v])) for k, v in R.items()}
    se = {k: float(np.std([r["eps"] for r in v], ddof=1) / math.sqrt(len(v))) for k, v in R.items()}
    w = {k: float(np.mean([r["wall"] for r in v])) for k, v in R.items()}
    s = lambda x, y: math.hypot(se[x], se[y])  # noqa: E731
    for k in ARMS:
        lines.append(f"  {k:7s} eps {e[k]:.4e} +- {se[k]:.1e}   wall {w[k]:.1f} s   per seed "
                     + " ".join(f"{r['eps']:.3e}" for r in R[k]))
    lines.append(f"  e_base4/e_base = {e['base4'] / e['base']:.3f}; e_spring/e_base = {e['spring'] / e['base']:.3f}; "
                 f"wall ratio spring/base = {w['spring'] / w['base']:.3f}, base4/base = {w['base4'] / w['base']:.3f}")
    if w["spring"] > 1.15 * w["base"]:
        return "INCOMPLETE-WALL", lines
    if e["ctrl"] >= e["base"] - 2 * s("ctrl", "base"):
        return "NO-TARGET", lines
    if e["spring"] >= e["base"] - 2 * s("spring", "base") or e["spring"] >= e["base4"]:
        return "KILL", lines
    if e["spring"] <= 0.8 * e["base"] and e["spring"] < e["base4"] - 2 * s("spring", "base4"):
        return "PASS", lines
    return "AMBIGUOUS", lines


def check():
    def mk(vals, wall=10.0, S2=0.0):
        return {k: [dict(eps=x, wall=(wall if k != "base4" else 10.0) if k != "spring" else wall * 1.0,
                         S2=S2, finite=True) for x in v] for k, v in vals.items()}
    base = [1.0e-2, 1.1e-2, 0.9e-2]
    cases = [
        ("PASS", dict(base=base, spring=[5e-3, 5.2e-3, 4.8e-3], base4=[8e-3, 8.2e-3, 7.8e-3], ctrl=[5e-3] * 3)),
        ("KILL", dict(base=base, spring=[0.98e-2, 1.0e-2, 1.02e-2], base4=[8e-3] * 3, ctrl=[5e-3, 5.1e-3, 4.9e-3])),
        ("KILL", dict(base=base, spring=[6e-3, 6.1e-3, 5.9e-3], base4=[5e-3, 5.1e-3, 4.9e-3], ctrl=[4e-3] * 3)),
        ("NO-TARGET", dict(base=base, spring=[5e-3] * 3, base4=[9e-3] * 3, ctrl=[1.0e-2, 0.95e-2, 1.05e-2])),
        ("AMBIGUOUS", dict(base=base, spring=[8.5e-3, 8.6e-3, 8.4e-3], base4=[9.5e-3, 9.6e-3, 9.4e-3],
                           ctrl=[5e-3] * 3)),
        ("INCOMPLETE", dict(base=base, spring=[5e-3] * 2, base4=[8e-3] * 3, ctrl=[5e-3] * 3)),
    ]
    ok = 0
    for want, vals in cases:
        got = rule(mk(vals))[0]
        ok += got == want
        print(f"  want {want:11s} got {got}")
    R = mk(dict(base=base, spring=[5e-3] * 3, base4=[8e-3] * 3, ctrl=[5e-3] * 3))
    for r in R["spring"]:
        r["wall"] = 13.0
    got = rule(R)[0]
    ok += got == "INCOMPLETE-WALL"
    print(f"  want INCOMPLETE-WALL got {got}")
    R = mk(dict(base=base, spring=[5e-3] * 3, base4=[8e-3] * 3, ctrl=[5e-3] * 3), S2=1e-6)
    got = rule(R)[0]
    ok += got == "INVALID"
    print(f"  want INVALID     got {got}")
    print(f"CHECK {'PASS' if ok == 8 else 'FAIL'} {ok}/8")


def load(outdir):
    R = {}
    for f in sorted(glob.glob(os.path.join(outdir, "*_seed*.json"))):
        r = json.load(open(f))
        R.setdefault(r["arm"], []).append(r)
    return R


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["timing", "run", "all", "report", "check"])
    p.add_argument("--arm", default="base", choices=list(ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=250)
    p.add_argument("--mu", type=float, default=0.9)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--budget-min", type=float, default=9.5)
    p.add_argument("--outdir", default=os.path.join(HERE, "results"))
    a = p.parse_args()
    assert a.threads <= 4
    torch.set_num_threads(a.threads)
    os.makedirs(a.outdir, exist_ok=True)
    if a.mode == "check":
        check()
        return
    if a.mode == "timing":
        m = model()
        for arm in ("base", "spring", "base4"):
            b = ns(arm, a)
            run = SpringRun(m, b, 0, a.mu) if ARMS[arm][2] else V.Run(m, b, 0)
            th = m.init(0)
            run.start_chains(th)
            t0 = time.time()
            for _ in range(20):
                th, _ = run.step(th)
            print(f"{arm}: N_s {b.n_chains}: {(time.time() - t0) / 20:.4f} s/step ({a.threads} threads)")
        return
    if a.mode == "report":
        R = load(a.outdir)
        v, lines = rule(R)
        print("\n".join(lines))
        print(f"VERDICT: {v}")
        return
    jobs = [(a.arm, a.seed)] if a.mode == "run" else [(arm, s) for s in (0, 1, 2) for arm in ARMS]
    m = model()
    sec = V.exact_sector(m)
    used = 0.0
    for arm, seed in jobs:
        f = os.path.join(a.outdir, f"{arm}_seed{seed}.json")
        if os.path.exists(f):
            print(f"{f} exists; skipping")
            continue
        t0 = time.time()
        r = train(arm, seed, a, sec, m)
        used += (time.time() - t0) * a.threads / 60
        json.dump(r, open(f, "w"))
        print(f"{arm} seed {seed}: eps {r.get('eps', float('nan')):.4e} (T/2 current {r.get('eps_half_current')}) "
              f"wall {r.get('wall', 0):.1f} s; CPU-min used {used:.2f}", flush=True)
        if used > a.budget_min:
            print(f"BUDGET {a.budget_min} CPU-min exceeded; stopping")
            break
    if a.mode == "all":
        v, lines = rule(load(a.outdir))
        print("\n".join(lines))
        print(f"VERDICT: {v}")


if __name__ == "__main__":
    main()
