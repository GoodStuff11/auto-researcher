#!/usr/bin/env python3
"""
P5.1 round 15, route R51.7 `rmavg`: step decay with Polyak-Ruppert averaging from R51.8's step-120 states.
=========================================================================================================
Rule: RULE_r14_decay.md (pre-registered).  Run from this directory with the lit_env interpreter:
    ~/.conda/envs/lit_env/bin/python ns_decay.py --src results/<R51.8 arm> --mode {cont,dec} [options]

Continues one R51.8 arm (`ns_run.py` results dir; read only: run.json for the lattice, sampler, N, seed0 and start;
state.npz for the step-120 theta) for --extra steps (default 60) with exactly ns_run.py's step (same GCtx, stats,
d_shift; imported from ns_run.py, not copied), step n = 120 + k drawing with seed seed0 + n, k = 1..60 (common random
numbers: CONT and DEC of one arm use the same seed at the same n).
  cont : metric length tau0 (fixed, 0.005); scored energy E(theta_n).                 (== ns_run.py --steps 180)
  dec  : length tau_k = tau0 (1 + k/10)^(-0.75); running Polyak-Ruppert average
         thbar_k = mean(theta_121 .. theta_{120+k}); scored energy E(thbar_k) (one extra exact energy per step).
Every step records E_iter = E(theta_n) (both modes; it is also the next step's |psi|^2) and, in dec, E_avg.
`E_score` is the rule's scored energy (E_iter in cont, E_avg in dec).

Options
  --src DIR        R51.8 arm (must hold a DONE run.json with 120 steps and state.npz at step 120).
  --mode           cont | dec.
  --extra INT      continuation steps (default 60).     --tau0 FLOAT (default 0.005).
  --cpu-cap FLOAT  CPU-minutes for this invocation; stop before a step that would exceed it.
  --res DIR        default results/r14_decay/<basename(src)>_<mode>.
Output: <res>/run.json (status DONE | TRUNCATED | RUNNING; per step n, k, tau, E_iter, E_avg, E_score, n_unique, dth,
cpu_step) and <res>/state.npz (theta, running parameter sum for the average, history, CPU), saved atomically after
every step.  Rerunning the same command resumes (same seeds).  SIGUSR1 / SIGTERM: finish the current step, save
TRUNCATED, exit 0 (sbatch: `#SBATCH --signal=B:USR1@600` and a trap forwarding the signal to this process).
Identity test (RULE_r14_decay.md): with --mode cont the energies equal ns_run.py resumed from a copy of the same
state.npz with --steps 180 (checked to 1e-12 for 3 steps; leaves/P5.1/log/round-15.md).
"""
import argparse
import json
import os
import signal
import time

import ns_run as NR  # sets the thread env vars, imports trsr/torch; main() is guarded
import numpy as np
import torch

HERE = NR.HERE
log, cpu_min = NR.log, NR.cpu_min
NSTEP0 = 120  # R51.8's run length: the continuation starts from the step-120 state


def parse():
    pa = argparse.ArgumentParser()
    pa.add_argument("--src", required=True)
    pa.add_argument("--mode", required=True, choices=["cont", "dec"])
    pa.add_argument("--extra", type=int, default=60)
    pa.add_argument("--tau0", type=float, default=0.005)
    pa.add_argument("--cpu-cap", type=float, default=600.0)
    pa.add_argument("--res", default=None)
    a = pa.parse_args()
    a.src = os.path.abspath(a.src)
    if a.res is None:
        a.res = os.path.join(HERE, "results", "r14_decay", f"{os.path.basename(a.src.rstrip('/'))}_{a.mode}")
    return a


def tau_of(a, k):
    return a.tau0 if a.mode == "cont" else a.tau0 * (1.0 + k / 10.0) ** (-0.75)


def main():
    a = parse()
    src = json.load(open(os.path.join(a.src, "run.json")))
    assert src["status"] == "DONE" and len(src["steps"]) == NSTEP0, (a.src, src["status"], len(src["steps"]))
    assert abs(src["tau"] - a.tau0) < 1e-15, (src["tau"], a.tau0)
    z0 = np.load(os.path.join(a.src, "state.npz"), allow_pickle=False)
    assert len(json.loads(str(z0["hist"]))) == NSTEP0
    th120 = np.array(z0["theta"])
    exact, sampler, nsamp, seed0 = src["exact"], src["sampler"], src["nsamp"], src["seed0"]
    start = src["start"] if os.path.isabs(src["start"]) else os.path.join(HERE, src["start"])
    c00 = cpu_min()
    ctx = NR.GCtx(src["Lx"], src["Ly"], src["Ne"], start)
    log(f"setup {cpu_min() - c00:.2f} CPU-min; src {a.src}; mode {a.mode}")
    os.makedirs(a.res, exist_ok=True)
    t0 = time.time()
    fn, fst = os.path.join(a.res, "run.json"), os.path.join(a.res, "state.npz")
    th = torch.tensor(th120)
    thsum = np.zeros_like(th120)
    hist, cpu_prev = [], 0.0
    if os.path.exists(fst):
        z = np.load(fst, allow_pickle=False)
        th = torch.tensor(z["theta"])
        thsum = np.array(z["thsum"])
        hist = json.loads(str(z["hist"]))
        cpu_prev = float(z["cpu_prev"])
        log(f"resumed from {fst} at k = {len(hist)} (previous runs {cpu_prev:.2f} CPU-min)")
    stop = {"why": None}

    def on_signal(signum, frame):
        stop["why"] = f"signal {signum}"
    signal.signal(signal.SIGUSR1, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    E, pr, phi0 = ctx.energy(th)
    if not hist:  # consistency: the step-120 state reproduces R51.8's recorded step-120 energy
        dE = abs(E - src["steps"][-1]["E"])
        log(f"E(theta_120) {E:.12f}, R51.8 recorded {src['steps'][-1]['E']:.12f} (|diff| {dE:.1e})")
        assert dE < 1e-10, dE

    def save(status):
        rec = dict(src=a.src, mode=a.mode, Lx=src["Lx"], Ly=src["Ly"], Ne=src["Ne"], exact=exact, sampler=sampler,
                   nsamp=nsamp, seed0=seed0, tau0=a.tau0, schedule="fixed" if a.mode == "cont" else
                   "tau0*(1+k/10)^-0.75 + Polyak-Ruppert average from step 121", E_start=src["E_start"],
                   E_120=src["steps"][-1]["E"], steps=hist, status=status, steps_target=a.extra,
                   cpu_min=cpu_prev + cpu_min(), wall=time.time() - t0)
        tmp = fst[:-4] + ".tmp.npz"
        np.savez(tmp, theta=th.numpy(), thsum=thsum, hist=json.dumps(hist), cpu_prev=cpu_prev + cpu_min())
        os.replace(tmp, fst)
        json.dump(rec, open(fn + ".tmp", "w"), indent=1)
        os.replace(fn + ".tmp", fn)

    per = None
    status = "RUNNING"
    tag = "exact" if exact else f"{sampler} N {nsamp}"
    while len(hist) < a.extra:
        c0 = cpu_min()
        if per is not None and c0 + per > a.cpu_cap:
            stop["why"] = f"CPU cap ({c0:.1f} + {per:.2f} > {a.cpu_cap})"
        if stop["why"]:
            status = "TRUNCATED"
            log(f"stop before k {len(hist) + 1}: {stop['why']}; state saved, resume by rerunning the same command")
            break
        k = len(hist) + 1
        n = NSTEP0 + k
        tau = tau_of(a, k)
        ST, f, nu = NR.stats(ctx, th, phi0, pr, nsamp, seed0 + n, sampler, exact)
        w, U = np.linalg.eigh(ST)
        d = NR.T.d_shift(w, U, f, tau)[0]
        th = th + torch.tensor(d)
        En, pr, phi0 = ctx.energy(th)
        rec = dict(n=n, k=k, tau=tau, E_iter=En, dE=E - En, n_unique=nu, dth=float(np.linalg.norm(d)))
        if a.mode == "dec":
            thsum = thsum + th.numpy()
            rec["E_avg"] = ctx.energy(torch.tensor(thsum / k))[0]
            rec["E_score"] = rec["E_avg"]
        else:
            rec["E_score"] = En
        E = En
        hist.append(rec)
        per = cpu_min() - c0
        rec["cpu_step"] = per
        log(f"S{src['Lx']}x{src['Ly']} {tag} {a.mode} step {n} (k {k}, tau {tau:.5f}): E_iter {En:.12f}"
            + (f" E_avg {rec['E_avg']:.12f}" if a.mode == "dec" else "")
            + f"; unique {nu} [{cpu_min():.2f} CPU-min, step {per:.3f}]")
        save("RUNNING")
    if len(hist) >= a.extra:
        status = "DONE"
    save(status)
    log(f"{status}: {len(hist)} of {a.extra} steps, {cpu_prev + cpu_min():.2f} CPU-min in all, "
        f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
