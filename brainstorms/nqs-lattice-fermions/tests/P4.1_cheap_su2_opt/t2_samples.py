#!/usr/bin/env python3
"""
P4.1 round 9, route R15 T2 (prepared, not submitted): is the 24-site `che` energy limited by the SR noise floor of
the 1024-sample protocol? Warm-started continuation at equal wall time, 1024 against 4096 samples per step.
================================================================================================================

Run with the lit_env interpreter:
    ~/.conda/envs/lit_env/bin/python t2_samples.py --mode MODE [options]

IMPORTS ../P4.1_cheap_su2_t2/vmc_t2.py (and through it the frozen vmc.py) read-only and calls vmc_t2.main with an
argument list; neither file is modified. The warm start uses vmc_t2's own resume path: this script writes a
checkpoint file <outdir>/<tag>_ckpt.npz with theta = the W-24 Polyak-averaged parameters (job 698224 task 0, che,
H = 16, 1000 steps, E = -16.43073 +- 0.00477), step 0, empty trajectory and average; vmc_t2.train then "resumes"
from step 0 and runs the full schedule of --iters steps from that state.

Arms (array index; both: 8x3 strip, N_e = 20, (+,+), che H = 16, seed 0, same schedule SHAPE as 686990 - constant lr
for the first half, cosine decay to 10 %, Polyak average of the last 25 % - with peak lr 0.02 (a warm start; the
same for both arms), diag shift 1e-3, final energy from 65,536 samples with chain-blocked error)
  0  A  1024 chains, T_A steps   (the 686990 sample count: the foil, "same ansatz, baseline optimiser")
  1  B  4096 chains, T_B steps   with T_B = round(T_A * t_A / t_B) from --mode timing (equal wall time)

Modes
-----
  timing   seconds per SR step at 24 sites, 4 threads, 1024 and 4096 chains (2 steps each after a short
           5-sweep thermalisation); writes <outdir>/timing.json. Used to fix T_B before submission.
  warm     write the warm-start checkpoint for --arm (if neither the result nor a checkpoint exists), then run
           vmc_t2 train. This is what the sbatch array runs.
  report   apply the pre-registered rule to <outdir>/{A,B}/*.json.
  check    logic check of the rule (CHECK PASS n/n).
  preflight  toy end-to-end on 4x2 (N_e = 6): trains a 40-step che, uses it as the warm start, runs both arms for
           (16, 8) steps with (64, 256) chains, then reports (exercises warm start, resume path, JSON, rule).

Pre-registered rule (written before any T2 run; `rule()`)
---------------------------------------------------------
  d = E_A - E_B, sigma = sqrt(dE_A^2 + dE_B^2); r = realised wall ratio (B training seconds / A training seconds).
  INCOMPLETE  if either result is missing or capped.
  INCOMPLETE-WALL if r > 1.15 (B had more than 15 % more wall time: rerun A at round(T_A * r) steps).
  INVALID     if E_B or E_A is above the warm-start energy -16.43073 by more than 0.05 (the continuation diverged).
  KILL        if d <= 2 sigma: at equal wall time 4x the samples does not lower the 24-site energy beyond 2 sigma,
              so the 1024-sample noise floor is not the 24-site limiter (R15 KILLED(T2)).
  PASS        if d > 2 sigma and d >= 0.01 (at least 0.01 in energy, i.e. 6e-4 in eps, about 5 % of the
              eps_che(24) = 1.05e-2 target): the protocol is sample-starved at 24 sites.
  AMBIGUOUS   otherwise (d > 2 sigma but below 0.01).
  Context, never gating: E_A - (-16.43073) (does more of the baseline optimiser help at all?), eps against the
  chi = 1000 DMRG -16.60459 (and against the 687302 reference when it exists), per-step times.

Options
-------
  --mode {timing, warm, report, check, preflight}  (required)
  --arm {A, B} (A)
  --iters INT        steps for this arm (required for warm)
  --threads INT (4)  --max-hours FLOAT (0.45)  --n-eval INT (65536)  --lr FLOAT (0.02)
  --outdir PATH (<script dir>/results_t2)
"""
import argparse
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
T2DIR = os.path.join(os.path.dirname(HERE), "P4.1_cheap_su2_t2")
sys.path.insert(0, T2DIR)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "P4.1_cheap_su2_scaling"))

import numpy as np  # noqa: E402

E_WARM = -16.43073
E_DMRG = -16.60459
W24 = os.path.join(T2DIR, "results/w24/S8x3_N20_U8_tp-0.25_pg11_che_H16_seed0")
TAG = "S8x3_N20_U8_tp-0.25_pg11_che_H16_seed0"
CHAINS = {"A": 1024, "B": 4096}


def rule(A, B):
    if A is None or B is None or A.get("capped") or B.get("capped"):
        return "INCOMPLETE", "missing or capped"
    tA = A["traj"][-1][3] if A["traj"] else A["wall"]
    tB = B["traj"][-1][3] if B["traj"] else B["wall"]
    r = tB / tA
    d = A["E"] - B["E"]
    s = math.hypot(A["dE"], B["dE"])
    msg = (f"E_A {A['E']:.5f} +- {A['dE']:.5f} ({A['iters']} steps, {A['hyper']['n_chains']} chains); "
           f"E_B {B['E']:.5f} +- {B['dE']:.5f} ({B['iters']} steps, {B['hyper']['n_chains']} chains); "
           f"d = {d:+.5f} +- {s:.5f} ({d / s:+.2f} sigma); wall ratio B/A {r:.3f}; "
           f"E_A - E_warm {A['E'] - E_WARM:+.5f}; eps_A {(A['E'] - E_DMRG) / abs(E_DMRG):.3e}, "
           f"eps_B {(B['E'] - E_DMRG) / abs(E_DMRG):.3e} (chi = 1000 DMRG)")
    if r > 1.15:
        return "INCOMPLETE-WALL", msg
    if A["E"] > E_WARM + 0.05 or B["E"] > E_WARM + 0.05:
        return "INVALID", msg
    if d <= 2 * s:
        return "KILL", msg
    if d >= 0.01:
        return "PASS", msg
    return "AMBIGUOUS", msg


def check():
    def mk(E, dE=0.0034, it=300, wall=1000.0, nc=1024, capped=False):
        return dict(E=E, dE=dE, iters=it, capped=capped, wall=wall, traj=[[it, E, 1.0, wall]],
                    hyper=dict(n_chains=nc))
    cases = [("PASS", mk(-16.44), mk(-16.47, nc=4096)),
             ("KILL", mk(-16.44), mk(-16.445, nc=4096)),
             ("KILL", mk(-16.46), mk(-16.44, nc=4096)),
             ("AMBIGUOUS", mk(-16.44, dE=0.002), mk(-16.4485, dE=0.002, nc=4096)),
             ("INCOMPLETE", mk(-16.44), None),
             ("INCOMPLETE", mk(-16.44, capped=True), mk(-16.47)),
             ("INCOMPLETE-WALL", mk(-16.44), mk(-16.47, wall=1200.0)),
             ("INVALID", mk(-16.37), mk(-16.47))]
    ok = 0
    for want, A, B in cases:
        got = rule(A, B)[0]
        ok += got == want
        print(f"  want {want:15s} got {got}")
    print(f"CHECK {'PASS' if ok == len(cases) else 'FAIL'} {ok}/{len(cases)}")


def write_warm(outdir, theta, tag=TAG):
    os.makedirs(outdir, exist_ok=True)
    ck = os.path.join(outdir, tag + "_ckpt.npz")
    if os.path.exists(ck) or os.path.exists(os.path.join(outdir, tag + ".json")):
        return False
    np.savez(ck, theta=theta, it=0, t_used=0.0, traj=json.dumps([]), cke=json.dumps({}),
             avg_sum=np.zeros_like(theta), avg_n=0)
    return True


def vt2_args(arm, iters, a, outdir, Lx=8, extra=()):
    return ["--mode", "train", "--arm", "che", "--width", "16", "--Lx", str(Lx), "--Ly", "3" if Lx == 8 else "2",
            "--pg=1,1", "--seed", "0", "--iters", str(iters), "--n-chains", str(CHAINS[arm] if Lx == 8 else
                                                                                 {"A": 64, "B": 256}[arm]),
            "--lr", str(a.lr), "--n-eval", str(a.n_eval), "--threads", str(a.threads),
            "--max-hours", str(a.max_hours), "--outdir", outdir, *extra]


def load(path):
    f = glob.glob(os.path.join(path, "*.json"))
    return json.load(open(f[0])) if f else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["timing", "warm", "report", "check", "preflight"])
    p.add_argument("--arm", default="A", choices=["A", "B"])
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-hours", type=float, default=0.45)
    p.add_argument("--n-eval", type=int, default=65536)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--outdir", default=os.path.join(HERE, "results_t2"))
    a = p.parse_args()
    assert a.threads <= 4
    if a.mode == "check":
        check()
        return
    if a.mode == "report":
        v, msg = rule(load(os.path.join(a.outdir, "A")), load(os.path.join(a.outdir, "B")))
        print(msg)
        print(f"VERDICT: {v}")
        return
    import torch
    torch.set_num_threads(a.threads)
    import vmc_t2 as VT
    import vmc as V
    if a.mode == "timing":
        out = {}
        for arm in ("A", "B"):
            b = VT.parse_args(vt2_args(arm, 10, a, a.outdir) + ["--therm-sweeps", "5"])
            m = VT.build(b)
            run = V.Run(m, b, 0)
            th = torch.tensor(np.load(W24 + "_theta.npy"))
            run.start_chains(th)
            run.step(th, 0.0)  # warm-up step, not timed
            t0 = time.time()
            for _ in range(2):
                run.step(th, 0.0)
            out[arm] = (time.time() - t0) / 2
            print(f"{arm}: {CHAINS[arm]} chains: {out[arm]:.2f} s/step ({a.threads} threads)", flush=True)
        out["ratio"] = out["A"] / out["B"]
        print(f"t_A / t_B = {out['ratio']:.4f}")
        json.dump(out, open(os.path.join(a.outdir, "timing.json"), "w"))
        return
    if a.mode == "warm":
        assert a.iters, "--iters is required"
        od = os.path.join(a.outdir, a.arm)
        if write_warm(od, np.load(W24 + "_theta.npy")):
            print(f"wrote warm-start checkpoint in {od} (theta = W-24 averaged parameters)", flush=True)
        VT.main(vt2_args(a.arm, a.iters, a, od))
        return
    if a.mode == "preflight":
        base = os.path.join(a.outdir, "preflight")
        seedd = os.path.join(base, "seedrun")
        a.n_eval, a.max_hours = 2048, 0.2
        VT.main(["--mode", "train", "--arm", "che", "--width", "16", "--Lx", "4", "--Ly", "2", "--pg=1,1",
                 "--iters", "40", "--n-chains", "64", "--n-eval", "2048", "--threads", str(a.threads),
                 "--outdir", seedd])
        th = np.load(os.path.join(seedd, "S4x2_N6_U8_tp-0.25_pg11_che_H16_seed0_theta.npy"))
        for arm, it in (("A", 16), ("B", 8)):
            od = os.path.join(base, arm)
            print(f"warm checkpoint written for {arm}: {write_warm(od, th, 'S4x2_N6_U8_tp-0.25_pg11_che_H16_seed0')}")
            VT.main(vt2_args(arm, it, a, od, Lx=4))
        A, B = load(os.path.join(base, "A")), load(os.path.join(base, "B"))
        print(f"preflight: A iters_done {A['iters_done']} chains {A['hyper']['n_chains']}; "
              f"B iters_done {B['iters_done']} chains {B['hyper']['n_chains']}")
        print(f"preflight rule output (toy, meaningless energies): {rule(A, B)[0]}")
        return


if __name__ == "__main__":
    main()
