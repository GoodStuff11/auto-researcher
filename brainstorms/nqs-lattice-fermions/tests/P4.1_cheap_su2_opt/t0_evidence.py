#!/usr/bin/env python3
"""
P4.1 round 9, route R15 (optimisation, not capacity, limits `che` at 24 sites): T0 evidence from existing runs.
===============================================================================================================

Reads ONLY finished or checkpointed VMC trajectories already on disk (read-only; no job file is touched):
  * W-24 (job 698224/700568): tests/P4.1_cheap_su2_t2/results/w24/*.json   (che, H = 16 and 48, 1000 steps)
  * W-12 / T2a (job 698223):  tests/P4.1_cheap_su2_t2/results/{w12,t2a}/*.json (12 sites, exact checkpoints)
  * T2b (job 699824, running): tests/P4.1_cheap_su2_t2b/results/t2b/*_ckpt.npz (3000-step protocol; the
    checkpoint energies at steps 750 and 1500 use 32,768 fresh samples; trajectory every 10 steps)
  * 12-site pilot: tests/P4.1_cheap_su2_scaling/logs/pilot/pg_che/*.json

For every trajectory (sampled energy of the CURRENT parameters at every 10th step, 1024 samples each) it prints
  - 250-step block means with standard errors (scatter of the ~25 points in a block / sqrt(n)),
  - the least-squares slope dE/dstep (per 1000 steps) over the last 250 logged steps of the constant-lr phase
    and over the last 250 steps of the run, with its standard error,
  - the learning rate at the end of each window (vmc.lr_at schedule).
It also prints the 24-site T2b checkpoint energies against the W-24 1000-step averaged energy, and the
epsilon values against the chi = 1000 DMRG (+,+) energy -16.60459 (the extrapolated 687302 value is not
available yet; it can only be lower, which raises every epsilon by the same amount).

Usage (any python3 with numpy; no options):
    python3 t0_evidence.py > logs/t0_evidence_output.txt
"""
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.dirname(HERE)
E_DMRG24 = -16.60459  # chi = 1000 DMRG, clean (+,+) characters (leaf: R2 T2b power sizing)
E0_12 = -7.827899034271753


def lr_at(it, iters, lr=0.05, dec=0.1):
    if it <= iters / 2:
        return lr
    x = (it - iters / 2) / (iters / 2)
    return lr * (dec + (1 - dec) * 0.5 * (1 + math.cos(math.pi * x)))


def slope(tr, lo, hi):
    t = np.array([r for r in tr if lo < r[0] <= hi])
    if len(t) < 5:
        return None
    x, y = t[:, 0] / 1000.0, t[:, 1]
    A = np.vstack([x, np.ones_like(x)]).T
    c, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    s2 = r @ r / (len(y) - 2)
    se = math.sqrt(s2 / np.sum((x - x.mean()) ** 2))
    return c[0], se, len(y)


def blocks(tr, iters, w=250):
    out = []
    for lo in range(0, iters, w):
        y = np.array([r[1] for r in tr if lo < r[0] <= lo + w])
        if len(y) >= 3:
            out.append((lo, lo + w, y.mean(), y.std(ddof=1) / math.sqrt(len(y))))
    return out


def report(name, tr, iters, L):
    done = max(r[0] for r in tr)
    print(f"\n== {name}: iters {iters}, logged up to step {done}")
    for lo, hi, m, se in blocks(tr, iters):
        eps = (m - (E_DMRG24 if L == 24 else E0_12)) / abs(E_DMRG24 if L == 24 else E0_12)
        print(f"   steps ({lo:4d},{hi:4d}]  <E_sampled> {m:11.5f} +- {se:.5f}   eps {eps:.3e}   lr(end) {lr_at(hi, iters):.4f}")
    half = iters // 2
    for lab, lo, hi in (("constant-lr tail", half - 250, min(half, done)), ("last 250 logged", done - 250, done)):
        s = slope(tr, lo, hi)
        if s:
            print(f"   slope {lab:17s} steps ({lo},{hi}]: dE/d(1000 steps) = {s[0]:+.4f} +- {s[1]:.4f} "
                  f"(t = {s[0] / s[1]:+.2f}, n = {s[2]})")


def main():
    print("R15 T0 evidence: is the 24-site che energy still falling at the end of its budget?")
    for f in sorted(glob.glob(os.path.join(T, "P4.1_cheap_su2_t2/results/w24/*.json"))):
        d = json.load(open(f))
        report(os.path.basename(f) + f"  [final avg E {d['E']:.5f} +- {d['dE']:.5f}, capped {d['capped']}]",
               d["traj"], d["iters"], d["L"])
    for f in sorted(glob.glob(os.path.join(T, "P4.1_cheap_su2_t2b/results/t2b/S8x3*_ckpt.npz"))):
        z = np.load(f, allow_pickle=True)
        tr, cke = json.loads(str(z["traj"])), json.loads(str(z["cke"]))
        report(os.path.basename(f) + " [running, 3000-step protocol]", tr, 3000, 24)
        for k, v in sorted(cke.items(), key=lambda kv: int(kv[0])):
            print(f"   CHECKPOINT step {k}: E {v['E']:.5f} +- {v['dE']:.5f} ({v['n']} samples), "
                  f"eps {(v['E'] - E_DMRG24) / abs(E_DMRG24):.3e}")
    for sub in ("w12", "t2a"):
        for f in sorted(glob.glob(os.path.join(T, f"P4.1_cheap_su2_t2/results/{sub}/*.json"))):
            d = json.load(open(f))
            ck = " ".join(f"{k}:{v['eps']:.3e}" for k, v in sorted(d["ckpt_exact"].items(), key=lambda kv: int(kv[0])))
            print(f"\n== 12 sites {os.path.basename(f)}: exact eps at checkpoints {ck}; final (avg) eps {d['eps']:.3e}")
    f = glob.glob(os.path.join(T, "P4.1_cheap_su2_t2b/results/t2b/S4x3*pcg*_ckpt.npz"))
    for g in f:
        z = np.load(g, allow_pickle=True)
        cke = json.loads(str(z["cke"]))
        print(f"\n== 12 sites {os.path.basename(g)} (3000 steps, running): exact eps "
              + " ".join(f"{k}:{v['eps']:.3e}" for k, v in sorted(cke.items(), key=lambda kv: int(kv[0]))))
    p = os.path.join(T, "P4.1_cheap_su2_scaling/logs/pilot/pg_che/S4x3_N10_U8_tp-0.25_pg11_che_seed0.json")
    d = json.load(open(p))
    report(f"12-site pilot pg_che [final avg exact E {d['E']:.6f}, eps {(d['E'] - E0_12) / abs(E0_12):.3e}]",
           d["traj"], d["iters"], 12)
    # direct comparison: W-24 averaged final vs T2b che checkpoints
    w = json.load(open(os.path.join(T, "P4.1_cheap_su2_t2/results/w24/S8x3_N20_U8_tp-0.25_pg11_che_H16_seed0.json")))
    z = np.load(os.path.join(T, "P4.1_cheap_su2_t2b/results/t2b/S8x3_N20_U8_tp-0.25_pg11_che_H16_seed0_ckpt.npz"),
                allow_pickle=True)
    cke = json.loads(str(z["cke"]))
    print("\n== 24-site che, H = 16: W-24 averaged final (1000 steps) vs T2b current-parameter checkpoints")
    for k, v in sorted(cke.items(), key=lambda kv: int(kv[0])):
        dd = v["E"] - w["E"]
        s = math.hypot(v["dE"], w["dE"])
        print(f"   E(T2b step {k}) - E(W-24 avg) = {dd:+.5f} +- {s:.5f}  ({dd / s:+.2f} sigma)")
    if "750" in cke and "1500" in cke:
        dd = cke["1500"]["E"] - cke["750"]["E"]
        s = math.hypot(cke["1500"]["dE"], cke["750"]["dE"])
        print(f"   E(1500) - E(750) at constant lr 0.05 = {dd:+.5f} +- {s:.5f}  ({dd / s:+.2f} sigma)")


if __name__ == "__main__":
    main()
