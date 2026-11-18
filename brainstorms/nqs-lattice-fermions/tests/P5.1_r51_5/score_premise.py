#!/usr/bin/env python3
"""R51.5 `mom` premise (T1) scorer, rule pre-registered on card R51.5 (2026-11-17, before any data).
Arm: fixed-step guided SR, 4x3 `che`, N = 4096 per step, tau 0.005, seed0 9000, 240 steps (0.98M draws).
R = (E_start - E)/gap with R51.1's own start and gap, read from arm A's run.json, so R is on the scale of the card's
0.494 (arm A's stored R at step 60).  NB ../P5.1_r05_gen/score_common.py uses E0 from info_S4x3_N10.json, a larger gap
(0.02311 vs 0.02237): on that scale arm A's step-60 R is 0.478, not 0.494.
KILLED if R(step 240) < 0.247 (half of R51.1 arm A's 0.494 at step 60 = 0.98M draws) or if more than a third of
the 240 steps raise the exact energy (> 80 rises); SURVIVES otherwise (releases the T2 K = 4 median arm);
INCOMPLETE if fewer than 240 steps exist.
Usage: python3 score_premise.py [results_dir]   (default results/guided_N4096_tau0.005_s9000)"""
import json
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
_A = json.load(open(os.path.join(H, "..", "P5.1_noise_floor_r01", "results", "N16384_tau0.005_s9000", "run.json")))
ES, GAP = _A["E_theta0"], _A["gap"]  # R51.1 arm A: start energy and gap (its R = (ES - E)/GAP)
STEPS, R_MIN, RISE_MAX = 240, 0.247, 240 / 3

res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "results", "guided_N4096_tau0.005_s9000")
d = json.load(open(os.path.join(res, "run.json")))
E = np.array([s["E"] for s in d["steps"]])
assert abs(d["E_start"] - ES) < 1e-9, "not R51.1's start"
R = (ES - E) / GAP
rises_all = np.diff(np.concatenate([[d["E_start"]], E])) > 0
n = len(E)
out = dict(rule="card R51.5 premise (T1)", res=os.path.relpath(res, H), nsamp=d["nsamp"], steps_done=n,
           draws=int(n * d["nsamp"]))
if n < STEPS:
    out.update(verdict="INCOMPLETE", R_last=float(R[-1]) if n else None, rises=int(rises_all.sum()))
else:
    r240, rises = float(R[STEPS - 1]), int(rises_all[:STEPS].sum())
    kill_R, kill_rise = r240 < R_MIN, rises > RISE_MAX
    out.update(R_240=r240, rises_240=rises, rise_frac=rises / STEPS,
               R_last20_mean=float(R[STEPS - 20:STEPS].mean()),
               kill_R=bool(kill_R), kill_rises=bool(kill_rise),
               verdict="KILLED" if (kill_R or kill_rise) else "SURVIVES")
print(json.dumps(out, indent=1))
if n >= STEPS:
    json.dump(out, open(os.path.join(H, "results", "score_premise.json"), "w"), indent=1)
