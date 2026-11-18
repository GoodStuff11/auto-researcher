#!/usr/bin/env python3
"""Score RULE_qopt_T2.md revision 1 (R51.11): sustained first passage (score_common.py); needs submit_cmp task 0."""
import json
import os

import score_common as C

assert os.path.exists(os.path.join(C.H, "results", "guided_N16384_tau0.005_cont", "run.json")), "cmp task 0 first"
ARMS = {"q1_16k": ("results/q1_N16384_tau0.005", 16384), "q1_65k": ("results/q1_N65536_tau0.005", 65536)}
out = {}
for frac in (0.90, 0.80):
    rs = frac * C.RP
    tg, tb_guided = C.guided_T(rs)
    arms = {}
    for k, (p, n) in ARMS.items():
        R = C.rseries(p)
        arms[k] = dict(T=C.T(R, n, rs), steps=len(R), R_end=float(R[-1]))
    tq = min(a["T"] for a in arms.values())
    r = tq / tg
    v = "Q-FEWER" if r <= 0.5 else ("NO-GAIN" if r >= 0.8 else "MIXED")
    out[frac] = dict(arms=arms, T_B=tb_guided, T_guided=tg, T_q=tq, ratio=r, verdict=v)
    print(frac, json.dumps(out[frac]))
json.dump(out, open(os.path.join(C.H, "results", "score_qopt.json"), "w"), indent=1)
