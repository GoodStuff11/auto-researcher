#!/usr/bin/env python3
"""Score RULE_cmp.md revision 1 (R51.16): sustained first passage (score_common.py)."""
import json
import os

import score_common as C

assert os.path.exists(os.path.join(C.H, "results", "guided_N16384_tau0.005_cont", "run.json")), "task 0 first"
BUDGET = 14.8e6
ARMS = {"born16k": ("results/born_N16384_tau0.005", 16384), "born65k_tau0.01": ("results/born_N65536_tau0.01", 65536)}
out = {}
for frac in (0.90, 0.80):
    rs = frac * C.RP
    tg, tb_guided = C.guided_T(rs)
    arms = {}
    for k, (p, n) in ARMS.items():
        R = C.rseries(p)
        arms[k] = dict(T=C.T(R, n, rs), steps=len(R), R_end=float(R[-1]))
    tb = min(a["T"] for a in arms.values())
    ratio = min(tb, BUDGET) / tg
    v = "GUIDED-FEWER" if ratio >= 2 else ("NO-GAIN" if ratio <= 1.25 else "MIXED")
    out[frac] = dict(arms=arms, T_B=tb_guided, T_guided=tg, T_born=tb, ratio_lower_bound=ratio, verdict=v)
    print(frac, json.dumps(out[frac]))
json.dump(out, open(os.path.join(C.H, "results", "score_cmp.json"), "w"), indent=1)
