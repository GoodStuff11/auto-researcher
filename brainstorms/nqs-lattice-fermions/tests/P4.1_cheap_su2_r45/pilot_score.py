#!/usr/bin/env python3
"""Score the R45 from-init pilot (RULE_pilot.md): power-law extrapolation of best-so-far 1-F to n = 2000."""
import json
import os
import sys

import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_pilot")
TARGET, NMAX, NCAP = 3e-4, 2000, 6000
out, ok = {}, True
for arm in ("che", "nnbf"):
    st = json.load(open(os.path.join(RES, f"state_{arm}_s0.json")))
    inf = np.minimum.accumulate(np.array([h["infid"] for h in st["hist"]]))
    n = np.arange(1, len(inf) + 1)
    sel = (n >= 50) & (n <= 150)
    if len(inf) < 150:
        out[arm] = dict(invalid="fewer than 150 evaluations", n=len(inf))
        ok = False
        continue
    slope, icpt = np.polyfit(np.log(n[sel]), np.log(inf[sel]), 1)
    b = -slope
    pred = float(np.exp(icpt) * NMAX ** slope)
    nstar = float((TARGET / np.exp(icpt)) ** (1 / slope)) if b > 0 else float("inf")
    out[arm] = dict(b=float(b), infid_150=float(inf[149]), pred_2000=pred, n_star=nstar,
                    cpu_min=st["cpu_min_train"], invalid=None if b > 0 else "not decreasing")
    ok = ok and b > 0
if not ok:
    v = "INVALID"
elif all(out[a]["pred_2000"] <= TARGET for a in out):
    v = "PASS"
else:
    ns = max(out[a]["n_star"] for a in out)
    v = f"FAIL (raise maxfun to {int(np.ceil(ns))})" if ns <= NCAP else "FAIL (warm-start or withdraw)"
out["verdict"] = v
json.dump(out, open(os.path.join(RES, "pilot_verdict.json"), "w"), indent=1)
print("PILOT", json.dumps(out))
