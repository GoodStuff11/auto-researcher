#!/usr/bin/env python3
"""Score RULE_B after unit 2 (job 761860): R_plat, Delta, ratios, margins, exponents, equal-total-sample view.
Run: python3 score_unit2.py  (reads results/*/run.json; writes results/score_unit2.json)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RPOL, SD = 0.7475, 0.015
ARMS = {"A": "N16384_tau0.005_s9000", "B": "N65536_tau0.005_s9000", "C": "N262144_tau0.005_s9000",
        "D": "N16384_tau0.0025_s9000"}
d = {k: json.load(open(os.path.join(HERE, "results", v, "run.json"))) for k, v in ARMS.items()}
out = {}
for k, r in d.items():
    R = np.array([s["R"] for s in r["steps"]])
    dR = np.array([s["dR"] for s in r["steps"]])
    out[k] = dict(N=r["nsamp"], tau=r["tau"], steps=len(R), R_last=float(R[-1]), R_plat=float(R[-20:].mean()),
                  slope=float(dR[-20:].mean()), rises=int((dR < 0).sum()), Delta=float(RPOL - R[-20:].mean()),
                  cpu_min=r["cpu_min"], unique_mean=float(np.mean([s["n_unique"] for s in r["steps"]])))
D = {k: out[k]["Delta"] for k in out}
res = dict(arms=out, ratio_BA=D["B"] / D["A"], ratio_CB=D["C"] / D["B"], ratio_CA=D["C"] / D["A"],
           ratio_DA=D["D"] / D["A"],
           a_AB=float(np.log(D["A"] / D["B"]) / np.log(4)), a_BC=float(np.log(D["B"] / D["C"]) / np.log(4)),
           a_AC=float(np.log(D["A"] / D["C"]) / np.log(16)))
lN = np.log([16384, 65536, 262144])
res["a_fit"] = float(-np.polyfit(lN, np.log([D["A"], D["B"], D["C"]]), 1)[0])
# constant + b/N noise-ball model from A and B, predicted for C
b = (D["A"] - D["B"]) / (1 - 0.25)
res["ball_model"] = dict(Delta_prog=D["A"] - b, b=b, Delta_C_pred=D["A"] - b + b / 16, Delta_C_obs=D["C"])
# margins (in R units) of each arm to each threshold, against 2 replica spreads (0.03)
res["margins"] = {
    "NF1_A": (RPOL - D["B"] / 0.62) - out["A"]["R_plat"],
    "NF1_B": out["B"]["R_plat"] - (RPOL - 0.62 * D["A"]),
    "NF2_C": out["C"]["R_plat"] - (RPOL - 0.62 * D["B"]),
    "NF2_B": (RPOL - D["C"] / 0.62) - out["B"]["R_plat"],
    "SN_D": (RPOL - 0.75 * D["A"]) - out["D"]["R_plat"],
    "SN_A": out["A"]["R_plat"] - (RPOL - D["D"] / 0.75),
}
# equal total samples
eq = {}
for tot in [0.98e6, 1.97e6, 3.93e6, 7.86e6, 15.7e6, 31.5e6]:
    row = {}
    for k, r in d.items():
        s = int(round(tot / r["nsamp"]))
        if 0 < s <= len(r["steps"]):
            row[k] = (s, r["steps"][s - 1]["R"])
    eq[f"{tot / 1e6:.2f}M"] = row
res["equal_total_samples"] = eq
first = {}
for thr in (0.9 * RPOL, 0.95 * RPOL):
    for k, r in d.items():
        st = [i + 1 for i, x in enumerate(r["steps"]) if x["R"] >= thr]
        first[f"{thr:.4f}_{k}"] = (st[0], st[0] * r["nsamp"]) if st else None
res["first_reach"] = first
json.dump(res, open(os.path.join(HERE, "results", "score_unit2.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
