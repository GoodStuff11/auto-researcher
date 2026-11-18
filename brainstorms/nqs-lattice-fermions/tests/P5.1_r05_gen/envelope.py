#!/usr/bin/env python3
"""P5.1 round 5, R51.15 `adaptN` T1 from stored data: can switching samples-per-step between R51.1's arms beat the
best single arm in total samples?  Markov approximation: progress rate at level R depends only on R and N.
For each arm, n_N(R) = samples to first reach R (running max of exact R); in bins of R the envelope pays the
cheapest arm's samples for that bin.  Reports total samples to 0.80, 0.90 and 0.95 of R_pol = 0.7475.
Rule: RULE_adaptN.md (written before this ran)."""
import json
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(H, "..", "P5.1_noise_floor_r01", "results")
arms = {"A": ("N16384_tau0.005_s9000", 16384), "B": ("N65536_tau0.005_s9000", 65536),
        "C": ("N262144_tau0.005_s9000", 262144), "D": ("N16384_tau0.0025_s9000", 16384)}
RP = 0.7475
cur = {}
for k, (d, N) in arms.items():
    R = np.array([s["R"] for s in json.load(open(os.path.join(B, d, "run.json")))["steps"]])
    cur[k] = (np.maximum.accumulate(R), N)


def first(k, r):
    R, N = cur[k]
    i = np.nonzero(R >= r)[0]
    return np.inf if len(i) == 0 else (i[0] + 1) * N


out = {}
for frac in (0.80, 0.90, 0.95):
    tgt = frac * RP
    single = {k: first(k, tgt) for k in cur}
    grid = np.linspace(0.05, tgt, 60)
    env, choice = 0.0, []
    for a, b in zip(grid[:-1], grid[1:]):
        costs = {k: first(k, b) - first(k, a) for k in cur if np.isfinite(first(k, b))}
        if not costs:
            env = np.inf
            break
        kk = min(costs, key=costs.get)
        env += costs[kk]
        choice.append(kk)
    best = min(single.values())
    sw = [(round(float(g), 3), c) for g, c in zip(grid[1:], choice)]
    chg = [sw[0]] + [sw[i] for i in range(1, len(sw)) if sw[i][1] != sw[i - 1][1]]
    out[frac] = dict(single={k: float(v) for k, v in single.items()}, envelope=float(env),
                     gain=float(best / env) if env > 0 else None, switches=chg)
    print(frac, json.dumps(out[frac]))
json.dump(out, open(os.path.join(H, "envelope.json"), "w"), indent=1)
