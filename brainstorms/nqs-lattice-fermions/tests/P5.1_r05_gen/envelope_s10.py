#!/usr/bin/env python3
"""Second-size check for R51.15 (diagnostic, after RULE_adaptN.md was scored at 4x3): the same switching envelope
and best monotone schedule on R51.8's finished S10 guided arms (1k..256k), targets = 0.80/0.90/0.95 of the S10 exact
arm's R at step 120 (R measured against E0_pg).  Does not compute any R51.8 statistic."""
import json
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(H, "..", "P5.1_size_scaling_r04", "results")
E0 = -6.550697008720983
cur = {}
for N in (1024, 4096, 16384, 65536, 262144):
    d = json.load(open(os.path.join(B, f"S5x2_N8_guided_N{N}_tau0.005_s9000", "run.json")))
    R = np.array([(d["E_start"] - s["E"]) / (d["E_start"] - E0) for s in d["steps"]])
    cur[N] = np.maximum.accumulate(R)
dx = json.load(open(os.path.join(B, "S5x2_N8_exact_tau0.005_s9000", "run.json")))
RX = (dx["E_start"] - dx["steps"][-1]["E"]) / (dx["E_start"] - E0)


def first(N, r):
    i = np.nonzero(cur[N] >= r)[0]
    return np.inf if len(i) == 0 else (i[0] + 1) * N


out = dict(R_exact120=RX)
Ns = sorted(cur)
for frac in (0.80, 0.90, 0.95):
    tgt = frac * RX
    single = {N: first(N, tgt) for N in Ns}
    g = np.round(np.arange(0.02, tgt, 0.01), 3)
    best = (np.inf, None)
    for i, r1 in enumerate(g):  # monotone: N_a up to r1, N_b up to tgt (one switch, any pair a < b)
        for ia, na in enumerate(Ns):
            for nb in Ns[ia + 1:]:
                c = first(na, r1) + first(nb, tgt) - first(nb, r1)
                if c < best[0]:
                    best = (c, (float(r1), na, nb))
    bs = min(single.values())
    out[frac] = dict(single={str(k): float(v) for k, v in single.items()}, mono=float(best[0]), sched=best[1],
                     gain=float(bs / best[0]) if np.isfinite(bs) else None)
    print(frac, out[frac])
json.dump(out, open(os.path.join(H, "envelope_s10.json"), "w"), indent=1)
