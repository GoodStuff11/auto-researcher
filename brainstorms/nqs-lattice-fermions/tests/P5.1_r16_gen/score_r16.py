#!/usr/bin/env python3
"""Round-16 scorer for RULE_snrcut.md (R51.26) and RULE_minsr.md (R51.31), 5x2.  Read only on R51.8's 5x2 arms
(../P5.1_size_scaling_r04/results, seed 9000) and on results/ here; writes results/score_r16.json."""
import json
import math
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
NSR = os.path.join(os.path.dirname(H), "P5.1_size_scaling_r04", "results")
ex = np.array([s["E"] for s in json.load(open(os.path.join(NSR, "S5x2_N8_exact_tau0.005_s9000", "run.json")))["steps"]])
Es = json.load(open(os.path.join(NSR, "S5x2_N8_exact_tau0.005_s9000", "run.json")))["E_start"]
Ewx = ex[60:120].mean()
G = Es - Ewx
assert abs(G - 0.001601132808884742) < 1e-12


def E_of(path):
    d = json.load(open(path))
    return np.array([s["E"] for s in d["steps"]]), d


def s_win(E):
    return (E[60:120].mean() - Ewx) / G


plain = []
for N in (1024, 4096, 16384, 65536, 262144):
    E, _ = E_of(os.path.join(NSR, f"S5x2_N8_guided_N{N}_tau0.005_s9000", "run.json"))
    plain.append((N, s_win(E)))


def n_eq(s):
    for (na, sa), (nb, sb) in zip(plain, plain[1:]):
        if sa >= s >= sb:
            return math.exp(math.log(na) + (math.log(s) - math.log(sa)) * (math.log(nb) - math.log(na))
                            / (math.log(sb) - math.log(sa)))
    return float("inf") if s < plain[-1][1] else 0.0


out = dict(G_ref=G, plain=plain, snr={})
for N, rho in ((4096, 1.0), (16384, 1.0), (4096, 2.0)):
    p = os.path.join(H, "results", f"snr_guided_N{N}_rho{rho}_tau0.005_s9000", "run.json")
    E, d = E_of(p)
    assert d["status"] == "DONE" and len(E) == 120, p
    s = s_win(E)
    ne = n_eq(s)
    out["snr"][f"N{N}_rho{rho}"] = dict(s=s, N_eq=ne, g=ne / N, kept_median=float(np.median([x["kept"] for x in d["steps"]])),
                                         R_last20=(Es - E[100:120].mean()) / G)
g = [out["snr"]["N4096_rho1.0"]["g"], out["snr"]["N16384_rho1.0"]["g"]]
out["snr_verdict"] = "PASS" if min(g) >= 2 else ("KILL" if max(g) <= 1.25 else "MARGINAL")
p = os.path.join(H, "results", "snrW_guided_N4096_rho1.0_tau0.005_s9000", "run.json")
if os.path.exists(p):  # RULE_snrcut.md addendum (Wiener repair)
    E, d = E_of(p)
    assert d["status"] == "DONE" and len(E) == 120, p
    s = s_win(E)
    gw = n_eq(s) / 4096
    out["wiener_N4096"] = dict(s=s, g=gw, R_last20=(Es - E[100:120].mean()) / G,
                               verdict="PASS" if gw >= 2 else ("KILL" if gw <= 1.25 else "MARGINAL"))

p = os.path.join(H, "results", "minsr_born_N1024_tau0.00125_s9000", "run.json")
if os.path.exists(p):
    E, d = E_of(p)
    R = (Es - E) / G
    m = np.convolve(R, np.ones(20) / 20, mode="valid")
    cross = np.nonzero(m >= 0.9)[0]
    Rm = float(m[-1])
    ref = {}
    for samp in ("guided", "born"):
        Eg, _ = E_of(os.path.join(NSR, f"S5x2_N8_{samp}_N16384_tau0.005_s9000", "run.json"))
        ref[samp] = float(((Es - Eg) / G)[30:50].mean())
    diff = ref["guided"] - Rm
    out["minsr"] = dict(steps=len(E), draws=len(E) * 1024, status=d["status"], R20_final=Rm, ref_R20_at_819k=ref,
                        first_sustained90_draws=None if len(cross) == 0 else int((cross[0] + 20) * 1024),
                        verdict="GUIDED-AHEAD" if diff >= 0.10 else ("MINSR-AHEAD" if diff <= -0.10 else "TIE"))
json.dump(out, open(os.path.join(H, "results", "score_r16.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
