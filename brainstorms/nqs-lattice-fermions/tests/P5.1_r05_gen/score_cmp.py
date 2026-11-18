#!/usr/bin/env python3
"""Score RULE_cmp.md (R51.16) from the arms' run.json files and R51.1's arm B."""
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
E0 = -7.828637164695285  # lowest (+,+) level, results/info_S4x3_N10.json of R51.8
ES = -7.805526273788163  # R51.1's 4x3 start energy
RP = 0.7475
R = lambda E: (ES - E) / (ES - E0)  # noqa: E731
ARMS = {"born16k": ("results/born_N16384_tau0.005", 16384, 0, 13.5e6),
        "born65k": ("results/born_N65536_tau0.005", 65536, 0, 13.5e6),
        "born65k_tau0.01": ("results/born_N65536_tau0.01", 65536, 0, 13.5e6),
        "born262k": ("results/born_N262144_tau0.005", 262144, 0, 13.5e6),
        "guided16k_cont": ("results/guided_N16384_tau0.005_cont", 16384, 120, 6.75e6)}


def first(path, N, off, frac):
    st = json.load(open(os.path.join(H, path, "run.json")))["steps"]
    for i, s in enumerate(st):
        if R(s["E"]) >= frac * RP:
            return (off + i + 1) * N, R(st[-1]["E"]), st and len(st)
    return float("inf"), R(st[-1]["E"]), len(st)


out = {}
for frac in (0.90, 0.80):
    t = {k: first(*v[:3], frac) for k, v in ARMS.items()}
    tb = min(v[0] for k, v in t.items() if k.startswith("born"))
    tb_eff = tb if tb < float("inf") else 13.5e6
    tg = min(6.75e6 if frac == 0.90 else 3.67e6, t["guided16k_cont"][0])
    ratio = tb_eff / tg
    v = "GUIDED-FEWER" if ratio >= 2 else ("NO-GAIN" if ratio <= 1.25 else "MIXED")
    out[frac] = dict(arms={k: dict(T=x[0], R_end=x[1], steps=x[2]) for k, x in t.items()}, T_born=tb, T_guided=tg,
                     ratio_lower_bound=ratio, verdict=v)
    print(frac, json.dumps(out[frac]))
json.dump(out, open(os.path.join(H, "results", "score_cmp.json"), "w"), indent=1)
