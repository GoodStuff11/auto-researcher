#!/usr/bin/env python3
"""
P5.1 round 11: sensitivity of the 6x2 SUPPORT-LIKE verdict to a second seed (zero compute, read only).
Usage: python3 sens_r11_seed2.py            (no options; reads results/score_r07.json and score_r08.json)
Seed-to-seed sd of s is estimated from the three S10 rungs that have two seeds (16k, 65k, 262k), pooled, under
two scalings (relative: sd proportional to s; absolute: same sd in s).  The verdict margin is ln N_needed(6x2) -
ln 110,100.  The seed MEAN of two seeds moves by sd/sqrt(2).  Tail probabilities use Student t with 3 dof (three
pooled one-pair estimates) and the normal.  Prints the flip probability SUPPORT-LIKE -> AMBIGUOUS.
"""
import json
import math
import os

from scipy import stats

H = os.path.dirname(os.path.abspath(__file__))
r7 = json.load(open(os.path.join(H, "results/score_r07.json")))
r8 = json.load(open(os.path.join(H, "results/score_r08.json")))
sL, sU = r8["rev2"]["s_lower"], r8["rev2"]["s_upper"]
NL = r8["rev2"]["bracket"][0]
thr = 3.52 * r8["N_needed_5x2"]
L = math.log


def lnN(a, b):
    return L(NL) + L(4) * (a - L(0.10)) / (a - b)


a, b = L(sL), L(sU)
base = lnN(a, b)
margin = base - L(thr)
dA = (lnN(a + 1e-7, b) - base) / 1e-7
dB = (lnN(a, b + 1e-7) - base) / 1e-7
print(f"N_needed(6x2) {math.exp(base):.0f}; SUPPORT threshold {thr:.0f}; margin {margin:.4f} ln-units")
print(f"dlnN/dln s(65k) {dA:.3f}; dlnN/dln s(262k) {dB:.3f}")
rel, ab = [], []
for row in r7["rows"]["guided"]:
    if row.get("seed_diff") is not None:
        rel.append(row["seed_diff"] / row["s"] / math.sqrt(2))
        ab.append(row["seed_diff"] / math.sqrt(2))
        print(f"S10 N {row['N']}: seed diff {row['seed_diff']:.4f}, s {row['s']:.4f}")
srel = math.sqrt(sum(x * x for x in rel) / len(rel))
sab = math.sqrt(sum(x * x for x in ab) / len(ab))
print(f"pooled single-seed sd: relative {srel:.4f} (ln s), absolute {sab:.4f} (s)")
for name, sl, su in (("relative", srel, srel), ("absolute", sab / sL, sab / sU)):
    for k in (1.0, 2.0):
        one = abs(dA) * k * sl / math.sqrt(2)
        both = math.hypot(one, dB * k * su / math.sqrt(2))
        for lab, sd in (("65k only", one), ("65k+262k", both)):
            z = margin / sd
            print(f"{name:8s} noise x{k:.0f} {lab:9s}: sd(lnN) {sd:.4f} z {z:.2f} "
                  f"P(flip) t3 {stats.t.sf(z, 3):.3f} normal {stats.norm.sf(z):.3f}")
