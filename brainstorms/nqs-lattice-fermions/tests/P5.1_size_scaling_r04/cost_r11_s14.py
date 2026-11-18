#!/usr/bin/env python3
"""
P5.1 round 11: cost of the S14 (7x2) R51.8 ladder (zero compute, read only).
Usage: python3 cost_r11_s14.py           (no options)
Model: in-job CPU-min per step = a + b x unique draws, fitted on the 6x2 in-job runs of job 769376
(results/S6x2_N10_*_s9000/run.json, per-step cpu_step and n_unique, guided arms + exact).  Scaled to 7x2 by
  a7 = a6 x (exact energy CPU-min 7x2 / 6x2, results/info_*.json 'cpu_min_energy'),
  b7 = b6 x (per-row CPU-min 7x2 / 6x2, 'cpu_min_per_row'),
and cross-checked for the exact arm against job 763903's in-job kappa at 7x2 (two exact row passes + two energies,
results/kappa_S7x2_start.json 'cpu_min'), scaled by the 6x2 ratio exact-step / kappa-pass (763902 vs 769376).
Unique-draw counts at 7x2: results/info_S7x2_N12.json 'uniq_guided_born' (probe start).
Prints per-arm CPU-h for 120 steps (expected) and x1.3 (cap), and decision thresholds for the three-size class.
"""
import glob
import json
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
R = lambda p: json.load(open(os.path.join(H, p)))  # noqa: E731
i6, i7 = R("results/info_S6x2_N10.json"), R("results/info_S7x2_N12.json")
k6, k7 = R("results/kappa_S6x2_start.json"), R("results/kappa_S7x2_start.json")
xs, ys = [], []
for d in glob.glob(os.path.join(H, "results/S6x2_N10_*_s9000")):
    if "born" in d:
        continue
    h = R(os.path.relpath(d, H) + "/run.json")["steps"]
    xs.append(np.mean([s["n_unique"] for s in h]))
    ys.append(np.mean([s["cpu_step"] for s in h[1:]]))
b6, a6 = np.polyfit(xs, ys, 1)
print(f"6x2 in-job fit: a = {a6:.3f} CPU-min/step, b = {b6:.3e} per unique ({len(xs)} arms)")
exact6 = [y for x, y in zip(xs, ys) if x > 1e5][0]
a7 = a6 * i7["cpu_min_energy"] / i6["cpu_min_energy"]
b7 = b6 * i7["cpu_min_per_row"] / i6["cpu_min_per_row"]
ex7_fit = a7 + b7 * i7["reps"]
ratio = exact6 / (k6["cpu_min"] / 2)
ex7_k = ratio * k7["cpu_min"] / 2
print(f"7x2: a7 {a7:.2f}, b7 {b7:.2e}; exact step fit {ex7_fit:.1f}, kappa-based {ex7_k:.1f} CPU-min "
      f"(6x2 exact-step/kappa-pass {ratio:.2f})")
tot = {}
for lab, per in (("exact (fit)", ex7_fit), ("exact (kappa)", ex7_k)):
    print(f"  {lab:14s}: {per * 120 / 60:6.1f} CPU-h expected, {per * 120 / 60 * 1.3:6.1f} cap; "
          f"wall {per * 120 / 60 / 24:.1f} days on 1 CPU")
for N, (ug, ub) in sorted(i7["uniq_guided_born"].items(), key=lambda t: int(t[0])):
    for s, u in (("guided", ug), ("born", ub)):
        c = (a7 + b7 * u) * 120 / 60
        tot[(s, int(N))] = c
        print(f"  {s:6s} {int(N):8d}: unique {u:7d}, {a7 + b7 * u:5.2f} CPU-min/step, {c:5.1f} CPU-h, cap {c * 1.3:5.1f}")
ex = (ex7_fit + ex7_k) / 2 * 2  # CPU-h for 120 steps at the midpoint
mn = ex + tot[("guided", 262144)] + tot[("guided", 1048576)]
print(f"minimal decisive ladder (exact + guided 262k + 1M): {mn:.0f} CPU-h expected, cap {mn * 1.3:.0f}")
full = mn + tot[("guided", 65536)] + tot[("born", 262144)] + tot[("born", 1048576)]
print(f"with guided 65k + Born 262k/1M (comparator): {full:.0f} expected, cap {full * 1.3:.0f}")
n10, n12 = 31283.7, 118626.2
gH2 = i7["expH"] / R("results/info_S5x2_N8.json")["expH"]
print(f"thresholds: SUPPORT vs S10 needs N(7x2) >= {gH2:.1f} x {n10:.0f} = {gH2 * n10:.0f}; vs S12' "
      f">= {i7['expH'] / i6['expH']:.2f} x {n12:.0f} = {i7['expH'] / i6['expH'] * n12:.0f}; "
      f"POLY-3 needs <= 2.744 x {n10:.0f} = {2.744 * n10:.0f}; a need 3-4x 6x2's = {3 * n12:.0f}-{4 * n12:.0f}")
