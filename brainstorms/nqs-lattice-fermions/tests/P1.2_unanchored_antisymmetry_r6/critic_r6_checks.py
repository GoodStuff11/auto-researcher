#!/usr/bin/env python3
"""
P1.2 round 6 per-leaf critic: checks of the worker's claims from stored JSONs (no new physics).
=====================================================================================================
  (1) Bound on the job-553724 final verdict. The final rule (r5 micro_test.aggregate, final=True) needs
      rho_sign = med_5 dEs(frdK) / med_5 dEs(nnbf6) <= 0.7. All ten nnbf6 runs (5 seeds x 2 shifts) are
      on disk; per seed the lower-energy shift is kept, exactly as aggregate() does. frdK has 3 seeds;
      whatever seeds 44 and 55 give, med_5 dEs(frdK) >= min of the three stored values, so
      rho_sign >= min_3 / med_5(nnbf6). If that lower bound exceeds 0.7, the verdict is FAIL now.
  (2) T2-a numbers: v = (dEs_abs - dEs_K) / gap from results/t2_task0.json, and the energy shift.
  (3) Lift versus precision: lift = precision / w_b, so lift <= 1 / w_b. Prints precision = lift * w_b
      for the toy rows (t1_vote.json, U and G) and the trained V = 16 rows (t2_task0.json), to show
      that lift values from different bases and sizes are not a size trend.
Usage: python3 critic_r6_checks.py   (reads ../P1.2_unanchored_antisymmetry_r5/results and results/)
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(os.path.dirname(HERE), "P1.2_unanchored_antisymmetry_r5", "results")


def main():
    sel = {}
    for f in glob.glob(os.path.join(R5, "inst0_*_s*_sh*.json")):
        r = json.load(open(f))
        if r["arm"].endswith("_mc") or not (r.get("budgets") or {}).get("P"):
            continue
        k = (r["arm"], r["seed"])
        if k not in sel or r["budgets"]["P"]["E"] < sel[k]["budgets"]["P"]["E"]:
            sel[k] = r
    c = sorted((k[1], v["budgets"]["P"]["dE_sign"], v["shift"]) for k, v in sel.items() if k[0] == "nnbf6")
    x = sorted((k[1], v["budgets"]["P"]["dE_sign"], v["shift"]) for k, v in sel.items() if k[0] == "frdK")
    print("(1) nnbf6 selected per seed:", [(s, round(d, 5), sh) for s, d, sh in c])
    print("    frdK  selected per seed:", [(s, round(d, 5), sh) for s, d, sh in x])
    medc = float(np.median([d for _, d, _ in c]))
    lb = min(d for _, d, _ in x) / medc
    print(f"    nnbf6 seeds {len(c)}, med_5 dEs = {medc:.5f}; frdK seeds {len(x)}; "
          f"rho_sign lower bound = {lb:.3f} (3-seed frdK median gives {np.median([d for _, d, _ in x]) / medc:.3f})")
    print(f"    553724 final verdict: {'FAIL is certain (bound > 0.7)' if lb > 0.7 else 'still open'}")
    t0 = json.load(open(os.path.join(HERE, "results", "t2_task0.json")))
    gap = t0["dEs_C"] - t0["dEs_K_stored"]
    print(f"(2) T2-a: dEs_K {t0['dEs_K_stored']:.6f} dEs_abs {t0['dEs_abs']:.6f} gap {gap:.5f} "
          f"v = {(t0['dEs_abs'] - t0['dEs_K_stored']) / gap:.4f}; E_abs - E_ident = {t0['E_abs'] - t0['E_ident']:.2e}")
    tv = json.load(open(os.path.join(HERE, "results", "t1_vote.json")))
    rows = tv["results"]
    print("(3) precision = lift * w_b (upper bound of lift is 1/w_b)")
    for r in (rows if isinstance(rows, list) else []):
        for lab in ("U", "G"):
            d = r.get(lab)
            if not d:
                continue
            wb = d["K"]["w_b"] if "K" in d else d.get("w_b")
            print(f"    toy V={r['V']} {lab}: w_b {wb:.4f} 1/w_b {1 / wb:.1f} | " + " ".join(
                f"{F}: lift {d[F]['lift']:.2f} prec {d[F]['lift'] * wb:.3f}" for F in ("K", "X", "H")))
    for lab in ("U", "G"):
        wb = t0[f"w_b_{lab}"]
        print(f"    trained V=16 {lab}: w_b {wb:.4f} 1/w_b {1 / wb:.1f} | " + " ".join(
            f"{F}: lift {t0[f'lift_{F}_{lab}']:.2f} prec {t0[f'lift_{F}_{lab}'] * wb:.3f} "
            f"recall {t0[f'recall_{F}_{lab}']:.3f}" for F in ("K", "X", "H")))


if __name__ == "__main__":
    main()
