#!/usr/bin/env python3
"""
P3.1 Round 12: verdict of Route 12.3u (unconditional block bracket) from the saved t1_r12.py JSONs.
==================================================================================================

No computation beyond reading results/r12_<size>[_d0.02].json (written by t1_r12.py --mode main).
For every scored item of each of the six instances (4x2, 5x2, 4x3; native and gap surgery 0.02 t) it scores the
block interval of the witness pass, for the OBS chain (deciding) and the DAV chain (W2 foil):
  covered      a_0 in [lo - 0.1|delta|, hi + 0.1|delta|]
  width ratio  (hi - lo) / w_ex, w_ex = eigenvalue range of A on the exact plane span{psi_0, psi_1}
Rule (leaf, "Route 12.3u", written before this script was run): KILL if coverage < 0.5 on any instance or the
median width ratio > 3 on any instance; PASS if coverage >= 0.8 and median width ratio <= 1.5 on every instance;
UNCLEAR otherwise.

Usage
  python3 t1_r12_cbru.py          # no options
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("r12", os.path.join(HERE, "t1_r12.py"))
r12 = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = [sys.argv[0]]
_spec.loader.exec_module(r12)
sys.argv = _argv


def score(rows, arm):
    it = r12.items(rows)
    cov = sum(1 for r, k in it if r12.covered(r, k, arm)) / len(it)
    wr = [(r[f"br_{arm}"][k][1] - r[f"br_{arm}"][k][0]) / r["wex"][k] for r, k in it]
    wd = [(r[f"br_{arm}"][k][1] - r[f"br_{arm}"][k][0]) / abs(r["delta"][k]) for r, k in it]
    return cov, float(np.median(wr)), float(np.median(wd)), len(it)


def main():
    res = {}
    for arm in ("OBS", "DAV"):
        for s in r12.SIZES:
            for tag in (s, f"{s}_d{r12.STRESS}"):
                rows = r12.load(tag)
                res[(arm, tag)] = c = score(rows, arm)
                print(f"  {arm} {tag:10s} n={c[3]:2d} coverage {c[0]:.2f} median width/w_ex {c[1]:.2f} "
                      f"median width/|delta| {c[2]:.2f}")

    def v(arm):
        X = [x for (a, _), x in res.items() if a == arm]
        if any(x[0] < 0.5 or x[1] > 3 for x in X):
            return "KILL"
        if all(x[0] >= 0.8 and x[1] <= 1.5 for x in X):
            return "PASS"
        return "UNCLEAR"
    vo, vd = v("OBS"), v("DAV")
    print(f"VERDICT 12.3u: {vo}  (DAV block under the same rule: {vd}; W2 label: "
          f"{'adds nothing' if vd == 'PASS' else 'observable sourcing adds'})")


if __name__ == "__main__":
    main()
