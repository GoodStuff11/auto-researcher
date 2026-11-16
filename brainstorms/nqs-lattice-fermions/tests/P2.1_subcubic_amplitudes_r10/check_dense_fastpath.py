#!/usr/bin/env python3
"""
Check that DenseHFPSAnsatz's fast path (shared feature vector computed once) equals the generic
HFPSAnsatz patch path with patch = all sites, for amplitudes and per-sample gradients, and that the
summed per-sample gradient equals the full gradient. Toy instances 9 and 8. No options.
Usage: python3 check_dense_fastpath.py > check_dense_fastpath_output.txt
"""
import sys

import numpy as np

import t3_hfps as T


def main():
    mt = T.mt
    ok = True
    for inst in (9, 8):
        m = mt.make_model(inst)
        refs = mt.references(m)
        a = T.DenseHFPSAnsatz(m, refs[1], 3, 1, seed=3)
        rng = np.random.default_rng(1)
        x = a.flat(); a.set_flat(x + 0.05 * rng.standard_normal(len(x)))
        b = T.T2.HFPSAnsatz(m, refs[1], "site", 3, 1, seed=3)
        b.P = a.P.copy(); b.p = {k: v.copy() for k, v in a.p.items()}; b.keys = list(a.keys)
        idx = rng.choice(m.D, size=min(30, m.D), replace=False)
        oc = a._occ(idx)
        coef = rng.standard_normal(len(idx))
        ca, cb = a.forward(*oc), b.forward(*oc)
        ga, gb, fa = a.grad(ca, coef, True), b.grad(cb, coef, True), a.grad(ca, coef, False)
        e1 = np.max(np.abs(ca["psi"] - cb["psi"])) / np.max(np.abs(cb["psi"]))
        e2 = np.max(np.abs(ga - gb)) / np.max(np.abs(gb))
        e3 = np.max(np.abs(ga.sum(0) - fa)) / np.max(np.abs(fa))
        print(f"inst {inst}: fast vs generic amplitude {e1:.1e}; per-sample grad {e2:.1e}; "
              f"summed per-sample vs full {e3:.1e}")
        ok &= max(e1, e2, e3) < 1e-12
    print("FASTPATH CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
