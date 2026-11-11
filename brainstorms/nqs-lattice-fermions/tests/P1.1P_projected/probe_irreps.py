#!/usr/bin/env python3
"""
Probe for P1.1-P: psi_0's space-group irrep, total spin, and projector costs on the
pre-committed instances.

Usage
-----
    python3 probe_irreps.py [--instances LIST]

Options
-------
  --instances LIST  Comma-separated instance indices from micro_test.INSTANCES
                    (0, 1, 2 or 3; default "1,0,2").
"""
import argparse
import sys
import time

import numpy as np

import micro_test as M
import p11_core as P


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="psi_0 irrep / spin probe (see docstring)")
    ap.add_argument("--instances", type=str, default="1,0,2")
    a = ap.parse_args(argv)
    a.inst = [int(k) for k in a.instances.split(",") if k.strip()]
    return a


def main(argv=None):
    a = parse_args(argv)
    for K in a.inst:
        t0 = time.time()
        m = M.make_model(K)
        W, Vv, deg = P.solve_ed(m, 3, 1e-8, 1234)
        t1 = time.time()
        sg = M.SpaceGroup(m)
        t2 = time.time()
        lam = sg.irrep_of(Vv[:, 0])
        spn = M.Spin(m)
        t3 = time.time()
        sg.set_irrep(lam)
        v = np.random.default_rng(0).standard_normal(m.D)
        t4 = time.time(); sg.project(v); t5 = time.time()
        spn.project(v, 0); t6 = time.time()
        print(f"instance {K} D={m.D} E={W.tolist()} deg={deg} irrep={lam} "
              f"<S2>(psi0)={spn.expect(Vv[:, 0]):.3e} Smax={spn.Smax}  ED {t1 - t0:.0f}s "
              f"group {t2 - t1:.1f}s spin {t3 - t2:.1f}s P_chi {t5 - t4:.2f}s P_S {t6 - t5:.2f}s",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
