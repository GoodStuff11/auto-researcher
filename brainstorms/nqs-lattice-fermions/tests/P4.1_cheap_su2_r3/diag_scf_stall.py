#!/usr/bin/env python3
"""Diagnostic for the round-2 critic objection "SCF stalls at eps = 2.4e-2 on the 400-state
2x3 cluster": is the stall an iteration-budget artefact, a bond-dimension limit, or a limit
of the linear charge conditioning W(c) = W0 + U h(c)?  Uses the round-2 SCF class verbatim
(copied file r2_micro_test.py in this folder; nothing imported across folders).

Usage: python diag_scf_stall.py      (no options; ~5-10 min on one core)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_micro_test as M  # noqa: E402


def main():
    sec = M.make_sector(2, 3, 6, 6.0, -0.3)
    E0 = M.exact_states(sec, 1, k=1)[0][1]
    print(f"2x3 Ne=6 U=6 t'=-0.3 dim={sec.dim} E0={E0:.8f}", flush=True)
    for mult, hid, it in [((2, 2, 1, 1), 8, 150), ((2, 2, 1, 1), 8, 3000),
                          ((4, 4, 2, 2), 8, 3000), ((2, 2, 1, 1), 32, 3000)]:
        arm = M.SCF(sec, hid, 0, mult)
        st = M.State(sec, arm, False, 1, 16384)
        t = time.time()
        th, E, n, _ = M.optimise(st, arm.theta0, it, 0.3, logf=lambda s: None)
        print(f"mult={mult} hidden={hid} maxiter={it} P={arm.nparam} D={arm.D} "
              f"eps={(E - E0) / abs(E0):.2e} iters={n} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
