#!/usr/bin/env python3
"""Login-node timing of the expensive operations of micro_test.py on one production instance.

Usage: python3 time_ops.py [--instance K]   (K in 0, 1, 2; default 0, the D = 627264 cluster)
Prints: model/group/spin build times, one A and one H application, one P_S P_chi projection,
one projected SR step (P and PS arms, width 8), and one response-network Adam step plus its
final evaluation. Output is recorded in timing_output.txt and used for the sbatch estimate.
"""
import argparse
import sys
import time

import numpy as np

import micro_test as T
import p11_core as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=int, default=0, choices=[0, 1, 2])
    k = ap.parse_args().instance
    t = time.time(); m = T.make_model(k); m.pair_setup(1.5); A = T.PairOp(m, 1.5)
    print(f"instance {k} D={m.D}: model {time.time() - t:.1f}s")
    t = time.time(); sg = T.SpaceGroup(m); print(f"space group {time.time() - t:.1f}s")
    t = time.time(); spn = T.Spin(m); print(f"spin {time.time() - t:.1f}s")
    v = np.random.default_rng(0).standard_normal(m.D)
    t = time.time(); A.apply(v); print(f"A apply {time.time() - t:.3f}s")
    t = time.time(); m.H @ v; print(f"H apply {time.time() - t:.3f}s")
    t = time.time(); spn.project(sg.project(v), 0); print(f"P_S P_chi {time.time() - t:.3f}s")
    Pu, Pd = P.uhf_orbitals(m, 1)
    ans = P.Ansatz(m, Pu, Pd, "nnbf", width=8, seed=1)
    args = argparse.Namespace(sr_iters=1, chunk=65536, o_chunk=256, sr_samples=2000, sr_shift=1e-3,
                              sr_lr=0.05, zr_width=8, zr_steps=1, zr_lr=0.02)
    for mode in ("P", "PS"):
        pr = T.Projector(sg, spn, 0, mode)
        t = time.time(); T.optimize_sr_proj(ans, m.H, sg, pr, args, np.random.default_rng(0), "t")
        print(f"SR step {mode}: {(time.time() - t) / 2:.2f}s (one step + final energy, halved)")
    psi = sg.project(ans.amplitudes(65536)); psi /= np.linalg.norm(psi)
    E = psi @ (m.H @ psi); w = A.apply(psi); w -= (psi @ w) * psi
    Kt = lambda q: (lambda z: m.H @ z - E * z - (psi @ (m.H @ z - E * z)) * psi)(q - (psi @ q) * psi)
    zr = T.ResponseNet(m, ans, T.Projector(sg, spn, 0, "P"), args, 3)
    t = time.time(); zr.train(psi, w, Kt); print(f"ZR one Adam step + final evaluation {time.time() - t:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
