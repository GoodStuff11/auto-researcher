#!/usr/bin/env python3
"""ED reflection characters of the 4x3 open, 5+5, U=8, t'=-0.25 ground state (lowest singlet of H + 5 S^2,
irrep read from the generators Mx, My by r3_micro_test.ground_state). Saves psi0 to results/psi0_S4x3.npy.
Usage: python3 ed_irrep_4x3.py   (no options)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import r3_micro_test as M3
sec = M3.make_sector(4, 3, 10, 8.0, -0.25, "obc", "obc")
sg = M3.SG(sec)
P0, E0, s2, lam, nxt = M3.ground_state(sec, sg)
print(f"E0 = {E0:.12f}  <S^2> = {s2:.1e}  characters {lam}  next level (H + 5 S^2) {nxt[1] + 5 * nxt[2]:.6f}")
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "psi0_S4x3.npy"), P0)
