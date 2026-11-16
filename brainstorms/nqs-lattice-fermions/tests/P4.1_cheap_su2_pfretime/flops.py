#!/usr/bin/env python3
"""
Flop model of one amplitude (no P_G; P_G multiplies both arms by 4) for `che` (H = 16) and `pcg` (matched H) on open
Lx x 3 strips, N_e = vmc.default_ne(L). No arguments: python3 flops.py  (output saved to logs/flops_output.txt).
che: MLP 2(2L)H + 2H(L N_e); solve + product + slogdet of N_up blocks; n_q slogdets of N_up x N_up.
pcg: MLP 2(2L)H + 2H(2 n_mask); f_c = B1 f0 B2^T dense 4L^3; X(u) assembly 3 n_q N_e^2;
     Pfaffian (full-block Parlett-Reid, as pf_fast.py) n_q (2/3) N_e^3  [triangular update: n_q (1/3) N_e^3].
"""
import math
ne = lambda L: int(2 * round(5 * L / 12))
for Lx in (4, 8, 12, 16, 32, 64, 256):
    L = 3 * Lx; Ne = ne(L); Nu = Ne // 2; nq = math.ceil((Nu + 1) / 2); H = 16; Hp = {4: 15, 8: 25, 12: 35}.get(Lx, 46)
    nmask = sum(1 for i in range(L) for j in range(L) if abs(i // 3 - j // 3) <= 1 and abs(i % 3 - j % 3) <= 1)
    che = 4 * L * H + 2 * H * L * Ne + (2 / 3 + 2 + 2 + 2 / 3) * Nu ** 3 + nq * (2 / 3 * Nu ** 3 + 2 * Nu ** 2)
    pf = nq * 2 / 3 * Ne ** 3
    pcg = 4 * L * Hp + 4 * Hp * nmask + 4 * L ** 3 + 3 * nq * Ne ** 2 + pf
    pcg_tri = pcg - pf / 2
    print(f"L={L:4d} Ne={Ne:4d} n_q={nq:4d}: che {che:.3g}  pcg {pcg:.3g} (Pf {pf:.3g}, f_c {4 * L ** 3:.3g})  "
          f"ratio {pcg / che:.2f}  (triangular Pf: {pcg_tri / che:.2f})")
