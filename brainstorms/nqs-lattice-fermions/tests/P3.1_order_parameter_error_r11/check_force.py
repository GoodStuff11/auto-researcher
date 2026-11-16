#!/usr/bin/env python3
"""Consistency check of sampled_force.sampled_force against the exact-sum T3 force on 4x2.
At large Ns the sampled force must converge to the exact one (cosine -> 1, norm ratio -> 1) and the sampled R to
the exact R. Prints both at Ns = 2000 and 50000 (3 draws each). Usage: python3 check_force.py"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv += ["--threads", "1"]
spec = importlib.util.spec_from_file_location("sf", os.path.join(HERE, "sampled_force.py"))
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)
np, t3, unit = sf.np, sf.t3, sf.unit


class A:
    size, width, gs_iters, ns = "4x2", 2, 40, 2000
    sr_lr, sr_shift, sr_samples, o_chunk, chunk = 0.05, 1e-3, 2000, 8, 32768


a = A()
C, ans_gs = sf.setup(a)
psi = unit(C.psi0 + 0.2 * C.U[:, 0] + 0.1 * C.U[:, 2])
s = C.shift(psi)
for k in sf.REAL:
    b = C.source(k, psi)
    ans = t3.perturbed_copy(C, ans_gs, a.width, 5, 0.05)
    R, yP, _, g = t3.resp_eval(ans, C, psi, b, s, a)
    fe = ans.force(g, a.chunk)
    Hphi = C.m.H @ psi
    for ns in (2000, 50000):
        for d in range(3):
            rng = np.random.default_rng(d)
            pc = yP ** 2
            idx = rng.choice(len(pc), size=ns, p=pc / pc.sum())
            jdx = rng.choice(len(psi), size=ns, p=psi ** 2)
            O = ans.per_sample_O_proj(C.sg, idx, yP, a.o_chunk)
            Rs, fs = sf.sampled_force(C, ans, yP, psi, b, s, Hphi, float(psi @ Hphi), idx, jdx, O)
            print(f"{k:6s} Ns {ns:6d} draw {d}: R exact {R:.4f} sampled {Rs:.4f} | cos(f_s, f_e) "
                  f"{fs @ fe / np.linalg.norm(fs) / np.linalg.norm(fe):+.3f} |f_s|/|f_e| "
                  f"{np.linalg.norm(fs) / np.linalg.norm(fe):.3f}", flush=True)
