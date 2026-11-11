"""Time one projected SR step on the real instance for widths 2, 4, 8 (cost estimate for
submit_test.sbatch). Usage: python3 time_step.py"""
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import micro_test as MT  # noqa: E402

inst = MT.Instance(1.11394, 8.0)
tp = inst.tpc + 0.002
inst.set_T(tp)
H = inst.H(tp)
args = SimpleNamespace(sr_iters=3, sr_lr=0.05, sr_shift=1e-3, sr_samples=2000, chunk=65536,
                       o_chunk=256)
for w in (2, 4, 8):
    Pu, Pd = MT.uhf_pattern(inst.m, "neel", 1234)
    a = MT.P.Ansatz(inst.m, Pu, Pd, "nnbf", width=w, seed=1234 + w)
    rng = np.random.default_rng(0)
    MT.PC.sr_step_proj(a, H, inst.sg, args, rng)
    t0 = time.time()
    for _ in range(3):
        MT.PC.sr_step_proj(a, H, inst.sg, args, rng)
    print(f"width {w} n_params {a.n_params()}: {(time.time() - t0) / 3:.2f} s per projected SR step",
          flush=True)
