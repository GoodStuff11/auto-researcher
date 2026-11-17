"""Timing and correctness probe for exgrad.energy_grad (1 thread). Usage: probe_timing.py ARM"""
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exgrad as X  # noqa: E402

arm = sys.argv[1]
ctx = X.setup(arm)
th = ctx.th0
t = time.process_time()
E, _ = X.energy_grad(ctx, th, need_grad=False)
te = time.process_time() - t
t = time.process_time()
E2, G = X.energy_grad(ctx, th)
tg = time.process_time() - t
rng = np.random.default_rng(0)
v = rng.standard_normal(G.size)
v /= np.linalg.norm(v)
h = 1e-5
Ep, _ = X.energy_grad(ctx, th + h * torch.tensor(v), need_grad=False)
Em, _ = X.energy_grad(ctx, th - h * torch.tensor(v), need_grad=False)
fd = (Ep - Em) / (2 * h)
print(f"{arm}: P {G.size} E {E:.10f} (ctx.energy {ctx.energy(th)[0]:.10f}) |g| {np.linalg.norm(G):.3e} "
      f"dir-deriv {G @ v:.6e} fd {fd:.6e}; t_E {te / 60:.3f} t_EG {tg / 60:.3f} CPU-min")
