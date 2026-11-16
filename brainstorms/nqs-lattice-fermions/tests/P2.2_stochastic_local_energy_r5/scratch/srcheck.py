"""Scratch: SR with exact pi weights (all configurations) vs sampled, H4 K=8."""
import os
import sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import micro_test as m  # noqa: E402

K, h, hj, warm, steps = 8, 16, 8, 40, 60
inst = m.Inst(4, 4.4, 0.25, 1, K)
net = m.make_net(K, 4, 'NNBFJ', h, hj, 0)
th0 = minimize(lambda t: net.energy_grad(t, inst.H, inst.occ, inst.oi), net.theta0, jac=True,
               method='L-BFGS-B', options={'maxiter': warm}).x
for mode in ["exactpi", "B200", "B2000"]:
    for lr in [0.02, 0.1, 0.5]:
        th = th0.copy(); rng = np.random.default_rng(0); es = []
        for t in range(steps):
            psi, _ = net.amplitudes(th, inst.occ, inst.oi); wv = psi ** 2 / (psi @ psi)
            EL = (inst.H @ psi) / psi
            es.append(1e3 * (wv @ EL - inst.E0))
            if mode == "exactpi":
                s = np.arange(inst.D); w = wv
            else:
                B = int(mode[1:]); s = rng.choice(inst.D, B, p=wv); w = np.full(B, 1 / B)
            _, O = m.logderiv(net, th, inst.occ[s], inst.oi[s])
            th = th + m.sr_update(O, w, EL[s], lr, 1e-4, 1.0)
        print(f"{mode} lr={lr}: {es[0]:.3f} -> {es[-1]:.3f} (min {min(es):.3f})", flush=True)
