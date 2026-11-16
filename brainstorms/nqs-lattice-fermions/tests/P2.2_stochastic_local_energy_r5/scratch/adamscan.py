"""Scratch: Adam step-size scan for Part L (EXACT estimator).
Usage: python3 adamscan.py K hidden hj warm_iters B steps"""
import os
import sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import micro_test as m  # noqa: E402

K, h, hj, warm, B, steps = map(int, sys.argv[1:7])
inst = m.Inst(4, 4.4, 0.25, 1, K)
net = m.make_net(K, 4, 'NNBFJ', h, hj, 0)
th0 = minimize(lambda t: net.energy_grad(t, inst.H, inst.occ, inst.oi), net.theta0, jac=True,
               method='L-BFGS-B', options={'maxiter': warm}).x
for lr in [3e-4, 1e-3, 3e-3]:
    th = th0.copy(); rng = np.random.default_rng(0); es = []
    mm = np.zeros_like(th); vv = np.zeros_like(th)
    for t in range(1, steps + 1):
        psi, _ = net.amplitudes(th, inst.occ, inst.oi); wv = psi ** 2 / (psi @ psi)
        es.append(1e3 * (wv @ ((inst.H @ psi) / psi) - inst.E0))
        s = rng.choice(inst.D, B, p=wv); e = (inst.H @ psi)[s] / psi[s]
        _, O = m.logderiv(net, th, inst.occ[s], inst.oi[s])
        g = 2 * (O - O.mean(0)).T @ (e - e.mean()) / B
        mm = 0.9 * mm + 0.1 * g; vv = 0.999 * vv + 0.001 * g * g
        th = th - lr * (mm / (1 - 0.9 ** t)) / (np.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
    print(f"adam lr={lr}: dE start {es[0]:.3f} -> mean last 20 {np.mean(es[-20:]):.3f} min {min(es):.3f}", flush=True)
