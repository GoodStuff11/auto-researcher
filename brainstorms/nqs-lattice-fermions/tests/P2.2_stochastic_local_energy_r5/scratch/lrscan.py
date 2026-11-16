"""Scratch: SR hyperparameter scan for Part L (EXACT estimator).
Usage: python3 lrscan.py K hidden hj warm_iters B steps"""
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
for lr in [0.5, 2.0, 5.0]:
    for sh in [1e-4, 1e-3]:
        for md in [0.05, 0.2]:
            th = th0.copy(); rng = np.random.default_rng(0); es = []
            for t in range(steps):
                psi, _ = net.amplitudes(th, inst.occ, inst.oi); wv = psi ** 2 / (psi @ psi)
                es.append(1e3 * (wv @ ((inst.H @ psi) / psi) - inst.E0))
                s = rng.choice(inst.D, B, p=wv); e = (inst.H @ psi)[s] / psi[s]
                _, O = m.logderiv(net, th, inst.occ[s], inst.oi[s])
                th = th + m.sr_update(O, np.full(B, 1 / B), e, lr, sh, md)
            print(f"lr={lr} shift={sh} maxd={md}: dE start {es[0]:.3f} -> mean last 10 {np.mean(es[-10:]):.3f} "
                  f"min {min(es):.3f}", flush=True)
