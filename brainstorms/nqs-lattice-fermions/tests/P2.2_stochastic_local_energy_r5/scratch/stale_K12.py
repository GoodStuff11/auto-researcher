"""Scratch pre-check: ALE0/ALE1 variance increment (over Var_pi(E_L)) vs anchor
age along an EXACT-SR trajectory, H4 R=4.4 K=12, NNBF-J h=32, warm 100."""
import os
import sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import micro_test as m  # noqa: E402

inst = m.Inst(4, 4.4, 0.25, 1, 12)
net = m.make_net(12, 4, 'NNBFJ', 32, 16, 0)
thw = minimize(lambda t: net.energy_grad(t, inst.H, inst.occ, inst.oi), net.theta0, jac=True,
               method='L-BFGS-B', options={'maxiter': 100}).x
for Kd in (4, 32):
    print(f"Kd={Kd}: age |dtheta| inc_ALE0/Var_pi inc_ALE1/Var_pi", flush=True)
    for t, nd, r0, r1 in m.staleness_precheck(inst, net, thw, B=100, Kd=Kd, steps=16, lr=0.5):
        if t in (1, 2, 4, 8, 16):
            print(f"   {t:2d} {nd:.3f} {r0:.3e} {r1:.3e}", flush=True)
