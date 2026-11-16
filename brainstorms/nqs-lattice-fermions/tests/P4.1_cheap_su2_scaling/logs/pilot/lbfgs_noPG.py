# diagnostic: exact-sum L-BFGS of `che` on 4x3 open WITHOUT space-group projection (first 100 iterations)
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy.optimize import minimize
import r3_micro_test as M3, fix_test as F
sec = M3.make_sector(4, 3, 10, 8.0, -0.25, "obc", "obc")
arm = F.build_arm_fix(sec, "che", 16, 0)
st = M3.State(sec, arm, True, None, 16384)
it = [0]; t0 = time.time()
def fg(th):
    E, g, _ = st.energy_grad(th); return E, g
def cb(th):
    it[0] += 1
    if it[0] % 10 == 0:
        print(f"it {it[0]} E {st.energy_grad(th)[0]:.6f} ({time.time()-t0:.0f}s)", flush=True)
minimize(fg, arm.theta0, jac=True, method="L-BFGS-B", callback=cb, options=dict(maxiter=100, maxcor=50))
