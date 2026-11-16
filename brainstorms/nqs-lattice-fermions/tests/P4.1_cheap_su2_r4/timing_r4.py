"""Time ONE energy + gradient per accuracy-test arm on the production instance (4x3 open, 5+5, U=8,
t'=-0.25), at NNBF width 16, with P_G (and P_S where needed). Usage: python timing_r4.py  (no options)."""
import time
import numpy as np
import r4_test as T
M3 = T.M3
t = time.time()
sec = M3.make_sector(4, 3, 10, 8.0, -0.25, "obc", "obc")
sg = M3.SG(sec)
print(f"sector dim={sec.dim} nC={sec.nC} built in {time.time() - t:.0f}s", flush=True)
for kind in T.RUN_ARMS:
    arm = T.build_arm(sec, kind, 16, 0)
    st = M3.State(sec, arm, T.SPROJ[kind], sg, 16384)
    t = time.time()
    E, g, _ = st.energy_grad(arm.theta0)
    print(f"  {kind:8s} P={arm.nparam:5d} E(init)={E:+.4f} energy+grad {time.time() - t:.1f}s", flush=True)
