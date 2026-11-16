#!/usr/bin/env python3
"""Timing probe for the P2.2 closed-leaf falsifier (cold-start ALE). No options; prints the wall
time of a few cold-start SR steps per arm on H4 R=4.4 K=12 (NNBFJ hidden 32, hJ 16), 1 thread."""
import os
import sys
import time
os.environ["OMP_NUM_THREADS"] = "1"
R5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "P2.2_stochastic_local_energy_r5")
sys.path.insert(0, R5)
import keyx_core as kc  # noqa: E402
import micro_test as mt  # noqa: E402


class A:
    pass


t = time.time()
inst = mt.Inst(4, 4.4, 0.25, 1, 12)
print("inst", round(time.time() - t, 1))
net = mt.make_net(12, 4, "NNBFJ", 32, 16, 0)
th = net.theta0.copy()
print("P", th.size)
a = A()
a.lr, a.shift, a.max_dtheta, a.ess_min, a.batch, a.max_steps_factor = 0.5, 1e-4, 1.0, 0.5, 500, 1
costs = {"amp_ops": kc.amp_ops(net), "cal": {"a": 0, "b": 0, "ag": 0, "bg": 0, "ts": 0}}
for arm, n in (("EXACT", 3), ("ALE0:4:16", 4), ("ALE1:4:16", 4)):
    a.n_ref = n
    t = time.time()
    tr = mt.run_sr(inst, net, th, mt.parse_arm(arm), 0, a, costs, 1e30, 1e30)
    print(arm, len(tr["step"]), "steps", round(time.time() - t, 2), "s; dE_mHa", [round(x, 2) for x in tr["dE_mHa"]])
