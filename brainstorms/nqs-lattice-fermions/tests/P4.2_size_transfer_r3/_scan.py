import sys
import numpy as np
import micro_test as m
lam, steps, C, Hf, Hb = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
a = m.parse_args(["--steps-src", str(steps), "--channels", str(C), "--field-hidden", str(Hf), "--bf-hidden", str(Hb)])
a.tp_sel, a.J_sel = -0.25, 0.4
src = m.build_source(a, -0.25, 0.4, a.sources.split(","))
ths = []
for seed in range(3):
    th, info = m.train_source(a, src, a.channels, lam, seed, f"lam={lam} seed {seed}")
    ths.append(th)
ths = np.array(ths)
print("dist", [round(np.linalg.norm(ths[i] - ths[j]) / np.linalg.norm(ths[i]), 3) for i in range(3) for j in range(i + 1, 3)])
