#!/usr/bin/env python3
"""CHECK.md item 3, iterated: up to 25 capped LM steps (polish.step_dir, tau 0.05, line search) at lower cutoffs."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys, time  # noqa: E402,E401
import numpy as np  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import polish as P  # noqa: E402
lr_, cc, rc = P.lr, P.cc, P.rc
src = json.load(open(os.path.join(HERE, "results", "sr3000_c8h32.json")))
args = cc.parse_args(["--mode", "train"] + lr_.VARIANTS["c8h32"] + ["--out", "x"])
T, _ = cc.setup_target(args)
net = cc.make_net(T["lat"], T["nup"], args)
cl = rc.Cluster(net, T["basis"], T["U"], T["HG"])
E0, aE0 = T["E0"], abs(T["E0"])
rel = lambda E: (E - E0) / aE0  # noqa: E731
c0 = time.process_time()
for thr in [float(x) for x in sys.argv[1:]] or (1e-8, 1e-9):
    th = np.asarray(src["theta"], float)
    E = lr_.energy(cl, th, args.chunk)
    hist = []
    for it in range(25):
        Et, h0, HT, ST = lr_.tangent(cl, net, th, args.chunk)
        d, pred, rank, lam, zn = P.step_dir(Et, h0, HT, ST, thr, 0.05)
        ok = False
        for s in (1, .5, .25):
            En = lr_.energy(cl, th + s * d, args.chunk)
            if np.isfinite(En) and En < E:
                th, E, ok = th + s * d, En, True
                break
        hist.append(f"{rel(E):.4%}/r{rank}/z{zn:.3f}")
        if not ok:
            break
    print(f"thr {thr:g}: {len(hist)} steps {hist[::4]} final {rel(E):.5%} "
          f"[{time.process_time() - c0:.0f} CPU-s]", flush=True)
