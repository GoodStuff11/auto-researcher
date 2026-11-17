#!/usr/bin/env python3
"""Validity check of polish.py's zero LM step on the SR-3000 c8h32 state (CHECK.md). Exact sums, 1 thread."""
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
th0 = np.asarray(src["theta"], float)
rel = lambda E: (E - E0) / aE0  # noqa: E731
c0 = time.process_time()
out = {}

# 1. gradient and its split over the kept metric subspace
E, n2, g, S = rc.reduced(cl, th0, args.chunk, True)
Et, h0, HT, ST = lr_.tangent(cl, net, th0, args.chunk)
print(f"E={E:.12f} relerr={rel(E):.5%}  |g|={np.linalg.norm(g):.3e}  |2 h0|={np.linalg.norm(2*h0):.3e} "
      f"|g-2h0|={np.linalg.norm(g-2*h0):.1e}  |S-ST|={np.abs(S-ST).max():.1e}")
w, U = np.linalg.eigh(0.5 * (ST + ST.T))
print(f"metric eig: max {w.max():.3e}, min {w.min():.3e}; #>1e-5max {np.sum(w>1e-5*w.max())}, "
      f">1e-8max {np.sum(w>1e-8*w.max())}, >1e-11max {np.sum(w>1e-11*w.max())}")
for thr in (1e-5, 1e-6, 1e-7, 1e-8):
    k = w > thr * w.max()
    gin = U[:, k].T @ g
    print(f"  thr {thr:g}: rank {k.sum()}  |g_in|^2/|g|^2 = {gin@gin/(g@g):.4f}  "
          f"natural-grad norm^2 g S^-1 g (kept) = {np.sum(gin**2/w[k]):.3e}")
out["g"] = float(np.linalg.norm(g))

# 3. LM at several cutoffs (polish.step_dir), with cap and line search
for thr in (1e-5, 1e-6, 1e-7):
    d, pred, rank, lam, zn = P.step_dir(Et, h0, HT, ST, thr, 0.05)
    tr = [(s, rel(lr_.energy(cl, th0 + s * d, args.chunk))) for s in (1, .5, .25)]
    print(f"LM thr {thr:g}: rank {rank} |z|={zn:.3e} lam={lam:.2e} |dth|={np.linalg.norm(d):.3e} "
          f"pred {rel(pred):.5%} trials {[(s, f'{r:.5%}') for s, r in tr]}")

# 2a. gradient descent with line search
th, Ec = th0.copy(), E
best = rel(E)
for it in range(15):
    Ec, _, gg, _ = rc.reduced(cl, th, args.chunk, True)
    step = None
    for eta in [10.0 * 0.5 ** j for j in range(30)]:
        En = lr_.energy(cl, th - eta * gg, args.chunk)
        if En < Ec:
            step = (eta, En); break
    if step is None:
        print(f"GD it {it}: no descent"); break
    th = th - step[0] * gg
    best = rel(step[1])
print(f"GD 15 its: relerr {rel(E):.5%} -> {best:.5%}  [{time.process_time()-c0:.0f} CPU-s]")

# 2b. SR as in training (diag shift, lr) for 40 steps with backtracking
th, Ec = th0.copy(), E
for it in range(40):
    Ec, _, gg, SS = rc.reduced(cl, th, args.chunk, True)
    Sr = SS + args.diag_shift * np.diag(np.diag(SS)) + 1e-6 * np.diag(SS).max() * np.eye(len(SS))
    d = np.linalg.solve(Sr, gg)
    eta = args.lr
    for _ in range(12):
        En = lr_.energy(cl, th - eta * d, args.chunk)
        if En < Ec:
            th = th - eta * d; break
        eta *= 0.5
srE = lr_.energy(cl, th, args.chunk)
print(f"SR 40 steps (lr {args.lr}, shift {args.diag_shift}): relerr {rel(E):.5%} -> {rel(srE):.5%}  "
      f"[{time.process_time()-c0:.0f} CPU-s]")
