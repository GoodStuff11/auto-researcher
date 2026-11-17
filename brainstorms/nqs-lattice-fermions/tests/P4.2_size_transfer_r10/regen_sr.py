#!/usr/bin/env python3
r"""
Regenerate a 704844 SR-3000 state WITH its parameters (critic_capacity.train does not save theta).
Same code path as critic_capacity.train (init rng 2000+seed, r5_common.optimise_ckpt, projected exact sums),
so on one thread it should reproduce the logged energy; the script checks it against results_t2/sr_<v>.json.
NOT RUN (needs about 0.75-1.95 CPU-h per arm; must go through compute_gate.py).

Usage: python3 regen_sr.py --variant base|spin_jas2cs|c8h32 [--steps 3000]
Writes results/sr3000_<variant>.json with key 'theta' (read by: polish.py --theta-json <that file>).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CF = os.path.join(HERE, "..", "P4.2_closed_falsifier")
sys.path.insert(0, CF)
import lm_retrain as lr  # noqa: E402
cc, rc = lr.cc, lr.rc

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=list(lr.VARIANTS), required=True)
ap.add_argument("--steps", type=int, default=3000)
a = ap.parse_args()
args = cc.parse_args(["--mode", "train", "--steps", str(a.steps)] + lr.VARIANTS[a.variant] + ["--out", "x"])
T, tgt = cc.setup_target(args)
net = cc.make_net(T["lat"], T["nup"], args)
th = net.init(np.random.default_rng(2000 + args.seed), 2 * T["nup"] / T["lat"].N, args.t, args.tp, args.delta0, args.eta)
cl = rc.Cluster(net, T["basis"], T["U"], T["HG"])
th, h = cc.optimise_ckpt(cl, th, args.steps, args.lr, args.diag_shift, args.chunk, "train", T["E0"], T["V"])
E = lr.energy(cl, th, args.chunk)
ref = json.load(open(os.path.join(CF, "results_t2", f"sr_{a.variant}.json")))
rc.log(f"regen {a.variant}: E={E:.10f} logged 704844 E={ref['E']:.10f} diff={E - ref['E']:.2e}")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(dict(variant=a.variant, E=E, E_704844=ref["E"], relerr=(E - T["E0"]) / abs(T["E0"]), hist=h,
               theta=th.tolist()), open(os.path.join(HERE, "results", f"sr3000_{a.variant}.json"), "w"), default=float)
