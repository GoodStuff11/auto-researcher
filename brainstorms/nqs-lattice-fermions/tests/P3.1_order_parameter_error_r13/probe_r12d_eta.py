#!/usr/bin/env python3
"""VOI probe (login node, not a pre-registered test): t1_r12d.py (R12.3d T1 toys) unchanged except grade-R noise
ETA set by --eta (default 0.9 0.5). Output JSON goes to ./results/probe_r12d_eta<in>_<size>_d<delta>.json; the r12
results directory is not touched. Usage: python3 probe_r12d_eta.py --size 5x2 --delta 0.05 [--eta 0.9 0.5]"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, importlib.util, json, resource  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--size", default="5x2"); p.add_argument("--delta", type=float, default=0.05)
p.add_argument("--eta", type=float, nargs=2, default=[0.9, 0.5]); p.add_argument("--cpu-limit", type=int, default=120)
a = p.parse_args()
resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_limit, a.cpu_limit + 5))
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("r12d", os.path.join(HERE, "..", "P3.1_order_parameter_error_r12", "t1_r12d.py"))
r12d = importlib.util.module_from_spec(spec); spec.loader.exec_module(r12d)
r12d.dav.ETA = tuple(a.eta)
I, info = r12d.dav.build(a.size, 15, a.delta)
out = os.path.join(HERE, "results", f"probe_r12d_eta{a.eta[0]}_{a.size}_d{a.delta}.json")
rows = r12d.run(I, 8, a.cpu_limit, out)
json.dump(dict(info=info, eta=a.eta, rows=rows), open(out, "w"), indent=1, default=str)
for arm in ("DAV", "L1R"):
    print(f"ETA={tuple(a.eta)} {arm} {a.size} delta {a.delta}: n, coverage, median w/wex, median w/|delta| =",
          r12d.score(rows, arm))
