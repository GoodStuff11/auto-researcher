#!/usr/bin/env python3
"""
P4.1 round 23: rerun the Round 10 T1 tangent rule (R16 `sq`, R17 `nj`) at the ladder-polished parents of job 742206.
==================================================================================================================

Run with the lit_env interpreter, from any directory:
    ~/.conda/envs/lit_env/bin/python t1_polished.py --mode MODE [--parent P] [--threads N] [--chunk C]

This is a thin driver around ../P4.1_cheap_su2_r10/tangent.py (read only; its pre-registered rule, families,
checks and controls are used unchanged). It only replaces the parent parameters and the stored reference energy,
and writes results to ./results instead of r10's. Rule and the unconverged-parent caveat: RULE.md (this directory).

Options
  --mode {rule, prep, run, report} (required)
        rule    print the r10 pre-registered rule.
        prep    write results/<parent>_theta.npy and results/<parent>.json ({"E": stored energy}) for che and pcg
                from ../P4.1_cheap_su2_r22/results/theta_<parent>.npy and state_<parent>.json (last step E_new).
        run     run the r10 T1 computation for --parent and write results/tangent_<parent>.json.
        report  apply the r10 rule to results/tangent_{che,pcg}.json and print verdicts (INCOMPLETE if a parent
                is missing).
  --parent {che, pcg}   parent for mode run (default che). che = V.Model('che', 16); pcg = T2.DetModel('pcg', 15).
  --threads INT         torch/BLAS threads (default 1).
  --chunk INT           configurations per autograd batch (default 4096).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
R22 = os.path.join(TESTS, "P4.1_cheap_su2_r22", "results")
RES = os.path.join(HERE, "results")


def parse_args():
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", required=True, choices=["rule", "prep", "run", "report"])
    pa.add_argument("--parent", default="che", choices=["che", "pcg"])
    pa.add_argument("--threads", default="1")
    pa.add_argument("--chunk", type=int, default=4096)
    return pa.parse_args()


def main():
    a = parse_args()
    sys.argv = [sys.argv[0], "--threads", a.threads]  # tangent.py reads --threads at import
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = a.threads  # must precede the numpy import (round-23 che run lacked this: 988% CPU)
    sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r10"))
    import numpy as np
    import tangent as T
    T.HERE = HERE  # results (and the cached exact ground state gs_4x3.npy) live in ./results
    T.PAR = {p: os.path.join(RES, p) for p in ("che", "pcg")}
    if a.mode == "rule":
        print(T.RULE)
    elif a.mode == "prep":
        for p in ("che", "pcg"):
            st = json.load(open(os.path.join(R22, f"state_{p}.json")))
            E = st["steps"][-1]["E_new"]
            np.save(T.PAR[p] + "_theta.npy", np.load(os.path.join(R22, f"theta_{p}.npy")))
            json.dump({"E": E, "source": f"r22 state_{p}.json step {len(st['steps'])} E_new"},
                      open(T.PAR[p] + ".json", "w"), indent=1)
            print(p, "steps", len(st["steps"]), "E", E, "eps", st["steps"][-1]["eps_new"])
    elif a.mode == "run":
        a.mode, a.parent = "run", a.parent
        T.run(a)
    else:
        T.report(a)


if __name__ == "__main__":
    main()
