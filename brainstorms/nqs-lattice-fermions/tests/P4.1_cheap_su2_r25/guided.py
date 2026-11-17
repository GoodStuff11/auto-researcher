"""R38 `guided` (RULE_C.md, RULE_D.md): R21's shift arm (r24 trsr.py, read only) with S_T and f drawn from
q = 0.8 p + 0.2 p_flat, self-normalised weights p/q.
Usage: guided.py [STEPS=15] [SEED0=5000] [RESDIR=results_guided]
The RULE_C run (15 steps, seed0 5000, results_guided) was made before BLAS threads were pinned (12.0 CPU-min for
247 s wall); thread variables are now set before numpy is imported."""
import os
import sys

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
R24 = os.path.join(os.path.dirname(HERE), "P4.1_cheap_su2_r24")
STEPS = sys.argv[1] if len(sys.argv) > 1 else "15"
SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
RESD = sys.argv[3] if len(sys.argv) > 3 else "results_guided"
sys.argv = [sys.argv[0], "--mode", "run", "--arm", "shift", "--steps", STEPS, "--nsamp", "16384", "--threads", "1",
            "--cpu-cap", "60", "--res", os.path.join(HERE, RESD)]
sys.path.insert(0, R24)
import numpy as np  # noqa: E402
import trsr as T  # noqa: E402

_count = {"n": 0}


def sr_stats_guided(ctx, th, phi0, pr, r):
    _count["n"] += 1
    p = ctx.size * pr * pr
    p = p / p.sum()
    pf = ctx.size * np.abs(pr)
    pf = pf / pf.sum()
    q = 0.8 * p + 0.2 * pf
    rg = np.random.default_rng(SEED0 + _count["n"]).choice(len(q), size=len(r), p=q)
    uq, inv = np.unique(rg, return_inverse=True)
    O = T.orows(ctx, th, phi0, pr, uq)[inv]
    w = p[rg] / q[rg]
    w = w / w.sum()
    elg = (ctx.Hr @ pr)[rg] / pr[rg]
    Ob = w @ O
    Oc = O - Ob
    ST = (Oc * w[:, None]).T @ Oc
    f = (Oc * w[:, None]).T @ (elg - w @ elg)
    el = (ctx.Hr @ pr)[r] / pr[r]          # Born-sample local energies, only for the INVALID check
    return 0.5 * (ST + ST.T), f, el


T.sr_stats = sr_stats_guided
T.main()
