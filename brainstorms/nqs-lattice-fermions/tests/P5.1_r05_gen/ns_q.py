#!/usr/bin/env python3
"""
P5.1 round 5, R51.11 T2 driver: R51.8's ns_run.py unchanged (read only), with the guided sampler's mixture weight
set by --qmix (q = (1 - qmix) p + qmix p_flat; 0.2 reproduces ns_run's guided, 1.0 is q ~ |psi|).
    ~/.conda/envs/lit_env/bin/python ns_q.py --qmix 1.0 <ns_run.py options with --sampler guided>
"""
import os
import sys

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P5.1_size_scaling_r04"))
import numpy as np  # noqa: E402
import ns_run as NS  # noqa: E402

i = sys.argv.index("--qmix")
QMIX = float(sys.argv[i + 1])
del sys.argv[i:i + 2]
T = NS.T


def stats(ctx, th, phi0, pr, N, seed, sampler, exact):
    assert sampler == "guided" and not exact
    p = ctx.size * pr * pr
    p = p / p.sum()
    pf = ctx.size * np.abs(pr)
    q = (1 - QMIX) * p + QMIX * pf / pf.sum()
    rg = np.random.default_rng(seed).choice(len(q), size=N, p=q)
    uq, inv = np.unique(rg, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(uq)).astype(float)
    w = cnt * p[uq] / q[uq]
    w = w / w.sum()
    O = T.orows(ctx, th, phi0, pr, uq)
    elg = (ctx.Hr @ pr)[uq] / pr[uq]
    Oc = O - w @ O
    ST = (Oc * w[:, None]).T @ Oc
    f = (Oc * w[:, None]).T @ (elg - w @ elg)
    return 0.5 * (ST + ST.T), f, len(uq)


NS.stats = stats
if __name__ == "__main__":
    NS.main()
