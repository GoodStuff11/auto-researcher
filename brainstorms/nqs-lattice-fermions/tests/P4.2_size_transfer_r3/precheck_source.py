#!/usr/bin/env python3
"""Login-node pre-check (about a minute): train the decisive SOURCE (4x2 torus + 8-site tilted
cluster, 2 holes each, t'=-0.25, J=0.4) for 3 seeds, both the TCMN way (joint, min-norm) and
the round-2 way (4x2 only, no penalty). Reports source energies, the null dimension of the
joint SR metric (identifiability), and the seed-to-seed spread of the trained parameters.
It says nothing about transfer; the target-side seed spread is measured by the Slurm run.

Usage: python precheck_source.py [STEPS]   (STEPS = SR steps per training, default 400)
"""
import sys

import numpy as np

import micro_test as m


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    a = m.parse_args(["--steps-src", str(steps)])
    a.tp_sel, a.J_sel = -0.25, 0.4
    src = m.build_source(a, -0.25, 0.4, a.sources.split(","))
    m.log("source clusters: " + ", ".join(f"{s['lat'].name()} dim={len(s['basis'])} E0={s['E0']:.8f}" for s in src))
    for tag, clusters, lam in (("TCMN", src, a.lam), ("r2-style", src[:1], 0.0)):
        ths = []
        for seed in range(3):
            th, info = m.train_source(a, clusters, a.channels, lam, seed, f"{tag} seed {seed}")
            ths.append(th)
        ths = np.array(ths)
        d = [np.linalg.norm(ths[i] - ths[j]) / np.linalg.norm(ths[i]) for i in range(3) for j in range(i + 1, 3)]
        m.log(f"{tag}: relative parameter distance between seeds {['%.3f' % x for x in d]}")


if __name__ == "__main__":
    main()
