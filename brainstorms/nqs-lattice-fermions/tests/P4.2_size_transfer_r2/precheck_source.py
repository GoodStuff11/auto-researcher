#!/usr/bin/env python3
"""Login-node pre-check (seconds): train MFNG and the foil on the decisive SOURCE (4x2 torus,
2 holes, t'=-0.25, J=0.4) and report energy and sign-budget (E_sign) excess in the k=0 block.
This checks only that the source training reaches a regime where the route's sign beats the
foil's at the source; it says nothing about transfer.

Usage: python precheck_source.py [STEPS]   (STEPS = SR steps per training, default 300)
"""
import sys

import numpy as np
import scipy.sparse as sp

import micro_test as m


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    a = m.parse_args(["--lbfgs-iters", "2000"])
    lat, bas, nup = m.system(4, 2, 2, a)
    H = m.build_H(lat, bas)
    mz = m.MomentumZero(lat, bas)
    Hk = (mz.U.T @ H @ mz.U).tocsr()
    w, _ = m.ed_k0(Hk)
    E0 = w[0]
    m.log(f"4x2 source: dim={len(bas)} orbits={mz.n_orb} E0(k=0)={E0:.8f} E1={w[1]:.8f}")
    for kind, C in (("route", a.channels), ("foil", 0)):
        net = m.new_net(lat, nup, a, C)
        th0 = net.init(np.random.default_rng(0), 6 / 8, a.t, a.tp, a.delta0, a.eta)
        th, E, _ = m.optimise(net, th0, bas, H, steps, a.lr, a.diag_shift, a.chunk, kind, E0)
        psi = m.state_vector(net, th, bas, a.chunk)
        sb = m.sign_budget(Hk, mz.reduce(psi), E0, a)
        m.log(f"{kind}: npar={net.npar} dE/V={(E - E0) / 8:.4e} A_sign/V={sb['A'] / 8:.4e} "
              f"f_sign={sb['A'] / (E - E0):.3f}")


if __name__ == "__main__":
    main()
