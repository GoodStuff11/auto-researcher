#!/usr/bin/env python3
"""Context only, written AFTER the t1i KILL (no decision weight): does arm S stop early (nit ~40 of 300)?
Re-trains S at size 2 (largest), both seeds, with ftol 1e-15 and gtol 1e-12, and prints the L-BFGS message.
Usage: python3 t1i_tol_posthoc.py"""
import os, sys, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1i_phaseS as X
from scipy.optimize import minimize
resource.setrlimit(resource.RLIMIT_CPU, (300, 310))
for seed in (11, 22):
    d = X.build(2, seed); H, psi = d["H"], d["psiF"]
    B, lam, v = X.basis(H, psi, d["S"])
    best = None
    for s in (1.0, -1.0):
        r = minimize(X.obj, s * 0.1 * v, args=(B, H, psi, False), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=300, ftol=1e-15, gtol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    sc = X.score(d, X.phase(best.x, B, False))
    X.P.log(f"size 2 seed {seed}: tight tol dE_complex {best.fun - d['E0']:.5f} dEs {sc['dEs']:.5f} "
            f"nit {best.nit} msg {best.message}")
