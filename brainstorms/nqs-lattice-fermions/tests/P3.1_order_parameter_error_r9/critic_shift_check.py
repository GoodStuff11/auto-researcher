#!/usr/bin/env python3
"""
P3.1 Round 9 per-leaf critic (2026-11-16): is the response shift doing inverse iteration toward psi_0?

The response of a state phi in t1_r10.py / t2_resp_premise.py uses s = E_phi - 2 (E_phi - E_L1R(phi)) = 2 E_L1R - E_phi.
When one Lanczos step removes more than half of the energy error, s < E_0 and s - E_0 can be much smaller than
omega_1, so (Q(H - s)Q)^{-1} amplifies the psi_0 component of Q(A - a)phi: a shift-and-invert step toward the
ground state, not observable-specific information. This script measures, on the calibrated T1b toys
(--overlap2 0.92, the same trial states as t1_r10.py), per trial state:
  (s - E_0)/omega_1, |<psi_0|unit(y_A(psi))>|^2 (median over A), and noise-free ratios of
  RWRK0 / SCR2-0 (fixed base, as coded) with (a) the default shift, (b) s = E_phi - 1.0 t (well below E_0),
  (c) the default shift with the psi_0 component removed from every y (attribution only).
Usage: python3 critic_shift_check.py  (4 threads, CPU limit 600 s)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
import importlib.util  # noqa: E402
import resource  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("t1_r10", os.path.join(HERE, "t1_r10.py"))
r10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r10)
t1, REAL, unit, krylov, ritz_vec = r10.t1, r10.REAL, r10.unit, r10.krylov, r10.ritz_vec


def resp(I, phi, mode):
    phi = unit(phi)
    E = float(phi @ (I.m.H @ phi))
    r = I.m.H @ phi - E * phi
    _, EL1 = t1.ritz(I, [phi, r], [])
    s = E - 1.0 if mode == "far" else E - 2.0 * (E - EL1)
    ys = []
    for k in REAL:
        a = float(phi @ I.apply(k, phi))
        y = I.solve(I.apply(k, phi) - a * phi, s, orth=phi)
        if mode == "nopsi0":
            y = y - I.psi0 * (I.psi0 @ y)
        ys.append(unit(y))
    return ys, s


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (600, 605))
    r10.OV2 = 0.92
    tot = {m: [0, 0, 0, 0] for m in ("default", "far", "nopsi0")}   # tracked RWRK0, n, tracked SCR, n
    for size in ("3x3", "6x2", "4x3"):
        I = t1.Inst(size, 60)
        rng = np.random.default_rng(7)
        a0 = {k: float(I.psi0 @ I.apply(k, I.psi0)) for k in REAL}
        om1 = float(I.U[:, 0] @ (I.m.H @ I.U[:, 0])) - I.E0
        for lab, Vop in t1.perturbations(I).items():
            psi = r10.trial_state(I, Vop, rng)
            psi = psi if psi @ I.psi0 > 0 else -psi
            E = float(psi @ (I.m.H @ psi)); r = I.m.H @ psi - E * psi
            ap = {k: float(psi @ I.apply(k, psi)) for k in REAL}
            dl = {k: a0[k] - ap[k] for k in REAL}
            sc = [k for k in REAL if abs(dl[k]) >= t1.SCORE_REL * abs(a0[k])]
            line = f"{size} {lab:7s}"
            for mode in ("default", "far", "nopsi0"):
                y, s = resp(I, psi, mode)
                ph = ritz_vec(I, [psi, r] + y + [krylov(I, v) for v in y])
                rk = {k: (ph @ I.apply(k, ph) - ap[k]) / dl[k] for k in REAL}
                for _ in range(1):
                    y1, _s = resp(I, ph, mode)
                    ph = ritz_vec(I, [psi, r] + y1 + [krylov(I, v) for v in y1])
                scr = {k: (ph @ I.apply(k, ph) - ap[k]) / dl[k] for k in REAL}
                for k in sc:
                    tot[mode][0] += 0.5 <= rk[k] <= 2; tot[mode][1] += 1
                    tot[mode][2] += 0.5 <= scr[k] <= 2; tot[mode][3] += 1
                if mode == "default":
                    ov = np.median([(I.psi0 @ v) ** 2 for v in y])
                    line += f" (s-E0)/w1 {(s - I.E0) / om1:+.3f} |<psi0|y>|^2 {ov:.3f} |"
                line += f" {mode}: RWRK0 " + " ".join(f"{rk[k]:+.2f}" for k in sc) \
                    + " SCR0 " + " ".join(f"{scr[k]:+.2f}" for k in sc) + " |"
            print(line, flush=True)
    for mode, (a, n, b, m) in tot.items():
        print(f"TOTAL {mode:8s}: RWRK0 tracked {a}/{n}  SCR0 tracked {b}/{m}")


if __name__ == "__main__":
    main()
