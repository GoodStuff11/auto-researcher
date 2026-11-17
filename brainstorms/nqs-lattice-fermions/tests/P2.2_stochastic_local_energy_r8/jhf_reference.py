#!/usr/bin/env python3
"""P2.2 R7.4: measured delta and compensator excess for a cheap Jastrow-HF reference on H8 STO-3G LOW R=3.0.

Reference psi_ref(x) = D_RHF(x) * exp(sum_k theta_k F_k(x)), D_RHF = RHF determinant (Loewdin-basis minors),
theta optimised by full-sum exact energy minimisation (L-BFGS, analytic gradient). Variants:
  G   : Gutzwiller only, F = sum_i n_iu n_id                                      (1 parameter)
  JD  : Gutzwiller + distance Jastrow sum_{|i-j|=d} n_i n_j, d = 1..K-1           (K parameters)
  JF  : site-resolved on-site g_i + all pair v_ij n_i n_j (i<j)                   (K + K(K-1)/2 parameters)
Compensator Y_ref = D_ref^-1 L_ref D_ref^-1 = path gadget build_L with psi_ref in place of psi0 (pair_data on
psi_ref). Excess on a state phi (normalised): phi^T Y_ref phi = u^T L_ref u, u = phi / psi_ref.
delta definitions (both reported):
  delta_rev = psi_ref^2-weighted RMS of (psi0/psi_ref - 1)   [psi0 = psi_ref(1 + delta xi): the relative-noise model
              that gives E[excess] = delta^2 tr L_ref for Y built from psi_ref and evaluated on psi0]
  delta_fwd = psi0^2-weighted RMS of (psi_ref/psi0 - 1)      [= d_eff_rel of real_error.py]
Usage: jhf_reference.py [--nn-cpu 60]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from itertools import combinations  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import path_gadget as pg  # noqa: E402  (RLIMIT_CPU 600 s)
rt = pg.rt
mt = rt.mt
_spec = importlib.util.spec_from_file_location(
    "keyx_core", os.path.join(HERE, "..", "P2.2_stochastic_local_energy_r5", "keyx_core.py"))
kc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kc)

TAU = 2e-4
EPS = {"eps_S": 2.2387211385683377e-3, "grid_end": 0.1}
NAT, R = 8, 3.0
THR = 1e-9


def rhf_vector(occ, K, nocc):
    X = mt.geometry(NAT, R, "chain")
    S, hao, Gao, enuc = mt.ao_integrals(X)
    Cm, Eel = mt.rhf(S, hao, Gao, nocc)
    s, U = np.linalg.eigh(S)
    C = (U @ np.diag(s ** 0.5) @ U.T) @ Cm[:, :nocc]  # occupied MOs in the Loewdin basis
    n = occ.shape[0]
    psi = np.empty(n)
    for x in range(n):
        a = np.nonzero(occ[x, :K])[0]
        b = np.nonzero(occ[x, K:])[0]
        psi[x] = np.linalg.det(C[a]) * np.linalg.det(C[b])
    return psi / np.linalg.norm(psi), Eel + enuc


def features(occ, K, kind):
    na, nb = occ[:, :K].astype(float), occ[:, K:].astype(float)
    nn = na + nb
    db = na * nb
    if kind == "G":
        return db.sum(1, keepdims=True)
    if kind == "JD":
        cols = [db.sum(1)] + [sum(nn[:, i] * nn[:, i + d] for i in range(K - d)) for d in range(1, K)]
        return np.stack(cols, 1)
    if kind == "JF":
        cols = [db[:, i] for i in range(K)] + [nn[:, i] * nn[:, j] for i, j in combinations(range(K), 2)]
        return np.stack(cols, 1)
    raise ValueError(kind)


def optimise(H, d0, F):
    def eg(t):
        p = d0 * np.exp(F @ t)
        nrm = p @ p
        Hp = H @ p
        E = (p @ Hp) / nrm
        g = 2 * F.T @ (p * (Hp - E * p)) / nrm
        return E, g
    r = minimize(eg, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B",
                 options={"maxiter": 2000, "gtol": 1e-10, "ftol": 1e-15})
    p = d0 * np.exp(F @ r.x)
    return p / np.linalg.norm(p), r


def excess(L, ref, phi, nzr):
    u = np.where(nzr, phi / np.where(nzr, ref, 1), 0.0)
    return float(u @ (L @ u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-cpu", type=float, default=60.0)
    a = ap.parse_args()
    H, codes, occ, K, nocc = rt.build(NAT, R, "LOW")
    w, v = rt.lowest(H, 1)
    E0, psi0 = float(w[0]), v[:, 0]
    n = H.shape[0]
    occ = np.asarray(occ).astype(np.int8)
    d0, Ehf = rhf_vector(occ, K, nocc)
    d0 *= np.sign(d0 @ psi0)
    Ed0 = float(d0 @ (H @ d0))
    print(f"H8 n={n} E0={E0:.8f} E_RHF(SCF)={Ehf:.8f} E[RHF vector]={Ed0:.8f} (check) "
          f"dE_RHF={1e3*(Ed0-E0):.2f} mHa cpu={pg.cpu():.0f}s", flush=True)
    refs = {"RHF": d0}
    info = {"RHF": dict(nparam=0)}
    for kind in ("G", "JD", "JF"):
        F = features(occ, K, kind)
        p, r = optimise(H, d0, F)
        p *= np.sign(p @ psi0)
        refs[kind] = p
        info[kind] = dict(nparam=F.shape[1], nit=int(r.nit), theta=[float(t) for t in r.x])
        print(f" {kind}: P={F.shape[1]} nit={r.nit} E={float(p @ (H @ p)):.8f} cpu={pg.cpu():.0f}s", flush=True)

    # trained NNBF-J (A1 architecture h4/hj4, seed 0), snapshot at dE <= 0.74 mHa and final
    oi = np.array([np.nonzero(o)[0] for o in occ])

    class A_:
        pass
    aa = A_(); aa.hidden, aa.hj, aa.window, aa.groups = 4, 4, None, 2
    net = kc.make_net(K, NAT, "NNBFJ", aa, 0)
    states = {}
    t1 = pg.cpu(); cnt = [0]

    def cb(xk):
        cnt[0] += 1
        if pg.cpu() - t1 > a.nn_cpu:
            raise StopIteration
        if cnt[0] % 10 or "NNBFJ_0.74" in states:
            return
        ph, _ = net.amplitudes(xk, occ, oi)
        ph = ph / np.linalg.norm(ph); ph *= np.sign(ph @ psi0)
        if 1e3 * (float(ph @ (H @ ph)) - E0) <= 0.74:
            states["NNBFJ_0.74"] = (ph, cnt[0])
    r = minimize(lambda t: net.energy_grad(t, H, occ, oi), net.theta0, jac=True, method="L-BFGS-B", callback=cb,
                 options={"maxiter": 1500, "maxcor": 30, "gtol": 1e-12, "ftol": 1e-16})
    ph, _ = net.amplitudes(r.x, occ, oi)
    ph = ph / np.linalg.norm(ph); ph *= np.sign(ph @ psi0)
    states["NNBFJ_final"] = (ph, int(r.nit))
    print(f" NNBF-J h4 trained: nit={r.nit} cpu={pg.cpu()-t1:.0f}s; snapshots {list(states)}", flush=True)

    targets = {"psi0": psi0}
    for k, (p, it) in states.items():
        targets[k] = p
    tE = {k: 1e3 * (float(p @ (H @ p)) - E0) for k, p in targets.items()}
    print(" target dE (mHa): " + " ".join(f"{k}={v:.3f}" for k, v in tE.items()), flush=True)

    out = dict(E0=E0, E_RHF=Ed0, refs={}, target_dE_mHa=tE, tau=TAU, eps=EPS)
    for name, ref in refs.items():
        nzr = np.abs(ref) > THR
        nz0 = np.abs(psi0) > THR
        Eref = float(ref @ (H @ ref))
        both = nzr & nz0
        d_rev = float(np.sqrt(np.sum(ref[nzr] ** 2 * (psi0[nzr] / ref[nzr] - 1) ** 2)))
        d_fwd = float(np.sqrt(np.sum(psi0[nz0] ** 2 * (ref[nz0] / psi0[nz0] - 1) ** 2)))
        rr = psi0[both] / ref[both]
        lost = float(np.sum(psi0[~nzr] ** 2))  # psi0 weight on psi_ref-zero dets (certificate undefined there)
        rec = dict(dE_mHa=1e3 * (Eref - E0), infidelity=float(1 - (ref @ psi0) ** 2), delta_rev=d_rev,
                   delta_fwd=d_fwd, n_ref_zero=int((~nzr).sum()), psi0_weight_on_ref_zero=lost,
                   ratio_psi0_over_ref_quantiles=[float(q) for q in np.quantile(np.abs(rr), [0.01, 0.5, 0.99])],
                   sign_mismatch_weight=float(np.sum(psi0[both] ** 2 * (rr < 0))), **info[name])
        P = rt.pair_data(H, ref, occ, K)
        for ek, eps in EPS.items():
            mask = np.abs(P["h"]) <= eps
            L, st = pg.build_L(P, mask, nzr, n)
            trL = float(L.diagonal().sum())
            b_ref = excess(L, ref, ref, nzr)
            ex = {k: excess(L, ref, p, nzr) for k, p in targets.items()}
            rec[ek] = dict(trL_ref=trL, needed_delta=float(np.sqrt(TAU / trL)), excess_on_ref=b_ref,
                           pred_rel=d_rev ** 2 * trL, unreachable=st["unreachable"],
                           excess={k: v_ for k, v_ in ex.items()},
                           excess_over_tau={k: v_ / TAU for k, v_ in ex.items()})
            print(f" ref={name} {ek}: trL_ref={trL:.3e} need delta<={np.sqrt(TAU/trL):.3f} pred d_rev^2 trL="
                  f"{d_rev**2*trL/TAU:.2f} tau | excess/tau: "
                  + " ".join(f"{k}={v_/TAU:.3g}" for k, v_ in ex.items()) + f" (on ref {b_ref:.1e}) cpu={pg.cpu():.0f}s",
                  flush=True)
        out["refs"][name] = rec
        print(f" ref={name}: dE={rec['dE_mHa']:.2f} mHa infid={rec['infidelity']:.3e} delta_rev={d_rev:.3f} "
              f"delta_fwd={d_fwd:.3f} ref-zero={rec['n_ref_zero']} psi0-wt-there={lost:.1e} "
              f"|psi0/ref| q01/50/99={rec['ratio_psi0_over_ref_quantiles']} sign-mismatch wt={rec['sign_mismatch_weight']:.2e}",
              flush=True)
    out["cpu_s"] = pg.cpu()
    with open(os.path.join(HERE, "results", "jhf_reference_H8.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"total cpu {pg.cpu():.0f}s")


if __name__ == "__main__":
    main()
