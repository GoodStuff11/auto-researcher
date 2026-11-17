#!/usr/bin/env python3
"""P2.2 R7.4 T1b (rule: REAL_ERROR.md). Excess psi_NN^T Y psi_NN of the path-gadget compensator on NNBF-J states
trained by full-sum L-BFGS (r5 keyx_core.Ansatz, arm NNBFJ, hidden 64, hj 32) on H6/H8 STO-3G LOW, R = 3.0.
Usage: real_error.py --n 6 --seeds 0 1 2 --maxiter 3000 [--n 8 ...]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402  (BLAS initialised single-threaded before path_gadget sets "4")
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
try:
    from threadpoolctl import threadpool_limits  # noqa: E402
    threadpool_limits(1)
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import path_gadget as pg  # noqa: E402  (sets RLIMIT_CPU 600 s)
rt = pg.rt
_spec = importlib.util.spec_from_file_location(
    "keyx_core", os.path.join(HERE, "..", "P2.2_stochastic_local_energy_r5", "keyx_core.py"))
kc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kc)

TAU = 2e-4
EPS = {"eps_S": 2.2387211385683377e-3, "grid_end": 0.1}


class A_:
    pass


def analyse(psi0, phi, L, nz, H, mask_h, P, Y_diag):
    """phi normalized and sign aligned. Returns dict of excess and decomposition."""
    n = len(psi0)
    u = np.where(nz, phi / np.where(nz, psi0, 1), 0.0)
    exc = float(u @ (L @ u))
    r = np.where(nz, u - 1, 0.0)
    w0 = psi0 ** 2
    d_eff = float(np.sqrt(w0[nz] @ r[nz] ** 2))
    diff = phi - psi0
    add_norm2 = float(diff @ diff)
    trL = float(L.diagonal().sum())
    Yd = Y_diag
    order = np.argsort(-Yd)
    top = order[: max(1, int(0.01 * nz.sum()))]
    big = np.where(Yd > 1e3)[0]
    # true dropped-term bias of the screened estimator on phi
    rd, cd, hd = P["r"][mask_h], P["c"][mask_h], P["h"][mask_h]
    scr = float(2 * (hd * phi[rd] * phi[cd]).sum())
    # per-det excess share: u^T L u = sum_a u_a (L u)_a ; contribution of top-Y dets
    Lu = L @ u
    share_top = float((u[top] * Lu[top]).sum() / exc) if exc != 0 else 0.0
    out = dict(excess=exc, excess_over_tau=exc / TAU, screen_bias_on_state=scr,
               d_eff_rel=d_eff, pred_relative=d_eff ** 2 * trL,
               additive_norm=float(np.sqrt(add_norm2)),
               pred_additive=add_norm2 / nz.sum() * float(Yd.sum()),
               trL=trL, maxYii=float(Yd.max()),
               top1pct=dict(n=int(len(top)), Yii_median=float(np.median(Yd[top])),
                            psi0_abs_median=float(np.median(np.abs(psi0[top]))),
                            r_abs_median=float(np.median(np.abs(r[top]))), r_abs_max=float(np.abs(r[top]).max()),
                            diff_abs_median=float(np.median(np.abs(diff[top]))),
                            excess_share=share_top),
               Yii_gt_1e3=dict(n=int(len(big)),
                               r_abs_median=float(np.median(np.abs(r[big]))) if len(big) else None,
                               r_abs_max=float(np.abs(r[big]).max()) if len(big) else None,
                               psi0_abs_median=float(np.median(np.abs(psi0[big]))) if len(big) else None))
    return out


def run(nat, seeds, maxiter, seed_cpu=1e9):
    t0 = time.time()
    H, codes, occ, K, nocc = rt.build(nat, 3.0, "LOW")
    w, v = rt.lowest(H, 1)
    E0, psi0 = float(w[0]), v[:, 0]
    n = H.shape[0]
    P = rt.pair_data(H, psi0, occ, K)
    nz = np.abs(psi0) > 1e-9
    occ = np.asarray(occ).astype(np.int8)
    oi = np.array([np.nonzero(o)[0] for o in occ])
    # rel. abundances of psi0 magnitudes
    print(f"H{nat} n={n} E0={E0:.8f} |psi0| quantiles 1/10/50%: "
          f"{np.quantile(np.abs(psi0[nz]), [0.01, 0.1, 0.5])} cpu={pg.cpu():.0f}s", flush=True)
    Ls = {}
    for k, eps in EPS.items():
        mask = np.abs(P["h"]) <= eps
        L, st = pg.build_L(P, mask, nz, n)
        Yd = np.where(nz, L.diagonal() / np.where(nz, psi0, 1) ** 2, 0.0)
        Ls[k] = (L, mask, Yd)
    res = dict(n_atoms=nat, n=n, E0=E0, runs=[])
    for seed in seeds:
        a = A_(); a.hidden, a.hj, a.window, a.groups = 64, 32, None, 2
        net = kc.make_net(K, nat, "NNBFJ", a, seed)
        t1 = pg.cpu()
        snaps = {}  # DIAGNOSTIC (not in the rule): first iterate (checked every 10) with dE <= target
        cnt = [0]

        def cb(xk):
            cnt[0] += 1
            if pg.cpu() - t1 > seed_cpu:  # per-seed CPU cap (REAL_ERROR.md: "or until the CPU cap")
                raise StopIteration
            if cnt[0] % 10:
                return
            ph, _ = net.amplitudes(xk, occ, oi)
            ph = ph / np.linalg.norm(ph); ph *= np.sign(ph @ psi0)
            dE = 1e3 * (float(ph @ (H @ ph)) - E0)
            for tgt in (2.0, 0.74, 0.2):
                if dE <= tgt and str(tgt) not in snaps:
                    L, mask, Yd = Ls["eps_S"]
                    a_ = analyse(psi0, ph, L, nz, H, mask, P, Yd)
                    snaps[str(tgt)] = dict(it=cnt[0], dE_mHa=dE, excess=a_["excess"], d_eff_rel=a_["d_eff_rel"],
                                           pred_relative=a_["pred_relative"], pred_additive=a_["pred_additive"],
                                           top1_r_med=a_["top1pct"]["r_abs_median"])
        r = minimize(lambda t: net.energy_grad(t, H, occ, oi), net.theta0, jac=True, method="L-BFGS-B", callback=cb,
                     options={"maxiter": maxiter, "maxcor": 30, "gtol": 1e-12, "ftol": 1e-16})
        phi, _ = net.amplitudes(r.x, occ, oi)
        phi = phi / np.linalg.norm(phi)
        phi *= np.sign(phi @ psi0)
        E = float(phi @ (H @ phi))
        run = dict(seed=seed, nit=int(r.nit), dE_mHa=1e3 * (E - E0), infidelity=float(1 - (phi @ psi0) ** 2),
                   train_cpu_s=pg.cpu() - t1, admissible=bool(1e3 * (E - E0) < 1.6), snapshots=snaps)
        for k, (L, mask, Yd) in Ls.items():
            run[k] = analyse(psi0, phi, L, nz, H, mask, P, Yd)
        res["runs"].append(run)
        s = run["eps_S"]; g = run["grid_end"]
        print(f" seed={seed} nit={run['nit']} dE={run['dE_mHa']:.3f} mHa infid={run['infidelity']:.2e} "
              f"| eps_S: excess={s['excess']:.3e} ({s['excess_over_tau']:.2f} tau) screen-bias={s['screen_bias_on_state']:.2e} "
              f"d_eff={s['d_eff_rel']:.2e} pred_rel={s['pred_relative']:.2e} pred_add={s['pred_additive']:.2e} "
              f"top1%: |r| med {s['top1pct']['r_abs_median']:.2e} max {s['top1pct']['r_abs_max']:.2e} "
              f"share {s['top1pct']['excess_share']:.2f} "
              f"| end: excess={g['excess']:.3e} ({g['excess_over_tau']:.2f} tau) "
              f"| snaps " + " ".join(f"dE<={k}:it{v['it']} exc={v['excess']:.2e} d_eff={v['d_eff_rel']:.1e} rel={v['pred_relative']:.1e} add={v['pred_additive']:.1e}" for k, v in snaps.items())
              + f" | train cpu {run['train_cpu_s']:.0f}s total cpu {pg.cpu():.0f}s", flush=True)
    adm = [r_["eps_S"]["excess"] for r_ in res["runs"] if r_["admissible"]]
    res["median_excess_eps_S"] = float(np.median(adm)) if adm else None
    res["median_excess_grid_end"] = float(np.median([r_["grid_end"]["excess"] for r_ in res["runs"] if r_["admissible"]])) if adm else None
    res["cpu_s"] = pg.cpu(); res["wall_s"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--seed-cpu", type=float, default=1e9)
    a = ap.parse_args()
    res = run(a.n, a.seeds, a.maxiter, a.seed_cpu)
    m = res["median_excess_eps_S"]
    print(f"H{a.n}: median excess at eps_S = {m} Ha ({(m or 0)/TAU:.2f} tau); grid end {res['median_excess_grid_end']}")
    with open(os.path.join(HERE, "results", f"real_error_H{a.n}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
