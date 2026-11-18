#!/usr/bin/env python3
"""P5.1 round 17: mechanism test of R51.26 (RULE_snrcut.md addendum 2).  rho_c = 1 soft cut, guided N = 4096, 5x2,
seeds 9000 + n, 120 steps, tau 0.005, but the filtered direction keeps the plain SR step's normalisation
tau / sqrt(x^T S_N x) (x unfiltered), so no kept mode is lengthened.  Reuses snr_run.py (read only).
    ~/.conda/envs/lit_env/bin/python snr_plainnorm.py
Output results/snrP_guided_N4096_rho1.0_tau0.005_s9000/run.json (per-step E, kept modes, len_ratio = |d_f|_S/tau)."""
import json
import os
import time

import numpy as np
import torch

import snr_run as S

NS, B, SHIFT = S.NS, S.B, S.SHIFT
N, RHO, TAU, SEED0, STEPS = 4096, 1.0, 0.005, 9000, 120


def step(ctx, th, phi0, pr, seed):
    uq, inv, r, Z, w, Oc, ec = S.rows(ctx, th, phi0, pr, N, seed, "guided")
    ST = (Oc * w[:, None]).T @ Oc
    f = (Oc * w[:, None]).T @ ec
    wv, U = np.linalg.eigh(0.5 * (ST + ST.T))
    bidx = np.repeat(np.arange(B), N // B)
    cb = np.zeros((B, len(uq)))
    np.add.at(cb, (bidx, inv), 1.0)
    Fb = (Oc.T @ (cb * (r * ec)).T) * (B / Z)
    fk = U.T @ f
    sd = (U.T @ Fb).std(1, ddof=1) / np.sqrt(B)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(sd > 0, np.abs(fk) / sd, 0.0)
        filt = np.where(rho > 0, 1.0 / (1.0 + (RHO / rho) ** 6), 0.0)
    x0 = fk / (wv + SHIFT)
    ml0 = np.sqrt(max(float(np.sum(wv * x0 * x0)), 1e-300))  # plain step's normalisation
    xk = x0 * filt
    lr = np.sqrt(max(float(np.sum(wv * xk * xk)), 0.0)) / ml0  # |d_f|_S / tau
    return -TAU * (U @ xk) / ml0, len(uq), int((filt > 0.5).sum()), lr


def main():
    ctx = NS.GCtx(5, 2, 8, S.START)
    E00, pr, phi0 = ctx.energy(ctx.th0)
    th = ctx.th0.clone()
    tag = f"snrP_guided_N{N}_rho{RHO}_tau{TAU}_s{SEED0}"
    res = os.path.join(S.HERE, "results", tag)
    os.makedirs(res, exist_ok=True)
    hist, c0 = [], time.process_time()
    for n in range(1, STEPS + 1):
        d, nu, kept, lr = step(ctx, th, phi0, pr, SEED0 + n)
        th = th + torch.tensor(d)
        E, pr, phi0 = ctx.energy(th)
        hist.append(dict(E=E, n_unique=nu, kept=kept, len_ratio=lr))
        if n % 20 == 0 or n == STEPS:
            NS.log(f"{tag} step {n}: E {E:.7f} (E-E_start {E - E00:+.3e}) kept {kept} len_ratio {lr:.3f} "
                   f"[{(time.process_time() - c0) / 60:.2f} CPU-min]")
            json.dump(dict(tag=tag, E_start=E00, nsamp=N, tau=TAU, rho=RHO, seed0=SEED0, norm="plain", steps=hist,
                           status="DONE" if n == STEPS else "RUNNING", cpu_min=(time.process_time() - c0) / 60),
                      open(os.path.join(res, "run.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
