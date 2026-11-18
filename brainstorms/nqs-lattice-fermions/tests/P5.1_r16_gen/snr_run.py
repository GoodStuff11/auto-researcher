#!/usr/bin/env python3
"""
P5.1 round 16: T1 drivers for R51.26 (`snrcut`, noise-level spectral filter of the SR step) and R51.31 (`minsr`,
Born MinSR with the SR shift at N < P).  Rules: RULE_snrcut.md, RULE_minsr.md (written before any run).
    ~/.conda/envs/lit_env/bin/python snr_run.py --mode snr --nsamp 4096 --rho 1.0 [--steps 120]
    ~/.conda/envs/lit_env/bin/python snr_run.py --mode minsr --nsamp 1024 --tau 0.00125 --max-draws 819200
    ~/.conda/envs/lit_env/bin/python snr_run.py --mode check     (identity test against ns_run.py, 3 steps)
Imports ../P5.1_size_scaling_r04/ns_run.py (GCtx, stats, T.orows, T.d_shift) read only, so the 5x2 instance, start,
draws (seed0 + n, rng.choice on q) and step family (shift 1e-3, length tau in the sample metric) are R51.8's.
snr:   draws split by position into B = 16 equal batches; per-batch forces f_b (self-normalised with the global
       weights, so mean_b f_b = f); per-mode SNR rho_k = |u_k.f| / (sd_b(u_k.f_b)/sqrt(B)); x = sum_k u_k (u_k.f)
       /(w_k + shift) * 1/(1 + (rho_c/rho_k)^6) (Schmitt & Heyl's soft cut); d = -tau x/sqrt(x S_T x).  rho_c = 0 is
       plain SR (checked against ns_run in --mode check).
minsr: same shifted step solved in sample space (push-through): A = diag(sqrt w) Oc, x = A^T (A A^T + shift)^-1 b,
       b = sqrt(w) e_c, |x|_S = |A x|; identical to SR with the shift at any N (checked in --mode check).
Output results/<tag>/run.json (per-step E, n_unique, kept modes).  Nothing else written.
"""
import argparse
import json
import os
import sys
import time

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NSDIR = os.path.join(os.path.dirname(HERE), "P5.1_size_scaling_r04")
sys.path.insert(0, NSDIR)
import ns_run as NS  # noqa: E402
import torch  # noqa: E402

T = NS.T
SHIFT = 1e-3
START = os.path.join(NSDIR, "starts", "S5x2_N8_U8_tp-0.25_pg11_che_seed0_theta.npy")
B = 16
WIENER = False  # --wiener: rho^2/(rho_c^2 + rho^2) shrinkage (translation repair, RULE_snrcut.md addendum)


def draws(ctx, pr, N, seed, sampler):
    p = ctx.size * pr * pr
    p = p / p.sum()
    if sampler == "guided":
        pf = ctx.size * np.abs(pr)
        q = 0.8 * p + 0.2 * pf / pf.sum()
    else:
        q = p
    rg = np.random.default_rng(seed).choice(len(q), size=N, p=q)
    return rg, p, q


def rows(ctx, th, phi0, pr, N, seed, sampler):
    rg, p, q = draws(ctx, pr, N, seed, sampler)
    uq, inv = np.unique(rg, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(uq)).astype(float)
    r = p[uq] / q[uq]
    Z = (cnt * r).sum()
    w = cnt * r / Z
    O = T.orows(ctx, th, phi0, pr, uq)
    elg = (ctx.Hr @ pr)[uq] / pr[uq]
    Oc = O - w @ O
    ec = elg - w @ elg
    return uq, inv, r, Z, w, Oc, ec


def snr_step(ctx, th, phi0, pr, N, seed, sampler, tau, rho_c):
    uq, inv, r, Z, w, Oc, ec = rows(ctx, th, phi0, pr, N, seed, sampler)
    ST = (Oc * w[:, None]).T @ Oc
    f = (Oc * w[:, None]).T @ ec
    ST = 0.5 * (ST + ST.T)
    wv, U = np.linalg.eigh(ST)
    if rho_c <= 0:
        return T.d_shift(wv, U, f, tau)[0], len(uq), len(wv)
    bidx = np.repeat(np.arange(B), N // B)
    cb = np.zeros((B, len(uq)))
    np.add.at(cb, (bidx, inv), 1.0)
    Fb = (Oc.T @ (cb * (r * ec)).T) * (B / Z)  # P x B; mean over b equals f
    fk = U.T @ f
    sd = (U.T @ Fb).std(1, ddof=1) / np.sqrt(B)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(sd > 0, np.abs(fk) / sd, 0.0)
        filt = np.where(rho > 0, 1.0 / (1.0 + (rho_c / rho) ** (2 if WIENER else 6)), 0.0)
    xk = fk / (wv + SHIFT) * filt
    ml = np.sqrt(max(float(np.sum(wv * xk * xk)), 1e-300))
    return -tau * (U @ xk) / ml, len(uq), int((filt > 0.5).sum())


def minsr_step(ctx, th, phi0, pr, N, seed, sampler, tau):
    uq, inv, r, Z, w, Oc, ec = rows(ctx, th, phi0, pr, N, seed, sampler)
    sw = np.sqrt(w)
    A = Oc * sw[:, None]
    x = A.T @ np.linalg.solve(A @ A.T + SHIFT * np.eye(len(uq)), sw * ec)
    return -tau * x / float(np.linalg.norm(A @ x)), len(uq), len(uq)


def run(a):
    ctx = NS.GCtx(5, 2, 8, START)
    E00, pr, phi0 = ctx.energy(ctx.th0)
    th = ctx.th0.clone()
    tag = (f"snr{'W' if WIENER else ''}_{a.sampler}_N{a.nsamp}_rho{a.rho}_tau{a.tau}_s{a.seed0}" if a.mode == "snr"
           else f"minsr_{a.sampler}_N{a.nsamp}_tau{a.tau}_s{a.seed0}")
    res = os.path.join(HERE, "results", tag)
    os.makedirs(res, exist_ok=True)
    hist, c00 = [], time.process_time()
    nsteps = a.steps if a.mode == "snr" else a.max_draws // a.nsamp
    for n in range(1, nsteps + 1):
        if a.mode == "snr":
            d, nu, kept = snr_step(ctx, th, phi0, pr, a.nsamp, a.seed0 + n, a.sampler, a.tau, a.rho)
        else:
            d, nu, kept = minsr_step(ctx, th, phi0, pr, a.nsamp, a.seed0 + n, a.sampler, a.tau)
        th = th + torch.tensor(d)
        E, pr, phi0 = ctx.energy(th)
        hist.append(dict(E=E, n_unique=nu, kept=kept))
        if n % 20 == 0 or n == nsteps:
            NS.log(f"{tag} step {n}: E {E:.7f} (E-E_start {E - E00:+.3e}) unique {nu} kept {kept} "
                   f"[{(time.process_time() - c00) / 60:.2f} CPU-min]")
            json.dump(dict(tag=tag, E_start=E00, nsamp=a.nsamp, tau=a.tau, rho=a.rho, seed0=a.seed0, steps=hist,
                           status="DONE" if n == nsteps else "RUNNING", cpu_min=(time.process_time() - c00) / 60),
                      open(os.path.join(res, "run.json"), "w"), indent=1)


def check():
    """3 steps: snr with rho_c = 0, a vanishing rho_c, and minsr must reproduce ns_run's shifted guided step."""
    ctx = NS.GCtx(5, 2, 8, START)
    th = ctx.th0.clone()
    E, pr, phi0 = ctx.energy(th)
    for n in range(1, 4):
        ST, f, nu = NS.stats(ctx, th, phi0, pr, 4096, 9000 + n, "guided", False)
        wv, U = np.linalg.eigh(ST)
        d0 = T.d_shift(wv, U, f, 0.005)[0]
        d1 = snr_step(ctx, th, phi0, pr, 4096, 9000 + n, "guided", 0.005, 0.0)[0]
        d2 = minsr_step(ctx, th, phi0, pr, 4096, 9000 + n, "guided", 0.005)[0]
        d3 = snr_step(ctx, th, phi0, pr, 4096, 9000 + n, "guided", 0.005, 1e-9)[0]

        def rel(x):
            return float(np.linalg.norm(x - d0) / np.linalg.norm(d0))
        NS.log(f"check step {n}: snr0 {rel(d1):.2e}, minsr {rel(d2):.2e}, snr(rho_c 1e-9) {rel(d3):.2e}")
        assert rel(d1) < 1e-12 and rel(d2) < 1e-8 and rel(d3) < 1e-6
        th = th + torch.tensor(d0)
        E, pr, phi0 = ctx.energy(th)
    ref = json.load(open(os.path.join(NSDIR, "results", "S5x2_N8_guided_N4096_tau0.005_s9000", "run.json")))
    ref = ref["steps"][2]["E"]
    NS.log(f"check: E(step 3) {E:.12f} vs R51.8 arm {ref:.12f}")
    assert abs(E - ref) < 1e-10


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", choices=["snr", "minsr", "check"], required=True)
    pa.add_argument("--nsamp", type=int, default=4096)
    pa.add_argument("--rho", type=float, default=1.0)
    pa.add_argument("--tau", type=float, default=0.005)
    pa.add_argument("--steps", type=int, default=120)
    pa.add_argument("--max-draws", type=int, default=819200)
    pa.add_argument("--sampler", default=None, choices=["guided", "born"])
    pa.add_argument("--seed0", type=int, default=9000)
    pa.add_argument("--wiener", action="store_true")
    a = pa.parse_args()
    WIENER = a.wiener
    if a.sampler is None:
        a.sampler = "born" if a.mode == "minsr" else "guided"
    check() if a.mode == "check" else run(a)
