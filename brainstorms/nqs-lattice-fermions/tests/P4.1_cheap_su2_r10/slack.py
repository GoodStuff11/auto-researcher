#!/usr/bin/env python3
"""
P4.1 Round 10, premise L (optimisation slack) at T1: exact full-tangent linear-method step on the trained 12-site che.
=================================================================================================================

Run with the lit_env interpreter from this directory:
    ~/.conda/envs/lit_env/bin/python slack.py --mode MODE [--threads N]

Question
--------
Is the VMC-SR-trained 12-site `che` (logs/pilot/pg_che, H = 16, 2637 parameters, 1000 SR steps, eps 2.858e-3) limited by
its optimiser or by its function class?  One exact linear-method (LM) step in the FULL tangent space of the
parameters, span{Psi, dPsi/dtheta_k : k = 1..2637}, gives the lowest exact energy reachable at first order from the
trained point.  The fraction L = Delta_full / g (g = E(Psi) - E0) is a LOWER bound on the optimisation slack: a
nonlinear optimum of the same ansatz can only be lower.  Every sum is exact over all 627,264 configurations.

Per-parameter log-derivatives O_k(x) = d log phi0(x) / d theta_k of the unprojected (P_S-included) amplitude are
assembled from the orbital gradient Gamma(x) = d log phi0 / d out (tangent.py), the hidden layer h(x) and the
features F(x):  P0_o, b2_o: Gamma_o;  jv_k: F_k;  W2[k,o]: h_k Gamma_o;  b1_k: (1 - h_k^2) sum_o W2[k,o] Gamma_o;
W1[i,k]: x_i (1 - h_k^2) sum_o W2[k,o] Gamma_o.  The assembly is checked against torch autograd on 32 configurations.
Tangents are then P_G-projected by index maps (tangent.pg_maps).

Pre-registered reading (written before any 'run' output existed)
  L = Delta_full / g, reported with the sub-blocks (jv; P0 + b2; W2; W1 + b1).
  * L >= 0.30: premise L HOLDS at 12 sites: the trained che is optimiser-limited by at least 30% of its gap; a route
    that improves the optimiser for the exact-S family (not its class) is the next T0 target, and R15d's T0 kill
    (cost per LM step) is re-priced against the number of steps saved.
  * L <= 0.10: premise L FAILS: the trained point is within 10% of its first-order optimum; the 12-site plateau is
    class-limited (frame or backflow), and optimiser routes are deprioritised.
  * otherwise PARTIAL (reported; no promotion).
  INVALID if the autograd check exceeds 1e-8 relative, E(Psi) differs from the stored exact energy by > 1e-8, or the
  positive control (adding the exact ground state) misses g by > 1e-8.
  Amendment (before any LM output existed; the first run was killed by the 20 GB job memory limit while building
  the tangent matrix): the 2639 tangent columns are stored in float32 and every product is formed in float64, so
  the positive-control tolerance becomes 1e-4 g (float32 rounding); the autograd check stays at 1e-8, now on
  float64 orbital gradients (the killed run printed 5.2e-8 because it used the float32 storage of tangent.py).

R18 T1 (--step; pre-registered before any --step output existed)
  Route R18 `blm` (block linear method for the exact-S families) needs the first-order slack to be reachable by a
  REAL parameter update.  With --step, the lowest LM eigenvector of the tangent block (all columns except the
  exact-ground-state control) is mapped back to a parameter update dtheta (theta + s dtheta, s in {1, 0.5, 0.25}),
  and each updated state's exact energy is computed.  R = (E(theta) - min_s E) / g:
    PASS if R >= 0.10 (one real update removes >= 10% of the gap that 1000 SR steps left);
    KILL if R < 0.03 (the tangent gain does not survive the nonlinearity);  AMBIGUOUS otherwise.
  Run on the dry-run block (--w2-units 1, 838 directions) under the T1 cap; the block is a pre-registered part of the
  route (first-layer, constant-orbital, Jastrow and one output unit), not a reading of premise L.
  Result of the unshifted step: KILL (|dtheta| = 17.7; the energy rises; logs/r18_t1_unshifted_output.txt).
  R18' (--step --shift; pre-registered after that KILL and before any --shift output): the standard stabilisation of
  the linear method, a level shift lambda on the complement of Psi, with lambda = 1e-3 * 2^j for the smallest j giving
  |dtheta| <= 0.5 (the SR protocol's own update cap), then s in {1, 0.5}.  Same PASS (R >= 0.10) and KILL (R < 0.03).

Modes
  run     compute and write results/slack_che.json (and print the reading).
Options
  --mode {run} (required)   --threads INT (4)   --chunk INT (4096)
  --step                    also apply the R18 T1 parameter-step test (see above)
  --shift                   with --step: the level-shifted R18' variant
  --w2-units INT (all 16)   dry run only: keep the W2 tangents of the first INT hidden units (result written to
                            results/slack_che_dryrun.json and labelled DRY-RUN; never a premise reading)
"""
import argparse
import json
import os
import sys
import time

THR = "4"
for i, s in enumerate(sys.argv):
    if s == "--threads" and i + 1 < len(sys.argv):
        THR = sys.argv[i + 1]
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = THR

import numpy as np  # noqa: E402
import torch  # noqa: E402

import tangent as TG  # noqa: E402

V = TG.V
log = V.log
torch.set_num_threads(int(THR))


def run(a):
    t0 = time.time()
    m = TG.build("che")
    th = torch.tensor(np.load(TG.PAR["che"] + "_theta.npy"))
    stored = json.load(open(TG.PAR["che"] + ".json"))["E"]
    sec = V.exact_sector(m)
    nU, nD = V.exact_configs(sec)
    l, s, G = TG.phi0_and_grad(m, th, nU, nD, "che", a.chunk, keep64=True)
    p = m.unpack(th)
    F, x = m.feats(nU, nD)
    h = torch.tanh(x @ p["W1"] + p["b1"])
    W2 = p["W2"]  # (H, nout)
    # autograd check of the assembled log-derivatives on 32 configurations (unprojected amplitude)
    rng = np.random.default_rng(1)
    sel = torch.tensor(rng.choice(sec.dim, 32, replace=False))
    err = 0.0
    for j in sel.tolist():
        tt = th.clone().requires_grad_(True)
        lj, _ = m.logpsi0(tt, nU[j:j + 1], nD[j:j + 1])
        (gj,) = torch.autograd.grad(lj.sum(), tt)
        Gj, hj, xj, Fj = G[j], h[j], x[j], F[j]
        back = (W2 @ Gj) * (1 - hj ** 2)  # (H,)
        parts = {"P0": Gj, "jv": Fj, "W1": torch.outer(xj, back).ravel(), "b1": back,
                 "W2": torch.outer(hj, Gj).ravel(), "b2": Gj}
        asm = torch.cat([parts[n] for n, _ in m.shapes])
        err = max(err, float((asm - gj).abs().max() / gj.abs().max()))
    log(f"assembly vs autograd: max relative error {err:.2e} ({time.time() - t0:.0f} s)")
    lmax = float(l.max())
    phi0 = (s * torch.exp(l - lmax)).numpy()
    maps = TG.pg_maps(m, sec)
    Psi = TG.project(maps, phi0)
    nrm = np.sqrt(np.sum(Psi * Psi))
    Psi, phi0 = Psi / nrm, phi0 / nrm
    EPsi = float(np.sum(Psi * sec.H(Psi.reshape(sec.shape)).ravel()))
    gap = EPsi - TG.E0
    log(f"E(Psi) {EPsi:.10f} (stored {stored:.10f}); gap {gap:.6e}")
    # columns in theta order
    back = (G @ W2.T) * (1 - h ** 2)  # (dim, H)
    blocks = {}
    nw2 = a.w2_units if a.w2_units is not None else m.H
    ncol = m.nparam - (m.H - nw2) * m.nout + 2
    X = np.empty((sec.dim, ncol), dtype=np.float32)  # float32 storage (20 GB job limit); products in float64
    X[:, 0] = Psi
    k = 1

    off = dict(zip([n for n, _ in m.shapes], np.cumsum([0] + m.sizes[:-1]).tolist()))
    pmap = []  # parameter index of each tangent column 1..

    def add(name, arr2d, pidx):
        nonlocal k
        pmap.extend(pidx)
        blocks.setdefault(name, []).extend(range(k, k + arr2d.shape[1]))
        for c in range(arr2d.shape[1]):
            X[:, k] = TG.project(maps, arr2d[:, c].numpy() * phi0)
            k += 1

    for n, _ in m.shapes:
        o0 = off[n]
        if n in ("P0", "b2"):
            add("P0+b2", G, [o0 + o for o in range(m.nout)])
        elif n == "jv":
            add("jv", F, [o0 + q for q in range(m.nf)])
        elif n == "W1":
            for i in range(x.shape[1]):
                add("W1+b1", x[:, i:i + 1] * back, [o0 + i * m.H + q for q in range(m.H)])
        elif n == "b1":
            add("W1+b1", back, [o0 + q for q in range(m.H)])
        elif n == "W2":
            for hk in range(nw2):
                add("W2", h[:, hk:hk + 1] * G, [o0 + hk * m.nout + o for o in range(m.nout)])
    assert k == ncol - 1
    gs = TG.sec_ground(sec, m)
    X[:, k] = gs
    log(f"tangent matrix {X.shape} built ({time.time() - t0:.0f} s)")
    Hs = sec.Hsparse()
    Sg = np.zeros((ncol, ncol))
    Hg = np.zeros((ncol, ncol))
    R = 65536
    for c0 in range(0, ncol, 256):
        cb = slice(c0, min(c0 + 256, ncol))
        Xb = X[:, cb].astype(np.float64)
        HXb = Hs @ Xb
        for r0 in range(0, sec.dim, R):
            Xr = X[r0:r0 + R].astype(np.float64)
            Sg[:, cb] += Xr.T @ Xb[r0:r0 + R]
            Hg[:, cb] += Xr.T @ HXb[r0:r0 + R]
        del Xb, HXb
    Hg = 0.5 * (Hg + Hg.T)
    Sg = 0.5 * (Sg + Sg.T)
    del X
    log(f"Gram and H matrices ({time.time() - t0:.0f} s)")
    res = {}
    gidx = ncol - 1
    for name, idx in [("full", list(range(1, gidx)))] + list(blocks.items()):
        ii = [0] + idx
        d, r = TG.lm_sub(Sg[np.ix_(ii, ii)], Hg[np.ix_(ii, ii)], EPsi)
        res[name] = dict(delta=d, frac=d / gap, rank=r, K=len(ii))
        log(f"Delta({name}) = {d:.6e} = {d / gap:.4f} of gap (rank {r} of {len(ii)})")
    dpos, _ = TG.lm_sub(Sg[np.ix_([0, gidx], [0, gidx])], Hg[np.ix_([0, gidx], [0, gidx])], EPsi)
    Lf = res["full"]["frac"]
    ok = err < 1e-8 and abs(EPsi - stored) < 1e-8 and abs(dpos - gap) < 1e-4 * gap
    reading = "DRY-RUN (subset; not a reading)" if nw2 != m.H else "INVALID" if not ok else ("HOLDS" if Lf >= 0.30 else ("FAILS" if Lf <= 0.10 else "PARTIAL"))
    log(f"positive control |Delta - g| = {abs(dpos - gap):.1e}; PREMISE L: {reading} (L = {Lf:.4f})")
    step = None
    if a.step:
        ii = list(range(0, gidx))
        S_, H_ = Sg[np.ix_(ii, ii)], Hg[np.ix_(ii, ii)]
        w, Uv = np.linalg.eigh(S_)
        keep = w > 1e-11 * w.max()
        Pm = Uv[:, keep] / np.sqrt(w[keep])
        Hp = Pm.T @ H_ @ Pm
        e, Y = np.linalg.eigh(Hp)
        # basis vector of Psi in the orthonormal basis; level shift acts on its orthogonal complement
        q0 = Pm.T @ S_[:, 0]
        q0 = q0 / np.linalg.norm(q0)
        Pc = np.eye(len(q0)) - np.outer(q0, q0)

        def dtheta(lam):
            ee, YY = np.linalg.eigh(Hp + lam * Pc)
            cc = Pm @ YY[:, 0]
            d = np.zeros(m.nparam)
            np.add.at(d, np.array(pmap), cc[1:] / cc[0])
            return d

        if a.shift:
            lam = 1e-3
            dth = dtheta(lam)
            while np.linalg.norm(dth) > 0.5 and lam < 1e6:
                lam *= 2.0
                dth = dtheta(lam)
            log(f"level shift lambda = {lam:.3e} gives |dtheta| = {np.linalg.norm(dth):.3f}")
            scales = (1.0, 0.5)
        else:
            dth = dtheta(0.0)
            scales = (1.0, 0.5, 0.25)
        step = {}
        for sc in scales:
            Es = V.exact_energy(m, th + sc * torch.tensor(dth), sec)[0]
            step[str(sc)] = Es
            log(f"exact LM parameter step x{sc}: E = {Es:.8f} (realised {(EPsi - Es) / gap:.4f} of gap; first-order "
                f"{(EPsi - e[0]) / gap:.4f}; |dtheta| {sc * np.linalg.norm(dth):.3e})")
        Rr = (EPsi - min(step.values())) / gap
        rv = "PASS" if Rr >= 0.10 else ("KILL" if Rr < 0.03 else "AMBIGUOUS")
        log(f"R18{chr(39) if a.shift else ''} T1 VERDICT: {rv} (realised fraction R = {Rr:.4f})")
        step = dict(E=step, R=Rr, verdict=rv, first_order=(EPsi - e[0]) / gap)
    out = dict(step=step, E_Psi=EPsi, gap=gap, eps=gap / abs(TG.E0), autograd_err=err, pos_err=abs(dpos - gap), sets=res,
               reading=reading, wall=time.time() - t0, threads=int(THR))
    out["w2_units"] = nw2
    json.dump(out, open(os.path.join(TG.HERE, "results", "slack_che.json" if nw2 == m.H else "slack_che_dryrun.json"), "w"), indent=1)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", required=True, choices=["run"])
    pa.add_argument("--threads", default="4")
    pa.add_argument("--chunk", type=int, default=4096)
    pa.add_argument("--w2-units", type=int, default=None)
    pa.add_argument("--step", action="store_true")
    pa.add_argument("--shift", action="store_true")
    run(pa.parse_args())


if __name__ == "__main__":
    main()
