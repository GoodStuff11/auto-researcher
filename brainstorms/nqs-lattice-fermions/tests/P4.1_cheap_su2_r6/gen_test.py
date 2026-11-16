#!/usr/bin/env python3
"""
P4.1 follow-up to the round-5 rank ladder (job 508440): generality test of che_kK and the che_rbm extension.
==========================================================================================================

Two pre-committed actions of the leaf artifact ("Rank-Ladder Result (job 508440, 2026-11-14)"):
  A. family k PASSED on 4x3 open, 5+5  ->  run the 4x3 cylinder and a second filling before any promotion.
  B. family rbm AMBIGUOUS              ->  extend che_rbm1..3 and che to 4000 iterations from the saved parameters.
Pre-registered rules: leaf artifact, section "Generality and rbm-extension tests (2026-11-14)".

Local copies only: r5_test.py, fix_test.py, r4_test.py, r3_micro_test.py, r2_micro_test.py are verbatim copies of
tests/P4.1_cheap_su2_r5/. Data READ (not code):
  * cached ED states psi0_<tag>.npz from --psi0-dir (default ../P4.1_cheap_su2_r4/results), else from --outdir;
    a missing state is computed by ED (lowest singlet, fix_test.load_psi0) and cached in --outdir;
  * (B only) the job-508440 rbm runs <tag>_w16_che_rbmM_seed<s>.json and their _theta.npy from --r5-dir
    (default ../P4.1_cheap_su2_r5/results), the job-451886 che / nnbf_SG JSONs from --ref-dir
    (default ../P4.1_cheap_su2_fix/results; used only for the reproduction check) and the per-amplitude cost
    file cost_r5.json from --r5-dir (the round-5 cost gate, which every arm passed).

Instances of A (both 4x3, U = 8, t' = -0.25, width-16 reference arms, exact S = 0 via P_G and exact P_S)
------------------------------------------------------------------------------------------------------
  cyl  : periodic along x (length 4), open along y (length 3), N_up = N_dn = 5, dimension 627264. Periodic along
         the length-4 direction because (i) the ring stays even (bipartite nearest-neighbour hopping, Neel not
         frustrated, unlike a length-3 ring), and (ii) the 5+5 tight-binding Fermi sea is closed-shell there
         (gap 1.17), whereas periodic along y gives an open shell (gap 0) and no common reference start.
  fill : open 4x3, N_up = N_dn = 4 (filling 2/3), dimension 245025. Chosen over 6+6 because the round-3 pilot
         (job 411494, same cluster) measured a resolved che -> nnbf_SG gap there (ratio 2.10, disjoint seed
         ranges), and it is 3.5x smaller than 6+6 (853776).
Arms per instance: che and nnbf_SG (fix_test.build_arm_fix, width 16, P_ref parameters), che_k2, che_k4, che_k8
(r5_test.CheK, common che start, hidden width H_K = the largest H with parameter count <= P_ref), seeds 0-2,
2000 L-BFGS iterations; plus che_drift4000 (che, seed 0, 4000 iterations).

Extension B (instance: 4x3 open, 5+5, as job 508440)
----------------------------------------------------
  che_rbm1..3 seeds 0-2: load the saved best parameters of job 508440 (iteration 2000), re-evaluate E (must match
  the stored E to 1e-9 relative, else abort), continue L-BFGS from iteration 2000 to 4000. The L-BFGS curvature
  history is NOT saved, so it restarts empty at iteration 2000 (one restart).
  che, nnbf_SG seeds 0-2: job 451886 did not save parameters, so the run restarts from the common reference
  start (deterministic: che_drift4000 seed 0 reproduced 451886's eps(2000) = 3.43e-4), runs 2000 iterations,
  then restarts L-BFGS with an empty history and continues to 4000. Every extended arm therefore has exactly one
  L-BFGS restart, at iteration 2000. eps(2000) is compared with the 451886 JSON (reproduction check, printed).
  nnbf_SG is extended too, so that the gap fractions at 4000 use a reference at the same budget (stricter than
  reusing nnbf_SG(2000)).

Usage
-----
  python gen_test.py --mode MODE [options]

--mode {prep, smoke, run, extend, report, report-ext, check}   (required)
  prep       : one instance (--instance). Builds the sector; checks the tight-binding Fermi-sea gap (closed shell
               required for the common start); loads or computes the lowest-singlet ED state psi_0 (cached in
               --outdir); runs a second, UNPENALISED Lanczos on H in the whole S_z = 0 space to check that the
               true ground state is that singlet (lowest level, its <S^2>, and the gap to the next level); prints
               the space-group irrep, the reference count P_ref and the hidden widths H_K of che_k2/4/8; with
               --prep-timing, also one energy + gradient of che, nnbf_SG and che_k8 (sizing). Writes
               <outdir>/prep_<instance>.json.
  smoke      : <= 2 min code check on small clusters: (a) a 4x2 cylinder (pbc x / obc y, N_e = 10; N_e = 6 is open-shell there) and a 4x2
               open cluster at N_e = 4: che, nnbf_SG, che_k2, che_k4, che_k8 at small widths from the common
               start: <S^2> = 0, analytic vs finite-difference gradient on every parameter block, E_init of every
               k arm within 5% of che; (b) on the 4x2 cylinder: polynomial per-amplitude projection (K closed-form
               Che evaluations) vs full-vector P_S, to 1e-10; (c) segmented optimiser: 0 -> 60 with a restart at
               30 equals (0 -> 30, save theta to .npy, load, 30 -> 60) to 1e-12 with iteration labels continuing
               at 40, 50, 60; (d) every saved job-508440 rbm theta has the length and block shapes of its
               production arm; (e) production parameter counts of both instances.
  run        : train one arm on one instance (A). --instance {cyl, fill}; --ansatz {che, nnbf_SG, che_k2, che_k4,
               che_k8}; --seed; --maxiter (2000); --drift (only with --ansatz che: label che_drift<maxiter>,
               use --maxiter 4000). Writes <outdir>/<tag>_w16_<label>_seed<seed>.json (E_best trajectory every 50
               iterations, eps, <S^2>, capped, ...) and <...>_theta.npy. Skips if the JSON exists (--overwrite).
  extend     : extension B for one arm. --ansatz {che_rbm1, che_rbm2, che_rbm3, che, nnbf_SG}; --seed;
               --maxiter (4000, the final iteration). Writes <outdir>/<tag>_w16_<ansatz>_ext<maxiter>_seed<s>.json
               (trajectory from iteration 50, the job-508440 part included for rbm) and _theta.npy.
  report     : pre-registered verdict of A: per-instance premise and qualification, then the promotion rule.
  report-ext : pre-registered verdict of B: the rbm family rule evaluated at iteration 4000 (check at 3000).
  check      : runs both report functions on synthetic result sets, one per verdict path, and prints the
               verdict lines. No options used.

Options
  --instance {cyl, fill} (cyl)     cyl = 4x3 pbc-x/obc-y 5+5; fill = 4x3 open 4+4 (prep, run, report uses both)
  --ansatz (che)                   see run / extend
  --seed (0)  --maxiter (2000 for run; extend needs 4000)  --drift (flag)
  --max-hours (72)                 wall-clock cap of the whole optimisation; capping is recorded in the JSON
  --chunk (16384)                  configurations per batch in forward / vjp
  --prep-timing (flag)             prep: also time one energy + gradient of che, nnbf_SG, che_k8
  --outdir (<script dir>/results)  --psi0-dir (../P4.1_cheap_su2_r4/results)
  --r5-dir (../P4.1_cheap_su2_r5/results)  --ref-dir (../P4.1_cheap_su2_fix/results)
  --overwrite (flag)               rerun even if the output JSON exists
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r5_test as F5  # noqa: E402  (local verbatim copy)

F, T, M3, R = F5.F, F5.T, F5.M3, F5.R
log = R.log

U_, TP_, WIDTH = 8.0, -0.25, 16
INSTANCES = {"cyl": dict(Lx=4, Ly=3, Ne=10, bc_x="pbc", bc_y="obc"),
             "fill": dict(Lx=4, Ly=3, Ne=8, bc_x="obc", bc_y="obc")}
BASE = dict(Lx=4, Ly=3, Ne=10, bc_x="obc", bc_y="obc")      # job-508440 instance (extension B)
GEN_REF = ["che", "nnbf_SG"]
K_ARMS = ["che_k2", "che_k4", "che_k8"]
DRIFT = "che_drift4000"
RBM_ARMS = ["che_rbm1", "che_rbm2", "che_rbm3"]
EXT_ARMS = RBM_ARMS + GEN_REF

# pre-registered thresholds (leaf artifact, "Generality and rbm-extension tests (2026-11-14)")
N_FINAL, N_CHECK, N_DRIFT = 2000, 1000, 4000            # A: as the round-5 rule
N_EXT, N_EXT_CHECK, N_PREV = 4000, 3000, 2000           # B: verdict at 4000, check at 3000, drift 2000 -> 4000
PASS_GAIN, FAIL_GAIN, STAB_TOL = F5.PASS_GAIN, F5.FAIL_GAIN, F5.STAB_TOL   # 0.5, 0.2, 0.2
GAP_RATIO = 1.5                                         # P_gap: m_che >= 1.5 m_SG and lo_che > hi_SG
S2_TOL = 1e-10


# ============================================================================ instances and arms
def inst_ns(a, spec):
    """namespace with the fields fix_test.tag_of / load_psi0 read."""
    return argparse.Namespace(**spec, U=U_, tp=TP_, tag=None, psi0_dir=a.psi0_dir, outdir=a.outdir)


def tag(spec):
    return F.tag_of(argparse.Namespace(**spec, U=U_, tp=TP_, tag=None))


def counts(L, Nu, Nd):
    """P_ref (che = nnbf_SG at width 16) and H_K = largest H with CheK count <= P_ref."""
    nf = L * (L - 1) // 2 + 11                       # R.charge_features: n_i n_j (i<j), sum D, 10 d-h counts
    pref = R.NNBF.count(L, nf, Nu, Nd, WIDTH)
    hk = {}
    for k in K_ARMS:
        K = F5.ARM_SPEC[k]["K"]
        hk[k] = max(H for H in range(1, 200) if F5.CheK.count(L, nf, Nu, Nd, H, K) <= pref)
    return pref, hk, {k: F5.CheK.count(L, nf, Nu, Nd, hk[k], F5.ARM_SPEC[k]["K"]) for k in K_ARMS}


def build(sec, kind, seed, chunk, H=None):
    """che / nnbf_SG: fix_test.build_arm_fix (width 16); che_kK: r5_test.build_r5 at width H (common start)."""
    if kind in GEN_REF:
        return F5.build_r5(sec, kind, seed, chunk=chunk, width=WIDTH)
    return F5.build_r5(sec, kind, seed, chunk=chunk, H=H)


def setup(a, spec):
    sec = M3.make_sector(spec["Lx"], spec["Ly"], spec["Ne"], U_, TP_, spec["bc_y"], spec["bc_x"])
    sg = M3.SG(sec)
    P0, E0 = F.load_psi0(inst_ns(a, spec), sec, sg)
    return sec, sg, P0, E0


# ============================================================================ segmented L-BFGS
def lbfgs(st, th0, it0, n, max_s, t0, logf, every):
    """R.optimise with an iteration offset it0 and a shared wall clock; same L-BFGS-B options."""
    best = {"E": np.inf, "th": th0.copy()}
    it = [it0]

    def f(th):
        E, g, _ = st.energy_grad(th)
        if not np.isfinite(E):
            return 1e3, np.zeros_like(th)
        if E < best["E"]:
            best["E"], best["th"] = E, th.copy()
        return E, g

    def cb(th):
        it[0] += 1
        if it[0] % every == 0:
            logf(f"  iter {it[0]:5d}  E_best = {best['E']:.10f}  ({(time.time() - t0) / 60:.1f} min)")
        if time.time() - t0 > max_s:
            raise StopIteration
    capped = False
    try:
        minimize(f, th0, jac=True, method="L-BFGS-B", callback=cb,
                 options=dict(maxiter=n, maxcor=30, ftol=1e-15, gtol=1e-10))
    except StopIteration:
        capped = True
        logf("  wall-clock cap reached")
    return best["th"], best["E"], it[0], capped


def optimise_seg(st, th0, it0, maxiter, restart_at, max_hours, logf, every=50):
    """L-BFGS from th0 (iteration counter it0) to maxiter; the history restarts empty at restart_at."""
    t0 = time.time()
    bounds = ([restart_at] if restart_at and it0 < restart_at < maxiter else []) + [maxiter]
    th, E, it, capped = th0, np.inf, it0, False
    for b in bounds:
        if b - it <= 0:
            continue
        th, E, it, capped = lbfgs(st, th, it, b - it, 3600 * max_hours, t0, logf, every)
        if capped:
            break
        if b != maxiter:
            logf(f"  L-BFGS history restarted at iteration {it}")
    return th, E, it, time.time() - t0, capped


def traj_logger(traj):
    import re

    def logf(msg):
        log(msg)
        mt = re.search(r"iter\s+(\d+)\s+E_best = (\S+)", msg)
        if mt:
            traj.append([int(mt.group(1)), float(mt.group(2))])
    return logf


def save_run(fn, sec, st, arm, th, P0, E0, base):
    phi, _ = arm.forward(th, st.chunk)
    psi = st.psi(phi)
    nrm = np.sum(psi * psi)
    out = dict(base, P=arm.nparam, H=arm.H, E0=E0, eps=(base["E"] - E0) / abs(E0),
               S2=float(np.sum(psi * sec.S2(psi)) / nrm),
               overlap2=float(np.sum(psi * P0) ** 2 / (nrm * np.sum(P0 * P0))),
               shapes=[[k, list(s)] for k, s in arm.shapes])
    if hasattr(arm, "terms"):
        p = arm.unpack(th)
        tv = arm.terms(th, st.chunk)
        out.update(ak=p["ak"].tolist(),
                   term_norm_ratio=[float(np.linalg.norm(tv[k]) / np.linalg.norm(tv[0])) for k in range(1, arm.K)])
    np.save(fn.replace(".json", "_theta.npy"), th)
    log(json.dumps({k: v for k, v in out.items() if k not in ("traj", "shapes")}))
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=1)


# ============================================================================ prep
def prep(a):
    spec = INSTANCES[a.instance]
    t0 = time.time()
    sec = M3.make_sector(spec["Lx"], spec["Ly"], spec["Ne"], U_, TP_, spec["bc_y"], spec["bc_x"])
    sg = M3.SG(sec)
    _, gap = F.fermi_sea(sec)                          # raises if open-shell
    P0, E0 = F.load_psi0(inst_ns(a, spec), sec, sg)
    s2_0 = float(np.sum(P0 * sec.S2(P0)) / np.sum(P0 * P0))
    shp = sec.shape
    op = R.LinearOperator((sec.dim, sec.dim), dtype=float, matvec=lambda x: sec.H(x.reshape(shp)).ravel())
    w, v = R.eigsh(op, k=3, which="SA", v0=np.random.default_rng(2).standard_normal(sec.dim), tol=1e-10)
    o = np.argsort(w)
    lev = []
    for kk in o:
        Pk = v[:, kk].reshape(shp)
        lev.append(dict(E=float(w[kk]), S2=float(np.sum(Pk * sec.S2(Pk)) / np.sum(Pk * Pk))))
    gs_singlet = abs(lev[0]["E"] - E0) < 1e-7 and abs(lev[0]["S2"]) < 1e-6
    L = sec.lat.L
    pref, hk, ck = counts(L, sec.Nu, sec.Nd)
    out = dict(instance=a.instance, tag=tag(spec), dim=sec.dim, fermi_gap=gap, E0_singlet=E0, S2_psi0=s2_0,
               irrep=sg.lam, unpenalised_levels=lev, ground_state_is_singlet=bool(gs_singlet), P_ref=pref,
               H_K=hk, P_K=ck, P_K_dev={k: ck[k] / pref - 1 for k in K_ARMS})
    log(f"prep {a.instance} ({out['tag']}): dim={sec.dim} Fermi gap={gap:.4f} E0(singlet)={E0:.10f} "
        f"<S^2>(psi0)={s2_0:+.1e} irrep={sg.lam}")
    log("  unpenalised lowest levels (whole S_z=0 space): " +
        ", ".join(f"E={x['E']:.10f} <S^2>={x['S2']:.4f}" for x in lev) +
        f" -> ground state is the singlet: {gs_singlet}")
    log(f"  P_ref (che = nnbf_SG, width 16) = {pref}; " +
        ", ".join(f"{k}: H={hk[k]} P={ck[k]} ({100 * (ck[k] / pref - 1):+.2f}%)" for k in K_ARMS))
    if a.prep_timing:
        tm = {}
        for kind in ["che", "nnbf_SG", "che_k8"]:
            arm = build(sec, kind, 0, a.chunk, hk.get(kind))
            st = M3.State(sec, arm, True, sg, a.chunk)
            assert arm.nparam == (pref if kind in GEN_REF else ck[kind]), (kind, arm.nparam)
            t = time.time()
            st.energy_grad(arm.theta0)
            tm[kind] = time.time() - t
            log(f"  timing {kind}: P={arm.nparam} one energy+gradient {tm[kind]:.1f}s "
                f"(threads {os.environ.get('OMP_NUM_THREADS')})")
        out["timing_sec"] = tm
    out["prep_seconds"] = time.time() - t0
    with open(os.path.join(a.outdir, f"prep_{a.instance}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


# ============================================================================ smoke
SMOKE_H = {"che": None, "nnbf_SG": None, "che_k2": 4, "che_k4": 3, "che_k8": 2}


def smoke(a):
    t0 = time.time()
    ok = True
    small = {"cyl4x2": (4, 2, 10, "pbc", "obc"), "open4x2_Ne4": (4, 2, 4, "obc", "obc")}
    for name, (Lx, Ly, Ne, bx, by) in small.items():
        sec = M3.make_sector(Lx, Ly, Ne, 6.0, -0.3, by, bx)
        sg = M3.SG(sec)
        _, gap = F.fermi_sea(sec)
        _, E0, *_ = M3.ground_state(sec, sg)
        log(f"(a) {name}: {Lx}x{Ly} {bx}/{by} Ne={Ne} dim={sec.dim} Fermi gap={gap:.3f} E0={E0:.6f} irrep={sg.lam}")
        Eche = None
        for kind in GEN_REF + K_ARMS:
            if kind in GEN_REF:
                arm = F5.build_r5(sec, kind, 0, chunk=4096, width=4)
            else:
                arm = F5.build_r5(sec, kind, 0, chunk=4096, H=SMOKE_H[kind])
            st = M3.State(sec, arm, True, sg, 4096)
            phi, _ = arm.forward(arm.theta0, 4096)
            Ei, _ = F.energy_of(sec, st.psi(phi))
            th = arm.theta0 + 0.1 * np.random.default_rng(3).standard_normal(arm.nparam)
            phi, _ = arm.forward(th, 4096)
            _, s2 = F.energy_of(sec, st.psi(phi))
            rel = F5.fd_blocks(st, arm, th)
            if kind == "che":
                Eche = Ei
            good = abs(s2) < S2_TOL and rel < 1e-5 and (kind == "nnbf_SG" or abs(Ei - Eche) < 0.05 * abs(Eche))
            ok &= good
            log(f"  {kind:8s} P={arm.nparam:4d} E_init={Ei:.6f} (che {Eche:.6f}) <S^2>={s2:+.1e} "
                f"grad rel.err (all blocks)={rel:.1e} [{'PASS' if good else 'FAIL'}]")
    # (b) polynomial per-amplitude projection vs full-vector P_S on the small cylinder
    s2c = M3.make_sector(4, 2, 10, 6.0, -0.3, "obc", "pbc")
    PS = s2c.projS(0)
    L, Nu, Nd = s2c.lat.L, s2c.Nu, s2c.Nd
    for kind in K_ARMS:
        arm = F5.build_r5(s2c, kind, 1, chunk=4096, H=SMOKE_H[kind])
        p = arm.unpack(arm.theta0 + 0.2 * np.random.default_rng(5).standard_normal(arm.nparam))
        p["jv"][:] = 0
        phi, _ = arm.forward(arm.pack(p), 4096)
        ref = PS(phi).ravel()
        e1 = 0.0
        for i in range(0, s2c.dim, max(1, s2c.dim // 60)):
            nU = s2c.up.bits[s2c.iu[i]].astype(int)
            nD = s2c.dn.bits[s2c.id[i]].astype(int)
            e1 = max(e1, abs(F5.single(kind, p, nU, nD, L, Nu, Nd) - ref[i]))
        scale = np.abs(ref).max()
        good = e1 < 1e-10 * max(1, scale)
        ok &= good
        log(f"(b) cyl4x2 {kind:8s}: |polynomial single - full-vector P_S| {e1:.1e} (max|amp| {scale:.2e}) "
            f"[{'PASS' if good else 'FAIL'}]")
    # (c) segmented optimiser = save / load / resume, iteration labels continue
    sec = M3.make_sector(4, 2, 6, 6.0, -0.3, "obc", "obc")
    sg = M3.SG(sec)
    M3.ground_state(sec, sg)
    arm = F5.build_r5(sec, "che_k2", 0, chunk=4096, H=4)
    st = M3.State(sec, arm, True, sg, 4096)
    tA, tB, tC = [], [], []
    thA, EA, itA, _, _ = optimise_seg(st, arm.theta0, 0, 60, 30, 1, traj_logger(tA), every=10)
    th1, _, it1, _, _ = optimise_seg(st, arm.theta0, 0, 30, None, 1, traj_logger(tB), every=10)
    with tempfile.TemporaryDirectory() as td:
        np.save(os.path.join(td, "x_theta.npy"), th1)
        th1 = np.load(os.path.join(td, "x_theta.npy"))
    thB, EB, itB, _, _ = optimise_seg(st, th1, it1, 60, None, 1, traj_logger(tB), every=10)
    optimise_seg(st, arm.theta0, 0, 60, None, 1, traj_logger(tC), every=10)
    dE = max(abs(x[1] - y[1]) for x, y in zip(tA, tB))
    labels_ok = [x[0] for x in tA] == [x[0] for x in tB] == list(range(10, 61, 10))
    good = labels_ok and dE < 1e-12 and itA == itB == 60 and abs(EA - EB) < 1e-12
    ok &= good
    log(f"(c) restart-at-30 run vs save/load/resume: labels {[x[0] for x in tB]} max|dE_best|={dE:.1e} "
        f"final it {itA}/{itB} [{'PASS' if good else 'FAIL'}]; no-restart run at 60: E_best={tC[-1][1]:.10f} vs "
        f"restarted {tA[-1][1]:.10f} (history restart changes the path, as documented)")
    # (d) saved job-508440 rbm parameters match their production arms
    L, nf, Nu, Nd = 12, 77, 5, 5
    t5 = tag(BASE)
    for kind in RBM_ARMS:
        for s in range(3):
            fn = os.path.join(a.r5_dir, f"{t5}_w{WIDTH}_{kind}_seed{s}.json")
            r = json.load(open(fn))
            th = np.load(fn.replace(".json", "_theta.npy"))
            n_shapes = sum(int(np.prod(sh)) for _, sh in r["shapes"])
            good = len(th) == n_shapes == F5.arm_count(kind, L, nf, Nu, Nd) == r["P"] and r["iters"] == 2000
            ok &= good
            log(f"(d) {kind} seed {s}: len(theta)={len(th)} shapes={n_shapes} count={F5.arm_count(kind, L, nf, Nu, Nd)} "
                f"P={r['P']} iters={r['iters']} [{'PASS' if good else 'FAIL'}]")
    # (e) production counts
    for name, spec in INSTANCES.items():
        pref, hk, ck = counts(spec["Lx"] * spec["Ly"], spec["Ne"] // 2, spec["Ne"] // 2)
        good = all(-0.04 <= ck[k] / pref - 1 <= 0 for k in K_ARMS)
        ok &= good
        log(f"(e) {name} ({tag(spec)}): P_ref={pref}; " + ", ".join(
            f"{k} H={hk[k]} P={ck[k]} ({100 * (ck[k] / pref - 1):+.2f}%)" for k in K_ARMS) +
            f" [{'PASS' if good else 'FAIL'}: every k arm within [-4%, 0] of P_ref]")
    log(f"SMOKE {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.1f}s)")
    return ok


# ============================================================================ run (A) and extend (B)
def run(a):
    spec = INSTANCES[a.instance]
    if a.ansatz not in GEN_REF + K_ARMS:
        raise SystemExit(f"run: --ansatz must be one of {GEN_REF + K_ARMS}")
    if a.drift and a.ansatz != "che":
        raise SystemExit("run: --drift only with --ansatz che")
    os.makedirs(a.outdir, exist_ok=True)
    label = f"che_drift{a.maxiter}" if a.drift else a.ansatz
    fn = os.path.join(a.outdir, f"{tag(spec)}_w{WIDTH}_{label}_seed{a.seed}.json")
    if os.path.exists(fn) and not a.overwrite:
        log(f"exists: {fn}")
        return
    sec, sg, P0, E0 = setup(a, spec)
    pref, hk, ck = counts(sec.lat.L, sec.Nu, sec.Nd)
    arm = build(sec, a.ansatz, a.seed, a.chunk, hk.get(a.ansatz))
    assert arm.nparam == (pref if a.ansatz in GEN_REF else ck[a.ansatz]), (arm.nparam, pref, ck)
    st = M3.State(sec, arm, True, sg, a.chunk)
    phi, _ = arm.forward(arm.theta0, a.chunk)
    E_init, _ = F.energy_of(sec, st.psi(phi))
    log(f"{a.instance} {label} seed {a.seed}: P={arm.nparam} (P_ref {pref}) H={arm.H} E_init={E_init:.8f} "
        f"E0={E0:.8f}")
    traj = []
    th, E, it, sec_, capped = optimise_seg(st, arm.theta0, 0, a.maxiter, None, a.max_hours, traj_logger(traj))
    base = dict(ansatz=label, instance=a.instance, seed=a.seed, E=E, E_init=E_init, iters=it, seconds=sec_,
                capped=bool(capped), rank=1 if a.ansatz in GEN_REF else F5.rank(a.ansatz), traj=traj)
    save_run(fn, sec, st, arm, th, P0, E0, base)


def extend(a):
    if a.ansatz not in EXT_ARMS:
        raise SystemExit(f"extend: --ansatz must be one of {EXT_ARMS}")
    os.makedirs(a.outdir, exist_ok=True)
    t5 = tag(BASE)
    label = f"{a.ansatz}_ext{a.maxiter}"
    fn = os.path.join(a.outdir, f"{t5}_w{WIDTH}_{label}_seed{a.seed}.json")
    if os.path.exists(fn) and not a.overwrite:
        log(f"exists: {fn}")
        return
    sec, sg, P0, E0 = setup(a, BASE)
    traj = []
    extra = {}
    if a.ansatz in RBM_ARMS:
        src = os.path.join(a.r5_dir, f"{t5}_w{WIDTH}_{a.ansatz}_seed{a.seed}.json")
        r = json.load(open(src))
        arm = F5.build_r5(sec, a.ansatz, a.seed, a.chunk)
        if [[k, list(s)] for k, s in arm.shapes] != r["shapes"]:
            raise SystemExit(f"shape mismatch: {arm.shapes} vs {r['shapes']}")
        th0 = np.load(src.replace(".json", "_theta.npy"))
        st = M3.State(sec, arm, True, sg, a.chunk)
        E_chk = st.energy_grad(th0)[0]
        if abs(E_chk - r["E"]) > 1e-9 * abs(r["E"]):
            raise SystemExit(f"re-evaluated E {E_chk:.12f} != stored {r['E']:.12f}")
        log(f"extend {a.ansatz} seed {a.seed}: loaded {src} (it {r['iters']}, E={r['E']:.12f}, re-evaluated "
            f"{E_chk:.12f}); continuing to {a.maxiter} with an empty L-BFGS history")
        traj = [list(x) for x in r["traj"]]
        it0, restart, E_init = r["iters"], None, r["E_init"]
        extra = dict(resumed_from=os.path.relpath(src, a.outdir), E_resume_check=E_chk)
    else:
        arm = build(sec, a.ansatz, a.seed, a.chunk)
        st = M3.State(sec, arm, True, sg, a.chunk)
        th0 = arm.theta0
        phi, _ = arm.forward(th0, a.chunk)
        E_init, _ = F.energy_of(sec, st.psi(phi))
        it0, restart = 0, N_PREV
        log(f"extend {a.ansatz} seed {a.seed}: from the common start to {N_PREV}, then an L-BFGS restart and on to "
            f"{a.maxiter} (job 451886 saved no parameters)")
    th, E, it, sec_, capped = optimise_seg(st, th0, it0, a.maxiter, restart, a.max_hours, traj_logger(traj))
    if a.ansatz in GEN_REF:
        ref = os.path.join(a.ref_dir, f"{t5}_w{WIDTH}_{a.ansatz}_seed{a.seed}.json")
        e2 = dict((int(i), e) for i, e in traj).get(N_PREV)
        if os.path.exists(ref) and e2 is not None:
            r = json.load(open(ref))
            extra.update(repro_eps2000=(e2 - E0) / abs(E0), ref451886_eps2000=r["eps"])
            log(f"  reproduction: eps(2000) = {(e2 - E0) / abs(E0):.6e} vs job 451886 {r['eps']:.6e}")
    base = dict(ansatz=label, seed=a.seed, E=E, E_init=E_init, iters=it, seconds=sec_, capped=bool(capped),
                rank=1 if a.ansatz in GEN_REF else F5.rank(a.ansatz), lbfgs_restart_at=N_PREV, traj=traj, **extra)
    save_run(fn, sec, st, arm, th, P0, E0, base)


# ============================================================================ verdicts (pure functions)
def stats(rs, arms, n):
    e = {k: [F5.eps_at(r, n) for r in rs[k]] for k in arms}
    if any(x is None for v in e.values() for x in v):
        return None
    e = {k: np.array(v) for k, v in e.items()}
    return ({k: float(np.median(v)) for k, v in e.items()}, {k: float(v.min()) for k, v in e.items()},
            {k: float(v.max()) for k, v in e.items()})


def ladder(rs, ref_che, ref_sg, arms, n_fin, n_chk, d_drift):
    """common core: P_run-checked runs -> (status or None, per-n stats, fails(k, n))."""
    allarms = [ref_che, ref_sg] + arms
    n_eq = min(n_fin, min(F5.last_iter(r) for k in allarms for r in rs[k]))
    n_eq -= n_eq % 50
    if n_eq < n_chk:
        return ("INCOMPLETE", f"a capped arm stopped at iteration {n_eq} < {n_chk}"), None, None, n_eq
    st = {}
    for n in sorted({n_eq, n_chk}):
        s = stats(rs, allarms, n)
        if s is None:
            return ("INCOMPLETE", f"trajectory missing iteration {n}"), None, None, n_eq
        m, lo, hi = s
        if not (m[ref_che] >= GAP_RATIO * m[ref_sg] and lo[ref_che] > hi[ref_sg]):
            return ("NO-TARGET", f"P_gap fails at iteration {n}: m_che/m_SG={m[ref_che] / m[ref_sg]:.2f}, "
                                 f"lo_che={lo[ref_che]:.3e}, hi_SG={hi[ref_sg]:.3e}"), None, None, n_eq
        gm = {k: (m[ref_che] - m[k]) / (m[ref_che] - m[ref_sg]) for k in arms}
        gb = {k: (lo[ref_che] - lo[k]) / (lo[ref_che] - lo[ref_sg]) for k in arms}
        st[n] = dict(m=m, lo=lo, hi=hi, gm=gm, gb=gb)

    def fails(k, n):
        s = st[n]
        f = []
        if s["gm"][k] < PASS_GAIN or s["gb"][k] < PASS_GAIN:
            f.append("gap fraction < 0.5")
        if s["hi"][k] >= s["lo"][ref_che]:
            f.append("seed ranges overlap che")
        if s["m"][ref_che] - s["m"][k] <= d_drift:
            f.append(f"gain <= che drift {d_drift:.2e}")
        return f
    return None, st, fails, n_eq


def instance_verdict(rs):
    """A, one instance: (status in INCOMPLETE / NO-TARGET / QUALIFIES / FAILS, detail)."""
    for k in GEN_REF + K_ARMS:
        v = rs.get(k, [])
        if len(v) < 3 or any(abs(r["S2"]) >= S2_TOL for r in v):
            return "INCOMPLETE", f"{k}: {len(v)} seeds or S^2 not exact"
    if not rs.get(DRIFT):
        return "INCOMPLETE", f"{DRIFT} missing"
    che = rs["che"]
    dd = [F5.eps_at(r, N_CHECK) - F5.eps_at(r, N_FINAL) for r in che
          if F5.eps_at(r, N_CHECK) is not None and F5.eps_at(r, N_FINAL) is not None]
    if len(dd) < 3:
        return "INCOMPLETE", "che trajectory lacks iteration 1000 or 2000"
    d_drift = float(np.median(dd))
    bad, st, fails, n_eq = ladder(rs, "che", "nnbf_SG", K_ARMS, N_FINAL, N_CHECK, d_drift)
    if bad:
        return bad
    s = st[n_eq]
    g = [s["gm"][k] for k in K_ARMS]
    best = K_ARMS[int(np.argmax(g))]
    why = [f"{best} at it {n}: " + ", ".join(fails(best, n)) for n in sorted({n_eq, N_CHECK}, reverse=True)
           if fails(best, n)]
    dr = rs[DRIFT][0]
    e_drift = F5.eps_at(dr, N_DRIFT)
    e_drift = dr["eps"] if e_drift is None else e_drift
    if e_drift <= s["m"][best]:
        why.append(f"che_drift4000 eps(4000)={e_drift:.3e} reaches {best} m={s['m'][best]:.3e}")
    stable = all(g[i + 1] >= g[i] - STAB_TOL for i in range(len(g) - 1))
    det = (f"best {best}; g_m by rank " + ", ".join(f"{x:+.2f}" for x in g) + "; g_b by rank " +
           ", ".join(f"{s['gb'][k]:+.2f}" for k in K_ARMS) + f"; it {N_CHECK} g_m " +
           ", ".join(f"{st[N_CHECK]['gm'][k]:+.2f}" for k in K_ARMS) +
           f"; m_che={s['m']['che']:.3e} m_SG={s['m']['nnbf_SG']:.3e} m_best={s['m'][best]:.3e}; "
           f"drift {d_drift:.2e}; che_drift4000={e_drift:.3e}; rank-stable={stable} (reported, not gating)"
           + (f"; compared at equal iteration {n_eq}" if n_eq < N_FINAL else ""))
    return ("QUALIFIES", det) if not why else ("FAILS", "; ".join(why) + " | " + det)


def promotion(vs):
    """A, overall (pre-registered)."""
    if any(s == "INCOMPLETE" for s, _ in vs.values()):
        return "INCOMPLETE"
    valid = [i for i, (s, _) in vs.items() if s != "NO-TARGET"]
    if not valid:
        return "NO-TARGET on both instances (generality untested; P4.1 stays a single-instance narrow PASS)"
    if any(vs[i][0] == "FAILS" for i in valid):
        return "KILL-GENERALITY (fails qualification on premise-valid instance(s): " + ", ".join(
            i for i in valid if vs[i][0] == "FAILS") + ")"
    return "PROMOTE (qualifies on premise-valid instance(s): " + ", ".join(valid) + ")"


def ext_verdict(rs, cost_r5):
    """B: the round-5 rbm family rule at iteration 4000 (check 3000; drift = che median gain 2000 -> 4000)."""
    ref_che, ref_sg = "che_ext4000", "nnbf_SG_ext4000"
    rbm = [f"{k}_ext4000" for k in RBM_ARMS]
    for k in [ref_che, ref_sg] + rbm:
        v = rs.get(k, [])
        if len(v) < 3 or any(abs(r["S2"]) >= S2_TOL for r in v):
            return {"ALL": f"INCOMPLETE ({k}: {len(v)} seeds or S^2 not exact)"}
    dd = [F5.eps_at(r, N_PREV) - F5.eps_at(r, N_EXT) for r in rs[ref_che]
          if F5.eps_at(r, N_PREV) is not None and F5.eps_at(r, N_EXT) is not None]
    if len(dd) < 3:
        return {"ALL": "INCOMPLETE (che_ext trajectory lacks iteration 2000 or 4000)"}
    d_drift = float(np.median(dd))
    bad, st, fails, n_eq = ladder(rs, ref_che, ref_sg, rbm, N_EXT, N_EXT_CHECK, d_drift)
    if bad:
        return {"ALL": f"{bad[0]} ({bad[1]})"}
    s = st[n_eq]
    out = {"drift": f"che_ext median gain 2000 -> 4000 = {d_drift:.2e}"}
    for k in rbm:
        out[k] = (f"m={s['m'][k]:.3e} [{s['lo'][k]:.3e}, {s['hi'][k]:.3e}] g_m={s['gm'][k]:+.2f} g_b={s['gb'][k]:+.2f}"
                  f" | it {N_EXT_CHECK}: g_m={st[N_EXT_CHECK]['gm'][k]:+.2f} g_b={st[N_EXT_CHECK]['gb'][k]:+.2f}")
    g = [s["gm"][k] for k in rbm]
    best = rbm[int(np.argmax(g))]
    stable = all(g[i + 1] >= g[i] - STAB_TOL for i in range(len(g) - 1))

    def costok(k):
        b = k.replace("_ext4000", "")
        return (cost_r5 is not None and cost_r5["ratios"][b] <= F5.COST_FACTOR * F5.rank(b)
                and cost_r5["slopes"][b] <= F5.COST_SLOPE)
    why = []
    if not all(costok(k) for k in rbm):
        v = "FAIL (cost gate)"
    elif max(g) < FAIL_GAIN and max(s["gb"][k] for k in rbm) < FAIL_GAIN:
        v = "FAIL"
    else:
        for n in sorted({n_eq, N_EXT_CHECK}, reverse=True):
            if fails(best, n):
                why.append(f"{best} at it {n}: " + ", ".join(fails(best, n)))
        if not stable:
            why.append("gain drops with rank")
        v = "PASS" if not why else "AMBIGUOUS (" + "; ".join(why) + ")"
    out["FAMILY rbm at 4000"] = (f"{v}; best rank {best}; g_m by rank " + ", ".join(f"{x:+.2f}" for x in g)
                                 + (f"; compared at equal iteration {n_eq}" if n_eq < N_EXT else ""))
    out["consequence"] = {"PASS": "attribution DEFERRED: extend che_k2/4/8 to 4000 the same way before crediting "
                                  "c-varying components",
                          "FAIL": "rbm closed: c-varying row scalings add nothing at 2x budget",
                          "AMBIGUOUS": "rbm closed as unresolved at 2x budget; no further extension"}[v.split(" ")[0]]
    return out


# ============================================================================ report / check
def load(outdir, pattern):
    rs = {}
    for f in glob.glob(os.path.join(outdir, pattern)):
        r = json.load(open(f))
        rs.setdefault(r["ansatz"], {})[r["seed"]] = r
    return {k: list(v.values()) for k, v in rs.items()}


def report(a):
    vs = {}
    for name, spec in INSTANCES.items():
        rs = load(a.outdir, f"{tag(spec)}_w{WIDTH}_*_seed*.json")
        for k in GEN_REF + K_ARMS + [DRIFT]:
            for r in sorted(rs.get(k, []), key=lambda r: r["seed"]):
                e1 = F5.eps_at(r, N_CHECK)
                print(f"{name:4s} {k:14s} seed={r['seed']} P={r['P']} eps={r['eps']:.3e} eps@{N_CHECK}="
                      f"{'n/a' if e1 is None else f'{e1:.3e}'} S2={r['S2']:+.1e} it={r['iters']} capped={r['capped']}")
        vs[name] = instance_verdict(rs)
        print(f"VERDICT {name}: {vs[name][0]} ({vs[name][1]})")
    print(f"VERDICT PROMOTION: {promotion(vs)}")


def report_ext(a):
    rs = load(a.outdir, f"{tag(BASE)}_w{WIDTH}_*_ext{N_EXT}_seed*.json")
    for k in sorted(rs):
        for r in sorted(rs[k], key=lambda r: r["seed"]):
            e = {n: F5.eps_at(r, n) for n in (N_PREV, N_EXT_CHECK, N_EXT)}
            print(f"{k:20s} seed={r['seed']} " + " ".join(f"eps@{n}={'n/a' if v is None else f'{v:.3e}'}"
                                                          for n, v in e.items())
                  + f" S2={r['S2']:+.1e} it={r['iters']} capped={r['capped']}"
                  + (f" repro(2000) {r['repro_eps2000']:.4e} vs 451886 {r['ref451886_eps2000']:.4e}"
                     if "repro_eps2000" in r else ""))
    cf = os.path.join(a.r5_dir, "cost_r5.json")
    for k, v in ext_verdict(rs, json.load(open(cf)) if os.path.exists(cf) else None).items():
        print(f"VERDICT {k}: {v}")


def check(_a):
    E0 = -7.0

    def mk(k, ef, ec, nf=2000, nc=1000, s2=0.0, cap_at=None, seeds=None):
        out = []
        for i, (xf, xc) in enumerate(zip(ef, ec)):
            if seeds is not None and i >= seeds:
                break
            traj = [[n, E0 + abs(E0) * (xc + (xf - xc) * (n - nc) / (nf - nc) if n >= nc else xc + 1e-4)]
                    for n in range(50, nf + 1, 50)]
            r = dict(ansatz=k, seed=i, P=2600, E0=E0, eps=xf, S2=s2, iters=nf, capped=False, traj=traj)
            if cap_at is not None and i == 0:
                r["traj"] = [t for t in traj if t[0] <= cap_at]
                r.update(capped=True, iters=cap_at, eps=(r["traj"][-1][1] - E0) / abs(E0))
            out.append(r)
        return out

    def plus(v, d):
        return [x + d for x in v]
    che2, che1 = [3.43e-4, 4.26e-4, 3.26e-4], [4.07e-4, 5.06e-4, 4.21e-4]
    sg2, sg1 = [2.31e-4, 2.07e-4, 1.89e-4], [2.79e-4, 2.55e-4, 2.30e-4]
    kk = {"che_k2": [2.26e-4, 2.11e-4, 2.32e-4], "che_k4": [1.74e-4, 1.51e-4, 1.85e-4],
          "che_k8": [1.35e-4, 1.12e-4, 1.46e-4]}

    def inst(drift4=3.06e-4, k=None, k1=None, sg=None, che=None, **mod):
        k = dict(kk, **(k or {}))
        d = {"che": mk("che", *(che or (che2, che1))), "nnbf_SG": mk("nnbf_SG", *(sg or (sg2, sg1)))}
        for a_, v in k.items():
            d[a_] = mk(a_, v, (k1 or {}).get(a_, plus(v, 0.6e-4)))
        d[DRIFT] = [dict(ansatz=DRIFT, seed=0, P=2637, E0=E0, eps=drift4, S2=0.0, iters=4000, capped=False,
                         traj=[[1000, E0 + abs(E0) * 4.07e-4], [2000, E0 + abs(E0) * 3.43e-4],
                               [4000, E0 + abs(E0) * drift4]])]
        d.update(mod)
        return d
    good = inst()
    notarget = inst(sg=(plus(che2, -0.5e-4), plus(che1, -0.5e-4)))
    flat = inst(k={a_: plus(che2, -0.1e-4) for a_ in K_ARMS})
    driftreach = inst(drift4=1.2e-4)
    late = inst(k1={a_: plus(che1, 0.0) for a_ in K_ARMS})
    missing = inst(che_k4=mk("che_k4", kk["che_k4"], plus(kk["che_k4"], 0.6e-4), seeds=2))
    s2bad = inst(che_k8=mk("che_k8", kk["che_k8"], plus(kk["che_k8"], 0.6e-4), s2=1e-6))
    capped = inst(che_k8=mk("che_k8", kk["che_k8"], plus(kk["che_k8"], 0.6e-4), cap_at=800))
    capped_ok = inst(che_k8=mk("che_k8", kk["che_k8"], plus(kk["che_k8"], 0.6e-4), cap_at=1500))
    sc = {"both qualify": (good, good), "qualify + NO-TARGET": (good, notarget),
          "both NO-TARGET": (notarget, notarget), "one flat (k = che)": (good, flat),
          "drift run reaches best k": (good, driftreach), "gain absent at 1000": (good, late),
          "missing seed": (good, missing), "S^2 not exact": (s2bad, good), "capped below 1000": (good, capped),
          "capped at 1500 (equal-iteration comparison)": (capped_ok, good), "flat + NO-TARGET": (flat, notarget)}
    for name, (c, f) in sc.items():
        vs = {"cyl": instance_verdict(c), "fill": instance_verdict(f)}
        print(f"[A: {name}] cyl {vs['cyl'][0]} | fill {vs['fill'][0]} -> {promotion(vs)}")
    # B: extension rule at 4000
    cost = dict(ratios={k: 1.1 * F5.rank(k) for k in RBM_ARMS}, slopes={k: 2.3 for k in RBM_ARMS})

    def ext(rbm4, rbm3=None, che4=None, sg4=None, che_2=None, seeds=3):
        che4 = che4 or [3.0e-4, 3.7e-4, 2.9e-4]
        che_2 = che_2 or che2
        d = {"che_ext4000": mk("che_ext4000", che4, che_2, 4000, 2000),
             "nnbf_SG_ext4000": mk("nnbf_SG_ext4000", sg4 or [1.9e-4, 1.8e-4, 1.7e-4], sg2, 4000, 2000)}
        for i, k in enumerate(RBM_ARMS):
            v4 = rbm4[i]
            v3 = (rbm3 or {}).get(i)
            d[f"{k}_ext4000"] = mk(f"{k}_ext4000", v4, v3 or plus(v4, 0.6e-4), 4000, 2000,
                                   seeds=seeds if i == 0 else None)
            if v3:   # check-iteration values given explicitly: rebuild so that eps(3000) = v3
                d[f"{k}_ext4000"] = mk(f"{k}_ext4000", v4, [2 * x3 - x4 for x3, x4 in zip(v3, v4)], 4000, 2000)
        return d
    near = [[3.1e-4, 3.6e-4, 2.95e-4]] * 3
    win = [[2.0e-4, 1.95e-4, 2.05e-4], [1.9e-4, 1.85e-4, 1.95e-4], [1.8e-4, 1.75e-4, 1.85e-4]]
    scb = {"rbm stays at che (g ~ 0) -> FAIL": (ext(near), cost),
           "rbm separates -> PASS": (ext(win), cost),
           "rbm modest gain (g ~ 0.3) -> AMBIGUOUS": (ext([[2.6e-4, 2.55e-4, 2.65e-4]] * 3), cost),
           "PASS at 4000 not at 3000 -> AMBIGUOUS": (ext(win, rbm3={i: [2.9e-4, 2.9e-4, 2.9e-4] for i in range(3)}),
                                                    cost),
           "che closes on SG at 4000 -> NO-TARGET": (ext(win, sg4=[2.6e-4, 2.7e-4, 2.5e-4]), cost),
           "missing seed -> INCOMPLETE": (ext(win, seeds=2), cost),
           "cost gate": (ext(win), dict(cost, ratios=dict(cost["ratios"], che_rbm2=9.0)))}
    for name, (rs, c) in scb.items():
        v = ext_verdict(rs, c)
        key = "FAMILY rbm at 4000" if "FAMILY rbm at 4000" in v else "ALL"
        print(f"[B: {name}] {v[key].split(';')[0]}" + (f" -> {v['consequence']}" if "consequence" in v else ""))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="P4.1 generality / rbm-extension tests (see module docstring)")
    p.add_argument("--mode", required=True, choices=["prep", "smoke", "run", "extend", "report", "report-ext", "check"])
    p.add_argument("--instance", default="cyl", choices=list(INSTANCES))
    p.add_argument("--ansatz", default="che", choices=GEN_REF + K_ARMS + RBM_ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=None)
    p.add_argument("--drift", action="store_true")
    p.add_argument("--max-hours", type=float, default=72.0, dest="max_hours")
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--prep-timing", action="store_true", dest="prep_timing")
    p.add_argument("--outdir", default=os.path.join(HERE, "results"))
    p.add_argument("--psi0-dir", default=os.path.join(HERE, "..", "P4.1_cheap_su2_r4", "results"), dest="psi0_dir")
    p.add_argument("--r5-dir", default=os.path.join(HERE, "..", "P4.1_cheap_su2_r5", "results"), dest="r5_dir")
    p.add_argument("--ref-dir", default=os.path.join(HERE, "..", "P4.1_cheap_su2_fix", "results"), dest="ref_dir")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    if a.maxiter is None:
        a.maxiter = N_EXT if a.mode == "extend" else N_FINAL
    if a.mode == "extend" and a.maxiter != N_EXT:
        p.error(f"extend: --maxiter must be {N_EXT} (pre-registered)")
    return a


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    dict(prep=prep, smoke=smoke, run=run, extend=extend, report=report, check=check,
         **{"report-ext": report_ext})[a.mode](a)


if __name__ == "__main__":
    main()
