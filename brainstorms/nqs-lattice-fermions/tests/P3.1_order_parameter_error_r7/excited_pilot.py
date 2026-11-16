#!/usr/bin/env python3
"""
P3.1 round 7, ESR0 excited-state pilot (ED-scale premise test; exact sums, no sampling of the Ritz matrices)
==========================================================================================================

Question. Route ESR0 (Ritz of H in span{psi, (H - E_psi) psi, phi_1..phi_K}) passed its injected check
(job 454376) with SYNTHETIC excited states. Can the SAME architecture as the ground state (projected NNBF,
width 6, P2.1 r3 arm "nnbf") be trained into in-irrep excited states good enough for Proposition S3, i.e.
eps_k / omega_k small, little weight outside the 24-level window and no psi_0 contamination?  This is an
ED-scale premise test: training uses exact sums over the full Hilbert space (D = 1681680), which is only
possible at ED size; the sampled (Metropolis) version is step 2.

Instance. "main": 4x4 OBC t-t'-J, N_up = N_dn = 6, t' = -0.25, J = 0.4 (P2.1 instance 0; built through
../P3.1_order_parameter_error_r6/micro_test.py -> r5_core).  psi = P_chi psi_theta of the trained width-6
NNBF, seed 11, loaded from ../P2.1_subcubic_amplitudes_r3/results/params/inst0_nnbf_s11.npz (NOT psi_0).
"small": 4x4 OBC 2+2 cluster (D = 10920), smoke only; there the penalty reference is the exact psi_0
(so the trained phi_k must reproduce ED's in-irrep excited states) and the ESR/ORC target is the r7
synthetic state (machinery only, no physics claim).

Method (sequential penalty VMC, Choo, Carleo, Regnault & Neupert, PRL 121, 167204 (2018)).
  phi_k = P_S P_chi psi_theta_k (S = S_0 = 0 projector of tj_core.TJ.spin_project, character projector
  P_chi of the space group; the P1.1-P / P2.1 projector machinery).  For k = 1..K minimise
      F_k(theta) = <phi|H'|phi>/<phi|phi>,   H' = H + beta sum_j u_j u_j^T,
  with u_0 = P_S psi / |psi| (so u_0^T phi = <psi_hat|phi> exactly, and H' commutes with P_S P_chi) and
  u_j = phi_hat_j for the previously trained states j < k.  The penalty term equals
  beta sum_j |<ref_j_hat|phi_hat>|^2.  Exact force: dF/dtheta = 2 sum_n r(n) d psi_theta(n)/d theta with
  r = (H' phi - F phi)/|phi|^2 (valid because H' commutes with the projector; checked by finite
  differences in --mode smoke).  SR exactly as the ground state (P2.1 r3 sr_step for the spin_chi arms):
  sampled metric from |P_chi psi_theta|^2 with projected log-derivatives, Woodbury solve, step 0.05,
  shift 1e-3, 2000 samples, trust region |step| <= 1, 1200 iterations, best-F iterate kept.
  Initialisation: fresh NNBF from the UHF reference with seed (seed + 100 k).

Exact outputs per k (levels |n>, n = 1..N_LOW, are the exact in-irrep (chi, S_0) eigenvectors; psi_0 = |0>):
  * matched level m(k) = argmax_n |<n|phi_k>|^2 over n = 0..N_LOW;
    eps_k = E[phi_k] - E_{m(k)}, omega_k = E_{m(k)} - E_0, ratio eps_k/omega_k (inf if m(k) = 0);
  * |<k|phi_k>|^2 (own index), the in-window weight ||P_3 phi_k||^2 (P_3 onto |1..3>), the out-of-window
    weight ||(1 - P_24) phi_k||^2 where P_24 projects onto span{psi_0, |1..24>} (psi_0 weight reported
    separately), and |<psi_0|phi_k>|^2;
  * principal angles between span{phi_1..phi_K} and span{|1..K>};
  * training wall time per state and its ratio to the ground-state training (P2.1 JSON t_sr).
  Then, with EXACT matrices on the real seed-11 state, delta = a_0 - a(psi) and delta_hat = a(Ritz) - a(psi):
  ESR0(K) span{psi, (H-E)psi, phi_1..phi_K};  ORC(K) span{psi, (H-E)psi, |1..K>};  L1R span{psi,(H-E)psi};
  delta_hat/delta for pair (K_1/2 = 2 on this state), spin and charge, and the ratio ESR0/ORC.
  Shrinkage-aware interval (critic fix b): delta_hat * [1, 1 + max_{k<=k_obs} epsh_k/omegah_k] where
  epsh_k = E[phi_k] - theta_k and omegah_k = theta_k - theta_0, theta = Ritz values of H in
  span{psi, (H-E)psi, phi_j, (H-E_j)phi_j : j <= K} (one Krylov cleaning step; epsh is an estimate that
  can be computed without ED; its calibration against the exact eps_k is printed).  k_obs = min(K_1/2, K)
  with K_1/2 = 2 (pair), 6 (spin), 11 (charge) from job 454376.  Abstain if the interval spans a factor > 2.

Pre-registered rule (printed at the end; decides whether step 2, sampled training on 3 seeds, is run):
  GO          iff ALL of: eps_k/omega_k <= 0.4 for k = 1..3; out-of-window weight <= 0.03 for k = 1..3;
              |<psi_0|phi_k>|^2 <= 0.02 for k = 1..3; exact-matrix pair ratio ESR0/ORC in [0.7, 1.3].
  MUTATE      iff eps_1/omega_1 > 1 or any out-of-window weight > 0.05.
  CONDITIONAL otherwise (step 2 scored on spin and pair only).

Usage
  python3 excited_pilot.py --mode smoke --instance small --results smoke_excpilot
      Finite-difference check of the penalty force (random directions, with two penalty references) and
      a K = 2 training on the small cluster against psi_0; prints SMOKE PASS/FAIL.
  python3 excited_pilot.py --mode run --instance main --K 3 --results results_excpilot   (Slurm)

Options
  --mode {run, smoke}     run: the pilot (train K states, then exact analysis); smoke: FD check + small
                          training with the smoke defaults below applied to unspecified training options.
  --instance {main, small}  main = 4x4 OBC 6+6 (P2.1 instance 0); small = 4x4 OBC 2+2 (smoke only).
  --K INT                 number of excited states trained sequentially (default 3).
  --beta FLOAT            penalty strength in units of t (default 2.0).
  --seed INT              ground-state seed whose P2.1 parameters define psi (default 11; file
                          inst0_nnbf_s{seed}.npz must exist for --instance main).
  --width INT             NNBF hidden width of psi and of every phi_k (default 6).
  --sr-iters INT          SR iterations per excited state (default 1200; smoke default 400).
  --sr-lr FLOAT           SR step (default 0.05).
  --sr-shift FLOAT        SR diagonal shift (default 1e-3).
  --sr-samples INT        samples for the SR metric (default 2000; smoke default 400).
  --chunk INT             amplitude chunk size for exact sums (default 32768).
  --o-chunk INT           sample chunk for projected log-derivatives (default 8).
  --n-low INT             number of exact in-irrep excited levels computed (default 30; must be >= 24
                          for the P_24 window on main; the small cluster uses min(n_low, available)).
  --log-every INT         iterations between exact progress lines (energy, overlaps) (default 50).
  --p21-params PATH       directory with the P2.1 parameter files.
  --p21-json PATH         P2.1 result JSON of the ground state (for the ground training wall time).
  --results PATH          output directory: params/phi{k}_s{seed}_w{width}.npz, phi{k}_*.json per state
                          (training record) and excited_pilot.json (all exact outputs and the verdict).
  --overwrite {0, 1}      0 (default): an existing phi_k parameter file is loaded and its training is
                          skipped (resume); 1: retrain every state.
"""
import argparse
import importlib.util
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("r7", os.path.join(HERE, "micro_test.py"))
r7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r7)
r6, rc, OBS, log = r7.r6, r7.rc, r7.OBS, r7.log

K_HALF = {"pair": 2, "spin": 6, "charge": 11}     # job 454376, seed-11 width-6 state
WIN = 24
RULE = dict(eps_go=0.4, out_go=0.03, psi0_go=0.02, pair_lo=0.7, pair_hi=1.3, eps_mut=1.0, out_mut=0.05)


# ============================================================================ penalty SR
class PenaltyState:
    """phi = P_S P_chi psi_theta and the penalised objective F = <phi|H'|phi>/<phi|phi>."""

    def __init__(self, m, sg, S0, refs, beta, chunk):
        self.m, self.sg, self.S0, self.beta, self.chunk = m, sg, S0, beta, chunk
        self.refs = [np.asarray(u) for u in refs]

    def vectors(self, ans):
        psi_t = ans.amplitudes(self.chunk)
        psi_c = self.sg.project(psi_t)
        phi = self.m.spin_project(psi_c, self.S0)
        return psi_c, phi

    def objective(self, phi):
        nn = float(phi @ phi)
        Hp = self.m.H @ phi
        E = float(phi @ Hp) / nn
        ov = np.array([float(u @ phi) for u in self.refs])
        pen = self.beta * float(np.sum(ov ** 2)) / nn
        Hp_eff = Hp.copy()
        for u, o in zip(self.refs, ov):
            Hp_eff += self.beta * o * u
        return E + pen, E, pen, Hp_eff, nn

    def force(self, ans, phi):
        F, E, pen, Hp_eff, nn = self.objective(phi)
        return F, E, pen, ans.force((Hp_eff - F * phi) / nn, self.chunk)   # dF/dtheta = 2 * force


def sr_step(ans, ps, args, rng):
    psi_c, phi = ps.vectors(ans)
    F, E, pen, f = ps.force(ans, phi)
    pc = psi_c ** 2
    idx = rng.choice(len(pc), size=args.sr_samples, p=pc / pc.sum())
    O = ans.per_sample_O_proj(ps.sg, idx, psi_c, args.o_chunk)
    Oc = (O - O.mean(0)) / np.sqrt(len(idx))
    lam = args.sr_shift
    Km = Oc @ Oc.T + lam * np.eye(len(idx))
    dx = (f - Oc.T @ np.linalg.solve(Km, Oc @ f)) / lam
    step = args.sr_lr * dx
    nrm = np.linalg.norm(dx)
    if nrm * args.sr_lr > 1.0:
        step *= 1.0 / (nrm * args.sr_lr)
    return F, E, pen, phi, step


def train_excited(ans, ps, args, rng, label, probe):
    """Penalty SR; keeps the best-F iterate. probe(phi) -> str for the progress line."""
    best_F, best_x = np.inf, ans.flat().copy()
    t0 = time.time()
    trace = []
    for it in range(args.sr_iters):
        x0 = ans.flat().copy()
        F, E, pen, phi, step = sr_step(ans, ps, args, rng)
        if not np.isfinite(F):
            log(f"{label}: non-finite objective at it {it}; restoring best")
            ans.set_flat(best_x)
            break
        if F < best_F:
            best_F, best_x = F, x0
        ans.set_flat(x0 - step)
        if it % args.log_every == 0 or it == args.sr_iters - 1:
            msg = probe(phi)
            trace.append(dict(it=it, F=F, E=E, pen=pen, probe=msg))
            log(f"{label}: it {it:5d}  F = {F:.8f}  E = {E:.8f}  pen = {pen:.2e}  {msg}  "
                f"({(time.time() - t0) / (it + 1):.2f} s/step)")
    F = ps.objective(ps.vectors(ans)[1])[0]
    if not (F <= best_F):
        ans.set_flat(best_x)
    wall = time.time() - t0
    return wall, trace


# ============================================================================ exact analysis
def unit(v):
    return v / np.linalg.norm(v)


def match_levels(H, phi, psi0, U, e_all):
    """Overlaps with [psi_0, |1..N>]; matched level and eps/omega."""
    ov = np.concatenate([[float(psi0 @ phi) ** 2], (U.T @ phi) ** 2])
    mk = int(np.argmax(ov))
    Ephi = float(phi @ (H @ phi))
    eps = Ephi - e_all[mk]
    om = e_all[mk] - e_all[0]
    return ov, mk, Ephi, eps, om, (eps / om if om > 1e-12 else math.inf)


def ritz_values(B, HB):
    S = B @ B.T; H = B @ HB.T
    S = 0.5 * (S + S.T); H = 0.5 * (H + H.T)
    lam, V = np.linalg.eigh(S)
    keep = lam > 1e-11 * lam.max()
    X = V[:, keep] / np.sqrt(lam[keep])
    return np.linalg.eigvalsh(X.T @ H @ X)



def analyse(m, sg, S0, ob, psi0, e_all, U, psi_ref, target, phis, K, out):
    """All exact outputs. psi_ref: penalty reference; target: ESR/ORC state (the same on main)."""
    a0 = ob.all(psi0)
    nl = U.shape[1]
    win = min(WIN, nl)
    per = []
    for k, phi in enumerate(phis, start=1):
        phi = unit(phi)
        ov, mk, Ephi, eps, om, rat = match_levels(m.H, phi, psi0, U, e_all)
        own = float(ov[k]) if k <= nl else float("nan")
        in3 = float(np.sum(ov[1:4]))
        outw = float(1.0 - ov[0] - np.sum(ov[1:win + 1]))
        rec = dict(k=k, E=Ephi, matched=mk, eps=eps, omega=om, eps_over_omega=rat, own_overlap=own,
                   in3=in3, out_window=outw, psi0_weight=float(ov[0]),
                   ref_overlap=float(unit(psi_ref) @ phi) ** 2, S2=m.S2_expect(phi),
                   chi_dev=float(np.linalg.norm(sg.project(phi) - phi)), top5=np.argsort(-ov)[:5].tolist())
        per.append(rec)
        log(f"phi_{k}: E={Ephi:.6f} matched level {mk} (omega={om:.4f}) eps={eps:+.4f} eps/omega={rat:.3f} | "
            f"|<{k}|phi>|^2={own:.4f} ||P3 phi||^2={in3:.4f} out-of-{win}-window={outw:.4f} "
            f"|<psi0|phi>|^2={ov[0]:.4f} |<psi|phi>|^2={rec['ref_overlap']:.2e} S2={rec['S2']:.2e} "
            f"top levels {rec['top5']}")
    Q, _ = np.linalg.qr(np.array([unit(p) for p in phis]).T)
    sv = np.linalg.svd(U[:, :K].T @ Q, compute_uv=False)
    ang = np.degrees(np.arccos(np.clip(sv, -1, 1))).tolist()
    log(f"principal angles span(phi_1..{K}) vs span(|1..{K}>): {np.round(ang, 2).tolist()} deg "
        f"(cosines {np.round(sv, 4).tolist()})")
    out["per_state"], out["principal_angles_deg"], out["principal_cos"] = per, ang, sv.tolist()
    # ---------------------------------------------------------------- ESR0 vs ORC (exact matrices)
    blk = r7.Block(m, ob)
    blk.add_state(target)
    pb = r7.Block(m, ob)
    for ph in phis:
        pb.add_state(ph)
    orc = r7.Block(m, ob)
    for n in range(K):
        orc.add(U[:, n])

    def shift(extra_b, extra_H, extra_A):
        B = np.array(blk.b + extra_b)
        HB = np.array(blk.Hb + extra_H)
        A = {o: B @ np.array(blk.Ab[o] + extra_A[o]).T for o in OBS}
        return r7.ritz_shift(B @ B.T, B @ HB.T, A, 1e-11)
    esr0 = shift([pb.b[2 * i] for i in range(K)], [pb.Hb[2 * i] for i in range(K)],
                 {o: [pb.Ab[o][2 * i] for i in range(K)] for o in OBS})
    oc = shift(orc.b, orc.Hb, orc.Ab)
    l1r = shift([], [], {o: [] for o in OBS})
    # estimated eps/omega from one Krylov cleaning step (no ED information)
    th = ritz_values(np.array(blk.b + pb.b), np.array(blk.Hb + pb.Hb))
    epsh = [float(per[k]["E"] - th[k + 1]) for k in range(K)]
    omh = [float(th[k + 1] - th[0]) for k in range(K)]
    rh = [e / o if o > 1e-12 else math.inf for e, o in zip(epsh, omh)]
    log(f"estimated epsh/omegah (one Krylov step) = {np.round(rh, 3).tolist()} vs exact eps/omega = "
        f"{np.round([p['eps_over_omega'] for p in per], 3).tolist()}")
    esr = {}
    for o in OBS:
        d = a0[o] - ob.expect(o, target)
        kk = min(K_HALF[o], K)
        r_ = max(rh[:kk])
        lo, hi = esr0[o], esr0[o] * (1 + r_)
        abst = bool((1 + r_) > 2 or not np.isfinite(r_))
        esr[o] = dict(delta=d, ESR0=esr0[o], ORC=oc[o], L1R=l1r[o], ESR0_ratio=esr0[o] / d, ORC_ratio=oc[o] / d,
                      L1R_ratio=l1r[o] / d, ESR0_over_ORC=esr0[o] / oc[o] if oc[o] != 0 else math.nan,
                      K_half=K_HALF[o], interval=[lo, hi], interval_ratio=[lo / d, hi / d], abstain=abst)
        log(f"{o}: delta={d:+.3e} | ESR0(K={K}) {esr0[o]:+.3e} ratio {esr0[o] / d:.3f} | ORC(K={K}) "
            f"{oc[o]:+.3e} ratio {oc[o] / d:.3f} | L1R ratio {l1r[o] / d:.3f} | ESR0/ORC "
            f"{esr[o]['ESR0_over_ORC']:.3f} (K_1/2={K_HALF[o]}) | S3 interval delta_hat*[1,{1 + r_:.2f}] = "
            f"ratio [{lo / d:.3f},{hi / d:.3f}] {'ABSTAIN' if abst else 'report'}")
    out["esr"] = esr
    out["eps_hat_over_omega_hat"] = rh
    return per, esr


def verdict(per, esr):
    e = [p["eps_over_omega"] for p in per[:3]]
    ow = [p["out_window"] for p in per[:3]]
    p0 = [p["psi0_weight"] for p in per[:3]]
    pr = esr["pair"]["ESR0_over_ORC"]
    go = (len(per) >= 3 and all(x <= RULE["eps_go"] for x in e) and all(x <= RULE["out_go"] for x in ow)
          and all(x <= RULE["psi0_go"] for x in p0) and RULE["pair_lo"] <= pr <= RULE["pair_hi"])
    mut = e[0] > RULE["eps_mut"] or any(x > RULE["out_mut"] for x in ow)
    v = "GO" if go else ("MUTATE" if mut else "CONDITIONAL")
    return v, dict(eps_over_omega=e, out_window=ow, psi0_weight=p0, pair_ESR0_over_ORC=pr)


def print_rule():
    log("PRE-REGISTERED RULE: GO to step 2 iff eps_k/omega_k <= 0.4 (k=1..3) AND out-of-24-window weight "
        "<= 0.03 (k=1..3) AND |<psi0|phi_k>|^2 <= 0.02 (k=1..3) AND exact pair ESR0/ORC in [0.7,1.3]; "
        "MUTATE iff eps_1/omega_1 > 1 or any out-of-window weight > 0.05; otherwise CONDITIONAL "
        "(step 2 scored on spin and pair only).")


# ============================================================================ smoke
def fd_check(m, sg, S0, psi0, U, args):
    ans = rc.nnbf(m, args.width, 5)
    rng = np.random.default_rng(9)
    ans.set_flat(ans.flat() + 0.05 * rng.standard_normal(ans.n_params()))
    refs = [m.spin_project(unit(sg.project(psi0 + 0.2 * U[:, 1])), S0), unit(U[:, 0] + 0.3 * U[:, 2])]
    ps = PenaltyState(m, sg, S0, refs, args.beta, args.chunk)
    _, phi = ps.vectors(ans)
    F0, E0, pen, g = ps.force(ans, phi)
    x0 = ans.flat().copy()
    errs = []
    for t in range(4):
        d = rng.standard_normal(len(x0)); d /= np.linalg.norm(d)
        h = 1e-5
        ans.set_flat(x0 + h * d); Fp = ps.objective(ps.vectors(ans)[1])[0]
        ans.set_flat(x0 - h * d); Fm = ps.objective(ps.vectors(ans)[1])[0]
        fd = (Fp - Fm) / (2 * h)
        an = 2 * float(g @ d)
        errs.append(abs(fd - an) / max(abs(fd), 1e-12))
        log(f"FD check dir {t}: FD {fd:+.8e} analytic {an:+.8e} rel err {errs[-1]:.1e} (pen {pen:.3e})")
    ans.set_flat(x0)
    return max(errs)


# ============================================================================ main
def run(args):
    t0 = time.time()
    m, E0, psi0, sg, S0, ob = r6.setup(args.instance)
    psi0 = psi0 if psi0.sum() > 0 else -psi0
    e, U = r6.sector_eigs(m, sg, S0, args.n_low + 1)
    assert abs(e[0] - E0) < 1e-6, (e[0], E0)
    e_all = e.copy()
    U = U[:, 1:]
    nl = U.shape[1]
    log(f"{nl} in-irrep (chi, S={S0:g}) levels: omega = {np.round(e_all[1:] - E0, 4).tolist()} "
        f"({time.time() - t0:.0f} s)")
    if args.instance == "main":
        assert nl >= WIN, f"need >= {WIN} levels for the P_24 window, got {nl}"
    print_rule()
    out = dict(args=vars(args), E0=E0, omega=(e_all[1:] - E0).tolist(), rule=RULE, K_half=K_HALF)
    os.makedirs(os.path.join(args.results, "params"), exist_ok=True)
    smoke_ok = True
    if args.mode == "smoke":
        err = fd_check(m, sg, S0, psi0, U, args)
        out["fd_max_rel_err"] = err
        smoke_ok &= err < 1e-5
        log(f"FD check max rel err {err:.1e} -> {'ok' if err < 1e-5 else 'FAIL'}")
    # ---------------------------------------------------------------- reference psi and target
    if args.instance == "main":
        f = os.path.join(args.p21_params, f"inst0_nnbf_s{args.seed}.npz")
        ans0 = rc.load_params(rc.nnbf(m, args.width, args.seed), f)
        psi = unit(sg.project(ans0.amplitudes(args.chunk)))
        psi = psi if psi @ psi0 > 0 else -psi
        psi_ref, target = psi, psi
        dE = float(psi @ (m.H @ psi) - E0)
        log(f"psi = P_chi NNBF(w{args.width}, s{args.seed}) from {f}: dE = {dE:.5f}, |<psi0|psi>|^2 = "
            f"{(psi0 @ psi) ** 2:.4f}, S2 = {m.S2_expect(psi):.4f}")
        out.update(psi_dE=dE, psi_ov0=float((psi0 @ psi) ** 2))
    else:
        psi_ref = psi0
        rng0 = np.random.default_rng(5)
        target = unit(sg.project(psi0 * np.exp(0.3 * rng0.standard_normal(m.D))))
        log("small: penalty reference = exact psi_0; ESR/ORC target = r7 synthetic state (machinery only)")
    t_ground = None
    if os.path.exists(args.p21_json):
        t_ground = json.load(open(args.p21_json)).get("t_sr")
    # ---------------------------------------------------------------- sequential training
    phis, recs = [], []
    for k in range(1, args.K + 1):
        tag = f"phi{k}_s{args.seed}_w{args.width}_{args.instance}"
        pfile = os.path.join(args.results, "params", tag + ".npz")
        jfile = os.path.join(args.results, tag + ".json")
        ans = rc.nnbf(m, args.width, args.seed + 100 * k)
        refs = [m.spin_project(psi_ref, S0)] + [unit(p) for p in phis]
        ps = PenaltyState(m, sg, S0, refs, args.beta, args.chunk)
        if os.path.exists(pfile) and os.path.exists(jfile) and not args.overwrite:
            rc.load_params(ans, pfile)
            rec = json.load(open(jfile))
            log(f"phi_{k}: loaded {pfile} (training skipped; wall {rec['wall']:.0f} s)")
        else:
            def probe(phi, k=k):
                v = unit(phi)
                ov = (U.T @ v) ** 2
                return (f"|<psi0|phi>|^2={(psi0 @ v) ** 2:.3e} |<{k}|phi>|^2={ov[k - 1]:.4f} "
                        f"P3={np.sum(ov[:3]):.4f} argmax={int(np.argmax(ov)) + 1}")
            rng = np.random.default_rng(args.seed + 7919 * k)
            wall, trace = train_excited(ans, ps, args, rng, f"phi_{k}", probe)
            np.savez(pfile, flat=ans.flat(), layout=json.dumps([[kk, list(ans.p[kk].shape)] for kk in ans.keys]))
            rec = dict(k=k, wall=wall, s_per_step=wall / max(args.sr_iters, 1), iters=args.sr_iters, trace=trace)
            json.dump(rec, open(jfile, "w"), indent=1, default=float)
        phi = ps.vectors(ans)[1]
        phis.append(unit(phi))
        rec["wall_over_ground"] = rec["wall"] / t_ground if t_ground else None
        recs.append(rec)
        log(f"phi_{k}: training wall {rec['wall']:.0f} s ({rec['s_per_step']:.2f} s/step); ground-state "
            f"training {t_ground if t_ground else float('nan'):.0f} s -> ratio "
            f"{rec['wall_over_ground'] if rec['wall_over_ground'] else float('nan'):.2f}")
    out["training"] = [{k_: v for k_, v in r.items() if k_ != "trace"} for r in recs]
    # ---------------------------------------------------------------- analysis
    per, esr = analyse(m, sg, S0, ob, psi0, e_all, U, psi_ref, target, phis, args.K, out)
    if args.mode == "smoke":
        ok1 = per[0]["own_overlap"] >= 0.95 and per[0]["eps_over_omega"] <= 0.1
        smoke_ok &= ok1
        log(f"smoke training: |<1|phi_1>|^2 = {per[0]['own_overlap']:.4f}, eps_1/omega_1 = "
            f"{per[0]['eps_over_omega']:.3f} (need >= 0.95 and <= 0.1) -> {'ok' if ok1 else 'FAIL'}")
        log(f"SMOKE {'PASS' if smoke_ok else 'FAIL'}")
        out["smoke_pass"] = bool(smoke_ok)
    if len(per) >= 3:
        v, why = verdict(per, esr)
        out["verdict"], out["verdict_inputs"] = v, why
        print_rule()
        log(f"EXCITED PILOT VERDICT: {v} | eps/omega {np.round(why['eps_over_omega'], 3).tolist()} "
            f"out-of-window {np.round(why['out_window'], 4).tolist()} psi0 weight "
            f"{np.round(why['psi0_weight'], 4).tolist()} pair ESR0/ORC {why['pair_ESR0_over_ORC']:.3f}")
    json.dump(out, open(os.path.join(args.results, "excited_pilot.json"), "w"), indent=1, default=float)
    log(f"done ({time.time() - t0:.0f} s)")
    return 0 if smoke_ok else 2


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["run", "smoke"], default="run")
    ap.add_argument("--instance", choices=["main", "small"], default="main")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--width", type=int, default=6)
    ap.add_argument("--sr-iters", type=int, default=None)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--o-chunk", type=int, default=8)
    ap.add_argument("--n-low", type=int, default=30)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--p21-params", default=os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3", "results", "params"))
    ap.add_argument("--p21-json", default=os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3", "results",
                                                       "inst0_nnbf_s11.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results_excpilot"))
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    a = ap.parse_args(argv)
    smoke = a.mode == "smoke"
    if a.sr_iters is None:
        a.sr_iters = 400 if smoke else 1200
    if a.sr_samples is None:
        a.sr_samples = 400 if smoke else 2000
    if a.mode == "run" and a.instance == "small":
        ap.error("--instance small is for --mode smoke only")
    return a


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
