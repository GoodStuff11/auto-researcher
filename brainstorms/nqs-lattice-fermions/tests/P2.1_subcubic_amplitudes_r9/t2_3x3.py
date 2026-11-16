#!/usr/bin/env python3
"""
P2.1 round-9 tier-2 (T2) pilot of Route 6.1'' (LH-HFDS with a 3x3 patch) on instance 4 (5x3 OBC 6+6
t-t'-J), seed 22, 75 P_chi-projected SR steps: the protocol of the Route 6.1' T2 (job 698229), plus a
compute-matched NNBF foil. The rule is in the leaf artifact (Round 9, "Pre-registered T2 rule, Route
6.1''"). ../P2.1_subcubic_amplitudes_r7/lh_train.py and its imports are imported READ-ONLY (jobs 699118
and 699305 use them); nothing there is modified.
=====================================================================================================

Arms (one Slurm task each; LHAnsatz / LRAnsatz from lh_train / tj_core, unchanged):
  task 0  lh_3x3   route: LH-HFDS, 3x3 patch, F = 5, P = 1625 (Jastrow fills to P_target = 1626)
  task 1  lh_plus  patch-size foil: plus-shaped 5-site patch, P = 1617
  task 2  lh_site  one-site arm (Tocchio-Becca lattice backflow with long-range K), P = 1641, reported
  task 3  nnbf     projected NNBF at matched parameters (width 6, P = 1626)
  task 4  nnbf_c   projected NNBF at matched COMPUTE: width H_c = the smallest width whose operation
                   count per amplitude (implemented forward) is >= lh_3x3's (amendment A1; H_c = 16)
  task 5  nnbf_w   projected NNBF at width H_w = 48, the largest width of the wall-time scan (reported
                   only; the wall-time match does not exist on instance 4, see amendment A1)
Amendment A1 (leaf, Round 9; written after the calibrate pre-flight, before any T2 data): the
pre-flight found the NNBF step time nearly flat in H (3.09 s at H = 6, 3.91 s at H = 48, against
10.34 s for lh_3x3), so no width in the pre-registered grid matches wall time, the rule's fallback
H_c = 48 would trip its own s/step-ratio guard (0.38 < 0.5) by construction, and exact-sum wall time
measures width-independent work (420420-configuration determinants, projection, SR solve), not the
networks' compute. H_c is therefore set by the operation count per amplitude, and the s/step-ratio
guard is dropped (the ratio is reported).

Rule (summary; the leaf is authoritative): rho_N = dEs(lh_3x3)/dEs(nnbf), rho_C = dEs(lh_3x3)/dEs(nnbf_c),
rho_F = dEs(lh_3x3)/dEs(lh_plus), dEs = projected dE_sign (P1.1 upper estimate) after 75 steps.
  1 INCOMPLETE if lh_3x3, lh_plus, nnbf or nnbf_c is missing.
  2 INVALID: calibration > 1e-3 dE_tot; multi-start spread >= 0.05; step-0 energy != pre-flight value
    (1e-6); nnbf_c width != configured H_c; lh_3x3 / lh_plus / nnbf not within 2V+2 of P = 1626; an NNBF
    arm stopped early.   (instance-4 guards; off on toys)
  3 KILL if lh_3x3 early-stopped at SR step 50 (E(50) - E0 > 2 (E_pilot(50) - E0) = 0.681).
  4 UNINFORMATIVE if dEs(nnbf) < 0.005, or dEs(nnbf_c) < 0.005 and dEs(lh_3x3) < 0.005.
  5 KILL if rho_C >= 1 or rho_F > 0.85.
  6 PROMOTE if rho_N <= 0.75 and rho_C <= 0.75.
  7 AMBIGUOUS otherwise (action: seed 33 for the five arms).

Modes
  --mode calibrate  (pre-flight, instance 4, run once on ONE node at 4 threads) validity asserts (E0,
                    |G| = 4, S = 0, dim_S_chi = 15080, spectral gap E1 - E0 >= 0.02 t, matched parameter
                    counts); step-0 energy of every arm at --seed; t_st = mean wall time of
                    --calib-steps SR steps for lh_3x3, then for NNBF widths in --widths (ascending; the
                    scan stops at the first width whose t_st >= t_st(lh_3x3)); writes H_c, the step-0
                    energies and the timings to --config. Refuses to overwrite without --overwrite 1.
  --mode opmatch    (amendment A1) reads --config (written by calibrate), counts the operations per
                    amplitude of the implemented forwards (lh_3x3: V F (2p + N_up + N_dn) + N_up V N_up
                    + N_dn V N_dn; NNBF width H: 2V H + H V (N_up + N_dn); both + the two N x N
                    determinants), sets H_c = smallest H in --widths with count >= lh_3x3's, H_w = 48,
                    computes the step-0 energies of nnbf_c and nnbf_w at --seed, and rewrites --config
                    (the wall-time scan is kept under "t_st").
  --mode check      lh_3x3 (and nnbf at width --check-H): analytic energy gradient vs central finite
                    differences (no projection, P_chi) on toy instances 9 and 8; lh_3x3 projected
                    per-sample O vs finite differences and fast carried-table E_loc (round-7 engine) ==
                    (H psi)/psi on every configuration of instance 8. Exit code 2 on failure.
  --mode train      one task: --task {0..5} or --arm NAME; --instance (4) --seed (22). nnbf_c reads
                    H_c from --Hc, nnbf_w reads H_w from --Hw (both must equal the config on instance 4). Writes
                    <results>/t2_<arm>_inst<I>_s<seed>.json and params/*.npy; refuses to overwrite
                    without --overwrite 1. Early stop at SR step 50 for lh_* arms on instance 4.
  --mode aggregate  verdict from <results> (instance --instance, seed --seed) and --config.
  --mode synthetic  verdict on synthetic results covering every rule branch.
  --mode controls   toy instance 7: trains nnbf_c (width --Hc, toy settings) into --toy-results, reads
                    the r7 toy arms (lh_plus, lh_site, nnbf, sj, lh_3x3; seed 11) READ-ONLY from
                    ../P2.1_subcubic_amplitudes_r7/results_preflight, builds the sign-exact positive
                    control (ground state times a positive random amplitude factor, P_chi-projected),
                    and checks: positive control (route := sign-exact state) -> PROMOTE; negative
                    control A (route := Slater x Jastrow) -> KILL; negative control B (route := nnbf_c
                    itself) -> KILL (rho_C = 1).

Options: --seed (22) --F (5) --Hc (H_c; required for task 4 and controls) --Hw (H_w; required for task 5)
--widths "8,12,16,20,24,32,48"
--calib-steps (2) --check-H (24) --config (./preflight_config.json) --results (./results)
--toy-results (./results_preflight) --sr-iters (75) --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000
--lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3 (the pilot's values) --chunk 32768 --o-chunk 8
--r-min 1.5 --overwrite {0,1}.
Threads: OMP_NUM_THREADS etc. (set by the sbatch file; <= 4 for local runs, under ulimit -t 600).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
R7 = os.path.join(os.path.dirname(HERE), "P2.1_subcubic_amplitudes_r7")
sys.path.insert(0, R7)
import lh_train as LT  # noqa: E402  (read-only; registers toy instances 7 and 8, adds r5 to the path)
import micro_test as mt  # noqa: E402
from p11_core import budget, energy, sign_constrained_min, solve_ed  # noqa: E402
from tj_core import LRAnsatz  # noqa: E402

log = LT.log
ARMS = ["lh_3x3", "lh_plus", "lh_site", "nnbf", "nnbf_c", "nnbf_w"]
MATCHED = ["lh_3x3", "lh_plus", "nnbf"]
RNG_OFFSET = dict(LT.RNG_OFFSET, nnbf_c=105, nnbf_w=106)
# ---- pre-registered T2 rule (leaf, Round 9) ----
KILL_C, PROMOTE_N, PROMOTE_C, FOIL_MAX = 1.0, 0.75, 0.75, 0.85
PREMISE_SIGN = 0.005
GAP_MIN = 0.02
STEP0_TOL = 1e-6
PIL = LT.PILOT[4]


# ============================================================================ arms
def make_arm(arm, m, refs, seed, F, Hc=None, match_width=6):
    """nnbf_c and nnbf_w: NNBF of width Hc (the caller passes H_c or H_w)."""
    if arm in ("nnbf_c", "nnbf_w"):
        assert Hc is not None and Hc > 0, f"{arm} needs a width"
        target = LRAnsatz(m, refs[0], "full", K=1, Hs=match_width).n_params()
        a = LRAnsatz(m, refs[0], "full", K=1, Hs=Hc, seed=seed)
        return a, dict(mode="full", Hs=Hc), target
    return LT.make_arm(arm, m, refs, seed, F, match_width)


def setup_instance(inst, args, k=4):
    m = mt.make_model(inst)
    m.spin_setup(); m.pair_setup(args.r_min)
    W, Vv, deg = solve_ed(m, k, 1e-8, mt.UHF_SEED)
    psi0 = mt.resolve_ground(m, Vv, deg)
    sg, gen = mt.build_symmetry(m, psi0)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * m.S2_expect(psi0)) - 1)))
    dim_chi, dim_S_chi = mt.sector_dims(m, sg, S0)
    return m, W, Vv, deg, psi0, sg, gen, S0, dim_S_chi


def step0_energy(a, m, sg, chunk):
    return float(energy(m.H, mt.state_of(a, m, sg, "chi", 0, chunk)[2])[0])


# ============================================================================ calibrate (pre-flight)
def calibrate(args):
    if os.path.exists(args.config) and not args.overwrite:
        log(f"{args.config} exists; use --overwrite 1"); return False
    m, W, Vv, deg, psi0, sg, gen, S0, d_S = setup_instance(4, args)
    gap = float(W[1] - W[0])
    ok_inst = (abs(W[0] - PIL["E0"]) < 1e-8 and sg.G == PIL["G"] and S0 == 0 and d_S == PIL["dim_S_chi"]
               and deg == 1 and gap >= GAP_MIN)
    log(f"instance 4 asserts: E0={W[0]:.10f} (pilot {PIL['E0']:.10f}) |G|={sg.G} S={S0:g} dim_S_chi={d_S:.0f} "
        f"deg={deg} gap E1-E0={gap:.5f} (>= {GAP_MIN}: {gap >= GAP_MIN}) -> {'PASS' if ok_inst else 'FAIL'}")
    refs = mt.references(m)
    ta = parse_args(["--mode", "train", "--instance", "4", "--arm", "nnbf"])
    target = LRAnsatz(m, refs[0], "full", K=1, Hs=6).n_params()
    e0, npar, ok_p = {}, {}, True
    for arm in ["lh_3x3", "lh_plus", "lh_site", "nnbf"]:
        a, kw, _ = make_arm(arm, m, refs, args.seed, args.F)
        e0[arm], npar[arm] = step0_energy(a, m, sg, args.chunk), a.n_params()
        good = abs(a.n_params() - target) <= 2 * m.V + 2
        if arm in MATCHED:
            ok_p &= good
        log(f"arm {arm:8s} {kw} P={a.n_params()} (target {target}, matched {good}) E(it0)={e0[arm]:.8f}")

    def t_step(a, arm):
        rng = np.random.default_rng(args.seed + 1000 * RNG_OFFSET[arm])
        ts = []
        for _ in range(args.calib_steps):
            t0 = time.time()
            E, f, step = mt.sr_step(a, m, sg, "chi", 0, ta, rng)
            a.set_flat(a.flat() - step); ts.append(time.time() - t0)
        return float(np.mean(ts)), ts
    a, _, _ = make_arm("lh_3x3", m, refs, args.seed, args.F)
    t_route, ts = t_step(a, "lh_3x3")
    log(f"t_st(lh_3x3) = {t_route:.2f} s (steps {', '.join(f'{x:.2f}' for x in ts)}) at "
        f"{os.environ.get('OMP_NUM_THREADS', '?')} threads")
    a, _, _ = make_arm("nnbf", m, refs, args.seed, args.F)
    t_nnbf6, _ = t_step(a, "nnbf")
    log(f"t_st(nnbf, H=6) = {t_nnbf6:.2f} s; ratio lh_3x3 / nnbf = {t_route / t_nnbf6:.2f}")
    widths = [int(x) for x in args.widths.split(",")]
    scan, Hc = {}, widths[-1]
    for H in widths:
        a, _, _ = make_arm("nnbf_c", m, refs, args.seed, args.F, Hc=H)
        scan[H], _ = t_step(a, "nnbf_c")
        log(f"t_st(nnbf, H={H}) = {scan[H]:.2f} s  P={a.n_params()}  (>= t_st(lh_3x3): {scan[H] >= t_route})")
        if scan[H] >= t_route:
            Hc = H
            break
    else:
        log(f"no width matched; H_c = {Hc} by the rule")
    a, kw, _ = make_arm("nnbf_c", m, refs, args.seed, args.F, Hc=Hc)
    e0["nnbf_c"], npar["nnbf_c"] = step0_energy(a, m, sg, args.chunk), a.n_params()
    log(f"H_c = {Hc}: nnbf_c P={npar['nnbf_c']} E(it0)={e0['nnbf_c']:.8f}; t_st ratio nnbf_c/lh_3x3 = "
        f"{scan.get(Hc, float('nan')) / t_route:.2f}")
    cfg = dict(instance=4, seed=args.seed, F=args.F, Hc=Hc, step0=e0, n_params=npar, P_target=target,
               t_st=dict(lh_3x3=t_route, nnbf6=t_nnbf6, scan={str(k): v for k, v in scan.items()}),
               gap=gap, E0=float(W[0]), threads=os.environ.get("OMP_NUM_THREADS", "?"),
               host=os.uname().nodename, asserts_pass=bool(ok_inst and ok_p))
    json.dump(cfg, open(args.config, "w"), indent=1)
    log(f"wrote {args.config}; asserts {'PASS' if ok_inst and ok_p else 'FAIL'}")
    return bool(ok_inst and ok_p)


# ============================================================================ opmatch (amendment A1)
def ops_lh(m, p, F):
    return m.V * F * (2 * p + m.nup + m.ndn) + m.nup * m.V * m.nup + m.ndn * m.V * m.ndn


def ops_nnbf(m, H):
    return 2 * m.V * H + H * m.V * (m.nup + m.ndn)


def opmatch(args):
    cfg = json.load(open(args.config))
    m, W, Vv, deg, psi0, sg, gen, S0, d_S = setup_instance(4, args, k=2)
    det = (m.nup ** 3 + m.ndn ** 3) / 3.0
    o_route = ops_lh(m, 9, args.F) + det
    widths = [int(x) for x in args.widths.split(",")]
    Hc = next((H for H in widths if ops_nnbf(m, H) + det >= o_route), widths[-1])
    log(f"ops per amplitude: lh_3x3 {o_route:.0f}; NNBF " + ", ".join(f"H={H}: {ops_nnbf(m, H) + det:.0f}" for H in widths)
        + f" -> H_c = {Hc}")
    refs = mt.references(m)
    for arm, H in (("nnbf_c", Hc), ("nnbf_w", 48)):
        a, kw, _ = make_arm(arm, m, refs, args.seed, args.F, Hc=H)
        cfg["step0"][arm], cfg["n_params"][arm] = step0_energy(a, m, sg, args.chunk), a.n_params()
        log(f"{arm}: {kw} P={a.n_params()} E(it0)={cfg['step0'][arm]:.8f}")
    cfg.update(Hc=Hc, Hw=48, Hc_rule="operation count per amplitude (amendment A1)",
               ops=dict(lh_3x3=o_route, nnbf_c=ops_nnbf(m, Hc) + det, nnbf6=ops_nnbf(m, 6) + det))
    json.dump(cfg, open(args.config, "w"), indent=1)
    log(f"rewrote {args.config}: H_c = {Hc} (op-count matched), H_w = 48 (reported only)")
    return True


# ============================================================================ check
def run_check(args, seed=5):
    import t1_r7
    from lh_geom import neighbours4
    ok = True
    rng = np.random.default_rng(seed)
    for inst in (9, 8):
        m = mt.make_model(inst)
        refs = mt.references(m)
        W, Vv, deg = solve_ed(m, 2, 1e-8, 1234)
        sg, gen = mt.build_symmetry(m, mt.resolve_ground(m, Vv, deg))
        for arm in ("lh_3x3", "nnbf_c"):
            a, kw, tgt = make_arm(arm, m, refs, seed, 3, Hc=args.check_H)
            x = a.flat(); a.set_flat(x + 0.05 * rng.standard_normal(len(x)))   # generic point
            for proj in ("none", "chi"):
                ok &= LT.fd_check(a, m, sg, proj, rng, f"{arm} {kw} inst {inst} P={a.n_params()}")
            if inst != 8 or arm != "lh_3x3":
                continue
            psi_c = sg.project(a.amplitudes(4096))
            ii = rng.choice(m.D, size=6, replace=False)
            O = a.per_sample_O_proj(sg, ii, psi_c)
            d = rng.standard_normal(a.n_params()); x0 = a.flat().copy(); eps = 1e-6
            a.set_flat(x0 + eps * d); pp = sg.project(a.amplitudes(4096))[ii]
            a.set_flat(x0 - eps * d); pm = sg.project(a.amplitudes(4096))[ii]
            a.set_flat(x0)
            e_O = float(np.max(np.abs(O @ d - (pp - pm) / (2 * eps) / psi_c[ii])))
            log(f"check O_P [lh_3x3 inst 8] vs fd: {e_O:.1e}")
            ok &= e_O < 1e-5
            # fast carried-table E_loc (round-7 engine, lam = 0) == (H psi)/psi on every configuration
            o = t1_r7.SHHFDS.__new__(t1_r7.SHHFDS)
            o.V, o.n, o.F, o.ops = m.V, m.nup, a.F, 0
            o.loc = t1_r7.Local(a.P, a.F, rng); o.loc.W, o.loc.b = a.p["Wl"], a.p["bl"]
            o.nb = neighbours4(m.Lx, m.Ly, (m.bcx, m.bcy))
            o.P0, o.K, o.Om = [a.p["P0u"], a.p["P0d"]], [a.p["Ku"], a.p["Kd"]], [a.p["Omu"], a.p["Omd"]]
            o.lam = [np.zeros((m.V, 4, a.F))] * 2
            psi_t = a.forward(*a._occ(np.arange(m.D)))["S"]
            Hp = m.H @ psi_t
            emax, nskip = 0.0, 0
            for k in range(m.D):
                if abs(psi_t[k]) < 1e-8 * np.max(np.abs(psi_t)):
                    nskip += 1
                    continue
                u, d_ = (z[0].astype(int) for z in a._occ(np.array([k]))[:2])
                o.setup(u, d_)
                diag, cl = t1_r7.conn_list(m.T, m.nn, u, d_)
                Ef = diag + sum(cc * o.ratio(mm) for (cc, mm) in cl)
                emax = max(emax, abs(Ef - Hp[k] / psi_t[k]) / (1 + abs(Hp[k] / psi_t[k])))
            log(f"check fast E_loc == (H psi)/psi on {m.D - nskip} of {m.D} configs [lh_3x3, 3x3 OBC]: {emax:.1e}")
            ok &= emax < 1e-9
    log("ALL CHECKS PASSED" if ok else "CHECK FAILED")
    return bool(ok)


# ============================================================================ train
def train(args, toy=False):
    m, W, Vv, deg, psi0, sg, gen, S0, dim_S_chi = setup_instance(args.instance, args)
    assert S0 == 0.0, "the projected SR here assumes a singlet ground state"
    on4 = args.instance == 4
    if on4:
        assert abs(W[0] - PIL["E0"]) < 1e-8 and dim_S_chi == PIL["dim_S_chi"] and sg.G == PIL["G"]
        assert W[1] - W[0] >= GAP_MIN, W[1] - W[0]
    log(f"instance {args.instance} D={m.D} arm={args.arm} seed={args.seed} E0={W[0]:.10f} gap={W[1] - W[0]:.5f} "
        f"|G|={sg.G} dim_S_chi={dim_S_chi:.0f}")
    rng_b = np.random.default_rng(args.seed + 1000003)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng_b.random(m.D))], args.lbfgs_iters)
    calib = float(Ec - W[0])
    refs = mt.references(m)
    a, kw, target = make_arm(args.arm, m, refs, args.seed, args.F, Hc=args.Hw if args.arm == "nnbf_w" else args.Hc)
    if on4 and args.arm in MATCHED:
        assert abs(a.n_params() - target) <= 2 * m.V + 2, (a.n_params(), target)
    log(f"arm {args.arm}: {kw} n_params={a.n_params()} target={target} calib={calib:.2e}")
    rng = np.random.default_rng(args.seed + 1000 * RNG_OFFSET[args.arm])
    early = dict(E0=PIL["E0"], E_it50=PIL["E_it50"]) if (on4 and args.arm.startswith("lh")) else None
    traj, stop, sps = LT.optimize_traj(a, m, sg, args, rng, args.arm, early)
    psi_t, psi_c, psi = mt.state_of(a, m, sg, "chi", S0, args.chunk)
    budgets = {}
    for lab, v in (("P", psi_c), ("P_S", m.spin_project(psi_c, S0))):
        t2 = time.time()
        b = budget(m, v, W, psi0[:, None] if deg > 1 else Vv, 1 if deg > 1 else deg, args, rng_b)
        vn = v / np.linalg.norm(v)
        b.update(S2=m.S2_expect(vn), ov0=float((psi0 @ vn) ** 2))
        for k_ in ("P0", "Ptheta", "Psign", "fn_history", "sign_starts"):
            b.pop(k_, None)
        budgets[lab] = b
        log(f"{args.arm} [{lab}]: dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} f_E={b['f_E_up']:.3f} "
            f"spread={b['spread']:.1e} ov0={b['ov0']:.3f} ({time.time() - t2:.0f} s)")
    pdir = os.path.join(args.results, "params"); os.makedirs(pdir, exist_ok=True)
    np.save(os.path.join(pdir, f"t2_{args.arm}_inst{args.instance}_s{args.seed}.npy"), a.flat())
    return dict(instance=args.instance, arm=args.arm, seed=args.seed, arm_kwargs=kw, n_params=a.n_params(),
                P_target=target, E0=float(W[0]), gap=float(W[1] - W[0]), calib_err=calib, G=sg.G,
                dim_S_chi=dim_S_chi, irrep=gen, traj=traj, steps_done=len(traj), early_stop=stop,
                s_per_step=sps, threads=os.environ.get("OMP_NUM_THREADS", "?"), host=os.uname().nodename,
                budgets=budgets, params={k: v for k, v in vars(args).items()})


# ============================================================================ verdict
def verdict(R, cfg=None):
    """Pre-registered rule (leaf, Round 9). R: arm -> result dict or None; cfg: pre-flight config
    (instance-4 guards; None on toys). Returns (verdict, reasons, numbers)."""
    for arm in ("lh_3x3", "lh_plus", "nnbf", "nnbf_c"):
        if R.get(arm) is None:
            return "INCOMPLETE", [f"{arm} result missing"], {}
    why = []
    for arm in ("lh_3x3", "lh_plus", "nnbf", "nnbf_c"):
        r = R[arm]
        b = r["budgets"].get("P")
        if b is None or not np.isfinite(b["dE_sign"]):
            return "INVALID", [f"{arm}: no P budget"], {}
        if r["calib_err"] > LT.CALIB_TOL * b["dE_tot"]:
            why.append(f"{arm}: calibration {r['calib_err']:.1e} > {LT.CALIB_TOL} dE_tot")
        if not (b["spread"] < LT.SPREAD_TOL):
            why.append(f"{arm}: multi-start spread {b['spread']:.3f} >= {LT.SPREAD_TOL}")
        if arm.startswith("nnbf") and r["steps_done"] < r["params"]["sr_iters"]:
            why.append(f"{arm} stopped early")
        if cfg is not None:
            if abs(r["traj"][0] - cfg["step0"][arm]) > STEP0_TOL:
                why.append(f"{arm}: E(it 0) {r['traj'][0]:.8f} != pre-flight {cfg['step0'][arm]:.8f}")
            if arm in MATCHED and abs(r["n_params"] - cfg["P_target"]) > 2 * 15 + 2:
                why.append(f"{arm}: P = {r['n_params']} not matched to {cfg['P_target']}")
    if cfg is not None and R["nnbf_c"]["arm_kwargs"].get("Hs") != cfg["Hc"]:
        why.append(f"nnbf_c width {R['nnbf_c']['arm_kwargs'].get('Hs')} != configured H_c {cfg['Hc']}")
    if why:
        return "INVALID", why, {}
    route = R["lh_3x3"]
    if route.get("early_stop") and str(route["early_stop"]).startswith("EARLY KILL"):
        return "KILL", [route["early_stop"]], {}
    for arm in ("lh_plus", "lh_site"):
        r = R.get(arm)
        if r is not None and r.get("early_stop") and str(r["early_stop"]).startswith("EARLY KILL"):
            why.append(f"{arm} early-stopped (its budget is taken at the stop point)")
    ds = {k: R[k]["budgets"]["P"]["dE_sign"] for k in ("lh_3x3", "lh_plus", "nnbf", "nnbf_c")}
    nums = dict(dE_sign_route=ds["lh_3x3"], dE_sign_nnbf=ds["nnbf"], dE_sign_nnbf_c=ds["nnbf_c"],
                dE_sign_plus=ds["lh_plus"], rho_N=ds["lh_3x3"] / ds["nnbf"],
                rho_C=ds["lh_3x3"] / max(ds["nnbf_c"], 1e-300), rho_F=ds["lh_3x3"] / ds["lh_plus"])
    for arm in ("lh_site", "nnbf_w"):      # reported only
        if R.get(arm) is not None:
            nums["rho_" + arm] = ds["lh_3x3"] / R[arm]["budgets"]["P"]["dE_sign"]
    if ds["nnbf"] < PREMISE_SIGN:
        return "UNINFORMATIVE", why + [f"dE_sign(nnbf) {ds['nnbf']:.4f} < {PREMISE_SIGN}"], nums
    if ds["nnbf_c"] < PREMISE_SIGN and ds["lh_3x3"] < PREMISE_SIGN:
        return "UNINFORMATIVE", why + ["dE_sign(nnbf_c) and dE_sign(lh_3x3) both < 0.005"], nums
    kill = []
    if nums["rho_C"] >= KILL_C:
        kill.append(f"rho_C = {nums['rho_C']:.3f} >= {KILL_C} (compute-matched NNBF at least as good)")
    if nums["rho_F"] > FOIL_MAX:
        kill.append(f"rho_F = {nums['rho_F']:.3f} > {FOIL_MAX} (3x3 does not beat the plus patch)")
    if kill:
        return "KILL", why + kill, nums
    if nums["rho_N"] <= PROMOTE_N and nums["rho_C"] <= PROMOTE_C:
        return "PROMOTE", why + [f"rho_N = {nums['rho_N']:.3f}, rho_C = {nums['rho_C']:.3f} <= 0.75; "
                                 f"rho_F = {nums['rho_F']:.3f} <= {FOIL_MAX}"], nums
    return "AMBIGUOUS", why + [f"rho_N = {nums['rho_N']:.3f}, rho_C = {nums['rho_C']:.3f} (not both <= 0.75, "
                               f"rho_C < 1); action: seed 33"], nums


def load(d, arm, inst, seed):
    f = os.path.join(d, f"t2_{arm}_inst{inst}_s{seed}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def aggregate(args):
    cfg = json.load(open(args.config)) if (args.instance == 4 and os.path.exists(args.config)) else None
    if args.instance == 4 and cfg is None:
        log(f"no pre-flight config {args.config}: the instance-4 guards cannot run -> INVALID"); return "INVALID"
    R = {arm: load(args.results, arm, args.instance, args.seed) for arm in ARMS}
    for arm, r in R.items():
        if r is None:
            log(f"{arm:8s} missing"); continue
        b = r["budgets"]["P"]
        log(f"{arm:8s} steps={r['steps_done']:3d} stop={r['early_stop']} P={r['n_params']} {r['arm_kwargs']} "
            f"dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} spread={b['spread']:.1e} "
            f"E(50)={r['traj'][50] if len(r['traj']) > 50 else float('nan'):.6f} s/step={r['s_per_step']:.2f} "
            f"host={r.get('host', '?')}")
    v, why, nums = verdict(R, cfg)
    if R.get("nnbf_c") is not None and R.get("lh_3x3") is not None:
        log(f"reported: in-job s/step ratio nnbf_c/lh_3x3 = {R['nnbf_c']['s_per_step'] / R['lh_3x3']['s_per_step']:.2f}")
    log(f"ROUTE 6.1'' (LH-HFDS 3x3) T2 VERDICT: {v}  " + "; ".join(why) + "  "
        + " ".join(f"{k}={val:.4g}" for k, val in nums.items()))
    return v


def synthetic():
    """Every branch of the rule from synthetic results (instance-4 guards on)."""
    cfg = dict(Hc=16, P_target=1626, step0={a: -8.0 - 0.01 * i for i, a in enumerate(ARMS)})

    def res(arm, ds, dt=0.2, stop=None, spread=0.0, calib=0.0, steps=75, P=None, Hs=None, sps=None, t0=None):
        P = P if P is not None else (1626 if arm in MATCHED else 5000 if arm == "nnbf_c" else 1641)
        kw = dict(Hs=Hs if Hs is not None else {"nnbf_c": 16, "nnbf_w": 48}.get(arm, 6))
        return dict(arm=arm, calib_err=calib, early_stop=stop, steps_done=steps, params=dict(sr_iters=75),
                    n_params=P, arm_kwargs=kw, s_per_step=sps if sps is not None else 16.0,
                    traj=[t0 if t0 is not None else cfg["step0"][arm]] + [0.0] * 74,
                    budgets=dict(P=dict(dE_sign=ds, dE_tot=dt, spread=spread)))

    def case(route=0.020, plus=0.040, site=0.050, nnbf=0.040, nnbf_c=0.040, nnbf_w=0.010, **over):
        R = {a: res(a, v) for a, v in zip(ARMS, (route, plus, site, nnbf, nnbf_c, nnbf_w))}
        for k, v in over.items():
            arm, field = k.split("__")
            if v is None and field == "drop":
                R[arm] = None
            else:
                R[arm] = res(arm, R[arm]["budgets"]["P"]["dE_sign"], **{field: v})
        return R
    cases = [("PROMOTE", case()),
             ("KILL", case(nnbf_c=0.020)),                                   # rho_C = 1
             ("KILL", case(nnbf_c=0.015)),                                   # rho_C > 1
             ("KILL", case(plus=0.022)),                                     # rho_F > 0.85
             ("AMBIGUOUS", case(nnbf=0.025)),                                # rho_N = 0.8
             ("AMBIGUOUS", case(nnbf_c=0.024)),                              # rho_C = 0.83 < 1
             ("KILL", case(lh_3x3__stop="EARLY KILL at it 50: ...")),
             ("UNINFORMATIVE", case(route=0.003, plus=0.004, nnbf=0.004, nnbf_c=0.004)),
             ("UNINFORMATIVE", case(route=0.004, plus=0.010, nnbf=0.010, nnbf_c=0.003)),
             ("INCOMPLETE", case(nnbf_c__drop=None)),
             ("INCOMPLETE", case(lh_plus__drop=None)),
             ("PROMOTE", case(lh_site__drop=None)),                          # lh_site is reported only
             ("PROMOTE", case(nnbf_w__drop=None)),                           # nnbf_w is reported only
             ("INVALID", case(nnbf__calib=1.0)),
             ("INVALID", case(lh_3x3__spread=0.2)),
             ("INVALID", case(nnbf_c__t0=-7.0)),                             # step-0 not reproduced
             ("INVALID", case(nnbf_c__Hs=48)),                               # wrong width
             ("INVALID", case(lh_plus__P=2000)),                             # P not matched
             ("INVALID", case(nnbf__steps=60)),                              # NNBF stopped early
             ("INVALID", case(nnbf_c__steps=60))]                            # compute-matched NNBF stopped early
    ok = True
    for want, R in cases:
        got, why, _ = verdict(R, cfg)
        log(f"synthetic: expected {want:13s} got {got:13s} ({'; '.join(why)})")
        ok &= got == want
    log("SYNTHETIC: every branch returns its intended verdict" if ok else "SYNTHETIC: FAILED")
    return ok


# ============================================================================ controls (toy instance 7)
def sign_exact_result(args, seed=7):
    """Positive control: psi = P_chi[psi0 * (1 + 0.5 u)], u ~ U(0,1): exact sign, wrong amplitudes."""
    m, W, Vv, deg, psi0, sg, gen, S0, d_S = setup_instance(7, args, k=2)
    rng = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 1000003)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng_b.random(m.D))], args.lbfgs_iters)
    v = sg.project(psi0 * (1.0 + 0.5 * rng.random(m.D)))
    b = budget(m, v, W, psi0[:, None] if deg > 1 else Vv, 1 if deg > 1 else deg, args, rng_b)
    for k_ in ("P0", "Ptheta", "Psign", "fn_history", "sign_starts"):
        b.pop(k_, None)
    log(f"sign-exact control: dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.2e} spread={b['spread']:.1e}")
    return dict(arm="sign_exact", calib_err=float(Ec - W[0]), early_stop=None, steps_done=0,
                params=dict(sr_iters=0), n_params=0, arm_kwargs={}, s_per_step=0.0, traj=[0.0],
                budgets=dict(P=b))


def controls(args):
    T7 = os.path.join(R7, "results_preflight")
    R = {a: load(T7, a, 7, 11) for a in ("lh_3x3", "lh_plus", "lh_site", "nnbf", "sj")}
    out = os.path.join(args.toy_results, f"t2_nnbf_c_inst7_s11.json")
    if os.path.exists(out) and not args.overwrite:
        R["nnbf_c"] = json.load(open(out))
    else:
        ta = parse_args(["--mode", "train", "--instance", "7", "--arm", "nnbf_c", "--seed", "11",
                         "--Hc", str(args.Hc), "--sr-iters", "30", "--sr-samples", "400", "--lbfgs-iters", "1500",
                         "--fn-iters", "4", "--results", args.toy_results])
        R["nnbf_c"] = train(ta)
        json.dump(R["nnbf_c"], open(out, "w"), indent=1, default=float)
    for a, r in R.items():
        log(f"toy {a:8s} " + ("missing" if r is None else
                              f"P={r['n_params']} steps={r['steps_done']} dE_sign={r['budgets']['P']['dE_sign']:.5f} "
                              f"dE_tot={r['budgets']['P']['dE_tot']:.5f} s/step={r['s_per_step']:.2f}"))
    ok = True
    pos = verdict(dict(R, lh_3x3=sign_exact_result(args)))
    negA = verdict(dict(R, lh_3x3=R["sj"]))
    negB = verdict(dict(R, lh_3x3=R["nnbf_c"]))
    real = verdict(R)
    log(f"toy positive control (route := sign-exact state): {pos[0]} {'; '.join(pos[1])}")
    log(f"toy negative control A (route := Slater x Jastrow): {negA[0]} {'; '.join(negA[1])}")
    log(f"toy negative control B (route := nnbf_c itself): {negB[0]} {'; '.join(negB[1])}")
    log(f"toy pipeline (lh_3x3; verdict meaningless at toy size): {real[0]} {'; '.join(real[1])}")
    ok &= pos[0] == "PROMOTE" and negA[0] == "KILL" and negB[0] == "KILL"
    log("CONTROLS: intended verdicts" if ok else "CONTROLS: FAILED")
    return ok


# ============================================================================ cli
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["calibrate", "opmatch", "check", "train", "aggregate", "synthetic", "controls"],
                    default="synthetic")
    ap.add_argument("--task", type=int, default=-1, choices=[-1, 0, 1, 2, 3, 4, 5])
    ap.add_argument("--arm", default="", choices=[""] + ARMS)
    ap.add_argument("--instance", type=int, default=4)
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--F", type=int, default=5)
    ap.add_argument("--Hc", type=int, default=0)
    ap.add_argument("--Hw", type=int, default=0)
    ap.add_argument("--widths", default="8,12,16,20,24,32,48")
    ap.add_argument("--calib-steps", type=int, default=2)
    ap.add_argument("--check-H", type=int, default=24)
    ap.add_argument("--config", default=os.path.join(HERE, "preflight_config.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--toy-results", default=os.path.join(HERE, "results_preflight"))
    ap.add_argument("--sr-iters", type=int, default=75)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--o-chunk", type=int, default=8)
    ap.add_argument("--lbfgs-iters", type=int, default=4000)
    ap.add_argument("--fn-iters", type=int, default=10)
    ap.add_argument("--fn-floor", type=float, default=1e-3)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    a = ap.parse_args(argv)
    if a.task >= 0:
        a.arm = ARMS[a.task]
    if a.mode == "train" and not a.arm:
        ap.error("--mode train needs --task or --arm")
    if (a.arm == "nnbf_c" or a.mode == "controls") and a.Hc <= 0:
        ap.error("nnbf_c and --mode controls need --Hc")
    if a.arm == "nnbf_w" and a.Hw <= 0:
        ap.error("nnbf_w needs --Hw")
    return a


def main(argv=None):
    args = parse_args(argv)
    for d in (args.results, args.toy_results):
        os.makedirs(d, exist_ok=True)
    if args.mode == "calibrate":
        return 0 if calibrate(args) else 2
    if args.mode == "opmatch":
        return 0 if opmatch(args) else 2
    if args.mode == "check":
        return 0 if run_check(args) else 2
    if args.mode == "synthetic":
        return 0 if synthetic() else 2
    if args.mode == "controls":
        return 0 if controls(args) else 2
    if args.mode == "aggregate":
        aggregate(args); return 0
    if args.instance == 4 and args.arm in ("nnbf_c", "nnbf_w"):
        cfg = json.load(open(args.config))
        assert cfg["Hc"] == args.Hc and cfg["Hw"] == args.Hw and cfg["seed"] == args.seed, \
            (cfg["Hc"], args.Hc, cfg.get("Hw"), args.Hw, cfg["seed"], args.seed)
    out = os.path.join(args.results, f"t2_{args.arm}_inst{args.instance}_s{args.seed}.json")
    if os.path.exists(out) and not args.overwrite:
        log(f"{out} exists; use --overwrite 1"); return 0
    res = train(args)
    json.dump(res, open(out, "w"), indent=1, default=float)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
