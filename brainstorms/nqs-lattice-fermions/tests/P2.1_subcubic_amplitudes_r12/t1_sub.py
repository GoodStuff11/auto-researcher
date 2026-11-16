#!/usr/bin/env python3
"""
P2.1 Round 12, tier 1 (T1): sign-capacity test of the sublattice-hidden-row variants of Route 7.1.
==================================================================================================

Question (leaf "Round 11c ... Consequence"): do hidden rows on a sublattice (V_h < V) keep the sign
capacity of the all-sites route (7.1, LH-HFPS, plus patch) at fixed F? Exact sums, converged training
(not 75 steps), kill-only. The pre-registered rule is in the leaf ("Round 12 ... T1 rule"); KILL/PASS
logic is implemented in decide() below and exercised by --mode synthetic.

Arms (all at the same F, same tanh-Jastrow width Ha, same two-body density Jastrow vJ, same pairing
reference F0 = round-5 projected d-wave BCS):
  full     Route 7.1: t2_routes.HFPSAnsatz, plus patch, one hidden row per site (v_h = 1)
  half     hidden rows on the checkerboard x + y even (v_h ~ 1/2)
  quarter  hidden rows on x and y both even (v_h ~ 1/4; some sites lie in no patch)
  dense    t3_hfps.DenseHFPSAnsatz (every hidden row sees the whole configuration), reported only
The sublattice variants are HFPSAnsatz with the columns K[:, m] and the blocks Omega[m] of the
non-hidden sites m fixed at zero. This is exactly the 11b/11c sublattice ansatz
psi = e^J det(F0 + sum_{m in S_h} K[:, m] h_m Omega_m)[up, dn]; the masked entries receive an
identically zero force and log-derivative (checked in --mode check and asserted after training), so
SR never moves them. Parameter counts are reported without the masked entries.

Toy sizes (t-t'-J, t' = -0.25, J = 0.4, OBC, N_up = N_dn, distinct V, gap asserted >= 0.02 t):
  20: 4x2 3+3 (V = 8)   21: 3x3 3+3 (V = 9)   22: 5x2 2+2 (V = 10)   23: 4x3 5+5 (V = 12)
  (5x2 4+4 is excluded: E1 - E0 = 0.0041 t, the P1.2 near-degenerate case; 5x2 3+3 has a triplet ground state.)

Protocol: exact-force SR of rounds 5-10 (micro_test.sr_step; lr 0.05, shift 1e-3, --sr-samples) with
--proj none (default, scored: the projected sectors of these toys have dimension 44-724, below the
parameter counts, so projected training is in the sector-fitting regime) or --proj chi; trained to a plateau: from step --min-steps on, every 50 steps, stop when the best
projected energy improved by less than --plateau-tol (t) over the last --plateau-window steps.
A run that hits --max-steps or --max-cpu before the plateau is marked converged = false.
Budget: one budget of the trained state (p11_core.budget: L-BFGS-B 4 starts, ten Haaf fixed node), stored
under key 'P' for either --proj; dE_sign scored.

Modes
  --mode check      (i) masked full == HFPSAnsatz at the same seed (bitwise amplitudes); (ii) half and
                    quarter amplitudes == an explicit site-by-site det(F0 + sum_{m in S_h} ...) build
                    (rel. err <= 1e-12); (iii) analytic chi-projected energy gradient == central finite
                    differences (rel <= 1e-5) for half and quarter on instance 20; (iv) force and
                    per-sample log-derivatives vanish exactly on masked entries; (v) gap, singlet and
                    sector asserts on every toy instance. Exit 2 on failure.
  --mode train      --arm {full,half,quarter,dense} --instance {20,21,22,23} --seed S; writes
                    <results>/t1_<arm>_inst<I>_s<S>.json. Refuses to overwrite unless --overwrite 1.
  --mode aggregate  pre-registered verdict (KILL / PASS-TO-T2 / INCONCLUSIVE) per variant.
  --mode synthetic  decide() on synthetic results covering every rule branch.
Options: --F 4 --Ha 0 --proj none --timing-only 0 (1: SR loop only, no budget, no output file) --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 1000 --min-steps 300 --max-steps 3000
         --plateau-window 200 --plateau-tol 2e-4 --max-cpu 480 (CPU-s for the SR loop; the budget runs
         after it) --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3 --chunk 32768 --o-chunk 8
         --r-min 1.5 --results DIR (./results) --overwrite {0,1}
Threads: set OMP/OPENBLAS/MKL_NUM_THREADS=1 and `ulimit -t 600` per process (<= 4 processes).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("P2.1_subcubic_amplitudes_r10", "P2.1_subcubic_amplitudes_r7"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import t2_routes as T2  # noqa: E402  (read-only)
import lh_train as LT  # noqa: E402  (read-only)
import t3_hfps as T3  # noqa: E402  (read-only)
from lh_geom import patch_input  # noqa: E402
import micro_test as mt  # noqa: E402
from p11_core import budget, energy, sign_constrained_min, solve_ed  # noqa: E402

log = LT.log
TOYS = {20: dict(Lx=4, Ly=2, bcx="obc", bcy="obc", nup=3, ndn=3, tp=-0.25, J=0.4),
        21: dict(Lx=3, Ly=3, bcx="obc", bcy="obc", nup=3, ndn=3, tp=-0.25, J=0.4),
        22: dict(Lx=5, Ly=2, bcx="obc", bcy="obc", nup=2, ndn=2, tp=-0.25, J=0.4),
        23: dict(Lx=4, Ly=3, bcx="obc", bcy="obc", nup=5, ndn=5, tp=-0.25, J=0.4)}
mt.INSTANCES.update(TOYS)
ARMS = ("full", "half", "quarter", "dense")
VARIANTS = ("half", "quarter")
RNG_OFF = {"full": 301, "half": 302, "quarter": 303, "dense": 304}
# ---- pre-registered T1 rule (leaf, Round 12) ----
GAP_MIN = 0.02          # counted size needs E1 - E0 >= 0.02 t
PREMISE = 1e-3          # counted size needs dE_sign(full) >= 1e-3 t (else the size cannot test sign capacity)
KILL_RATIO = 1.25       # KILL if rho = dE_sign(variant)/dE_sign(full) > 1.25 at >= 2 counted sizes
TREND_RATIO = 1.15      # ... or rho strictly increasing in V over >= 3 counted sizes with rho(V_max) > 1.15
MIN_COUNTED = 3
CALIB_TOL, SPREAD_TOL = 1e-3, 0.05


def hidden_sites(Lx, Ly, sub):
    xs, ys = np.arange(Lx * Ly) % Lx, np.arange(Lx * Ly) // Lx
    m = {"full": np.ones(Lx * Ly, bool), "half": (xs + ys) % 2 == 0,
         "quarter": (xs % 2 == 0) & (ys % 2 == 0)}[sub]
    return np.nonzero(m)[0]


class SubHFPSAnsatz(T2.HFPSAnsatz):
    """HFPSAnsatz (plus patch) with hidden rows only on S_h: K[:, m] = 0 and Omega[m] = 0 for m not in S_h."""

    def __init__(self, model, F0, F, Ha, sub, seed=0):
        super().__init__(model, F0, "plus", F, Ha, seed=seed)
        self.sub = sub
        self.hs = hidden_sites(model.Lx, model.Ly, sub)
        off = np.setdiff1d(np.arange(model.V), self.hs)
        self.off = off
        mask = {k: np.ones_like(v, bool) for k, v in self.p.items()}
        mask["K"][:, off] = False
        mask["Om"][off] = False
        self.mask_flat = np.concatenate([mask[k].ravel() for k in self.keys])
        self.p["K"][:, off] = 0.0
        self.p["Om"][off] = 0.0
        self.max_masked_seen = 0.0

    def set_flat(self, x):
        bad = np.abs(x[~self.mask_flat]).max() if (~self.mask_flat).any() else 0.0
        self.max_masked_seen = max(self.max_masked_seen, float(bad))
        x = x.copy(); x[~self.mask_flat] = 0.0
        super().set_flat(x)

    def n_eff(self):
        return int(self.mask_flat.sum())


def make(arm, m, F0, F, Ha, seed):
    if arm == "dense":
        a = T3.DenseHFPSAnsatz(m, F0, F, Ha, seed=seed)
        a.n_eff = a.n_params
        return a
    return SubHFPSAnsatz(m, F0, F, Ha, arm, seed=seed)


def setup(inst, r_min=1.5):
    m = mt.make_model(inst)
    m.spin_setup(); m.pair_setup(r_min)
    W, Vv, deg = solve_ed(m, 4, 1e-8, mt.UHF_SEED)
    assert deg == 1, f"instance {inst}: degenerate ground state"
    gap = float(W[1] - W[0])
    psi0 = mt.resolve_ground(m, Vv, deg)
    sg, gen = mt.build_symmetry(m, psi0)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * m.S2_expect(psi0)) - 1)))
    assert S0 == 0.0, f"instance {inst}: ground state not a singlet"
    dim_chi, dim_S_chi = mt.sector_dims(m, sg, S0)
    return m, W, Vv, deg, gap, psi0, sg, gen, dim_chi, dim_S_chi


# ============================================================================ checks
def explicit_amp(a, ou, od):
    """Independent site-by-site build of e^J det(F0 + sum_{m in S_h} K[:, m] (h_m . Omega_m))[up, dn]."""
    p = a.p
    up, dn = np.nonzero(ou)[0], np.nonzero(od)[0]
    A = p["F0"][np.ix_(up, dn)].copy()
    for mm in a.hs:
        xp = patch_input(ou[None], od[None], a.P[mm:mm + 1])[0, 0]
        h = np.tanh(xp @ p["Wl"] + p["bl"])
        A += np.outer(p["K"][up, mm], h @ p["Om"][mm][:, dn])
    nt = (ou + od).astype(float)
    J = sum(p["vJ"][k] * nt[i] * nt[j] for k, (i, j) in enumerate(zip(*a.iu)))
    if a.Ha > 0:
        x = np.concatenate([ou, od]).astype(float) * 2 - 1
        J += np.tanh(x @ p["W1a"] + p["b1a"]) @ p["w2a"]
    return np.exp(J) * np.linalg.det(A)


def run_check(args):
    ok = True
    rng = np.random.default_rng(5)
    for inst in TOYS:
        m, W, Vv, deg, gap, psi0, sg, gen, dchi, dS = setup(inst)
        g_ok = gap >= GAP_MIN
        ok &= g_ok
        hs = {s: len(hidden_sites(m.Lx, m.Ly, s)) for s in ("half", "quarter")}
        log(f"check inst {inst} {TOYS[inst]}: D={m.D} E0={W[0]:.8f} gap={gap:.4f} (>= {GAP_MIN}: {g_ok}) "
            f"|G|={sg.G} dim_S_chi={dS:.0f} V_h half={hs['half']} quarter={hs['quarter']}")
    m = mt.make_model(21)
    refs = mt.references(m)
    base = T2.HFPSAnsatz(m, refs[1], "plus", args.F, 2, seed=7)
    full = SubHFPSAnsatz(m, refs[1], args.F, 2, "full", seed=7)
    idx = np.arange(m.D)
    d = np.max(np.abs(base.amplitudes(4096) - full.amplitudes(4096)))
    log(f"check (i) masked full == HFPSAnsatz: max |diff| = {d:.1e}"); ok &= d == 0.0
    for inst in (21, 23):
        m = mt.make_model(inst); refs = mt.references(m)
        for sub in VARIANTS:
            a = SubHFPSAnsatz(m, refs[1], args.F, 2, sub, seed=3)
            x = a.flat(); x = x + 0.3 * rng.standard_normal(len(x)) * a.mask_flat; a.set_flat(x)
            sel = rng.choice(m.D, 40, replace=False)
            ou, od, ru, rd = a._occ(sel)
            fast = a.forward(ou, od, ru, rd)["psi"]
            ref = np.array([explicit_amp(a, ou[k], od[k]) for k in range(len(sel))])
            # amplitude ordering convention: rows/cols in increasing site order (same in both builds)
            err = np.max(np.abs(fast - ref)) / np.max(np.abs(ref))
            log(f"check (ii) inst {inst} {sub}: explicit amplitude rel err {err:.1e}"); ok &= err <= 1e-12
    m = mt.make_model(20); m.spin_setup()
    refs = mt.references(m)
    W, Vv, deg = solve_ed(m, 2, 1e-8, mt.UHF_SEED)
    sg, gen = mt.build_symmetry(m, mt.resolve_ground(m, Vv, deg))
    for sub in VARIANTS:
        a = SubHFPSAnsatz(m, refs[1], args.F, 2, sub, seed=4)
        x = a.flat(); a.set_flat(x + 0.05 * rng.standard_normal(len(x)) * a.mask_flat)
        g_ok = LT.fd_check(a, m, sg, "chi", rng, f"{sub} inst20")
        ok &= g_ok
        psi_t, psi_c, psi = mt.state_of(a, m, sg, "chi", 0, 4096)
        E, Hp, nn = energy(m.H, psi)
        f = a.force(sg.project((Hp - E * psi) / nn), 4096)
        O = a.per_sample_O_proj(sg, np.arange(0, m.D, 7), psi_c, 8)
        z = max(np.abs(f[~a.mask_flat]).max(), np.abs(O[:, ~a.mask_flat]).max())
        log(f"check (iv) {sub}: max |force|, |O| on masked entries = {z:.1e}"); ok &= z == 0.0
    log(f"CHECK {'PASS' if ok else 'BUG'}")
    return ok


# ============================================================================ training
def train(args):
    t_start = time.process_time()
    m, W, Vv, deg, gap, psi0, sg, gen, dchi, dS = setup(args.instance, args.r_min)
    assert gap >= GAP_MIN, f"gap {gap} < {GAP_MIN}"
    log(f"instance {args.instance} {TOYS[args.instance]} D={m.D} E0={W[0]:.10f} gap={gap:.4f} |G|={sg.G} "
        f"chi={gen} dim_S_chi={dS:.0f} arm={args.arm} seed={args.seed}")
    rng_b = np.random.default_rng(args.seed + 1000003)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng_b.random(m.D))], args.lbfgs_iters)
    calib = float(Ec - W[0])
    refs = mt.references(m)
    a = make(args.arm, m, refs[1], args.F, args.Ha, args.seed)
    log(f"arm {args.arm}: F={args.F} Ha={args.Ha} n_eff={a.n_eff()} P/dim_S_chi={a.n_eff() / dS:.2f} "
        f"calib={calib:.2e}")
    rng = np.random.default_rng(args.seed + 1000 * RNG_OFF[args.arm])
    best_E, best_x, traj, best_hist = np.inf, a.flat().copy(), [], []
    stop, converged = "max-steps", False
    t0 = time.process_time()
    for it in range(args.max_steps):
        x0 = a.flat().copy()
        E, f, step = mt.sr_step(a, m, sg, args.proj, 0, args, rng)
        traj.append(E)
        if not np.isfinite(E):
            stop = f"non-finite energy at it {it}"; break
        if E < best_E:
            best_E, best_x = E, x0
        best_hist.append(best_E)
        a.set_flat(x0 - step)
        if it % 50 == 0:
            log(f"{args.arm}: it {it:5d} E-E0 = {E - W[0]:.6f} best = {best_E - W[0]:.6f} "
                f"({(time.process_time() - t0) / (it + 1):.3f} CPU-s/step)")
        if it + 1 >= args.min_steps and (it + 1) % 50 == 0:
            if best_hist[-1 - args.plateau_window] - best_E < args.plateau_tol:
                stop, converged = f"plateau at it {it + 1}", True
                break
        if time.process_time() - t0 > args.max_cpu:
            stop = f"CPU cap at it {it + 1}"; break
    a.set_flat(best_x)
    sr_cpu = time.process_time() - t0
    log(f"{args.arm}: stop = {stop}; best E - E0 = {best_E - W[0]:.6f}; SR CPU {sr_cpu:.0f} s")
    psi_t, psi_c, psi = mt.state_of(a, m, sg, args.proj, 0, args.chunk)
    if args.timing_only:
        return None
    b = budget(m, psi_c, W, Vv, deg, args, rng_b)
    for k_ in ("P0", "Ptheta", "Psign", "fn_history", "sign_starts"):
        b.pop(k_, None)
    masked = getattr(a, "max_masked_seen", 0.0)
    log(f"{args.arm} [P]: dE_tot={b['dE_tot']:.6f} dE_sign={b['dE_sign']:.6f} f_E={b['f_E_up']:.3f} "
        f"spread={b['spread']:.1e}; max |masked entry| proposed by SR = {masked:.1e}")
    return dict(instance=args.instance, V=m.V, D=m.D, gap=gap, arm=args.arm, seed=args.seed, F=args.F,
                Ha=args.Ha, n_eff=a.n_eff(), dim_S_chi=dS, G=sg.G, E0=float(W[0]), calib_err=calib,
                converged=converged, stop=stop, steps_done=len(traj), traj=traj, sr_cpu=sr_cpu,
                total_cpu=time.process_time() - t_start, max_masked=masked, budgets=dict(P=b),
                params={k: v for k, v in vars(args).items()})


# ============================================================================ verdict
def decide(R):
    """R: dict (arm, instance) -> result. Returns {variant: (verdict, reasons, table)}."""
    out = {}
    for var in VARIANTS:
        why, rows = [], []
        for inst in sorted(TOYS):
            fr, vr = R.get(("full", inst)), R.get((var, inst))
            if fr is None or vr is None:
                rows.append((inst, None, "missing")); continue
            bf, bv = fr["budgets"]["P"], vr["budgets"]["P"]
            V = fr["V"]
            if fr["gap"] < GAP_MIN:
                rows.append((inst, V, "not counted: gap")); continue
            bad = [lab for lab, r, b in (("full", fr, bf), (var, vr, bv))
                   if r["calib_err"] > CALIB_TOL * b["dE_tot"] or not (b["spread"] < SPREAD_TOL)]
            if bad:
                rows.append((inst, V, f"not counted: calibration/spread ({','.join(bad)})")); continue
            if not (fr["converged"] and vr["converged"]):
                rows.append((inst, V, "not counted: unconverged")); continue
            if bf["dE_sign"] < PREMISE:
                rows.append((inst, V, f"not counted: premise dE_sign(full) = {bf['dE_sign']:.2e} < {PREMISE}"))
                continue
            rows.append((inst, V, bv["dE_sign"] / bf["dE_sign"]))
        counted = [(V, r) for (_, V, r) in rows if isinstance(r, float)]
        n_over = sum(r > KILL_RATIO for _, r in counted)
        if n_over >= 2:
            out[var] = ("KILL", [f"rho > {KILL_RATIO} at {n_over} counted sizes"], rows); continue
        cs = sorted(counted)
        if len(cs) >= 3 and all(cs[i][1] < cs[i + 1][1] for i in range(len(cs) - 1)) and cs[-1][1] > TREND_RATIO:
            out[var] = ("KILL", [f"rho grows with V over {len(cs)} counted sizes and rho(V={cs[-1][0]}) = "
                                 f"{cs[-1][1]:.2f} > {TREND_RATIO}"], rows); continue
        if len(cs) < MIN_COUNTED:
            out[var] = ("INCONCLUSIVE", [f"{len(cs)} counted sizes < {MIN_COUNTED}"], rows); continue
        out[var] = ("PASS-TO-T2", [f"no kill clause on {len(cs)} counted sizes"], rows)
    return out


def load_all(results):
    R = {}
    for fn in sorted(os.listdir(results)):
        if fn.startswith("t1_") and fn.endswith(".json"):
            r = json.load(open(os.path.join(results, fn)))
            R[(r["arm"], r["instance"])] = r
    return R


def aggregate(args):
    R = load_all(args.results)
    print(f"{'inst':>4} {'V':>3} {'gap':>6} {'arm':>8} {'n_eff':>6} {'P/dS':>5} {'conv':>5} {'steps':>6} "
          f"{'dE_tot':>9} {'dE_sign':>9} {'f_E':>6} {'spread':>8} {'cpu_s':>6}")
    for inst in sorted(TOYS):
        for arm in ARMS:
            r = R.get((arm, inst))
            if r is None:
                continue
            b = r["budgets"]["P"]
            print(f"{inst:>4} {r['V']:>3} {r['gap']:6.3f} {arm:>8} {r['n_eff']:>6} {r['n_eff'] / r['dim_S_chi']:5.2f} "
                  f"{str(r['converged']):>5} {r['steps_done']:>6} {b['dE_tot']:9.2e} {b['dE_sign']:9.2e} "
                  f"{b['f_E_up']:6.3f} {b['spread']:8.1e} {r['total_cpu']:6.0f}")
    print("dense / full dE_sign (reported only):",
          {inst: round(R[("dense", inst)]["budgets"]["P"]["dE_sign"] / R[("full", inst)]["budgets"]["P"]["dE_sign"], 3)
           for inst in TOYS if ("dense", inst) in R and ("full", inst) in R
           and R[("full", inst)]["budgets"]["P"]["dE_sign"] > 0})
    print(f"total CPU-s: {sum(r['total_cpu'] for r in R.values()):.0f}")
    for var, (v, why, rows) in decide(R).items():
        print(f"{var}: {v}  ({'; '.join(why)})")
        for inst, V, r in rows:
            print(f"    inst {inst} V={V}: " + (f"rho = {r:.3f}" if isinstance(r, float) else r))


def synthetic(args):
    def res(arm, inst, ds, conv=True, gap=0.1, calib=0.0, spread=0.0):
        return dict(arm=arm, instance=inst, V=TOYS[inst]["Lx"] * TOYS[inst]["Ly"], gap=gap, converged=conv,
                    calib_err=calib, budgets=dict(P=dict(dE_sign=ds, dE_tot=0.05, spread=spread)))

    def case(full, half, quarter, kw=None):
        kw = kw or {}
        R = {}
        for inst, f, h, q in zip(sorted(TOYS), full, half, quarter):
            R[("full", inst)] = res("full", inst, f, **kw.get(inst, {}))
            R[("half", inst)] = res("half", inst, h)
            R[("quarter", inst)] = res("quarter", inst, q)
        return decide(R)
    cases = [
        ("both pass", case([.01] * 4, [.011] * 4, [.012] * 4), "PASS-TO-T2", "PASS-TO-T2"),
        ("half KILL ratio at 2", case([.01] * 4, [.013, .013, .01, .01], [.01] * 4), "KILL", "PASS-TO-T2"),
        ("ratio > 1.25 at 1 only", case([.01] * 4, [.013, .01, .01, .01], [.01] * 4), "PASS-TO-T2", "PASS-TO-T2"),
        ("quarter KILL trend", case([.01] * 4, [.01] * 4, [.010, .0105, .011, .012]), "PASS-TO-T2", "KILL"),
        ("trend but small", case([.01] * 4, [.01] * 4, [.010, .0103, .0106, .011]), "PASS-TO-T2", "PASS-TO-T2"),
        ("premise fails at 2 -> INCONCLUSIVE", case([.0005, .0005, .01, .01], [.01] * 4, [.01] * 4),
         "INCONCLUSIVE", "INCONCLUSIVE"),
        ("unconverged full at 2", case([.01] * 4, [.01] * 4, [.01] * 4, kw={20: dict(conv=False), 21: dict(conv=False)}),
         "INCONCLUSIVE", "INCONCLUSIVE"),
        ("gap fail + premise fail", case([.01, .0001, .01, .01], [.01] * 4, [.01] * 4, kw={20: dict(gap=0.01)}),
         "INCONCLUSIVE", "INCONCLUSIVE"),
        ("calibration fail at 1, kill at 2 others", case([.01] * 4, [.02] * 4, [.01] * 4, kw={20: dict(calib=1.0)}),
         "KILL", "PASS-TO-T2"),
    ]
    ok = True
    for name, out, eh, eq in cases:
        good = out["half"][0] == eh and out["quarter"][0] == eq
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {name}: half {out['half'][0]} (want {eh}), quarter {out['quarter'][0]} (want {eq})")
    print(f"SYNTHETIC {'PASS' if ok else 'FAIL'} ({len(cases)} cases)")
    return ok


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["check", "train", "aggregate", "synthetic"], required=True)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--instance", type=int, choices=sorted(TOYS))
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--F", type=int, default=4)
    p.add_argument("--Ha", type=int, default=0)
    p.add_argument("--proj", choices=["none", "chi"], default="none")
    p.add_argument("--timing-only", type=int, default=0)
    p.add_argument("--sr-lr", type=float, default=0.05)
    p.add_argument("--sr-shift", type=float, default=1e-3)
    p.add_argument("--sr-samples", type=int, default=1000)
    p.add_argument("--min-steps", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--plateau-window", type=int, default=200)
    p.add_argument("--plateau-tol", type=float, default=2e-4)
    p.add_argument("--max-cpu", type=float, default=480.0)
    p.add_argument("--lbfgs-iters", type=int, default=4000)
    p.add_argument("--fn-iters", type=int, default=10)
    p.add_argument("--fn-floor", type=float, default=1e-3)
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--o-chunk", type=int, default=8)
    p.add_argument("--r-min", type=float, default=1.5)
    p.add_argument("--results", default=os.path.join(HERE, "results"))
    p.add_argument("--overwrite", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "check":
        sys.exit(0 if run_check(args) else 2)
    if args.mode == "synthetic":
        sys.exit(0 if synthetic(args) else 2)
    if args.mode == "aggregate":
        aggregate(args); return
    os.makedirs(args.results, exist_ok=True)
    fn = os.path.join(args.results, f"t1_{args.arm}_inst{args.instance}_s{args.seed}.json")
    if os.path.exists(fn) and not args.overwrite:
        log(f"{fn} exists; refusing to overwrite"); return
    r = train(args)
    if r is None:
        return
    json.dump(r, open(fn, "w"))
    log(f"wrote {fn}")


if __name__ == "__main__":
    main()
