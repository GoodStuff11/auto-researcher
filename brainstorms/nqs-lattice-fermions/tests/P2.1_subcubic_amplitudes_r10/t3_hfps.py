#!/usr/bin/env python3
"""
P2.1 round 10: T3 decisive test of Route 7.1 (LH-HFPS, patch-local hidden rows in the pairing frame)
with the cubic pairing-frame baseline demanded by the Round 8 per-leaf critic (leaf artifact,
"Round 8 per-leaf critic on 7.1 PROMOTE" and "T3 rule, Route 7.1"). The T2 code
(../P2.1_subcubic_amplitudes_r7/t2_routes.py, lh_train.py and their imports) is imported READ-ONLY:
job 699305 uses it.
=================================================================================================

Instances (micro_test.INSTANCES): 0 = 4x4 OBC 6+6, 3 = 4x4 OBC 5+5, 4 = 5x3 OBC 6+6 (t-t'-J, t' = -0.25,
J = 0.4). Seeds 11, 22, 33. Protocol = round 5 / T2 (P_chi-projected exact-sum SR, step 0.05, shift 1e-3,
2000 samples; budget with 4000 L-BFGS and 10 fixed-node iterations) with --sr-iters 1200 steps (the
round-5 horizon, so the round-5 nnbf / agpj / skagp results on instances 0 and 3 are reused unchanged).

Arms (P_target = parameter count of NNBF width 6 on the instance: 1734, 1478, 1626 on 0, 3, 4):
  hfps_plus     route: HFPSAnsatz, plus patch, F = F_r(instance), Jastrow fill (Ha)
  hfps_site     foil (one-site features = Tocchio-Becca pairing backflow with long-range K)
  hfps_dense    CUBIC PAIRING-FRAME BASELINE (scored): the same ansatz with every site's feature vector
                computed from the WHOLE configuration (patch = all V sites, absolute order), i.e.
                F(n) = F0 + K C(n), C(n) = sum_f g_f(n) Omega_f, g = tanh(W x + b): a dense hidden-fermion
                pairing state / pairing-frame NNBF of width F. Every move changes every row of C, so a
                ratio needs a fresh N x N determinant (O(N^3) per connection). Same F, J_d and tanh
                Jastrow as the route; P is the route's within max(2V + 2, 5% of P_target), never below.
  hfps_dense_w  reported only: hfps_dense at width --F-wide (the VMC-cost-matched width from Round 8c);
                not trained when --F-wide <= F_r.
  nnbf          projected NNBF width 6 (Slater frame; lh_train.make_arm)
  agpj, skagp   round-5 arms (micro_test.arm_kwargs; K = 4, hs_sign = 6)
  F_r(instance) = the largest F <= 4 with base(plus, F, Ha = 0) <= P_target + 2V + 2 (4, 3, 4).

Array layout (--task, 0..44; see task_table()): new tasks only. Reused (not run): nnbf, agpj, skagp on
instances 0 and 3 (../P2.1_subcubic_amplitudes_r5/results/inst{0,3}_{arm}_s{seed}.json, 1200 steps) and
nnbf seed 11 on instance 4 (r5 results/inst4_nnbf_s11.json, pilot 457482, 1200 steps).

Modes
  --mode train       --task K (array index) or --arm A --instance I --seed S. Checkpoints every
                     --ckpt-every SR steps to <results>/ckpt/<tag>.npz (params, best, trajectory, RNG
                     state) and resumes from it; writes <results>/t3_<arm>_inst<I>_s<S>.json; refuses to
                     overwrite an existing result without --overwrite 1.
  --mode tasks       print the task table (index, arm, instance, seed) and exit.
  --mode check       DenseHFPSAnsatz: amplitude == explicit det(F0 + K C) on random configurations;
                     analytic gradients == central finite differences (no projection, P_chi) on toy
                     instances 9 and 8 (lh_train.fd_check); parameter counts on instances 0, 3, 4.
  --mode preflight   instance validity asserts and per-arm SR-step timing on --instance (default 4) for
                     --preflight-steps steps; prints E(it 0), P, s/step.
  --mode checkpoint  futility rule at a checkpoint step --at (300 or 600): reads <results>/ckpt and
                     prints KILL-EARLY / CONTINUE (pre-registered; the orchestrator cancels on KILL-EARLY).
  --mode aggregate   pre-registered T3 verdict from <results> (+ reused round-5 results).
  --mode synthetic   verdict on synthetic results covering every rule branch.
  --mode toy         pre-flight pipeline on toy instance 7: trains hfps_dense (30 steps, toy budget) into
                     ./results_preflight, reads the round-7 toy arms read-only, runs the positive and two
                     negative controls through the T3 rule and a synthetic futility test.
Options: --sr-iters (1200) --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 --lbfgs-iters 4000 --fn-iters 10
--fn-floor 1e-3 --chunk 32768 --o-chunk 8 --r-min 1.5 --ckpt-every 50 --F-wide (0 = arm off)
--results DIR (default ./results) --r5-results DIR (default ../P2.1_subcubic_amplitudes_r5/results)
--overwrite {0,1}. Threads: OMP/OPENBLAS/MKL_NUM_THREADS set by the sbatch file (<= 4 for local runs).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
R7 = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r7")
sys.path.insert(0, R7)
import t2_routes as T2  # noqa: E402  (read-only)
import lh_train as LT  # noqa: E402  (read-only)
from lh_geom import patch_index  # noqa: E402
import micro_test as mt  # noqa: E402
from p11_core import budget, energy, sign_constrained_min, solve_ed  # noqa: E402
from tj_core import LRAnsatz  # noqa: E402

log = LT.log
INSTANCES = (0, 3, 4)
SEEDS = (11, 22, 33)
NEW_ARMS = ("hfps_plus", "hfps_site", "hfps_dense")
RNG_OFF = {"hfps_plus": 201, "hfps_site": 202, "agpj": 203, "hfps_dense": 211, "hfps_dense_w": 212,
           "nnbf": 0, "skagp": 213}
# ---- pre-registered T3 rule (leaf, Round 10 "T3 rule, Route 7.1") ----
D_PASS, D_KILL, D_INST_MAX = 1.10, 1.30, 1.35      # rho_D (route / dense), pooled geometric mean
F_MAX = 0.90                                        # rho_F (route / one-site foil), pooled
N_PASS, N_KILL = 1.00, 1.15                         # rho_N (route / nnbf), pooled
PREMISE = 0.005                                     # dense baseline must keep dE_sign >= 0.005 t
CALIB_TOL, SPREAD_TOL = 1e-3, 0.05
FUT_RATIO = 1.5                                     # futility: (E_route - E0) / (E_dense - E0), medians
R5 = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r5", "results")


# ============================================================================ dense baseline
class DenseHFPSAnsatz(T2.HFPSAnsatz):
    """hfps_dense: HFPSAnsatz whose feature vector at every site is g(n) = tanh(W x + b) of the whole
    configuration (patch = all sites, absolute order). C(n) = sum_f g_f Omega[:, f, :]; cubic per ratio."""

    def __init__(self, model, F0, F, Ha, seed=0, om_scale=1e-3):
        super().__init__(model, F0, "site", F, Ha, seed=seed, om_scale=om_scale)
        V = model.V
        rng = np.random.default_rng(seed + 17)
        self.P = np.tile(np.arange(V, dtype=np.int64), (V, 1))
        self.p["Wl"] = rng.standard_normal((2 * V, F)) / np.sqrt(2 * V)
        self.keys = list(self.p.keys())

    # Fast path: the feature vector is the same at every site, so it is computed once per configuration
    # (the generic patch path would build a (B, V, 2V) input and is V times slower). Identical algebra to
    # HFPSAnsatz.forward / grad with patch = all sites; --mode check compares against an explicit build.
    def forward(self, ou, od, ru, rd):
        p, V = self.p, self.V
        B, N = ou.shape[0], ru.shape[1]
        xg = np.concatenate([ou, od], 1).astype(float) * 2 - 1                  # (B, 2V)
        g = np.tanh(xg @ p["Wl"] + p["bl"])                                    # (B, F)
        h = np.broadcast_to(g[:, None, :], (B, V, self.F))
        C = np.einsum("bf,mfv->bmv", g, p["Om"])                               # (B, V, V)
        Crd = np.take_along_axis(C, np.broadcast_to(rd[:, None, :], (B, V, N)), 2)
        Kr = p["K"][ru]
        A = p["F0"][ru[:, :, None], rd[:, None, :]] + Kr @ Crd
        det = np.linalg.det(A)
        J, Q, x, ha = self.jastrow(ou, od)
        return dict(xg=xg, h=h, ru=ru, rd=rd, B=B, Crd=Crd, Kr=Kr, A=A, Q=Q, x=x, ha=ha, J=J, S=det,
                    psi=np.exp(J) * det)

    def grad(self, c, coef, per_sample):
        """HFPSAnsatz.grad with the Wl / bl blocks taken through the shared feature vector."""
        p, B, V = self.p, c["B"], self.V
        ru, rd = c["ru"], c["rd"]
        bi = np.arange(B)[:, None]
        red = (lambda a: a) if per_sample else (lambda a: a.sum(0))
        tk = coef * c["psi"]
        G = tk[:, None, None] * self._invT(c["A"])
        gK = G @ np.swapaxes(c["Crd"], 1, 2)
        dCrd = np.swapaxes(c["Kr"], 1, 2) @ G
        dC = np.zeros((B, V, V))
        dC[bi, :, rd] = np.swapaxes(dCrd, 1, 2)
        g1 = c["h"][:, 0, :]                                                   # (B, F)
        g = {}
        if per_sample:
            Z = np.zeros((B, V, V)); Z[bi[:, :, None], ru[:, :, None], rd[:, None, :]] = G
            ZK = np.zeros((B, V, V)); ZK[bi, ru] = gK
            g["F0"], g["K"] = Z, ZK
            g["Om"] = np.einsum("bf,bmv->bmfv", g1, dC)
        else:
            fi = (ru[:, :, None] * V + rd[:, None, :]).ravel()
            g["F0"] = np.bincount(fi, weights=G.ravel(), minlength=V * V).reshape(V, V)
            Sc = sp.csr_matrix((np.ones(ru.size), (ru.ravel(), np.arange(ru.size))), shape=(V, ru.size))
            g["K"] = Sc @ gK.reshape(-1, V)
            g["Om"] = np.einsum("bf,bmv->mfv", g1, dC)
        dg = np.einsum("bmv,mfv->bf", dC, p["Om"])                             # sum over sites
        dpre = dg * (1 - g1 ** 2)                                              # (B, F)
        xg = c["xg"]
        if per_sample:
            g["Wl"] = xg[:, :, None] * dpre[:, None, :]
            g["bl"] = dpre
        else:
            g["Wl"] = xg.T @ dpre
            g["bl"] = dpre.sum(0)
        g["vJ"] = tk[:, None] * c["Q"] if per_sample else c["Q"].T @ tk
        if self.Ha > 0:
            x, ha = c["x"], c["ha"]
            dprea = tk[:, None] * p["w2a"][None] * (1 - ha ** 2)
            g["W1a"] = (x[:, :, None] * dprea[:, None, :]) if per_sample else x.T @ dprea
            g["b1a"] = red(dprea)
            g["w2a"] = red(tk[:, None] * ha)
        if per_sample:
            return np.concatenate([g[k].reshape(B, -1) for k in self.keys], 1)
        return np.concatenate([g[k].ravel() for k in self.keys])


def plus_base(m, F):
    return T2.hfps_count(m, "plus", F, 0)


def route_F(m, target):
    return max(F for F in (1, 2, 3, 4) if plus_base(m, F) <= target + 2 * m.V + 2)


def dense_count(m, F, Ha):
    V = m.V
    return 2 * V * V + V * F * V + 2 * V * F + F + V * (V + 1) // 2 + (2 * V + 2) * Ha


def make_arm(arm, m, refs, seed, F_wide=0):
    target = LRAnsatz(m, refs[0], "full", K=1, Hs=6).n_params()
    Fr = route_F(m, target)
    if arm == "nnbf":
        a, kw, t = LT.make_arm("nnbf", m, refs, seed, 5)
        return a, kw, target
    if arm in ("agpj", "skagp"):
        ns = argparse.Namespace(match_width=6, hs_sign=6, nnbfj_width=4, K=4)
        kw, _ = mt.arm_kwargs(arm, m, ns, refs)
        return LRAnsatz(m, refs[1], seed=seed, **kw), kw, target
    if arm in ("hfps_plus", "hfps_site"):
        patch = "plus" if arm == "hfps_plus" else "site"
        Ha = max(0, int(round((target - T2.hfps_count(m, patch, Fr, 0)) / (2 * m.V + 2))))
        a = T2.HFPSAnsatz(m, refs[1], patch, Fr, Ha, seed=seed)
        assert a.n_params() == T2.hfps_count(m, patch, Fr, Ha)
        return a, dict(mode="hfps", patch=patch, F=Fr, Ha=Ha), target
    F = Fr if arm == "hfps_dense" else F_wide
    Ha = max(0, int(round((target - dense_count(m, F, 0)) / (2 * m.V + 2))))
    a = DenseHFPSAnsatz(m, refs[1], F, Ha, seed=seed)
    assert a.n_params() == dense_count(m, F, Ha), (a.n_params(), dense_count(m, F, Ha))
    return a, dict(mode="hfps_dense", F=F, Ha=Ha), target


def p_ok(arm, P, target, V):
    if arm == "hfps_dense_w":
        return True
    tol = max(2 * V + 2, int(0.05 * target)) if arm == "hfps_dense" else 2 * V + 2
    return abs(P - target) <= tol


def task_table(F_wide=0):
    T = []
    for inst in INSTANCES:
        for arm in NEW_ARMS + (("hfps_dense_w",) if F_wide else ()):
            for s in SEEDS:
                T.append((arm, inst, s))
    for arm in ("nnbf", "agpj", "skagp"):
        for s in SEEDS:
            if not (arm == "nnbf" and s == 11):
                T.append((arm, 4, s))
    return T


# ============================================================================ train with checkpoints
def optimize_ckpt(a, m, sg, args, rng, tag):
    cdir = os.path.join(args.results, "ckpt"); os.makedirs(cdir, exist_ok=True)
    cf = os.path.join(cdir, tag + ".npz")
    best_E, best_x, traj, it0 = np.inf, a.flat().copy(), [], 0
    if os.path.exists(cf):
        z = np.load(cf, allow_pickle=True)
        a.set_flat(z["x"]); best_x = z["best_x"]; best_E = float(z["best_E"]); traj = list(z["traj"])
        rng.bit_generator.state = json.loads(str(z["rng"]))
        it0 = len(traj)
        log(f"{tag}: resumed from checkpoint at step {it0}")
    t0, stop = time.time(), None
    for it in range(it0, args.sr_iters):
        x0 = a.flat().copy()
        E, f, step = mt.sr_step(a, m, sg, "chi", 0, args, rng)
        traj.append(E)
        if not np.isfinite(E):
            a.set_flat(best_x); stop = f"non-finite energy at it {it}"; break
        if E < best_E:
            best_E, best_x = E, x0
        a.set_flat(x0 - step)
        if (it + 1) % args.ckpt_every == 0 or it == args.sr_iters - 1:
            np.savez(cf + ".tmp.npz", x=a.flat(), best_x=best_x, best_E=best_E, traj=np.array(traj),
                     rng=json.dumps(rng.bit_generator.state))
            os.replace(cf + ".tmp.npz", cf)
        if it % 50 == 0 or it == args.sr_iters - 1:
            log(f"{tag}: it {it:5d}  E = {E:.8f}  |f| = {np.linalg.norm(f):.2e}  "
                f"({(time.time() - t0) / (it + 1 - it0):.2f} s/step)")
    E = energy(m.H, mt.state_of(a, m, sg, "chi", 0, args.chunk)[2])[0]
    if not (E <= best_E):
        a.set_flat(best_x)
    return traj, stop, (time.time() - t0) / max(len(traj) - it0, 1)


def setup_instance(inst, args):
    m = mt.make_model(inst)
    m.spin_setup(); m.pair_setup(args.r_min)
    W, Vv, deg = solve_ed(m, 4, 1e-8, mt.UHF_SEED)
    psi0 = mt.resolve_ground(m, Vv, deg)
    sg, gen = mt.build_symmetry(m, psi0)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * m.S2_expect(psi0)) - 1)))
    assert S0 == 0.0, "singlet ground state assumed"
    dim_chi, dim_S_chi = mt.sector_dims(m, sg, S0)
    if inst in LT.PILOT:
        assert abs(W[0] - LT.PILOT[inst]["E0"]) < 1e-8 and dim_S_chi == LT.PILOT[inst]["dim_S_chi"]
    r5 = os.path.join(args.r5_results, f"inst{inst}_nnbf_s11.json")
    if os.path.exists(r5):
        assert sg.G == json.load(open(r5))["G"], "symmetry group differs from round 5"
    return m, W, Vv, deg, psi0, sg, gen, S0, dim_S_chi


def train(args):
    m, W, Vv, deg, psi0, sg, gen, S0, dS = setup_instance(args.instance, args)
    log(f"instance {args.instance} {mt.INSTANCES[args.instance]} D={m.D} E0={W[0]:.10f} |G|={sg.G} "
        f"dim_S_chi={dS:.0f} arm={args.arm} seed={args.seed}")
    e0f = os.path.join(args.results, f"E0_inst{args.instance}.json")
    if not os.path.exists(e0f):
        json.dump(dict(E0=float(W[0])), open(e0f, "w"))
    rng_b = np.random.default_rng(args.seed + 1000003)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng_b.random(m.D))], args.lbfgs_iters)
    calib = float(Ec - W[0])
    refs = mt.references(m)
    a, kw, target = make_arm(args.arm, m, refs, args.seed, args.F_wide)
    assert p_ok(args.arm, a.n_params(), target, m.V), (args.arm, a.n_params(), target)
    log(f"arm {args.arm}: {kw} n_params={a.n_params()} target={target} calib={calib:.2e}")
    rng = np.random.default_rng(args.seed + 1000 * RNG_OFF[args.arm])
    tag = f"t3_{args.arm}_inst{args.instance}_s{args.seed}"
    traj, stop, sps = optimize_ckpt(a, m, sg, args, rng, tag)
    psi_t, psi_c, psi = mt.state_of(a, m, sg, "chi", S0, args.chunk)
    budgets = {}
    for lab, v in (("P", psi_c),):
        b = budget(m, v, W, psi0[:, None] if deg > 1 else Vv, 1 if deg > 1 else deg, args, rng_b)
        vn = v / np.linalg.norm(v)
        b.update(ov0=float((psi0 @ vn) ** 2))
        for k_ in ("P0", "Ptheta", "Psign", "fn_history", "sign_starts"):
            b.pop(k_, None)
        budgets[lab] = b
        log(f"{args.arm} [{lab}]: dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} spread={b['spread']:.1e}")
    pdir = os.path.join(args.results, "params"); os.makedirs(pdir, exist_ok=True)
    np.save(os.path.join(pdir, tag + ".npy"), a.flat())
    return dict(instance=args.instance, arm=args.arm, seed=args.seed, arm_kwargs=kw, n_params=a.n_params(),
                P_target=target, E0=float(W[0]), calib_err=calib, G=sg.G, dim_S_chi=dS, traj=traj,
                steps_done=len(traj), early_stop=stop, s_per_step=sps, budgets=budgets,
                threads=os.environ.get("OMP_NUM_THREADS", "?"), params={k: v for k, v in vars(args).items()})


# ============================================================================ verdict
def load_all(args):
    R = {}
    for inst in INSTANCES:
        for s in SEEDS:
            for arm in NEW_ARMS + ("hfps_dense_w", "nnbf", "agpj", "skagp"):
                f = os.path.join(args.results, f"t3_{arm}_inst{inst}_s{s}.json")
                g = os.path.join(args.r5_results, f"inst{inst}_{arm}_s{s}.json")
                if os.path.exists(f):
                    R[(arm, inst, s)] = json.load(open(f))
                elif arm in ("nnbf", "agpj", "skagp") and os.path.exists(g):
                    r = json.load(open(g))
                    r.setdefault("steps_done", r["params"].get("sr_iters"))
                    R[(arm, inst, s)] = r
    return R


def decide(R, sr_iters=1200):
    """Pre-registered T3 rule. R: {(arm, inst, seed): result}. Returns (verdict, reasons, numbers)."""
    why = []
    req = [(a, i, s) for a in ("hfps_plus", "hfps_site", "hfps_dense", "nnbf") for i in INSTANCES for s in SEEDS]
    miss = [k for k in req if k not in R]
    if miss:
        return "INCOMPLETE", [f"missing {len(miss)} scored results, e.g. {miss[:3]}"], {}
    for k in req:
        r = R[k]; b = r["budgets"].get("P")
        if b is None or not np.isfinite(b["dE_sign"]):
            why.append(f"{k}: no P budget"); continue
        if r["calib_err"] > CALIB_TOL * b["dE_tot"]:
            why.append(f"{k}: calibration {r['calib_err']:.1e}")
        if not (b["spread"] < SPREAD_TOL):
            why.append(f"{k}: multi-start spread {b['spread']:.3f}")
        if k[0] != "hfps_plus" and r.get("early_stop"):
            why.append(f"{k}: stopped early ({r['early_stop']})")
        if (r.get("steps_done") or 0) < sr_iters and not r.get("early_stop"):
            why.append(f"{k}: {r.get('steps_done')} < {sr_iters} steps")
    if why:
        return "INVALID", why, {}
    med = {(a, i): float(np.median([R[(a, i, s)]["budgets"]["P"]["dE_sign"] for s in SEEDS]))
           for a in ("hfps_plus", "hfps_site", "hfps_dense", "nnbf") for i in INSTANCES}
    if any(R[("hfps_plus", i, s)].get("early_stop") for i in INSTANCES for s in SEEDS):
        why.append("a route task stopped early (non-finite); its best state is scored")
    rD = {i: med[("hfps_plus", i)] / med[("hfps_dense", i)] for i in INSTANCES}
    rF = {i: med[("hfps_plus", i)] / med[("hfps_site", i)] for i in INSTANCES}
    rN = {i: med[("hfps_plus", i)] / med[("nnbf", i)] for i in INSTANCES}
    gm = lambda d: float(np.exp(np.mean(np.log(list(d.values())))))  # noqa: E731
    nums = dict(gD=gm(rD), gF=gm(rF), gN=gm(rN), maxD=max(rD.values()),
                **{f"rD{i}": rD[i] for i in INSTANCES}, **{f"rF{i}": rF[i] for i in INSTANCES},
                **{f"rN{i}": rN[i] for i in INSTANCES})
    if sum(med[("hfps_dense", i)] < PREMISE for i in INSTANCES) >= 2:
        return "UNINFORMATIVE", why + ["dense baseline dE_sign < 0.005 t on >= 2 instances"], nums
    kills = []
    if nums["gD"] >= D_KILL:
        kills.append(f"rho_D = {nums['gD']:.3f} >= {D_KILL} (accuracy loss against the cubic pairing baseline)")
    if nums["gF"] > F_MAX:
        kills.append(f"rho_F = {nums['gF']:.3f} > {F_MAX} (patch does not beat one-site pairing backflow)")
    if nums["gN"] >= N_KILL:
        kills.append(f"rho_N = {nums['gN']:.3f} >= {N_KILL} (worse than Slater-frame NNBF)")
    if kills:
        return "KILL", why + kills, nums
    if nums["gD"] <= D_PASS and nums["maxD"] <= D_INST_MAX and nums["gN"] <= N_PASS:
        return "PASS", why + [f"rho_D = {nums['gD']:.3f} (max {nums['maxD']:.3f}), rho_F = {nums['gF']:.3f}, "
                              f"rho_N = {nums['gN']:.3f}"], nums
    return "AMBIGUOUS", why + [f"rho_D = {nums['gD']:.3f} (max {nums['maxD']:.3f}), rho_N = {nums['gN']:.3f}"], nums


def aggregate(args):
    R = load_all(args)
    for (a, i, s), r in sorted(R.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        b = r["budgets"]["P"]
        log(f"inst {i} {a:12s} s{s} P={r['n_params']} steps={r.get('steps_done')} dE_sign={b['dE_sign']:.5f} "
            f"dE_tot={b['dE_tot']:.5f} spread={b['spread']:.1e} s/step={r.get('s_per_step', float('nan')):.1f}")
    v, why, nums = decide(R, args.sr_iters)
    log(f"ROUTE 7.1 T3 VERDICT: {v}  " + "; ".join(why) + "  " + " ".join(f"{k}={x:.4g}" for k, x in nums.items()))
    for arm in ("agpj", "skagp", "hfps_dense_w"):
        rr = [R[(arm, i, s)]["budgets"]["P"]["dE_sign"] for i in INSTANCES for s in SEEDS if (arm, i, s) in R]
        if rr:
            log(f"reported only: {arm} n={len(rr)} median dE_sign={np.median(rr):.5f}")
    return v


def synthetic(args):
    def res(ds, stop=None, spread=0.0, steps=1200):
        return dict(calib_err=0.0, early_stop=stop, steps_done=steps, budgets=dict(P=dict(dE_sign=ds, dE_tot=0.1,
                                                                                         spread=spread)))

    def case(p, d, f, n, **kw):
        R = {}
        for i in INSTANCES:
            for s in SEEDS:
                R[("hfps_plus", i, s)] = res(p, **kw); R[("hfps_dense", i, s)] = res(d)
                R[("hfps_site", i, s)] = res(f); R[("nnbf", i, s)] = res(n)
        return R
    cases = [("PASS", case(0.010, 0.010, 0.013, 0.012)),
             ("KILL", case(0.014, 0.010, 0.020, 0.012)),          # rho_D 1.4
             ("KILL", case(0.010, 0.010, 0.0105, 0.012)),         # rho_F 0.95
             ("KILL", case(0.016, 0.015, 0.030, 0.012)),          # rho_N 1.33
             ("AMBIGUOUS", case(0.012, 0.010, 0.020, 0.013)),     # rho_D 1.2
             ("AMBIGUOUS", case(0.0125, 0.012, 0.020, 0.0115)),   # rho_N 1.09
             ("UNINFORMATIVE", case(0.004, 0.004, 0.006, 0.006)),
             ("INVALID", case(0.010, 0.010, 0.013, 0.012, spread=0.2)),
             ("INVALID", case(0.010, 0.010, 0.013, 0.012, steps=600))]
    R = case(0.010, 0.010, 0.013, 0.012); R[("hfps_dense", 3, 11)]["budgets"]["P"]["dE_sign"] = 0.005
    R[("hfps_dense", 3, 22)]["budgets"]["P"]["dE_sign"] = 0.005; R[("hfps_dense", 3, 33)]["budgets"]["P"]["dE_sign"] = 0.005
    cases.append(("AMBIGUOUS", R))                                # one instance rho_D 2.0, pooled 1.26
    R = case(0.010, 0.010, 0.013, 0.012); del R[("nnbf", 4, 33)]
    cases.append(("INCOMPLETE", R))
    ok = True
    for want, R in cases:
        got, why, nums = decide(R)
        log(f"synthetic: expected {want:13s} got {got:13s} ({'; '.join(why)[:150]})")
        ok &= got == want
    log("SYNTHETIC: every branch returns its intended verdict" if ok else "SYNTHETIC: FAILED")
    return ok


def checkpoint_rule(args):
    """Futility at step --at: KILL-EARLY if on >= 2 instances the seed-median of E_route(at) - E0 exceeds
    FUT_RATIO x the seed-median of E_dense(at) - E0 (best energy up to that step)."""
    cdir = os.path.join(args.results, "ckpt")
    E0 = {0: None, 3: None, 4: LT.PILOT[4]["E0"]}
    bad, have = 0, 0
    for inst in INSTANCES:
        g = {}
        for arm in ("hfps_plus", "hfps_dense"):
            v = []
            for s in SEEDS:
                f = os.path.join(cdir, f"t3_{arm}_inst{inst}_s{s}.npz")
                if os.path.exists(f):
                    tr = np.load(f, allow_pickle=True)["traj"]
                    if len(tr) >= args.at:
                        v.append(float(np.min(tr[:args.at])))
            g[arm] = v
        e0 = E0[inst]
        if e0 is None:
            r = os.path.join(args.results, f"E0_inst{inst}.json")
            e0 = json.load(open(r))["E0"] if os.path.exists(r) else None
        if e0 is None or len(g["hfps_plus"]) < 2 or len(g["hfps_dense"]) < 2:
            log(f"instance {inst}: not enough checkpoints at step {args.at} (or E0 unknown)"); continue
        have += 1
        q = (np.median(g["hfps_plus"]) - e0) / (np.median(g["hfps_dense"]) - e0)
        bad += q > FUT_RATIO
        log(f"instance {inst}: (E_route - E0)/(E_dense - E0) at step {args.at} = {q:.3f}")
    v = "KILL-EARLY" if bad >= 2 else ("CONTINUE" if have else "NO-DATA")
    log(f"FUTILITY at step {args.at}: {v} ({bad} of {have} instances above {FUT_RATIO})")
    return v


def toy(args):
    """Pre-flight pipeline on toy instance 7 (4x3 OBC 4+4): hfps_dense trained here (30 steps, toy budget);
    hfps_plus, hfps_site, agpj and nnbf read from the round-7 toy runs (read-only). Each toy result is
    replicated over the (instance, seed) keys of the T3 rule. Controls: positive (route := sign-exact state)
    must PASS; negative A (route := hfps_site) and B (route := agpj, no backflow) must KILL. Also a synthetic
    futility test on fabricated checkpoints (one KILL-EARLY, one CONTINUE)."""
    tdir = os.path.join(HERE, "results_preflight"); os.makedirs(tdir, exist_ok=True)
    r7 = {a: os.path.join(R7, d, f"t2_{a}_inst7_s11.json") for a, d in
          (("hfps_plus", "results_hfps_preflight"), ("hfps_site", "results_hfps_preflight"),
           ("agpj", "results_hfps_preflight"), ("nnbf", "results_preflight"))}
    T = {a: json.load(open(f)) for a, f in r7.items()}
    out = os.path.join(tdir, "t3_hfps_dense_inst7_s11.json")
    if os.path.exists(out) and not args.overwrite:
        T["hfps_dense"] = json.load(open(out))
    else:
        ta = parse_args(["--mode", "train", "--arm", "hfps_dense", "--instance", "7", "--seed", "11",
                         "--sr-iters", "30", "--sr-samples", "400", "--lbfgs-iters", "1500", "--fn-iters", "4",
                         "--results", tdir])
        T["hfps_dense"] = train(ta)
        json.dump(T["hfps_dense"], open(out, "w"), indent=1, default=float)
    # sign-exact positive control on instance 7
    m, W, Vv, deg, psi0, sg, gen, S0, dS = setup_instance(7, args)
    rng_b = np.random.default_rng(1000010)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng_b.random(m.D))], 1500)
    v = sg.project(psi0 * (1.0 + 0.5 * np.random.default_rng(7).random(m.D)))
    ta = parse_args(["--mode", "train", "--arm", "hfps_plus", "--lbfgs-iters", "1500", "--fn-iters", "4"])
    b = budget(m, v, W, psi0[:, None] if deg > 1 else Vv, 1 if deg > 1 else deg, ta, rng_b)
    for k_ in ("P0", "Ptheta", "Psign", "fn_history", "sign_starts"):
        b.pop(k_, None)
    T["sign_exact"] = dict(calib_err=float(Ec - W[0]), early_stop=None, steps_done=30, budgets=dict(P=b))
    for a, r in T.items():
        log(f"toy {a:11s} dE_sign={r['budgets']['P']['dE_sign']:.2e} dE_tot={r['budgets']['P']['dE_tot']:.5f} "
            f"spread={r['budgets']['P']['spread']:.1e} steps={r.get('steps_done')}")

    def rep(route):
        R = {}
        for i in INSTANCES:
            for s in SEEDS:
                for a in ("hfps_site", "hfps_dense", "nnbf"):
                    R[(a, i, s)] = dict(T[a], steps_done=30)
                R[("hfps_plus", i, s)] = dict(T[route], steps_done=30, early_stop=None)
        return R
    ok = True
    for lab, route, want in (("positive (route := sign-exact state)", "sign_exact", "PASS"),
                             ("negative A (route := hfps_site)", "hfps_site", "KILL"),
                             ("negative B (route := agpj, no backflow)", "agpj", "KILL"),
                             ("pipeline (hfps_plus; meaningless at toy size)", "hfps_plus", None)):
        v, why, nums = decide(rep(route), sr_iters=30)
        log(f"toy control {lab}: {v} ({'; '.join(why)})")
        if want is not None:
            ok &= v == want
    # futility rule on fabricated checkpoints
    fdir = os.path.join(tdir, "fut")
    for case, (er, ed, want) in {"kill": (0.30, 0.10, "KILL-EARLY"), "cont": (0.12, 0.10, "CONTINUE")}.items():
        cdir = os.path.join(fdir, case, "ckpt"); os.makedirs(cdir, exist_ok=True)
        for inst in INSTANCES:
            json.dump(dict(E0=-10.0), open(os.path.join(fdir, case, f"E0_inst{inst}.json"), "w"))
            for arm, gap in (("hfps_plus", er), ("hfps_dense", ed)):
                for s in SEEDS:
                    e0 = LT.PILOT[4]["E0"] if inst == 4 else -10.0
                    np.savez(os.path.join(cdir, f"t3_{arm}_inst{inst}_s{s}.npz"), traj=np.full(700, e0 + gap))
        fa = parse_args(["--mode", "checkpoint", "--at", "600", "--results", os.path.join(fdir, case)])
        got = checkpoint_rule(fa)
        log(f"futility synthetic [{case}]: expected {want} got {got}")
        ok &= got == want
    log("TOY PIPELINE AND CONTROLS: intended verdicts" if ok else "TOY PIPELINE: FAILED")
    return ok


def check(args):
    ok = True
    rng = np.random.default_rng(5)
    for inst in (9, 8):
        m = mt.make_model(inst)
        refs = mt.references(m)
        W, Vv, deg = solve_ed(m, 2, 1e-8, 1234)
        sg, gen = mt.build_symmetry(m, mt.resolve_ground(m, Vv, deg))
        a = DenseHFPSAnsatz(m, refs[1], 3, 1, seed=3)
        x = a.flat(); a.set_flat(x + 0.05 * rng.standard_normal(len(x)))
        idx = rng.choice(m.D, size=min(40, m.D), replace=False)
        ou, od, ru, rd = a._occ(idx)
        ps = a.forward(ou, od, ru, rd)["S"]
        p = a.p
        pb = []
        for k in range(len(idx)):
            xk = np.concatenate([2.0 * ou[k] - 1, 2.0 * od[k] - 1])
            gk = np.tanh(xk @ p["Wl"] + p["bl"])
            C = np.einsum("f,mfv->mv", gk, p["Om"])
            A = p["F0"][np.ix_(ru[k], rd[k])] + p["K"][ru[k]] @ C[:, rd[k]]
            pb.append(np.linalg.det(A))
        pb = np.array(pb)
        e = float(np.max(np.abs(ps - pb)) / np.max(np.abs(pb)))
        log(f"check dense amplitude vs explicit det(F0 + K C(g(n))) inst {inst}: {e:.1e}")
        ok &= e < 1e-12
        for proj in ("none", "chi"):
            ok &= LT.fd_check(a, m, sg, proj, rng, f"hfps_dense inst {inst}")
    for inst in INSTANCES:
        m = mt.make_model(inst)
        refs = mt.references(m)
        for arm in ("hfps_plus", "hfps_site", "hfps_dense", "nnbf", "agpj", "skagp"):
            a, kw, tgt = make_arm(arm, m, refs, 11)
            good = p_ok(arm, a.n_params(), tgt, m.V)
            ok &= good
            log(f"instance {inst} {arm:11s} {kw} P={a.n_params()} target={tgt} matched={good}")
    log("ALL CHECKS PASSED" if ok else "CHECK FAILED")
    return ok


def preflight(args):
    m, W, Vv, deg, psi0, sg, gen, S0, dS = setup_instance(args.instance, args)
    json.dump(dict(E0=float(W[0])), open(os.path.join(args.results, f"E0_inst{args.instance}.json"), "w"))
    log(f"instance {args.instance}: D={m.D} E0={W[0]:.10f} |G|={sg.G} S={S0:g} dim_S_chi={dS:.0f} deg={deg}")
    refs = mt.references(m)
    ta = parse_args(["--mode", "train", "--arm", "hfps_plus", "--instance", str(args.instance)])
    for arm in args.arm_list:
        a, kw, tgt = make_arm(arm, m, refs, 11, args.F_wide)
        rng = np.random.default_rng(11 + 1000 * RNG_OFF[arm])
        t0 = time.time(); Es = []
        for _ in range(args.preflight_steps):
            E, f, step = mt.sr_step(a, m, sg, "chi", 0, ta, rng)
            a.set_flat(a.flat() - step); Es.append(E)
        dt = (time.time() - t0) / args.preflight_steps
        log(f"instance {args.instance} {arm:12s} {kw} P={a.n_params()} (target {tgt}, matched "
            f"{p_ok(arm, a.n_params(), tgt, m.V)}) E(it0)={Es[0]:.8f} {dt:.2f} s/step at "
            f"{os.environ.get('OMP_NUM_THREADS', '?')} threads")
    return True


# ============================================================================ cli
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["train", "tasks", "check", "preflight", "checkpoint", "aggregate", "toy",
                                       "synthetic"], default="check")
    ap.add_argument("--task", type=int, default=-1)
    ap.add_argument("--arm", default="", choices=["", "hfps_plus", "hfps_site", "hfps_dense", "hfps_dense_w",
                                                  "nnbf", "agpj", "skagp"])
    ap.add_argument("--arms", default="hfps_plus,hfps_site,hfps_dense,nnbf,agpj,skagp")
    ap.add_argument("--instance", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--F-wide", type=int, default=0)
    ap.add_argument("--sr-iters", type=int, default=1200)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--o-chunk", type=int, default=8)
    ap.add_argument("--lbfgs-iters", type=int, default=4000)
    ap.add_argument("--fn-iters", type=int, default=10)
    ap.add_argument("--fn-floor", type=float, default=1e-3)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--at", type=int, default=600)
    ap.add_argument("--preflight-steps", type=int, default=2)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--r5-results", default=R5)
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    a = ap.parse_args(argv)
    a.arm_list = [x for x in a.arms.split(",") if x]
    if a.task >= 0:
        a.arm, a.instance, a.seed = task_table(a.F_wide)[a.task]
    if a.mode == "train" and not a.arm:
        ap.error("--mode train needs --task or --arm")
    return a


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.results, exist_ok=True)
    if args.mode == "tasks":
        for k, t in enumerate(task_table(args.F_wide)):
            print(k, *t)
        return 0
    if args.mode == "check":
        return 0 if check(args) else 2
    if args.mode == "synthetic":
        return 0 if synthetic(args) else 2
    if args.mode == "preflight":
        return 0 if preflight(args) else 2
    if args.mode == "checkpoint":
        checkpoint_rule(args); return 0
    if args.mode == "toy":
        return 0 if toy(args) else 2
    if args.mode == "aggregate":
        aggregate(args); return 0
    out = os.path.join(args.results, f"t3_{args.arm}_inst{args.instance}_s{args.seed}.json")
    if os.path.exists(out) and not args.overwrite:
        log(f"{out} exists; use --overwrite 1"); return 0
    res = train(args)
    json.dump(res, open(out, "w"), indent=1, default=float)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
