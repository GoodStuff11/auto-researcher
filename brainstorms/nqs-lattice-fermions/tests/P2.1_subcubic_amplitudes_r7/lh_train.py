#!/usr/bin/env python3
"""
P2.1 round-7 tier-2 (T2) pilot of Route 6.1' (LH-HFDS) on instance 4, with the redesign of the
round-6 per-leaf critic (Major 5): route arm = plus-shaped 5-site patch, foil = one-site features
(the Tocchio-Becca long-range lattice-backflow class), reference = projected NNBF width 6, seed 11,
the protocol of pilot job 457482. Decision at the pre-registered 25% checkpoint (75 projected SR
steps of a 300-step plan). The kill rule is in the leaf artifact (Round 7, "T2 rule").
=============================================================================================

Ansatz (LHAnsatz; the interface of ../P2.1_subcubic_amplitudes_r5/tj_core.LRAnsatz, so the
round-5 exact-sum projected SR machinery (micro_test.sr_step, P_chi projection, budget()) is
reused unchanged):
    psi(n) = exp(J(n)) prod_s det A_s(n),  A_s = Phi0_s[occ_s] + K_s[occ_s, :] C_s(n),
    C_s[m, :] = h_m(n) Omega_{s,m}   (Omega_{s,m}: F x N_s, site dependent),
    h_m = tanh(W x_patch(m) + b),    x_patch from lh_geom.patch_input (OBC-aware, 0 = outside),
    J = w2a . tanh(W1a x + b1a)      (x = 2n - 1, width Ha fills the parameter budget).
  Phi0 = UHF orbitals (micro_test.references), K ~ N(0, 1/V), Omega ~ 1e-3 N(0, 1): the start
  is the UHF Slater determinant (same reference as the NNBF arm).

Arms (all at P_target = parameter count of NNBF width 6 on the instance, 1626 on instance 4; F = 5;
Ha = round((P_target - base) / (2V + 2)), so P is matched within one Jastrow unit):
    task 0  lh_plus  plus-shaped 5-site patch (route arm)
    task 1  lh_site  one-site features (foil: lattice backflow with long-range K)
    task 2  nnbf     projected NNBF width 6, seed 11, RNG stream of pilot 457482 (reference rerun
                     to step 75; its step-0 energy must reproduce the pilot's -8.06278092)
    task 3  lh_3x3   3x3 patch (reported only)
  Controls used only by the pre-flight: sj (Slater x tanh Jastrow, Omega = 0 class), and the
  "identical partner" control (the nnbf result relabelled as the route).

Modes
  --mode check      geometry asserts; LHAnsatz amplitude == independent augmented-matrix amplitude
                    (t1_r7.SHHFDS with lam = 0) on random configurations; analytic gradients
                    against central finite differences (no projection, P_chi projection) on toy
                    instances 9 and 8; fast carried-table E_loc == (H psi)/psi on every
                    configuration of toy instance 8. Exit code 2 on any failure.
  --mode train      one task: --task {0,1,2,3} or --arm {lh_plus,lh_site,nnbf,lh_3x3,sj};
                    --instance (default 4); --sr-iters (default 75). Writes
                    <results>/t2_<arm>_inst<I>_s<seed>.json (trajectory, budgets, timing) and the
                    parameters. Pre-registered early stop for lh_* arms on instance 4: at SR step
                    50, stop if E(50) - E0 > 2 (E_pilot(50) - E0) (E_pilot(50) = -9.45132438).
  --mode aggregate  verdict from <results> (instance --instance) by the pre-registered rule.
  --mode synthetic  aggregate verdict on synthetic results covering every rule branch.
  --mode preflight  --part toy: check + synthetic + toy-instance pipeline (instance 7, 4x3 OBC
                    4+4: all arms and both controls, --toy-steps SR steps, short budgets);
                    --part inst4: instance-4 validity asserts and per-arm step timing
                    (--preflight-steps SR steps each); --part all (default): both.
Other options: --seed (11), --F (5), --results DIR (default ./results), --threads info only,
--sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3
(the pilot's values), --overwrite {0,1}.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lh_geom import check_geometry, patch_index, patch_input  # noqa: E402  (adds r5 to path)
import micro_test as mt  # noqa: E402
from p11_core import budget, energy, sign_constrained_min, solve_ed  # noqa: E402
from tj_core import TJ, LRAnsatz, connections  # noqa: E402

mt.INSTANCES[8] = dict(Lx=3, Ly=3, bcx="obc", bcy="obc", nup=3, ndn=3, tp=-0.25, J=0.4)   # FD toy
mt.INSTANCES[7] = dict(Lx=4, Ly=3, bcx="obc", bcy="obc", nup=4, ndn=4, tp=-0.25, J=0.4)   # control toy

ARMS = ["lh_plus", "lh_site", "nnbf", "lh_3x3"]
PATCH = {"lh_plus": "plus", "lh_site": "site", "lh_3x3": "3x3"}
RNG_OFFSET = {"nnbf": 0, "lh_plus": 101, "lh_site": 102, "lh_3x3": 103, "sj": 104}
# ---- pre-registered T2 rule (leaf, Round 7 "T2 rule") ----
KILL_NNBF, PROMOTE_NNBF, FOIL_MAX = 1.5, 1.15, 0.85
PREMISE_SIGN = 0.005          # reference dE_sign at the checkpoint must be >= 0.005 t
EARLY_STEP, EARLY_FACTOR = 50, 2.0
PILOT = {4: dict(E_it0=-8.06278092, E_it50=-9.45132438, E0=-9.791858330133278, dim_S_chi=15080.0,
                 G=4, n_params=1626, s_per_step_16cpu=5.20)}
CALIB_TOL, SPREAD_TOL = 1e-3, 0.05


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ ansatz
class LHAnsatz(LRAnsatz):
    """LH-HFDS (Route 6.1') with OBC-aware patches; see module docstring."""

    def __init__(self, model, ref, patch, F, Ha, seed=0, om_scale=1e-3):
        rng = np.random.default_rng(seed)
        m = model
        self.m, self.V, self.nu, self.nd = m, m.V, m.nup, m.ndn
        self.mode, self.K, self.r, self.Kref, self.trunk = "lh", 1, 0, 1, False
        self.F, self.Ha, self.Hs = F, Ha, 0
        self.P = patch_index(m.Lx, m.Ly, (m.bcx, m.bcy), patch)
        p_sz = self.P.shape[1]
        V = m.V
        Pu, Pd = ref
        p = {"P0u": Pu.copy(), "P0d": Pd.copy(),
             "Ku": rng.standard_normal((V, V)) / np.sqrt(V), "Kd": rng.standard_normal((V, V)) / np.sqrt(V),
             "Omu": om_scale * rng.standard_normal((V, F, m.nup)),
             "Omd": om_scale * rng.standard_normal((V, F, m.ndn)),
             "Wl": rng.standard_normal((2 * p_sz, F)) / np.sqrt(2 * p_sz), "bl": np.zeros(F)}
        if Ha > 0:
            p["W1a"] = rng.standard_normal((2 * V, Ha)) / np.sqrt(2 * V)
            p["b1a"] = np.zeros(Ha)
            p["w2a"] = np.zeros(Ha)
        self.p = p
        self.keys = list(p.keys())

    def forward(self, ou, od, ru, rd):
        p = self.p
        B = ou.shape[0]
        xp = patch_input(ou, od, self.P)                       # (B, V, 2p)
        h = np.tanh(xp @ p["Wl"] + p["bl"])                    # (B, V, F)
        c = dict(xp=xp, h=h, ru=ru, rd=rd, B=B, x=np.concatenate([ou, od], 1).astype(float) * 2 - 1)
        det = np.ones(B)
        for s, rows in (("u", ru), ("d", rd)):
            if rows.shape[1] == 0:
                continue
            C = np.swapaxes(np.matmul(np.swapaxes(h, 0, 1), p["Om" + s]), 0, 1)   # (B, V, N)
            Kr = p["K" + s][rows]                              # (B, N, V)
            A = p["P0" + s][rows] + Kr @ C
            c["C" + s], c["Kr" + s], c["A" + s] = C, Kr, A
            det = det * np.linalg.det(A)
        if self.Ha > 0:
            ha = np.tanh(c["x"] @ p["W1a"] + p["b1a"])
            J = ha @ p["w2a"]
            c["ha"] = ha
        else:
            J = np.zeros(B)
        c["J"], c["S"] = J, det
        c["psi"] = np.exp(J) * det
        return c

    def grad(self, c, coef, per_sample):
        p, B, V, F = self.p, c["B"], self.V, self.F
        bi = np.arange(B)[:, None]
        red = (lambda a: a) if per_sample else (lambda a: a.sum(0))
        tk = coef * c["psi"]
        g = {}
        dh = np.zeros((B, V, F))
        hT = np.swapaxes(c["h"], 0, 1)                            # (V, B, F)
        for s, rows in (("u", c["ru"]), ("d", c["rd"])):
            N = rows.shape[1]
            if N == 0:
                g["P0" + s] = np.zeros((B, V, 0)) if per_sample else np.zeros((V, 0))
                continue
            G = tk[:, None, None] * self._invT(c["A" + s])       # d psi / d A (B, N, N)
            gK = G @ np.swapaxes(c["C" + s], 1, 2)                # (B, N, V)
            dC = np.swapaxes(c["Kr" + s], 1, 2) @ G               # (B, V, N)
            if per_sample:
                Z = np.zeros((B, V, N)); Z[bi, rows] = G
                ZK = np.zeros((B, V, V)); ZK[bi, rows] = gK
                g["P0" + s], g["K" + s] = Z, ZK
                g["Om" + s] = np.einsum("bvf,bvn->bvfn", c["h"], dC)
            else:
                Sc = sp.csr_matrix((np.ones(rows.size), (rows.ravel(), np.arange(rows.size))), shape=(V, rows.size))
                Z = Sc @ G.reshape(-1, N)
                ZK = Sc @ gK.reshape(-1, V)
                g["P0" + s], g["K" + s] = Z, ZK
                g["Om" + s] = np.matmul(np.swapaxes(hT, 1, 2), np.swapaxes(dC, 0, 1))   # (V, F, N)
            dh += np.swapaxes(np.matmul(np.swapaxes(dC, 0, 1), np.swapaxes(p["Om" + s], 1, 2)), 0, 1)
        dpre = dh * (1 - c["h"] ** 2)                             # (B, V, F)
        if per_sample:
            g["Wl"] = np.einsum("bvi,bvf->bif", c["xp"], dpre)
            g["bl"] = dpre.sum(1)
        else:
            g["Wl"] = np.einsum("bvi,bvf->if", c["xp"], dpre)
            g["bl"] = dpre.sum((0, 1))
        if self.Ha > 0:
            x, ha = c["x"], c["ha"]
            dprea = tk[:, None] * p["w2a"][None] * (1 - ha ** 2)
            g["W1a"] = (x[:, :, None] * dprea[:, None, :]) if per_sample else x.T @ dprea
            g["b1a"] = red(dprea)
            g["w2a"] = red(tk[:, None] * ha)
        if per_sample:
            return np.concatenate([g[k].reshape(B, -1) for k in self.keys], 1)
        return np.concatenate([g[k].ravel() for k in self.keys])


def lh_count(m, patch, F, Ha):
    V, p_sz = m.V, patch_index(m.Lx, m.Ly, "obc", patch).shape[1]
    return V * (m.nup + m.ndn) + 2 * V * V + V * F * (m.nup + m.ndn) + 2 * p_sz * F + F + (2 * V + 2) * Ha


def make_arm(arm, m, refs, seed, F, match_width=6):
    target = LRAnsatz(m, refs[0], "full", K=1, Hs=match_width).n_params()
    if arm == "nnbf":
        a = LRAnsatz(m, refs[0], "full", K=1, Hs=match_width, seed=seed)
        return a, dict(mode="full", Hs=match_width), target
    if arm == "sj":
        Ha = int(round((target - LRAnsatz(m, refs[0], "lr", K=1, r=0, Ha=0).n_params()) / (2 * m.V + 2)))
        a = LRAnsatz(m, refs[0], "lr", K=1, r=0, Hs=0, Ha=Ha, seed=seed)
        return a, dict(mode="sj", Ha=Ha), target
    patch = PATCH[arm]
    base = lh_count(m, patch, F, 0)
    Ha = max(0, int(round((target - base) / (2 * m.V + 2))))
    a = LHAnsatz(m, refs[0], patch, F, Ha, seed=seed)
    assert a.n_params() == lh_count(m, patch, F, Ha)
    return a, dict(mode="lh", patch=patch, F=F, Ha=Ha), target


# ============================================================================ checks
def fd_check(a, m, sg, proj, rng, label, eps=1e-6):
    """Analytic energy gradient (exact force) against central finite differences, one direction."""
    def E_of(x):
        a.set_flat(x)
        v = mt.state_of(a, m, sg, proj, 0, 4096)[2]
        return energy(m.H, v)[0]
    x0 = a.flat().copy()
    a.set_flat(x0)
    psi_t, psi_c, psi = mt.state_of(a, m, sg, proj, 0, 4096)
    E, Hp, nn = energy(m.H, psi)
    rc = (Hp - E * psi) / nn
    if proj == "chi":    # P_chi is symmetric, idempotent: d<E>/dtheta = 2 sum (P (H-E) psi)/nn dpsi_t
        rc = sg.project(rc)
    f = 2 * a.force(rc, 4096)
    d = rng.standard_normal(len(x0))
    an = f @ d
    fd = (E_of(x0 + eps * d) - E_of(x0 - eps * d)) / (2 * eps)
    a.set_flat(x0)
    rel = abs(an - fd) / max(1e-12, abs(fd))
    log(f"check grad [{label}, proj={proj}]: analytic {an:.10e} fd {fd:.10e} rel {rel:.1e}")
    return rel < 1e-5


def run_check(seed=5):
    ok = check_geometry()
    log(f"check geometry (OBC patch indexing, no wrap-around): {ok}")
    rng = np.random.default_rng(seed)
    import t1_r7
    for inst in (9, 8):
        m = mt.make_model(inst)
        refs = mt.references(m)
        W, Vv, deg = solve_ed(m, 2, 1e-8, 1234)
        sg, gen = mt.build_symmetry(m, mt.resolve_ground(m, Vv, deg))
        for arm in ("lh_plus", "lh_site", "lh_3x3"):
            a, kw, tgt = make_arm(arm, m, refs, seed, 3)
            x = a.flat(); a.set_flat(x + 0.05 * rng.standard_normal(len(x)))   # generic point
            # (1) amplitude against an independent augmented-matrix build (t1_r7.SHHFDS, lam = 0)
            o = t1_r7.SHHFDS.__new__(t1_r7.SHHFDS)
            o.V, o.n, o.F, o.ops = m.V, m.nup, a.F, 0
            o.loc = t1_r7.Local(a.P, a.F, rng); o.loc.W, o.loc.b = a.p["Wl"], a.p["bl"]
            from lh_geom import neighbours4
            o.nb = neighbours4(m.Lx, m.Ly, (m.bcx, m.bcy))
            o.P0, o.K, o.Om = [a.p["P0u"], a.p["P0d"]], [a.p["Ku"], a.p["Kd"]], [a.p["Omu"], a.p["Omd"]]
            o.lam = [np.zeros((m.V, 4, a.F))] * 2
            idx = rng.choice(m.D, size=min(40, m.D), replace=False)
            ou, od, ru, rd = a._occ(idx)
            ps = a.forward(ou, od, ru, rd)["S"]
            pb = np.array([o.amp(ou[k], od[k]) for k in range(len(idx))])
            e_amp = float(np.max(np.abs(ps - pb)) / np.max(np.abs(pb)))
            log(f"check amplitude [{arm} inst {inst}] LHAnsatz vs augmented det: {e_amp:.1e}")
            ok &= e_amp < 1e-12
            for proj in ("none", "chi"):
                ok &= fd_check(a, m, sg, proj, rng, f"{arm} inst {inst} Ha={a.Ha}")
            # (2) projected per-sample O vs finite differences of log psi_P (one parameter direction)
            if inst == 8:
                psi_c = sg.project(a.amplitudes(4096))
                ii = rng.choice(m.D, size=6, replace=False)
                O = a.per_sample_O_proj(sg, ii, psi_c)
                d = rng.standard_normal(a.n_params()); x0 = a.flat().copy(); eps = 1e-6
                a.set_flat(x0 + eps * d); pp = sg.project(a.amplitudes(4096))[ii]
                a.set_flat(x0 - eps * d); pm = sg.project(a.amplitudes(4096))[ii]
                a.set_flat(x0)
                e_O = float(np.max(np.abs(O @ d - (pp - pm) / (2 * eps) / psi_c[ii])))
                log(f"check O_P [{arm}] vs fd: {e_O:.1e}")
                ok &= e_O < 1e-5
            # (3) fast carried-table E_loc (round-7 engine) == (H psi)/psi on every configuration
            if inst == 8:
                psi_t = a.forward(*a._occ(np.arange(m.D)))["S"]           # determinant part only
                Hp = m.H @ psi_t
                T = m.T; nn = m.nn
                emax = 0.0
                for k in range(m.D):
                    if abs(psi_t[k]) < 1e-8 * np.max(np.abs(psi_t)):
                        continue
                    oc = a._occ(np.array([k]))
                    u, d_ = oc[0][0], oc[1][0]
                    o.setup(u.astype(int), d_.astype(int))
                    diag, cl = t1_r7.conn_list(T, nn, u.astype(int), d_.astype(int))
                    Ef = diag + sum(cc * o.ratio(mm) for (cc, mm) in cl)
                    emax = max(emax, abs(Ef - Hp[k] / psi_t[k]) / (1 + abs(Hp[k] / psi_t[k])))
                log(f"check fast E_loc == (H psi)/psi on all {m.D} configs [{arm}, OBC 3x3]: {emax:.1e}")
                ok &= emax < 1e-9
    log("ALL CHECKS PASSED" if ok else "CHECK FAILED")
    return bool(ok)


# ============================================================================ train
def optimize_traj(a, m, sg, args, rng, label, early=None):
    """micro_test.optimize with a per-step energy trajectory and the pre-registered early stop."""
    best_E, best_x = np.inf, a.flat().copy()
    traj, stop = [], None
    t0 = time.time()
    for it in range(args.sr_iters):
        x0 = a.flat().copy()
        E, f, step = mt.sr_step(a, m, sg, "chi", 0, args, rng)
        traj.append(E)
        if not np.isfinite(E):
            a.set_flat(best_x); stop = f"non-finite energy at it {it}"; break
        if E < best_E:
            best_E, best_x = E, x0
        if early is not None and it == EARLY_STEP and E - early["E0"] > EARLY_FACTOR * (early["E_it50"] - early["E0"]):
            stop = (f"EARLY KILL at it {it}: E - E0 = {E - early['E0']:.5f} > "
                    f"{EARLY_FACTOR} x pilot NNBF {early['E_it50'] - early['E0']:.5f}")
            log(f"{label}: {stop}")
            break
        a.set_flat(x0 - step)
        if it % 25 == 0 or it == args.sr_iters - 1:
            log(f"{label}: it {it:5d}  E = {E:.8f}  |f| = {np.linalg.norm(f):.2e}  "
                f"({(time.time() - t0) / (it + 1):.2f} s/step)")
    E = energy(m.H, mt.state_of(a, m, sg, "chi", 0, args.chunk)[2])[0]
    if not (E <= best_E):
        a.set_flat(best_x)
    return traj, stop, (time.time() - t0) / max(len(traj), 1)


def train(args):
    m = mt.make_model(args.instance)
    m.spin_setup(); m.pair_setup(args.r_min)
    log(f"instance {args.instance} {mt.INSTANCES[args.instance]} D={m.D} arm={args.arm} seed={args.seed}")
    W, Vv, deg = solve_ed(m, 4, 1e-8, mt.UHF_SEED)
    psi0 = mt.resolve_ground(m, Vv, deg)
    sg, gen = mt.build_symmetry(m, psi0)
    S2_0 = m.S2_expect(psi0)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * S2_0) - 1)))
    dim_chi, dim_S_chi = mt.sector_dims(m, sg, S0)
    assert S0 == 0.0, "the projected SR here assumes a singlet ground state"
    pil = PILOT.get(args.instance)
    if pil is not None:   # validity asserts against the pilot record (inst4_nnbf_s11.json)
        assert abs(W[0] - pil["E0"]) < 1e-8, (W[0], pil["E0"])
        assert dim_S_chi == pil["dim_S_chi"] and sg.G == pil["G"], (dim_S_chi, sg.G)
    log(f"E0={W[0]:.10f} deg={deg} |G|={sg.G} chi={gen} dim_S_chi={dim_S_chi:.0f}")
    rng_b = np.random.default_rng(args.seed + 1000003)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng_b.random(m.D))], args.lbfgs_iters)
    calib = float(Ec - W[0])
    refs = mt.references(m)
    a, kw, target = make_arm(args.arm, m, refs, args.seed, args.F)
    if args.arm != "sj" and pil is not None:
        assert abs(a.n_params() - target) <= 2 * m.V + 2, (a.n_params(), target)
    log(f"arm {args.arm}: {kw} n_params={a.n_params()} target={target} calib={calib:.2e}")
    rng = np.random.default_rng(args.seed + 1000 * RNG_OFFSET[args.arm])
    early = None
    if args.arm.startswith("lh") and pil is not None:
        early = dict(E0=pil["E0"], E_it50=pil["E_it50"])
    traj, stop, sps = optimize_traj(a, m, sg, args, rng, args.arm, early)
    if args.arm == "nnbf" and pil is not None:
        d0 = abs(traj[0] - pil["E_it0"])
        log(f"reference reproduction: E(it 0) = {traj[0]:.8f} vs pilot {pil['E_it0']:.8f} (|d| = {d0:.1e})"
            + (f"; E(it 50) = {traj[50]:.8f} vs pilot {pil['E_it50']:.8f}" if len(traj) > 50 else ""))
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
                P_target=target, E0=float(W[0]), calib_err=calib, G=sg.G, dim_S_chi=dim_S_chi, irrep=gen,
                traj=traj, steps_done=len(traj), early_stop=stop, s_per_step=sps,
                threads=os.environ.get("OMP_NUM_THREADS", "?"), budgets=budgets,
                params={k: v for k, v in vars(args).items()})


# ============================================================================ verdict
def verdict(route, ref, foil, pilot_check=True):
    """Pre-registered T2 rule (leaf, Round 7). Inputs: result dicts. Returns (verdict, reasons, numbers)."""
    why = []
    for lab, r in (("route", route), ("reference", ref), ("foil", foil)):
        if r is None:
            return "INCOMPLETE", [f"{lab} result missing"], {}
        b = r["budgets"].get("P")
        if b is None or not np.isfinite(b["dE_sign"]):
            return "INVALID", [f"{lab}: no P budget"], {}
        if r["calib_err"] > CALIB_TOL * b["dE_tot"]:
            why.append(f"{lab}: calibration {r['calib_err']:.1e} > {CALIB_TOL} dE_tot")
        if not (b["spread"] < SPREAD_TOL):
            why.append(f"{lab}: multi-start spread {b['spread']:.3f} >= {SPREAD_TOL}")
    if pilot_check and route["instance"] in PILOT:
        pil = PILOT[route["instance"]]
        if abs(ref["traj"][0] - pil["E_it0"]) > 1e-6:
            why.append(f"reference does not reproduce pilot E(it 0): {ref['traj'][0]:.8f}")
        if ref["steps_done"] < ref["params"]["sr_iters"]:
            why.append("reference stopped early")
    if why:
        return "INVALID", why, {}
    if route.get("early_stop") and str(route["early_stop"]).startswith("EARLY KILL"):
        return "KILL", [route["early_stop"]], {}
    if foil.get("early_stop") and str(foil["early_stop"]).startswith("EARLY KILL"):
        why.append("foil early-stopped (its budget is taken at the stop point)")
    sr, sn, sf = (x["budgets"]["P"]["dE_sign"] for x in (route, ref, foil))
    nums = dict(dE_sign_route=sr, dE_sign_ref=sn, dE_sign_foil=sf, rho_N=sr / sn, rho_F=sr / sf,
                rho_t=route["budgets"]["P"]["dE_tot"] / ref["budgets"]["P"]["dE_tot"])
    if sn < PREMISE_SIGN:
        return "UNINFORMATIVE", why + [f"reference dE_sign {sn:.4f} < {PREMISE_SIGN}"], nums
    if nums["rho_N"] >= KILL_NNBF:
        return "KILL", why + [f"rho_N = {nums['rho_N']:.3f} >= {KILL_NNBF}"], nums
    if nums["rho_F"] > FOIL_MAX:
        return "KILL", why + [f"rho_F = {nums['rho_F']:.3f} > {FOIL_MAX} (does not beat the one-site foil)"], nums
    if nums["rho_N"] <= PROMOTE_NNBF:
        return "PROMOTE", why + [f"rho_N = {nums['rho_N']:.3f} <= {PROMOTE_NNBF} and rho_F = {nums['rho_F']:.3f} <= {FOIL_MAX}"], nums
    return "AMBIGUOUS", why + [f"{PROMOTE_NNBF} < rho_N = {nums['rho_N']:.3f} < {KILL_NNBF}; action: seed 22"], nums


def load(results, arm, inst, seed):
    f = os.path.join(results, f"t2_{arm}_inst{inst}_s{seed}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def aggregate(args):
    R = {arm: load(args.results, arm, args.instance, args.seed) for arm in ARMS + ["sj"]}
    for arm, r in R.items():
        if r is None:
            continue
        b = r["budgets"]["P"]
        log(f"{arm:8s} steps={r['steps_done']:3d} stop={r['early_stop']} P={r['n_params']} "
            f"dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} spread={b['spread']:.1e} "
            f"E(50)={r['traj'][50] if len(r['traj']) > 50 else float('nan'):.6f} s/step={r['s_per_step']:.2f}")
    v, why, nums = verdict(R["lh_plus"], R["nnbf"], R["lh_site"], pilot_check=args.instance in PILOT)
    log(f"ROUTE 6.1' (LH-HFDS) T2 VERDICT: {v}  " + "; ".join(why) + "  "
        + " ".join(f"{k}={val:.4g}" for k, val in nums.items()))
    if R["lh_3x3"] is not None and R["nnbf"] is not None:
        log(f"reported only: lh_3x3 rho_N = {R['lh_3x3']['budgets']['P']['dE_sign'] / R['nnbf']['budgets']['P']['dE_sign']:.3f}")
    return v


def synthetic():
    """Every branch of the rule from synthetic results."""
    def res(ds, dt=0.2, stop=None, inst=4, traj0=-8.06278092, spread=0.0, calib=0.0, steps=75):
        return dict(instance=inst, calib_err=calib, early_stop=stop, steps_done=steps, params=dict(sr_iters=75),
                    traj=[traj0] + [0.0] * 74, budgets=dict(P=dict(dE_sign=ds, dE_tot=dt, spread=spread)))
    cases = [("PROMOTE", res(0.020), res(0.019), res(0.030)),
             ("KILL", res(0.030), res(0.019), res(0.050)),                      # rho_N >= 1.5
             ("KILL", res(0.020), res(0.019), res(0.022)),                      # does not beat foil
             ("AMBIGUOUS", res(0.025), res(0.019), res(0.040)),
             ("KILL", res(0.020, stop="EARLY KILL at it 50: ..."), res(0.019), res(0.030)),
             ("UNINFORMATIVE", res(0.003), res(0.004), res(0.005)),
             ("INVALID", res(0.020), res(0.019, traj0=-8.0), res(0.030)),       # reference not reproduced
             ("INVALID", res(0.020, spread=0.2), res(0.019), res(0.030)),
             ("INVALID", res(0.020), res(0.019, steps=60), res(0.030)),
             ("INCOMPLETE", res(0.020), None, res(0.030))]
    ok = True
    for want, r, n, f in cases:
        got, why, _ = verdict(r, n, f)
        log(f"synthetic: expected {want:13s} got {got:13s} ({'; '.join(why)})")
        ok &= got == want
    log("SYNTHETIC: every branch returns its intended verdict" if ok else "SYNTHETIC: FAILED")
    return ok


def preflight(args):
    ok = True
    if args.part in ("all", "toy"):
        ok &= preflight_toy(args)
    if args.part in ("all", "inst4"):
        ok &= preflight_inst4(args)
    log("PREFLIGHT PASSED" if ok else "PREFLIGHT FAILED")
    return ok


def preflight_toy(args):
    ok = run_check()
    ok &= synthetic()
    # toy pipeline (instance 7, 4x3 OBC 4+4): every arm and both controls, short budget
    tdir = os.path.join(HERE, "results_preflight")
    os.makedirs(tdir, exist_ok=True)
    toy = {}
    for arm in ARMS + ["sj"]:
        ta = parse_args(["--mode", "train", "--instance", "7", "--arm", arm, "--sr-iters", str(args.toy_steps),
                         "--sr-samples", "400", "--lbfgs-iters", "1500", "--fn-iters", "4", "--results", tdir])
        toy[arm] = train(ta)
        json.dump(toy[arm], open(os.path.join(tdir, f"t2_{arm}_inst7_s{ta.seed}.json"), "w"), indent=1, default=float)
    pos, _, nums_p = verdict(toy["nnbf"], toy["nnbf"], toy["sj"], pilot_check=False)
    neg, _, nums_n = verdict(toy["sj"], toy["nnbf"], toy["lh_site"], pilot_check=False)
    real, why, nums_r = verdict(toy["lh_plus"], toy["nnbf"], toy["lh_site"], pilot_check=False)
    log(f"toy positive control (route := reference itself, foil := Slater x J): {pos} {nums_p}")
    log(f"toy negative control (route := Slater x J): {neg} {nums_n}")
    log(f"toy pipeline (lh_plus vs nnbf, foil lh_site; verdict meaningless at toy size): {real} {nums_r}")
    ok &= pos == "PROMOTE" and neg == "KILL"
    return ok


def preflight_inst4(args):
    """Instance-4 validity asserts and per-arm step timing."""
    ok = True
    m = mt.make_model(4)
    W, Vv, deg = solve_ed(m, 2, 1e-8, mt.UHF_SEED)
    psi0 = mt.resolve_ground(m, Vv, deg)
    sg, gen = mt.build_symmetry(m, psi0)
    m.spin_setup()
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * m.S2_expect(psi0)) - 1)))
    d_chi, d_S = mt.sector_dims(m, sg, S0)
    log(f"instance 4: E0={W[0]:.10f} (pilot {PILOT[4]['E0']:.10f}) |G|={sg.G} S={S0:g} dim_S_chi={d_S:.0f} deg={deg}")
    ok &= abs(W[0] - PILOT[4]["E0"]) < 1e-8 and sg.G == 4 and S0 == 0 and d_S == PILOT[4]["dim_S_chi"]
    refs = mt.references(m)
    ta = parse_args(["--mode", "train", "--instance", "4", "--arm", "nnbf"])
    times = {}
    for arm in ARMS:
        a, kw, tgt = make_arm(arm, m, refs, 11, args.F)
        rng = np.random.default_rng(11 + 1000 * RNG_OFFSET[arm])
        t0 = time.time()
        Es = []
        for _ in range(args.preflight_steps):
            E, f, step = mt.sr_step(a, m, sg, "chi", 0, ta, rng)
            a.set_flat(a.flat() - step); Es.append(E)
        times[arm] = (time.time() - t0) / args.preflight_steps
        log(f"instance 4 {arm:8s} {kw} P={a.n_params()} (target {tgt}) E(it0)={Es[0]:.8f} "
            f"{times[arm]:.2f} s/step at {os.environ.get('OMP_NUM_THREADS', '?')} threads")
        if arm == "nnbf":
            ok &= abs(Es[0] - PILOT[4]["E_it0"]) < 1e-6
            log(f"reference reproduces pilot E(it 0) = {PILOT[4]['E_it0']}: {abs(Es[0] - PILOT[4]['E_it0']) < 1e-6}")
        else:
            ok &= abs(a.n_params() - tgt) <= 2 * m.V + 2
    log(f"step-time ratio route/NNBF = {times['lh_plus'] / times['nnbf']:.2f} (budget rule: <= 1.5 for 16 CPUs)")
    return ok


# ============================================================================ cli
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["check", "train", "aggregate", "synthetic", "preflight"], default="check")
    ap.add_argument("--task", type=int, default=-1, choices=[-1, 0, 1, 2, 3])
    ap.add_argument("--arm", default="", choices=["", "lh_plus", "lh_site", "nnbf", "lh_3x3", "sj"])
    ap.add_argument("--instance", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--F", type=int, default=5)
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
    ap.add_argument("--toy-steps", type=int, default=40)
    ap.add_argument("--preflight-steps", type=int, default=1)
    ap.add_argument("--part", choices=["all", "toy", "inst4"], default="all")
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    a = ap.parse_args(argv)
    if a.task >= 0 and not a.arm:
        a.arm = ARMS[a.task]
    if a.mode == "train" and not a.arm:
        ap.error("--mode train needs --task or --arm")
    return a


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.results, exist_ok=True)
    if args.mode == "check":
        return 0 if run_check() else 2
    if args.mode == "synthetic":
        return 0 if synthetic() else 2
    if args.mode == "aggregate":
        aggregate(args); return 0
    if args.mode == "preflight":
        return 0 if preflight(args) else 2
    out = os.path.join(args.results, f"t2_{args.arm}_inst{args.instance}_s{args.seed}.json")
    if os.path.exists(out) and not args.overwrite:
        log(f"{out} exists; use --overwrite 1"); return 0
    res = train(args)
    json.dump(res, open(out, "w"), indent=1, default=float)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
