#!/usr/bin/env python3
"""
P2.1 Round 15, route R15.6, T1 kill-only: learned sparse gating of hidden rows at V = 12.
=========================================================================================

Pre-registered rule: RULE.md in this directory (written before any train run). Summary: train the gated ansatz on
instance 23 (4x3 OBC 5+5) with the Round 12 protocol (unprojected exact-force SR, lr 0.05, shift 1e-3, 1000 metric
samples, seed 11, F = 4, plateau test, SR CPU cap 480 s) plus a dual-ascent L1 gate penalty holding the exact
active fraction at 0.5; score one unchanged budget; rho(12) = dE_sign / dE_sign(full_F4 = 7.95e-3).
KILL if rho > 1.59, or rho > 1.25 and converged; INCONCLUSIVE if the scored active fraction > 0.55 or no plateau in
480 SR CPU-s; otherwise no kill (go to the T0 cost check).

Ansatz (SparseGateAnsatz):
    psi = e^J det(F0 + sum_m K[:, m] g_m(n) h_m(n) . Omega_m)[up, dn],   g_m = ReLU(w . x_m + c_{s(m)})
x_m: plus-patch input of row m (10 entries, +-1 or 0 off-lattice); w: 10 shared weights ("gw"); c: one bias per
checkerboard sublattice s = (x + y) mod 2 ("gc"). All V rows exist; HFPSAnsatz parameters start from the seed-11
values used by full_F4; gw ~ N(0, 1/10) from rng(seed + 7), gc = 0.

Imported read-only (nothing upstream is edited): ../P2.1_subcubic_amplitudes_r12/t1_sub.py (SubHFPSAnsatz, setup,
make, parse_args defaults, constants), and through it t2_routes, micro_test, p11_core (budget, energy,
sign_constrained_min), lh_geom.patch_input, lh_train.fd_check.

Modes
  --mode check      (i) amplitudes == an explicit site-by-site gated determinant build (rel <= 1e-12) on 40 random
                    configurations of instances 20 and 23 with perturbed parameters; (ii) gate forced open
                    (w = 0, c = (1, 1)) reproduces t2_routes.HFPSAnsatz bit for bit (instance 21); (iii) w = 0,
                    c = (1, -1) reproduces t1_sub's `half` amplitudes (rel <= 1e-12, instance 23); (iv) analytic
                    energy gradient == central finite differences (lh_train.fd_check, proj none, rel <= 1e-5,
                    instance 20); (v) analytic penalty gradient == central finite differences (rel <= 1e-5).
                    Exit 2 on failure.
  --mode probe      --probe-steps N: N penalised SR steps on instance 23, prints CPU-s per step. No output file.
  --mode train      Trains, scores, records hop statistics, writes results/t1_sgate_inst23_s11.json (refuses to
                    overwrite unless --overwrite 1) and prints the verdict.
  --mode verdict    Re-prints the verdict from the result file.
  --mode synthetic  The verdict function on synthetic inputs covering every branch. Exit 2 on failure.
Controller (RULE.md addendum A): --controller project (default, run 2: no penalty; after every SR step both
  sublattice biases are shifted by one delta so the exact active fraction under the pre-step |psi|^2 is 0.5) or
  dual (run 1: L1 gate penalty with lambda <- lambda * lam-mult^sign(a - target); unstable, kept for the record).
Options: --instance (23; 20 allowed for a smoke run, which is never scored) --max-cpu 480 (SR CPU-s cap)
  --max-steps 6000 --lam0 1e-3 --lam-mult 1.2 --target 0.5 --act-max 0.55 --overwrite {0,1}
  --results DIR (./results). Every other option is t1_sub's pre-registered default.
Threads: OMP/OPENBLAS/MKL_NUM_THREADS=1; run under `ulimit -t 600`.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
R12 = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r12")
R12B = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r12b")
sys.path.insert(0, R12)
import t1_sub as TS  # noqa: E402  (read-only)
from lh_geom import patch_input  # noqa: E402
from p11_core import budget, energy, sign_constrained_min, solve_ed  # noqa: E402

T2, mt, LT, log = TS.T2, TS.mt, TS.LT, TS.log
REF_FULL = os.path.join(R12B, "results", "t1_full_inst23_s11.json")
HALF_RHO_23 = 1.587
KILL_HALF, KILL_CONV = 1.59, 1.25
RNG_OFF = 307


class SparseGateAnsatz(TS.SubHFPSAnsatz):
    def __init__(self, model, F0, F, Ha, seed=0):
        super().__init__(model, F0, F, Ha, "full", seed=seed)
        xs, ys = np.arange(model.V) % model.Lx, np.arange(model.V) // model.Lx
        self.sub = ((xs + ys) % 2).astype(int)                 # s(m): 0 on x+y even, 1 on x+y odd
        npi = 2 * self.P.shape[1]
        rg = np.random.default_rng(seed + 7)
        self.p["gw"] = rg.standard_normal(npi) / np.sqrt(npi)
        self.p["gc"] = np.zeros(2)
        self.keys = list(self.p.keys())
        self.mask_flat = np.concatenate([self.mask_flat, np.ones(npi + 2, bool)])
        self.sub_name = "sgate"

    def gate_z(self, xp):
        return xp @ self.p["gw"] + self.p["gc"][self.sub][None, :]

    def forward(self, ou, od, ru, rd):
        p, V = self.p, self.V
        B, N = ou.shape[0], ru.shape[1]
        xp = patch_input(ou, od, self.P)
        hr = np.tanh(xp @ p["Wl"] + p["bl"])
        z = self.gate_z(xp)
        g = np.maximum(z, 0.0)
        h = hr * g[:, :, None]
        C = np.swapaxes(np.matmul(np.swapaxes(h, 0, 1), p["Om"]), 0, 1)
        Crd = np.take_along_axis(C, np.broadcast_to(rd[:, None, :], (B, V, N)), 2)
        Kr = p["K"][ru]
        A = p["F0"][ru[:, :, None], rd[:, None, :]] + Kr @ Crd
        det = np.linalg.det(A)
        J, Q, x, ha = self.jastrow(ou, od)
        return dict(xp=xp, h=h, hr=hr, g=g, z=z, ru=ru, rd=rd, B=B, Crd=Crd, Kr=Kr, A=A, Q=Q, x=x, ha=ha, J=J,
                    S=det, psi=np.exp(J) * det)

    def grad(self, c, coef, per_sample):
        """t2_routes.HFPSAnsatz.grad with h -> g h, plus the gate parameters gw, gc."""
        p, B, V = self.p, c["B"], self.V
        ru, rd = c["ru"], c["rd"]
        bi = np.arange(B)[:, None]
        tk = coef * c["psi"]
        G = tk[:, None, None] * self._invT(c["A"])
        gK = G @ np.swapaxes(c["Crd"], 1, 2)
        dCrd = np.swapaxes(c["Kr"], 1, 2) @ G
        dC = np.zeros((B, V, V))
        dC[bi, :, rd] = np.swapaxes(dCrd, 1, 2)
        h = c["h"]
        g = {}
        if per_sample:
            Z = np.zeros((B, V, V)); Z[bi[:, :, None], ru[:, :, None], rd[:, None, :]] = G
            ZK = np.zeros((B, V, V)); ZK[bi, ru] = gK
            g["F0"], g["K"] = Z, ZK
            g["Om"] = np.einsum("bmf,bmv->bmfv", h, dC)
        else:
            fi = (ru[:, :, None] * V + rd[:, None, :]).ravel()
            g["F0"] = np.bincount(fi, weights=G.ravel(), minlength=V * V).reshape(V, V)
            Sc = sp.csr_matrix((np.ones(ru.size), (ru.ravel(), np.arange(ru.size))), shape=(V, ru.size))
            g["K"] = Sc @ gK.reshape(-1, V)
            g["Om"] = np.matmul(np.swapaxes(np.swapaxes(h, 0, 1), 1, 2), np.swapaxes(dC, 0, 1))
        dh = np.swapaxes(np.matmul(np.swapaxes(dC, 0, 1), np.swapaxes(p["Om"], 1, 2)), 0, 1)   # d/d(g h)
        dpre = dh * (1 - c["hr"] ** 2) * c["g"][:, :, None]
        dz = (dh * c["hr"]).sum(-1) * (c["z"] > 0)                                              # (B, V)
        if per_sample:
            g["Wl"] = np.einsum("bvi,bvf->bif", c["xp"], dpre)
            g["bl"] = dpre.sum(1)
            g["gw"] = np.einsum("bvi,bv->bi", c["xp"], dz)
            g["gc"] = np.stack([dz[:, self.sub == s].sum(1) for s in (0, 1)], 1)
        else:
            g["Wl"] = np.einsum("bvi,bvf->if", c["xp"], dpre)
            g["bl"] = dpre.sum((0, 1))
            g["gw"] = np.einsum("bvi,bv->i", c["xp"], dz)
            g["gc"] = np.array([dz[:, self.sub == s].sum() for s in (0, 1)])
        g["vJ"] = tk[:, None] * c["Q"] if per_sample else c["Q"].T @ tk
        assert self.Ha == 0
        if per_sample:
            return np.concatenate([g[k].reshape(B, -1) for k in self.keys], 1)
        return np.concatenate([g[k].ravel() for k in self.keys])


class GateTables:
    """Parameter-independent patch inputs of every basis state, for exact gate statistics."""

    def __init__(self, a, chunk=8192):
        D = a.m.D
        ou, od, _, _ = a._occ(np.arange(D))
        self.ou, self.od = ou, od
        self.xp = patch_input(ou, od, a.P)                       # (D, V, 10)

    def stats(self, a):
        z = self.xp @ a.p["gw"] + a.p["gc"][a.sub][None, :]
        act = z > 0
        return np.maximum(z, 0.0), act                          # (D, V) gate values, active mask


def penalty_parts(a, tab, psi, nn):
    """A = <mean_m g_m>, a = <mean_m 1[g>0]> under |psi|^2; rcoef for the score part; explicit gate gradient."""
    gv, act = tab.stats(a)
    V = a.V
    Gbar = gv.mean(1)
    w = psi ** 2 / nn
    A = float(w @ Gbar)
    afrac = float(w @ act.mean(1))
    r_pen = psi * (Gbar - A) / nn                               # 2 force(r_pen) = score part of dA/dtheta
    wa = (w[:, None] * act) / V                                 # (D, V)
    ex = {k: np.zeros_like(v) for k, v in a.p.items()}
    ex["gw"] = np.einsum("dvi,dv->i", tab.xp, wa)
    ex["gc"] = np.array([wa[:, a.sub == s].sum() for s in (0, 1)])
    ex_flat = np.concatenate([ex[k].ravel() for k in a.keys])
    return A, afrac, r_pen, ex_flat


def project_bias(a, tab, psi, target):
    """Shift both sublattice biases by one delta so the exact active fraction under |psi|^2 is <= target (quantile)."""
    w = psi ** 2 / (psi @ psi)
    z = tab.xp @ a.p["gw"] + a.p["gc"][a.sub][None, :]
    zz = z.ravel()
    ww = np.repeat(w / a.V, a.V)
    o = np.argsort(-zz)
    cw = np.cumsum(ww[o])
    k = int(np.searchsorted(cw, target, side="right"))          # entries o[:k] stay active, weight <= target
    thr = zz[o[k]] if k < len(o) else zz[o[-1]] - 1.0           # largest z that must become inactive
    delta = -thr if k < len(o) else 0.0
    a.p["gc"] = a.p["gc"] + delta
    return float(delta)


def sr_step_pen(a, m, tab, lam, args, rng):
    """micro_test.sr_step (proj none) with the force of E + lam * A. Returns E, A, a, step."""
    psi = a.amplitudes(args.chunk)
    E, Hp, nn = energy(m.H, psi)
    A, afrac, r_pen, ex = penalty_parts(a, tab, psi, nn)
    f = a.force((Hp - E * psi) / nn + lam * r_pen, args.chunk) + 0.5 * lam * ex
    pc = psi ** 2
    idx = rng.choice(len(pc), size=args.sr_samples, p=pc / pc.sum())
    O = a.per_sample_O(idx)
    Oc = (O - O.mean(0)) / np.sqrt(len(idx))
    sh = args.sr_shift
    Km = Oc @ Oc.T + sh * np.eye(len(idx))
    dx = (f - Oc.T @ np.linalg.solve(Km, Oc @ f)) / sh
    step = args.sr_lr * dx
    nrm = np.linalg.norm(dx)
    if nrm * args.sr_lr > 1.0:
        step *= 1.0 / (nrm * args.sr_lr)
    return E, A, afrac, step, psi


# ============================================================================ hop statistics
def hop_stats(a, m, tab, psi):
    """Exact expectations over accepted single-electron hops (MH, uniform proposal over H-connected hops)."""
    Hc = m.H.tocoo()
    i, j = Hc.row, Hc.col
    nt = (tab.ou + tab.od)
    keep = (i != j) & np.any(nt[i] != nt[j], 1)                 # charge moves only (not spin exchange)
    i, j = i[keep], j[keep]
    deg = np.bincount(i, minlength=m.D).astype(float)
    pi = psi ** 2 / (psi @ psi)
    wgt = np.minimum(pi[i] / deg[i], pi[j] / deg[j])            # pi_i q(j|i) A(i->j)
    changed = (tab.ou[i] != tab.ou[j]) | (tab.od[i] != tab.od[j])   # (M, V) sites whose occupation changes
    Inc = np.zeros((m.V, m.V), bool)                            # Inc[s, row] = s in patch(row)
    for row in range(m.V):
        for s in a.P[row]:
            if s >= 0:
                Inc[s, row] = True
    touched = (changed.astype(int) @ Inc.astype(int)) > 0       # (M, V) rows whose input changes
    _, act = tab.stats(a)
    ai, aj = act[i], act[j]
    rows_changed = (touched & (ai | aj)).sum(1)
    rows_both = (touched & ai & aj).sum(1)
    flips = (ai != aj).sum(1)
    b_full = touched.sum(1)
    Z = wgt.sum()
    ev = lambda q: float(wgt @ q / Z)                           # noqa: E731
    acc_rate = float(Z)                                          # sum_i pi_i sum_j q A = acceptance per proposal
    return dict(rows_changed=ev(rows_changed), rows_active_both=ev(rows_both), gate_flips=ev(flips),
                rows_touched_full=ev(b_full), acceptance=acc_rate, n_moves=int(len(i)),
                rows_changed_over_full=ev(rows_changed) / ev(b_full))


# ============================================================================ verdict
def verdict(r, ref):
    bf, bv = ref["budgets"]["P"], r["budgets"]["P"]
    rho = bv["dE_sign"] / bf["dE_sign"]
    guards = []
    for lab, rr, b in (("full_F4", ref, bf), ("sgate", r, bv)):
        if rr["calib_err"] > TS.CALIB_TOL * b["dE_tot"] or not (b["spread"] < TS.SPREAD_TOL):
            guards.append(lab)
    if rho > KILL_HALF:
        return "KILL", rho, f"rho(12) = {rho:.3f} > {KILL_HALF} (no better than half_F4, rho {HALF_RHO_23})"
    if rho > KILL_CONV and r["converged"]:
        return "KILL", rho, f"rho(12) = {rho:.3f} > {KILL_CONV} and converged"
    if guards:
        return "INCONCLUSIVE", rho, f"calibration/spread guard failed ({','.join(guards)})"
    if r["act_scored"] > 0.55:
        return "INCONCLUSIVE", rho, f"scored active fraction {r['act_scored']:.3f} > 0.55"
    if not r["converged"]:
        return "INCONCLUSIVE", rho, f"no plateau within {r['params']['max_cpu']:.0f} SR CPU-s (rho {rho:.3f})"
    return "NO-KILL", rho, f"rho(12) = {rho:.3f} <= {KILL_CONV}, converged: proceed to the T0 cost check"


def synthetic(_):
    ref = dict(calib_err=0.0, budgets=dict(P=dict(dE_sign=7.95e-3, dE_tot=0.047, spread=0.0)))

    def res(rho, conv=True, act=0.5, spread=0.0):
        return dict(calib_err=0.0, converged=conv, act_scored=act, params=dict(max_cpu=480.0),
                    budgets=dict(P=dict(dE_sign=rho * 7.95e-3, dE_tot=0.05, spread=spread)))
    cases = [("rho 1.7 unconverged", res(1.7, conv=False), "KILL"),
             ("rho 1.4 converged", res(1.4), "KILL"),
             ("rho 1.4 unconverged", res(1.4, conv=False), "INCONCLUSIVE"),
             ("rho 1.0 converged", res(1.0), "NO-KILL"),
             ("rho 1.0 active 0.6", res(1.0, act=0.6), "INCONCLUSIVE"),
             ("rho 1.0 unconverged", res(1.0, conv=False), "INCONCLUSIVE"),
             ("rho 1.0 spread fail", res(1.0, spread=0.1), "INCONCLUSIVE"),
             ("rho 1.6 active 0.6", res(1.6, act=0.6), "KILL")]
    ok = True
    for name, r, want in cases:
        got = verdict(r, ref)[0]
        ok &= got == want
        print(f"{'OK ' if got == want else 'BAD'} {name}: {got} (want {want})")
    print(f"SYNTHETIC {'PASS' if ok else 'FAIL'} ({len(cases)} cases)")
    return ok


# ============================================================================ checks
def explicit_amp(a, ou, od):
    p = a.p
    up, dn = np.nonzero(ou)[0], np.nonzero(od)[0]
    A = p["F0"][np.ix_(up, dn)].copy()
    for mm in range(a.V):
        xp = patch_input(ou[None], od[None], a.P[mm:mm + 1])[0, 0]
        gm = max(0.0, float(xp @ p["gw"] + p["gc"][a.sub[mm]]))
        h = np.tanh(xp @ p["Wl"] + p["bl"])
        A += gm * np.outer(p["K"][up, mm], h @ p["Om"][mm][:, dn])
    nt = (ou + od).astype(float)
    J = sum(p["vJ"][k] * nt[i] * nt[j] for k, (i, j) in enumerate(zip(*a.iu)))
    return np.exp(J) * np.linalg.det(A)


def check(args):
    ok = True
    rng = np.random.default_rng(5)
    for inst in (20, 23):
        m = mt.make_model(inst); refs = mt.references(m)
        a = SparseGateAnsatz(m, refs[1], 4, 0, seed=3)
        x = a.flat(); a.set_flat(x + 0.3 * rng.standard_normal(len(x)))
        sel = rng.choice(m.D, 40, replace=False)
        ou, od, ru, rd = a._occ(sel)
        fast = a.forward(ou, od, ru, rd)
        ref = np.array([explicit_amp(a, ou[k], od[k]) for k in range(len(sel))])
        err = np.max(np.abs(fast["psi"] - ref)) / np.max(np.abs(ref))
        frac = float((fast["z"] > 0).mean())
        log(f"check (i) inst {inst}: explicit amplitude rel err {err:.1e} (active fraction on sample {frac:.2f})")
        ok &= err <= 1e-12 and 0.05 < frac < 0.95
    m = mt.make_model(21); refs = mt.references(m)
    base = T2.HFPSAnsatz(m, refs[1], "plus", 4, 0, seed=7)
    a = SparseGateAnsatz(m, refs[1], 4, 0, seed=7)
    a.p["gw"][:] = 0.0; a.p["gc"][:] = 1.0
    d = np.max(np.abs(base.amplitudes(4096) - a.amplitudes(4096)))
    log(f"check (ii) gate open == HFPSAnsatz: max |diff| = {d:.1e}"); ok &= d == 0.0
    m = mt.make_model(23); refs = mt.references(m)
    half = TS.SubHFPSAnsatz(m, refs[1], 4, 0, "half", seed=7)
    a = SparseGateAnsatz(m, refs[1], 4, 0, seed=7)
    a.p["gw"][:] = 0.0; a.p["gc"][:] = (1.0, -1.0)
    for k in half.keys:
        a.p[k] = half.p[k].copy()
    ph, pa = half.amplitudes(4096), a.amplitudes(4096)
    d = np.max(np.abs(ph - pa)) / np.max(np.abs(ph))
    log(f"check (iii) gate = checkerboard == half: rel diff {d:.1e}"); ok &= d <= 1e-12
    m = mt.make_model(20); m.spin_setup(); refs = mt.references(m)
    W, Vv, deg = solve_ed(m, 2, 1e-8, mt.UHF_SEED)
    sg, gen = mt.build_symmetry(m, mt.resolve_ground(m, Vv, deg))
    a = SparseGateAnsatz(m, refs[1], 4, 0, seed=4)
    x = a.flat(); a.set_flat(x + 0.05 * rng.standard_normal(len(x)))
    ok &= LT.fd_check(a, m, sg, "none", rng, "sgate inst20")
    # (v) penalty gradient
    tab = GateTables(a)
    x0 = a.flat().copy()

    def A_of(xx):
        a.set_flat(xx); ps = a.amplitudes(4096)
        return penalty_parts(a, tab, ps, ps @ ps)[0]
    a.set_flat(x0); ps = a.amplitudes(4096)
    A0, af, r_pen, ex = penalty_parts(a, tab, ps, ps @ ps)
    gA = 2 * a.force(r_pen, 4096) + ex
    dvec = rng.standard_normal(len(x0))
    eps = 1e-6
    fd = (A_of(x0 + eps * dvec) - A_of(x0 - eps * dvec)) / (2 * eps)
    a.set_flat(x0)
    rel = abs(gA @ dvec - fd) / max(1e-12, abs(fd))
    log(f"check (v) penalty gradient: analytic {gA @ dvec:.10e} fd {fd:.10e} rel {rel:.1e} (A = {A0:.4f}, a = {af:.3f})")
    ok &= rel < 1e-5
    log(f"CHECK {'PASS' if ok else 'BUG'}")
    return ok


# ============================================================================ training
def train(args):
    t_start = time.process_time()
    m, W, Vv, deg, gap, psi0, sg, gen, dchi, dS = TS.setup(args.instance, args.r_min)
    assert gap >= TS.GAP_MIN
    rng_b = np.random.default_rng(args.seed + 1000003)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng_b.random(m.D))], args.lbfgs_iters)
    calib = float(Ec - W[0])
    refs = mt.references(m)
    a = SparseGateAnsatz(m, refs[1], args.F, args.Ha, seed=args.seed)
    tab = GateTables(a)
    if args.controller == "project":
        lam = 0.0
        d0 = project_bias(a, tab, a.amplitudes(args.chunk), args.target)
        log(f"controller project: initial bias shift {d0:.4f}")
    log(f"instance {args.instance} D={m.D} E0={W[0]:.10f} gap={gap:.4f} n_eff={a.n_eff()} calib={calib:.2e} "
        f"threads={os.environ.get('OMP_NUM_THREADS')}")
    rng = np.random.default_rng(args.seed + 1000 * RNG_OFF)
    lam = args.lam0 if args.controller == "dual" else 0.0
    best_E, best_x, best_act, traj, acts, lams, best_hist = np.inf, a.flat().copy(), np.nan, [], [], [], []
    stop, converged = "max-steps", False
    t0 = time.process_time()
    n_max = args.probe_steps if args.probe_steps else args.max_steps
    for it in range(n_max):
        x0 = a.flat().copy()
        E, A, af, step, psi_pre = sr_step_pen(a, m, tab, lam, args, rng)
        traj.append(E); acts.append(af); lams.append(lam)
        if not np.isfinite(E):
            stop = f"non-finite energy at it {it}"; break
        if E < best_E and af <= args.act_max:
            best_E, best_x, best_act = E, x0, af
        best_hist.append(best_E)
        a.set_flat(x0 - step)
        if args.controller == "dual":
            lam *= args.lam_mult ** np.sign(af - args.target)
        else:
            project_bias(a, tab, psi_pre, args.target)
        if it % 50 == 0:
            log(f"it {it:5d} E-E0 = {E - W[0]:.6f} best = {best_E - W[0]:.6f} active = {af:.3f} A = {A:.4f} "
                f"lam = {lam:.2e} ({(time.process_time() - t0) / (it + 1):.3f} CPU-s/step)")
        if it + 1 >= args.min_steps and (it + 1) % 50 == 0:
            if best_hist[-1 - args.plateau_window] - best_E < args.plateau_tol:
                stop, converged = f"plateau at it {it + 1}", True
                break
        if time.process_time() - t0 > args.max_cpu:
            stop = f"CPU cap at it {it + 1}"; break
    sr_cpu = time.process_time() - t0
    if args.probe_steps:
        log(f"PROBE {len(traj)} steps, {sr_cpu / len(traj):.3f} CPU-s/step, setup {t0 - t_start:.1f} CPU-s")
        return None
    a.set_flat(best_x)
    log(f"stop = {stop}; best E - E0 = {best_E - W[0]:.6f} at active fraction {best_act:.3f}; SR CPU {sr_cpu:.0f} s")
    psi = a.amplitudes(args.chunk)
    b = budget(m, psi, W, Vv, deg, args, rng_b)
    for k_ in ("P0", "Ptheta", "Psign", "fn_history", "sign_starts"):
        b.pop(k_, None)
    hs = hop_stats(a, m, tab, psi)
    log(f"[P]: dE_tot={b['dE_tot']:.6f} dE_sign={b['dE_sign']:.6f} spread={b['spread']:.1e}; hops: {hs}")
    return dict(instance=args.instance, V=m.V, D=m.D, gap=gap, arm="sgate", seed=args.seed, F=args.F, Ha=args.Ha,
                n_eff=a.n_eff(), E0=float(W[0]), calib_err=calib, converged=converged, stop=stop,
                steps_done=len(traj), traj=traj, active=acts, lam=lams, act_scored=best_act, sr_cpu=sr_cpu,
                total_cpu=time.process_time() - t_start, budgets=dict(P=b), hops=hs,
                gate=dict(gw=a.p["gw"].tolist(), gc=a.p["gc"].tolist()),
                params={k: v for k, v in vars(args).items()})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["check", "probe", "train", "verdict", "synthetic"], required=True)
    p.add_argument("--instance", type=int, choices=[20, 23], default=23)
    p.add_argument("--probe-steps", type=int, default=0)
    p.add_argument("--max-cpu", type=float, default=480.0)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--controller", choices=["dual", "project"], default="project")
    p.add_argument("--lam0", type=float, default=1e-3)
    p.add_argument("--lam-mult", type=float, default=1.2)
    p.add_argument("--target", type=float, default=0.5)
    p.add_argument("--act-max", type=float, default=0.55)
    p.add_argument("--results", default=os.path.join(HERE, "results"))
    p.add_argument("--overwrite", type=int, default=0)
    a = p.parse_args(argv)
    base = TS.parse_args(["--mode", "train"])                    # t1_sub's pre-registered defaults
    for k, v in vars(base).items():
        if not hasattr(a, k):
            setattr(a, k, v)
    a.max_steps = a.max_steps
    return a


def main(argv=None):
    args = parse_args(argv)
    fn = os.path.join(args.results, f"t1_sgate_inst{args.instance}_s{args.seed}.json")
    if args.mode == "check":
        sys.exit(0 if check(args) else 2)
    if args.mode == "synthetic":
        sys.exit(0 if synthetic(args) else 2)
    if args.mode == "probe":
        assert args.probe_steps > 0
        train(args); return
    if args.mode == "verdict":
        r, ref = json.load(open(fn)), json.load(open(REF_FULL))
        v, rho, why = verdict(r, ref)
        print(f"VERDICT R15.6 T1 (V=12): {v}; {why}")
        print(f"hop statistics on the scored state: {r['hops']}")
        return
    if args.instance != 23:
        log("train is scored on instance 23 only"); sys.exit(2)
    os.makedirs(args.results, exist_ok=True)
    if os.path.exists(fn) and not args.overwrite:
        log(f"{fn} exists; refusing to overwrite"); return
    r = train(args)
    json.dump(r, open(fn, "w"))
    log(f"wrote {fn}")
    ref = json.load(open(REF_FULL))
    v, rho, why = verdict(r, ref)
    log(f"VERDICT R15.6 T1 (V=12): {v}; {why}")


if __name__ == "__main__":
    main()
