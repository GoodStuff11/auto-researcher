#!/usr/bin/env python3
"""
P1.3 round-5 premise test: is the residual projected sign error on the 4x4 OBC t-t'-J instance
a BASIN (optimisation trapping) or a CAPACITY limit?
================================================================================================

Physics question
----------------
On the shared instance (4x4 OBC, N_up = N_dn = 6, t' = -0.25, J = 0.4; D = 1681680; projected
sector dim range(P_S P_chi) = 30220) a space-group-projected NNBF of width 6 (1734 parameters)
keeps a nodal error dE_sign = 0.0125 t, overlap 0.936 with psi_0 (P2.1 round-3 pilot, job 410988,
seed 11). P1.3 has a target only if that error is a basin: a better-signed state exists AT THE
SAME WIDTH, is stable under projected SR, and is not found from cold starts. If instead the
oracle-guided state relaxes back to the cold level, or the oracle optimum is no better, the error
is a capacity limit and the leaf hands over to P2.1 / P1.2.

Arms (all: NNBF "full" mode of tj_core.LRAnsatz, K = 1, no Jastrow, width w; trained state
psi_P = P_chi psi_theta, projected SR with the exact force and the sampled projected metric,
exactly as P2.1's "nnbf" arm, which Cuhf at w = 6 reproduces step for step):
  Cuhf  cold start, reference = UHF orbitals of the U = 8 Hubbard model (p11_core.uhf_orbitals,
        seed 1234; P2.1's reference). This is the PROJECTED CONTROL. --pre + --rel energy steps.
  Cfs   cold start, reference = Fermi sea of -T (lowest N_sigma orbitals, 1e-3 random diagonal
        tie-break with seed + 31). --pre + --rel energy steps.
  Ccdw  cold start, reference = UHF from a charge-stripe initial density (column x = 0 depleted by
        40%, other columns enriched, staggered moment 0.3; P1.3 round-3 "cdw"). --pre + --rel.
  W     oracle arm, paired with Cuhf (same reference, same network initialisation, same SR random
        stream): --pre steps of projected SR on -log F, F = <psi_0|psi_P>^2/<psi_P|psi_P>
        (half-gradient force with coefficient psi_P/<psi_P|psi_P> - psi_0/<psi_0|psi_P>),
        keeping the best-fidelity iterate ("W0"); then RELEASED to --rel steps of plain projected
        energy SR. Every arm therefore takes the same total number of SR steps (--pre + --rel).
Width ladder: w in {6, 4, 10} (1734, 1156, 2890 parameters). Seeds 11, 22, 33.
Snapshots with full P1.1 budgets (p11_core.budget: E_sign upper estimate = min of 4 L-BFGS-B
starts and the ten Haaf fixed node, dE_tot, dE_sign, f_E, w_sign, ...), all on the best-energy
iterate so far:
  W0   (W only) end of the fidelity phase, state P;
  mid  step --pre + --rel/2, state P;
  end  step --pre + --rel, states P and P_S (post-hoc exact total-spin projection).
Each snapshot also records ov_k = |<E_k|psi>|^2 with the 4 lowest ED levels and their sector
labels (space-group characters, <S^2>), so a same-sector competitor would be visible.
Energy and fidelity traces are stored per step; "converged" means the mean energy of the last
100 steps is at most CONV_TOL = 1e-3 t below the mean of the 100 steps before (2% of the
control's dE_tot = 0.052); "W0 plateau" means the fidelity of the last 50 fidelity steps gained
at most 2e-3 over the 50 before.

Pre-committed decision rule (implemented in --mode aggregate; primary state P, metric dE_sign)
---------------------------------------------------------------------------------------------
Per width w, with n complete seeds (n >= 3 required, otherwise UNCLEAR):
  c_a      = median over seeds of dE_sign(cold arm a, end);  Cbest = min_a c_a
  sigma_c  = median over the 3 cold arms of the seed range (max - min) of dE_sign(end)
  sigma_E  = the same for dE_tot(end)
  Wr, Wm, W0 = medians of dE_sign(W) at end, mid, W0
  B1 (lower)      Wr <= Cbest - max(0.25 Cbest, 2 sigma_c)
  B2 (paired)     dE_sign(W, end, s) < min_a dE_sign(a, end, s) on >= ceil(2n/3) seeds
  B3 (stays)      median_s [dE_sign(W,end,s) - dE_sign(W,mid,s)] <= 0.25 (Cbest - Wr)
  B4 (stable)     median dE_tot(W, end) <= min_a median dE_tot(a, end) + max(2 sigma_E, 1e-3)
                  (W is not a sign-for-amplitude trade that SR would leave)
  Bc (converged)  every cold arm's median run is converged (energy plateau)
  BASIN(w)        B1 and B2 and B3 and B4 and Bc
  K1 (oracle no better)  W0 >= 0.75 Cbest, the W0 fidelity phase plateaued on >= ceil(2n/3)
                         seeds, and median ov0(W0) >= median ov0(Cuhf, end) (the oracle phase
                         reached at least the control's overlap, so it is not itself trapped)
  K2 (relaxed back)      Wr >= Cbest - max(0.10 Cbest, sigma_c)
  K3 (trade-off)         B1 and B2 hold but B4 fails
  CAPACITY(w)     not BASIN(w) and (K1 or K2 or K3)
  UNCLEAR(w)      otherwise (also: B1, B2, B3, B4 hold but Bc fails)
Global (instance 0):
  PREMISE GATE    median dE_sign(Cuhf, w = 6, end) >= 0.005 t over its seeds; else NO-TARGET
                  (no residual sign error to trap: leaf closed on this instance too).
  BASIN           BASIN(w) at >= 2 widths, at least one of them w = 6 or w = 10.
  CAPACITY        CAPACITY(w) at >= 2 widths and BASIN(w) at none.
  UNCLEAR         otherwise. Pre-committed action: seeds 44, 55, 66 at every width that is
                  not CAPACITY, then re-apply; if still not BASIN, the verdict is CAPACITY.
  Consequence     BASIN -> P1.3 has a target; an escape route (not OCP, not SNO-FN) is tested
                  against Cuhf/Cfs/Ccdw at the BASIN widths. CAPACITY -> the leaf reduces to a
                  capacity question and is handed to P2.1 (amplitude/sign capacity at fixed cost)
                  and P1.2 (sign-layer capacity).
Reported, not gated: Cbest(w) and W0(w) versus width (width-limited if Cbest(10) <= 0.5 Cbest(4)),
the P_S budgets, overlaps with the ED levels, P2.1's seeds of the same control if present.
Hard Core: SIGN HALF ONLY. ED levels 1-3 are labelled, but no same-sector competitor is claimed.

Usage
-----
  python3 micro_test.py --mode train --task ID [options]  one Slurm array task (0..35)
  python3 micro_test.py --mode train --instance I --arm A --width W --seed S [options]
  python3 micro_test.py --mode aggregate [--results DIR] [--instance I]
  python3 micro_test.py --mode smoke                       checks + tiny pipeline (instance 8)
  python3 micro_test.py --mode time --instance 0           one energy SR step per width

Options
-------
  --mode {train,aggregate,smoke,time}  What to run (default smoke).
  --task INT       Array index 0..35: width = WIDTHS[task // 12] with WIDTHS = 6, 4, 10;
                   arm = ARMS[(task % 12) // 3] with ARMS = Cuhf, Cfs, Ccdw, W;
                   seed = SEEDS[task % 3] with SEEDS = 11, 22, 33. Explicit --arm / --width /
                   --seed override the decoded values. Instance defaults to 0.
  --instance INT   0 (production: 4x4 OBC 6+6) or 8 (smoke: 3x3 OBC 3+3, D = 1680).
  --arm NAME       Cuhf, Cfs, Ccdw or W (see Arms).
  --width INT      NNBF hidden width (default 6).
  --seed INT       Network initialisation and SR sampling seed (default 11).
  --pre INT        Fidelity steps of W; every cold arm takes --pre extra energy steps (default 200).
  --rel INT        Energy steps after the fidelity phase (default 700). Mid snapshot at
                   --pre + --rel // 2.
  --sr-lr FLOAT (0.05)  --sr-shift FLOAT (1e-3)  --sr-samples INT (2000): projected SR settings
                   (P2.1's). The same step size is used for the fidelity phase.
  --chunk INT      Configurations per forward chunk (default 32768).
  --o-chunk INT    Samples per chunk of projected log-derivatives (default 8).
  --lbfgs-iters INT (4000)  --fn-iters INT (10)  --fn-floor FLOAT (1e-3)  --r-min FLOAT (1.5):
                   budget settings, as P1.1 / P2.1.
  --init-from PATH Optional .npz of saved parameters (results/params) to resume from; the run
                   then does --rel further energy steps (pre-committed extension action).
  --results DIR    Results directory (default results/ next to this script).
  --overwrite {0,1} Recompute an existing result JSON (default 0).

Outputs
-------
  train: results/inst{I}_{arm}_w{W}_s{S}.json with params, n_params, ED levels and their sector
         labels, irrep, sector dims, calib_err, traces (E, F per step), converged flags, and
         snapshots {W0, mid, end, end_PS}: P1.1 budget fields + S2, ov0, ovk, step;
         results/params/inst{I}_{arm}_w{W}_s{S}.npz (flat parameters of the final state).
  aggregate: printed table, per-width and global verdicts; results/aggregate_inst{I}.json.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p11_core import budget, energy, sign_constrained_min, solve_ed, uhf_orbitals  # noqa: E402
from tj_core import TJ, LRAnsatz, SpaceGroup  # noqa: E402

INSTANCES = {0: dict(Lx=4, Ly=4, bcx="obc", bcy="obc", nup=6, ndn=6, tp=-0.25, J=0.4),
             8: dict(Lx=3, Ly=3, bcx="obc", bcy="obc", nup=3, ndn=3, tp=-0.25, J=0.4)}
WIDTHS = [6, 4, 10]
ARMS = ["Cuhf", "Cfs", "Ccdw", "W"]
COLD = ARMS[:3]
SEEDS = [11, 22, 33]
NTASK = len(WIDTHS) * len(ARMS) * len(SEEDS)
UHF_SEED = 1234
# pre-committed thresholds
PREMISE_MIN = 0.005
B1_FRAC, B1_SIG, B3_FRAC, B4_MIN, B4_SIG = 0.25, 2.0, 0.25, 1e-3, 2.0
K1_FRAC, K2_FRAC = 0.75, 0.10
CONV_TOL, CONV_WIN, FID_TOL, FID_WIN = 1e-3, 100, 2e-3, 50
P21_RESULTS = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3", "results")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ setup (from P2.1 r3)
def make_model(k):
    c = INSTANCES[k]
    return TJ(c["Lx"], c["Ly"], c["bcx"], c["bcy"], c["nup"], c["ndn"], 1.0, c["tp"], c["J"])


def resolve_ground(m, Vv, deg):
    """psi_0 with definite characters under {Mx, My} (and translations) inside a degenerate level."""
    if deg == 1:
        return Vv[:, 0]
    sg = SpaceGroup(m, diag=False)
    gens = [k for k, e in enumerate(sg.elems) if e in ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0),
                                                        (0, 0, 1, 0, 0), (0, 0, 0, 1, 0))]
    Q = Vv[:, :deg]
    Ms = [Q.T @ np.column_stack([sg.apply(Q[:, i], k) for i in range(deg)]) for k in gens]
    rng = np.random.default_rng(7)
    A = sum(rng.standard_normal() * Mg for Mg in Ms)
    w, U = np.linalg.eig(A)
    for j in range(deg):
        q = np.real(U[:, j]); q /= np.linalg.norm(q)
        if all(np.linalg.norm(Mg @ q - np.round(q @ Mg @ q) * q) < 1e-8 for Mg in Ms):
            return Q @ q
    raise RuntimeError("could not resolve a 1-D irrep inside the degenerate ground level")


def build_symmetry(m, psi0):
    for diag in (True, False):
        sg = SpaceGroup(m, diag=diag)
        try:
            gen = sg.set_from(psi0)
            return sg, gen
        except RuntimeError:
            continue
    raise RuntimeError("psi_0 not in a 1-D irrep of the abelian subgroup")


def level_labels(m, sg, Vv):
    """Characters <v|U_g|v> under the group generators and <S^2> of every ED level."""
    out = []
    for k in range(Vv.shape[1]):
        v = Vv[:, k] / np.linalg.norm(Vv[:, k])
        ch = {}
        for kk, e in enumerate(sg.elems):
            name = {(1, 0, 0, 0, 0): "Tx", (0, 1, 0, 0, 0): "Ty", (0, 0, 1, 0, 0): "Mx",
                    (0, 0, 0, 1, 0): "My", (0, 0, 0, 0, 1): "D"}.get(tuple(e))
            if name:
                ch[name] = round(float(v @ sg.apply(v, kk)), 4)
        out.append(dict(chars=ch, S2=round(m.S2_expect(v), 4),
                        in_sector=bool(np.linalg.norm(sg.project(v) - v) < 1e-6)))
    return out


# ============================================================================ references
def uhf_from(m, nu, nd, iters=2000, mix=0.3):
    """P1.1 UHF loop (p11_core.uhf_orbitals) from given initial densities (P1.3 r3 copy)."""
    T, U = m.T, m.U
    for _ in range(iters):
        eu, vu = np.linalg.eigh(-T + np.diag(U * nd))
        ed, vd = np.linalg.eigh(-T + np.diag(U * nu))
        Pu, Pd = vu[:, :m.nup], vd[:, :m.ndn]
        nu_new, nd_new = np.sum(Pu ** 2, 1), np.sum(Pd ** 2, 1)
        if max(np.abs(nu_new - nu).max(), np.abs(nd_new - nd).max()) < 1e-10:
            break
        nu, nd = (1 - mix) * nu + mix * nu_new, (1 - mix) * nd + mix * nd_new
    return Pu.copy(), Pd.copy()


def reference(m, arm, seed):
    rng = np.random.default_rng(seed + 31)
    if arm in ("Cuhf", "W"):
        return uhf_orbitals(m, UHF_SEED)
    if arm == "Cfs":
        out = []
        for n in (m.nup, m.ndn):
            _, v = np.linalg.eigh(-m.T + np.diag(1e-3 * rng.standard_normal(m.V)))
            out.append(v[:, :n].copy())
        return tuple(out)
    if arm == "Ccdw":
        x = np.array([c[0] for c in m.coords]); y = np.array([c[1] for c in m.coords])
        stag = (-1.0) ** (x + y)
        hole = np.where(x == 0, -0.4, 0.4 / (m.Lx - 1))
        nu = (m.nup / m.V) * (1 + hole + 0.3 * stag) + 0.01 * rng.standard_normal(m.V)
        nd = (m.ndn / m.V) * (1 + hole - 0.3 * stag) + 0.01 * rng.standard_normal(m.V)
        return uhf_from(m, nu, nd)
    raise ValueError(arm)


def make_ansatz(m, arm, width, seed):
    return LRAnsatz(m, reference(m, arm, seed), "full", K=1, r=0, Hs=width, Ha=0, shared_ref=True, seed=seed)


# ============================================================================ projected SR
def proj_state(ans, sg, chunk):
    psi_t = ans.amplitudes(chunk)
    return psi_t, sg.project(psi_t)


def sr_step(ans, m, sg, psi0, args, rng, fidelity):
    """One projected SR step on psi_P = P_chi psi_theta. Energy force (Hpsi - E psi)/nn, or the
    fidelity half-gradient psi/nn - psi0/<psi0|psi> of -log F. Returns E, F, |f|, step."""
    psi_t, psi = proj_state(ans, sg, args.chunk)
    E, Hp, nn = energy(m.H, psi)
    ov = float(psi0 @ psi)
    F = ov * ov / nn
    rc = psi / nn - psi0 / ov if fidelity else (Hp - E * psi) / nn
    f = ans.force(rc, args.chunk)
    pc = psi ** 2
    idx = rng.choice(len(pc), size=args.sr_samples, p=pc / pc.sum())
    O = ans.per_sample_O_proj(sg, idx, psi, args.o_chunk)
    Oc = (O - O.mean(0)) / np.sqrt(len(idx))
    lam = args.sr_shift
    Km = Oc @ Oc.T + lam * np.eye(len(idx))
    dx = (f - Oc.T @ np.linalg.solve(Km, Oc @ f)) / lam
    step = args.sr_lr * dx
    nrm = np.linalg.norm(dx)
    if nrm * args.sr_lr > 1.0:
        step *= 1.0 / (nrm * args.sr_lr)
    return E, F, float(np.linalg.norm(f)), step


class Tracker:
    """Runs SR phases and keeps the best iterate (lowest E, or highest F in the fidelity phase)."""

    def __init__(self, ans, m, sg, psi0, args, rng, label):
        self.ans, self.m, self.sg, self.psi0, self.a, self.rng, self.label = ans, m, sg, psi0, args, rng, label
        self.E, self.F, self.phase = [], [], []
        self.best = None
        self.t0 = time.time()

    def run(self, nsteps, fidelity):
        if fidelity or self.best is None or self.best[2] != "E":
            self.best = (np.inf, self.ans.flat().copy(), "F" if fidelity else "E")
        for _ in range(nsteps):
            x0 = self.ans.flat().copy()
            E, F, fn, step = sr_step(self.ans, self.m, self.sg, self.psi0, self.a, self.rng, fidelity)
            if not np.isfinite(E) or not np.isfinite(F):
                log(f"{self.label}: non-finite value; restoring best")
                self.ans.set_flat(self.best[1])
                break
            key = -F if fidelity else E
            if key < self.best[0]:
                self.best = (key, x0, self.best[2])
            self.ans.set_flat(x0 - step)
            self.E.append(E); self.F.append(F); self.phase.append("F" if fidelity else "E")
            it = len(self.E) - 1
            if it % 50 == 0:
                log(f"{self.label}: it {it:5d} [{'fid' if fidelity else 'E'}] E = {E:.8f} F = {F:.6f} "
                    f"|f| = {fn:.2e} ({(time.time() - self.t0) / (it + 1):.2f} s/step)")

    def best_params(self):
        """Best iterate of the current phase, including the current parameters."""
        psi = proj_state(self.ans, self.sg, self.a.chunk)[1]
        E, _, nn = energy(self.m.H, psi)
        F = float(self.psi0 @ psi) ** 2 / nn
        key = -F if self.best[2] == "F" else E
        return self.ans.flat().copy() if key <= self.best[0] else self.best[1].copy()


def window_drop(x, win):
    x = np.asarray(x, float)
    if len(x) < 2 * win:
        return float("nan")
    return float(np.mean(x[-2 * win:-win]) - np.mean(x[-win:]))


# ============================================================================ task
def snapshot(ctx, ans, x, lab, step, args, with_ps=False):
    m, sg, W, Vv, deg, psi0, S0, rng_b = (ctx[k] for k in ("m", "sg", "W", "Vv", "deg", "psi0", "S0", "rng_b"))
    keep = ans.flat().copy()
    ans.set_flat(x)
    psi = proj_state(ans, sg, args.chunk)[1]
    ans.set_flat(keep)
    states = [(lab, psi)] + ([(lab + "_PS", m.spin_project(psi, S0))] if with_ps else [])
    out = {}
    for name, v in states:
        t2 = time.time()
        Vref = psi0[:, None] if deg > 1 else Vv
        b = budget(m, v, W, Vref, 1 if deg > 1 else deg, args, rng_b)
        vn = v / np.linalg.norm(v)
        b.update(S2=m.S2_expect(vn), ov0=float((psi0 @ vn) ** 2), ovk=[float((Vv[:, k] @ vn) ** 2)
                                                                       for k in range(Vv.shape[1])], step=step)
        for k_ in ("P0", "Ptheta", "Psign", "sign_starts"):
            b.pop(k_, None)
        out[name] = b
        log(f"{ctx['label']} [{name} @ {step}]: dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} "
            f"f_E={b['f_E_up']:.3f} spread={b['spread']:.1e} S2={b['S2']:.3f} ov0={b['ov0']:.4f} "
            f"ovk={[round(o, 4) for o in b['ovk']]} ({time.time() - t2:.0f} s)")
    return out


def train_task(args):
    m = make_model(args.instance)
    m.spin_setup()
    m.pair_setup(args.r_min)
    label = f"{args.arm}-w{args.width}-s{args.seed}"
    log(f"instance {args.instance} {INSTANCES[args.instance]} D={m.D} arm={args.arm} width={args.width} "
        f"seed={args.seed} pre={args.pre} rel={args.rel}")
    t0 = time.time()
    W, Vv, deg = solve_ed(m, 4, 1e-8, UHF_SEED)
    psi0 = resolve_ground(m, Vv, deg)
    psi0 = psi0 / np.linalg.norm(psi0)
    sg, gen = build_symmetry(m, psi0)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * m.S2_expect(psi0)) - 1)))
    labels = level_labels(m, sg, Vv)
    log(f"ED levels {W.tolist()} deg {deg}; |G|={sg.G} chi(psi_0)={gen}; S(psi_0)={S0:g}; "
        f"level labels {labels} ({time.time() - t0:.0f} s)")
    rng_b = np.random.default_rng(args.seed + 1000003)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng_b.random(m.D))], args.lbfgs_iters)
    calib = float(Ec - W[0])
    ans = make_ansatz(m, args.arm, args.width, args.seed)
    if args.init_from:
        ans.set_flat(np.load(args.init_from)["flat"])
        log(f"resumed parameters from {args.init_from}")
    psi_t, psi = proj_state(ans, sg, args.chunk)
    log(f"{label}: n_params={ans.n_params()} calib={calib:.2e} init |P psi|/|psi|="
        f"{np.linalg.norm(psi) / np.linalg.norm(psi_t):.3e} init F={float(psi0 @ psi) ** 2 / (psi @ psi):.4f}")
    rng = np.random.default_rng(args.seed + 1000 * (0 if args.arm == "W" else ARMS.index(args.arm)))
    ctx = dict(m=m, sg=sg, W=W, Vv=Vv, deg=deg, psi0=psi0, S0=S0, rng_b=rng_b, label=label)
    tr = Tracker(ans, m, sg, psi0, args, rng, label)
    snaps = {}
    fid_plateau = None
    if args.init_from:
        n_pre = 0
    elif args.arm == "W":
        n_pre = args.pre
        tr.run(n_pre, fidelity=True)
        xw = tr.best_params()
        ans.set_flat(xw)                                   # release from the best-fidelity iterate
        fid_plateau = bool(window_drop([-f for f in tr.F], FID_WIN) <= FID_TOL)
        snaps.update(snapshot(ctx, ans, xw, "W0", n_pre, args))
    else:
        n_pre = args.pre
        tr.run(n_pre, fidelity=False)
    half = args.rel // 2
    tr.run(half, fidelity=False)
    snaps.update(snapshot(ctx, ans, tr.best_params(), "mid", n_pre + half, args))
    tr.run(args.rel - half, fidelity=False)
    xf = tr.best_params()
    ans.set_flat(xf)
    snaps.update(snapshot(ctx, ans, xf, "end", n_pre + args.rel, args, with_ps=True))
    Erel = [e for e, ph in zip(tr.E, tr.phase) if ph == "E"]
    drop = window_drop(Erel, CONV_WIN)
    conv = bool(np.isfinite(drop) and drop <= CONV_TOL)
    log(f"{label}: energy-window drop {drop:.2e} (tol {CONV_TOL:g}) converged={conv} fid_plateau={fid_plateau} "
        f"total {time.time() - t0:.0f} s")
    pdir = os.path.join(args.results, "params")
    os.makedirs(pdir, exist_ok=True)
    np.savez(os.path.join(pdir, f"inst{args.instance}_{args.arm}_w{args.width}_s{args.seed}.npz"), flat=xf)
    return dict(params=vars(args), instance=args.instance, arm=args.arm, width=args.width, seed=args.seed,
                n_params=ans.n_params(), ed_levels=W.tolist(), level_labels=labels, degeneracy=deg,
                irrep=gen, G=sg.G, S_psi0=S0, calib_err=calib, trace_E=tr.E, trace_F=tr.F,
                trace_phase="".join(tr.phase), energy_drop=drop, converged=conv, fid_plateau=fid_plateau,
                snapshots=snaps, wall=time.time() - t0)


# ============================================================================ aggregate
def _med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def width_verdict(R, w, seeds):
    """R[(arm, seed)] = result dict at width w. Returns (verdict, info dict)."""
    def g(arm, s, snap="end", key="dE_sign"):
        r = R.get((arm, s))
        if r is None or snap not in r["snapshots"] or r["snapshots"][snap] is None:
            return float("nan")
        return float(r["snapshots"][snap][key])
    n = len(seeds)
    need = math.ceil(2 * n / 3)
    c = {a: _med([g(a, s) for s in seeds]) for a in COLD}
    cE = {a: _med([g(a, s, key="dE_tot") for s in seeds]) for a in COLD}
    Cbest = min(c.values())
    sig_c = _med([max(g(a, s) for s in seeds) - min(g(a, s) for s in seeds) for a in COLD])
    sig_E = _med([max(g(a, s, key="dE_tot") for s in seeds) - min(g(a, s, key="dE_tot") for s in seeds)
                  for a in COLD])
    Wr = _med([g("W", s) for s in seeds])
    Wm = _med([g("W", s, "mid") for s in seeds])
    W0 = _med([g("W", s, "W0") for s in seeds])
    WE = _med([g("W", s, key="dE_tot") for s in seeds])
    paired = sum(g("W", s) < min(g(a, s) for a in COLD) for s in seeds)
    drift = _med([g("W", s) - g("W", s, "mid") for s in seeds])
    conv = {a: _med([float(R[(a, s)]["converged"]) for s in seeds if (a, s) in R]) >= 0.5 for a in COLD}
    plateau = sum(bool(R[("W", s)].get("fid_plateau")) for s in seeds if ("W", s) in R)
    B1 = Wr <= Cbest - max(B1_FRAC * Cbest, B1_SIG * sig_c)
    B2 = paired >= need
    B3 = drift <= B3_FRAC * (Cbest - Wr)
    B4 = WE <= min(cE.values()) + max(B4_SIG * sig_E, B4_MIN)
    Bc = all(conv.values())
    ovW0 = _med([g("W", s, "W0", "ov0") for s in seeds])
    ovC = _med([g("Cuhf", s, "end", "ov0") for s in seeds])
    K1 = W0 >= K1_FRAC * Cbest and plateau >= need and ovW0 >= ovC
    K2 = Wr >= Cbest - max(K2_FRAC * Cbest, sig_c)
    K3 = B1 and B2 and not B4
    info = dict(width=w, n=n, c=c, cE=cE, Cbest=Cbest, sigma_c=sig_c, sigma_E=sig_E, Wr=Wr, Wm=Wm, W0=W0,
                W_dE_tot=WE, ov0_W0=ovW0, ov0_Cuhf=ovC, paired=paired, drift=drift, converged=conv, fid_plateau=plateau,
                B1=B1, B2=B2, B3=B3, B4=B4, Bc=Bc, K1=K1, K2=K2, K3=K3)
    if n < 3 or not np.isfinite(Cbest) or not np.isfinite(Wr):
        return "UNCLEAR", info
    if B1 and B2 and B3 and B4 and Bc:
        return "BASIN", info
    if K1 or K2 or K3:
        return "CAPACITY", info
    return "UNCLEAR", info


def aggregate(d, inst):
    R = {}
    for fn in sorted(os.listdir(d)):
        if fn.startswith(f"inst{inst}_") and fn.endswith(".json") and "aggregate" not in fn:
            r = json.load(open(os.path.join(d, fn)))
            R[(r["width"], r["arm"], r["seed"])] = r
    if not R:
        print(f"no results for instance {inst} in {d}")
        return 1
    widths = [w for w in WIDTHS if any(k[0] == w for k in R)] + sorted({k[0] for k in R} - set(WIDTHS))
    any_r = next(iter(R.values()))
    print(f"===== P1.3 r5 instance {inst}: ED {np.round(any_r['ed_levels'], 6).tolist()} irrep {any_r['irrep']}")
    print(f"      ED level labels {any_r['level_labels']}")
    out = dict(instance=inst, widths={})
    verdicts = {}
    for w in widths:
        seeds = sorted({k[2] for k in R if k[0] == w and all((w, a, k[2]) in R for a in ARMS)})
        Rw = {(a, s): R[(w, a, s)] for a in ARMS for s in seeds}
        npar = R[(w, "Cuhf", seeds[0])]["n_params"] if seeds else "?"
        print(f"--- width {w} (n_params {npar}), complete seeds {seeds}")
        for s in seeds:
            row = []
            for a in ARMS:
                sn = Rw[(a, s)]["snapshots"]
                extra = f" W0 {sn['W0']['dE_sign']:.4f}/{sn['W0']['ov0']:.3f}" if a == "W" else ""
                row.append(f"{a}: sign {sn['end']['dE_sign']:.4f} (mid {sn['mid']['dE_sign']:.4f}) "
                           f"tot {sn['end']['dE_tot']:.4f} ov0 {sn['end']['ov0']:.3f} "
                           f"PS-sign {sn['end_PS']['dE_sign']:.4f} conv {int(Rw[(a, s)]['converged'])}{extra}")
            print(f"  seed {s}: " + " | ".join(row))
        v, info = width_verdict(Rw, w, seeds) if seeds else ("UNCLEAR", dict(n=0))
        verdicts[w] = v
        if seeds:
            print(f"  cold medians dE_sign {dict((a, round(x, 5)) for a, x in info['c'].items())} Cbest {info['Cbest']:.5f} "
                  f"sigma_c {info['sigma_c']:.5f}; dE_tot {dict((a, round(x, 5)) for a, x in info['cE'].items())} "
                  f"sigma_E {info['sigma_E']:.5f}")
            print(f"  W: W0 {info['W0']:.5f} mid {info['Wm']:.5f} end {info['Wr']:.5f} dE_tot {info['W_dE_tot']:.5f}; "
                  f"paired wins {info['paired']}/{info['n']}; drift {info['drift']:+.5f}; fid plateau "
                  f"{info['fid_plateau']}/{info['n']}; ov0 W0 {info['ov0_W0']:.4f} vs Cuhf {info['ov0_Cuhf']:.4f}; "
                  f"cold converged {info['converged']}")
            print(f"  B1 {info['B1']} B2 {info['B2']} B3 {info['B3']} B4 {info['B4']} Bc {info['Bc']} | "
                  f"K1 {info['K1']} K2 {info['K2']} K3 {info['K3']}  -> {v}")
        out["widths"][w] = dict(verdict=v, **{k: (x if not isinstance(x, (np.bool_,)) else bool(x))
                                              for k, x in info.items()})
    # premise gate on the projected control
    ref_w = 6 if 6 in widths else widths[0]
    ctrl = [R[(ref_w, "Cuhf", s)]["snapshots"]["end"]["dE_sign"] for s in SEEDS if (ref_w, "Cuhf", s) in R]
    p21 = []
    for s in SEEDS:
        fn = os.path.join(P21_RESULTS, f"inst{inst}_nnbf_s{s}.json")
        if inst == 0 and os.path.exists(fn):
            b = json.load(open(fn))["budgets"].get("P")
            if b:
                p21.append((s, round(b["dE_sign"], 5)))
    premise = _med(ctrl) >= PREMISE_MIN
    print(f"PREMISE GATE: median dE_sign(Cuhf, w={ref_w}) = {_med(ctrl):.5f} over {len(ctrl)} seeds "
          f"(>= {PREMISE_MIN}): {premise}; P2.1 r3 control (1200 steps) {p21}")
    nb = [w for w in widths if verdicts[w] == "BASIN"]
    nc = [w for w in widths if verdicts[w] == "CAPACITY"]
    if not premise:
        g = "NO-TARGET"
    elif len(nb) >= 2 and (6 in nb or 10 in nb):
        g = "BASIN"
    elif len(nc) >= 2 and not nb:
        g = "CAPACITY"
    else:
        g = "UNCLEAR"
    cb = {w: out["widths"][w].get("Cbest") for w in widths}
    if 4 in cb and 10 in cb and cb[4] and np.isfinite(cb[4]):
        print(f"width trend: Cbest {cb}; width-limited (Cbest(10) <= 0.5 Cbest(4)): {cb[10] <= 0.5 * cb[4]}")
    print(f"GLOBAL VERDICT: {g}  (BASIN widths {nb}, CAPACITY widths {nc})")
    if g == "UNCLEAR":
        print("  pre-committed action: seeds 44, 55, 66 at widths "
              f"{[w for w in widths if verdicts[w] != 'CAPACITY']}; still not BASIN -> CAPACITY")
    out.update(premise=premise, control=ctrl, p21_control=p21, verdict=g)
    json.dump(out, open(os.path.join(d, f"aggregate_inst{inst}.json"), "w"), indent=1, default=float)
    return 0


# ============================================================================ checks
def run_checks(seed):
    ok = True
    m = make_model(8)
    m.spin_setup(); m.pair_setup(1.0)
    W, U = np.linalg.eigh(m.H.toarray())
    psi0 = U[:, 0]
    deg = int(np.sum(W[:4] - W[0] < 1e-8))
    psi0 = resolve_ground(m, U[:, :4], deg)
    psi0 /= np.linalg.norm(psi0)
    sg, gen = build_symmetry(m, psi0)
    e_p0 = np.linalg.norm(sg.project(psi0) - psi0)
    log(f"check smoke instance: D={m.D} E0={W[0]:.8f} deg={deg} |G|={sg.G} chi={gen} P_chi psi0-psi0 {e_p0:.1e}")
    ok &= e_p0 < 1e-10
    rng = np.random.default_rng(seed)
    h = 1e-6
    for arm in ARMS:
        a = make_ansatz(m, arm, 4, seed)
        a.p["W2"] = 0.3 * rng.standard_normal(a.p["W2"].shape)
        x0 = a.flat().copy()
        d = rng.standard_normal(len(x0)); d /= np.linalg.norm(d)
        psi_t, psi = proj_state(a, sg, 4096)
        E, Hp, nn = energy(m.H, psi)
        ov = psi0 @ psi
        gE = 2 * a.force((Hp - E * psi) / nn, 4096)
        gF = 2 * a.force(psi / nn - psi0 / ov, 4096)      # gradient of -log F

        def fE(x):
            a.set_flat(x); return energy(m.H, proj_state(a, sg, 4096)[1])[0]

        def fF(x):
            a.set_flat(x); p = proj_state(a, sg, 4096)[1]; return -np.log((psi0 @ p) ** 2 / (p @ p))
        fdE = (fE(x0 + h * d) - fE(x0 - h * d)) / (2 * h)
        fdF = (fF(x0 + h * d) - fF(x0 - h * d)) / (2 * h)
        a.set_flat(x0)
        e1 = abs(gE @ d - fdE) / max(1.0, abs(fdE)); e2 = abs(gF @ d - fdF) / max(1.0, abs(fdF))
        orth = max(np.abs(r.T @ r - np.eye(r.shape[1])).max() for r in reference(m, arm, seed))
        log(f"check [{arm}] ref orthonormal {orth:.1e}; grad E analytic {gE @ d:.10e} fd {fdE:.10e}; "
            f"grad -logF analytic {gF @ d:.10e} fd {fdF:.10e}; init F={ov ** 2 / nn:.4f}")
        ok &= e1 < 1e-6 and e2 < 1e-6 and orth < 1e-10
    log(f"ALL CHECKS {'PASSED' if ok else 'FAILED'}")
    return bool(ok)


# ============================================================================ timing
def time_step(args):
    m = make_model(args.instance)
    t0 = time.time()
    W, Vv, deg = solve_ed(m, 4, 1e-8, UHF_SEED)
    psi0 = resolve_ground(m, Vv, deg); psi0 /= np.linalg.norm(psi0)
    sg, gen = build_symmetry(m, psi0)
    log(f"instance {args.instance} D={m.D} ED+group {time.time() - t0:.0f}s |G|={sg.G} chi={gen}")
    for w in WIDTHS:
        ans = make_ansatz(m, "Cuhf", w, 11)
        rng = np.random.default_rng(0)
        for fid in (False, True):
            t1 = time.time()
            sr_step(ans, m, sg, psi0, args, rng, fid)
            log(f"width {w}: n_params={ans.n_params()} one {'fidelity' if fid else 'energy'} SR step "
                f"{time.time() - t1:.2f}s (threads {os.environ.get('OMP_NUM_THREADS', '?')})")


# ============================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.3 round-5 basin-vs-capacity test (see module docstring)")
    ap.add_argument("--mode", choices=["train", "aggregate", "smoke", "time"], default="smoke")
    ap.add_argument("--task", type=int, default=-1)
    ap.add_argument("--instance", type=int, default=0, choices=sorted(INSTANCES))
    ap.add_argument("--arm", default="", choices=[""] + ARMS)
    ap.add_argument("--width", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--pre", type=int, default=200)
    ap.add_argument("--rel", type=int, default=700)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--o-chunk", type=int, default=8)
    ap.add_argument("--lbfgs-iters", type=int, default=4000)
    ap.add_argument("--fn-iters", type=int, default=10)
    ap.add_argument("--fn-floor", type=float, default=1e-3)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--init-from", default="")
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    a = ap.parse_args(argv)
    if a.task >= 0:
        if a.task >= NTASK:
            ap.error(f"--task must be in 0..{NTASK - 1}")
        if a.width < 0:
            a.width = WIDTHS[a.task // (len(ARMS) * len(SEEDS))]
        if not a.arm:
            a.arm = ARMS[(a.task % (len(ARMS) * len(SEEDS))) // len(SEEDS)]
        if a.seed < 0:
            a.seed = SEEDS[a.task % len(SEEDS)]
    a.width = 6 if a.width < 0 else a.width
    a.seed = SEEDS[0] if a.seed < 0 else a.seed
    if a.mode == "train" and not a.arm:
        ap.error("--mode train needs --task or --arm")
    return a


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.results, exist_ok=True)
    if args.mode == "aggregate":
        return aggregate(args.results, args.instance)
    if args.mode == "time":
        time_step(args)
        return 0
    if args.mode == "smoke":
        t0 = time.time()
        if not run_checks(UHF_SEED):
            return 2
        sdir = os.path.join(HERE, "results_smoke")
        os.makedirs(sdir, exist_ok=True)
        for w in WIDTHS:
            for arm in ARMS:
                for seed in SEEDS:
                    sm = parse_args(["--mode", "train", "--instance", "8", "--arm", arm, "--width", str(w),
                                     "--seed", str(seed), "--pre", "30", "--rel", "60", "--sr-samples", "200",
                                     "--lbfgs-iters", "300", "--fn-iters", "2", "--r-min", "1.0",
                                     "--results", sdir])
                    res = train_task(sm)
                    json.dump(res, open(os.path.join(sdir, f"inst8_{arm}_w{w}_s{seed}.json"), "w"),
                              indent=1, default=float)
        aggregate(sdir, 8)
        log(f"smoke total {time.time() - t0:.1f} s")
        return 0
    out = os.path.join(args.results, f"inst{args.instance}_{args.arm}_w{args.width}_s{args.seed}.json")
    if os.path.exists(out) and not args.overwrite:
        log(f"{out} exists; use --overwrite 1"); return 0
    if not run_checks(UHF_SEED):
        return 2
    res = train_task(args)
    json.dump(res, open(out, "w"), indent=1, default=float)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
