#!/usr/bin/env python3
"""
P4.1 R40: charge-only vs spin-reading backflow, both exactly projected onto S = 0, ladder-polished, 12 sites.
=============================================================================================================
Rule: RULE.md (pre-registered, this folder). Run with the lit_env interpreter:
    ~/.conda/envs/lit_env/bin/python r40.py --mode MODE [--arm ARM] [options]

Arms (identical NNBF architecture, H = 16, P = 2637; only the backflow input differs)
  che   V.Model('che', 16): input x = 2[d, h] - 1 (doublon/holon occupations), exact P_S in closed form (Che
        quadrature). Start: ../P4.1_cheap_su2_r5/results/..._che_drift4000_seed0_theta.npy (seed-0 common
        reference, 4000 exact-gradient L-BFGS iterations; eps 3.0585e-4).
  nnbf  V.Model('nnbf', 16) = `nnbf_SG`: input x = 2[n_up, n_dn] - 1 (reads the spins), exact P_S applied to the
        full 627,264-dim vector by a block-diagonal S = 0 projector (S^2 is block diagonal in the charge
        configuration; blocks <= 252). Start: results/theta_nnbf_pre.npy from --mode pretrain (seed-0 common
        reference, the SAME init and L-BFGS options as the legacy nnbf_SG / che runs, 4000 iterations).
Both carry P_G (+,+) through the r11 orbit reduction (157,216 representatives).

Modes
  check     validity checks (writes results/checks.json): block P_S == sec.projS(0) on random vectors; nnbf seed-0
            init energy == legacy nnbf_SG E_init (-6.282599213109587); che drift4000 energy == its record
            (-7.825504891193657); finite-difference check of the nnbf energy gradient; nnbf ladder tangent column
            == finite difference of the projected state.
  time      per-component timings for the per-step cost model (partial build, extrapolated; writes
            results/timing.json). Interactive use only under the 10 CPU-minute allowance.
  pretrain  nnbf L-BFGS-B (maxcor 30, ftol 1e-15, gtol 1e-10; legacy r2_micro_test.optimise) for --pre-iters
            iterations from V.Model('nnbf').init(0); keeps the best theta; trajectory every 50 iterations.
  arm       cutoff-ladder polish of --arm (protocol of ../P4.1_cheap_su2_r22/ladder.py, unchanged) until STOP-RULE,
            STOP-NOGAIN or --steps (hard cap 30); resumable from results/state_<arm>.json.
  verdict   RULE.md: rho = eps_che / eps_nnbf at the endpoints and tail-extrapolated (RULE_amend2 guard).
"""
import argparse
import json
import os
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check", "time", "pretrain", "arm", "verdict"])
    ap.add_argument("--arm", choices=["che", "nnbf"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--pre-iters", type=int, default=4000, dest="pre_iters")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--cpu-cap", type=float, default=1e9, dest="cpu_cap")
    ap.add_argument("--res", default=None)
    a = ap.parse_args()
    if a.mode == "arm" and a.arm is None:
        ap.error("--arm is required for mode arm")
    return a


A = parse_args()
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = A.threads  # L63: before the first numpy import
sys.argv = [sys.argv[0], "--threads", A.threads]

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
R19 = os.path.join(TESTS, "P4.1_cheap_su2_r19")
sys.path.insert(0, R19)
import polish as P  # noqa: E402  (read only; imports r11 realise, r10 tangent, scaling vmc)

R, V, log = P.R, P.V, P.log
E0 = P.E0
TAU, MIN_GAIN, STOP_FRAC = 0.05, 1e-7, 1e-3
LADDER = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
HARD_CAP = 30
G_COMMON = 2.5e-4 * abs(E0)  # RULE.md: one absolute reference gap for both arms' STOP-RULE
NO_LOSS, LOSS = 1.1, 1.5     # RULE.md thresholds on rho = eps_che / eps_nnbf
CHE_START = os.path.join(TESTS, "P4.1_cheap_su2_r5/results/L4x3xobcobc_N10_U8_tp-0.25_w16_che_drift4000_seed0")
# Legacy fix_test.py code (current version) evaluated at its own nnbf_SG seed-0 init (identical theta to
# V.Model('nnbf').init(0)); run locally 2026-11-17 (logs/legacy_init.txt). The value recorded by job 451886
# (-6.282599213109587) is not reproduced by the current legacy code either (its che E_init -6.27701 vs -6.28046 now),
# so the 451886 init differed slightly; the check is against the current legacy code, not the 451886 record.
E_INIT_LEGACY_NNBF = -6.2824656105282575
RES = A.res or os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
c0 = time.process_time()
cpu = lambda: (time.process_time() - c0) / 60  # noqa: E731


def model(arm):
    return V.Model(arm, 4, 3, 10, 16, -0.25, 8.0, (1, 1))


# ------------------------------------------------------------------ exact S = 0 projector (block diagonal in c)
class SpinProj:
    def __init__(self, sec):
        dim = sec.dim
        bu = sec.up.bits[sec.iu].astype(float)
        bd = sec.dn.bits[sec.id].astype(float)
        diag = (bu * (1 - bd)).sum(1) + (sec.Sz ** 2 - sec.Sz)
        M = sp.diags(diag).tocsr()
        for Eu, Ed in sec.spsm_pairs():
            M = M - sp.kron(Eu, Ed, format="csr")  # S2(P) -= Eu P Ed^T  <=>  vec: kron(Eu, Ed)
        self.S2 = M.tocsr()
        cid = sec.cidx
        order = np.argsort(cid, kind="stable")
        cnt = np.bincount(cid)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        pos = np.empty(dim, dtype=np.int64)
        pos[order] = np.arange(dim) - start[cid[order]]
        coo = M.tocoo()
        assert np.all(cid[coo.row] == cid[coo.col]), "S^2 not block diagonal in c"
        self.groups = []
        for d in np.unique(cnt):
            cs = np.nonzero(cnt == d)[0]
            gi = np.full(len(cnt), -1)
            gi[cs] = np.arange(len(cs))
            idxs = order[start[cs][:, None] + np.arange(d)[None, :]]
            B = np.zeros((len(cs), d, d))
            sel = cnt[cid[coo.row]] == d
            np.add.at(B, (gi[cid[coo.row[sel]]], pos[coo.row[sel]], pos[coo.col[sel]]), coo.data[sel])
            w, U = np.linalg.eigh(B)
            keep = (np.abs(w) < 0.5).astype(float)  # S(S+1) = 0; next level 2
            Pb = np.einsum("bik,bk,bjk->bij", U, keep, U)
            self.groups.append((idxs, Pb))
        self.sumd2 = int(sum(Pb.shape[0] * Pb.shape[1] ** 2 for _, Pb in self.groups))

    def apply(self, X):
        Y = np.empty_like(X)
        for idxs, Pb in self.groups:
            Y[idxs] = Pb @ X[idxs] if X.ndim == 2 else np.einsum("bij,bj->bi", Pb, X[idxs])
        return Y


# ------------------------------------------------------------------ contexts
class NnbfCtx(R.Ctx):
    """r11 orbit-reduced sector with the spin-reading NNBF and the full-vector exact P_S."""

    def __init__(self):
        super().__init__()
        self.m = model("nnbf")
        self.arm = "nnbf"
        self.PS = SpinProj(self.sec)

    def phi(self, th):
        ls, ss = [], []
        with torch.no_grad():
            for s0 in range(0, self.sec.dim, 8192):
                l, s = self.m.logpsi0(th, self.nU[s0:s0 + 8192], self.nD[s0:s0 + 8192])
                ls.append(l)
                ss.append(s)
        l, s = torch.cat(ls), torch.cat(ss)
        lmax = float(l.max())
        return (s * torch.exp(l - lmax)).numpy(), lmax

    def reduce(self, q):
        return 0.25 * (self.sg[:, self.reps] * q[self.idx[:, self.reps]]).sum(0)

    def energy(self, th):
        """exact energy of P_G P_S phi(theta); (E, normalised representative vector, phi / nrm)."""
        phi, _ = self.phi(th)
        pr = self.reduce(self.PS.apply(phi))
        nrm = np.sqrt(np.sum(self.size * pr * pr))
        pr = pr / nrm
        return float(np.sum(self.size * pr * (self.Hr @ pr))), pr, phi / nrm

    def energy_grad(self, th_np, CH=8192):
        """E and dE/dtheta: grad = (2/|P phi|^2) sum_x [(H - E) P phi](x) phi(x) d log|phi(x)| / dtheta."""
        th = torch.tensor(th_np)
        phi, lmax = self.phi(th)
        pr = self.reduce(self.PS.apply(phi))
        n2 = float(np.sum(self.size * pr * pr))
        Hpr = self.Hr @ pr
        E = float(np.sum(self.size * pr * Hpr)) / n2
        v = self.E @ (Hpr - E * pr)  # (H - E) P phi on the full space (P_G P_S-symmetric, so P v = v)
        w = torch.tensor(2.0 * v * phi / n2)
        thg = th.clone().requires_grad_(True)
        for s0 in range(0, self.sec.dim, CH):
            l, _ = self.m.logpsi0(thg, self.nU[s0:s0 + CH], self.nD[s0:s0 + CH])
            (w[s0:s0 + CH] * l).sum().backward()
        return E, thg.grad.numpy().copy()


def build_nnbf(ctx, th, CH=2048, CB=256, rows=None, cols=None):
    """Y = orbit-weighted [Psi, P_G P_S d phi / d theta] / nrm and HY over all representatives (as polish.build).
    rows / cols restrict the work (timing only)."""
    E, pr, phi0 = ctx.energy(th)
    Pn, nrep = ctx.m.nparam, len(ctx.reps)
    nr = nrep if rows is None else rows
    Y = np.empty((nrep, 1 + Pn))
    Y[:, 0] = pr * ctx.sw
    for r0 in range(0, nr, CH):
        rr = ctx.reps[r0:min(r0 + CH, nr)]
        acc = np.zeros((len(rr), Pn))
        for g in range(4):
            ix = ctx.idx[g, rr]
            acc += (0.25 * ctx.sg[g, rr] * phi0[ix])[:, None] * P.logder(ctx.m, th, ctx.nU[ix], ctx.nD[ix])
        Y[r0:r0 + len(rr), 1:] = acc
    nc = Pn if cols is None else cols
    for c in range(1, 1 + nc, CB):
        cb = slice(c, min(c + CB, 1 + nc))
        Y[:, cb] = ctx.PS.apply(ctx.E @ Y[:, cb])[ctx.reps] * ctx.sw[:, None]
    HY = np.empty_like(Y)
    for c in range(0, 1 + nc, 512):
        cb = slice(c, min(c + 512, 1 + nc))
        HY[:, cb] = (ctx.Hr @ (Y[:, cb] / ctx.sw[:, None])) * ctx.sw[:, None]
    return E, Y, HY, pr


def che_ctx():
    ctx = R.Ctx()
    ctx.m = model("che")
    ctx.arm = "che"
    return ctx


def setup(arm):
    """(ctx, build, theta_start, checks)."""
    if arm == "che":
        ctx = che_ctx()
        th = torch.tensor(np.load(CHE_START + "_theta.npy"))
        Erec = json.load(open(CHE_START + ".json"))["E"]
        d = abs(ctx.energy(th)[0] - Erec)
        return ctx, P.build, th, dict(start_vs_record=d, valid=bool(d < 1e-8))
    ctx = NnbfCtx()
    f = os.path.join(RES, "pretrain_nnbf.json")
    if not os.path.exists(f):
        return ctx, build_nnbf, None, dict(valid=False, why="no pretrain_nnbf.json")
    pre = json.load(open(f))
    th = torch.tensor(np.load(os.path.join(RES, "theta_nnbf_pre.npy")))
    d = abs(ctx.energy(th)[0] - pre["E"])
    return ctx, build_nnbf, th, dict(start_vs_pretrain=d, pretrain_valid=pre["valid"],
                                     valid=bool(d < 1e-10 and pre["valid"]))


def dump(fn, d):
    json.dump(d, open(fn, "w"), indent=1)


# ------------------------------------------------------------------ modes
def check(a):
    out = {}
    ctx = NnbfCtx()
    sec, PS = ctx.sec, ctx.PS
    rng = np.random.default_rng(0)
    projS = sec.projS(0)
    errs = []
    for _ in range(2):
        x = rng.standard_normal(sec.dim)
        errs.append(float(np.abs(PS.apply(x) - projS(x.reshape(sec.shape)).ravel()).max()))
    out["PS_vs_projS"] = max(errs)
    th0 = ctx.m.init(0)
    out["nnbf_init_vs_legacy"] = abs(ctx.energy(th0)[0] - E_INIT_LEGACY_NNBF)
    E, g = ctx.energy_grad(th0.numpy())
    k = rng.choice(ctx.m.nparam, 4, replace=False)
    fd = []
    for j in k:
        e = np.zeros(ctx.m.nparam)
        e[j] = 1e-5
        fd.append((ctx.energy(th0 + torch.tensor(e))[0] - ctx.energy(th0 - torch.tensor(e))[0]) / 2e-5)
    out["grad_fd_rel"] = float(np.abs(np.array(fd) - g[k]).max() / np.abs(g).max())
    cctx = che_ctx()
    out["che_start_vs_record"] = abs(cctx.energy(torch.tensor(np.load(CHE_START + "_theta.npy")))[0]
                                     - json.load(open(CHE_START + ".json"))["E"])
    out["sum_block_d2"] = PS.sumd2
    out["valid"] = bool(out["PS_vs_projS"] < 1e-10 and out["nnbf_init_vs_legacy"] < 1e-8 and out["grad_fd_rel"] < 1e-5
                        and out["che_start_vs_record"] < 1e-8)
    out["cpu_min"] = cpu()
    log(f"checks {out}")
    dump(os.path.join(RES, "checks.json"), out)


def timing(a):
    out = {}
    t = cpu()
    ctx = NnbfCtx()
    out["setup_nnbf"] = cpu() - t
    th = ctx.m.init(0)
    t = cpu()
    ctx.energy(th)
    out["energy_nnbf"] = cpu() - t
    t = cpu()
    ctx.energy_grad(th.numpy())
    out["energy_grad_nnbf"] = cpu() - t
    nrep, Pn = len(ctx.reps), ctx.m.nparam
    rows, cols = 8192, 256
    t = cpu()
    build_nnbf(ctx, th, rows=rows, cols=0)
    t_rows = cpu() - t - out["energy_nnbf"]
    t = cpu()
    build_nnbf(ctx, th, rows=0, cols=cols)
    t_cols = cpu() - t - out["energy_nnbf"]
    # tangent-column check, in two parts (a full build is not run interactively):
    # (i) the build's projection pipeline PS(E @ reduce(v))[reps] equals reduce(PS(v)) for v = d phi / d theta_k
    #     (central finite difference on the full space); (ii) phi * logder equals that finite difference on the
    #     first 8192 configurations.
    def phis(t):
        l, s = ctx.m.logpsi0(t, ctx.nU, ctx.nD)
        return (s * torch.exp(l - 50.0)).numpy()
    errs, errs2 = [], []
    with torch.no_grad():
        p0 = phis(th)
    G = P.logder(ctx.m, th, ctx.nU[:8192], ctx.nD[:8192])
    for k in (5, 700, 2000):
        e = torch.zeros(Pn, dtype=th.dtype)
        e[k] = 1e-5
        with torch.no_grad():
            v = (phis(th + e) - phis(th - e)) / 2e-5
        ref = ctx.reduce(ctx.PS.apply(v))
        pipe = ctx.PS.apply(ctx.E @ ctx.reduce(v))[ctx.reps]
        errs.append(float(np.abs(pipe - ref).max() / np.abs(ref).max()))
        errs2.append(float(np.abs(p0[:8192] * G[:, k] - v[:8192]).max() / np.abs(v[:8192]).max()))
    out["pipeline_rel"] = max(errs)
    out["logder_fd_rel"] = max(errs2)
    out["logder_rows_per_8192"] = t_rows
    out["PS_and_H_per_256_cols"] = t_cols
    tY = t_rows * nrep / rows
    tP = t_cols * Pn / cols
    # S = Y^T Y, H = Y^T HY (two (nrep x P)^T (nrep x P) products), and the tangent eigendecomposition: from r22
    # che steps (same nrep, P 2637): step total minus che's build and line-search energies is the dense part.
    out["build_extrapolated"] = out["energy_nnbf"] + tY + tP
    out["cpu_min_total"] = cpu()
    log(f"timing {out}")
    dump(os.path.join(RES, "timing.json"), out)


def pretrain(a):
    from scipy.optimize import minimize
    ctx = NnbfCtx()
    th0 = ctx.m.init(0).numpy()
    E_init = ctx.energy(torch.tensor(th0))[0]
    best = dict(E=np.inf, th=th0.copy())
    it, traj, t0, flag = [0], [], time.time(), dict(capped=False)

    def f(th):
        E, g = ctx.energy_grad(th)
        if not np.isfinite(E):
            return 1e3, np.zeros_like(th)
        if E < best["E"]:
            best["E"], best["th"] = E, th.copy()
        return E, g

    def cb(th):
        it[0] += 1
        if it[0] % 50 == 0:
            traj.append([it[0], best["E"]])
            log(f"  iter {it[0]:5d}  E_best {best['E']:.10f}  eps {(best['E'] - E0) / abs(E0):.4e}  "
                f"[{cpu():.1f} CPU-min]")
            np.save(os.path.join(RES, "theta_nnbf_pre_ckpt.npy"), best["th"])
        if cpu() > a.cpu_cap:
            flag["capped"] = True  # newer scipy swallows StopIteration from callbacks, so record it here
            raise StopIteration
    try:
        minimize(f, th0, jac=True, method="L-BFGS-B", callback=cb,
                 options=dict(maxiter=a.pre_iters, maxcor=30, ftol=1e-15, gtol=1e-10))
    except StopIteration:
        flag["capped"] = True
    capped = flag["capped"]
    np.save(os.path.join(RES, "theta_nnbf_pre.npy"), best["th"])
    eps = (best["E"] - E0) / abs(E0)
    e_at = {n: (E - E0) / abs(E0) for n, E in traj}
    # RULE.md INVALID: init energy must equal the legacy nnbf_SG E_init; the run must not be capped; eps at
    # iteration 2000 must lie in [0.8 x 1.885e-4, 1.2 x 2.306e-4] (legacy 3-seed range, widened 20%).
    e2000 = e_at.get(2000)
    valid = bool(abs(E_init - E_INIT_LEGACY_NNBF) < 1e-8 and not capped and e2000 is not None
                 and 0.8 * 1.885e-4 <= e2000 <= 1.2 * 2.306e-4)
    out = dict(E=best["E"], eps=eps, E_init=E_init, iters=it[0], capped=capped, eps_at_1000=e_at.get(1000),
               eps_at_2000=e2000, valid=valid, traj=traj, cpu_min=cpu(), wall_s=time.time() - t0)
    dump(os.path.join(RES, "pretrain_nnbf.json"), out)
    log(f"pretrain: eps {eps:.5e} after {it[0]} iterations; valid {valid} [{cpu():.1f} CPU-min]")


def run_arm(a):
    t0 = time.time()
    fn, fth = os.path.join(RES, f"state_{a.arm}.json"), os.path.join(RES, f"theta_{a.arm}.npy")
    ctx, build, th, chk = setup(a.arm)
    log(f"{a.arm}: P {ctx.m.nparam}; checks {chk} [{cpu():.2f} CPU-min]")
    st = dict(arm=a.arm, g=G_COMMON, checks=chk, ladder=LADDER, tau=TAU, min_gain=MIN_GAIN, stop_frac=STOP_FRAC,
              cap=a.steps, steps=[], stop=None)
    if os.path.exists(fn):
        old = json.load(open(fn))
        st.update(steps=old["steps"], stop=old["stop"], E_start=old["E_start"])
        if st["stop"]:
            log(f"{a.arm}: already finished ({st['stop']}); nothing to do")
            return
        if old["steps"]:
            th = torch.tensor(np.load(fth))
    if not chk["valid"]:
        st["stop"] = "INVALID"
        dump(fn, st)
        log(f"{a.arm}: INVALID checks; stop")
        return
    E = ctx.energy(th)[0]
    st.setdefault("E_start", E)
    Pn, per = ctx.m.nparam, None
    while st["stop"] is None:
        if len(st["steps"]) >= min(a.steps, HARD_CAP):
            st["stop"] = "STOP-CAP"
            break
        s0 = cpu()
        if per is not None and s0 + per > a.cpu_cap:
            log(f"stop: next step would exceed the CPU cap ({s0:.1f} + {per:.1f} > {a.cpu_cap}); resumable")
            break
        Eb, Y, HY, pr = build(ctx, th)
        assert abs(Eb - E) < 1e-9, (Eb, E)
        S, Hm = P.exact_SH(Y, HY)
        del Y, HY, pr
        tan = R.Tan(S, Hm, E, list(range(Pn)))
        del S, Hm
        rungs, best, prev = [], None, None
        for c in LADDER:
            r = tan.reduce(c)
            if r["rank"] == prev:
                rungs.append(dict(cut=c, rank=r["rank"], skipped=True))
                continue
            prev = r["rank"]
            z, lam = tan.candidate(r, "lm", TAU)
            d = tan.dtheta(r, z, Pn)
            acc, trials = None, []
            for s in (1.0, 0.5, 0.25):
                Es = ctx.energy(th + s * torch.tensor(d))[0]
                trials.append((s, (Es - E0) / abs(E0)))
                if np.isfinite(Es) and (E - Es) / abs(E0) > MIN_GAIN:
                    acc = (s, Es)
                    break
            row = dict(cut=c, rank=r["rank"], zn=float(np.linalg.norm(z)), lam=lam,
                       pred_full=E - tan.pred(r, z), trials=trials, s=None)
            if acc:
                row.update(s=acc[0], E_new=acc[1], pred=E - tan.pred(r, acc[0] * z))
                if best is None or acc[1] < best["E_new"]:
                    best = dict(row, d=d)
            rungs.append(row)
        rec = dict(E=E, eps=(E - E0) / abs(E0), rungs=rungs, cpu_min=None)
        if best is None:
            rec["accepted"] = False
            st["steps"].append(rec)
            st["stop"] = "STOP-NOGAIN"
            break
        th = th + best["s"] * torch.tensor(best["d"])
        E = best["E_new"]
        per = cpu() - s0
        rec.update(accepted=True, cut=best["cut"], s=best["s"], E_new=E, eps_new=(E - E0) / abs(E0),
                   gain=rec["E"] - E, pred=best["pred"], pred_over_g=best["pred"] / G_COMMON, cpu_min=per)
        st["steps"].append(rec)
        np.save(fth, th.numpy())
        if best["pred"] < STOP_FRAC * G_COMMON:
            st["stop"] = "STOP-RULE"
        log(f"step {len(st['steps'])}: cut {best['cut']:.0e} s {best['s']} eps {rec['eps']:.5e} -> "
            f"{rec['eps_new']:.5e}; predicted gain {rec['pred_over_g']:.2e} g [{cpu():.1f} CPU-min, step {per:.2f}, "
            f"wall {time.time() - t0:.0f} s]{' ' + st['stop'] if st['stop'] else ''}")
        st["cpu_min_total"] = cpu()
        dump(fn, st)
    st["cpu_min_total"] = cpu()
    dump(fn, st)
    log(f"{a.arm}: stop {st['stop']}; {len(st['steps'])} steps [{cpu():.1f} CPU-min]")


FIT_N, FIT_RMS_MAX = 5, float(np.log(2.0))  # RULE_amend2 guard of r22, unchanged


def tail_fit(gains):
    g = np.asarray(gains[-FIT_N:], float)
    out = dict(n=len(g), r=None, rms=None, tail=np.inf, unsettled=False, why=[])
    if len(g) < FIT_N or np.any(g <= 0):
        out.update(unsettled=True, why=["fewer than 5 positive gains"])
        return out
    k = np.arange(len(g), dtype=float)
    b, a_ = np.polyfit(k, np.log(g), 1)
    rms = float(np.sqrt(np.mean((np.log(g) - (a_ + b * k)) ** 2)))
    r = float(np.exp(b))
    out.update(r=r, rms=rms)
    if r >= 1:
        out["why"].append(f"fitted ratio {r:.3f} >= 1")
    else:
        out["tail"] = float(np.exp(a_ + b * k[-1]) * r / (1 - r))
    if rms > FIT_RMS_MAX:
        out["why"].append(f"RMS log-residual {rms:.3f} > ln 2")
    out["unsettled"] = bool(out["why"])
    return out


def endpoint(st):
    acc = [s for s in st["steps"] if s.get("accepted")]
    E = acc[-1]["E_new"] if acc else st["E_start"]
    eps = (E - E0) / abs(E0)
    tail, fit = 0.0, None
    if st["stop"] == "STOP-CAP":
        fit = tail_fit([s["gain"] for s in acc])
        tail = fit["tail"]
        if not np.isfinite(tail) or eps - tail / abs(E0) <= 0:
            fit["unsettled"] = True
            fit["why"].append("eps_inf <= 0 or tail unbounded")
    eps_inf = max(eps - tail / abs(E0), 0.0) if np.isfinite(tail) else 0.0
    return eps, eps_inf, len(acc), fit


def cls(rho):
    return "NO-LOSS" if rho <= NO_LOSS else "LOSS" if rho >= LOSS else "INTERMEDIATE"


def verdict(a):
    S = {k: json.load(open(os.path.join(RES, f"state_{k}.json"))) for k in ("che", "nnbf")
         if os.path.exists(os.path.join(RES, f"state_{k}.json"))}
    out = dict(arms={})
    for k, st in S.items():
        eps, eps_inf, n, fit = endpoint(st)
        out["arms"][k] = dict(stop=st["stop"], steps=n, eps_start=(st["E_start"] - E0) / abs(E0), eps=eps,
                              eps_inf=eps_inf, tail_fit=fit, checks=st["checks"])
        print(f"{k}: stop {st['stop']}; {n} steps; eps {out['arms'][k]['eps_start']:.5e} -> {eps:.5e} "
              f"(tail-extrapolated {eps_inf:.5e}); fit {fit}")
    stops = [st["stop"] for st in S.values()]
    if len(S) < 2 or any(s is None for s in stops):
        out["verdict"] = "INCOMPLETE"
    elif "INVALID" in stops:
        out["verdict"] = "INVALID"
    else:
        c, n = out["arms"]["che"], out["arms"]["nnbf"]
        rho = c["eps"] / n["eps"]
        rho_inf = c["eps_inf"] / n["eps_inf"] if n["eps_inf"] > 0 else float("inf")
        bad = [k for k in ("che", "nnbf") if (out["arms"][k]["tail_fit"] or {}).get("unsettled")]
        c1, c2 = cls(rho), cls(rho_inf)
        v = c1 if (c1 == c2 and not bad) else "UNSETTLED"
        if v == "UNSETTLED" and all(out["arms"][k]["steps"] >= HARD_CAP for k in out["arms"]
                                    if out["arms"][k]["stop"] == "STOP-CAP"):
            v = f"AMBIGUOUS ({c1} at endpoints, {c2} extrapolated)"
        out.update(rho=rho, rho_inf=rho_inf, cls_end=c1, cls_inf=c2, unsettled_arms=bad, verdict=v)
        print(f"rho = eps_che / eps_nnbf = {rho:.4f} (tail-extrapolated {rho_inf:.4f})")
    print(f"VERDICT: {out['verdict']}")
    dump(os.path.join(RES, "verdict.json"), out)


if __name__ == "__main__":
    {"check": check, "time": timing, "pretrain": pretrain, "arm": run_arm, "verdict": verdict}[A.mode](A)
