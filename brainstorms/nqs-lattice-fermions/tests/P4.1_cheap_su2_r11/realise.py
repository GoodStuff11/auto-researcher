#!/usr/bin/env python3
"""
P4.1 Round 11 T1: realise the tangent-space gain of the trained 12-site `che` as real parameter updates.
=======================================================================================================

Run with the lit_env interpreter from this directory:
    ~/.conda/envs/lit_env/bin/python realise.py --mode MODE [options]

Question
--------
R18'' (job 707268) found that one exact linear-method (LM) Ritz step in the full 2637-parameter tangent space of the
trained 12-site `che` (open 4x3 strip, N_e = 10, U = 8, t' = -0.25, irrep (+,+), exact S = 0) removes 0.49 of its
gap, mostly along W2.  R18 and R18' failed to turn that into a parameter update because the gain-carrying tangent
combinations have tiny overlap eigenvalues, so dtheta blows up.  This script tests damped, truncated-metric updates
with EXACT energies (every sum over all 627,264 configurations; no sampling, no seeds).

Pre-registered rule (written before any 'screen' or 'seq' output existed; mode 'rule' prints it)
  g = E(theta0) - E0 is the parent's exact gap (2.2373e-2).  A "step" is one update theta <- theta + s * dtheta built
  from S and H in the tangent space at the current theta (rebuilt after every accepted step), with one of the
  variants below, and a line search s in {1, 1/2, 1/4} on the exact energy (the first s that lowers E is taken; if
  none lowers E the step is rejected and the sequence stops).  Realised fraction R = (E(theta0) - E_final) / g.
    REALISED      if some variant's sequence of at most 20 steps reaches R >= 0.25 with no accepted step raising
                  E by more than 1e-4 g (rejected trial points are reported, they are not steps);
    NOT-REALISED  if no variant reaches R >= 0.10;
    PARTIAL       otherwise.
  Budget clause: the leaf caps this T1 at 30 CPU-minutes.  If the cap ends a sequence before 20 steps while R is
  still rising and below 0.25, the verdict is PARTIAL (budget-limited), never NOT-REALISED, and the remaining steps
  are a prepared job.
  Variant selection: step 1 is screened over the variant grid at theta0 (first-order predicted energies for all,
  exact energies for the best predicted variant per (block, mode, threshold)); the sequence then uses the variant
  with the lowest exact step-1 energy.  That is a line search over the damping, stated as such.
  INVALID if: E(theta0) differs from the stored exact energy by > 1e-8; the orbit-reduced Gram reproduces the
  R18'' full Ritz fraction 0.4895 by more than 0.002 (threshold 1e-11); the positive control (exact ground state
  added as one direction) misses g by more than 1e-6 g; or the orbit reduction fails its checks (below).

Variants (per step)
  block   full (all 2637 parameters) or W2 (the 1920 second-layer weights only; other parameters frozen).
  thr     relative eigenvalue cutoff of the tangent metric S_T (tangents orthogonalised against Psi):
          1e-4, 1e-5, 1e-6, 1e-7, 1e-8.  Kept eigenpairs (w_i, u_i) with w_i > thr * max(w); z = diag(sqrt w) U^T dtheta
          are orthonormal tangent coordinates, and |z| is the metric (Fubini-Study, first-order) norm of the update.
  mode    lm: lowest eigenvector of [[E, g~^T], [g~, H~ + lam]] (level shift lam on the tangent complement of Psi)
              with lam the smallest value giving |z| <= tau (lam = 0 if already inside);
          ng: natural gradient z = -tau g~ / |g~| (the truncated-metric SR direction at metric length tau).
  tau     trust radius in the metric norm: 0.02, 0.05, 0.1, 0.2, 0.4 (lm also 'inf', the undamped truncated LM).
  First-order predicted energy of a candidate: (E + 2 z.g~ + z^T H~ z) / (1 + |z|^2).

Implementation notes
  * P_G = (1/4) sum_g chi_g U_g (vmc.Model perms/chars, by the index maps of tangent.pg_maps).  Every tangent and
    H tangent lies in the (+,+) image, so inner products are sums over orbit representatives weighted by orbit size.
    The reduced H is Hr = H[reps, :] @ E, with E the sparse expansion rep -> full (sign sigma(x)).  Checks: Psi
    expanded from its representatives equals Psi to 1e-12; the weighted norm is 1 to 1e-12; Hr Psi_rep equals
    (H Psi)[reps] to 1e-10.
  * Log-derivatives in theta order as in slack.py (P0, jv, W1, b1, W2, b2), assembled from Gamma = dlog phi0/d out,
    the hidden layer h and the features; slack.py checked the assembly against autograd (4.2e-17).
  * The CPU time of the process is logged after every stage (time.process_time), and the run stops before any
    rebuild that would push it past --cpu-cap minutes.

Modes
  rule     print the rule above.
  screen   build S, H at theta0 (orbit-reduced), run the validations and controls, predict every variant, evaluate
           exact step-1 energies for the best predicted variant per (block, mode, thr), and write
           results/screen.json plus results/S_H_theta0.npz.
  seq      run the step sequence with the variant given by --block --mthd --thr --tau (from the screen), starting
           at theta0 (or at the last accepted point of results/seq_<tag>.json with --resume), and write
           results/seq_<tag>.json and the parameters results/theta_<tag>.npy.
  verdict  apply the rule to results/screen.json and every results/seq_*.json and print the verdict.
Options
  --mode {rule, screen, seq, verdict} (required)
  --threads INT (8)          torch/BLAS threads
  --cpu-cap FLOAT (30)       CPU-minutes for this process; seq stops before a rebuild that would exceed it
  --block {full, W2}         seq only (full)
  --mthd {lm, ng}            seq only (lm)
  --thr FLOAT                seq only (1e-6)
  --tau FLOAT                seq only (0.1; use 0 for 'inf')
  --steps INT (20)           seq only: maximum number of steps
  --resume                   seq only: continue from the last accepted point of results/seq_<tag>.json
"""
import argparse
import json
import os
import sys
import time

THR = "8"
for i, s in enumerate(sys.argv):
    if s == "--threads" and i + 1 < len(sys.argv):
        THR = sys.argv[i + 1]
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = THR

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "P4.1_cheap_su2_r10"))
import tangent as TG  # noqa: E402  (read only)

V = TG.V
log = V.log
torch.set_num_threads(int(THR))
RULE = __doc__.split("Pre-registered rule")[1].split("Variants (per step)")[0]
THRS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
TAUS = [0.02, 0.05, 0.1, 0.2, 0.4]
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def cpu_min():
    return time.process_time() / 60.0


class Ctx:
    """sector, configurations, orbit reduction (built once)."""

    def __init__(self):
        self.m = TG.build("che")
        self.sec = V.exact_sector(self.m)
        self.nU, self.nD = V.exact_configs(self.sec)
        maps = TG.pg_maps(self.m, self.sec)
        self.idx = np.stack([mp[0] for mp in maps])  # (4, dim)
        self.sg = np.stack([mp[1] for mp in maps])
        dim = self.sec.dim
        gstar = np.argmin(self.idx, 0)
        rep_of = self.idx[gstar, np.arange(dim)]
        self.sigma = self.sg[gstar, np.arange(dim)]
        self.reps = np.unique(rep_of)
        pos = np.full(dim, -1)
        pos[self.reps] = np.arange(len(self.reps))
        self.pos_of = pos[rep_of]
        srt = np.sort(self.idx[:, self.reps], 0)
        self.size = 1.0 + (np.diff(srt, axis=0) != 0).sum(0)
        self.sw = np.sqrt(self.size)
        self.E = sp.csr_matrix((self.sigma, (np.arange(dim), self.pos_of)), shape=(dim, len(self.reps)))
        Hs = self.sec.Hsparse()
        self.Hr = (Hs[self.reps, :] @ self.E).tocsr()
        self.gs = TG.sec_ground(self.sec, self.m)
        self.stored = json.load(open(TG.PAR["che"] + ".json"))["E"]
        self.th0 = torch.tensor(np.load(TG.PAR["che"] + "_theta.npy"))
        off = np.cumsum([0] + self.m.sizes[:-1])
        self.off = dict(zip([n for n, _ in self.m.shapes], off.tolist()))
        log(f"orbit reduction: {len(self.reps)} representatives of {dim}; Hr nnz {self.Hr.nnz}")

    def energy(self, th):
        """exact energy of P_G phi0(theta) (normalised), with the representative vector."""
        ls, ss = [], []
        with torch.no_grad():
            for s0 in range(0, self.sec.dim, 8192):
                l, s = self.m.logpsi0(th, self.nU[s0:s0 + 8192], self.nD[s0:s0 + 8192])
                ls.append(l)
                ss.append(s)
        l, s = torch.cat(ls), torch.cat(ss)
        phi0 = (s * torch.exp(l - l.max())).numpy()
        pr = 0.25 * (self.sg[:, self.reps] * phi0[self.idx[:, self.reps]]).sum(0)
        nrm = np.sqrt(np.sum(self.size * pr * pr))
        pr = pr / nrm
        return float(np.sum(self.size * pr * (self.Hr @ pr))), pr, phi0 / nrm


def build(ctx, th, extra_gs=False, checks=False):
    """S and H over [Psi, theta-order tangents (, exact gs)] in the orbit-weighted inner product."""
    m = ctx.m
    l, s, G = TG.phi0_and_grad(m, th, ctx.nU, ctx.nD, "che", 4096, keep64=True)
    phi0 = (s * torch.exp(l - l.max())).numpy()
    pr = 0.25 * (ctx.sg[:, ctx.reps] * phi0[ctx.idx[:, ctx.reps]]).sum(0)
    nrm = np.sqrt(np.sum(ctx.size * pr * pr))
    pr, phi0 = pr / nrm, phi0 / nrm
    EPsi = float(np.sum(ctx.size * pr * (ctx.Hr @ pr)))
    out = {"E": EPsi}
    if checks:
        full = TG.project([(ctx.idx[g], ctx.sg[g]) for g in range(4)], phi0)
        out["chk_expand"] = float(np.abs(ctx.E @ pr - full).max())
        out["chk_norm"] = float(abs(np.sum(full * full) - 1))
        Hf = ctx.sec.Hsparse() @ full
        out["chk_Hr"] = float(np.abs(ctx.Hr @ pr - Hf[ctx.reps]).max())
        log(f"orbit checks: expand {out['chk_expand']:.1e}, norm {out['chk_norm']:.1e}, Hr {out['chk_Hr']:.1e}")
    p = m.unpack(th)
    F, x = m.feats(ctx.nU, ctx.nD)
    with torch.no_grad():
        h = torch.tanh(x @ p["W1"] + p["b1"])
        back = (G @ p["W2"].T) * (1 - h ** 2)
    G, h, back, F, x = (t.numpy() for t in (G, h, back, F, x))
    nrep, P = len(ctx.reps), m.nparam
    ncol = 1 + P + (1 if extra_gs else 0)
    Y = np.empty((nrep, ncol))
    Y[:, 0] = pr * ctx.sw
    CH = 4096
    for r0 in range(0, nrep, CH):
        rr = ctx.reps[r0:r0 + CH]
        acc = np.zeros((len(rr), P))
        for g in range(4):
            ix = ctx.idx[g, rr]
            w = 0.25 * ctx.sg[g, rr] * phi0[ix]
            Gi, hi, bi, Fi, xi = G[ix], h[ix], back[ix], F[ix], x[ix]
            O = np.concatenate([Gi, Fi, (xi[:, :, None] * bi[:, None, :]).reshape(len(rr), -1), bi,
                                (hi[:, :, None] * Gi[:, None, :]).reshape(len(rr), -1), Gi], 1)
            acc += w[:, None] * O
        Y[r0:r0 + CH, 1:1 + P] = acc * ctx.sw[r0:r0 + CH, None]
    if extra_gs:
        Y[:, -1] = ctx.gs[ctx.reps] * ctx.sw
    del G, h, back, F, x
    HY = np.empty_like(Y)
    for c0 in range(0, ncol, 512):
        cb = slice(c0, min(c0 + 512, ncol))
        HY[:, cb] = (ctx.Hr @ (Y[:, cb] / ctx.sw[:, None])) * ctx.sw[:, None]
    S = Y.T @ Y
    Hm = Y.T @ HY
    del Y, HY
    out["S"], out["H"] = 0.5 * (S + S.T), 0.5 * (Hm + Hm.T)
    return out


def ritz(S, Hm, E, thr=1e-11):
    w, U = np.linalg.eigh(S)
    k = w > thr * w.max()
    Pm = U[:, k] / np.sqrt(w[k])
    return E - float(np.linalg.eigvalsh(Pm.T @ Hm @ Pm)[0])


class Tan:
    """tangent model at one theta for a parameter block: orthogonalised against Psi, eigendecomposed metric."""

    def __init__(self, S, Hm, E, cols):
        c = np.asarray(cols) + 1
        s = S[0, c]
        b = Hm[0, c]
        self.ST = S[np.ix_(c, c)] - np.outer(s, s)
        self.HT = Hm[np.ix_(c, c)] - np.outer(s, b) - np.outer(b, s) + E * np.outer(s, s)
        self.g = b - E * s
        self.E = E
        self.w, self.U = np.linalg.eigh(0.5 * (self.ST + self.ST.T))
        self.cols = np.asarray(cols)

    def reduce(self, thr):
        k = self.w > thr * self.w.max()
        Pm = self.U[:, k] / np.sqrt(self.w[k])
        gt = Pm.T @ self.g
        Ht = Pm.T @ self.HT @ Pm
        mu, Q = np.linalg.eigh(0.5 * (Ht + Ht.T))
        return dict(Pm=Pm, gt=gt, Ht=Ht, mu=mu, Q=Q, gq=Q.T @ gt, rank=int(k.sum()))

    def pred(self, r, z):
        return (self.E + 2 * z @ r["gt"] + z @ r["Ht"] @ z) / (1 + z @ z)

    def lm_z(self, r, lam):
        """lowest root of the shifted LM problem by the secular equation; returns z (orthonormal coordinates)."""
        mu, gq, E = r["mu"] + lam, r["gq"], self.E
        f = lambda eps: E - eps - np.sum(gq ** 2 / (mu - eps))  # noqa: E731  decreasing in eps below mu[0]
        hi = mu[0] - 1e-14 * max(1.0, abs(mu[0]))
        lo = min(E, mu[0]) - 1.0 - np.linalg.norm(gq)
        while f(lo) < 0:
            lo -= 10 * (1 + abs(lo))
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        eps = 0.5 * (lo + hi)
        return r["Q"] @ (-gq / (mu - eps))

    def candidate(self, r, mthd, tau):
        if mthd == "ng":
            z = -tau * r["gt"] / np.linalg.norm(r["gt"])
            return z, 0.0
        lam = 0.0
        z = self.lm_z(r, lam)
        if tau and np.linalg.norm(z) > tau:
            lo, hi = 0.0, 1.0
            while np.linalg.norm(self.lm_z(r, hi)) > tau:
                hi *= 4.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if np.linalg.norm(self.lm_z(r, mid)) > tau:
                    lo = mid
                else:
                    hi = mid
            lam = hi
            z = self.lm_z(r, lam)
        return z, lam

    def dtheta(self, r, z, P):
        d = np.zeros(P)
        d[self.cols] = r["Pm"] @ z
        return d


def block_cols(ctx, block):
    if block == "full":
        return list(range(ctx.m.nparam))
    o = ctx.off["W2"]
    return list(range(o, o + ctx.m.H * ctx.m.nout))


def line_search(ctx, th, d, E):
    trials = []
    for s in (1.0, 0.5, 0.25):
        Es = ctx.energy(th + s * torch.tensor(d))[0]
        trials.append((s, Es))
        if Es < E:
            return s, Es, trials
    return None, None, trials


def screen(a):
    t0 = time.time()
    ctx = Ctx()
    b = build(ctx, ctx.th0, extra_gs=True, checks=True)
    E, S, Hm = b["E"], b["S"], b["H"]
    gap = E - TG.E0
    P = ctx.m.nparam
    full = list(range(1, P + 1))
    L_full = ritz(S[np.ix_([0] + full, [0] + full)], Hm[np.ix_([0] + full, [0] + full)], E) / gap
    pos = ritz(S[np.ix_([0, P + 1], [0, P + 1])], Hm[np.ix_([0, P + 1], [0, P + 1])], E)
    ok = (abs(E - ctx.stored) < 1e-8 and abs(L_full - 0.4895) < 0.002 and abs(pos - gap) < 1e-6 * gap
          and b["chk_expand"] < 1e-12 and b["chk_norm"] < 1e-12 and b["chk_Hr"] < 1e-10)
    log(f"E(theta0) {E:.10f} (stored {ctx.stored:.10f}); gap {gap:.6e}; full Ritz fraction {L_full:.4f} "
        f"(R18'' 0.4895); positive control |Delta - g|/g {abs(pos - gap) / gap:.1e}; VALID {ok} "
        f"[{time.time() - t0:.0f} s, {cpu_min():.1f} CPU-min]")
    np.savez(os.path.join(RES, "S_H_theta0.npz"), S=S[:P + 1, :P + 1], H=Hm[:P + 1, :P + 1], E=E)
    rows = []
    for block in ("full", "W2"):
        tan = Tan(S, Hm, E, block_cols(ctx, block))
        L_b = ritz(tan.ST, tan.HT, 0.0) if False else None  # noqa: F841 (block Ritz below via thr 1e-11)
        for thr in THRS:
            r = tan.reduce(thr)
            ritz_frac = (E - tan.pred(r, tan.lm_z(r, 0.0))) / gap
            log(f"{block} thr {thr:.0e}: rank {r['rank']} of {len(tan.cols)}; truncated Ritz fraction {ritz_frac:.4f}")
            for mthd in ("lm", "ng"):
                best = None
                for tau in TAUS + ([0.0] if mthd == "lm" else []):
                    z, lam = tan.candidate(r, mthd, tau)
                    d = tan.dtheta(r, z, P)
                    pf = (E - tan.pred(r, z)) / gap
                    row = dict(block=block, thr=thr, mthd=mthd, tau=tau, lam=lam, zn=float(np.linalg.norm(z)),
                               dth=float(np.linalg.norm(d)), pred=float(pf), rank=r["rank"], ritz=float(ritz_frac))
                    rows.append(row)
                    if best is None or pf > best[0]["pred"]:
                        best = (row, d)
                row, d = best
                s, Es, trials = line_search(ctx, ctx.th0, d, E)
                row["exact"] = dict(s=s, E=Es, frac=None if Es is None else (E - Es) / gap,
                                    trials=[(t, (E - e) / gap) for t, e in trials])
                log(f"  {block} {mthd} thr {thr:.0e} tau {row['tau']}: predicted {row['pred']:.4f}, |z| "
                    f"{row['zn']:.3f}, |dtheta| {row['dth']:.3f}; exact trials (s, fraction) "
                    + ", ".join(f"({t:g}, {f:.4f})" for t, f in row["exact"]["trials"])
                    + f" [{cpu_min():.1f} CPU-min]")
    json.dump(dict(E0_theta=E, gap=gap, L_full=L_full, pos_err=abs(pos - gap) / gap, valid=ok,
                   checks={k: b[k] for k in ("chk_expand", "chk_norm", "chk_Hr")}, rows=rows,
                   cpu_min=cpu_min(), wall=time.time() - t0), open(os.path.join(RES, "screen.json"), "w"), indent=1)
    log(f"wrote results/screen.json ({time.time() - t0:.0f} s, {cpu_min():.1f} CPU-min)")


def seq(a):
    t0 = time.time()
    ctx = Ctx()
    tag = f"{a.block}_{a.mthd}_thr{a.thr:.0e}_tau{a.tau:g}"
    fn = os.path.join(RES, f"seq_{tag}.json")
    th = ctx.th0.clone()
    E00 = ctx.energy(th)[0]
    gap = E00 - TG.E0
    hist = []
    if a.resume and os.path.exists(fn):
        hist = json.load(open(fn))["steps"]
        th = torch.tensor(np.load(os.path.join(RES, f"theta_{tag}.npy")))
    P = ctx.m.nparam
    per_build = None
    while len(hist) < a.steps:
        c0 = cpu_min()
        if per_build is not None and c0 + per_build > a.cpu_cap:
            log(f"stop: next step would exceed the CPU cap ({c0:.1f} + {per_build:.1f} > {a.cpu_cap})")
            break
        b = build(ctx, th)
        E = b["E"]
        tan = Tan(b["S"], b["H"], E, block_cols(ctx, a.block))
        r = tan.reduce(a.thr)
        z, lam = tan.candidate(r, a.mthd, a.tau)
        d = tan.dtheta(r, z, P)
        pf = (E - tan.pred(r, z)) / gap
        rz = (E - tan.pred(r, tan.lm_z(r, 0.0))) / gap
        s, Es, trials = line_search(ctx, th, d, E)
        rec = dict(E=E, R_before=(E00 - E) / gap, pred=pf, ritz=rz, lam=lam, zn=float(np.linalg.norm(z)),
                   dth=float(np.linalg.norm(d)), rank=r["rank"], s=s, E_new=Es,
                   trials=[(t, (E00 - e) / gap) for t, e in trials])
        if s is None:
            rec["accepted"] = False
            hist.append(rec)
            log(f"step {len(hist)}: no lowering trial; stop. trials {rec['trials']}")
            break
        th = th + s * torch.tensor(d)
        rec["accepted"] = True
        rec["R"] = (E00 - Es) / gap
        rec["rise_over_g"] = max(0.0, (Es - E) / gap)
        hist.append(rec)
        per_build = cpu_min() - c0
        log(f"step {len(hist)}: R {rec['R']:.4f} (predicted gain {pf:.4f} of g, truncated Ritz {rz:.4f}); s {s}; "
            f"|z| {rec['zn']:.3f}; |dtheta| {s * rec['dth']:.3f}; rank {r['rank']} [{cpu_min():.1f} CPU-min]")
        np.save(os.path.join(RES, f"theta_{tag}.npy"), th.numpy())
        json.dump(dict(tag=tag, E_theta0=E00, gap=gap, steps=hist, cpu_min=cpu_min(), wall=time.time() - t0),
                  open(fn, "w"), indent=1)
    json.dump(dict(tag=tag, E_theta0=E00, gap=gap, steps=hist, cpu_min=cpu_min(), wall=time.time() - t0),
              open(fn, "w"), indent=1)
    log(f"wrote {os.path.basename(fn)} ({time.time() - t0:.0f} s, {cpu_min():.1f} CPU-min)")


def verdict(a):
    sc = json.load(open(os.path.join(RES, "screen.json")))
    best, bt = 0.0, None
    for row in sc["rows"]:
        ex = row.get("exact")
        if ex and ex["frac"] is not None and ex["frac"] > best:
            best, bt = ex["frac"], f"screen {row['block']} {row['mthd']} thr {row['thr']:.0e} tau {row['tau']}"
    seqs = {}
    for f in sorted(os.listdir(RES)):
        if f.startswith("seq_") and f.endswith(".json"):
            d = json.load(open(os.path.join(RES, f)))
            acc = [s for s in d["steps"] if s.get("accepted")]
            R = acc[-1]["R"] if acc else 0.0
            rise = max([s["rise_over_g"] for s in acc], default=0.0)
            seqs[d["tag"]] = (R, len(acc), rise)
            if rise <= 1e-4 and len(acc) <= 20 and R > best:
                best, bt = R, f"seq {d['tag']} ({len(acc)} steps)"
            print(f"{d['tag']}: R {R:.4f} after {len(acc)} accepted steps; max rise {rise:.1e} g")
    v = "REALISED" if best >= 0.25 else ("NOT-REALISED" if best < 0.10 else "PARTIAL")
    print(f"VALID {sc['valid']}; best realised fraction {best:.4f} ({bt}); VERDICT: {v}")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", required=True, choices=["rule", "screen", "seq", "verdict"])
    pa.add_argument("--threads", default="8")
    pa.add_argument("--cpu-cap", type=float, default=30.0)
    pa.add_argument("--block", default="full", choices=["full", "W2"])
    pa.add_argument("--mthd", default="lm", choices=["lm", "ng"])
    pa.add_argument("--thr", type=float, default=1e-6)
    pa.add_argument("--tau", type=float, default=0.1)
    pa.add_argument("--steps", type=int, default=20)
    pa.add_argument("--resume", action="store_true")
    a = pa.parse_args()
    if a.mode == "rule":
        print(RULE)
    else:
        {"screen": screen, "seq": seq, "verdict": verdict}[a.mode](a)


if __name__ == "__main__":
    main()
