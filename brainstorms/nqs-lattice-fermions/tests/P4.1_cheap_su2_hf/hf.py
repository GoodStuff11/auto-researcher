#!/usr/bin/env python3
"""
P4.1 R3 variant (iii) `che_hf`: hidden-fermion (rational) head under closed-form exact spin projection.
=========================================================================================================

Run with the lit_env interpreter (torch, float64, CPU):
    ~/.conda/envs/lit_env/bin/python hf.py --mode MODE [options]

This module IMPORTS (read only, never modifies) the frozen ../P4.1_cheap_su2_scaling/vmc.py (job 686990) and
../P4.1_cheap_su2_t2/vmc_t2.py (jobs 698224/700568 and, through t2b.py, 699824). It adds one arm and an exact-sum
trainer, and reuses vmc_t2.train for the VMC (T2) mode by swapping vmc_t2.build in THIS process only.

Arm `che_hf` (hidden-fermion determinant with a charge-only hidden network)
---------------------------------------------------------------------------
  For spin s in {up, dn} with N_s electrons and M hidden fermions:
    visible orbitals P_s = [Pa_s | Pb_s]  (L x (N_s + M), free parameters)
    hidden rows      [C_s(c) | D_s(c)] = reshape(W2^T h(c) + b2)   (M x (N_s + M)),  h(c) = tanh(W1^T x(c) + b1),
                     x(c) = 2 [doublon, holon indicators] - 1 (charge only; the same input as `che`)
    augmented det    det [[Pa_s[R_s], Pb_s[R_s]], [C_s, D_s]] = det D_s(c) * det Phieff_s(c)[R_s]
    effective orbitals (Schur complement, rational in h):  Phieff_s(c) = Pa_s - Pb_s D_s(c)^{-1} C_s(c)
  psi(x) = P_G P_S [ e^{J(c)} det D_up(c) det D_dn(c) det Phieff_up(c)[U] det Phieff_dn(c)[D] ].
  Every factor except the two Phieff determinants depends on the charge configuration c only, so P_S acts on one
  c-conditioned UHF determinant and is evaluated EXACTLY per amplitude by the closed-form Che quadrature of
  vmc.Model (the `che` code path, K = 1). Cost per amplitude O(M^3 + L M N_s) for the head plus the che projection.
  Default M = N_s (hidden count = visible count per spin); H is matched to che's parameter count at --match-width.
  Init: Pa = Fermi sea, Pb = 0.3 * N(0,1)/sqrt(L), b2 = [0 | identity], W2 = 1e-3 N(0,1)  (so Phieff ~ Fermi sea,
  det D ~ 1 at start, as che starts with W2 = 1e-3 N(0,1)).

Modes (--mode)
--------------
  smoke   pre-flight checks (save stdout to logs/smoke_output.txt):
          (v) validity: closed-shell Fermi sea; parameter counts below the (+,+) S = 0 dimension at the T1 instance;
          (a) 4x2 N_e = 6, no P_G: the closed-form projected che_hf vector equals the full-vector P_S (sec.projS(0)) of
              its unprojected vector, <= 1e-9 (after normalisation);
          (b) Schur identity: det D * det Phieff[R] equals the augmented (N_s+M) determinant, <= 1e-10 relative;
          (c) |<S^2>| of the projected vector <= 1e-12;
          (d) exact-sum gradient (chunked autograd) against central finite differences along 3 random directions,
              with P_G (1,1), <= 1e-6 relative;
          (e) positive control: che_hf with Pb = 0 and W2 = 0 equals vmc che with P0 = Pa, W2 = b2 = 0 (overlap 1);
              negative control: the projection changes a generic unprojected che_hf state (1 - overlap > 1e-3).
  t1      one exact-sum run: exact energy and exact gradient over the whole S^z = 0 sector with P_G, scipy L-BFGS
          (r2_micro_test.optimise, the T1 protocol of t1_scaling.py), --maxiter iterations, CPU cap --cpu-min.
          Arms: che (vmc.Model, --width), che_hf (--match-width), r1 and pcg (vmc_t2.DetModel, --match-width).
          Writes <outdir>/t1_<Lx>x<Ly>_<arm>_H<H>_s<seed>.json.
  t1report  applies the pre-registered T1 rule (docstring of t1_rule) to <outdir> and prints the verdict.
  train   VMC-SR training of ONE arm with vmc_t2.train (same protocol, checkpoints, futility, cap and resume as T2a);
          --arm che_hf is built here, any other arm by vmc_t2.build. All vmc_t2 options are accepted after "--".
          Example: hf.py --mode train -- --Lx 4 --Ly 3 --pg=1,1 --iters 1000 --exact-eval ...  (no T2 sbatch was kept:
          the T1 rule returned KILL on 2026-11-16, so the 12-site T2 is not run)
  t2report  applies the pre-registered T2 rule (docstring of t2_rule) to the 12-site results and prints the verdict.

Options
-------
  --mode {smoke, t1, t1report, train, t2report}   (required)
  --arm {che, che_hf, r1, pcg} (che_hf)   --Lx INT (5)  --Ly INT (2)  --Ne INT (auto: vmc.default_ne)
  --U FLOAT (8)  --tp FLOAT (-0.25)  --pg STR ("1,1"; "none" for no point-group projection)
  --width INT (4)          che hidden width (t1)
  --match-width INT (4)    che_hf / r1 / pcg are parameter-matched to che at this width
  --M INT (0)              hidden fermions per spin for che_hf; 0 means M = N_s
  --seed INT (0)  --maxiter INT (300)  --threads INT (2)  --cpu-min FLOAT (10)
  --outdir PATH (<script dir>/results)
"""
import argparse
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_thr = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--threads"), "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = _thr
sys.path.insert(0, os.path.join(HERE, "..", "P4.1_cheap_su2_t2"))
sys.path.insert(0, os.path.join(HERE, "..", "P4.1_cheap_su2_scaling"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vmc as V  # noqa: E402  (frozen, read only)
import vmc_t2 as T2  # noqa: E402  (frozen, read only)

torch.set_default_dtype(torch.float64)
log = V.log
E0_5x2 = -6.549056922754901  # ED, open 5x2, N_e = 8, U = 8, t' = -0.25, (+,+) singlet (t1_scaling.py; rechecked in smoke)


# ============================================================================ arm
def hf_count(L, Ne, H, M):
    Ns = Ne // 2
    nf = L * (L - 1) // 2 + 11
    nh = 2 * M * (Ns + M)
    return 2 * L * (Ns + M) + nf + 2 * L * H + H + (H + 1) * nh


def hf_width(L, Ne, M, Hmatch):
    target = V.count("che", L, Ne, Hmatch)
    return min(range(1, 400), key=lambda H: abs(hf_count(L, Ne, H, M) - target))


class HFModel(V.Model):
    def __init__(self, Lx, Ly, Ne, H, M, tp, U, pg=None, project=True):
        super().__init__("che", Lx, Ly, Ne, 1, tp, U, pg)
        L, Ns = self.L, Ne // 2
        self.arm, self.H, self.M, self.project = "che_hf", H, M, project
        self.nh = 2 * M * (Ns + M)
        self.shapes = [("Pu", (L, Ns + M)), ("Pd", (L, Ns + M)), ("jv", (self.nf,)), ("W1", (2 * L, H)),
                       ("b1", (H,)), ("W2", (H, self.nh)), ("b2", (self.nh,))]
        self.sizes = [int(np.prod(s)) for _, s in self.shapes]
        self.nparam = sum(self.sizes)
        assert self.nparam == hf_count(L, Ne, H, M)

    def init(self, seed, eps=None):
        L, Ns, M = self.L, self.Nu, self.M
        e, Uh = np.linalg.eigh(-self.Tnp)
        assert e[Ns] - e[Ns - 1] > 1e-8, "open-shell Fermi sea"
        Phi = Uh[:, :Ns]
        rng = np.random.default_rng(30_000 + seed)
        p = {}
        for s in ("Pu", "Pd"):
            p[s] = np.concatenate([Phi, 0.3 * rng.standard_normal((L, M)) / np.sqrt(L)], 1)
        jv = np.zeros(self.nf)
        jv[L * (L - 1) // 2] = -1.0
        p["jv"] = jv
        p["W1"] = rng.standard_normal((2 * L, self.H)) / np.sqrt(2 * L)
        p["b1"] = np.zeros(self.H)
        p["W2"] = 1e-3 * rng.standard_normal((self.H, self.nh))
        blk = np.concatenate([np.zeros((M, Ns)), np.eye(M)], 1).ravel()
        p["b2"] = np.concatenate([blk, blk])
        return torch.tensor(np.concatenate([np.asarray(p[n]).ravel() for n, _ in self.shapes]))

    def heads(self, p, nU, nD, rep):
        """effective orbitals Ou, Od (B, L, Ns) and log|det D_up det D_dn|, sign."""
        _, x = self.feats(nU, nD)
        B, L, Ns, M = nU.shape[0], self.L, self.Nu, self.M
        if rep:
            h = torch.tanh(torch.einsum("bi,bih->bh", x, p["W1"]) + p["b1"])
            out = torch.einsum("bh,bho->bo", h, p["W2"]) + p["b2"]
        else:
            h = torch.tanh(x @ p["W1"] + p["b1"])
            out = h @ p["W2"] + p["b2"]
        rows = out.reshape(B, 2, M, Ns + M)
        orbs, ls, ss = [], 0.0, 1.0
        for k, P in ((0, p["Pu"]), (1, p["Pd"])):
            C, Dm = rows[:, k, :, :Ns], rows[:, k, :, Ns:]
            Y = torch.linalg.solve(Dm, C)  # (B, M, Ns)
            Pa, Pb = (P[..., :Ns], P[..., Ns:])
            orbs.append(Pa - Pb @ Y if not rep else Pa - torch.einsum("blm,bmn->bln", Pb, Y))
            sD, lD = torch.linalg.slogdet(Dm)
            ls, ss = ls + lD, ss * sD
        return orbs[0], orbs[1], ls, ss

    def logpsi0(self, th, nU, nD, rep=False):
        p = self.unpack(th, rep)
        B, Nu, Nd = nU.shape[0], self.Nu, self.Nd
        F, _ = self.feats(nU, nD)
        J = (F * p["jv"]).sum(1) if rep else F @ p["jv"]
        Ou, Od, lD, sDt = self.heads(p, nU, nD, rep)
        U = torch.nonzero(nU > 0.5)[:, 1].reshape(B, Nu)
        D = torch.nonzero(nD > 0.5)[:, 1].reshape(B, Nd)
        A = torch.gather(Ou, 1, U[:, :, None].expand(B, Nu, Nu))
        Dm = torch.gather(Od, 1, D[:, :, None].expand(B, Nd, Nd))
        if not self.project:
            sA, lA = torch.linalg.slogdet(A)
            sB, lB = torch.linalg.slogdet(Dm)
            return J + lD + lA + lB, sDt * sA * sB
        Bm = torch.gather(Od, 1, U[:, :, None].expand(B, Nu, Nd))
        C = torch.gather(Ou, 1, D[:, :, None].expand(B, Nd, Nu))
        X = C @ torch.linalg.solve(A, Bm)
        sA, lA = torch.linalg.slogdet(A)
        Mq = Dm[:, None] + self.tq[None, :, None, None] * X[:, None]
        sM, lM = torch.linalg.slogdet(Mq)
        sk, lk = self.slse(sM, lM + self.lq[None], 1)
        return J + lD + lA + lk, sDt * sA * sk


def build(arm, Lx, Ly, Ne, tp, U, pg, width, match_width, M=0, project=True):
    L = Lx * Ly
    Ne = V.default_ne(L) if Ne is None else Ne
    if arm == "che":
        return V.Model("che", Lx, Ly, Ne, width, tp, U, pg)
    if arm == "che_hf":
        M = M if M > 0 else Ne // 2
        return HFModel(Lx, Ly, Ne, hf_width(L, Ne, M, match_width), M, tp, U, pg, project)
    H = T2.det_width(arm, L, Ne, int(T2.obc_mask(Lx, Ly).sum()), match_width)
    return T2.DetModel(arm, Lx, Ly, Ne, H, tp, U, pg)


# ============================================================================ exact sums
class ExactState:
    """exact E and gradient over the whole S^z = 0 sector of psi = model.logpsi (P_G and P_S inside the model)."""

    def __init__(self, m, sec, chunk=8192):
        self.m, self.sec, self.chunk = m, sec, chunk
        self.nU, self.nD = V.exact_configs(sec)

    def psi(self, th):
        l, s = self.m.logpsi_batched(th, self.nU, self.nD, self.chunk)
        c = float(l[torch.isfinite(l)].max())
        return (s * torch.exp(l - c)).numpy(), c

    def energy_grad(self, thn):
        th = torch.tensor(thn)
        psi, c = self.psi(th)
        P = psi.reshape(self.sec.shape)
        nrm = float(np.sum(P * P))
        Hp = self.sec.H(P)
        E = float(np.sum(P * Hp) / nrm)
        v = (2.0 * (Hp - E * P) / nrm).ravel()
        fin = np.nonzero(psi != 0.0)[0]
        g = torch.zeros_like(th)
        for s0 in range(0, len(fin), self.chunk):
            idx = torch.tensor(fin[s0:s0 + self.chunk])
            tt = th.clone().requires_grad_(True)
            l, s = self.m.logpsi(tt, self.nU[idx], self.nD[idx])
            (gk,) = torch.autograd.grad((torch.tensor(v[idx.numpy()]) * s * torch.exp(l - c)).sum(), tt)
            g += gk
        return E, g.numpy(), psi


def t1(a):
    import r2_micro_test as R
    torch.set_num_threads(a.threads)
    os.makedirs(a.outdir, exist_ok=True)
    m = build(a.arm, a.Lx, a.Ly, a.Ne, a.tp, a.U, V.pg_of(a), a.width, a.match_width, a.M)
    fout = os.path.join(a.outdir, f"t1_{a.Lx}x{a.Ly}_{a.arm}_H{m.H}_s{a.seed}.json")
    if os.path.exists(fout):
        log(f"{fout} exists; skipping")
        return
    sec = V.exact_sector(m)
    st = ExactState(m, sec)
    th0 = m.init(a.seed).numpy()
    E_init = st.energy_grad(th0)[0]
    log(f"t1 {a.arm} {a.Lx}x{a.Ly} Ne={m.Ne} H={m.H} M={getattr(m, 'M', None)} P={m.nparam} dim={sec.dim} "
        f"E_init={E_init:.6f} threads={a.threads}")
    c0 = time.process_time()
    th, E, it, wall = R.optimise(st, th0, a.maxiter, a.cpu_min / 60.0 / a.threads, logf=log)
    cpu = (time.process_time() - c0) / 60
    _, psi = V.exact_energy(m, torch.tensor(th), sec)
    S2 = float(np.sum(psi * sec.S2(psi)))
    E0 = E0_5x2 if (a.Lx, a.Ly) == (5, 2) else None
    res = dict(arm=a.arm, Lx=a.Lx, Ly=a.Ly, Ne=m.Ne, H=m.H, M=getattr(m, "M", None), P=m.nparam, seed=a.seed,
               E=E, E0=E0, eps=(E - E0) / abs(E0) if E0 else None, iters=it, maxiter=a.maxiter,
               capped=bool(it < a.maxiter), wall_s=wall, cpu_min=cpu, S2=S2)
    log(json.dumps(res))
    with open(fout, "w") as f:
        json.dump(res, f)


def t1_rule(rs):
    """Pre-registered T1 rule for che_hf (written in the leaf before any t1 output existed).

    Instance: open 5x2, N_e = 8, (+,+), 300 L-BFGS iterations, seed 0. eps_X = (E_X - E0)/|E0|.
    Arms: che (H = 4, 620 parameters), che_hf (M = 4, H matched: 620), r1 (matched), che8 (che at H = 8).
      INVALID   if any |<S^2>| > 1e-10.
      INCOMPLETE if any arm is missing. [Amendment 1: every arm is compared at the largest common logged iteration;
                 INCOMPLETE if that is below 100. Original text: "or capped before 150 iterations".]
      NO-TARGET if eps_che8 > 0.8 eps_che (width does not bind on this toy: the rational head has nothing to relieve).
      KILL      if eps_hf >= eps_che, or eps_hf >= eps_r1.
      PASS      if eps_hf <= 0.8 eps_che and eps_hf < eps_r1.
      AMBIGUOUS otherwise.
    """
    need = ["che", "che_hf", "r1", "che8"]
    if any(k not in rs for k in need):
        return "INCOMPLETE", "missing " + ",".join(k for k in need if k not in rs)
    if any(abs(r["S2"]) > 1e-10 for r in rs.values()):
        return "INVALID", "S^2"
    # Amendment 1 (recorded in the leaf; made when the che_hf run had logged only iteration 50): the 7-CPU-minute cap
    # stops every arm before 300 iterations, so every arm is compared at the largest common LOGGED iteration n* (the
    # rule of the round-6 T1, t1_scaling.py); INCOMPLETE if n* < 100.
    common = max(set.intersection(*[set(rs[k]["traj"]) for k in need]) | {0})
    if common < 100:
        return "INCOMPLETE", f"largest common logged iteration {common} < 100"
    e = {k: (rs[k]["traj"][common] - rs[k]["E0"]) / abs(rs[k]["E0"]) for k in need}
    msg = f"at iteration {common}: " + ", ".join(f"eps_{k} {v:.3e}" for k, v in e.items()) + \
        f", rho_hf {e['che_hf'] / e['che']:.3f}"
    if e["che8"] > 0.8 * e["che"]:
        return "NO-TARGET", msg
    if e["che_hf"] >= e["che"] or e["che_hf"] >= e["r1"]:
        return "KILL", msg
    if e["che_hf"] <= 0.8 * e["che"] and e["che_hf"] < e["r1"]:
        return "PASS", msg
    return "AMBIGUOUS", msg


def t1report(a):
    import re
    rs = {}
    for f in glob.glob(os.path.join(a.outdir, f"t1_{a.Lx}x{a.Ly}_*_s{a.seed}.json")):
        r = json.load(open(f))
        key = "che8" if (r["arm"] == "che" and r["H"] == 8) else r["arm"]
        if r["arm"] == "che" and r["H"] not in (4, 8):
            continue
        # trajectory: E_best at every 50th iteration from the run log logs/t1_<arm>[_H<H>].out (and the final E)
        lg = os.path.join(HERE, "logs", f"t1_{r['arm']}" + (f"_H{r['H']}" if r["arm"] == "che" else "") + ".out")
        traj = {int(m_.group(1)): float(m_.group(2))
                for m_ in re.finditer(r"iter\s+(\d+)\s+E_best = (\S+)", open(lg).read())} if os.path.exists(lg) else {}
        traj[r["iters"]] = r["E"]
        r["traj"] = traj
        rs[key] = r
    for k, r in sorted(rs.items()):
        log(f"  {k:7s} H={r['H']} M={r['M']} P={r['P']} eps={r['eps']:.4e} iters={r['iters']} cpu={r['cpu_min']:.1f} min "
            f"S2={r['S2']:.1e}")
    v, msg = t1_rule(rs)
    log(f"VERDICT T1 (R3 variant iii che_hf): {v}  [{msg}]")


# ============================================================================ T2 (VMC, 12 sites)
def t2_rule(e_hf, e_che, e_r1, e_che32, e_pcg, capped=False, early=False):
    """Pre-registered T2 rule for che_hf at 12 sites (open 4x3, N_e = 10, (+,+); written before any T2 output).

    eps from the exact final energy; che = pilot logs/pilot/pg_che (H = 16, 2.858e-3), r1 = T2a task 1 (2.313e-3),
    che32 = W-12 task 5 (che at H = 32, 4957 parameters, 1.352e-3), pcg = T2a task 0 (1.284e-3, reference only).
      KILL       if the futility stop fired (eps(250) > 2.29e-2 or eps(500) > 1.43e-2, the T2a multiples).
      INCOMPLETE if che_hf is missing or capped.
      KILL       if eps_hf >= 0.8 eps_che, or eps_hf >= eps_r1  (no matched-parameter gain over the affine head or the
                 projection-free singlet foil).
      PASS       if eps_hf <= eps_che32  (the rational head at 2637 parameters reaches the width-doubled affine head at
                 4957: it relieves the width bottleneck at fixed parameter count).
      AMBIGUOUS  otherwise (gain over che and r1, but below the width-doubled control): run seed 1, best of seeds; if
                 still ambiguous, AMBIGUOUS-FINAL (no promotion).
    """
    if early:
        return "KILL", "futility"
    if e_hf is None or capped:
        return "INCOMPLETE", "che_hf missing or capped"
    msg = (f"eps_hf {e_hf:.3e}, rho_che {e_hf / e_che:.3f}, hf/r1 {e_hf / e_r1:.3f}, hf/che32 {e_hf / e_che32:.3f}, "
           f"hf/pcg {e_hf / e_pcg:.3f} (reference)")
    if e_hf >= 0.8 * e_che or e_hf >= e_r1:
        return "KILL", msg
    if e_hf <= e_che32:
        return "PASS", msg
    return "AMBIGUOUS", msg


T2REF = dict(che=2.858e-3, r1=None, che32=None, pcg=None)


def _eps(path):
    if not os.path.exists(path):
        return None, False, False
    r = json.load(open(path))
    return r.get("eps"), bool(r.get("capped")), bool(r.get("early_kill"))


def t2report(a):
    t2dir = os.path.join(HERE, "..", "P4.1_cheap_su2_t2", "results")
    e_r1 = _eps(os.path.join(t2dir, "t2a", "S4x3_N10_U8_tp-0.25_pg11_r1_H25_seed0.json"))[0]
    e_pcg = _eps(os.path.join(t2dir, "t2a", "S4x3_N10_U8_tp-0.25_pg11_pcg_H15_seed0.json"))[0]
    e_c32 = _eps(os.path.join(t2dir, "w12", "S4x3_N10_U8_tp-0.25_pg11_che_H32_seed0.json"))[0]
    best = None
    for f in sorted(glob.glob(os.path.join(a.outdir, "t2", "S4x3_N10_*_che_hf_H*_seed*.json"))):
        e, cap, early = _eps(f)
        log(f"  {os.path.basename(f)}: eps {e} capped {cap} early_kill {early}")
        if best is None or (e is not None and (best[0] is None or e < best[0])):
            best = (e, cap, early)
    best = best or (None, False, False)
    v, msg = t2_rule(best[0], T2REF["che"], e_r1, e_c32, e_pcg, best[1], best[2])
    log(f"VERDICT T2 (R3 variant iii che_hf, 12 sites): {v}  [{msg}]")


def _check_t2_rule():
    ok = True
    for args, want in [((1.0e-3, 2.858e-3, 2.313e-3, 1.352e-3, 1.284e-3), "PASS"),
                       ((2.0e-3, 2.858e-3, 2.313e-3, 1.352e-3, 1.284e-3), "AMBIGUOUS"),
                       ((2.5e-3, 2.858e-3, 2.313e-3, 1.352e-3, 1.284e-3), "KILL"),
                       ((2.2e-3, 2.858e-3, 2.1e-3, 1.352e-3, 1.284e-3), "KILL"),
                       ((None, 2.858e-3, 2.313e-3, 1.352e-3, 1.284e-3), "INCOMPLETE")]:
        got = t2_rule(*args)[0]
        ok &= got == want
        log(f"  t2_rule{args[:1]}: {got} (want {want})")
    got = t2_rule(1e-3, 2.858e-3, 2.313e-3, 1.352e-3, 1.284e-3, early=True)[0]
    ok &= got == "KILL"
    log(f"  t2_rule futility: {got} (want KILL)")
    return ok


# ============================================================================ smoke
def smoke(a):
    import r3_micro_test as M3
    torch.set_num_threads(a.threads)
    t00 = time.time()
    ok_all = True

    def chk(name, val, tol, greater=False):
        nonlocal ok_all
        ok = bool(val > tol) if greater else bool(val <= tol)
        ok_all &= ok
        log(f"  {name}: {val:.2e} ({'>' if greater else '<='} {tol:.0e}) [{'PASS' if ok else 'FAIL'}]")

    # (v) validity at the T1 instance (5x2, N_e = 8) and the T2 instance (4x3, N_e = 10)
    from math import comb
    for (Lx, Ly, Ne, Hm) in [(5, 2, 8, 4), (4, 3, 10, 16)]:
        L, Ns = Lx * Ly, Ne // 2
        dimS0 = comb(L, Ns) ** 2 - comb(L, Ns + 1) * comb(L, Ns - 1)
        e = np.linalg.eigh(-V.strip_T(Lx, Ly, -0.25)[0])[0]
        chk(f"(v) {Lx}x{Ly} Fermi-sea gap", float(e[Ns] - e[Ns - 1]), 1e-8, greater=True)
        for arm in ("che", "che_hf", "r1"):
            m = build(arm, Lx, Ly, Ne, -0.25, 8.0, (1, 1), Hm, Hm)
            ok = m.nparam < dimS0 // 4
            ok_all &= ok
            log(f"  (v) {Lx}x{Ly} {arm:6s} H={m.H} M={getattr(m, 'M', '-')} P={m.nparam} (che P="
                f"{V.count('che', L, Ne, Hm)}) < (+,+) S=0 dim ~{dimS0 // 4}: [{'PASS' if ok else 'FAIL'}]")
    # ED reference of the T1 instance
    m = build("che", 5, 2, 8, -0.25, 8.0, (1, 1), 4, 4)
    sec = V.exact_sector(m)
    sg = M3.SG(sec)
    E0 = M3.ground_state(sec, sg)[1]
    chk("(v) ED E0(5x2, (+,+)) vs stored", abs(E0 - E0_5x2), 1e-9)

    # (a)-(c), (e) on 4x2, N_e = 6, no P_G
    mP = build("che_hf", 4, 2, 6, -0.25, 8.0, None, 4, 4)
    mR = HFModel(4, 2, 6, mP.H, mP.M, -0.25, 8.0, None, project=False)
    sec = V.exact_sector(mP)
    nU, nD = V.exact_configs(sec)
    th = mP.init(1) + 0.1 * torch.randn(mP.nparam, generator=torch.Generator().manual_seed(7))
    _, psiP = V.exact_energy(mP, th, sec)
    l, s = mR.logpsi_batched(th, nU, nD, 8192)
    raw = (s * torch.exp(l - l.max())).numpy().reshape(sec.shape)
    ref = sec.projS(0)(raw)
    ref = ref / np.linalg.norm(ref)
    psiP = psiP / np.linalg.norm(psiP)
    if np.sum(ref * psiP) < 0:
        ref = -ref
    log(f"4x2 Ne=6 che_hf H={mP.H} M={mP.M} P={mP.nparam} dim={sec.dim}")
    chk("(a) |closed-form P_S psi - full-vector P_S psi|_inf", float(np.abs(ref - psiP).max()), 1e-9)
    # (b) Schur identity on 64 configurations
    p = mR.unpack(th)
    idx = torch.arange(0, len(nU), max(1, len(nU) // 64))
    _, x = mR.feats(nU[idx], nD[idx])
    out = torch.tanh(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
    rows = out.reshape(len(idx), 2, mR.M, mR.Nu + mR.M)
    big = torch.ones(len(idx))
    for k, (P, occ) in enumerate(((p["Pu"], nU[idx]), (p["Pd"], nD[idx]))):
        R_ = torch.nonzero(occ > 0.5)[:, 1].reshape(len(idx), -1)
        top = P[R_]
        big = big * torch.linalg.det(torch.cat([top, rows[:, k]], 1))
    lr_, sr_ = mR.logpsi(th, nU[idx], nD[idx])
    F, _ = mR.feats(nU[idx], nD[idx])
    small = sr_ * torch.exp(lr_ - F @ p["jv"])
    chk("(b) Schur: max |det D det Phieff[R] - det(augmented)| / max|.|",
        float((small - big).abs().max() / big.abs().max()), 1e-10)
    chk("(c) |<S^2>| of the projected vector", abs(float(np.sum(psiP * sec.S2(psiP.reshape(sec.shape))))), 1e-12)
    rawn = raw.ravel() / np.linalg.norm(raw)
    chk("(e-) generic che_hf: 1 - |<P_S phi|phi>| (projection acts)", 1 - abs(float(rawn @ psiP.ravel())), 1e-3,
        greater=True)
    # (e+) Pb = 0, W2 = 0 -> che with P0 = Pa, W2 = b2 = 0
    q = {k: v.clone() for k, v in mP.unpack(th).items()}
    q["Pu"][:, mP.Nu:] = 0.0
    q["Pd"][:, mP.Nu:] = 0.0
    q["W2"] = torch.zeros_like(q["W2"])
    thq = torch.cat([q[n].reshape(-1) for n, _ in mP.shapes])
    mc = V.Model("che", 4, 2, 6, mP.H, -0.25, 8.0, None)
    pc = mc.unpack(mc.init(0))
    pc = {k: v.clone() for k, v in pc.items()}
    pc["P0"] = torch.cat([q["Pu"][:, :mP.Nu].reshape(-1), q["Pd"][:, :mP.Nu].reshape(-1)])
    pc["jv"], pc["W1"], pc["b1"] = q["jv"], q["W1"], q["b1"]
    pc["W2"] = torch.zeros_like(pc["W2"])
    pc["b2"] = torch.zeros_like(pc["b2"])
    thc = torch.cat([pc[n].reshape(-1) for n, _ in mc.shapes])
    _, a1 = V.exact_energy(mP, thq, sec)
    _, a2 = V.exact_energy(mc, thc, sec)
    chk("(e+) che_hf(Pb=0, W2=0) vs che(P0=Pa): 1 - |overlap|", 1 - abs(float(a1.ravel() @ a2.ravel())), 1e-12)
    # (d) exact-sum gradient vs finite differences, with P_G
    mG = build("che_hf", 4, 2, 6, -0.25, 8.0, (1, 1), 4, 4)
    stG = ExactState(mG, sec)
    th0 = th.numpy()
    E, g, _ = stG.energy_grad(th0)
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(3):
        d = rng.standard_normal(len(th0))
        d /= np.linalg.norm(d)
        h = 1e-5
        fd = (stG.energy_grad(th0 + h * d)[0] - stG.energy_grad(th0 - h * d)[0]) / (2 * h)
        worst = max(worst, abs(fd - g @ d) / max(abs(fd), 1e-12))
    chk("(d) exact gradient vs central FD (3 directions, P_G (1,1))", worst, 1e-6)
    log("(r) T2 rule logic:")
    okr = _check_t2_rule()
    ok_all &= okr
    log(f"SMOKE {'PASS' if ok_all else 'FAIL'} ({time.time() - t00:.0f} s)")


# ============================================================================ VMC train (T2)
def train(a, rest):
    targs = T2.parse_args(["--mode", "train"] + [r for r in rest if r != "--"])
    targs.arm = a.arm
    targs.threads = a.threads
    if a.arm == "che_hf":
        def _build(b, arm=None, Lx=None, width=None):
            return build("che_hf", b.Lx, b.Ly, b.Ne, b.tp, b.U, V.pg_of(b), b.width, b.match_width, a.M)
        T2.build = _build  # this process only; vmc_t2.py on disk is unchanged
    T2.train(targs)


# ============================================================================ CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["smoke", "t1", "t1report", "train", "t2report"])
    p.add_argument("--arm", default="che_hf", choices=["che", "che_hf", "r1", "pcg"])
    p.add_argument("--Lx", type=int, default=5)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--Ne", type=int, default=None)
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--tp", type=float, default=-0.25)
    p.add_argument("--pg", default="1,1")
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--match-width", type=int, default=4)
    p.add_argument("--M", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--cpu-min", type=float, default=10.0, dest="cpu_min")
    p.add_argument("--outdir", default=os.path.join(HERE, "results"))
    return p.parse_known_args(argv)


def main(argv=None):
    a, rest = parse_args(argv)
    if a.mode == "train":
        train(a, rest)
        return
    {"smoke": smoke, "t1": t1, "t1report": t1report, "t2report": t2report}[a.mode](a)


if __name__ == "__main__":
    main()
