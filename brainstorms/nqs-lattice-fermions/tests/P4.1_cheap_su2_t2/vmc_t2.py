#!/usr/bin/env python3
"""
P4.1 round-7 T2 pilots: VMC-SR for R2 `pcg-scal` (T2a) and for the width premise W of R3, on open Lx x 3 strips.
================================================================================================================

Run with the lit_env interpreter (torch, float64, CPU):
    ~/.conda/envs/lit_env/bin/python vmc_t2.py --mode MODE [options]

This script IMPORTS the frozen size-scaling module ../P4.1_cheap_su2_scaling/vmc.py (geometry, Metropolis, local
energy, SR step, P_G wrapper, exact sums) and never modifies it: job 686990 runs that file. It adds three arms
(Model subclass `DetModel`) and a training driver with exact checkpoint energies and a pre-registered futility stop.

Arms (--arm)
------------
  che    unchanged vmc.Model("che"): closed-form exact-S projection of one charge-backflowed UHF determinant,
         global MLP backflow of hidden width H = --width (the pilot logs/pilot/pg_che is che at H = 16).
  pcg    R2 `pcg-scal` = round-2 r2_micro_test.DetArm("pcg") with exact PER-AMPLITUDE spin projection:
           psi(x) = P_G P_S e^{J(c)} det f_c[U, D],  f_c = B1(c) f0 B2(c)^T,  B_a(c) = 1 + eta_a(c),
           eta_a(c)_{ij} = (tanh(W [d, h] + b) . V_a)_{ij} on the 3x3 neighbourhood mask (|dx|, |dy| <= 1),
           f0 a general L x L pairing matrix (singlet + triplet). Exact S = 0 by
           <x|P_0|phi> = (1/2) sum_q w_q Pf X(u_q), Gauss-Legendre in u = cos(beta) with ceil((N_up + 1)/2) nodes
           (exact: the integrand is a polynomial of degree <= N_up in u), where for c^2 = (1+u)/2, s^2 = (1-u)/2,
           cs = sqrt(1-u^2)/2 and f = f_c, f_a = f - f^T:
             X(u) = [[ -cs f_a[U,U],               c^2 f[U,D] + s^2 f^T[U,D] ],
                     [ -(c^2 f[U,D] + s^2 f^T[U,D])^T,   cs f_a[D,D]            ]]   (N_e x N_e, antisymmetric).
           Pfaffian by batched Parlett-Reid with partial pivoting (log-domain, differentiable). Cost per amplitude
           O(n_q N_e^3) plus O(L^3) for f_c.
  r1     the mandatory singlet charge-backflow foil = DetArm("r1"): det F_c[U, D], F_c = B(c) F0 B(c)^T, F0 = (f0 +
         f0^T)/2, one backflow head. Exact singlet with NO projection (Round 4).
  foil   the mVMC-like foil = DetArm("foil"): P_S e^{J(c)} det f0[U, D] (no backflow), native size (221 parameters).
  Every arm: J(c) = F(c) . jv (vmc.Model.feats), and P_G over {E, Mx, My, MxMy} if --pg is given.
  Width of pcg / r1: the H whose parameter count is closest to che's at H = --match-width (default 16), as in
  r2_micro_test.build_arm. Initialisation: DetArm.init (numpy rng(seed)), bit-identical parameters, so the torch
  amplitudes are checked against the numpy DetArm with full-vector P_S.

Modes (--mode)
--------------
  smoke   Pre-flight verification (save stdout to logs/smoke_output.txt). For pcg, r1, foil:
          (v) validity asserts: target sector exists, N_up = N_dn = N_e/2 <= L, closed-shell Fermi sea, parameter
              count below the (+,+) S = 0 sector dimension at the T2 instance (12 sites) [round-7 minor 7];
          (a) 4x2 (N_e = 6) and 4x3 (N_e = 10), no P_G: torch amplitudes equal numpy DetArm (+ full-vector P_S for
              pcg and foil) to <= 1e-9 (after normalisation);
          (b) 4x2 with P_G (1,1): amplitudes equal numpy P_G [P_S] DetArm (r3_micro_test.SG) to <= 1e-9;
          (c) exact-weighted mean of E_loc equals <psi|H|psi>/<psi|psi> (4x2 and 4x3), <= 1e-9 relative;
          (d) exact gradient from per-sample log-derivatives equals the numpy State gradient (4x2, with and
              without P_G), <= 1e-7 relative  [exact-gradient check of the new ansatz code];
          (e) |<S^2>| of the full torch vector <= 1e-12 (4x3) for pcg, r1 and foil;
          (f) structural positive control: pcg with V2 = V1 and symmetric f0 equals r1 (P_S of an exact singlet is
              itself) to <= 1e-9; negative control: pcg at generic parameters is NOT equal to its unprojected
              determinant (|1 - overlap| > 1e-3), i.e. the projection does something.
  timing  Per-SR-step wall time of che (H = 16), pcg, r1 at Lx in --timing-sizes (Ly = 3), --n-chains chains,
          --timing-steps steps (median), with --pg applied at every size; plus che at --timing-wide (comma list
          "Lx:H", e.g. "8:48") for the W-24 sizing. Writes <outdir>/timing_t2_<host>_t<threads>.json.
  train   Optimise ONE (--arm, --Lx, --width/--match-width, --seed) for --iters SR steps with the vmc.py SR
          settings (lr schedule, Polyak average of the last --avg-frac, checkpoint every --ckpt-every, resume).
          --exact-ckpts "250,500,750": exact energy of the CURRENT parameters at those steps (Lx * Ly <= 12 only).
          --futility "250:0.0229,500:0.0143": pre-registered early KILL: if the exact eps at that step exceeds the
          value, the run stops, is flagged early_kill = step and still writes its JSON (eps of the current state).
          Writes <outdir>/<tag>.json with tag = vmc tag + "_H<H>"; final energy exact (--exact-eval) or sampled
          (--n-eval); exact runs also store <S^2> of the final state. A capped run (--max-hours) writes its JSON with
          capped = true; resubmitting the same command resumes from <tag>_ckpt.npz and overwrites it.
  pipeline  Toy end-to-end pre-flight: trains che, pcg, r1 (and che at H = 2, 16) on 4x2 for --iters steps into
          <outdir>, then the caller runs t2_report.py --mode preflight on that directory.

Options
-------
  --mode {smoke, timing, train, pipeline}     (required)
  --arm {che, pcg, r1, foil}  (che)           arm for train
  --Lx INT (4)  --Ly INT (3)  --Ne INT (auto: vmc.default_ne)  --U FLOAT (8)  --tp FLOAT (-0.25)
  --pg STR ("none")           "cMx,cMy" characters of the target irrep, or "none"
  --width INT (16)            che hidden width H
  --match-width INT (16)      pcg / r1 width is matched to che's parameter count at this H
  --seed INT (0)  --iters INT (1000)  --lr FLOAT (0.05)  --lr-decay FLOAT (0.1)  --avg-frac FLOAT (0.25)
  --sr-norm {0,1} (0)  --diag-shift FLOAT (1e-3)  --max-step FLOAT (0.5)
  --n-chains INT (1024)  --sweeps-per-step FLOAT (0.5)  --therm-sweeps INT (50)
  --n-eval INT (65536)  --exact-eval  --exact-ckpts STR ("")  --futility STR ("")
  --chunk INT (4096)  --grad-chunk INT (256)  --threads INT (3)  --ckpt-every INT (50)  --max-hours FLOAT (2.0)
  --timing-sizes STR ("4,8,16")  --timing-steps INT (2)  --timing-wide STR ("8:48")
  --outdir PATH (<script dir>/results)
"""
import argparse
import json
import math
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCAL = os.path.join(HERE, "..", "P4.1_cheap_su2_scaling")
sys.path.insert(0, SCAL)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vmc as V  # noqa: E402  (frozen module of job 686990; read only)

torch.set_default_dtype(torch.float64)
ARMS = ["che", "pcg", "r1", "foil"]
log = V.log


# ============================================================================ Pfaffian
def slogpf(A):
    """sign and log|Pf| of a batch of antisymmetric matrices A (..., m, m), m even (Parlett-Reid, pivoting)."""
    sh = A.shape[:-2]
    m = A.shape[-1]
    M = A.reshape(-1, m, m)
    nb = M.shape[0]
    sgn = torch.ones(nb, dtype=A.dtype)
    lab = torch.zeros(nb, dtype=A.dtype)
    r = torch.arange(nb)
    while m > 0:
        p = torch.argmax(M[:, 0, 1:].abs().detach(), dim=1) + 1  # pivot column in 1..m-1
        idx = torch.arange(m).repeat(nb, 1)
        idx[r, 1] = p
        idx[r, p] = 1
        sgn = sgn * torch.where(p != 1, -1.0, 1.0)
        M = torch.gather(M, 1, idx[:, :, None].expand(nb, m, m))
        M = torch.gather(M, 2, idx[:, None, :].expand(nb, m, m))
        piv = M[:, 0, 1]
        sgn = sgn * torch.sign(piv)
        lab = lab + torch.log(torch.abs(piv))
        if m == 2:
            break
        tau = M[:, 0, 2:] / piv[:, None]
        c1 = M[:, 2:, 1]
        M = M[:, 2:, 2:] + tau[:, :, None] * c1[:, None, :] - c1[:, :, None] * tau[:, None, :]
        m -= 2
    return sgn.reshape(sh), lab.reshape(sh)


# ============================================================================ new arms
def obc_mask(Lx, Ly):
    L = Lx * Ly
    mask = np.zeros((L, L), bool)
    for x in range(Lx):
        for y in range(Ly):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if 0 <= x + dx < Lx and 0 <= y + dy < Ly:
                        mask[x * Ly + y, (x + dx) * Ly + y + dy] = True
    return mask


def det_count(kind, L, nmask, H):
    nf = L * (L - 1) // 2 + 11
    n = L * L + nf
    if kind == "pcg":
        n += H * (2 * L + 1) + 2 * H * nmask
    elif kind == "r1":
        n += H * (2 * L + 1) + H * nmask
    return n


def det_width(kind, L, Ne, nmask, Hmatch):
    if kind == "foil":
        return 0
    target = V.count("che", L, Ne, Hmatch)
    return min(range(1, 400), key=lambda H: abs(det_count(kind, L, nmask, H) - target))


class DetModel(V.Model):
    """pcg / r1 / foil on the vmc.Model machinery (P_G wrapper, E_loc, Metropolis, log-derivatives)."""

    def __init__(self, arm, Lx, Ly, Ne, H, tp, U, pg=None):
        super().__init__("che", Lx, Ly, Ne, 1, tp, U, pg)  # geometry, features, P_G permutations
        L = self.L
        self.arm, self.K, self.charge = arm, 1, True
        self.H = H
        mask = obc_mask(Lx, Ly)
        mi, mj = np.nonzero(mask)
        self.mi, self.mj, self.nmask = torch.tensor(mi), torch.tensor(mj), len(mi)
        shapes = [("f0", (L, L)), ("jv", (self.nf,))]
        if arm in ("pcg", "r1"):
            shapes += [("W", (H, 2 * L)), ("b", (H,)), ("V1", (H, self.nmask))]
            if arm == "pcg":
                shapes += [("V2", (H, self.nmask))]
        self.shapes = shapes
        self.sizes = [int(np.prod(s)) for _, s in shapes]
        self.nparam = sum(self.sizes)
        assert self.nparam == det_count(arm, L, self.nmask, H)
        from scipy.special import roots_legendre
        u, w = roots_legendre(int(np.ceil((self.Nu + 1) / 2)))
        self.c2 = torch.tensor((1 + u) / 2)
        self.s2 = torch.tensor((1 - u) / 2)
        self.cs = torch.tensor(np.sqrt(1 - u ** 2) / 2)
        self.lw = torch.tensor(np.log(w) + np.log(0.5))

    def init(self, seed, eps=None):
        """DetArm.init, parameter for parameter (numpy rng(seed); seed >= 1 non-r1 arms start from UHF F)."""
        import r2_micro_test as R
        import r3_micro_test as M3
        sec = V.exact_sector(self) if self.L <= 12 else None
        if sec is not None:
            na = R.DetArm(sec, self.arm, self.H, seed)
            assert na.nparam == self.nparam
            return torch.tensor(na.theta0.copy())
        # large clusters (timing only): same distributions without the exact sector
        rng = np.random.default_rng(seed)
        L = self.L
        e, Uh = np.linalg.eigh(-self.Tnp)
        F = Uh[:, :self.Nd] @ Uh[:, :self.Nd].T
        if seed >= 1 and self.arm != "r1":
            lat = M3.CylLattice(self.Lx, self.Ly, self.tp, "obc", "obc")
            Pu, Pd = R.uhf(lat, self.Nu, self.Nd, self.U, seed)
            F = Pu @ Pd.T
        noise = 0.02 if seed == 0 else 0.05
        Rm = rng.standard_normal((L, L))
        p = {"f0": F + noise * (0.5 * (Rm + Rm.T) if self.arm == "r1" else Rm), "jv": np.zeros(self.nf)}
        p["jv"][L * (L - 1) // 2] = -1.0
        if self.arm in ("pcg", "r1"):
            p["W"] = 0.3 * rng.standard_normal((self.H, 2 * L))
            p["b"] = 0.1 * rng.standard_normal(self.H)
            p["V1"] = 0.01 * rng.standard_normal((self.H, self.nmask))
            if self.arm == "pcg":
                p["V2"] = 0.01 * rng.standard_normal((self.H, self.nmask))
        return torch.tensor(np.concatenate([np.asarray(p[n]).ravel() for n, _ in self.shapes]))

    def _bf(self, hh, Vm, rep):
        B = hh.shape[0]
        eta = torch.einsum("bh,bhm->bm", hh, Vm) if rep else hh @ Vm
        E = torch.zeros(B, self.L, self.L, dtype=hh.dtype)
        E = E.index_put((torch.arange(B)[:, None].expand(B, self.nmask), self.mi[None].expand(B, self.nmask),
                         self.mj[None].expand(B, self.nmask)), eta)
        return E + torch.eye(self.L, dtype=hh.dtype)[None]

    def fc(self, p, nU, nD, rep):
        B, L = nU.shape[0], self.L
        f0 = p["f0"]
        if self.arm == "r1":
            f0 = 0.5 * (f0 + f0.transpose(-1, -2))
        if self.arm == "foil":
            return f0.expand(B, L, L) if not rep else f0
        n = nU + nD
        feat = torch.cat([(n == 2).double(), (n == 0).double()], 1)
        if rep:
            hh = torch.tanh(torch.einsum("bhi,bi->bh", p["W"], feat) + p["b"])
        else:
            hh = torch.tanh(feat @ p["W"].T + p["b"])
        B1 = self._bf(hh, p["V1"], rep)
        B2 = self._bf(hh, p["V2"], rep) if self.arm == "pcg" else B1
        return B1 @ f0 @ B2.transpose(1, 2)

    def logpsi0(self, th, nU, nD, rep=False):
        p = self.unpack(th, rep)
        B, n = nU.shape[0], self.Nu
        F, _ = self.feats(nU, nD)
        J = (F * p["jv"]).sum(1) if rep else F @ p["jv"]
        f = self.fc(p, nU, nD, rep)
        U = torch.nonzero(nU > 0.5)[:, 1].reshape(B, n)
        D = torch.nonzero(nD > 0.5)[:, 1].reshape(B, n)
        bi = torch.arange(B)[:, None, None]
        fUD = f[bi, U[:, :, None], D[:, None, :]]
        if self.arm == "r1":
            s, l = torch.linalg.slogdet(fUD)
            return J + l, s
        fTUD = f[bi, D[:, None, :], U[:, :, None]]  # (f^T)[U, D]
        fa = f - f.transpose(1, 2)
        aUU = fa[bi, U[:, :, None], U[:, None, :]]
        aDD = fa[bi, D[:, :, None], D[:, None, :]]
        c2, s2, cs = (t[None, :, None, None] for t in (self.c2, self.s2, self.cs))
        G = c2 * fUD[:, None] + s2 * fTUD[:, None]  # (B, Q, n, n)
        top = torch.cat([-cs * aUU[:, None].expand_as(G), G], 3)
        bot = torch.cat([-G.transpose(2, 3), cs * aDD[:, None].expand_as(G)], 3)
        X = torch.cat([top, bot], 2)
        sP, lP = slogpf(X)
        s, l = self.slse(sP, lP + self.lw[None], 1)
        return J + l, s


def build(a, arm=None, Lx=None, width=None):
    arm = a.arm if arm is None else arm
    Lx = a.Lx if Lx is None else Lx
    L = Lx * a.Ly
    Ne = a.Ne if (Lx == a.Lx and a.Ne is not None) else V.default_ne(L)
    pg = V.pg_of(a)
    if arm == "che":
        return V.Model("che", Lx, a.Ly, Ne, a.width if width is None else width, a.tp, a.U, pg)
    H = det_width(arm, L, Ne, int(obc_mask(Lx, a.Ly).sum()), a.match_width)
    return DetModel(arm, Lx, a.Ly, Ne, H, a.tp, a.U, pg)


def tag_of(a, m):
    pg = "" if V.pg_of(a) is None else "_pg" + a.pg.replace(",", "")
    return f"S{m.Lx}x{m.Ly}_N{m.Ne}_U{a.U:g}_tp{a.tp:g}{pg}_{m.arm}_H{m.H}_seed{a.seed}"


# ============================================================================ train
def train(a):
    torch.set_num_threads(a.threads)
    os.makedirs(a.outdir, exist_ok=True)
    m = build(a)
    tag = tag_of(a, m)
    fout = os.path.join(a.outdir, tag + ".json")
    if os.path.exists(fout):
        old = json.load(open(fout))
        if not (old.get("capped") and old.get("iters_done", 0) < a.iters):
            log(f"{fout} exists; skipping")
            return
        log(f"{fout} was capped at step {old['iters_done']}; resuming from the checkpoint (resubmission)")
    log(f"{tag}: L={m.L} Ne={m.Ne} H={m.H} P={m.nparam} (che@{a.match_width} P={V.count('che', m.L, m.Ne, a.match_width)})"
        f" threads={a.threads} host={socket.gethostname()}")
    exact_ok = m.L <= 12
    sec = V.exact_sector(m) if (exact_ok and (a.exact_eval or a.exact_ckpts)) else None
    ckpts = sorted(int(s) for s in a.exact_ckpts.split(",") if s.strip())
    fut = {int(k): float(v) for k, v in (s.split(":") for s in a.futility.split(",") if s.strip())}
    assert not fut or sec is not None, "futility needs exact checkpoint energies (<= 12 sites)"
    E0 = V.E0_4x3 if (m.Lx, m.Ly, m.Ne) == (4, 3, 10) else None
    if E0 is None and sec is not None and m.L <= 8:  # toy clusters: global S^z = 0 ground energy (ED)
        from scipy.sparse.linalg import eigsh
        E0 = float(eigsh(sec.Hsparse(), k=1, which="SA")[0][0])
    ck = os.path.join(a.outdir, tag + "_ckpt.npz")
    run = V.Run(m, a, a.seed)
    traj, t_used, it0, cke = [], 0.0, 0, {}
    avg_start = int(round(a.iters * (1.0 - a.avg_frac)))
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        th = torch.tensor(z["theta"])
        it0, t_used = int(z["it"]), float(z["t_used"])
        traj = json.loads(str(z["traj"]))
        cke = json.loads(str(z["cke"]))
        avg_sum, avg_n = torch.tensor(z["avg_sum"]), int(z["avg_n"])
        log(f"resumed from {ck} at step {it0} (t_used {t_used / 3600:.2f} h)")
    else:
        th = m.init(a.seed)
        avg_sum, avg_n = torch.zeros_like(th), 0
    acc0 = run.start_chains(th)
    log(f"thermalised: acceptance {acc0:.3f}")
    t_start = time.time()
    capped, early_kill, it = False, None, it0
    for it in range(it0 + 1, a.iters + 1):
        th, info = run.step(th, V.lr_at(a, it))
        if it > avg_start:
            avg_sum += th
            avg_n += 1
        el = t_used + time.time() - t_start
        if it % 10 == 0 or it == 1:
            traj.append([it, info["E"], info["Estd"], el])
        if it % 50 == 0 or it == 1:
            log(f"it {it} E {info['E']:.6f} std {info['Estd']:.3f} acc {info['acc']:.3f} |upd| {info['upd']:.2e} "
                f"t(mcmc/eloc/grad/sr) {info['t_mcmc']:.2f}/{info['t_eloc']:.2f}/{info['t_grad']:.2f}/"
                f"{info['t_sr']:.2f}s elapsed {el / 3600:.3f} h")
        if not np.isfinite(info["E"]):
            raise RuntimeError("non-finite energy")
        if it in ckpts and str(it) not in cke:
            t0 = time.time()
            Ec = V.exact_energy(m, th, sec)[0]
            epsc = (Ec - E0) / abs(E0) if E0 is not None else None
            cke[str(it)] = dict(E=Ec, eps=epsc, wall=el)
            log(f"CHECKPOINT it {it}: exact E {Ec:.7f} eps {epsc if epsc is None else f'{epsc:.4e}'} "
                f"({time.time() - t0:.0f} s)")
            if it in fut and epsc is not None and epsc > fut[it]:
                early_kill = it
                log(f"EARLY-KILL (pre-registered futility): eps({it}) = {epsc:.4e} > {fut[it]:.4e}")
        if it % a.ckpt_every == 0 or it == a.iters or early_kill:
            np.savez(ck, theta=th.numpy(), it=it, t_used=el, traj=json.dumps(traj), cke=json.dumps(cke),
                     avg_sum=avg_sum.numpy(), avg_n=avg_n)
        if early_kill:
            break
        if el > a.max_hours * 3600:
            capped = True
            log(f"wall cap {a.max_hours} h reached at step {it}")
            break
    wall = t_used + time.time() - t_start
    np.save(os.path.join(a.outdir, tag + "_theta_last.npy"), th.numpy())
    if avg_n > 0 and not early_kill:
        th = avg_sum / avg_n
    np.save(os.path.join(a.outdir, tag + "_theta.npy"), th.numpy())
    t0 = time.time()
    S2 = None
    if a.exact_eval:
        E, psi = V.exact_energy(m, th, sec)
        S2 = float(np.sum(psi * sec.S2(psi)))
        dE, ns = 0.0, sec.dim
    else:
        E, dE, ns = run.evaluate(th, a.n_eval)
    log(f"final E ({'current state' if early_kill else f'average of last {avg_n} steps'}) {E:.7f} +- {dE:.7f} "
        f"({ns} samples, {time.time() - t0:.0f} s)" + (f", <S^2> {S2:.1e}" if S2 is not None else ""))
    res = dict(tag=tag, arm=m.arm, Lx=m.Lx, Ly=m.Ly, L=m.L, Ne=m.Ne, U=a.U, tp=a.tp, pg=a.pg, seed=a.seed, H=m.H,
               nparam=m.nparam, iters_done=it, iters=a.iters, capped=capped, early_kill=early_kill, wall=wall,
               threads=a.threads, E=E, dE=dE, n_eval=ns, exact_eval=bool(a.exact_eval), S2=S2, E0=E0,
               eps=((E - E0) / abs(E0) if E0 is not None else None), ckpt_exact=cke, avg_n=avg_n, traj=traj,
               hyper=dict(lr=a.lr, lr_decay=a.lr_decay, avg_frac=a.avg_frac, diag_shift=a.diag_shift,
                          max_step=a.max_step, n_chains=a.n_chains, sweeps_per_step=a.sweeps_per_step),
               host=socket.gethostname())
    with open(fout, "w") as f:
        json.dump(res, f)
    log(f"wrote {fout}")


# ============================================================================ timing
def timing(a):
    torch.set_num_threads(a.threads)
    os.makedirs(a.outdir, exist_ok=True)
    out = dict(host=socket.gethostname(), threads=a.threads, n_chains=a.n_chains, pg=a.pg, rows=[])
    jobs = [(int(s), arm, None) for s in a.timing_sizes.split(",") for arm in ("che", "pcg", "r1")]
    jobs += [(int(s.split(":")[0]), "che", int(s.split(":")[1])) for s in a.timing_wide.split(",") if s.strip()]
    for Lx, arm, Hw in jobs:
        b = argparse.Namespace(**vars(a))
        b.Lx, b.Ne = Lx, V.default_ne(Lx * a.Ly)
        m = build(b, arm, Lx, Hw)
        run = V.Run(m, b, 0)
        th = m.init(0)
        t0 = time.time()
        run.nU, run.nD = m.random_configs(b.n_chains, run.gen)
        run.la, run.sa = m.logpsi_batched(th, run.nU, run.nD, b.chunk)
        bad = ~torch.isfinite(run.la)
        while bad.any():
            u, d = m.random_configs(int(bad.sum()), run.gen)
            run.nU[bad], run.nD[bad] = u, d
            run.la, run.sa = m.logpsi_batched(th, run.nU, run.nD, b.chunk)
            bad = ~torch.isfinite(run.la)
        ts = []
        for _ in range(a.timing_steps):
            t1 = time.time()
            th, info = run.step(th)
            ts.append((time.time() - t1, info))
        tstep = float(np.median([t for t, _ in ts]))
        inf = ts[-1][1]
        row = dict(Lx=Lx, L=m.L, Ne=m.Ne, arm=arm, H=m.H, P=m.nparam, t_step=tstep, t_mcmc=inf["t_mcmc"],
                   t_eloc=inf["t_eloc"], t_grad=inf["t_grad"], t_sr=inf["t_sr"], wall_total=time.time() - t0)
        out["rows"].append(row)
        log(f"L={m.L:2d} Ne={m.Ne} {arm:5s} H={m.H:3d} P={m.nparam:6d} t_step={tstep:8.3f}s (mcmc {inf['t_mcmc']:.2f} "
            f"eloc {inf['t_eloc']:.2f} grad {inf['t_grad']:.2f} sr {inf['t_sr']:.2f}) [{n_ch(a)} chains]")
    for Lx in sorted({r["Lx"] for r in out["rows"]}):
        t = {(r["arm"], r["H"]): r["t_step"] for r in out["rows"] if r["Lx"] == Lx}
        che = t.get(("che", a.width))
        for (arm, H), v in sorted(t.items()):
            if che and (arm, H) != ("che", a.width):
                log(f"L={Lx * a.Ly}: t_step({arm}, H={H}) / t_step(che, H={a.width}) = {v / che:.2f}")
    fn = os.path.join(a.outdir, f"timing_t2_{socket.gethostname()}_t{a.threads}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {fn}")


def n_ch(a):
    return a.n_chains


# ============================================================================ smoke (pre-flight)
def smoke(a):
    import r2_micro_test as R
    import r3_micro_test as M3
    torch.set_num_threads(a.threads)
    t00 = time.time()
    ok_all = True

    def chk(name, val, tol, greater=False):
        nonlocal ok_all
        ok = bool(val > tol) if greater else bool(val <= tol)
        ok_all &= ok
        log(f"  {name}: {val:.2e} ({'>' if greater else '<='} {tol:.0e}) [{'PASS' if ok else 'FAIL'}]")

    # (v) validity asserts at the T2 instance
    b = argparse.Namespace(**vars(a))
    b.Lx, b.Ly, b.Ne, b.pg = 4, 3, 10, "1,1"
    sec12 = V.exact_sector(build(b, "che"))
    dim_sz0 = sec12.dim
    from math import comb
    L, Nu = 12, 5
    dim_sz1 = comb(L, Nu + 1) * comb(L, Nu - 1)
    dimS0 = dim_sz0 - dim_sz1
    log(f"(v) 4x3 open N_e=10: dim(S^z=0) = {dim_sz0}, dim(S=0) = {dimS0}; the (+,+) part is about dim(S=0)/4 = "
        f"{dimS0 // 4}")
    for arm in ("pcg", "r1", "foil"):
        m = build(b, arm)
        assert m.Nu == m.Nd == m.Ne // 2 <= m.L
        ok = m.nparam < dimS0 // 4
        ok_all &= ok
        log(f"  (v) {arm:4s} H={m.H} P={m.nparam} < (+,+) S=0 dimension ~{dimS0 // 4}: [{'PASS' if ok else 'FAIL'}]")
    e = np.linalg.eigh(-V.strip_T(4, 3, -0.25)[0])[0]
    chk("(v) 4x3 Fermi-sea gap (closed shell)", float(e[5] - e[4]), 1e-8, greater=True)

    def numpy_ref(sec, arm, H, th, sgp=None):
        na = R.DetArm(sec, arm, H, 0)
        assert na.nparam == len(th)
        st = M3.State(sec, na, arm in ("pcg", "foil"), sgp, 16384)
        phi, _ = na.forward(th.numpy(), 16384)
        ref = st.psi(phi).ravel()
        return na, st, ref / np.linalg.norm(ref), phi.ravel()

    for (Lx, Ly, Ne) in [(4, 2, 6), (4, 3, 10)]:
        for pg in ([None, (1, 1)] if Lx * Ly <= 8 else [None]):
            b = argparse.Namespace(**vars(a))
            b.Lx, b.Ly, b.Ne, b.pg = Lx, Ly, Ne, ("none" if pg is None else f"{pg[0]},{pg[1]}")
            sec = V.exact_sector(build(b, "che"))
            nU, nD = V.exact_configs(sec)
            sgp = None
            if pg is not None:
                sgp = M3.SG(sec)
                assert [e_[2:] for e_, _ in sgp.el] == [(0, 0), (0, 1), (1, 0), (1, 1)]
                sgp.chi = np.array([1.0, pg[1], pg[0], pg[0] * pg[1]])
            log(f"{Lx}x{Ly} open Ne={Ne} dim={sec.dim} P_G={pg}")
            for arm in ("pcg", "r1", "foil"):
                m = build(b, arm)
                th = m.init(1) + 0.05 * torch.randn(m.nparam, generator=torch.Generator().manual_seed(5))
                na, st, ref, phi_raw = numpy_ref(sec, arm, m.H, th, sgp)
                E_t, psi_t = V.exact_energy(m, th, sec)
                psi_t = psi_t.ravel()
                if psi_t @ ref < 0:
                    psi_t = -psi_t
                lab = "(b)" if pg is not None else "(a)"
                chk(f"{lab} {arm:4s} H={m.H} P={m.nparam} |psi_torch - psi_numpy|_inf", float(np.abs(psi_t - ref).max()), 1e-9)
                l, s = m.logpsi_batched(th, nU, nD, 8192)
                fin = torch.isfinite(l)
                El = torch.zeros(len(l))
                El[fin] = m.local_energy(th, nU[fin], nD[fin], l[fin], s[fin], 8192)
                w = psi_t ** 2
                chk(f"(c) {arm:4s} |sum w E_loc - <H>|/|<H>|", abs(float(np.sum(w * El.numpy())) - E_t) / abs(E_t), 1e-9)
                if Lx * Ly <= 8:
                    E_n, g_n, _ = st.energy_grad(th.numpy())
                    O = torch.zeros(len(l), m.nparam)
                    O[fin] = m.logderiv(th, nU[fin], nD[fin], 512)
                    g_t = 2.0 * O.numpy().T @ (w * (El.numpy() - E_t))
                    chk(f"(d) {arm:4s} |g_torch - g_numpy|/|g_numpy| (E diff {abs(E_n - E_t):.1e})",
                        float(np.linalg.norm(g_t - g_n) / np.linalg.norm(g_n)), 1e-7)
                if pg is None:
                    P = psi_t.reshape(sec.shape)
                    chk(f"(e) {arm:4s} |<S^2>|", abs(float(np.sum(P * sec.S2(P)))), 1e-12)
                if arm == "pcg" and pg is None:
                    # (f) negative control: projection changes a generic pcg state
                    raw = phi_raw / np.linalg.norm(phi_raw)
                    chk("(f-) pcg generic: 1 - |<P_S phi|phi>| (projection acts)", 1 - abs(float(raw @ psi_t)), 1e-3,
                        greater=True)
                    # (f) positive control: symmetric f0 and V2 = V1 -> exact singlet, equals r1 with same params
                    p = {k: v_.clone() for k, v_ in m.unpack(th).items()}
                    p["f0"] = 0.5 * (p["f0"] + p["f0"].T)
                    p["V2"] = p["V1"].clone()
                    thp = torch.cat([p[n].reshape(-1) for n, _ in m.shapes])
                    mr = DetModel("r1", Lx, Ly, Ne, m.H, a.tp, a.U, None)
                    thr = torch.cat([p[n].reshape(-1) for n, _ in mr.shapes])
                    _, pp = V.exact_energy(m, thp, sec)
                    _, pr = V.exact_energy(mr, thr, sec)
                    chk("(f+) pcg(sym f0, V2=V1) vs r1: 1 - |overlap|", 1 - abs(float(pp.ravel() @ pr.ravel())), 1e-9)
    # Pfaffian unit check against det: Pf([[0, M], [-M^T, 0]]) = (-1)^{n(n-1)/2} det M
    Mt = torch.randn(7, 5, 5, generator=torch.Generator().manual_seed(3))
    Z = torch.zeros_like(Mt)
    Xt = torch.cat([torch.cat([Z, Mt], 2), torch.cat([-Mt.transpose(1, 2), Z], 2)], 1)
    sp, lp = slogpf(Xt)
    sd, ld = torch.linalg.slogdet(Mt)
    chk("(p) Pf vs det (block form), max |diff|", float((sp * torch.exp(lp) - (-1) ** 10 * sd * torch.exp(ld)).abs().max()), 1e-12)
    At = torch.randn(4, 8, 8, generator=torch.Generator().manual_seed(4))
    At = At - At.transpose(1, 2)
    sp, lp = slogpf(At)
    chk("(p) Pf^2 vs det (general antisymmetric), max rel diff",
        float(((torch.exp(2 * lp) - torch.linalg.det(At)) / torch.linalg.det(At)).abs().max()), 1e-10)
    log(f"SMOKE {'PASS' if ok_all else 'FAIL'} ({time.time() - t00:.0f} s)")


# ============================================================================ toy pipeline
def pipeline(a):
    os.makedirs(a.outdir, exist_ok=True)
    base = ["--mode", "train", "--Lx", "4", "--Ly", "2", "--Ne", "6", "--iters", str(a.iters), "--n-chains", "256",
            "--threads", str(a.threads), "--outdir", a.outdir, "--exact-eval", "--pg", a.pg]
    h = a.iters // 2
    for arm, extra in [("che", ["--width", "16"]), ("pcg", []), ("r1", []), ("foil", []), ("che", ["--width", "2"])]:
        train(parse_args(base + ["--arm", arm] + extra + ["--exact-ckpts", f"{h}", "--futility", f"{h}:1.0"]))
    # futility path control: a threshold below any possible eps must stop the run at step h with early_kill = h
    train(parse_args(base[:-5] + ["--outdir", os.path.join(a.outdir, "futility_ctrl"), "--exact-eval", "--pg", a.pg,
                                  "--arm", "pcg", "--exact-ckpts", f"{h}", "--futility", f"{h}:-1.0"]))


# ============================================================================ CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="P4.1 round-7 T2 pilots (see module docstring)")
    p.add_argument("--mode", required=True, choices=["smoke", "timing", "train", "pipeline"])
    p.add_argument("--arm", choices=ARMS, default="che")
    p.add_argument("--Lx", type=int, default=4)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--Ne", type=int, default=None)
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--tp", type=float, default=-0.25)
    p.add_argument("--pg", default="none")
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--match-width", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--sr-norm", type=int, default=0, choices=[0, 1])
    p.add_argument("--lr-decay", type=float, default=0.1)
    p.add_argument("--avg-frac", type=float, default=0.25)
    p.add_argument("--diag-shift", type=float, default=1e-3)
    p.add_argument("--max-step", type=float, default=0.5)
    p.add_argument("--n-chains", type=int, default=1024)
    p.add_argument("--sweeps-per-step", type=float, default=0.5)
    p.add_argument("--therm-sweeps", type=int, default=50)
    p.add_argument("--n-eval", type=int, default=65536)
    p.add_argument("--exact-eval", action="store_true")
    p.add_argument("--exact-ckpts", default="")
    p.add_argument("--futility", default="")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--grad-chunk", type=int, default=256)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--max-hours", type=float, default=2.0)
    p.add_argument("--timing-sizes", default="4,8,16")
    p.add_argument("--timing-steps", type=int, default=2)
    p.add_argument("--timing-wide", default="8:48")
    p.add_argument("--outdir", default=os.path.join(HERE, "results"))
    a = p.parse_args(argv)
    if a.Ne is None:
        a.Ne = V.default_ne(a.Lx * a.Ly)
    return a


def main(argv=None):
    a = parse_args(argv)
    {"smoke": smoke, "timing": timing, "train": train, "pipeline": pipeline}[a.mode](a)


if __name__ == "__main__":
    main()
