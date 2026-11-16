#!/usr/bin/env python3
"""
P4.1 size-scaling decisive test: VMC (Metropolis + stochastic reconfiguration) for the exact-S
rank-K projected mixture `che_kK` and the plain spin-resolved NNBF, on open Lx x 3 Hubbard strips.
====================================================================================================

Run with the lit_env interpreter (torch 2.11, float64, CPU):
    ~/.conda/envs/lit_env/bin/python vmc.py --mode MODE [options]

Physics (fixed for the whole ladder)
------------------------------------
  Hubbard model on an Lx x Ly strip, OPEN along x and y, site index i = x*Ly + y,
  t = 1 on nearest-neighbour bonds, t' (--tp) on the diagonal bonds, on-site U (--U).
  N_e = 5/6 * Lx * Ly electrons (N_up = N_dn = N_e / 2, S^z = 0). Basis convention (identical
  to tests/P4.1_cheap_su2_r5): |x> = (up string in site order) x (down string in site order),
  so a spin-sigma hop j -> i carries (-1)^{# sigma electrons strictly between i and j}.
  Ladder: Lx in {4, 8, 12, 16}, Ly = 3 -> N_sites = 12, 24, 36, 48; N_e = 10, 20, 30, 40.
  Lx = 4 is the instance of jobs 451886 / 508440 (exact reference E_0 = -7.827899034271753).

Arms (--arm)
------------
  che      K = 1: closed-form exact-S (Che) projection of one charge-conditioned UHF determinant.
           phi(x) = e^{J(c)} <x|P_{S=0}|Phi_up(c), Phi_dn(c)>, Phi(c) = P0 + W2 tanh(W1 x_c + b1) + b2,
           x_c = 2 [doublon_i, holon_i] - 1 (charge-only input, 2 L features).
  che_k2, che_k4, che_k8
           rank-K mixture: phi(x) = e^{J(c)} sum_k a_k <x|P_{S=0}|Phi^(k)(c)>, a_0 = 1,
           Phi^(k)(c) = P0_k + W2 h(c) + b2 (ONE shared backflow, fixed offsets P0_k).
           Same parameter layout as r5_test.CheK. Cost: K closed-form projections per amplitude.
  nnbf     plain spin-resolved NNBF (S^z conserved, NO spin projection, the scalable baseline):
           phi(x) = e^{J(c)} det Phi_up(x)[U] det Phi_dn(x)[D], input 2 [n_up, n_dn] - 1.
           Same layout as r2_micro_test.NNBF.
  Every arm: J(c) = F(c) . jv, F = [n_i n_j (i<j), sum_i d_i, 10 doublon-holon neighbour counts]
  (r2_micro_test.charge_features). No space-group projection anywhere (all sizes, all arms).
  Closed form used: <x|P_0|Phi> = (1/2) det A sum_q w_q c_q^{N_e} det(Dm + t_q^2 C A^{-1} B), with
  A = Phi_up[U], B = Phi_dn[U], C = Phi_up[D], Dm = Phi_dn[D], Gauss-Legendre in u = cos(beta) with
  ceil((N_up + 1)/2) nodes (exact; r3_micro_test.che_single). Cost O(N^3) per component.
  Widths: nnbf and che use H = --width (16). che_kK uses the largest H whose parameter count does not
  exceed that of nnbf at the same size (matched-or-fewer parameters at every size).
  Initialisation (common start, as r5_test.build_r5): P0 = tight-binding Fermi sea for both spins,
  jv = Gutzwiller (-1 on the double-occupancy count), W2 ~ 1e-3 N(0,1), W1 ~ N(0, 1/(2L)), b = 0;
  che_kK offsets P0_k = P0 + 0.05 rms(P0) xi_k, a_k = 1. RNG streams 10000+seed and 20000+seed.

Optimisation (identical for every arm and size)
------------------------------------------------
  Metropolis on |psi|^2: --n-chains persistent chains; one proposal = pick a bond uniformly (NN or
  diagonal) and one of {hop up (0.4), hop down (0.4), exchange of opposite single spins (0.2)};
  symmetric proposal, rejected if not applicable. --sweeps-per-step x N_e proposals per chain between
  SR steps; one sample per chain per step (N_samples = n_chains).
  SR (sample-space Woodbury form): dtheta = -lr * Ob^T (Ob Ob^T + diag_shift I)^{-1} eb,
  Ob = (O - <O>)/sqrt(N), eb = (E_loc - <E_loc>)/sqrt(N), O = d log|psi| / d theta per sample.
  Update norm capped at --max-step. Final energy: --n-eval samples (chain-blocked error bar), or the
  exact sum over the whole sector when --exact-eval is given (Lx * Ly <= 12 only).

Modes (--mode)
--------------
  smoke   Verification (all output to stdout; save it to logs/smoke_output.txt):
          (a) hopping matrix equals r3_micro_test.CylLattice(obc, obc) on 4x3;
          (b) 4x2 open, N_e = 6 and 4x3 open, N_e = 10: torch amplitudes of every arm equal the
              numpy reference (r5_test.CheK / r3 ChargeNNBF with full-vector P_S; r2 NNBF) up to
              normalisation, to <= 1e-9;
          (c) exact-weighted mean of E_loc equals <psi|H|psi>/<psi|psi> from the sparse sector H
              (checks local energy and fermion signs), <= 1e-9 relative;
          (d) exact gradient from the per-sample log-derivatives equals the numpy State gradient
              (4x2, every arm), <= 1e-7 relative;
          (e) 4x3: Metropolis-sampled energy and gradient vs the exact ones (z-scores);
          (f) 4x3: a short SR run of `che` lowers the exact energy (code runs; not evidence).
  smoke-pg  P_G verification against exact sums (numpy r3_micro_test.SG projector) for every arm:
          4x2 (N_e = 6) in irreps (Mx, My) = (1,1), (-1,1), (1,-1): amplitudes, E_loc identity and exact
          gradient vs numpy State; 4x3 (N_e = 10) in the irrep given by --pg: amplitudes, E_loc identity,
          Metropolis-sampled energy and gradient vs exact (save to logs/smoke_pg_output.txt).
  timing  Per-sample wall time (one E_loc + one log-derivative + MCMC share) and per-SR-step wall
          time for every arm at every ladder size, with --timing-steps SR steps; writes
          <outdir>/timing_<host>.json. The nnbf equal-wall-time iteration count is derived from it.
  train   Optimise ONE (--Lx, --arm, --seed) for --iters SR steps (checkpoint every --ckpt-every
          steps to <outdir>/<tag>_ckpt.npz; restart resumes from it). If --extra-checkpoints is
          given (comma list of step numbers), the parameters at those steps are saved and evaluated
          too (used for nnbf: equal-iteration and equal-wall-time points). Writes <outdir>/<tag>.json.
  report  Aggregate result JSONs + DMRG references (dmrg/*.json) and print the pre-registered
          size-scaling verdict (see scaling_report.py; this mode just calls it).

Options
-------
  --Lx INT (4)            strip length; ladder uses 4, 8, 12, 16
  --Ly INT (3)            strip width (3 for the whole ladder; 2 allowed for smoke clusters)
  --Ne INT (auto)         electrons; default round(5/6 * Lx * Ly) to the nearest even integer
  --U FLOAT (8.0)  --tp FLOAT (-0.25)
  --arm {che, che_k2, che_k4, che_k8, nnbf}   (train; default che)
  --pg STR ("none")       point-group projection of EVERY arm: "none", or "cMx,cMy" with cMx, cMy in {1, -1},
                          the characters of the target irrep under the x-mirror and the y-mirror; then
                          psi(y) = (1/4) sum_{g in E,Mx,My,MxMy} chi(g) sigma_g(y) phi(g y) (4 amplitudes per psi)
  --width INT (16)        reference width H (nnbf, che); che_kK width is derived (see above)
  --seed INT (0)
  --iters INT (3000)      SR steps (train)
  --extra-checkpoints STR ("")   e.g. "2000" (nnbf: also evaluate at this step)
  --lr FLOAT (0.05)       SR learning rate (peak; see --lr-decay)
  --sr-norm {0, 1} (0)    1 = scale-invariant SR (each parameter normalised by sqrt(S_kk), so diag-shift is
                          relative to the S diagonal); 0 = plain SR with an absolute diag-shift
  --lr-decay FLOAT (0.1)  lr is constant for the first half of --iters, then cosine-decays to lr * lr_decay
  --avg-frac FLOAT (0.25) the reported state is the uniform average of the parameters over the last
                          avg_frac * iters steps (Polyak averaging; removes SR noise from the final state);
                          the last-step parameters are saved as <tag>_theta_last.npy
  --diag-shift FLOAT (1e-3)
  --max-step FLOAT (0.5)  cap on ||lr * dtheta||_2
  --n-chains INT (2048)   chains = samples per SR step
  --sweeps-per-step FLOAT (0.5)   proposals per chain per step = ceil(value * N_e)
  --therm-sweeps INT (50) initial thermalisation sweeps (x N_e proposals)
  --n-eval INT (65536)    final-evaluation samples (spread over the chains, 2 sweeps apart)
  --exact-eval            final energy by exact sum (only Lx*Ly <= 12)
  --chunk INT (4096)      configurations per amplitude batch
  --grad-chunk INT (256)  samples per log-derivative batch
  --threads INT (16)      torch threads
  --ckpt-every INT (50)
  --max-hours FLOAT (150) wall cap: the run stops, evaluates and is flagged "capped"
  --timing-sizes STR ("4,8,12,16")  --timing-steps INT (3)
  --outdir PATH (<script dir>/results)
"""
import argparse
import json
import math
import os
import socket
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)
ARMS = ["che", "che_k2", "che_k4", "che_k8", "nnbf"]
LADDER = [4, 8, 12, 16]
E0_4x3 = -7.827899034271753


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ geometry
def default_ne(L):
    return int(2 * round(5 * L / 12))


def strip_T(Lx, Ly, tp):
    L = Lx * Ly
    T = np.zeros((L, L))
    NN = np.zeros((L, L))
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            for dx, dy, a in [(1, 0, 1.0), (0, 1, 1.0), (1, 1, tp), (1, -1, tp)]:
                X, Y = x + dx, y + dy
                if 0 <= X < Lx and 0 <= Y < Ly:
                    j = X * Ly + Y
                    T[i, j] += a
                    T[j, i] += a
                    if a == 1.0 and (dx == 0 or dy == 0):
                        NN[i, j] = NN[j, i] = 1.0
    return T, NN


# ============================================================================ model
def kofarm(arm):
    return 1 if arm in ("che", "nnbf") else int(arm.split("_k")[1])


def count(arm, L, Ne, H):
    nf = L * (L - 1) // 2 + 11
    nout = L * Ne
    K = kofarm(arm)
    base = nout + nf + 2 * L * H + H + H * nout + nout
    return base + (K - 1) * nout + (K - 1)


def width_for(arm, L, Ne, Href):
    if arm in ("che", "nnbf"):
        return Href
    P = count("nnbf", L, Ne, Href)
    ok = [H for H in range(1, Href + 1) if count(arm, L, Ne, H) <= P]
    return max(ok) if ok else 1  # 1 only on tiny smoke clusters


class Model:
    def __init__(self, arm, Lx, Ly, Ne, H, tp, U, pg=None):
        self.arm, self.Lx, self.Ly, self.L, self.Ne = arm, Lx, Ly, Lx * Ly, Ne
        self.Nu = self.Nd = Ne // 2
        self.H, self.U, self.tp = H, U, tp
        self.K = kofarm(arm)
        self.charge = arm != "nnbf"
        L = self.L
        T, NN = strip_T(Lx, Ly, tp)
        self.Tnp = T
        self.NN = torch.tensor(NN)
        iu, ju = np.triu_indices(L, 1)
        self.iu, self.ju = torch.tensor(iu), torch.tensor(ju)
        bi, bj = np.nonzero(np.triu(T, 1))
        self.bi, self.bj = torch.tensor(bi), torch.tensor(bj)
        self.bt = torch.tensor(T[bi, bj])
        self.nf = L * (L - 1) // 2 + 11
        self.nout = L * Ne
        P0shape = (self.nout,) if self.K == 1 else (self.K, self.nout)
        self.shapes = [("P0", P0shape), ("jv", (self.nf,)), ("W1", (2 * L, H)), ("b1", (H,)),
                       ("W2", (H, self.nout)), ("b2", (self.nout,))]
        if self.K > 1:
            self.shapes.append(("ak", (self.K - 1,)))
        self.sizes = [int(np.prod(s)) for _, s in self.shapes]
        self.nparam = sum(self.sizes)
        assert self.nparam == count(arm, L, Ne, H)
        from scipy.special import roots_legendre
        u, w = roots_legendre(int(np.ceil((self.Nu + 1) / 2)))
        self.tq = torch.tensor((1 - u) / (1 + u))
        self.lq = torch.tensor(np.log(w) + 0.5 * Ne * np.log(0.5 * (1 + u)) + np.log(0.5))
        # point-group projection P_G = (1/4) sum_g chi(g) U_g over {E, Mx, My, MxMy} (pg = (chi_Mx, chi_My))
        self.pg = pg
        if pg is not None:
            perms, chars = [], []
            for c, d, ch in [(0, 0, 1.0), (1, 0, pg[0]), (0, 1, pg[1]), (1, 1, pg[0] * pg[1])]:
                p = np.array([(Lx - 1 - x if c else x) * Ly + (Ly - 1 - y if d else y)
                              for x in range(Lx) for y in range(Ly)])
                perms.append(torch.tensor(p))
                chars.append(float(ch))
            self.perms, self.chars = perms, chars

    # ------------------------------------------------------------------ parameters
    def unpack(self, th, rep=False):
        out, k = {}, 0
        for (name, shp), sz in zip(self.shapes, self.sizes):
            out[name] = th[..., k:k + sz].reshape(*((th.shape[0],) if rep else ()), *shp)
            k += sz
        return out

    def init(self, seed, eps=0.05):
        L = self.L
        e, Uh = np.linalg.eigh(-self.Tnp)
        gap = e[self.Nu] - e[self.Nu - 1]
        assert gap > 1e-8, "open-shell Fermi sea"
        Phi = Uh[:, :self.Nu]
        P0 = np.concatenate([Phi.ravel(), Phi.ravel()])
        rng = np.random.default_rng(10_000 + seed)
        rc = np.random.default_rng(20_000 + seed)
        p = {}
        p["W2"] = 1e-3 * rng.standard_normal((self.H, self.nout))
        p["W1"] = rng.standard_normal((2 * L, self.H)) / np.sqrt(2 * L)
        p["b1"] = np.zeros(self.H)
        p["b2"] = np.zeros(self.nout)
        jv = np.zeros(self.nf)
        jv[L * (L - 1) // 2] = -1.0
        p["jv"] = jv
        if self.K == 1:
            p["P0"] = P0
        else:
            rms = np.sqrt(np.mean(P0 ** 2))
            p["P0"] = P0[None] + eps * rms * rc.standard_normal((self.K, P0.size))
            p["ak"] = np.ones(self.K - 1)
        return torch.tensor(np.concatenate([np.asarray(p[n]).ravel() for n, _ in self.shapes]))

    # ------------------------------------------------------------------ amplitude
    def feats(self, nU, nD):
        n = nU + nD
        d = (n == 2).double()
        h = (n == 0).double()
        pair = n[:, self.iu] * n[:, self.ju]
        hn = h @ self.NN
        dn = d @ self.NN
        dh = [(d * (hn == m)).sum(1) for m in range(5)] + [(h * (dn == m)).sum(1) for m in range(5)]
        F = torch.cat([pair, d.sum(1, keepdim=True), torch.stack(dh, 1)], 1)
        if self.charge:
            x = 2.0 * torch.cat([d, h], 1) - 1.0
        else:
            x = 2.0 * torch.cat([nU, nD], 1) - 1.0
        return F, x

    @staticmethod
    def slse(s, l, dim):
        m = l.max(dim, keepdim=True).values.detach()
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        S = (s * torch.exp(l - m)).sum(dim)
        return torch.sign(S), m.squeeze(dim) + torch.log(torch.abs(S))

    @staticmethod
    def perm_sign(n, p):
        """fermion sign of the site permutation p on configurations n (as r2_micro_test.Species.perm)."""
        B = n.shape[0]
        occ = torch.nonzero(n > 0.5)[:, 1].reshape(B, -1)
        mp = p[occ]
        inv = torch.triu((mp[:, :, None] > mp[:, None, :]).to(torch.int64), 1).sum((1, 2))
        return 1.0 - 2.0 * (inv % 2).double()

    def logpsi(self, th, nU, nD, rep=False):
        """log|psi_G(x)| and sign; psi_G(y) = (1/4) sum_g chi(g) sigma_g psi(g^-1 y) (reflections: g^-1 = g)."""
        if self.pg is None:
            return self.logpsi0(th, nU, nD, rep)
        ls, ss = [], []
        for p, ch in zip(self.perms, self.chars):
            nU2 = torch.empty_like(nU)
            nD2 = torch.empty_like(nD)
            nU2[:, p] = nU
            nD2[:, p] = nD
            l, s = self.logpsi0(th, nU2, nD2, rep)
            ls.append(l)
            ss.append(s * ch * self.perm_sign(nU2, p) * self.perm_sign(nD2, p))
        L4 = torch.stack(ls, 1)
        s, l = self.slse(torch.stack(ss, 1), L4, 1)
        # symmetry zeros: cancellation to < 1e-11 of the largest term is an exact zero (rounding residue otherwise)
        zero = (s == 0) | (l < L4.max(1).values.detach() - 25.0)
        s = torch.where(zero, torch.zeros_like(s), s)
        l = torch.where(zero, torch.full_like(l, -float("inf")), l - math.log(4.0))
        return l, s

    def logpsi0(self, th, nU, nD, rep=False):
        """log|psi(x)| and sign for a batch of configurations nU, nD (B, L) float 0/1."""
        p = self.unpack(th, rep)
        B, L, Nu, Nd, K = nU.shape[0], self.L, self.Nu, self.Nd, self.K
        F, x = self.feats(nU, nD)
        if rep:
            J = (F * p["jv"]).sum(1)
            h = torch.tanh(torch.einsum("bi,bih->bh", x, p["W1"]) + p["b1"])
            shared = torch.einsum("bh,bho->bo", h, p["W2"]) + p["b2"]
            P0 = p["P0"].reshape(B, K, self.nout)
        else:
            J = F @ p["jv"]
            h = torch.tanh(x @ p["W1"] + p["b1"])
            shared = h @ p["W2"] + p["b2"]
            P0 = p["P0"].reshape(1, K, self.nout)
        out = shared[:, None, :] + P0
        Ou = out[..., :L * Nu].reshape(B, K, L, Nu)
        Od = out[..., L * Nu:].reshape(B, K, L, Nd)
        U = torch.nonzero(nU > 0.5)[:, 1].reshape(B, Nu)
        D = torch.nonzero(nD > 0.5)[:, 1].reshape(B, Nd)
        iU = U[:, None, :, None].expand(B, K, Nu, Nu)
        iD = D[:, None, :, None].expand(B, K, Nd, Nd)
        A = torch.gather(Ou, 2, iU)
        Dm = torch.gather(Od, 2, iD)
        if not self.charge:
            sA, lA = torch.linalg.slogdet(A[:, 0])
            sD, lD = torch.linalg.slogdet(Dm[:, 0])
            return J + lA + lD, sA * sD
        Bm = torch.gather(Od, 2, U[:, None, :, None].expand(B, K, Nu, Nd))
        C = torch.gather(Ou, 2, D[:, None, :, None].expand(B, K, Nd, Nu))
        X = C @ torch.linalg.solve(A, Bm)
        sA, lA = torch.linalg.slogdet(A)
        M = Dm[:, :, None] + self.tq[None, None, :, None, None] * X[:, :, None]
        sM, lM = torch.linalg.slogdet(M)
        sk, lk = self.slse(sM, lM + self.lq, 2)
        sk, lk = sk * sA, lk + lA
        if K > 1:
            a = torch.cat([torch.ones(1, dtype=th.dtype), p["ak"]], -1) if not rep else \
                torch.cat([torch.ones(B, 1, dtype=th.dtype), p["ak"]], 1)
            sk = sk * torch.sign(a)
            lk = lk + torch.log(torch.abs(a))
        s, l = self.slse(sk, lk, 1)
        return J + l, s

    def logpsi_batched(self, th, nU, nD, chunk):
        ls, ss = [], []
        with torch.no_grad():
            for s0 in range(0, nU.shape[0], chunk):
                l, s = self.logpsi(th, nU[s0:s0 + chunk], nD[s0:s0 + chunk])
                ls.append(l)
                ss.append(s)
        if not ls:
            return torch.zeros(0), torch.zeros(0)
        return torch.cat(ls), torch.cat(ss)

    def logderiv(self, th, nU, nD, chunk):
        out = []
        for s0 in range(0, nU.shape[0], chunk):
            b = min(chunk, nU.shape[0] - s0)
            rep = th.detach().unsqueeze(0).repeat(b, 1).requires_grad_(True)
            l, _ = self.logpsi(rep, nU[s0:s0 + b], nD[s0:s0 + b], rep=True)
            l.sum().backward()
            out.append(rep.grad)
        return torch.cat(out)

    # ------------------------------------------------------------------ local energy
    def local_energy(self, th, nU, nD, la, sa, chunk):
        B = nU.shape[0]
        E = self.U * (nU * nD).sum(1)
        for spin in (0, 1):
            occ = nU if spin == 0 else nD
            oi, oj = occ[:, self.bi], occ[:, self.bj]
            valid = oi != oj
            cs = torch.cumsum(occ, 1)
            between = cs[:, self.bj - 1] - cs[:, self.bi]
            hel = -self.bt[None, :] * (1.0 - 2.0 * torch.remainder(between, 2))
            b, k = torch.nonzero(valid, as_tuple=True)
            new = occ[b].clone()
            ii, jj = self.bi[k], self.bj[k]
            r = torch.arange(len(b))
            new[r, ii] = 1.0 - new[r, ii]
            new[r, jj] = 1.0 - new[r, jj]
            if spin == 0:
                l2, s2 = self.logpsi_batched(th, new, nD[b], chunk)
            else:
                l2, s2 = self.logpsi_batched(th, nU[b], new, chunk)
            ratio = s2 * sa[b] * torch.exp(l2 - la[b])
            E = E.index_add(0, b, hel[b, k] * ratio)
        return E

    # ------------------------------------------------------------------ Metropolis
    def random_configs(self, n, gen):
        L = self.L
        nU = torch.zeros(n, L)
        nD = torch.zeros(n, L)
        for c in range(n):
            nU[c, torch.randperm(L, generator=gen)[:self.Nu]] = 1.0
            nD[c, torch.randperm(L, generator=gen)[:self.Nd]] = 1.0
        return nU, nD

    def mcmc(self, th, nU, nD, la, sa, nprop, gen, chunk):
        B = nU.shape[0]
        nb = len(self.bi)
        acc = 0
        tried = 0
        for _ in range(nprop):
            k = torch.randint(nb, (B,), generator=gen)
            typ = torch.rand(B, generator=gen)
            i, j = self.bi[k], self.bj[k]
            r = torch.arange(B)
            ui, uj, di, dj = nU[r, i], nU[r, j], nD[r, i], nD[r, j]
            hu = (typ < 0.4) & (ui != uj)
            hd = (typ >= 0.4) & (typ < 0.8) & (di != dj)
            ex = (typ >= 0.8) & (((ui == 1) & (di == 0) & (dj == 1) & (uj == 0)) |
                                 ((di == 1) & (ui == 0) & (uj == 1) & (dj == 0)))
            mu = hu | ex
            md = hd | ex
            mv = mu | md
            idx = torch.nonzero(mv)[:, 0]
            if len(idx) == 0:
                continue
            nU2, nD2 = nU[idx].clone(), nD[idx].clone()
            ri = torch.arange(len(idx))
            ii, jj = i[idx], j[idx]
            su, sd = mu[idx], md[idx]
            tu = nU2[ri, ii].clone()
            nU2[ri[su], ii[su]] = nU2[ri[su], jj[su]]
            nU2[ri[su], jj[su]] = tu[su]
            td = nD2[ri, ii].clone()
            nD2[ri[sd], ii[sd]] = nD2[ri[sd], jj[sd]]
            nD2[ri[sd], jj[sd]] = td[sd]
            l2, s2 = self.logpsi_batched(th, nU2, nD2, chunk)
            ok = torch.rand(len(idx), generator=gen) < torch.exp(2.0 * (l2 - la[idx]))
            sel = idx[ok]
            nU[sel], nD[sel], la[sel], sa[sel] = nU2[ok], nD2[ok], l2[ok], s2[ok]
            acc += int(ok.sum())
            tried += len(idx)
        return acc / max(tried, 1)


# ============================================================================ SR driver
def lr_at(a, it):
    """constant lr for the first half of the run, then cosine decay to lr * lr_decay at the last step."""
    T = a.iters
    if it <= T / 2:
        return a.lr
    x = (it - T / 2) / (T / 2)
    return a.lr * (a.lr_decay + (1 - a.lr_decay) * 0.5 * (1 + math.cos(math.pi * x)))


class Run:
    def __init__(self, model, a, seed):
        self.m, self.a = model, a
        self.gen = torch.Generator().manual_seed(1000 + seed)
        self.nprop = int(math.ceil(a.sweeps_per_step * model.Ne))

    def start_chains(self, th):
        m, a = self.m, self.a
        self.nU, self.nD = m.random_configs(a.n_chains, self.gen)
        self.la, self.sa = m.logpsi_batched(th, self.nU, self.nD, a.chunk)
        bad = ~torch.isfinite(self.la)
        while bad.any():  # resample configurations with zero amplitude
            u, d = m.random_configs(int(bad.sum()), self.gen)
            self.nU[bad], self.nD[bad] = u, d
            self.la, self.sa = m.logpsi_batched(th, self.nU, self.nD, a.chunk)
            bad = ~torch.isfinite(self.la)
        return m.mcmc(th, self.nU, self.nD, self.la, self.sa, a.therm_sweeps * m.Ne, self.gen, a.chunk)

    def step(self, th, lr=None):
        m, a = self.m, self.a
        lr = a.lr if lr is None else lr
        t0 = time.time()
        self.la, self.sa = m.logpsi_batched(th, self.nU, self.nD, a.chunk)
        acc = m.mcmc(th, self.nU, self.nD, self.la, self.sa, self.nprop, self.gen, a.chunk)
        t1 = time.time()
        El = m.local_energy(th, self.nU, self.nD, self.la, self.sa, a.chunk)
        t2 = time.time()
        O = m.logderiv(th, self.nU, self.nD, a.grad_chunk)
        t3 = time.time()
        N = El.shape[0]
        Ob = (O - O.mean(0)) / math.sqrt(N)
        eb = (El - El.mean()) / math.sqrt(N)
        if a.sr_norm:  # scale-invariant SR: parameters normalised by sqrt(diag S) (diag-relative shift)
            dS = torch.sqrt((Ob * Ob).sum(0)) + 1e-12
            Ob = Ob / dS
        Tm = Ob @ Ob.T
        Tm.diagonal().add_(a.diag_shift)
        dth = Ob.T @ torch.linalg.solve(Tm, eb)
        if a.sr_norm:
            dth = dth / dS
        upd = lr * dth
        nrm = float(torch.linalg.norm(upd))
        if nrm > a.max_step:
            upd = upd * (a.max_step / nrm)
        t4 = time.time()
        info = dict(E=float(El.mean()), Estd=float(El.std()), acc=acc, upd=nrm,
                    t_mcmc=t1 - t0, t_eloc=t2 - t1, t_grad=t3 - t2, t_sr=t4 - t3)
        return th - upd, info

    def evaluate(self, th, n_eval):
        m, a = self.m, self.a
        self.la, self.sa = m.logpsi_batched(th, self.nU, self.nD, a.chunk)
        m.mcmc(th, self.nU, self.nD, self.la, self.sa, 10 * m.Ne, self.gen, a.chunk)
        nrep = max(1, n_eval // a.n_chains)
        Es = []
        for _ in range(nrep):
            m.mcmc(th, self.nU, self.nD, self.la, self.sa, 2 * m.Ne, self.gen, a.chunk)
            Es.append(m.local_energy(th, self.nU, self.nD, self.la, self.sa, a.chunk))
        Es = torch.stack(Es)  # (nrep, chains)
        cm = Es.mean(0)
        return float(Es.mean()), float(cm.std() / math.sqrt(len(cm))), int(Es.numel())


# ============================================================================ exact sums (<= 12 sites)
def exact_sector(m):
    import r3_micro_test as M3
    import r2_micro_test as R
    lat = M3.CylLattice(m.Lx, m.Ly, m.tp, "obc", "obc")
    assert np.allclose(lat.T, m.Tnp)
    return R.Sector(lat, m.Nu, m.Nd, m.U)


def exact_configs(sec):
    nU = torch.tensor(sec.up.bits[sec.iu], dtype=torch.float64)
    nD = torch.tensor(sec.dn.bits[sec.id], dtype=torch.float64)
    return nU, nD


def exact_energy(m, th, sec, chunk=8192):
    nU, nD = exact_configs(sec)
    l, s = m.logpsi_batched(th, nU, nD, chunk)
    psi = (s * torch.exp(l - l.max())).numpy().reshape(sec.shape)
    nrm = np.sum(psi * psi)
    return float(np.sum(psi * sec.H(psi)) / nrm), psi / np.sqrt(nrm)


# ============================================================================ modes
def tag_of(a, arm=None, seed=None):
    arm = a.arm if arm is None else arm
    seed = a.seed if seed is None else seed
    pg = "" if pg_of(a) is None else "_pg" + a.pg.replace(",", "")
    return f"S{a.Lx}x{a.Ly}_N{a.Ne}_U{a.U:g}_tp{a.tp:g}{pg}_{arm}_seed{seed}"


def pg_of(a):
    """(chi_Mx, chi_My) from --pg, or None (no point-group projection)."""
    if a.pg in (None, "", "none"):
        return None
    cx, cy = (int(v) for v in a.pg.split(","))
    assert cx in (-1, 1) and cy in (-1, 1)
    return (cx, cy)


def build(a, arm=None):
    arm = a.arm if arm is None else arm
    L = a.Lx * a.Ly
    H = width_for(arm, L, a.Ne, a.width)
    return Model(arm, a.Lx, a.Ly, a.Ne, H, a.tp, a.U, pg_of(a))


def train(a):
    torch.set_num_threads(a.threads)
    os.makedirs(a.outdir, exist_ok=True)
    tag = tag_of(a)
    fout = os.path.join(a.outdir, tag + ".json")
    if os.path.exists(fout):
        log(f"{fout} exists; skipping")
        return
    m = build(a)
    log(f"{tag}: L={m.L} Ne={m.Ne} H={m.H} P={m.nparam} (nnbf P={count('nnbf', m.L, m.Ne, a.width)}) "
        f"K={m.K} threads={a.threads} host={socket.gethostname()}")
    ck = os.path.join(a.outdir, tag + "_ckpt.npz")
    extra = sorted(int(s) for s in a.extra_checkpoints.split(",") if s.strip())
    run = Run(m, a, a.seed)
    traj, t_used, it0, extras = [], 0.0, 0, {}
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        th = torch.tensor(z["theta"])
        it0, t_used = int(z["it"]), float(z["t_used"])
        traj = json.loads(str(z["traj"]))
        extras = json.loads(str(z["extras"]))
        log(f"resumed from {ck} at step {it0} (t_used {t_used / 3600:.2f} h)")
    else:
        th = m.init(a.seed)
    acc0 = run.start_chains(th)
    log(f"thermalised: acceptance {acc0:.3f}")
    t_start = time.time()
    capped = False
    it = it0
    avg_start = int(round(a.iters * (1.0 - a.avg_frac)))
    if os.path.exists(ck) and "avg_sum" in z.files:
        avg_sum, avg_n = torch.tensor(z["avg_sum"]), int(z["avg_n"])
    else:
        avg_sum, avg_n = torch.zeros_like(th), 0
    for it in range(it0 + 1, a.iters + 1):
        th, info = run.step(th, lr_at(a, it))
        if it > avg_start:
            avg_sum += th
            avg_n += 1
        el = t_used + time.time() - t_start
        if it % 10 == 0 or it == 1:
            traj.append([it, info["E"], info["Estd"], el])
        if it % 50 == 0 or it == 1:
            log(f"it {it} E {info['E']:.6f} std {info['Estd']:.3f} acc {info['acc']:.3f} |upd| {info['upd']:.2e} "
                f"t(mcmc/eloc/grad/sr) {info['t_mcmc']:.2f}/{info['t_eloc']:.2f}/{info['t_grad']:.2f}/"
                f"{info['t_sr']:.2f}s elapsed {el / 3600:.2f} h")
        if not np.isfinite(info["E"]):
            raise RuntimeError("non-finite energy")
        if it in extra:
            extras[str(it)] = dict(theta=th.numpy().tolist(), wall=el)
        if it % a.ckpt_every == 0 or it == a.iters:
            np.savez(ck, theta=th.numpy(), it=it, t_used=el, traj=json.dumps(traj), extras=json.dumps(extras),
                     avg_sum=avg_sum.numpy(), avg_n=avg_n)
        if el > a.max_hours * 3600:
            capped = True
            log(f"wall cap {a.max_hours} h reached at step {it}")
            break
    wall = t_used + time.time() - t_start
    np.save(os.path.join(a.outdir, tag + "_theta_last.npy"), th.numpy())
    if avg_n > 0:  # reported state: uniform parameter average over the last avg_frac of the steps
        th = avg_sum / avg_n
    np.save(os.path.join(a.outdir, tag + "_theta.npy"), th.numpy())

    def ev(theta):
        if a.exact_eval:
            sec = exact_sector(m)
            return exact_energy(m, theta, sec)[0], 0.0, sec.dim
        return run.evaluate(theta, a.n_eval)
    t0 = time.time()
    E, dE, ns = ev(th)
    log(f"final E (average of last {avg_n} steps) {E:.6f} +- {dE:.6f} ({ns} samples, {time.time() - t0:.0f} s)")
    E_last = None
    if a.exact_eval:
        E_last = ev(torch.tensor(np.load(os.path.join(a.outdir, tag + "_theta_last.npy"))))[0]
        log(f"last-step parameters: exact E {E_last:.6f} (reported only)")
    ex_res = {}
    for k, v in extras.items():
        Ek, dEk, nk = ev(torch.tensor(v["theta"]))
        ex_res[k] = dict(E=Ek, dE=dEk, n=nk, wall=v["wall"])
        log(f"checkpoint {k}: E {Ek:.6f} +- {dEk:.6f} (wall at checkpoint {v['wall'] / 3600:.2f} h)")
    res = dict(tag=tag, arm=a.arm, Lx=a.Lx, Ly=a.Ly, L=m.L, Ne=m.Ne, U=a.U, tp=a.tp, seed=a.seed, H=m.H,
               nparam=m.nparam, K=m.K, iters_done=it, iters=a.iters, capped=capped, wall=wall,
               E=E, dE=dE, n_eval=ns, E_last=E_last, avg_n=avg_n, exact_eval=bool(a.exact_eval), extras=ex_res, traj=traj,
               hyper=dict(lr=a.lr, lr_decay=a.lr_decay, avg_frac=a.avg_frac, diag_shift=a.diag_shift, max_step=a.max_step, n_chains=a.n_chains,
                          sweeps_per_step=a.sweeps_per_step), host=socket.gethostname())
    with open(fout, "w") as f:
        json.dump(res, f)
    log(f"wrote {fout}")


def timing(a):
    torch.set_num_threads(a.threads)
    os.makedirs(a.outdir, exist_ok=True)
    out = dict(host=socket.gethostname(), threads=a.threads, n_chains=a.n_chains, sizes={})
    for Lx in [int(s) for s in a.timing_sizes.split(",")]:
        L = Lx * a.Ly
        Ne = default_ne(L)
        b = argparse.Namespace(**vars(a))
        b.Lx, b.Ne = Lx, Ne
        out["sizes"][str(Lx)] = {}
        for arm in ARMS:
            m = build(b, arm)
            run = Run(m, b, 0)
            th = m.init(0)
            run.start_chains(th)
            ts = []
            for _ in range(a.timing_steps):
                t0 = time.time()
                th, info = run.step(th)
                ts.append((time.time() - t0, info))
            tstep = float(np.median([t for t, _ in ts]))
            inf = ts[-1][1]
            out["sizes"][str(Lx)][arm] = dict(H=m.H, P=m.nparam, t_step=tstep, t_sample=tstep / a.n_chains,
                                              t_mcmc=inf["t_mcmc"], t_eloc=inf["t_eloc"], t_grad=inf["t_grad"],
                                              t_sr=inf["t_sr"])
            log(f"L={L:2d} Ne={Ne} {arm:7s} H={m.H:2d} P={m.nparam:6d} t_step={tstep:7.2f}s "
                f"(mcmc {inf['t_mcmc']:.2f} eloc {inf['t_eloc']:.2f} grad {inf['t_grad']:.2f} sr {inf['t_sr']:.2f}) "
                f"t_sample={1e3 * tstep / a.n_chains:.2f} ms")
        r = out["sizes"][str(Lx)]
        log(f"L={L}: t_step(che_k8)/t_step(nnbf) = {r['che_k8']['t_step'] / r['nnbf']['t_step']:.2f}")
    fn = os.path.join(a.outdir, f"timing_{socket.gethostname()}_t{a.threads}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {fn}")


def smoke(a):
    import r2_micro_test as R
    import r3_micro_test as M3
    import r5_test as F5
    torch.set_num_threads(a.threads)
    t00 = time.time()
    ok_all = True

    def chk(name, val, tol):
        nonlocal ok_all
        ok = bool(val <= tol)
        ok_all &= ok
        log(f"  {name}: {val:.2e} (tol {tol:.0e}) [{'PASS' if ok else 'FAIL'}]")

    # (a) geometry
    T, _ = strip_T(4, 3, -0.25)
    chk("(a) |T_strip - CylLattice(obc,obc)| on 4x3", float(np.abs(T - M3.CylLattice(4, 3, -0.25, "obc", "obc").T).max()), 0)

    def numpy_arm(sec, arm, H, th):
        if arm == "nnbf":
            na = R.NNBF(sec, H, 0)
        elif arm == "che":
            na = M3.ChargeNNBF(sec, H, 0)
        else:
            na = F5.CheK(sec, H, 0, kofarm(arm))
        assert na.nparam == len(th), (na.nparam, len(th))
        return na

    for (Lx, Ly, Ne, Hs) in [(4, 2, 6, 3), (4, 3, 10, 4)]:
        b = argparse.Namespace(**vars(a))
        b.Lx, b.Ly, b.Ne = Lx, Ly, Ne
        m0 = Model("che", Lx, Ly, Ne, Hs, a.tp, a.U)
        sec = exact_sector(m0)
        nU, nD = exact_configs(sec)
        log(f"{Lx}x{Ly} open Ne={Ne} dim={sec.dim}")
        for arm in ARMS:
            H = width_for(arm, Lx * Ly, Ne, Hs)
            m = Model(arm, Lx, Ly, Ne, H, a.tp, a.U)
            th = m.init(1)
            th = th + 0.05 * torch.randn(len(th), generator=torch.Generator().manual_seed(5))
            na = numpy_arm(sec, arm, H, th)
            st = M3.State(sec, na, arm != "nnbf", None, 16384)
            phi, _ = na.forward(th.numpy(), 16384)
            ref = st.psi(phi).ravel()
            ref = ref / np.linalg.norm(ref)
            E_t, psi_t = exact_energy(m, th, sec)
            psi_t = psi_t.ravel()
            if psi_t @ ref < 0:
                psi_t = -psi_t
            chk(f"(b) {arm:7s} P={m.nparam} |psi_torch - psi_numpy|_inf", float(np.abs(psi_t - ref).max()), 1e-9)
            l, s = m.logpsi_batched(th, nU, nD, 8192)
            El = m.local_energy(th, nU, nD, l, s, 8192).numpy()
            w = psi_t ** 2
            chk(f"(c) {arm:7s} |sum w E_loc - <H>|/|<H>|", abs(np.sum(w * El) - E_t) / abs(E_t), 1e-9)
            if Lx * Ly <= 8:
                E_n, g_n, _ = st.energy_grad(th.numpy())
                O = m.logderiv(th, nU, nD, 512).numpy()
                g_t = 2.0 * O.T @ (w * (El - E_t))
                chk(f"(d) {arm:7s} |g_torch - g_numpy|/|g_numpy| (E diff {abs(E_n - E_t):.1e})",
                    float(np.linalg.norm(g_t - g_n) / np.linalg.norm(g_n)), 1e-7)
            else:
                # (e) sampled vs exact energy and gradient on 4x3
                E_n, g_n, _ = st.energy_grad(th.numpy())
                b.n_chains = 4096
                run = Run(m, b, 3)
                run.start_chains(th)
                Es, Os = [], []
                for _ in range(5):
                    run.la, run.sa = m.logpsi_batched(th, run.nU, run.nD, b.chunk)
                    m.mcmc(th, run.nU, run.nD, run.la, run.sa, 2 * m.Ne, run.gen, b.chunk)
                    Es.append(m.local_energy(th, run.nU, run.nD, run.la, run.sa, b.chunk))
                    Os.append(m.logderiv(th, run.nU, run.nD, 512))
                El_s = torch.cat(Es)
                O_s = torch.cat(Os)
                Ns = El_s.shape[0]
                Em = float(El_s.mean())
                cm = torch.stack(Es).mean(0)
                dEm = float(cm.std() / math.sqrt(len(cm)))
                z = (Em - E_n) / dEm
                log(f"  (e) {arm:7s} E_sampled {Em:.6f} +- {dEm:.6f} vs exact {E_n:.6f}: z = {z:+.2f} ({Ns} samples)")
                ok_all &= abs(z) < 4
                Oc = O_s - O_s.mean(0)
                gs = 2.0 * (Oc * (El_s - Em)[:, None]).mean(0)
                gvar = (4.0 * (Oc * (El_s - Em)[:, None]) ** 2).mean(0) / Ns * 5  # x5 for chain autocorrelation bound
                zg = (gs.numpy() - g_n) / np.sqrt(gvar.numpy() + 1e-300)
                cos = float(gs.numpy() @ g_n / np.linalg.norm(gs.numpy()) / np.linalg.norm(g_n))
                frac = float(np.mean(np.abs(zg) > 3))
                log(f"  (e) {arm:7s} gradient: cos(g_sampled, g_exact) = {cos:.4f}, frac |z|>3 = {frac:.4f}, "
                    f"median |z| = {np.median(np.abs(zg)):.2f} [{'PASS' if (cos > 0.9 and frac < 0.01) else 'FAIL'}]")
                ok_all &= (cos > 0.9 and frac < 0.01)
    # (f) short SR run of che on 4x3, exact energy before / after
    b = argparse.Namespace(**vars(a))
    b.Lx, b.Ly, b.Ne, b.n_chains = 4, 3, 10, 512
    m = build(b, "che")
    sec = exact_sector(m)
    th = m.init(0)
    E_init = exact_energy(m, th, sec)[0]
    run = Run(m, b, 0)
    run.start_chains(th)
    for it in range(1, 61):
        th, info = run.step(th)
        if it % 20 == 0:
            log(f"  (f) che 4x3 SR step {it}: E_sampled {info['E']:.5f} (exact start {E_init:.5f})")
    E_fin = exact_energy(m, th, sec)[0]
    log(f"  (f) che 4x3: exact E {E_init:.6f} -> {E_fin:.6f} after 60 SR steps (E0 {E0_4x3:.6f}) "
        f"[{'PASS' if E_fin < E_init else 'FAIL'}]")
    ok_all &= E_fin < E_init
    log(f"SMOKE {'PASS' if ok_all else 'FAIL'} ({time.time() - t00:.0f} s)")


def smoke_pg(a):
    """P_G verification against exact sums (numpy r3 SG projector), every arm:
    4x2 (N_e=6), irreps (cMx,cMy) in {(1,1), (-1,1), (1,-1)}: amplitudes, E_loc identity, exact gradient vs numpy State;
    4x3 (N_e=10), irrep --pg: amplitudes, E_loc identity, Metropolis energy and gradient vs exact."""
    import r2_micro_test as R
    import r3_micro_test as M3
    import r5_test as F5
    torch.set_num_threads(a.threads)
    t00 = time.time()
    ok_all = True

    def chk(name, val, tol):
        nonlocal ok_all
        ok = bool(val <= tol)
        ok_all &= ok
        log(f"  {name}: {val:.2e} (tol {tol:.0e}) [{'PASS' if ok else 'FAIL'}]")

    def numpy_arm(sec, arm, H):
        if arm == "nnbf":
            return R.NNBF(sec, H, 0)
        if arm == "che":
            return M3.ChargeNNBF(sec, H, 0)
        return F5.CheK(sec, H, 0, kofarm(arm))
    gs = tuple(int(v) for v in a.pg.split(","))
    for (Lx, Ly, Ne, Hs, irreps) in [(4, 2, 6, 3, [(1, 1), (-1, 1), (1, -1)]), (4, 3, 10, 4, [gs])]:
        m0 = Model("che", Lx, Ly, Ne, Hs, a.tp, a.U)
        sec = exact_sector(m0)
        nU, nD = exact_configs(sec)
        for pg in irreps:
            sg = M3.SG(sec)
            assert [e[2:] for e, _ in sg.el] == [(0, 0), (0, 1), (1, 0), (1, 1)]
            sg.chi = np.array([1.0, pg[1], pg[0], pg[0] * pg[1]])
            log(f"{Lx}x{Ly} open Ne={Ne} dim={sec.dim} irrep (Mx, My) = {pg}")
            for arm in ARMS:
                H = width_for(arm, Lx * Ly, Ne, Hs)
                m = Model(arm, Lx, Ly, Ne, H, a.tp, a.U, pg)
                th = m.init(1) + 0.05 * torch.randn(m.nparam, generator=torch.Generator().manual_seed(5))
                na = numpy_arm(sec, arm, H)
                st = M3.State(sec, na, arm != "nnbf", sg, 16384)
                phi, _ = na.forward(th.numpy(), 16384)
                ref = st.psi(phi).ravel()
                ref = ref / np.linalg.norm(ref)
                E_t, psi_t = exact_energy(m, th, sec)
                psi_t = psi_t.ravel()
                if psi_t @ ref < 0:
                    psi_t = -psi_t
                chk(f"(b) {arm:7s} |psi_torch - P_G psi_numpy|_inf", float(np.abs(psi_t - ref).max()), 1e-9)
                l, s = m.logpsi_batched(th, nU, nD, 8192)
                fin = torch.isfinite(l)
                El = torch.zeros(len(l))
                El[fin] = m.local_energy(th, nU[fin], nD[fin], l[fin], s[fin], 8192)
                w = psi_t ** 2
                chk(f"(c) {arm:7s} |sum w E_loc - <H>|/|<H>|", abs(float(np.sum(w * El.numpy())) - E_t) / abs(E_t), 1e-9)
                E_n, g_n, _ = st.energy_grad(th.numpy())
                if Lx * Ly <= 8:
                    O = torch.zeros(len(l), m.nparam)
                    O[fin] = m.logderiv(th, nU[fin], nD[fin], 512)
                    g_t = 2.0 * O.numpy().T @ (w * (El.numpy() - E_t))
                    chk(f"(d) {arm:7s} |g_torch - g_numpy|/|g_numpy| (E diff {abs(E_n - E_t):.1e})",
                        float(np.linalg.norm(g_t - g_n) / np.linalg.norm(g_n)), 1e-7)
                else:
                    b = argparse.Namespace(**vars(a))
                    b.Lx, b.Ly, b.Ne, b.n_chains = Lx, Ly, Ne, 4096
                    run = Run(m, b, 3)
                    run.start_chains(th)
                    Es, Os = [], []
                    for _ in range(5):
                        run.la, run.sa = m.logpsi_batched(th, run.nU, run.nD, b.chunk)
                        m.mcmc(th, run.nU, run.nD, run.la, run.sa, 2 * m.Ne, run.gen, b.chunk)
                        Es.append(m.local_energy(th, run.nU, run.nD, run.la, run.sa, b.chunk))
                        Os.append(m.logderiv(th, run.nU, run.nD, 512))
                    El_s, O_s = torch.cat(Es), torch.cat(Os)
                    Ns = El_s.shape[0]
                    Em = float(El_s.mean())
                    dEm = float(torch.stack(Es).mean(0).std() / math.sqrt(b.n_chains))
                    z = (Em - E_n) / dEm
                    log(f"  (e) {arm:7s} E_sampled {Em:.6f} +- {dEm:.6f} vs exact {E_n:.6f}: z = {z:+.2f} ({Ns} samples)")
                    ok_all &= abs(z) < 4
                    Oc = O_s - O_s.mean(0)
                    gs_ = 2.0 * (Oc * (El_s - Em)[:, None]).mean(0)
                    gvar = (4.0 * (Oc * (El_s - Em)[:, None]) ** 2).mean(0) / Ns * 5
                    zg = (gs_.numpy() - g_n) / np.sqrt(gvar.numpy() + 1e-300)
                    cos = float(gs_.numpy() @ g_n / np.linalg.norm(gs_.numpy()) / np.linalg.norm(g_n))
                    frac = float(np.mean(np.abs(zg) > 3))
                    okg = cos > 0.9 and frac < 0.01
                    ok_all &= okg
                    log(f"  (e) {arm:7s} gradient: cos(g_sampled, g_exact) = {cos:.4f}, frac |z|>3 = {frac:.4f} "
                        f"[{'PASS' if okg else 'FAIL'}]")
    log(f"SMOKE-PG {'PASS' if ok_all else 'FAIL'} ({time.time() - t00:.0f} s)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="P4.1 size-scaling VMC (see module docstring)")
    p.add_argument("--mode", required=True, choices=["smoke", "smoke-pg", "timing", "train", "report"])
    p.add_argument("--Lx", type=int, default=4)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--Ne", type=int, default=None)
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--tp", type=float, default=-0.25)
    p.add_argument("--arm", choices=ARMS, default="che")
    p.add_argument("--pg", default="none")
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--extra-checkpoints", default="")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--sr-norm", type=int, default=0, choices=[0, 1])
    p.add_argument("--lr-decay", type=float, default=0.1)
    p.add_argument("--avg-frac", type=float, default=0.25)
    p.add_argument("--diag-shift", type=float, default=1e-3)
    p.add_argument("--max-step", type=float, default=0.5)
    p.add_argument("--n-chains", type=int, default=2048)
    p.add_argument("--sweeps-per-step", type=float, default=0.5)
    p.add_argument("--therm-sweeps", type=int, default=50)
    p.add_argument("--n-eval", type=int, default=65536)
    p.add_argument("--exact-eval", action="store_true")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--grad-chunk", type=int, default=256)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--max-hours", type=float, default=150.0)
    p.add_argument("--timing-sizes", default="4,8,12,16")
    p.add_argument("--timing-steps", type=int, default=3)
    p.add_argument("--outdir", default=os.path.join(HERE, "results"))
    a = p.parse_args(argv)
    if a.Ne is None:
        a.Ne = default_ne(a.Lx * a.Ly)
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.mode == "smoke":
        smoke(a)
    elif a.mode == "smoke-pg":
        smoke_pg(a)
    elif a.mode == "timing":
        timing(a)
    elif a.mode == "train":
        train(a)
    elif a.mode == "report":
        import scaling_report
        scaling_report.main(["--resdir", a.outdir])


if __name__ == "__main__":
    main()
