"""Symmetry machinery for the P1.3 round-3 test.

The block between the COPIED markers is copied VERBATIM from
tests/P1.1P_projected/micro_test.py lines 211-452 (group, species_perm, SpaceGroup,
Spin, _local_parts, per_sample_O_proj, proj_energy_grad, sr_step_proj). It was verified
there: projector idempotent to 5e-16, [P,H] to 7e-17, projected gradient and O_P against
finite differences. Nothing is imported across folders.

Additions after the copied block:
  SectorProjector  P = P_S P_chi (space-group irrep chi of psi_0 AND total spin S of psi_0).
                   Both factors are real symmetric, idempotent and commute with H and with
                   each other, so P is an orthogonal projector commuting with H.
  sector_lowest    lowest k eigenpairs of H inside range(P), by Lanczos on A = H P (equal
                   to P H P because [H,P]=0); vectors outside the sector map to eigenvalue 0,
                   which lies above every in-sector level of interest (all < 0 here).
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh

import p11_core as P

# ------------------------------------------------------------------ COPIED (begin)
# ============================================================================ symmetry
def group(model):
    """All space-group elements g = T^(a,b) M^(c,d) as (exponents, site perm); copied
    from P1.3 r2 (reflect first, then translate)."""
    Lx, Ly, V = model.Lx, model.Ly, model.V
    out = []
    for c in (0, 1):
        for d in (0, 1):
            for a in range(Lx):
                for b in range(Ly):
                    p = np.empty(V, dtype=np.int64)
                    for y in range(Ly):
                        for x in range(Lx):
                            x2 = (-x) % Lx if c else x
                            y2 = (-y) % Ly if d else y
                            x2, y2 = (x2 + a) % Lx, (y2 + b) % Ly
                            p[x + Lx * y] = x2 + Lx * y2
                    out.append(((a, b, c, d), p))
    return out


def species_perm(states, rows, lookup_V, p):
    """Index map and fermionic sign for one species under site permutation p (copied
    from P1.3 r2 diag_sectors.py)."""
    newidx = np.empty(len(states), dtype=np.int64)
    sgn = np.empty(len(states))
    for k, occ in enumerate(rows):
        img = p[occ]
        perm = np.argsort(img, kind="stable")
        seen = np.zeros(len(perm), bool); par = 0
        for i in range(len(perm)):
            if not seen[i]:
                j, L = i, 0
                while not seen[j]:
                    seen[j] = True; j = perm[j]; L += 1
                par += L - 1
        b = 0
        for i in img:
            b |= 1 << int(i)
        newidx[k] = lookup_V[b]
        sgn[k] = -1.0 if par % 2 else 1.0
    return newidx, sgn


class SpaceGroup:
    """Unitary representation U_g of the space group on the fixed-(N_up, N_dn) basis and
    the 1-D irrep projector P_chi = (1/|G|) sum_g chi(g) U_g.

    U_g psi (a, b) = isu_g[a] isd_g[b] psi(inv_u_g[a], inv_d_g[b]), a/b up/down indices."""

    GEN = {(1, 0, 0, 0): "Tx", (0, 1, 0, 0): "Ty", (0, 0, 1, 0): "Mx", (0, 0, 0, 1): "My"}

    def __init__(self, model):
        self.m = model
        st_u, lk_u, _, rows_u = P.species_basis(model.V, model.nup)
        st_d, lk_d, _, rows_d = P.species_basis(model.V, model.ndn)
        self.elems, inv_u, isu, inv_d, isd = [], [], [], [], []
        for e, p in group(model):
            mu, su = species_perm(st_u, rows_u, lk_u, p)
            md, sd = species_perm(st_d, rows_d, lk_d, p)
            iu = np.empty_like(mu); iu[mu] = np.arange(len(mu))
            idd = np.empty_like(md); idd[md] = np.arange(len(md))
            self.elems.append(e)
            inv_u.append(iu); isu.append(su[iu]); inv_d.append(idd); isd.append(sd[idd])
        self.inv_u, self.isu = np.array(inv_u), np.array(isu)
        self.inv_d, self.isd = np.array(inv_d), np.array(isd)
        self.G = len(self.elems)
        self.chi = np.ones(self.G)
        self.lam = dict(Tx=1, Ty=1, Mx=1, My=1)

    def apply(self, psi, k):
        m = self.m
        Psi = psi.reshape(m.Du, m.Dd)
        out = Psi[np.ix_(self.inv_u[k], self.inv_d[k])] * self.isu[k][:, None] * self.isd[k][None, :]
        return out.ravel()

    def gen_expect(self, psi):
        psi = psi / np.linalg.norm(psi)
        return {self.GEN[e]: float(psi @ self.apply(psi, k))
                for k, e in enumerate(self.elems) if e in self.GEN}

    def set_irrep(self, lam):
        self.lam = dict(lam)
        self.chi = np.array([lam["Tx"] ** a * lam["Ty"] ** b * lam["Mx"] ** c * lam["My"] ** d
                             for (a, b, c, d) in self.elems], float)

    def irrep_of(self, psi0):
        ev = self.gen_expect(psi0)
        if any(abs(abs(v) - 1) > 1e-6 for v in ev.values()):
            raise RuntimeError(f"psi_0 is not in a real 1-D irrep: {ev}")
        return {k: int(round(v)) for k, v in ev.items()}

    def project(self, psi):
        out = np.zeros_like(psi)
        for k in range(self.G):
            out += self.chi[k] * self.apply(psi, k)
        return out / self.G


# ============================================================================ spin
class Spin:
    """S^2 = S^- S^+ + S_z^2 + S_z on the fixed-(N_up, N_dn) basis via a sparse S^+, and
    the Loewdin projector onto S0 = |S_z|."""

    def __init__(self, model):
        m = self.m = model
        V, nu, nd = m.V, m.nup, m.ndn
        self.Sz = 0.5 * (nu - nd)
        Ntot = nu + nd
        self.Smax = 0.5 * min(Ntot, 2 * V - Ntot)
        self.S0 = abs(self.Sz)
        if nd == 0 or nu == V:
            self.Sp = sp.csr_matrix((1, m.D))
            return
        su, sd = m.su, m.sd
        st_u2, lk_u2, _, _ = P.species_basis(V, nu + 1)
        st_d2, lk_d2, _, _ = P.species_basis(V, nd - 1)
        Dd2 = len(st_d2)
        pc = lambda s, i: np.array([bin(int(x) & ((1 << i) - 1)).count("1") for x in s])
        r, c, v = [], [], []
        for i in range(V):
            au = np.nonzero(((su >> i) & 1) == 0)[0]
            ad = np.nonzero(((sd >> i) & 1) == 1)[0]
            if len(au) == 0 or len(ad) == 0:
                continue
            newu = lk_u2[su[au] | (1 << i)]
            newd = lk_d2[sd[ad] ^ (1 << i)]
            sg_u = (-1.0) ** pc(su[au], i)
            sg_d = (-1.0) ** pc(sd[ad], i)
            # c^dag_{i up} c_{i dn}: sign (-1)^(N_up + #dn below i + #up below i); the
            # global (-1)^N_up cancels in S^- S^+ and is dropped
            r.append((newu[:, None] * Dd2 + newd[None, :]).ravel())
            c.append((au[:, None] * m.Dd + ad[None, :]).ravel())
            v.append((sg_u[:, None] * sg_d[None, :]).ravel())
        r, c, v = np.concatenate(r), np.concatenate(c), np.concatenate(v)
        self.Sp = sp.csr_matrix((v, (r, c)), shape=(len(st_u2) * Dd2, m.D))

    def S2(self, psi):
        return self.Sp.T @ (self.Sp @ psi) + (self.Sz ** 2 + self.Sz) * psi

    def expect(self, psi):
        return float(psi @ self.S2(psi) / (psi @ psi))

    def project(self, psi, S):
        """Loewdin projector onto total spin S (S0 <= S <= Smax, S - S0 integer)."""
        out = psi.copy()
        tgt = S * (S + 1)
        s = self.S0
        while s <= self.Smax + 1e-9:
            if abs(s - S) > 1e-9:
                ss = s * (s + 1)
                out = (self.S2(out) - ss * out) / (tgt - ss)
            s += 1
        return out


# ============================================================================ projected SR
def _local_parts(ans, flat_idx):
    """Per-config pieces of dlog psi_theta for basis indices flat_idx (cf. per_sample_O_fast)."""
    iu0, id0 = ans.iu, ans.idd
    ans.iu, ans.idd = iu0[flat_idx], id0[flat_idx]
    try:
        n = len(flat_idx)
        sl = slice(0, n)
        Ou, Od, x, h = ans._orbitals(sl)
        Au, Ad, ru, rd = ans._mats(Ou, Od, sl)
        Gu, Gd = ans._inv_T(Au), ans._inv_T(Ad)
        dOu = np.zeros((n, ans.V, ans.nu)); dOd = np.zeros((n, ans.V, ans.nd))
        bi = np.arange(n)[:, None]
        dOu[bi, ru] = Gu  # rows of one config are distinct, so assignment == add.at
        dOd[bi, rd] = Gd
        jf = ans._jfeat(sl) if ans.jas else None
        return dOu, dOd, x, h, jf
    finally:
        ans.iu, ans.idd = iu0, id0


def per_sample_O_proj(ans, sg, idx, psi_t, psi_P, o_chunk):
    """O_P(n)_k = d log psi_P(n) / d theta_k for sampled basis indices idx, with
    psi_P = P_chi psi_theta: O_P(n) = sum_g c_g(n) O_theta(n_g)."""
    m = ans.m
    Dd, G = m.Dd, sg.G
    out = np.empty((len(idx), ans.n_params()))
    for s in range(0, len(idx), o_chunk):
        ii = idx[s:s + o_chunk]
        B = len(ii)
        a, b = ii // Dd, ii % Dd
        img = (sg.inv_u[:, a] * Dd + sg.inv_d[:, b]).T                       # (B, G)
        fac = (sg.chi[:, None] * sg.isu[:, a] * sg.isd[:, b]).T / G          # (B, G)
        coef = fac * psi_t[img] / psi_P[ii][:, None]                          # (B, G)
        dOu, dOd, x, h, jf = _local_parts(ans, img.ravel())
        c = coef.ravel()
        parts = {"Pu": (dOu.reshape(B * G, -1) * c[:, None]).reshape(B, G, -1).sum(1),
                 "Pd": (dOd.reshape(B * G, -1) * c[:, None]).reshape(B, G, -1).sum(1)}
        if ans.jas:
            parts["vj"] = (jf * c[:, None]).reshape(B, G, -1).sum(1)
        if ans.kind == "nnbf":
            dout = np.concatenate([dOu.reshape(B * G, -1), dOd.reshape(B * G, -1)], 1)
            nout, H, nin = dout.shape[1], h.shape[1], x.shape[1]
            parts["b2"] = (dout * c[:, None]).reshape(B, G, nout).sum(1)
            hc = (h * c[:, None]).reshape(B, G, H)
            parts["W2"] = np.matmul(hc.transpose(0, 2, 1), dout.reshape(B, G, nout)).reshape(B, -1)
            dpre = (dout @ ans.p["W2"].T) * (1 - h ** 2)
            parts["b1"] = (dpre * c[:, None]).reshape(B, G, H).sum(1)
            xc = (x * c[:, None]).reshape(B, G, nin)
            parts["W1"] = np.matmul(xc.transpose(0, 2, 1), dpre.reshape(B, G, H)).reshape(B, -1)
        out[s:s + B] = np.concatenate([parts[k] for k in ans.keys], 1)
    return out


def proj_energy_grad(ans, H, sg, chunk):
    """E[psi_P] and the P1.1-normalised force f = sum_n r(n) d psi_theta(n),
    r = (H - E) psi_P / ||psi_P||^2 (the true gradient is 2 f)."""
    psi_t = ans.amplitudes(chunk)
    psi = sg.project(psi_t)
    E, Hp, nn = P.energy(H, psi)
    r = (Hp - E * psi) / nn
    f = ans.grad_weighted(psi_t * r, chunk)
    return E, f, psi_t, psi, nn


def sr_step_proj(ans, H, sg, args, rng, tm=None):
    """One projected SR step (P1.1 update rule with psi -> psi_P). Returns E before the step."""
    t0 = time.time()
    E, f, psi_t, psi, nn = proj_energy_grad(ans, H, sg, args.chunk)
    t1 = time.time()
    p = psi ** 2 / nn
    idx = rng.choice(len(psi), size=args.sr_samples, p=p)
    O = per_sample_O_proj(ans, sg, idx, psi_t, psi, args.o_chunk)
    t2 = time.time()
    Oc = (O - O.mean(0)) / np.sqrt(len(idx))
    lam = args.sr_shift
    Kmat = Oc @ Oc.T + lam * np.eye(len(idx))
    dx = (f - Oc.T @ np.linalg.solve(Kmat, Oc @ f)) / lam
    nrm = np.linalg.norm(dx)
    step = args.sr_lr * dx
    if nrm * args.sr_lr > 1.0:
        step *= 1.0 / (nrm * args.sr_lr)
    ans.set_flat(ans.flat() - step)
    if tm is not None:
        tm.update(force=t1 - t0, metric_O=t2 - t1, solve=time.time() - t2)
    return E, f
# ------------------------------------------------------------------ COPIED (end)


class SectorProjector:
    """P = P_S P_chi onto psi_0's space-group irrep chi and total spin S."""

    def __init__(self, model, sg, spn, lam, S):
        self.m, self.sg, self.spn, self.S = model, sg, spn, float(S)
        sg.set_irrep(lam)
        self.lam = dict(lam)

    def chi_only(self, psi):
        return self.sg.project(psi)

    def spin_only(self, psi):
        return self.spn.project(psi, self.S)

    def __call__(self, psi):
        return self.spn.project(self.sg.project(psi), self.S)


def sector_lowest(model, proj, k, seed, tol=1e-10):
    """Lowest k eigenpairs of H restricted to range(proj)."""
    D = model.D
    H = model.H

    def mv(v):
        v = np.asarray(v).ravel()
        return H @ proj(v)
    A = LinearOperator((D, D), matvec=mv, dtype=float)
    v0 = proj(np.random.default_rng(seed).standard_normal(D))
    w, U = eigsh(A, k=k, which="SA", v0=v0, tol=tol, maxiter=20000)
    o = np.argsort(w)
    w, U = w[o], U[:, o]
    for j in range(k):  # clean residual out-of-sector noise
        u = proj(U[:, j])
        U[:, j] = u / np.linalg.norm(u)
    return w, U
