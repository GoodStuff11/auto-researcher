#!/usr/bin/env python3
"""
P4.1 Round 10 T1: exact tangent-space (linear-method) test of two nested extensions of trained exact-S states.
==============================================================================================================

Run with the lit_env interpreter (torch, float64, CPU), from this directory:
    ~/.conda/envs/lit_env/bin/python tangent.py --mode MODE [--parent P] [--threads N]

Question
--------
At a TRAINED 12-site exact-S state Psi (open 4x3 strip, N_e = 10, U = 8, t' = -0.25, irrep (+,+) via P_G, S = 0),
do the new degrees of freedom of a candidate route lower the exact energy at first order more than a parameter-
matched generic extension does?  For a set D of tangent directions T_k = dPsi/dphi_k (phi = 0 reproduces Psi exactly)
the linear method gives the lowest exact energy in span{Psi, T_k : k in D}:
    E_LM(D) = min generalised eigenvalue of (<v_a|H|v_b>, <v_a|v_b>), v = (Psi, T_1, ..., T_K),
with every sum exact over all 627,264 configurations (no sampling, no seeds).  Delta(D) = E(Psi) - E_LM(D) >= 0.

Parents (--parent; parameters are files written by earlier, already scored runs, read only)
  che  : logs/pilot/pg_che (vmc.Model "che", H = 16, 1000 SR steps; eps 2.858e-3)
  pcg  : P4.1_cheap_su2_t2/results/t2a pcg H = 15 (eps 1.28e-3)
  r1   : P4.1_cheap_su2_t2/results/t2a r1  H = 25 (eps 2.31e-3)

Direction families (all charge-only or orbital-level, so every T_k is an exact S = 0 vector; checked)
  C1   the parent's own pair/doublon-holon Jastrow features F(c) (vmc.Model.feats; 77 directions): slack control.
  C3   site-indexed constant orbital shifts (the parent's own P0 / f0 directions; 120 for che, 144 for pcg, r1).
  SQ   R16 squeezed-frame directions: the orbital row of an occupied site i gets Q[s(i,c)], where s(i,c) is the
       rank of i among the occupied sites along the snake path of the strip (columns traversed alternately up and
       down).  che: Q is L x N_e (120); pcg: f_c[i,j] += Q[s(i),s(j)] (144); r1: the same with Q symmetrised.
  W    matched foil for SQ: one new tanh hidden unit h(c) of the charge one-hot input (fixed random first-layer
       weights, seed 7) times a free orbital vector (che 120, pcg/r1 144 directions).
  NJ   R17 extensive local charge Jastrow: translation-summed indicator of every 2x2-plaquette charge pattern
       (n in {0,1,2}^4, 81 directions; size-independent parameter count).
  RF   matched foil for NJ: 81 random global tanh charge features (seed 11), the "global neural Jastrow".

Pre-registered rule (written before any 'run' output existed; mode 'rule' prints it)
  Gains, in units of the parent's exact gap g = E(Psi) - E0:
    G_SQ = [Delta(C1+C3+SQ) - Delta(C1+C3)] / g        G_W  = [Delta(C1+C3+W) - Delta(C1+C3)] / g
    G_NJ = [Delta(C1+NJ)    - Delta(C1)]    / g        G_RF = [Delta(C1+RF)   - Delta(C1)]    / g
  Route R16 (SQ against W), route R17 (NJ against RF); per route, over the parents che and pcg (r1 reported only):
    KILL      if G_route <= G_foil for both che and pcg;
    PASS      if G_route >= 2 G_foil and G_route >= 0.05 for both che and pcg;
    AMBIGUOUS otherwise (no promotion; T2 only after a written reason).
  INVALID if a validation check fails (P_G map against vmc.logpsi overlap < 1 - 1e-10; |<S^2>| of any T_k
  relative to <T_k|T_k> > 1e-10; E(Psi) differs from the stored exact energy by > 1e-8).
  Controls (must behave or the run is INVALID): positive, adding the exact ground state as one direction gives
  Delta = g to 1e-8; negative, 81 random charge-only multipliers r(c) Psi (r i.i.d. Gaussian per charge configuration) give G < 0.05.
  A first-order (tangent) gain is a necessary condition, not a trained gain: T1 can KILL, and a PASS only licenses
  a T2 fine-tune at 24 sites.

Modes
  rule      print the rule above.
  run       compute the families for --parent and write results/tangent_<parent>.json.
  report    apply the rule to results/tangent_{che,pcg,r1}.json and print the verdicts.

Options
  --mode {rule, run, report} (required)   --parent {che, pcg, r1} (che)   --threads INT (4)   --chunk INT (4096)
"""
import argparse
import json
import os
import sys
import time

THR = "4"
for i, s in enumerate(sys.argv):
    if s == "--threads" and i + 1 < len(sys.argv):
        THR = sys.argv[i + 1]
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = THR

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_t2"))
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_scaling"))
import vmc as V  # noqa: E402  (frozen; read only)
import vmc_t2 as T2  # noqa: E402  (frozen; read only)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(THR))
log = V.log
E0 = V.E0_4x3
RULE = __doc__.split("Pre-registered rule")[1].split("Modes")[0]
PAR = {
    "che": os.path.join(TESTS, "P4.1_cheap_su2_scaling/logs/pilot/pg_che/S4x3_N10_U8_tp-0.25_pg11_che_seed0"),
    "pcg": os.path.join(TESTS, "P4.1_cheap_su2_t2/results/t2a/S4x3_N10_U8_tp-0.25_pg11_pcg_H15_seed0"),
    "r1": os.path.join(TESTS, "P4.1_cheap_su2_t2/results/t2a/S4x3_N10_U8_tp-0.25_pg11_r1_H25_seed0"),
}


# ---------------------------------------------------------------- models with an orbital-level probe "delta"
class CheProbe(V.Model):
    def logpsi0(self, th, nU, nD, rep=False, delta=None):
        p = self.unpack(th)
        B, L, Nu, Nd = nU.shape[0], self.L, self.Nu, self.Nd
        F, x = self.feats(nU, nD)
        J = F @ p["jv"]
        h = torch.tanh(x @ p["W1"] + p["b1"])
        out = h @ p["W2"] + p["b2"] + p["P0"][None]
        if delta is not None:
            out = out + delta
        Ou = out[:, :L * Nu].reshape(B, L, Nu)
        Od = out[:, L * Nu:].reshape(B, L, Nd)
        U = torch.nonzero(nU > 0.5)[:, 1].reshape(B, Nu)
        D = torch.nonzero(nD > 0.5)[:, 1].reshape(B, Nd)
        A = torch.gather(Ou, 1, U[:, :, None].expand(B, Nu, Nu))
        Dm = torch.gather(Od, 1, D[:, :, None].expand(B, Nd, Nd))
        Bm = torch.gather(Od, 1, U[:, :, None].expand(B, Nu, Nd))
        C = torch.gather(Ou, 1, D[:, :, None].expand(B, Nd, Nu))
        X = C @ torch.linalg.solve(A, Bm)
        sA, lA = torch.linalg.slogdet(A)
        M = Dm[:, None] + self.tq[None, :, None, None] * X[:, None]
        sM, lM = torch.linalg.slogdet(M)
        sk, lk = self.slse(sM, lM + self.lq, 1)
        return J + lk + lA, sk * sA


class DetProbe(T2.DetModel):
    delta = None

    def fc(self, p, nU, nD, rep):
        f = super().fc(p, nU, nD, rep)
        if self.delta is not None:
            d = self.delta if self.arm != "r1" else 0.5 * (self.delta + self.delta.transpose(1, 2))
            f = f + d
        return f


def build(parent):
    if parent == "che":
        return CheProbe("che", 4, 3, 10, 16, -0.25, 8.0, (1, 1))
    H = {"pcg": 15, "r1": 25}[parent]
    return DetProbe(parent, 4, 3, 10, H, -0.25, 8.0, (1, 1))


def snake_rank(n, Lx, Ly):
    """s(i,c): rank of site i among occupied sites (n_i > 0) along the snake path; -1 for empty sites."""
    path = []
    for x in range(Lx):
        ys = range(Ly) if x % 2 == 0 else range(Ly - 1, -1, -1)
        path += [x * Ly + y for y in ys]
    path = torch.tensor(path)
    occ = (n[:, path] > 0).double()
    rk = torch.cumsum(occ, 1) - occ
    s = torch.full_like(n, -1.0)
    s[:, path] = torch.where(occ > 0, rk, torch.full_like(rk, -1.0))
    return s.long()


# ---------------------------------------------------------------- unprojected amplitudes and orbital gradients
def phi0_and_grad(m, th, nU, nD, parent, chunk, keep64=False):
    """log phi0, sign, and Gamma = d log phi0 / d(orbital object) per configuration (unprojected, P_S included)."""
    ls, ss, gs = [], [], []
    for s0 in range(0, nU.shape[0], chunk):
        u, d = nU[s0:s0 + chunk], nD[s0:s0 + chunk]
        B = u.shape[0]
        shp = (B, m.nout) if parent == "che" else (B, m.L, m.L)
        dl = torch.zeros(shp, requires_grad=True)
        if parent == "che":
            l, s = m.logpsi0(th, u, d, delta=dl)
        else:
            m.delta = dl
            l, s = m.logpsi0(th, u, d)
            m.delta = None
        (g,) = torch.autograd.grad(l.sum(), dl)
        ls.append(l.detach())
        ss.append(s.detach())
        gs.append(g.reshape(B, -1) if keep64 else g.reshape(B, -1).float())  # float32 storage; tangents are rebuilt in float64 below
    return torch.cat(ls), torch.cat(ss), torch.cat(gs)


# ---------------------------------------------------------------- P_G by index maps
def pg_maps(m, sec):
    maps = []
    for p, ch in zip(m.perms, m.chars):
        pn = p.numpy()
        mu, su = sec.up.perm(pn)
        md, sd = sec.dn.perm(pn)
        idx = mu[sec.iu] * sec.dn.dim + md[sec.id]
        maps.append((idx, ch * su[sec.iu] * sd[sec.id]))
    return maps


def project(maps, t):
    out = np.zeros_like(t)
    for idx, sg in maps:
        out += sg * t[idx]
    return 0.25 * out


# ---------------------------------------------------------------- features
def plaquette_feats(n, Lx, Ly):
    cols = []
    for x in range(Lx - 1):
        for y in range(Ly - 1):
            sites = [x * Ly + y, x * Ly + y + 1, (x + 1) * Ly + y, (x + 1) * Ly + y + 1]
            code = sum(n[:, s].long() * 3 ** k for k, s in enumerate(sites))
            cols.append(code)
    code = torch.stack(cols, 1)
    return torch.stack([(code == P).double().sum(1) for P in range(81)], 1)


def charge_onehot(n):
    return 2.0 * torch.cat([(n == 2).double(), (n == 0).double()], 1) - 1.0


# ---------------------------------------------------------------- linear method
def lm_delta(sec, Psi, vecs, EPsi):
    """E(Psi) - lowest energy in span{Psi, vecs}; exact. vecs: list of full-space arrays."""
    V_ = [Psi] + vecs
    K = len(V_)
    X = np.stack([v.ravel() for v in V_], 1)
    HX = np.stack([sec.H(v.reshape(sec.shape)).ravel() for v in V_], 1)
    S = X.T @ X
    Hm = X.T @ HX
    Hm = 0.5 * (Hm + Hm.T)
    w, Uv = np.linalg.eigh(S)
    keep = w > 1e-11 * w.max()
    Pm = Uv[:, keep] / np.sqrt(w[keep])
    e = np.linalg.eigvalsh(Pm.T @ Hm @ Pm)
    return EPsi - float(e[0]), int(keep.sum()), K


def lm_sub(S, Hm, EPsi):
    w, Uv = np.linalg.eigh(S)
    keep = w > 1e-11 * w.max()
    Pm = Uv[:, keep] / np.sqrt(w[keep])
    e = np.linalg.eigvalsh(Pm.T @ Hm @ Pm)
    return EPsi - float(e[0]), int(keep.sum())


def s2_rel(sec, v):
    vv = v.reshape(sec.shape)
    return float(abs(np.sum(vv * sec.S2(vv))) / max(np.sum(vv * vv), 1e-300))


def run(a):
    t0 = time.time()
    parent = a.parent
    m = build(parent)
    th = torch.tensor(np.load(PAR[parent] + "_theta.npy"))
    stored = json.load(open(PAR[parent] + ".json"))["E"]
    sec = V.exact_sector(m)
    nU, nD = V.exact_configs(sec)
    n = nU + nD
    log(f"{parent}: P={m.nparam}, dim={sec.dim}, threads={THR}")
    l, s, G = phi0_and_grad(m, th, nU, nD, parent, a.chunk)
    log(f"phi0 and orbital gradients done ({time.time() - t0:.0f} s)")
    lmax = float(l.max())
    phi0 = (s * torch.exp(l - lmax)).numpy()
    maps = pg_maps(m, sec)
    Psi = project(maps, phi0)
    nrm = np.sqrt(np.sum(Psi * Psi))
    Psi, phi0 = Psi / nrm, phi0 / nrm
    EPsi = float(np.sum(Psi * sec.H(Psi.reshape(sec.shape)).ravel()))
    # validation 1: P_G map against vmc's own projected amplitudes
    lv, sv = m.logpsi_batched(th, nU, nD, a.chunk)
    ref = (sv * torch.exp(lv - lv.max())).numpy()
    ov = abs(np.sum(ref * Psi)) / np.sqrt(np.sum(ref * ref))
    checks = dict(pg_overlap=float(ov), E_Psi=EPsi, E_stored=stored, dE_stored=abs(EPsi - stored),
                  S2_Psi=s2_rel(sec, Psi))
    log(f"validation: P_G overlap {ov:.15f}, E(Psi) {EPsi:.10f} (stored {stored:.10f}), <S^2> {checks['S2_Psi']:.1e}")
    gap = EPsi - E0
    L, Ne = m.L, m.Ne
    # families (unprojected tangents t(x), then P_G)
    fam = {}
    F, _ = m.feats(nU, nD)
    fam["C1"] = [F[:, k].numpy() * phi0 for k in range(F.shape[1])]
    Gd = G.double()
    ncol = Ne if parent == "che" else L
    if parent == "che":
        fam["C3"] = [Gd[:, o].numpy() * phi0 for o in range(m.nout)]
    else:
        fam["C3"] = [Gd[:, o].numpy() * phi0 for o in range(L * L)]
    sr = snake_rank(n, m.Lx, m.Ly)  # (dim, L)
    sq = []
    if parent == "che":
        Gm = Gd.reshape(-1, L, Ne)  # [config, site, column]; up columns 0..Nu-1 then down (out layout)
        Gm = torch.cat([Gd[:, :L * m.Nu].reshape(-1, L, m.Nu), Gd[:, L * m.Nu:].reshape(-1, L, m.Nd)], 2)
        for r in range(L):
            hit = (sr == r).double()  # (dim, L): site with squeezed index r
            row = torch.einsum("bi,bic->bc", hit, Gm)
            sq += [row[:, c].numpy() * phi0 for c in range(Ne)]
    else:
        Gm = Gd.reshape(-1, L, L)
        hits = [(sr == r).double() for r in range(L)]
        for r in range(L):
            tmp = torch.einsum("bi,bij->bj", hits[r], Gm)
            for r2 in range(L):
                sq.append((tmp * hits[r2]).sum(1).numpy() * phi0)
    fam["SQ"] = sq
    gen = torch.Generator().manual_seed(7)
    x = charge_onehot(n)
    hW = torch.tanh(x @ (torch.randn(2 * L, 1, generator=gen) / np.sqrt(2 * L)) + 0.1 * torch.randn(1, generator=gen))
    fam["W"] = [(hW[:, 0] * Gd[:, o]).numpy() * phi0 for o in range(Gd.shape[1])]
    PF = plaquette_feats(n, m.Lx, m.Ly)
    fam["NJ"] = [PF[:, k].numpy() * phi0 for k in range(81)]
    gen = torch.Generator().manual_seed(11)
    RF = torch.tanh(x @ (torch.randn(2 * L, 81, generator=gen) / np.sqrt(2 * L)) + 0.5 * torch.randn(81, generator=gen))
    fam["RF"] = [RF[:, k].numpy() * phi0 for k in range(81)]
    del G, Gd, Gm
    log(f"families built ({time.time() - t0:.0f} s): " + ", ".join(f"{k} {len(v)}" for k, v in fam.items()))
    proj = {}
    for k in list(fam):
        proj[k] = [project(maps, t) for t in fam.pop(k)]
    # validation 2: exact S of tangent vectors (sampled check: 5 per family)
    s2max = 0.0
    rng = np.random.default_rng(0)
    for k, v in proj.items():
        for j in rng.choice(len(v), size=min(5, len(v)), replace=False):
            if np.sum(v[j] ** 2) > 1e-20:
                s2max = max(s2max, s2_rel(sec, v[j]))
    checks["S2_tangent_max"] = s2max
    log(f"max relative <S^2> of sampled tangents {s2max:.1e}")
    names = ["C1", "C3", "SQ", "W", "NJ", "RF"]
    cols = {"Psi": [0]}
    allv = [Psi]
    for k in names:
        cols[k] = list(range(len(allv), len(allv) + len(proj[k])))
        allv += proj.pop(k)
    X = np.stack(allv, 1)
    del allv
    Sg = X.T @ X
    Hg = np.zeros_like(Sg)
    for j in range(X.shape[1]):
        Hg[:, j] = X.T @ sec.H(X[:, j].reshape(sec.shape)).ravel()
    Hg = 0.5 * (Hg + Hg.T)
    log(f"Gram and H matrices over {X.shape[1]} vectors ({time.time() - t0:.0f} s)")
    sets = {"C1": ["C1"], "C1+C3": ["C1", "C3"], "C1+C3+SQ": ["C1", "C3", "SQ"], "C1+C3+W": ["C1", "C3", "W"],
            "C1+NJ": ["C1", "NJ"], "C1+RF": ["C1", "RF"]}
    res = {}
    for name, fs in sets.items():
        idx = [0] + [c for f in fs for c in cols[f]]
        d, rank = lm_sub(Sg[np.ix_(idx, idx)], Hg[np.ix_(idx, idx)], EPsi)
        res[name] = dict(delta=d, rank=rank, K=len(idx))
        log(f"Delta({name}) = {d:.6e} (= {d / gap:.4f} of gap; rank {rank} of {len(idx)})")
    del X
    # controls
    psi_ex = sec_ground(sec, m)
    dpos, _, _ = lm_delta(sec, Psi, [psi_ex], EPsi)
    noise = []
    for j in range(81):
        rc = rng.standard_normal(sec.nC)[sec.cidx]  # random charge-only function: exact S = 0
        z = project(maps, rc * phi0)
        noise.append(z)
    dneg, _, _ = lm_delta(sec, Psi, noise, EPsi)
    checks.update(pos_delta_over_gap=dpos / gap, pos_abs_err=abs(dpos - gap), neg_G=dneg / gap)
    log(f"controls: positive Delta/g = {dpos / gap:.10f}; negative (81 random S=0 vectors) G = {dneg / gap:.4f}")
    out = dict(parent=parent, E_Psi=EPsi, E0=E0, gap=gap, eps=gap / abs(E0), sets=res, checks=checks,
               wall=time.time() - t0, threads=int(THR))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", f"tangent_{parent}.json"), "w"), indent=1)
    log(f"wrote results/tangent_{parent}.json ({time.time() - t0:.0f} s)")


def sec_ground(sec, m):
    """exact (+,+) S=0 ground state of the sector (cached)."""
    f = os.path.join(HERE, "results", "gs_4x3.npy")
    if os.path.exists(f):
        return np.load(f)
    from scipy.sparse.linalg import eigsh
    w, v = eigsh(sec.Hsparse(), k=1, which="SA", tol=1e-12)
    assert abs(w[0] - E0) < 1e-8, w[0]
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.save(f, v[:, 0])
    return v[:, 0]


def verdicts(R):
    out = {}
    for route, rs, fs, base in [("R16 squeezed frame", "C1+C3+SQ", "C1+C3+W", "C1+C3"),
                                ("R17 local charge Jastrow", "C1+NJ", "C1+RF", "C1")]:
        g = {}
        for p in ("che", "pcg", "r1"):
            if p not in R:
                continue
            r = R[p]
            b = r["sets"][base]["delta"]
            g[p] = ((r["sets"][rs]["delta"] - b) / r["gap"], (r["sets"][fs]["delta"] - b) / r["gap"])
        if not all(p in g for p in ("che", "pcg")):
            out[route] = ("INCOMPLETE", g)
            continue
        bad = [p for p in R if not valid(R[p]["checks"])]
        if bad:
            out[route] = (f"INVALID ({','.join(bad)})", g)
        elif all(g[p][0] <= g[p][1] for p in ("che", "pcg")):
            out[route] = ("KILL", g)
        elif all(g[p][0] >= 2 * g[p][1] and g[p][0] >= 0.05 for p in ("che", "pcg")):
            out[route] = ("PASS", g)
        else:
            out[route] = ("AMBIGUOUS", g)
    return out


def valid(c):
    return (c["pg_overlap"] > 1 - 1e-10 and c["dE_stored"] < 1e-8 and c["S2_tangent_max"] < 1e-10
            and c["pos_abs_err"] < 1e-8 and c["neg_G"] < 0.05)


def report(a):
    R = {}
    for p in ("che", "pcg", "r1"):
        f = os.path.join(HERE, "results", f"tangent_{p}.json")
        if os.path.exists(f):
            R[p] = json.load(open(f))
    for p, r in R.items():
        print(f"{p}: eps {r['eps']:.4e}, gap {r['gap']:.6e}, checks {r['checks']}")
        for k, v in r["sets"].items():
            print(f"   Delta({k}) / gap = {v['delta'] / r['gap']:.4f}  (rank {v['rank']} of {v['K']})")
    for route, (v, g) in verdicts(R).items():
        print(f"VERDICT {route}: {v}   " + "; ".join(f"{p}: G_route {x:.4f}, G_foil {y:.4f}" for p, (x, y) in g.items()))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", required=True, choices=["rule", "run", "report"])
    pa.add_argument("--parent", default="che", choices=["che", "pcg", "r1"])
    pa.add_argument("--threads", default="4")
    pa.add_argument("--chunk", type=int, default=4096)
    a = pa.parse_args()
    if a.mode == "rule":
        print(RULE)
    elif a.mode == "run":
        run(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
