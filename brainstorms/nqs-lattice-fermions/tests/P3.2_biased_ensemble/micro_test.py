#!/usr/bin/env python3
"""
P3.2 decisive micro-test: does a biased-branch zero-variance extrapolation rule
pick the correct ground-state "order" of a doped t-t'-U Hubbard cluster, with a
correct error bar, and does the jointly optimized two-state (block / ensemble)
version fix the failure modes of independent branches?

Physics set-up
--------------
Hubbard model on an Lx x Ly square cluster,
    H = -t sum_<ij>,s c+_is c_js  - t' sum_<<ij>>,s c+_is c_js + U sum_i n_iu n_id ,
with N_up = N_dn electrons (doped, so the sign problem / non-stoquasticity is
present; t' != 0 breaks particle-hole symmetry). A lattice mirror R splits the
Hilbert space into R-even and R-odd sectors. The two "competing orders" are the
lowest R-even state (label A) and the lowest R-odd state (label B); their exact
energies E_A*, E_B* come from Lanczos ED inside each sector. The truth at each
control parameter g (= t') is sign(delta) with delta = E_B* - E_A*.
The decision rules never see the sector projectors: they only see variational
states, their energies, energy variances, and one order observable <R> (stand-in
for the long-range order correlation C(r) a real VMC run would measure).

Variational branches (cheap emulation of a biased NQS run)
----------------------------------------------------------
Biased reference for order X in {A,B} (Y is the other one):
    phi_X = normalize( mask_f0( psi_X* ) * signflips(q)  +  eta * psi_Y* ),
mask_f0 keeps the f0 fraction of configurations with the largest |psi_X*|;
signflips(q) flips the sign of a random fraction q of the kept configurations
taken from the smaller-amplitude half (emulates residual sign errors); eta is
drawn uniformly in [0, eta_max] per seed (emulates contamination by the
competing order). Accuracy levels p = 0..P:
  * independent branch X at level p: lowest Ritz vector of the Krylov space
    K_p(H, phi_X) = span{phi_X, H phi_X, ..., H^p phi_X} (VMC + p Lanczos steps);
  * joint (two-state ensemble) at level p: lowest two Ritz vectors of the block
    Krylov space K_p(H, {phi_A, phi_B}) (same number of H applications per state).

Decision rules compared
-----------------------
  R0   status quo: label branches by initialization; for each branch fit E vs
       sigma^2 over the last `--fit-points` levels (linear), intercept I_X with
       error err_X = sqrt(SE_lin^2 + (I_lin - I_quad)^2) (I_quad: quadratic fit
       intercept, only when >= 3 points); decide A if I_B - I_A > z*err,
       B if I_A - I_B > z*err, else UNDECIDED.
  R0+  independent branches relabelled by measured character sign(<R>) at the
       last level; UNDECIDED if either has |<R>| < --char-min; if both carry
       the same character c (crossover), decide c; otherwise
       same intercept comparison as R0 with the measured labels.
  R4   joint two-state ensemble: Ritz roots chi_0, chi_1 at every level, each
       root extrapolated (same fit and error model) along its own
       (sigma^2, theta) path; roots labelled by measured character; same
       decision logic as R0+.

Scalar metric and verdict
-------------------------
For each rule: CW = fraction of instances with a confident wrong decision;
DEC = fraction of instances with any confident decision; COV = fraction of
(branch, instance) pairs whose intercept interval I +- z*err covers the exact
energy of the eigenstate whose character the branch carries.
Primary metric M = CW(R4). Secondary gain condition
    G = [DEC(R4) - DEC(R0+) >= 0.10] or [CW(R0+) - CW(R4) >= 0.05].
PASS      : M <= 0.02 and DEC(R4) >= 0.50 and G
FAIL      : M > 0.10, or (M <= 0.10 and not G)  [joint step adds nothing]
AMBIGUOUS : anything else -> pre-committed action: rerun with the other cluster
            geometry (--lx/--ly swapped or 2x6) and --n-samples 10000; if still
            AMBIGUOUS, drop the joint step (keep R0+) and mark R4 KILLED.

Command-line options (all optional)
-----------------------------------
  --lx INT            cluster length along x (mirror direction for --mirror x).
                      Default 6. Any integer >= 2.
  --ly INT            cluster length along y. Default 2. Any integer >= 1.
                      (Default 6x2 'po' = two-leg ladder, periodic legs, 12
                      sites; with 5+5 electrons and t'=g the lowest R-even and
                      R-odd states cross between g=-0.4 (odd lower, delta=-0.055)
                      and g=0 (even lower, delta=+0.0025): a genuine
                      near-degenerate competition of order 2e-4 t per site.)
  --bc {pp,po,op,oo}  boundary conditions (x then y): p = periodic, o = open.
                      Default po. For a periodic direction of length 2 the
                      wrap-around bond coincides with the direct bond and its
                      amplitude is added (doubled hopping).
  --nup INT           number of up electrons. Default 5.
  --ndn INT           number of down electrons. Default 5. Must equal --nup if
                      --mirror is used with spin-symmetric sectors (it is not
                      required by the code, any value 0..Lx*Ly works).
  --U FLOAT           on-site repulsion U/t. Default 8.0.
  --t FLOAT           nearest-neighbour hopping. Default 1.0.
  --g-min FLOAT, --g-max FLOAT, --n-g INT
                      scan of the control parameter g = t' on a uniform grid of
                      n-g points in [g-min, g-max]. Defaults -0.4, 0.4, 17.
  --mirror {x,y}      mirror used to define the two competitor sectors:
                      x: site (x,y) -> (Lx-1-x, y);  y: (x,y) -> (x, Ly-1-y).
                      Default x.
  --seeds INT         number of random biased-reference draws per g. Default 8.
  --P INT             highest Lanczos level p (levels 0..P). Default 4.
  --fit-points INT    number of most-accurate levels used in each extrapolation
                      (2..P+1). Default 3.
  --f0 FLOAT          fraction of configurations kept in the biased reference,
                      0 < f0 <= 1. Default 0.1.
  --q FLOAT           fraction of kept configurations (from the smaller-amplitude
                      half) whose sign is flipped, 0 <= q <= 1. Default 0.1.
  --eta-max FLOAT     max admixture amplitude of the competing order. Default 0.5.
  --z FLOAT           confidence multiplier on error bars. Default 2.0.
  --char-min FLOAT    minimum |<R>| for a root/branch to count as having a
                      definite character; R0+ and R4 return UNDECIDED whenever
                      either compared branch/root is below it. Default 0.6.
  --n-samples INT     0 (default) = exact expectation values. >0 adds Gaussian
                      Monte Carlo noise emulating N independent samples:
                      std(E) = sigma/sqrt(N), std(sigma^2) = sigma^2*sqrt(2/N),
                      std(<R>) = 1/sqrt(N).
  --rng INT           master random seed. Default 1234.
  --out PATH          JSON file for the full per-instance record. Default
                      results_<lx>x<ly>_<bc>_n<nup>_<ndn>.json next to this script.
  --smoke             tiny preset for a login-node check: 4x2 'po' cluster,
                      3+3 electrons, n-g 3, seeds 2, P 3 (overrides those options).
                      Also runs the operator self-checks (H Hermitian, [H,R]=0,
                      R^2=1, sector energies vs dense diagonalisation).

Outputs: a per-g table of delta and per-rule decisions on stdout, the summary
(CW, DEC, COV per rule; M; G; verdict) on stdout, and the JSON file.
Requires only numpy and scipy.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ----------------------------------------------------------------------------
# argument parsing
# ----------------------------------------------------------------------------
def parse_args(argv):
    p = argparse.ArgumentParser(description="P3.2 biased-ensemble decision-rule test")
    p.add_argument("--lx", type=int, default=6)
    p.add_argument("--ly", type=int, default=2)
    p.add_argument("--bc", choices=["pp", "po", "op", "oo"], default="po")
    p.add_argument("--nup", type=int, default=5)
    p.add_argument("--ndn", type=int, default=5)
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--g-min", type=float, default=-0.4)
    p.add_argument("--g-max", type=float, default=0.4)
    p.add_argument("--n-g", type=int, default=17)
    p.add_argument("--mirror", choices=["x", "y"], default="x")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--P", type=int, default=4)
    p.add_argument("--fit-points", type=int, default=3)
    p.add_argument("--f0", type=float, default=0.1)
    p.add_argument("--q", type=float, default=0.1)
    p.add_argument("--eta-max", type=float, default=0.5)
    p.add_argument("--z", type=float, default=2.0)
    p.add_argument("--char-min", type=float, default=0.6)
    p.add_argument("--n-samples", type=int, default=0)
    p.add_argument("--rng", type=int, default=1234)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    if a.smoke:
        a.lx, a.ly, a.bc, a.nup, a.ndn = 4, 2, "po", 3, 3
        a.n_g, a.seeds, a.P = 3, 2, 3
    if not (2 <= a.fit_points <= a.P + 1):
        p.error("--fit-points must be in 2..P+1")
    if not (0 < a.f0 <= 1):
        p.error("--f0 must be in (0,1]")
    if a.out is None:
        a.out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"results_{a.lx}x{a.ly}_{a.bc}_n{a.nup}_{a.ndn}"
                             + ("_smoke" if a.smoke else "") + ".json")
    return a


# ----------------------------------------------------------------------------
# lattice and single-species operators
# ----------------------------------------------------------------------------
def site(x, y, ly):
    return x * ly + y


def bonds(lx, ly, bc, t, tp):
    """dict (i,j) i<j -> hopping amplitude (positive number multiplying -c+c)."""
    px, py = bc[0] == "p", bc[1] == "p"
    b = {}

    def add(x1, y1, x2, y2, amp):
        if not px and not (0 <= x2 < lx):
            return
        if not py and not (0 <= y2 < ly):
            return
        x2 %= lx
        y2 %= ly
        i, j = site(x1, y1, ly), site(x2, y2, ly)
        if i == j:
            return
        k = (min(i, j), max(i, j))
        b[k] = b.get(k, 0.0) + amp

    for x in range(lx):
        for y in range(ly):
            add(x, y, x + 1, y, t)
            add(x, y, x, y + 1, t)
            if tp != 0.0:
                add(x, y, x + 1, y + 1, tp)
                add(x, y, x + 1, y - 1, tp)
    # remove double-counting from periodic length-1/2 wraps that hit the same pair twice
    return b


def species_basis(ns, n):
    states = [sum(1 << i for i in c) for c in itertools.combinations(range(ns), n)]
    states = np.array(sorted(states), dtype=np.int64)
    index = {s: k for k, s in enumerate(states)}
    return states, index


def popcount(x):
    return bin(x).count("1")


def hop_matrix(states, index, bnds):
    """Single-species kinetic matrix T with T = -sum amp (c+_i c_j + h.c.)."""
    rows, cols, vals = [], [], []
    for k, s in enumerate(states):
        s = int(s)
        for (i, j), amp in bnds.items():
            for a, b_ in ((i, j), (j, i)):  # c+_a c_b
                if (s >> b_) & 1 and not (s >> a) & 1:
                    lo, hi = min(a, b_), max(a, b_)
                    mask = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
                    sign = -1.0 if popcount(s & mask) % 2 else 1.0
                    s2 = s ^ (1 << b_) ^ (1 << a)
                    rows.append(index[s2])
                    cols.append(k)
                    vals.append(-amp * sign)
    n = len(states)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def perm_matrix(states, index, perm):
    """Single-species operator for the site permutation i -> perm[i] with fermion sign."""
    rows, cols, vals = [], [], []
    for k, s in enumerate(states):
        s = int(s)
        occ = [i for i in range(len(perm)) if (s >> i) & 1]
        img = [perm[i] for i in occ]
        # sign of permutation sorting img
        inv = sum(1 for a in range(len(img)) for b in range(a + 1, len(img)) if img[a] > img[b])
        s2 = sum(1 << i for i in img)
        rows.append(index[s2])
        cols.append(k)
        vals.append(-1.0 if inv % 2 else 1.0)
    n = len(states)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


class Hubbard:
    """Matrix-free H on psi stored as (Du, Dd) array: H V = Tu V + V Td^T + U D * V."""

    def __init__(self, lx, ly, bc, nup, ndn, t, tp, U, mirror):
        ns = lx * ly
        self.su, iu = species_basis(ns, nup)
        self.sd, idn = species_basis(ns, ndn)
        b = bonds(lx, ly, bc, t, tp)
        self.Tu = hop_matrix(self.su, iu, b)
        self.Td = hop_matrix(self.sd, idn, b)
        dbl = np.array([[popcount(int(u) & int(d)) for d in self.sd] for u in self.su], dtype=float)
        self.UD = U * dbl
        if mirror == "x":
            perm = [site(lx - 1 - x, y, ly) for x in range(lx) for y in range(ly)]
        else:
            perm = [site(x, ly - 1 - y, ly) for x in range(lx) for y in range(ly)]
        self.Ru = perm_matrix(self.su, iu, perm)
        self.Rd = perm_matrix(self.sd, idn, perm)
        self.shape2 = (len(self.su), len(self.sd))
        self.dim = self.shape2[0] * self.shape2[1]

    def H(self, v):
        V = v.reshape(self.shape2)
        out = self.Tu @ V + (self.Td @ V.T).T + self.UD * V
        return out.reshape(-1)

    def R(self, v):
        V = v.reshape(self.shape2)
        out = self.Ru @ V
        out = (self.Rd @ out.T).T
        return out.reshape(-1)

    def sector_ground(self, parity, v0=None):
        def mv(v):
            w = 0.5 * (v + parity * self.R(v))
            w = self.H(w)
            return 0.5 * (w + parity * self.R(w))
        op = spla.LinearOperator((self.dim, self.dim), matvec=mv, dtype=float)
        e, x = spla.eigsh(op, k=1, which="SA", tol=1e-10, v0=v0)
        x = x[:, 0]
        x = 0.5 * (x + parity * self.R(x))
        x /= np.linalg.norm(x)
        return float(e[0]), x


# ----------------------------------------------------------------------------
# variational branches
# ----------------------------------------------------------------------------
def biased_reference(psiX, psiY, f0, q, eta, rng):
    n = psiX.size
    keep = max(4, int(round(f0 * n)))
    order = np.argsort(-np.abs(psiX))
    kept = order[:keep]
    phi = np.zeros_like(psiX)
    phi[kept] = psiX[kept]
    small = kept[keep // 2:]
    nflip = int(round(q * small.size))
    if nflip > 0:
        flip = rng.choice(small, size=nflip, replace=False)
        phi[flip] *= -1.0
    phi /= np.linalg.norm(phi)
    phi = phi + eta * psiY
    return phi / np.linalg.norm(phi)


def block_krylov_ritz(ham, starts, P, nroots):
    """For p = 0..P return list of (thetas, vectors) for lowest nroots Ritz pairs of
    span{H^k s : s in starts, k <= p}."""
    basis = []
    Hbasis = []
    out = []
    current = [s.copy() for s in starts]
    for p in range(P + 1):
        for v in current:
            w = v.copy()
            for _ in range(2):  # full re-orthogonalisation, twice
                for b in basis:
                    w -= (b @ w) * b
            nrm = np.linalg.norm(w)
            if nrm < 1e-10:
                continue
            w /= nrm
            basis.append(w)
            Hbasis.append(ham.H(w))
        B = np.array(basis)
        HB = np.array(Hbasis)
        Hs = B @ HB.T
        Hs = 0.5 * (Hs + Hs.T)
        th, c = np.linalg.eigh(Hs)
        res = []
        for r in range(min(nroots, len(th))):
            chi = c[:, r] @ B
            Hchi = c[:, r] @ HB
            res.append((float(th[r]), chi, Hchi))
        out.append(res)
        current = [ham.H(v) for v in current]
    return out


def observe(ham, chi, Hchi, nsamp, rng):
    e = float(chi @ Hchi)
    var = max(float(Hchi @ Hchi) - e * e, 0.0)
    r = float(chi @ ham.R(chi))
    if nsamp > 0:
        s = np.sqrt(var)
        e += rng.normal() * s / np.sqrt(nsamp)
        var = abs(var + rng.normal() * var * np.sqrt(2.0 / nsamp))
        r = float(np.clip(r + rng.normal() / np.sqrt(nsamp), -1, 1))
    return e, var, r


# ----------------------------------------------------------------------------
# extrapolation and decision rules
# ----------------------------------------------------------------------------
def extrapolate(var, en, npts):
    x = np.asarray(var[-npts:])
    y = np.asarray(en[-npts:])
    X = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    I_lin = coef[0]
    dof = len(x) - 2
    if dof > 0:
        resid = y - X @ coef
        s2 = resid @ resid / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(max(cov[0, 0], 0.0))
    else:
        se = 0.0
    sysq = 0.0
    if len(x) >= 3:
        Xq = np.vstack([np.ones_like(x), x, x * x]).T
        cq, *_ = np.linalg.lstsq(Xq, y, rcond=None)
        sysq = abs(cq[0] - I_lin)
    return float(I_lin), float(np.sqrt(se ** 2 + sysq ** 2))


def compare(IA, eA, IB, eB, z):
    d = IB - IA
    err = np.sqrt(eA ** 2 + eB ** 2)
    if d > z * err:
        return "A"
    if -d > z * err:
        return "B"
    return "U"


def labelled_rule(branches, z, char_min):
    """branches: list of dicts with I, err, R (character at last level).
    Label by sign(R); handle crossover (same label)."""
    if any(abs(b["R"]) < char_min for b in branches):
        return "U"
    lab = ["A" if b["R"] >= 0 else "B" for b in branches]
    if lab[0] == lab[1]:
        return lab[0]
    bA = branches[lab.index("A")]
    bB = branches[lab.index("B")]
    return compare(bA["I"], bA["err"], bB["I"], bB["err"], z)


# ----------------------------------------------------------------------------
def self_check(a):
    ham = Hubbard(a.lx, a.ly, a.bc, a.nup, a.ndn, a.t, 0.3, a.U, a.mirror)
    n = ham.dim
    I = np.eye(n)
    Hd = np.array([ham.H(I[:, k]) for k in range(n)]).T
    Rd = np.array([ham.R(I[:, k]) for k in range(n)]).T
    print(f"# self-check dim={n} |H-H^T|={np.abs(Hd-Hd.T).max():.1e} "
          f"|HR-RH|={np.abs(Hd@Rd-Rd@Hd).max():.1e} |R^2-1|={np.abs(Rd@Rd-I).max():.1e}")
    for par in (+1, -1):
        Pp = 0.5 * (I + par * Rd)
        w, V = np.linalg.eigh(Pp)
        Q = V[:, w > 0.5]
        e_dense = np.linalg.eigvalsh(Q.T @ Hd @ Q)[0]
        e_lan, _ = ham.sector_ground(par)
        print(f"# self-check parity {par:+d}: dense {e_dense:.10f} lanczos {e_lan:.10f}")
    print(f"# self-check global ground (dense) {np.linalg.eigvalsh(Hd)[0]:.10f}")


def run(a):
    t0 = time.time()
    if a.smoke:
        self_check(a)
    rng = np.random.default_rng(a.rng)
    gs = np.linspace(a.g_min, a.g_max, a.n_g)
    records = []
    print(f"# cluster {a.lx}x{a.ly} bc={a.bc} Nup={a.nup} Ndn={a.ndn} U={a.U} mirror={a.mirror}")
    print(f"# P={a.P} fit_points={a.fit_points} f0={a.f0} q={a.q} eta_max={a.eta_max} "
          f"z={a.z} n_samples={a.n_samples} seeds={a.seeds}")
    for g in gs:
        ham = Hubbard(a.lx, a.ly, a.bc, a.nup, a.ndn, a.t, float(g), a.U, a.mirror)
        EA, psiA = ham.sector_ground(+1)
        EB, psiB = ham.sector_ground(-1)
        delta = EB - EA
        truth = "A" if delta > 0 else "B"
        line = f"g={g:+.3f} dim={ham.dim} E_A={EA:.6f} E_B={EB:.6f} delta={delta:+.2e} truth={truth} |"
        for sd in range(a.seeds):
            etaA, etaB = rng.uniform(0, a.eta_max, size=2)
            phiA = biased_reference(psiA, psiB, a.f0, a.q, etaA, rng)
            phiB = biased_reference(psiB, psiA, a.f0, a.q, etaB, rng)
            indep = {}
            for lab, phi in (("A", phiA), ("B", phiB)):
                path = block_krylov_ritz(ham, [phi], a.P, 1)
                obs = [observe(ham, lev[0][1], lev[0][2], a.n_samples, rng) for lev in path]
                en = [o[0] for o in obs]
                var = [o[1] for o in obs]
                I, err = extrapolate(var, en, a.fit_points)
                indep[lab] = dict(I=I, err=err, R=obs[-1][2], E=en, var=var)
            joint_path = block_krylov_ritz(ham, [phiA, phiB], a.P, 2)
            roots = []
            for r in range(2):
                obs = [observe(ham, lev[r][1], lev[r][2], a.n_samples, rng) for lev in joint_path]
                en = [o[0] for o in obs]
                var = [o[1] for o in obs]
                I, err = extrapolate(var, en, a.fit_points)
                roots.append(dict(I=I, err=err, R=obs[-1][2], E=en, var=var))
            d0 = compare(indep["A"]["I"], indep["A"]["err"], indep["B"]["I"], indep["B"]["err"], a.z)
            d0p = labelled_rule([indep["A"], indep["B"]], a.z, a.char_min)
            d4 = labelled_rule(roots, a.z, a.char_min)

            def cov(b):
                Etrue = EA if b["R"] >= 0 else EB
                return bool(abs(b["I"] - Etrue) <= a.z * b["err"])

            rec = dict(g=float(g), seed=sd, EA=EA, EB=EB, delta=delta, truth=truth,
                       etaA=float(etaA), etaB=float(etaB),
                       R0=d0, R0p=d0p, R4=d4,
                       cov_indep=[cov(indep["A"]), cov(indep["B"])],
                       cov_joint=[cov(roots[0]), cov(roots[1])],
                       indep={k: {kk: vv for kk, vv in v.items()} for k, v in indep.items()},
                       joint=roots)
            records.append(rec)
            line += f" {d0}{d0p}{d4}"
        print(line, flush=True)

    def stats(key):
        n = len(records)
        dec = [r for r in records if r[key] != "U"]
        cw = [r for r in dec if r[key] != r["truth"]]
        return dict(CW=len(cw) / n, DEC=len(dec) / n, n=n)

    S = {k: stats(k) for k in ("R0", "R0p", "R4")}
    S["R0"]["COV"] = S["R0p"]["COV"] = float(np.mean([c for r in records for c in r["cov_indep"]]))
    S["R4"]["COV"] = float(np.mean([c for r in records for c in r["cov_joint"]]))
    M = S["R4"]["CW"]
    G = (S["R4"]["DEC"] - S["R0p"]["DEC"] >= 0.10) or (S["R0p"]["CW"] - M >= 0.05)
    if M <= 0.02 and S["R4"]["DEC"] >= 0.50 and G:
        verdict = "PASS"
    elif M > 0.10 or not G:
        verdict = "FAIL"
    else:
        verdict = "AMBIGUOUS"
    print("# per-instance decision triples above: R0 R0+ R4 (A/B/U)")
    for k in ("R0", "R0p", "R4"):
        print(f"# {k:4s} CW={S[k]['CW']:.3f} DEC={S[k]['DEC']:.3f} COV={S[k]['COV']:.3f} n={S[k]['n']}")
    print(f"# M = CW(R4) = {M:.3f}   G = {G}   VERDICT = {verdict}   wall = {time.time()-t0:.1f}s")
    with open(a.out, "w") as f:
        json.dump(dict(args=vars(a), summary=S, M=M, G=G, verdict=verdict, records=records), f)
    print(f"# wrote {a.out}")
    return verdict


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    run(a)


if __name__ == "__main__":
    main()
