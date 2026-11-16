"""
r5_core.py -- shared physics for the P3.1 round-5 test (no command line; imported by micro_test.py)
====================================================================================================

Instance (the shared round-5 target, built exactly as P2.1 round 3 instance 0):
  4x4 OBC t-t'-J, N_up = N_dn = 6, t = 1, t' = -0.25, J = 0.4, D = 1681680. psi_0 is a
  non-degenerate singlet with characters Mx = My = +1, D = -1 (|G| = 8).
Code reuse: tj_core.py and p11_core.py are verbatim copies of
tests/P2.1_subcubic_amplitudes_r3/{tj_core,p11_core}.py. resolve_ground, build_symmetry,
references and the NNBF construction below are copied from P2.1 round 3 micro_test.py, so that
the parameter files written by P2.1 jobs (results/params/inst0_nnbf_s{seed}.npz) load into the
identical network.

Observables (all Hermitian, linear in the density matrix, commuting with the space group and
with total spin; "largest r" classes fixed before any data were seen):
  pair   A = sum_{b,b'} M_bb' Delta_b^dag Delta_b', Delta_b = (c_i^up c_j^dn + c_j^up c_i^dn)/sqrt 2
         on NN bond b = (i, j); M_bb' = eta_b eta_b' / (2 n) for every unordered pair of bonds
         whose bond-centre distance is >= R_PAIR (eta = +1 x bond, -1 y bond: d-wave form factor),
         n = number of such pairs. <A> = mean_{pairs} eta eta' <Delta_b^dag Delta_b'>.
  spin   A = (1/n) sum_{(i,j): |r_i - r_j| >= R_SITE} S_i . S_j
  charge A = (1/n) sum_{(i,j): |r_i - r_j| >= R_SITE} (n_i - nbar)(n_j - nbar), nbar = N_e / V.
         The fully connected <n_i n_j> - <n_i><n_j> is reported as a diagnostic only.
"""
import numpy as np
import scipy.sparse as sp

from p11_core import solve_ed, uhf_orbitals
from tj_core import TJ, LRAnsatz, SpaceGroup, between_mask, popcount

INST0 = dict(Lx=4, Ly=4, bcx="obc", bcy="obc", nup=6, ndn=6, tp=-0.25, J=0.4)
UHF_SEED = 1234
R_PAIR = 3.0     # bond-centre distance for the long-range pair correlation
R_SITE = 3.5     # site distance for spin / charge (classes (3,2), (2,3), (3,3))
OBS = ("pair", "spin", "charge")


def make_model(cfg=INST0):
    c = cfg
    m = TJ(c["Lx"], c["Ly"], c["bcx"], c["bcy"], c["nup"], c["ndn"], 1.0, c["tp"], c["J"])
    m.spin_setup()
    return m


def resolve_ground(m, Vv, deg):
    if deg == 1:
        return Vv[:, 0]
    raise RuntimeError("degenerate ground level not supported in round 5 (instance 0 is non-degenerate)")


def build_symmetry(m, psi0):
    for diag in (True, False):
        sg = SpaceGroup(m, diag=diag)
        try:
            gen = sg.set_from(psi0)
            return sg, gen
        except RuntimeError:
            continue
    raise RuntimeError("psi_0 not in a 1-D irrep of the abelian subgroup")


def nnbf(m, width, seed):
    """NNBF of P2.1 r3 arm 'nnbf' (full, K=1, Hs=width, Ha=0) from the UHF reference."""
    Pu, Pd = uhf_orbitals(m, UHF_SEED)
    return LRAnsatz(m, (Pu, Pd), "full", K=1, r=0, Hs=width, Ha=0, shared_ref=True, seed=seed)


def load_params(ans, path):
    z = np.load(path)
    import json
    lay = json.loads(str(z["layout"]))
    assert [k for k, _ in lay] == ans.keys, (lay, ans.keys)
    for k, shp in lay:
        assert list(ans.p[k].shape) == shp, (k, shp, ans.p[k].shape)
    ans.set_flat(z["flat"])
    return ans


# ============================================================================ observables
class Observables:
    def __init__(self, m):
        self.m = m
        V = m.V
        self.coords = m.coords
        # ---- pair
        m.pair_setup(0.0)
        self.bonds = m.pair_bonds
        self.D_ops = m.pair_ops
        cen = [m.bond_center(b) for b in self.bonds]
        eta = np.array([1.0 if b[2] == "x" else -1.0 for b in self.bonds])
        nb = len(self.bonds)
        pairs = [(a, b) for a in range(nb) for b in range(a + 1, nb)
                 if np.hypot(*(cen[a] - cen[b])) >= R_PAIR - 1e-9]
        M = np.zeros((nb, nb))
        for a, b in pairs:
            M[a, b] = M[b, a] = eta[a] * eta[b] / (2 * len(pairs))
        self.M, self.n_pair = M, len(pairs)
        # ---- site pairs
        sp_ = [(i, j) for i in range(V) for j in range(i + 1, V)
               if np.hypot(*(m.coords[i] - m.coords[j])) >= R_SITE - 1e-9]
        self.site_pairs = sp_
        self.n_site = len(sp_)
        self.nbar = (m.nup + m.ndn) / V
        nu = np.array([(m.bu >> i) & 1 for i in range(V)], dtype=float)  # (V, D)
        nd = np.array([(m.bd >> i) & 1 for i in range(V)], dtype=float)
        self.n_site_occ = nu + nd
        szs = 0.5 * (nu - nd)
        diag_s = np.zeros(m.D)
        diag_c = np.zeros(m.D)
        for i, j in sp_:
            diag_s += szs[i] * szs[j]
            diag_c += (self.n_site_occ[i] - self.nbar) * (self.n_site_occ[j] - self.nbar)
        self.Aspin = (sp.diags(diag_s) + self._exchange(sp_)).tocsr() / len(sp_)
        self.diag_charge = diag_c / len(sp_)
        # ---- resolved pieces (one per point-group-invariant distance class of the same correlator;
        #      used as the Galerkin basis of route OSG)
        self.pieces = {"pair": [], "spin": [], "charge": []}
        allb = [(a, b) for a in range(nb) for b in range(a, nb)]
        dist_b = {}
        for a, b in allb:
            dist_b.setdefault(round(float(np.hypot(*(cen[a] - cen[b]))), 6), []).append((a, b))
        for dd in sorted(dist_b):
            Mc = np.zeros((nb, nb))
            for a, b in dist_b[dd]:
                f = eta[a] * eta[b] / len(dist_b[dd])
                if a == b:
                    Mc[a, a] += f
                else:
                    Mc[a, b] += f / 2; Mc[b, a] += f / 2
            self.pieces["pair"].append(Mc)
        alls = [(i, j) for i in range(V) for j in range(i + 1, V)]
        dist_s = {}
        for i, j in alls:
            dist_s.setdefault(round(float(np.hypot(*(m.coords[i] - m.coords[j]))), 6), []).append((i, j))
        for dd in sorted(dist_s):
            prs = dist_s[dd]
            ds = np.zeros(m.D); dc = np.zeros(m.D)
            for i, j in prs:
                ds += szs[i] * szs[j]
                dc += (self.n_site_occ[i] - self.nbar) * (self.n_site_occ[j] - self.nbar)
            self.pieces["spin"].append(((sp.diags(ds) + self._exchange(prs)) / len(prs)).tocsr())
            self.pieces["charge"].append(dc / len(prs))
        self.piece_dists = {"pair": sorted(dist_b), "spin": sorted(dist_s), "charge": sorted(dist_s)}

    def _exchange(self, pairs):
        """sum over pairs of (S+_i S-_j + S-_i S+_j)/2 with Jordan-Wigner signs (as tj_core H)."""
        m = self.m
        bu, bd = m.bu, m.bd
        one = np.int64(1)
        r_, c_, v_ = [], [], []
        for (i, j) in pairs:
            for (a, b) in ((i, j), (j, i)):   # up at b, down at a -> up at a, down at b
                src = np.nonzero(((bu >> b) & 1) & ((bd >> a) & 1))[0]
                if len(src) == 0:
                    continue
                bm = between_mask(a, b)
                nu_ = bu[src] ^ (one << b) ^ (one << a)
                nd_ = bd[src] ^ (one << a) ^ (one << b)
                sg = (1.0 - 2.0 * (popcount(bu[src] & bm) & 1)) * (1.0 - 2.0 * (popcount(bd[src] & bm) & 1))
                tgt = m.index(nu_, nd_)
                assert (tgt >= 0).all()
                r_.append(tgt); c_.append(src); v_.append(-0.5 * sg)
        return sp.csr_matrix((np.concatenate(v_), (np.concatenate(r_), np.concatenate(c_))),
                             shape=(m.D, m.D))

    def apply(self, name, psi):
        if name == "pair":
            vs = np.array([Dop @ psi for Dop in self.D_ops])      # (nb, D')
            u = self.M @ vs
            out = np.zeros_like(psi)
            for b, Dop in enumerate(self.D_ops):
                out += Dop.T @ u[b]
            return out
        if name == "spin":
            return self.Aspin @ psi
        if name == "charge":
            return self.diag_charge * psi
        raise KeyError(name)

    def apply_piece(self, name, P, psi):
        if name == "pair":
            return self.apply_pieces_all("pair", psi, [P])[0]
        if name == "spin":
            return P @ psi
        return P * psi

    def apply_pieces_all(self, name, psi, plist=None):
        plist = self.pieces[name] if plist is None else plist
        if name != "pair":
            return [self.apply_piece(name, P, psi) for P in plist]
        vs = np.array([Dop @ psi for Dop in self.D_ops])
        outs = []
        for P in plist:
            u = P @ vs
            out = np.zeros_like(psi)
            for b, Dop in enumerate(self.D_ops):
                if np.any(P[b] != 0):
                    out += Dop.T @ u[b]
            outs.append(out)
        return outs

    def expect(self, name, psi):
        p = psi / np.linalg.norm(psi)
        return float(p @ self.apply(name, p))

    def connected_charge(self, psi):
        p2 = (psi / np.linalg.norm(psi)) ** 2
        n = self.n_site_occ @ p2
        return float(np.mean([(self.n_site_occ[i] * self.n_site_occ[j]) @ p2 - n[i] * n[j]
                              for i, j in self.site_pairs]))

    def all(self, psi):
        return {k: self.expect(k, psi) for k in OBS}


def setup(n_ed=4, log=print):
    import time
    t0 = time.time()
    m = make_model()
    W, Vv, deg = solve_ed(m, n_ed, 1e-8, UHF_SEED)
    psi0 = resolve_ground(m, Vv, deg)
    psi0 = psi0 / np.linalg.norm(psi0)
    sg, gen = build_symmetry(m, psi0)
    S2 = m.S2_expect(psi0)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * S2) - 1)))
    log(f"ED {W.tolist()} deg {deg} |G|={sg.G} chi={gen} S0={S0} ({time.time() - t0:.0f} s)")
    ob = Observables(m)
    log(f"observables: pair pairs (R>={R_PAIR}) {ob.n_pair}, site pairs (R>={R_SITE}) {ob.n_site}")
    return m, W, psi0, sg, S0, ob
