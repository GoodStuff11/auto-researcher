import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tj_core import *
from p11_core import energy, uhf_orbitals
m = TJ(2, 1, 'obc', 'obc', 1, 1, 1.0, 0.0, 0.4); print('2site', np.linalg.eigvalsh(m.H.toarray()))
m = TJ(3, 2, 'pbc', 'obc', 2, 2, 1.0, -0.25, 0.4)
H = m.H; print('D', m.D, 'herm', abs(H - H.T).max())
m.spin_setup(); v = np.random.default_rng(0).standard_normal(m.D)
print('[H,S2]', np.linalg.norm(m.S2(H @ v) - H @ m.S2(v)))
w, U = np.linalg.eigh(H.toarray()); p0 = U[:, 0]; print('E0', w[:3], 'S2', m.S2_expect(p0))
sg = SpaceGroup(m); print('G', sg.G, max(np.linalg.norm(sg.apply(H @ v, k) - H @ sg.apply(v, k)) for k in range(sg.G)))
print(sg.set_from(p0)); Pv = sg.project(v); print('idem', np.linalg.norm(sg.project(Pv) - Pv), 'P psi0', np.linalg.norm(sg.project(p0) - p0))
Pu, Pd = uhf_orbitals(m, 1)
F = dwave_bcs_F(m)
iu, idd = m.iu_id()
for mode, kw, ref in [('lr', dict(K=3, r=2, Hs=5, Ha=3), (Pu, Pd)), ('lr', dict(K=1, r=0, Hs=0, Ha=3), (Pu, Pd)),
                      ('agp', dict(K=3, r=2, Hs=5, Ha=3), F), ('agp', dict(K=1, r=0, Hs=0, Ha=3), F),
                      ('full', dict(K=2, Hs=4, Ha=0, shared_ref=False), (Pu, Pd))]:
    a = LRAnsatz(m, ref, mode, seed=3, w2_scale=0.3, **kw)
    rng = np.random.default_rng(1)
    if a.Ha: a.p['w2a'] = 0.3 * rng.standard_normal(a.Ha)
    if a.K > 1: a.p['wk'] = 1 + 0.5 * rng.standard_normal(a.K)
    x0 = a.flat().copy(); psi = a.amplitudes(4096); E, Hp, nn = energy(H, psi)
    g = 2 * a.force((Hp - E * psi) / nn, 4096); d = rng.standard_normal(len(x0)); d /= np.linalg.norm(d); h = 1e-6
    a.set_flat(x0 + h * d); Ep = energy(H, a.amplitudes(4096))[0]; a.set_flat(x0 - h * d); Em = energy(H, a.amplitudes(4096))[0]; a.set_flat(x0)
    O = a.per_sample_O(np.arange(m.D)); g2 = 2 * (((Hp - E * psi) / nn) * psi) @ O
    print(mode, kw, 'grad', g @ d, (Ep - Em) / 2 / h, 'Opath', np.abs(g2 - g).max())
    Hpsi = H @ psi; err = 0; errb = 0
    for j in range(m.D):
        ou = m.occ_u[iu[j]].astype(int); od = m.occ_d[idd[j]].astype(int)
        ref_ = Hpsi[j] / psi[j]
        eb = eloc_brute(a, m.T, m.nn, m.J, ou, od); errb = max(errb, abs(eb - ref_) / (1 + abs(ref_)))
        if mode == 'lr': ef = LRCarried(a, m.T, m.nn, m.J, ou, od).eloc()
        elif mode == 'agp': ef = eloc_fast_agp(a, m.T, m.nn, m.J, ou, od)[0]
        else: continue
        err = max(err, abs(ef - ref_) / (1 + abs(ref_)))
    print('  eloc fast err', err, 'brute err', errb)
