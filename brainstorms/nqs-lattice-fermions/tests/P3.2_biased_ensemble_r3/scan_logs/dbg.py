import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import ed_scan as E

geom, nu, nd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cl = E.Cluster(geom, nu, nd)
H = cl.H(-0.3, 8)
x = np.random.default_rng(0).standard_normal(cl.D)
bad = [(e, np.linalg.norm(H @ cl.Ug(k, x) - cl.Ug(k, H @ x))) for k, e in enumerate(cl.elems)]
print('G', cl.G, 'max [H,Ug]', max(b[1] for b in bad))
chi = np.ones(cl.G)
cl.build_irrep(chi, 0)
B = cl.B
print('BtB-I', abs((B.T @ B).toarray() - np.eye(B.shape[1])).max())
y = np.random.default_rng(1).standard_normal(B.shape[1])
v = B @ y
print('HB closure', np.linalg.norm(H @ v - B @ (B.T @ (H @ v))))
# check P idempotent
Pv = sum(cl.Ug(k, x) for k in range(cl.G)) / cl.G
PPv = sum(cl.Ug(k, Pv) for k in range(cl.G)) / cl.G
print('P idem', np.linalg.norm(PPv - Pv))
print('BBt x vs Px', np.linalg.norm(B @ (B.T @ x) - Pv))
Q = cl.Q
print('dB dS', cl.dB, cl.dS, 'QtQ-I', abs(Q.T @ Q - np.eye(Q.shape[1])).max())
y = np.random.default_rng(2).standard_normal(cl.dS)
v = cl.full_vec(y)
print('S2 of spin basis vec', cl.spin.S2exp(v))
Hp = cl.parts[0] - 0.3 * cl.parts[1] + 8 * cl.parts[2]
print('closure with spin', np.linalg.norm(H @ v - cl.full_vec(Hp @ y)))
Sp = cl.spin.Sp
print('S2 commut', np.linalg.norm(Sp.T @ (Sp @ (H @ x)) - H @ (Sp.T @ (Sp @ x))))
