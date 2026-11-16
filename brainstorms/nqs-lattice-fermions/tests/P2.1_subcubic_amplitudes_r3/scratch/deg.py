import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from scipy.sparse.linalg import eigsh
from tj_core import TJ, SpaceGroup
tp = float(sys.argv[1]) if len(sys.argv) > 1 else -0.25
m = TJ(4, 4, "pbc", "pbc", 7, 7, 1.0, tp, 0.4)
w, v = eigsh(m.H, k=4, which="SA", tol=1e-12, v0=np.random.default_rng(0).standard_normal(m.D))
o = np.argsort(w); w, v = w[o], v[:, o]
print(w)
sg = SpaceGroup(m, diag=True)
Q = v[:, :2]
for k, e in enumerate(sg.elems):
    if e in ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)):
        Mg = Q.T @ np.column_stack([sg.apply(Q[:, i], k) for i in range(2)])
        print(e, np.round(Mg, 4).tolist(), np.round(np.linalg.eigvals(Mg), 4))
