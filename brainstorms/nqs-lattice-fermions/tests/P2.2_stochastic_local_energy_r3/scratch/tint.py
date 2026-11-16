import numpy as np, importlib.util, os, time, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ints
spec = importlib.util.spec_from_file_location("r2", os.path.join(HERE, "../../P2.2_stochastic_local_energy_r2/micro_test.py"))
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
X = r2.geometry(4, 2.0, "ring")
S0, h0, G0, e0 = r2.ao_integrals(X, 2)
basis = ints.STO3G_H + [(0, [0.08], [1.0])]
S, h, G, e, _ = ints.integrals(X, basis)
print("s-only vs r2:", abs(S - S0).max(), abs(h - h0).max(), abs(G - G0).max())
t = time.time()
X = r2.geometry(4, 2.8, "ring"); S, h, G, e, lab = ints.integrals(X, ints.H631GSS)
print("K=", S.shape[0], "time", time.time() - t)
K = S.shape[0]
print("ERI sym", abs(G - G.transpose(1, 0, 2, 3)).max(), abs(G - G.transpose(2, 3, 0, 1)).max(),
      "min eig", np.linalg.eigvalsh(G.reshape(K * K, K * K)).min())
C, E, _ = r2.rhf(S, h, G, 2); print("E_RHF H4 ring 6-31G**", E + e)
rng = np.random.default_rng(0); Qm, _ = np.linalg.qr(rng.standard_normal((3, 3)))
S2, h2, G2, e2, _ = ints.integrals(X @ Qm.T + np.array([0.3, -0.2, 0.5]), ints.H631GSS)
C2, E2, _ = r2.rhf(S2, h2, G2, 2); print("rotated", E2 + e2, "diff", E2 + e2 - E - e)
X2 = np.array([[0, 0, 0], [1.4, 0, 0.]]); S, h, G, e, _ = ints.integrals(X2, ints.H631GSS)
C, E, _ = r2.rhf(S, h, G, 1); print("H2 6-31G** R=1.4 RHF", E + e)
