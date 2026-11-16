import sys, numpy as np, importlib.util, os
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r2", os.path.join(HERE, "../../P2.2_stochastic_local_energy_r2/micro_test.py"))
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
ns = int(sys.argv[1]); natom = int(sys.argv[2]); iters = int(sys.argv[3]); hid = int(sys.argv[4])
rng = np.random.default_rng(1)
h, g, enuc, Erhf, K = r2.build_instance(natom, ns, 2.8, "ring")
N = natom; nocc = N // 2
ham = r2.SQHam(h, g, enuc, K, N); codes, occ, oi = r2.build_space(K, nocc, nocc)
H = r2.build_hamiltonian(ham, codes, oi); E0 = r2.ground_states(H, 1)[0][0]
M = 2 * K; Phi0 = np.zeros((M, N)); Phi0[:K, :nocc] = np.eye(K)[:, :nocc]; Phi0[K:, nocc:] = Phi0[:K, :nocc]
net = r2.NNBF(M, N, hid, Phi0, 0.05, rng)
r = minimize(lambda t: net.energy_grad(t, H, occ, oi), net.theta0, jac=True, method="L-BFGS-B",
             options={"maxiter": iters, "maxcor": 30, "gtol": 1e-10, "ftol": 1e-15})
th = r.x; psi, _ = net.amplitudes(th, occ, oi); w = psi ** 2 / (psi @ psi); Ev = psi @ (H @ psi) / (psi @ psi)
EL = (H @ psi) / psi; V = w @ (EL - Ev) ** 2
print(f"K={K} dim={len(codes)} dE={1e3*(Ev-E0):.3f} mHa Var(EL)={V:.3e}")
P = net.unpack(th); W2r = P["W2"].reshape(M, N, hid)
keys = ["FB", "TS", "TSH", "LRX"]
res = {k: [] for k in keys}; dl = {k: [] for k in keys}; absx = []
for n in range(len(codes)):
    rr = r2.SQHam.row(ham, codes[n], oi[n]); j = np.searchsorted(codes, rr["code"]); hh = rr["h"]
    z, Phi, G = net.setup_one(P, oi[n]); A = Phi[oi[n]]; Ai = np.linalg.inv(A)
    x = hh * psi[j] / psi[n]
    fb = r2.frozen_ratio(G, oi[n], oi[j], rr["rem"], rr["add"])
    s = 1 - np.tanh(z) ** 2
    Z = r2.lrx_setup(W2r, Ai, oi[n]); Y = np.einsum("kkh->h", Z)
    gvec = P["W1"].T @ (s * Y)
    rem, add = rr["rem"], rr["add"]; d2 = rem[:, 1] >= 0
    ex = 2 * (gvec[add[:, 0]] - gvec[rem[:, 0]]); ex[d2] += 2 * (gvec[add[d2, 1]] - gvec[rem[d2, 1]])
    ts = fb * np.exp(ex)
    W1T = P["W1"].T; dz = 2 * (W1T[add[:, 0]] - W1T[rem[:, 0]]); dz[d2] += 2 * (W1T[add[d2, 1]] - W1T[rem[d2, 1]])
    dt = np.tanh(z[None] + dz) - np.tanh(z)[None]
    tsh = fb * np.exp(dt @ Y)
    lx = r2.lrx_ratio(P, W2r, z, Ai, Z, oi[n], rem, add)
    for k, y in (("FB", fb), ("TS", ts), ("TSH", tsh), ("LRX", lx)):
        res[k].append(np.abs(x - hh * y).sum()); dl[k].append((x - hh * y).sum())
    absx.append(np.abs(x).sum())
absx = np.array(absx)
for k in keys:
    a = np.array(res[k]); d = np.array(dl[k]); md = w @ d
    print(f"{k:4s} rho={w@a/(w@absx):.3f}  E[d^2]={w@d**2:.3e}  Var(d)={w@(d-md)**2:.3e}  "
          f"Var(d)/Var(EL)={w@(d-md)**2/V:.3g}  mean d={1e3*md:.2f} mHa")
