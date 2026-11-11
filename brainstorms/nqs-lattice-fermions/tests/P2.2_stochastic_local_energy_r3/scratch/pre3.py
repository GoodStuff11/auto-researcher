"""Pre-check: Cholesky sum-of-squares (Jackson-Feenberg-type) energy estimator."""
import sys, numpy as np, importlib.util, os
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r2", os.path.join(HERE, "../../P2.2_stochastic_local_energy_r2/micro_test.py"))
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
ns = int(sys.argv[1]); natom = int(sys.argv[2]); iters = [int(x) for x in sys.argv[3].split(",")]; hid = int(sys.argv[4])
geom = sys.argv[5] if len(sys.argv) > 5 else "ring"
rng = np.random.default_rng(1)
h, g, enuc, Erhf, K = r2.build_instance(natom, ns, 2.8, geom)
N = natom; nocc = N // 2
ham = r2.SQHam(h, g, enuc, K, N); codes, occ, oi = r2.build_space(K, nocc, nocc)
H = r2.build_hamiltonian(ham, codes, oi); E0 = r2.ground_states(H, 1)[0][0]
M = 2 * K
# Cholesky-like factorisation of (pq|rs)
Vm = g.reshape(K * K, K * K); ev, U = np.linalg.eigh(Vm); keep = ev > 1e-10
L = (U[:, keep] * np.sqrt(ev[keep])).T.reshape(-1, K, K)  # L[gamma,p,q]
hp = h - 0.5 * np.einsum("pqqs->ps", g)
print("Ng=", L.shape[0], "factor err", np.abs(np.einsum("gpq,grs->pqrs", L, L) - g).max())
# one-body local Green function g_pq(m) = <m|E_pq|psi>/psi(m) for all m (full space)
D = len(codes)
def Gloc(psi):
    out = np.zeros((D, K, K))
    for s in (0, 1):
        for p in range(K):
            P_ = p + s * K
            for q in range(K):
                Q = q + s * K
                if p == q:
                    out[:, p, p] += occ[:, P_]; continue
                m = (occ[:, P_] == 1) & (occ[:, Q] == 0)
                idx = np.nonzero(m)[0]
                o = occ[idx].astype(np.int64)
                newc = codes[idx] - (1 << P_) + (1 << Q)
                j = np.searchsorted(codes, newc)
                # <m| c+_P c_Q |m'> with m' = m - P + Q
                o2 = o.copy(); o2[:, P_] = 0; o2[:, Q] = 1
                s1 = o2[:, :Q].sum(1)  # c_Q on m'
                o2[:, Q] = 0
                s2 = o2[:, :P_].sum(1)  # c+_P
                out[idx, p, q] += (-1.0) ** (s1 + s2) * psi[j] / psi[idx]
    return out
Phi0 = np.zeros((M, N)); Phi0[:K, :nocc] = np.eye(K)[:, :nocc]; Phi0[K:, nocc:] = Phi0[:K, :nocc]
net = r2.NNBF(M, N, hid, Phi0, 0.05, rng)
th = net.theta0; done = 0
nsing = 2 * nocc * (K - nocc)
ncon = np.diff(H.indptr) - 1
for it in iters:
    if it > done:
        r = minimize(lambda t: net.energy_grad(t, H, occ, oi), th, jac=True, method="L-BFGS-B",
                     options={"maxiter": it - done, "maxcor": 30, "gtol": 1e-12, "ftol": 1e-16})
        th = r.x; done = it
    psi, _ = net.amplitudes(th, occ, oi); w = psi ** 2 / (psi @ psi); Ev = psi @ (H @ psi) / (psi @ psi)
    EL = (H @ psi) / psi; V = w @ (EL - Ev) ** 2
    Gl = Gloc(psi)
    E1 = np.einsum("mpq,pq->m", Gl, hp)
    ell = np.einsum("mpq,gpq->mg", Gl, L)
    EJ = enuc + E1 + 0.5 * (ell ** 2).sum(1)
    VJ = w @ (EJ - w @ EJ) ** 2
    cn = w @ ncon
    print(f"it={it} K={K} dim={D} dE={1e3*(Ev-E0):.3f} mHa  <EJF>-E={w@EJ-Ev:.2e}  Var(EL)={V:.3e}  Var(EJF)={VJ:.3e}  "
          f"ratio={VJ/V:.3g}  <|C|>={cn:.0f} singles={nsing}  cost ratio (fwd only)={VJ/V*(1+nsing)/(1+cn):.3g}")
    # exact ground state
psi0 = r2.ground_states(H, 1)[1][:, 0]
w = psi0 ** 2; Gl = Gloc(psi0)
EJ = enuc + np.einsum("mpq,pq->m", Gl, hp) + 0.5 * (np.einsum("mpq,gpq->mg", Gl, L) ** 2).sum(1)
print(f"exact psi0: <EJF>-E0={w@EJ-E0:.2e} Var(EJF)={w@(EJ-w@EJ)**2:.3e}")
