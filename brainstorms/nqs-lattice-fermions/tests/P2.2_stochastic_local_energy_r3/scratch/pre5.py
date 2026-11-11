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

for KW in (4,6,8,10):
    if KW>=K: continue
    inW=np.zeros(M,bool); inW[:KW]=True; inW[K:K+KW]=True
    dl=[]; kept=[]; ELr=[]
    for n in range(len(codes)):
        rr = r2.SQHam.row(ham, codes[n], oi[n]); j = np.searchsorted(codes, rr["code"]); hh = rr["h"]
        x = hh * psi[j] / psi[n]; add=rr["add"]
        m = inW[add[:,0]] & ((add[:,1]<0) | inW[np.maximum(add[:,1],0)])
        dl.append(x[~m].sum()); kept.append(m.mean())
    d=np.array(dl); md=w@d; v=w@(d-md)**2; kf=w@np.array(kept)
    # MF cost ratio with identity coefficient, cheap model cost = kept fraction
    q=min(1,np.sqrt(v/max(V-v,1e-30)*kf)) if V>v else 1
    ratio=(np.sqrt(max(V-v,0)*kf)+np.sqrt(v))**2/V
    print(f"K={K} KW={KW} kept={kf:.3f} bias={1e3*md:.2f} mHa Var(dtail)/Var(EL)={v/V:.3g} MF ratio={ratio:.3g}")
