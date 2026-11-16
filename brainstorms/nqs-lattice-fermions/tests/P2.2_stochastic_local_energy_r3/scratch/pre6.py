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

KWs=[k for k in (4,6,8) if k<K]
rho={k:[] for k in KWs}; rhoFB=[]; absx=[]; nkey={k:[] for k in KWs}; tau={k:[] for k in KWs}
W1=P["W1"]; b1=P["b1"]
rows=np.random.default_rng(5).choice(len(codes),300,p=w)
for n in rows:
    rr = r2.SQHam.row(ham, codes[n], oi[n]); j = np.searchsorted(codes, rr["code"]); hh = rr["h"]
    x = hh * psi[j] / psi[n]; absx.append(np.abs(x).sum())
    z, Phi, G = net.setup_one(P, oi[n])
    fb = r2.frozen_ratio(G, oi[n], oi[j], rr["rem"], rr["add"]); rhoFB.append(np.abs(x-hh*fb).sum())
    xn=2.0*occ[n]-1; xj=2.0*occ[j]-1
    for KW in KWs:
        inW=np.zeros(M,bool); inW[:KW]=True; inW[K:K+KW]=True
        xm=np.where(inW[None],xj,xn[None])
        hid=np.tanh(xm@W1.T+b1)
        Ph=P["Phi0"][None]+(hid@P["W2"].T).reshape(-1,M,N)
        Aj=np.take_along_axis(Ph,oi[j][:,:,None],axis=1)
        y=hh*np.linalg.det(Aj)/psi[n]
        rho[KW].append(np.abs(x-y).sum())
        nkey[KW].append(len(np.unique(codes[j]&sum(1<<int(p) for p in np.nonzero(inW)[0]))))
        r_=x-y; a=np.abs(r_).sum(); tau[KW].append(a*a-r_.sum()**2)  # oracle-IS floor of residual
absx=np.array(absx); w=np.ones(len(rows))/len(rows)
print("rho_FB", w@np.array(rhoFB)/(w@absx), " <|C|>", w@(np.diff(H.indptr)-1)[rows])
for KW in KWs:
    print(f"KW={KW}: rho_PF={w@np.array(rho[KW])/(w@absx):.4f}  <#keys>={w@np.array(nkey[KW]):.1f}  E[tau2_oracle]/Var(EL)={w@np.array(tau[KW])/V:.3g}")
