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
feats=[]; ELr=[]; tailrho={k:[] for k in ("FB","FITL","FITX")}; absT=[]
fr=float(sys.argv[5])
for n in range(len(codes)):
    rr = r2.SQHam.row(ham, codes[n], oi[n]); j = np.searchsorted(codes, rr["code"]); hh = rr["h"]
    z, Phi, G = net.setup_one(P, oi[n])
    x = hh * psi[j] / psi[n]
    fb = r2.frozen_ratio(G, oi[n], oi[j], rr["rem"], rr["add"]); y=hh*fb
    rem, add = rr["rem"], rr["add"]; d2 = rem[:, 1] >= 0
    order=np.argsort(-np.abs(y)); nD=max(int(fr*len(y)),M+2); D=order[:nD]; T=order[nD:]
    F=np.zeros((len(y),M+1)); F[:,M]=1
    np.add.at(F,(np.arange(len(y)),add[:,0]),1.); np.add.at(F,(np.arange(len(y)),rem[:,0]),-1.)
    F[d2,add[d2,1]]+=1; F[d2,rem[d2,1]]-=1
    # linear multiplicative fit x ~ y (1 + F c): least squares on D
    Xd=y[D,None]*F[D]; c,*_=np.linalg.lstsq(Xd, x[D]-y[D], rcond=1e-8)
    yl=y*(1+F@c)
    # log fit
    ok=np.abs(y[D])>1e-14; rat=x[D][ok]/y[D][ok]; pos=rat>0
    wgt=np.abs(y[D][ok][pos]); cl,*_=np.linalg.lstsq(F[D][ok][pos]*wgt[:,None], np.log(rat[pos])*wgt, rcond=1e-8)
    yx=y*np.exp(np.clip(F@cl,-5,5))
    absT.append(np.abs(x[T]).sum())
    for k,yy in (("FB",y),("FITL",yl),("FITX",yx)): tailrho[k].append(np.abs(x[T]-yy[T]).sum())
    hd=x[D].sum()
    feats.append([rr["Hnn"], y.sum(), hd, hd+y[T].sum(), hd+yl[T].sum(), hd+yx[T].sum()])
    ELr.append(rr["Hnn"]+x.sum())
F=np.array(feats); ELr=np.array(ELr); absT=np.array(absT)
print("frac exact", fr)
for k in tailrho: print(k, "tail rho", w@np.array(tailrho[k])/(w@absT))
names=["Hnn","FBsum","head","head+FB","head+FITL","head+FITX"]
def wr2(X):
    X1=np.column_stack([X,np.ones(len(X))]); sw=np.sqrt(w)
    b,*_=np.linalg.lstsq(X1*sw[:,None], ELr*sw, rcond=None); res=ELr-X1@b; mr=w@res
    return 1-(w@(res-mr)**2)/V
for i,nm in enumerate(names): 
    d=ELr-F[:,i]; md=w@d
    print(f"{nm:10s} R2(regression)={wr2(F[:,[i]]):.4f}  identity Var(d)/V={(w@(d-md)**2)/V:.3g} bias={1e3*md:.2f} mHa")
print("all feats R2", wr2(F))
