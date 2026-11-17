"""Round 22 sub-question A optimiser probes (RULE.md). Usage: run.py ARM ROUTE BUDGET_CPU_MIN
ROUTE in {lbfgs, ncg, plbfgs}. Writes results/<arm>_<route>.json and theta."""
import json
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exgrad as X  # noqa: E402

P = X.P
arm, route, budget = sys.argv[1], sys.argv[2], float(sys.argv[3])
ctx = X.setup(arm)
th0 = ctx.th0.clone()
E0 = P.E0
st = json.load(open(os.path.join(X.R22, f"results/state_{arm}.json")))
acc = [s for s in st["steps"] if s.get("accepted")]
g5 = [s["gain"] / abs(E0) for s in acc[-5:]]
cpus = [s["cpu_min"] for s in st["steps"][-5:]]
gLM = np.mean(g5) / np.mean(cpus)
cpu0 = time.process_time() / 60
hist = []
best = {"E": np.inf, "th": None}


class Stop(Exception):
    pass


M = None
if route == "plbfgs":
    _, Y, _, _ = P.build(ctx, th0)
    O = Y[:, 1:]
    s = Y[:, 0] @ O
    ST = O.T @ O - np.outer(s, s)
    w, U = np.linalg.eigh(0.5 * (ST + ST.T))
    w = np.clip(w, 0, None)
    M = U / np.sqrt(w + 1e-7 * w.max())
    del Y, O
    hist.append(dict(cpu=time.process_time() / 60 - cpu0, E=None, note="metric build"))


def to_th(z):
    return th0 + torch.tensor(M @ z if M is not None else z)


def fg(z):
    cpu = time.process_time() / 60 - cpu0
    if cpu > budget:
        raise Stop
    E, G = X.energy_grad(ctx, to_th(z))
    eps = (E - E0) / abs(E0)
    hist.append(dict(cpu=time.process_time() / 60 - cpu0, E=E, eps=eps))
    print(f"call {len(hist)} cpu {hist[-1]['cpu']:.3f} eps {eps:.6e}", flush=True)
    if E < best["E"]:
        best.update(E=E, th=to_th(z).numpy().copy(), gnorm=float(np.linalg.norm(G)))
    return E, (M.T @ G if M is not None else G)


gcache = {}


def fg_c(z):
    E, g = fg(z)
    gcache["z"], gcache["g"] = z.copy(), g
    return E, g


def hessp(z, p):
    nrm = np.linalg.norm(p)
    if nrm == 0:
        return np.zeros_like(p)
    h = 1e-4
    g0 = gcache["g"] if "z" in gcache and np.array_equal(gcache["z"], z) else fg(z)[1]
    g1 = fg(z + h * p / nrm)[1]
    return nrm * (g1 - g0) / h


z0 = np.zeros(th0.numel())
try:
    if route in ("lbfgs", "plbfgs"):
        minimize(fg, z0, jac=True, method="L-BFGS-B", options=dict(maxcor=30, maxiter=10 ** 6, maxfun=10 ** 6,
                                                                   ftol=0, gtol=0))
    else:
        minimize(fg_c, z0, jac=True, hessp=hessp, method="trust-ncg", options=dict(maxiter=10 ** 6, gtol=0))
except Stop:
    pass
Es = [h for h in hist if h.get("E") is not None]
eps_s = Es[0]["eps"]
eps_e = min(h["eps"] for h in Es)
half = [h for h in Es if h["cpu"] <= budget / 2]
eps_half = min(h["eps"] for h in half) if half else eps_s
tot = Es[-1]["cpu"]
gam = (eps_s - eps_e) / tot
gam_late = (eps_half - eps_e) / (tot - budget / 2) if tot > budget / 2 else float("nan")
v = "PASS-T1" if gam >= 3 * gLM and gam_late >= 2 * gLM else ("KILL" if gam <= gLM else "AMBIGUOUS")
out = dict(arm=arm, route=route, budget=budget, gamma_LM=gLM, eps_start=eps_s, eps_half=eps_half, eps_end=eps_e,
           cpu=tot, gamma=gam, gamma_late=gam_late, ratio=gam / gLM, ratio_late=gam_late / gLM, calls=len(Es),
           gnorm_end=best.get("gnorm"), verdict=v,
           traj=[(round(h["cpu"], 3), h["eps"]) for h in Es])
json.dump(out, open(os.path.join(HERE, f"results/{arm}_{route}.json"), "w"), indent=1)
np.save(os.path.join(HERE, f"results/theta_{arm}_{route}.npy"), best["th"])
print(f"{arm} {route}: eps {eps_s:.4e} -> half {eps_half:.4e} -> end {eps_e:.4e} in {tot:.2f} CPU-min, {len(Es)} calls;"
      f" gamma/gLM {gam / gLM:.2f}, late {gam_late / gLM:.2f}; gLM {gLM:.3e}; VERDICT {v}")
