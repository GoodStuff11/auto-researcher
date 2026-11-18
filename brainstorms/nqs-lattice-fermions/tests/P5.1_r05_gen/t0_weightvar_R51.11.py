import os, sys, json, time
import numpy as np
WD = "/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests"
sys.path.insert(0, WD + "/P5.1_size_scaling_r04"); os.chdir(WD + "/P5.1_size_scaling_r04")
import ns_run as NS, torch
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
R = WD + "/P5.1_size_scaling_r04/results/"
cases = {"S5x2": (5, 2, 8, R + "S5x2_N8_exact_tau0.005_s9000/theta.npy"),
         "S4x3": (4, 3, 10, R + "preflight_S4x3_exact/theta.npy"),
         "S6x2": (6, 2, 10, R + "S6x2_N10_exact_tau0.005_s9000/theta.npy")}
only = sys.argv[1:] or list(cases)


def stats(size, pr):
    a = np.abs(pr); m = a > 1e-14 * a.max()
    size, a = size[m], a[m]
    Z1, Z2, Z3 = (size * a).sum(), (size * a**2).sum(), (size * a**3).sum()
    pc = a**2 / Z2; q1 = a / Z1; qg = 0.8 * pc + 0.2 * q1
    H = -(size * pc * np.log(pc)).sum()
    pu = size * pc; qk = np.sqrt(size) * a; qk /= qk.sum()  # kq.py convention
    return dict(D=float(size.sum()), expH=float(np.exp(H)), Ew2_q1=float(Z1 * Z3 / Z2**2),
                Ew2_guided=float((size * pc**2 / qg).sum()), Ew2_q1_kqconv=float((pu**2 / qk).sum()))


for k in only:
    Lx, Ly, Ne, tr = cases[k]
    t = time.process_time()
    st = json.load(open(R + f"kappa_{k}_start.json"))["theta"]
    ctx = NS.GCtx(Lx, Ly, Ne, st)
    o = {}
    _, pr, _ = ctx.energy(ctx.th0); o["start"] = stats(ctx.size, pr)
    th = torch.tensor(np.load(tr)); _, pr, _ = ctx.energy(th); o["trained"] = stats(ctx.size, pr)
    K = sp.diags(ctx.sw) @ ctx.Hr @ sp.diags(1.0 / ctx.sw); K = 0.5 * (K + K.T)
    e, v = eigsh(K, k=1, which="SA", tol=1e-10); o["ground"] = stats(ctx.size, v[:, 0] / ctx.sw); o["E0"] = float(e[0])
    o["V"] = Lx * Ly; o["cpu_s"] = time.process_time() - t
    print("RESULT", k, json.dumps(o), flush=True)
