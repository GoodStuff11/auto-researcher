#!/usr/bin/env python3
"""
P1.2 reopening review (2026-11-16): T1 of route RPC, the ratio-fed PHASE carrier. Rules written here BEFORE any run.
====================================================================================================================

Why this route. The Saturation Diagnostic says ratio features LOCATE the trained foil's wrong signs (held-out AUC
0.93 at V = 12) but every continuous REAL carrier (second pass, tied re-entry, low-rank kernel LRK) ties width:
SR spends the new capacity on amplitudes. A real carrier can change a sign only by driving psi through zero, where
log|psi| diverges (barrier (b) of the leaf). RPC moves the sign by a PHASE, so |psi| never vanishes:
    psi_RPC(n) = psi_F(n) * exp(i phi(n)),   phi(n) = pi * sum_v [ tanh(phi_v(n) Wk + bk) . a + c ],
with psi_F the trained t1c foil (frozen; NNBF width 6) and phi_v(n) = (x_up_v, x_dn_v, f_0v, f_1v, f_2v) the
per-site input of LRK (frame-H frozen Green's-function ratio features of the foil, t1b_drf / t1c FastDRF).
K = 8 hidden units, weight-shared over sites, so phi is a sum of per-site terms (extensive) and the carrier has
5K + K + K + 1 = 57 parameters at every V. For real symmetric H,
    E[phi] = sum_{nm} psi_n psi_m H_nm cos(phi_n - phi_m) / sum_n psi_n^2 ,
which couples the real and imaginary parts (unlike the killed complex detour 6.4, whose two parts were
independent networks). At phi = 0 the gradient vanishes identically (T0 lemma), and the second-order change is
-(1/2) <[g,[g,H]]> = (1/2) sum_{n != m} (-psi_n psi_m H_nm)(g_n - g_m)^2 / N: negative curvature exists exactly
along phase patterns whose gradients sit on frustrated bonds (psi_n psi_m H_nm > 0). The kernel starts at a small
random a (scale 0.1) to leave the saddle.
Read-out (real, sign only): s(n) = sgn(psi_F(n) cos phi(n)); metric dEs = t1_vote.dE_sign(H, s, psi0, E0, 3000),
the same sign-constrained minimum used for every arm in this leaf.

Arms (sizes 0, 1, 2 of t1c; foil seeds 11 and 22; unprojected, as the diagnostic's scope):
  rpc   the carrier above, features G (frozen ratios from O1 A^-1).
  rpcO  attribution control: identical, features O (G replaced by O1; one change, as lrkO/wsO).
  free  premise ceiling: an unrestricted phase per configuration (D parameters; not extensive; decides nothing
        about RPC's expressivity, only whether ANY phase optimisation on the foil's frozen amplitudes beats width).
Optimiser: scipy L-BFGS-B on the exact energy, maxiter 300 (rpc, rpcO: 2 random inits per foil seed, the lower
ENERGY kept, no oracle), maxiter 500 for free (init phi = 0.1 * N(0,1)).
Comparator: the stored t1e `wid` arm (the same foil widened to 1255 parameters and trained 300 more SR steps;
results/t1e/A_<size>_wid_<seed>.json). It has more added parameters (145) than rpc (57), which biases toward KILL.

RULES (median over the two foil seeds; counted size iff the foil's median dEs >= 0.005, so sizes 1 and 2):
  PREMISE  r_free = dEs[free] / dEs[wid]. PREMISE-KILL if r_free >= 0.9 at both counted sizes: phase optimisation
           on frozen foil amplitudes cannot beat width even unrestricted, and RPC dies with its family.
  RPC      r = dEs[rpc] / dEs[wid]. KILL if r >= 0.9 at >= 2 counted sizes including the largest (size 2).
           PASS-W if r <= 0.7 at every counted size and r(2) <= r(1) + 0.1. UNDECIDED otherwise.
  ATTRIB   after PASS-W only: credited to the inverse iff dEs[rpc] <= 0.9 dEs[rpcO] at size 2.
  PASS-W with ATTRIB promotes RPC to a T2 design (value-of-information note first). Nothing is submitted here.
Validity (else INVALID): t1c gap asserts (via t1c_warm.model); CHECK A the copied pass 1 reproduces the foil
  amplitudes (rel < 1e-10); CHECK B kernel gradient vs central finite differences (rel < 1e-5) on size 0;
  CHECK C E[phi = 0] equals the foil energy (< 1e-12).
Budget (enforced before the heavy part): RLIMIT_CPU per invocation (--cpu_cap, default 580 s); 5 timed energy+
  gradient evaluations project each arm (x 1.3 for line-search overhead); abort if the projection exceeds 90% of
  the invocation cap or if the ledger (results/t1f/ledger.json) plus projection exceeds --task_cap (1800 CPU-s).
  Threads 1 by default, at most 4.

Usage
-----
  python3 t1f_phase.py --mode checks
  python3 t1f_phase.py --mode train --sizes 0,1,2 --arms free,rpc
  python3 t1f_phase.py --mode train --sizes 2 --arms rpcO
  python3 t1f_phase.py --mode verdict
Options: --mode {checks,train,verdict}
           checks : CHECKs A, B, C (prints them, writes results/t1f/checks.json)
           train  : the listed arms at the listed sizes, one JSON per (size, arm, seed) in results/t1f/
           verdict: applies PREMISE, RPC and ATTRIB to the JSONs present and the stored t1e wid JSONs
         --sizes LIST   subset of 0,1,2 (default 0,1,2)
         --arms LIST    subset of free,rpc,rpcO (default free,rpc)
         --seeds LIST   foil seeds (default 11,22; only seeds with a saved t1c foil exist)
         --threads INT  1..4 (default 1)
         --cpu_cap FLOAT per-invocation RLIMIT_CPU in CPU-s (default 580)
         --task_cap FLOAT ledger cap in CPU-s for this test (default 1800)
"""
import os
import sys


def _pre_threads():
    n = 1
    for i, a in enumerate(sys.argv):
        if a.startswith("--threads="):
            n = int(a.split("=")[1])
        if a == "--threads" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
    n = max(1, min(4, n))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(n)
    return n


NTHREADS = _pre_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R6 = os.path.join(os.path.dirname(HERE), "P1.2_unanchored_antisymmetry_r6")
sys.path.insert(0, R6)
import t1c_warm as C  # noqa: E402
import t1b_drf as B  # noqa: E402
from t1_vote import dE_sign  # noqa: E402

OUT = os.path.join(HERE, "results", "t1f")
LEDGER = os.path.join(OUT, "ledger.json")
T1E = os.path.join(R6, "results", "t1e")
KH = 8


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ data per (size, seed)
def setup(k, seed, feat):
    """Foil amplitudes psiF (D,), per-site inputs Phi (D, V, 5), H, psi0, E0, and the CHECK A residual."""
    m, psi0, E0, ref, gap = C.model(k)
    nb = C.load_foil(k, seed)
    psiF = nb.amplitudes(8192)
    dr = C.FastDRF(m, ref, B.WC, 1, "H", seed + 2000, feat)
    C.copy_pass1(dr, nb)
    dr.p["W4"][:] = 0.0
    V, D = m.V, m.D
    Phi = np.empty((D, V, 5))
    amp = np.empty(D)
    for s in range(0, D, 4096):
        idx = np.arange(s, min(s + 4096, D))
        c, _, _ = dr._fw(idx)
        x, f = c["x"], c["z"][:, 2 * V:]
        Phi[idx] = np.stack([x[:, :V], x[:, V:], f[:, :V], f[:, V:2 * V], f[:, 2 * V:]], 2)
        amp[idx] = c["psi"]
    chkA = float(np.max(np.abs(amp - psiF)) / np.max(np.abs(psiF)))
    return dict(m=m, H=m.H.tocsr(), psi0=psi0, E0=E0, psiF=psiF, Phi=Phi, chkA=chkA, gap=gap)


# ============================================================================ energy of psiF exp(i phi)
def e_and_dphi(H, psi, phi):
    c, s = psi * np.cos(phi), psi * np.sin(phi)
    Hc, Hs = H @ c, H @ s
    N = psi @ psi
    E = (c @ Hc + s @ Hs) / N
    g = 2.0 / N * (-s * Hc + c * Hs)
    return float(E), g


# ============================================================================ kernel carrier
def unpack(th):
    Wk = th[:5 * KH].reshape(5, KH)
    bk = th[5 * KH:6 * KH]
    a = th[6 * KH:7 * KH]
    c = th[7 * KH]
    return Wk, bk, a, c


def n_kernel():
    return 7 * KH + 1


def phi_kernel(th, Phi):
    Wk, bk, a, c = unpack(th)
    h = np.tanh(Phi @ Wk + bk)                      # (D, V, K)
    phi = np.pi * (h @ a + c).sum(1)                # (D,)
    return phi, h


def obj_kernel(th, H, psi, Phi):
    phi, h = phi_kernel(th, Phi)
    E, gphi = e_and_dphi(H, psi, phi)
    Wk, bk, a, c = unpack(th)
    gp = np.pi * gphi                               # dE / d(sum_v (h.a + c))
    ga = np.einsum("n,nvk->k", gp, h)
    gc = gp.sum() * Phi.shape[1]
    dpre = gp[:, None, None] * a[None, None, :] * (1 - h * h)   # (D, V, K)
    gW = np.einsum("nvi,nvk->ik", Phi, dpre)
    gb = dpre.sum((0, 1))
    return E, np.concatenate([gW.ravel(), gb, ga, [gc]])


def init_kernel(rng):
    th = np.zeros(n_kernel())
    th[:5 * KH] = rng.standard_normal(5 * KH) / np.sqrt(5)
    th[6 * KH:7 * KH] = 0.1 * rng.standard_normal(KH)
    return th


def readout(psiF, phi):
    return np.where(psiF * np.cos(phi) >= 0, 1.0, -1.0)


# ============================================================================ ledger and budget
def ledger_used():
    if not os.path.exists(LEDGER):
        return 0.0
    return float(sum(r["cpu"] for r in json.load(open(LEDGER))))


def ledger_add(mode, cpu, note):
    rows = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    rows.append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), mode=mode, cpu=cpu, note=note))
    json.dump(rows, open(LEDGER, "w"), indent=1)


def time_eval(fn, n=5):
    fn()
    t0 = time.process_time()
    for _ in range(n):
        fn()
    return (time.process_time() - t0) / n


# ============================================================================ modes
def run_checks(args):
    rng = np.random.default_rng(0)
    d = setup(0, 11, "G")
    E_f = float(d["psiF"] @ (d["H"] @ d["psiF"]) / (d["psiF"] @ d["psiF"]))
    E0phi, _ = e_and_dphi(d["H"], d["psiF"], np.zeros(len(d["psiF"])))
    th = init_kernel(rng) + 0.05 * rng.standard_normal(n_kernel())
    _, g = obj_kernel(th, d["H"], d["psiF"], d["Phi"])
    fd = np.empty_like(th)
    for i in range(len(th)):
        e = np.zeros_like(th)
        e[i] = 1e-6
        fd[i] = (obj_kernel(th + e, d["H"], d["psiF"], d["Phi"])[0]
                 - obj_kernel(th - e, d["H"], d["psiF"], d["Phi"])[0]) / 2e-6
    chkB = float(np.max(np.abs(fd - g)) / np.max(np.abs(g)))
    chkC = abs(E0phi - E_f)
    _, g0 = obj_kernel(np.concatenate([th[:6 * KH], np.zeros(KH), [0.0]]), d["H"], d["psiF"], d["Phi"])
    res = dict(chkA=d["chkA"], chkB=chkB, chkC=chkC, grad_at_phi0=float(np.max(np.abs(g0))),
               valid=bool(d["chkA"] < 1e-10 and chkB < 1e-5 and chkC < 1e-12))
    log(f"CHECK A copied pass 1 vs foil rel {d['chkA']:.2e}; CHECK B kernel grad vs FD rel {chkB:.2e}; "
        f"CHECK C E[phi=0]-E_foil {chkC:.2e}; T0 lemma max|grad| at phi=0 {res['grad_at_phi0']:.2e}; "
        f"valid {res['valid']}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "checks.json"), "w"), indent=1)
    ledger_add("checks", time.process_time(), "")
    return 0


def train_one(arm, d, seed):
    H, psiF, Phi = d["H"], d["psiF"], d["Phi"]
    rng = np.random.default_rng(seed + 7)
    if arm == "free":
        x0 = 0.1 * rng.standard_normal(len(psiF))
        r = minimize(lambda p: e_and_dphi(H, psiF, p), x0, jac=True, method="L-BFGS-B",
                     options=dict(maxiter=500))
        phi, E, npar, nit = r.x, r.fun, len(psiF), r.nit
    else:
        best = None
        for init in range(2):
            th0 = init_kernel(rng)
            r = minimize(lambda t: obj_kernel(t, H, psiF, Phi), th0, jac=True, method="L-BFGS-B",
                         options=dict(maxiter=300))
            if best is None or r.fun < best.fun:
                best = r
        phi, E, npar, nit = phi_kernel(best.x, Phi)[0], best.fun, n_kernel(), best.nit
    s = readout(psiF, phi)
    sF = np.where(psiF >= 0, 1.0, -1.0)
    w = d["psi0"] ** 2 / (d["psi0"] @ d["psi0"])
    flipw = float(w[s != sF].sum())
    dEs = dE_sign(d["m"].H, s, d["psi0"], d["E0"], 3000)
    cosw = float((psiF ** 2 * np.abs(np.cos(phi))).sum() / (psiF @ psiF))
    return dict(E=float(E), dE_complex=float(E - d["E0"]), dEs=float(dEs), n_params=int(npar), nit=int(nit),
                flip_weight_psi0=flipw, mean_abs_cos=cosw)


def run_train(args):
    os.makedirs(OUT, exist_ok=True)
    for k in args.sizes:
        for seed in args.seeds:
            for feat in sorted({("O" if a == "rpcO" else "G") for a in args.arms}):
                d = setup(k, seed, feat)
                assert d["chkA"] < 1e-10, "CHECK A failed"
                arms = [a for a in args.arms if (a == "rpcO") == (feat == "O")]
                per = []
                for a in arms:
                    if a == "free":
                        x = 0.1 * np.random.default_rng(0).standard_normal(len(d["psiF"]))
                        per.append(time_eval(lambda: e_and_dphi(d["H"], d["psiF"], x)) * 500 * 1.3)
                    else:
                        th = init_kernel(np.random.default_rng(0))
                        per.append(time_eval(lambda: obj_kernel(th, d["H"], d["psiF"], d["Phi"])) * 600 * 1.3)
                proj = sum(per) + 20.0
                used = time.process_time()
                log(f"budget size {k} seed {seed} arms {arms}: projected {proj:.0f} CPU-s + {used:.0f} used; "
                    f"invocation cap {args.cpu_cap:.0f}; ledger {ledger_used():.0f} of {args.task_cap:.0f}")
                if proj + used > 0.9 * args.cpu_cap or ledger_used() + used + proj > args.task_cap:
                    log("BUDGET-ABORT")
                    ledger_add("train-abort", time.process_time(), f"{k} {seed} {arms}")
                    return 1
                for a in arms:
                    t0 = time.process_time()
                    res = train_one(a, d, seed)
                    fo = json.load(open(os.path.join(C.OUT, f"{k}_foil_{seed}.json")))
                    res.update(size=k, seed=seed, arm=a, gap=d["gap"], foil_dEs=fo["dEs"],
                               cpu=time.process_time() - t0)
                    json.dump(res, open(os.path.join(OUT, f"{k}_{a}_{seed}.json"), "w"), indent=1)
                    log(f"size {k} seed {seed} {a}: dEs {res['dEs']:.5f} (foil {fo['dEs']:.5f}) "
                        f"dE_complex {res['dE_complex']:.5f} flipw {res['flip_weight_psi0']:.4f} "
                        f"|cos| {res['mean_abs_cos']:.3f} nit {res['nit']} cpu {res['cpu']:.0f}")
    ledger_add("train", time.process_time(), f"sizes {args.sizes} arms {args.arms} seeds {args.seeds}")
    return 0


def med(vals):
    return float(np.median(vals)) if vals else None


def run_verdict(args):
    rows = {}
    for k in (0, 1, 2):
        foil = med([json.load(open(os.path.join(C.OUT, f"{k}_foil_{s}.json")))["dEs"] for s in (11, 22)])
        wid = med([json.load(open(os.path.join(T1E, f"A_{k}_wid_{s}.json")))["dEs"] for s in (11, 22)])
        r = dict(foil=foil, wid=wid, counted=foil >= 0.005)
        for a in ("free", "rpc", "rpcO"):
            fs = [os.path.join(OUT, f"{k}_{a}_{s}.json") for s in (11, 22)]
            if all(os.path.exists(f) for f in fs):
                r[a] = med([json.load(open(f))["dEs"] for f in fs])
                r["r_" + a] = r[a] / wid
        rows[k] = r
        log(f"size {k}: foil {foil:.5f} wid {wid:.5f} " + " ".join(
            f"{a} {r[a]:.5f} r_{a} {r['r_' + a]:.3f}" for a in ("free", "rpc", "rpcO") if a in r)
            + f" counted {r['counted']}")
    cnt = [k for k in rows if rows[k]["counted"]]
    out = {}
    if all("r_free" in rows[k] for k in cnt):
        out["PREMISE"] = "PREMISE-KILL" if all(rows[k]["r_free"] >= 0.9 for k in cnt) else "PREMISE-OK"
    if all("r_rpc" in rows[k] for k in cnt):
        rr = [rows[k]["r_rpc"] for k in cnt]
        if sum(x >= 0.9 for x in rr) >= 2 and rows[max(cnt)]["r_rpc"] >= 0.9:
            out["RPC"] = "KILL"
        elif all(x <= 0.7 for x in rr) and rows[max(cnt)]["r_rpc"] <= rows[min(cnt)]["r_rpc"] + 0.1:
            out["RPC"] = "PASS-W"
        else:
            out["RPC"] = "UNDECIDED"
        if out["RPC"] == "PASS-W" and "rpcO" in rows[max(cnt)]:
            out["ATTRIB"] = "INVERSE" if rows[max(cnt)]["rpc"] <= 0.9 * rows[max(cnt)]["rpcO"] else "NOT-INVERSE"
    log(f"VERDICT {out}")
    json.dump(dict(rows=rows, verdict=out), open(os.path.join(OUT, "verdict.json"), "w"), indent=1)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["checks", "train", "verdict"])
    p.add_argument("--sizes", default="0,1,2")
    p.add_argument("--arms", default="free,rpc")
    p.add_argument("--seeds", default="11,22")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cpu_cap", type=float, default=580.0)
    p.add_argument("--task_cap", type=float, default=1800.0)
    a = p.parse_args(argv)
    a.sizes = [int(x) for x in a.sizes.split(",")]
    a.arms = a.arms.split(",")
    a.seeds = [int(x) for x in a.seeds.split(",")]
    assert set(a.sizes) <= {0, 1, 2} and set(a.arms) <= {"free", "rpc", "rpcO"}
    return a


def main(argv=None):
    args = parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (int(args.cpu_cap), int(args.cpu_cap) + 10))
    return {"checks": run_checks, "train": run_train, "verdict": run_verdict}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
