#!/usr/bin/env python3
"""
P1.2 r11 (2026-11-17): t1c training run of the linear phase carrier on the r10 span S. Rule in RULE.md (written
before any run). Reuses t1f_phase (energy, setup, read-out), t1h_curv2 (span S, edges, edge ratios, clean, rank_w)
and t1_vote.dE_sign unchanged.

Arms (sizes 1 and 2; foil seeds 11 and 22):
  S     phi = B c, B the psi^2-centred, psi^2-whitened basis of span S; start +-0.1 v (v the lambda(S)
        eigenvector, unit psi^2-rms), L-BFGS-B maxiter 300, lower energy of the two signs kept.
  rand  the same on an equal-rank random +-1 span (r10 control), started along its own lambda eigenvector.
  Sb    sign-capacity ablation: phi = (pi/2)(1-1e-6) tanh(B c) on span S; cannot flip a sign.
  S-nn  read-out of the trained S with flips reverted on N = {n : max_e |R_e(n)| >= 10}.
Comparator: stored t1e wid JSONs (../P1.2_unanchored_antisymmetry_r6/results/t1e/A_<size>_wid_<seed>.json).

Usage
-----
  python3 t1i_phaseS.py --mode checks                 CHECK A, E[phi=0], gradient FD (size 1 seed 11)
  python3 t1i_phaseS.py --mode time                   setup + 5 timed objective evaluations per arm, per
                                                      (size, seed); prints the projected CPU-s of a full run
  python3 t1i_phaseS.py --mode train [--sizes 1,2] [--seeds 11,22] [--arms S,rand,Sb]
                                                      trains, writes results/<size>_<arm>_<seed>.json
  python3 t1i_phaseS.py --mode verdict                applies RULE.md to the JSONs present
Options:
  --mode {checks,time,train,verdict}  (required)
  --sizes LIST    subset of 1,2 (default 1,2)
  --seeds LIST    subset of 11,22 (default 11,22; only these seeds have saved t1c foils)
  --arms LIST     subset of S,rand,Sb (default S,rand,Sb; S-nn is computed with S and needs no flag)
  --cpu_cap INT   RLIMIT_CPU of this invocation in CPU-s (default 600)
One thread always (OMP/OPENBLAS/MKL = 1).
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sl  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R10 = os.path.join(os.path.dirname(HERE), "P1.2_unanchored_antisymmetry_r10")
R6 = os.path.join(os.path.dirname(HERE), "P1.2_unanchored_antisymmetry_r6")
sys.path.insert(0, R10)
import t1h_curv2 as T  # noqa: E402
P = T.P
C = P.C
from t1_vote import dE_sign  # noqa: E402

OUT = os.path.join(HERE, "results")
T1E = os.path.join(R6, "results", "t1e")
AMP_B = 0.5 * np.pi * (1.0 - 1e-6)
NN_THRESH = 10.0


def build(k, seed):
    """Foil data, span S (r10 code), the rand control, the near-node mask."""
    d = P.setup(k, seed, "G")
    m, H, psi, Phi = d["m"], d["H"], d["psiF"], d["Phi"]
    D, V = Phi.shape[0], Phi.shape[1]
    w = psi ** 2
    ratio = Phi.reshape(D, V * 5)
    prods = [Phi[:, :, i] * Phi[:, :, j] for i in range(5) for j in range(i + 1, 5)]
    hb, E = T.edges(m)
    cross = np.concatenate([(Phi[:, i, :, None] * Phi[:, j, None, :]).reshape(D, 25) for (i, j) in hb], 1)
    G = T.edge_ratios(H, psi, E)
    pairs = [(p, q) for p in range(len(E)) for q in range(p + 1, len(E)) if set(E[p][1]) & set(E[q][1])]
    EP = np.stack([G[:, p] * G[:, q] for (p, q) in pairs], 1)
    S = np.concatenate([T.clean(np.concatenate([ratio] + prods + [cross], 1), w),
                        T.clean(np.concatenate([G, EP], 1), w)], 1)
    rS = T.rank_w(S, psi)
    rand = np.random.default_rng(seed).choice([-1.0, 1.0], size=(D, rS))
    absR = np.abs(G) / np.maximum(1.0 - np.abs(G), 1e-300)
    nn = absR.max(1) >= NN_THRESH
    d.update(S=S, rand=rand, rank=rS, nn=nn)
    return d


def basis(H, psi, S):
    """psi^2-centred, whitened basis B (D, r) and the lowest curvature eigenvector v in B coordinates."""
    N = psi @ psi
    Sc = S - (psi ** 2 @ S / N)[None, :]
    Wm = (Sc * (psi ** 2)[:, None]).T @ Sc / N
    ev, U = np.linalg.eigh(Wm)
    keep = ev > 1e-10 * ev.max()
    B = Sc @ (U[:, keep] / np.sqrt(ev[keep]))
    A = sp.diags(psi) @ H @ sp.diags(psi)
    A = A - sp.diags(A.diagonal())
    deg = np.asarray(A.sum(1)).ravel()
    Q = -(B.T @ (deg[:, None] * B - A @ B)) / N
    w, X = sl.eigh(Q)
    return B, float(w[0]), X[:, 0]


def obj(c, B, H, psi, bounded):
    z = B @ c
    if bounded:
        t = np.tanh(z)
        E, g = P.e_and_dphi(H, psi, AMP_B * t)
        return E, B.T @ (g * AMP_B * (1.0 - t * t))
    E, g = P.e_and_dphi(H, psi, z)
    return E, B.T @ g


def phase(c, B, bounded):
    z = B @ c
    return AMP_B * np.tanh(z) if bounded else z


def train(B, v, H, psi, bounded):
    best = None
    for sgn in (1.0, -1.0):
        r = minimize(obj, sgn * 0.1 * v, args=(B, H, psi, bounded), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=300))
        if best is None or r.fun < best.fun:
            best = r
    return best


def score(d, phi, mask_revert=None):
    psiF = d["psiF"]
    sF = np.where(psiF >= 0, 1.0, -1.0)
    s = np.where(psiF * np.cos(phi) >= 0, 1.0, -1.0)
    if mask_revert is not None:
        s = np.where(mask_revert, sF, s)
    w0 = d["psi0"] ** 2 / (d["psi0"] @ d["psi0"])
    fl = s != sF
    return dict(dEs=float(dE_sign(d["m"].H, s, d["psi0"], d["E0"], 3000)), n_flip=int(fl.sum()),
                flip_weight_psi0=float(w0[fl].sum()), flip_weight_psi0_nn=float(w0[fl & d["nn"]].sum()))


def run_checks(args):
    d = build(1, 11)
    H, psi = d["H"], d["psiF"]
    B, lam, v = basis(H, psi, d["S"])
    E_f = float(psi @ (H @ psi) / (psi @ psi))
    chkC = abs(obj(np.zeros(B.shape[1]), B, H, psi, False)[0] - E_f)
    rng = np.random.default_rng(0)
    c = 0.1 * v + 0.01 * rng.standard_normal(B.shape[1])
    out = {}
    for bounded in (False, True):
        _, g = obj(c, B, H, psi, bounded)
        idx = rng.choice(B.shape[1], 5, replace=False)
        fd = []
        for i in idx:
            e = np.zeros_like(c)
            e[i] = 1e-6
            fd.append((obj(c + e, B, H, psi, bounded)[0] - obj(c - e, B, H, psi, bounded)[0]) / 2e-6)
        out["Sb" if bounded else "S"] = float(np.max(np.abs(np.array(fd) - g[idx])) / np.max(np.abs(g[idx])))
    res = dict(chkA=d["chkA"], chkC=chkC, fd_rel=out, lam_S=lam, rank=d["rank"], nn_count=int(d["nn"].sum()),
               valid=bool(d["chkA"] < 1e-10 and chkC < 1e-12 and max(out.values()) < 1e-5))
    P.log(f"CHECK A {d['chkA']:.2e}; E[phi=0]-E_foil {chkC:.2e}; grad FD rel {out}; lambda(S) {lam:+.4f} "
          f"(r10: -9.7317); rank {d['rank']}; |N| {res['nn_count']}; valid {res['valid']}; "
          f"CPU {time.process_time():.0f} s")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "checks.json"), "w"), indent=1)
    return 0 if res["valid"] else 1


def run_time(args):
    tot = 0.0
    for k in args.sizes:
        for seed in args.seeds:
            t0 = time.process_time()
            d = build(k, seed)
            Bs = {"S": basis(d["H"], d["psiF"], d["S"]), "rand": basis(d["H"], d["psiF"], d["rand"])}
            t_setup = time.process_time() - t0
            per = {}
            for a in args.arms:
                B = Bs["rand" if a == "rand" else "S"][0]
                c = 0.1 * Bs["rand" if a == "rand" else "S"][2]
                t1 = time.process_time()
                for _ in range(5):
                    obj(c, B, d["H"], d["psiF"], a == "Sb")
                per[a] = (time.process_time() - t1) / 5
            t2 = time.process_time()
            score(d, np.zeros(len(d["psiF"])))
            t_score = time.process_time() - t2
            proj = t_setup + sum(per[a] * 2 * 300 * 1.3 for a in per) + t_score * (len(per) + 1)
            tot += proj
            P.log(f"size {k} seed {seed}: setup {t_setup:.1f} s; per eval " +
                  " ".join(f"{a} {per[a] * 1e3:.1f} ms" for a in per) +
                  f"; dE_sign {t_score:.1f} s; projected (2 inits x 300 it x 1.3) {proj:.0f} CPU-s")
    P.log(f"PROJECTED full run {tot:.0f} CPU-s ({tot / 60:.1f} CPU-min); timing CPU {time.process_time():.0f} s")
    return 0


def run_train(args):
    os.makedirs(OUT, exist_ok=True)
    for k in args.sizes:
        for seed in args.seeds:
            d = build(k, seed)
            H, psi = d["H"], d["psiF"]
            assert d["chkA"] < 1e-10, "CHECK A failed"
            fo = json.load(open(os.path.join(C.OUT, f"{k}_foil_{seed}.json")))
            Bs = {"S": basis(H, psi, d["S"])}
            if "rand" in args.arms:
                Bs["rand"] = basis(H, psi, d["rand"])
            for a in args.arms:
                t0 = time.process_time()
                B, lam, v = Bs["rand" if a == "rand" else "S"]
                r = train(B, v, H, psi, a == "Sb")
                phi = phase(r.x, B, a == "Sb")
                res = dict(size=k, seed=seed, arm=a, n_params=int(B.shape[1]), lam=lam, E=float(r.fun),
                           dE_complex=float(r.fun - d["E0"]), nit=int(r.nit), foil_dEs=fo["dEs"],
                           mean_abs_cos=float((psi ** 2 * np.abs(np.cos(phi))).sum() / (psi @ psi)),
                           nn_count=int(d["nn"].sum()))
                res.update(score(d, phi))
                if a == "S":
                    res["nn_revert"] = score(d, phi, d["nn"])
                res["cpu"] = time.process_time() - t0
                json.dump(res, open(os.path.join(OUT, f"{k}_{a}_{seed}.json"), "w"), indent=1)
                extra = (f" | S-nn dEs {res['nn_revert']['dEs']:.5f} flips {res['nn_revert']['n_flip']}"
                         if a == "S" else "")
                P.log(f"size {k} seed {seed} {a}: dEs {res['dEs']:.5f} (foil {fo['dEs']:.5f}) dE_complex "
                      f"{res['dE_complex']:.5f} flips {res['n_flip']} (psi0 weight {res['flip_weight_psi0']:.4f}, "
                      f"near-node {res['flip_weight_psi0_nn']:.4f}) |cos| {res['mean_abs_cos']:.4f} nit {res['nit']} "
                      f"params {res['n_params']} cpu {res['cpu']:.0f}" + extra)
    P.log(f"train CPU {time.process_time():.0f} s")
    return 0


def med(x):
    return float(np.median(x))


def run_verdict(args):
    rows = {}
    for k in (1, 2):
        ld = lambda a, s: json.load(open(os.path.join(OUT, f"{k}_{a}_{s}.json")))  # noqa: E731
        wid = med([json.load(open(os.path.join(T1E, f"A_{k}_wid_{s}.json")))["dEs"] for s in (11, 22)])
        foil = med([json.load(open(os.path.join(C.OUT, f"{k}_foil_{s}.json")))["dEs"] for s in (11, 22)])
        r = dict(foil=foil, wid=wid, counted=foil >= 0.005)
        for a in ("S", "rand", "Sb"):
            if all(os.path.exists(os.path.join(OUT, f"{k}_{a}_{s}.json")) for s in (11, 22)):
                js = [ld(a, s) for s in (11, 22)]
                r[a] = med([j["dEs"] for j in js])
                r["dEc_" + a] = med([j["dE_complex"] for j in js])
                r["r_" + a] = r[a] / wid
                if a == "S":
                    r["S-nn"] = med([j["nn_revert"]["dEs"] for j in js])
                    r["r_S-nn"] = r["S-nn"] / wid
        rows[k] = r
        P.log(f"size {k}: " + " ".join(f"{x} {y:.5f}" if isinstance(y, float) else f"{x} {y}"
                                       for x, y in r.items()))
    cnt = [k for k in rows if rows[k]["counted"]]
    rr = [rows[k]["r_S"] for k in cnt]
    if all(x >= 0.9 for x in rr) and len(cnt) >= 2:
        v = "KILL"
    elif all(x <= 0.7 for x in rr) and rows[max(cnt)]["r_S"] <= rows[min(cnt)]["r_S"] + 0.1:
        v = "PASS-W"
    else:
        v = "UNDECIDED"
    out = dict(verdict=v)
    if v == "PASS-W":
        a_ = all(rows[k]["S"] <= 0.9 * rows[k]["rand"] for k in cnt)
        b_ = all(rows[k]["r_S-nn"] <= 0.7 for k in cnt)
        c_ = all(rows[k]["dEc_S"] < rows[k]["dEc_Sb"] for k in cnt)
        out.update(capacity=a_, flips_removed=b_, sign_capacity=c_,
                   credit="SIGN-CREDITED" if (a_ and b_ and c_) else "NOT CREDITED")
    P.log(f"VERDICT linear phase carrier on span S: {out}")
    json.dump(dict(rows=rows, verdict=out), open(os.path.join(OUT, "verdict.json"), "w"), indent=1)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["checks", "time", "train", "verdict"])
    p.add_argument("--sizes", default="1,2")
    p.add_argument("--seeds", default="11,22")
    p.add_argument("--arms", default="S,rand,Sb")
    p.add_argument("--cpu_cap", type=int, default=600)
    a = p.parse_args(argv)
    a.sizes = [int(x) for x in a.sizes.split(",")]
    a.seeds = [int(x) for x in a.seeds.split(",")]
    a.arms = a.arms.split(",")
    assert set(a.sizes) <= {1, 2} and set(a.seeds) <= {11, 22} and set(a.arms) <= {"S", "rand", "Sb"}
    return a


def main(argv=None):
    a = parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_cap, a.cpu_cap + 10))
    return {"checks": run_checks, "time": run_time, "train": run_train, "verdict": run_verdict}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
