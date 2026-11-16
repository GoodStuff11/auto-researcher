#!/usr/bin/env python3
"""
P3.1 round 9: tier-2 premise test shared by Routes 10.1 (SCR) and 10.2 (TMW-SCR), on the real trained states
=============================================================================================================

Premise. Both live response routes assume that a Ritz step with the (Krylov-cleaned) response of a reference
state recovers the real-space tail error of a REAL converged width-6 projected NNBF state. On the toys
(t1_r10.py, calibrated T1b) the noise-free arms track 0.80-0.93 at 4x3. This test removes training noise
(exact responses, minres) and asks the question on the shared 4x4 OBC 6+6 instance (D = 1681680), seeds
11/22/33 (P2.1 r3 parameters; the same targets as job 698163). If the noise-free arms fail here, both routes
die without any response training.

Arms (Ritz of H, exact sums; ratio = (a(Ritz vector) - a(psi)) / delta for the tails pair / spin / charge;
r = (H - E_psi) psi; K(v) = P_chi[(H - <H>_v) v]; responses y_A(phi) = (Q(H - s)Q)^{-1} Q P_chi (A - a_phi) phi,
Q = projector on chi minus |phi><phi|, s = E_phi - 2 (E_phi - E_L1R(phi)), as t1_r10.py):
  L1R       span{psi, r}                                                            (blind control)
  RWRstarK  span{psi, r, y_A(psi_0)-type: (H-E_0)^{-1} Q_0 (A - a_0) psi_0, K(.)}   (oracle ceiling)
  RWRK0     span{psi, r, y_A(psi), K(y_A(psi))}                                      (route 9.1 noise-free)
  SCR0      phi_1 = Ritz(RWRK0); span{psi, r, y_A(phi_1), K(y_A(phi_1))}             (route 10.1, noise-free)
  TMWK      span{psi, r, v_m, K(v_m)}, v_m = P_chi(O_m^dag O_m - <.>) psi, rho/S/Delta_d, all 15 OBC modes
  TMWSCR0   phi_T = Ritz(TMWK); TMWK span + y_A(phi_T) + K(y_A(phi_T))                (route 10.2, noise-free)
ORC3 on the real states is read from job 698163 (results/t2_sf_premise.json of round 8), not recomputed.

Pre-registered rule (leaf, Round 9, "T2 premise" block; written before any main run):
  Items: the real-space tails with |delta| >= 0.10 |a_0| (job 698163: pair x3, spin seeds 11/22, charge x3 = 8).
  Tracked: ratio in [1/2, 2].
  KILL both routes  if RWRstarK tracks < 0.75 of the items (the first-order lemma does not hold on the real
                    error), or if SCR0 and TMWSCR0 both track < 0.5.
  PASS 10.1         if SCR0 tracks >= 0.75 of the items and at least one spin item.
  PASS 10.2         if TMWSCR0 tracks >= 0.75 of the items, at least one spin item, and >= SCR0.
  KILL 10.1 / 10.2  separately if its arm tracks < 0.5.
  AMBIGUOUS         otherwise (hold; decide with the master after job 618646).
  Early stop (per task): if RWRstarK tracks none of that seed's scored items, the other arms are skipped and
  the seed is recorded as ceiling-fail.
  Validity: delta for every item must reproduce job 698163 within 1e-3 relative, else INVALID.

Usage
  python3 t2_resp_premise.py --instance main --seed 11 --out results/t2_resp_s11.json     (Slurm array task)
  python3 t2_resp_premise.py --aggregate results/t2_resp_s11.json,results/t2_resp_s22.json,results/t2_resp_s33.json
  python3 t2_resp_premise.py --instance small --seed 0 --out results/dry_small.json        (pre-flight dry run)
  python3 t2_resp_premise.py --selftest-verdict                                            (every verdict branch)
Options
  --instance {main, small}  main: 4x4 OBC 6+6 with the real P2.1 state of --seed.
                            small: 4x4 OBC 2+2 (D = 10920), machinery only; seed 0 is a positive-control
                            target psi_0 + 0.05 e (e a first-order error: the oracle must track it) and seed 1 a
                            negative-control target exp(0.8 xi)-distorted state (large, non-first-order error: L1R must not track); no physics claim.
  --seed INT                one target per task.
  --chunk INT               amplitude-evaluation chunk size (default 32768).
  --out PATH                per-seed JSON.
  --aggregate LIST          comma list of per-seed JSONs: print the verdict and exit (no computation).
  --selftest-verdict        synthetic records covering every verdict branch.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
R8 = os.path.join(HERE, "..", "P3.1_order_parameter_error_r8")
_spec = importlib.util.spec_from_file_location("t2sf", os.path.join(R8, "t2_sf_premise.py"))
t2sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t2sf)
r6, rc, log, P21 = t2sf.r6, t2sf.rc, t2sf.log, t2sf.P21
REAL = ("pair", "spin", "charge")
REF = os.path.join(R8, "results", "t2_sf_premise.json")
ARMS = ("L1R", "RWRstarK", "RWRK0", "SCR0", "TMWK", "TMWSCR0")


def unit(v):
    return v / np.linalg.norm(v)


class Ctx:
    def __init__(self, m, sg, ob):
        self.m, self.sg, self.ob = m, sg, ob

    def apply(self, k, v):
        return self.ob.apply(k, v)

    def K(self, v):
        v = unit(v)
        w = self.m.H @ v
        return self.sg.project(w - (v @ w) * v)

    def solve(self, rhs, shift, orth):
        m, sg = self.m, self.sg

        def q(v):
            v = sg.project(v)
            return v - orth * (orth @ v)

        def op(v):
            x = q(v)
            return q(m.H @ x - shift * x) + (v - x)
        L = sla.LinearOperator((m.D, m.D), matvec=op, dtype=float)
        b = q(rhs)
        x, info = sla.minres(L, b, rtol=1e-9, maxiter=8000)
        res = np.linalg.norm(op(x) - b) / np.linalg.norm(b)
        assert res < 1e-6, ("minres", info, res)
        return q(x)

    def ritz_vec(self, vecs):
        B = np.array([unit(v) for v in vecs])
        S = B @ B.T
        H = B @ np.array([self.m.H @ b for b in B]).T
        S, H = 0.5 * (S + S.T), 0.5 * (H + H.T)
        lam, V = np.linalg.eigh(S)
        keep = lam > 1e-11 * lam.max()
        X = V[:, keep] / np.sqrt(lam[keep])
        ev, Y = np.linalg.eigh(X.T @ H @ X)
        return unit((X @ Y[:, 0]) @ B), float(ev[0])

    def responses(self, phi):
        phi = unit(phi)
        E = float(phi @ (self.m.H @ phi))
        r = self.m.H @ phi - E * phi
        _, EL1 = self.ritz_vec([phi, r])
        s = E - 2.0 * (E - EL1)
        out = []
        for k in REAL:
            a = float(phi @ self.apply(k, phi))
            out.append(unit(self.solve(self.apply(k, phi) - a * phi, s, phi)))
        return out

    def two_mode_vectors(self, psi):
        """P_chi(O_m^dag O_m - <.>) psi for rho, S, Delta_d standing waves, all modes m != (0,0), on the fly."""
        m, ob = self.m, self.ob
        modes = [(a, b) for a in range(m.Lx) for b in range(m.Ly) if (a, b) != (0, 0)]

        def f(mode, xy):
            return (math.cos(math.pi * mode[0] * (xy[0] + 0.5) / m.Lx)
                    * math.cos(math.pi * mode[1] * (xy[1] + 0.5) / m.Ly))
        nu = np.array([(m.bu >> i) & 1 for i in range(m.V)], dtype=float)
        nd = np.array([(m.bd >> i) & 1 for i in range(m.V)], dtype=float)
        szs = 0.5 * (nu - nd)
        eta = np.array([1.0 if b[2] == "x" else -1.0 for b in m.pair_bonds])
        cen = [m.bond_center(b) for b in m.pair_bonds]
        exv = {}
        for i in range(m.V):
            for j in range(i + 1, m.V):
                exv[(i, j)] = ob._exchange([(i, j)]) @ psi
        onsite = [(0.75 * (ob.n_site_occ[i] % 2) - szs[i] ** 2) for i in range(m.V)]
        vecs = []
        for mode in modes:
            fs = np.array([f(mode, m.coords[i]) for i in range(m.V)])
            rho = fs @ (ob.n_site_occ - ob.nbar)
            vs = [rho ** 2 * psi]
            sv = (fs @ szs) ** 2 * psi + sum(fs[i] ** 2 * onsite[i] for i in range(m.V)) * psi
            sv = sv + sum(2 * fs[i] * fs[j] * w for (i, j), w in exv.items())
            vs.append(sv)
            fb = eta * np.array([f(mode, c) for c in cen])
            vs.append(ob.apply_pieces_all("pair", psi, [np.outer(fb, fb)])[0])
            for v in vs:
                v = self.sg.project(v - (psi @ v) * psi)
                if np.linalg.norm(v) > 1e-10:
                    vecs.append(v)
        return vecs


def tracked(x):
    return x is not None and 0.5 <= x <= 2.0


def verdict(recs):
    """recs: list of per-seed dicts {obs: {scored, ratios{arm: x}}}. Returns (verdict dict, details)."""
    items = [(r["seed"], k, r[k]) for r in recs for k in REAL if r[k]["scored"]]
    n = len(items)
    if n == 0:
        return {"all": "NO-TARGET"}, {}
    fr = {a: sum(1 for _, _, d in items if tracked(d["ratios"].get(a))) / n for a in ARMS}
    spin = {a: sum(1 for _, k, d in items if k == "spin" and tracked(d["ratios"].get(a))) for a in ARMS}
    v = {}
    if fr["RWRstarK"] < 0.75 or (fr["SCR0"] < 0.5 and fr["TMWSCR0"] < 0.5):
        v["10.1"] = v["10.2"] = "KILL"
    else:
        v["10.1"] = ("PASS" if fr["SCR0"] >= 0.75 and spin["SCR0"] >= 1
                     else "KILL" if fr["SCR0"] < 0.5 else "AMBIGUOUS")
        v["10.2"] = ("PASS" if fr["TMWSCR0"] >= 0.75 and spin["TMWSCR0"] >= 1 and fr["TMWSCR0"] >= fr["SCR0"]
                     else "KILL" if fr["TMWSCR0"] < 0.5 else "AMBIGUOUS")
    return v, dict(n_items=n, tracked=fr, spin_tracked=spin)


def selftest():
    def rec(seed, vals, scored=(True, True, True)):
        return dict(seed=seed, **{k: dict(scored=s, ratios=dict(zip(ARMS, vals[i])))
                                  for i, (k, s) in enumerate(zip(REAL, scored))})
    good = (0.0, 1.0, 0.9, 1.0, 0.6, 1.1)
    bad = (0.0, 1.0, 0.1, 0.2, 0.1, 0.1)
    gnt = (0.0, 1.0, 0.9, 1.0, 0.6, 0.1)   # SCR0 tracked, TMWSCR0 not
    cases = {
        "PASS/PASS": [rec(11, [good] * 3), rec(22, [good] * 3)],
        "KILL/KILL (ceiling)": [rec(11, [(0, 0.1, 1, 1, 1, 1)] * 3), rec(22, [(0, 0.1, 1, 1, 1, 1)] * 3)],
        "KILL/KILL (both arms)": [rec(11, [bad] * 3), rec(22, [bad] * 3)],
        "AMBIGUOUS/KILL": [rec(11, [gnt, bad, bad]), rec(22, [gnt, gnt, bad])],
    }
    want = {"PASS/PASS": ("PASS", "PASS"), "KILL/KILL (ceiling)": ("KILL", "KILL"),
            "KILL/KILL (both arms)": ("KILL", "KILL"), "AMBIGUOUS/KILL": ("AMBIGUOUS", "KILL")}
    ok = True
    for lab, recs in cases.items():
        v, det = verdict(recs)
        got = (v["10.1"], v["10.2"])
        ok &= got == want[lab]
        print(f"selftest {lab:24s} expected {want[lab]} got {got} {det['tracked']}")
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def run(args):
    t0 = time.time()
    m, E0, psi0, sg, S0, ob = r6.setup(args.instance)
    psi0 = psi0 if psi0.sum() > 0 else -psi0
    if args.instance == "main":
        assert m.D == 1681680 and S0 == 0, (m.D, S0)
    C = Ctx(m, sg, ob)
    a0 = {k: float(psi0 @ C.apply(k, psi0)) for k in REAL}
    # ------------------------------------------------ target
    if args.instance == "main":
        f = os.path.join(P21, f"inst0_nnbf_s{args.seed}.npz")
        assert os.path.exists(f), f
        ans = rc.load_params(rc.nnbf(m, 6, args.seed), f)
        psi = unit(sg.project(ans.amplitudes(args.chunk)))
    else:
        rng = np.random.default_rng(5)
        if args.seed == 0:   # positive control: small first-order error built from a wrong-Hamiltonian response
            w = sg.project(C.apply("spin", psi0) + C.apply("charge", psi0))
            w -= psi0 * (psi0 @ w)
            e = unit(-C.solve(w, E0, psi0))
            psi = unit(psi0 + 0.05 * e)
        else:                # negative control: random multiplicative distortion
            psi = unit(sg.project(psi0 * np.exp(0.8 * rng.standard_normal(m.D))))
    psi = psi if psi @ psi0 > 0 else -psi
    a_psi = {k: float(psi @ C.apply(k, psi)) for k in REAL}
    delta = {k: a0[k] - a_psi[k] for k in REAL}
    out = dict(args=vars(args), seed=args.seed, E0=E0, a0=a0, overlap2=float((psi @ psi0) ** 2),
               dE=float(psi @ (m.H @ psi)) - E0)
    for k in REAL:
        out[k] = dict(a_psi=a_psi[k], delta=delta[k], scored=bool(abs(delta[k]) >= 0.10 * abs(a0[k])), ratios={})
    log(f"seed {args.seed}: overlap^2 {out['overlap2']:.4f} dE {out['dE']:.5f} | "
        + " ".join(f"{k} delta {delta[k]:+.3e} (rel {delta[k] / a0[k]:+.3f})" for k in REAL))
    # validity against job 698163
    if args.instance == "main" and os.path.exists(REF):
        ref = json.load(open(REF))["seeds"].get(str(args.seed))
        if ref:
            dev = {k: abs(delta[k] - ref[k]["delta"]) / abs(ref[k]["delta"]) for k in REAL}
            out["valid"] = all(x <= 1e-3 for x in dev.values())
            out["ORC3_698163"] = {k: ref[k]["ORC"][2] / 1.0 for k in REAL}
            log(f"validity vs job 698163: rel dev {dev} -> {'OK' if out['valid'] else 'INVALID'}")
    E = float(psi @ (m.H @ psi))
    base = [psi, m.H @ psi - E * psi]

    def score(arm, phi):
        for k in REAL:
            out[k]["ratios"][arm] = float(phi @ C.apply(k, phi) - a_psi[k]) / delta[k]
        log(f"  {arm:9s} " + " ".join(f"{k} {out[k]['ratios'][arm]:+.3f}" for k in REAL)
            + f"  ({time.time() - t0:.0f} s)")
        json.dump(out, open(args.out, "w"), indent=1, default=float)
    score("L1R", C.ritz_vec(base)[0])
    ystar = [C.solve(C.apply(k, psi0) - a0[k] * psi0, E0, psi0) for k in REAL]
    score("RWRstarK", C.ritz_vec(base + ystar + [C.K(v) for v in ystar])[0])
    sc = [k for k in REAL if out[k]["scored"]]
    if sc and not any(tracked(out[k]["ratios"]["RWRstarK"]) for k in sc):
        out["early_stop"] = "ceiling-fail"
        log("EARLY STOP: the oracle-response ceiling tracks none of this seed's scored items")
        json.dump(out, open(args.out, "w"), indent=1, default=float)
        return 0
    y = C.responses(psi)
    phi1 = C.ritz_vec(base + y + [C.K(v) for v in y])[0]
    score("RWRK0", phi1)
    y1 = C.responses(phi1)
    score("SCR0", C.ritz_vec(base + y1 + [C.K(v) for v in y1])[0])
    tm = C.two_mode_vectors(psi)
    tmk = base + tm + [C.K(v) for v in tm]
    phiT = C.ritz_vec(tmk)[0]
    score("TMWK", phiT)
    yT = C.responses(phiT)
    score("TMWSCR0", C.ritz_vec(tmk + yT + [C.K(v) for v in yT])[0])
    out["n_tmw"] = len(tm)
    log(f"seed {args.seed} done ({time.time() - t0:.0f} s), {len(tm)} two-mode vectors")
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    return 0


def aggregate(paths):
    recs = []
    for p in paths.split(","):
        if not os.path.exists(p):
            log(f"missing {p}")
            continue
        r = json.load(open(p))
        if r.get("valid") is False:
            log(f"INVALID: {p} does not reproduce job 698163")
            return 1
        recs.append(r)
        log(f"seed {r['seed']}: " + " | ".join(
            f"{k}{'' if r[k]['scored'] else '(ns)'} " + " ".join(f"{a} {x:+.2f}" for a, x in r[k]["ratios"].items())
            for k in REAL) + (f" [{r['early_stop']}]" if r.get("early_stop") else ""))
    v, det = verdict(recs)
    log(f"T2 PREMISE VERDICT: {v} | {det}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", choices=("main", "small"), default="main")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--out", default=None)
    p.add_argument("--aggregate", default=None)
    p.add_argument("--selftest-verdict", action="store_true")
    a = p.parse_args(argv)
    if a.selftest_verdict:
        return selftest()
    if a.aggregate:
        return aggregate(a.aggregate)
    a.out = a.out or os.path.join(HERE, "results", f"t2_resp_s{a.seed}.json")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
