#!/usr/bin/env python3
"""
P3.1 Round 12: T2 premise run for Route 12.3d (Davidson-block bracket) on the real 4x4 states. PREPARED, NOT SUBMITTED.
=======================================================================================================================

Instance and targets as the T2-v2 run (job 699756): 4x4 OBC 6+6 (D = 1681680), the real width-6 projected NNBF
states of seeds 11, 22, 33; ED, targets, validity check against job 698163, Ctx machinery (sector projector,
minres, observables) imported unchanged from ../P3.1_order_parameter_error_r10/t2v2_scru_premise.py (class Run).

Arms per seed (exact sums; intervals = eigenvalue range of A compressed onto the two lowest Ritz vectors):
  L1R block   span{psi, (H-E)psi}                                                         Gate 2 foil
  DAV-R block 10.4 chain (sources Q(H-E)^p phi, p = 1..3; T = 4; shift E_L1R - 0.1 t) with grade-R noise on every
              generated vector (0.3 in-window over the in-sector levels with omega <= 1 t, 0.5 junk), then the
              witness pass span{phi_4, (H-E)phi_4, y~_p, K y~_p}                          deciding
  DAV-0 block the same without noise                                                      reported, if budget allows
Levels: the lowest --n-low in-sector levels (r6.sector_eigs); psi_1 = the lowest excited level gives w_ex, the
eigenvalue range of A on span{psi_0, psi_1}; the window levels carry the in-window noise. Junk: x = P(base * xi),
j = P((H - E_base) x) with psi_0 and the computed levels removed (t1_portfolio.Inst.junk).

Scored items: |delta| >= 0.10 |a_0| (job 698163: 8 items). Covered: a_0 in [lo - 0.1|delta|, hi + 0.1|delta|].
Width ratio: (hi - lo) / w_ex.
Pre-registered rule (leaf, Round 12, "T2 premise run for 12.3d"):
  INVALID        delta does not reproduce job 698163 within 1e-3 relative, or the window is not covered
  KILL (Gate 2)  the L1R block covers >= 7 of 8 with median width ratio <= 1.5
  PASS           DAV-R covers >= 7 of 8 and its median width ratio is <= 1.5
  KILL           DAV-R covers < 5 of 8, or its median width ratio > 3
  AMBIGUOUS      otherwise
Early stop: after each seed, if DAV-R can no longer reach 5 covered items, stop with KILL.

Usage
  python3 t2_block_premise.py --instance main --out results/t2_block_main.json     (the Slurm task)
  python3 t2_block_premise.py --instance small --out results/t2_block_dry.json     (pre-flight dry run)
  python3 t2_block_premise.py --aggregate results/t2_block_main.json               (verdict from the JSON)
  python3 t2_block_premise.py --selftest-verdict                                   (every verdict branch)
Options
  --instance {main, small}  main: the real states; small: 4x4 OBC 2+2 (D = 10920), "seed" 0 = positive control
                            psi_0 + 0.05 e, "seed" 1 = random distortion (machinery only, no validity check)
  --n-low INT               in-sector levels to compute (default 16; must reach omega > 1 t, else INVALID)
  --window-all              accept a window that ends below 1 t: the in-window noise is spread over every computed
                            level (the real 4x4 instance needs about 45 levels for 1 t; 31 levels reach 0.84 t)
  --chunk INT               amplitude-evaluation chunk size (default 32768)
  --cpu-budget FLOAT        CPU-seconds; DAV-0 starts only if it projects to fit (default 6600)
  --out PATH                JSON output, rewritten after every arm
  --aggregate PATH          print the verdict from an output JSON and exit
  --selftest-verdict        synthetic records for every verdict branch
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "t2v2", os.path.join(HERE, "..", "P3.1_order_parameter_error_r10", "t2v2_scru_premise.py"))
t2v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t2v2)
np, r6, log, unit, cpu, REAL, REF = t2v2.np, t2v2.r6, t2v2.log, t2v2.unit, t2v2.cpu, t2v2.REAL, t2v2.REF
ETA = (0.3, 0.5)


def ritz_block(m, vecs):
    B = np.array([unit(v) for v in vecs])
    S = B @ B.T; S = 0.5 * (S + S.T)
    H = B @ np.array([m.H @ b for b in B]).T; H = 0.5 * (H + H.T)
    lam, V = np.linalg.eigh(S)
    keep = lam > 1e-11 * lam.max()
    X = V[:, keep] / np.sqrt(lam[keep])
    ev, Y = np.linalg.eigh(X.T @ H @ X)
    C = X @ Y
    return ev, [unit(C[:, k] @ B) for k in range(2)]


def bracket(C, k, vecs):
    AV = [C.apply(k, v) for v in vecs]
    M = np.array([[float(a @ b) for b in AV] for a in vecs])
    ev = np.linalg.eigvalsh(0.5 * (M + M.T))
    return float(ev[0]), float(ev[-1])


def count(rec, arm):
    it = [(s, k) for s, d in rec.items() for k in REAL if d[k]["scored"] and arm in d[k]["br"]]
    cov = sum(1 for s, k in it if rec[s][k]["br"][arm][0] - 0.1 * abs(rec[s][k]["delta"]) <= rec[s][k]["a0"]
              <= rec[s][k]["br"][arm][1] + 0.1 * abs(rec[s][k]["delta"]))
    wr = [(rec[s][k]["br"][arm][1] - rec[s][k]["br"][arm][0]) / rec[s][k]["wex"] for s, k in it]
    return cov, (float(np.median(wr)) if wr else float("nan")), len(it)


def verdict(rec):
    cl, wl, _ = count(rec, "L1R")
    cd, wd, nd = count(rec, "DAV-R")
    det = dict(L1R=(cl, wl), DAVR=(cd, wd, nd))
    if cl >= 7 and wl <= 1.5:
        return "KILL (Gate 2)", det
    if cd >= 7 and wd <= 1.5:
        return "PASS", det
    if cd < 5 or wd > 3:
        return "KILL", det
    return "AMBIGUOUS", det


def selftest():
    def rec(cov_l, w_l, cov_d, w_d, n=8):
        out = {}
        for i in range(n):
            s, k = str(i // 3), REAL[i % 3]
            d = out.setdefault(s, {kk: dict(scored=False, br={}, delta=1.0, a0=0.0, wex=1.0) for kk in REAL})
            d[k] = dict(scored=True, delta=1.0, a0=0.0, wex=1.0,
                        br={"L1R": (-w_l / 2, w_l / 2) if i < cov_l else (5.0, 5.0 + w_l),
                            "DAV-R": (-w_d / 2, w_d / 2) if i < cov_d else (5.0, 5.0 + w_d)})
        return out
    cases = [("KILL (Gate 2)", rec(8, 1.0, 8, 1.0)), ("PASS", rec(4, 1.0, 8, 1.2)), ("KILL", rec(4, 1.0, 4, 1.0)),
             ("KILL", rec(4, 1.0, 8, 3.5)), ("AMBIGUOUS", rec(4, 1.0, 6, 1.0)), ("AMBIGUOUS", rec(4, 1.0, 8, 2.0))]
    ok = True
    for want, r in cases:
        got = verdict(r)[0]
        ok &= got == want
        print(f"  selftest want {want:14s} got {got:14s} {'ok' if got == want else 'MISMATCH'}")
    print("SELFTEST " + ("PASS" if ok else "FAIL"))


class BlockRun(t2v2.Run):
    def levels(self):
        m, sg = self.m, self.sg
        S0 = float(round(0.5 * (np.sqrt(1 + 4 * m.S2_expect(self.psi0)) - 1)))
        e, U = r6.sector_eigs(m, sg, S0, self.a.n_low)
        assert abs(e[0] - self.E0) < 1e-6, (e[0], self.E0)
        om = e[1:] - self.E0
        U = U[:, 1:]
        log(f"{len(om)} in-sector excited levels: omega = {np.round(om, 4).tolist()} (cpu {cpu():.0f} s)")
        if om[-1] <= 1.0 and not self.a.window_all:
            return None
        # --window-all: the window is every computed level (on the real 4x4 instance 30 levels reach only 0.84 t)
        self.om, self.U, self.Uw = om, U, U[:, om <= 1.0]
        self.out["window_top_omega"] = float(om[om <= 1.0][-1])
        self.out["omega"] = om.tolist()
        return True

    def junk(self, rng, base):
        m = self.m
        x = self.sg.project(base * rng.standard_normal(m.D))
        j = self.sg.project(m.H @ x - float(base @ (m.H @ base)) * x)
        j -= self.U @ (self.U.T @ j)
        j -= self.psi0 * (self.psi0 @ j)
        return unit(j)

    def noisy(self, y, rng, base):
        if rng is None:
            return unit(y)
        g = unit(self.Uw @ rng.standard_normal(self.Uw.shape[1]))
        return unit(unit(y) + ETA[0] * g + ETA[1] * self.junk(rng, base))

    def sources(self, phi):
        m, out, v = self.m, [], phi
        E = float(phi @ (m.H @ phi))
        for _ in range(3):
            v = self.sg.project(m.H @ v - E * v)
            out.append(v - phi * (phi @ v))
        return out

    def ys(self, phi, rng):
        C, m = self.C, self.m
        E = float(phi @ (m.H @ phi))
        _, EL1 = C.ritz_vec([phi, m.H @ phi - E * phi])
        s = EL1 - 0.1
        assert s < self.E0, ("VALIDITY: shift not below E0", s, self.E0)
        return [self.noisy(C.solve(b, s, phi), rng, phi) for b in self.sources(phi)]

    def chain_block(self, psi, rng):
        C, m = self.C, self.m
        phi = unit(psi)
        for t in range(1, 5):
            ys = self.ys(phi, rng)
            if t == 1:
                phi = C.ritz_vec([phi, m.H @ phi - float(phi @ (m.H @ phi)) * phi] + ys + [C.K(v) for v in ys])[0]
            else:
                phi = C.ritz_vec([phi] + ys)[0]
            phi = unit(phi)
        ys = self.ys(phi, rng)
        ev, blk = ritz_block(m, [phi, m.H @ phi - float(phi @ (m.H @ phi)) * phi] + ys + [C.K(v) for v in ys])
        return blk, float(ev[1] - ev[0])

    def run(self):
        a, C, m = self.a, self.C, self.m
        seeds = t2v2.SEEDS[a.instance]
        if self.levels() is None:
            self.out["verdict"] = "INVALID (window omega <= 1 t not covered; raise --n-low)"
            self.dump(); return 1
        plane = [self.psi0, self.U[:, 0]]
        wex = {k: (lambda b: b[1] - b[0])(bracket(C, k, plane)) for k in REAL}
        ref = json.load(open(REF))["seeds"] if a.instance == "main" else {}
        for s in seeds:
            psi = self.target(s)
            self.psi[s] = psi
            ap = {k: float(psi @ C.apply(k, psi)) for k in REAL}
            dl = {k: self.a0[k] - ap[k] for k in REAL}
            self.out["seeds"][str(s)] = {k: dict(a_psi=ap[k], delta=dl[k], a0=self.a0[k], wex=wex[k], br={},
                                                 scored=bool(abs(dl[k]) >= 0.10 * abs(self.a0[k]))) for k in REAL}
            if a.instance == "main":
                dev = {k: abs(dl[k] - ref[str(s)][k]["delta"]) / abs(ref[str(s)][k]["delta"]) for k in REAL}
                log(f"VALIDITY seed {s} vs job 698163: rel dev {dev}")
                if not all(x <= 1e-3 for x in dev.values()):
                    self.out["verdict"] = "INVALID"
                    self.dump(); return 1
        n_sc = sum(1 for s in seeds for k in REAL if self.out["seeds"][str(s)][k]["scored"])
        log(f"scored items: {n_sc}")
        cov_so_far, left = 0, n_sc
        per = 0.0
        for s in seeds:
            c0 = cpu()
            d, psi = self.out["seeds"][str(s)], self.psi[s]
            L = [unit(psi), unit(m.H @ psi - float(psi @ (m.H @ psi)) * psi)]
            for k in REAL:
                d[k]["br"]["L1R"] = bracket(C, k, L)
            try:
                blk, g = self.chain_block(psi, np.random.default_rng(50_000 + s))
            except AssertionError as err:      # shift not below E0: the functional is unbounded (validity)
                d["invalid"] = str(err)
                log(f"seed {s}: INVALID ({err})")
                if a.instance == "main":
                    self.out["verdict"] = "INVALID"
                    self.dump(); return 1
                continue
            d["ghat_DAV-R"] = g
            for k in REAL:
                d[k]["br"]["DAV-R"] = bracket(C, k, blk)
            per = max(per, cpu() - c0)
            self.dump()
            log(f"seed {s}: g-hat {g:.4f} | " + " | ".join(
                f"{k}{'' if d[k]['scored'] else '(ns)'} delta {d[k]['delta']:+.3e} L1R [{d[k]['br']['L1R'][0]:+.3e},"
                f"{d[k]['br']['L1R'][1]:+.3e}] DAV-R [{d[k]['br']['DAV-R'][0]:+.3e},{d[k]['br']['DAV-R'][1]:+.3e}] "
                f"a0 {d[k]['a0']:+.3e} w_ex {d[k]['wex']:.3e}" for k in REAL) + f" (cpu {cpu() - c0:.0f} s)")
            sc = [k for k in REAL if d[k]["scored"]]
            left -= len(sc)
            lo = lambda k: d[k]["br"]["DAV-R"][0] - 0.1 * abs(d[k]["delta"])  # noqa: E731
            hi = lambda k: d[k]["br"]["DAV-R"][1] + 0.1 * abs(d[k]["delta"])  # noqa: E731
            cov_so_far += sum(1 for k in sc if lo(k) <= d[k]["a0"] <= hi(k))
            if a.instance == "main" and cov_so_far + left < 5:
                log("EARLY STOP: DAV-R can no longer cover 5 of 8 -> KILL")
                break
        for s in seeds:
            if cpu() + per > a.cpu_budget:
                log(f"soft budget: DAV-0 skipped from seed {s}"); break
            d = self.out["seeds"][str(s)]
            if "invalid" in d:
                continue
            blk, g = self.chain_block(self.psi[s], None)
            d["ghat_DAV-0"] = g
            for k in REAL:
                d[k]["br"]["DAV-0"] = bracket(C, k, blk)
            self.dump()
        v, det = verdict(self.rec())
        self.out["verdict"], self.out["verdict_details"] = v, det
        log(f"T2 12.3d VERDICT: {v} | {det} | cpu {cpu():.0f} s")
        self.dump()
        return 0

    def rec(self):
        return {s: {k: d[k] for k in REAL} for s, d in self.out["seeds"].items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", choices=("main", "small"), default="main")
    p.add_argument("--n-low", type=int, default=16)
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--window-all", action="store_true")
    p.add_argument("--cpu-budget", type=float, default=6600.0)
    p.add_argument("--out", default=None)
    p.add_argument("--aggregate", default=None)
    p.add_argument("--selftest-verdict", action="store_true")
    a = p.parse_args()
    if a.selftest_verdict:
        return selftest()
    if a.aggregate:
        out = json.load(open(a.aggregate))
        rec = {s: {k: d[k] for k in REAL} for s, d in out["seeds"].items()}
        print(verdict(rec))
        return
    assert a.out, "--out is required"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sys.exit(BlockRun(a).run())


if __name__ == "__main__":
    main()
