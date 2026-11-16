#!/usr/bin/env python3
"""
P2.1 Round 13, T1 accuracy: sign capacity of the hole-gated (hg) and checkerboard-plus-holes (cbh) variants.
============================================================================================================

Pre-registered in the leaf ("Round 13 ... Pre-registered T1 rules"). The Round 12 protocol is used unchanged:
t1_sub.train (imported read-only from ../P2.1_subcubic_amplitudes_r12; nothing there is edited) with this module's
ansatz injected at run time into t1_sub.make / t1_sub.RNG_OFF of this process only.

Ansatz: t2_routes.HFPSAnsatz (plus patch, F = 4, Ha = 0, pairing reference F0 = round-5 projected d-wave BCS,
two-body density Jastrow) with the hidden features gated per configuration:
    psi = e^J det(F0 + sum_m K[:, m] g_m(n) h_m(n) . Omega_m)[up, dn]
    hg : g_m = 1 - n_m             (hidden term only on empty sites)
    cbh: g_m = 1 on the checkerboard (x + y even); g_m = 1 - n_m on the other sublattice
All parameters stay trainable (n_eff = the full_F4 count). g is a 0/1 constant per configuration, so the
analytic gradient is the HFPSAnsatz gradient with h -> g h and d(pre) -> g (1 - h^2) dh.

Scored (rule in the leaf): rho = dE_sign(route) / dE_sign(full_F4) on sizes 20 and 23; references from
../P2.1_subcubic_amplitudes_r12/results (size 20) and ../P2.1_subcubic_amplitudes_r12b/results (size 23).
KILL if rho > 1.25 at both sizes; cbh also KILL if rho(cbh) >= rho(half) at both sizes; T1-PASS if both sizes count
and no kill clause fires; INCONCLUSIVE if fewer than 2 sizes count.

Modes
  --mode check      (i) amplitudes == an explicit site-by-site gated det build (rel <= 1e-12), instances 20, 23;
                    (ii) analytic chi-projected gradient == central finite differences (lt.fd_check), instance 20;
                    (iii) with all g = 1 forced, the amplitude equals HFPSAnsatz bit for bit. Exit 2 on failure.
  --mode train      --arm {hg,cbh} --instance {20,23} [--max-cpu 480]: t1_sub.train with the Round 12 options;
                    writes results/t1_<arm>_inst<I>_s11.json (refuses to overwrite).
  --mode probe      --arm --instance --probe-steps N: CPU-s per SR step, no output file.
  --mode aggregate  the pre-registered verdicts.
  --mode synthetic  the verdict function on synthetic inputs covering every branch.
Threads: OMP/OPENBLAS/MKL_NUM_THREADS=1; `ulimit -t 600` per run.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
R12 = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r12")
R12B = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r12b")
sys.path.insert(0, R12)
import t1_sub as TS  # noqa: E402  (read-only)
from lh_geom import patch_input  # noqa: E402

T2 = TS.T2
mt = TS.mt
log = TS.log
GATED = ("hg", "cbh")
HALF_RHO = {20: 2.439, 23: 1.587}     # Round 12 / 704752


class GatedAnsatz(TS.SubHFPSAnsatz):
    def __init__(self, model, F0, F, Ha, mode, seed=0):
        super().__init__(model, F0, F, Ha, "full", seed=seed)
        self.gmode = mode
        xs, ys = np.arange(model.V) % model.Lx, np.arange(model.V) // model.Lx
        cb = (xs + ys) % 2 == 0
        self.static = {"hg": np.zeros(model.V, bool), "cbh": cb, "ones": np.ones(model.V, bool)}[mode]

    def gate(self, ou, od):
        return np.where(self.static[None], 1.0, 1.0 - (ou + od)).astype(float)    # (B, V)

    def forward(self, ou, od, ru, rd):
        p, V = self.p, self.V
        B, N = ou.shape[0], ru.shape[1]
        xp = patch_input(ou, od, self.P)
        hr = np.tanh(xp @ p["Wl"] + p["bl"])
        g = self.gate(ou, od)
        h = hr * g[:, :, None]
        C = np.swapaxes(np.matmul(np.swapaxes(h, 0, 1), p["Om"]), 0, 1)
        Crd = np.take_along_axis(C, np.broadcast_to(rd[:, None, :], (B, V, N)), 2)
        Kr = p["K"][ru]
        A = p["F0"][ru[:, :, None], rd[:, None, :]] + Kr @ Crd
        det = np.linalg.det(A)
        J, Q, x, ha = self.jastrow(ou, od)
        return dict(xp=xp, h=h, hr=hr, g=g, ru=ru, rd=rd, B=B, Crd=Crd, Kr=Kr, A=A, Q=Q, x=x, ha=ha, J=J, S=det,
                    psi=np.exp(J) * det)

    def grad(self, c, coef, per_sample):
        # T2.HFPSAnsatz.grad with h -> g h (stored in c["h"]) and dpre -> g (1 - hr^2) dh
        p, B, V = self.p, c["B"], self.V
        ru, rd = c["ru"], c["rd"]
        bi = np.arange(B)[:, None]
        red = (lambda a: a) if per_sample else (lambda a: a.sum(0))
        tk = coef * c["psi"]
        G = tk[:, None, None] * self._invT(c["A"])
        gK = G @ np.swapaxes(c["Crd"], 1, 2)
        dCrd = np.swapaxes(c["Kr"], 1, 2) @ G
        dC = np.zeros((B, V, V))
        dC[bi, :, rd] = np.swapaxes(dCrd, 1, 2)
        h = c["h"]
        g = {}
        if per_sample:
            Z = np.zeros((B, V, V)); Z[bi[:, :, None], ru[:, :, None], rd[:, None, :]] = G
            ZK = np.zeros((B, V, V)); ZK[bi, ru] = gK
            g["F0"], g["K"] = Z, ZK
            g["Om"] = np.einsum("bmf,bmv->bmfv", h, dC)
        else:
            fi = (ru[:, :, None] * V + rd[:, None, :]).ravel()
            g["F0"] = np.bincount(fi, weights=G.ravel(), minlength=V * V).reshape(V, V)
            Sc = sp.csr_matrix((np.ones(ru.size), (ru.ravel(), np.arange(ru.size))), shape=(V, ru.size))
            g["K"] = Sc @ gK.reshape(-1, V)
            g["Om"] = np.matmul(np.swapaxes(np.swapaxes(h, 0, 1), 1, 2), np.swapaxes(dC, 0, 1))
        dh = np.swapaxes(np.matmul(np.swapaxes(dC, 0, 1), np.swapaxes(p["Om"], 1, 2)), 0, 1)
        dpre = dh * (1 - c["hr"] ** 2) * c["g"][:, :, None]
        if per_sample:
            g["Wl"] = np.einsum("bvi,bvf->bif", c["xp"], dpre)
            g["bl"] = dpre.sum(1)
        else:
            g["Wl"] = np.einsum("bvi,bvf->if", c["xp"], dpre)
            g["bl"] = dpre.sum((0, 1))
        g["vJ"] = tk[:, None] * c["Q"] if per_sample else c["Q"].T @ tk
        if self.Ha > 0:
            x, ha = c["x"], c["ha"]
            dprea = tk[:, None] * p["w2a"][None] * (1 - ha ** 2)
            g["W1a"] = (x[:, :, None] * dprea[:, None, :]) if per_sample else x.T @ dprea
            g["b1a"] = red(dprea)
            g["w2a"] = red(tk[:, None] * ha)
        if per_sample:
            return np.concatenate([g[k].reshape(B, -1) for k in self.keys], 1)
        return np.concatenate([g[k].ravel() for k in self.keys])


def explicit_amp(a, ou, od):
    p = a.p
    up, dn = np.nonzero(ou)[0], np.nonzero(od)[0]
    A = p["F0"][np.ix_(up, dn)].copy()
    for mm in range(a.V):
        if not (a.static[mm] or ou[mm] + od[mm] == 0):
            continue
        xp = patch_input(ou[None], od[None], a.P[mm:mm + 1])[0, 0]
        h = np.tanh(xp @ p["Wl"] + p["bl"])
        A += np.outer(p["K"][up, mm], h @ p["Om"][mm][:, dn])
    nt = (ou + od).astype(float)
    J = sum(p["vJ"][k] * nt[i] * nt[j] for k, (i, j) in enumerate(zip(*a.iu)))
    return np.exp(J) * np.linalg.det(A)


def install():
    old = TS.make

    def make(arm, m, F0, F, Ha, seed):
        if arm in GATED:
            return GatedAnsatz(m, F0, F, Ha, arm, seed=seed)
        return old(arm, m, F0, F, Ha, seed)
    TS.make = make
    TS.RNG_OFF.update({"hg": 305, "cbh": 306})


def check(args):
    ok = True
    rng = np.random.default_rng(5)
    for inst in (20, 23):
        m = mt.make_model(inst); refs = mt.references(m)
        for mode in GATED:
            a = GatedAnsatz(m, refs[1], 4, 0, mode, seed=3)
            x = a.flat(); a.set_flat(x + 0.3 * rng.standard_normal(len(x)))
            sel = rng.choice(m.D, 40, replace=False)
            ou, od, ru, rd = a._occ(sel)
            fast = a.forward(ou, od, ru, rd)["psi"]
            ref = np.array([explicit_amp(a, ou[k], od[k]) for k in range(len(sel))])
            err = np.max(np.abs(fast - ref)) / np.max(np.abs(ref))
            log(f"check (i) inst {inst} {mode}: explicit amplitude rel err {err:.1e}"); ok &= err <= 1e-12
    m = mt.make_model(21); refs = mt.references(m)
    base = T2.HFPSAnsatz(m, refs[1], "plus", 4, 0, seed=7)
    ones = GatedAnsatz(m, refs[1], 4, 0, "ones", seed=7)
    d = np.max(np.abs(base.amplitudes(4096) - ones.amplitudes(4096)))
    log(f"check (iii) gate == 1 reproduces HFPSAnsatz: max |diff| = {d:.1e}"); ok &= d == 0.0
    m = mt.make_model(20); m.spin_setup()
    refs = mt.references(m)
    W, Vv, deg = TS.solve_ed(m, 2, 1e-8, mt.UHF_SEED)
    sg, gen = mt.build_symmetry(m, mt.resolve_ground(m, Vv, deg))
    for mode in GATED:
        a = GatedAnsatz(m, refs[1], 4, 0, mode, seed=4)
        x = a.flat(); a.set_flat(x + 0.05 * rng.standard_normal(len(x)))
        ok &= TS.LT.fd_check(a, m, sg, "chi", rng, f"{mode} inst20")
    log(f"CHECK {'PASS' if ok else 'BUG'}")
    return ok


def ref_path(inst):
    return os.path.join(R12 if inst == 20 else R12B, "results", f"t1_full_inst{inst}_s11.json")


def decide(R, Rf, half_rho=HALF_RHO):
    """R: {(arm, inst): result}; Rf: {inst: full_F4 result}. Returns {arm: (verdict, rows)}."""
    out = {}
    for arm in GATED:
        rows = []
        for inst in (20, 23):
            fr, vr = Rf.get(inst), R.get((arm, inst))
            if fr is None or vr is None:
                rows.append((inst, "missing")); continue
            bf, bv = fr["budgets"]["P"], vr["budgets"]["P"]
            if fr["gap"] < TS.GAP_MIN:
                rows.append((inst, "not counted: gap")); continue
            bad = [lab for lab, r, b in (("full", fr, bf), (arm, vr, bv))
                   if r["calib_err"] > TS.CALIB_TOL * b["dE_tot"] or not (b["spread"] < TS.SPREAD_TOL)]
            if bad:
                rows.append((inst, f"not counted: calibration/spread ({','.join(bad)})")); continue
            if not (fr["converged"] and vr["converged"]):
                rows.append((inst, "not counted: unconverged")); continue
            if bf["dE_sign"] < TS.PREMISE:
                rows.append((inst, "not counted: premise")); continue
            rows.append((inst, bv["dE_sign"] / bf["dE_sign"]))
        cnt = [(i, r) for i, r in rows if isinstance(r, float)]
        if len(cnt) == 2 and all(r > TS.KILL_RATIO for _, r in cnt):
            v = "KILL (rho > 1.25 at both sizes)"
        elif arm == "cbh" and len(cnt) == 2 and all(r >= half_rho[i] for i, r in cnt):
            v = "KILL (Gate 2: no better than half at both sizes)"
        elif len(cnt) < 2:
            v = f"INCONCLUSIVE ({len(cnt)} counted sizes)"
        else:
            v = "T1-PASS (licenses the sizes 24 and 25 ladder)"
        out[arm] = (v, rows)
    return out


def aggregate(args):
    R = {}
    for fn in sorted(os.listdir(args.results)):
        if fn.startswith("t1_") and fn.endswith(".json"):
            r = json.load(open(os.path.join(args.results, fn)))
            R[(r["arm"], r["instance"])] = r
    Rf = {i: json.load(open(ref_path(i))) for i in (20, 23)}
    for (arm, inst), r in sorted(R.items()):
        b = r["budgets"]["P"]
        print(f"{arm:4s} inst {inst}: n_eff={r['n_eff']} conv={r['converged']} steps={r['steps_done']} "
              f"dE_tot={b['dE_tot']:.3e} dE_sign={b['dE_sign']:.3e} spread={b['spread']:.1e} cpu={r['total_cpu']:.0f}s")
    for i, r in Rf.items():
        print(f"full_F4 inst {i}: dE_tot={r['budgets']['P']['dE_tot']:.3e} dE_sign={r['budgets']['P']['dE_sign']:.3e}")
    for arm, (v, rows) in decide(R, Rf).items():
        print(f"VERDICT {arm}: {v}")
        for inst, r in rows:
            print(f"    inst {inst}: " + (f"rho = {r:.3f} (half {HALF_RHO[inst]:.3f})" if isinstance(r, float) else r))


def synthetic(args):
    def res(ds, conv=True, gap=0.1):
        return dict(gap=gap, converged=conv, calib_err=0.0, budgets=dict(P=dict(dE_sign=ds, dE_tot=0.05, spread=0.0)))
    Rf = {20: res(0.01), 23: res(0.01)}
    cases = [
        ("both pass", {("hg", 20): res(.011), ("hg", 23): res(.012), ("cbh", 20): res(.01), ("cbh", 23): res(.01)},
         "T1-PASS", "T1-PASS"),
        ("hg kill both", {("hg", 20): res(.013), ("hg", 23): res(.02), ("cbh", 20): res(.01), ("cbh", 23): res(.013)},
         "KILL", "T1-PASS"),
        ("cbh worse than half", {("hg", 20): res(.01), ("hg", 23): res(.01), ("cbh", 20): res(.025),
                                 ("cbh", 23): res(.016)}, "T1-PASS", "KILL"),
        ("cbh one over", {("hg", 20): res(.01), ("hg", 23): res(.01), ("cbh", 20): res(.02),
                          ("cbh", 23): res(.012)}, "T1-PASS", "T1-PASS"),
        ("unconverged", {("hg", 20): res(.01), ("hg", 23): res(.03, conv=False), ("cbh", 20): res(.01)},
         "INCONCLUSIVE", "INCONCLUSIVE"),
    ]
    ok = True
    for name, R, eh, ec in cases:
        o = decide(R, Rf)
        good = o["hg"][0].startswith(eh) and o["cbh"][0].startswith(ec)
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {name}: hg {o['hg'][0]} (want {eh}); cbh {o['cbh'][0]} (want {ec})")
    print(f"SYNTHETIC {'PASS' if ok else 'FAIL'} ({len(cases)} cases)")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["check", "train", "probe", "aggregate", "synthetic"], required=True)
    ap.add_argument("--arm", choices=GATED, default="hg")
    ap.add_argument("--instance", type=int, choices=[20, 23], default=20)
    ap.add_argument("--max-cpu", type=float, default=480.0)
    ap.add_argument("--probe-steps", type=int, default=20)
    ap.add_argument("--results", default=os.path.join(HERE, "results_acc"))
    a = ap.parse_args(argv)
    if a.mode == "check":
        sys.exit(0 if check(a) else 2)
    if a.mode == "synthetic":
        sys.exit(0 if synthetic(a) else 2)
    if a.mode == "aggregate":
        aggregate(a); return
    install()
    targs = TS.parse_args(["--mode", "train", "--arm", "full", "--instance", str(a.instance),
                           "--max-cpu", str(a.max_cpu), "--results", a.results])
    targs.arm = a.arm
    if a.mode == "probe":
        targs.timing_only, targs.max_steps, targs.min_steps = 1, a.probe_steps, a.probe_steps + 1
        t0 = time.process_time()
        TS.train(targs)
        log(f"probe {a.arm} inst {a.instance}: {(time.process_time() - t0) / a.probe_steps:.3f} CPU-s per step "
            f"(incl. setup)")
        return
    os.makedirs(a.results, exist_ok=True)
    fn = os.path.join(a.results, f"t1_{a.arm}_inst{a.instance}_s11.json")
    if os.path.exists(fn):
        log(f"{fn} exists; refusing to overwrite"); return
    r = TS.train(targs)
    json.dump(r, open(fn, "w"))
    log(f"wrote {fn}")


if __name__ == "__main__":
    main()
