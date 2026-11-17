#!/usr/bin/env python3
"""
P4.1 r22: cutoff-ladder polish of three 12-site arms (R19 full ladder + R20 step 2). Rule: RULE.md (pre-registered).
=================================================================================================================

Run with the lit_env interpreter:
    ~/.conda/envs/lit_env/bin/python ladder.py --mode MODE [--arm ARM] [options]

Modes (--mode, required)
  check     run only the INVALID checks for --arm (start energy; for rank5 also the rank-12 identity and the
            chain-rule log-derivative test) and print them; writes nothing.
  arm       polish --arm with the cutoff ladder until STOP-RULE, STOP-NOGAIN or the step cap (--steps); resumes
            from <res>/state_<arm>.json and <res>/theta_<arm>.npy if they exist (an arm with a final stop is not
            touched again). Runs the INVALID checks first and stores them in the state file.
  verdict   apply RULE.md to <res>/state_{pcg,che,rank5}.json: endpoint eps, convergence guard (geometric tail),
            rho_pol = eps_pcg/eps_che and rho_abl = eps_pcg/eps_rank5 with their classes, and the joint outcome;
            prints it and writes <res>/verdict.json. Missing or unfinished arms give INCOMPLETE.
Arms (--arm; required for check and arm)
  pcg     R2 DetModel('pcg', H=15) from ../P4.1_cheap_su2_r19/results/theta_pcg.npy (r19 N8 endpoint).
  che     V.Model('che', 16) from ../P4.1_cheap_su2_r19/results/theta_che.npy (r19 N8 endpoint).
  rank5   pcg with f0 = U_f D_f^T (U_f, D_f 12x5), started from the balanced SVD factors of pcg's N8 f0
          (R20 step 2; its energy must equal ../P4.1_cheap_su2_r21/results/truncate.json trunc[5].E).
Options
  --steps INT      step cap per arm (default 12, RULE.md)
  --threads INT    torch/BLAS threads (default 1, as r19/r21, so CPU times compare)
  --cpu-cap FLOAT  CPU-minutes; before each step, stop (state left resumable, stop = null) if the elapsed CPU time
                   plus the last step's CPU time would exceed it (default 74, under the 80-min sbatch limit)
  --res DIR        results directory (default ./results; local probes use ./results_probe)

Per step (identical for all arms, RULE.md): one exact S, H build (r19 polish.build / exact_SH) and one tangent
eigendecomposition (r11 realise.Tan), reused for every cutoff in LADDER = 1e-5..1e-9; at each cutoff the LM
candidate with |z| <= 0.05 and the line search s in {1, 1/2, 1/4} (first s lowering eps by > 1e-7); the rung with
the lowest exact energy is accepted. Predicted gain = E - E_LM(s z) at the accepted rung. Stops: STOP-NOGAIN (no
rung qualifies; nothing applied), STOP-RULE (accepted step's predicted gain < 1e-3 g; the step is applied),
STOP-CAP (--steps reached). g = r19 SR gap of pcg (pcg, rank5) or che (che).
"""
import argparse
import json
import os
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check", "arm", "verdict"])
    ap.add_argument("--arm", choices=["pcg", "che", "rank5"])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--threads", default="1")
    ap.add_argument("--cpu-cap", type=float, default=74.0)
    ap.add_argument("--res", default=None)
    a = ap.parse_args()
    if a.mode != "verdict" and a.arm is None:
        ap.error("--arm is required for modes check and arm")
    return a


A = parse_args()
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = A.threads
sys.argv = [sys.argv[0], "--threads", A.threads]  # polish.py / realise.py read --threads at import

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
R19 = os.path.join(TESTS, "P4.1_cheap_su2_r19")
R21 = os.path.join(TESTS, "P4.1_cheap_su2_r21")
sys.path.insert(0, R19)
import polish as P  # noqa: E402  (read only; imports r11 realise, r10 tangent, t2 vmc_t2)

R, T2, log = P.R, P.T2, P.log
E0, TAU, MIN_GAIN, STOP_FRAC, K = P.E0, 0.05, 1e-7, 1e-3, 5
LADDER = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
KEEP, RETRACT = 0.60, 0.85
RES = A.res or os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
c0 = time.process_time()
cpu = lambda: (time.process_time() - c0) / 60  # noqa: E731


# ------------------------------------------------------------------ R20 reparametrised model (rank-k f0)
class RankK(T2.DetModel):
    """pcg with f0 = U_f D_f^T: same backflow, Jastrow, P_G, P_S, quadrature; only the f0 block is replaced."""

    def __init__(self, k):
        super().__init__("pcg", 4, 3, 10, 15, -0.25, 8.0, (1, 1))
        L, self.k = self.L, k
        self.shapes = [("Uf", (L, k)), ("Df", (L, k))] + [s for s in self.shapes if s[0] != "f0"]
        self.sizes = [int(np.prod(s)) for _, s in self.shapes]
        self.nparam = sum(self.sizes)

    def unpack(self, th, rep=False):
        p = super().unpack(th, rep)
        p["f0"] = p.pop("Uf") @ p.pop("Df").transpose(-1, -2)
        return p


def split(m, th):
    off = np.cumsum([0] + m.sizes)
    return {n: th[off[i]:off[i + 1]].reshape(s) for i, (n, s) in enumerate(m.shapes)}


def pcg_to_rank(mp, th_pcg, k):
    """balanced SVD factors of pcg's f0 (top k triplets); other parameters copied unchanged."""
    p = split(mp, th_pcg.numpy())
    U, s, Vt = np.linalg.svd(p["f0"])
    mk = RankK(k)
    p["Uf"], p["Df"] = U[:, :k] * np.sqrt(s[:k]), Vt[:k].T * np.sqrt(s[:k])
    return mk, torch.tensor(np.concatenate([p[n].ravel() for n, _ in mk.shapes]))


# ------------------------------------------------------------------ arms
def setup(arm):
    """(ctx, theta_start, E_recorded_or_reference, g, checks)."""
    seqp = json.load(open(os.path.join(R19, "results/seq_pcg.json")))
    ctx = P.ctx_for("che" if arm == "che" else "pcg")
    th_r19 = torch.tensor(np.load(os.path.join(R19, f"results/theta_{'che' if arm == 'che' else 'pcg'}.npy")))
    seq = json.load(open(os.path.join(R19, f"results/seq_{'che' if arm == 'che' else 'pcg'}.json")))
    Erec = seq["steps"][-1]["E_new"]
    chk = {}
    if arm in ("pcg", "che"):
        E = ctx.energy(th_r19)[0]
        chk["start_vs_r19"] = abs(E - Erec)
        chk["valid"] = chk["start_vs_r19"] < 1e-8
        return ctx, th_r19, seq["gap"], chk
    # rank5: identity at full rank, start energy vs r21 truncate.json, chain-rule log-derivatives
    mp = ctx.m
    m12, th12 = pcg_to_rank(mp, th_r19, 12)
    ctx.m = m12
    chk["rank12_vs_pcg"] = abs(ctx.energy(th12)[0] - Erec)
    m5, th5 = pcg_to_rank(mp, th_r19, K)
    ctx.m = m5
    E5 = ctx.energy(th5)[0]
    Etr = json.load(open(os.path.join(R21, "results/truncate.json")))["trunc"]["5"]["E"]
    chk["start_vs_truncate"] = abs(E5 - Etr)
    # chain rule: d/dU_f = G D_f, d/dD_f = G^T U_f with G = d log psi / d f0 of pcg at the truncated point
    p5 = split(m5, th5.numpy())
    pf = split(mp, th_r19.numpy())
    pf["f0"] = p5["Uf"] @ p5["Df"].T
    thf = torch.tensor(np.concatenate([pf[n].ravel() for n, _ in mp.shapes]))
    ix = ctx.reps[:64]
    Gf = P.logder(mp, thf, ctx.nU[ix], ctx.nD[ix])
    G5 = P.logder(m5, th5, ctx.nU[ix], ctx.nD[ix])
    L, B = mp.L, len(ix)
    G = Gf[:, :L * L].reshape(B, L, L)  # f0 is the first block of pcg's shapes
    assert mp.shapes[0][0] == "f0"
    exp5 = np.concatenate([(G @ p5["Df"]).reshape(B, -1), (G.transpose(0, 2, 1) @ p5["Uf"]).reshape(B, -1),
                           Gf[:, L * L:]], 1)
    chk["chain_rule_rel"] = float(np.abs(G5 - exp5).max() / np.abs(exp5).max())
    chk["nparam"] = m5.nparam
    chk["valid"] = chk["rank12_vs_pcg"] < 1e-10 and chk["start_vs_truncate"] < 1e-10 and chk["chain_rule_rel"] < 1e-9
    ctx.arm = "rank5"
    return ctx, th5, seqp["gap"], chk


def dump(fn, d):
    json.dump(d, open(fn, "w"), indent=1)


def run_arm(a):
    t0 = time.time()
    fn, fth = os.path.join(RES, f"state_{a.arm}.json"), os.path.join(RES, f"theta_{a.arm}.npy")
    ctx, th, g, chk = setup(a.arm)
    log(f"{a.arm}: P {ctx.m.nparam}; checks {chk} [{cpu():.2f} CPU-min]")
    st = dict(arm=a.arm, g=g, checks=chk, ladder=LADDER, tau=TAU, min_gain=MIN_GAIN, stop_frac=STOP_FRAC,
              cap=a.steps, steps=[], stop=None)
    if os.path.exists(fn):
        old = json.load(open(fn))
        st.update(steps=old["steps"], stop=old["stop"], E_start=old["E_start"])
        if st["stop"]:
            log(f"{a.arm}: already finished ({st['stop']}); nothing to do")
            return
        if old["steps"]:
            th = torch.tensor(np.load(fth))
    if not chk["valid"]:
        st["stop"] = "INVALID"
        dump(fn, st)
        log(f"{a.arm}: INVALID checks; stop")
        return
    E = ctx.energy(th)[0]
    st.setdefault("E_start", E)
    Pn, per = ctx.m.nparam, None
    while st["stop"] is None:
        if len(st["steps"]) >= a.steps:
            st["stop"] = "STOP-CAP"
            break
        s0 = cpu()
        if per is not None and s0 + per > a.cpu_cap:
            log(f"stop: next step would exceed the CPU cap ({s0:.1f} + {per:.1f} > {a.cpu_cap}); resumable")
            break
        Eb, Y, HY, pr = P.build(ctx, th)
        assert abs(Eb - E) < 1e-9, (Eb, E)
        S, Hm = P.exact_SH(Y, HY)
        del Y, HY, pr
        tan = R.Tan(S, Hm, E, list(range(Pn)))
        del S, Hm
        rungs, best, prev = [], None, None
        for c in LADDER:
            r = tan.reduce(c)
            if r["rank"] == prev:
                rungs.append(dict(cut=c, rank=r["rank"], skipped=True))
                continue
            prev = r["rank"]
            z, lam = tan.candidate(r, "lm", TAU)
            d = tan.dtheta(r, z, Pn)
            acc, trials = None, []
            for s in (1.0, 0.5, 0.25):
                Es = ctx.energy(th + s * torch.tensor(d))[0]
                trials.append((s, (Es - E0) / abs(E0)))
                if np.isfinite(Es) and (E - Es) / abs(E0) > MIN_GAIN:
                    acc = (s, Es)
                    break
            row = dict(cut=c, rank=r["rank"], zn=float(np.linalg.norm(z)), lam=lam,
                       pred_full=E - tan.pred(r, z), trials=trials, s=None)
            if acc:
                row.update(s=acc[0], E_new=acc[1], pred=E - tan.pred(r, acc[0] * z))
                if best is None or acc[1] < best["E_new"]:
                    best = dict(row, d=d)
            rungs.append(row)
        rec = dict(E=E, eps=(E - E0) / abs(E0), rungs=rungs, cpu_min=None)
        if best is None:
            rec["accepted"] = False
            st["steps"].append(rec)
            st["stop"] = "STOP-NOGAIN"
            log(f"step {len(st['steps'])}: no rung lowers eps by > {MIN_GAIN:g}; STOP-NOGAIN")
            break
        th = th + best["s"] * torch.tensor(best["d"])
        E = best["E_new"]
        per = cpu() - s0
        rec.update(accepted=True, cut=best["cut"], s=best["s"], E_new=E, eps_new=(E - E0) / abs(E0),
                   gain=rec["E"] - E, pred=best["pred"], pred_over_g=best["pred"] / g, cpu_min=per)
        st["steps"].append(rec)
        np.save(fth, th.numpy())
        if best["pred"] < STOP_FRAC * g:
            st["stop"] = "STOP-RULE"
        log(f"step {len(st['steps'])}: cut {best['cut']:.0e} s {best['s']} eps {rec['eps']:.5e} -> "
            f"{rec['eps_new']:.5e} (drop {100 * (1 - rec['eps_new'] / rec['eps']):.2f}%); predicted gain "
            f"{rec['pred_over_g']:.2e} g; rungs " +
            " ".join(f"{x['cut']:.0e}:{'skip' if x.get('skipped') else x['s']}" for x in rungs) +
            f" [{cpu():.1f} CPU-min, step {per:.2f}, wall {time.time() - t0:.0f} s]{' ' + st['stop'] if st['stop'] else ''}")
        st["cpu_min_total"] = cpu()
        dump(fn, st)
    st["cpu_min_total"] = cpu()
    dump(fn, st)
    log(f"{a.arm}: stop {st['stop']}; {len(st['steps'])} steps; wrote {fn} [{cpu():.1f} CPU-min, "
        f"{time.time() - t0:.0f} s wall]")


# ------------------------------------------------------------------ verdict
def endpoint(st):
    acc = [s for s in st["steps"] if s.get("accepted")]
    E = acc[-1]["E_new"] if acc else st["E_start"]
    eps = (E - E0) / abs(E0)
    tail = 0.0
    if st["stop"] == "STOP-CAP":
        gains = [s["gain"] for s in acc]
        r = gains[-1] / gains[-2] if len(gains) >= 2 and gains[-2] > 0 else np.inf
        tail = gains[-1] * r / (1 - r) if r < 1 else np.inf
    return eps, max(eps - tail / abs(E0), 0.0), len(acc)  # eps >= 0: an unbounded tail clips to 0


def cls(rho, names):
    return names[0] if rho <= KEEP else names[2] if rho >= RETRACT else names[1]


JOINT = {
    ("KEEP-LIVE", "PAIRING-KEPT"): "R2 live, mechanism attributed at 12 sites; 24-site rung needs R21 VIABLE + critic",
    ("KEEP-LIVE", "ABL-CONFOUNDED"): "R2 live, mechanism unattributed; any 24-site rung must carry the rank5 arm",
    ("KEEP-LIVE", "PAIRING-RETRACTED"): "R2 pairing claim retracted; advantage is the backflow form; open new route (L51)",
    ("CONFOUNDED", "PAIRING-KEPT"): "R2 at T2a, pairing effect real, cross-architecture ratio confounded; 12-site question closed",
    ("CONFOUNDED", "ABL-CONFOUNDED"): "R2 closed at 12 sites as unresolved at this budget",
    ("CONFOUNDED", "PAIRING-RETRACTED"): "R2 killed (mechanism retracted, no clean advantage)",
    ("RETRACT", "PAIRING-KEPT"): "R2 killed as competitive ansatz; library entry: pairing helps a weaker backflow",
    ("RETRACT", "ABL-CONFOUNDED"): "R2 killed (r19 rule RETRACT)",
    ("RETRACT", "PAIRING-RETRACTED"): "R2 killed on both counts; Q1 has no live route",
}


def verdict(a):
    S = {}
    for arm in ("pcg", "che", "rank5"):
        f = os.path.join(RES, f"state_{arm}.json")
        if os.path.exists(f):
            S[arm] = json.load(open(f))
    out = dict(arms={})
    for arm, st in S.items():
        eps, eps_inf, n = endpoint(st)
        out["arms"][arm] = dict(stop=st["stop"], steps=n, eps_start=(st["E_start"] - E0) / abs(E0), eps=eps,
                                eps_inf=eps_inf, checks=st["checks"])
        print(f"{arm:5s}: stop {st['stop']}; {n} accepted steps; eps {out['arms'][arm]['eps_start']:.5e} -> "
              f"{eps:.5e} (tail-extrapolated {eps_inf:.5e}); checks valid {st['checks'].get('valid')}")
    stops = [S[k]["stop"] for k in S]
    if len(S) < 3 or any(s is None for s in stops):
        out["verdict"] = "INCOMPLETE"
    elif any(s == "INVALID" for s in stops):
        out["verdict"] = "INVALID"
    else:
        A_ = out["arms"]
        res = {}
        for key, den, names in (("rho_pol", "che", ("KEEP-LIVE", "CONFOUNDED", "RETRACT")),
                                ("rho_abl", "rank5", ("PAIRING-KEPT", "ABL-CONFOUNDED", "PAIRING-RETRACTED"))):
            rho = A_["pcg"]["eps"] / A_[den]["eps"]
            num_i, den_i = A_["pcg"]["eps_inf"], A_[den]["eps_inf"]
            ok = num_i > 0 or den_i > 0  # both clipped to 0: nothing can be said
            rho_inf = (num_i / den_i if den_i > 0 else float("inf")) if ok else float("nan")
            c1 = cls(rho, names)
            c2 = cls(rho_inf, names) if ok else None
            res[key] = dict(rho=rho, rho_inf=rho_inf, cls=c1 if c1 == c2 else "UNSETTLED", cls_end=c1, cls_inf=c2)
            print(f"{key} = eps_pcg/eps_{den} = {rho:.4f} (tail-extrapolated {rho_inf:.4f}) -> {res[key]['cls']}")
        out.update(res)
        jp, ja = res["rho_pol"]["cls"], res["rho_abl"]["cls"]
        out["verdict"] = f"{jp} / {ja}"
        out["action"] = JOINT.get((jp, ja), "UNSETTLED ratio: resume the cap-limited arm(s) (new gated job); no change")
    print(f"VERDICT: {out['verdict']}" + (f"; ACTION: {out['action']}" if "action" in out else ""))
    dump(os.path.join(RES, "verdict.json"), out)


if __name__ == "__main__":
    if A.mode == "check":
        _, _, g, chk = setup(A.arm)
        log(f"{A.arm}: g {g:.6e}; checks {chk} [{cpu():.2f} CPU-min]")
    elif A.mode == "arm":
        run_arm(A)
    else:
        verdict(A)
