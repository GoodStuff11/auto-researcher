#!/usr/bin/env python3
"""
P3.1 Round 10: T3 decisive test of Route 10.1 in its scalable form 10.1d, with TRAINED response networks
=======================================================================================================

Question. The T2-v2 premise run (job 699756) showed that a self-consistent response Ritz with the base updated
each pass recovers the real-space tail errors of the real 4x4 width-6 NNBF states when the responses are EXACT
(minres). The scalable carrier 10.1d (SCRd0, T = 4) tracked 8 of 8 items there. This test replaces every exact
response by a trained Hylleraas response network of the same architecture and width as the ground state (NNBF,
exact-sum force, SR metric sampled from |P y_theta|^2 with 2000 samples, exactly as the real ground states were
trained), and adds the mandatory matched-compute foils (Gate 2).

Instance. The real 4x4 states cannot be used inside the brainstorm's compute limit: one width-6 network on the
4x4 OBC 6+6 instance costs about 68 CPU-h (P2.1 r3: 12.7 s/step x 16 CPUs x 1200 steps), and 10.1d at T = 4
trains 12 response networks per seed. The test therefore runs on the 4x3 OBC t-t'-J cluster (4+4, n = 2/3,
t' = -0.25, J = 0.4, D = 34650, S_0 = 0, chi = (Mx, My) = (+1, +1), omega_1 = 0.16 t, 11 levels with
omega <= 1 t; Round 10 census), with ground states trained here at the same width and with the same optimizer,
and the tails defined as in t1_portfolio.py (cut-offs 0.82 x the largest distance: 10 bond pairs, 9 site pairs).
Phase A checks that the trained states carry scorable tail errors (the target gate, applied at --aggregate).

Per task (one ground-state seed; phases in order, JSON rewritten after every phase):
  A  Ground state: NNBF width --gs-width, --gs-iters SR steps (P2.1 r3 settings: step 0.05, shift 1e-3,
     2000 samples, trust region 1, best-E iterate kept). Checkpoint file reused if present (resume).
     Items: tails pair / spin / charge; delta = a_0 - a(psi); scored if |delta| >= 0.10 |a_0|.
     If the seed has no scored item, phases B-D are skipped (nothing to score).
  B  Exact references (no training): L1R span{psi, (H-E)psi} (negative control), ORC3 span{psi, (H-E)psi,
     |1>,|2>,|3>} (reported), SCRd0 = 10.1d at T = 4 with exact responses (the T2-v2 premise on this
     instance). Shift of every response: s = E_L1R(phi) - 0.1 t; validity: s < E_0 (functional bounded).
  C  Deciding arm SCRd-NN: 10.1d with trained responses.
       pass 1: three response networks u_A (A = pair, spin, charge), cold start from the ground-state
               parameters plus a seeded perturbation (--resp-init), --resp-cold SR steps each;
               phi_1 = lowest Ritz vector of span{phi_0, (H-E)phi_0, u_A, K u_A}, K v = (H - <H>_v) v.
       pass t = 2..T: the three networks warm-start from pass t-1, --resp-warm SR steps each;
               phi_t = lowest Ritz vector of span{phi_{t-1}, u_A(phi_{t-1})}.   (H enters only in pass 1.)
     Hylleraas ratio minimised per network: R(u) = <u|(H - s)|u> / <u|b>^2, u = Q P_chi y_theta,
     Q = P_chi - |phi><phi|, b = unit(Q P_chi (A - a_phi) phi); argmin R = (Q(H-s)Q)^{-1} b up to scale.
     Force: dR/dtheta = 2 sum_n g(n) dy_theta(n)/dtheta, g = Q[(H-s)u - R <u|b> b] / <u|b>^2 (checked by
     finite differences in --mode gradcheck). Reported per network: R, the exact optimum R*, the response
     error eta = 1 - cos^2(u, y_exact) and the in-window error eta_low (same on the levels with omega <= 1 t).
     Scored: SCRd-NN at T = 4 (deciding), T = 1 (RWRK-NN) and T = 2 (reported). NN0 = pass 1 with the
     untrained (initial) networks: because they start from perturbed copies of the ground-state network, NN0 is a
     Ritz step in random tangent directions of the ground-state manifold, i.e. a zero-cost "more of the same
     state" foil (F0), and it enters the Gate 2 comparison.
  D  Matched-compute foils (Gate 2), each reporting its own observable shift as an estimate of delta:
       F1 "train the state further": continue the ground-state SR for as many steps as phase C used in total
          (3 x (cold + (T-1) warm)); ratio = (a(psi') - a(psi)) / delta.
       F2 "wider state": a fresh NNBF of width 2 x --gs-width trained until its CPU time equals the CPU time of
          phase A plus phase C (the response route's total), capped at --gs-iters + F1 steps.
       F3 "Davidson with trained correction networks" (added by the T3 design critic, 2026-11-16): the phase-C
          pipeline with the observable sources replaced by b_p = unit(Q (H - E)^p phi), p = 1, 2, 3 (no observable
          information), same init, steps and passes; ratio from its phi_4. Its exact analogue SCRdE0 (phase B,
          reported) tracks 14 of 14 4x3 T1 items (critic_t3_foil_check.py): an exact shift-invert of the residual
          is inverse iteration, so the exact references cannot show that the observable sources matter.
  MC (reported, critic 2026-11-16): the SCRd-NN Ritz chain and the readout delta-hat with S, H and <A> estimated
     from Ns exact draws (Ns = 1e4 and 1e5, 5 draws each; median ratio scored as MC10000 and MC100000).
Scored item: |delta| >= 0.10 |a_0|. Tracked: ratio in [1/2, 2].

Pre-registered rule (copied into the leaf, "T3 design for route 10.1 (2026-11-16)", before submission):
  n = number of scored items over all tasks (6 seeds: at most 18).
  NO-TARGET      n < 9. The instance does not keep the target at this width; no route verdict (the target
                 probe, --checkpoint target, applies the same threshold before the full run).
  PREMISE-FAIL   SCRd0 (exact responses) tracks < 0.75 n. The T2-v2 premise does not transfer to this
                 instance; no route verdict.
  KILL (10.1d)   SCRd-NN (T = 4) tracks < 0.5 n.
  KILL (Gate 2)  a foil (F0 = NN0, F1, F2 or F3) tracks at least as many items as SCRd-NN.
  PASS           SCRd-NN tracks >= 0.75 n, including >= 1 spin item if any spin item is scored, AND
                 SCRd-NN tracks >= max(F0, F1, F2, F3) + ceil(0.25 n). Scope label of a PASS: "sampled Ritz
                 and readout" if MC100000 tracks >= 0.75 n, else "exact-sum Ritz and readout only".
  AMBIGUOUS      otherwise.
  INVALID        any validity assert failed (s >= E_0, a non-finite objective, a control that fails).
  INCOMPLETE     n >= 6 but some task has not reached phase "done" (resubmit that seed; its ground state is
                 reloaded from the checkpoint).
  Negative control (must hold for a PASS or AMBIGUOUS to stand, else INVALID (control)): L1R tracks <= 0.25 n.
  Positive control: SCRd0 (exact responses) is the PREMISE-FAIL test above.
Checkpointed early stop (--aggregate on partial JSONs, the "pass2" checkpoint = 50 % of the response budget):
  KILL early if, over the items of every task that has finished pass 2, fewer than 0.5 n_done are still
  reachable, where an item is reachable if its SCRd-NN pass-2 ratio lies in [0.5 - 0.30, 2 + 0.60]. The
  margin 0.30 is 2.5 x the largest pass-2 -> pass-4 change of any exact item in job 699756 (0.12, charge,
  seed 11). It is evaluated only after the cold-start budget of every network is spent, so a slow starter
  cannot trigger it. Partial results may kill early, never promote. The check prints WAIT until every task has
  passed pass 2 (or finished with no scored item).
Width-ladder guard (critic 2026-11-16): a probe at a width below 6 is a valid target only if the median overlap^2
  is >= 0.85 (the real 4x4 states have 0.91-0.94 at V = 16); otherwise the ladder stops and the trained-response
  step is recorded as untested. A KILL at a width below 6 is recorded as "KILL at width W".

Usage
  python3 t3_scrd.py --mode run --size 4x3 --seed 11       (one Slurm task; JSON results/t3/t3_4x3_w6_s11.json)
  python3 t3_scrd.py --mode run --size 4x3 --seed 11 --stop-after A   (target probe; then --checkpoint target)
  python3 t3_scrd.py --aggregate results/t3/t3_4x3_w6_s11.json,...,results/t3/t3_4x3_w6_s66.json   (verdict)
  python3 t3_scrd.py --aggregate ... --checkpoint pass2                                 (early-stop check)
  python3 t3_scrd.py --mode dry                                                         (pre-flight, 3x3 toy)
  python3 t3_scrd.py --mode gradcheck                                                   (FD check, 3x3 toy)
  python3 t3_scrd.py --mode timing --size 4x3                                           (per-step CPU cost)
  python3 t3_scrd.py --selftest-verdict                                                 (every verdict branch)
  python3 t3_scrd.py --mode power                                                       (power sizing)
Options
  --mode {run, dry, gradcheck, timing, power}
        run        the four phases above for one --seed on --size.
        dry        pre-flight on the 3x3 3+3 toy (D = 1680) with short budgets: the run pipeline on a
                   briefly trained width-2 ground state, plus a positive-control target psi_0 + 0.05 e
                   (first-order error; the exact-response arm must track it); writes --out, aggregates it.
        gradcheck  finite-difference check of the Hylleraas force and of the ground-state force on 3x3.
        timing     CPU seconds per SR step (ground state width w and 2w, response network) on --size.
        power      Monte Carlo of the verdict under a PASS world and a KILL world (see the leaf).
  --size {3x3, 4x3}         instance (4x3 for the T3; 3x3 for the toy modes).
  --seed INT                ground-state seed (task); response networks use seed + 100 k, F2 uses seed + 500.
  --gs-width INT            ground-state NNBF width (default 6); F2 uses twice this.
  --gs-iters INT            ground-state SR steps (default 1200).
  --resp-cold INT           SR steps of each pass-1 response network (default 600).
  --resp-warm INT           SR steps of each response network in passes 2..T (default 200).
  --T INT                   number of passes (default 4).
  --resp-init {gs, uhf}     gs (default): ground-state parameters plus Gaussian noise of 0.05 x the rms of each
                            tensor (seed + 100 k); uhf: a fresh NNBF from the UHF reference.
  --sr-lr, --sr-shift, --sr-samples, --o-chunk, --chunk   SR settings (defaults 0.05, 1e-3, 2000, 8, 32768).
  --params-dir PATH         checkpoint directory for network parameters (default results/t3/params).
  --out PATH                JSON output (rewritten after every phase).
  --aggregate LIST          comma list of task JSONs: print the verdict (or the checkpoint check) and exit.
  --checkpoint {final, pass2, target}  which rule --aggregate applies (default final): pass2 = the early-stop
                            check; target = the target probe (scored items over the seeds, need >= 9).
  --selftest-verdict        synthetic records covering every verdict branch.
  --cpu-budget FLOAT        unused guard kept for the record (default 1e9 s).
  --target {trained, posctrl}  trained (default): the target is the phase-A ground state. posctrl (smoke only):
                            the target is psi_0 + 0.05 e (first-order error) and the phase-A net only seeds the
                            response networks and the foils; no physics claim.
  --stop-after {A, done}    A: train and score the ground state only (target probe; the parameters are saved
                            and reused by the later full task); done (default): all phases.
"""
import argparse
import importlib.util
import json
import math
import os
import resource
import sys
import time

import numpy as np
import scipy.sparse.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
R8 = os.path.join(HERE, "..", "P3.1_order_parameter_error_r8")
_spec = importlib.util.spec_from_file_location("t1p", os.path.join(R8, "t1_portfolio.py"))
t1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1)
rc, log, unit = t1.rc, t1.log, t1.unit
REAL = ("pair", "spin", "charge")
KRYLOV = ("kry1", "kry2", "kry3")     # observable-free sources of the critic foil F3 (SCRdE)
WINDOW = 1.0
MC_NS, MC_DRAWS = (10_000, 100_000), 5   # sampled-Ritz replay (reported; critic 2026-11-16)
MARGIN_LO, MARGIN_HI = 0.30, 0.60
OV2_MIN = 0.85   # width-ladder guard (critic 2026-11-16): real 4x4 states 0.91-0.94 at V = 16


def cpu():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def tracked(x):
    return x is not None and np.isfinite(x) and 0.5 <= x <= 2.0


# ============================================================================ instance and exact algebra
class Ctx:
    def __init__(self, size, n_low):
        I = t1.Inst(size, n_low)
        self.I, self.m, self.sg, self.E0, self.psi0 = I, I.m, I.sg, I.E0, I.psi0
        self.U, self.om = I.U, I.om
        self.Uw = I.U[:, I.om <= WINDOW]
        self.a0 = {k: float(self.psi0 @ I.apply(k, self.psi0)) for k in REAL}
        assert I.S0 == 0 or size == "3x3", ("4x3 must be a singlet", I.S0)
        if size == "4x3":
            assert I.om[-1] > WINDOW and self.Uw.shape[1] >= 3, ("window levels not all found", I.om[-1])

    def apply(self, k, v):
        return self.I.apply(k, v)

    def a(self, k, v):
        v = unit(v)
        return float(v @ self.apply(k, v))

    def E(self, v):
        v = unit(v)
        return float(v @ (self.m.H @ v))

    def Kv(self, v):
        v = unit(v)
        w = self.m.H @ v
        return self.sg.project(w - (v @ w) * v)

    def ritz_vec(self, vecs):
        B = np.array([unit(v) for v in vecs])
        S = B @ B.T
        H = B @ np.array([self.m.H @ b for b in B]).T
        S, H = 0.5 * (S + S.T), 0.5 * (H + H.T)
        lam, V = np.linalg.eigh(S)
        keep = lam > 1e-11 * lam.max()
        X = V[:, keep] / np.sqrt(lam[keep])
        ev, Y = np.linalg.eigh(X.T @ H @ X)
        v = unit((X @ Y[:, 0]) @ B)
        return (v if v @ vecs[0] >= 0 else -v), float(ev[0])

    def Q(self, v, phi):
        v = self.sg.project(v)
        return v - phi * (phi @ v)

    def shift(self, phi):
        phi = unit(phi)
        E = self.E(phi)
        _, EL1 = self.ritz_vec([phi, self.m.H @ phi - E * phi])
        s = EL1 - 0.1
        assert s < self.E0, ("VALIDITY: shift not below E_0; Hylleraas functional unbounded", s, self.E0)
        return s

    def source(self, k, phi):
        phi = unit(phi)
        if k in KRYLOV:                       # critic foil F3: b_p = unit(Q (H - E)^p phi), no observable
            E, v = self.E(phi), phi
            for _ in range(KRYLOV.index(k) + 1):
                v = self.sg.project(self.m.H @ v - E * v)
            return unit(self.Q(v, phi))
        b = self.Q(self.apply(k, phi) - self.a(k, phi) * phi, phi)
        return unit(b)

    def solve(self, b, s, phi):
        m = self.m

        def op(v):
            x = self.Q(v, phi)
            return self.Q(m.H @ x - s * x, phi) + (v - x)
        L = sla.LinearOperator((m.D, m.D), matvec=op, dtype=float)
        x, info = sla.minres(L, b, rtol=1e-10, maxiter=20000)
        res = np.linalg.norm(op(x) - b) / np.linalg.norm(b)
        assert res < 1e-6, ("minres", info, res)
        return self.Q(x, phi)

    def R_of(self, u, b, s):
        return float((u @ (self.m.H @ u) - s * (u @ u)) / (u @ b) ** 2)


# ============================================================================ SR machinery
def sr_solve(ans, sg, yP, f, args, rng):
    pc = yP ** 2
    idx = rng.choice(len(pc), size=args.sr_samples, p=pc / pc.sum())
    O = ans.per_sample_O_proj(sg, idx, yP, args.o_chunk)
    Oc = (O - O.mean(0)) / np.sqrt(len(idx))
    lam = args.sr_shift
    Km = Oc @ Oc.T + lam * np.eye(len(idx))
    dx = (f - Oc.T @ np.linalg.solve(Km, Oc @ f)) / lam
    step = args.sr_lr * dx
    nrm = np.linalg.norm(dx)
    if nrm * args.sr_lr > 1.0:
        step *= 1.0 / (nrm * args.sr_lr)
    return step


def gs_eval(ans, C, args):
    yP = C.sg.project(ans.amplitudes(args.chunk))
    Hp = C.m.H @ yP
    nn = float(yP @ yP)
    E = float(yP @ Hp) / nn
    return E, yP, (Hp - E * yP) / nn


def resp_eval(ans, C, phi, b, s, args):
    yP = C.sg.project(ans.amplitudes(args.chunk))
    u = yP - phi * (phi @ yP)
    Hu = C.m.H @ u
    ub = float(u @ b)
    R = float(u @ Hu - s * (u @ u)) / ub ** 2
    g = C.Q(Hu - s * u - R * ub * b, phi) / ub ** 2
    return R, yP, u, g


def train(ans, objective, C, args, rng, iters, label, cpu_cap=None):
    """Generic SR loop. objective(ans) -> (value, yP, g); force = ans.force(g) (= half the gradient).
    Keeps the best iterate. Returns dict(steps, best, cpu, trace)."""
    best, best_x = np.inf, ans.flat().copy()
    c0 = cpu()
    trace = []
    it = 0
    for it in range(iters):
        x0 = ans.flat().copy()
        val, yP, g = objective(ans)
        assert np.isfinite(val), f"VALIDITY: non-finite objective in {label} at step {it}"
        if val < best:
            best, best_x = val, x0
        f = ans.force(g, args.chunk)
        ans.set_flat(x0 - sr_solve(ans, C.sg, yP, f, args, rng))
        if it % 100 == 0:
            trace.append((it, val))
        if cpu_cap is not None and cpu() - c0 >= cpu_cap:
            break
    val = objective(ans)[0]
    if not (val <= best):
        ans.set_flat(best_x)
        val = best
    out = dict(steps=it + 1, best=float(val), cpu=cpu() - c0, trace=trace)
    log(f"  {label}: {out['steps']} steps, best {val:.8f}, {out['cpu']:.0f} CPU-s "
        f"({out['cpu'] / max(out['steps'], 1):.3f} CPU-s/step)")
    return out


def save(ans, path):
    np.savez(path, flat=ans.flat(), layout=json.dumps([[k, list(ans.p[k].shape)] for k in ans.keys]))


def perturbed_copy(C, ans_gs, width, seed, scale):
    ans = rc.nnbf(C.m, width, seed)
    ans.set_flat(ans_gs.flat().copy())
    rng = np.random.default_rng(seed)
    for k in ans.keys:
        p = ans.p[k]
        rms = float(np.sqrt(np.mean(p ** 2))) or 1.0
        p += scale * rms * rng.standard_normal(p.shape)
    return ans


def mc_ritz(C, vecs, ns, rng):
    """Lowest Ritz vector of span(vecs) with S and H estimated from ns exact draws of the mixture guide."""
    B = np.array([unit(v) for v in vecs])
    HB = np.array([C.m.H @ b for b in B])
    pi = np.mean(B ** 2, axis=0)
    pi /= pi.sum()
    x = rng.choice(len(pi), size=ns, p=pi)
    w = 1.0 / pi[x]
    Bx, HBx = B[:, x], HB[:, x]
    S = (Bx * w) @ Bx.T / ns
    H = (Bx * w) @ HBx.T / ns
    S, H = 0.5 * (S + S.T), 0.5 * (H + H.T)
    lam, V = np.linalg.eigh(S)
    keep = lam > 1e-11 * lam.max()
    X = V[:, keep] / np.sqrt(lam[keep])
    ev, Y = np.linalg.eigh(X.T @ H @ X)
    v = unit((X @ Y[:, 0]) @ B)
    return v if v @ vecs[0] >= 0 else -v


def mc_delta(C, k, phi, psi, ns, rng):
    """a(phi) - a(psi) from ns shared draws of g = (phi^2 + psi^2) / 2 (correlated reweighting)."""
    phi, psi = unit(phi), unit(psi)
    g = 0.5 * (phi ** 2 + psi ** 2)
    g /= g.sum()
    x = rng.choice(len(g), size=ns, p=g)
    w = 1.0 / g[x]
    out = []
    for v in (phi, psi):
        Av = C.apply(k, v)
        out.append(float(np.sum(w * v[x] * Av[x]) / np.sum(w * v[x] ** 2)))
    return out[0] - out[1]


# ============================================================================ one task
class Task:
    def __init__(self, args, C=None):
        self.a = args
        self.C = C or Ctx(args.size, max(3, args.n_low))
        self.out = dict(args=vars(args), size=args.size, seed=args.seed, E0=self.C.E0, a0=self.C.a0,
                        items={}, ratios={}, resp=[], cpu={}, phase="start")
        self.t0 = time.time()

    def dump(self):
        self.out["cpu"]["total"] = cpu()
        if self.a.out:
            json.dump(self.out, open(self.a.out, "w"), indent=1, default=float)

    def score(self, arm, phi):
        r = {}
        for k in REAL:
            it = self.out["items"][k]
            r[k] = (self.C.a(k, phi) - it["a_psi"]) / it["delta"]
        self.out["ratios"][arm] = r
        log(f"  seed {self.a.seed} {arm:12s} " + " ".join(f"{k} {r[k]:+.3f}" for k in REAL)
            + f"  (cpu {cpu():.0f} s)")
        self.dump()
        return r

    def ground_state(self):
        a, C = self.a, self.C
        os.makedirs(a.params_dir, exist_ok=True)
        f = os.path.join(a.params_dir, f"gs_{a.size}_w{a.gs_width}_s{a.seed}.npz")
        ans = rc.nnbf(C.m, a.gs_width, a.seed)
        c0 = cpu()
        if os.path.exists(f) and not a.dry_target:
            rc.load_params(ans, f)
            log(f"phase A: loaded {f}")
            rec = dict(loaded=True)
        else:
            rec = train(ans, lambda x: gs_eval(x, C, a), C, a, np.random.default_rng(a.seed), a.gs_iters,
                        f"phase A ground state w{a.gs_width}")
            save(ans, f)
        self.out["cpu"]["A"] = cpu() - c0
        self.out["gs_train"] = rec
        return ans

    def set_target(self, psi):
        C = self.C
        psi = unit(psi)
        psi = psi if psi @ C.psi0 > 0 else -psi
        self.psi = psi
        for k in REAL:
            ap = C.a(k, psi)
            d = C.a0[k] - ap
            self.out["items"][k] = dict(a_psi=ap, delta=d, scored=bool(abs(d) >= 0.10 * abs(C.a0[k])))
        self.out["overlap2"] = float((psi @ C.psi0) ** 2)
        self.out["dE"] = C.E(psi) - C.E0
        log(f"seed {self.a.seed}: |<psi0|psi>|^2 {self.out['overlap2']:.4f} dE {self.out['dE']:.5f} | " + " ".join(
            f"{k} delta {self.out['items'][k]['delta']:+.3e} ({self.out['items'][k]['delta'] / C.a0[k]:+.2f} a0) "
            f"{'scored' if self.out['items'][k]['scored'] else 'ns'}" for k in REAL))
        self.dump()

    def exact_refs(self):
        C, psi = self.C, self.psi
        c0 = cpu()
        r = C.m.H @ psi - C.E(psi) * psi
        self.score("L1R", C.ritz_vec([psi, r])[0])
        self.score("ORC3", C.ritz_vec([psi, r] + [C.U[:, i] for i in range(3)])[0])
        for arm, srcs in (("SCRd0", REAL), ("SCRdE0", KRYLOV)):   # SCRdE0: exact F3 analogue (reported)
            phi = psi
            for t in range(1, self.a.T + 1):
                s = C.shift(phi)
                ys = [C.solve(C.source(k, phi), s, phi) for k in srcs]
                if t == 1:
                    phi = C.ritz_vec([phi, C.m.H @ phi - C.E(phi) * phi] + ys + [C.Kv(y) for y in ys])[0]
                else:
                    phi = C.ritz_vec([phi] + ys)[0]
                if t == self.a.T:
                    self.score(f"{arm}_T{t}", phi)
        self.out["cpu"]["B"] = cpu() - c0

    def responses(self, ans_gs, srcs=REAL, tag="SCRdNN", seed_off=0):
        """Phase C (srcs = REAL, tag SCRdNN) or the critic foil F3 (srcs = KRYLOV, tag F3): identical pipeline,
        budget and initialisation; only the sources differ. Returns the total response SR steps."""
        a, C, psi = self.a, self.C, self.psi
        c0 = cpu()
        nets, phi = [], psi
        steps_total = 0
        chain = []
        for t in range(1, a.T + 1):
            s = C.shift(phi)
            us = []
            for j, k in enumerate(srcs):
                b = C.source(k, phi)
                if t == 1:
                    if a.resp_init == "gs":
                        ans = perturbed_copy(C, ans_gs, a.gs_width, a.seed + 100 * (j + 1) + seed_off, 0.05)
                    else:
                        ans = rc.nnbf(C.m, a.gs_width, a.seed + 100 * (j + 1) + seed_off)
                    nets.append(ans)
                ans = nets[j]
                if t == 1 and j == 0:
                    u0 = []
                if t == 1:
                    u0.append(resp_eval(ans, C, phi, b, s, a)[2])
                iters = a.resp_cold if t == 1 else a.resp_warm
                rec = train(ans, lambda x, b=b, s=s, phi=phi: self._resp_obj(x, phi, b, s), C, a,
                            np.random.default_rng(a.seed + 1000 * t + j + seed_off), iters,
                            f"{tag} pass {t} response {k}")
                steps_total += rec["steps"]
                R, _, u, _ = resp_eval(ans, C, phi, b, s, a)
                y = C.solve(b, s, phi)
                Rstar = C.R_of(y, b, s)
                eta = 1.0 - float(unit(u) @ unit(y)) ** 2
                uw, yw = C.Uw.T @ u, C.Uw.T @ y
                eta_low = 1.0 - float(uw @ yw) ** 2 / float((uw @ uw) * (yw @ yw))
                self.out["resp"].append(dict(arm=tag, pass_=t, obs=k, R=R, R_star=Rstar, R_over_Rstar=R / Rstar,
                                             eta=eta, eta_low=eta_low, steps=rec["steps"], cpu=rec["cpu"],
                                             s_minus_E0=s - C.E0))
                log(f"    {tag} pass {t} {k}: R/R* {R / Rstar:.4f} eta {eta:.3f} eta_low {eta_low:.3f}")
                us.append(u)
            chain.append(us)
            last = "F3" if tag == "F3" else f"{tag}_T{t}"
            name = last if t == a.T else f"{tag}_T{t}"
            if t == 1:
                if tag == "SCRdNN":
                    self.score("NN0", C.ritz_vec([phi, C.m.H @ phi - C.E(phi) * phi] + u0
                                                 + [C.Kv(v) for v in u0])[0])
                phi = C.ritz_vec([phi, C.m.H @ phi - C.E(phi) * phi] + us + [C.Kv(v) for v in us])[0]
            else:
                phi = C.ritz_vec([phi] + us)[0]
            self.score(name, phi)
            if tag == "SCRdNN":
                self.out["phase"] = f"pass{t}"
            self.out["cpu"][f"{tag}_pass{t}"] = cpu() - c0
            self.dump()
        self.out["cpu"]["C" if tag == "SCRdNN" else tag] = cpu() - c0
        self.out[f"{tag}_steps_total"] = steps_total
        if tag == "SCRdNN":
            self.out["resp_steps_total"] = steps_total
            self.chain = chain
        return steps_total

    def mc_replay(self):
        """Reported arm (critic 2026-11-16): the SCRd-NN Ritz chain and the readout delta-hat with every matrix
        element SAMPLED, emulating VMC local estimators with exact draws (no Markov chain). The trained vectors are
        reused (they were trained against the exact-Ritz phi_t). Ritz matrices: x ~ pi = mean_i B_i(x)^2 over the
        normalised basis; S_ij = <B_i B_j / pi>, H_ij = <B_i (H B_j) / pi>. Readout: correlated reweighting from
        g = (phi_T^2 + psi^2) / 2, a(v) = <v (A v) / g> / <v^2 / g>. MC_DRAWS draws at each Ns in MC_NS."""
        C, psi = self.C, self.psi
        try:
            c0 = cpu()
            res = {}
            for ns in MC_NS:
                draws = []
                for d in range(MC_DRAWS):
                    rng = np.random.default_rng(10_000 * self.a.seed + 10 * ns // 1000 + d)
                    phi = psi
                    for t, us in enumerate(self.chain, start=1):
                        if t == 1:
                            basis = [phi, C.m.H @ phi - C.E(phi) * phi] + us + [C.Kv(v) for v in us]
                        else:
                            basis = [phi] + us
                        phi = mc_ritz(C, basis, ns, rng)
                    r = {}
                    for k in REAL:
                        it = self.out["items"][k]
                        r[k] = mc_delta(C, k, phi, psi, ns, rng) / it["delta"]
                    draws.append(r)
                med = {k: float(np.median([r[k] for r in draws])) for k in REAL}
                self.out["ratios"][f"MC{ns}"] = med
                res[str(ns)] = draws
                log(f"  seed {self.a.seed} MC{ns:<7d} median " + " ".join(f"{k} {med[k]:+.3f}" for k in REAL)
                    + " | draws " + " ".join(f"{k} [" + ",".join(f"{r[k]:+.2f}" for r in draws) + "]" for k in REAL))
            self.out["mc_draws"] = res
            self.out["cpu"]["MC"] = cpu() - c0
        except Exception as e:                 # a reported arm never stops the decisive run
            self.out["mc_error"] = repr(e)
            log(f"  MC replay failed (reported arm only): {e!r}")
        self.dump()

    def _resp_obj(self, ans, phi, b, s):
        R, yP, _, g = resp_eval(ans, self.C, phi, b, s, self.a)
        return R, yP, g

    def foils(self, ans_gs, steps_total):
        a, C = self.a, self.C
        c0 = cpu()
        ans1 = rc.nnbf(C.m, a.gs_width, a.seed)
        ans1.set_flat(ans_gs.flat().copy())
        rec1 = train(ans1, lambda x: gs_eval(x, C, a), C, a, np.random.default_rng(a.seed + 7), steps_total,
                     f"F1 continue ground state ({steps_total} steps)")
        self.score("F1", gs_eval(ans1, C, a)[1])
        self.out["cpu"]["F1"] = cpu() - c0
        c1 = cpu()
        cap = self.out["cpu"].get("A", 0.0) + self.out["cpu"]["C"]
        if self.out.get("gs_train", {}).get("loaded"):
            cap = self.out["cpu"]["C"] * (1.0 + a.gs_iters / max(steps_total, 1))
        ans2 = rc.nnbf(C.m, 2 * a.gs_width, a.seed + 500)
        rec2 = train(ans2, lambda x: gs_eval(x, C, a), C, a, np.random.default_rng(a.seed + 8),
                     a.gs_iters + steps_total, f"F2 width {2 * a.gs_width} (CPU cap {cap:.0f} s)", cpu_cap=cap)
        self.score("F2", gs_eval(ans2, C, a)[1])
        self.out["foil_train"] = dict(F1=rec1, F2=rec2, F2_cpu_cap=cap)
        self.out["cpu"]["F2"] = cpu() - c1

    def run(self, psi=None, ans_gs=None):
        try:
            return self._run(psi, ans_gs)
        except AssertionError as e:
            self.out["invalid"] = str(e)
            self.out["phase"] = "invalid"
            log(f"INVALID: {e}")
            self.dump()
            raise

    def _run(self, psi=None, ans_gs=None):
        if psi is None:
            ans_gs = self.ground_state()
            psi = gs_eval(ans_gs, self.C, self.a)[1]
        self.set_target(psi)
        self.out["phase"] = "A"
        if self.a.stop_after == "A":
            log(f"seed {self.a.seed}: --stop-after A (target probe); phases B-D not run")
            self.dump()
            return self.out
        if not any(self.out["items"][k]["scored"] for k in REAL):
            log(f"seed {self.a.seed}: no scored item; phases B-D skipped")
            self.out["phase"] = "done"
            self.dump()
            return self.out
        self.exact_refs()
        self.out["phase"] = "B"
        self.dump()
        steps = self.responses(ans_gs)
        self.mc_replay()
        self.responses(ans_gs, srcs=KRYLOV, tag="F3", seed_off=50)   # critic foil F3 (Gate 2), same budget
        self.foils(ans_gs, steps)
        self.out["phase"] = "done"
        log(f"seed {self.a.seed}: done (wall {time.time() - self.t0:.0f} s, cpu {cpu():.0f} s); "
            f"CPU by phase {json.dumps({k: round(v) for k, v in self.out['cpu'].items()})}")
        self.dump()
        return self.out


# ============================================================================ verdict
def items_of(recs):
    out = []
    for r in recs:
        for k in REAL:
            it = r["items"].get(k)
            if it and it["scored"]:
                out.append((r["seed"], k, r["ratios"]))
    return out


def count(items, arm):
    return sum(1 for _, k, rr in items if arm in rr and tracked(rr[arm][k]))


def verdict(recs):
    if any(r.get("invalid") for r in recs):
        return "INVALID", {}
    items = items_of(recs)
    n = len(items)
    det = dict(n_items=n)
    if n < 9:
        return "NO-TARGET", det
    if any(r.get("phase") != "done" for r in recs):
        det["incomplete_seeds"] = [r["seed"] for r in recs if r.get("phase") != "done"]
        return "INCOMPLETE", det
    c = {arm: count(items, arm) for arm in ("L1R", "ORC3", "SCRd0_T4", "SCRdE0_T4", "NN0", "SCRdNN_T1",
                                           "SCRdNN_T2", "SCRdNN_T4", "F1", "F2", "F3", "MC10000", "MC100000")}
    det["tracked"] = c
    spin_items = [rr for _, k, rr in items if k == "spin"]
    spin_ok = (not spin_items) or any(tracked(rr.get("SCRdNN_T4", {}).get("spin")) for rr in spin_items)
    det["spin_items"], det["spin_tracked_NN"] = len(spin_items), sum(
        1 for rr in spin_items if tracked(rr.get("SCRdNN_T4", {}).get("spin")))
    controls_ok = c["L1R"] <= 0.25 * n
    det["controls_ok"] = controls_ok
    nn, foil = c["SCRdNN_T4"], max(c["NN0"], c["F1"], c["F2"], c["F3"])
    if c["SCRd0_T4"] < 0.75 * n:
        v = "PREMISE-FAIL"
    elif nn < 0.5 * n:
        v = "KILL (10.1d)"
    elif foil >= nn:
        v = "KILL (Gate 2)"
    elif nn >= 0.75 * n and spin_ok and nn >= foil + math.ceil(0.25 * n):
        v = "PASS"
    else:
        v = "AMBIGUOUS"
    if not controls_ok and v in ("PASS", "AMBIGUOUS"):
        v = "INVALID (control)"
    if v == "PASS":   # scope label (critic 2026-11-16): sampled Ritz and readout at Ns = 1e5
        det["scope"] = ("sampled Ritz and readout" if c["MC100000"] >= 0.75 * n
                        else "exact-sum Ritz and readout only")
    return v, det


def checkpoint_pass2(recs):
    done = [r for r in recs if "SCRdNN_T2" in r.get("ratios", {})
            or (r.get("phase") == "done" and not any(r["items"][k]["scored"] for k in REAL))]
    if len(done) < len(recs):   # the leaf reads this check only when every task has passed pass 2
        return "WAIT", dict(tasks_at_pass2=len(done), tasks=len(recs))
    items = items_of(done)
    n = len(items)
    reach = sum(1 for _, k, rr in items if 0.5 - MARGIN_LO <= rr["SCRdNN_T2"][k] <= 2.0 + MARGIN_HI)
    kill = n > 0 and reach < 0.5 * n
    return ("EARLY KILL (10.1d)" if kill else "CONTINUE"), dict(n_done_items=n, reachable=reach,
                                                                 tasks_at_pass2=len(done), tasks=len(recs))


def synth(spec):
    """spec: list of (seed, obs, scored, {arm: ratio}). Builds task records."""
    recs = {}
    for s, k, sc, rr in spec:
        r = recs.setdefault(s, dict(seed=s, items={}, ratios={}, phase="done"))
        r["items"][k] = dict(scored=sc, delta=1.0, a_psi=0.0)
        for arm, x in rr.items():
            r["ratios"].setdefault(arm, {kk: 0.0 for kk in REAL})[k] = x
    return list(recs.values())


def selftest():
    obs = [(s, k) for s in (11, 22, 33, 44, 55, 66) for k in REAL]          # 18 items
    N = len(obs)

    def mk(nn, f1=0.1, f2=0.1, ex=1.0, l1=0.1, n0=0.1, scored=None, spin_nn=None, f3=0.1, mc=None):
        spec = []
        for i, (s, k) in enumerate(obs):
            x = nn[i]
            if k == "spin" and spin_nn is not None:
                x = spin_nn
            spec.append((s, k, True if scored is None else scored[i],
                         {"L1R": l1, "ORC3": 1.0, "SCRd0_T4": ex, "NN0": n0, "SCRdNN_T1": x, "SCRdNN_T2": x,
                          "SCRdNN_T4": x, "F1": f1, "F2": f2, "F3": f3, "SCRdE0_T4": 1.0,
                          "MC10000": x if mc is None else mc, "MC100000": x if mc is None else mc}))
        return synth(spec)
    good = [1.0] * N
    foil_close = mk(good)
    for r in foil_close:
        for k in REAL:
            if obs.index((r["seed"], k)) < 14:
                r["ratios"]["F2"][k] = 1.0
    incomplete = mk(good)
    incomplete[0]["phase"] = "pass2"
    one_spin = [k != "spin" or s == 11 for s, k in obs]
    cases = [
        ("PASS", mk(good), "PASS"),
        ("NO-TARGET 8/18", mk(good, scored=[i < 8 for i in range(N)]), "NO-TARGET"),
        ("INCOMPLETE", incomplete, "INCOMPLETE"),
        ("PREMISE-FAIL", mk(good, ex=0.1), "PREMISE-FAIL"),
        ("KILL 10.1d 8/18", mk([1.0] * 8 + [0.1] * 10), "KILL (10.1d)"),
        ("KILL Gate 2 (F1)", mk(good, f1=1.0), "KILL (Gate 2)"),
        ("KILL Gate 2 (NN0)", mk(good, n0=1.0), "KILL (Gate 2)"),
        ("KILL Gate 2 (F3)", mk(good, f3=1.0), "KILL (Gate 2)"),
        ("PASS exact-sum scope", mk(good, mc=0.1), "PASS"),
        ("AMBIGUOUS 12/18", mk([1.0] * 12 + [0.1] * 6), "AMBIGUOUS"),
        ("AMBIGUOUS no spin", mk(good, spin_nn=0.1, scored=one_spin), "AMBIGUOUS"),
        ("AMBIGUOUS foil 14/18", foil_close, "AMBIGUOUS"),
        ("INVALID control", mk(good, l1=1.0), "INVALID (control)"),
    ]
    ok = True
    for lab, recs, want in cases:
        v, det = verdict(recs)
        ok &= v == want
        if lab.startswith("PASS"):
            ok &= det.get("scope") == ("exact-sum Ritz and readout only" if "scope" in lab
                                       else "sampled Ritz and readout")
        print(f"selftest {lab:22s} expected {want:18s} got {v:18s} n={det.get('n_items')} "
              f"scope={det.get('scope')} {det.get('tracked', '')}")
    wait = mk([0.1] * N)
    wait[0]["phase"] = "A"
    del wait[0]["ratios"]["SCRdNN_T2"]
    for lab, recs, want in (("checkpoint continue", mk([0.3] * N), "CONTINUE"),
                            ("checkpoint kill", mk([0.1] * N), "EARLY KILL (10.1d)"),
                            ("checkpoint wait", wait, "WAIT")):
        v, det = checkpoint_pass2(recs)
        ok &= v == want
        print(f"selftest {lab:22s} expected {want:18s} got {v:18s} {det}")
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def aggregate(paths, which):
    recs = [json.load(open(p)) for p in paths]
    for r in recs:
        log(f"seed {r['seed']} phase {r.get('phase')} ov2 {r.get('overlap2', float('nan')):.4f}: " + " | ".join(
            f"{k}{'' if r['items'][k]['scored'] else '(ns)'} " + " ".join(
                f"{a} {x[k]:+.2f}" for a, x in r["ratios"].items()) for k in REAL if k in r["items"]))
        for q in r.get("resp", []):
            log(f"   {q.get('arm', 'SCRdNN')} pass {q['pass_']} {q['obs']}: R/R* {q['R_over_Rstar']:.4f} eta {q['eta']:.3f} "
                f"eta_low {q['eta_low']:.3f} steps {q['steps']}")
    if which == "target":
        n = len(items_of(recs))
        ov = [round(r.get("overlap2", float("nan")), 4) for r in recs]
        v = "TARGET-OK" if n >= 9 else "NO-TARGET"
        med = float(np.median([r.get("overlap2", float("nan")) for r in recs]))
        rep = med >= OV2_MIN
        log(f"T3 TARGET PROBE: {v} | scored items {n} (need >= 9) over {len(recs)} seeds; overlap^2 {ov}; median "
            f"{med:.4f} {'representative' if rep else 'NOT representative'} (critic guard: median >= {OV2_MIN}, "
            f"binding for widths below 6)")
        return 0
    if which == "pass2":
        v, det = checkpoint_pass2(recs)
        log(f"T3 CHECKPOINT (pass 2): {v} | {det}")
    else:
        v, det = verdict(recs)
        log(f"T3 VERDICT: {v} | {det}")
    return 0


# ============================================================================ toy modes
def gradcheck(args):
    C = Ctx("3x3", 3)
    a = args
    ok = True
    ans = rc.nnbf(C.m, 2, 3)
    rng = np.random.default_rng(0)
    ans.set_flat(ans.flat() + 0.05 * rng.standard_normal(ans.n_params()))
    psi = gs_eval(ans, C, a)[1]
    phi = unit(psi)
    s = C.E0 - 0.3
    for name, obj in (("ground state E", lambda x: gs_eval(x, C, a)),
                      ("response R (spin)", lambda x: (lambda R, yP, u, g: (R, yP, g))(
                          *resp_eval(x, C, phi, C.source("spin", phi), s, a)))):
        ans2 = rc.nnbf(C.m, 2, 5)
        ans2.set_flat(ans2.flat() + 0.05 * np.random.default_rng(1).standard_normal(ans2.n_params()))
        x0 = ans2.flat().copy()
        v0, _, g = obj(ans2)
        grad = 2.0 * ans2.force(g, a.chunk)
        worst = 0.0
        for trial in range(4):
            d = np.random.default_rng(10 + trial).standard_normal(len(x0))
            d /= np.linalg.norm(d)
            h = 1e-5
            ans2.set_flat(x0 + h * d); vp = obj(ans2)[0]
            ans2.set_flat(x0 - h * d); vm = obj(ans2)[0]
            ans2.set_flat(x0)
            fd = (vp - vm) / (2 * h)
            an = float(grad @ d)
            rel = abs(fd - an) / max(abs(fd), 1e-12)
            worst = max(worst, rel)
            log(f"gradcheck {name}: dir {trial} FD {fd:+.8e} analytic {an:+.8e} rel {rel:.2e}")
        ok &= worst < 1e-5
    # the minimiser of R is the exact response
    b = C.source("spin", phi)
    y = C.solve(b, s, phi)
    rng = np.random.default_rng(2)
    worse = all(C.R_of(C.Q(y + 0.05 * rng.standard_normal(C.m.D), phi), b, s) > C.R_of(y, b, s) for _ in range(5))
    log(f"gradcheck: R(y_exact) = {C.R_of(y, b, s):.6f} is below R of 5 random perturbations: {worse}")
    ok &= worse
    print(f"GRADCHECK {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def timing(args):
    C = Ctx(args.size, 3)
    a = args
    out = {}
    for lab, w in (("gs_w", a.gs_width), ("gs_2w", 2 * a.gs_width)):
        ans = rc.nnbf(C.m, w, 1)
        rec = train(ans, lambda x: gs_eval(x, C, a), C, a, np.random.default_rng(0), a.timing_steps, f"timing {lab}={w}")
        out[lab] = rec["cpu"] / rec["steps"]
    ans = rc.nnbf(C.m, a.gs_width, 1)
    phi = unit(C.psi0 + 0.05 * C.U[:, 0])
    s = C.shift(phi)
    b = C.source("spin", phi)
    pert = perturbed_copy(C, ans, a.gs_width, 7, 0.05)
    rec = train(pert, lambda x: (lambda R, yP, u, g: (R, yP, g))(*resp_eval(x, C, phi, b, s, a)), C, a,
                np.random.default_rng(0), a.timing_steps, "timing response")
    out["resp_w"] = rec["cpu"] / rec["steps"]
    c0 = cpu()
    C.solve(b, s, phi)
    out["minres"] = cpu() - c0
    log(f"TIMING {args.size}: CPU-s per SR step {json.dumps({k: round(v, 3) for k, v in out.items()})}")
    return 0


def posctrl_target(C, a, seed):
    """psi_0 + 0.05 e with e the normalised first-order response of psi_0 to pair + spin + charge."""
    w = C.sg.project(C.apply("spin", C.psi0) + C.apply("charge", C.psi0) + C.apply("pair", C.psi0))
    w -= C.psi0 * (C.psi0 @ w)
    e = unit(-C.solve(unit(w), C.E0 - 0.05, C.psi0))
    ans_gs = rc.nnbf(C.m, a.gs_width, seed)
    train(ans_gs, lambda x: gs_eval(x, C, a), C, a, np.random.default_rng(seed), a.gs_iters, "posctrl net")
    return unit(C.psi0 + 0.05 * e), ans_gs


def dry(args):
    """Pre-flight on 3x3: (1) positive control target psi_0 + 0.05 e; (2) a briefly trained width-2 state."""
    C = Ctx("3x3", 3)
    rdir = os.path.join(HERE, "results", "t3_dry")
    os.makedirs(rdir, exist_ok=True)
    recs = []
    # (1) positive control: first-order error with exact responses; trained arms on a toy state
    for seed, kind in ((1, "posctrl"), (2, "trained"), (3, "trained")):
        a = argparse.Namespace(**vars(args))
        a.seed, a.size, a.out = seed, "3x3", os.path.join(rdir, f"dry_{kind}_s{seed}.json")
        a.params_dir = os.path.join(rdir, "params")
        T = Task(a, C)
        if kind == "posctrl":
            psi, ans_gs = posctrl_target(C, a, seed)
            T.out["gs_train"] = dict(loaded=False, note="positive control: target is psi_0 + 0.05 e, not this net")
            out = T.run(psi=psi, ans_gs=ans_gs)
        else:
            a.dry_target = True
            T.a = a
            out = T.run()
        recs.append(out)
    paths = [os.path.join(rdir, f) for f in sorted(os.listdir(rdir)) if f.endswith(".json")]
    log("aggregate of the dry-run JSONs (machinery only; no physics claim):")
    aggregate(paths, "final")
    aggregate(paths, "pass2")
    pc = recs[0]["ratios"]
    ok = all(tracked(pc["SCRd0_T4"][k]) for k in REAL if recs[0]["items"][k]["scored"])
    log(f"positive control (exact responses track the first-order error on every scored item): {ok}")
    print(f"DRY {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def power(args):
    """Monte Carlo of the verdict and of the tracked fraction f of SCRd-NN. Per item, log-ratio = mu_obs + tau z_obs
    (shared by every seed of a replicate: the coherent part, as the real seeds share the sign of delta) + sigma z_item
    (training noise). mu_obs = log of the job-699756 SCRd0 means (pair 0.93, spin 1.23, charge 0.65). Spin is scored
    on 2 of every 3 seeds (as on the real states). Foils, L1R and NN0 are fixed at non-tracking values here."""
    rng = np.random.default_rng(0)
    mu0 = {"pair": math.log(0.93), "spin": math.log(1.23), "charge": math.log(0.65)}
    worlds = [("PASS-world sigma 0.15 tau 0.10", 0.15, 0.10, 0.0),
              ("PASS-world sigma 0.30 tau 0.15", 0.30, 0.15, 0.0),
              ("noisy sigma 0.50 tau 0.20", 0.50, 0.20, 0.0),
              ("charge lost (mu_charge = log 0.40)", 0.30, 0.15, math.log(0.40 / 0.65)),
              ("KILL-world (ratio ~ 0.15)", None, None, None)]
    reps = 3000
    for n_seeds in (4, 6):
        for lab, sig, tau, dch in worlds:
            tally, fr = {}, []
            for rep in range(reps):
                zo = {k: rng.standard_normal() for k in REAL}
                recs = []
                for s in range(n_seeds):
                    rr = {}
                    for k in REAL:
                        if sig is None:
                            x = 0.15 * math.exp(0.5 * rng.standard_normal())
                        else:
                            x = math.exp(mu0[k] + (dch if k == "charge" else 0.0) + tau * zo[k]
                                         + sig * rng.standard_normal())
                        rr[k] = x
                    recs.append(dict(seed=s, phase="done", items={k: dict(scored=not (k == "spin" and s % 3 == 2), delta=1.0,
                                                            a_psi=0.0) for k in REAL},
                                     ratios={"L1R": {k: 0.1 for k in REAL}, "NN0": {k: 0.1 for k in REAL},
                                             "ORC3": {k: 1.0 for k in REAL}, "SCRd0_T4": {k: 1.0 for k in REAL},
                                             "SCRdNN_T4": rr, "F1": {k: 0.2 for k in REAL},
                                             "F2": {k: 0.2 for k in REAL}}))
                v, det = verdict(recs)
                tally[v] = tally.get(v, 0) + 1
                fr.append(det["tracked"]["SCRdNN_T4"] / det["n_items"])
            fr = np.array(fr)
            key = f"{n_seeds} seeds, {lab}"
            log(f"POWER {key}: f mean {fr.mean():.3f} sd {fr.std():.3f} | " + json.dumps(
                {k: round(v / reps, 3) for k, v in sorted(tally.items())}))
    return 0


# ============================================================================ CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("run", "dry", "gradcheck", "timing", "power"), default="run")
    p.add_argument("--size", choices=("3x3", "4x3"), default="4x3")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--gs-width", type=int, default=6)
    p.add_argument("--gs-iters", type=int, default=1200)
    p.add_argument("--resp-cold", type=int, default=600)
    p.add_argument("--resp-warm", type=int, default=200)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--resp-init", choices=("gs", "uhf"), default="gs")
    p.add_argument("--sr-lr", type=float, default=0.05)
    p.add_argument("--sr-shift", type=float, default=1e-3)
    p.add_argument("--sr-samples", type=int, default=2000)
    p.add_argument("--o-chunk", type=int, default=8)
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--n-low", type=int, default=15)
    p.add_argument("--timing-steps", type=int, default=5)
    p.add_argument("--params-dir", default=os.path.join(HERE, "results", "t3", "params"))
    p.add_argument("--out", default=None)
    p.add_argument("--aggregate", default=None)
    p.add_argument("--checkpoint", choices=("final", "pass2", "target"), default="final")
    p.add_argument("--selftest-verdict", action="store_true")
    p.add_argument("--cpu-budget", type=float, default=1e9)
    p.add_argument("--stop-after", choices=("A", "done"), default="done")
    p.add_argument("--target", choices=("trained", "posctrl"), default="trained")
    a = p.parse_args(argv)
    a.dry_target = False
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.selftest_verdict:
        return selftest()
    if a.aggregate:
        return aggregate(a.aggregate.split(","), a.checkpoint)
    if a.mode == "gradcheck":
        return gradcheck(a)
    if a.mode == "timing":
        return timing(a)
    if a.mode == "power":
        return power(a)
    if a.mode == "dry":
        return dry(a)
    if a.out is None:
        a.out = os.path.join(HERE, "results", "t3", f"t3_{a.size}_w{a.gs_width}_s{a.seed}{'_probe' if a.stop_after == 'A' else ''}.json")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    T = Task(a)
    if a.target == "posctrl":
        psi, ans_gs = posctrl_target(T.C, a, a.seed)
        T.out["gs_train"] = dict(loaded=False, note="positive control: target is psi_0 + 0.05 e, not this net")
        T.run(psi=psi, ans_gs=ans_gs)
    else:
        T.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
