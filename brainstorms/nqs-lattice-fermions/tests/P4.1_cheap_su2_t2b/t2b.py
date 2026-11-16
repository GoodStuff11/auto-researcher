#!/usr/bin/env python3
"""
P4.1 R2 `pcg-scal` T2b: does the 12-site gain over `che` survive at 24 sites? VMC-SR driver (2026-11-16).
=========================================================================================================

Run with the lit_env interpreter, at most 4 threads (torch and numba use the same count):
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 ~/.conda/envs/lit_env/bin/python t2b.py --mode MODE [options]

This script IMPORTS, read-only, ../P4.1_cheap_su2_t2/vmc_t2.py (arms che, pcg, r1; parameter matching; P_G) and through
it the frozen ../P4.1_cheap_su2_scaling/vmc.py (geometry, Metropolis, local energy, SR step, exact sums), plus
../P4.1_cheap_su2_pfretime/pf_fast.py (numba-compiled Pfaffian, verified in that directory) and, for the reference,
../P4.1_cheap_su2_scaling/scaling_report.py (load_refs / select_ref). None of those files is modified: jobs 686990,
687302 and 698224 run them. The only change of code path is that, inside this process, vmc_t2.slogpf (torch
Parlett-Reid) is replaced by pf_fast.slogpf_fast when --pf fast (the default), exactly as retime.py does.

What is new here (compared with vmc_t2.py --mode train):
  * checkpoint energies at sizes above 12 sites are SAMPLED (vmc_t2 only allows exact checkpoints, <= 12 sites):
    at each step in --ckpts a separate Markov chain set (its own generator, seed 7000 + seed, so the training chains
    and RNG stream are untouched) is thermalised on the current parameters and --ckpt-eval samples give E +- dE
    (chain-blocked). At <= 12 sites the checkpoint energy is exact (vmc.exact_energy) unless --force-sampled.
  * a reference-aware futility stop for pcg at 24 sites (pre-registered, see the leaf section "R2 T2b design"):
    at a step t in --futility "t:mult,...", rho_ck = (E_pcg(t) - E_ref)/(E_cmp - E_ref), with E_cmp the comparator
    `che` energy from --fut-che (the finished 1000-step W-24 che run, H = 16, same instance and irrep) and E_ref the
    24-site (+,+) reference (scaling_report.load_refs(--dmrgdir), status OK) or --ref-override. The run stops with
    early_kill = t if rho_ck - 2 sigma(rho_ck) > mult. If the reference or the comparator is missing at that step,
    the clause is DISARMED for that step (logged; the run continues). The clause can kill, never promote.
  * --pf {fast, torch} selects the Pfaffian; results are identical to rounding (pfretime/logs/verify_output.txt).

Modes (--mode)
--------------
  train     Optimise ONE (--arm, --Lx, --seed) for --iters SR steps (vmc.py SR settings, Polyak average of the last
            --avg-frac, checkpoint file every --ckpt-every steps, resume). Final energy exact (--exact-eval, <= 12
            sites) or sampled (--n-eval). Writes <outdir>/<tag>.json, tag = vmc_t2 tag (S<Lx>x<Ly>_N.._pg11_<arm>_H<H>
            _seed<s>). A run capped by --max-hours writes capped = true; resubmitting the same command resumes from
            <tag>_ckpt.npz. A finished JSON is skipped.
  validate  Validity asserts for the production instances (no training): 8x3 and 4x3 open strips, N_e = 20 and 10,
            N_up = N_dn, closed-shell Fermi sea, (+,+) irrep = the DMRG characters of ../P4.1_cheap_su2_scaling/
            dmrg_irrep at 24 sites, parameter counts of che / pcg / r1 (pcg and r1 matched to che at H = 16), the
            reference loader (prints OK or PENDING), and the futility comparator file (prints present or PENDING).
  gradcheck On 4x2 (N_e = 6) with P_G (1,1) and without: for pcg (compiled Pfaffian) and r1 (che: 686990 smoke (d)), the exact gradient
            from per-sample log-derivatives equals the numpy round-3 State gradient (<= 1e-7 relative), and the pcg
            log-derivatives with the compiled Pfaffian equal the torch-Pfaffian ones (<= 1e-9 relative).
  pipeline  Toy end-to-end pre-flight on 4x2 into --outdir: che, pcg, r1 with --force-sampled checkpoints and sampled
            final energies ("24-site slot", subdir s24) and with exact checkpoints and final energies ("12-site slot",
            subdir s12), --iters steps each; plus a futility control (pcg with --ref-override and mult -1, which must stop
            at its first checkpoint with early_kill set). Then run t2b_report.py --mode preflight --resdir <outdir>.

Options
-------
  --mode {train, validate, gradcheck, pipeline}   (required)
  --pf {fast, torch} (fast)        Pfaffian kernel for pcg
  --ckpts STR ("")                 comma list of steps with a checkpoint energy, e.g. "750,1500,2250"
  --ckpt-eval INT (32768)          samples per sampled checkpoint energy
  --force-sampled                  sampled checkpoints even at <= 12 sites (toy pipeline only)
  --futility STR ("")              "t:mult,..." pre-registered futility multiples (pcg only; see above)
  --fut-che PATH ("")              comparator che JSON for the futility clause (vmc_t2 W-24 H = 16 result)
  --dmrgdir PATH (../P4.1_cheap_su2_scaling/dmrg)   DMRG reference JSONs (687302)
  --ref-override STR ("")          "E,sigma": reference used instead of the DMRG loader (toy pipeline only)
  every vmc_t2.py train option is accepted unchanged: --arm {che, pcg, r1, foil}, --Lx, --Ly, --Ne, --U, --tp,
  --pg, --width, --match-width, --seed, --iters, --lr, --lr-decay, --avg-frac, --sr-norm, --diag-shift, --max-step,
  --n-chains, --sweeps-per-step, --therm-sweeps, --n-eval, --exact-eval, --chunk, --grad-chunk, --threads,
  --ckpt-every, --max-hours, --outdir (default <script dir>/results).
"""
import argparse
import json
import math
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
T2 = os.path.join(HERE, "..", "P4.1_cheap_su2_t2")
SCAL = os.path.join(HERE, "..", "P4.1_cheap_su2_scaling")
PFR = os.path.join(HERE, "..", "P4.1_cheap_su2_pfretime")
for p in (T2, PFR, SCAL):
    sys.path.insert(0, p)

import numba  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import vmc_t2 as T  # noqa: E402  (read only; job 698224 runs it)
from pf_fast import slogpf_fast  # noqa: E402

V = T.V  # frozen vmc.py (read only; job 686990 runs it)
log = V.log
SLOGPF_TORCH = T.slogpf
torch.set_default_dtype(torch.float64)


def use_pf(kind):
    T.slogpf = slogpf_fast if kind == "fast" else SLOGPF_TORCH


# ============================================================================ reference and ratio
def load_ref(dmrgdir, L, override=""):
    """(E_ref, sigma_ref) for the (+,+) sector at L sites, or None if not available / not valid."""
    if override:
        E, s = (float(v) for v in override.split(","))
        return E, s
    if L == 12:
        return V.E0_4x3, 0.0
    try:
        import scaling_report as SR
        r = SR.load_refs(dmrgdir).get(L)
    except Exception as e:  # noqa: BLE001
        log(f"reference loader failed: {e}")
        return None
    if r is None or r[2] != "OK":
        return None
    return float(r[0]), float(r[1])


def rho_sigma(Ep, dEp, Ec, dEc, Er, sr):
    """rho = (Ep - Er)/(Ec - Er) and its 1-sigma error, with the shared reference error propagated exactly
    (d rho/d Er = (rho - 1)/(Ec - Er))."""
    den = Ec - Er
    rho = (Ep - Er) / den
    sig = math.sqrt(dEp ** 2 + (rho * dEc) ** 2 + ((1.0 - rho) * sr) ** 2) / abs(den)
    return rho, sig


# ============================================================================ checkpoint energy
def ckpt_energy(m, b, th, seed, n_eval, exact_sec):
    """exact energy (dE = 0) if exact_sec is given, else sampled with a fresh chain set (training chains untouched)."""
    if exact_sec is not None:
        return float(V.exact_energy(m, th, exact_sec)[0]), 0.0, int(exact_sec.dim)
    run = V.Run(m, b, 7000 + seed)
    run.start_chains(th)
    E, dE, ns = run.evaluate(th, n_eval)
    return float(E), float(dE), int(ns)


# ============================================================================ train
def train(a):
    torch.set_num_threads(a.threads)
    numba.set_num_threads(a.threads)
    use_pf(a.pf)
    os.makedirs(a.outdir, exist_ok=True)
    m = T.build(a)
    tag = T.tag_of(a, m)
    fout = os.path.join(a.outdir, tag + ".json")
    if os.path.exists(fout):
        old = json.load(open(fout))
        if not (old.get("capped") and old.get("iters_done", 0) < a.iters):
            log(f"{fout} exists; skipping")
            return
        log(f"{fout} was capped at step {old['iters_done']}; resuming from the checkpoint (resubmission)")
    log(f"{tag}: L={m.L} Ne={m.Ne} H={m.H} P={m.nparam} (che@{a.match_width} P={V.count('che', m.L, m.Ne, a.match_width)})"
        f" threads={a.threads} pf={a.pf} host={socket.gethostname()}")
    exact_ok = m.L <= 12
    ckpts = sorted(int(s) for s in a.ckpts.split(",") if s.strip())
    fut = {int(k): float(v) for k, v in (s.split(":") for s in a.futility.split(",") if s.strip())}
    assert set(fut) <= set(ckpts), "every futility step must be a checkpoint step"
    assert not fut or m.arm == "pcg", "the futility clause is pre-registered for pcg only"
    need_exact = exact_ok and (a.exact_eval or (ckpts and not a.force_sampled))
    sec = V.exact_sector(m) if need_exact else None
    E0 = V.E0_4x3 if (m.Lx, m.Ly, m.Ne) == (4, 3, 10) else None
    if E0 is None and exact_ok and m.L <= 8:  # toy cluster: global S^z = 0 ground energy (ED), as vmc_t2
        from scipy.sparse.linalg import eigsh
        s_ = sec if sec is not None else V.exact_sector(m)
        E0 = float(eigsh(s_.Hsparse(), k=1, which="SA")[0][0])
    ck = os.path.join(a.outdir, tag + "_ckpt.npz")
    run = V.Run(m, a, a.seed)
    traj, t_used, it0, cke = [], 0.0, 0, {}
    avg_start = int(round(a.iters * (1.0 - a.avg_frac)))
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        th = torch.tensor(z["theta"])
        it0, t_used = int(z["it"]), float(z["t_used"])
        traj = json.loads(str(z["traj"]))
        cke = json.loads(str(z["cke"]))
        avg_sum, avg_n = torch.tensor(z["avg_sum"]), int(z["avg_n"])
        log(f"resumed from {ck} at step {it0} (t_used {t_used / 3600:.2f} h)")
    else:
        th = m.init(a.seed)
        avg_sum, avg_n = torch.zeros_like(th), 0
    acc0 = run.start_chains(th)
    log(f"thermalised: acceptance {acc0:.3f}")
    t_start = time.time()
    capped, early_kill, it = False, None, it0
    for it in range(it0 + 1, a.iters + 1):
        th, info = run.step(th, V.lr_at(a, it))
        if it > avg_start:
            avg_sum += th
            avg_n += 1
        el = t_used + time.time() - t_start
        if it % 10 == 0 or it == 1:
            traj.append([it, info["E"], info["Estd"], el])
        if it % 50 == 0 or it == 1:
            log(f"it {it} E {info['E']:.6f} std {info['Estd']:.3f} acc {info['acc']:.3f} |upd| {info['upd']:.2e} "
                f"t(mcmc/eloc/grad/sr) {info['t_mcmc']:.2f}/{info['t_eloc']:.2f}/{info['t_grad']:.2f}/"
                f"{info['t_sr']:.2f}s elapsed {el / 3600:.3f} h")
        if not np.isfinite(info["E"]):
            raise RuntimeError("non-finite energy")
        if it in ckpts and str(it) not in cke:
            t0 = time.time()
            Ec_, dEc_, ns_ = ckpt_energy(m, a, th, a.seed, a.ckpt_eval, None if a.force_sampled else sec)
            rec = dict(E=Ec_, dE=dEc_, n=ns_, wall=el,
                       eps=((Ec_ - E0) / abs(E0) if E0 is not None else None))
            msg = f"CHECKPOINT it {it}: E {Ec_:.6f} +- {dEc_:.6f} ({ns_} {'exact' if dEc_ == 0 else 'samples'})"
            if it in fut:
                ref = load_ref(a.dmrgdir, m.L, a.ref_override)
                cmpf = json.load(open(a.fut_che)) if a.fut_che and os.path.exists(a.fut_che) else None
                if ref is None or cmpf is None or cmpf.get("capped"):
                    rec["futility"] = "disarmed (" + ("no valid reference" if ref is None else "no comparator") + ")"
                    msg += f"; futility at {it} DISARMED ({rec['futility']})"
                else:
                    rho, sg = rho_sigma(Ec_, dEc_, cmpf["E"], cmpf["dE"], ref[0], ref[1])
                    rec.update(rho_ck=rho, rho_ck_sigma=sg, mult=fut[it], E_ref=ref[0], sigma_ref=ref[1],
                               E_cmp=cmpf["E"], dE_cmp=cmpf["dE"])
                    msg += f"; rho_ck {rho:.3f} +- {sg:.3f} vs futility multiple {fut[it]}"
                    if rho - 2 * sg > fut[it]:
                        early_kill = it
                        rec["futility"] = "fired"
                        msg += " -> EARLY-KILL (pre-registered futility)"
                    else:
                        rec["futility"] = "armed, not fired"
            cke[str(it)] = rec
            log(msg + f" ({time.time() - t0:.0f} s)")
        if it % a.ckpt_every == 0 or it == a.iters or early_kill:
            np.savez(ck, theta=th.numpy(), it=it, t_used=el, traj=json.dumps(traj), cke=json.dumps(cke),
                     avg_sum=avg_sum.numpy(), avg_n=avg_n)
        if early_kill:
            break
        if el > a.max_hours * 3600:
            capped = True
            log(f"wall cap {a.max_hours} h reached at step {it}")
            break
    wall = t_used + time.time() - t_start
    np.save(os.path.join(a.outdir, tag + "_theta_last.npy"), th.numpy())
    if avg_n > 0 and not early_kill:
        th = avg_sum / avg_n
    np.save(os.path.join(a.outdir, tag + "_theta.npy"), th.numpy())
    t0 = time.time()
    S2 = None
    if a.exact_eval and exact_ok:
        sec = sec if sec is not None else V.exact_sector(m)
        E, psi = V.exact_energy(m, th, sec)
        S2 = float(np.sum(psi * sec.S2(psi)))
        dE, ns = 0.0, sec.dim
    else:
        E, dE, ns = run.evaluate(th, a.n_eval)
    log(f"final E ({'current state' if early_kill else f'average of last {avg_n} steps'}) {E:.7f} +- {dE:.7f} "
        f"({ns} samples, {time.time() - t0:.0f} s)" + (f", <S^2> {S2:.1e}" if S2 is not None else ""))
    res = dict(tag=tag, arm=m.arm, Lx=m.Lx, Ly=m.Ly, L=m.L, Ne=m.Ne, U=a.U, tp=a.tp, pg=a.pg, seed=a.seed, H=m.H,
               nparam=m.nparam, iters_done=it, iters=a.iters, capped=capped, early_kill=early_kill, wall=wall,
               threads=a.threads, pf=a.pf, E=float(E), dE=float(dE), n_eval=ns, exact_eval=bool(a.exact_eval and exact_ok),
               S2=S2, E0=E0, eps=((E - E0) / abs(E0) if E0 is not None else None), ckpt=cke, avg_n=avg_n, traj=traj,
               hyper=dict(lr=a.lr, lr_decay=a.lr_decay, avg_frac=a.avg_frac, diag_shift=a.diag_shift,
                          max_step=a.max_step, n_chains=a.n_chains, sweeps_per_step=a.sweeps_per_step),
               host=socket.gethostname())
    with open(fout, "w") as f:
        json.dump(res, f)
    log(f"wrote {fout}")


# ============================================================================ validate
def validate(a):
    torch.set_num_threads(a.threads)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        log(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    for Lx, Ne in ((8, 20), (4, 10)):
        b = parse_args(["--mode", "validate", "--Lx", str(Lx), "--pg=1,1"])
        L = 3 * Lx
        chk(f"{Lx}x3: N_e = vmc.default_ne(L) (filling 5/6)", V.default_ne(L) == Ne, f"N_e = {V.default_ne(L)}")
        e = np.linalg.eigh(-V.strip_T(Lx, 3, -0.25)[0])[0]
        chk(f"{Lx}x3: closed-shell Fermi sea", e[Ne // 2] - e[Ne // 2 - 1] > 1e-8, f"gap {e[Ne // 2] - e[Ne // 2 - 1]:.3f}")
        P = {}
        for arm in ("che", "pcg", "r1"):
            m = T.build(b, arm, Lx)
            chk(f"{Lx}x3 {arm}: N_up = N_dn = N_e/2 <= L and P_G = (+,+)",
                m.Nu == m.Nd == m.Ne // 2 <= m.L and V.pg_of(b) == (1, 1), f"H={m.H} P={m.nparam}")
            P[arm] = m.nparam
        chk(f"{Lx}x3: pcg and r1 parameter counts within 5% of che (H = 16)",
            all(abs(P[k] / P["che"] - 1) < 0.05 for k in ("pcg", "r1")), str(P))
    d = json.load(open(os.path.join(SCAL, "dmrg_irrep", "dmrg_S8x3_N20_Sz20_U8_tp-0.25.json")))
    chk("24 sites: DMRG characters <Inv>, <My> >= 0.95 -> irrep (+,+) = the arms' --pg=1,1",
        d["chi_inv"] >= 0.95 and d["chi_My"] >= 0.95, f"<Inv> {d['chi_inv']:.4f}, <My> {d['chi_My']:.4f}")
    r = load_ref(a.dmrgdir, 24)
    log(f"  24-site (+,+) reference (687302, scaling_report.select_ref): "
        + ("PENDING (scoring will be INCOMPLETE-REFERENCE until it lands)" if r is None else f"E {r[0]:.6f} +- {r[1]:.1e}"))
    f = os.path.join(T2, "results", "w24", "S8x3_N20_U8_tp-0.25_pg11_che_H16_seed0.json")
    if os.path.exists(f):
        w = json.load(open(f))
        chk("futility comparator (W-24 che H=16) is the same instance and irrep, uncapped",
            (w["Lx"], w["Ne"], w["pg"], w["H"], w["capped"]) == (8, 20, "1,1", 16, False),
            f"E {w['E']:.5f} +- {w['dE']:.5f}")
    else:
        log("  futility comparator (W-24 che H=16): PENDING (the clause stays disarmed until it exists)")
    log(f"VALIDATE {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================ gradcheck (toy)
def gradcheck(a):
    import r3_micro_test as M3
    import r2_micro_test as R
    torch.set_num_threads(a.threads)
    numba.set_num_threads(a.threads)
    ok = True
    for pg in (None, (1, 1)):
        b = parse_args(["--mode", "gradcheck", "--Lx", "4", "--Ly", "2", "--Ne", "6",
                        "--pg", "none" if pg is None else "1,1"])
        sec = V.exact_sector(T.build(b, "che", 4))
        nU, nD = V.exact_configs(sec)
        sgp = None
        if pg is not None:
            sgp = M3.SG(sec)
            sgp.chi = np.array([1.0, pg[1], pg[0], pg[0] * pg[1]])
        for arm in ("pcg", "r1"):
            use_pf("fast")
            m = T.build(b, arm, 4)
            th = m.init(1) + 0.05 * torch.randn(m.nparam, generator=torch.Generator().manual_seed(5))
            E_t, psi_t = V.exact_energy(m, th, sec)
            psi_t = psi_t.ravel()
            l, s = m.logpsi_batched(th, nU, nD, 8192)
            fin = torch.isfinite(l)
            El = torch.zeros(len(l))
            El[fin] = m.local_energy(th, nU[fin], nD[fin], l[fin], s[fin], 8192)
            O = torch.zeros(len(l), m.nparam)
            O[fin] = m.logderiv(th, nU[fin], nD[fin], 512)
            w = psi_t ** 2
            g_t = 2.0 * O.numpy().T @ (w * (El.numpy() - E_t))
            na = R.DetArm(sec, arm, m.H, 0)
            st = M3.State(sec, na, arm == "pcg", sgp, 16384)
            E_n, g_n, _ = st.energy_grad(th.numpy())
            rel = float(np.linalg.norm(g_t - g_n) / np.linalg.norm(g_n))
            good = rel <= 1e-7 and abs(E_n - E_t) <= 1e-9 * abs(E_t)
            ok &= good
            log(f"  [{'PASS' if good else 'FAIL'}] P_G={pg} {arm}: |g_fastpf - g_numpy|/|g_numpy| = {rel:.1e} (<= 1e-7), "
                f"|E diff| {abs(E_n - E_t):.1e}")
            if arm == "pcg":
                use_pf("torch")
                O2 = torch.zeros(len(l), m.nparam)
                O2[fin] = m.logderiv(th, nU[fin], nD[fin], 512)
                rel2 = float((O - O2).abs().max() / O2.abs().max())
                good = rel2 <= 1e-9
                ok &= good
                log(f"  [{'PASS' if good else 'FAIL'}] P_G={pg} pcg: max |O_fast - O_torch| / max |O_torch| = {rel2:.1e} (<= 1e-9)")
                use_pf("fast")
    log(f"GRADCHECK {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================ pipeline (toy pre-flight)
def pipeline(a):
    base = ["--mode", "train", "--Lx", "4", "--Ly", "2", "--Ne", "6", "--pg=1,1", "--iters", str(a.iters),
            "--n-chains", "128", "--therm-sweeps", "10", "--threads", str(a.threads), "--ckpt-every", "5"]
    h = max(1, a.iters // 4)
    for slot, extra in (("s24", ["--force-sampled", "--ckpt-eval", "2048", "--n-eval", "4096"]),
                        ("s12", ["--exact-eval"])):
        for arm in ("che", "pcg", "r1"):
            train(parse_args(base + ["--arm", arm, "--outdir", os.path.join(a.outdir, slot), "--ckpts", f"{h},{2 * h}"]
                             + extra))
    # futility control: a real pcg run whose pre-registered multiple is -1 must stop at its first checkpoint
    che = json.load(open(next(os.path.join(a.outdir, "s24", f) for f in os.listdir(os.path.join(a.outdir, "s24"))
                               if "_che_" in f and f.endswith(".json"))))
    train(parse_args(base + ["--arm", "pcg", "--outdir", os.path.join(a.outdir, "futility_ctrl"), "--force-sampled",
                             "--ckpt-eval", "2048", "--n-eval", "4096", "--ckpts", f"{h},{2 * h}",
                             "--futility", f"{h}:-1.0", f"--ref-override={che['E0']},0.0",
                             "--fut-che", os.path.join(a.outdir, "s24", che["tag"] + ".json")]))
    # disarmed control: same, but no reference available (Lx=4, Ly=2 has none in the DMRG loader) -> must NOT stop
    train(parse_args(base + ["--arm", "pcg", "--outdir", os.path.join(a.outdir, "disarmed_ctrl"), "--force-sampled",
                             "--ckpt-eval", "2048", "--n-eval", "4096", "--ckpts", f"{h},{2 * h}",
                             "--futility", f"{h}:-1.0", "--dmrgdir", os.path.join(a.outdir, "no_such_dir"),
                             "--fut-che", os.path.join(a.outdir, "s24", che["tag"] + ".json")]))


# ============================================================================ args
def parse_args(argv=None):
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--mode", required=True, choices=["train", "validate", "gradcheck", "pipeline"])
    p.add_argument("--pf", default="fast", choices=["fast", "torch"])
    p.add_argument("--ckpts", default="")
    p.add_argument("--ckpt-eval", type=int, default=32768)
    p.add_argument("--force-sampled", action="store_true")
    p.add_argument("--futility", default="")
    p.add_argument("--fut-che", default="")
    p.add_argument("--dmrgdir", default=os.path.join(SCAL, "dmrg"))
    p.add_argument("--ref-override", default="")
    mine, rest = p.parse_known_args(argv)
    if "--outdir" not in " ".join(rest):
        rest = rest + ["--outdir", os.path.join(HERE, "results")]
    a = T.parse_args(["--mode", "train"] + rest)
    for k, v in vars(mine).items():
        setattr(a, k, v)
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.mode == "train":
        train(a)
    elif a.mode == "validate":
        sys.exit(0 if validate(a) else 1)
    elif a.mode == "gradcheck":
        sys.exit(0 if gradcheck(a) else 1)
    else:
        pipeline(a)


if __name__ == "__main__":
    main()
