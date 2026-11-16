#!/usr/bin/env python3
"""
P1.2 round-7 per-leaf critic: shared T1 premise of refill routes 8.2 (DRF-E, exact ratios) and 8.3
(DRF-WS, warm-started DRF on frozen ratios), measured on the TRAINED matched-width foil, not the anchor.
=====================================================================================================

Why: the round-7 anti-pattern says an information test on the starting determinant does not predict a
trained gain. Both refill routes feed ratio features of a trained pass 1 into a second backflow pass. A
second pass can only beat width if those features tell it where the TRAINED foil's signs are still wrong,
beyond what the foil already computes (its occupations and its own hidden units). This is the cheapest
test that decides the premise shared by 8.2 and 8.3 (SKILL 3.5.1: a shared premise is tested once, first).

Base: nnbfW of t1b_drf.py (LRAnsatz full, width 6, UHF start, 300 exact-sum SR steps, seeds 11 and 22),
retrained with the identical deterministic recipe; validity: dEs(base) equals results/t1b/<k>_nnbfW_<seed>.json
to 1e-9. Toys V = 8, 10, 12 (t1_vote.SIZES).
Label y(n) = [sgn psi_W(n) != sgn psi_0(n)] (globally aligned), weight p(n) = psi_0(n)^2; configurations with
|psi_W| < 1e-10 max dropped. Random half split (seed 7). Weighted L2-logistic models (t1_drf.fit):
  A  : occupations x = 2n - 1
  A2 : x plus the foil's own hidden activations tanh(x W1 + b1) (what width already computes)
  C  : x plus as many random +-1 features as B adds (capacity control, seed 9)
  Bex: x plus g(R_e) for every H-frame edge (hole hops and exchanges), R_e EXACT = (hc/scale) psi_W(m_e)/psi_W(n)
  Bfr: x plus g(R_e) with R_e FROZEN = the Green's-function ratios of the trained orbitals (t1b_drf DRF pass 1)
  g(R) = R / sqrt(1 + R^2), 0 on inactive edges.
Statistic per size: dAUC_ex = AUC_Bex - max(AUC_A, AUC_A2, AUC_C); dAUC_fr likewise; median over seeds.
PRE-REGISTERED RULES (written 2026-11-16 before the first run):
  counted size iff med dEs(base) >= 0.005 (the T1b rule; sizes 1 and 2 are expected to count).
  Route 8.2 premise: KILL if med dAUC_ex < 0.03 at >= 2 counted sizes including the largest;
                     PASS if med dAUC_ex >= 0.05 at every counted size and dAUC_ex(largest) >= 0.5 dAUC_ex(smallest);
                     UNDECIDED otherwise.
  Route 8.3 premise: the same rule on dAUC_fr.
  A PASS only promotes the route to its trained T1 (information is not trainability, round-7 anti-pattern);
  a KILL kills it.
  Validity: base reproduces t1b; AUC of the true label is 1; frozen ratios of the copied pass 1 at zero
  backflow output equal the exact ratios (inherited t1b CHECK 1), and the copied pass 1 reproduces the foil's
  amplitudes (max relative error < 1e-10, checked here).
Budget: RLIMIT_CPU 600 s, 1 thread.
Usage: python3 critic_r7_premise.py   (writes results/critic_r7_premise.json)
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t1b_drf as B  # noqa: E402
from t1_vote import frame_tables, dE_sign  # noqa: E402
from t1_drf import wauc, fit  # noqa: E402
from p11_core import solve_ed, uhf_orbitals  # noqa: E402
import p21_harness as H21  # noqa: E402


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (600, 630))
    sra = SimpleNamespace(sr_iters=300, sr_lr=0.05, sr_shift=1e-3, sr_samples=600, chunk=8192, o_chunk=4)
    rows, valid = [], True
    for k in (0, 1, 2):
        m = B.model(k)
        W, Vv, _ = solve_ed(m, 2, 1e-8, H21.UHF_SEED)
        psi0, E0 = Vv[:, 0], float(W[0])
        ref = uhf_orbitals(m, H21.UHF_SEED)
        Mi, hc = frame_tables(m, "H")
        for seed in (11, 22):
            nb, T = B.make("nnbfW", m, ref, seed)
            H21.optimize(nb, m, None, "none", 0, sra, np.random.default_rng(seed + 5), f"s{k} base {seed}")
            psi = nb.amplitudes(8192)
            if psi0 @ psi < 0:
                psi = -psi
            d_base = dE_sign(m.H, np.where(psi >= 0, 1.0, -1.0), psi0, E0, 3000)
            ok_t1b = abs(d_base - json.load(open(os.path.join(B.OUT, f"{k}_nnbfW_{seed}.json")))["dEs"]) < 1e-9
            dr = B.DRF(m, ref, B.WC, 1, "H", 0)
            for key in ("W1", "b1", "W2", "b2"):
                dr.p[key] = nb.p[key].copy()
            dr.p["P0u"], dr.p["P0d"] = nb.p["P0u"][0].copy(), nb.p["P0d"][0].copy()
            dr.p["W4"][:] = 0.0
            c = dr._fw(np.arange(m.D))[0]
            psi_dr = c["psi"] * np.sign(psi @ c["psi"])
            amp_err = float(np.max(np.abs(psi_dr - psi)) / np.max(np.abs(psi)))
            ok_amp = amp_err < 1e-10
            good = np.abs(psi) > 1e-10 * np.abs(psi).max()
            act = Mi >= 0
            Rex = np.where(act & good[:, None], (hc / dr.scale[None]) * psi[np.maximum(Mi, 0)], 0.0) \
                / np.where(good, psi, 1.0)[:, None]
            Rfr = c["R"]
            gex, gfr = Rex / np.sqrt(1 + Rex ** 2), Rfr / np.sqrt(1 + Rfr ** 2)
            x = c["x"]
            hid = c["h1"]
            y = (np.sign(psi) != np.sign(psi0)).astype(float)
            p = psi0 ** 2 / (psi0 @ psi0)
            sel = good & (p > 0)
            rnd = np.random.default_rng(9).choice([-1.0, 1.0], size=gex.shape)
            tr = np.random.default_rng(7).random(sel.sum()) < 0.5
            te = ~tr
            yy, pp = y[sel], p[sel]
            out = dict(size=k, V=m.V, seed=seed, dEs_base=d_base, base_matches_t1b=ok_t1b, amp_err=amp_err,
                       w_wrong=float((pp * yy).sum()), auc_true=wauc(yy[te], yy[te], pp[te]))
            feats = {"A": x, "A2": np.concatenate([x, hid], 1), "C": np.concatenate([x, rnd], 1),
                     "Bex": np.concatenate([x, gex], 1), "Bfr": np.concatenate([x, gfr], 1)}
            for lab, X in feats.items():
                X = X[sel]
                b = fit(X[tr], yy[tr], pp[tr])
                sc = np.concatenate([X[te], np.ones((te.sum(), 1))], 1) @ b
                out[f"auc_{lab}"] = wauc(sc, yy[te], pp[te])
            base_auc = max(out["auc_A"], out["auc_A2"], out["auc_C"])
            out["dAUC_ex"] = out["auc_Bex"] - base_auc
            out["dAUC_fr"] = out["auc_Bfr"] - base_auc
            valid &= ok_t1b and ok_amp and abs(out["auc_true"] - 1) < 1e-12
            rows.append(out)
            print(f"[{time.strftime('%H:%M:%S')}] " + " ".join(
                f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}" for a, b in out.items()), flush=True)
    summ = []
    for k in (0, 1, 2):
        rr = [r for r in rows if r["size"] == k]
        mb = float(np.median([r["dEs_base"] for r in rr]))
        summ.append(dict(size=k, med_base=mb, counted=mb >= 0.005,
                         dAUC_ex=float(np.median([r["dAUC_ex"] for r in rr])),
                         dAUC_fr=float(np.median([r["dAUC_fr"] for r in rr]))))
        s = summ[-1]
        print(f"  size {k}: med dEs base {mb:.5f} counted {s['counted']} | med dAUC_ex {s['dAUC_ex']:.4f} "
              f"| med dAUC_fr {s['dAUC_fr']:.4f}")
    cr = [s for s in summ if s["counted"]]
    verdicts = {}
    for key, route in (("dAUC_ex", "8.2 DRF-E"), ("dAUC_fr", "8.3 DRF-WS")):
        d = [s[key] for s in cr]
        if not valid:
            v = "INVALID"
        elif len(cr) < 2:
            v = "NO-TARGET"
        elif sum(t < 0.03 for t in d) >= 2 and d[-1] < 0.03:
            v = "KILL"
        elif all(t >= 0.05 for t in d) and d[-1] >= 0.5 * d[0]:
            v = "PASS"
        else:
            v = "UNDECIDED"
        verdicts[route] = v
        print(f"VERDICT route {route} premise on the trained matched-width foil: {v} (counted {key} "
              f"{[round(t, 4) for t in d]})")
    print(f"valid {valid}; CPU {time.process_time():.0f} s")
    json.dump(dict(rows=rows, summary=summ, verdicts=verdicts, valid=valid),
              open(os.path.join(HERE, "results", "critic_r7_premise.json"), "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
