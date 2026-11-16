"""Synthetic check of micro_test.width_verdict, global_verdict and pilot_decision. Covers every
per-width label (BASIN, SIGN-DEGENERATE, CAPACITY via K2 and K3, NO-TARGET(w), UNCLEAR-CONV for
cold non-convergence / W non-convergence / cold mid->end drift, UNCLEAR-MEAS for spread and
calibration, UNCLEAR-ORACLE for K1 alone and for a missed fidelity target, UNCLEAR-NOISE,
INCOMPLETE), the global rule (NO-TARGET(w) excluded from both counts, width-trend tie-break)
and the width-6 pilot rule. No physics; decision logic only. Each line prints the label, the
expected label and OK/MISMATCH; the last line says whether all matched.
Usage: python3 verdict_logic_check.py"""
import micro_test as M

ALL_OK = True


def res(sign_end, sign_mid, tot, conv=True, w0=None, spread=1e-12, calib=1e-12, reached=True):
    sn = {"end": dict(dE_sign=sign_end, dE_tot=tot, ov0=0.94, spread=spread),
          "mid": dict(dE_sign=sign_mid, dE_tot=tot, ov0=0.94, spread=spread)}
    if w0 is not None:
        sn["W0"] = dict(dE_sign=w0, dE_tot=tot + 0.02, ov0=0.96)
    return dict(snapshots=sn, converged=conv, calib_err=calib, fid_reached=reached, n_fid=100)


def check(name, got, want):
    global ALL_OK
    ok = got == want
    ALL_OK &= ok
    print(f"{name:40s} -> {got:16s} (want {want:16s}) {'OK' if ok else 'MISMATCH'}")


def case(name, want, W_end, W_mid, W_tot, W0=0.004, cold_conv=True, W_conv=True, cold_mid=0.0002,
         spread=1e-12, calib=1e-12, base=(0.0125, 0.0140, 0.0160), reached=True, seeds=None):
    """Cold arm a, seed i: dE_sign = base_a + 0.001 i, mid = end + cold_mid, dE_tot = 0.052 + 0.001 i.
    So Cbest = base_0 + 0.001, sigma_c = 0.002, min cE = 0.053, sigma_E = 0.002, tolE = 0.004."""
    seeds = M.SEEDS if seeds is None else seeds
    R = {}
    for i, s in enumerate(seeds):
        for a, b in zip(M.COLD, base):
            R[(a, s)] = res(b + 0.001 * i, b + 0.001 * i + cold_mid, 0.052 + 0.001 * i, cold_conv,
                            spread=spread, calib=calib)
        R[("W", s)] = res(W_end + 0.0005 * i, W_mid + 0.0005 * i, W_tot + 0.001 * i, W_conv, W0,
                          spread=spread, reached=reached)
    v, info = M.width_verdict(R, 6, seeds)
    check(name, v, want)
    return v, info


print("--- per width (Cbest = 0.0135, margin = 0.004, min cE = 0.053, tolE = 0.004)")
case("basin: lower sign, strictly lower E", "BASIN", 0.005, 0.005, 0.045)
case("sign-degenerate: lower sign, equal E", "SIGN-DEGENERATE", 0.005, 0.005, 0.053)
case("sign-degenerate: E within tol above", "SIGN-DEGENERATE", 0.005, 0.005, 0.056)
case("capacity K3: lower sign, worse E", "CAPACITY", 0.005, 0.004, 0.070)
case("capacity K2: W0 lower, relaxed back", "CAPACITY", 0.0128, 0.008, 0.053)
case("unclear-oracle: K1 alone (W0 no better)", "UNCLEAR-ORACLE", 0.0115, 0.011, 0.053, W0=0.012)
case("unclear-oracle: fid target missed", "UNCLEAR-ORACLE", 0.0105, 0.0105, 0.053, reached=False)
case("unclear-noise: between B1 and K2", "UNCLEAR-NOISE", 0.0105, 0.0105, 0.053)
case("unclear-conv: a cold arm unconverged", "UNCLEAR-CONV", 0.005, 0.005, 0.045, cold_conv=False)
case("unclear-conv: W unconverged", "UNCLEAR-CONV", 0.005, 0.005, 0.045, W_conv=False)
case("unclear-conv: cold mid->end drift 1e-3", "UNCLEAR-CONV", 0.005, 0.005, 0.045, cold_mid=0.001)
case("unclear-meas: spread*dE_tot > 0.2 margin", "UNCLEAR-MEAS", 0.005, 0.005, 0.045, spread=0.02)
case("unclear-meas: |calib| > 1e-4", "UNCLEAR-MEAS", 0.005, 0.005, 0.045, calib=2e-4)
case("no-target(w): Cbest < 0.005", "NO-TARGET", 0.001, 0.001, 0.045, base=(0.003, 0.004, 0.005))
case("incomplete: 2 seeds", "INCOMPLETE", 0.005, 0.005, 0.045, seeds=M.SEEDS[:2])

print("--- global rule")
G = M.global_verdict
check("basin at 6 and 4", G({6: "BASIN", 4: "BASIN", 10: "CAPACITY"}, True, {})[0], "BASIN")
check("capacity at 6 and 10", G({6: "CAPACITY", 4: "UNCLEAR-NOISE", 10: "CAPACITY"}, True, {})[0], "CAPACITY")
check("NO-TARGET(4) excluded: 6 CAP, 10 CAP", G({6: "CAPACITY", 4: "NO-TARGET", 10: "CAPACITY"}, True, {})[0],
      "CAPACITY")
check("only width 10 eligible, BASIN", G({6: "NO-TARGET", 4: "NO-TARGET", 10: "BASIN"}, True, {})[0], "BASIN")
check("only width 4 eligible, BASIN (not 6/10)", G({6: "NO-TARGET", 4: "BASIN", 10: "NO-TARGET"}, True, {})[0],
      "UNCLEAR")
check("SD at 6 + CAP at 4", G({6: "SIGN-DEGENERATE", 4: "CAPACITY", 10: "UNCLEAR-CONV"}, True, {})[0],
      "SIGN-DEGENERATE")
check("every width NO-TARGET(w)", G({6: "NO-TARGET", 4: "NO-TARGET", 10: "NO-TARGET"}, True, {})[0], "NO-TARGET")
check("premise gate fails", G({6: "BASIN", 4: "BASIN", 10: "BASIN"}, False, {})[0], "NO-TARGET")
g, note = G({6: "BASIN", 4: "CAPACITY", 10: "UNCLEAR-ORACLE"}, True, {4: 0.02, 10: 0.008})
check("mixed -> UNCLEAR, tie-break width-limited", g, "UNCLEAR")
print(f"    note: {note}")
check("  tie-break note names CAPACITY", str(note.endswith("-> CAPACITY")), "True")

print("--- width-6 pilot rule")
P = M.pilot_decision
for v6, prem, want in [("CAPACITY", True, "CLOSE P1.3: CAPACITY"), ("SIGN-DEGENERATE", True, "CLOSE P1.3: no target"),
                       ("NO-TARGET", True, "CLOSE P1.3: NO-TARGET"), ("BASIN", False, "CLOSE P1.3: NO-TARGET"),
                       ("BASIN", True, "SUBMIT tasks 12-35"), ("UNCLEAR-CONV", True, "SUBMIT tasks 12-35"),
                       ("UNCLEAR-ORACLE", True, "SUBMIT tasks 12-35"), ("UNCLEAR-NOISE", True, "SUBMIT tasks 12-35"),
                       ("INCOMPLETE", True, "WAIT")]:
    d = P(v6, prem)
    check(f"pilot {v6} (premise {prem})", d[:len(want)], want)
print(f"    e.g. {P('UNCLEAR-CONV', True)}")
print(f"ALL LOGIC CHECKS {'PASSED' if ALL_OK else 'FAILED'}")
