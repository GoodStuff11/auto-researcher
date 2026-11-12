"""Synthetic check of micro_test.width_verdict: BASIN, CAPACITY (relaxed back, trade-off, oracle no
better) and UNCLEAR (unconverged cold arms) cases. No physics; decision logic only.
Usage: python3 verdict_logic_check.py"""
import micro_test as M


def res(sign_end, sign_mid, tot, conv=True, w0=None, ov0w0=0.99, plateau=True):
    sn = {"end": dict(dE_sign=sign_end, dE_tot=tot, ov0=0.94), "mid": dict(dE_sign=sign_mid, dE_tot=tot, ov0=0.94)}
    if w0 is not None:
        sn["W0"] = dict(dE_sign=w0, dE_tot=tot + 0.02, ov0=ov0w0)
    return dict(snapshots=sn, converged=conv, fid_plateau=plateau)


def case(name, W_end, W_mid, W_tot, conv=True, W0=0.004):
    R = {}
    for i, s in enumerate(M.SEEDS):
        for a, base in zip(M.COLD, (0.0125, 0.0140, 0.0160)):
            R[(a, s)] = res(base + 0.001 * i, base + 0.001 * i + 0.001, 0.052 + 0.001 * i, conv)
        R[("W", s)] = res(W_end + 0.0005 * i, W_mid + 0.0005 * i, W_tot + 0.001 * i, True, W0)
    v, info = M.width_verdict(R, 6, M.SEEDS)
    print(f"{name:28s} -> {v}  (B1 {info['B1']} B2 {info['B2']} B3 {info['B3']} B4 {info['B4']} "
          f"Bc {info['Bc']} K1 {info['K1']} K2 {info['K2']} K3 {info['K3']})")


case("basin (lower, stable)", 0.005, 0.005, 0.050)
case("capacity: relaxed back", 0.0128, 0.008, 0.053)
case("capacity: trade-off", 0.005, 0.004, 0.070)
case("unclear: cold unconverged", 0.005, 0.005, 0.050, conv=False)
case("capacity: oracle no better", 0.0115, 0.011, 0.053, W0=0.012)
