#!/usr/bin/env python3
"""R51.33 two-sided regime-matched 10 -> 12 size ratio (RULE.md in this directory). Written before any 6x2 arm ran.
Read-only on its inputs except the files named below. Modes:
  s262 TAU : s of the 5x2 seed-9000 262,144 rung at TAU (job 1 runs a 1,048,576 rung iff s > 0.10)
  release  : the 6x2 tau 0.0025 exact arm is usable (OK) or not (STOP ...)
  bracket  : m6 of that arm, the two 5x2 grid points that bracket it, and the 5x2 seed-19000 runs owed at their
             defining rungs, printed as TAU:N pairs (or NOT-MATCHED)
  place    : the two 6x2 rungs placed from N5(m6) (writes results/rungs6.json; prints "N_P N_S" or a reason)
  score    : the verdict; writes results/score_r18_match2.json
  owed     : 6x2 rungs where a second seed (19000) is owed (job 3); prints NONE if none
  curve    : the 5x2 regime curve (tau, m, g1, N_needed)
  identity : R51.32 reproduced by this code path (5x2 tau 0.0025 98,387.30; 6x2 tau 0.005 118,626.2; 1.2057)
  test     : synthetic arms exercise every verdict branch
Statistic: R51.8 rev. 2 as in R51.32 (score_r17_regime.py), unchanged except tau and rungs: E_win = mean exact
energy over steps 61-120 of the exact arm at the same tau; G_ref = E_start - E_win; s = (E_win(arm) - E_win(exact))
/ G_ref, seed mean; N_needed at s = 0.10 by ln s - ln N interpolation; censoring and class logic from
score_r16_qladder.py; conditioning gap > 2 sd_w on a bracket. Matching: m = mean |dE| / G_ref over steps 61-120 of
each exact arm; N5(m6) by ln N - ln m interpolation between the two 5x2 grid points that bracket m6.
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(H)
sys.path.insert(0, os.path.join(TESTS, 'P5.1_r05_gen'))
from score_r16_qladder import need, classify, allowed_classes, ln_dist, wsd, s_win  # noqa: E402

THR, G3, GH1, MARGIN, RUNG_LN = 0.10, 1.728, 3.52, 0.1, 0.3
NSTEP, WIN = 120, slice(60, 120)
TAU6 = 0.0025
R04 = os.path.join(TESTS, 'P5.1_size_scaling_r04', 'results')
R17 = os.path.join(TESTS, 'P5.1_r17_regime', 'results')
RES = os.path.join(H, 'results')
SIZE = {'5x2': dict(pre='S5x2_N8', LxLyNe=(5, 2, 8), start='S5x2_N8_U8_tp-0.25_pg11_che_seed0_theta.npy'),
        '6x2': dict(pre='S6x2_N10', LxLyNe=(6, 2, 10), start='S6x2_N10_U8_tp-0.25_pg11_che_seed0_theta.npy')}
RUNGS5 = (4096, 16384, 65536, 262144)
BIG5 = 1048576            # 5x2 extra rung, run only where s(262,144) > 0.10
SEEDS = (9000, 19000)
GRID5 = (0.00375, 0.0025, 0.00125, 0.000625)  # tau 0.005 is stored (R51.8)
PLACE_P, PLACE_S, N_MAX6 = math.exp(-0.14), math.exp(0.13), 2097152
N5_0025 = 98387.29735765654      # R51.32 (773759) N_needed(5x2, tau 0.0025), two seeds at both defining rungs
N6_FIXED = 118626.20197989585  # R51.8 rev. 2 N_needed(6x2, tau 0.005), one seed
G6_FIXED, M6_FIXED, G1_6_FIXED = 0.0070106742743636374, 0.04848712809722166, 0.06886883533618318
STORED_5x2 = dict(tau=0.005, m=0.15332736583126177, g1=0.3010, N=31283.67280062243, N_lo=31283.67280062243,
                  N_hi=31283.67280062243, seeds='[9000, 19000] at 16k/65k/262k (R51.8)')


def jl(p):
    return json.load(open(p))


def dirs5(tau):
    """(exact-arm directory, xsx log, ladder directory) for a 5x2 grid tau."""
    if tau == 0.000625:
        return RES, os.path.join(RES, f'xsx_exact_tau{tau}.jsonl'), RES
    return R17, os.path.join(R17, f'xsx_exact_tau{tau}.jsonl'), (R17 if tau == 0.0025 else RES)


def exact_ref(res, size, tau, xsx=None, need_xsx=True):
    z = SIZE[size]
    fn = os.path.join(res, f"{z['pre']}_exact_tau{tau}_s9000", 'run.json')
    if not os.path.exists(fn):
        return None, 'missing'
    d = jl(fn)
    if d.get('status') != 'DONE' or len(d['steps']) < NSTEP:
        return None, f"{d.get('status')} at step {len(d['steps'])}"
    assert d['exact'] and abs(d['tau'] - tau) < 1e-15 and (d['Lx'], d['Ly'], d['Ne']) == z['LxLyNe'], fn
    assert os.path.basename(d['start']) == z['start'], fn
    E = np.array([s['E'] for s in d['steps'][:NSTEP]])
    if not np.all(np.isfinite(E)):
        return None, 'non-finite energy'
    Ew = float(E[WIN].mean())
    G = d['E_start'] - Ew
    if G <= 0:
        return None, f'G_ref {G:.3g} <= 0'
    R = dict(size=size, tau=tau, E_start=d['E_start'], E_win=Ew, G_ref=G,
             m=float(np.mean(np.abs([s['dE'] for s in d['steps'][60:NSTEP]]))) / G)
    t = np.arange(61, 121)
    c, cov = np.polyfit(t, E[WIN], 1, cov=True)
    se = math.sqrt(cov[0, 0])
    R['exact_slope'] = dict(slope=float(c[0]), z=float(c[0] / se), over60_in_G=float(c[0] * 60 / G),
                            flag='exact still descending' if c[0] < 0 and abs(c[0]) > 2 * se else 'not flagged')
    if xsx:
        L = [json.loads(x) for x in open(xsx) if x.strip()] if os.path.exists(xsx) else []
        if len(L) == NSTEP:
            R['g1'] = float(np.mean([2 * tau * r['ml'] for r in L[60:NSTEP]])) / G
        else:
            R['g1_problem'] = f'xsx log has {len(L)} lines, not {NSTEP}'
            if need_xsx:
                return None, R['g1_problem']
    return R, None


def rung(res, size, tau, N, sd, R):
    z = SIZE[size]
    fn = os.path.join(res, f"{z['pre']}_guided_N{N}_tau{tau}_s{sd}", 'run.json')
    if not os.path.exists(fn):
        return None, 'missing'
    d = jl(fn)
    if d.get('status') != 'DONE' or len(d['steps']) < NSTEP:
        return None, f"{d.get('status')} at step {len(d['steps'])}"
    assert not d['exact'] and d['sampler'] == 'guided' and d['nsamp'] == N and d['seed0'] == sd, fn
    assert abs(d['tau'] - tau) < 1e-15 and abs(d['E_start'] - R['E_start']) < 1e-12, fn
    assert (d['Lx'], d['Ly'], d['Ne']) == z['LxLyNe'], fn
    return np.array([s['E'] for s in d['steps'][:NSTEP]]), None


def ladder(res, size, tau, R, rungs, seeds=SEEDS):
    """Rows as R51.32: seed 9000 required at every rung; seed 19000 used where present."""
    rows, bad = [], []
    for N in rungs:
        per = {}
        for sd in seeds:
            E, why = rung(res, size, tau, N, sd, R)
            if E is not None:
                per[sd] = dict(s=s_win(E, R), sd_win_detr=wsd(E, R, True), sd_win_raw=wsd(E, R, False))
            elif sd == 9000:
                bad.append(f'{size} tau {tau} N{N} s{sd}: {why}')
        row = dict(N=N, per=per)
        if 9000 in per:
            row['s'] = float(np.mean([p['s'] for p in per.values()]))
            row['seed_diff'] = abs(per[9000]['s'] - per[19000]['s']) if len(per) == 2 else None
            row['sd_win_detr'] = float(np.mean([p['sd_win_detr'] for p in per.values()]))
        rows.append(row)
    return rows, bad


def size_need(res, size, tau, R, rungs, seeds=SEEDS):
    rows, bad = ladder(res, size, tau, R, rungs, seeds)
    if bad:
        return dict(status='INCOMPLETE', not_done=bad, rows=rows)
    nn, lo, hi, kind, br, bound = need(rows)
    o = dict(status='SCORED', rows=rows, N_needed=nn, N_lo=lo, N_hi=hi, kind=kind)
    if br is not None:
        a, b = rows[br[0]], rows[br[1]]
        seed_term = b['seed_diff'] / math.sqrt(2) if b['seed_diff'] is not None else 0.0
        sdw = max(b['sd_win_detr'], seed_term)
        gap = a['s'] - b['s']
        o.update(defining_rungs=[a['N'], b['N']], gap=gap, sd_w=sdw, gap_over_sdw=gap / sdw if sdw > 0 else math.inf,
                 conditioning='RESOLVABLE' if gap > 2 * sdw else 'UNRESOLVABLE')
    else:
        o.update(defining_rungs=[rows[bound]['N']], conditioning='no bracket (censored; not conditioned)')
    return o


def need5(tau):
    """5x2 point at a grid tau, or (None, reason)."""
    ex, xl, ld = dirs5(tau)
    R, why = exact_ref(ex, '5x2', tau, xl)
    if R is None:
        return None, f'5x2 exact tau {tau}: {why}'
    rungs = RUNGS5
    rows9, _ = ladder(ld, '5x2', tau, R, (262144,), (9000,))
    if rows9[0].get('s', 0) > THR:
        rungs = RUNGS5 + (BIG5,)
    o = size_need(ld, '5x2', tau, R, rungs)
    if o['status'] != 'SCORED':
        return None, '; '.join(o['not_done'])
    seeds = {N: sorted(r['per']) for N in o['defining_rungs'] for r in o['rows'] if r['N'] == N}
    return dict(tau=tau, m=R['m'], g1=R['g1'], G_ref=R['G_ref'], N=o['N_needed'], N_lo=o['N_lo'], N_hi=o['N_hi'],
                kind=o['kind'], conditioning=o['conditioning'], defining_seeds=seeds,
                rows=[(r['N'], r.get('s')) for r in o['rows']]), None


def curve5():
    pts, problems = [dict(STORED_5x2)], {}
    for tau in GRID5:
        p, why = need5(tau)
        if p:
            pts.append(p)
        else:
            problems[str(tau)] = why
    return sorted(pts, key=lambda p: p['m']), problems


def bracket_pts(pts, key, x):
    P = sorted(pts, key=lambda p: p[key])
    if x is None or x < P[0][key] or x > P[-1][key]:
        return None
    for a, b in zip(P, P[1:]):
        if a[key] <= x <= b[key]:
            return a, b
    return None


def interp_N(pts, key, x):
    """ln N against ln key between the two bracketing 5x2 points; interval if either is censored."""
    ab = bracket_pts(pts, key, x)
    if ab is None:
        return None
    a, b = ab
    w = (math.log(x) - math.log(a[key])) / (math.log(b[key]) - math.log(a[key])) if b[key] != a[key] else 0.0

    def f(u, v):
        if u == 0 or v == 0:
            return 0.0
        if not (math.isfinite(u) and math.isfinite(v)):
            return math.inf
        return math.exp((1 - w) * math.log(u) + w * math.log(v))
    return dict(lo=f(a['N_lo'], b['N_lo']), hi=f(a['N_hi'], b['N_hi']), between=(a['tau'], b['tau']), w=w)


def ratio(n6, d_lo, d_hi):
    lo = n6['N_lo'] / d_hi if d_hi > 0 else math.inf
    hi = n6['N_hi'] / d_lo if d_lo > 0 else math.inf
    return lo, hi


def place(pts, m6):
    iv = interp_N(pts, 'm', m6)
    if iv is None:
        return None, 'NOT-MATCHED (m6 outside the 5x2 range)'
    if not (0 < iv['lo'] and math.isfinite(iv['hi'])):
        return None, 'NOT-MATCHED (5x2 need censored at the bracketing point)'
    mid = math.sqrt(iv['lo'] * iv['hi'])
    n_p = 4096 * max(1, math.floor(G3 * PLACE_P * mid / 4096))
    n_s = min(N_MAX6, 4096 * math.ceil(GH1 * PLACE_S * mid / 4096))
    return dict(N5_star=mid, N5_lo=iv['lo'], N5_hi=iv['hi'], between_tau=iv['between'], m6=m6, rungs=[n_p, n_s],
                clamped=n_s == N_MAX6), None


def ref6(res):
    return exact_ref(res, '6x2', TAU6, os.path.join(res, f'xsx_exact6_tau{TAU6}.jsonl'), need_xsx=False)


def verdict(res, pts=None, rungs6=None):
    R6, why = ref6(res)
    out = dict(rule='tests/P5.1_r18_match2/RULE.md', ref6=R6)
    if R6 is None:
        out.update(verdict='INCOMPLETE', reason=f'6x2 exact tau {TAU6}: {why}')
        return out
    if pts is None:
        pts, problems = curve5()
        out['curve_problems'] = problems
    out['curve5'] = pts
    if rungs6 is None:
        fn = os.path.join(res, 'rungs6.json')
        if not os.path.exists(fn):
            out.update(verdict='INCOMPLETE', reason='rungs6.json missing (job 1 not finished)')
            return out
        rungs6 = jl(fn)['rungs']
    o6 = size_need(res, '6x2', TAU6, R6, tuple(rungs6))
    out['s12'] = o6
    if o6['status'] != 'SCORED':
        out.update(verdict='INCOMPLETE', reason='6x2 rungs not done')
        return out
    # reported: same-tau pair, tau effect and reference depth at 12 sites, common-reference check, g1 match
    a2, b2 = ratio(o6, N5_0025, N5_0025)
    out.update(same_tau=dict(r_lo=a2, r_hi=b2, cls=classify(a2, b2), allowed=allowed_classes(a2, b2)),
               tau_effect_6x2=dict(lo=o6['N_lo'] / N6_FIXED, hi=o6['N_hi'] / N6_FIXED),
               ref_depth_6x2=R6['G_ref'] / G6_FIXED)
    rows005, _ = ladder(R04, '6x2', 0.005, R6, (4096, 16384, 65536, 262144), (9000,))
    out['common_ref_tau005_rungs'] = [(r['N'], r.get('s')) for r in rows005]
    for key in ('m', 'g1'):
        x = R6.get(key)
        iv = interp_N(pts, key, x) if x is not None else None
        if iv is None:
            out[f'{key}_matched'] = dict(x6=x, status='outside the 5x2 range' if x is not None else 'not measured')
            continue
        a, b = ratio(o6, iv['lo'], iv['hi'])
        out[f'{key}_matched'] = dict(x6=x, N5_lo=iv['lo'], N5_hi=iv['hi'], between_tau=iv['between'], r_lo=a, r_hi=b,
                                     cls=classify(a, b), allowed=allowed_classes(a, b))
    mm = out['m_matched']
    if 'r_lo' not in mm:
        out.update(verdict='NOT-MATCHED', reason=mm['status'])
    elif o6['conditioning'] == 'UNRESOLVABLE':
        out.update(verdict='UNRESOLVABLE', reason='CONDITIONING')
    else:
        lo, hi = mm['r_lo'], mm['r_hi']
        out.update(r_mm=math.sqrt(lo * hi) if 0 < lo and math.isfinite(hi) else None, r_mm_lo=lo, r_mm_hi=hi)
        c = classify(lo, hi)
        if c is None:
            out.update(verdict='UNRESOLVABLE', reason='CENSORED-UNDECIDED', classes_allowed=allowed_classes(lo, hi))
        else:
            d = {str(L): ln_dist(lo, hi, L) for L in (GH1, G3)}
            gm = out['g1_matched']
            sens = 'r_lo' in gm and c not in gm['allowed']
            out.update(verdict=c, provisional=any(v <= MARGIN for v in d.values()), margin_ln_dist=d,
                       regime_sensitive=bool(sens))
    # second-seed clause (L72; R51.8's rung clause): owed at a single-seed defining 6x2 rung if PROVISIONAL or its
    # s lies within 0.3 ln-units of 0.10 (job 3, separately gated)
    owed = []
    for N in o6.get('defining_rungs', []):
        r = next(x for x in o6['rows'] if x['N'] == N)
        if len(r['per']) == 1 and r['s'] > 0 and (out.get('provisional') or abs(math.log(r['s'] / THR)) <= RUNG_LN):
            owed.append(N)
    out['second_seed_owed_at'] = owed
    txt = out['verdict']
    if out.get('provisional'):
        txt = 'provisional ' + txt
    if out.get('regime_sensitive'):
        g = out['g1_matched']
        txt += f" (REGIME-SENSITIVE: g1-matched ratio [{g['r_lo']:.2f}, {g['r_hi']:.2f}])"
    out['verdict_text'] = txt
    return out


def dumpj(o, fn):
    json.dump(o, open(fn, 'w'), indent=1, default=lambda v: float(v) if np.isfinite(v) else str(v))


# ---------------------------------------------------------------- synthetic test
def _write_run(d, size, tau, E_start, E, exact, N=None, sd=9000):
    z = SIZE[size]
    os.makedirs(d, exist_ok=True)
    st = [dict(E=float(e), dE=float((E[i - 1] if i else E_start) - e)) for i, e in enumerate(E)]
    Lx, Ly, Ne = z['LxLyNe']
    json.dump(dict(Lx=Lx, Ly=Ly, Ne=Ne, nsamp=N, exact=exact, sampler='guided', tau=tau, seed0=sd,
                   start='starts/' + z['start'], E_start=E_start, status='DONE', steps=st), open(d + '/run.json', 'w'))


def selftest():
    tmp = tempfile.mkdtemp(dir=os.environ.get('TMPDIR'))
    Es, G = -7.605891062343777, 0.0070
    rng = np.random.default_rng(1)
    # alternation 0.01 G per step gives m ~ 0.02; xsx ml set so g1 ~ 0.02
    base = Es - G * (1 - np.exp(-np.arange(1, 121) / 25.0)) * 1.03 + 0.005 * G * (-1.0) ** np.arange(120)
    Ew = base[WIN].mean()
    Gref = Es - Ew
    _write_run(f'{tmp}/S6x2_N10_exact_tau{TAU6}_s9000', '6x2', TAU6, Es, base, True)
    with open(f'{tmp}/xsx_exact6_tau{TAU6}.jsonl', 'w') as fh:
        for _ in range(120):
            fh.write(json.dumps(dict(ml=0.02 * Gref / (2 * TAU6), fx=0, fd=0, tau=TAU6)) + '\n')
    R6, _ = ref6(tmp)
    # synthetic 5x2 curve: N5 = 200k at m 0.0121 and 100k at m 0.051 (g1 0.018 / 0.070); 400k at m 0.003
    pts = [dict(tau=0.000625, m=0.003, g1=0.005, N=4e5, N_lo=4e5, N_hi=4e5),
           dict(tau=0.00125, m=0.0121, g1=0.018, N=2e5, N_lo=2e5, N_hi=2e5),
           dict(tau=0.0025, m=0.051, g1=0.070, N=1e5, N_lo=1e5, N_hi=1e5),
           dict(STORED_5x2)]
    pl, why = place(pts, R6['m'])
    assert pl, why
    rungs6 = pl['rungs']
    ok = True
    print(f"synthetic m6 {R6['m']:.4f} g1_6 {R6['g1']:.4f} -> N5* {pl['N5_star']:.0f}, rungs {rungs6}")

    def arm(N, sd, s):
        E = base + s * Gref + rng.normal(0, 1e-6, 120)
        E = E - (E[WIN].mean() - (Ew + s * Gref))
        _write_run(f'{tmp}/S6x2_N10_guided_N{N}_tau{TAU6}_s{sd}', '6x2', TAU6, Es, E, False, N, sd)

    def run(sa, sb, want, owed=None, extra=None, label=''):
        nonlocal ok
        for N, s in zip(rungs6, (sa, sb)):
            arm(N, 9000, s)
        o = verdict(tmp, pts, rungs6)
        good = o['verdict'] == want and (owed is None or o['second_seed_owed_at'] == owed)
        if extra:
            good &= extra(o)
        ok &= good
        print(f"synthetic {label or (sa, sb)}: {o.get('verdict_text', o['verdict'])} "
              f"r_mm=[{o.get('r_mm_lo', 0):.3f},{o.get('r_mm_hi', 0):.3f}] "
              f"same-tau {o.get('same_tau', {}).get('cls')} owed {o.get('second_seed_owed_at')}"
              f" -> {'ok' if good else 'FAIL (want ' + want + ')'}")
        return o
    run(0.08, 0.04, 'POLY-CONSISTENT', [rungs6[0]])
    run(0.05, 0.03, 'POLY-CONSISTENT', [])
    run(0.20, 0.12, 'SUPPORT-LIKE', [rungs6[1]])
    run(0.30, 0.20, 'SUPPORT-LIKE', [])
    run(0.12, 0.08, 'AMBIGUOUS', [rungs6[0], rungs6[1]])
    run(0.105, 0.0999, 'UNRESOLVABLE')
    # provisional: interpolated N6 at 3.52 x N5*
    n_t = 0.97 * GH1 * pl['N5_star']
    sa = 0.16
    sb = math.exp(math.log(sa) + (math.log(THR) - math.log(sa)) * math.log(rungs6[1] / rungs6[0])
                  / math.log(n_t / rungs6[0]))
    run(sa, sb, 'AMBIGUOUS', extra=lambda o: o.get('provisional') is True, label='near 3.52 (0.97 x line)')
    # regime-sensitive: g1-matched denominator made much smaller by a shifted g1 curve
    pts_g = [dict(p, g1=0.021) if p['tau'] == 0.0025 else p for p in pts]
    for N, s in zip(rungs6, (0.12, 0.08)):
        arm(N, 9000, s)
    o = verdict(tmp, pts_g, rungs6)
    good = o.get('regime_sensitive') is True
    ok &= good
    print(f"synthetic regime-sensitive: {o['verdict_text']} -> {'ok' if good else 'FAIL'}")
    # second seed present at the lower rung: owed list loses it
    arm(rungs6[0], 19000, 0.08)
    run(0.08, 0.04, 'POLY-CONSISTENT', [], label='seed 19000 at the lower rung')
    # not matched: m6 below the curve
    o = verdict(tmp, pts[1:], rungs6)
    good = o['verdict'] == 'NOT-MATCHED' if R6['m'] < pts[1]['m'] else True
    ok &= good
    print(f"synthetic m6 outside curve: {o['verdict']} -> {'ok' if good else 'FAIL'}")
    shutil.rmtree(f'{tmp}/S6x2_N10_guided_N{rungs6[1]}_tau{TAU6}_s9000')
    o = verdict(tmp, pts, rungs6)
    good = o['verdict'] == 'INCOMPLETE'
    ok &= good
    print(f"synthetic missing rung: {o['verdict']} -> {'ok' if good else 'FAIL'}")
    shutil.rmtree(tmp)
    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return ok


def identity():
    R5, why = exact_ref(R17, '5x2', 0.0025, os.path.join(R17, 'xsx_exact_tau0.0025.jsonl'))
    assert R5, why
    o5 = size_need(R17, '5x2', 0.0025, R5, RUNGS5)
    R6, why = exact_ref(R04, '6x2', 0.005)
    assert R6, why
    o6 = size_need(R04, '6x2', 0.005, R6, RUNGS5, (9000,))
    r = o6['N_needed'] / o5['N_needed']
    ok = (abs(o5['N_needed'] - N5_0025) < 0.01 and abs(o6['N_needed'] - N6_FIXED) < 0.01 and
          abs(r - 1.2057064599) < 1e-6 and abs(R6['m'] - M6_FIXED) < 1e-9)
    print(f"identity: N5(tau 0.0025) {o5['N_needed']:.2f} (gap/sd_w {o5['gap_over_sdw']:.1f}), "
          f"N6(tau 0.005) {o6['N_needed']:.2f} (gap/sd_w {o6['gap_over_sdw']:.1f}), ratio {r:.4f}, m6 {R6['m']:.4f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    assert abs(jl(os.path.join(R17, 'score_r17_regime.json'))['s10']['N_needed'] - N5_0025) < 0.01
    return ok


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('mode', choices=['s262', 'release', 'bracket', 'place', 'score', 'owed', 'curve', 'identity',
                                     'test'])
    pa.add_argument('tau', nargs='?', type=float)
    pa.add_argument('--out', default=os.path.join(RES, 'score_r18_match2.json'))
    a = pa.parse_args()
    if a.mode == 'test':
        sys.exit(0 if selftest() else 1)
    if a.mode == 'identity':
        sys.exit(0 if identity() else 1)
    if a.mode == 's262':
        ex, xl, ld = dirs5(a.tau)
        R, why = exact_ref(ex, '5x2', a.tau, xl)
        rows, bad = ladder(ld, '5x2', a.tau, R, (262144,), (9000,)) if R else (None, [why])
        print(rows[0]['s'] if R and not bad else f'INCOMPLETE ({why if not R else bad})')
        return
    if a.mode == 'curve':
        pts, problems = curve5()
        for p in pts:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()})
        print('problems:', problems)
        return
    if a.mode in ('release', 'bracket', 'place'):
        R6, why = ref6(RES)
        if R6 is None:
            print(f'STOP ({why})')
            return
        if a.mode == 'release':
            print('OK')
            return
        pts, problems = curve5()
        ab = bracket_pts(pts, 'm', R6['m'])
        if ab is None:
            print('NOT-MATCHED')
            return
        if a.mode == 'bracket':
            owed = [f"{p['tau']}:{N}" for p in ab if p['tau'] != 0.005 for N, sds in p['defining_seeds'].items()
                    if 19000 not in sds]
            print(' '.join(owed) or 'NONE')
            return
        pl, why = place(pts, R6['m'])
        if pl is None:
            print(why)
            return
        pl.update(g1_6=R6.get('g1'), curve5=pts, note='rungs placed by RULE.md formula from N5(m6)')
        dumpj(pl, os.path.join(RES, 'rungs6.json'))
        print(' '.join(str(n) for n in pl['rungs']))
        return
    o = verdict(RES)
    if a.mode == 'owed':
        print(' '.join(str(n) for n in o.get('second_seed_owed_at', [])) or 'NONE')
        return
    dumpj(o, a.out)
    print('verdict:', o.get('verdict_text', o['verdict']), o.get('reason', ''),
          "r'_mm", o.get('r_mm'), '[', o.get('r_mm_lo'), ',', o.get('r_mm_hi'), ']',
          'same-tau', o.get('same_tau'), 'g1-matched', o.get('g1_matched'), 'owed', o.get('second_seed_owed_at'))


if __name__ == '__main__':
    main()
