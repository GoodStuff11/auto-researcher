#!/usr/bin/env python3
"""R51.32 regime-matched size comparison (RULE.md in this directory). Written before any arm of the rule ran.
Read-only on its inputs. Three modes:
  select  : stage-1 choice of the 5x2 step length tau* from the exact arms (prints tau* or NOT-MATCHED)
  owed    : the seed-9000 rungs that define N_needed(5x2, tau*) (bracketing pair, or the bounding rung); the job
            runs seed 19000 at exactly these
  score   : the verdict; writes results/score_r17_regime.json
  python3 score_r17_regime.py {select,owed,score} [--res-dir DIR] [--ref-dir DIR] [--tau T] [--grid ...]
Identity test (round 17, before submission): `score --res-dir ../P5.1_size_scaling_r04/results --tau 0.005
--grid 0.005` must reproduce score_r07.json's N_needed(5x2) = 31,283.67 (two seeds at 16k, 65k, 262k).
Statistic = R51.8 rev. 2 + addenda, unchanged except tau: E_win = mean exact energy over steps 61-120 of the exact
arm AT THE SAME tau; G_ref(tau) = E_start - E_win; s = (E_win(arm) - E_win(exact)) / G_ref; per rung the seed mean;
N_needed at s = 0.10 by ln s - ln N interpolation; censored -> interval; conditioning gap > 2 sd_w (detrended
window sd of the upper rung, max with |seed diff| / sqrt 2). Censoring and class logic imported from
score_r16_qladder.py (interval ratio; a line inside the interval -> UNRESOLVABLE, CENSORED-UNDECIDED).
Matching statistic: g1(tau) = mean over steps 61-120 of 2 tau sqrt(x^T S x) / G_ref(tau) on the exact arm
(xsx_run.py log; round-17 definition). Target T = 0.0689 = mean of the two measured 6x2 states (0.0220, 0.1157,
results/r17_gain/six.json); band [0.0220, 0.1157]. tau* = grid value minimising |ln(g1 / T)|; MATCHED iff
g1(tau*) lies in the band.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(H)
sys.path.insert(0, os.path.join(TESTS, 'P5.1_r05_gen'))
from score_r16_qladder import need, classify, allowed_classes, ln_dist, wsd, s_win  # noqa: E402

THR, G3, GH1, MARGIN = 0.10, 1.728, 3.52, 0.1
NSTEP, WIN = 120, slice(60, 120)
GRID = (0.00375, 0.0025, 0.00125)
RUNGS = (4096, 16384, 65536, 262144)
SEEDS = (9000, 19000)
START = 'S5x2_N8_U8_tp-0.25_pg11_che_seed0_theta.npy'
REF = os.path.join(TESTS, 'P5.1_size_scaling_r04', 'results')
SIX = json.load(open(os.path.join(REF, 'r17_gain', 'six.json')))
G6 = [r['gain'] for r in SIX]
TARGET, BAND = float(np.mean(G6)), (min(G6), max(G6))
N12, N10_FIXED = 118626.20, 31283.67  # score_r08.json / score_r07.json rev2 N_needed (asserted below)


def jl(p):
    return json.load(open(p))


def exact_ref(res, tau):
    fn = os.path.join(res, f'S5x2_N8_exact_tau{tau}_s9000', 'run.json')
    if not os.path.exists(fn):
        return None, 'missing'
    d = jl(fn)
    if d.get('status') != 'DONE' or len(d['steps']) < NSTEP:
        return None, f"{d.get('status')} at step {len(d['steps'])}"
    assert d['exact'] and abs(d['tau'] - tau) < 1e-15 and (d['Lx'], d['Ly'], d['Ne']) == (5, 2, 8), fn
    assert os.path.basename(d['start']) == START, fn
    E = np.array([s['E'] for s in d['steps'][:NSTEP]])
    Ew = float(E[WIN].mean())
    R = dict(tau=tau, E_start=d['E_start'], E_win=Ew, G_ref=d['E_start'] - Ew,
             m=float(np.mean(np.abs([s['dE'] for s in d['steps'][60:NSTEP]]))) / (d['E_start'] - Ew))
    t = np.arange(61, 121)
    c, cov = np.polyfit(t, E[WIN], 1, cov=True)
    se = math.sqrt(cov[0, 0])
    R['exact_slope'] = dict(slope=float(c[0]), z=float(c[0] / se), over60_in_G=float(c[0] * 60 / R['G_ref']),
                            flag='exact still descending' if c[0] < 0 and abs(c[0]) > 2 * se else 'not flagged')
    xl = os.path.join(res, f'xsx_exact_tau{tau}.jsonl')
    if tau == 0.005 and not os.path.exists(xl):  # stored round-17 rerun of R51.8's arm
        R['g1'] = float(np.mean(jl(os.path.join(REF, 'r17_gain', 'gain5x2.json'))['gain_61_120']))
    elif os.path.exists(xl):
        L = [json.loads(x) for x in open(xl) if x.strip()]
        if len(L) != NSTEP:
            return None, f'xsx log has {len(L)} lines, not {NSTEP}'
        R['g1'] = float(np.mean([2 * tau * r['ml'] for r in L[60:NSTEP]])) / R['G_ref']
    else:
        return None, 'xsx log missing'
    return R, None


def select(res, grid):
    out = dict(target=TARGET, band=BAND, per_tau={}, stored_tau0005=None)
    ok = {}
    for tau in grid:
        R, why = exact_ref(res, tau)
        out['per_tau'][str(tau)] = R if R else dict(problem=why)
        if R and R['G_ref'] > 0:
            ok[tau] = R
    if len(ok) != len(grid):
        out['selection'] = 'INCOMPLETE'
        return out, None
    ts = min(ok, key=lambda t: abs(math.log(ok[t]['g1'] / TARGET)))
    matched = BAND[0] <= ok[ts]['g1'] <= BAND[1]
    out.update(tau_star=ts, g1_star=ok[ts]['g1'], selection='MATCHED' if matched else 'NOT-MATCHED')
    return out, (ts if matched else None)


def rung(res, tau, N, sd, R):
    fn = os.path.join(res, f'S5x2_N8_guided_N{N}_tau{tau}_s{sd}', 'run.json')
    if not os.path.exists(fn):
        return None, 'missing'
    d = jl(fn)
    if d.get('status') != 'DONE' or len(d['steps']) < NSTEP:
        return None, f"{d.get('status')} at step {len(d['steps'])}"
    assert not d['exact'] and d['sampler'] == 'guided' and d['nsamp'] == N and d['seed0'] == sd, fn
    assert abs(d['tau'] - tau) < 1e-15 and abs(d['E_start'] - R['E_start']) < 1e-12, fn
    return np.array([s['E'] for s in d['steps'][:NSTEP]]), None


def ladder(res, tau, R, seeds_only=None):
    rows, bad = [], []
    for N in RUNGS:
        per = {}
        for sd in (seeds_only or SEEDS):
            E, why = rung(res, tau, N, sd, R)
            if E is not None:
                per[sd] = dict(s=s_win(E, R), sd_win_detr=wsd(E, R, True), sd_win_raw=wsd(E, R, False))
            elif sd == 9000:
                bad.append(f'N{N} s{sd}: {why}')
        row = dict(N=N, per=per)
        if 9000 in per:
            row['s'] = float(np.mean([p['s'] for p in per.values()]))
            row['seed_diff'] = abs(per[9000]['s'] - per[19000]['s']) if len(per) == 2 else None
            row['sd_win_detr'] = float(np.mean([p['sd_win_detr'] for p in per.values()]))
        rows.append(row)
    return rows, bad


def defining(rows):
    nn, lo, hi, kind, br, bound = need(rows)
    return (nn, lo, hi, kind, br, bound), ([rows[i]['N'] for i in br] if br else [rows[bound]['N']])


def score(res, tau, R):
    rows, bad = ladder(res, tau, R)
    out = dict(tau=tau, ref=R, rows=rows)
    if bad:
        out.update(status='INCOMPLETE', not_done=bad)
        return out
    (nn, lo, hi, kind, br, bound), dr = defining(rows)
    out.update(status='SCORED', N_needed=nn, N_lo=lo, N_hi=hi, kind=kind, defining_rungs=dr,
               seeds_at_defining=[sorted(next(r for r in rows if r['N'] == N)['per']) for N in dr])
    if br is not None:
        a, b = rows[br[0]], rows[br[1]]
        seed_term = b['seed_diff'] / math.sqrt(2) if b['seed_diff'] is not None else 0.0
        sdw = max(b['sd_win_detr'], seed_term)
        gap = a['s'] - b['s']
        out.update(gap=gap, sd_w=sdw, gap_over_sdw=gap / sdw if sdw > 0 else math.inf,
                   conditioning='RESOLVABLE' if gap > 2 * sdw else 'UNRESOLVABLE')
    else:
        out['conditioning'] = 'no bracket (censored; not conditioned)'
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('mode', choices=['select', 'owed', 'score'])
    pa.add_argument('--res-dir', default=os.path.join(H, 'results'))
    pa.add_argument('--tau', type=float, default=None, help='skip selection (identity test only)')
    pa.add_argument('--grid', type=float, nargs='+', default=list(GRID))
    pa.add_argument('--out', default=os.path.join(H, 'results', 'score_r17_regime.json'))
    a = pa.parse_args()
    s7, s8 = jl(os.path.join(REF, 'score_r07.json')), jl(os.path.join(REF, 'score_r08.json'))
    assert abs(s7['rev2']['N_needed'] - N10_FIXED) < 0.1 and abs(s8['rev2']['N_needed'] - N12) < 0.1
    sel, ts = select(a.res_dir, a.grid)
    if a.tau is not None:
        ts = a.tau
    if a.mode == 'select':
        print(ts if ts is not None else sel['selection'])
        json.dump(sel, open(os.path.join(a.res_dir, 'select_r17.json'), 'w'), indent=1, default=float)
        return
    if ts is None:
        print(sel['selection'])
        return
    R, why = exact_ref(a.res_dir, ts)
    assert R, why
    if a.mode == 'owed':
        rows, bad = ladder(a.res_dir, ts, R, seeds_only=(9000,))
        if bad:
            print('INCOMPLETE')
            return
        print(' '.join(str(n) for n in defining(rows)[1]))
        return
    o = score(a.res_dir, ts, R)
    out = dict(rule='tests/P5.1_r17_regime/RULE.md', selection=sel, tau_star=ts, s10=o, N_needed_6x2=N12,
               reported=dict(m_6x2=0.0485, m_5x2_tau0005=0.1533, g1_5x2_tau0005=0.3010, N_needed_5x2_tau0005=N10_FIXED))
    if o['status'] == 'INCOMPLETE':
        out['verdict'] = 'INCOMPLETE'
    else:
        lo = N12 / o['N_hi'] if o['N_hi'] > 0 else math.inf
        hi = N12 / o['N_lo'] if o['N_lo'] > 0 else math.inf
        out.update(r_m=N12 / o['N_needed'] if o['N_needed'] else None, r_m_lo=lo, r_m_hi=hi,
                   tau_effect_5x2=(o['N_needed'] / N10_FIXED) if o['N_needed'] else None)
        if o['conditioning'] == 'UNRESOLVABLE':
            out.update(verdict='UNRESOLVABLE', reason='CONDITIONING')
        else:
            c = classify(lo, hi)
            if c is None:
                out.update(verdict='UNRESOLVABLE', reason='CENSORED-UNDECIDED', classes_allowed=allowed_classes(lo, hi))
            else:
                d = {str(L): ln_dist(lo, hi, L) for L in (GH1, G3)}
                prov = any(v <= MARGIN for v in d.values())
                out.update(verdict=c, provisional=prov, margin_ln_dist=d)
    json.dump(out, open(a.out, 'w'), indent=1, default=lambda v: float(v) if np.isfinite(v) else str(v))
    print('verdict:', ('provisional ' if out.get('provisional') else '') + out['verdict'], out.get('reason', ''),
          "r'_m", out.get('r_m'), '[', out.get('r_m_lo'), ',', out.get('r_m_hi'), '] tau*', ts,
          'N_needed(5x2)', o.get('N_needed'), o.get('kind'), o.get('conditioning'),
          [(r['N'], round(r['s'], 4) if 's' in r else None) for r in o['rows']])


if __name__ == '__main__':
    main()
