#!/usr/bin/env python3
"""Round 15: score R51.7's decay test (RULE_r14_decay.md, written before any arm ran) from the 18 continuations
written by ns_decay.py (job 773528) to results/r14_decay/<R51.8 arm>_<cont|dec>/run.json.
Read-only on its inputs; writes results/score_r15_decay.json (or --out). Written before any of 773528's data was read.
  python3 score_r15_decay.py [--res-dir results/r14_decay] [--ref-dir results] [--out FILE]
References (R51.8's registered ones, recomputed from the R51.8 exact arms' run.json and checked against
score_r07.json / score_r08.json): E_win(exact, fixed, steps 61-120) and G_ref(V) = E_start - E_win.
Statistic, per arm and mode: E_tail = mean of E_score (CONT: E(theta_n); DEC: E(thbar_k)) over k = 31-60;
s = (E_tail - E_win) / G_ref. Per size and mode: s per rung = seed mean where two seeds exist; N^mode(V) at s = 0.10
by ln s - ln N interpolation between the first rung with s <= 0.10 and the rung below it. Censored otherwise, as an
interval: (0, N_min] if the lowest rung has s <= 0.10; (N_max, inf) if no rung does; (N_lo, N_up] if the upper
bracketing rung has s <= 0 (ln s undefined). Conditioning (rev 2, three 10-step detrended windows over k 31-60):
fit a line to the upper rung's E_score over k = 31-60, subtract it, sd (ddof 1) of the means over k 31-40, 41-50,
51-60, in G_ref units (averaged over seeds where two exist, as score_r07.py); sd_w = max(that, |seed difference of
the upper rung's s| / sqrt 2); the size is UNRESOLVABLE if, in either mode with a bracket, s(lower) - s(upper)
<= 2 sd_w. rho(V) = N^DEC / N^CONT as an interval from the two N intervals ("censored bounds count where they
decide"). Per size: HELPS if rho_hi <= 0.5, NO-GAIN if rho_lo >= 0.8, MID if the interval lies in (0.5, 0.8),
UNDECIDED if it straddles a threshold. Verdict precedence: INCOMPLETE (a bracketing arm, or an arm whose bracketing
role cannot be excluded, is not DONE with 60 steps) > UNRESOLVABLE > DECAY-HELPS (HELPS at both) > DECAY-NO-GAIN
(NO-GAIN at both) > MIXED (otherwise; if a size is UNDECIDED the JSON lists the classes the bounds still allow).
Reported only: bounce amplitude A at S10 exact, rms over k = 32-59 of E_k - (E_{k-1} + E_{k+1}) / 2 on E_score
(and on E_iter), BOUNCE-REMOVED if A_DEC <= 0.5 A_CONT; size ratios r'_DEC, r'_CONT = N(S12') / N(S10) against
1.728 and 3.52, labelled "180-step continuation, not the registered statistic"; the DECAY-HELPS sub-branch
(r'_DEC < 3.52 or >= 3.52) is given for the editor.
"""
import argparse
import json
import math
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
THR, RHO_HELP, RHO_NOGAIN, G3, GH1 = 0.10, 0.5, 0.8, 1.728, 3.52
NEXTRA, TAIL = 60, (31, 60)
SIZES = {
    'S10': dict(exact='S5x2_N8_exact_tau0.005_s9000', ref_score='score_r07.json',
                rungs=[(4096, [9000]), (16384, [9000, 19000]), (65536, [9000, 19000])],
                arm=lambda N, sd: f'S5x2_N8_guided_N{N}_tau0.005_s{sd}'),
    'S12p': dict(exact='S6x2_N10_exact_tau0.005_s9000', ref_score='score_r08.json',
                 rungs=[(16384, [9000]), (65536, [9000]), (262144, [9000])],
                 arm=lambda N, sd: f'S6x2_N10_guided_N{N}_tau0.005_s{sd}'),
}
MODES = ('cont', 'dec')


def ref(ref_dir, sz):
    """R51.8's E_win(exact, 61-120), G_ref, E_start; asserted equal to the registered score json."""
    d = json.load(open(os.path.join(ref_dir, sz['exact'], 'run.json')))
    assert d['status'] == 'DONE' and len(d['steps']) == 120, sz['exact']
    E = np.array([s['E'] for s in d['steps']])
    Ew = float(E[60:120].mean())
    G = d['E_start'] - Ew
    sc = json.load(open(os.path.join(ref_dir, sz['ref_score'])))
    assert abs(sc['E_win_exact'] - Ew) < 1e-12 and abs(sc['G_ref_rev2'] - G) < 1e-12, (sz['ref_score'], Ew, G)
    return dict(E_start=d['E_start'], E_win=Ew, G_ref=G)


def load(res_dir, name, mode, R):
    """E_score / E_iter over k = 1..60 or None (with the reason) if the arm is not DONE with 60 steps."""
    fn = os.path.join(res_dir, f'{name}_{mode}', 'run.json')
    if not os.path.exists(fn):
        return None, 'missing'
    d = json.load(open(fn))
    st = d['steps']
    if d.get('status') != 'DONE' or len(st) < NEXTRA:
        return None, f"{d.get('status')} at k = {len(st)}"
    assert d['mode'] == mode and abs(d['tau0'] - 0.005) < 1e-15, (fn, d['mode'], d['tau0'])
    assert [s['k'] for s in st[:NEXTRA]] == list(range(1, NEXTRA + 1)), fn
    assert [s['n'] for s in st[:NEXTRA]] == list(range(121, 121 + NEXTRA)), fn
    assert abs(d['E_start'] - R['E_start']) < 1e-12, fn
    for s in st[:NEXTRA]:
        want = 0.005 if mode == 'cont' else 0.005 * (1 + s['k'] / 10) ** -0.75
        assert abs(s['tau'] - want) < 1e-15, (fn, s['k'], s['tau'])
        assert abs(s['E_score'] - (s['E_iter'] if mode == 'cont' else s['E_avg'])) == 0.0, (fn, s['k'])
    return dict(E=np.array([s['E_score'] for s in st[:NEXTRA]]),
                Ei=np.array([s['E_iter'] for s in st[:NEXTRA]])), None


def tail_s(E, R):
    return (float(E[TAIL[0] - 1:TAIL[1]].mean()) - R['E_win']) / R['G_ref']


def wsd_detr(E, R):
    k = np.arange(TAIL[0], TAIL[1] + 1)
    e = E[TAIL[0] - 1:TAIL[1]]
    r = e - np.polyval(np.polyfit(k, e, 1), k)
    return float(np.std([r[a:a + 10].mean() for a in (0, 10, 20)], ddof=1) / R['G_ref'])


def need(rows):
    """N at s = 0.10 as (value or None, lo, hi, kind, bracket index pair). rows: ascending N, all present."""
    for i, r in enumerate(rows):
        if r['s'] <= THR:
            if i == 0:
                return None, 0.0, float(r['N']), f"censored: <= {r['N']} (lowest rung s <= 0.10)", None
            a = rows[i - 1]
            if r['s'] <= 0:
                return None, float(a['N']), float(r['N']), 'censored: upper bracketing rung s <= 0', (i - 1, i)
            la, lb = math.log(a['N']), math.log(r['N'])
            nn = math.exp(la + (math.log(THR) - math.log(a['s'])) * (lb - la) / (math.log(r['s']) - math.log(a['s'])))
            return nn, nn, nn, 'interpolated', (i - 1, i)
    return None, float(rows[-1]['N']), math.inf, f"censored: > {rows[-1]['N']} (no rung s <= 0.10)", None


def score_mode(res_dir, sz, mode, R):
    rows, incomplete = [], []
    for N, seeds in sz['rungs']:
        per, bad = {}, []
        for sd in seeds:
            x, why = load(res_dir, sz['arm'](N, sd), mode, R)
            if x is None:
                bad.append(f"{sz['arm'](N, sd)}_{mode}: {why}")
            else:
                per[sd] = dict(s=tail_s(x['E'], R), sd_w_detr=wsd_detr(x['E'], R))
        row = dict(N=N, seeds=seeds, per=per, missing=bad)
        if not bad:
            row['s'] = float(np.mean([p['s'] for p in per.values()]))
            row['seed_diff'] = abs(per[9000]['s'] - per[19000]['s']) if len(seeds) == 2 else None
            row['sd_w_detr'] = float(np.mean([p['sd_w_detr'] for p in per.values()]))
        else:
            incomplete += bad
        rows.append(row)
    out = dict(rows=rows)
    if incomplete:
        # A missing rung is harmless only if a complete rung below it already has s <= 0.10 (the bracket is lower).
        first_bad = next(i for i, r in enumerate(rows) if r['missing'])
        if not any('s' in r and r['s'] <= THR for r in rows[:first_bad]):
            out.update(status='INCOMPLETE', not_done=incomplete)
            return out
        out['not_done_but_not_bracketing'] = incomplete
        rows = rows[:first_bad]
    nn, lo, hi, kind, br = need(rows)
    out.update(status='SCORED', N=nn, N_lo=lo, N_hi=hi, kind=kind)
    if br is not None:
        a, b = rows[br[0]], rows[br[1]]
        seed_term = (b['seed_diff'] / math.sqrt(2)) if b['seed_diff'] is not None else 0.0
        sdw = max(b['sd_w_detr'], seed_term)
        gap = a['s'] - b['s']
        out.update(bracket=[a['N'], b['N']], gap=gap, sd_w=sdw, sd_w_detr_upper=b['sd_w_detr'], seed_term=seed_term,
                   gap_over_sdw=gap / sdw if sdw > 0 else math.inf,
                   conditioning='RESOLVABLE' if gap > 2 * sdw else 'UNRESOLVABLE')
    else:
        out['conditioning'] = 'no bracket (censored; not conditioned)'
    return out


def ratio_iv(num, den):
    """Interval of num/den from two N intervals [lo, hi] (hi may be inf, lo may be 0)."""
    lo = num['N_lo'] / den['N_hi'] if den['N_hi'] > 0 else math.inf
    hi = num['N_hi'] / den['N_lo'] if den['N_lo'] > 0 else math.inf
    return lo, hi


def classify_rho(lo, hi):
    if hi <= RHO_HELP:
        return 'HELPS'
    if lo >= RHO_NOGAIN:
        return 'NO-GAIN'
    if lo > RHO_HELP and hi < RHO_NOGAIN:
        return 'MID'
    return 'UNDECIDED'


def bounce(res_dir, ref_dir):
    sz = SIZES['S10']
    R = ref(ref_dir, sz)
    out = {}
    for mode in MODES:
        x, why = load(res_dir, sz['exact'], mode, R)
        if x is None:
            out[mode] = dict(status=why)
            continue
        d = {}
        for key, E in (('E_score', x['E']), ('E_iter', x['Ei'])):
            k = np.arange(32, 60)  # k = 32..59 -> indices k-1
            dd = E[k - 1] - 0.5 * (E[k - 2] + E[k])
            d[key] = dict(A=float(np.sqrt(np.mean(dd ** 2))), A_over_G=float(np.sqrt(np.mean(dd ** 2)) / R['G_ref']))
        d['s_tail'] = tail_s(x['E'], R)
        out[mode] = d
    if all('E_score' in out[m] for m in MODES):
        for key in ('E_score', 'E_iter'):
            out[f'A_DEC_over_A_CONT_{key}'] = out['dec'][key]['A'] / out['cont'][key]['A']
        out['bounce_verdict'] = ('BOUNCE-REMOVED' if out['dec']['E_score']['A'] <= 0.5 * out['cont']['E_score']['A']
                                 else 'bounce not removed')
        out['note'] = 'primary A on E_score (the rule\'s scored energy); E_iter variant reported alongside'
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--res-dir', default=os.path.join(H, 'results', 'r14_decay'))
    pa.add_argument('--ref-dir', default=os.path.join(H, 'results'))
    pa.add_argument('--out', default=os.path.join(H, 'results', 'score_r15_decay.json'))
    a = pa.parse_args()
    out = dict(rule='RULE_r14_decay.md', res_dir=a.res_dir, sizes={})
    for V, sz in SIZES.items():
        R = ref(a.ref_dir, sz)
        o = dict(ref=R, modes={m: score_mode(a.res_dir, sz, m, R) for m in MODES})
        cm, dm = o['modes']['cont'], o['modes']['dec']
        if 'INCOMPLETE' in (cm['status'], dm['status']):
            o['size_status'] = 'INCOMPLETE'
        elif 'UNRESOLVABLE' in (cm.get('conditioning'), dm.get('conditioning')):
            o['size_status'] = 'UNRESOLVABLE'
        else:
            lo, hi = ratio_iv(dm, cm)
            o.update(size_status='SCORED', rho=(dm['N'] / cm['N']) if dm['N'] and cm['N'] else None,
                     rho_lo=lo, rho_hi=hi, rho_class=classify_rho(lo, hi))
        out['sizes'][V] = o
    st = [o['size_status'] for o in out['sizes'].values()]
    if 'INCOMPLETE' in st:
        out['verdict'] = 'INCOMPLETE'
    elif 'UNRESOLVABLE' in st:
        out['verdict'] = 'UNRESOLVABLE'
    else:
        cl = [o['rho_class'] for o in out['sizes'].values()]
        if all(c == 'HELPS' for c in cl):
            out['verdict'] = 'DECAY-HELPS'
        elif all(c == 'NO-GAIN' for c in cl):
            out['verdict'] = 'DECAY-NO-GAIN'
        else:
            out['verdict'] = 'MIXED'
            if 'UNDECIDED' in cl:  # the bounds do not fix the class; list what they still allow
                opts = [{'HELPS', 'MID', 'NO-GAIN'} if c == 'UNDECIDED' else {c} for c in cl]
                allowed = set()
                for x in opts[0]:
                    for y in opts[1]:
                        allowed.add('DECAY-HELPS' if x == y == 'HELPS' else 'DECAY-NO-GAIN' if x == y == 'NO-GAIN'
                                    else 'MIXED')
                out['censoring_undecided'] = dict(per_size=cl, classes_allowed=sorted(allowed))
                # Addendum 1 (RULE_r14_decay.md): an undecided size gets no class; the verdict stands only if the
                # bounds allow a single class, else UNRESOLVABLE (CENSORED-UNDECIDED)
                out['verdict'] = allowed.pop() if len(allowed) == 1 else 'UNRESOLVABLE'
                if out['verdict'] == 'UNRESOLVABLE':
                    out['unresolvable_reason'] = 'CENSORED-UNDECIDED'
        # reported only: size ratios over the 180-step continuation
        rr = {}
        for m in MODES:
            n10, n12 = out['sizes']['S10']['modes'][m], out['sizes']['S12p']['modes'][m]
            lo, hi = ratio_iv(n12, n10)
            rr[m] = dict(r=(n12['N'] / n10['N']) if n12['N'] and n10['N'] else None, r_lo=lo, r_hi=hi,
                         vs_g3_1728=('<=' if hi <= G3 else '>' if lo > G3 else 'straddles'),
                         vs_gH1_352=('<' if hi < GH1 else '>=' if lo >= GH1 else 'straddles'))
        out['size_ratios_reported'] = dict(label='180-step continuation, not the registered statistic', **rr)
        if out['verdict'] == 'DECAY-HELPS':
            out['helps_branch'] = {'<': "r'_DEC < 3.52: R51.7 LIVE at T2, next a 120-step rev-3 ladder",
                                   '>=': "r'_DEC >= 3.52: constant-factor saving at most; R51.7 settled",
                                   'straddles': "r'_DEC straddles 3.52 under censoring: branch undecided"}[
                rr['dec']['vs_gH1_352']]
    out['bounce_S10_exact_reported'] = bounce(a.res_dir, a.ref_dir)
    json.dump(out, open(a.out, 'w'), indent=1, default=lambda v: float(v) if np.isfinite(v) else str(v))
    print('verdict:', out['verdict'], out.get('censoring_undecided', ''))
    for V, o in out['sizes'].items():
        ms = o['modes']
        print(V, o['size_status'], {m: (ms[m].get('N'), ms[m].get('N_lo'), ms[m].get('N_hi'), ms[m].get('kind'),
                                        ms[m].get('conditioning'), [(r['N'], round(r['s'], 4) if 's' in r else None)
                                                                    for r in ms[m]['rows']]) for m in MODES},
              'rho', o.get('rho'), o.get('rho_lo'), o.get('rho_hi'), o.get('rho_class'))
    print('size ratios (reported):', out.get('size_ratios_reported'))
    print('bounce (reported):', out['bounce_S10_exact_reported'].get('bounce_verdict'),
          out['bounce_S10_exact_reported'].get('A_DEC_over_A_CONT_E_score'))


if __name__ == '__main__':
    main()
