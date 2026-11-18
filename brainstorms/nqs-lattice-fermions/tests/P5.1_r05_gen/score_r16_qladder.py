#!/usr/bin/env python3
"""Round 16: score R51.11's size ladder with q ~ |psi| (RULE_r14_qladder.md incl. Addendum 1; job 773500).
Written before any of 773500's output was read. Read-only on its inputs; writes results/score_r16_qladder.json (or --out).
  python3 score_r16_qladder.py [--res-dir results] [--ref-dir ../P5.1_size_scaling_r04/results] [--out FILE]
Arms (ns_q.py --qmix 1.0 via submit_r14_qladder.sbatch): <res-dir>/q1L_S{5,6}x2_N<N>_s<seed>/run.json (ns_run.py's
record). S10 4,096 seed 9000 is the reused storm.py rung <res-dir>/storm_N4096_a1.0_q1.0/run.json (no status field;
complete iff it has 120 steps).
References: R51.8's exact arms, E_win(exact) = mean E over steps 61-120, G_ref = E_start - E_win, recomputed and
asserted equal to score_r07.json / score_r08.json to 1e-12.
Statistic (R51.8 rev 2 + addenda): s = (E_win(arm) - E_win(exact)) / G_ref; per rung s = seed mean where two seeds
exist (each seed its own arm, the storm rung included); N_q(V) at s = 0.10 by ln s - ln N interpolation between the
first rung with s <= 0.10 and the rung below it. Censored N is an interval (score_r15_decay interpretation (a)):
(0, N_min] if the lowest rung has s <= 0.10; (N_max, inf) if none does; (N_lo, N_up] if the upper bracketing rung has
s <= 0. Conditioning only where a bracket exists ((b)): sd_w = max(sd_win_detr of the upper rung (sd, ddof 1, of its
three 20-step window means over steps 61-120 after removing a linear fit, in G_ref units, averaged over seeds as
score_r07.py), |seed difference of the upper rung's s| / sqrt 2); the size fails (UNRESOLVABLE) unless
s(lower) - s(upper) > 2 sd_w. Raw (undetrended) sd_w reported alongside. A not-DONE arm makes the size INCOMPLETE
unless a complete lower rung already has s <= 0.10 ((c)).
Verdict: r'_q = N_q(S12') / N_q(S10) as an interval; POLY-CONSISTENT if r_hi <= 1.728, SUPPORT-LIKE if r_lo >= 3.52,
AMBIGUOUS if 1.728 < r_lo and r_hi < 3.52; an interval straddling a line gives no class -> UNRESOLVABLE with reason
CENSORED-UNDECIDED (as RULE_r14_decay.md Addendum 1). Precedence INCOMPLETE > UNRESOLVABLE > class.
Addendum 1 clauses: PROVISIONAL if some r in [r_lo, r_hi] has |ln r - ln 3.52| <= 0.1 or |ln r - ln 1.728| <= 0.1
(for a point: the plain clause); rung clause: an S12' rung defining N_q(S12') (both bracketing rungs; the single
bounding rung if censored) with s > 0 and |ln s - ln 0.10| <= 0.3 owes a second seed at that rung.
Reported only: gain N_guided(V) / N_q(V) (guided 31,280 and 118,600, as the rule) as an interval.
"""
import argparse
import json
import math
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
THR, G3, GH1, MARGIN, RUNG_LN = 0.10, 1.728, 3.52, 0.1, 0.3
NSTEP, WIN = 120, slice(60, 120)
STORM = 'storm_N4096_a1.0_q1.0'
SIZES = {
    'S10': dict(exact='S5x2_N8_exact_tau0.005_s9000', ref_score='score_r07.json', geo=(5, 2, 8),
                start='S5x2_N8_U8_tp-0.25_pg11_che_seed0_theta.npy', guided=31280,
                rungs=[(4096, [9000, 19000]), (16384, [9000, 19000]), (65536, [9000])]),
    'S12p': dict(exact='S6x2_N10_exact_tau0.005_s9000', ref_score='score_r08.json', geo=(6, 2, 10),
                 start='S6x2_N10_U8_tp-0.25_pg11_che_seed0_theta.npy', guided=118600,
                 rungs=[(4096, [9000]), (16384, [9000]), (65536, [9000])]),
}


def arm_name(V, N, sd):
    if V == 'S10' and N == 4096 and sd == 9000:
        return STORM
    lx, ly, _ = SIZES[V]['geo']
    return f'q1L_S{lx}x{ly}_N{N}_s{sd}'


def ref(ref_dir, sz):
    d = json.load(open(os.path.join(ref_dir, sz['exact'], 'run.json')))
    assert d['status'] == 'DONE' and len(d['steps']) == NSTEP, sz['exact']
    E = np.array([s['E'] for s in d['steps']])
    Ew = float(E[WIN].mean())
    G = d['E_start'] - Ew
    sc = json.load(open(os.path.join(ref_dir, sz['ref_score'])))
    assert abs(sc['E_win_exact'] - Ew) < 1e-12 and abs(sc['G_ref_rev2'] - G) < 1e-12, (sz['ref_score'], Ew, G)
    return dict(E_start=d['E_start'], E_win=Ew, G_ref=G, N_guided_registered=sc['rev2']['N_needed'])


def load(res_dir, V, N, sd, R):
    """Energies over steps 1-120, or None and the reason if the arm is missing / not complete."""
    name = arm_name(V, N, sd)
    fn = os.path.join(res_dir, name, 'run.json')
    if not os.path.exists(fn):
        return None, 'missing'
    d = json.load(open(fn))
    st = d['steps']
    if name == STORM:  # storm.py writes run.json once, at the end; no status field
        if len(st) < NSTEP:
            return None, f'storm rung with {len(st)} steps'
        assert d['qmix'] == 1.0 and d['a'] == 1.0 and d['mode'] == 'x' and d['nsamp'] == 4096 and d['seed0'] == 9000, fn
    else:
        if d.get('status') != 'DONE' or len(st) < NSTEP:
            return None, f"{d.get('status')} at step {len(st)}"
        lx, ly, ne = SIZES[V]['geo']
        assert (d['Lx'], d['Ly'], d['Ne']) == (lx, ly, ne) and not d['exact'] and d['sampler'] == 'guided', fn
        assert d['nsamp'] == N and d['seed0'] == sd and os.path.basename(d['start']) == SIZES[V]['start'], fn
    assert abs(d['tau'] - 0.005) < 1e-15, fn
    assert abs(d['E_start'] - R['E_start']) < 1e-12, (fn, d['E_start'], R['E_start'])
    return np.array([s['E'] for s in st[:NSTEP]]), None


def s_win(E, R):
    return (float(E[WIN].mean()) - R['E_win']) / R['G_ref']


def wsd(E, R, detr):
    t = np.arange(61, 121)
    e = E[WIN]
    if detr:
        e = e - np.polyval(np.polyfit(t, e, 1), t)
    return float(np.std([e[a:a + 20].mean() for a in (0, 20, 40)], ddof=1) / R['G_ref'])


def need(rows):
    """(value or None, lo, hi, kind, bracket index pair or None, index of the bounding rung)."""
    for i, r in enumerate(rows):
        if r['s'] <= THR:
            if i == 0:
                return None, 0.0, float(r['N']), f"censored: <= {r['N']} (lowest rung s <= 0.10)", None, i
            a = rows[i - 1]
            if r['s'] <= 0:
                return None, float(a['N']), float(r['N']), 'censored: upper bracketing rung s <= 0', (i - 1, i), None
            la, lb = math.log(a['N']), math.log(r['N'])
            nn = math.exp(la + (math.log(THR) - math.log(a['s'])) * (lb - la) / (math.log(r['s']) - math.log(a['s'])))
            return nn, nn, nn, 'interpolated', (i - 1, i), None
    return None, float(rows[-1]['N']), math.inf, f"censored: > {rows[-1]['N']} (no rung s <= 0.10)", None, \
        len(rows) - 1


def score_size(res_dir, V, R):
    sz = SIZES[V]
    rows, incomplete = [], []
    for N, seeds in sz['rungs']:
        per, bad = {}, []
        for sd in seeds:
            E, why = load(res_dir, V, N, sd, R)
            if E is None:
                bad.append(f'{arm_name(V, N, sd)}: {why}')
            else:
                per[sd] = dict(arm=arm_name(V, N, sd), s=s_win(E, R), sd_win_detr=wsd(E, R, True),
                               sd_win_raw=wsd(E, R, False))
        row = dict(N=N, seeds=seeds, per=per, missing=bad)
        if not bad:
            row['s'] = float(np.mean([p['s'] for p in per.values()]))
            row['seed_diff'] = abs(per[9000]['s'] - per[19000]['s']) if len(seeds) == 2 else None
            row['sd_win_detr'] = float(np.mean([p['sd_win_detr'] for p in per.values()]))
            row['sd_win_raw'] = float(np.mean([p['sd_win_raw'] for p in per.values()]))
        else:
            incomplete += bad
        rows.append(row)
    out = dict(ref=R, rows=rows)
    use = rows
    if incomplete:
        first_bad = next(i for i, r in enumerate(rows) if r['missing'])
        if not any('s' in r and r['s'] <= THR for r in rows[:first_bad]):
            out.update(status='INCOMPLETE', not_done=incomplete)
            return out
        out['not_done_but_not_bracketing'] = incomplete
        use = rows[:first_bad]
    nn, lo, hi, kind, br, bound = need(use)
    out.update(status='SCORED', N_q=nn, N_lo=lo, N_hi=hi, kind=kind)
    if br is not None:
        a, b = use[br[0]], use[br[1]]
        seed_term = b['seed_diff'] / math.sqrt(2) if b['seed_diff'] is not None else 0.0
        sdw = max(b['sd_win_detr'], seed_term)
        sdw_raw = max(b['sd_win_raw'], seed_term)
        gap = a['s'] - b['s']
        out.update(bracket=[a['N'], b['N']], s_lower=a['s'], s_upper=b['s'], gap=gap, sd_w=sdw,
                   sd_win_detr_upper=b['sd_win_detr'], seed_term=seed_term, gap_over_sdw=gap / sdw if sdw > 0 else math.inf,
                   conditioning='RESOLVABLE' if gap > 2 * sdw else 'UNRESOLVABLE',
                   reported_raw=dict(sd_win_raw_upper=b['sd_win_raw'], sd_w_raw=sdw_raw,
                                     conditioning_raw='RESOLVABLE' if gap > 2 * sdw_raw else 'UNRESOLVABLE'))
        out['defining_rungs'] = [a, b]
    else:
        out['conditioning'] = 'no bracket (censored; not conditioned)'
        out['defining_rungs'] = [use[bound]]
    g = sz['guided']
    out['gain_reported'] = dict(N_guided=g, gain=(g / nn) if nn else None, gain_lo=g / hi if hi > 0 else math.inf,
                                gain_hi=g / lo if lo > 0 else math.inf,
                                N_guided_registered=R['N_guided_registered'])
    return out


def ratio_iv(num, den):
    lo = num['N_lo'] / den['N_hi'] if den['N_hi'] > 0 else math.inf
    hi = num['N_hi'] / den['N_lo'] if den['N_lo'] > 0 else math.inf
    return lo, hi


def classify(lo, hi):
    if hi <= G3:
        return 'POLY-CONSISTENT'
    if lo >= GH1:
        return 'SUPPORT-LIKE'
    if lo > G3 and hi < GH1:
        return 'AMBIGUOUS'
    return None


def allowed_classes(lo, hi):
    c = set()
    if lo <= G3:
        c.add('POLY-CONSISTENT')
    if hi >= GH1:
        c.add('SUPPORT-LIKE')
    if hi > G3 and lo < GH1:
        c.add('AMBIGUOUS')
    return sorted(c)


def ln_dist(lo, hi, line):
    """min over r in [lo, hi] of |ln r - ln line| (0 if the line is inside)."""
    if lo <= line <= hi:
        return 0.0
    return math.log(lo / line) if lo > line else math.log(line / hi)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--res-dir', default=os.path.join(H, 'results'))
    pa.add_argument('--ref-dir', default=os.path.join(H, '..', 'P5.1_size_scaling_r04', 'results'))
    pa.add_argument('--out', default=os.path.join(H, 'results', 'score_r16_qladder.json'))
    a = pa.parse_args()
    out = dict(rule='RULE_r14_qladder.md (+ Addendum 1)', res_dir=a.res_dir, sizes={})
    for V in SIZES:
        out['sizes'][V] = score_size(a.res_dir, V, ref(a.ref_dir, SIZES[V]))
    s10, s12 = out['sizes']['S10'], out['sizes']['S12p']
    st = [s10['status'], s12['status']]
    if 'INCOMPLETE' in st:
        out['verdict'] = 'INCOMPLETE'
    elif 'UNRESOLVABLE' in (s10.get('conditioning'), s12.get('conditioning')):
        out['verdict'] = 'UNRESOLVABLE'
        out['unresolvable_reason'] = 'CONDITIONING'
    if 'INCOMPLETE' not in st:
        lo, hi = ratio_iv(s12, s10)
        out.update(r_q=(s12['N_q'] / s10['N_q']) if s12['N_q'] and s10['N_q'] else None, r_q_lo=lo, r_q_hi=hi)
        if 'verdict' not in out:
            c = classify(lo, hi)
            if c is None:
                out.update(verdict='UNRESOLVABLE', unresolvable_reason='CENSORED-UNDECIDED',
                           classes_allowed=allowed_classes(lo, hi))
            else:
                d = {str(L): ln_dist(lo, hi, L) for L in (GH1, G3)}
                prov = any(v <= MARGIN for v in d.values())
                out.update(verdict=c, provisional=prov, margin_ln_dist=d,
                           verdict_text=('provisional ' if prov else '') + c)
                if prov:
                    out['margin_clause'] = ('PROVISIONAL: Answer not changed; second seed at S12\' (bracketing '
                                            'rungs only, capped, separately gated) owed')
    if s12['status'] == 'SCORED':
        owed = []
        for r in s12['defining_rungs']:
            dl = abs(math.log(r['s']) - math.log(THR)) if r['s'] > 0 else None
            r['ln_dist_to_0.10'] = dl
            if dl is not None and dl <= RUNG_LN:
                owed.append(r['N'])
        out['rung_clause'] = dict(rungs=[(r['N'], r['s'], r['ln_dist_to_0.10']) for r in s12['defining_rungs']],
                                  second_seed_owed_at=owed)
    for V in SIZES:
        out['sizes'][V].pop('defining_rungs', None)
    json.dump(out, open(a.out, 'w'), indent=1, default=lambda v: float(v) if np.isfinite(v) else str(v))
    print('verdict:', out.get('verdict_text', out['verdict']), out.get('unresolvable_reason', ''),
          out.get('classes_allowed', ''))
    print("r'_q:", out.get('r_q'), '[', out.get('r_q_lo'), ',', out.get('r_q_hi'), ']')
    for V, o in out['sizes'].items():
        print(V, o['status'], o.get('N_q'), o.get('N_lo'), o.get('N_hi'), o.get('kind'), o.get('conditioning'),
              o.get('gap_over_sdw'), [(r['N'], round(r['s'], 4) if 's' in r else None) for r in o['rows']],
              'gain', o.get('gain_reported'), o.get('not_done', ''))
    print('rung clause:', out.get('rung_clause'))


if __name__ == '__main__':
    main()
