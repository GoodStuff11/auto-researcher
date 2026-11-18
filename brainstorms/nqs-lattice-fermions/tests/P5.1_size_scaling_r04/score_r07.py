"""Round-07 scoring of R51.8 at 5x2 (S10) under RULE.md revision 2, with second seeds (job 769098, seed0 19000)
and the seed-9000 arms + exact arm (job 763968). Read-only on results/; writes results/score_r07.json.
Also reports (not scored): 5%/last-20 N_needed with two seeds, exact-arm 61-120 slope and 2-sigma flag,
guided-vs-born sustained-90% total draws, and the quantities of the critic's rev-2 addendum (sustained 90% of the
size's own exact gain; sd_w from detrended window means)."""
import json, math, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, 'results')
NS = [1024, 4096, 16384, 65536, 262144]; SEED2 = {16384, 65536, 262144}
def E_of(name):
    d = json.load(open(os.path.join(R, name, 'run.json')))
    assert d.get('status') == 'DONE' and len(d['steps']) == 120, name
    return np.array([s['E'] for s in d['steps']]), d['E_start']
def arm(samp, N, seed=9000): return f'S5x2_N8_{samp}_N{N}_tau0.005_s{seed}'
Ex, Es = E_of('S5x2_N8_exact_tau0.005_s9000')
win = slice(60, 120); last20 = slice(100, 120)
Ewx = Ex[win].mean(); G = Es - Ewx; Elx = Ex[last20].mean(); G20 = Es - Elx
def s_win(E): return (E[win].mean() - Ewx) / G
def s_l20(E): return (E[last20].mean() - Elx) / G20
def wmeans(E): return [E[a:a + 20].mean() for a in (60, 80, 100)]
def wmeans_detr(E):
    t = np.arange(61, 121); E2 = E[win]; c = np.polyfit(t, E2, 1); r = E2 - np.polyval(c, t)
    return [r[a:a + 20].mean() for a in (0, 20, 40)]
def interp(rows, key, thr):
    for k, r in enumerate(rows):
        if r[key] <= thr:
            if k == 0: return None, 'below 1024', None
            a = rows[k - 1]
            if r[key] <= 0: return None, f'upper s<=0', (a['N'], r['N'])
            la, lb = math.log(a['N']), math.log(r['N'])
            nn = math.exp(la + (math.log(thr) - math.log(a[key])) * (lb - la) / (math.log(r[key]) - math.log(a[key])))
            return nn, None, (a['N'], r['N'])
    return None, 'above 262144', None
def tstar(E, Gain, frac=0.9):  # sustained: first t>=20 with trailing 20-step mean of (Es-E) >= frac*Gain
    g = Es - E; m = np.convolve(g, np.ones(20) / 20, mode='valid'); i = np.nonzero(m >= frac * Gain)[0]
    return None if len(i) == 0 else int(i[0] + 20)
out = dict(E_start=Es, E_win_exact=Ewx, G_ref_rev2=G, E_last20_exact=Elx, G_ref_last20=G20)
# exact-arm slope over 61-120
t = np.arange(61, 121); c, cov = np.polyfit(t, Ex[win], 1, cov=True); se = math.sqrt(cov[0, 0])
out['exact_slope'] = dict(slope_per_step=c[0], se=se, z=c[0] / se, slope_over_60_in_G=c[0] * 60 / G,
                          flag='exact still descending' if c[0] < 0 and abs(c[0]) > 2 * se else 'not flagged')
rows = {}
for samp in ['guided', 'born']:
    rr = []
    for N in NS:
        seeds = [9000] + ([19000] if samp == 'guided' and N in SEED2 else [])
        per = {}
        for sd in seeds:
            E, es = E_of(arm(samp, N, sd)); assert abs(es - Es) < 1e-12
            per[sd] = dict(s=s_win(E), s_l20=s_l20(E), wm=wmeans(E), wmd=wmeans_detr(E),
                           t90_rev2=tstar(E, G), t90_l20=tstar(E, G20))
        r = dict(N=N, seeds=seeds, per=per,
                 s=float(np.mean([p['s'] for p in per.values()])),
                 s_l20=float(np.mean([p['s_l20'] for p in per.values()])))
        r['seed_diff'] = abs(per[9000]['s'] - per[19000]['s']) if len(seeds) == 2 else None
        # within-run window sd in s units, averaged over seeds (rule: sd of upper rung's three window means)
        r['sd_win'] = float(np.mean([np.std(np.array(p['wm']) / G, ddof=1) for p in per.values()]))
        r['sd_win_detr'] = float(np.mean([np.std(np.array(p['wmd']) / G, ddof=1) for p in per.values()]))
        rr.append(r)
    rows[samp] = rr
out['rows'] = rows
g = rows['guided']
nn, cens, br = interp(g, 's', 0.10)
out['rev2'] = dict(N_needed=nn, censored=cens, bracket=br)
if br:
    lo = next(r for r in g if r['N'] == br[0]); up = next(r for r in g if r['N'] == br[1])
    sdseed = up['seed_diff'] / math.sqrt(2) if up['seed_diff'] is not None else 0.0
    sd_w = max(up['sd_win'], sdseed); gap = lo['s'] - up['s']
    out['rev2'].update(s_lower=lo['s'], s_upper=up['s'], gap=gap, sd_win_upper=up['sd_win'], sd_seed=sdseed,
                       sd_w=sd_w, ratio_gap_over_sdw=gap / sd_w,
                       verdict='RESOLVABLE' if gap > 2 * sd_w else 'UNRESOLVABLE')
    sd_wd = max(up['sd_win_detr'], sdseed)
    out['addendum_sdw_detrended'] = dict(sd_win_detr_upper=up['sd_win_detr'], sd_w=sd_wd,
                                         verdict='RESOLVABLE' if gap > 2 * sd_wd else 'UNRESOLVABLE')
# reported only: 5%/last-20 with two seeds
nn5, c5, b5 = interp(g, 's_l20', 0.05)
out['five_pct_last20'] = dict(N_needed=nn5, censored=c5, bracket=b5)
# sustained-90% total draws (trailing 20-step mean, against the rev-2 exact gain and the last-20 gain)
def Ttot(samp, key):
    best = None; per = []
    for r in rows[samp]:
        for sd, p in r['per'].items():
            ts = p[key]; T = None if ts is None else r['N'] * ts; per.append((r['N'], sd, ts, T))
    for seed in (9000, 19000):
        pass
    return per
out['sustained90'] = {}
for key in ('t90_rev2', 't90_l20'):
    d = {}
    for samp in ('guided', 'born'):
        per = Ttot(samp, key)
        d[samp + '_per_arm'] = per
        for seed in (9000, 19000):
            Ts = [T for (N, sd, ts, T) in per if T is not None and (sd == seed or (seed == 19000 and sd == 9000 and N not in SEED2 and samp == 'guided'))]
            d[f'{samp}_Tmin_s{seed}'] = min(Ts) if Ts else None
    d['ratio_born_over_guided_s9000'] = (d['born_Tmin_s9000'] / d['guided_Tmin_s9000']) if d['born_Tmin_s9000'] and d['guided_Tmin_s9000'] else None
    d['ratio_born_over_guided_s19000set'] = (d['born_Tmin_s9000'] / d['guided_Tmin_s19000']) if d['born_Tmin_s9000'] and d['guided_Tmin_s19000'] else None
    out['sustained90'][key] = d
json.dump(out, open(os.path.join(R, 'score_r07.json'), 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
