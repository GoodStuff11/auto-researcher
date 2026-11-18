"""Round-11 scoring of R51.8 unit 1c (6x2 = S12', job 769376) under RULE.md revision 2 plus both appended addenda.
Read-only on results/; writes results/score_r08.json. One seed (9000) per arm.
Primary: E_win = mean exact energy steps 61-120; G_ref = E_start - E_win(exact); s per rung; N_needed at s = 0.10
by ln s - ln N interpolation; conditioning gap > 2 sd_w with sd_w from detrended window means of the upper rung
(rev-2 addendum; one seed, so no seed term). r' = N_needed(6x2)/N_needed(5x2), N_needed(5x2) from score_r07.json.
Comparator (addendum): first crossing of E <= E_start - 0.9 G_ref; censored Born >= largest Born N x 120.
Reported only: 5%/last-20 N_needed, raw (undetrended) sd_w, exact-arm 61-120 slope flag, sustained-90% draws."""
import json, math, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, 'results')
GN = [4096, 16384, 65536, 262144]; BN = [65536, 262144]
def E_of(name):
    d = json.load(open(os.path.join(R, name, 'run.json')))
    assert d.get('status') == 'DONE' and len(d['steps']) == 120, name
    return np.array([s['E'] for s in d['steps']]), d['E_start']
def arm(samp, N): return f'S6x2_N10_{samp}_N{N}_tau0.005_s9000'
Ex, Es = E_of('S6x2_N10_exact_tau0.005_s9000')
info = json.load(open(os.path.join(R, 'info_S6x2_N10.json'))) if os.path.exists(os.path.join(R, 'info_S6x2_N10.json')) else {}
win = slice(60, 120); last20 = slice(100, 120)
Ewx = Ex[win].mean(); G = Es - Ewx; Elx = Ex[last20].mean(); G20 = Es - Elx
def s_win(E): return (E[win].mean() - Ewx) / G
def s_l20(E): return (E[last20].mean() - Elx) / G20
def wsd(E, detr):
    t = np.arange(61, 121); E2 = E[win]
    if detr: E2 = E2 - np.polyval(np.polyfit(t, E2, 1), t)
    return float(np.std([E2[a:a + 20].mean() for a in (0, 20, 40)], ddof=1) / G)
def interp(rows, key, thr):
    for k, r in enumerate(rows):
        if r[key] <= thr:
            if k == 0: return None, f'below {rows[0]["N"]}', None
            a = rows[k - 1]
            if r[key] <= 0: return None, 'upper s<=0', (a['N'], r['N'])
            la, lb = math.log(a['N']), math.log(r['N'])
            nn = math.exp(la + (math.log(thr) - math.log(a[key])) * (lb - la) / (math.log(r[key]) - math.log(a[key])))
            return nn, None, (a['N'], r['N'])
    return None, f'above {rows[-1]["N"]}', None
def tfirst(E):
    i = np.nonzero(E <= Es - 0.9 * G)[0]; return None if len(i) == 0 else int(i[0] + 1)
def tsust(E):
    g = Es - E; m = np.convolve(g, np.ones(20) / 20, mode='valid'); i = np.nonzero(m >= 0.9 * G)[0]
    return None if len(i) == 0 else int(i[0] + 20)
out = dict(E_start=Es, E_win_exact=Ewx, G_ref_rev2=G, E_last20_exact=Elx, G_ref_last20=G20, E0_pg=info.get('E0_pg'))
if info.get('E0_pg') is not None:
    out['gap_to_E0'] = Es - info['E0_pg']; out['G_ref_over_gap'] = G / (Es - info['E0_pg'])
t = np.arange(61, 121); c, cov = np.polyfit(t, Ex[win], 1, cov=True); se = math.sqrt(cov[0, 0])
out['exact_slope'] = dict(slope_per_step=c[0], se=se, z=c[0] / se, slope_over_60_in_G=c[0] * 60 / G,
                          flag='exact still descending' if c[0] < 0 and abs(c[0]) > 2 * se else 'not flagged')
out['exact_t_first90'] = tfirst(Ex)
rows = {}
for samp, NS in (('guided', GN), ('born', BN)):
    rr = []
    for N in NS:
        E, es = E_of(arm(samp, N)); assert abs(es - Es) < 1e-12
        rr.append(dict(N=N, s=s_win(E), s_l20=s_l20(E), sd_win=wsd(E, False), sd_win_detr=wsd(E, True),
                       t_first90=tfirst(E), t_sust90=tsust(E),
                       T_first=None if tfirst(E) is None else N * tfirst(E),
                       T_sust=None if tsust(E) is None else N * tsust(E)))
    rows[samp] = rr
out['rows'] = rows
g = rows['guided']
nn, cens, br = interp(g, 's', 0.10)
out['rev2'] = dict(N_needed=nn, censored=cens, bracket=br)
if br:
    lo = next(r for r in g if r['N'] == br[0]); up = next(r for r in g if r['N'] == br[1])
    gap = lo['s'] - up['s']
    out['rev2'].update(s_lower=lo['s'], s_upper=up['s'], gap=gap,
                       sd_w_detr=up['sd_win_detr'], ratio_gap_over_sdw_detr=gap / up['sd_win_detr'],
                       cond_verdict='RESOLVABLE' if gap > 2 * up['sd_win_detr'] else 'UNRESOLVABLE',
                       sd_w_raw=up['sd_win'], ratio_raw=gap / up['sd_win'],
                       cond_raw='RESOLVABLE' if gap > 2 * up['sd_win'] else 'UNRESOLVABLE')
s7 = json.load(open(os.path.join(R, 'score_r07.json')))
n10 = s7['rev2']['N_needed']; out['N_needed_5x2'] = n10
if nn:
    rp = nn / n10; out['r_prime'] = rp
    out['verdict'] = ('UNRESOLVABLE' if out['rev2']['cond_verdict'] == 'UNRESOLVABLE' else
                      'POLY-CONSISTENT' if rp <= 1.728 else 'SUPPORT-LIKE' if rp >= 3.52 else 'AMBIGUOUS')
nn5, c5, b5 = interp(g, 's_l20', 0.05)
out['five_pct_last20'] = dict(N_needed=nn5, censored=c5, bracket=b5)
def Tmin(samp, key):
    v = [r[key] for r in rows[samp] if r[key] is not None]; return min(v) if v else None
tg = Tmin('guided', 'T_first'); tb = Tmin('born', 'T_first')
cmp_ = dict(guided=tg, born=tb, born_censored=tb is None)
if tb is None: tb_lb = max(BN) * 120; cmp_['born_lower_bound'] = tb_lb; cmp_['ratio_ge'] = tb_lb / tg if tg else None
else: cmp_['ratio'] = tb / tg if tg else None
out['comparator_first'] = cmp_
out['sustained90_reported'] = dict(guided=Tmin('guided', 'T_sust'), born=Tmin('born', 'T_sust'))
json.dump(out, open(os.path.join(R, 'score_r08.json'), 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
