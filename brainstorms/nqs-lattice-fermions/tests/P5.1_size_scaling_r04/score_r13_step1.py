"""Round 13: re-score R51.8 step 1 (S10 -> S12') with the S12' guided 65,536 second seed (seed0 19000), per the
RULE.md addendum of 2026-11-18 ("before any 7x2 data"). Read-only on results/; writes results/step1_r13.json.
Called by submit_r13_s14_ladder.sbatch after the seed arm is DONE; exit 0 if r' is still SUPPORT-LIKE, 3 otherwise
(the job then writes results/STOP_r13 and runs no S14 arm). Exit 2 if an input is missing or not DONE.
  python3 score_r13_step1.py            score (needs S6x2_N10_guided_N65536_tau0.005_s19000 DONE)
  python3 score_r13_step1.py --dry-run  seed 9000 only (must reproduce score_r08.json: N 118,626, r' 3.79)
Statistic: rev 2 (s on steps 61-120 vs the exact arm's 61-120 mean; N_needed at s = 0.10 by ln s - ln N
interpolation). s(65k) = two-seed mean; 262k one seed. Conditioning: gap > 2 sd_w, sd_w = max(detrended window sd
of the upper rung, |seed difference at 65k| / sqrt(2)). r' = N_needed(S12') / N_needed(S10) (score_r07.json)
against g3 = 1.728 and gH1 = expH(S12') / expH(S10) (info_*.json)."""
import json, math, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, 'results')
DRY = '--dry-run' in sys.argv
def E_of(name):
    fn = os.path.join(R, name, 'run.json')
    if not os.path.exists(fn): print(f'missing {fn}'); sys.exit(2)
    d = json.load(open(fn))
    if d.get('status') != 'DONE' or len(d['steps']) != 120: print(f'{name} not DONE'); sys.exit(2)
    return np.array([s['E'] for s in d['steps']]), d['E_start']
Ex, Es = E_of('S6x2_N10_exact_tau0.005_s9000')
win = slice(60, 120); Ewx = Ex[win].mean(); G = Es - Ewx
def s_win(E): return (E[win].mean() - Ewx) / G
def wsd_detr(E):
    t = np.arange(61, 121); E2 = E[win] - np.polyval(np.polyfit(t, E[win], 1), t)
    return float(np.std([E2[a:a + 20].mean() for a in (0, 20, 40)], ddof=1) / G)
g = lambda N, sd: f'S6x2_N10_guided_N{N}_tau0.005_s{sd}'
rows = []
for N in (16384, 65536, 262144):
    seeds = [9000] + ([19000] if (N == 65536 and not DRY) else [])
    per = {}
    for sd in seeds:
        E, es = E_of(g(N, sd)); assert abs(es - Es) < 1e-12
        per[sd] = dict(s=s_win(E), sd_w_detr=wsd_detr(E))
    s = float(np.mean([v['s'] for v in per.values()]))
    diff = abs(per[9000]['s'] - per[19000]['s']) if 19000 in per else None
    rows.append(dict(N=N, s=s, per=per, seed_diff=diff, sd_w_detr=per[9000]['sd_w_detr']))
k = next((i for i, r in enumerate(rows) if r['s'] <= 0.10), None)
out = dict(dry_run=DRY, E_start=Es, E_win_exact=Ewx, G_ref=G, rows=rows)
n10 = json.load(open(os.path.join(R, 'score_r07.json')))['rev2']['N_needed']
gH1 = json.load(open(os.path.join(R, 'info_S6x2_N10.json')))['expH'] / json.load(open(os.path.join(R, 'info_S5x2_N8.json')))['expH']
out.update(N_needed_5x2=n10, gH1=gH1, g3=1.728)
if k is None or k == 0:
    out['verdict'] = 'AMBIGUOUS'; out['why'] = 'no bracket (censored)'
else:
    a, b = rows[k - 1], rows[k]
    la, lb = math.log(a['N']), math.log(b['N'])
    nn = math.exp(la + (math.log(0.10) - math.log(a['s'])) * (lb - la) / (math.log(b['s']) - math.log(a['s'])))
    seed_term = max((r['seed_diff'] or 0.0) for r in rows) / math.sqrt(2)  # only the 65k rung has two seeds
    sdw = max(b['sd_w_detr'], seed_term)
    gap = a['s'] - b['s']
    rp = nn / n10
    out.update(bracket=[a['N'], b['N']], N_needed=nn, gap=gap, sd_w=sdw, seed_term=seed_term, r_prime=rp,
               margin=rp / gH1)
    out['verdict'] = ('UNRESOLVABLE' if gap <= 2 * sdw else 'POLY-CONSISTENT' if rp <= 1.728 else
                      'SUPPORT-LIKE' if rp >= gH1 else 'AMBIGUOUS')
if not DRY:
    json.dump(out, open(os.path.join(R, 'step1_r13.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1, default=float))
print('s per rung:', [(r['N'], round(r['s'], 4), r['seed_diff']) for r in rows])
sys.exit(0 if out['verdict'] == 'SUPPORT-LIKE' else 3)
