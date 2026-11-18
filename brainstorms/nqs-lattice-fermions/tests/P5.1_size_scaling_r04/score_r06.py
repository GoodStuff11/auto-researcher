"""Round-06 scoring of R51.8 unit 1a (5x2 ladder) and R51.10 kappa (RULE.md rev1, RULE_kappa.md rev1, voi_unit1a.md).
Read-only on results/; writes results/score_r06.json."""
import json, math, os
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, 'results')
def run(name): return json.load(open(os.path.join(R, name, 'run.json')))
NS = [1024, 4096, 16384, 65536, 262144]; P10 = 1832
ex = run('S5x2_N8_exact_tau0.005_s9000')
def plat(d): E = [s['E'] for s in d['steps']]; return sum(E[-20:]) / 20, len(E), d.get('status')
Es = ex['E_start']; Ex, nex, stx = plat(ex); G = Es - Ex
out = {'E_start': Es, 'E_plat_exact': Ex, 'G_ref': G, 'exact_steps': nex, 'exact_status': stx,
       'E0_pg': json.load(open(os.path.join(R, 'info_S5x2_N8.json')))['E0_pg']}
out['R_plat_exact'] = G / (Es - out['E0_pg'])
def first_cross(d, thr):
    for i, s in enumerate(d['steps']):
        if s['E'] <= thr: return i + 1
    return None
thr = Es - 0.9 * G
for samp in ['guided', 'born']:
    rows = []
    for N in NS:
        d = run(f'S5x2_N8_{samp}_N{N}_tau0.005_s9000'); Ep, n, st = plat(d)
        s = (Ep - Ex) / G; fc = first_cross(d, thr)
        rows.append(dict(N=N, E_plat=Ep, s=s, R=(Es - Ep) / (Es - out['E0_pg']), steps=n, status=st,
                         NltP=N < P10, first90=fc, T=None if fc is None else N * fc,
                         uniq_frac_last=d['steps'][-1].get('n_unique', float('nan')) / N))
    nn = None; cens = None
    for k, r in enumerate(rows):
        if r['s'] <= 0.05:
            if k == 0: cens = 'below 1024'; break
            a, b = rows[k - 1], r
            if b['s'] <= 0: cens = f'bracket ({a["N"]},{b["N"]}], s<=0 at upper rung'; break
            la, lb = math.log(a['N']), math.log(b['N']); sa, sb = math.log(a['s']), math.log(b['s'])
            nn = math.exp(la + (math.log(0.05) - sa) * (lb - la) / (sb - sa))
            break
    else:
        cens = 'above 262144'
    Ts = [r['T'] for r in rows if r['T'] is not None]
    out[samp] = dict(rows=rows, N_needed=nn, censored=cens, T_tot=min(Ts) if Ts else None)
g, b = out['guided'], out['born']
out['comparator_ratio_Tborn_over_Tguided'] = (b['T_tot'] / g['T_tot']) if (g['T_tot'] and b['T_tot']) else None
nn = g['N_needed']
if g['censored'] == 'above 262144' or (nn and nn >= 151700):
    stage = 'POLY-CONSISTENT (r<=1.728; 4x3 exact arm never bought)'
elif (nn is not None and nn <= 16850) or g['censored'] == 'below 1024':
    stage = 'SUPPORT-LIKE (r>=3.89; 4x3 arm never bought; S12 ladder confirms)'
else:
    stage = 'IN-BETWEEN (4x3 reference arm released only if geometry matters)'
out['staged_decision'] = stage
if nn: out['r_bracket_vs_4x3'] = [65536 / nn, 262144 / nn]
k = {n: json.load(open(os.path.join(R, f'kappa_{n}_start.json'))) for n in ['S5x2', 'S6x2', 'S4x3']}
k1 = k['S6x2']['kappa_guided'] / k['S5x2']['kappa_guided']; gH1 = k['S6x2']['expH'] / k['S5x2']['expH']
out['kappa'] = dict(kg={n: v['kappa_guided'] for n, v in k.items()}, kb={n: v['kappa_born'] for n, v in k.items()},
                    d_eff={n: v['d_eff'] for n, v in k.items()}, expH={n: v['expH'] for n, v in k.items()},
                    k1=k1, gH1=gH1,
                    k1_class='POLY-side (<=1.728)' if k1 <= 1.728 else ('SUPPORT-side (>=gH1)' if k1 >= gH1 else 'AMBIG-side'),
                    geom_ratio_6x2_over_4x3=k['S6x2']['kappa_guided'] / k['S4x3']['kappa_guided'])
gr = out['kappa']['geom_ratio_6x2_over_4x3']; out['kappa']['geometry_flag'] = not (1 / 1.5 <= gr <= 1.5)
json.dump(out, open(os.path.join(R, 'score_r06.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
