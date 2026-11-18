"""Round-06 T0 (read-only on results/): is the 5x2 guided shortfall s ~ 0.04 a floor or an unfinished descent?
For each arm: E(t) tail slopes (OLS over last 20 and 40 steps, in units of G_ref per step), lagged shortfall
s(t) = (E_arm(t) - E_exact(t))/G in 20-step windows, and a fit E(t) = E_inf + A exp(-t/T) to steps 41-120.
Writes results/t0_floor_r06.json."""
import json, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, 'results')
def load(name): return json.load(open(os.path.join(R, name, 'run.json')))
def E(name): return np.array([s['E'] for s in load(name)['steps']])
ex = E('S5x2_N8_exact_tau0.005_s9000'); Es = load('S5x2_N8_exact_tau0.005_s9000')['E_start']
Ex = ex[-20:].mean(); G = Es - Ex
def slope(y, n):
    t = np.arange(len(y))[-n:].astype(float); yy = y[-n:]; b = np.polyfit(t, yy, 1); r = yy - np.polyval(b, t)
    se = np.sqrt((r**2).sum() / (n - 2) / ((t - t.mean())**2).sum()); return float(b[0] / G), float(se / G)
out = {'G_ref': float(G)}
arms = {'exact': ex}
for samp in ['guided', 'born']:
    for N in [1024, 4096, 16384, 65536, 262144]:
        arms[f'{samp}_{N}'] = E(f'S5x2_N8_{samp}_N{N}_tau0.005_s9000')
for k, y in arms.items():
    d = {}
    d['slope40_perstep'], d['slope40_se'] = slope(y, 40)
    d['slope20_perstep'], d['slope20_se'] = slope(y, 20)
    lag = (y - ex) / G
    d['s_win'] = {f'{a+1}-{a+20}': float(lag[a:a+20].mean()) for a in range(0, 120, 20)}
    tt = np.arange(20.)
    d['resid_sd_last20'] = float(np.std(y[-20:] - np.polyval(np.polyfit(tt, y[-20:], 1), tt)) / G)
    t = np.arange(40, 120).astype(float); yy = y[40:]; best = None
    for T in np.geomspace(3, 3000, 400):
        X = np.c_[np.ones_like(t), np.exp(-t / T)]; c = np.linalg.lstsq(X, yy, rcond=None)[0]
        rss = ((X @ c - yy)**2).sum()
        if best is None or rss < best[0]: best = (rss, T, c)
    _, T, c = best
    d['expfit_T'] = float(T); d['expfit_Einf_minus_Eplat_exact_overG'] = float((c[0] - Ex) / G)
    out[k] = d
json.dump(out, open(os.path.join(R, 't0_floor_r06.json'), 'w'), indent=1)
for k, d in out.items():
    if k == 'G_ref': continue
    print(f"{k:13s} sl40 {d['slope40_perstep']:+.2e}+-{d['slope40_se']:.1e} sl20 {d['slope20_perstep']:+.2e}+-{d['slope20_se']:.1e} "
          f"s_win {[round(v,3) for v in d['s_win'].values()]} sd20 {d['resid_sd_last20']:.3f} T {d['expfit_T']:.0f} Einf {d['expfit_Einf_minus_Eplat_exact_overG']:+.3f}")
