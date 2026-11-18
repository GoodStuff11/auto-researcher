"""6x2 exact arm: first-order gain at the step-120 end state and one exact step later (phase check)."""
import sys, json, numpy as np, torch
D='/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04'
sys.path.insert(0, D); import ns_run as NS
ctx = NS.GCtx(6, 2, 10, f'{D}/starts/S6x2_N10_U8_tp-0.25_pg11_che_seed0_theta.npy')
th = torch.tensor(np.load(f'{D}/results/S6x2_N10_exact_tau0.005_s9000/theta.npy'))
G = 0.0070106742743636374; out = []
E, pr, phi0 = ctx.energy(th)
for k in range(2):
    ST, f, nu = NS.stats(ctx, th, phi0, pr, 0, 0, 'guided', True)
    w, U = np.linalg.eigh(ST); d, ml = NS.T.d_shift(w, U, f, 0.005)
    out.append(dict(E=E, xSx=ml**2, gain=2*0.005*ml/G, fo=float(-2*f@d)/G)); NS.log(json.dumps(out[-1]))
    if k == 0:
        th = th + torch.tensor(d); E, pr, phi0 = ctx.energy(th)
json.dump(out, open('six.json', 'w'), indent=1)
