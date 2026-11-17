#!/usr/bin/env python3
"""P4.1 round 22: exact-sum energy and gradient for the 12-site arms (che, pcg) at the 742206 endpoints.

Read-only reuse of ../P4.1_cheap_su2_r19/polish.py (ctx_for, which builds Round 11's orbit-reduced Ctx).
E(theta) = pr^T W Hr pr / pr^T W pr with pr = A phi(theta) (point-group orbit projection), phi = s exp(l - lmax).
dE/dpr = 2 (W Hr pr - E W pr)/(pr^T W pr);  dE/dphi = A^T dE/dpr;  then one chunked backward through phi(theta).
Checked against a central finite difference in probe_timing.py.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r19"))
import polish as P  # noqa: E402

R22 = os.path.join(TESTS, "P4.1_cheap_su2_r22")


def setup(arm):
    ctx = P.ctx_for(arm)
    ctx.th0 = torch.tensor(np.load(os.path.join(R22, f"results/theta_{arm}.npy")))
    return ctx


def energy_grad(ctx, th, CH=8192, need_grad=True):
    dim = ctx.sec.dim
    ls, ss = [], []
    with torch.no_grad():
        for s0 in range(0, dim, CH):
            l, s = ctx.m.logpsi0(th, ctx.nU[s0:s0 + CH], ctx.nD[s0:s0 + CH])
            ls.append(l)
            ss.append(s)
    l, s = torch.cat(ls), torch.cat(ss)
    lmax = float(l.max())
    phi = (s * torch.exp(l - lmax)).numpy()
    reps = ctx.reps
    pr = 0.25 * (ctx.sg[:, reps] * phi[ctx.idx[:, reps]]).sum(0)
    W = ctx.size
    nrm2 = np.sum(W * pr * pr)
    Hp = ctx.Hr @ pr
    E = float(np.sum(W * pr * Hp) / nrm2)
    if not need_grad:
        return E, None
    c = 2.0 * (W * Hp - E * W * pr) / nrm2
    cphi = np.zeros(dim)
    for g in range(4):
        np.add.at(cphi, ctx.idx[g, reps], 0.25 * ctx.sg[g, reps] * c)
    nz = np.nonzero(cphi)[0]
    thg = th.detach().clone().requires_grad_(True)
    G = torch.zeros_like(thg)
    cphi_t = torch.tensor(cphi)
    for k0 in range(0, len(nz), CH):
        ix = nz[k0:k0 + CH]
        l, s = ctx.m.logpsi0(thg, ctx.nU[ix], ctx.nD[ix])
        loss = (cphi_t[ix] * s * torch.exp(l - lmax)).sum()
        G += torch.autograd.grad(loss, thg)[0]
    return E, G.detach().numpy()
