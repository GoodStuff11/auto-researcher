"""
Compiled batched Pfaffian for the R2 `pcg` arm (P4.1 compiled-Pfaffian re-timing, 2026-11-16).
==============================================================================================

slogpf_fast(A) is a drop-in replacement for vmc_t2.slogpf(A): sign and log|Pf| of a batch (..., m, m) of real
antisymmetric matrices, m even. Same algorithm as vmc_t2.slogpf (Parlett-Reid elimination, partial pivoting on the
first row of the trailing block), but compiled with numba and parallel over the batch (numba prange; the thread count
is numba.set_num_threads, set by the caller to the same value as torch.set_num_threads).

Differentiable through a torch.autograd.Function: d log|Pf X| = (1/2) tr(X^{-1} dX), so the backward pass returns
grad_X = (1/2) g X^{-T} (LAPACK batched inverse). The sign carries no gradient.

Not a script: imported by retime.py.
"""
import numba
import numpy as np
import torch


@numba.njit(parallel=True, cache=True, fastmath=False)
def _slogpf_batch(A):
    nb, m, _ = A.shape
    sgn = np.ones(nb)
    lab = np.zeros(nb)
    for b in numba.prange(nb):
        M = A[b].copy()
        s = 1.0
        la = 0.0
        tau = np.empty(m)
        c1 = np.empty(m)
        for k in range(0, m - 1, 2):
            # pivot: largest |M[k, j]|, j > k (same rule as vmc_t2.slogpf)
            kp = k + 1
            best = abs(M[k, k + 1])
            for j in range(k + 2, m):
                v = abs(M[k, j])
                if v > best:
                    best = v
                    kp = j
            if kp != k + 1:
                for j in range(k, m):  # swap rows k+1, kp
                    t = M[k + 1, j]
                    M[k + 1, j] = M[kp, j]
                    M[kp, j] = t
                for i in range(k, m):  # swap columns k+1, kp
                    t = M[i, k + 1]
                    M[i, k + 1] = M[i, kp]
                    M[i, kp] = t
                s = -s
            piv = M[k, k + 1]
            if piv == 0.0:
                s = 0.0
                la = -np.inf
                break
            if piv < 0.0:
                s = -s
            la += np.log(abs(piv))
            for i in range(k + 2, m):
                tau[i] = M[k, i] / piv
                c1[i] = M[i, k + 1]
            for i in range(k + 2, m):
                ti = tau[i]
                ci = c1[i]
                for j in range(k + 2, m):
                    M[i, j] += ti * c1[j] - ci * tau[j]
        sgn[b] = s
        lab[b] = la
    return sgn, lab


class _LogPf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        Xn = np.ascontiguousarray(X.detach().cpu().numpy().reshape(-1, X.shape[-1], X.shape[-1]))
        s, l = _slogpf_batch(Xn)
        sh = X.shape[:-2]
        ctx.save_for_backward(X)
        return torch.from_numpy(l).reshape(sh), torch.from_numpy(s).reshape(sh)

    @staticmethod
    def backward(ctx, gl, gs):
        (X,) = ctx.saved_tensors
        Xi = torch.linalg.inv(X)
        return 0.5 * gl[..., None, None] * Xi.transpose(-1, -2)


def slogpf_fast(A):
    """(sign, log|Pf A|) for a batch of antisymmetric A (..., m, m); drop-in for vmc_t2.slogpf."""
    if A.requires_grad:
        l, s = _LogPf.apply(A)
        return s.detach(), l
    Xn = np.ascontiguousarray(A.detach().cpu().numpy().reshape(-1, A.shape[-1], A.shape[-1]))
    s, l = _slogpf_batch(Xn)
    sh = A.shape[:-2]
    return torch.from_numpy(s).reshape(sh), torch.from_numpy(l).reshape(sh)
