"""
r5_common.py -- helpers shared by close31.py and obc_transfer.py (P4.2 round 5). No command line.
=================================================================================================

* optimise_ckpt   the projected exact-sum SR loop of r3_core.optimise (same backtracking, same
                  trust region |lr0 d| <= MAX_STEP, same descent fallback, energies therefore
                  non-increasing), run as ONE continuous trajectory with (i) a callback at the
                  checkpoint steps and (ii) a wall-clock deadline. r3_core.optimise cannot be
                  used in segments because every call resets the adaptive step to lr0.
                  Returns (theta, eps_history), eps_history[n] = (E_n - E0)/V after n accepted
                  or rejected steps (n = 0 is the starting state).
                  Optional (round-5 critic fixes; defaults keep the close31 behaviour):
                    mask_steps  the mask applies only to steps 1..mask_steps (then all parameters);
                    clock       a list; clock[n] = cumulative SR wall seconds after step n, with the
                                checkpoint-callback time EXCLUDED (clock[0] = initial gradient pass);
                    on_step     callback(n, hist) after every step (used for incremental dumps).
* identified_fraction  |P_id d| / |d| for d = theta_src - theta_init, where P_id projects onto the
                  eigenvectors of the joint source SR metric with eigenvalue >= tol * max eig
                  (the source-identified subspace).
* null_filter     source-null filtering (Route R5.1): theta_f = theta_src - P_null (theta_src -
                  theta_init), where P_null projects onto the eigenvectors of the joint source SR
                  metric S = sum_c S_c / V_c (at theta_src) with eigenvalue < tol * max eig.
                  Directions the sources do not constrain are reset to the source run's own
                  mean-field initialisation; constrained directions keep what the sources learned.
* steps_to        first index n with hist[n] <= level (None if never reached).
* med             median of a list (nan if empty).
"""
import time

import numpy as np

import r3_core as rc


def optimise_ckpt(cluster, th, steps, lr, shift, chunk, tag, E0, V, ckpts=(), on_ckpt=None,
                  deadline=None, mask=None, mask_steps=None, clock=None, on_step=None):
    th = th.copy()
    P = len(th)
    idx_m = np.arange(P) if mask is None else np.nonzero(mask)[0]
    idx_all = np.arange(P)
    tc = time.time()

    def lg(t_, need):
        r = rc.reduced(cluster, t_, chunk, need)
        if need:
            E, _, g, S = r
            if g is None:
                return E, np.zeros(P), np.zeros((P, P))
            return E, g, S
        return r[0], None, None

    E, g, S = lg(th, True)
    hist = [(E - E0) / V]
    if clock is not None:
        clock.append(time.time() - tc)
    if on_ckpt is not None and 0 in ckpts:
        t_cb = time.time()
        on_ckpt(0, th)
        tc += time.time() - t_cb
    lr0 = lr
    for it in range(1, steps + 1):
        if deadline is not None and time.time() > deadline:
            rc.log(f"  [{tag}] wall-clock deadline reached after {it - 1} steps; stopping")
            break
        idx = idx_m if (mask_steps is None or it <= mask_steps) else idx_all
        Ss = S[np.ix_(idx, idx)]
        Sr = Ss + shift * np.diag(np.diag(Ss)) + 1e-6 * max(1e-12, np.diag(Ss).max()) * np.eye(len(idx))
        d = np.zeros(P)
        d[idx] = np.linalg.solve(Sr, g[idx])
        nd_ = np.linalg.norm(d)
        if nd_ > rc.MAX_STEP / lr0:
            d *= rc.MAX_STEP / (lr0 * nd_)
        if g @ d <= 0:
            d = np.zeros(P)
            d[idx] = g[idx] * rc.MAX_STEP / (lr0 * max(1e-300, np.linalg.norm(g[idx])))
        tried = 0
        while True:
            tn = th - lr * d
            En, _, _ = lg(tn, False)
            if (np.isfinite(En) and En <= E + 1e-13) or tried >= 12:
                break
            lr *= 0.5
            tried += 1
        if np.isfinite(En) and En <= E + 1e-13:
            th = tn
            E, g, S = lg(th, True)
            lr = min(lr * 1.1, 4 * lr0)
        hist.append((E - E0) / V)
        if clock is not None:
            clock.append(time.time() - tc)
        if it % max(1, steps // 10) == 0 or it == steps:
            rc.log(f"  [{tag}] step {it}/{steps} eps={hist[-1]:.5e} lr={lr:.3g}")
        if on_ckpt is not None and it in ckpts:
            t_cb = time.time()
            on_ckpt(it, th)
            tc += time.time() - t_cb
        if on_step is not None:
            t_cb = time.time()
            on_step(it, hist)
            tc += time.time() - t_cb
    return th, hist


def null_filter(clusters, th_src, th_init, chunk, tol):
    S = np.zeros((len(th_src), len(th_src)))
    for cl in clusters:
        S += rc.reduced(cl, th_src, chunk, True)[3] / cl.V
    w, U = np.linalg.eigh(S)
    null = w < tol * w.max()
    Un = U[:, null]
    d = th_src - th_init
    th_f = th_src - Un @ (Un.T @ d)
    info = dict(n_null=int(null.sum()), npar=len(th_src), tol=tol,
                frac_shift_removed=float(np.linalg.norm(Un.T @ d) / max(1e-300, np.linalg.norm(d))),
                dist_to_init=float(np.linalg.norm(th_f - th_init)))
    return th_f, info


def identified_fraction(clusters, d, chunk, tol):
    """Fraction of |d| in the source-identified subspace (joint SR metric sum_c S_c / V_c at the
    clusters' current parameters d; eigenvalues >= tol * max eig)."""
    S = np.zeros((len(d), len(d)))
    for cl in clusters:
        S += rc.reduced(cl, d, chunk, True)[3] / cl.V
    w, U = np.linalg.eigh(S)
    ide = w >= tol * w.max()
    nd = max(1e-300, float(np.linalg.norm(d)))
    return dict(tol=tol, n_identified=int(ide.sum()), npar=len(d),
                frac_identified=float(np.linalg.norm(U[:, ide].T @ d) / nd))


def steps_to(hist, level):
    for n, e in enumerate(hist):
        if e <= level:
            return n
    return None


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")
