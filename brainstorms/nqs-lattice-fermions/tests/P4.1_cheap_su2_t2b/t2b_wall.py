#!/usr/bin/env python3
"""
P4.1 R2 T2b equal-wall-time amendment: thin wrapper around t2b.py (2026-11-16).
================================================================================

Why: T2b (job 699824) scores pcg against che and r1 at matched parameters and an equal step count (3000), but pcg
costs about 2.6-2.8 times che per step at 24 sites. This wrapper runs the equal-wall-time arms `che_wall` and
`r1_wall`: the same arm, instance, seed and protocol as T2b, with --iters raised so that the training wall time equals
that of pcg (leaf section "R2 T2b equal-wall-time amendment (2026-11-16)").

Run with the lit_env interpreter, at most 4 threads:
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 ~/.conda/envs/lit_env/bin/python t2b_wall.py [wrapper options] \
        --mode train [every t2b.py option]

What it does: imports t2b.py READ-ONLY (t2b.py, vmc_t2.py, vmc.py and pf_fast.py are not modified; job 699824 runs
them) and, inside this process only, replaces vmc_t2.tag_of by tag_of(...) + --tag-suffix. Everything else (model,
SR step, learning-rate schedule, Polyak average, checkpoints, final evaluation, resume, cap) is t2b.train unchanged.
Schedules are fractions of --iters (vmc.lr_at: constant for the first half, cosine to lr * lr_decay; Polyak average
over the last --avg-frac = 25%), so a run with --iters N is the 3000-step protocol stretched to N steps.

Wrapper options (parsed in parse_wrapper_args, everything else is passed to t2b.parse_args unchanged)
-----------------------------------------------------------------------------------------------------
  --tag-suffix STR ("_wall")  appended to the result tag, so <outdir>/<tag><suffix>.json, _ckpt.npz, _theta.npy,
                              _theta_last.npy never collide with T2b's own files. Must be non-empty.
  --print-only                build the model, print the tag, iters, checkpoint steps, Polyak window and learning rate
                              at the schedule's corners, then exit without training (used by the pre-flight to check
                              every array line).
Safety: --outdir must not be the T2b folder results/t2b (t2b_report.py globs results/t2b/S*x3_*.json and groups by
the JSON field arm, so a che_wall file there would be read as a che seed). The sbatch file uses results/t2b_wall.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2b  # noqa: E402  (read only; job 699824 runs it)

T, V = t2b.T, t2b.V
_TAG_OF = T.tag_of
FORBIDDEN = os.path.realpath(os.path.join(HERE, "results", "t2b"))


def parse_wrapper_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--tag-suffix", default="_wall")
    p.add_argument("--print-only", action="store_true")
    mine, rest = p.parse_known_args(argv)
    if not mine.tag_suffix:
        raise SystemExit("--tag-suffix must be non-empty")
    return mine, rest


def main(argv=None):
    mine, rest = parse_wrapper_args(sys.argv[1:] if argv is None else argv)
    T.tag_of = lambda a, m: _TAG_OF(a, m) + mine.tag_suffix
    a = t2b.parse_args(rest)
    if os.path.realpath(a.outdir) == FORBIDDEN:
        raise SystemExit(f"refusing --outdir {a.outdir}: that is the T2b folder read by t2b_report.py")
    if mine.print_only:
        m = T.build(a)
        ck = [int(s) for s in a.ckpts.split(",") if s.strip()]
        avg_start = int(round(a.iters * (1.0 - a.avg_frac)))
        print(f"tag {T.tag_of(a, m)} arm {m.arm} L {m.L} H {m.H} P {m.nparam} seed {a.seed} iters {a.iters} "
              f"ckpts {ck} (fractions {[round(c / a.iters, 3) for c in ck]}) polyak steps {avg_start + 1}-{a.iters} "
              f"lr(1) {V.lr_at(a, 1):.4f} lr(N/2) {V.lr_at(a, a.iters // 2):.4f} lr(N) {V.lr_at(a, a.iters):.4f} "
              f"max_hours {a.max_hours} exact_eval {a.exact_eval} n_eval {a.n_eval} outdir {a.outdir}")
        return
    t2b.main(rest)


if __name__ == "__main__":
    main()
