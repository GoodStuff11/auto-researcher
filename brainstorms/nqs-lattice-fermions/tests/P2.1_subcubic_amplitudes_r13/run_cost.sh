#!/bin/bash
# Round 13 T1 cost runs (pre-registered): wave A then wave B, single-threaded, ulimit -t 600 each.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/jek_numba_r13
cd /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.1_subcubic_amplitudes_r13
mkdir -p logs
wave() {
  tag=$1; shift
  for a in "$@"; do
    ( ulimit -t 600; /usr/bin/time -f "CPU %U+%S s" python3 schur.py --mode run --arm "$a" --tag "$tag" > "logs/run_${a}_${tag}.log" 2>&1 ) &
  done
  wait
}
wave A dense2 full_S hg_S full_11c
wave B dense2 cbh_S half_S
