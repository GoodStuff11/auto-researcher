#!/bin/bash
# Round 13 (amendment A1): hg accuracy on sizes 20 and 23, and 13.1' cost at L = 24 (dense2, full_S); 4 processes.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/jek_numba_r13
cd /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.1_subcubic_amplitudes_r13
mkdir -p logs
for inst in 20 23; do
  ( ulimit -t 600; /usr/bin/time -f "CPU %U+%S s" python3 t1_gated.py --mode train --arm hg --instance $inst --max-cpu 480 > "logs/acc_hg_inst${inst}.log" 2>&1 ) &
done
for a in dense2 full_S; do
  ( ulimit -t 600; /usr/bin/time -f "CPU %U+%S s" python3 schur.py --mode run --arm "$a" --tag C24 --sizes 24 --budgets 360 > "logs/run_${a}_C24.log" 2>&1 ) &
done
wait
