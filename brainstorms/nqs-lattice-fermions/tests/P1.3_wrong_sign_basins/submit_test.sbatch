#!/bin/bash
#SBATCH --job-name=p13_sign_basins_4x3
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P1.3_wrong_sign_basins/p13_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P1.3_wrong_sign_basins/p13_%j.err
#
# P1.3 decisive test (wrong-sign basins, orbital-swap basin hopping).
# Main instance: 4x3 PBC Hubbard, N_up = N_dn = 5 (doping 1/6), t' = -0.3, U = 8.
# Runs the NQS arm (16 threads) and the Jastrow-Slater foil arm (8 threads) in
# parallel. Each prints a SUMMARY line with eps_SR, eps_OS, eps_best, G, M and
# the PASS/FAIL/AMBIGUOUS verdict, and writes a results_*.json next to this file.
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P1.3_wrong_sign_basins
PY=/share/apps/software/anaconda3/bin/python3
cd "$DIR"

COMMON="--lx 4 --ly 3 --bc pbc --nup 5 --ndn 5 --t 1.0 --tp -0.3 --U 8.0 \
        --n-mf 3 --n-rand 3 --steps 800 --steps-pre 300 --relax-steps 15 --window 2"

OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  $PY micro_test.py $COMMON --ansatz nqs --ha 4 --hb 1 > "$DIR/p13_nqs_${SLURM_JOB_ID}.log" 2>&1 &
PID_NQS=$!
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  $PY micro_test.py $COMMON --ansatz foil > "$DIR/p13_foil_${SLURM_JOB_ID}.log" 2>&1 &
PID_FOIL=$!
wait $PID_NQS
wait $PID_FOIL
grep SUMMARY "$DIR/p13_nqs_${SLURM_JOB_ID}.log" "$DIR/p13_foil_${SLURM_JOB_ID}.log"
