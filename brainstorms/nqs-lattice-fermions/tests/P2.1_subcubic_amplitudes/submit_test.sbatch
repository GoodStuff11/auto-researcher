#!/bin/bash
#SBATCH --job-name=P2.1_lgb_backflow_scaling
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.1_subcubic_amplitudes/P2.1_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.1_subcubic_amplitudes/P2.1_%j.err
# P2.1 decisive test: exact fast ratios for Local+Global-Bottleneck backflow vs brute force.
# CPU only, single thread so that wall-time exponents are not distorted by BLAS threading.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.1_subcubic_amplitudes
cd "$DIR"
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --sizes 8,12,16,20,24,28,32 --delta 0.125 --tp -0.2 --U 8 \
    --r 1 --F 16 --K 4 --nconf 3 --fit-last 4 --seed 1234 \
    --out "$DIR/results_P2.1.json"
