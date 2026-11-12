#!/bin/bash
#SBATCH --job-name=P4.2_r5_cap
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r5/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r5/%x_%j.err
# P4.2 round 5 CAPACITY PILOT (critic fix 1): the scr arm only, seed 0, 600 projected-SR steps of
# MFNG-O (426 parameters) on the 4x4 OBC t-t'-J 6+6 target. Read-out and pre-committed rule:
#   python3 obc_transfer.py --capacity results_capacity
#   relerr_inf > 0.03 -> MUTATE (do not run submit_obc.sbatch); <= 0.03 -> run submit_obc.sbatch.
# Cost: 110-140 s per step on 8 threads (timing_output.txt) -> 600 steps 18-23 h, plus ~5 min
# target setup and 8 checkpoint records with the sign budget. --wall-hours 28 stops the SR loop
# cleanly inside the 30 h limit; the JSON is rewritten every 10 steps.
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r5
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
cd "$DIR"
mkdir -p results_capacity
$PY obc_transfer.py --arm scr --seed 0 --steps 600 --ckpts 10,20,50,100,150,300,450,600 \
    --wall-hours 28 --out "$DIR/results_capacity/scr_s0.json"
