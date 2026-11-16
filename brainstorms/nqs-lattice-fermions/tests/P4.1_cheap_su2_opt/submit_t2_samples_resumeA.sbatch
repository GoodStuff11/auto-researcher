#!/bin/bash
# One-off resume (2026-11-16): arm A was capped at step 288 of 300 by the cumulative 0.45 h wall cap; it already had more wall time than arm B, so finishing its last 12 steps can only favour A (against the PASS direction). Rule unchanged.
# P4.1 round 9, route R15b `ns-rich` T2 (prepared, NOT submitted by the worker): is the 24-site che energy limited by
# the SR noise floor of the 1024-sample protocol? Warm start = W-24 che H = 16 averaged parameters (job 698224 task 0).
#   task 0 (A): 1024 chains, 300 steps     task 1 (B): 4096 chains, 75 steps  (equal wall: t_A/t_B = 0.2485,
#   logs/t2_timing24_output.txt, 4 threads, this node). Peak lr 0.02 for both, 686990 schedule shape, n_eval 65,536.
# Budget per task: train about 20 min + evaluation about 4.6 min + thermalisation <= 2 min, at 4 threads: about
# 1.8 CPU-hours (T2 cap 2). --max-hours 0.55 caps the training loop. No in-job futility: a task is under 30 min.
# Afterwards (login node): python3 t2_samples.py --mode report
#SBATCH --job-name=P41nsA_resume
#SBATCH --partition=kim
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:50:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_opt/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_opt/logs/%x_%A_%a.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_opt
cd "$DIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
ARMS=(A B)
ITERS=(300 75)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
IT=${ITERS[$SLURM_ARRAY_TASK_ID]}
ARGS="--mode warm --arm $ARM --iters $IT --threads 4 --max-hours 0.55 --n-eval 65536 --lr 0.02 --outdir $DIR/results_t2"
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): t2_samples.py $ARGS"
~/.conda/envs/lit_env/bin/python -u t2_samples.py $ARGS
