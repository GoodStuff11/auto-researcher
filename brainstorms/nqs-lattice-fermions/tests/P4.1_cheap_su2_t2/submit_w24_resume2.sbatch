#!/bin/bash
# One-off resume (2026-11-16): the --max-hours cap is cumulative (t_used + elapsed), so the H=48 task, capped at step 956 of 1000, can only finish with a larger cumulative cap. Same command otherwise; no rule change.
# P4.1 round-7 premise W, W-24 (written exception of the round-7 critic: about 5 CPU-hours per task, 2 tasks).
# Open 8x3 strip, N_e = 20, U = 8, t' = -0.25, (+,+) irrep (P_G (1,1), as job 686990 at Lx = 8).
#   task 0: che at H = 16      task 1: che at H = 48
# Protocol = the 12-site pilot: 1000 SR steps, 1024 chains, lr 0.05 with cosine decay, Polyak average of the last 250
# steps; final energy from 65,536 Metropolis samples (chain-blocked error bar). No early stop (premise measurement).
# Reason for the exception: the bottleneck is a statement about H against N, and 24 sites is the smallest size with
# N/H >= 1.5 at H = 16. Sizing (logs/timing_L12_24_output.txt): t_step(24 sites)/t_step(12 sites) = 3.6 for che at H = 16,
# H = 48 within noise of H = 16, so about 4-4.5 CPU-hours per task from the 1.0 CPU-hour 12-site pilot. Hard cap
# --max-hours 2.0 at 4 threads = 6 CPU-hours; a capped task writes capped = true and resumes on resubmission.
# Afterwards (login node): python3 t2_report.py --mode report  (W needs the 24-site (+,+) reference of job 687302)
#SBATCH --job-name=P41w24r2
#SBATCH --partition=kim
#SBATCH --array=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_t2/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_t2/logs/%x_%A_%a.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_t2
cd "$DIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
HS=(16 48)
H=${HS[$SLURM_ARRAY_TASK_ID]}
ARGS="--mode train --arm che --width $H --Lx 8 --Ly 3 --pg=1,1 --iters 1000 --n-chains 1024 --n-eval 65536"
ARGS="$ARGS --threads 4 --max-hours 2.0 --outdir $DIR/results/w24"
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): vmc_t2.py $ARGS"
~/.conda/envs/lit_env/bin/python -u vmc_t2.py $ARGS
