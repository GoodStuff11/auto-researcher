#!/bin/bash
# P4.1 R43 `flip-even-24`, stage 1 (seeds 0-1), arm nnbf_even only. Rule: RULE.md. Card: leaves/P4.1/cards/R43.md.
# Prepared by the round-29 worker, NOT submitted; submit only through compute_gate.py submit with voi.md.
# Task k: arm nnbf_even, seed k. Compared against R42's che (job 759907, ../P4.1_cheap_su2_r42/results), not rerun.
# Pinned to kim-cpu-01, where all four R42 tasks ran (removes the cross-machine step-rate confound).
# Equal compute (L35): each task trains for exactly 180 CPU-min (process_time) on ONE core, then scores.
# Wall time (measured 2026-11-17 on kim-compute-03, 1 thread, logs/probe_local.txt): nnbf_even 0.168 CPU-min/step
# (1.98x nnbf's 0.085 measured alongside), start-up 0.69 CPU-min. Training is capped by CPU time, so cluster slowness
# lowers the step count (expected ~600-640 steps), not the wall time. Uncapped parts x 1.8 (in-job/interactive):
# start-up 0.69 x 1.8 = 1.2 min; scoring: R42 nnbf took 23.9 min in-job; interactive estimate for nnbf_even
# 17 x 1.98 = 34 min, x 1.8 = 61 min (R42-scaled 23.9 x 1.98 = 47; take 61). Total 1.2 + 180 + 61 = 242 min,
# x 1.3 margin = 315 min -> 05:15:00. R42 wall = CPU to 1% (sacct 759907). Peak RSS 0.86 GB (probe) -> 3G.
# SIGUSR1 at 600 s before the limit: checkpoint written, exit 4 (INCOMPLETE); resubmitting resumes from it.
#SBATCH --job-name=P41r43
#SBATCH --partition=kim
#SBATCH --nodelist=kim-cpu-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=05:15:00
#SBATCH --array=0-1
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r43/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r43/logs/%x_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r43
cd "$DIR"
SEED=${SLURM_ARRAY_TASK_ID:-0}
echo "r43 task $SEED: arm nnbf_even seed $SEED on $(hostname), job ${SLURM_JOB_ID:-local}"
# exec: python replaces the batch shell, so --signal=B:USR1 (batch shell) reaches python directly
exec ~/.conda/envs/lit_env/bin/python -u r43.py --mode train --arm nnbf_even --seed "$SEED" --Lx 8 --guided 1 \
    --cpu-min 180 --n-chains 1024 --n-eval 131072 --n-s2 32768 --n-half 32768 --outdir "$DIR/results"
