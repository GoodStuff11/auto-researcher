#!/bin/bash
# P4.1 R45 `sup-fit` T2, stage 1 (seed 0): task 0 = che, task 1 = nnbf. Rule: RULE_T2.md. Card: leaves/P4.1/cards/R45.md.
# Prepared by the round-30 worker, NOT submitted; submit only through compute_gate.py submit with voi.md.
# Wall time (measured 2026-11-17 interactively, 1 thread, logs/check_local.txt): che 0.117, nnbf 0.065 CPU-min per
# function evaluation; setup 0.3 min. 2000 evaluations: che 234, nnbf 130 CPU-min interactive; x 1.8 in-job slowdown
# (R42/R43 measured 1.5-1.8x) = 421 / 234; safety CPU caps 485 / 270 (x 1.15). Time limit = che cap + 10 min, x 1.1
# -> 09:00:00 for both array tasks. Cap 2 x 9 = 18 CPU-h; expected about 7.0 + 3.9 = 10.9 CPU-h.
# Memory: peak RSS measured locally (logs/rss_local.txt) -> --mem below.
# SIGUSR1 at 600 s before the limit: checkpoint, exit 4; resubmitting resumes from the best theta.
#SBATCH --job-name=P41r45
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=09:00:00
#SBATCH --array=0-1
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r45/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r45/logs/%x_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r45
cd "$DIR"
T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" = "0" ]; then ARM=che; CAP=485; else ARM=nnbf; CAP=270; fi
echo "r45 task $T: arm $ARM seed 0 on $(hostname), job ${SLURM_JOB_ID:-local}"
# exec: python replaces the batch shell, so --signal=B:USR1 reaches python directly
exec ~/.conda/envs/lit_env/bin/python -u r45.py --mode train --arm "$ARM" --seed 0 --maxfun 2000 --cpu-min "$CAP" \
    --threads 1 --res "$DIR/results"
