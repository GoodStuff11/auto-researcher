#!/bin/bash
# P4.1 rbm extension (pre-committed after the rbm-family AMBIGUOUS of job 508440). Rules: leaf artifact,
# "Generality and rbm-extension tests (2026-11-14)". Instance: 4x3 open, 5+5, U = 8, t' = -0.25 (as 508440).
#   tasks 0-14: arm = ARMS[task / 3], seed = task % 3, ARMS = che_rbm1 che_rbm2 che_rbm3 che nnbf_SG.
#   rbm: resume from the saved 508440 parameters (iteration 2000) to 4000, empty L-BFGS history.
#   che, nnbf_SG: 451886 saved no parameters -> deterministic rerun from the common start to 2000, L-BFGS restart,
#   on to 4000 (one restart at 2000 for every arm; eps(2000) checked against 451886).
# Sizing: rbm took ~4.2 h per 2000 iterations; che 26.3 h for 4000 (drift run); nnbf_SG <= 12.4 h per 2000.
# --max-hours 72 >= 2.7x every estimate.
# Afterwards (login node): python3 gen_test.py --mode report-ext
#SBATCH --job-name=P4.1ext
#SBATCH --partition=kim
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3-08:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r6/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r6/logs/%x_%A_%a.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r6
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
ARMS=(che_rbm1 che_rbm2 che_rbm3 che nnbf_SG)
ID=${SLURM_ARRAY_TASK_ID}
cd "$DIR"
$PY -u gen_test.py --mode extend --ansatz "${ARMS[$((ID / 3))]}" --seed $((ID % 3)) --maxiter 4000 --max-hours 72
