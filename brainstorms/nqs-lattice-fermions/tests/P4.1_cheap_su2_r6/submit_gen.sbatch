#!/bin/bash
# P4.1 generality test (pre-committed after the k-family PASS of job 508440). Rules: leaf artifact,
# "Generality and rbm-extension tests (2026-11-14)". Two instances, 16 tasks each (array 0-31):
#   tasks  0-15: cyl  = 4x3, periodic x (length 4) / open y, 5+5, U = 8, t' = -0.25 (dim 627264)
#   tasks 16-31: fill = 4x3 open, 4+4, U = 8, t' = -0.25 (dim 245025)
#   within an instance (j = task % 16): j = 0-14 -> arm = ARMS[j / 3], seed = j % 3, 2000 L-BFGS iterations,
#   ARMS = che nnbf_SG (width 16, P_ref) che_k2 che_k4 che_k8 (H_K = largest H with count <= P_ref);
#   j = 15 -> che drift control (seed 0, 4000 iterations).
# Sizing: on the 627k open instance che took <= 12.3 h, che_k8 <= 22.9 h, che_drift4000 26.3 h (16 CPUs);
# --max-hours 72 is >= 2.7x every estimate (cylinder timing: logs/prep_cyl_output.txt). Capping is recorded.
# Afterwards (login node): python3 gen_test.py --mode report
#SBATCH --job-name=P4.1gen
#SBATCH --partition=kim
#SBATCH --array=0-31
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
INST=(cyl fill)
ARMS=(che nnbf_SG che_k2 che_k4 che_k8)
ID=${SLURM_ARRAY_TASK_ID}
I=${INST[$((ID / 16))]}
J=$((ID % 16))
cd "$DIR"
if [ "$J" -eq 15 ]; then
    $PY -u gen_test.py --mode run --instance "$I" --ansatz che --drift --seed 0 --maxiter 4000 --max-hours 72
else
    $PY -u gen_test.py --mode run --instance "$I" --ansatz "${ARMS[$((J / 3))]}" --seed $((J % 3)) \
        --maxiter 2000 --max-hours 72
fi
