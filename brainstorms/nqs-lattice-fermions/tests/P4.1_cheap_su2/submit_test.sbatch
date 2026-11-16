#!/bin/bash
# P4.1 decisive test: exact-SU(2) charge-conditioned singlet geminal (r1, r1x) vs
# spin-projected Jastrow-pair-product foil, 3x4 torus Hubbard, Ne=10, U=8, t'=-0.3, S=0.
# Array: 5 ansaetze x 3 seeds = 15 independent CPU tasks (exact sums, no sampling).
# After all tasks finish, run the report (cheap, login node is fine):
#   /share/apps/software/anaconda3/bin/python3 micro_test.py --mode report --route r1x
#   /share/apps/software/anaconda3/bin/python3 micro_test.py --mode report --route r1
#SBATCH --job-name=P4.1_su2
#SBATCH --partition=kim
#SBATCH --array=0-14
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:30:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2/logs/P4.1_su2_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2/logs/P4.1_su2_%A_%a.err

set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

ANSATZE=(foil agp r1 r1x r2)
ANSATZ=${ANSATZE[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))

cd "$DIR"
$PY -u micro_test.py --mode run --Lx 3 --Ly 4 --Ne 10 --U 8 --tp -0.3 --spin 0 \
    --ansatz "$ANSATZ" --seed "$SEED" --hidden 16 --maxiter 4000 --max-hours 21
