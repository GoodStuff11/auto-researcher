#!/bin/bash
# P4.1 round-4 fix: decisive accuracy pilot (NOT submitted by the fix worker).
# Instance: 4x3 OPEN Hubbard cluster, N_up = N_dn = 5, U = 8, t' = -0.25 (dim 627264; N_s = 10 configurations
# carry ~72% of psi_0's weight). Arms param-matched to NNBF width 16, all trained with P_G from the SAME
# reference (Fermi sea x Gutzwiller); che, sfb, nnbf_SG also with exact P_S:
#   tocchio (1 symmetric charge-conditioned AGP), kagp2 (Route 4.2), che (charge-only, exact S),
#   sfb (Route 4.1: che + one sublattice spin feature, exact S by bucketing), nnbf_SG (spin-resolved, exact S:
#   the reference sfb must approach).
# Array 0-14: arm = task / 3, seed = task % 3. Afterwards (login node):
#   python3 fix_test.py --mode report --Ne 10
# Requires results/complete_L4x3xobcobc_N10_U8_tp-0.25.json (submit_complete.sbatch) for the Route 4.2 premise.
#SBATCH --job-name=P4.1_acc_fix
#SBATCH --partition=kim
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_fix/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_fix/logs/%x_%A_%a.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_fix
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
ARMS=(tocchio kagp2 che sfb nnbf_SG)
ID=${SLURM_ARRAY_TASK_ID}
cd "$DIR"
$PY -u fix_test.py --mode run --Ne 10 --ansatz "${ARMS[$((ID / 3))]}" --seed $((ID % 3)) \
    --nnbf-hidden 16 --maxiter 2000 --max-hours 15
