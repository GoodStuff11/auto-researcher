#!/bin/bash
# P4.1 round 5: decisive test (NOT submitted by the round-5 worker).
# Instance: 4x3 OPEN Hubbard, N_up = N_dn = 5, U = 8, t' = -0.25 (dim 627264), identical to job 451886.
# Arms (both exact S = 0, both 4 closed-form Che projections per amplitude, param-matched to NNBF width 16):
#   che_rbm (Route 5.5: che x prod_{m<2} 2cosh(b_m + w_m(c).sigma), exact P_S by hidden-unit enumeration)
#   che_k4  (Route 5.4 / equal-cost Gaussian-rank foil: 4 charge-backflowed UHF components, shared backflow)
# References (read, not rerun): che and nnbf_SG from ../P4.1_cheap_su2_fix/results (job 451886).
# Array 0-5: arm = task / 3, seed = task % 3. Afterwards (login node):
#   python3 r5_test.py --mode report
#SBATCH --job-name=P4.1r5
#SBATCH --partition=kim
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r5/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r5/logs/%x_%A_%a.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r5
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
ARMS=(che_rbm che_k4)
ID=${SLURM_ARRAY_TASK_ID}
cd "$DIR"
$PY -u r5_test.py --mode run --ansatz "${ARMS[$((ID / 3))]}" --seed $((ID % 3)) \
    --nnbf-hidden 16 --maxiter 2000 --max-hours 22
