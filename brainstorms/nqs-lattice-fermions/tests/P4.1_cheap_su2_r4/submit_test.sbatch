#!/bin/bash
# P4.1 round-4 decisive ACCURACY test (the cost half is already measured: logs/cost_output.txt,
# results/cost_r4.json). Instance: 4x3 OPEN cluster, Hubbard, N_up = N_dn = 5, U = 8, t' = -0.25, dim 627264;
# N_s = 10 configurations carry weight, where one AGP per charge configuration is provably incomplete
# (Jacobian rank 26 < singlet dimension 42, logs/dims_output.txt). SIGN HALF ONLY.
# Arms (param-matched to NNBF width 16, all trained with P_G; che, sfb, nnbf_SG also with exact P_S):
#   tocchio (1 symmetric charge-conditioned det), kagp2 (Route 4.2), che (Che 2608.22925 family),
#   sfb (Route 4.1: charge NNBF + one sublattice spin feature, exact S by bucketing), nnbf_SG (control).
# Array 0-14: arm = task / 3, seed = task % 3. Run the pilot first (seed 0 of every arm):
#   sbatch --array=0,3,6,9,12 submit_test.sbatch      then the rest:  sbatch --array=1,2,4,5,7,8,10,11,13,14 submit_test.sbatch
# Requires results/psi0_L4x3xobcobc_N10_U8_tp-0.25.npz from submit_agp.sbatch (job 411774); otherwise each task
# recomputes ED (~minutes). Measured energy+gradient: 4.9 s (tocchio) ... 11.8 s (sfb) on 8 CPUs
# (logs/timing_output.txt) -> <= ~7.5 h for 2000 L-BFGS iterations.
# Afterwards (login node): python3 r4_test.py --mode report --Ne 10
#SBATCH --job-name=P4.1_acc_r4
#SBATCH --partition=kim
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r4/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r4/logs/%x_%A_%a.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r4
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
ARMS=(tocchio kagp2 che sfb nnbf_SG)
ID=${SLURM_ARRAY_TASK_ID}
cd "$DIR"
$PY -u r4_test.py --mode run --Lx 4 --Ly 3 --Ne 10 --U 8 --tp -0.25 --bc-x obc --bc-y obc \
    --ansatz "${ARMS[$((ID / 3))]}" --seed $((ID % 3)) --nnbf-hidden 16 --maxiter 2000 --max-hours 10.5
