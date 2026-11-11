#!/bin/bash
# P4.1 round-3 decisive test: structurally exact-SU(2) routes (sqrvb = spin-charge factorised RVB
# with charge-conditioned spinon pairing; scf = charge determinant x charge-conditioned SU(2) MPS)
# against the PROJECTED control nnbf_SG (spin-resolved NNBF trained as P_G P_S phi), the exact-S
# charge-input control che (Che arXiv:2608.22925 family, P_G P_S), nnbf_G (P_G only), the sqrvb0
# ablation (uncoupled spinon pairing) and the mVMC-like foil (P_G P_S).
# Instance: 4x3 PBC Hubbard, N_up = N_dn = 4, U = 8, t' = -0.25 (dim 245025), SIGN HALF ONLY.
# Array 0-20: arm = task / 3 in (sqrvb sqrvb0 scf nnbf_SG nnbf_G che foil), seed = task % 3.
# Budget: every arm gets the same L-BFGS iteration cap (2000) and the same 20 h safety cap;
# see logs/timing_output.txt for the measured cost per iteration (scf is the slowest arm).
# Before the array (login node, seconds, already done): python3 micro_test.py --mode cost
# Afterwards (login node, seconds): python3 micro_test.py --mode report
#SBATCH --job-name=P4.1_su2_r3
#SBATCH --partition=kim
#SBATCH --array=0-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r3/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r3/logs/%x_%A_%a.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r3
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
ARMS=(sqrvb sqrvb0 scf nnbf_SG nnbf_G che foil)
ID=${SLURM_ARRAY_TASK_ID}
ARM=${ARMS[$((ID / 3))]}
SEED=$((ID % 3))
cd "$DIR"
$PY -u micro_test.py --mode run --Lx 4 --Ly 3 --Ne 8 --U 8 --tp -0.25 \
    --ansatz "$ARM" --seed "$SEED" --nnbf-hidden 16 --mps-mult 2,2,1,1 \
    --maxiter 2000 --max-hours 20 --chunk 16384 --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3
