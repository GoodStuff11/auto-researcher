#!/bin/bash
# P4.1 round 5 (fix): rank-ladder test of charge-conditioned projected UHF mixtures.
# Instance: 4x3 OPEN Hubbard, N_up = N_dn = 5, U = 8, t' = -0.25 (dim 627264), identical to job 451886.
# All arms exact S = 0 (P_G + exact P_S), common che start + independent component noise, 2000 L-BFGS iterations,
# parameters within +-1% of 2589 (see r5_test.py ARM_SPEC):
#   tasks  0-20: arm = NEW_ARMS[task / 3], seed = task % 3, NEW_ARMS = che_rbm1 che_rbm2 che_rbm3 (rank 2, 4, 8;
#                c-varying row scalings), che_k2 che_k4 che_k8 (fixed offsets, shared backflow), che_kh4 (containing
#                foil, own backflow head per component).
#   task     21: che (H = 16, 2637 parameters), seed 0, 4000 iterations: drift control.
# References (read, not rerun): che and nnbf_SG JSONs and logs of job 451886 (../P4.1_cheap_su2_fix).
# Sizing: login timing (logs/timing_output.txt) puts the slowest arm (che_k8) at 1.72x che per energy+gradient;
# che took <= 12.3 h for 2000 iterations on this partition, so che_k8 ~ 21 h and the drift run ~ 25 h.
# --max-hours 72 is >= 2.9x every estimate, so no task should be capped (capping is recorded and handled by report).
# Afterwards (login node): python3 r5_test.py --mode report
#SBATCH --job-name=P4.1r5fix
#SBATCH --partition=kim
#SBATCH --array=0-21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3-08:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r5/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r5/logs/%x_%A_%a.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r5
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
ARMS=(che_rbm1 che_rbm2 che_rbm3 che_k2 che_k4 che_k8 che_kh4)
ID=${SLURM_ARRAY_TASK_ID}
cd "$DIR"
if [ "$ID" -eq 21 ]; then
    $PY -u r5_test.py --mode run --ansatz che --seed 0 --nnbf-hidden 16 --maxiter 4000 --max-hours 72
else
    $PY -u r5_test.py --mode run --ansatz "${ARMS[$((ID / 3))]}" --seed $((ID % 3)) \
        --nnbf-hidden 16 --maxiter 2000 --max-hours 72
fi
