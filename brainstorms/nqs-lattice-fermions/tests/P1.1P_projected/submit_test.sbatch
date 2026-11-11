#!/bin/bash
#SBATCH --job-name=P1.1P_proj
#SBATCH --partition=kim
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.1-P: symmetry-projected sign budget. CPU only. One array task per (instance, seed, arm):
#   t = 0..17 -> instance = t // 6 (0: 4x3 5+5 U=8; 1: 4x3 4+4 U=8; 2: 4x3 5+5 U=4; t'=-0.25),
#                seed = (1234, 2345, 3456)[(t // 2) % 3], arm = (U, P)[t % 2].
#   Arm U trains plain NNBF (P1.1 protocol) and budgets U, U->P, U_S, UP_S;
#   arm P trains projected-SR NNBF and budgets P, P_S. Widths 8, 32, 128; 1500 SR steps each
#   (P1.1 settings, matched in both arms).
# Submit from this directory so the logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
# Cost (measured, one SR step, 16 threads, login node; timing_output.txt):
#   instance 1 (D=245025): U 1.1 / 1.4 / 2.1 s, P 1.1 / 1.3 / 1.9 s for widths 8 / 32 / 128;
#   instance 0 (D=627264): U 2.5 / 3.3 / 4.7 s, P 2.9 / 3.3 / 4.8 s.
#   P1.1 job 329057 (8 CPUs) measured 4.9 / 5.5 / 7.9 s per step on instance 0. Worst case,
#   using the P1.1 rates: 1500 x (4.9 + 5.5 + 7.9) s = 7.6 h of SR, plus budgets (~4 min
#   each on D=627264; 12 for arm U, 6 for arm P) and ED/calibration (~5 min), so <= ~9 h per
#   task on instances 0 and 2 and ~4 h on instance 1; 24 h leaves > 2.5x margin.
#   Memory: P1.1 peaked at 3.4 GB on D=627264; 16G gives > 4x margin.
# Afterwards (any node, seconds): python3 micro_test.py --aggregate results
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --widths 8,32,128 --sr-iters 1500 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
    --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3 --rng-replay 1
