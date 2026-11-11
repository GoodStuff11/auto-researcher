#!/bin/bash
#SBATCH --job-name=P1.3r2_basins
#SBATCH --partition=kim
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.3 round-2 decisive test. One array task = one paired seed running all arms
# S, N, A, AE, B, W, NNBF width 32, 1200 SR steps per arm (W: +300 fidelity steps). CPU only.
#   tasks 0-7 : instance main (4x3 PBC, 4+4, U=8, t'=-0.25 = P1.1 instance 1; SIGN HALF ONLY,
#               no same-sector competitor), seeds 1234..1241
#   tasks 8-15: instance comp (same cluster at t'=-0.32, inside the same-sector avoided
#               crossing, in-sector gap ~0.27t), seeds 1234..1241
# Submit from this directory so logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
# Cost estimate (measured 2026-11-11 on kim-compute-04, 16 threads): one NNBF-32 SR step
# on D=245025 takes 1.37 s. Per task ~7500 step-equivalents (6 arms x 1200, W pretrain and
# FCD excursions at ~2x force cost) = ~2.9 h, plus ~20 FN scoring solves and 6 final
# evaluations (fixed node + 3 L-BFGS-B starts) plus the irrep scan, measured in the 400-step
# pre-diagnostic (see artifact) -> ~3.5-4 h per task; 12 h leaves > 3x margin.
# Afterwards (seconds): python3 micro_test.py --aggregate results
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
if [ "${SLURM_ARRAY_TASK_ID}" -lt 8 ]; then INST=main; IDX=${SLURM_ARRAY_TASK_ID}; else INST=comp; IDX=$((SLURM_ARRAY_TASK_ID - 8)); fi
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --instance "${INST}" --seed $((1234 + IDX)) --arms S,N,A,AE,B,W \
    --width 32 --budget 1200 --warmup 300 --cycles 2 --relax 40 --excursion 60 \
    --caps 0.8,0.5 --lam 2.0 --null-k 3 --pretrain 300 --final-window 100 \
    --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 --fn-iters 4 --lbfgs-iters 3000
