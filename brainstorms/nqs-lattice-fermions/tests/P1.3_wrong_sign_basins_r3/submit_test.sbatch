#!/bin/bash
#SBATCH --job-name=P1.3r3
#SBATCH --partition=kim
#SBATCH --array=0-119%30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.3 round 3: wrong-sign basins after symmetry projection. CPU only.
# One array task per (instance, seed, arm): t = 0..119 ->
#   instance = (c1, c2)[t // 60]   c1: 4x3 PBC 4+4 U=6 t'=-0.37 (gap_in 0.234t, smallest found)
#                                  c2: 4x3 PBC 4+4 U=8 t'=-0.34 (gap_in 0.273t)
#   seed     = 1234 + (t % 60) // 10   (6 seeds)
#   arm      = (U, PSaf, PSfs, PScdw, PW0, PW1, N, A, O, O0)[t % 10]
# Cost: P1.1-P job 407525 measured projected SR of NNBF width 32 on D=245025 at 1.64 s/step
#   (16 CPUs, kim). The P_S factor adds 0.02 s per step (timing on the interactive node).
#   Per task: 1500 steps (1800 for PW0/PW1) x ~1.8 s = 45-55 min, plus ED and sector
#   Lanczos (~1 min) and the E_sign evaluation (L-BFGS-B, 3 starts, plus ten Haaf; ~5 min;
#   two evaluations for arm U), so <= ~1.2 h per task; 6 h leaves a 5x margin.
#   Memory: P1.1-P peaked at 3.4 GB on D=627264; D=245025 needs well under 8 GB.
# Submit from this directory so the logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
# Afterwards (seconds, any node): python3 micro_test.py --aggregate results
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 micro_test.py --task-id "${SLURM_ARRAY_TASK_ID}" \
    --width 32 --budget 1500 --warmup 300 --cycles 2 --relax 40 --null-k 3 \
    --probe-at 600 --probe 300 --lam 2.0 --pretrain 300 --acc-samples 2000 \
    --final-window 100 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
    --fn-iters 4 --fn-floor 1e-3 --lbfgs-iters 3000
