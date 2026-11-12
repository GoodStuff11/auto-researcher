#!/bin/bash
#SBATCH --job-name=P1.3r5_basin
#SBATCH --partition=kim
#SBATCH --array=0-35%12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.3 round 5: basin versus capacity on the shared 4x4 OBC t-t'-J 6+6 instance (see micro_test.py
# docstring for arms and the pre-committed rule).
# Task id -> width = [6, 4, 10][id // 12], arm = [Cuhf, Cfs, Ccdw, W][(id % 12) // 3],
#            seed = [11, 22, 33][id % 3]. Tasks 0-11 are width 6 (premise gate + reference width).
# Each task: 200 fidelity (W) or energy (cold) SR steps + 700 energy SR steps, exact sums on
# D = 1681680, plus 4 (W) or 3 (cold) P1.1 budgets and ED.
# Cost (timing_output.txt; P2.1 job 410988 measured 12.7 s/step for width 6 at 16 CPUs over 1200
# steps): 900 steps x 10-20 s = 2.5-5 h, budgets 4 x ~6 min, ED + calibration ~5 min.
# Worst case (width 10) ~6 h; 14 h leaves > 2x margin.
# Submit from this directory so logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
# Afterwards (any node, seconds): python3 micro_test.py --mode aggregate --instance 0
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
${PY} micro_test.py --mode train --instance 0 --task "${SLURM_ARRAY_TASK_ID}" --pre 200 --rel 700 \
    --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3
