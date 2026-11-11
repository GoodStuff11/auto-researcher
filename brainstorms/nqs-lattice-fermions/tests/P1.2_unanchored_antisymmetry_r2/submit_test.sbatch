#!/bin/bash
#SBATCH --job-name=P1.2r2_frd
#SBATCH --partition=kim
#SBATCH --array=0-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.2 round 2: factorized-ratio dressing (frdK kinetic frame, frdX exchange frame) versus
# NNBF x Jastrow at matched parameter count (H = 8, 32), trained by exact-force SR from the same
# UHF start, on the three P1.1 clusters. One array task per (instance, arm, width, SR shift):
#   instance = SLURM_ARRAY_TASK_ID / 12   (0, 1, 2 = P1.1 instances)
#   task     = SLURM_ARRAY_TASK_ID % 12   (row-major over arms {nnbfj,frdK,frdX} x widths {8,32}
#                                          x shifts {1e-3,1e-2})
# Cost estimate (measured on the login node, 4 threads, instance 1, D = 245025, H = 32):
#   nnbfj 2.5 s/it, frdX 5.8 s/it, frdK 8.5 s/it  ->  1500 its = 1.0 / 2.4 / 3.5 h.
#   Instances 0 and 2 have D = 627264 (x2.6): frdK ~9 h worst case at 4 threads, less at 8.
#   Plus ED (minutes) and one P1.1 sign budget per task (P1.1 measured ~35 min on D = 627264).
#   Worst task ~10 h, so 24 h leaves > 2x margin. P1.1 ran the same sizes in 32G.
# Submit from this directory so the logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
# Afterwards (any node, seconds): python3 micro_test.py --aggregate results
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
INST=$(( SLURM_ARRAY_TASK_ID / 12 ))
TASK=$(( SLURM_ARRAY_TASK_ID % 12 ))
/share/apps/software/anaconda3/bin/python3 micro_test.py --mode checks
/share/apps/software/anaconda3/bin/python3 micro_test.py --mode run \
    --instance "${INST}" --task "${TASK}" \
    --arms nnbfj,frdK,frdX --widths 8,32 --sr-shifts 1e-3,1e-2 \
    --sr-iters 1500 --sr-lr 0.05 --sr-samples 2000 \
    --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3 --seed 1234
