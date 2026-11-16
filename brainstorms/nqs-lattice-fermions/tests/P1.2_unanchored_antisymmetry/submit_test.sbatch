#!/bin/bash
#SBATCH --job-name=P1.2_walsh_sign
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# P1.2 decisive test (CPU only). Submit from this directory so logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
$PY micro_test.py --mode smoke
$PY micro_test.py --mode holon_check
$PY micro_test.py --mode pairwise_check
$PY micro_test.py --mode decisive \
    --instances "4x4p:2:-0.3,4x4p:2:0.3,3x4o:2:0.0" \
    --J 0.4 --degree 3 --ntrain 4000 --l2 1e-3 --maxfev 80 --starts 2 \
    --fn-opt-iters 2 --fn-iters 10 --seed 0 \
    --out "P1.2_decisive_${SLURM_JOB_ID}.json"
