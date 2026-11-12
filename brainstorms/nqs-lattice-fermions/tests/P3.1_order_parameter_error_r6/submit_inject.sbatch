#!/bin/bash
#SBATCH --job-name=P3.1r6_inject
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# P3.1 round 6: Saturation measurement on the real converged width-6 states (seeds present at run
# time) and the pre-registered injected low-energy check for route CLX on the shared 4x4 OBC
# t-t'-J instance (micro_test.py docstring). Writes results/inject.json only. Expected < 1.5 h.
# Submit from this directory: sbatch submit_inject.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 micro_test.py --mode inject --instance main --n-low 9 --n-seeds 5 \
    --mc-samples 200000 --n-boot 100 --lanczos-steps 120 --results results
