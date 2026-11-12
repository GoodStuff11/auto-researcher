#!/bin/bash
#SBATCH --job-name=P3.1r5_target
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# P3.1 round 5, step 1: target measurement (pair / spin / charge error of the converged projected
# NNBF width-6 states vs ED) on all three seeds of the shared 4x4 OBC t-t'-J instance. Seeds 22 and
# 33 are produced by P2.1 job 451706, so submit with --dependency=afterany:451706.
# Submit from this directory: sbatch --dependency=afterany:451706 submit_target.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 target_probe.py --seeds 11,22,33 --spectrum 0 --out results/target_probe_all.json
