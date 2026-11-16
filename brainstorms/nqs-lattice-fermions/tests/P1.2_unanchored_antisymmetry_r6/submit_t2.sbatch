#!/bin/bash
#SBATCH --job-name=P1.2r6_T2readout
#SBATCH --partition=kim
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.2 round 6, T2 one-seed pilot (2026-11-16): read-outs of the trained round-5 FRD-K state (seed 11,
# shift 1e-3) on the 4x4 OBC t-t'-J instance, no training. 4 CPUs x 30 min caps each task at 2 CPU-hours.
# task 0: identity energy check + |F| read-out budget (T2-a, vote share v) + V = 16 vote lift (Gate 4)
# task 1: identity energy check + frozen-frame (FRD-F) read-out budget (T2-b, u)
# Afterwards: python3 t2_readout.py --mode aggregate   (pre-registered rules in the t2_readout.py docstring)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 t2_readout.py --mode task --task "${SLURM_ARRAY_TASK_ID}"
