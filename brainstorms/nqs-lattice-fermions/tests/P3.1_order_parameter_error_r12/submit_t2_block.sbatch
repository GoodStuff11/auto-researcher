#!/bin/bash
#SBATCH --job-name=P3.1r12_t2block
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# P3.1 Round 12: T2 premise run for Route 12.3d (Davidson-block bracket), 2026-11-16. PREPARED, NOT SUBMITTED.
# One task, 2 CPUs x 2 h = 4 CPU-h allocated (expected about 2.5 CPU-h; above the 2 CPU-h T2 cap, exception
# in the leaf's value-of-information note). ED, 31 in-sector levels (window = all 30 excited levels, omega <= 0.84 t), then seeds 11, 22, 33 of the real width-6 NNBF states on the
# 4x4 OBC 6+6 instance; validity against job 698163 before any minres solve. Rule and decision tree: leaf, Round 12,
# "T2 premise run for 12.3d", and the t2_block_premise.py docstring.
# Verdict afterwards: python3 t2_block_premise.py --aggregate results/t2_block_main.json
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 t2_block_premise.py --instance main --n-low 31 --window-all --cpu-budget 13800 \
    --out results/t2_block_main.json
