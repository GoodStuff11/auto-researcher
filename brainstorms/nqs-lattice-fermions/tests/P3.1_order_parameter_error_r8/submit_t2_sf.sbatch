#!/bin/bash
#SBATCH --job-name=P3.1r8_t2sf
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# P3.1 round 8, Route 8.3 (structure-factor targeting), tier-2 premise pilot (2026-11-16).
# Exact sums on the 4x4 OBC 6+6 instance: 30 in-irrep levels, real P2.1 width-6 states (seeds 11/22/33),
# trained phi_1..3 (job 456094; phi_4..6 used only if job 618646 has written them). No training.
# Budget exception (written in the leaf, Round 8 tier log): about 6 CPU-hours, above the 2 CPU-hour T2 cap,
# because the only trained-NQS states with a measurable error exist on this D = 1.68e6 instance.
# Pre-registered rule: t2_sf_premise.py docstring and the leaf, Route 8.3.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 t2_sf_premise.py --instance main --n-low 30 --seeds 11,22,33 \
    --out results/t2_sf_premise.json
