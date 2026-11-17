#!/bin/bash
#SBATCH --job-name=P3.1r12_t2block_resume
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:32:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# P3.1 Round 12, route 12.3d T2 (Davidson-block interval), RESUME of cancelled job 707298, prepared 2026-11-17.
# PREPARED, NOT SUBMITTED: submit only via compute_gate.py submit with voi_t2_block_resume.md.
# Seed 11 is reused from results/t2_block_main.json (--reuse), not recomputed; seeds 22 and 33 (5 scored items) run.
# Both seeds in one task: neither seed alone can decide the verdict (voi note, cheapest-version), and a second job
# would repeat the ED + 31-level eigensolve (about 0.4 CPU-h). DAV-0 is skipped (--cpu-budget 0; it decides nothing).
# Wall time from 707298: ED + 31 levels + validity 11.7 min, seed 11 6.2 min -> 11.7 + 2 x 6.2 = 24.1 min, +30% = 32 min.
# Memory: 707298 MaxRSS 3.8 GB.
# Verdict afterwards: python3 t2_block_premise.py --aggregate results/t2_block_main_resume.json
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 t2_block_premise.py --instance main --n-low 31 --window-all --cpu-budget 0 \
    --reuse results/t2_block_main.json --out results/t2_block_main_resume.json
