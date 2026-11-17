#!/bin/bash
#SBATCH --job-name=P3.1r13_t2block_eta20
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:42:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# P3.1 R12.3d T2 RERUN at the measured trained noise level ETA = (2.0, 1.2) (matches measured total eta 0.85; see RULE), prepared 2026-11-17.
# PREPARED, NOT SUBMITTED: submit only via compute_gate.py submit with voi.md. Rule: RULE_R12.3d_eta20.md.
# Seeds 11, 22, 33 all recomputed (no --reuse: seed 11 on disk was computed at ETA = (0.3, 0.5)). DAV-0 skipped.
# Single arm per seed (DAV-R at ETA = (2.0, 1.2)); the noise amplitude does not change the solves, so ~6.1 min/seed.
# Wall time from job 731418 (elapsed 25:45): ED + 31 levels + validity 13.6 min (start to 02:19:31), seeds 22 and 33
# 6.0 and 6.1 min -> 13.6 + 3 x 6.1 = 31.9 min, +30% = 41.5 min -> 42 min. Memory: 731418 MaxRSS 3.8 GB.
# Verdict afterwards: python3 t2_block_premise_eta.py --aggregate results/t2_block_eta20.json
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 t2_block_premise_eta.py --instance main --n-low 31 --window-all --cpu-budget 0 \
    --eta 2.0 1.2 --out results/t2_block_eta20.json
