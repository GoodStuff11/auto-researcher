#!/bin/bash
#SBATCH --job-name=P2.1r13_hg
#SBATCH --partition=kim
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# P2.1 Round 13: hole-gated hidden rows (13.2), ladder at sizes 23 (T1 completion), 24 and 25 (leaf "Round 13").
# Rule: Round 12 rule over sizes 20 (counted, rho = 0.83), 23, 24, 25. Verdict: python3 hg_ladder.py --mode aggregate
# Caps: 6600 SR CPU-s per task (+ one budget), under 2 CPU-h per task; 6000 steps; checkpoint every 250 steps (resume).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/share/apps/software/anaconda3/bin/python3
${PY} hg_ladder.py --mode train --task "${SLURM_ARRAY_TASK_ID}"
