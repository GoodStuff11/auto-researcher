#!/bin/bash
#SBATCH --job-name=P3.1r9_t2resp
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.1 round 9: tier-2 premise test shared by Routes 10.1 (SCR) and 10.2 (TMW-SCR) (2026-11-16).
# Exact sums on the 4x4 OBC 6+6 instance, one real P2.1 width-6 state per array task (seeds 11/22/33),
# exact (noise-free) responses by minres; no training. Pre-registered rule: t2_resp_premise.py docstring and
# the leaf (Round 9, "T2 premise" block). Early stop per task if the oracle-response ceiling fails.
# Budget exception (written in the leaf): estimated 4-6 CPU-hours per task (about 12 minres solves of
# D = 1.68e6 plus ED), above the 2 CPU-hour T2 cap; the only trained-NQS states with a measurable tail error
# exist on this instance (same reason as the Route 8.3 T2, job 698163).
# Aggregate after all three tasks:
#   python3 t2_resp_premise.py --aggregate results/t2_resp_s11.json,results/t2_resp_s22.json,results/t2_resp_s33.json
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
SEEDS=(11 22 33)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
/share/apps/software/anaconda3/bin/python3 t2_resp_premise.py --instance main --seed "$S" \
    --out "results/t2_resp_s${S}.json"
