#!/bin/bash
#SBATCH --job-name=P3.1r10_t2v2
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# P3.1 Round 10: T2-v2 premise run for Route 10.1u (2026-11-16). One task, 2 CPUs x 1 h = 2 CPU-h (the T2 cap,
# no exception). ED once; seeds 11, 22, 33 of the real width-6 projected NNBF on the 4x4 OBC 6+6 instance;
# exact responses (minres), no training. Rule, early stops and decision tree: t2v2_scru_premise.py docstring
# and the leaf (Round 10, "T2-v2"). Validity against job 698163 is checked before any minres solve.
# Verdict afterwards: python3 t2v2_scru_premise.py --aggregate results/t2v2_main.json
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 t2v2_scru_premise.py --instance main --cpu-budget 6600 \
    --wall-budget 3300 --out results/t2v2_main.json
