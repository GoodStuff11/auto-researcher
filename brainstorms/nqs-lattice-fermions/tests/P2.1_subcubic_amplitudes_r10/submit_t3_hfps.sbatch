#!/bin/bash
#SBATCH --job-name=P2.1r10_T3hfps
#SBATCH --partition=kim
#SBATCH --array=0-34
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# P2.1 round 10, T3 of Route 7.1 (local-feature HFPS) against the cubic pairing-frame baseline
# (hfps_dense), the one-site pairing foil (hfps_site) and projected NNBF width 6, on instances 0, 3, 4,
# seeds 11, 22, 33, 1200 projected SR steps. Rule: leaf artifact, Round 10 "T3 rule, Route 7.1".
# Task table: python3 t3_hfps.py --mode tasks
#   0-26  hfps_plus, hfps_site, hfps_dense x seeds on instance 0 (0-8), 3 (9-17), 4 (18-26)
#   27-34 instance-4 references: nnbf s22, s33; agpj s11-33; skagp s11-33
# Reused, not run: round-5 nnbf / agpj / skagp on instances 0 and 3 and nnbf s11 on instance 4 (1200 steps).
# Checkpoints every 50 steps (results/ckpt); a resubmitted task resumes. Existing result JSONs are skipped.
# Futility checks (orchestrator, any node, seconds): python3 t3_hfps.py --mode checkpoint --at 300 / --at 600
# Verdict: python3 t3_hfps.py --mode aggregate
# Budget (written exception in the leaf): instance-0/3 HFPS tasks about 25-45 CPU-h each at 4 CPUs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
${PY} t3_hfps.py --mode train --task "${SLURM_ARRAY_TASK_ID}" --sr-iters 1200 --sr-lr 0.05 --sr-shift 1e-3 \
    --sr-samples 2000 --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3 --ckpt-every 50
