#!/bin/bash
#SBATCH --job-name=P2.1r15_resume_half_V15
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# P2.1 route R15.1 resume (RULE_resume.md): job 742029 was INCONCLUSIVE only because half_F16 hit its 480 CPU-s cap
# still falling (3.7e-4 > 2e-4 over 50 it). Resume half_F16 ONLY from its polished x_pol, L-BFGS to the convergence
# guard (<= 2e-4 t over 50 it; full_F4 already meets it, 2.7e-5) or 7200 CPU-s / 3000 it, then the unchanged budget.
# Thresholds unchanged (KILL > 1.25, PASS <= 1.15); INCONCLUSIVE again -> AMBIGUOUS, no more compute.
# Wall time from the local probe (resume_probe_output.txt, single thread): setup 3 s, 1.89-2.56 CPU-s/iteration
#   (2.56 = job 742029 average), budget 9-16 s (x2), L-BFGS cap 7200 s (+ <= 3 s for the iteration in progress)
#   -> <= 7240 s; +30% = 9412 s = 2 h 37 min -> 02:40:00 (cap 2.67 CPU-h).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/share/apps/software/anaconda3/bin/python3
${PY} polish_resume.py --mode resume --max-iter 3000 --max-cpu 7200
${PY} polish_resume.py --mode aggregate
