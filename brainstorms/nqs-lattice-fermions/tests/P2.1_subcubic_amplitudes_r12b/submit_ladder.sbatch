#!/bin/bash
#SBATCH --job-name=P2.1r12b_ladder
#SBATCH --partition=kim
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# P2.1 Round 12b: exact-sum ladder of the sublattice sign-capacity test (leaf "Round 12b: exact-sum ladder").
# Rule: Round 12 T1 rule, unchanged (t1_sub.decide). Task table: python3 ladder.py --mode tasks
#   0-2 instance 23 (4x3 5+5, V=12) full/half/quarter; 3-5 instance 24 (7x2 6+6, V=14); 6-8 instance 25 (5x3 7+7, V=15)
# Convergence: plateau test of Round 12 (from step 300, every 50 steps, best E improved < 2e-4 t over 200 steps);
#   cap 6000 steps or 10800 SR CPU-s, else converged = false. Checkpoints every 250 steps; a resubmitted task resumes.
# Budget (probe, 1 thread): about 0.26 / 1.68 / 1.16 CPU-s per full step at V = 12 / 14 / 15;
#   worst case per task 0.45 / 3.0 / 2.0 CPU-h (step or CPU cap), at most about 16.5 CPU-h for the array.
# Anytime report: python3 ladder.py --mode progress    Verdict: python3 ladder.py --mode aggregate
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/share/apps/software/anaconda3/bin/python3
${PY} ladder.py --mode train --task "${SLURM_ARRAY_TASK_ID}"
