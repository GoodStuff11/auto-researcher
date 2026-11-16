#!/bin/bash
#SBATCH --job-name=P2.1r12d_halfF
#SBATCH --partition=kim
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# P2.1 Round 12d: `half` at larger hidden width F on the Round 12b ladder (leaf "Round 12d: half at larger F").
# Rule: Round 12 T1 rule, unchanged (t1_sub.decide); primary comparison A = half_F16 vs full_F4 (704752 / Round 12).
# Task table (python3 ladder_d.py --mode tasks): 0-3 half_F16, 4-7 half_F8, 8-11 full_F8 on instances 20, 23, 24, 25.
# Convergence: Round 12 plateau test unchanged; cap 6000 steps or 10800 SR CPU-s, else converged = false.
# Checkpoints every 250 steps; a resubmitted task resumes. Results: results/<arm_key>/.
# Budget (1-thread probe, CPU-s per step): half_F16 0.40 / 2.28 / 1.57 at V = 12 / 14 / 15; full_F8 1.82 / 1.27 at V = 14 / 15.
#   Worst case per task about 3.0 CPU-h (V = 14, CPU cap binds); about 18.5 CPU-h for the array. Expected 4 to 6 CPU-h.
# Anytime report: python3 ladder_d.py --mode progress    Verdict: python3 ladder_d.py --mode aggregate
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/share/apps/software/anaconda3/bin/python3
${PY} ladder_d.py --mode train --task "${SLURM_ARRAY_TASK_ID}"
