#!/bin/bash
# P4.1 R45 from-init pilot (auditor CHANGES REQUIRED on T2 digest 08d370f1663e): task 0 = che, task 1 = nnbf, seed 0,
# 150 fidelity evaluations each from the T2's V.Model init. Rule: RULE_pilot.md; score: pilot_score.py (run after).
# Prepared by the round-30 worker, NOT submitted; submit only through compute_gate.py submit with voi_pilot.md.
# Wall time (measured 2026-11-17 interactively, logs/check_local.txt): che 0.117, nnbf 0.065 CPU-min per evaluation,
# setup 0.3. 150 evaluations: che 17.6, nnbf 9.8 CPU-min; x 1.8 in-job = 31.6 / 17.6; + setup, x 1.3 margin
# -> 00:45:00 for both array tasks. Cap 2 x 0.75 = 1.5 CPU-h; expected about 0.85 CPU-h. Peak RSS 1.66 GB -> 5G.
#SBATCH --job-name=P41r45p
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:45:00
#SBATCH --array=0-1
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r45/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r45/logs/%x_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r45
cd "$DIR"
T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" = "0" ]; then ARM=che; else ARM=nnbf; fi
echo "r45 pilot task $T: arm $ARM seed 0 on $(hostname), job ${SLURM_JOB_ID:-local}"
exec ~/.conda/envs/lit_env/bin/python -u r45.py --mode train --arm "$ARM" --seed 0 --maxfun 150 --cpu-min 40 \
    --threads 1 --res "$DIR/results_pilot"
