#!/bin/bash
# P4.1 round-4 fix: fast per-configuration AGP completeness (replaces the timed-out tabular job 411996).
# 4x3 OPEN Hubbard cluster, U = 8, t' = -0.25: N_e = 10 (decisive instance, 45 min fit budget), N_e = 8
# (pilot-411494 instance, 25 min), N_e = 12 (half filling, ED + 20 min; last, optional). Each writes
# results/complete_<tag>.json on completion, so a time-out in a later cluster does not lose earlier ones.
# Login-node probe (4 processes, 2 min) in logs/complete_probe_N10_login.txt.
#SBATCH --job-name=P4.1_complete_fix
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:58:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_fix/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_fix/logs/%x_%j.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_fix
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$DIR"
$PY -u fix_test.py --mode complete --Ne 10 --procs 16 --budget-min 45
$PY -u fix_test.py --mode complete --Ne 8 --procs 16 --budget-min 25
$PY -u fix_test.py --mode complete --Ne 12 --procs 16 --budget-min 20
