#!/bin/bash
# P4.1 round 21, route R21 trSR T1 at 12 sites (che only). Rule: RULE.md (card rule verbatim + reported-only ladder arm).
# Prepared, NOT submitted by the worker; submit only through compute_gate.py submit with voi.md.
# Arms: trsr (cutoff 1e-4 rel, metric length 0.05), shift (abs 1e-3 Tikhonov, same length), ladder (reported only).
# Wall time (measured, 1 thread, logs/probe_*_1step_1thread.txt): set-up 0.25 CPU-min per arm; trsr/shift step
# 0.18 CPU-min; ladder step 0.38 CPU-min. 3 x 0.25 + 15 x (0.18 + 0.18 + 0.38) = 11.9 min; x1.3 = 15.5 -> 0:20:00.
# Memory: 2.0 GB peak RSS (probe) -> --mem=4G.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r24/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r24/logs/%x_%j.err
set -uo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r24
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
echo "r24 (R21 trSR) on $(hostname), job ${SLURM_JOB_ID:-local}"
# Do not overwrite a finished run; partial arms resume from results/*.json.
if [ -f results/verdict.json ] && grep -qE '"verdict": "(VIABLE|MARGINAL|NOT-VIABLE)"' results/verdict.json; then
    echo "results/verdict.json already holds a verdict; nothing to do"; exit 0
fi
set -e
$PY -u trsr.py --mode run --arm trsr   --steps 15 --nsamp 16384 --threads 1 --cpu-cap 5  | tee -a logs/trsr_output.txt
$PY -u trsr.py --mode run --arm shift  --steps 15 --nsamp 16384 --threads 1 --cpu-cap 5  | tee -a logs/shift_output.txt
$PY -u trsr.py --mode run --arm ladder --steps 15 --nsamp 16384 --threads 1 --cpu-cap 9  | tee -a logs/ladder_output.txt
$PY -u trsr.py --mode verdict | tee logs/verdict_output.txt
