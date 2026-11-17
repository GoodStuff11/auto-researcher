#!/bin/bash
# P4.1 round 21, route R21 trSR T1 at 12 sites (che only). Rule: RULE.md as amended by RULE_amend1.md.
# Prepared, NOT submitted by the worker; submit only through compute_gate.py submit with voi.md.
# Arms: trsr (cutoff 1e-4 rel), shift (abs 1e-3 Tikhonov), both with a held-out metric-length ladder
# {0.005,0.01,0.02,0.05}; ladder (cutoff ladder at 0.05, reported only).
# Wall time (measured, 1 thread): set-up 0.25 CPU-min per arm; trsr/shift step 0.30 CPU-min
# (logs/probe_amend1_trsr_1step_1thread.txt); ladder step 0.38 (logs/probe_ladder_1step_1thread.txt).
# 3 x 0.25 + 15 x (0.30 + 0.30 + 0.38) = 15.5 min; x1.3 = 20 -> 0:25:00.
# Memory: 2.0 GB peak RSS (probe) -> --mem=4G.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:25:00
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
$PY -u trsr.py --mode run --arm trsr   --steps 15 --nsamp 16384 --threads 1 --cpu-cap 7  | tee -a logs/trsr_output.txt
$PY -u trsr.py --mode run --arm shift  --steps 15 --nsamp 16384 --threads 1 --cpu-cap 7  | tee -a logs/shift_output.txt
$PY -u trsr.py --mode run --arm ladder --steps 15 --nsamp 16384 --threads 1 --cpu-cap 9  | tee -a logs/ladder_output.txt
$PY -u trsr.py --mode verdict | tee logs/verdict_output.txt
