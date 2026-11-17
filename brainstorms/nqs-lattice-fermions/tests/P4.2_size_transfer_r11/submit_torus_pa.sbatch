#!/bin/bash
#SBATCH --job-name=P4.2_r11_torus_pa
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:12:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r11/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r11/logs/%x_%j.err
# P4.2 R9.5 via R5.1 on the 4x4 torus: anchored copy (pa), seed 0, 50 projected-SR steps; control = job 451914 scr.
# Rule: RULE.md. Cost basis: 52.6 s/step (451914 seed 0, slowest node) x 50 + setup 148 s + source ladder <= 185 s
# + records <= 240 s ~ 0.89 h; x1.3 = 1.16 h -> 1.2 h. A stepping guard (--wall-hours 1.05) records before the limit.
# Afterwards: python3 torus_pa.py --aggregate results/pa_s0.json
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r11
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
cd "$DIR"
mkdir -p results logs
$PY torus_pa.py --seed 0 --wall-hours 1.05 --out "$DIR/results/pa_s0.json"
$PY torus_pa.py --aggregate "$DIR/results/pa_s0.json"
