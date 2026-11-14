#!/bin/bash
#SBATCH --job-name=p22r5crit
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-17
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r5/critic_r5/crit_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r5/critic_r5/crit_%A_%a.err
# P2.2 round-5 per-leaf critic: exact SCREEN bias of trained WIBJ vs NNBFJ (H4 R=4.4, K=12,16,20,
# seeds 0-2, Part W settings: hidden 64, hJ 32, 6000 L-BFGS). Aggregate:
#   python3 critic_screen_bias.py --aggregate critic_r5
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r5
cd "$DIR"
PY=/share/apps/software/anaconda3/bin/python3
T=$SLURM_ARRAY_TASK_ID
KS=(12 16 20); ARMS=(NNBFJ WIBJ)
K=${KS[$((T / 6))]}; ARM=${ARMS[$(((T / 3) % 2))]}; S=$((T % 3))
$PY critic_screen_bias.py --n-atoms 4 --bond-length 4.4 --K $K --arm $ARM --seed $S --maxiter 6000 \
    --out "$DIR/critic_r5/critic_sb_H4_K${K}_${ARM}_s${S}.json"
