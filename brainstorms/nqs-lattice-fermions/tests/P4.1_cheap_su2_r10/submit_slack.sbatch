#!/bin/bash
# P4.1 round 10, premise L (optimisation slack) T2 (prepared, NOT submitted by the worker).
# One exact linear-method step in the FULL 2637-direction tangent space of the trained 12-site che
# (logs/pilot/pg_che), every sum exact over 627,264 configurations. Reading pre-registered in slack.py:
#   L >= 0.30 HOLDS, L <= 0.10 FAILS, otherwise PARTIAL (the dry run already gives L >= 0.199 on a subspace).
# Budget: tangent build about 2 min, Gram and H products about 17 TFlop (about 5 min at 8 threads): under
# 1 CPU-hour (T2 cap 2). Memory: 6.6 GB float32 tangent matrix plus float64 blocks; 32 GB requested.
# Afterwards: read logs/slack_che_output_job.txt or results/slack_che.json.
#SBATCH --job-name=P41slack
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r10/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r10/logs/%x_%j.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r10
cd "$DIR"
echo "slack.py --mode run --threads 8 on $(hostname)"
~/.conda/envs/lit_env/bin/python -u slack.py --mode run --threads 8 | tee logs/slack_che_output_job.txt
