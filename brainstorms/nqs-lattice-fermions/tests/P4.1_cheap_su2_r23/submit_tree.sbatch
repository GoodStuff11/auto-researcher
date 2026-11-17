#!/bin/bash
# P4.1 r23 stage 1 (RULE_amend1.md): R22 `spin-tree` falsifier 2, seed 0 only. Rule: RULE.md + RULE_amend1.md.
# Prepared, NOT submitted by the worker; submit only through compute_gate.py submit with voi.md.
# 1 CPU: stage A seed 0 (7000 exact-sum Adam steps) into results/, then copy it to results/oneseed/ and run the
# r22 ladder polish (<= 12 steps), the sign diagnostic and the one-seed verdict there. Seeds 1-2 are released only if
# verdict1 says RELEASE (separate gated job; the 3-seed polish then runs in results/).
# Wall time (measured 1 thread, logs/time.log): Adam 0.94 CPU-s/step -> 110 min; ladder 0.54 CPU-min/step -> 6.5 min;
# set-up + sign ~2 min. 119 min x 1.3 = 155 min -> 02:40:00; 1 CPU -> 2.67 CPU-h cap.
# Rerun-safe: pretrain skips if results/stageA_seed0.json exists; polish resumes from results/oneseed/state_tree.json.
#SBATCH --partition=kim
#SBATCH --job-name=p41-r23-tree-s0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:40:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r23/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r23/logs/%x_%j.err
set -uo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r23
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
echo "r23 tree arm seed 0 on $(hostname), job ${SLURM_JOB_ID:-local}"
$PY -u -W ignore tree_nqs.py --mode pretrain --seed 0 --nA 7000 --threads 1 | tee logs/pretrain_seed0.txt
mkdir -p results/oneseed
cp -n results/stageA_seed0.json results/thetaA_seed0.npy results/oneseed/
$PY -u -W ignore tree_nqs.py --mode polish --seeds 1 --steps 12 --cpu-cap 30 --threads 1 --res results/oneseed | tee logs/polish_seed0.txt
$PY -u -W ignore tree_nqs.py --mode sign --threads 1 --res results/oneseed | tee logs/sign_seed0.txt
$PY -u -W ignore tree_nqs.py --mode verdict1 --res results/oneseed | tee logs/verdict1_output.txt
