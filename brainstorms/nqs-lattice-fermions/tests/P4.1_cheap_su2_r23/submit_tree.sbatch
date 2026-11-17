#!/bin/bash
# P4.1 r23: R22 `spin-tree` falsifier 2 (tree-basis NQS vs ladder-polished `che`, 12 sites, exact sums). Rule: RULE.md.
# Prepared, NOT submitted by the worker; submit only through compute_gate.py submit with voi.md.
# One task, 3 CPUs: stage A (3 seeds x 7000 exact-sum Adam steps) in parallel, one process per CPU; then stage B
# (r22 ladder polish of the best seed, <= 12 steps), the Gate-2 sign diagnostic and the verdict, serially.
# Wall time (measured 1 thread, logs/time.log): Adam step 0.94 CPU-s -> 7000 steps 110 min per seed; ladder step
# 0.54 CPU-min -> 12 steps 6.5 min; set-up and sign ~2 min. ~119 min x 1.3 = 155 min -> 02:40:00. 3 CPU x 2.67 h = 8.0 CPU-h cap.
# Rerun-safe: pretrain skips a seed whose results/stageA_seed<s>.json exists; polish resumes from state_tree.json.
#SBATCH --partition=kim
#SBATCH --job-name=p41-r23-tree
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=02:40:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r23/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r23/logs/%x_%j.err
set -uo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r23
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
echo "r23 tree arm on $(hostname), job ${SLURM_JOB_ID:-local}"
for s in 0 1 2; do
  $PY -u -W ignore tree_nqs.py --mode pretrain --seed $s --nA 7000 --threads 1 > "logs/pretrain_seed${s}.txt" 2>&1 &
done
wait
$PY -u -W ignore tree_nqs.py --mode polish --seeds 3 --steps 12 --cpu-cap 30 --threads 3 | tee logs/polish_output.txt
$PY -u -W ignore tree_nqs.py --mode sign --threads 3 | tee logs/sign_output.txt
$PY -u -W ignore tree_nqs.py --mode verdict | tee logs/verdict_output.txt
