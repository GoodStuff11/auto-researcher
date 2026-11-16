#!/bin/bash
# P4.1 size-scaling decisive test: VMC-SR training array (vmc.py --mode train).
# Task i runs line i+1 of tasks.txt: "<Lx> <arm> <seed> <iters> <extra_ckpt|-> <exact|-> <pg> <n_chains>" (make_tasks.py).
# Every arm is point-group projected (--pg) onto the ground-state irrep of its size (amendment 2026-11-16).
# Restarting a task resumes from results/<tag>_ckpt.npz; a finished task (results/<tag>.json) is skipped.
#SBATCH --job-name=P41scal_vmc
#SBATCH --partition=kim
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-39
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_scaling/logs/vmc_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_scaling/logs/vmc_%A_%a.err
cd /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_scaling
read LX ARM SEED ITERS EXTRA EXACT PG NC <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" tasks.txt)"
ARGS="--mode train --Lx $LX --arm $ARM --seed $SEED --iters $ITERS --threads 16 --max-hours 150 --pg=$PG --n-chains $NC"
[ "$EXTRA" != "-" ] && ARGS="$ARGS --extra-checkpoints $EXTRA"
[ "$EXACT" = "exact" ] && ARGS="$ARGS --exact-eval"
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): vmc.py $ARGS"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
~/.conda/envs/lit_env/bin/python vmc.py $ARGS
