#!/bin/bash
# P4.1 R2 T2b equal-wall-time amendment (leaf section "R2 T2b equal-wall-time amendment (2026-11-16)").
# Task i runs line i+1 of tasks_t2b_wall.txt: "<Lx> <arm> <seed> <iters> <exact|-> <n_eval> <max_hours> <ckpts>",
# through t2b_wall.py (t2b.py train unchanged, tag suffix _wall, outdir results/t2b_wall, never results/t2b).
#   0: che_wall, open 8x3, N_e = 20, (+,+): 8400 SR steps = 3000 x 2.8 (pcg/che cost per step at 24 sites). GATING.
#   1: che_wall, open 4x3, N_e = 10, (+,+): 5100 steps = 3000 x 1.7 (12-site ratio, 1024 chains). Context only.
#   2: r1_wall,  open 8x3: 16500 steps = 3000 x 5.5 (pcg/r1 at 24 sites). Optional, context only.
# Checkpoints at 25, 50 and 75% (the same wall times as pcg's 750, 1500 and 2250). No futility clause.
# Budget (written exception, Stage 3.5.1): about 41 CPU-hours for task 0, 4 for task 1, 40 for task 2.
# Default array 0-1 (gating plus 12-site context); task 2: sbatch --array=2 submit_t2b_wall.sbatch
# If T2b returns AMBIGUOUS: sbatch --export=ALL,SEED=1 --array=0 submit_t2b_wall.sbatch
# Afterwards (login node): python3 t2b_wall_report.py --mode report
#SBATCH --job-name=P41t2bW
#SBATCH --partition=kim
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_t2b/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_t2b/logs/%x_%A_%a.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_t2b
cd "$DIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
read LX ARM SEED0 ITERS EXACT NEVAL MAXH CKPTS <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" tasks_t2b_wall.txt)"
SEED=${SEED:-$SEED0}
ARGS="--mode train --arm $ARM --Lx $LX --Ly 3 --pg=1,1 --seed $SEED --iters $ITERS --n-chains 1024 --threads 4"
ARGS="$ARGS --max-hours $MAXH --ckpts $CKPTS --ckpt-eval 32768 --outdir $DIR/results/t2b_wall"
if [ "$EXACT" = "exact" ]; then ARGS="$ARGS --exact-eval"; else ARGS="$ARGS --n-eval $NEVAL"; fi
WRAP=${WRAP:-}
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): t2b_wall.py $WRAP $ARGS"
~/.conda/envs/lit_env/bin/python -u t2b_wall.py $WRAP $ARGS
