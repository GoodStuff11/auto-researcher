#!/bin/bash
# P5.1 R51.1 `nfloor` unit 3 (RULE_B.md release order: "a second seed (9500) of the arms whose verdict margin is
# within 2 replica spreads", spread 0.015 -> 0.030 in R).  Prepared round 4, NOT submitted by the worker; submit
# only through compute_gate.py submit with voi_unit3.md.  Margins after unit 2 (results/score_unit2.json):
#   C to NOISE-FLOOR's Delta(C) <= 0.62 Delta(B):      0.026  -> second seed
#   D to STEP-NOISE's Delta(D) <= 0.75 Delta(A):       0.018  -> second seed
#   A to STEP-NOISE (same inequality, A side):         0.025  -> second seed
#   B (0.038, 0.043) is outside 2 spreads -> not rerun.
# Three array tasks: 0 = A (N 16384, tau 0.005, 120 steps), 1 = C (N 262144, --dedup, 120), 2 = D (16384, 0.0025, 240).
# Wall time from unit 2's in-job rates (job 761860): A 0.359 CPU-min/step x 120 = 43 min; C 95.5 min for 120 steps;
# D 97.9 min for 240 steps.  Longest (D) 98 x 1.3 = 127 min -> --time 02:15:00; --cpu-cap 130 per task.
# Memory: unit 2 MaxRSS C 3.3 GB, A/D ~2.4 GB -> 8G.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:15:00
#SBATCH --array=0-2
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_noise_floor_r01/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_noise_floor_r01/logs/%x_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_noise_floor_r01
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
case ${SLURM_ARRAY_TASK_ID:-0} in
  0) N=16384;  TAU=0.005;  STEPS=120; EXTRA="" ;;
  1) N=262144; TAU=0.005;  STEPS=120; EXTRA="--dedup" ;;
  2) N=16384;  TAU=0.0025; STEPS=240; EXTRA="" ;;
esac
echo "P5.1 R51.1 nfloor unit 3 task ${SLURM_ARRAY_TASK_ID:-0}: N $N tau $TAU steps $STEPS seed0 9500 $EXTRA on $(hostname), job ${SLURM_JOB_ID:-local}"
$PY -u nf_run.py --nsamp "$N" --tau "$TAU" --steps "$STEPS" --seed0 9500 --cpu-cap 130 $EXTRA | tee -a "logs/nf_N${N}_tau${TAU}_s9500.txt"
