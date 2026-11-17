#!/bin/bash
# P5.1 R51.1 `nfloor` unit 2 (RULE_B.md), prepared round 3, NOT submitted by the worker; submit only through
# compute_gate.py submit with voi_unit2.md.  Three array tasks:
#   0  arm B completion: resume N=65536 tau 0.005 from results/N65536_tau0.005_s9000/run.json (83 -> 120 steps;
#      unit 1 was TRUNCATED by its CPU cap, so RULE_B cannot score Delta(B) or Delta(C)/Delta(B) without it)
#   1  arm C: N=262144 (16N) tau 0.005, 120 steps, --dedup (exact rewrite, logs/dedup_check.txt: step d equal to
#      5.7e-13; same R at preflight steps 1-2; 3.3 GB instead of 17.6 GB and 0.46 instead of 1.21 CPU-min/step)
#   2  arm D: N=16384 tau 0.0025, 240 steps (same code path as arm A)
# Wall time: in-job rates measured by unit 1 (job 759974, kim-cpu-04): 0.359 CPU-min/step at N, 0.790 at 4N, i.e.
# 1.62x and 1.79x the interactive pre-flight. Arm C pre-flight (dedup, interactive) 0.460 -> x1.79 = 0.82 in-job.
#   B: 37 x 0.790 + 0.6 = 29.8 min;  C: 120 x 0.82 + 1.0 = 99.4 min;  D: 240 x 0.359 + 0.5 = 86.7 min.
#   Longest (C) x1.3 = 129 min -> --time 02:20:00; per-task CPU cap 135 min (above every arm's need x1.3, below
#   --time, so no arm is truncated by its cap unless the node is >35% slower than kim-cpu-04, and the script then
#   stops cleanly and saves).
# Memory: B 5.4 GB measured (unit 1 MaxRSS), C 3.3 GB (dedup pre-flight), D ~2.4 GB -> 12G.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:20:00
#SBATCH --array=0-2
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_noise_floor_r01/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_noise_floor_r01/logs/%x_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_noise_floor_r01
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
case ${SLURM_ARRAY_TASK_ID:-0} in
  0) N=65536;  TAU=0.005;  STEPS=120; EXTRA="" ;;
  1) N=262144; TAU=0.005;  STEPS=120; EXTRA="--dedup" ;;
  2) N=16384;  TAU=0.0025; STEPS=240; EXTRA="" ;;
esac
echo "P5.1 R51.1 nfloor unit 2 task ${SLURM_ARRAY_TASK_ID:-0}: N $N tau $TAU steps $STEPS $EXTRA on $(hostname), job ${SLURM_JOB_ID:-local}"
$PY -u nf_run.py --nsamp "$N" --tau "$TAU" --steps "$STEPS" --seed0 9000 --cpu-cap 135 $EXTRA | tee -a "logs/nf_N${N}_tau${TAU}_s9000.txt"
