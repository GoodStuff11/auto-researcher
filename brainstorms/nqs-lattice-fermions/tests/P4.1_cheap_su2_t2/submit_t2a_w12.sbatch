#!/bin/bash
# P4.1 round-7 T2 pilots at 12 sites (open 4x3, N_e = 10, U = 8, t' = -0.25, (+,+) irrep, exact final energy).
# Line i+1 of tasks_t2a_w12.txt = "<test> <arm> <H>":
#   tasks 0-2  T2a (R2): pcg (H matched to che@16: 15, P = 2696), r1 (H = 25, P = 2596), foil (P = 221)
#   tasks 3-5  W-12 (R3 premise): che at H = 4, 8, 32 (H = 16 is the pilot logs/pilot/pg_che, not rerun)
# Protocol = the 12-site pilot: 1000 SR steps, 1024 chains, lr 0.05 (cosine to 0.005 over the second half), Polyak
# average of the last 250 steps, diag shift 1e-3, P_G (1,1), 3 threads. Exact checkpoint energies at 250, 500, 750.
# pcg only: pre-registered futility stop (leaf, "Round 7 T2 pilots"): eps(250) > 2.29e-2 or eps(500) > 1.43e-2.
# Cap: 2 CPU-hours per task (--max-hours 0.65 wall at 3 threads) for r1 and che. pcg and foil (Pfaffian path, measured
# 2.1x che per step at 12 sites, logs/timing_L12_24_output.txt) get the written exception of the leaf: 2.55 CPU-hours
# (--max-hours 0.85). A capped task writes capped = true; resubmitting the same array index resumes from its checkpoint.
# Afterwards (login node): python3 t2_report.py --mode report
#SBATCH --job-name=P41t2a_w12
#SBATCH --partition=kim
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_t2/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_t2/logs/%x_%A_%a.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_t2
cd "$DIR"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
SEED=${SEED:-0}  # AMBIGUOUS T2a: sbatch --export=ALL,SEED=1 --array=0-1 submit_t2a_w12.sbatch
read TEST ARM H <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" tasks_t2a_w12.txt)"
ARGS="--mode train --arm $ARM --Lx 4 --Ly 3 --pg=1,1 --seed $SEED --iters 1000 --n-chains 1024 --exact-eval --exact-ckpts 250,500,750"
MAXH=0.65
if [ "$ARM" = "pcg" ] || [ "$ARM" = "foil" ]; then MAXH=0.85; fi
ARGS="$ARGS --threads 3 --max-hours $MAXH --outdir $DIR/results/$TEST"
if [ "$ARM" = "che" ]; then ARGS="$ARGS --width $H"; else ARGS="$ARGS --match-width $H"; fi
if [ "$ARM" = "pcg" ]; then ARGS="$ARGS --futility 250:2.29e-2,500:1.43e-2"; fi
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): vmc_t2.py $ARGS"
~/.conda/envs/lit_env/bin/python -u vmc_t2.py $ARGS
